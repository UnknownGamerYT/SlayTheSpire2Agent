from __future__ import annotations

from sts2sim.mechanics.card_effects import (
    card_effect_plan,
    normalize_card_effect_steps,
    normalize_card_spec,
)


def test_report_unknown_cards_use_known_specs() -> None:
    expected_types = {
        "rampage": "attack",
        "dominate": "attack",
        "greed": "curse",
        "iron_wave": "attack",
    }

    for card_id, card_type in expected_types.items():
        plan = card_effect_plan({"id": card_id, "name": card_id, "type": "unknown"})

        assert plan.card["type"] == card_type
        assert plan.steps

    greed = card_effect_plan({"id": "greed", "name": "greed", "type": "unknown", "cost": 1})
    assert "eternal" in greed.card["tags"]
    assert greed.card["cost"] == -1
    assert greed.card["target"] == "none"
    assert greed.card["custom"]["frontloaded_gold"] == 333


def test_catalog_style_multi_hit_card_normalizes_to_damage_sequence() -> None:
    plan = card_effect_plan(
        {
            "id": "CELESTIAL_MIGHT",
            "name": "Celestial Might",
            "type": "Attack",
            "target": "AnyEnemy",
            "cost": 2,
            "damage": 6,
            "hit_count": 3,
        }
    )

    assert plan.card["card_id"] == "celestial_might"
    assert plan.card["target"] == "enemy"
    assert plan.steps == ({"damage": 6}, {"damage": 6}, {"damage": 6})
    assert plan.card["effects"] == {"sequence": [{"damage": 6}, {"damage": 6}, {"damage": 6}]}
    assert plan.events[0]["kind"] == "card_effects_normalized"


def test_all_enemy_multi_hit_uses_all_damage_steps() -> None:
    steps = normalize_card_effect_steps(
        {
            "id": "DAGGER_SPRAY",
            "type": "Attack",
            "target": "AllEnemies",
            "damage": 4,
            "hit_count": 2,
        }
    )

    assert steps == ({"all_damage": 4}, {"all_damage": 4})


def test_powers_applied_become_status_steps_with_targets() -> None:
    plan = card_effect_plan(
        {
            "id": "ASSASSINATE",
            "name": "Assassinate",
            "type": "Attack",
            "target": "AnyEnemy",
            "cost": 0,
            "damage": 10,
            "powers_applied": [
                {"amount": 1, "power": "Vulnerable", "power_key": "Vulnerable"},
            ],
        }
    )

    assert plan.steps == (
        {"damage": 10},
        {"apply_status": {"target": "enemy", "vulnerable": 1}},
    )


def test_self_power_card_statuses_are_kept_as_self_markers() -> None:
    plan = card_effect_plan(
        {
            "id": "ABRASIVE",
            "name": "Abrasive",
            "type": "Power",
            "target": "Self",
            "cost": 3,
            "powers_applied": [
                {"amount": 4, "power": "Thorns", "power_key": "Thorns"},
                {"amount": 1, "power": "Dexterity", "power_key": "Dexterity"},
            ],
        }
    )

    assert plan.steps == (
        {"apply_status": {"target": "self", "thorns": 4}},
        {"apply_status": {"target": "self", "dexterity": 1}},
    )


def test_generated_card_steps_include_temporary_card_descriptors() -> None:
    plan = card_effect_plan(
        {
            "id": "BLADE_DANCE",
            "name": "Blade Dance",
            "type": "Skill",
            "target": "Self",
            "cost": 1,
            "spawns_cards": ["SHIV"],
        },
        card_library={
            "SHIV": {
                "id": "SHIV",
                "name": "Shiv",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 4,
            }
        },
    )

    generated = plan.steps[0]["add_card_to_hand"]["card"]
    assert generated["card_id"] == "shiv"
    assert generated["cost"] == 0
    assert generated["custom"]["temporary"] is True
    assert generated["custom"]["generated"] is True
    assert generated["effects"] == {"sequence": [{"damage": 4}]}


def test_special_orb_text_is_merged_into_card_effect_plan() -> None:
    plan = card_effect_plan(
        {
            "id": "ORB_ROUTINE",
            "name": "Orb Routine",
            "type": "Skill",
            "target": "Self",
            "cost": 1,
            "description": "Channel 1 Lightning. Evoke your rightmost Orb twice.",
        }
    )

    assert plan.steps == (
        {"channel_orb": {"orb": "lightning", "amount": 1}},
        {"evoke_orb": {"selector": "rightmost", "amount": 2}},
    )
    assert plan.card["effects"] == {
        "sequence": [
            {"channel_orb": {"orb": "lightning", "amount": 1}},
            {"evoke_orb": {"selector": "rightmost", "amount": 2}},
        ]
    }
    assert any(event["kind"] == "card_special_normalized" for event in plan.events)


def test_special_resource_text_does_not_duplicate_description_resource_step() -> None:
    plan = card_effect_plan(
        {
            "id": "CALL_HELP",
            "name": "Call Help",
            "type": "Skill",
            "target": "Self",
            "cost": 0,
            "description": "[gold]Summon[/gold] 4.",
        }
    )

    assert plan.steps == ({"player_resource": {"resource": "summon", "amount": 4}},)


def test_special_stance_mantra_and_soul_text_merge_into_card_effect_plan() -> None:
    plan = card_effect_plan(
        {
            "id": "SOUL_STANCE",
            "name": "Soul Stance",
            "type": "Skill",
            "target": "Self",
            "cost": 0,
            "description": "Enter Calm. Gain 3 Mantra. Add a Soul+ into your Hand.",
        }
    )

    assert plan.steps[0:2] == (
        {"apply_status": {"target": "self", "stance_calm": 1}},
        {"player_resource": {"resource": "mantra", "amount": 3, "source": "card_special"}},
    )
    soul_step = plan.steps[2]
    assert tuple(soul_step) == ("add_card_to_hand",)
    generated = soul_step["add_card_to_hand"][0]["card"]
    assert generated["id"] == "SOUL"
    assert generated["draw"] == 3
    assert generated["upgraded"] is True


def test_named_cost_and_end_of_combat_upgrade_effects_become_status_markers() -> None:
    steps = normalize_card_effect_steps(
        [
            {"effect": "power_cost_reduction", "amount": 1},
            {"effect": "end_of_combat_upgrade_random", "upgrade_random_count": 2},
        ],
        card_type="power",
        target="self",
    )

    assert steps == (
        {"apply_status": {"target": "self", "power_cost_reduction": 1}},
        {"apply_status": {"target": "self", "end_of_combat_upgrade_random": 2}},
    )


def test_lifecycle_modifier_text_becomes_engine_steps() -> None:
    steps = normalize_card_effect_steps(
        {
            "id": "CORRUPTION",
            "type": "Power",
            "target": "Self",
            "description": "Skills cost 0 [energy:1]. Whenever you play a Skill, Exhaust it.",
        }
    )

    assert steps == (
        {"apply_status": {"target": "self", "skill_cost_zero": 1}},
        {"apply_status": {"target": "self", "skill_exhaust_on_play": 1}},
    )


def test_drum_of_battle_exhaust_energy_is_triggered_not_immediate() -> None:
    plan = card_effect_plan(
        {
            "id": "DRUM_OF_BATTLE",
            "name": "Drum of Battle",
            "type": "Skill",
            "target": "Self",
            "cost": 1,
            "draw": 2,
            "energy_gain": 2,
            "description": "Draw 2 cards.\nWhen this card is Exhausted, gain [energy:2].",
        }
    )

    assert plan.card["cost"] == 1
    assert plan.steps == (
        {"draw": 2},
        {
            "combat_trigger": {
                "trigger": "card_exhausted",
                "duration": "once",
                "effects": ({"energy": 2},),
                "text": "when this card is exhausted, gain [energy:2]",
                "condition": {"card_id": "drum_of_battle"},
            }
        },
    )


def test_fusion_upgrade_removes_exhaust_from_normalized_card() -> None:
    base = {
        "id": "FUSION",
        "name": "Fusion",
        "type": "Skill",
        "target": "Self",
        "cost": 1,
        "keywords_key": ("Exhaust",),
        "description": "Gain [energy:2]. Exhaust.",
        "energy_gain": 2,
        "upgrade": {"remove_exhaust": 1},
    }

    assert normalize_card_spec(base)["exhausts"] is True
    assert normalize_card_spec({**base, "upgraded": True})["exhausts"] is False


def test_howl_from_beyond_exhaust_pile_timing_is_preserved() -> None:
    plan = card_effect_plan(
        {
            "id": "HOWL_FROM_BEYOND",
            "name": "Howl From Beyond",
            "type": "Skill",
            "target": "Self",
            "cost": 0,
            "description": "At the end of your turn, if this is in your Exhaust Pile, play it.",
        }
    )

    assert plan.steps == (
        {
            "combat_trigger": {
                "trigger": "turn_end",
                "duration": "combat",
                "condition": {"zone": "exhaust_pile", "card_id": "howl_from_beyond"},
                "effects": (
                    {
                        "play_named_card": {
                            "card_id": "howl_from_beyond",
                            "from_zone": "exhaust_pile",
                        }
                    },
                ),
                "text": (
                    "at the end of your turn, if this is in your exhaust pile, play it"
                ),
            }
        },
    )


def test_resource_power_triggers_normalize_to_combat_triggers() -> None:
    genesis = card_effect_plan(
        {
            "id": "GENESIS",
            "name": "Genesis",
            "type": "Power",
            "target": "Self",
            "description": "At the start of your turn, gain [star:2].",
        }
    )
    subroutine = card_effect_plan(
        {
            "id": "SUBROUTINE",
            "name": "Subroutine",
            "type": "Power",
            "target": "Self",
            "description": "Whenever you play a Power, gain [energy:1].",
        }
    )
    throne = card_effect_plan(
        {
            "id": "THE_SEALED_THRONE",
            "name": "The Sealed Throne",
            "type": "Power",
            "target": "Self",
            "description": "Whenever you play a card, gain [star:1].",
        }
    )

    assert genesis.steps == (
        {
            "combat_trigger": {
                "trigger": "turn_start",
                "duration": "combat",
                "effects": ({"player_resource": {"resource": "star", "amount": 2}},),
            }
        },
    )
    assert subroutine.steps == (
        {
            "combat_trigger": {
                "trigger": "card_played",
                "duration": "combat",
                "condition": {"card_type": "power"},
                "effects": ({"energy": 1},),
            }
        },
    )
    assert throne.steps == (
        {
            "combat_trigger": {
                "trigger": "card_played",
                "duration": "combat",
                "condition": {"card_type": "any"},
                "effects": ({"player_resource": {"resource": "star", "amount": 1}},),
            }
        },
    )


def test_direct_sovereign_blade_cards_ignore_stale_power_and_spawn_fields() -> None:
    parry = card_effect_plan(
        {
            "id": "PARRY",
            "name": "Parry",
            "type": "Power",
            "target": "Self",
            "cost": 1,
            "powers_applied": [{"amount": 10, "power": "Parry", "power_key": "Parry"}],
            "spawns_cards": ["SOVEREIGN_BLADE"],
            "description": "[gold]Sovereign Blade[/gold] now gains 10 [gold]Block[/gold].",
        }
    )
    sage = card_effect_plan(
        {
            "id": "SWORD_SAGE",
            "name": "Sword Sage",
            "type": "Power",
            "target": "Self",
            "cost": 2,
            "powers_applied": [
                {"amount": 1, "power": "Sword Sage", "power_key": "SwordSage"}
            ],
            "spawns_cards": ["SOVEREIGN_BLADE"],
            "description": "[gold]Sovereign Blade[/gold] gains [gold]Replay[/gold] 1.",
        }
    )

    assert parry.steps == ({"sovereign_blade": {"action": "gain_block", "amount": 10}},)
    assert sage.steps == (
        {
            "add_keyword_to_matching_cards": {
                "keyword": "replay",
                "amount": 1,
                "filter": {
                    "card_id_contains": "sovereign_blade",
                    "exclude_keyword": "replay",
                },
                "zones": ("hand", "draw_pile", "discard_pile", "exhaust_pile"),
            }
        },
    )


def test_upgraded_description_drives_tesla_coil_and_parry_plus_markers() -> None:
    tesla_base = {
        "id": "TESLA_COIL",
        "name": "Tesla Coil",
        "type": "Attack",
        "target": "AnyEnemy",
        "cost": 0,
        "damage": 3,
        "description": "Deal 3 damage.\nTrigger all Lightning against the enemy.",
        "upgrade_description": "Deal 4 damage.\nTrigger all Lightning against the enemy twice.",
    }
    parry_base = {
        "id": "PARRY",
        "name": "Parry",
        "type": "Power",
        "target": "Self",
        "cost": 1,
        "powers_applied": [{"amount": 10, "power": "Parry", "power_key": "Parry"}],
        "spawns_cards": ["SOVEREIGN_BLADE"],
        "description": "[gold]Sovereign Blade[/gold] now gains 10 [gold]Block[/gold].",
        "upgrade_description": (
            "[gold]Sovereign Blade[/gold] now gains 14 [gold]Block[/gold]."
        ),
    }

    tesla = card_effect_plan({**tesla_base, "upgraded": True, "damage": 4})
    parry = card_effect_plan({**parry_base, "upgraded": True})

    assert tesla.steps == (
        {"damage": 4},
        {"trigger_orb_passive": {"selector": "all", "amount": 2, "orb_filter": "lightning"}},
    )
    assert parry.steps == ({"sovereign_blade": {"action": "gain_block", "amount": 14}},)


def test_next_card_extra_play_duration_follows_this_turn_text() -> None:
    persistent = normalize_card_effect_steps(
        {
            "id": "PERSISTENT_NEXT",
            "type": "Skill",
            "target": "Self",
            "description": "Your next Skill is played an extra time.",
        }
    )
    temporary = normalize_card_effect_steps(
        {
            "id": "TEMPORARY_NEXT",
            "type": "Skill",
            "target": "Self",
            "description": "This turn, your next card is played an extra time.",
        }
    )

    assert persistent == (
        {
            "next_card_extra_play": {
                "card_type": "skill",
                "amount": 1,
                "duration": "combat",
            }
        },
    )
    assert temporary == (
        {
            "next_card_extra_play": {
                "card_type": "card",
                "amount": 1,
                "duration": "turn",
            }
        },
    )


def test_dynamic_value_text_becomes_formula_steps() -> None:
    body_slam = normalize_card_effect_steps(
        {
            "id": "BODY_SLAM",
            "type": "Attack",
            "target": "AnyEnemy",
            "description": "Deal damage equal to your Block.",
        }
    )
    stack = normalize_card_effect_steps(
        {
            "id": "STACK",
            "type": "Skill",
            "target": "Self",
            "description": "Gain Block equal to the number of cards in your Discard Pile.",
        }
    )

    assert body_slam == ({"damage_formula": {"formula": "player_block"}},)
    assert stack == ({"block_formula": {"formula": "discard_pile_count"}},)


def test_more_dynamic_and_generated_text_becomes_engine_steps() -> None:
    double_energy = normalize_card_effect_steps(
        {
            "id": "DOUBLE_ENERGY",
            "type": "Skill",
            "target": "Self",
            "description": "Double your Energy.",
        }
    )
    infernal_blade = normalize_card_effect_steps(
        {
            "id": "INFERNAL_BLADE",
            "type": "Skill",
            "target": "Self",
            "description": (
                "Add a random Attack into your [gold]Hand[/gold]. "
                "It's free to play this turn."
            ),
        }
    )
    scrawl = normalize_card_effect_steps(
        {
            "id": "SCRAWL",
            "type": "Skill",
            "target": "Self",
            "description": "Draw cards until your [gold]Hand[/gold] is full.",
        }
    )
    prolong = normalize_card_effect_steps(
        {
            "id": "PROLONG",
            "type": "Skill",
            "target": "Self",
            "description": (
                "Next turn, gain [gold]Block[/gold] equal to your current [gold]Block[/gold]."
            ),
        }
    )
    echo_form = normalize_card_effect_steps(
        {
            "id": "ECHO_FORM",
            "type": "Power",
            "target": "Self",
            "description": "The first card you play each turn is played an extra time.",
        }
    )
    apotheosis = normalize_card_effect_steps(
        {
            "id": "APOTHEOSIS",
            "type": "Skill",
            "target": "Self",
            "description": "[gold]Upgrade[/gold] ALL your cards.",
        }
    )
    alchemize = normalize_card_effect_steps(
        {
            "id": "ALCHEMIZE",
            "type": "Skill",
            "target": "Self",
            "description": "Procure a random potion.",
        }
    )
    enlightenment = normalize_card_effect_steps(
        {
            "id": "ENLIGHTENMENT",
            "type": "Skill",
            "target": "Self",
            "description": (
                "Reduce the cost of ALL cards in your [gold]Hand[/gold] to 1 this turn."
            ),
        }
    )

    assert double_energy == ({"energy_formula": {"formula": "current_energy"}},)
    assert infernal_blade == (
        {
            "add_random_card_to_hand": {
                "count": 1,
                "card_types": ("attack",),
                "free_to_play_this_turn": True,
            }
        },
    )
    assert scrawl == ({"draw_formula": {"formula": "hand_space"}},)
    assert prolong == ({"next_turn": {"block": {"formula": "player_block"}}},)
    assert echo_form == (
        {
            "combat_trigger": {
                "trigger": "turn_start",
                "effects": (
                    {
                        "next_card_extra_play": {
                            "card_type": "card",
                            "amount": 1,
                            "duration": "turn",
                        }
                    },
                ),
            }
        },
    )
    assert apotheosis == ({"upgrade_all_combat_cards": True},)
    assert alchemize == ({"add_random_potion": {"count": 1}},)
    assert enlightenment == (
        {"set_hand_cost": {"cost": 1, "max_cost_only": True, "duration": "turn"}},
    )


def test_status_loss_and_enemy_defense_text_becomes_engine_steps() -> None:
    dark_shackles = normalize_card_effect_steps(
        {
            "id": "DARK_SHACKLES",
            "type": "Skill",
            "target": "AnyEnemy",
            "description": "Enemy loses 9 [gold]Strength[/gold] this turn.",
        }
    )
    shockwave = normalize_card_effect_steps(
        {
            "id": "SHOCKWAVE",
            "type": "Skill",
            "target": "AllEnemies",
            "description": (
                "Apply 3 [gold]Weak[/gold] and [gold]Vulnerable[/gold] to ALL enemies."
            ),
        }
    )
    expose = normalize_card_effect_steps(
        {
            "id": "EXPOSE",
            "type": "Skill",
            "target": "AnyEnemy",
            "description": (
                "Remove all [gold]Artifact[/gold] and [gold]Block[/gold] from the enemy.\n"
                "Apply 2 [gold]Vulnerable[/gold]."
            ),
        }
    )
    malaise = normalize_card_effect_steps(
        {
            "id": "MALAISE",
            "type": "Skill",
            "target": "AnyEnemy",
            "description": (
                "Enemy loses X [gold]Strength[/gold]. Apply X [gold]Weak[/gold]."
            ),
        }
    )
    times_up = normalize_card_effect_steps(
        {
            "id": "TIMES_UP",
            "type": "Attack",
            "target": "AnyEnemy",
            "description": "Deal damage equal to the enemy's [gold]Doom[/gold].",
        }
    )

    assert dark_shackles == (
        {"apply_status": {"target": "enemy", "temporary_strength": -9}},
    )
    assert shockwave == (
        {"apply_status": {"target": "all_enemies", "weak": 3, "vulnerable": 3}},
    )
    assert expose == (
        {"remove_status": {"target": "enemy", "statuses": ("artifact",)}},
        {"remove_block": {"target": "enemy"}},
        {"apply_status": {"target": "enemy", "vulnerable": 2}},
    )
    assert malaise == (
        {
            "apply_status": {
                "target": "enemy",
                "strength": {"amount": 0, "per_energy": -1},
                "weak": {"amount": 0, "per_energy": 1},
            }
        },
    )
    assert times_up == ({"damage_formula": {"formula": "target_doom"}},)


def test_no_effect_status_and_hand_end_turn_cards_emit_explicit_markers() -> None:
    wound = normalize_card_effect_steps(
        {"id": "WOUND", "type": "Status", "target": "None", "cost": -1}
    )
    regret = normalize_card_effect_steps(
        {
            "id": "REGRET",
            "type": "Curse",
            "target": "None",
            "cost": -1,
            "description": (
                "At the end of your turn, if this is in your [gold]Hand[/gold], "
                "lose 1 HP for each card in your [gold]Hand[/gold]."
            ),
        }
    )
    enthralled = normalize_card_effect_steps(
        {
            "id": "ENTHRALLED",
            "type": "Curse",
            "target": "Self",
            "cost": 2,
            "description": (
                "If this is in your [gold]Hand[/gold], it must be played before other cards."
            ),
        }
    )
    lantern_key = normalize_card_effect_steps(
        {
            "id": "LANTERN_KEY",
            "type": "Quest",
            "target": "None",
            "cost": -1,
            "description": "Unlocks a special event in the next Act.",
        }
    )

    assert wound == ({"noop": True},)
    assert regret == ({"end_turn_hand_effect": True},)
    assert enthralled == ({"force_play_priority": True},)
    assert lantern_key == ({"noop": {"reason": "non_combat_quest_card"}},)
