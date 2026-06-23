from __future__ import annotations

from sts2sim.mechanics.card_specials import (
    card_special_blockers,
    card_special_events,
    card_special_plan,
    normalize_card_special_steps,
)


def _soul_payload(*, upgraded: bool = False) -> dict:
    draw = 3 if upgraded else 2
    return {
        "card": {
            "id": "SOUL",
            "name": "Soul+" if upgraded else "Soul",
            "type": "Skill",
            "target": "Self",
            "cost": 1,
            "draw": draw,
            "description": f"Draw {draw} cards.",
            "keywords_key": ("Exhaust",),
            "exhausts": True,
            "upgraded": upgraded,
        },
        "temporary": True,
    }


def test_channel_and_evoke_text_become_executable_orb_markers() -> None:
    plan = card_special_plan(
        {
            "id": "ORB_ROUTINE",
            "description": (
                "[gold]Channel[/gold] 1 [gold]Lightning[/gold].\n"
                "[gold]Evoke[/gold] your rightmost Orb twice."
            ),
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
        {"channel_orb": {"orb": "lightning", "amount": 1}},
        {"evoke_orb": {"selector": "rightmost", "amount": 2}},
    )
    assert [event["metadata"]["classification"] for event in plan.events] == [
        "executable",
        "executable",
    ]


def test_if_kill_resource_gain_becomes_post_damage_resource_marker() -> None:
    plan = card_special_plan(
        {
            "id": "FINISHING_BLOW",
            "description": "Deal 24 damage.\nIf this kills an enemy, gain [energy:3].",
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
        {
            "if_kill_resource": {
                "resource": "energy",
                "amount": 3,
                "trigger": "card_kills_enemy",
            }
        },
    )
    assert plan.blockers == ()
    assert plan.events[0]["metadata"]["classification"] == "executable"


def test_summon_osty_and_soul_text_emit_character_markers() -> None:
    card_spec = {
        "id": "NECRO_ASSIST",
        "description": (
            "[gold]Summon[/gold] 4.\n"
            "[gold]Osty[/gold] deals 6 damage to a random enemy.\n"
            "Add 2 [gold]Souls[/gold] into your [gold]Draw Pile[/gold]."
        ),
    }
    steps = normalize_card_special_steps(card_spec)

    assert steps == (
        {"player_resource": {"resource": "summon", "amount": 4, "source": "card_special"}},
        {"osty_action": {"action": "damage", "amount": 6, "target": "random_enemy"}},
        {"add_card_to_draw": (_soul_payload(), _soul_payload())},
    )
    assert card_special_blockers(card_spec) == ()


def test_stance_and_mantra_text_emit_executable_status_and_resource_steps() -> None:
    plan = card_special_plan(
        {
            "id": "INNER_FIRE",
            "description": "Enter Wrath.\nGain 4 Mantra.\nExit your Stance.",
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
        {"apply_status": {"target": "self", "stance_wrath": 1}},
        {"player_resource": {"resource": "mantra", "amount": 4, "source": "card_special"}},
        {"apply_status": {"target": "self", "stance_none": 1}},
    )


def test_forge_and_sovereign_blade_text_emit_executable_markers() -> None:
    plan = card_special_plan(
        {
            "id": "BLADE_ORDER",
            "description": (
                "[gold]Forge[/gold] 3.\n"
                "[gold]Sovereign Blade[/gold] deals double damage to the enemy this turn.\n"
                "[gold]Sovereign Blade[/gold] now hits an additional time."
            ),
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
        {"player_resource": {"resource": "forge", "amount": 3, "source": "card_special"}},
        {
            "sovereign_blade": {
                "action": "temporary_modifier",
                "modifier": "double_damage",
                "duration": "turn",
                "target": "enemy",
            }
        },
        {"sovereign_blade": {"action": "add_hit", "amount": 1}},
    )
    assert plan.blockers == ()


def test_new_sovereign_blade_text_emits_parry_and_replay_markers() -> None:
    plan = card_special_plan(
        {
            "id": "BLADE_UPDATE_TEXT",
            "description": (
                "[gold]Sovereign Blade[/gold] now gains 14 Block.\n"
                "[gold]Sovereign Blade[/gold] gains Replay 1."
            ),
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
        {"sovereign_blade": {"action": "gain_block", "amount": 14}},
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
    assert plan.blockers == ()


def test_v107_orb_special_text_emits_new_repeat_markers() -> None:
    shatter = card_special_plan(
        {"id": "SHATTER", "description": "Evoke all of your Orbs twice."}
    )
    tesla = card_special_plan(
        {
            "id": "TESLA_COIL",
            "description": "Deal 4 damage. Trigger all Lightning against the enemy twice.",
        }
    )

    assert shatter.status == "executable"
    assert shatter.steps == ({"evoke_orb": {"selector": "all", "amount": 2}},)
    assert tesla.status == "executable"
    assert tesla.steps == (
        {"trigger_orb_passive": {"selector": "all", "amount": 2, "orb_filter": "lightning"}},
    )


def test_self_exhaust_energy_text_emits_card_specific_trigger() -> None:
    plan = card_special_plan(
        {
            "id": "DRUM_OF_BATTLE",
            "description": "Draw 2 cards.\nWhen this card is Exhausted, gain [energy:2].",
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
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
    assert plan.blockers == ()


def test_choose_card_required_text_is_classified_as_blocker_with_reason() -> None:
    blockers = card_special_blockers(
        {
            "id": "MINION_SWAP",
            "description": (
                "Choose 2 cards in your [gold]Draw Pile[/gold] to [gold]Transform[/gold]."
            ),
        }
    )

    assert blockers == (
        {
            "explicit_blocker": {
                "kind": "choose_card_required",
                "reason": (
                    "requires choosing card targets from a combat zone before the card can resolve"
                ),
                "metadata": {
                    "text": "choose 2 cards in your draw pile to transform",
                    "zone": "draw_pile",
                    "count": 2,
                    "choices": None,
                    "action": "transform",
                },
            }
        },
    )


def test_supported_transform_choice_text_becomes_executable_marker() -> None:
    plan = card_special_plan(
        {
            "id": "MINION_SWAP",
            "description": (
                "Choose 2 cards in your [gold]Draw Pile[/gold] to [gold]Transform[/gold] "
                "into [gold]Minion Dive Bombs[/gold]."
            ),
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
        {
            "choose_card": {
                "action": "transform",
                "zone": "draw_pile",
                "count": 2,
                "text": (
                    "choose 2 cards in your draw pile to transform into minion dive bombs"
                ),
                "target_card_id": "minion_dive_bomb",
            }
        },
    )
    assert plan.blockers == ()


def test_timed_choose_card_text_becomes_executable_marker() -> None:
    plan = card_special_plan(
        {
            "id": "ENTROPY",
            "description": "At the start of your turn, Transform 1 card in your Hand.",
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
        {
            "timed_choice": {
                "trigger": "turn_start",
                "repeat": True,
                "choose_card": {
                    "action": "transform",
                    "zone": "hand",
                    "count": 1,
                    "random_transform": True,
                    "text": "at the start of your turn, transform 1 card in your hand",
                },
                "text": "at the start of your turn, transform 1 card in your hand",
            }
        },
    )
    assert plan.blockers == ()


def test_tyranny_draw_then_exhaust_text_becomes_timed_choice_marker() -> None:
    plan = card_special_plan(
        {
            "id": "TYRANNY",
            "description": (
                "At the start of your turn, draw 1 card and Exhaust 1 card from your Hand."
            ),
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
        {
            "timed_choice": {
                "trigger": "turn_start",
                "repeat": True,
                "pre_effects": ({"draw": 1},),
                "choose_card": {
                    "action": "exhaust",
                    "zone": "hand",
                    "count": 1,
                    "text": (
                        "at the start of your turn, draw 1 card and exhaust 1 card "
                        "from your hand"
                    ),
                },
                "text": "at the start of your turn, draw 1 card and exhaust 1 card from your hand",
            }
        },
    )
    assert plan.blockers == ()


def test_shuffle_choose_card_text_becomes_timed_choice_marker() -> None:
    plan = card_special_plan(
        {
            "id": "STRATAGEM",
            "description": (
                "Whenever you shuffle your Draw Pile, choose a card from it to put into your Hand."
            ),
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
        {
            "timed_choice": {
                "trigger": "draw_pile_shuffled",
                "repeat": True,
                "choose_card": {
                    "action": "move_to_hand",
                    "zone": "draw_pile",
                    "count": 1,
                    "destination": "hand",
                    "text": (
                        "whenever you shuffle your draw pile, choose a card from it "
                        "to put into your hand"
                    ),
                },
                "text": (
                    "whenever you shuffle your draw pile, choose a card from it to put into "
                    "your hand"
                ),
            }
        },
    )
    assert plan.blockers == ()


def test_nightmare_choice_schedules_next_turn_copies_marker() -> None:
    plan = card_special_plan(
        {
            "id": "NIGHTMARE",
            "description": "Choose a card. Next turn, add 3 copies of that card into your Hand.",
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
        {
            "choose_card": {
                "action": "copy_to_hand_next_turn",
                "zone": "hand",
                "count": 1,
                "copy_count": 3,
                "destination": "hand",
                "text": "choose a card",
            }
        },
    )
    assert plan.blockers == ()


def test_combat_trigger_text_becomes_executable_marker() -> None:
    rage = card_special_plan(
        {
            "id": "RAGE",
            "description": "Whenever you play an Attack this turn, gain 3 Block.",
        }
    )
    panache = card_special_plan(
        {
            "id": "PANACHE",
            "description": (
                "Every time you play 5 cards in a single turn, deal 10 damage to ALL enemies."
            ),
        }
    )

    assert rage.steps == (
        {
            "combat_trigger": {
                "trigger": "card_played",
                "duration": "turn",
                "condition": {"card_type": "attack"},
                "effects": ({"block": 3},),
                "text": "whenever you play an attack this turn, gain 3 block",
            }
        },
    )
    assert panache.steps == (
        {
            "combat_trigger": {
                "trigger": "card_played",
                "duration": "combat",
                "counter_scope": "turn",
                "every": 5,
                "effects": ({"all_damage": 10},),
                "text": (
                    "every time you play 5 cards in a single turn, deal 10 damage to "
                    "all enemies"
                ),
            }
        },
    )


def test_persistent_modifier_text_becomes_status_marker() -> None:
    plan = card_special_plan(
        {
            "id": "BARRICADE",
            "description": "Block is not removed at the start of your turn.",
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
        {"apply_status": {"target": "self", "retain_block": 1}},
    )


def test_supported_choose_card_text_becomes_executable_marker() -> None:
    plan = card_special_plan(
        {
            "id": "SNAP_TEST",
            "description": "Add Retain to a card in your Hand.",
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
        {
            "choose_card": {
                "action": "add_retain",
                "zone": "hand",
                "count": 1,
                "text": "add retain to a card in your hand",
            }
        },
    )
    assert plan.blockers == ()


def test_generated_discovery_choice_text_becomes_executable_marker() -> None:
    plan = card_special_plan(
        {
            "id": "DISCOVERY_TEST",
            "description": (
                "Choose 1 of 3 random Colorless cards to add into your Hand. "
                "It's free to play this turn."
            ),
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
        {
            "choose_card": {
                "action": "move_to_hand",
                "zone": "generated",
                "count": 1,
                "choices": 3,
                "destination": "hand",
                "pool": "colorless",
                "free_to_play_this_turn": True,
                "text": (
                    "choose 1 of 3 random colorless cards to add into your hand"
                ),
            }
        },
    )


def test_copy_and_play_choice_text_become_executable_markers() -> None:
    copy_plan = card_special_plan(
        {
            "id": "DUAL_WIELD_TEST",
            "description": (
                "Choose an Attack or Power card. "
                "Add a copy of that card into your Hand."
            ),
        }
    )
    play_plan = card_special_plan(
        {
            "id": "DECISIONS_TEST",
            "description": "Choose a Skill in your Hand and play it 3 times.",
        }
    )

    assert copy_plan.steps == (
        {
            "choose_card": {
                "action": "copy_to_hand",
                "zone": "hand",
                "count": 1,
                "copy_count": 1,
                "destination": "hand",
                "text": "choose an attack or power card",
                "card_types": ("attack", "power"),
            }
        },
    )
    assert play_plan.steps == (
        {
            "choose_card": {
                "action": "play",
                "zone": "hand",
                "count": 1,
                "text": "choose a skill in your hand and play it 3 times",
                "card_types": ("skill",),
                "play_times": 3,
            }
        },
    )


def test_choose_card_source_zone_uses_from_pile_not_destination_pile() -> None:
    plan = card_special_plan(
        {
            "id": "HEADBUTT_TEST",
            "description": "Put a card from your Discard Pile on top of your Draw Pile.",
        }
    )

    assert plan.steps == (
        {
            "choose_card": {
                "action": "move_to_draw_top",
                "zone": "discard_pile",
                "count": 1,
                "text": (
                    "put a card from your discard pile on top of your draw pile"
                ),
                "destination": "draw_pile_top",
            }
        },
    )


def test_timed_draw_then_discard_text_becomes_pending_choice_marker() -> None:
    plan = card_special_plan(
        {
            "id": "TOOLS_OF_THE_TRADE",
            "description": "At the start of your turn, draw 1 card and discard 1 card.",
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
        {
            "timed_choice": {
                "trigger": "turn_start",
                "repeat": True,
                "pre_effects": ({"draw": 1},),
                "choose_card": {
                    "action": "discard",
                    "zone": "hand",
                    "count": 1,
                    "text": (
                        "at the start of your turn, draw 1 card and discard 1 card"
                    ),
                },
                "text": "at the start of your turn, draw 1 card and discard 1 card",
            }
        },
    )


def test_top_card_and_draw_top_modifier_text_become_executable_markers() -> None:
    mayhem = card_special_plan(
        {
            "id": "MAYHEM",
            "description": "At the start of your turn, play the top card of your Draw Pile.",
        }
    )
    nostalgia = card_special_plan(
        {
            "id": "NOSTALGIA",
            "description": (
                "The first Attack or Skill you play each turn is placed on top "
                "of your Draw Pile."
            ),
        }
    )

    assert mayhem.steps == (
        {
            "combat_trigger": {
                "trigger": "turn_start",
                "duration": "combat",
                "effects": ({"play_top_card": {"zone": "draw_pile"}},),
                "text": (
                    "at the start of your turn, play the top card of your draw pile"
                ),
            }
        },
    )
    assert nostalgia.steps == (
        {"apply_status": {"target": "self", "first_attack_skill_to_draw_top": 1}},
    )


def test_dynamic_orb_text_becomes_state_scaled_channel_marker() -> None:
    plan = card_special_plan(
        {
            "id": "ORB_TRADE",
            "description": (
                "Lose 1 Orb Slot.\n"
                "[gold]Channel[/gold] 1 [gold]Frost[/gold] for each enemy."
            ),
        }
    )

    assert plan.status == "executable"
    assert plan.steps == (
        {"orb_slot_delta": {"amount": -1}},
        {"dynamic_channel_orb": {"orb": "frost", "formula": "alive_enemy_count"}},
    )
    assert plan.blockers == ()


def test_card_special_events_helper_returns_same_event_shape() -> None:
    events = card_special_events(
        {
            "id": "SOUL_CALL",
            "description": "Add a [gold]Soul[/gold] into your [gold]Hand[/gold].",
        }
    )

    assert events == (
        {
            "kind": "card_special_normalized",
            "source_id": "soul_call",
            "target_id": None,
            "amount": 1,
            "metadata": {
                "classification": "executable",
                "special": "add_soul",
                "effect": {
                    "add_card_to_hand": (_soul_payload(),),
                },
            },
        },
    )
