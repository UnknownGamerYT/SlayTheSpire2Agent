from __future__ import annotations

from sts2sim.mechanics.card_effects import (
    card_effect_plan,
    normalize_card_effect_steps,
)


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
