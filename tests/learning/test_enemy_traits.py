from __future__ import annotations

from sts2sim.mechanics.enemy_traits import (
    ENEMY_TRAIT_AGGREGATE_KEYS,
    ENEMY_TRAIT_KEYS,
    enemy_slots_from_payload,
    enemy_trait_aggregate,
    enemy_trait_aggregate_vector,
    enemy_trait_summary,
    enemy_trait_vector,
)


def test_attack_multi_hit_enemy_uses_visible_damage_without_known_id() -> None:
    summary = enemy_trait_summary(
        {
            "monster_id": "totally_new_enemy",
            "hp": 40,
            "max_hp": 80,
            "intent": "Attack",
            "intent_damage": 6,
            "hit_count": 3,
            "statuses": {},
            "metadata": {},
        }
    )

    assert summary["intent_attack"] == 1.0
    assert summary["multi_hit"] == 1.0
    assert summary["single_hit_damage"] == 6.0
    assert summary["hit_count"] == 3.0
    assert summary["incoming_damage"] == 18.0
    assert summary["behavior_attack_frequency"] == 1.0
    assert summary["unknown_behavior"] == 0.0
    assert len(enemy_trait_vector(summary)) == len(ENEMY_TRAIT_KEYS)


def test_scaling_buff_enemy_exposes_strength_and_scaling_speed_from_keywords() -> None:
    summary = enemy_trait_summary(
        {
            "monster_id": "new_scaler",
            "hp": 50,
            "max_hp": 50,
            "intent": "Buff",
            "move_id": "GROW_STRENGTH",
            "intent_damage": 0,
            "hit_count": 1,
            "statuses": {"ritual": 2},
            "metadata": {
                "move_powers": (
                    {"power_id": "strength", "amount": 3, "target": "self"},
                ),
            },
        }
    )

    assert summary["intent_buff"] == 1.0
    assert summary["scaling_strength"] == 1.0
    assert summary["behavior_buff_frequency"] == 1.0
    assert summary["behavior_scaling_speed"] == 1.0
    assert summary["unknown_behavior"] == 0.0


def test_unknown_enemy_still_exposes_current_intent_and_unknown_behavior() -> None:
    summary = enemy_trait_summary(
        {
            "monster_id": "mystery",
            "hp": 10,
            "max_hp": 20,
            "block": 4,
            "intent": "Observe",
            "intent_damage": 0,
            "intent_block": 0,
            "statuses": {"poison": 3, "weak": 1},
            "metadata": {},
        }
    )

    assert summary["alive"] == 1.0
    assert summary["hp_fraction"] == 0.5
    assert summary["block"] == 4.0
    assert summary["current_poison"] == 3.0
    assert summary["current_weak"] == 1.0
    assert summary["intent_attack"] == 0.0
    assert summary["unknown_behavior"] == 1.0


def test_aggregate_vector_length_is_stable_and_averages_across_enemies() -> None:
    payload = {
        "combat": {
            "monsters": [
                {
                    "monster_id": "a",
                    "hp": 20,
                    "max_hp": 40,
                    "intent": "Attack",
                    "intent_damage": 5,
                    "hit_count": 2,
                    "statuses": {},
                    "metadata": {},
                },
                {
                    "monster_id": "b",
                    "hp": 30,
                    "max_hp": 60,
                    "intent": "Defend",
                    "intent_block": 12,
                    "statuses": {"poison": 4},
                    "metadata": {},
                },
            ]
        }
    }

    slots = enemy_slots_from_payload(payload)
    aggregate = enemy_trait_aggregate(payload)
    vector = enemy_trait_aggregate_vector(payload)

    assert len(slots) == 2
    assert aggregate["enemy_count"] == 2.0
    assert aggregate["alive_count"] == 2.0
    assert aggregate["total_incoming_damage"] == 10.0
    assert aggregate["average_incoming_damage"] == 5.0
    assert aggregate["average_hp_fraction"] == 0.5
    assert aggregate["average_intent_attack"] == 0.5
    assert aggregate["average_current_poison"] == 2.0
    assert len(vector) == len(ENEMY_TRAIT_AGGREGATE_KEYS)
    assert vector == tuple(aggregate[key] for key in ENEMY_TRAIT_AGGREGATE_KEYS)
