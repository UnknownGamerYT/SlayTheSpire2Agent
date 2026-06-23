from __future__ import annotations

from sts2sim.mechanics.trigger_visibility import (
    TRIGGER_VISIBILITY_KEYS,
    potion_slot_summary,
    trigger_visibility_summary,
    trigger_visibility_vector,
)


def test_timed_card_triggers_expose_delay_repeating_and_remaining_uses() -> None:
    payload = {
        "combat": {
            "turn": 3,
            "metadata": {
                "timed_card_triggers": (
                    {
                        "source_card_id": "setup",
                        "trigger": "turn_start",
                        "duration": "combat",
                        "delay": 2,
                        "remaining_uses": 3,
                        "every": 2,
                    },
                )
            },
        }
    }

    summary = trigger_visibility_summary(payload)

    assert summary["trigger_count"] == 1.0
    assert summary["start_turn_count"] == 1.0
    assert summary["delayed_count"] == 1.0
    assert summary["repeating_count"] == 1.0
    assert summary["periodic_count"] == 1.0
    assert summary["turns_until_next_effect_min"] == 2.0
    assert summary["remaining_uses_total"] == 3.0


def test_potion_capacity_pressure_for_two_of_three_slots() -> None:
    payload = {
        "player": {"max_potion_slots": 3},
        "potions": ["fire_potion", "block_potion"],
    }

    slots = potion_slot_summary(payload)
    summary = trigger_visibility_summary(payload)

    assert slots["capacity"] == 3
    assert slots["filled"] == 2
    assert slots["empty"] == 1
    assert slots["potions"][2] == {"slot_index": 2, "id": "", "empty": True}
    assert summary["potion_slots"] == 3.0
    assert summary["potion_slots_filled"] == 2.0
    assert summary["potion_slots_empty"] == 1.0
    assert summary["potion_capacity_pressure"] == 0.6667


def test_potion_capacity_pressure_for_three_of_five_slots() -> None:
    payload = {
        "potion_slots": 5,
        "potions": ["fire_potion", "block_potion", "swift_potion"],
    }

    slots = potion_slot_summary(payload)
    summary = trigger_visibility_summary(payload)

    assert slots["capacity"] == 5
    assert slots["filled"] == 3
    assert slots["empty"] == 2
    assert len(slots["potions"]) == 5
    assert summary["potion_capacity_pressure"] == 0.6


def test_potion_capacity_uses_default_ascension_and_relic_slots() -> None:
    reduced_payload = {
        "ascension": 11,
        "potions": ["fire_potion"],
    }
    belt_payload = {
        "ascension": 11,
        "relics": ["potion_belt"],
        "potions": ["fire_potion", "block_potion", "swift_potion"],
    }

    reduced_slots = potion_slot_summary(reduced_payload)
    belt_slots = potion_slot_summary(belt_payload)

    assert reduced_slots["capacity"] == 2
    assert reduced_slots["empty"] == 1
    assert belt_slots["capacity"] == 4
    assert belt_slots["empty"] == 1


def test_relic_and_status_keyword_matching_exposes_modifiers() -> None:
    payload = {
        "relics": [
            "membership_card",
            "question_card",
            "potion_belt",
            {"id": "opening_bell", "description": "At the start of each combat, trigger."},
        ],
        "player": {
            "statuses": {
                "next_turn_energy": 1,
                "reward_card_choice_bonus": 1,
                "shop_discount_status": 1,
            }
        },
        "reward": {"modifiers": [{"id": "reward_extra_card_group"}]},
    }

    summary = trigger_visibility_summary(payload)

    assert summary["shop_modifier_count"] >= 1.0
    assert summary["reward_modifier_count"] >= 1.0
    assert summary["potion_modifier_count"] >= 1.0
    assert summary["start_combat_count"] >= 1.0
    assert summary["start_turn_count"] >= 1.0
    assert summary["next_shop_modifier"] >= 1.0
    assert summary["next_reward_modifier"] >= 1.0
    assert summary["next_combat_modifier"] >= 1.0


def test_vector_length_matches_keys_and_missing_payload_is_safe() -> None:
    summary = trigger_visibility_summary({})
    vector = trigger_visibility_vector(summary)

    assert list(summary) == list(TRIGGER_VISIBILITY_KEYS)
    assert len(vector) == len(TRIGGER_VISIBILITY_KEYS)
    assert all(isinstance(value, float) for value in vector)
    assert vector == tuple(summary[key] for key in TRIGGER_VISIBILITY_KEYS)
