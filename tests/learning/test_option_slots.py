from __future__ import annotations

from sts2sim.mechanics.option_slots import (
    OPTION_SLOT_KEYS,
    option_slot_vector,
    option_slots_vector,
    reward_option_slots,
    shop_option_slots,
)


def test_reward_card_options_are_separate_slots() -> None:
    payload = {
        "player": {"gold": 120, "hp": 50, "max_hp": 80},
        "master_deck": [{"card_id": "strike", "type": "attack", "effects": {"damage": 6}}],
        "reward": {
            "reward_id": "r1",
            "source": "combat",
            "card_options": ["pommel_strike", "shrug_it_off"],
        },
    }

    slots = reward_option_slots(payload)
    cards = [slot for slot in slots if slot["kind"] == "card"]

    assert [slot["content_id"] for slot in cards] == ["pommel_strike", "shrug_it_off"]
    assert cards[0]["selection_set_size"] == 2
    assert cards[0]["values"]["card_gain"] == 1.0
    assert cards[0]["values"] != cards[1]["values"]
    assert len(option_slot_vector(cards[0])) == len(OPTION_SLOT_KEYS)


def test_reward_slots_include_mixed_items_and_skip_proceed() -> None:
    payload = {
        "player": {"gold": 80},
        "reward": {
            "reward_id": "r2",
            "source": "event",
            "gold": 25,
            "relic_ids": ["anchor"],
            "potion_ids": ["fire_potion"],
        },
    }

    slots = reward_option_slots(payload)
    kinds = {slot["kind"] for slot in slots}

    assert {"gold", "relic", "potion", "skip", "proceed"} <= kinds
    assert any(slot["leaves_other_choices_open"] for slot in slots if slot["kind"] == "relic")
    assert any(slot["values"]["gold_gain"] == 25.0 for slot in slots)


def test_optional_card_removal_declines_with_proceed_not_skip_slot() -> None:
    payload = {
        "master_deck": [
            {"instance_id": "strike-1", "card_id": "strike"},
            {"instance_id": "defend-1", "card_id": "defend"},
        ],
        "reward": {
            "reward_id": "optional-removal",
            "source": "combat",
            "metadata": {
                "optional_remove_card_count": 1,
                "optional_remove_card_instance_ids": ("strike-1", "defend-1"),
                "optional_remove_card_ids": ("strike", "defend"),
            },
        },
    }

    slots = reward_option_slots(payload)

    assert [slot["kind"] for slot in slots if slot["kind"] == "card_removal"]
    assert not any(
        slot["kind"] == "skip" and slot["content_id"] == "card_removal"
        for slot in slots
    )
    assert any(slot["kind"] == "proceed" and slot["skip_action"] for slot in slots)


def test_shop_slots_include_price_affordability_and_pressure() -> None:
    payload = {
        "player": {"gold": 100},
        "master_deck": [
            {"card_id": "strike", "type": "attack", "effects": {"damage": 6}},
            {"card_id": "defend", "type": "skill", "effects": {"block": 5}},
        ],
        "shop": {
            "items": [
                {
                    "slot_id": "shop:0",
                    "item_id": "anchor",
                    "kind": "relic",
                    "rarity": "common",
                    "price": 175,
                },
                {
                    "slot_id": "shop:1",
                    "item_id": "pommel_strike",
                    "kind": "card",
                    "rarity": "common",
                    "price": 50,
                },
                {
                    "slot_id": "shop:2",
                    "item_id": "fire_potion",
                    "kind": "potion",
                    "rarity": "common",
                    "price": 50,
                },
                {
                    "slot_id": "shop:remove",
                    "item_id": "card_removal",
                    "kind": "card_removal",
                    "price": 75,
                },
            ],
        },
    }

    slots = shop_option_slots(payload)
    by_id = {slot["content_id"]: slot for slot in slots}

    assert by_id["anchor"]["affordable"] is False
    assert by_id["anchor"]["values"]["gold_pressure"] > 0
    assert by_id["pommel_strike"]["affordable"] is True
    assert by_id["fire_potion"]["values"]["potion_gain"] == 1.0
    assert by_id["card_removal"]["values"]["card_remove"] == 1.0


def test_option_slots_vector_is_fixed_and_padded() -> None:
    payload = {"reward": {"reward_id": "empty", "source": "combat"}}
    vector = option_slots_vector(payload, reward_limit=3, shop_limit=2)

    assert len(vector) == (3 + 2) * len(OPTION_SLOT_KEYS)
    assert vector == option_slots_vector(payload, reward_limit=3, shop_limit=2)
