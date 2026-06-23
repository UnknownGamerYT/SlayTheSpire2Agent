from __future__ import annotations

from sts2sim.mechanics.synergy import (
    SYNERGY_VALUE_KEYS,
    action_synergy_profile,
    profile_value_vector,
)


def test_reward_card_improves_block_need() -> None:
    state_payload = {
        "player": {"hp": 50, "max_hp": 80, "gold": 100},
        "master_deck": [
            {"card_id": "strike", "type": "attack", "effects": {"damage": 6}},
            {"card_id": "strike", "type": "attack", "effects": {"damage": 6}},
        ],
        "combat": {
            "player": {"hp": 50, "max_hp": 80, "block": 0},
            "monsters": [{"monster_id": "jaw_worm", "hp": 40, "intent_damage": 12}],
        },
    }
    descriptor = {
        "type": "take_reward_card",
        "card": {
            "card_id": "brace_up",
            "type": "skill",
            "target": "self",
            "effects": {"block": 8},
        },
        "reward_choice": {"kind": "card", "content_id": "brace_up"},
    }

    profile = action_synergy_profile(state_payload, descriptor)

    assert profile["values"]["block"] == 8.0
    assert profile["values"]["improves_current_need"] > 0
    assert profile["deck_context"]["improves_current_need"] > 0
    assert "synergy:block" in profile["tags"]
    assert "context:improves_current_need" in profile["tags"]
    assert len(profile_value_vector(profile)) == len(SYNERGY_VALUE_KEYS)


def test_duplicate_attack_adds_bloat_and_duplicate_signal() -> None:
    state_payload = {
        "player": {"hp": 70, "max_hp": 80, "gold": 100},
        "master_deck": [
            {"card_id": "strike", "type": "attack", "effects": {"damage": 6}},
            {"card_id": "strike", "type": "attack", "effects": {"damage": 6}},
            {"card_id": "defend", "type": "skill", "effects": {"block": 5}},
        ],
        "combat": {
            "player": {"hp": 70, "max_hp": 80, "block": 0},
            "monsters": [{"monster_id": "cultist", "hp": 48, "intent_damage": 0}],
        },
    }
    descriptor = {
        "type": "take_reward_card",
        "card": {
            "card_id": "strike",
            "type": "attack",
            "target": "enemy",
            "effects": {"damage": 6},
        },
        "reward_choice": {"kind": "card", "content_id": "strike"},
    }

    profile = action_synergy_profile(state_payload, descriptor)

    assert profile["values"]["frontload"] == 6.0
    assert profile["values"]["duplicates_existing_engine"] >= 1.0
    assert profile["values"]["adds_bloat"] > 0
    assert profile["deck_context"]["adds_bloat"] > 0
    assert "deck:duplicate_content" in profile["tags"]
    assert "context:duplicates_existing_engine" in profile["tags"]


def test_shop_relic_cost_sets_gold_pressure() -> None:
    state_payload = {
        "player": {"hp": 70, "max_hp": 80, "gold": 100},
        "master_deck": [
            {"card_id": "strike", "type": "attack", "effects": {"damage": 6}},
            {"card_id": "defend", "type": "skill", "effects": {"block": 5}},
        ],
        "relics": (),
        "potions": (),
    }
    descriptor = {
        "type": "shop_buy",
        "item": {
            "item_id": "anchor",
            "kind": "relic",
            "price": 175,
            "base_price": 175,
        },
        "relic": {"relic_id": "anchor"},
    }

    profile = action_synergy_profile(state_payload, descriptor)

    assert profile["values"]["relic_synergy"] >= 1.0
    assert profile["values"]["gold_pressure"] > 1.0
    assert profile["opportunity_cost"]["gold_pressure"] == profile["values"]["gold_pressure"]
    assert "opportunity:gold_pressure" in profile["tags"]
    assert "cost:spends_gold" in profile["tags"]
