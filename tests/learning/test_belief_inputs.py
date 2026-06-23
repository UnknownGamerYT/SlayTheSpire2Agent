from __future__ import annotations

from sts2sim.mechanics.belief import (
    BELIEF_OBSERVATION_KEYS,
    belief_summary,
    belief_vector,
)


def test_belief_summary_draw_and_combat_odds_are_deterministic() -> None:
    payload = {
        "phase": "combat",
        "player": {"hp": 30, "max_hp": 70, "block": 3},
        "master_deck": [
            {"card_id": "strike", "type": "attack", "effects": {"damage": 6}},
            {"card_id": "defend", "type": "skill", "effects": {"block": 5}},
            {"card_id": "bash", "type": "attack", "effects": {"damage": 8}},
            {"card_id": "footwork", "type": "power", "effects": {"dexterity": 2}},
        ],
        "combat": {
            "draw_per_turn": 4,
            "player": {"hp": 30, "max_hp": 70, "block": 3},
            "hand": [
                {"card_id": "strike", "type": "attack", "effects": {"damage": 6}},
                {"card_id": "bash", "type": "attack", "effects": {"damage": 8}},
            ],
            "draw_pile": [
                {"card_id": "defend", "type": "skill", "effects": {"block": 5}},
                {"card_id": "strike", "type": "attack", "effects": {"damage": 6}},
            ],
            "discard_pile": [
                {"card_id": "footwork", "type": "power", "effects": {"dexterity": 2}},
                {"card_id": "strike", "type": "attack", "effects": {"damage": 6}},
            ],
            "monsters": [
                {
                    "monster_id": "cultist",
                    "hp": 13,
                    "max_hp": 48,
                    "block": 1,
                    "intent_damage": 8,
                    "hit_count": 1,
                }
            ],
        },
    }

    summary = belief_summary(payload)

    assert summary["draw_attack_chance"] == 0.5
    assert summary["draw_block_chance"] == 0.25
    assert summary["draw_damage_chance"] == 0.5
    assert summary["draw_setup_chance"] == 0.5
    assert summary["deck_cycle_distance"] == 0.5
    assert summary["reshuffle_risk"] == 0.5
    assert summary["visible_lethal_now"] == 1.0
    assert summary["likely_damage_taken_after_end_turn"] == 5
    assert summary["survival_margin"] == 25
    assert summary["turns_to_kill_estimate"] == 0.0


def test_belief_summary_route_and_reward_placeholders() -> None:
    payload = {
        "player": {"hp": 70, "max_hp": 80},
        "master_deck": [
            {"card_id": "defend", "type": "skill", "effects": {"block": 5}},
            {"card_id": "defend_2", "type": "skill", "effects": {"block": 5}},
        ],
        "map": {
            "current_node_id": "start",
            "completed_node_ids": ["start"],
            "nodes": [
                {"node_id": "start", "floor": 0, "lane": 0, "kind": "start"},
                {"node_id": "monster", "floor": 1, "lane": 0, "kind": "monster"},
                {"node_id": "shop", "floor": 1, "lane": 1, "kind": "shop"},
                {"node_id": "elite", "floor": 2, "lane": 0, "kind": "elite"},
                {"node_id": "rest", "floor": 2, "lane": 1, "kind": "rest"},
                {"node_id": "boss", "floor": 3, "lane": 0, "kind": "boss"},
            ],
            "edges": [
                {"from_id": "start", "to_id": "monster"},
                {"from_id": "start", "to_id": "shop"},
                {"from_id": "monster", "to_id": "elite"},
                {"from_id": "shop", "to_id": "rest"},
                {"from_id": "elite", "to_id": "boss"},
                {"from_id": "rest", "to_id": "boss"},
            ],
        },
        "reward": {
            "card_options": ["strike", "pommel_strike", "bash"],
            "relic_id": "anchor",
            "potion_ids": ["fire_potion"],
        },
    }

    summary = belief_summary(payload)

    assert summary["route_expected_fights_before_boss"] == 1.0
    assert summary["route_expected_elites_before_boss"] == 0.5
    assert summary["route_expected_rests_before_boss"] == 0.5
    assert summary["route_expected_shops_before_boss"] == 0.5
    assert summary["route_expected_rewards_before_boss"] == 1.5
    assert summary["reward_visible_card_count"] == 3
    assert summary["reward_visible_relic_count"] == 1
    assert summary["reward_visible_potion_count"] == 1
    assert summary["reward_card_attack_ev"] > summary["reward_card_block_ev"]
    assert summary["reward_relic_ev"] == 0.425
    assert summary["reward_potion_ev"] == 0.325


def test_belief_vector_matches_key_order_and_missing_fields_are_safe() -> None:
    summary = belief_summary({})
    vector = belief_vector(summary)

    assert list(summary) == list(BELIEF_OBSERVATION_KEYS)
    assert len(vector) == len(BELIEF_OBSERVATION_KEYS)
    assert all(isinstance(value, float) for value in vector)
    assert vector == [summary[key] for key in BELIEF_OBSERVATION_KEYS]
