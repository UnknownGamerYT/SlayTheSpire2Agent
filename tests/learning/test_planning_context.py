from __future__ import annotations

from sts2sim.mechanics.planning_context import (
    REWARD_PLAN_KEYS,
    ROUTE_PLAN_KEYS,
    reward_plan_summary,
    reward_plan_vector,
    route_plan_summary,
    route_plan_vector,
)


def test_reward_plan_summary_keeps_mixed_visible_reward_context() -> None:
    payload = {
        "phase": "reward",
        "reward": {
            "reward_id": "reward:mixed",
            "source": "combat",
            "gold": 42,
            "gold_claimed": True,
            "relic_ids": ["anchor", "bag_of_marbles"],
            "claimed_relic_ids": ["anchor"],
            "card_options": ["strike_plus", "defend_plus"],
            "card_option_groups": [
                ["pommel_strike", "shrug_it_off"],
                ["anger", "twin_strike"],
            ],
            "claimed_card_option_group_indices": [0],
            "card_ids": ["ritual_dagger", "bandage_up"],
            "skipped_card_indices": [1],
            "potion_ids": ["fire_potion", "block_potion"],
            "claimed_potion_indices": [0],
        },
    }

    summary = reward_plan_summary(payload)
    by_set = {
        selection_set["selection_set_id"]: selection_set
        for selection_set in summary["selection_sets"]
    }

    assert summary["reward_open"] is True
    assert summary["can_skip"] is True
    assert summary["can_take_multiple_items"] is True
    assert summary["available_counts"] == {
        "total": 7,
        "selection_sets": 5,
        "cards": 2,
        "card_groups": 1,
        "fixed_cards": 1,
        "card_removals": 0,
        "relics": 1,
        "potions": 1,
        "gold": 0,
    }
    assert summary["claimed_counts"]["total"] == 4
    assert summary["claimed_groups"]["card_group_indices"] == [0]
    assert summary["claimed_groups"]["relic_ids"] == ["anchor"]
    assert summary["skipped_counts"]["fixed_cards"] == 1
    assert summary["skipped_groups"]["fixed_card_indices"] == [1]
    assert by_set["card_options"]["content_ids"] == ["strike_plus", "defend_plus"]
    assert by_set["card_group:1"]["exclusive_within_set"] is True
    assert by_set["relic:1"]["content_ids"] == ["bag_of_marbles"]
    assert "bandage_up" not in summary["available_content_ids"]
    assert len(reward_plan_vector(payload)) == len(REWARD_PLAN_KEYS)


def test_reward_plan_summary_exposes_optional_card_removal_choices() -> None:
    payload = {
        "phase": "reward",
        "reward": {
            "reward_id": "reward:remove",
            "source": "combat",
            "metadata": {
                "optional_remove_card_count": 1,
                "optional_remove_card_instance_ids": ["strike-1", "defend-1"],
                "optional_remove_card_ids": ["strike", "defend"],
            },
        },
    }

    summary = reward_plan_summary(payload)

    assert summary["available_counts"]["card_removals"] == 2
    assert summary["available_counts"]["total"] == 2
    assert summary["claimed_counts"]["card_removals"] == 0
    assert summary["skipped_counts"]["card_removals"] == 0
    assert summary["selection_sets"][0]["selection_set_id"] == "card_removal"
    assert summary["selection_sets"][0]["content_ids"] == ["strike", "defend"]
    assert summary["available_choices"][0]["card_instance_id"] == "strike-1"
    assert summary["available_choices"][0]["target_id"] == "reward:remove_card:0"
    assert len(reward_plan_vector(payload)) == len(REWARD_PLAN_KEYS)
    assert reward_plan_vector(payload)[
        REWARD_PLAN_KEYS.index("available_card_removals")
    ] == 2


def test_route_plan_summary_scores_small_map_by_style() -> None:
    payload = {
        "phase": "map",
        "map": {
            "current_node_id": "start",
            "completed_node_ids": ["start"],
            "boss_node_id": "boss",
            "nodes": [
                {"node_id": "start", "act": 1, "floor": 0, "lane": 0, "kind": "start"},
                {"node_id": "elite", "act": 1, "floor": 1, "lane": 0, "kind": "elite"},
                {"node_id": "shop", "act": 1, "floor": 1, "lane": 1, "kind": "shop"},
                {"node_id": "rest", "act": 1, "floor": 1, "lane": 2, "kind": "rest"},
                {
                    "node_id": "monster",
                    "act": 1,
                    "floor": 2,
                    "lane": 0,
                    "kind": "monster",
                },
                {"node_id": "rest2", "act": 1, "floor": 2, "lane": 1, "kind": "rest"},
                {"node_id": "shop2", "act": 1, "floor": 2, "lane": 2, "kind": "shop"},
                {"node_id": "boss", "act": 1, "floor": 3, "lane": 0, "kind": "boss"},
            ],
            "edges": [
                {"from_id": "start", "to_id": "elite"},
                {"from_id": "start", "to_id": "shop"},
                {"from_id": "start", "to_id": "rest"},
                {"from_id": "elite", "to_id": "monster"},
                {"from_id": "shop", "to_id": "shop2"},
                {"from_id": "rest", "to_id": "rest2"},
                {"from_id": "monster", "to_id": "boss"},
                {"from_id": "shop2", "to_id": "boss"},
                {"from_id": "rest2", "to_id": "boss"},
            ],
        },
    }

    summary = route_plan_summary(payload)
    styles = summary["styles"]

    assert summary["route_open"] is True
    assert summary["reachable_next_node_ids"] == ["elite", "rest", "shop"]
    assert summary["reachable_path_count"] == 3
    assert summary["boss_path_count"] == 3
    assert styles["aggressive"]["first_node_id"] == "elite"
    assert styles["elite_heavy"]["first_node_id"] == "elite"
    assert styles["safe"]["first_node_id"] == "rest"
    assert styles["upgrade_heavy"]["first_node_id"] == "rest"
    assert styles["shop_heavy"]["first_node_id"] == "shop"
    assert styles["aggressive"]["score"] > styles["safe"]["score"]
    assert styles["shop_heavy"]["counts"]["shops"] == 2
    assert styles["upgrade_heavy"]["counts"]["rests"] == 2
    assert len(route_plan_vector(payload)) == len(ROUTE_PLAN_KEYS)


def test_route_plan_exposes_pending_map_changing_flags() -> None:
    payload = {
        "act": 1,
        "flags": {
            "golden_compass_act2_map": True,
            "spoils_map_pending_act": 2,
            "spoils_map_target_node_id": "a2:5:1",
        },
        "map": {
            "act": 1,
            "current_node_id": "start",
            "completed_node_ids": ("start",),
            "nodes": [
                {"node_id": "start", "floor": 0, "kind": "start"},
                {"node_id": "monster", "floor": 1, "kind": "monster"},
                {"node_id": "boss", "floor": 2, "kind": "boss"},
            ],
            "edges": [
                {"from_id": "start", "to_id": "monster"},
                {"from_id": "monster", "to_id": "boss"},
            ],
        },
    }

    summary = route_plan_summary(payload)
    effects = summary["map_effects"]
    vector = route_plan_vector(payload)

    assert effects["next_act_map_change_pending"] is True
    assert effects["golden_compass_act2_map"] is True
    assert effects["spoils_map_pending"] is True
    assert effects["spoils_map_target_act"] == 2
    assert effects["spoils_map_target_known"] is True
    assert vector[ROUTE_PLAN_KEYS.index("next_act_map_change_pending")] == 1.0
    assert vector[ROUTE_PLAN_KEYS.index("spoils_map_target_act")] == 2.0
