from __future__ import annotations

from sts2sim.live_agent import (
    choose_live_action,
    compare_live_state_to_simulator,
    live_character_id,
    live_state_summary,
)


def test_live_character_id_maps_display_names() -> None:
    assert live_character_id("The Ironclad") == "IRONCLAD"
    assert live_character_id("The Necrobinder") == "NECROBINDER"
    assert live_character_id("REGENT") == "REGENT"


def test_choose_live_action_chooses_first_map_node() -> None:
    state = {
        "state_type": "map",
        "map": {"next_options": [{"index": 2, "type": "Monster"}]},
    }

    decision = choose_live_action(state)

    assert decision.action == {"action": "choose_map_node", "index": 2}
    assert decision.reason == "choose_first_map_node"


def test_choose_live_action_prefers_playable_attack_with_target() -> None:
    state = {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [{"entity_id": "JAW_WORM_0", "hp": 44}],
        },
        "player": {
            "hand": [
                {
                    "index": 0,
                    "id": "DEFEND_R",
                    "type": "Skill",
                    "target_type": "Self",
                    "can_play": True,
                },
                {
                    "index": 1,
                    "id": "STRIKE_R",
                    "type": "Attack",
                    "target_type": "AnyEnemy",
                    "can_play": True,
                },
            ]
        },
    }

    decision = choose_live_action(state)

    assert decision.action == {
        "action": "play_card",
        "card_index": 1,
        "target": "JAW_WORM_0",
    }


def test_choose_live_action_ends_turn_without_playable_cards() -> None:
    state = {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [{"entity_id": "JAW_WORM_0", "hp": 44}],
        },
        "player": {
            "hand": [
                {
                    "index": 0,
                    "id": "WOUND",
                    "type": "Status",
                    "target_type": "None",
                    "can_play": False,
                }
            ]
        },
    }

    decision = choose_live_action(state)

    assert decision.action == {"action": "end_turn"}
    assert decision.reason == "no_playable_cards"


def test_rewards_claim_non_card_rewards_before_leaving() -> None:
    state = {
        "state_type": "rewards",
        "rewards": {
            "items": [
                {"index": 0, "type": "card"},
                {"index": 1, "type": "gold", "gold_amount": 17},
            ],
            "can_proceed": True,
        },
        "player": {"potions": [], "max_potion_slots": 3},
    }

    decision = choose_live_action(state)

    assert decision.action == {"action": "claim_reward", "index": 1}
    assert decision.reason == "claim_gold_reward"


def test_card_reward_skips_optional_card_pick() -> None:
    state = {
        "state_type": "card_reward",
        "card_reward": {"cards": [{"index": 0, "id": "STRIKE_R"}], "can_skip": True},
    }

    decision = choose_live_action(state)

    assert decision.action == {"action": "skip_card_reward"}


def test_live_state_summary_uses_top_level_run_and_player() -> None:
    state = {
        "state_type": "map",
        "run": {"act": 1, "floor": 0, "ascension": 0},
        "player": {
            "character": "The Ironclad",
            "hp": 80,
            "max_hp": 80,
            "gold": 99,
            "relics": [{"id": "BURNING_BLOOD"}],
            "potions": [],
            "max_potion_slots": 3,
        },
    }

    summary = live_state_summary(state)

    assert summary["phase"] == "map"
    assert summary["character_id"] == "IRONCLAD"
    assert summary["player"]["gold"] == 99
    assert summary["relics"] == ["BURNING_BLOOD"]


def test_compare_live_state_to_simulator_reports_baseline_gaps() -> None:
    state = {
        "state_type": "map",
        "run": {"act": 1, "floor": 0, "ascension": 0},
        "player": {
            "character": "The Ironclad",
            "hp": 80,
            "max_hp": 80,
            "gold": 99,
            "relics": [{"id": "BURNING_BLOOD"}],
            "potions": [],
        },
    }

    report = compare_live_state_to_simulator(state, simulator_seed=0)

    assert report.supported is True
    assert report.character_id == "IRONCLAD"
    assert report.mismatch_count > 0
    assert {mismatch.path for mismatch in report.mismatches} >= {"phase", "player.gold"}
