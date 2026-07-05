from __future__ import annotations

import pytest

from sts2sim.gymnasium_env import Sts2Env
from sts2sim.learning.rewards import (
    DEFAULT_REWARD_CONFIG,
    LearningRewardTracker,
    aggression_weights,
    learning_reward,
    learning_reward_breakdown,
)


def test_reward_defaults_remove_sparse_shortcuts() -> None:
    config = DEFAULT_REWARD_CONFIG

    assert config.step_penalty == 0.0
    assert config.death_penalty == 0.0
    assert config.win_reward == 0.0


@pytest.mark.parametrize(
    ("pressure", "hp_weight", "damage_weight", "prevented_weight"),
    [
        (0.0, 0.09, 0.01, 0.05),
        (0.5, 0.05, 0.05, 0.0325),
        (1.0, 0.01, 0.09, 0.015),
    ],
)
def test_aggression_weights_interpolate_exactly(
    pressure: float,
    hp_weight: float,
    damage_weight: float,
    prevented_weight: float,
) -> None:
    weights = aggression_weights(pressure)

    assert weights["hp_loss_weight"] == hp_weight
    assert weights["enemy_damage_weight"] == damage_weight
    assert weights["prevented_hp_weight"] == prevented_weight


def test_no_direct_death_or_complete_reward() -> None:
    previous = _base_payload(phase="map", hp=40)
    failed = _base_payload(phase="failed", hp=40)
    complete = _base_payload(phase="complete", hp=40)

    assert learning_reward_breakdown(previous, failed).total == 0.0
    assert learning_reward_breakdown(previous, complete).total == 0.0


def test_gold_reward_is_positive_and_capped() -> None:
    previous = _base_payload(gold=0)
    current = _base_payload(gold=999)

    breakdown = learning_reward_breakdown(previous, current)

    assert breakdown.gold_reward == 0.5
    assert breakdown.total == 0.5


def test_cards_relics_removals_and_skips_have_no_direct_reward() -> None:
    previous = _base_payload()
    current = _base_payload()
    current["reward"] = {
        "reward_id": "test",
        "source": "combat",
        "relic_id": "some_relic",
        "relic_claimed": True,
        "card_options": ["strike", "defend", "bash"],
        "card_claimed": True,
        "claimed_card_indices": [0],
        "skipped_card_indices": [1],
    }

    assert learning_reward_breakdown(previous, current).total == 0.0


def test_node_progress_reward_is_tiny_and_once_per_tracker() -> None:
    tracker = LearningRewardTracker()
    previous = _base_payload(room_history=("a1:0:0",))
    current = _base_payload(room_history=("a1:0:0", "a1:1:0"))

    first = learning_reward_breakdown(previous, current, tracker=tracker)
    second = learning_reward_breakdown(previous, current, tracker=tracker)

    assert first.node_progress_reward == 0.1
    assert second.node_progress_reward == 0.0


@pytest.mark.parametrize(
    ("kind", "act", "combat_reward", "boss_reward"),
    [
        ("monster", 1, 2.0, 0.0),
        ("elite", 1, 10.0, 0.0),
        ("boss", 1, 0.0, 50.0),
        ("boss", 2, 0.0, 150.0),
        ("boss", 3, 0.0, 400.0),
    ],
)
def test_combat_win_rewards_scale_by_room_and_act(
    kind: str,
    act: int,
    combat_reward: float,
    boss_reward: float,
) -> None:
    previous = _combat_payload(kind=kind, act=act, turn=10)
    current = _base_payload(phase="reward", act=act, floor=1, kind=kind)

    breakdown = learning_reward_breakdown(previous, current)

    assert breakdown.combat_win_reward == combat_reward
    assert breakdown.boss_reward == boss_reward
    assert breakdown.combat_pace_reward == 0.0


def test_fast_combat_bonus_only_applies_before_turn_ten() -> None:
    previous = _combat_payload(kind="elite", turn=2)
    current = _base_payload(phase="reward", kind="elite")

    breakdown = learning_reward_breakdown(previous, current)

    assert breakdown.combat_pace_reward == pytest.approx(2.4)


def test_enemy_hp_progress_cannot_be_refarmed_after_healing() -> None:
    tracker = LearningRewardTracker(max_enemy_hp_progress={"0:test_monster": 20})
    previous = _combat_payload(monster_hp=80)
    current = _combat_payload(monster_hp=70)

    first = learning_reward_breakdown(previous, current, tracker=tracker)
    healed = _combat_payload(monster_hp=90)
    second = learning_reward_breakdown(current, healed, tracker=tracker)
    redamaged = _combat_payload(monster_hp=70)
    third = learning_reward_breakdown(healed, redamaged, tracker=tracker)

    assert first.enemy_hp_progress_reward > 0
    assert second.enemy_hp_progress_reward == 0.0
    assert third.enemy_hp_progress_reward == 0.0


def test_prevented_hp_reward_requires_projected_hp_saved() -> None:
    previous = _combat_payload(player_block=0, intent_damage=12)
    current = _combat_payload(player_block=12, intent_damage=12)

    blocked = learning_reward_breakdown(
        previous,
        current,
        action_descriptor={"preview": {"projected_damage_taken_after_end": 0}},
    )
    unnecessary = learning_reward_breakdown(
        _combat_payload(player_block=20, intent_damage=12),
        _combat_payload(player_block=25, intent_damage=12),
        action_descriptor={"preview": {"projected_damage_taken_after_end": 0}},
    )

    assert blocked.prevented_hp_reward > 0
    assert unnecessary.prevented_hp_reward == 0.0


def test_wasteful_potion_discard_is_contextual() -> None:
    previous = _base_payload(potions=("fire_potion",))

    waste = learning_reward_breakdown(
        previous,
        _base_payload(potions=()),
        action_descriptor={"type": "discard_potion"},
    )
    slot_pressure = learning_reward_breakdown(
        previous,
        _base_payload(potions=()),
        action_descriptor={
            "type": "discard_potion",
            "potion_strategy": {"belt_full": True, "slot_pressure": 1.0},
        },
    )

    assert waste.potion_waste_penalty == -0.5
    assert slot_pressure.potion_waste_penalty == 0.0


def test_env_info_exposes_reward_breakdown() -> None:
    env = Sts2Env(seed=1, character_id="TEST", reward_fn=learning_reward)
    try:
        _observation, info = env.reset()
        action_id = info["action_space"][0]["id"]
        _observation, reward, _terminated, _truncated, info = env.step(action_id)
    finally:
        env.close()

    assert info["reward_breakdown"]["total"] == reward
    assert "aggression_pressure" in info["reward_breakdown"]
    assert "reward_aggression_pressure" in info["agent_memory"]["entries"][0]


def _base_payload(
    *,
    phase: str = "map",
    act: int = 1,
    floor: int = 1,
    hp: int = 50,
    max_hp: int = 80,
    gold: int = 0,
    kind: str = "monster",
    room_history: tuple[str, ...] = (),
    potions: tuple[str, ...] = (),
) -> dict:
    return {
        "phase": phase,
        "act": act,
        "floor": floor,
        "player": {"hp": hp, "max_hp": max_hp, "block": 0, "gold": gold},
        "potions": list(potions),
        "room_history": list(room_history),
        "map": {
            "current_node_id": "a1:1:0",
            "completed_node_ids": [],
            "nodes": [
                {
                    "node_id": "a1:1:0",
                    "act": act,
                    "floor": floor,
                    "lane": 0,
                    "kind": kind,
                }
            ],
        },
    }


def _combat_payload(
    *,
    kind: str = "monster",
    act: int = 1,
    turn: int = 1,
    hp: int = 50,
    max_hp: int = 80,
    player_block: int = 0,
    monster_hp: int = 80,
    intent_damage: int = 8,
) -> dict:
    payload = _base_payload(phase="combat", act=act, floor=1, hp=hp, max_hp=max_hp, kind=kind)
    payload["combat"] = {
        "turn": turn,
        "player": {"hp": hp, "max_hp": max_hp, "block": player_block, "gold": 0},
        "monsters": [
            {
                "monster_id": "test_monster",
                "name": "Test Monster",
                "hp": monster_hp,
                "max_hp": 100,
                "intent": "attack",
                "intent_damage": intent_damage,
                "hit_count": 1,
                "statuses": {},
                "metadata": {},
            }
        ],
        "metadata": {"room_kind": kind, "combat_id": "test-combat"},
    }
    return payload
