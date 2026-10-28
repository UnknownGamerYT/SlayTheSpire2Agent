from __future__ import annotations

import json

import pytest
from combat.test_fishing_rod import _enter_combat, _run, _win

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.agent_api import action_space
from sts2sim.engine import RewardState, RoomKind, RunPhase


@pytest.mark.parametrize("kind", [RoomKind.MONSTER, RoomKind.ELITE, RoomKind.BOSS, RoomKind.EVENT])
def test_pumpkin_candle_spends_a_charge_after_each_kind_of_combat_win(kind: RoomKind) -> None:
    state = _enter_combat(_run(relics=("PUMPKIN_CANDLE",)), kind)
    assert state.player.energy == state.player.max_energy + 1
    state = _win(state)
    assert state.flags["relic_counters"]["pumpkin_candle"] == 4


def test_pumpkin_candle_extinguishes_after_fifth_combat_and_persists() -> None:
    state = _run(relics=("pumpkin_candle",))
    for remaining in range(5, 0, -1):
        state = _enter_combat(state)
        assert state.player.energy == state.player.max_energy + 1
        # Turns do not consume charges, and every turn receives the energy.
        state = step(state, next(a for a in legal_actions(state) if a.type == "end_turn"))
        assert state.player.energy == state.player.max_energy + 1
        state = _win(state)
        assert state.flags["relic_counters"]["pumpkin_candle"] == remaining - 1
    state = load_state(json.loads(json.dumps(serialize(state))))
    state = _enter_combat(state)
    assert state.player.energy == state.player.max_energy
    state = step(state, next(a for a in legal_actions(state) if a.type == "end_turn"))
    assert state.player.energy == state.player.max_energy


@pytest.mark.parametrize("charges", [0, 2, 5, 10])
def test_kindle_adds_five_charges_and_spends_the_rest_site(charges: int) -> None:
    state = _enter_combat(_run(relics=("pumpkin_candle",)))
    state = state.model_copy(
        update={
            "phase": RunPhase.REST,
            "combat": None,
            "flags": {**state.flags, "relic_counters": {"pumpkin_candle": charges}},
        }
    )
    state = step(state, next(a for a in legal_actions(state) if a.type == "kindle"))
    assert state.phase == RunPhase.MAP
    assert state.flags["relic_counters"]["pumpkin_candle"] == charges + 5
    assert not any(a.type == "kindle" for a in legal_actions(state))
    state = _enter_combat(state)
    assert state.player.energy == state.player.max_energy + 1


def test_kindle_requires_pumpkin_candle() -> None:
    state = _run(relics=()).model_copy(update={"phase": RunPhase.REST})
    assert not any(action.type == "kindle" for action in legal_actions(state))


def _silken_tress_reward():
    state = _run(relics=())
    state = state.model_copy(
        update={
            "phase": RunPhase.REWARD,
            "player": state.player.model_copy(update={"gold": 75}),
            "reward": RewardState(
                reward_id="test:tress", source="treasure", relic_id="SILKEN_TRESS"
            ),
            "flags": {
                **state.flags,
                "combat_reward_card_count": 3,
                "card_pool": tuple(
                    {
                        "id": f"REWARD_{index}",
                        "name": f"Reward {index}",
                        "color": "test",
                        "rarity": "Common",
                        "type": "Attack",
                        "target": "AnyEnemy",
                        "cost": 1,
                        "damage": 6,
                    }
                    for index in range(4)
                ),
            },
        }
    )
    state = step(state, next(a for a in legal_actions(state) if a.type == "take_reward_relic"))
    assert state.player.gold == 0
    assert not state.flags.get("silken_tress_reward_used")
    return _win(_enter_combat(state))


def test_silken_tress_enchants_first_presented_group_and_selected_card() -> None:
    state = _silken_tress_reward()
    assert state.flags["silken_tress_reward_used"]
    assert len(state.reward.card_options) == 3
    descriptors = [d for d in action_space(state) if d["type"] == "take_reward_card"]
    assert len(descriptors) == 3
    assert all(d["card"]["enchantments"][0]["keyword"] == "glam" for d in descriptors)
    state = load_state(json.loads(json.dumps(serialize(state))))
    state = step(state, next(a for a in legal_actions(state) if a.type == "take_reward_card"))
    chosen = state.master_deck[-1]
    assert chosen.enchantments[0].keyword == "glam"
    assert chosen.enchantments[0].source_id == "silken_tress"
    assert "enchant:glam" in chosen.tags
    next_reward = _win(_enter_combat(state))
    assert "card_reward_enchantments" not in next_reward.reward.metadata


def test_silken_tress_first_reward_is_consumed_even_when_skipped() -> None:
    state = _silken_tress_reward()
    skip = next(
        a for a in legal_actions(state) if a.type == "skip_reward" and "card" in a.target_id
    )
    state = step(state, skip)
    state = _win(_enter_combat(state))
    assert "card_reward_enchantments" not in state.reward.metadata


def test_silken_tress_does_not_enchant_a_second_group_on_the_same_screen() -> None:
    state = _silken_tress_reward()
    state = state.model_copy(
        update={"reward": state.reward.model_copy(update={"card_option_groups": (("reward_0",),)})}
    )
    state = step(
        state,
        next(a for a in legal_actions(state) if a.target_id == "reward:card_group:0:0"),
    )
    assert not state.master_deck[-1].enchantments
