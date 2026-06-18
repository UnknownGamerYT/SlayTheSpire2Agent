from __future__ import annotations

from random import Random

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    MapEdgeState,
    MapNodeState,
    MapState,
    RewardState,
    RoomKind,
    RunPhase,
)
from sts2sim.mechanics import (
    TreasureContext,
    build_treasure_relic_pool,
    draw_treasure_reward,
)

TREASURE_POOL = (
    {"id": "ANCHOR", "name": "Anchor", "rarity_key": "Common", "pool": "shared"},
    {"id": "AKABEKO", "name": "Akabeko", "rarity_key": "Uncommon", "pool": "shared"},
    {"id": "OLD_COIN", "name": "Old Coin", "rarity_key": "Rare", "pool": "shared"},
)


def _treasure_state(*, seed: int = 1, ascension: int = 0, relics=()):
    state = new_run(
        seed=seed,
        character_id="TEST",
        ascension=ascension,
        source_data={
            "treasure_relic_pool": TREASURE_POOL,
            "map_floors": 4,
            "map_width": 1,
        },
    )
    start = MapNodeState(
        node_id="start",
        act=1,
        floor=0,
        lane=0,
        kind=RoomKind.START,
    )
    treasure = MapNodeState(
        node_id="treasure",
        act=1,
        floor=1,
        lane=0,
        kind=RoomKind.TREASURE,
    )
    map_state = MapState(
        act=1,
        nodes=(start, treasure),
        edges=(MapEdgeState(from_id=start.node_id, to_id=treasure.node_id),),
        current_node_id=start.node_id,
        completed_node_ids=(start.node_id,),
        boss_node_id=None,
    )
    return state.model_copy(
        update={
            "phase": RunPhase.MAP,
            "map": map_state,
            "relics": tuple(relics),
            "ancient": None,
            "room_history": (),
            "replay_log": (),
        }
    )


def _action(state, action_type: str, target_id: str | None = None):
    return next(
        action
        for action in legal_actions(state)
        if action.type == action_type
        and (target_id is None or action.target_id == target_id)
    )


def _enter_treasure(state):
    return step(state, _action(state, "choose_node", "treasure"))


def test_treasure_node_offers_gold_and_source_backed_relic() -> None:
    state = _enter_treasure(_treasure_state(seed=10))

    assert state.phase == RunPhase.TREASURE
    assert state.reward is not None
    assert 42 <= state.reward.gold <= 52
    assert state.reward.relic_id in {"anchor", "akabeko", "old_coin"}
    assert state.reward.metadata["relic_rarity"] in {"common", "uncommon", "rare"}
    assert state.flags["treasure_chests_opened"] == 1
    assert any(action.type == "take_reward_gold" for action in legal_actions(state))
    assert any(action.type == "take_reward_relic" for action in legal_actions(state))
    assert any(action.type == "proceed" for action in legal_actions(state))


def test_treasure_reward_can_be_taken_or_skipped() -> None:
    state = _enter_treasure(_treasure_state(seed=11))
    assert state.reward is not None
    gold = state.reward.gold
    relic_id = state.reward.relic_id

    state = step(state, _action(state, "take_reward_gold", "reward:gold"))
    state = step(state, _action(state, "take_reward_relic", "reward:relic"))

    assert state.player.gold == gold
    assert relic_id in state.relics
    assert state.reward is not None
    assert state.reward.gold_claimed is True
    assert state.reward.relic_claimed is True

    state = step(state, _action(state, "proceed"))

    assert state.phase == RunPhase.MAP
    assert state.reward is None


def test_treasure_poverty_ascension_reduces_chest_gold() -> None:
    state = _enter_treasure(_treasure_state(seed=12, ascension=3))

    assert state.reward is not None
    assert state.reward.gold == int(state.reward.metadata["base_gold"] * 0.75)


def test_silver_crucible_makes_first_treasure_chest_empty() -> None:
    state = _enter_treasure(_treasure_state(seed=13, relics=("silver_crucible",)))

    assert state.reward is not None
    assert state.reward.gold == 0
    assert state.reward.relic_id is None
    assert state.reward.metadata["empty"] is True
    assert state.reward.metadata["empty_reason"] == "first_chest_empty_relic"
    assert state.flags["treasure_chests_opened"] == 1
    assert not any(action.type == "take_reward_gold" for action in legal_actions(state))
    assert not any(action.type == "take_reward_relic" for action in legal_actions(state))
    assert any(action.type == "proceed" for action in legal_actions(state))


def test_treasure_relic_rarity_falls_forward_when_pool_is_exhausted() -> None:
    pool = build_treasure_relic_pool(
        (
            {"id": "ANCHOR", "rarity_key": "Common", "pool": "shared"},
            {"id": "AKABEKO", "rarity_key": "Uncommon", "pool": "shared"},
        ),
        character_id="TEST",
    )

    reward = draw_treasure_reward(
        Random(1),
        pool,
        TreasureContext(character_id="TEST", owned_relics=("anchor",)),
    )

    assert reward.relic_id == "akabeko"
    assert reward.relic_rarity is not None
    assert reward.relic_rarity.value == "uncommon"


def test_old_coin_treasure_pickup_grants_gold() -> None:
    state = new_run(seed=14, character_id="TEST", ascension=0)
    state = state.model_copy(
        update={
            "phase": RunPhase.TREASURE,
            "reward": RewardState(
                reward_id="treasure:test",
                source="treasure",
                relic_id="old_coin",
            ),
        }
    )
    before_gold = state.player.gold

    state = step(state, _action(state, "take_reward_relic", "reward:relic"))

    assert "old_coin" in state.relics
    assert state.player.gold == before_gold + 300
    assert [event.kind for event in state.replay_log[-1].events] == [
        "reward_relic_taken",
        "relic_gold_gained",
    ]
