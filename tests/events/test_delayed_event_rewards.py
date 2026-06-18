from __future__ import annotations

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    MapEdgeState,
    MapNodeState,
    MapState,
    RoomKind,
    RunPhase,
)

REWARD_CARDS = (
    {
        "id": "COMMON_ATTACK",
        "name": "Common Attack",
        "rarity": "Common",
        "color": "test",
        "type": "Attack",
        "target": "AnyEnemy",
        "damage": 6,
    },
)
REWARD_RELICS = (
    {"id": "ANCHOR", "name": "Anchor", "rarity_key": "Common", "pool": "shared"},
    {"id": "KUNAI", "name": "Kunai", "rarity_key": "Uncommon", "pool": "shared"},
    {"id": "SHOVEL", "name": "Shovel", "rarity_key": "Rare", "pool": "shared"},
    {
        "id": "BAG_OF_PREPARATION",
        "name": "Bag of Preparation",
        "rarity_key": "Common",
        "pool": "shared",
    },
)


def _choose_first_ancient(state):
    action = next(action for action in legal_actions(state) if action.type == "choose_ancient")
    return step(state, action)


def _action(state, action_type: str, target_id: str | None = None):
    return next(
        action
        for action in legal_actions(state)
        if action.type == action_type
        and (target_id is None or action.target_id == target_id)
    )


def _choose_event_action(state, option_id: str):
    return next(
        action
        for action in legal_actions(state)
        if action.type == "choose_event" and action.target_id == option_id
    )


def _force_next_room(state, room_kind: RoomKind):
    start = MapNodeState(node_id="start", act=state.act, floor=0, lane=0, kind=RoomKind.START)
    target = MapNodeState(node_id="target", act=state.act, floor=1, lane=0, kind=room_kind)
    game_map = MapState(
        act=state.act,
        nodes=(start, target),
        edges=(MapEdgeState(from_id=start.node_id, to_id=target.node_id),),
        current_node_id=start.node_id,
        completed_node_ids=(start.node_id,),
    )
    return state.model_copy(update={"phase": RunPhase.MAP, "map": game_map, "floor": 0})


def _win_forced_combat(state):
    state = _force_next_room(state, RoomKind.MONSTER)
    state = step(state, _action(state, "choose_node", "target"))
    return step(state, _action(state, "play_card"))


def test_wongo_mystery_box_pays_three_relics_after_five_combats() -> None:
    state = new_run(
        seed=1300,
        character_id="TEST",
        ascension=0,
        source_data={
            "event_id": "WELCOME_TO_WONGOS",
            "cards": REWARD_CARDS,
            "relic_pool": REWARD_RELICS,
            "potion_pool": ("fire_potion",),
            "combat_reward_card_count": 0,
            "combat_reward_gold": 0,
            "combat_reward_potion_chance_percent": 0,
            "deck": [
                {
                    "card_id": "debug_kill",
                    "name": "Debug Kill",
                    "type": "attack",
                    "cost": 0,
                    "target": "enemy",
                    "effects": {"damage": 999},
                }
            ],
            "player": {"hp": 80, "max_hp": 80, "gold": 300, "energy": 3, "max_energy": 3},
        },
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = step(state, _action(state, "choose_node", "target"))

    state = step(state, _choose_event_action(state, "MYSTERY_BOX"))

    assert state.player.gold == 0
    assert state.flags["delayed_event_rewards"][0]["reward_kind"] == "random_relic"
    assert state.flags["delayed_event_rewards"][0]["count"] == 3
    assert state.flags["delayed_event_rewards"][0]["remaining_combats"] == 5
    assert any(
        event.kind == "event_delayed_reward_scheduled" for event in state.replay_log[-1].events
    )

    state = step(state, _action(state, "proceed"))
    for expected_remaining in (4, 3, 2, 1):
        state = _win_forced_combat(state)
        assert state.phase == RunPhase.REWARD
        assert state.reward is not None
        assert state.reward.relic_ids == ()
        assert state.flags["delayed_event_rewards"][0]["remaining_combats"] == expected_remaining
        state = step(state, _action(state, "proceed"))

    state = _win_forced_combat(state)

    assert state.phase == RunPhase.REWARD
    assert state.reward is not None
    assert len(state.reward.relic_ids) == 3
    assert "delayed_event_rewards" not in state.flags
    assert state.reward.metadata["delayed_rewards"][0]["reward_kind"] == "random_relic"
    assert state.reward.metadata["delayed_rewards"][0]["relic_ids"] == state.reward.relic_ids
    assert any(
        event.kind == "delayed_event_reward_ready" for event in state.combat.last_events
    )

    for index, relic_id in enumerate(state.reward.relic_ids):
        state = step(state, _action(state, "take_reward_relic", f"reward:relic:{index}"))
        assert relic_id in state.relics
