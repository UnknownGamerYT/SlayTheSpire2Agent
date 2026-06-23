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
        "upgrade": {"damage": "+3"},
    },
)

REWARD_RELICS = (
    {"id": "COMMON_RELIC", "name": "Common Relic", "rarity_key": "Common", "pool": "shared"},
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


def _play_debug_kill_action(state):
    assert state.combat is not None
    debug_ids = {
        card.instance_id for card in state.combat.hand if card.card_id == "debug_kill"
    }
    return next(
        action
        for action in legal_actions(state)
        if action.type == "play_card" and action.card_instance_id in debug_ids
    )


def _force_next_room(state, room_kind: RoomKind):
    start = MapNodeState(node_id="start", act=1, floor=0, lane=0, kind=RoomKind.START)
    target = MapNodeState(node_id="target", act=1, floor=1, lane=0, kind=room_kind)
    game_map = MapState(
        act=1,
        nodes=(start, target),
        edges=(MapEdgeState(from_id=start.node_id, to_id=target.node_id),),
        current_node_id=start.node_id,
        completed_node_ids=(start.node_id,),
    )
    return state.model_copy(update={"phase": RunPhase.MAP, "map": game_map, "floor": 0})


def _reward_after_kill(
    *,
    seed: int,
    relics: tuple[str, ...],
    extra_source: dict | None = None,
):
    source_data = {
        "max_acts": 1,
        "map_floors": 4,
        "map_width": 1,
        "cards": REWARD_CARDS,
        "relic_pool": REWARD_RELICS,
        "potion_pool": ("fire_potion",),
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
        "player": {"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3},
    }
    if extra_source:
        source_data.update(extra_source)

    state = new_run(seed=seed, character_id="TEST", ascension=0, source_data=source_data)
    state = _choose_first_ancient(state)
    state = state.model_copy(update={"relics": relics})
    state = _force_next_room(state, RoomKind.MONSTER)
    state = step(state, _action(state, "choose_node", "target"))
    return step(state, _play_debug_kill_action(state))


def test_lava_lamp_upgrades_combat_card_reward_after_no_damage_fight() -> None:
    state = _reward_after_kill(
        seed=9401,
        relics=("lava_lamp",),
        extra_source={
            "combat_reward_card_options": ("COMMON_ATTACK",),
            "combat_reward_potion_chance_percent": 0,
        },
    )

    assert state.reward is not None
    state = step(state, _action(state, "take_reward_card", "reward:card:0"))

    gained = state.master_deck[-1]
    assert gained.card_id.lower() == "common_attack"
    assert gained.upgraded is True
    assert any(
        event.kind == "relic_card_upgraded" and event.source_id == "lava_lamp"
        for event in state.replay_log[-1].events
    )
