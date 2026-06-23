from __future__ import annotations

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    MapEdgeState,
    MapNodeState,
    MapState,
    RoomKind,
    RunPhase,
)

CARD_POOL = (
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
    {
        "id": "COMMON_SKILL",
        "name": "Common Skill",
        "rarity": "Common",
        "color": "test",
        "type": "Skill",
        "target": "Self",
        "block": 5,
        "upgrade": {"block": "+3"},
    },
)

RELIC_POOL = (
    {"id": "COMMON_RELIC", "name": "Common Relic", "rarity_key": "Common", "pool": "shared"},
    {
        "id": "UNCOMMON_RELIC",
        "name": "Uncommon Relic",
        "rarity_key": "Uncommon",
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
        if action.type == action_type and (target_id is None or action.target_id == target_id)
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


def _source_data(deck: tuple[dict, ...] | None = None) -> dict:
    return {
        "max_acts": 1,
        "map_floors": 4,
        "map_width": 1,
        "cards": CARD_POOL,
        "relic_pool": RELIC_POOL,
        "potion_pool": ("fire_potion", "skill_potion"),
        "deck": deck
        or (
            {
                "card_id": "debug_kill",
                "name": "Debug Kill",
                "type": "attack",
                "cost": 0,
                "target": "enemy",
                "effects": {"damage": 999},
            },
        ),
        "player": {"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3},
    }


def _run_at_room(
    *,
    seed: int,
    relics: tuple[str, ...],
    room_kind: RoomKind = RoomKind.MONSTER,
    deck: tuple[dict, ...] | None = None,
    potions: tuple[str, ...] = (),
    flags: dict | None = None,
):
    state = new_run(seed=seed, character_id="TEST", ascension=0, source_data=_source_data(deck))
    state = _choose_first_ancient(state)
    state = state.model_copy(
        update={
            "relics": relics,
            "potions": potions,
            "flags": {**state.flags, **dict(flags or {})},
        }
    )
    return _force_next_room(state, room_kind)


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


def _kill_current_combat(state):
    return step(state, _play_debug_kill_action(state))


def test_delicate_frond_fills_empty_potion_slots_at_combat_start() -> None:
    state = _run_at_room(seed=9501, relics=("delicate_frond",), potions=("blood_potion",))

    state = step(state, _action(state, "choose_node", "target"))

    assert len(state.potions) == 3
    assert state.potions[0] == "blood_potion"
    assert sum(event.kind == "relic_potion_obtained" for event in state.replay_log[-1].events) == 2


def test_petrified_toad_adds_potion_shaped_rock_at_combat_start() -> None:
    state = _run_at_room(seed=9502, relics=("petrified_toad",))

    state = step(state, _action(state, "choose_node", "target"))

    assert state.potions == ("potion_shaped_rock",)
    assert any(
        event.kind == "relic_potion_obtained" and event.target_id == "potion_shaped_rock"
        for event in state.replay_log[-1].events
    )


def test_gambling_chip_creates_optional_opening_discard_redraw_choice() -> None:
    deck = tuple(
        {
            "card_id": f"debug_card_{index}",
            "name": f"Debug Card {index}",
            "type": "skill",
            "cost": 0,
            "target": "self",
            "effects": {"block": 1},
        }
        for index in range(7)
    )
    state = _run_at_room(seed=9503, relics=("gambling_chip",), deck=deck)
    state = step(state, _action(state, "choose_node", "target"))

    assert state.combat is not None
    assert len(state.combat.hand) == 5
    assert state.combat.pending_choices
    choice = state.combat.pending_choices[0]
    assert choice.kind == "discard"
    assert choice.min_choices == 0

    discard_action = next(
        action for action in legal_actions(state) if action.type == "discard_card"
    )
    discarded_id = discard_action.card_instance_id
    state = step(state, discard_action)
    state = step(state, _action(state, "proceed"))

    assert state.combat is not None
    assert len(state.combat.hand) == 5
    assert any(card.instance_id == discarded_id for card in state.combat.discard_pile)
    assert any(event.kind == "card_choice_completed" for event in state.replay_log[-1].events)


def test_war_hammer_upgrades_four_deck_cards_after_elite_win() -> None:
    deck = (
        {
            "card_id": "debug_kill",
            "name": "Debug Kill",
            "type": "attack",
            "cost": 0,
            "target": "enemy",
            "effects": {"damage": 999},
        },
    ) + tuple(
        {
            "card_id": f"upgrade_target_{index}",
            "name": f"Upgrade Target {index}",
            "type": "skill",
            "cost": 1,
            "target": "self",
            "effects": {"block": 4},
            "upgrade": {"block": "+2"},
        }
        for index in range(5)
    )
    state = _run_at_room(seed=9504, relics=("war_hammer",), room_kind=RoomKind.ELITE, deck=deck)
    state = step(state, _action(state, "choose_node", "target"))

    state = _kill_current_combat(state)

    assert state.phase == RunPhase.REWARD
    assert sum(card.upgraded for card in state.master_deck) == 4
    assert sum(event.kind == "relic_deck_card_upgraded" for event in state.combat.last_events) == 4


def test_paels_tooth_adds_upgraded_card_removed_by_relic_after_combat() -> None:
    state = _run_at_room(
        seed=9505,
        relics=("paels_tooth",),
        flags={"relic_removed_card_ids": ("COMMON_ATTACK",)},
    )
    state = step(state, _action(state, "choose_node", "target"))

    state = _kill_current_combat(state)

    gained = state.master_deck[-1]
    assert gained.card_id == "common_attack"
    assert gained.upgraded is True
    assert any(
        event.kind == "relic_deck_card_added" and event.source_id == "paels_tooth"
        for event in state.combat.last_events
    )
