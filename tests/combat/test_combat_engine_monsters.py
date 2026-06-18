from __future__ import annotations

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    MapEdgeState,
    MapNodeState,
    MapState,
    RoomKind,
    RunPhase,
)

MONSTERS = (
    {
        "id": "TRAINING_AUTOMATON",
        "name": "Training Automaton",
        "type": "Normal",
        "min_hp": 30,
        "max_hp": 30,
        "min_hp_ascension": 40,
        "max_hp_ascension": 40,
        "moves": (
            {
                "id": "DOUBLE_STRIKE",
                "name": "Double Strike",
                "intent": "Attack",
                "damage": {"normal": 5, "ascension": 7, "hit_count": 2},
                "block": None,
                "heal": None,
                "powers": None,
            },
            {
                "id": "FORTIFY",
                "name": "Fortify",
                "intent": "Defend + Buff",
                "damage": None,
                "block": 4,
                "heal": None,
                "powers": ({"power_id": "STRENGTH", "amount": 2, "target": "self"},),
            },
        ),
        "attack_pattern": {
            "initial_move": "DOUBLE_STRIKE",
            "states": (
                {
                    "id": "DOUBLE_STRIKE_MOVE",
                    "move_id": "DOUBLE_STRIKE",
                    "next": "FORTIFY_MOVE",
                    "type": "move",
                },
                {
                    "id": "FORTIFY_MOVE",
                    "move_id": "FORTIFY",
                    "next": "DOUBLE_STRIKE_MOVE",
                    "type": "move",
                },
            ),
            "type": "cycle",
        },
    },
)

ENCOUNTERS = (
    {
        "id": "TRAINING_AUTOMATON_ENCOUNTER",
        "name": "Training Automaton",
        "act": "Act 1 - Overgrowth",
        "room_type": "Monster",
        "is_weak": True,
        "monsters": ({"id": "TRAINING_AUTOMATON"},),
    },
)


def _choose_first_ancient(state):
    action = next(action for action in legal_actions(state) if action.type == "choose_ancient")
    return step(state, action)


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


def _enter_training_combat(*, ascension: int = 0, deck: tuple[dict, ...] = ()):
    source_data = {
        "monsters": MONSTERS,
        "encounters": ENCOUNTERS,
        "combat_encounter_id": "TRAINING_AUTOMATON_ENCOUNTER",
        "deck": deck
        or (
            {
                "card_id": "defend",
                "name": "Defend",
                "type": "skill",
                "cost": 1,
                "target": "self",
                "effects": {"block": 5},
            },
        ),
        "player": {"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3},
    }
    state = new_run(
        seed=4100 + ascension,
        character_id="TEST",
        ascension=ascension,
        source_data=source_data,
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.MONSTER)
    action = next(action for action in legal_actions(state) if action.type == "choose_node")
    return step(state, action)


def _end_turn_action(state):
    return next(action for action in legal_actions(state) if action.type == "end_turn")


def test_combat_spawns_source_monster_with_scaled_initial_intent() -> None:
    state = _enter_training_combat()

    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.monster_id == "TRAINING_AUTOMATON"
    assert monster.hp == 30
    assert monster.move_id == "DOUBLE_STRIKE"
    assert monster.intent_damage == 10
    assert monster.hit_count == 2
    assert monster.metadata["encounter_id"] == "TRAINING_AUTOMATON_ENCOUNTER"

    ascension_state = _enter_training_combat(ascension=7)
    assert ascension_state.combat is not None
    ascension_monster = ascension_state.combat.monsters[0]
    assert ascension_monster.hp == 40
    assert ascension_monster.intent_damage == 14


def test_monster_turn_applies_multi_hit_damage_block_buff_and_next_intent() -> None:
    state = _enter_training_combat()
    assert state.combat is not None

    state = step(state, _end_turn_action(state))
    assert state.combat is not None
    assert state.combat.player.hp == 70
    monster = state.combat.monsters[0]
    assert monster.move_id == "FORTIFY"
    assert monster.intent_block == 4
    assert monster.intent_damage == 0

    state = step(state, _end_turn_action(state))
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.block == 4
    assert monster.statuses["strength"] == 2
    assert monster.move_id == "DOUBLE_STRIKE"
    assert monster.intent_damage == 14


def test_end_turn_cycles_discard_back_into_draw_pile() -> None:
    state = _enter_training_combat(
        deck=(
            {
                "card_id": "small_block",
                "name": "Small Block",
                "type": "skill",
                "cost": 1,
                "target": "self",
                "effects": {"block": 3},
            },
        )
    )

    assert state.combat is not None
    assert [card.card_id for card in state.combat.hand] == ["small_block"]
    assert state.combat.draw_pile == ()

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert [card.card_id for card in state.combat.hand] == ["small_block"]
    assert any(event.kind == "discard_shuffled" for event in state.combat.last_events)
    assert any(event.kind == "draw_pile_shuffled" for event in state.combat.last_events)
