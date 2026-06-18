from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import MapEdgeState, MapNodeState, MapState, OrbState, RoomKind, RunPhase

ORB_DUMMY_MONSTERS = (
    {
        "id": "ORB_DUMMY",
        "name": "Orb Dummy",
        "type": "Normal",
        "min_hp": 50,
        "max_hp": 50,
        "moves": (
            {
                "id": "STRIKE",
                "name": "Strike",
                "intent": "Attack",
                "damage": {"normal": 10, "ascension": 10, "hit_count": 1},
            },
        ),
        "attack_pattern": {
            "initial_move": "STRIKE",
            "states": ({"id": "STRIKE_MOVE", "move_id": "STRIKE", "type": "move"},),
        },
    },
)

ORB_DUMMY_ENCOUNTERS = (
    {
        "id": "ORB_DUMMY_ENCOUNTER",
        "name": "Orb Dummy",
        "act": "Act 1 - Overgrowth",
        "room_type": "Monster",
        "is_weak": True,
        "monsters": ({"id": "ORB_DUMMY"},),
    },
)


def _choose_first_ancient(state):
    return state.model_copy(update={"phase": RunPhase.MAP, "ancient": None})


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


def _enter_orb_combat(
    deck: Sequence[Mapping[str, Any]],
    *,
    orb_slots: int | None = 3,
    relics: Sequence[str] = (),
):
    source_data: dict[str, Any] = {
        "monsters": ORB_DUMMY_MONSTERS,
        "encounters": ORB_DUMMY_ENCOUNTERS,
        "combat_encounter_id": "ORB_DUMMY_ENCOUNTER",
        "deck": tuple(dict(card) for card in deck),
        "player": {"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3},
        "draw_per_turn": len(deck),
    }
    if orb_slots is not None:
        source_data["orb_slots"] = orb_slots
    state = new_run(
        seed=7300,
        character_id="TEST",
        ascension=0,
        source_data=source_data,
    )
    state = _choose_first_ancient(state)
    if relics:
        state = state.model_copy(update={"relics": tuple(relics)})
    state = _force_next_room(state, RoomKind.MONSTER)
    return step(
        state,
        next(action for action in legal_actions(state) if action.type == "choose_node"),
    )


def _orb_card(card_id: str, effects: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "card_id": card_id,
        "name": card_id.replace("_", " ").title(),
        "type": "Skill",
        "target": "Self",
        "cost": 0,
        "effects": dict(effects),
    }


def _play_card(state, card_id: str):
    assert state.combat is not None
    card = next(card for card in state.combat.hand if card.card_id == card_id)
    return step(
        state,
        next(
            action
            for action in legal_actions(state)
            if action.type == "play_card" and action.card_instance_id == card.instance_id
        ),
    )


def test_cracked_core_channels_opening_lightning_and_grants_slots() -> None:
    state = _enter_orb_combat((), orb_slots=None, relics=("cracked_core",))

    assert state.combat is not None
    assert state.combat.orb_slots == 3
    assert [orb.orb_id for orb in state.combat.orbs] == ["lightning"]
    assert any(event.kind == "orb_channeled" for event in state.combat.last_events)


def test_channel_and_evoke_orbs_with_focus() -> None:
    state = _enter_orb_combat(
        (
            _orb_card("zap", {"channel_orb": {"orb": "lightning", "amount": 1}}),
            _orb_card("dualcast", {"evoke_orb": {"selector": "rightmost", "amount": 2}}),
        )
    )
    assert state.combat is not None
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={"player": state.combat.player.model_copy(update={"statuses": {"focus": 2}})}
            )
        }
    )

    state = _play_card(state, "zap")
    assert state.combat is not None
    assert [orb.orb_id for orb in state.combat.orbs] == ["lightning"]

    starting_hp = state.combat.monsters[0].hp
    state = _play_card(state, "dualcast")

    assert state.combat is not None
    assert state.combat.orbs == ()
    assert state.combat.monsters[0].hp == starting_hp - 20


def test_source_description_orb_text_executes_in_combat() -> None:
    state = _enter_orb_combat(
        (
            {
                "card_id": "source_zap",
                "name": "Source Zap",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Channel 1 Lightning.",
            },
        )
    )

    state = _play_card(state, "source_zap")

    assert state.combat is not None
    assert [orb.orb_id for orb in state.combat.orbs] == ["lightning"]
    assert state.combat.last_events[-1].kind == "orb_channeled"


def test_channel_overflow_evokes_leftmost_orb_without_losing_new_orb() -> None:
    state = _enter_orb_combat(
        (
            _orb_card(
                "overfill",
                {
                    "sequence": (
                        {"channel_orb": {"orb": "frost", "amount": 1}},
                        {"channel_orb": {"orb": "frost", "amount": 1}},
                    )
                },
            ),
        ),
        orb_slots=1,
    )

    state = _play_card(state, "overfill")

    assert state.combat is not None
    assert [orb.orb_id for orb in state.combat.orbs] == ["frost"]
    assert state.combat.player.block == 5
    assert any(
        event.kind == "player_block" and event.metadata.get("reason") == "channel_overflow"
        for event in state.combat.last_events
    )


def test_orb_passives_fire_at_end_turn_with_focus() -> None:
    state = _enter_orb_combat((), orb_slots=3)
    assert state.combat is not None
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "orbs": (
                        OrbState(orb_id="frost"),
                        OrbState(orb_id="lightning"),
                        OrbState(orb_id="dark", value=6),
                    ),
                    "player": state.combat.player.model_copy(update={"statuses": {"focus": 1}}),
                }
            )
        }
    )

    state = step(
        state,
        next(action for action in legal_actions(state) if action.type == "end_turn"),
    )

    assert state.combat is not None
    assert state.combat.player.hp == 73
    assert state.combat.monsters[0].hp == 46
    assert state.combat.orbs[2].value == 13
    assert any(event.kind == "orb_dark_charged" for event in state.combat.last_events)
