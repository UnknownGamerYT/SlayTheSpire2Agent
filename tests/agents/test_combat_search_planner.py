from __future__ import annotations

from typing import Any

from sts2sim import legal_actions, new_run, step
from sts2sim.agents import StrategicAgent
from sts2sim.engine import MapEdgeState, MapNodeState, MapState, RoomKind, RunPhase


def test_combat_search_prefers_lethal_over_blocking() -> None:
    state = _enter_combat(
        (
            _attack("POKE", damage=6, cost=1),
            _block("GUARD", block=20, cost=1),
        ),
        energy=1,
        monster_hp=6,
        incoming_damage=30,
    )

    assert _chosen_card_id(state) == "poke"


def test_combat_search_prefers_block_when_damage_cannot_prevent_hp_loss() -> None:
    state = _enter_combat(
        (
            _attack("POKE", damage=6, cost=1),
            _block("GUARD", block=12, cost=1),
        ),
        player={"hp": 30, "max_hp": 80, "energy": 1, "max_energy": 1},
        energy=1,
        monster_hp=30,
        incoming_damage=12,
    )

    assert _chosen_card_id(state) == "guard"


def test_combat_search_values_card_specific_kill_payoff() -> None:
    state = _enter_combat(
        (
            {
                **_attack("FINISHING_BLOW", damage=5, cost=1),
                "description": "Deal 5 damage.\nIf this kills an enemy, gain [energy:3].",
            },
            _attack("PLAIN_KILL", damage=5, cost=1),
        ),
        energy=1,
        monster_hp=5,
        incoming_damage=0,
    )

    decision = StrategicAgent().choose_action(state)

    assert decision.chosen is not None
    assert _card_id_for_action(state, decision.chosen.action) == "finishing_blow"
    assert "line_values_card_specific_payoff" in decision.chosen.reasons


def _enter_combat(
    deck: tuple[dict[str, Any], ...],
    *,
    player: dict[str, Any] | None = None,
    energy: int = 3,
    monster_hp: int = 30,
    incoming_damage: int = 0,
) -> Any:
    state = new_run(
        seed=9200,
        character_id="TEST",
        ascension=0,
        source_data={
            "deck": deck,
            "player": player
            or {"hp": 80, "max_hp": 80, "energy": energy, "max_energy": energy},
            "flags": {"draw_per_turn": len(deck)},
        },
    )
    state = _force_next_room(state, RoomKind.MONSTER)
    state = step(
        state,
        next(action for action in legal_actions(state) if action.type == "choose_node"),
    )
    assert state.combat is not None
    monster = state.combat.monsters[0].model_copy(
        update={
            "hp": monster_hp,
            "max_hp": max(monster_hp, state.combat.monsters[0].max_hp),
            "intent": "attack" if incoming_damage else None,
            "intent_damage": incoming_damage,
            "hit_count": 1,
        }
    )
    combat = state.combat.model_copy(
        update={
            "monsters": (monster,),
            "player": state.combat.player.model_copy(
                update={"energy": energy, "max_energy": energy}
            ),
        }
    )
    return state.model_copy(update={"combat": combat, "player": combat.player})


def _force_next_room(state: Any, room_kind: RoomKind) -> Any:
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


def _chosen_card_id(state: Any) -> str:
    decision = StrategicAgent().choose_action(state)
    assert decision.chosen is not None
    return _card_id_for_action(state, decision.chosen.action)


def _card_id_for_action(state: Any, action: dict[str, Any]) -> str:
    assert state.combat is not None
    card_instance_id = str(action.get("card_instance_id", ""))
    for card in state.combat.hand:
        if card.instance_id == card_instance_id:
            return card.card_id
    return ""


def _attack(card_id: str, *, damage: int, cost: int) -> dict[str, Any]:
    return {
        "id": card_id,
        "name": card_id.replace("_", " ").title(),
        "type": "Attack",
        "target": "AnyEnemy",
        "cost": cost,
        "damage": damage,
    }


def _block(card_id: str, *, block: int, cost: int) -> dict[str, Any]:
    return {
        "id": card_id,
        "name": card_id.replace("_", " ").title(),
        "type": "Skill",
        "target": "Self",
        "cost": cost,
        "block": block,
    }
