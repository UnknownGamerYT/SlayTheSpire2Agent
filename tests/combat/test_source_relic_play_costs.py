"""Stable relic payment and persistent-counter behavior through public actions."""

from __future__ import annotations

import pytest

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    Action,
    ActionType,
    MapEdgeState,
    MapNodeState,
    MapState,
    RoomKind,
    RunPhase,
    load_state,
    serialize_state,
    transitions,
)
from sts2sim.engine.rng import random_from_state


def _enter(cards, relics):
    state = new_run("PAYMENT", "REGENT", 0, source_data={
        "deck": cards, "relics": relics, "draw_per_turn": 10,
        "player": {"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3, "stars": 0},
        "monsters": [{"id": "QUIET", "hp": 1000, "max_hp": 1000,
                      "moves": [{"id": "WAIT", "intent": "defend", "block": 0}]}],
        "combat_monster_ids": ["QUIET"],
    })
    start = MapNodeState(node_id="start", act=1, floor=0, lane=3, kind=RoomKind.START)
    node = MapNodeState(node_id="next", act=1, floor=1, lane=3, kind=RoomKind.MONSTER)
    state = state.model_copy(update={"phase": RunPhase.MAP, "ancient": None, "map": MapState(
        act=1, nodes=(start, node), edges=(MapEdgeState(from_id="start", to_id="next"),),
        current_node_id="start", completed_node_ids=("start",),
    )})
    return step(state, Action(type=ActionType.CHOOSE_NODE, target_id="next"))


def _play(state, card_id):
    assert state.combat is not None
    card = next(card for card in state.combat.hand if card.card_id == card_id)
    action = next(a for a in legal_actions(state)
                  if a.type == ActionType.PLAY_CARD and a.card_instance_id == card.instance_id)
    actual = step(state, action)
    restored = step(load_state(serialize_state(state)), action)
    assert serialize_state(actual) == serialize_state(restored)
    return actual


@pytest.mark.parametrize("star_cost", [0, 5])
def test_brilliant_scarf_frees_fifth_manual_card_without_randomizing_other_cards(star_cost) -> None:
    cards = [{"card_id": f"zero_{i}", "type": "skill", "target": "self", "cost": 0,
              "effects": {}} for i in range(4)]
    cards += [{"card_id": "expensive", "type": "skill", "target": "self", "cost": 9,
               "custom": {"star_cost": star_cost}, "effects": {}},
              {"card_id": "remaining", "type": "skill", "target": "self", "cost": 1,
               "effects": {}}]
    state = _enter(cards, ["brilliant_scarf"])
    for i in range(4):
        state = _play(state, f"zero_{i}")
    state = _play(state, "expensive")
    assert state.combat is not None
    assert state.combat.player.energy == 3
    assert state.combat.player.resources.get("star", 0) == 0
    remaining = next(c for c in state.combat.hand if c.card_id == "remaining")
    assert not remaining.custom.get("free_to_play_this_turn")
    assert not any(e.kind == "card_made_free_this_turn" for e in state.combat.last_events)


def test_brilliant_scarf_does_not_count_autoplays_or_reduce_x_payments() -> None:
    state = _enter([{"card_id": "x", "cost": -1, "is_x_cost": True, "custom": {
                     "star_cost": "X"}, "effects": {}}], ["brilliant_scarf"])
    assert state.combat is not None
    combat = transitions._set_turn_power_count(state.combat, "brilliant_scarf_manual", 4)
    combat = combat.model_copy(update={"metadata": {**combat.metadata,
                                                    "item_relic_ids": ("brilliant_scarf",)}})
    card = combat.hand[0]
    assert transitions._energy_cost(card, 3, combat=combat) == 3
    cost_card = transitions._card_for_void_form_cost(combat, card)
    assert transitions._card_resource_costs(cost_card, available_star=5) == {"star": 5}
    combat, _ = transitions._apply_after_card_play_hooks(
        combat, card, state.relics, rng=random_from_state(state.rng), state=state,
        is_auto_play=True,
    )
    assert transitions._turn_power_count(combat, "brilliant_scarf_manual") == 4


@pytest.mark.parametrize("paid_x,free", [(False, False), (False, True), (True, False)])
def test_payment_relics_use_actual_energy_and_helmet_block_precedes_card_effect(
    paid_x, free,
) -> None:
    spec = {"card_id": "paid", "type": "attack", "target": "enemy", "cost": -1 if paid_x else 3,
            "is_x_cost": paid_x, "custom": {"free_to_play_this_turn": free},
            "effects": {"damage": 1}}
    state = _play(_enter([spec], ["intimidating_helmet", "ivory_tile"]), "paid")
    assert state.combat is not None
    assert state.combat.player.block == (0 if free else 4)
    assert state.combat.player.energy == (3 if free else 1)
    if not free:
        events = state.combat.last_events
        block = next(i for i, e in enumerate(events) if e.source_id == "intimidating_helmet")
        damage = next(i for i, e in enumerate(events) if e.kind == "monster_damaged")
        assert block < damage


def test_iron_club_saved_counter_survives_turn_boundary() -> None:
    cards = [{"card_id": f"zero_{i}", "type": "skill", "target": "self", "cost": 0,
              "effects": {}} for i in range(6)]
    state = _enter(cards, ["iron_club"])
    for i in range(2):
        state = _play(state, f"zero_{i}")
    assert state.combat is not None
    assert state.combat.metadata["relic_counters"]["iron_club"] == 2
    state = step(state, next(a for a in legal_actions(state) if a.type == ActionType.END_TURN))
    for i in range(2):
        state = _play(state, f"zero_{i}")
    assert state.combat is not None
    assert state.combat.metadata["relic_counters"]["iron_club"] == 4
    assert any(e.kind == "relic_bonus_draw_applied" for e in state.combat.last_events)
