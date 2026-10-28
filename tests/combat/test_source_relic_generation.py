"""Actual source relic generation and stream/candidate boundaries."""

from __future__ import annotations

import pytest

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    Action,
    ActionType,
    CardInstance,
    CombatState,
    MapEdgeState,
    MapNodeState,
    MapState,
    MonsterState,
    PlayerState,
    RoomKind,
    RunPhase,
    load_state,
    serialize_state,
)
from sts2sim.engine.rng import RunRng, capture_random_state
from sts2sim.engine.transitions import _apply_combat_relic_marker
from sts2sim.mechanics.relic_combat import CombatRelicHook, CombatRelicMarker


def _combat() -> CombatState:
    return CombatState(player=PlayerState(hp=80, max_hp=80), monsters=tuple(
        MonsterState(monster_id=f"enemy_{i}", hp=100, max_hp=100) for i in range(3)
    ))


def test_random_damage_uses_combat_targets_and_can_hit_any_living_enemy() -> None:
    hits = set()
    for seed in range(20):
        rng = RunRng(str(seed))
        combat, _ = _apply_combat_relic_marker(_combat(), CombatRelicMarker(
            kind="random_damage", relic_id="forgotten_soul", amount=1,
            hook=CombatRelicHook.CARD_EXHAUSTED,
        ), rng=rng)
        hits.update(m.monster_id for m in combat.monsters if m.hp == 99)
        snapshot = capture_random_state(rng)
        assert snapshot.streams["combat_targets"].draws == 1
        assert "combat_card_selection" not in snapshot.streams
    assert hits == {"enemy_0", "enemy_1", "enemy_2"}


def test_bookmark_random_selection_excludes_x_and_zero_cost_cards() -> None:
    cards = tuple(CardInstance(instance_id=f"card_{i}", card_id=f"card_{i}", cost=cost,
                               custom={"retain": True})
                  for i, cost in enumerate((0, -1, 1, 2)))
    selected = set()
    for seed in range(12):
        rng = RunRng(str(seed))
        combat, _ = _apply_combat_relic_marker(
            _combat().model_copy(update={"hand": cards}), CombatRelicMarker(
                kind="reduce_retained_card_cost", relic_id="bookmark", amount=1,
                hook=CombatRelicHook.TURN_END,
            ), rng=rng,
        )
        selected.update(after.instance_id for before, after in zip(cards, combat.hand, strict=True)
                        if before.cost != after.cost)
        assert capture_random_state(rng).streams["combat_card_selection"].draws == 1
    assert selected == {"card_2", "card_3"}


def test_mummified_hand_includes_star_only_and_eventually_falls_back_to_zero_cost() -> None:
    star = CardInstance(instance_id="stars", card_id="stars", cost=0,
                        custom={"star_cost": 2, "source_spec": {"star_cost": 2}})
    zero = CardInstance(instance_id="zero", card_id="zero", cost=0)
    marker = CombatRelicMarker(kind="make_random_card_free_this_turn", relic_id="mummified_hand",
                               hook=CombatRelicHook.CARD_PLAYED)
    combat, _ = _apply_combat_relic_marker(
        _combat().model_copy(update={"hand": (zero, star)}), marker, rng=RunRng("STARS"),
    )
    assert combat.hand[1].custom["free_to_play_this_turn"]
    assert not combat.hand[0].custom.get("free_to_play_this_turn")
    combat, _ = _apply_combat_relic_marker(
        _combat().model_copy(update={"hand": (zero,)}), marker, rng=RunRng("ZERO"),
    )
    assert combat.hand[0].custom["free_to_play_this_turn"]


def _enter(relic: str, character: str = "NECROBINDER"):
    state = new_run("RELIC-CARDS", character, 0, source_data={
        "relics": [relic], "revealed_epochs": [],
    })
    start = MapNodeState(node_id="start", act=1, floor=0, lane=3, kind=RoomKind.START)
    target = MapNodeState(node_id="next", act=1, floor=1, lane=3, kind=RoomKind.MONSTER)
    state = state.model_copy(update={"phase": RunPhase.MAP, "ancient": None, "map": MapState(
        act=1, nodes=(start, target), edges=(MapEdgeState(from_id="start", to_id="next"),),
        current_node_id="start", completed_node_ids=("start",),
    )})
    action = Action(type=ActionType.CHOOSE_NODE, target_id="next")
    entered = step(state, action)
    restored = step(load_state(serialize_state(state)), action)
    assert serialize_state(entered) == serialize_state(restored)
    return entered


@pytest.mark.parametrize("relic", ["big_hat", "crossbow", "choices_paradox"])
def test_real_source_cards_are_generated_instead_of_placeholder_cards(relic) -> None:
    state = _enter(relic)
    assert state.combat is not None
    assert all(card.card_id != "generated_card" for card in state.combat.hand)
    if relic == "choices_paradox":
        choice = next(c for c in state.combat.pending_choices if c.zone == "generated")
        offers = choice.metadata["generated_cards"]
        assert len(offers) == 5 and len({c["card_id"] for c in offers}) == 5
        assert all(c["custom"]["retain"] for c in offers)
        selected = next(a for a in legal_actions(state) if a.type == ActionType.CHOOSE_CARD)
        restored = step(load_state(serialize_state(state)), selected)
        actual = step(state, selected)
        assert serialize_state(actual) == serialize_state(restored)
        assert actual.combat is not None
        assert any(card.custom.get("retain") for card in actual.combat.hand)
    else:
        generated = [card for card in state.combat.hand
                     if card.instance_id.startswith("choice_relic:")]
        assert len(generated) == (2 if relic == "big_hat" else 1)
        assert all(card.custom.get("ethereal") for card in generated) if relic == "big_hat" else (
            all(card.type.value == "attack" and card.custom.get("free_to_play_this_turn")
                for card in generated)
        )


def test_source_status_insertion_uses_shuffle_and_can_enter_opening_hand() -> None:
    state = _enter("funerary_mask")
    assert state.combat is not None
    assert any(card.card_id == "soul" for card in state.combat.hand)
    assert state.rng.streams["shuffle"].draws >= 3
