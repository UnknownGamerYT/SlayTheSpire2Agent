from __future__ import annotations

import json

import pytest

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import (
    Action,
    CardEnchantment,
    CardInstance,
    MapNodeState,
    MapState,
    PlayerState,
    RoomKind,
    RunPhase,
    RunState,
)
from sts2sim.engine.rng import new_rng_state
from sts2sim.engine.transitions import _cook_at_campfire


def _campfire(*, relics: tuple[str, ...] = ()) -> RunState:
    node = MapNodeState(node_id="rest", act=1, floor=4, lane=0, kind=RoomKind.REST)
    return RunState(
        seed=204,
        character_id="TEST",
        rng=new_rng_state(204),
        phase=RunPhase.REST,
        floor=4,
        player=PlayerState(hp=40, max_hp=80),
        relics=relics,
        map=MapState(act=1, nodes=(node,), current_node_id="rest"),
        master_deck=tuple(
            CardInstance(
                instance_id=card_id,
                card_id=card_id,
                type="skill",
                target="self",
                effects={"block": 5},
                tags=("eternal",) if card_id == "eternal" else (),
            )
            for card_id in ("first", "second", "third", "eternal")
        ),
    )


def _clone_campfire(*, relics: tuple[str, ...] = ()) -> RunState:
    state = _campfire(relics=relics)
    first, second, third, eternal = state.master_deck
    enchantment = CardEnchantment(keyword="clone", amount=1, source_id="paels_growth")
    first = first.model_copy(
        update={
            "upgraded": True,
            "enchantments": (enchantment,),
            "custom": {"nested": {"counter": 1}},
        }
    )
    second = second.model_copy(update={"enchantments": (enchantment,)})
    return state.model_copy(update={"master_deck": (first, second, third, eternal)})


def test_cook_removes_exactly_two_selected_removable_cards_and_gains_nine_max_hp() -> None:
    state = _campfire(relics=("MEAT_CLEAVER",))
    actions = [action for action in legal_actions(state) if action.type == "cook"]
    assert len(actions) == 3
    assert all("eternal" not in a.payload["card_instance_ids"] for a in actions)
    action = next(a for a in actions if set(a.payload["card_instance_ids"]) == {"first", "third"})
    state = step(state, action)

    assert [card.instance_id for card in state.master_deck] == ["second", "eternal"]
    assert (state.player.hp, state.player.max_hp) == (49, 89)
    assert state.phase == RunPhase.MAP
    assert sum(event.kind == "card_removed" for event in state.replay_log[-1].events) == 2


@pytest.mark.parametrize(
    "ids",
    [
        (),
        ("first",),
        ("first", "first"),
        ("first", "second", "third"),
        ("first", "missing"),
        ("first", "eternal"),
    ],
)
def test_cook_rejects_invalid_card_sets_without_any_changes(ids: tuple[str, ...]) -> None:
    state = _campfire(relics=("meat_cleaver",))
    result, events = _cook_at_campfire(
        state, Action(type="cook", payload={"card_instance_ids": ids})
    )
    assert result == state
    assert events == ()


def test_cook_requires_meat_cleaver_and_two_removable_cards() -> None:
    state = _campfire()
    assert not any(action.type == "cook" for action in legal_actions(state))
    result, events = _cook_at_campfire(
        state,
        Action(
            type="cook",
            payload={"card_instance_ids": ("first", "second")},
        ),
    )
    assert result == state and events == ()
    state = state.model_copy(
        update={
            "relics": ("meat_cleaver",),
            "master_deck": (state.master_deck[0], state.master_deck[-1]),
        }
    )
    assert not any(action.type == "cook" for action in legal_actions(state))


def test_clone_duplicates_each_enchanted_card_once_preserving_card_data_without_relic_gate() -> (
    None
):
    state = _clone_campfire()
    originals = state.master_deck
    state = step(state, next(action for action in legal_actions(state) if action.type == "clone"))

    assert state.phase == RunPhase.MAP
    assert state.master_deck[:4] == originals
    assert len(state.master_deck) == 6
    assert len({card.instance_id for card in state.master_deck}) == 6
    for original, duplicate in zip(originals[:2], state.master_deck[4:], strict=True):
        assert duplicate.model_copy(update={"instance_id": original.instance_id}) == original
    assert state.master_deck[4].custom is not originals[0].custom
    assert state.master_deck[4].custom["nested"] is not originals[0].custom["nested"]


def test_clone_uses_unique_ids_even_if_a_derived_id_already_exists() -> None:
    state = _clone_campfire()
    collision = state.master_deck[2].model_copy(update={"instance_id": "first:clone:1"})
    state = state.model_copy(update={"master_deck": state.master_deck + (collision,)})
    state = step(state, next(action for action in legal_actions(state) if action.type == "clone"))
    assert len({card.instance_id for card in state.master_deck}) == 7
    assert state.master_deck[-2].instance_id == "first:clone:2"


def test_clone_is_unavailable_without_a_clone_enchantment() -> None:
    assert not any(a.type == "clone" for a in legal_actions(_campfire(relics=("paels_growth",))))


@pytest.mark.parametrize("action_type", ["cook", "clone"])
def test_cook_and_clone_respect_miniature_tent_once_per_site_and_save_reload(
    action_type: str,
) -> None:
    relics = ("miniature_tent", "meat_cleaver")
    state = _clone_campfire(relics=relics) if action_type == "clone" else _campfire(relics=relics)
    restored = load_state(json.loads(json.dumps(serialize(state))))
    action = next(a for a in legal_actions(state) if a.type == action_type)
    assert serialize(step(state, action)) == serialize(step(restored, action))
    state = step(state, action)

    assert state.phase == RunPhase.REST
    assert not any(a.type == action_type for a in legal_actions(state))
    assert any(a.type == "rest" for a in legal_actions(state))
