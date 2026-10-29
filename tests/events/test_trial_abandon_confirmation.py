from dataclasses import replace

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.agent_api import action_space, encode_observation
from sts2sim.engine import Action, ActionType, EngineError, RoomKind, RunPhase
from sts2sim.engine import transitions as engine
from sts2sim.mechanics.event_flows import (
    event_flow_state,
    legal_event_flow_option_ids,
    resolve_event_flow_option,
)


def _trial():
    state = new_run("trial-confirmation", "IRONCLAD", 0)
    state = state.model_copy(update={
        "flags": {**state.flags, "event_id": "TRIAL"},
        "potions": ("fairy_in_a_bottle",),
        "relics": state.relics + ("tungsten_rod",),
        "player": state.player.model_copy(update={"statuses": {"buffer": 1}}),
    })
    return engine._enter_event_room(
        state, state.map.nodes[0].model_copy(update={"kind": RoomKind.EVENT}),
    )[0]


def _choose(state, option):
    return step(state, Action(type=ActionType.CHOOSE_EVENT, target_id=option))


def test_source_trial_double_down_opens_a_nonlethal_cancelable_confirmation():
    flow = event_flow_state("TRIAL", hp=37, max_hp=80, gold=123,
                            data={"source_runtime": True, "trial_case": "NOBLE"})
    rejected = resolve_event_flow_option(flow, "REJECT").state
    popup = resolve_event_flow_option(rejected, "DOUBLE_DOWN")
    assert popup.state.page_id == "ABANDON_CONFIRMATION"
    assert (popup.state.hp, popup.state.max_hp, popup.state.gold) == (37, 80, 123)
    assert not popup.terminal and not popup.markers
    assert legal_event_flow_option_ids(popup.state) == ("CONFIRM_ABANDON", "CANCEL_ABANDON")
    canceled = resolve_event_flow_option(popup.state, "CANCEL_ABANDON")
    assert canceled.state.page_id == "REJECT"
    assert not canceled.terminal and not canceled.markers
    assert legal_event_flow_option_ids(canceled.state) == ("ACCEPT", "DOUBLE_DOWN")
    assert resolve_event_flow_option(canceled.state, "ACCEPT").state.page_id == "NOBLE"
    # The reserved pure fixture path keeps its earlier explicit lethal action.
    legacy = replace(rejected, data={"trial_case": "NOBLE"})
    assert resolve_event_flow_option(legacy, "DOUBLE_DOWN").terminal


def test_public_confirmation_blocks_underlying_event_and_proceed_actions():
    rejected = _choose(_trial(), "REJECT")
    popup = _choose(rejected, "DOUBLE_DOWN")
    assert popup.phase is RunPhase.EVENT
    assert popup.player == rejected.player and popup.master_deck == rejected.master_deck
    assert popup.relics == rejected.relics and popup.potions == rejected.potions
    assert popup.rng == rejected.rng
    assert {action.target_id for action in legal_actions(popup)} == {
        "CONFIRM_ABANDON", "CANCEL_ABANDON",
    }
    with pytest.raises(EngineError):
        _choose(popup, "ACCEPT")
    with pytest.raises(EngineError):
        step(popup, Action(type=ActionType.PROCEED))


def test_cancel_reopens_original_repeatable_options_without_spending_trial_rng():
    rejected = _choose(_trial(), "REJECT")
    state = rejected
    for _ in range(3):
        state = _choose(load_state(serialize(state)), "DOUBLE_DOWN")
        state = _choose(load_state(serialize(state)), "CANCEL_ABANDON")
        assert state.player == rejected.player and state.rng == rejected.rng
        assert state.event.page_id == "REJECT"
        assert {a.target_id for a in legal_actions(state)} == {"ACCEPT", "DOUBLE_DOWN"}
    after_cancel = _choose(state, "ACCEPT")
    direct = _choose(rejected, "ACCEPT")
    assert after_cancel.event.page_id == direct.event.page_id
    assert after_cancel.event.options == direct.event.options
    assert after_cancel.rng == direct.rng
    assert after_cancel.rng.streams["TRIAL"].draws == 1


def test_abandon_modal_blocks_potion_use_and_discard_until_canceled():
    rejected = _choose(_trial(), "REJECT")
    rejected = rejected.model_copy(update={
        "potions": ("fruit_juice",),
        "flags": {**rejected.flags, "allow_free_potion_discard": True},
    })
    ordinary = legal_actions(rejected)
    assert {ActionType.USE_POTION, ActionType.DISCARD_POTION}.issubset(
        {action.type for action in ordinary},
    )
    popup = _choose(rejected, "DOUBLE_DOWN")
    assert {action.type for action in legal_actions(popup)} == {ActionType.CHOOSE_EVENT}
    for action in ordinary:
        if action.type in {ActionType.USE_POTION, ActionType.DISCARD_POTION}:
            with pytest.raises(EngineError):
                step(popup, action)
    canceled = _choose(popup, "CANCEL_ABANDON")
    assert {ActionType.USE_POTION, ActionType.DISCARD_POTION}.issubset(
        {action.type for action in legal_actions(canceled)},
    )
    assert canceled.potions == rejected.potions and canceled.player == rejected.player


@pytest.mark.parametrize("decision", ["CONFIRM_ABANDON", "CANCEL_ABANDON"])
def test_confirmation_choice_reload_and_actor_inputs_are_identical(decision):
    popup = _choose(_choose(_trial(), "REJECT"), "DOUBLE_DOWN")
    restored = load_state(serialize(popup))
    assert legal_actions(restored) == legal_actions(popup)
    assert action_space(restored) == action_space(popup)
    assert encode_observation(restored) == encode_observation(popup)
    descriptors = action_space(popup)
    assert {item["event_option"]["page_id"] for item in descriptors} == {
        "ABANDON_CONFIRMATION",
    }
    assert {item["event_option"]["metadata"]["confirmation_choice"]
            for item in descriptors} == {True, False}
    assert serialize(_choose(restored, decision)) == serialize(_choose(popup, decision))


def test_confirmed_abandon_force_kills_only_after_confirmation_and_is_terminal():
    popup = _choose(_choose(_trial(), "REJECT"), "DOUBLE_DOWN")
    result = _choose(load_state(serialize(popup)), "CONFIRM_ABANDON")
    assert result.phase is RunPhase.FAILED and result.player.hp == 0
    assert result.flags["run_abandoned"] is True
    assert result.event.page_id == "ABANDONED"
    assert result.rng == popup.rng
    assert result.potions == popup.potions  # Force kill bypasses Fairy's death prevention.
    assert result.player.statuses == popup.player.statuses
    assert not legal_actions(result)
    assert not legal_actions(load_state(serialize(result)))
    events = result.replay_log[-1].events
    assert sum(event.kind == "run_abandoned" for event in events) == 1
    assert not any(event.kind == "potion_used" for event in events)
