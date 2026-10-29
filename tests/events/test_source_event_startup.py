"""Stable event startup variables: independent C vectors and public reload/action checks."""
from __future__ import annotations

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import Action, ActionType, RoomKind, RunPhase
from sts2sim.engine import transitions as e
from sts2sim.engine.rng import RunRng, capture_random_state, model_random
from sts2sim.mechanics.source_events import source_event_startup


@pytest.mark.parametrize(('event_id', 'values', 'draws'), [
    ('CRYSTAL_SPHERE', {'uncover_future_cost': 63}, 1),
    ('DENSE_VEGETATION', {'gold': 63, 'heal': 24}, 1),
    ('LOST_WISP', {'gold': 74}, 1),
    ('LUMINOUS_CHOIR', {'gold': 141}, 1),
    ('PUNCH_OFF', {'gold': 93}, 1),
    ('SUNKEN_STATUE', {'gold': 111}, 1),
    ('SUNKEN_TREASURY', {'small_chest_gold': 64, 'large_chest_gold': 330}, 2),
    ('THIS_OR_THAT', {'gold': 43}, 1),
    ('WHISPERING_HOLLOW', {'gold': 44}, 1),
    ('JUNGLE_MAZE_ADVENTURE', {'solo_gold': 164, 'join_forces_gold': 64}, 2),
])
def test_startup_values_and_consumption_match_independent_c(
    event_id: str, values: dict[str, int], draws: int
) -> None:
    rng = model_random(RunRng('TEST'), event_id)
    assert source_event_startup(event_id, rng, max_hp=80, hp=50).values == values
    assert capture_random_state(rng).draws == draws


def _enter(event_id: str):
    state = new_run('TEST', 'IRONCLAD', 0)
    assert state.map is not None
    state = state.model_copy(update={'flags': {**state.flags, 'event_id': event_id},
        'player': state.player.model_copy(update={'hp': 50, 'max_hp': 80, 'gold': 125})})
    node = state.map.nodes[0].model_copy(update={'kind': RoomKind.EVENT})
    return e._enter_event_room(state, node)[0]


def test_sunken_treasury_rolls_both_values_once_and_applies_gold_immediately_after_reload() -> None:
    state = _enter('SUNKEN_TREASURY')
    assert state.event is not None
    assert state.event.metadata['source_numeric_values'] == {
        'small_chest_gold': 64, 'large_chest_gold': 330
    }
    assert state.rng.streams['SUNKEN_TREASURY'].draws == 2
    before = state.player.gold
    restored = load_state(serialize(state))
    next_state = step(restored, Action(type=ActionType.CHOOSE_EVENT, target_id='first_chest'))
    assert next_state.player.gold == before + 64
    assert next_state.reward is None
    assert next_state.rng.streams['SUNKEN_TREASURY'].draws == 2


def test_dense_rest_preserves_explicit_fight_page_and_actual_heal() -> None:
    state = _enter('DENSE_VEGETATION')
    state = step(state, Action(type=ActionType.CHOOSE_EVENT, target_id='rest'))
    assert state.phase == RunPhase.EVENT
    assert state.combat is None
    assert state.player.hp == 74
    assert state.event is not None and state.event.page_id == 'REST'
    assert [action.target_id for action in legal_actions(state)] == ['fight']


def test_luminous_choir_locks_the_option_against_actual_cost() -> None:
    state = _enter('LUMINOUS_CHOIR')
    assert state.event is not None
    assert state.event.metadata['source_numeric_values']['gold'] == 141
    assert all(action.target_id != 'offer_tribute' for action in legal_actions(state))
    assert any(option.option_id == 'offer_tribute_locked' for option in state.event.options)


def test_reentering_the_same_event_recreates_model_rng() -> None:
    first = _enter('THIS_OR_THAT')
    assert first.map is not None
    node = first.map.nodes[0].model_copy(update={'kind': RoomKind.EVENT})
    second = e._enter_event_room(first, node)[0]
    assert second.event is not None and first.event is not None
    assert (second.event.metadata['source_numeric_values']
            == first.event.metadata['source_numeric_values'])
    assert second.rng.streams['THIS_OR_THAT'].draws == 1
