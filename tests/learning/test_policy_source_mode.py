"""Public preview RNGs preserve source rules without retaining secret RNG state."""
from __future__ import annotations

from combat.test_card_play_status_runtime import _source, _state
from learning.test_policy_visibility import _combat

from sts2sim import action_space, legal_actions, serialize, step
from sts2sim.engine.rng import (
    PYTHON_RANDOM_ALGORITHM,
    STS2_STABLE_DERIVATION,
    RunRng,
    capture_random_state,
    run_random_from_seed,
)
from sts2sim.policy_visibility import policy_state


def test_source_projection_uses_constant_public_rng_without_hidden_banks_or_counters():
    first_rng = RunRng('hidden-seed-a', derivation=STS2_STABLE_DERIVATION)
    first_rng.for_stream('hidden_event_a').random()
    second_rng = RunRng('hidden-seed-b', derivation=STS2_STABLE_DERIVATION)
    for _ in range(11):
        second_rng.for_stream('hidden_event_b').random()
    first = _combat().model_copy(update={'rng': capture_random_state(first_rng)})
    second = first.model_copy(update={'rng': capture_random_state(second_rng)})
    expected = capture_random_state(run_random_from_seed(0))
    assert policy_state(first).rng == policy_state(second).rng == expected
    assert policy_state(first).rng.derivation == STS2_STABLE_DERIVATION
    assert policy_state(first).rng.seed == '0' and not policy_state(first).rng.streams
    assert policy_state(policy_state(first)).rng == expected
    assert action_space(first) == action_space(second)
    assert policy_state(first, oracle=True).rng == first.rng


def test_legacy_projection_remains_legacy_and_does_not_copy_custom_derivation():
    state = _combat().model_copy(update={'rng': capture_random_state(RunRng(
        'private-adapter-seed', derivation='sts2sim.sha256-splitmix64.v1'))})
    projected = policy_state(state)
    expected = capture_random_state(run_random_from_seed(0, algorithm=PYTHON_RANDOM_ALGORITHM))
    assert projected.rng == expected
    assert projected.rng.derivation != state.rng.derivation


def test_source_random_outcomes_and_unknown_draws_remain_suppressed_in_previews():
    state = _combat().model_copy(update={
        'rng': capture_random_state(run_random_from_seed('hidden-source-run')),
    })
    actions = {action['card_instance_id']: action for action in action_space(state)
               if action['type'] == 'play_card'}
    assert actions['hit']['preview']['target_hp_delta'] == -6
    assert actions['draw']['preview'] == {'preview_available': 0}
    assert actions['alchemy']['preview'] == {'preview_available': 0}


def test_source_entrench_preview_keeps_actual_unpowered_block_and_relic_rules():
    state = _state(_source('ENTRENCH'), statuses={
        'dexterity': 10, 'frail': 2, 'no_card_block_gain': 1, 'unmovable': 1,
    }, relics=('vambrace',))
    player = state.player.model_copy(update={'block': 5})
    state = state.model_copy(update={
        'player': player, 'combat': state.combat.model_copy(update={'player': player}),
        'rng': capture_random_state(RunRng('source-entrench-preview')),
    })
    original = serialize(state)
    descriptor = next(action for action in action_space(state)
                      if action['type'] == 'play_card')
    action = next(action for action in legal_actions(state) if action.type == 'play_card')
    actual = step(state, action)
    assert actual.player.block == 25
    assert descriptor['preview']['player_block_delta'] == 20
    assert serialize(state) == original
