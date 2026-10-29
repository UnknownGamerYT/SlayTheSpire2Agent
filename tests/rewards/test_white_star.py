"""WhiteStar.TryModifyRewards adds a distinct Boss-profile reward after Elites."""
from __future__ import annotations

import pytest
from rewards.test_combat_rewards_engine import REWARD_CARDS, _action, _reward_after_kill
from rewards.test_relic_deck_choices import _reward_state

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import RoomKind
from sts2sim.engine.rng import RunRng, capture_random_state, random_from_state
from sts2sim.engine.transitions import _combat_reward_state


def test_white_star_adds_independent_rare_choices_and_preserves_boss_reroll_context():
    state = _reward_after_kill(RoomKind.ELITE, seed=99621,
        relics=('white_star', 'driftwood'), extra_source={'cards': REWARD_CARDS})
    reward = state.reward
    assert len(reward.card_option_groups) == 1
    group = reward.card_option_groups[0]
    assert len(group) == 3 and len(set(group)) == 3
    assert all(identifier in {'rare_power_a', 'rare_power_b', 'rare_power_c'}
               for identifier in group)
    assert reward.metadata['card_reroll_contexts']['group:0']['encounter'] == 'boss'
    assert reward.metadata['white_star_card_reward_group_indices'] == (0,)
    action = _action(state, 'reroll_reward', 'reward:reroll:group:0')
    assert serialize(step(state, action)) == serialize(step(load_state(serialize(state)), action))
    state = step(state, action)
    assert all(identifier in {'rare_power_a', 'rare_power_b', 'rare_power_c'}
               for identifier in state.reward.card_option_groups[0])
    state = step(state, _action(state, 'take_reward_card', 'reward:card_group:0:0'))
    assert state.master_deck[-1].card_id.lower() in group
    assert any(a.type == 'take_reward_card' and a.target_id.startswith('reward:card:')
               for a in legal_actions(state))


@pytest.mark.parametrize('encounter,expected_count', [('normal', 0), ('elite', 1), ('boss', 0)])
def test_white_star_requires_elite_room_and_source_boss_factory_consumes_nine_draws(
    encounter, expected_count,
):
    state = _reward_state('anchor').model_copy(update={
        'rng': capture_random_state(RunRng('WHITE-STAR',
            derivation='sts2.v0.107.1.utf16-djb2-splitmix64')),
        'flags': {'combat_reward_encounter': encounter, 'combat_reward_card_count': 0,
                  'combat_reward_relic_count': 0}})
    baseline_rng = random_from_state(state.rng)
    baseline, _, _ = _combat_reward_state(state, baseline_rng)
    rng = random_from_state(state.rng)
    state = state.model_copy(update={'relics': ('white_star',)})
    reward, _, _ = _combat_reward_state(state, rng)
    assert len(reward.card_option_groups) == len(baseline.card_option_groups) + expected_count
    delta = (capture_random_state(rng).streams['rewards'].draws
             - capture_random_state(baseline_rng).streams['rewards'].draws)
    assert delta == expected_count * 9
    if expected_count:
        assert reward.metadata['card_group_rarities'][-1] == ('rare',)*3


def test_white_star_extra_reward_uses_card_count_modifiers_and_late_hooks_independently():
    state = _reward_after_kill(RoomKind.ELITE, seed=99622,
        relics=('white_star', 'question_card', 'silver_crucible'),
        extra_source={'cards': REWARD_CARDS})
    assert len(state.reward.card_options) == 4
    assert len(state.reward.card_option_groups[0]) == 4
    assert state.reward.metadata['card_reroll_contexts']['group:0']['count'] == 4
    assert state.flags['silver_crucible_times_used'] == 2
    assert all(state.reward.metadata['card_option_upgraded'])
    assert all(state.reward.metadata['card_option_group_upgraded'][0])
