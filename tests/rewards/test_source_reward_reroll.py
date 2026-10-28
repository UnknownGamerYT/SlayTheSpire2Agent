from __future__ import annotations

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.agent_api import action_space
from sts2sim.engine import ActionType, RoomKind, RunPhase
from sts2sim.engine import transitions as e
from sts2sim.engine.errors import IllegalActionError
from sts2sim.engine.models import RewardState
from sts2sim.engine.rng import capture_random_state, random_from_state
from sts2sim.engine.source_reward_reroll import card_reroll_context


def _event(event_id):
    state = new_run('TEST', 'IRONCLAD', 0)
    state = state.model_copy(update={'flags': {**state.flags, 'event_id': event_id},
                                    'relics': (*state.relics, 'driftwood')})
    return e._enter_event_room(state, state.map.nodes[0].model_copy(
        update={'kind': RoomKind.EVENT}))[0]


def _combat_reward():
    state = new_run('TEST', 'IRONCLAD', 0)
    state = state.model_copy(update={'relics': (*state.relics, 'driftwood')})
    rng = random_from_state(state.rng)
    reward, _, flags = e._combat_reward_state(state, rng)
    return state.model_copy(update={'phase': RunPhase.REWARD, 'reward': reward,
                                   'flags': flags, 'rng': capture_random_state(rng)})


def test_combat_driftwood_public_reroll_consumes_factory_draws_once_and_updates_pity():
    state = _combat_reward()
    action = next(a for a in legal_actions(state) if a.type == ActionType.REROLL_REWARD)
    draws = state.rng.streams['rewards'].draws
    old_pity = state.flags['card_non_rare_count']
    result = step(load_state(serialize(state)), action)
    assert result.reward.reward_id == state.reward.reward_id
    assert result.rng.streams['rewards'].draws == draws + 9
    assert result.flags['card_non_rare_count'] == old_pity + 3
    assert not any(a.type == ActionType.REROLL_REWARD for a in legal_actions(result))
    assert len([a for a in legal_actions(result) if a.type == ActionType.TAKE_REWARD_CARD]) == 3
    assert result.reward.metadata['card_reward_rerolled_groups'] == ('primary',)
    with pytest.raises(IllegalActionError):
        step(result, action)
    card_action = next(a for a in legal_actions(result) if a.type == ActionType.TAKE_REWARD_CARD)
    taken = step(load_state(serialize(result)), card_action)
    assert len(taken.master_deck) == len(result.master_deck) + 1


def test_each_colorful_group_rerolls_its_own_rarity_pool_and_can_still_be_skipped():
    state = _event('COLORFUL_PHILOSOPHERS')
    state = step(state, next(a for a in legal_actions(state) if a.type == ActionType.CHOOSE_EVENT))
    actions = [a for a in legal_actions(state) if a.type == ActionType.REROLL_REWARD]
    assert len(actions) == 3
    old_primary = state.reward.card_options
    old_rare = state.reward.card_option_groups[1]
    draws = state.rng.streams['rewards'].draws
    result = step(load_state(serialize(state)),
                  next(a for a in actions if 'group:0' in a.target_id))
    assert result.reward.card_options == old_primary
    assert result.reward.card_option_groups[1] == old_rare
    assert result.rng.streams['rewards'].draws == draws + 6
    assert all(e._reward_card_spec(result, card)['rarity'].lower() == 'uncommon'
               for card in result.reward.card_option_groups[0])
    assert len([a for a in legal_actions(result) if a.type == ActionType.REROLL_REWARD]) == 2
    assert any(a.type == ActionType.SKIP_REWARD for a in legal_actions(result))


def test_manual_card_reward_uses_explicit_reroll_options_and_original_option_count():
    state = _event('BRAIN_LEECH')
    reward = RewardState(reward_id='manual', source='event', card_options=('anger',), metadata={
        'card_reroll_contexts': {'primary': card_reroll_context(
            pools=('silent',), count=1, uniform=True, no_upgrade_roll=False, rarity='rare')}})
    state = state.model_copy(update={'phase': RunPhase.REWARD, 'reward': reward})
    result = step(state, next(a for a in legal_actions(state)
                              if a.type == ActionType.REROLL_REWARD))
    assert len(result.reward.card_options) == 1
    spec = e._reward_card_spec(result, result.reward.card_options[0])
    assert spec['rarity'].lower() == 'rare' and spec['color'].lower() == 'silent'
    assert result.rng.streams['rewards'].draws == 2


def test_reroll_model_override_and_force_upgrade_survive_serialization():
    state = _event('CRYSTAL_SPHERE')
    reward = RewardState(reward_id='override', source='event', card_options=('anger',), metadata={
        'card_reroll_contexts': {'primary': card_reroll_context(
            count=3, uniform=True, no_upgrade_roll=False, rarity='common',
            rng_stream='CRYSTAL_SPHERE', force_upgrade=True)}})
    state = state.model_copy(update={'phase': RunPhase.REWARD, 'reward': reward})
    draws = state.rng.streams['CRYSTAL_SPHERE'].draws
    result = step(load_state(serialize(state)), next(a for a in legal_actions(state)
                                                  if a.type == ActionType.REROLL_REWARD))
    assert result.rng.streams['CRYSTAL_SPHERE'].draws == draws + 6
    assert result.rng.streams['rewards'].draws == 0
    assert result.reward.metadata['card_option_upgraded'] == (True,) * 3


def test_fixed_special_cards_and_returned_stolen_cards_have_no_driftwood_reroll():
    state = _combat_reward()
    reward = RewardState(reward_id='special', source='combat', card_ids=('lantern_key',),
        card_option_groups=(('bash',),), metadata={'returned_card_option_specs': {'0': {}}})
    state = state.model_copy(update={'reward': reward})
    assert not any(a.type == ActionType.REROLL_REWARD for a in legal_actions(state))


def test_agent_reroll_descriptor_uses_a_stable_action_type_and_keeps_selection_open():
    state = _combat_reward()
    descriptors = action_space(state)
    descriptor = next(d for d in descriptors if d['type'] == 'reroll_reward')
    assert descriptor['action_type_id'] >= 0
    assert descriptor['reward_choice']['kind'] == 'reroll'
    assert not descriptor['reward_choice']['closes_selection_set']



def test_known_source_empty_custom_reroll_is_masked_and_keeps_its_diagnostic():
    state = _combat_reward()
    context = card_reroll_context(
        mode='source_empty_custom_pool', pools=(), card_ids=(), custom_pool=True,
        count=3, no_upgrade_roll=True, is_card_reward=False,
        unsupported_reason='source_empty_custom_card_pool',
    )
    reward = RewardState(reward_id='manual-empty-source', source='other',
        card_options=('anger', 'shrug_it_off', 'bash'), metadata={
            'card_reroll_contexts': {'primary': context}})
    state = load_state(serialize(state.model_copy(update={'reward': reward})))
    assert state.reward.metadata['card_reroll_contexts']['primary']['availability'] == 'unsupported'
    assert not any(action.type == ActionType.REROLL_REWARD for action in legal_actions(state))
    assert len([action for action in legal_actions(state)
                if action.type == ActionType.TAKE_REWARD_CARD]) == 3
    assert any(action.type == ActionType.SKIP_REWARD for action in legal_actions(state))
    from sts2sim.engine import Action
    with pytest.raises(IllegalActionError):
        step(state, Action(type=ActionType.REROLL_REWARD, target_id='reward:reroll:primary'))


def test_nonempty_manual_custom_reroll_still_uses_its_original_explicit_pool():
    state = _combat_reward()
    context = card_reroll_context(pools=(), custom_pool=True,
        card_ids=('anger', 'shrug_it_off', 'armaments'), count=2, uniform=True)
    state = state.model_copy(update={'reward': RewardState(
        reward_id='manual-explicit', source='other', card_options=('bash',),
        metadata={'card_reroll_contexts': {'primary': context}})})
    action = next(action for action in legal_actions(state)
                  if action.type == ActionType.REROLL_REWARD)
    result = step(load_state(serialize(state)), action)
    assert len(result.reward.card_options) == 2
    assert set(result.reward.card_options) <= {'anger', 'shrug_it_off', 'armaments'}
