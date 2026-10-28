"""Persist factory modifications across native event grids and direct deck adds."""
from __future__ import annotations

import pytest

from sts2sim import action_space, legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import Action, ActionType, RoomKind
from sts2sim.engine import transitions as t


def _event(event_id, relics, *, potions=()):
    state = new_run('EVENT-FACTORY', 'IRONCLAD', 0)
    state = state.model_copy(update={'relics': relics, 'potions': potions,
        'flags': {**state.flags, 'event_id': event_id}})
    return t._enter_event_room(state, state.map.nodes[0].model_copy(
        update={'kind': RoomKind.EVENT}))[0]


def _choose(state, target, *, selected=()):
    action = next(action for action in legal_actions(state)
                  if action.type == ActionType.CHOOSE_EVENT
                  and str(action.target_id).upper() == target
                  and (not selected or set(action.payload['card_instance_ids']) == set(selected)))
    assert serialize(step(state, action)) == serialize(step(load_state(serialize(state)), action))
    return step(state, action)


@pytest.mark.parametrize(('event_id', 'option_id', 'count', 'choose'), [
    ('BRAIN_LEECH', 'SHARE_KNOWLEDGE', 5, 1),
    ('ROOM_FULL_OF_CHEESE', 'GORGE', 8, 2),
])
def test_simple_grid_modifies_one_factory_batch_and_keeps_selected_card_previews(
    event_id, option_id, count, choose,
):
    state = _event(event_id, ('silver_crucible', 'silken_tress', 'molten_egg', 'toxic_egg',
                              'frozen_egg', 'wing_charm'))
    state = _choose(state, option_id)
    assert state.reward is None
    offered = state.event.metadata['generated_cards']
    upgrades = state.event.metadata['generated_card_upgraded']
    modifiers = state.event.metadata['generated_card_modifiers']
    assert len(offered) == count and all(upgrades)
    assert len(modifiers) == 1
    wing_index = int(next(iter(modifiers)))
    selected = tuple(dict.fromkeys((offered[wing_index], *offered)))[:choose]
    assert state.rng.streams['niche'].draws == 1
    assert not state.flags.get('silver_crucible_times_used')
    assert not state.flags.get('silken_tress_reward_used')
    preview = next(action for action in action_space(load_state(serialize(state)))
                   if action['type'] == 'choose_event'
                   and set(action['payload']['card_instance_ids']) == set(selected))
    assert len(preview['selected_cards']) == choose
    assert all(card['upgraded'] for card in preview['selected_cards'])
    assert sum(bool(card['enchantments']) for card in preview['selected_cards']) == 1
    result = _choose(state, 'CHOOSE_CARDS', selected=selected)
    obtained = result.master_deck[-choose:]
    assert {card.card_id for card in obtained} == set(selected)
    assert all(card.upgraded for card in obtained)
    assert sum(bool(card.enchantments) for card in obtained) == 1
    assert result.rng.streams['niche'].draws == 1
    assert not result.flags.get('silver_crucible_times_used')


def test_infested_immediate_factory_preserves_egg_and_wing_but_skips_reward_only_hooks():
    state = _event('INFESTED_AUTOMATON', (
        'silver_crucible', 'silken_tress', 'frozen_egg', 'wing_charm'))
    result = _choose(state, 'STUDY')
    card = result.master_deck[-1]
    assert card.type.value == 'power' and card.upgraded
    assert card.enchantments[0].keyword == 'swift'
    assert result.rng.streams['niche'].draws == 1
    assert not result.flags.get('silver_crucible_times_used')
    assert not result.flags.get('silken_tress_reward_used')


def test_brain_rip_retains_source_no_pool_modifications_on_reroll():
    state = _event('BRAIN_LEECH', ('prismatic_gem', 'dingy_rug', 'silver_crucible', 'driftwood'))
    state = _choose(state, 'RIP')
    assert not state.reward.card_option_groups  # Canonical RewardCount is exactly one.
    context = state.reward.metadata['card_reroll_contexts']['primary']
    assert context['no_pool_modifications'] and context['is_card_reward']
    assert state.flags['silver_crucible_times_used'] == 1
    action = next(action for action in legal_actions(state)
                  if action.type == ActionType.REROLL_REWARD)
    result = step(load_state(serialize(state)), action)
    assert result.flags['silver_crucible_times_used'] == 2
    assert all(t._reward_card_spec(result, card_id)['color'].lower() == 'colorless'
               for card_id in result.reward.card_options)


def test_trial_actual_card_rewards_use_prismatic_pool_but_simple_grids_do_not():
    state = _event('TRIAL', ('prismatic_gem',))
    state = _choose(state, 'ACCEPT')
    # Retain a real source-native page while selecting the Nondescript callback.
    state = state.model_copy(update={'event': t._source_event_next_page(
        state, state.event, 'NONDESCRIPT')})
    state = _choose(state, 'GUILTY')
    assert state.reward.card_options and len(state.reward.card_option_groups) == 1
    assert any(t._reward_card_spec(state, identifier)['color'].lower() != 'ironclad'
               for identifiers in (state.reward.card_options, *state.reward.card_option_groups)
               for identifier in identifiers)

    grid = _choose(_event('BRAIN_LEECH', ('prismatic_gem', 'dingy_rug')), 'SHARE_KNOWLEDGE')
    assert all(t._reward_card_spec(grid, identifier)['color'].lower() == 'ironclad'
               for identifier in grid.event.metadata['generated_cards'])


def test_future_of_potions_retains_no_pool_flag_and_forced_upgrades_on_reroll():
    state = _event('THE_FUTURE_OF_POTIONS', ('prismatic_gem', 'dingy_rug', 'driftwood'),
                   potions=('fire_potion',))
    state = _choose(state, 'POTION_0')
    context = state.reward.metadata['card_reroll_contexts']['primary']
    assert context['no_pool_modifications'] and context['force_upgrade']
    action = next(action for action in legal_actions(state)
                  if action.type == ActionType.REROLL_REWARD)
    result = step(load_state(serialize(state)), action)
    assert result.reward.metadata['card_option_upgraded'] == (True, True, True)
    assert all(t._reward_card_spec(result, card_id)['color'].lower() == 'ironclad'
               for card_id in result.reward.card_options)


def test_crystal_sphere_reward_allows_pool_hooks_before_rarity_filter():
    from sts2sim.engine.source_crystal_sphere import _sphere_event
    from sts2sim.mechanics.source_crystal_sphere import SphereBoard, SphereItem

    state = _choose(_event('CRYSTAL_SPHERE', ('prismatic_gem', 'driftwood')), 'PAYMENT_PLAN')
    board = SphereBoard((SphereItem('card', 'common', width=2, height=2, x=4, y=4,
                                    placed=True, subscriptions=1),), (), 1)
    state = state.model_copy(update={'event': _sphere_event(state.event, board)})
    state = step(load_state(serialize(state)), Action(type=ActionType.CHOOSE_EVENT,
                                                     target_id='big_5_5'))
    assert any(t._reward_card_spec(state, card_id)['color'].lower() != 'ironclad'
               for card_id in state.reward.card_options)
    context = state.reward.metadata['card_reroll_contexts']['primary']
    assert context['is_card_reward'] and not context['no_pool_modifications']
    assert all(rarity == 'common' for rarity in state.reward.metadata['card_group_rarities'][0])
