from __future__ import annotations

from rewards.test_relic_deck_choices import _reward_state

from sts2sim import action_space, legal_actions, load_state, serialize, step
from sts2sim.engine import RewardState, RunPhase
from sts2sim.engine.rng import RunRng, capture_random_state


def _source_state(relic_id, **flags):
    state = _reward_state(relic_id)
    return state.model_copy(update={'rng': capture_random_state(RunRng('PICKUP-REWARDS',
        derivation='sts2.v0.107.1.utf16-djb2-splitmix64')), 'flags': {**state.flags,
            'source_relic_bag_mode': 'source', 'player_relic_bag': {
                'Common': ('small_capsule', 'anchor', 'centennial_puzzle'),
                'Uncommon': ('orrery', 'horn_cleat'), 'Rare': ('old_coin', 'tungsten_rod')},
            **flags}})


def _act(state, kind, target=None):
    action = next(a for a in legal_actions(state) if a.type == kind
                  and (target is None or a.target_id == target))
    assert serialize(step(state, action)) == serialize(step(load_state(serialize(state)), action))
    return step(state, action)


def test_calling_bell_offers_separate_fixed_rarities_and_nested_reward_resumes_parent():
    state = _act(_source_state('calling_bell'), 'take_reward_relic')
    assert state.reward.relic_ids == ('small_capsule', 'orrery', 'old_coin')
    assert any(c.card_id == 'curse_of_the_bell' for c in state.master_deck)
    assert not {'small_capsule', 'orrery', 'old_coin'} & set(state.relics)
    assert len(state.flags['source_pickup_reward_stack']) == 1
    state = _act(state, 'take_reward_relic', 'reward:relic:0')
    assert state.reward.reward_id == 'pickup:small_capsule'
    assert len(state.flags['source_pickup_reward_stack']) == 2
    state = _act(state, 'proceed')
    assert state.reward.reward_id == 'pickup:calling_bell'
    assert state.reward.claimed_relic_ids == ('small_capsule',)
    state = _act(state, 'take_reward_relic', 'reward:relic:1')
    assert state.reward.reward_id == 'pickup:orrery'
    assert len(state.reward.card_options) == 3
    assert len(state.reward.card_option_groups) == 4
    state = _act(state, 'take_reward_card', 'reward:card:0')
    state = _act(state, 'proceed')
    assert state.reward.reward_id == 'pickup:calling_bell'
    state = _act(state, 'proceed')
    assert state.reward.reward_id == 'test-reward'
    assert state.reward.relic_claimed
    assert state.reward.gold == 77 and not state.reward.gold_claimed
    assert 'source_pickup_reward_stack' not in state.flags


def test_cauldron_preserves_five_potion_rewards_when_inventory_is_full():
    state = _source_state('cauldron').model_copy(update={'potions': ('strength_potion',)*3})
    state = _act(state, 'take_reward_relic')
    assert len(state.reward.potion_ids) == 5
    assert 'take_reward_potion' not in {a.type for a in legal_actions(state)}
    state = _act(state, 'discard_potion')
    state = _act(state, 'take_reward_potion')
    assert len(state.potions) == 3
    assert len(state.reward.claimed_potion_indices) == 1
    assert state.rng.streams['rewards'].draws == 10


def test_neows_bones_requires_both_rewards_and_defers_curse_until_child_rewards_complete():
    state = _act(_source_state('neows_bones', unlocked_character_ids=('ironclad',)),
                 'take_reward_relic')
    assert len(state.reward.relic_ids) == 2
    assert state.reward.forced
    assert not {'proceed', 'skip_reward'} & {a.type for a in legal_actions(state)}
    assert not any(c.type.value == 'curse' for c in state.master_deck)
    # Predetermined offers isolate mandatory completion from their pickup callbacks.
    state = state.model_copy(update={'reward': state.reward.model_copy(update={
        'relic_ids': ('anchor', 'small_capsule')})})
    state = _act(state, 'take_reward_relic', 'reward:relic:0')
    assert 'proceed' not in {a.type for a in legal_actions(state)}
    state = _act(state, 'take_reward_relic', 'reward:relic:1')
    assert state.reward.reward_id == 'pickup:small_capsule'
    assert not any(c.type.value == 'curse' for c in state.master_deck)
    state = _act(state, 'proceed')
    assert state.reward.reward_id == 'pickup:neows_bones'
    state = _act(state, 'proceed')
    assert sum(c.type.value == 'curse' for c in state.master_deck) == 1
    assert state.rng.streams['niche'].draws == 1


def test_orrery_consumes_source_other_rarity_selection_and_upgrade_draws():
    state = _act(_source_state('orrery'), 'take_reward_relic')
    assert len(state.reward.card_option_groups) == 4
    assert state.rng.streams['rewards'].draws == 45
    assert all(not v for v in state.reward.metadata['card_option_upgraded'])
    assert state.reward.metadata['card_reroll_contexts']['primary']['no_upgrade_roll'] is False


def test_glass_eye_nested_rewards_preserve_parent_and_source_uniform_reroll_contexts():
    state = _act(_source_state('glass_eye'), 'take_reward_relic')
    assert len(state.reward.card_option_groups) == 4
    assert state.rng.streams['rewards'].draws == 15
    contexts = state.reward.metadata['card_reroll_contexts']
    assert [v['rarity'] for v in contexts.values()] == [
        'common', 'common', 'uncommon', 'uncommon', 'rare']
    assert all(v['uniform'] and v['no_upgrade_roll'] for v in contexts.values())
    state = _act(state, 'proceed')
    assert state.reward.reward_id == 'test-reward'


def test_paels_wing_has_distinct_sacrifice_action_per_card_reward_and_rewards_every_two():
    state = _source_state('anchor').model_copy(update={'relics': ('paels_wing',),
        'reward': RewardState(reward_id='wing', card_options=('anger', 'bash'),
            card_option_groups=(('shrug_it_off',),), card_ids=('anger',))})
    choices = [a for a in action_space(state) if a['type'] == 'sacrifice_reward']
    assert len(choices) == 2
    assert all(a['reward_choice']['kind'] == 'sacrifice' for a in choices)
    state = _act(state, 'sacrifice_reward', 'reward:sacrifice:primary')
    assert state.flags['relic_counters']['paels_wing'] == 1
    assert not any(a.type == 'take_reward_card' and a.target_id.startswith('reward:card:')
                   for a in legal_actions(state))
    state = _act(state, 'sacrifice_reward', 'reward:sacrifice:group:0')
    assert state.flags['relic_counters']['paels_wing'] == 0
    assert len(state.relics) == 2
    assert state.flags['paels_wing_rewards_sacrificed'] == 2


def test_lost_coffer_generates_upgrade_rolls_then_potion_and_returns_to_parent():
    state = _act(_source_state('lost_coffer'), 'take_reward_relic')
    assert len(state.reward.card_options) == 3
    assert len(state.reward.potion_ids) == 1
    assert state.rng.streams['rewards'].draws == 11
    assert not state.reward.metadata['card_reroll_contexts']['primary']['no_upgrade_roll']
    state = _act(state, 'proceed')
    assert state.reward.reward_id == 'test-reward'


def test_exhausted_relic_bag_offers_three_separate_circlets_in_calling_bell():
    state = _act(_source_state('calling_bell', player_relic_bag={}), 'take_reward_relic')
    assert state.reward.relic_ids == ('circlet',)*3
    state = _act(state, 'take_reward_relic', 'reward:relic:0')
    assert len([a for a in legal_actions(state) if a.type == 'take_reward_relic']) == 2
    state = _act(state, 'skip_reward', 'reward:relic:1')
    state = _act(state, 'take_reward_relic', 'reward:relic:2')
    assert state.relics.count('circlet') == 2
    assert not any(a.type == 'take_reward_relic' for a in legal_actions(state))


def test_native_event_pickup_preserves_child_reward_and_resolves_parent_page():
    from sts2sim.engine import EventOptionState, EventState

    state = _source_state('cauldron').model_copy(update={'phase': RunPhase.EVENT, 'reward': None,
        'event': EventState(event_id='MORPHIC_GROVE', name='Fixture', options=(
            EventOptionState(option_id='pickup', title='Take', metadata={
                'source_native_rule': {'add_relics': ('cauldron',), 'selection': ''}}),))})
    state = _act(state, 'choose_event')
    assert state.reward.reward_id == 'pickup:cauldron'
    assert state.phase.value == 'reward'
    state = _act(state, 'proceed')
    assert state.phase.value == 'event'
    assert state.event.resolved_option_id == 'pickup'
    assert state.reward is None


def test_arcane_scroll_runs_single_nonreward_factory_wing_and_egg_hooks():
    state = _source_state('arcane_scroll').model_copy(update={
        'relics': ('wing_charm', 'silver_crucible', 'silken_tress',
                   'molten_egg', 'toxic_egg', 'frozen_egg')})
    state = _act(state, 'take_reward_relic')
    card = state.master_deck[-1]
    assert card.upgraded
    assert card.enchantments[0].keyword == 'swift'
    assert state.rng.streams['rewards'].draws == 1
    assert state.rng.streams['niche'].draws == 1
    assert not state.flags.get('silver_crucible_times_used')
    assert not state.flags.get('silken_tress_reward_used')



def test_paels_wing_stackable_circlet_fallback_can_be_obtained_repeatedly():
    state = _source_state('anchor', player_relic_bag={}).model_copy(update={
        'relics': ('paels_wing', 'circlet'), 'reward': RewardState(reward_id='wing',
            card_options=('anger',), card_option_groups=(('bash',),))})
    state = _act(state, 'sacrifice_reward', 'reward:sacrifice:primary')
    state = _act(state, 'sacrifice_reward', 'reward:sacrifice:group:0')
    assert state.relics.count('circlet') == 2
