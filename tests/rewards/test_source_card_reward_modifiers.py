"""Source late-hook flags, per-option provenance and repeat/population boundaries."""
from __future__ import annotations

from rewards.test_relic_deck_choices import _reward_state

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.agent_api import action_space
from sts2sim.engine import RewardState
from sts2sim.engine.rng import RunRng, capture_random_state, random_from_state
from sts2sim.engine.source_card_reward_modifiers import modify_generated_cards


def _state(relics):
    state = _reward_state('silver_crucible')
    return state.model_copy(update={
        'relics': relics,
        'rng': capture_random_state(RunRng('REWARD-LATE',
            derivation='sts2.v0.107.1.utf16-djb2-splitmix64')),
        'reward': RewardState(reward_id='batch:0', gold=1,
            card_options=('anger', 'shrug_it_off'), relic_id='silver_crucible'),
    })


def _act(state, kind, target=None):
    action = next(a for a in legal_actions(state) if a.type == kind
                  and (target is None or a.target_id == target))
    assert serialize(step(state, action)) == serialize(step(load_state(serialize(state)), action))
    return step(state, action)


def test_wing_charm_marks_only_one_option_and_upgrade_preview_matches_pickup():
    state = _act(_state(('wing_charm',)), 'take_reward_relic')
    assert state.flags['silver_crucible_times_used'] == 1
    assert state.flags['relic_counters']['silver_crucible'] == 2
    assert state.reward.metadata['card_option_upgraded'] == (True, True)
    assert state.rng.streams['niche'].draws == 1
    modifiers = state.reward.metadata['card_reward_option_modifiers']['primary']
    assert len(modifiers) == 1
    index = int(next(iter(modifiers)))
    options = [a for a in action_space(state) if a['type'] == 'take_reward_card']
    assert all(a['card']['upgraded'] for a in options)
    assert sum(bool(a['card']['enchantments']) for a in options) == 1
    selected = next(a for a in options if a['target_id'] == f'reward:card:{index}')
    assert selected['card']['enchantments'][0]['keyword'] == 'swift'
    state = _act(state, 'take_reward_card', f'reward:card:{index}')
    assert state.master_deck[-1].upgraded
    assert state.master_deck[-1].enchantments[0].keyword == 'swift'
    assert state.rng.streams['niche'].draws == 1
    assert state.flags['silver_crucible_times_used'] == 1


def test_silver_crucible_counts_reward_batches_once_including_repeated_actions():
    state = _act(_state(()), 'take_reward_relic')
    for number in range(1, 4):
        state = state.model_copy(update={'reward': RewardState(
            reward_id=f'batch:{number}', gold=1, card_options=('anger', 'shrug_it_off'),
        )})
        state = _act(state, 'take_reward_gold')
        assert state.reward.metadata['card_option_upgraded'] == ((True, True) if number < 3
                                                                  else (False, False))
        state = _act(state, 'take_reward_card', 'reward:card:0')
    assert state.flags['silver_crucible_times_used'] == 3


def test_manual_reward_no_model_modifications_still_runs_silver_and_egg_hooks():
    state = _state(('silver_crucible', 'frozen_egg'))
    rng = random_from_state(state.rng)
    state, upgraded, _, _ = modify_generated_cards(state, rng, ('anger',), (False,),
        is_card_reward=True, context={'no_card_model_modifications': True})
    assert upgraded == (True,)
    assert state.flags['silver_crucible_times_used'] == 1


def test_nonreward_factory_skips_silver_silken_but_applies_egg_and_wing():
    state = _state(('silver_crucible', 'silken_tress', 'molten_egg', 'wing_charm'))
    rng = random_from_state(state.rng)
    updated, upgrades, mods, _ = modify_generated_cards(state, rng, ('anger',), (False,),
        is_card_reward=False, context={'no_upgrade_roll': True})
    assert upgrades == (True,)
    assert not updated.flags.get('silver_crucible_times_used')
    assert not updated.flags.get('silken_tress_reward_used')
    assert mods['0']['enchantments'][0]['keyword'] == 'swift'
    assert capture_random_state(rng).streams['niche'].draws == 1


def test_no_modify_hooks_suppresses_all_hooks_and_no_hook_upgrades_only_suppresses_eggs():
    state = _state(('molten_egg', 'wing_charm'))
    rng = random_from_state(state.rng)
    updated, upgrades, mods, events = modify_generated_cards(state, rng, ('anger',), (False,),
        is_card_reward=False, context={'no_modify_hooks': True})
    assert updated == state and upgrades == (False,) and not mods and not events
    assert capture_random_state(rng) == state.rng
    _, upgrades, mods, _ = modify_generated_cards(state, rng, ('anger',), (False,),
        is_card_reward=False, context={'no_hook_upgrades': True})
    assert upgrades == (False,)
    assert mods['0']['enchantments'][0]['keyword'] == 'swift'


def test_new_relic_updates_existing_reward_without_repeating_old_late_hooks():
    state = _act(_state(('wing_charm',)), 'take_reward_relic')
    state = state.model_copy(update={'reward': state.reward.model_copy(update={
        'relic_id': 'toxic_egg', 'relic_claimed': False,
    })})
    state = _act(state, 'take_reward_relic')
    assert state.flags['silver_crucible_times_used'] == 1
    assert state.rng.streams['niche'].draws == 1


def test_fresnel_lens_modifies_only_enchantable_reward_options_before_they_are_taken():
    state = _state(('fresnel_lens',))
    state = _act(state, 'take_reward_gold')
    options = [a for a in action_space(state) if a['type'] == 'take_reward_card']
    attack = next(a for a in options if a['card']['card_id'] == 'anger')
    block = next(a for a in options if a['card']['card_id'] == 'shrug_it_off')
    assert not attack['card']['enchantments']
    assert block['card']['enchantments'][0]['keyword'] == 'nimble'
    assert block['card']['enchantments'][0]['amount'] == 2
