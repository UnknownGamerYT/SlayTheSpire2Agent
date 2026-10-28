"""Finite stable source contracts; the fixture records each reviewed method/hash."""
from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _state
from combat.test_glam import _play

from sts2sim import action_space, legal_actions, new_run, serialize, step
from sts2sim.engine import PlayerState, RewardState, RunPhase, RunState
from sts2sim.engine import source_run_passives as hooks
from sts2sim.engine import transitions as t
from sts2sim.engine.rng import RunRng, capture_random_state, random_from_state


def _act(state, kind, target=None):
    return step(state, next(a for a in legal_actions(state) if a.type == kind
                            and (target is None or a.target_id == target)))


def _reward(card_id, relics):
    return RunState(seed=1, character_id='IRONCLAD', phase=RunPhase.REWARD,
        rng=capture_random_state(RunRng('PASSIVE')), player=PlayerState(hp=10, max_hp=80),
        relics=relics,
        reward=RewardState(reward_id='passive', card_options=(card_id,)))


def test_darkstone_periapt_observes_each_curse_deck_entry_including_bing_copy():
    state = _reward('injury', ('darkstone_periapt', 'bing_bong'))
    result = _act(_reload(state), 'take_reward_card')
    assert len(result.master_deck) == 2
    assert all(c.type.value == 'curse' for c in result.master_deck)
    assert (result.player.max_hp, result.player.hp) == (92, 22)
    changes = [e for e in result.replay_log[-1].events if e.source_id == 'darkstone_periapt']
    assert [e.amount for e in changes] == [6, 6]
    assert serialize(_reload(result)) == serialize(result)


@pytest.mark.parametrize('card_id', ['bash', 'burn'])
def test_darkstone_does_not_trigger_for_attack_or_status(card_id):
    result = _act(_reward(card_id, ('darkstone_periapt',)), 'take_reward_card')
    assert (result.player.max_hp, result.player.hp) == (80, 10)


def test_dingy_rug_appends_source_ordered_unlocked_colorless_only_for_modifiable_rewards():
    state = new_run('RUG', 'IRONCLAD', 0, source_data={'relics': ['dingy_rug']})
    pool = t._combat_card_pool(state)
    expanded = hooks.modify_reward_pool(state, pool)
    assert tuple(expanded[:len(pool)]) == pool
    assert 'finesse' in {c.card_id for c in expanded}
    assert len({c.card_id for c in expanded}) == len(expanded)
    for flags in (
        {'is_card_reward': False}, {'no_pool_modifications': True}, {'custom_pool': True},
    ):
        assert hooks.modify_reward_pool(state, pool, **flags) == pool
    locked = state.model_copy(update={'flags': {**state.flags, 'revealed_epochs': []}})
    locked_ids = {c.card_id for c in hooks.modify_reward_pool(locked, t._combat_card_pool(locked))}
    assert locked_ids <= {c.card_id for c in expanded}
    assert len(locked_ids) < len(expanded)


def _candy_victory(combats_seen=1, *, extra_groups=0):
    kill = _simple('kill', effects={'all_damage': 999}, card_type='attack', target='all_enemies')
    state = _state(kill, relics=('lasting_candy', 'driftwood'))
    # Three cards force the base reward to include both powers; the source then
    # falls back to its entire Power pool and may duplicate one original choice.
    cards = (
        {'id': 'ANGER', 'type': 'Attack', 'rarity': 'Common', 'color': 'ironclad'},
        {'id': 'INFLAME', 'type': 'Power', 'rarity': 'Uncommon', 'color': 'ironclad'},
        {'id': 'DEMON_FORM', 'type': 'Power', 'rarity': 'Rare', 'color': 'ironclad'},
    )
    flags = {**state.flags, 'card_pool': cards, 'combat_reward_card_count': 3,
             'combat_reward_extra_card_groups': extra_groups,
             'relic_counters': {'lasting_candy': combats_seen}}
    state = state.model_copy(update={'flags': flags, 'combat': state.combat.model_copy(update={
        'metadata': {**state.combat.metadata, 'relic_counters': {'lasting_candy': combats_seen}},
    }), 'rng': capture_random_state(RunRng(
        'CANDY', derivation='sts2.v0.107.1.utf16-djb2-splitmix64'))})
    return _play(_reload(state), 'kill')


def test_lasting_candy_saved_counter_and_extra_power_on_second_completed_combat():
    first = _candy_victory(0)
    assert first.flags['relic_counters']['lasting_candy'] == 1
    assert len(first.reward.card_options) == 3
    second = _candy_victory(1)
    assert second.flags['relic_counters']['lasting_candy'] == 2
    assert len(second.reward.card_options) == 4
    assert second.reward.card_options[-1] in {'inflame', 'demon_form'}
    assert second.reward.card_options[-1] in second.reward.card_options[:3]
    assert len(second.reward.metadata['card_option_upgraded']) == 4
    assert serialize(_reload(second)) == serialize(second)


def test_lasting_candy_reroll_keeps_original_choice_count_then_adds_one_power():
    state = _candy_victory(1)
    assert state.reward.metadata['card_reroll_contexts']['primary']['count'] == 3
    result = _act(_reload(state), 'reroll_reward', 'reward:reroll:primary')
    assert len(result.reward.card_options) == 4
    assert result.reward.card_options[-1] in {'inflame', 'demon_form'}
    assert any(a['type'] == 'take_reward_card' for a in action_space(result))


def test_lasting_candy_single_rarity_power_pool_uses_only_item_and_upgrade_draws():
    state = _candy_victory(1)
    rng = random_from_state(state.rng)
    before = capture_random_state(rng)
    pool = tuple(card for card in t._combat_card_pool(state) if card.card_id == 'inflame')
    cards, rarities, upgrades = hooks.append_lasting_candy_power(
        state, rng, (), (), (), pool=pool)
    after = capture_random_state(rng)
    assert cards == ('inflame',) and rarities == ('uncommon',) and upgrades == (False,)
    assert after.streams['rewards'].draws - before.streams['rewards'].draws == 2
    assert state.flags['card_rarity_offset'] == _reload(state).flags['card_rarity_offset']


@pytest.mark.parametrize('unblockable,took_damage', [(False, True), (True, False)])
def test_lava_lamp_records_only_surviving_blockable_hp_loss(unblockable, took_damage):
    card = _simple('pay', effects={'hp_loss': 2} if unblockable else {'self_damage': 2})
    # Use the real command helper for a blockable command, as synthetic card
    # hp_loss explicitly denotes the source's unblockable property.
    state = _state(card, relics=('lava_lamp',))
    combat, events = t._damage_combat_player(state.combat, 2, 'source', unblockable=unblockable)
    combat = hooks.observe_lava_damage(combat, events)
    assert bool(combat.metadata.get('lava_lamp_took_damage')) == took_damage
    reward_state = state.model_copy(update={'combat': combat,
        'reward': RewardState(reward_id='combat', source='combat')})
    upgraded, _ = hooks.apply_late_group_relic(reward_state, 'lava_lamp', ('anger',), (False,),
        group_key='primary')
    assert upgraded == (not took_damage,)


def test_glitter_respects_existing_enchantment_and_noncardreward_factory_flags():
    state = _reward('anger', ('glitter',))
    upgraded, modifiers = hooks.apply_late_group_relic(state, 'glitter',
        ('anger', 'inflame', 'injury'), (False, False, False), group_key='primary',
        option_modifiers={'0': {'enchantments': ({'keyword': 'swift', 'amount': 1,
                                                'source_id': 'wing_charm'},)}})
    assert upgraded == (False, False, False)
    assert modifiers['0']['enchantments'][0]['keyword'] == 'swift'
    assert modifiers['1']['enchantments'][0]['keyword'] == 'glam'
    assert '2' not in modifiers
    _, ignored = hooks.apply_late_group_relic(state, 'glitter', ('anger',), (False,),
        group_key='primary', context={'no_modify_hooks': True})
    assert ignored == {}
