from __future__ import annotations

import random

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import CardInstance, CombatState, MonsterState, RunPhase
from sts2sim.engine import transitions as t
from sts2sim.engine.source_extra_rewards import KEY
from sts2sim.mechanics.source_rng import is_source_rng


class _AuditRng(random.Random):
    derivation = 'sts2.v0.107.1.utf16-djb2-splitmix64'

    def __init__(self):
        super().__init__(0)
        self.calls = []

    def random(self):
        self.calls.append(('float',))
        return .8

    def randrange(self, start, stop=None, step=1):
        self.calls.append(('integer', start, stop))
        return start

    def choice(self, sequence):
        self.calls.append(('choice',))
        return sequence[0]


def _fixture(*, relics=('lava_rock',), extras=()):
    state = new_run('SOURCE-POPULATION', 'IRONCLAD', 0)
    return state.model_copy(update={
        'phase': RunPhase.COMBAT, 'relics': relics,
        'flags': {**state.flags, 'combat_reward_encounter': 'boss'},
        'combat': CombatState(player=state.player,
            monsters=(MonsterState(monster_id='done', hp=0, max_hp=10),),
            metadata={KEY: extras}),
    })


def test_heist_existing_room_reward_populates_before_lava_rock_hook_relics():
    rng = _AuditRng()
    assert is_source_rng(rng)
    reward, _, flags = t._combat_reward_state(_fixture(extras=(
        {'kind': 'gold', 'amount': 7, 'source_id': 'heist'},)), rng)
    # GenerateRewardsFor rolls potion odds. Initial population is fixed Boss
    # gold then three card rarity/item/upgrade triplets. The room's fixed Gold
    # reward must then draw before Lava Rock's two relic-rarity draws.
    assert rng.calls[:2] == [('float',), ('integer', 100, 101)]
    assert rng.calls[2:11] == [('float',), ('choice',), ('float',)] * 3
    assert rng.calls[11:] == [('integer', 7, 8), ('float',), ('float',)]
    assert flags['lava_rock_triggered'] is True
    assert len(reward.relic_ids) == 2
    order = reward.metadata['source_reward_population_order']
    assert [(r['kind'], r['source_id']) for r in order[-3:]] == [
        ('gold', 'heist'), ('relic', 'lava_rock'), ('relic', 'lava_rock')]


def test_card_and_gold_room_extras_preserve_fifo_before_hook_additions():
    rng = _AuditRng()
    reward, _, _ = t._combat_reward_state(_fixture(extras=(
        {'kind': 'card', 'source_id': 'the_hunt', 'encounter': 'boss'},
        {'kind': 'gold', 'amount': 7, 'source_id': 'heist'},)), rng)
    assert rng.calls[11:20] == [('float',), ('choice',), ('float',)] * 3
    assert rng.calls[20:] == [('integer', 7, 8), ('float',), ('float',)]
    assert len(reward.card_option_groups) == 1
    assert reward.metadata['card_reroll_contexts']['group:0']['encounter'] == 'boss'


@pytest.mark.parametrize('relics,expected', [
    (('lava_rock', 'amethyst_aubergine'), ['lava_rock', 'lava_rock', 'amethyst_aubergine']),
    (('amethyst_aubergine', 'lava_rock'), ['amethyst_aubergine', 'lava_rock', 'lava_rock']),
])
def test_acquired_modifier_order_is_retained_and_appended_fixed_gold_is_not_repopulated(
    relics, expected,
):
    rng = _AuditRng()
    reward, _, _ = t._combat_reward_state(_fixture(relics=relics), rng)
    assert [r['source_id'] for r in reward.metadata['source_reward_population_order'][-3:]] == (
        expected)
    assert rng.calls[11:] == [('float',), ('float',)]
    amethyst = next(r for r in reward.metadata['source_reward_population_order']
                    if r['source_id'] == 'amethyst_aubergine')
    assert amethyst['populate_invoked'] is False
    assert reward.gold == 115


def _two_targets(*, hunt_kills_heist):
    state = new_run('SOURCE-EXTRA-FIFO', 'IRONCLAD', 0)
    hunt = t._card_from_spec(t._reward_card_spec(state, 'the_hunt'), 0)
    kill = CardInstance(instance_id='finish', card_id='finish', type='attack', target='enemy',
                        cost=0, effects={'damage': 999})
    fat = MonsterState(monster_id='fat', hp=1, max_hp=1,
        metadata={'source_monster_id': 'FAT_GREMLIN', 'stolen_gold': 7})
    other = MonsterState(monster_id='other', hp=1, max_hp=1)
    state = state.model_copy(update={
        'phase': RunPhase.COMBAT, 'relics': (),
        'flags': {**state.flags, 'combat_reward_encounter': 'normal'},
        'combat': CombatState(player=state.player, monsters=(fat, other), hand=(hunt, kill),
            metadata={KEY: (), 'source_reward_encounter': 'normal'}),
    })
    target = 'fat' if hunt_kills_heist else 'other'
    first = next(a for a in legal_actions(state)
                 if a.card_instance_id == hunt.instance_id and a.target_id == target)
    state = step(state, first)
    kinds = tuple(item['kind'] for item in state.combat.metadata[KEY])
    assert kinds == (('gold', 'card') if hunt_kills_heist else ('card',))
    target = 'other' if hunt_kills_heist else 'fat'
    last = next(a for a in legal_actions(state)
                if a.card_instance_id == 'finish' and a.target_id == target)
    restored = load_state(serialize(state))
    result = step(state, last)
    assert serialize(result) == serialize(step(restored, last))
    return result


@pytest.mark.parametrize('hunt_kills_heist', [False, True])
def test_public_fatal_and_heist_queue_preserves_source_before_death_and_cross_step_order(
    hunt_kills_heist,
):
    result = _two_targets(hunt_kills_heist=hunt_kills_heist)
    assert result.phase == RunPhase.REWARD
    source_order = [record['kind'] for record in
                    result.reward.metadata['source_reward_population_order']
                    if record['source_id'] in {'heist', 'the_hunt'}]
    assert source_order == (['gold', 'card'] if hunt_kills_heist else ['card', 'gold'])
    assert any(component['amount'] == 7 and component['source_id'] == 'heist'
               for component in result.reward.metadata['gold_reward_components'])
    assert len(result.reward.card_option_groups) == 1


@pytest.mark.parametrize('relics,expected', [
    (('black_star', 'white_star'), ['black_star', 'white_star']),
    (('white_star', 'black_star'), ['white_star', 'black_star']),
])
def test_black_star_is_a_hook_append_after_existing_extra_rewards(relics, expected):
    state = _fixture(relics=relics, extras=(
        {'kind': 'gold', 'amount': 7, 'source_id': 'heist'},))
    state = state.model_copy(update={
        'flags': {**state.flags, 'combat_reward_encounter': 'elite'}})
    rng = _AuditRng()
    reward, _, _ = t._combat_reward_state(state, rng)
    order = reward.metadata['source_reward_population_order']
    assert [(entry['kind'], entry['source_id']) for entry in order[:4]] == [
        ('gold', None), ('card', None), ('relic', None), ('gold', 'heist')]
    assert [entry['source_id'] for entry in order[4:]] == expected
    # Base cards consume their nine draws, then the base relic consumes one.
    # Heist must be populated before either acquired relic's added reward.
    assert rng.calls[11:13] == [('float',), ('integer', 7, 8)]
    assert len(reward.relic_ids) == 2
    assert len(set(reward.relic_ids)) == 2
    assert len(reward.card_option_groups) == 1
    assert reward.metadata['card_reroll_contexts']['group:0']['encounter'] == 'boss'


def _ticket_flags(state, remaining):
    ticket = 'wongos_mystery_ticket'
    return {**state.flags,
        'initialized_relic_lifecycles': (ticket,),
        'relic_counters': {ticket: remaining},
        'delayed_event_rewards': ({'reward_kind': 'random_relic', 'count': 3,
            'remaining_combats': remaining, 'qualifier': None,
            'metadata': {'source_relic_id': ticket}},)}


@pytest.mark.parametrize('relics,expected', [
    (('wongos_mystery_ticket', 'white_star'),
     ['wongos_mystery_ticket'] * 3 + ['white_star']),
    (('white_star', 'wongos_mystery_ticket'),
     ['white_star'] + ['wongos_mystery_ticket'] * 3),
])
def test_due_ticket_populates_in_acquired_hook_order_and_cannot_pay_twice(relics, expected):
    state = _fixture(relics=relics)
    state = state.model_copy(update={
        'flags': {**_ticket_flags(state, 0), 'combat_reward_encounter': 'elite'}})
    reward, events, flags = t._combat_reward_state(state, _AuditRng())
    order = reward.metadata['source_reward_population_order']
    assert [entry['source_id'] for entry in order[3:]] == expected
    assert len(reward.relic_ids) == len(set(reward.relic_ids)) == 4
    assert flags['wongos_mystery_ticket_used_up'] is True
    assert 'delayed_event_rewards' not in flags
    assert sum(event.kind == 'delayed_event_reward_ready' for event in events) == 1
    # A second combat retains the spent ticket and its disabled source hook.
    again, _, _ = t._combat_reward_state(state.model_copy(update={'flags': flags}), _AuditRng())
    assert len(again.relic_ids) == 1
    assert not any(entry['source_id'] == 'wongos_mystery_ticket'
                   for entry in again.metadata['source_reward_population_order'])


def test_final_boss_omits_base_and_room_extras_but_runs_ready_ticket_hook():
    state = _fixture(relics=('amethyst_aubergine', 'wongos_mystery_ticket'), extras=(
        {'kind': 'gold', 'amount': 7, 'source_id': 'royalties'},))
    state = state.model_copy(update={'act': 3, 'flags': _ticket_flags(state, 0)})
    rng = _AuditRng()
    reward, _, flags = t._combat_reward_state(state, rng)
    assert reward.metadata['final_act_boss'] is True
    assert reward.gold == 0 and not reward.card_options and reward.potion_id is None
    assert len(reward.relic_ids) == 3
    assert rng.calls == [('float',)] * 3
    assert flags['wongos_mystery_ticket_used_up'] is True
    assert [entry['source_id'] for entry in reward.metadata['source_reward_population_order']] == (
        ['wongos_mystery_ticket'] * 3)


def test_public_ticket_countdown_ticks_once_and_replays_before_source_ordered_payout():
    state = _fixture(relics=('wongos_mystery_ticket',))
    kill = CardInstance(instance_id='finish', card_id='finish', type='attack', target='enemy',
                        cost=0, effects={'damage': 999})
    state = state.model_copy(update={'flags': {
        **_ticket_flags(state, 2), 'combat_reward_encounter': 'normal'}})
    for remaining in (1, 0):
        state = state.model_copy(update={'phase': RunPhase.COMBAT, 'reward': None,
            'combat': CombatState(player=state.player, hand=(kill,),
                monsters=(MonsterState(monster_id='enemy', hp=1, max_hp=1),), metadata={
                    KEY: (), 'relic_counters': state.flags['relic_counters']})})
        action = next(a for a in legal_actions(state) if a.card_instance_id == 'finish')
        restored = load_state(serialize(state))
        result = step(state, action)
        assert serialize(result) == serialize(step(restored, action))
        state = result
        assert state.flags['relic_counters']['wongos_mystery_ticket'] == remaining
        if remaining:
            assert state.flags['delayed_event_rewards'][0]['remaining_combats'] == remaining
            assert not state.reward.relic_ids
        else:
            assert len(state.reward.relic_ids) == 3
            assert 'delayed_event_rewards' not in state.flags
            assert state.flags['wongos_mystery_ticket_used_up'] is True


def test_tutorial_elite_fixed_relic_precedes_manual_card_population_and_hook_appends():
    from sts2sim.engine.source_tutorial_rewards import HISTORY

    state = _fixture(relics=('black_star', 'silver_crucible'))
    state = state.model_copy(update={'flags': {**state.flags,
        'combat_reward_encounter': 'elite', 'number_of_runs': 0, 'revealed_epochs': (),
        HISTORY: {'1:elite': {'map_point_type': 'elite', 'room_types': ('elite',)}}}})
    rng = _AuditRng()
    reward, _, flags = t._combat_reward_state(state, rng)
    assert [(entry['kind'], entry['source_id']) for entry in
            reward.metadata['source_reward_population_order']] == [
        ('gold', None), ('potion', None), ('relic', None), ('card', None),
        ('relic', 'black_star')]
    assert reward.relic_ids[0] == 'vajra'
    assert reward.potion_id == 'block_potion'
    assert reward.card_options == ('bludgeon', 'pyre', 'evil_eye')
    assert reward.metadata['card_option_upgraded'] == (True, True, True)
    assert flags['silver_crucible_times_used'] == 1
    assert rng.calls == [('integer', 35, 46), ('float',)]
