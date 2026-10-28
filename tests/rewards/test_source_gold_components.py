"""Separate stable GoldReward claims, callbacks, descriptors and persistence."""
from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _state
from combat.test_glam import _play

from sts2sim import action_space, legal_actions, serialize, step
from sts2sim.engine import RewardState, RunPhase
from sts2sim.engine.rng import RunRng, capture_random_state
from sts2sim.mechanics.combat_rewards import CombatRewardContext, draw_combat_reward
from sts2sim.mechanics.option_slots import reward_option_slots
from sts2sim.mechanics.planning_context import reward_plan_summary
from sts2sim.mechanics.rewards import EncounterType


def _action(state, kind, target=None):
    return next(a for a in legal_actions(state) if a.type == kind
                and (target is None or a.target_id == target))


def _victory(*, base_gold=11, relics=(), forced=False, escaped_stolen=None, event=None):
    kill = _simple('kill', effects={'all_damage': 999}, card_type='attack', target='all_enemies')
    state = _state(kill, relics=('amethyst_aubergine', *relics))
    flags = {**state.flags, 'combat_reward_gold': base_gold, 'combat_reward_gold_is_base': True,
             'combat_reward_forced': forced}
    if event:
        flags['event_fight_id'] = event
        flags['fake_merchant_unsold_relic_ids'] = ()
    combat = state.combat
    if escaped_stolen is not None:
        gremlin = combat.monsters[0].model_copy(update={
            'metadata': {**combat.monsters[0].metadata, 'source_monster_id': 'FAT_GREMLIN',
                         'escaped': True, 'stolen_gold': escaped_stolen},
            'hp': 0,
        })
        combat = combat.model_copy(update={'monsters': (*combat.monsters, gremlin)})
    state = state.model_copy(update={'flags': flags, 'combat': combat,
        'rng': capture_random_state(RunRng('GOLD',
            derivation='sts2.v0.107.1.utf16-djb2-splitmix64'))})
    return _play(_reload(state), 'kill')


def test_amethyst_creates_separate_gold_actions_and_per_gain_bowler_dragon_callbacks():
    state = _victory(relics=('bowler_hat', 'dragon_fruit'))
    assert state.reward.gold == 26
    assert [(c['amount'], c['source_id']) for c in
            state.reward.metadata['gold_reward_components']] == [(11, None),
                                                                (15, 'amethyst_aubergine')]
    assert [a.target_id for a in legal_actions(state) if a.type == 'take_reward_gold'] == [
        'reward:gold:0', 'reward:gold:1']
    before = state.player
    rng_before = state.rng
    first = step(_reload(state), _action(state, 'take_reward_gold', 'reward:gold:1'))
    assert first.player.gold - before.gold == 18
    assert first.player.max_hp - before.max_hp == 1
    assert first.rng == rng_before
    result = step(_reload(first), _action(first, 'take_reward_gold', 'reward:gold:0'))
    assert result.player.gold - before.gold == 31  # floor(11*1.25)+floor(15*1.25)
    assert result.player.max_hp - before.max_hp == 2
    assert result.reward.gold_claimed
    assert result.rng == rng_before
    assert not any(a.type == 'take_reward_gold' for a in legal_actions(result))
    assert serialize(_reload(result)) == serialize(result)


def test_mixed_component_skip_and_claim_remain_independent_after_reload():
    state = _victory(relics=('dragon_fruit',))
    before = state.player
    first = step(state, _action(state, 'skip_reward', 'reward:gold:0'))
    assert first.player == before
    assert [a.target_id for a in legal_actions(first) if a.type == 'take_reward_gold'] == [
        'reward:gold:1']
    result = step(_reload(first), _action(first, 'take_reward_gold', 'reward:gold:1'))
    assert result.player.gold - before.gold == 15
    assert result.player.max_hp - before.max_hp == 1
    plan = reward_plan_summary(serialize(result))
    assert plan['claimed_counts']['gold'] == 1
    assert plan['skipped_counts']['gold'] == 1
    assert plan['available_counts']['gold'] == 0


def test_forced_components_block_proceed_until_every_component_is_claimed():
    state = _victory(forced=True)
    # Isolate the forced GoldReward set from the unrelated generated card options.
    state = state.model_copy(update={'reward': state.reward.model_copy(update={
        'card_options': (), 'card_option_groups': (), 'relic_ids': (), 'potion_ids': (),
        'potion_id': None, 'relic_id': None,
    })})
    assert not any(a.type in {'skip_reward', 'proceed'} for a in legal_actions(state))
    first = step(state, _action(state, 'take_reward_gold', 'reward:gold:0'))
    assert not any(a.type == 'proceed' for a in legal_actions(first))
    final = step(_reload(first), _action(first, 'take_reward_gold', 'reward:gold:1'))
    assert any(a.type == 'proceed' for a in legal_actions(final))


def test_gold_descriptors_and_planning_slots_keep_unique_component_identity_and_value():
    state = _victory()
    actions = [a for a in action_space(state) if a['type'] == 'take_reward_gold']
    assert [(a['target_id'], a['reward_choice']['content_id'], a['reward_choice']['gold'],
             a['reward_choice']['selection_set_id']) for a in actions] == [
        ('reward:gold:0', 'gold:0', 11, 'gold:0'),
        ('reward:gold:1', 'gold:1', 15, 'gold:1')]
    plan = reward_plan_summary(serialize(state))
    assert plan['available_counts']['gold'] == 2
    assert [(a['option_slot']['content_id'], a['option_slot']['values']['gold_gain'])
            for a in actions] == [('gold:0', 11.), ('gold:1', 15.)]
    assert [c['content_id'] for c in plan['available_choices'] if c['kind'] == 'gold'] == [
        'gold:0', 'gold:1']
    slots = [slot for slot in reward_option_slots(serialize(state)) if slot['kind'] == 'gold']
    assert [(s['content_id'], s['values']['gold_gain']) for s in slots] == [
        ('gold:0', 11.), ('gold:1', 15.)]
    skips = [a for a in action_space(state) if a['type'] == 'skip_reward'
             and a['target_id'].startswith('reward:gold:')]
    assert [a['reward_choice']['selection_set_id'] for a in skips] == ['gold:0', 'gold:1']


@pytest.mark.parametrize('stolen,expected', [(0, (5, 15)), (10, (15,))])
def test_escaped_gremlin_reduces_only_base_gold_amethyst_survives(stolen, expected):
    state = _victory(escaped_stolen=stolen)
    assert state.reward.gold == sum(expected)
    assert tuple(c['amount'] for c in state.reward.metadata['gold_reward_components']) == expected


def test_fake_merchant_fixed_base_gold_stays_separate_from_amethyst():
    state = _victory(base_gold=300, event='fake_merchant')
    assert state.reward.gold == 315
    assert tuple(c['amount'] for c in state.reward.metadata['gold_reward_components']) == (300, 15)


def test_legacy_single_gold_reward_keeps_its_public_target_and_behavior():
    state = _victory().model_copy(update={'reward': RewardState(reward_id='single', gold=11),
                                        'phase': RunPhase.REWARD})
    assert [a.target_id for a in legal_actions(state) if a.type == 'take_reward_gold'] == [
        'reward:gold']
    result = step(state, _action(state, 'take_reward_gold', 'reward:gold'))
    assert result.reward.gold_claimed


@pytest.mark.parametrize('ascension,encounter,proportion,endpoints', [
    (2, EncounterType.NORMAL, 1., (10, 20)),
    (3, EncounterType.NORMAL, 1., (7, 15)),
    (3, EncounterType.ELITE, 1., (26, 33)),
    (3, EncounterType.BOSS, 1., (75, 75)),
    (0, EncounterType.NORMAL, .5, (5, 10)),
    (3, EncounterType.NORMAL, .5, (4, 8)),
])
def test_source_gold_factory_scales_endpoints_before_rng_with_bankers_rounding(
    ascension, encounter, proportion, endpoints,
):
    class Ranges(RunRng):
        def __init__(self):
            super().__init__('GOLD-RANGE')
            self.endpoints = []

        def randint(self, low, high):
            self.endpoints.append((low, high))
            return super().randint(low, high)

    rng = Ranges()
    bundle = draw_combat_reward(rng, card_pool=(), potion_pool=(), relic_pool=(),
        boss_relic_pool=(), context=CombatRewardContext('IRONCLAD', encounter=encounter,
            ascension_level=ascension, gold_proportion=proportion), card_count=0, relic_count=0)
    assert rng.endpoints == [endpoints]
    assert endpoints[0] <= bundle.gold <= endpoints[1]


def test_source_zero_gold_proportion_omits_reward_rng_draw_but_still_rolls_potion_odds():
    rng = RunRng('GOLD-ZERO')
    bundle = draw_combat_reward(rng, card_pool=(), potion_pool=(), relic_pool=(),
        boss_relic_pool=(), context=CombatRewardContext('IRONCLAD', gold_proportion=0.),
        card_count=0, relic_count=0)
    assert bundle.gold == 0
    assert capture_random_state(rng).draws == 1
