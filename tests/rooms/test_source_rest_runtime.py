"""Stable normal and mimicked rest callbacks through retained public actions."""
from __future__ import annotations

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import RoomKind, RunPhase
from sts2sim.engine import transitions as t
from sts2sim.engine.rng import random_from_state, random_stream
from sts2sim.mechanics.source_rng import draw_source_potion


def _reload(state):
    return load_state(serialize(state))


def _action(state, kind, target=None):
    return next(a for a in legal_actions(state) if a.type == kind
                and (target is None or a.target_id == target))


def _state(relics=(), *, modifiers=(), hp=10, max_hp=83, act=1):
    state = new_run('REST-CALLBACKS', 'IRONCLAD', 0, source_data={
        'relics': relics, 'run_modifiers': modifiers,
    })
    return state.model_copy(update={'phase': RunPhase.REST, 'ancient': None, 'act': act,
        'player': state.player.model_copy(update={'hp': hp, 'max_hp': max_hp})})


def _dense(state):
    node = state.map.nodes[0].model_copy(update={'kind': RoomKind.EVENT})
    return t._enter_event_room(state.model_copy(update={'flags': {
        **state.flags, 'event_id': 'DENSE_VEGETATION',
    }}), node)[0]


def test_rest_uses_pillow_before_healing_and_humidifier_after_healing():
    state = _state(('regal_pillow', 'stone_humidifier'))
    result = step(_reload(state), _action(state, 'rest'))
    assert (result.player.hp, result.player.max_hp) == (54, 88)
    events = result.replay_log[-1].events
    assert [(e.kind, e.amount) for e in events[:2]] == [
        ('rest_healed', 39), ('relic_max_hp_changed', 5)]
    assert serialize(step(state, _action(state, 'rest'))) == serialize(result)


def test_dense_mimic_calls_pillow_humidifier_despite_coffee_dripper_then_shows_fight():
    state = _dense(_state(('coffee_dripper', 'regal_pillow', 'stone_humidifier')))
    rest = next(option for option in state.event.options if option.option_id == 'rest')
    assert rest.metadata['source_numeric_effect']['heal'] == 39
    result = step(_reload(state), _action(state, 'choose_event', 'rest'))
    assert (result.player.hp, result.player.max_hp) == (54, 88)
    assert result.event.page_id == 'REST'
    assert [(a.type.value, a.target_id) for a in legal_actions(result)] == [
        ('choose_event', 'fight')]
    assert result.replay_log[-1].events[1].metadata['is_mimicked']


@pytest.mark.parametrize('mimicked,expected', [(False, (83, 83)), (True, (88, 88))])
def test_night_terrors_overrides_pillow_and_loses_max_hp_only_for_actual_rest(mimicked, expected):
    state = _state(('regal_pillow', 'stone_humidifier'), modifiers=('NIGHT_TERRORS',))
    if mimicked:
        state = _dense(state)
        action = _action(state, 'choose_event', 'rest')
    else:
        action = _action(state, 'rest')
    result = step(_reload(state), action)
    assert (result.player.hp, result.player.max_hp) == expected


def test_night_terrors_cannot_reduce_resting_max_hp_below_one():
    state = _state(modifiers=('night_terrors',), hp=1, max_hp=2)
    result = step(state, _action(state, 'rest'))
    assert (result.player.hp, result.player.max_hp) == (1, 1)


def test_tiny_mailbox_populates_two_independent_potion_rewards_without_auto_procurement():
    state = _state(('tiny_mailbox', 'sozu'))
    state = state.model_copy(update={'potions': ('fire_potion', 'blood_potion', 'energy_potion')})
    rng = random_stream(random_from_state(state.rng), 'rewards')
    pool, rarities = t._item_potion_generation_pool(state, in_combat=False)
    expected = tuple(draw_source_potion(rng, pool, rarities) for _ in range(2))
    result = step(_reload(state), _action(state, 'rest'))
    assert result.phase == RunPhase.REWARD
    assert result.reward.potion_ids == expected
    assert result.potions == state.potions
    assert result.rng.streams['rewards'].draws - state.rng.streams.get('rewards').draws == 4


def test_dream_catcher_uses_encounter_pity_upgrades_candy_and_reroll_context():
    state = _state(('dream_catcher', 'lasting_candy', 'driftwood'), act=2)
    state = state.model_copy(update={'flags': {**state.flags,
        'relic_counters': {'lasting_candy': 2}, 'card_rarity_offset': .9}})
    result = step(_reload(state), _action(state, 'rest'))
    assert result.phase == RunPhase.REWARD
    assert len(result.reward.card_options) == 4
    assert t._reward_card_spec(result, result.reward.card_options[-1])['type'].lower() == 'power'
    assert result.flags['card_rarity_offset'] != .9
    assert len(result.reward.metadata['card_option_upgraded']) == 4
    context = result.reward.metadata['card_reroll_contexts']['primary']
    assert (context['mode'], context['encounter'], context['count']) == ('combat', 'normal', 3)
    rerolled = step(_reload(result), _action(result, 'reroll_reward', 'reward:reroll:primary'))
    assert len(rerolled.reward.card_options) == 4
    assert serialize(step(state, _action(state, 'rest'))) == serialize(result)


def test_mimicked_reward_proceed_returns_to_dense_fight_page_after_reload():
    state = _dense(_state(('coffee_dripper', 'dream_catcher', 'tiny_mailbox')))
    result = step(state, _action(state, 'choose_event', 'rest'))
    assert result.phase == RunPhase.REWARD
    assert len(result.reward.card_options) == 3 and len(result.reward.potion_ids) == 2
    result = step(_reload(result), _action(result, 'proceed'))
    assert result.phase == RunPhase.EVENT and result.reward is None
    assert result.event.page_id == 'REST'
    fight = _action(result, 'choose_event', 'fight')
    result = step(_reload(result), fight)
    assert result.phase == RunPhase.COMBAT and result.combat is not None
