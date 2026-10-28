"""Prismatic Gem's ordered option-pool hook respects source flags and unlocks."""
from __future__ import annotations

import pytest

from sts2sim import load_state, new_run, serialize
from sts2sim.engine import source_run_passives as hooks
from sts2sim.engine import transitions as t
from sts2sim.engine.source_reward_reroll import _pool


def _state(relics=('prismatic_gem',), **flags):
    state = new_run('PRISMATIC-POOL', 'SILENT', 0)
    return state.model_copy(update={'relics': relics, 'flags': {**state.flags, **flags}})


def test_prismatic_uses_unlocked_character_registry_order_before_original_pool():
    state = _state(unlocked_character_ids=('SILENT', 'IRONCLAD'))
    original = t._combat_card_pool(state)
    expanded = hooks.modify_reward_pool(state, original)
    ironclad = t._combat_card_pool(state.model_copy(update={'character_id': 'IRONCLAD'}))
    assert tuple(card.card_id for card in expanded) == tuple(
        card.card_id for card in (*ironclad, *original))
    assert len({card.card_id for card in expanded}) == len(expanded)
    assert hooks.modify_reward_pool(load_state(serialize(state)), original) == expanded


@pytest.mark.parametrize('flags', [
    {'is_card_reward': False}, {'no_pool_modifications': True}, {'custom_pool': True},
])
def test_prismatic_pool_hook_obeys_creation_flags(flags):
    state = _state()
    original = t._combat_card_pool(state)
    assert hooks.modify_reward_pool(state, original, **flags) == original


def test_prismatic_preserves_colorless_only_pool_including_empty_registered_pool():
    state = _state()
    original = _pool(state, {'pools': ('colorless',), 'no_pool_modifications': True})
    assert hooks.modify_reward_pool(state, original, pool_ids=('colorless',)) == original
    assert hooks.modify_reward_pool(state, (), pool_ids=('colorless',)) == ()


@pytest.mark.parametrize('relics', [
    ('dingy_rug', 'prismatic_gem'), ('prismatic_gem', 'dingy_rug'),
])
def test_prismatic_and_dingy_preserve_all_character_then_colorless_pool_order(relics):
    state = _state(relics, unlocked_character_ids=('IRONCLAD', 'SILENT'))
    expanded = hooks.modify_reward_pool(state, t._combat_card_pool(state))
    colors = tuple(dict.fromkeys(card.color for card in expanded))
    assert colors == ('ironclad', 'silent', 'colorless')
    assert len({card.card_id for card in expanded}) == len(expanded)


def test_reroll_pool_extends_before_filter_and_preserves_explicit_custom_scope():
    state = _state()
    context = {'pools': ('character',), 'card_type': 'power', 'rarity': 'uncommon'}
    expanded = _pool(state, context)
    assert 'inflame' in {card.card_id for card in expanded}
    assert all(t._reward_card_spec(state, card.card_id)['type'].lower() == 'power'
               and card.rarity.value == 'uncommon' for card in expanded)
    assert {card.card_id for card in _pool(state, {**context, 'card_ids': ('inflame',)})} == {
        'inflame',
    }
