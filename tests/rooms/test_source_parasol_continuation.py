"""Awaited merchant callbacks, original stock iteration and Courier timing."""
from __future__ import annotations

import pytest
from rewards.test_remaining_relic_reward_interactions import _force_next_room

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import RoomKind
from sts2sim.engine.models import ShopItemState
from sts2sim.engine.transitions import _apply_lords_parasol


def _enter(*relics, shop_relics=('orrery', 'dollys_mirror'), potions=()):
    state = new_run('PARASOL', 'IRONCLAD', 0, source_data={
        'shop_plan': {'colored_cards': 1, 'colorless_cards': 1, 'relics': len(shop_relics),
                      'potions': 1, 'include_card_removal': True},
        'shop_card_pool': [{'id': 'anger', 'kind': 'card', 'rarity': 'common', 'price': 50}],
        'shop_colorless_card_pool': [{'id': 'finesse', 'kind': 'colorless_card',
                                     'rarity': 'uncommon', 'price': 100}],
        'shop_relic_pool': [{'id': relic, 'kind': 'relic', 'rarity': 'shop', 'price': 100}
                            for relic in shop_relics],
        'shop_potion_pool': [{'id': 'fire_potion', 'kind': 'potion', 'rarity': 'common',
                             'price': 30}],
    })
    state = state.model_copy(update={
        'relics': (*state.relics, *relics), 'potions': potions,
        'flags': {**state.flags, 'source_relic_bag_mode': 'fixture'},
        'player': state.player.model_copy(update={'gold': 1000}),
    })
    state = _force_next_room(state, RoomKind.SHOP)
    return _act(state, 'choose_node')


def _act(state, kind, *, source=None, target=None):
    action = next(a for a in legal_actions(state) if a.type == kind
                  and (source is None or a.payload.get('source_relic_id') == source)
                  and (target is None or a.target_id == target))
    before = serialize(state)
    result = step(state, action)
    assert serialize(state) == before
    assert serialize(result) == serialize(step(load_state(before), action))
    return result


def test_parasol_waits_for_relic_choice_then_child_rewards_then_mandatory_removal():
    state = _enter('lords_parasol')
    # Fixed seed stocks Dolly before Orrery. Cards are bought first, potions last.
    assert [item.item_id for item in state.shop.items[2:4]] == ['dollys_mirror', 'orrery']
    assert len(state.master_deck) == 12
    assert 'dollys_mirror' in state.relics and 'orrery' not in state.relics
    assert state.potions == ()
    assert {a.type for a in legal_actions(state)} == {'choose_card'}
    state = _act(state, 'choose_card', source='dollys_mirror')
    assert state.reward.reward_id == 'pickup:orrery'
    assert state.potions == ()
    assert not state.shop.items[-1].purchased
    state = _act(state, 'proceed')
    assert state.potions == ('fire_potion',)
    assert state.shop.items[-1].purchased
    assert state.shop.card_removals_bought == 1
    assert state.flags['shop_card_removals_bought'] == 1
    assert {a.type for a in legal_actions(state)} == {'choose_card'}
    assert all(a.payload['source_relic_id'] == 'lords_parasol' for a in legal_actions(state))
    state = _act(state, 'choose_card', source='lords_parasol')
    assert 'source_parasol_purchase' not in state.flags
    assert 'source_merchant_purchase_pending' not in state.flags
    assert len(state.master_deck) == 12  # Two stock cards, one Dolly copy, one removal.
    assert state.player.gold == 1000
    assert all(item.purchased for item in state.shop.items)


def test_parasol_keeps_multiple_custom_pickups_separate_and_in_stock_order():
    state = _enter('lords_parasol', shop_relics=('cauldron', 'orrery'))
    first, second = (item.item_id for item in state.shop.items[2:4])
    assert state.reward.reward_id == f'pickup:{first}'
    assert second not in state.relics
    state = _act(state, 'proceed')
    assert state.reward.reward_id == f'pickup:{second}'
    assert state.potions == ()
    state = _act(state, 'proceed')
    assert state.potions == ('fire_potion',)
    assert all(a.payload['source_relic_id'] == 'lords_parasol' for a in legal_actions(state))
    state = _act(state, 'choose_card', source='lords_parasol')
    assert state.relics.count(first) == state.relics.count(second) == 1


def test_parasol_visits_each_original_entry_once_when_courier_refills_it():
    state = _enter('lords_parasol', 'the_courier', 'maw_bank', shop_relics=('old_coin',))
    purchases = [event for event in state.replay_log[-1].events if event.kind == 'shop_item_bought']
    assert len(purchases) == 4
    assert len({event.metadata['slot_id'] for event in purchases}) == 4
    assert all(event.metadata['price'] == 0 for event in purchases)
    assert len(state.master_deck) == 12
    assert not state.flags.get('maw_bank_disabled')
    assert all(not item.purchased for item in state.shop.items[:-1])
    state = _act(state, 'choose_card', source='lords_parasol')
    assert len(state.master_deck) == 11
    assert 'source_parasol_purchase' not in state.flags
    assert state.relics.count('old_coin') == 1


@pytest.mark.parametrize('blocked', ['sozu', 'full'])
def test_parasol_leaves_failed_potion_purchases_stocked(blocked):
    extra = ('sozu',) if blocked == 'sozu' else ()
    potions = () if blocked == 'sozu' else ('strength_potion',) * 3
    state = _enter('lords_parasol', *extra, shop_relics=('anchor',), potions=potions)
    assert not state.shop.items[-2].purchased
    assert state.potions == potions
    assert all(a.payload['source_relic_id'] == 'lords_parasol' for a in legal_actions(state))


def test_newly_obtained_bing_does_not_duplicate_preceding_merchant_card_entries():
    state = _enter('lords_parasol', shop_relics=('bing_bong',))
    assert 'bing_bong' in state.relics
    assert len(state.master_deck) == 12
    assert not any(event.kind == 'relic_deck_card_duplicated'
                   and event.source_id == 'bing_bong' for event in state.replay_log[-1].events)


def test_paid_orrery_purchase_awaits_reward_callbacks_before_courier_refill_rng():
    state = _enter('the_courier', 'maw_bank', shop_relics=('orrery',))
    rewards_before = state.rng.streams['rewards'].draws
    shops_before = state.rng.streams['shops'].draws
    state = _act(state, 'shop_buy', target='shop:2')
    assert state.reward.reward_id == 'pickup:orrery'
    assert state.rng.streams['rewards'].draws == rewards_before + 45
    assert state.rng.streams['shops'].draws == shops_before
    assert state.shop.items[2].purchased  # Awaiting its callback; no refill yet.
    assert not state.flags.get('maw_bank_disabled')
    assert 'source_merchant_purchase_pending' in state.flags
    state = _act(state, 'proceed')
    assert state.rng.streams['rewards'].draws == rewards_before + 46
    assert state.rng.streams['shops'].draws > shops_before
    assert not state.shop.items[2].purchased
    assert state.flags['maw_bank_disabled']
    assert 'source_merchant_purchase_pending' not in state.flags


def test_forced_new_courier_entry_uses_owned_after_callback_predicate_for_own_slot():
    # Courier is excluded by normal merchant generation. A supplied entry
    # isolates MerchantEntry's generic post-obtain ShouldRefill callback.
    state = _enter(shop_relics=('anchor',))
    items = list(state.shop.items)
    items[2] = ShopItemState(slot_id='shop:2', item_id='the_courier', kind='relic',
                             rarity='rare', price=100, base_price=100)
    state = state.model_copy(update={'shop': state.shop.model_copy(update={'items': tuple(items)})})
    state = _act(state, 'shop_buy', target='shop:2')
    assert 'the_courier' in state.relics
    assert state.shop.items[2].item_id == 'anchor'
    assert not state.shop.items[2].purchased


def test_parasol_rejects_event_room_inventory_even_with_a_previous_merchant_object():
    state = _enter()
    game_map = state.map.model_copy(update={'nodes': tuple(
        node.model_copy(update={'kind': RoomKind.EVENT})
        if node.node_id == state.map.current_node_id else node for node in state.map.nodes
    )})
    state = state.model_copy(update={'map': game_map, 'relics': (*state.relics, 'lords_parasol')})
    result, events = _apply_lords_parasol(state, state.shop)
    assert result is None and events == ()


def test_parasol_applies_again_in_a_later_normal_merchant_room():
    state = _enter('lords_parasol', shop_relics=('anchor',))
    state = _act(state, 'choose_card', source='lords_parasol')
    first_size = len(state.master_deck)
    state = _force_next_room(state, RoomKind.SHOP)
    state = _act(state, 'choose_node')
    assert len(state.master_deck) == first_size + 2
    assert state.flags['shop_card_removals_bought'] == 2
    state = _act(state, 'choose_card', source='lords_parasol')
    assert len(state.master_deck) == first_size + 1
    assert 'source_parasol_purchase' not in state.flags


def test_maw_bank_saved_used_state_notifies_once_across_later_purchases():
    state = _enter('maw_bank', shop_relics=('anchor',))
    state = _act(state, 'shop_buy', target='shop:0')
    assert state.flags['maw_bank_disabled']
    assert sum(e.kind == 'maw_bank_disabled' for e in state.replay_log[-1].events) == 1
    state = _act(state, 'shop_buy', target='shop:1')
    assert not any(e.kind == 'maw_bank_disabled' for e in state.replay_log[-1].events)


def test_callback_completion_refreshes_stock_after_a_nested_egg_reward_without_courier():
    state = _enter(shop_relics=('calling_bell',))
    state = _act(state, 'shop_buy', target='shop:2')
    # Isolate the allowed late merchant creation hook after a nested relic claim.
    state = state.model_copy(update={'reward': state.reward.model_copy(update={
        'relic_ids': ('molten_egg',), 'claimed_relic_indices': (),
        'claimed_relic_ids': (), 'skipped_relic_indices': (),
    })})
    state = _act(state, 'take_reward_relic', target='reward:relic:0')
    state = _act(state, 'proceed')
    assert state.shop.items[0].metadata['card']['upgraded'] is True
    assert not state.shop.items[0].purchased
