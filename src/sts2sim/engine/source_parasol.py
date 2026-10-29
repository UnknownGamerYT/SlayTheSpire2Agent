"""Lords Parasol's awaited, one-pass merchant entry purchases.

Each inventory entry is visited once. Pickup reward screens and deck choices
pause the serialized continuation before the next entry or Courier refill.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from sts2sim.mechanics.relics import RelicEffectMarker

from .models import Action, ActionType, EffectEvent, RoomKind, RunPhase, RunState, ShopState

_PENDING = 'source_parasol_purchase'


def begin(state: RunState, shop: ShopState) -> tuple[RunState | None, tuple[EffectEvent, ...]]:
    from . import transitions as t

    if not t._has_relic(state, 'lords_parasol', "lord's_parasol", 'lord_parasol'):
        return None, ()
    node = t._current_map_node(state)
    if node is not None and node.kind != RoomKind.SHOP:
        return None, ()
    kinds = ('card', 'colorless_card', 'relic', 'potion')
    indices = [i for kind in kinds for i, item in enumerate(shop.items)
               if item.kind == kind and not item.purchased]
    pending = {'node_id': shop.node_id, 'remaining_indices': indices,
               'item_ids': [], 'stage': 'purchases'}
    state = state.model_copy(update={
        'phase': RunPhase.SHOP, 'shop': shop, 'reward': None, 'combat': None,
        'flags': {**state.flags, _PENDING: deepcopy(pending)},
    })
    return resume(state)


def _blocked(state: RunState) -> bool:
    from . import transitions as t

    return (state.phase != RunPhase.SHOP or bool(t._pending_relic_deck_choices(state))
            or bool(state.flags.get('source_pickup_reward_stack'))
            or bool(state.flags.get('scroll_boxes_pending'))
            or bool(state.flags.get('source_merchant_purchase_pending')))


def _save(state: RunState, pending: dict[str, Any]) -> RunState:
    return state.model_copy(update={'flags': {**state.flags, _PENDING: deepcopy(pending)}})


def resume(state: RunState) -> tuple[RunState, tuple[EffectEvent, ...]]:
    from . import transitions as t
    from .source_card_reward_modifiers import apply_card_reward_modifiers

    raw = state.flags.get(_PENDING)
    if not isinstance(raw, dict) or _blocked(state) or state.shop is None:
        return state, ()
    pending = dict(raw)
    if pending['node_id'] != state.shop.node_id:
        return state, ()
    events: list[EffectEvent] = []
    remaining = list(pending['remaining_indices'])
    while remaining:
        index = int(remaining.pop(0))
        pending['remaining_indices'] = remaining
        state = _save(state, pending)
        assert state.shop is not None
        if index >= len(state.shop.items) or state.shop.items[index].purchased:
            continue
        item = state.shop.items[index]
        items = list(state.shop.items)
        items[index] = item.model_copy(update={'price': 0})
        free_state = state.model_copy(update={
            'shop': state.shop.model_copy(update={'items': tuple(items)}),
        })
        action = Action(type=ActionType.SHOP_BUY, target_id=f'shop:{index}')
        state, purchased = t._resolve_shop_action(free_state, action)
        succeeded = any(event.kind == 'shop_item_bought' for event in purchased)
        if not succeeded:
            # Full potion slots and Sozu leave the entry stocked; proceed to the next entry.
            if state.shop is not None:
                state = state.model_copy(update={'shop': t._apply_shop_price_modifiers(
                    state, state.shop,
                )})
            events.extend(purchased)
            continue
        pending['item_ids'] = [*pending['item_ids'], item.item_id]
        state = _save(state, pending)
        state, modified = apply_card_reward_modifiers(state)
        events.extend((*purchased, *modified))
        if _blocked(state):
            return state, tuple(events)
    if pending.get('stage') != 'removal':
        pending['stage'] = 'removal'
        state = _save(state, pending)
        assert state.shop is not None
        removal = next((i for i, item in enumerate(state.shop.items)
                        if item.kind == 'card_removal' and not item.purchased), None)
        # Synthetic TEST fixtures retain their established optional-service profile.
        if removal is not None and t._normalized_id(state.character_id) != 'test':
            marker = RelicEffectMarker(kind='remove_deck_cards', relic_id='lords_parasol',
                amount=1, metadata={'selection': 'chosen'})
            if t._relic_marker_deck_candidates(state.master_deck, marker):
                items = list(state.shop.items)
                items[removal] = items[removal].model_copy(update={'purchased': True})
                count = state.shop.card_removals_bought + 1
                state = state.model_copy(update={
                    'shop': state.shop.model_copy(update={'items': tuple(items),
                                                         'card_removals_bought': count}),
                    'flags': {**state.flags, 'shop_card_removals_bought': count},
                })
                state, emitted = t._queue_relic_deck_choice(state, 'lords_parasol', marker)
                events.extend(emitted)
                return state, tuple(events)
    flags = dict(state.flags)
    flags.pop(_PENDING, None)
    state = state.model_copy(update={'flags': flags})
    events.append(EffectEvent(kind='lords_parasol_claimed_shop', source_id='lords_parasol',
        amount=len(pending['item_ids']), metadata={'item_ids': tuple(pending['item_ids'])}))
    return state, tuple(events)
