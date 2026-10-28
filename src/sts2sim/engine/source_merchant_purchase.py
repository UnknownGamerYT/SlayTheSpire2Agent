"""Await merchant pickup callbacks before Courier refill and purchase completion."""
from __future__ import annotations

from typing import Any

from sts2sim.mechanics.shops import ShopItemKind

from .models import Action, EffectEvent, RunPhase, RunState
from .rng import capture_random_state, random_from_state, random_stream

PENDING = 'source_merchant_purchase_pending'


def _blocked(state: RunState) -> bool:
    from . import transitions as t

    return (state.phase != RunPhase.SHOP or bool(t._pending_relic_deck_choices(state))
            or bool(state.flags.get('source_pickup_reward_stack'))
            or bool(state.flags.get('scroll_boxes_pending')))


def _complete(state: RunState, purchase: dict[str, Any]) -> tuple[RunState, str | None]:
    from . import transitions as t

    identifier = None
    if (state.shop is not None and purchase['kind'] != 'card_removal'
            and t._has_relic(state, 'the_courier', 'courier')):
        rng = random_stream(random_from_state(state.rng), 'shops')
        shop, identifier, flags = t._restock_shop_slot(
            state, state.shop, int(purchase['index']), ShopItemKind(purchase['kind']), rng,
        )
        if shop is not None:
            shop = t._apply_shop_price_modifiers(state, shop)
        state = state.model_copy(update={
            'shop': shop, 'flags': flags, 'rng': capture_random_state(rng),
        })
    flags = dict(state.flags)
    flags.pop(PENDING, None)
    return state.model_copy(update={'flags': flags}), identifier


def prepare(
    before: RunState, state: RunState, action: Action, events: tuple[EffectEvent, ...], *,
    index: int, kind: str,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    from . import transitions as t
    from .source_card_reward_modifiers import apply_card_reward_modifiers

    # CardPileCmd.Add callbacks finish before MerchantEntry refills. Settle the
    # entry against the relics owned now, before later Parasol relic purchases.
    state, settled = t._apply_deck_add_relics(before, state, action, events)
    settled = tuple(event.model_copy(update={'metadata': {
        **event.metadata, 'source_deck_observers_applied': True,
    }}) for event in settled)
    state, modified = apply_card_reward_modifiers(state)
    purchase_event = next(event for event in settled
                          if event.kind in {'shop_item_bought', 'shop_card_removed'})
    purchase = {'index': index, 'kind': kind,
                'node_id': state.shop.node_id if state.shop else '',
                'event': purchase_event.model_dump(mode='json')}
    if _blocked(state):
        state = state.model_copy(update={'flags': {**state.flags, PENDING: purchase}})
        settled = tuple(event.model_copy(update={'metadata': {
            **event.metadata, 'merchant_callback_pending': True,
        }}) if event == purchase_event else event for event in settled)
        return state, (*settled, *modified)
    state, restocked = _complete(state, purchase)
    if restocked is not None:
        settled = tuple(event.model_copy(update={'metadata': {
            **event.metadata, 'restocked_item_id': restocked,
        }}) if event == purchase_event else event for event in settled)
    return state, (*settled, *modified)


def resume(state: RunState) -> tuple[RunState, tuple[EffectEvent, ...]]:
    purchase = state.flags.get(PENDING)
    if (not isinstance(purchase, dict) or _blocked(state) or state.shop is None
            or purchase['node_id'] != state.shop.node_id):
        return state, ()
    state, restocked = _complete(state, purchase)
    original = EffectEvent.model_validate(purchase['event'])
    return state, (EffectEvent(kind='merchant_purchase_completed',
        source_id=original.source_id, target_id=original.target_id, amount=original.amount,
        metadata={**original.metadata, 'restocked_item_id': restocked,
                  'after_pickup_completed': True}),)
