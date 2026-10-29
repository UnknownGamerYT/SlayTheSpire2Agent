"""Single-entry Courier refill; preserve entry constraints without rerolling sales."""
from __future__ import annotations

from random import Random
from typing import Any

from sts2sim.mechanics.rewards import CardRarity, PotionRarity
from sts2sim.mechanics.shop_rooms import build_basic_shop_inventory
from sts2sim.mechanics.shops import (
    ShopInventoryPlan,
    ShopItem,
    ShopItemKind,
    price_shop_item,
)
from sts2sim.mechanics.source_rng import draw_source_potion, float32, source_float

from .models import RunState, ShopState
from .rng import random_stream


def restock(
    state: RunState, shop: ShopState, index: int, kind: ShopItemKind, rng: Random,
) -> tuple[ShopState, str, dict[str, Any]]:
    from . import transitions as t

    rng = random_stream(rng, "shops")
    flags = dict(state.flags)
    existing = {item.item_id for i, item in enumerate(shop.items)
                if i != index and not item.purchased}
    old = shop.items[index]
    if kind in {ShopItemKind.CARD, ShopItemKind.COLORLESS_CARD}:
        # Restock calls Populate before ClearAfterPurchase, so this entry's
        # original CreationResult is still part of the source blacklist.
        existing.add(old.item_id)
        pool_kind = ShopItemKind.COLORLESS_CARD if kind is ShopItemKind.COLORLESS_CARD else kind
        pool_key = ("shop_colorless_card_pool" if kind is ShopItemKind.COLORLESS_CARD
                    else "shop_card_pool")
        default = (t._default_shop_colorless_pool() if kind is ShopItemKind.COLORLESS_CARD
                   else t._default_shop_card_pool())
        pool = tuple(item for item in t._shop_item_pool(state, pool_key, default,
            default_kind=pool_kind) if item.item_id not in existing)
        rewards = random_stream(rng, "rewards")
        if kind is ShopItemKind.CARD:
            card_type = t._merchant_card_instance(state, old).type.value
            sample = source_float(rewards)
            offset = t._flag_float(state, "card_rare_offset_percent",
                                   t._flag_float(state, "card_rarity_offset", -.05) * 100) / 100
            rare = float32(float32(.045 if state.ascension >= 7 else .09) + float32(offset))
            rarity = (CardRarity.RARE if sample < rare else CardRarity.UNCOMMON
                      if sample < float32(rare + float32(.37)) else CardRarity.COMMON)
            cycle = (CardRarity.COMMON, CardRarity.UNCOMMON, CardRarity.RARE)
            start = cycle.index(rarity)
            eligible: list[ShopItem] = []
            for delta in range(3):
                eligible = [item for item in pool if item.rarity is cycle[(start + delta) % 3]
                            and (item.card_type or t._card_from_spec(
                                t._reward_card_spec(state, item.item_id), 0).type.value
                            ).lower() == card_type]
                if eligible:
                    break
        else:
            rarity = CardRarity(str(old.rarity))
            eligible = [item for item in pool if item.rarity is rarity]
        if not eligible:
            raise ValueError(f"no valid source merchant refill for {kind.value} slot {index}")
        chosen = rng.choice(eligible)
        # CreateForMerchant uses Shops for selection and consumes a Rewards
        # upgrade roll with a deliberately impossible -999999999 base chance.
        source_float(rewards)
        priced = price_shop_item(chosen, rng=rng, ascension_level=state.ascension)
    elif kind is ShopItemKind.POTION:
        pool = t._shop_item_pool(state, "shop_potion_pool", t._default_shop_potion_pool(),
                                 default_kind=kind)
        # Unlike the initial three-model batch, RestockAfterPurchase passes an
        # empty blacklist, even when another stocked entry has the same model.
        options = tuple(item.item_id for item in pool)
        rarities = {item.item_id: item.rarity.value if item.rarity is not None else "common"
                    for item in pool}
        identifier = draw_source_potion(rng, options, rarities)
        chosen = next(item for item in pool if item.item_id == identifier)
        if chosen.rarity is None:
            chosen = ShopItem(chosen.item_id, kind, PotionRarity.COMMON, chosen.price)
        priced = price_shop_item(chosen, rng=rng, ascension_level=state.ascension)
    elif kind is ShopItemKind.RELIC:
        inventory = build_basic_shop_inventory(rng,
            relic_pool=t._shop_restock_pool(t._shop_item_pool(state, "shop_relic_pool",
                t._default_shop_relic_pool(), default_kind=kind), existing),
            relic_selector=t._source_shop_relic_selector(state, flags, excluded_ids=existing),
            relic_rarity_rng=random_stream(rng, "rewards"),
            plan=ShopInventoryPlan(colored_cards=0, colorless_cards=0, relics=1,
                                   potions=0, include_card_removal=False),
            ascension_level=state.ascension)
        if not inventory.items:
            raise ValueError("source merchant relic refill produced no model")
        priced = inventory.items[0]
    else:
        raise ValueError(f"unsupported source merchant refill kind {kind.value}")
    fresh = t._shop_item_state_from_priced(index, priced)
    # RestockAfterPurchase clears IsOnSale before Populate; price hooks run
    # later through the shared awaited merchant callback continuation.
    fresh = fresh.model_copy(update={"metadata": {**fresh.metadata, "on_sale": False}})
    return shop.model_copy(update={"items": tuple(
        fresh if i == index else item for i, item in enumerate(shop.items))}), fresh.item_id, flags
