"""Shop room action availability and deterministic purchase results."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from random import Random
from typing import Any

from sts2sim.content.sources import STS1_COMPAT_SOURCE, SourceRef

from .rewards import CardRarity, PotionRarity, RelicRarity
from .shops import (
    DEFAULT_SHOP_PRICING_RULES,
    PricedShopItem,
    ShopInventory,
    ShopInventoryPlan,
    ShopItem,
    ShopItemKind,
    ShopPricingRules,
    build_shop_inventory,
    price_shop_item,
)
from .source_rng import float32, is_source_rng, named_stream, source_float

SHOP_CARD_RARITY_WEIGHTS = {
    CardRarity.COMMON: 5400,
    CardRarity.UNCOMMON: 3700,
    CardRarity.RARE: 900,
}
SHOP_CARD_RARITY_WEIGHTS_A7 = {
    CardRarity.COMMON: 5850,
    CardRarity.UNCOMMON: 3700,
    CardRarity.RARE: 450,
}
SHOP_RELIC_RARITY_WEIGHTS = {
    RelicRarity.COMMON: 5000,
    RelicRarity.UNCOMMON: 3300,
    RelicRarity.RARE: 1700,
}
SHOP_POTION_RARITY_WEIGHTS = {
    PotionRarity.COMMON: 6500,
    PotionRarity.UNCOMMON: 2500,
    PotionRarity.RARE: 1000,
}
CHARACTER_CARD_SLOT_TYPES = ("attack", "attack", "skill", "skill", "power")
COLORLESS_CARD_RARITY_SLOTS = (CardRarity.UNCOMMON, CardRarity.RARE)
SHOP_RELIC_BLACKLIST = frozenset(
    {
        "the_courier",
        "old_coin",
        "lucky_fysh",
        "bowler_hat",
        "amethyst_aubergine",
    }
)


class ShopRoomAction(str, Enum):
    BUY_ITEM = "buy_item"
    LEAVE = "leave"


@dataclass(frozen=True, slots=True)
class ShopRoomState:
    gold: int
    inventory: ShopInventory
    removable_card_ids: frozenset[str] = frozenset()
    purchased_item_indices: frozenset[int] = frozenset()
    card_removals_bought: int = 0


@dataclass(frozen=True, slots=True)
class ShopRoomChoice:
    action: ShopRoomAction
    item_index: int | None = None
    target_card_id: str | None = None


@dataclass(frozen=True, slots=True)
class ShopRoomResult:
    choice: ShopRoomChoice
    state: ShopRoomState
    gold_delta: int = 0
    purchased_item_index: int | None = None
    purchased_item: PricedShopItem | None = None
    removed_card_id: str | None = None
    left_shop: bool = False
    source: SourceRef = STS1_COMPAT_SOURCE


def _draw_without_replacement(
    rng: Random,
    pool: Sequence[ShopItem],
    count: int,
) -> tuple[ShopItem, ...]:
    candidates = list(pool)
    rng.shuffle(candidates)
    return tuple(candidates[: max(0, count)])


def _draw_weighted_without_replacement(
    rng: Random,
    pool: Sequence[ShopItem],
    count: int,
    weights: Mapping[Any, int],
) -> tuple[ShopItem, ...]:
    candidates = list(pool)
    selected: list[ShopItem] = []
    for _ in range(max(0, count)):
        if not candidates:
            break
        item = _weighted_choice_item(rng, candidates, weights)
        selected.append(item)
        candidates.remove(item)
    return tuple(selected)


def _weighted_choice_item(
    rng: Random,
    candidates: Sequence[ShopItem],
    weights: Mapping[Any, int],
) -> ShopItem:
    total = sum(max(1, weights.get(item.rarity, 1)) for item in candidates)
    roll = rng.randrange(total)
    cursor = 0
    for item in candidates:
        cursor += max(1, weights.get(item.rarity, 1))
        if roll < cursor:
            return item
    return candidates[-1]


def _draw_shop_relics(
    rng: Random,
    relic_pool: Sequence[ShopItem],
    count: int,
    selector: Callable[[str], ShopItem] | None = None,
    rarity_rng: Random | None = None,
) -> tuple[ShopItem, ...]:
    if count <= 0:
        return ()
    if selector is not None:
        from .relic_bag_runtime import source_relic_rarity

        source_rng = rarity_rng or rng
        normal_count = count - 1 if count >= 3 else count
        rarities = [source_relic_rarity(source_rng) for _ in range(normal_count)]
        if count >= 3:
            rarities.append("shop")
        return tuple(selector(rarity) for rarity in rarities)

    filtered_pool = [
        item for item in relic_pool if _normalized_item_id(item.item_id) not in SHOP_RELIC_BLACKLIST
    ]
    shop_relics = [item for item in filtered_pool if item.rarity is RelicRarity.SHOP]
    normal_relics = [item for item in filtered_pool if item.rarity is not RelicRarity.SHOP]
    normal_count = count - 1 if shop_relics and count >= 3 else count
    normal_candidates = normal_relics or ([] if count >= 3 else shop_relics)
    selected = list(
        _draw_weighted_without_replacement(
            rng,
            normal_candidates,
            normal_count,
            SHOP_RELIC_RARITY_WEIGHTS,
        )
    )
    if shop_relics and count >= 3:
        remaining_shop_relics = [
            item
            for item in shop_relics
            if item.item_id not in {chosen.item_id for chosen in selected}
        ]
        selected.extend(_draw_without_replacement(rng, remaining_shop_relics, 1))
    return tuple(selected[:count])


def _draw_character_cards(
    rng: Random,
    card_pool: Sequence[ShopItem],
    count: int,
    weights: Mapping[CardRarity, int],
) -> tuple[ShopItem, ...]:
    if count != len(CHARACTER_CARD_SLOT_TYPES):
        return _draw_weighted_without_replacement(rng, card_pool, count, weights)

    typed_candidates = {
        card_type: [
            item
            for item in card_pool
            if item.kind is ShopItemKind.CARD
            and item.card_type is not None
            and item.card_type.lower() == card_type
        ]
        for card_type in set(CHARACTER_CARD_SLOT_TYPES)
    }
    if any(
        len(typed_candidates[card_type]) < CHARACTER_CARD_SLOT_TYPES.count(card_type)
        for card_type in typed_candidates
    ):
        return _draw_weighted_without_replacement(rng, card_pool, count, weights)

    selected: list[ShopItem] = []
    selected_ids: set[str] = set()
    for card_type in CHARACTER_CARD_SLOT_TYPES:
        candidates = [
            item for item in typed_candidates[card_type] if item.item_id not in selected_ids
        ]
        if not candidates:
            return _draw_weighted_without_replacement(rng, card_pool, count, weights)
        item = _weighted_choice_item(rng, candidates, weights)
        selected.append(item)
        selected_ids.add(item.item_id)
    return tuple(selected)


def _draw_colorless_cards(
    rng: Random,
    colorless_card_pool: Sequence[ShopItem],
    count: int,
) -> tuple[ShopItem, ...]:
    if count != len(COLORLESS_CARD_RARITY_SLOTS):
        return _draw_weighted_without_replacement(
            rng,
            colorless_card_pool,
            count,
            SHOP_CARD_RARITY_WEIGHTS,
        )

    selected: list[ShopItem] = []
    selected_ids: set[str] = set()
    for rarity in COLORLESS_CARD_RARITY_SLOTS:
        candidates = [
            item
            for item in colorless_card_pool
            if item.rarity is rarity and item.item_id not in selected_ids
        ]
        if not candidates:
            return _draw_weighted_without_replacement(
                rng,
                colorless_card_pool,
                count,
                SHOP_CARD_RARITY_WEIGHTS,
            )
        item = _draw_without_replacement(rng, candidates, 1)[0]
        selected.append(item)
        selected_ids.add(item.item_id)
    return tuple(selected)


def _normalized_item_id(item_id: str) -> str:
    return item_id.lower().replace("'", "").replace(" ", "_").replace("-", "_")


def shop_card_rarity_weights(
    *,
    ascension_level: int = 0,
    rare_offset_percent: float = 0.0,
) -> dict[CardRarity, int]:
    base = SHOP_CARD_RARITY_WEIGHTS_A7 if ascension_level >= 7 else SHOP_CARD_RARITY_WEIGHTS
    rare_raw = base[CardRarity.RARE] + int(rare_offset_percent * 100)
    rare = max(0, rare_raw)
    uncommon = base[CardRarity.UNCOMMON] + min(0, rare_raw)
    uncommon = max(0, uncommon)
    common = max(0, 10000 - rare - uncommon)
    return {
        CardRarity.COMMON: common,
        CardRarity.UNCOMMON: uncommon,
        CardRarity.RARE: rare,
    }


def build_basic_shop_inventory(
    rng: Random,
    *,
    card_pool: Sequence[ShopItem] = (),
    colorless_card_pool: Sequence[ShopItem] = (),
    relic_pool: Sequence[ShopItem] = (),
    relic_selector: Callable[[str], ShopItem] | None = None,
    relic_rarity_rng: Random | None = None,
    potion_pool: Sequence[ShopItem] = (),
    plan: ShopInventoryPlan = ShopInventoryPlan(),
    ascension_level: int = 0,
    rare_offset_percent: float = 0.0,
    card_removals_bought: int = 0,
    rules: ShopPricingRules = DEFAULT_SHOP_PRICING_RULES,
) -> ShopInventory:
    """Build a priced shop inventory by deterministically sampling candidate pools."""

    if is_source_rng(rng):
        return _build_source_shop_inventory(
            rng,
            card_pool=card_pool,
            colorless_card_pool=colorless_card_pool,
            relic_pool=relic_pool,
            potion_pool=potion_pool,
            plan=plan,
            ascension_level=ascension_level,
            rare_offset_percent=rare_offset_percent,
            card_removals_bought=card_removals_bought,
            rules=rules,
            relic_selector=relic_selector,
            relic_rarity_rng=relic_rarity_rng,
        )
    card_weights = shop_card_rarity_weights(
        ascension_level=ascension_level,
        rare_offset_percent=rare_offset_percent,
    )

    return build_shop_inventory(
        rng,
        colored_cards=_draw_character_cards(
            rng,
            card_pool,
            plan.colored_cards,
            card_weights,
        ),
        colorless_cards=_draw_colorless_cards(
            rng,
            colorless_card_pool,
            plan.colorless_cards,
        ),
        relics=_draw_shop_relics(
            rng, relic_pool, plan.relics, relic_selector, relic_rarity_rng,
        ),
        potions=_draw_weighted_without_replacement(
            rng,
            potion_pool,
            plan.potions,
            SHOP_POTION_RARITY_WEIGHTS,
        ),
        plan=plan,
        ascension_level=ascension_level,
        card_removals_bought=card_removals_bought,
        rules=rules,
    )


def _build_source_shop_inventory(
    rng: Random,
    *,
    card_pool: Sequence[ShopItem],
    colorless_card_pool: Sequence[ShopItem],
    relic_pool: Sequence[ShopItem],
    potion_pool: Sequence[ShopItem],
    plan: ShopInventoryPlan,
    ascension_level: int,
    rare_offset_percent: float,
    card_removals_bought: int,
    rules: ShopPricingRules,
    relic_selector: Callable[[str], ShopItem] | None,
    relic_rarity_rng: Random | None,
) -> ShopInventory:
    """MerchantInventory population order, with separate Rewards and Shops streams."""
    rewards = named_stream(rng, "rewards")
    priced: list[PricedShopItem] = []
    used: set[str] = set()

    def add(item: ShopItem, *, sale: bool = False) -> None:
        entry = price_shop_item(
            item,
            rng=rng,
            ascension_level=ascension_level,
            card_removals_bought=card_removals_bought,
            rules=rules,
        )
        if sale:
            # MerchantCardEntry.SetOnSale calls CalcCost again, then integer /2.
            entry = price_shop_item(
                item,
                rng=rng,
                ascension_level=ascension_level,
                card_removals_bought=card_removals_bought,
                sale_percent=50,
                rules=rules,
            )
        priced.append(entry)

    def choose_card(
        pool: Sequence[ShopItem], rarity: CardRarity, card_type: str | None = None
    ) -> ShopItem | None:
        available = [
            item
            for item in pool
            if item.item_id not in used
            and (card_type is None or item.card_type is None or item.card_type.lower() == card_type)
        ]
        cycle = (CardRarity.COMMON, CardRarity.UNCOMMON, CardRarity.RARE)
        start = cycle.index(rarity)
        for offset in range(3):
            candidates = [item for item in available if item.rarity is cycle[(start + offset) % 3]]
            if candidates:
                chosen = rng.choice(candidates)
                used.add(chosen.item_id)
                # Merchant upgrade odds are forcibly negative, but still draw Rewards.
                source_float(rewards)
                return chosen
        return None

    sale_index = rng.randrange(plan.colored_cards) if plan.colored_cards else None
    for index in range(plan.colored_cards):
        sample = source_float(rewards)
        rare = float32(
            float32(0.045 if ascension_level >= 7 else 0.09) + float32(rare_offset_percent / 100)
        )
        rarity = (
            CardRarity.RARE
            if sample < rare
            else (
                CardRarity.UNCOMMON if sample < float32(rare + float32(0.37)) else CardRarity.COMMON
            )
        )
        item = choose_card(
            card_pool, rarity, CHARACTER_CARD_SLOT_TYPES[index % len(CHARACTER_CARD_SLOT_TYPES)]
        )
        if item is not None:
            add(item, sale=index == sale_index)
    for index in range(plan.colorless_cards):
        item = choose_card(
            colorless_card_pool,
            COLORLESS_CARD_RARITY_SLOTS[index % len(COLORLESS_CARD_RARITY_SLOTS)],
        )
        if item is not None:
            add(item)

    rarity_rng = relic_rarity_rng or rewards
    normal_count = max(0, plan.relics - 1) if plan.relics >= 3 else plan.relics
    relic_rarities: list[RelicRarity] = []
    for _ in range(normal_count):
        value = source_float(rarity_rng)
        relic_rarities.append(
            RelicRarity.COMMON
            if value < float32(0.5)
            else (RelicRarity.UNCOMMON if value < float32(0.83) else RelicRarity.RARE)
        )
    if plan.relics >= 3:
        relic_rarities.append(RelicRarity.SHOP)
    for relic_rarity in relic_rarities:
        if relic_selector is not None:
            item = relic_selector(relic_rarity.value)
        else:
            available = [item for item in relic_pool if item.item_id not in used]
            candidates = [item for item in available if item.rarity is relic_rarity]
            if not candidates:
                candidates = available
            if not candidates:
                break
            item = rng.choice(candidates)
        used.add(item.item_id)
        add(item)

    # PotionFactory draws all three models before constructors calculate prices.
    potions: list[ShopItem] = []
    available_potions = list(potion_pool)
    for _ in range(plan.potions):
        if not available_potions:
            raise ValueError("source merchant potion creation pool is empty")
        value = source_float(rng)
        potion_rarity = (
            PotionRarity.RARE
            if value <= float32(0.1)
            else (PotionRarity.UNCOMMON if value <= float32(0.35) else PotionRarity.COMMON)
        )
        candidates = [item for item in available_potions if item.rarity is potion_rarity]
        if not candidates:
            raise ValueError(f"no {potion_rarity.value} potion in source merchant creation pool")
        item = rng.choice(candidates)
        available_potions.remove(item)
        potions.append(item)
    for item in potions:
        add(item)
    if plan.include_card_removal:
        add(ShopItem("card_removal", ShopItemKind.CARD_REMOVAL))
    return ShopInventory(tuple(priced), plan.include_card_removal, rules.source)


def available_shop_actions(state: ShopRoomState) -> tuple[ShopRoomChoice, ...]:
    choices: list[ShopRoomChoice] = []
    for index, priced_item in enumerate(state.inventory.items):
        if index in state.purchased_item_indices or priced_item.price > state.gold:
            continue
        if priced_item.item.kind is ShopItemKind.CARD_REMOVAL:
            choices.extend(
                ShopRoomChoice(
                    action=ShopRoomAction.BUY_ITEM,
                    item_index=index,
                    target_card_id=card_id,
                )
                for card_id in sorted(state.removable_card_ids)
            )
        else:
            choices.append(ShopRoomChoice(action=ShopRoomAction.BUY_ITEM, item_index=index))
    choices.append(ShopRoomChoice(action=ShopRoomAction.LEAVE))
    return tuple(choices)


def _item_at(state: ShopRoomState, item_index: int) -> PricedShopItem:
    if item_index < 0 or item_index >= len(state.inventory.items):
        raise ValueError(f"Shop item index is out of range: {item_index}")
    return state.inventory.items[item_index]


def resolve_shop_action(choice: ShopRoomChoice, state: ShopRoomState) -> ShopRoomResult:
    if choice.action is ShopRoomAction.LEAVE:
        if choice.item_index is not None or choice.target_card_id is not None:
            raise ValueError("Leaving the shop does not use an item or target.")
        return ShopRoomResult(
            choice=choice,
            state=state,
            left_shop=True,
            source=state.inventory.source,
        )

    if choice.action is not ShopRoomAction.BUY_ITEM:
        raise ValueError(f"Unsupported shop action: {choice.action.value}")

    item_index = choice.item_index
    if item_index is None:
        raise ValueError("Buying from a shop requires an item_index.")
    priced_item = _item_at(state, item_index)
    if item_index in state.purchased_item_indices:
        raise ValueError(f"Shop item has already been purchased: {item_index}")
    if priced_item.price > state.gold:
        raise ValueError(
            f"Cannot afford shop item {item_index}: price={priced_item.price} gold={state.gold}"
        )

    removed_card_id: str | None = None
    next_removable_card_ids = state.removable_card_ids
    next_card_removals_bought = state.card_removals_bought
    if priced_item.item.kind is ShopItemKind.CARD_REMOVAL:
        if choice.target_card_id is None:
            raise ValueError("Card removal requires a target_card_id.")
        if choice.target_card_id not in state.removable_card_ids:
            raise ValueError(f"Card removal target is not removable: {choice.target_card_id}")
        removed_card_id = choice.target_card_id
        next_removable_card_ids = frozenset(
            card_id for card_id in state.removable_card_ids if card_id != choice.target_card_id
        )
        next_card_removals_bought += 1
    elif choice.target_card_id is not None:
        raise ValueError("Only card removal purchases use target_card_id.")

    next_state = replace(
        state,
        gold=state.gold - priced_item.price,
        removable_card_ids=next_removable_card_ids,
        purchased_item_indices=state.purchased_item_indices | frozenset({item_index}),
        card_removals_bought=next_card_removals_bought,
    )
    return ShopRoomResult(
        choice=choice,
        state=next_state,
        gold_delta=-priced_item.price,
        purchased_item_index=item_index,
        purchased_item=priced_item,
        removed_card_id=removed_card_id,
        source=state.inventory.source,
    )
