"""Shop pricing and inventory plan helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from random import Random

from sts2sim.content.sources import STS1_COMPAT_SOURCE, SourceRef

from .rewards import CardRarity, PotionRarity, RelicRarity


class ShopItemKind(str, Enum):
    CARD = "card"
    COLORLESS_CARD = "colorless_card"
    POTION = "potion"
    RELIC = "relic"
    CARD_REMOVAL = "card_removal"


ShopRarity = CardRarity | PotionRarity | RelicRarity | None


def _default_card_prices() -> dict[CardRarity, tuple[int, int]]:
    return {
        CardRarity.COMMON: (48, 52),
        CardRarity.UNCOMMON: (71, 79),
        CardRarity.RARE: (142, 158),
    }


def _default_colorless_prices() -> dict[CardRarity, tuple[int, int]]:
    return {
        CardRarity.COMMON: (55, 60),
        CardRarity.UNCOMMON: (82, 91),
        CardRarity.RARE: (163, 182),
    }


def _default_potion_prices() -> dict[PotionRarity, tuple[int, int]]:
    return {
        PotionRarity.COMMON: (48, 52),
        PotionRarity.UNCOMMON: (71, 79),
        PotionRarity.RARE: (95, 105),
    }


def _default_relic_prices() -> dict[RelicRarity, tuple[int, int]]:
    return {
        RelicRarity.COMMON: (149, 201),
        RelicRarity.UNCOMMON: (191, 259),
        RelicRarity.RARE: (234, 316),
        RelicRarity.SHOP: (170, 230),
    }


@dataclass(frozen=True, slots=True)
class ShopPricingRules:
    card_prices: Mapping[CardRarity, tuple[int, int]] = field(
        default_factory=_default_card_prices
    )
    colorless_card_prices: Mapping[CardRarity, tuple[int, int]] = field(
        default_factory=_default_colorless_prices
    )
    potion_prices: Mapping[PotionRarity, tuple[int, int]] = field(
        default_factory=_default_potion_prices
    )
    relic_prices: Mapping[RelicRarity, tuple[int, int]] = field(
        default_factory=_default_relic_prices
    )
    removal_base_price: int = 75
    removal_increment: int = 25
    ascension_removal_base_price: int = 100
    ascension_removal_increment: int = 50
    min_price: int = 0
    source: SourceRef = STS1_COMPAT_SOURCE


@dataclass(frozen=True, slots=True)
class ShopItem:
    item_id: str
    kind: ShopItemKind
    rarity: ShopRarity = None
    price: int | None = None
    source: SourceRef = STS1_COMPAT_SOURCE
    card_type: str | None = None


@dataclass(frozen=True, slots=True)
class PricedShopItem:
    item: ShopItem
    price: int
    base_price: int
    source: SourceRef


@dataclass(frozen=True, slots=True)
class ShopInventoryPlan:
    colored_cards: int = 5
    colorless_cards: int = 2
    relics: int = 3
    potions: int = 3
    include_card_removal: bool = True


@dataclass(frozen=True, slots=True)
class ShopInventory:
    items: tuple[PricedShopItem, ...]
    includes_card_removal: bool
    source: SourceRef


DEFAULT_SHOP_PRICING_RULES = ShopPricingRules()


def roll_base_shop_price(
    rng: Random,
    kind: ShopItemKind,
    rarity: ShopRarity,
    *,
    rules: ShopPricingRules = DEFAULT_SHOP_PRICING_RULES,
) -> int:
    low: int
    high: int
    if kind is ShopItemKind.CARD:
        if not isinstance(rarity, CardRarity):
            raise ValueError("Card shop items require CardRarity.")
        low, high = rules.card_prices[rarity]
    elif kind is ShopItemKind.COLORLESS_CARD:
        if not isinstance(rarity, CardRarity):
            raise ValueError("Colorless card shop items require CardRarity.")
        low, high = rules.colorless_card_prices[rarity]
    elif kind is ShopItemKind.POTION:
        if not isinstance(rarity, PotionRarity):
            raise ValueError("Potion shop items require PotionRarity.")
        low, high = rules.potion_prices[rarity]
    elif kind is ShopItemKind.RELIC:
        if not isinstance(rarity, RelicRarity):
            raise ValueError("Relic shop items require RelicRarity.")
        low, high = rules.relic_prices[rarity]
    else:
        raise ValueError(f"Cannot roll base price for {kind.value}.")
    if low > high:
        raise ValueError(f"Invalid price range for {kind.value}/{rarity}: {(low, high)}")
    return rng.randint(low, high)


def price_shop_item(
    item: ShopItem,
    *,
    rng: Random,
    ascension_level: int = 0,
    sale_percent: int = 0,
    card_removals_bought: int = 0,
    rules: ShopPricingRules = DEFAULT_SHOP_PRICING_RULES,
) -> PricedShopItem:
    if item.kind is ShopItemKind.CARD_REMOVAL:
        if ascension_level >= 6:
            base_price = (
                rules.ascension_removal_base_price
                + rules.ascension_removal_increment * card_removals_bought
            )
        else:
            base_price = rules.removal_base_price + rules.removal_increment * card_removals_bought
    elif item.price is not None:
        base_price = item.price
    else:
        base_price = roll_base_shop_price(rng, item.kind, item.rarity, rules=rules)

    price = base_price
    if sale_percent:
        price = int(price * max(0, 100 - sale_percent) / 100)
    return PricedShopItem(
        item=item,
        price=max(rules.min_price, price),
        base_price=base_price,
        source=rules.source,
    )


def build_shop_inventory(
    rng: Random,
    *,
    colored_cards: Sequence[ShopItem] = (),
    colorless_cards: Sequence[ShopItem] = (),
    relics: Sequence[ShopItem] = (),
    potions: Sequence[ShopItem] = (),
    plan: ShopInventoryPlan = ShopInventoryPlan(),
    ascension_level: int = 0,
    card_removals_bought: int = 0,
    rules: ShopPricingRules = DEFAULT_SHOP_PRICING_RULES,
) -> ShopInventory:
    """Build a priced inventory from already-selected candidate items."""

    selected_colored_cards = list(colored_cards[: plan.colored_cards])
    sale_index = (
        rng.randrange(len(selected_colored_cards))
        if len(selected_colored_cards) >= 5
        else None
    )
    selected: list[ShopItem] = []
    selected.extend(selected_colored_cards)
    selected.extend(colorless_cards[: plan.colorless_cards])
    selected.extend(relics[: plan.relics])
    selected.extend(potions[: plan.potions])
    if plan.include_card_removal:
        selected.append(ShopItem("card_removal", ShopItemKind.CARD_REMOVAL))

    priced = tuple(
        price_shop_item(
            item,
            rng=rng,
            ascension_level=ascension_level,
            sale_percent=50 if index == sale_index else 0,
            card_removals_bought=card_removals_bought,
            rules=rules,
        )
        for index, item in enumerate(selected)
    )
    return ShopInventory(
        items=priced,
        includes_card_removal=plan.include_card_removal,
        source=rules.source,
    )
