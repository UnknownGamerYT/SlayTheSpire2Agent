from __future__ import annotations

from random import Random

from sts2sim.content.sources import STS1_COMPAT_SOURCE
from sts2sim.mechanics import (
    CardRarity,
    PotionRarity,
    PricedShopItem,
    RelicRarity,
    ShopInventory,
    ShopInventoryPlan,
    ShopItem,
    ShopItemKind,
    ShopRoomAction,
    ShopRoomChoice,
    ShopRoomState,
    available_shop_actions,
    build_basic_shop_inventory,
    resolve_shop_action,
)


def _priced(item: ShopItem, price: int) -> PricedShopItem:
    return PricedShopItem(
        item=item,
        price=price,
        base_price=price,
        source=STS1_COMPAT_SOURCE,
    )


def test_basic_shop_inventory_prices_are_seed_deterministic() -> None:
    cards = (
        ShopItem("strike_plus", ShopItemKind.CARD, CardRarity.COMMON),
        ShopItem("uppercut", ShopItemKind.CARD, CardRarity.UNCOMMON),
        ShopItem("feed", ShopItemKind.CARD, CardRarity.RARE),
    )
    relics = (
        ShopItem("anchor", ShopItemKind.RELIC, RelicRarity.COMMON),
        ShopItem("kunai", ShopItemKind.RELIC, RelicRarity.UNCOMMON),
    )
    potions = (
        ShopItem("fire_potion", ShopItemKind.POTION, PotionRarity.COMMON),
        ShopItem("focus_potion", ShopItemKind.POTION, PotionRarity.RARE),
    )
    plan = ShopInventoryPlan(
        colored_cards=2,
        colorless_cards=0,
        relics=1,
        potions=1,
        include_card_removal=True,
    )

    first = build_basic_shop_inventory(
        Random(19),
        card_pool=cards,
        relic_pool=relics,
        potion_pool=potions,
        plan=plan,
    )
    second = build_basic_shop_inventory(
        Random(19),
        card_pool=cards,
        relic_pool=relics,
        potion_pool=potions,
        plan=plan,
    )
    payload = tuple((item.item.item_id, item.item.kind.value, item.price) for item in first.items)

    assert first == second
    assert payload == (
        ("strike_plus", "card", 49),
        ("feed", "card", 154),
        ("anchor", "relic", 171),
        ("fire_potion", "potion", 52),
        ("card_removal", "card_removal", 75),
    )


def test_basic_shop_inventory_forces_third_relic_to_shop_rarity_when_available() -> None:
    relics = (
        ShopItem("anchor", ShopItemKind.RELIC, RelicRarity.COMMON, price=100),
        ShopItem("kunai", ShopItemKind.RELIC, RelicRarity.UNCOMMON, price=100),
        ShopItem("membership_card", ShopItemKind.RELIC, RelicRarity.SHOP, price=100),
    )
    inventory = build_basic_shop_inventory(
        Random(3),
        relic_pool=relics,
        plan=ShopInventoryPlan(
            colored_cards=0,
            colorless_cards=0,
            relics=3,
            potions=0,
            include_card_removal=False,
        ),
    )

    assert inventory.items[-1].item.rarity is RelicRarity.SHOP
    assert inventory.items[-1].item.item_id == "membership_card"


def test_full_shop_inventory_uses_typed_card_slots_sale_and_fixed_colorless_rarities() -> None:
    cards = (
        ShopItem(
            "attack_common",
            ShopItemKind.CARD,
            CardRarity.COMMON,
            price=50,
            card_type="attack",
        ),
        ShopItem(
            "attack_uncommon",
            ShopItemKind.CARD,
            CardRarity.UNCOMMON,
            price=75,
            card_type="attack",
        ),
        ShopItem("attack_rare", ShopItemKind.CARD, CardRarity.RARE, price=150, card_type="attack"),
        ShopItem("skill_common", ShopItemKind.CARD, CardRarity.COMMON, price=50, card_type="skill"),
        ShopItem(
            "skill_uncommon",
            ShopItemKind.CARD,
            CardRarity.UNCOMMON,
            price=75,
            card_type="skill",
        ),
        ShopItem("skill_rare", ShopItemKind.CARD, CardRarity.RARE, price=150, card_type="skill"),
        ShopItem(
            "power_uncommon",
            ShopItemKind.CARD,
            CardRarity.UNCOMMON,
            price=75,
            card_type="power",
        ),
        ShopItem("power_rare", ShopItemKind.CARD, CardRarity.RARE, price=150, card_type="power"),
    )
    colorless = (
        ShopItem("colorless_common", ShopItemKind.COLORLESS_CARD, CardRarity.COMMON, price=55),
        ShopItem("colorless_uncommon", ShopItemKind.COLORLESS_CARD, CardRarity.UNCOMMON, price=82),
        ShopItem("colorless_rare", ShopItemKind.COLORLESS_CARD, CardRarity.RARE, price=163),
    )

    inventory = build_basic_shop_inventory(
        Random(2),
        card_pool=cards,
        colorless_card_pool=colorless,
        relic_pool=(),
        potion_pool=(),
        plan=ShopInventoryPlan(
            colored_cards=5,
            colorless_cards=2,
            relics=0,
            potions=0,
            include_card_removal=False,
        ),
    )

    colored = inventory.items[:5]
    colorless_items = inventory.items[5:7]

    assert [item.item.card_type for item in colored].count("attack") == 2
    assert [item.item.card_type for item in colored].count("skill") == 2
    assert [item.item.card_type for item in colored].count("power") == 1
    assert sorted(item.item.rarity for item in colorless_items) == [
        CardRarity.RARE,
        CardRarity.UNCOMMON,
    ]
    assert sum(item.price == int(item.base_price * 0.5) for item in colored) == 1


def test_shop_inventory_uses_ascension_six_card_removal_inflation() -> None:
    inventory = build_basic_shop_inventory(
        Random(1),
        plan=ShopInventoryPlan(
            colored_cards=0,
            colorless_cards=0,
            relics=0,
            potions=0,
            include_card_removal=True,
        ),
        ascension_level=6,
        card_removals_bought=2,
    )

    assert inventory.items[-1].price == 200


def test_available_shop_actions_filters_to_affordable_unpurchased_items() -> None:
    inventory = ShopInventory(
        items=(
            _priced(ShopItem("strike_plus", ShopItemKind.CARD, CardRarity.COMMON), 80),
            _priced(ShopItem("kunai", ShopItemKind.RELIC, RelicRarity.UNCOMMON), 180),
            _priced(ShopItem("card_removal", ShopItemKind.CARD_REMOVAL), 75),
        ),
        includes_card_removal=True,
        source=STS1_COMPAT_SOURCE,
    )
    state = ShopRoomState(
        gold=100,
        inventory=inventory,
        removable_card_ids=frozenset({"strike", "defend"}),
        purchased_item_indices=frozenset({0}),
    )

    actions = available_shop_actions(state)

    assert ShopRoomChoice(ShopRoomAction.BUY_ITEM, item_index=0) not in actions
    assert ShopRoomChoice(ShopRoomAction.BUY_ITEM, item_index=1) not in actions
    assert ShopRoomChoice(
        ShopRoomAction.BUY_ITEM,
        item_index=2,
        target_card_id="defend",
    ) in actions
    assert ShopRoomChoice(
        ShopRoomAction.BUY_ITEM,
        item_index=2,
        target_card_id="strike",
    ) in actions
    assert actions[-1] == ShopRoomChoice(ShopRoomAction.LEAVE)


def test_buying_relic_reduces_gold_and_marks_item_purchased() -> None:
    relic = _priced(ShopItem("anchor", ShopItemKind.RELIC, RelicRarity.COMMON), 135)
    state = ShopRoomState(
        gold=200,
        inventory=ShopInventory(
            items=(relic,),
            includes_card_removal=False,
            source=STS1_COMPAT_SOURCE,
        ),
    )

    result = resolve_shop_action(
        ShopRoomChoice(ShopRoomAction.BUY_ITEM, item_index=0),
        state,
    )

    assert result.state.gold == 65
    assert result.gold_delta == -135
    assert result.purchased_item_index == 0
    assert result.purchased_item == relic
    assert result.state.purchased_item_indices == frozenset({0})
    assert state.gold == 200
    assert state.purchased_item_indices == frozenset()


def test_buying_card_removal_records_removed_card() -> None:
    removal = _priced(ShopItem("card_removal", ShopItemKind.CARD_REMOVAL), 75)
    state = ShopRoomState(
        gold=125,
        inventory=ShopInventory(
            items=(removal,),
            includes_card_removal=True,
            source=STS1_COMPAT_SOURCE,
        ),
        removable_card_ids=frozenset({"strike", "defend"}),
    )

    result = resolve_shop_action(
        ShopRoomChoice(
            ShopRoomAction.BUY_ITEM,
            item_index=0,
            target_card_id="strike",
        ),
        state,
    )

    assert result.state.gold == 50
    assert result.removed_card_id == "strike"
    assert result.state.removable_card_ids == frozenset({"defend"})
    assert result.state.card_removals_bought == 1


def test_leave_shop_returns_result_without_mutating_state() -> None:
    inventory = ShopInventory(
        items=(),
        includes_card_removal=False,
        source=STS1_COMPAT_SOURCE,
    )
    state = ShopRoomState(gold=42, inventory=inventory)

    result = resolve_shop_action(ShopRoomChoice(ShopRoomAction.LEAVE), state)

    assert result.left_shop is True
    assert result.state is state
    assert result.gold_delta == 0
    assert result.purchased_item is None
    assert result.removed_card_id is None
