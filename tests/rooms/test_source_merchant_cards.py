"""Merchant card creation modifiers are visible, saved, and retained on purchase."""
from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _state as _fight
from combat.test_glam import _play

from sts2sim import action_space, legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import RunPhase
from sts2sim.engine.models import CardInstance, ShopItemState, ShopState
from sts2sim.engine.transitions import (
    _apply_shop_price_modifiers,
    _enchant_card,
    _shop_gain_all_non_service_items,
)


def _shop(*relics: str):
    state = new_run(212, "TEST", 0, source_data={
        "cards": (
            {"id": "merchant_guard", "type": "skill", "cost": 1, "target": "self",
             "block": 5, "upgrade": {"block": "+3"}},
            {"id": "merchant_attack", "type": "attack", "target": "enemy",
             "damage": 6, "upgrade": {"damage": "+3"}},
        ),
        "shop_card_pool": [{"id": "merchant_attack", "kind": "card", "rarity": "common"}],
    })
    state = state.model_copy(update={
        "phase": RunPhase.SHOP, "relics": relics,
        "player": state.player.model_copy(update={"gold": 1000}),
    })
    shop = ShopState(node_id="shop", items=(
        ShopItemState(slot_id="shop:0", item_id="merchant_guard", kind="card",
                      rarity="common", price=50, base_price=50),
    ))
    return state.model_copy(update={"shop": _apply_shop_price_modifiers(state, shop)})


def _buy(state):
    return step(state, next(a for a in legal_actions(state) if a.type == "shop_buy"))


@pytest.mark.parametrize("relics,upgraded,enchanted", [
    ((), False, False), (("fresnel_lens",), False, True),
    (("toxic_egg", "fresnel_lens"), True, True),
    (("fresnel_lens", "toxic_egg"), True, True),
    (("glitter", "wing_charm", "silver_crucible", "silken_tress"), False, False),
])
def test_merchant_modifiers_are_visible_and_persist_once_on_purchase(
    relics, upgraded, enchanted,
):
    state = load_state(serialize(_shop(*relics)))
    offered = next(a for a in action_space(state) if a["type"] == "shop_buy")["card"]
    assert bool(offered["upgraded"]) is upgraded
    assert bool(offered["enchantments"]) is enchanted
    if enchanted:
        assert offered["enchantments"][0]["keyword"] == "nimble"
        assert offered["enchantments"][0]["amount"] == 2
    result = _buy(state)
    card = result.master_deck[-1]
    assert card.upgraded is upgraded
    assert len(card.enchantments) == int(enchanted)
    played = _play(_fight(card), card.instance_id)
    assert played.player.block == (8 if upgraded else 5) + (2 if enchanted else 0)


def test_merchant_update_preserves_existing_enchantment_and_is_idempotent():
    state = _shop()
    item = state.shop.items[0]
    saved = CardInstance.model_validate(item.metadata["card"])
    saved = _enchant_card(saved, "swift", 1, source_id="earlier")
    item = item.model_copy(update={"metadata": {"card": saved.model_dump(mode="json")}})
    state = state.model_copy(update={"relics": ("fresnel_lens",),
                                     "shop": state.shop.model_copy(update={"items": (item,)})})
    shop = _apply_shop_price_modifiers(state, state.shop)
    updated = CardInstance.model_validate(shop.items[0].metadata["card"])
    assert updated.enchantments == saved.enchantments
    assert _apply_shop_price_modifiers(state, shop) == shop


def test_courier_purchase_keeps_original_card_before_restocking_its_slot():
    state = _shop("fresnel_lens", "the_courier")
    result = _buy(load_state(serialize(state)))
    assert result.master_deck[-1].card_id == "merchant_guard"
    assert result.master_deck[-1].enchantments[0].amount == 2
    assert result.shop.items[0].item_id == "merchant_attack"
    assert result.shop.items[0].metadata["card"]["card_id"] == "merchant_attack"
    assert result.shop.items[0].metadata["card"]["enchantments"] == []


def test_lords_parasol_claims_the_modified_merchant_result():
    state = _shop("fresnel_lens", "toxic_egg")
    update, gained = _shop_gain_all_non_service_items(state, state.shop)
    assert gained == ["merchant_guard"]
    card = update["master_deck"][-1]
    assert card.upgraded and card.enchantments[0].amount == 2
    assert _play(_fight(card), card.instance_id).player.block == 10
