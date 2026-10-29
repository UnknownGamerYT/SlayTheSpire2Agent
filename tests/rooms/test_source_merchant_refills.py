"""Courier retains source entry constraints and consumes only refill RNG calls."""
from __future__ import annotations

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import MapEdgeState, MapNodeState, MapState, RunPhase, ShopItemState
from sts2sim.engine.rng import RunRng, capture_random_state
from sts2sim.engine.transitions import _modified_shop_price


def _shop(*extra):
    state = new_run("SOURCE_REFILL", "IRONCLAD", 0,
                    source_data={"relics": ("the_courier", *extra)})
    start = MapNodeState(node_id="start", kind="start", act=1, floor=0, lane=0)
    target = MapNodeState(node_id="shop", kind="shop", act=1, floor=1, lane=0)
    state = state.model_copy(update={"phase": RunPhase.MAP, "ancient": None,
        "map": MapState(nodes=(start, target),
            edges=(MapEdgeState(from_id="start", to_id="shop"),), current_node_id="start",
            completed_node_ids=("start",)),
        "player": state.player.model_copy(update={"gold": 10000})})
    return step(state, next(a for a in legal_actions(state) if a.type == "choose_node"))


def _buy(state, index):
    action = next(a for a in legal_actions(state)
                  if a.type == "shop_buy" and a.target_id == f"shop:{index}")
    result = step(state, action)
    assert serialize(result) == serialize(step(load_state(serialize(state)), action))
    return result


def _draws(state, key):
    return state.rng.streams[key].draws if key in state.rng.streams else 0


@pytest.mark.parametrize("index,kind", [(3, "skill"), (4, "power")])
def test_refill_keeps_original_colored_type_and_does_not_roll_sale_or_double_price(index, kind):
    before = _shop()
    assert before.shop.items[index].metadata["card"]["type"] == kind
    state = _buy(before, index)
    item = state.shop.items[index]
    assert item.metadata["card"]["type"] == kind
    assert not item.purchased and item.metadata["on_sale"] is False
    assert _draws(state, "shops") - _draws(before, "shops") == 2  # NextItem, one CalcCost.
    assert _draws(state, "rewards") - _draws(before, "rewards") == 2  # Rarity, upgrade.
    assert item.price == item.base_price * 4 // 5


def test_refilled_colorless_rare_entry_stays_rare_without_a_rarity_roll():
    before = _shop()
    assert before.shop.items[6].rarity == "rare"
    state = _buy(before, 6)
    assert state.shop.items[6].rarity == "rare"
    assert _draws(state, "shops") - _draws(before, "shops") == 2
    assert _draws(state, "rewards") - _draws(before, "rewards") == 1


def test_colored_refill_uses_saved_shop_rarity_offset_without_changing_it():
    before = _shop().model_copy(update={"rng": capture_random_state(RunRng("1"))})
    before = before.model_copy(update={"flags": {**before.flags, "card_rarity_offset": .4}})
    state = _buy(before, 3)
    # Independent source first Rewards sample .141266331: .09 +.4 wins Rare;
    # dropping the saved offset would incorrectly produce Uncommon.
    assert state.shop.items[3].rarity == "rare"
    assert state.flags["card_rarity_offset"] == .4
    assert _draws(state, "rewards") == 2 and _draws(state, "shops") == 2


def test_refilled_potion_can_duplicate_a_remaining_stocked_potion():
    before = _shop().model_copy(update={"rng": capture_random_state(RunRng("12"))})
    assert before.shop.items[12].item_id == "colorless_potion"
    state = _buy(before, 10)
    assert state.shop.items[10].item_id == state.shop.items[12].item_id == "colorless_potion"
    assert _draws(state, "shops") == 3  # Rarity, model, price.
    assert _draws(state, "rewards") == 0


def test_relic_refill_rolls_rarity_once_and_prices_one_back_bag_pull():
    before = _shop()
    state = _buy(before, 7)
    item = state.shop.items[7]
    assert before.shop.items[7].item_id == "candelabra"
    assert item.item_id == "war_paint" and not item.purchased
    assert item.item_id not in {x.item_id for x in before.shop.items[7:10]}
    assert _draws(state, "shops") - _draws(before, "shops") == 1
    assert _draws(state, "rewards") - _draws(before, "rewards") == 1
    assert item.price == item.base_price * 4 // 5


def test_source_sale_flag_survives_price_hooks_and_refill_clears_it():
    before = _shop("membership_card")
    sale = [item for item in before.shop.items[:5] if item.metadata.get("on_sale")]
    assert len(sale) == 1
    item = sale[0]
    assert item.price == (item.base_price // 2) * 2 // 5
    restocked = _buy(before, int(item.slot_id.split(":")[1]))
    fresh = restocked.shop.items[int(item.slot_id.split(":")[1])]
    assert fresh.metadata["on_sale"] is False
    assert fresh.price == fresh.base_price * 2 // 5
    # Source integer /2 happens before decimal discount multiplication.
    odd = ShopItemState(slot_id="shop:0", item_id="anger", kind="card",
                        price=37, base_price=75, metadata={"on_sale": True})
    assert _modified_shop_price(before, odd) == 14


def test_refill_applies_existing_merchant_creation_hooks_to_the_new_instance():
    before = _shop("toxic_egg", "fresnel_lens")
    before = before.model_copy(update={"flags": {**before.flags, "shop_card_pool": (
        {"id": "shrug_it_off", "kind": "card", "rarity": "common", "type": "skill"},
        {"id": "flame_barrier", "kind": "card", "rarity": "uncommon", "type": "skill"},
        {"id": "impervious", "kind": "card", "rarity": "rare", "type": "skill"},
    )}})
    state = _buy(before, 3)
    card = state.shop.items[3].metadata["card"]
    assert card["type"] == "skill" and card["upgraded"]
    assert len(card["enchantments"]) == 1
    assert card["enchantments"][0]["keyword"] == "nimble"


def test_refill_excludes_current_card_before_populate_clears_its_creation_result():
    before = _shop().model_copy(update={"rng": capture_random_state(RunRng("0"))})
    assert before.shop.items[3].item_id == "armaments"
    before = before.model_copy(update={"flags": {**before.flags, "shop_card_pool": (
        {"id": "armaments", "kind": "card", "rarity": "common", "type": "skill"},
        {"id": "shrug_it_off", "kind": "card", "rarity": "common", "type": "skill"},
    )}})
    # Seed0's Shops sample .1339 would choose Armaments if the old model
    # were omitted from MerchantCardEntry.Populate's current-entry blacklist.
    state = _buy(before, 3)
    assert state.shop.items[3].item_id == "shrug_it_off"
    assert _draws(state, "shops") == _draws(state, "rewards") == 2


def test_initial_colorless_fixed_rare_slot_does_not_fall_back_to_uncommon():
    from sts2sim.mechanics.rewards import CardRarity
    from sts2sim.mechanics.shop_rooms import build_basic_shop_inventory
    from sts2sim.mechanics.shops import ShopInventoryPlan, ShopItem, ShopItemKind

    bank = RunRng("FIXED_COLORLESS")
    with pytest.raises(ValueError, match="rare"):
        build_basic_shop_inventory(bank,
            colorless_card_pool=(ShopItem("finesse", ShopItemKind.COLORLESS_CARD,
                                          CardRarity.UNCOMMON),),
            plan=ShopInventoryPlan(colored_cards=0, colorless_cards=2, relics=0,
                                   potions=0, include_card_removal=False))
