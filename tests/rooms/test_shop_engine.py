from __future__ import annotations

from sts2sim import legal_actions, new_run, step


def _action(state, action_type: str, target_contains: str | None = None):
    return next(
        action
        for action in legal_actions(state)
        if action.type == action_type
        and (target_contains is None or (action.target_id and target_contains in action.target_id))
    )


def _play_kill_card(state):
    return next(action for action in legal_actions(state) if action.type == "play_card")


def _enter_shop(
    *,
    relics=(),
    potions=(),
    hp: int = 80,
    shop_relic_pool=None,
    ascension: int = 0,
):
    for seed in range(50):
        state = _try_enter_shop(
            seed,
            relics=relics,
            potions=potions,
            hp=hp,
            shop_relic_pool=shop_relic_pool,
            ascension=ascension,
        )
        if state is not None:
            return state
    raise AssertionError("Did not reach a shop for any searched seed")


def _try_enter_shop(
    seed: int,
    *,
    relics=(),
    potions=(),
    hp: int = 80,
    shop_relic_pool=None,
    ascension: int = 0,
):
    relic_pool = shop_relic_pool or [
        {"id": "anchor", "kind": "relic", "rarity": "common", "price": 100}
    ]
    state = new_run(
        seed=seed,
        character_id="TEST",
        ascension=ascension,
        source_data={
            "max_acts": 1,
            "map_floors": 8,
            "map_width": 5,
            "map_paths": 6,
            "shop_plan": {
                "colored_cards": 1,
                "colorless_cards": 1,
                "relics": 1,
                "potions": 1,
                "include_card_removal": True,
            },
            "shop_card_pool": [
                {"id": "shop_attack", "kind": "card", "rarity": "common", "price": 50}
            ],
            "shop_colorless_card_pool": [
                {
                    "id": "flash_of_steel",
                    "kind": "colorless_card",
                    "rarity": "uncommon",
                    "price": 120,
                }
            ],
            "shop_relic_pool": relic_pool,
            "shop_potion_pool": [
                {"id": "fire_potion", "kind": "potion", "rarity": "common", "price": 30}
            ],
            "cards": [
                {
                    "id": "shop_attack",
                    "name": "Shop Attack",
                    "type": "Attack",
                    "target": "AnyEnemy",
                    "damage": 6,
                    "upgrade": {"damage": "+3"},
                },
                {
                    "id": "flash_of_steel",
                    "name": "Flash of Steel",
                    "type": "Attack",
                    "target": "AnyEnemy",
                    "damage": 3,
                    "draw": 1,
                    "upgrade": {"damage": "+2"},
                },
            ],
            "deck": [
                {
                    "instance_id": "strike_1",
                    "card_id": "debug_kill",
                    "name": "Debug Kill",
                    "type": "attack",
                    "cost": 0,
                    "target": "enemy",
                    "effects": {"damage": 999},
                },
                {
                    "instance_id": "defend_1",
                    "card_id": "defend",
                    "name": "Defend",
                    "type": "skill",
                    "cost": 1,
                    "target": "self",
                    "effects": {"block": 5},
                },
            ],
            "player": {"hp": hp, "max_hp": 80, "gold": 300, "energy": 3, "max_energy": 3},
        },
    )
    state = step(state, _action(state, "choose_ancient"))
    state = state.model_copy(
        update={
            "player": state.player.model_copy(update={"hp": hp}),
            "relics": tuple(dict.fromkeys(state.relics + tuple(relics))),
            "potions": tuple(potions),
        }
    )

    for _ in range(40):
        if state.phase.value == "shop":
            return state
        if state.phase.value == "map":
            state = step(state, _next_map_action_preferring_shop(state))
        elif state.phase.value == "combat":
            state = step(state, _play_kill_card(state))
        elif state.phase.value in {"reward", "event", "treasure"}:
            state = step(state, _action(state, "proceed"))
        elif state.phase.value == "rest":
            state = step(state, _action(state, "rest"))
        else:
            break

    return None


def _next_map_action_preferring_shop(state):
    assert state.map is not None
    actions = [action for action in legal_actions(state) if action.type == "choose_node"]
    return next(
        (
            action
            for action in actions
            if action.target_id is not None
            and state.map.node_by_id[action.target_id].kind.value == "shop"
        ),
        actions[0],
    )


def test_shop_node_builds_inventory_and_offers_buy_and_leave_actions() -> None:
    state = _enter_shop()

    assert state.shop is not None
    assert [item.item_id for item in state.shop.items] == [
        "shop_attack",
        "flash_of_steel",
        "anchor",
        "fire_potion",
        "card_removal",
    ]
    assert [item.kind for item in state.shop.items[:2]] == ["card", "colorless_card"]
    action_types = {action.type for action in legal_actions(state)}
    assert {"shop_buy", "shop_leave", "proceed"} <= action_types


def test_shop_buying_relic_updates_gold_relics_and_inventory() -> None:
    state = _enter_shop()

    next_state = step(state, _action(state, "shop_buy", "shop:2"))

    assert next_state.phase.value == "shop"
    assert next_state.player.gold == state.player.gold - 100
    assert "anchor" in next_state.relics
    assert next_state.shop is not None
    assert next_state.shop.items[2].purchased is True
    assert not any(action.target_id == "shop:2" for action in legal_actions(next_state))
    assert next_state.replay_log[-1].events[0].kind == "shop_item_bought"


def test_shop_membership_card_discounts_prices() -> None:
    state = _enter_shop(relics=("membership_card",))

    assert state.shop is not None
    assert [item.price for item in state.shop.items] == [25, 60, 50, 15, 37]


def test_shop_buying_membership_card_reprices_remaining_items_once() -> None:
    state = _enter_shop(
        shop_relic_pool=[
            {"id": "membership_card", "kind": "relic", "rarity": "shop", "price": 100}
        ]
    )

    next_state = step(state, _action(state, "shop_buy", "shop:2"))

    assert next_state.shop is not None
    assert "membership_card" in next_state.relics
    assert [item.price for item in next_state.shop.items] == [25, 60, 50, 15, 37]

    after_card = step(next_state, _action(next_state, "shop_buy", "shop:0"))

    assert after_card.player.gold == next_state.player.gold - 25
    assert after_card.shop is not None
    assert [item.price for item in after_card.shop.items] == [25, 60, 50, 15, 37]


def test_shop_blacklists_courier_from_inventory() -> None:
    state = _enter_shop(
        shop_relic_pool=[
            {"id": "the_courier", "kind": "relic", "rarity": "shop", "price": 100},
            {"id": "membership_card", "kind": "relic", "rarity": "shop", "price": 100},
        ]
    )

    assert state.shop is not None
    item_ids = {item.item_id for item in state.shop.items}
    assert "the_courier" not in item_ids
    assert "membership_card" in item_ids


def test_shop_smiling_mask_sets_card_removal_price_to_fifty() -> None:
    state = _enter_shop(relics=("membership_card", "smiling_mask"))

    assert state.shop is not None
    assert state.shop.items[4].item_id == "card_removal"
    assert state.shop.items[4].price == 50


def test_shop_courier_discounts_and_restocks_purchased_items() -> None:
    state = _enter_shop(relics=("the_courier",))
    assert state.shop is not None
    assert state.shop.items[2].price == 80

    next_state = step(state, _action(state, "shop_buy", "shop:2"))

    assert next_state.player.gold == 220
    assert next_state.shop is not None
    assert next_state.shop.items[2].purchased is False
    assert next_state.shop.items[2].price == 80
    assert next_state.replay_log[-1].events[0].metadata["restocked_item_id"] == "anchor"


def test_shop_foul_potion_can_be_thrown_at_real_merchant_for_gold() -> None:
    state = _enter_shop(potions=("foul_potion",))

    assert any(action.type == "throw_potion_at_merchant" for action in legal_actions(state))

    next_state = step(state, _action(state, "throw_potion_at_merchant"))

    assert next_state.phase.value == "shop"
    assert next_state.player.gold == state.player.gold + 100
    assert next_state.potions == ()
    assert next_state.replay_log[-1].events[0].kind == "foul_potion_thrown_at_merchant"


def test_shop_meal_ticket_heals_on_entry() -> None:
    state = _enter_shop(relics=("meal_ticket",), hp=50)

    assert state.player.hp == 65
    assert any(event.kind == "meal_ticket_healed" for event in state.replay_log[-1].events)


def test_shop_lords_parasol_claims_all_non_service_items_on_entry() -> None:
    state = _enter_shop(relics=("lords_parasol",))

    assert state.shop is not None
    assert "anchor" in state.relics
    assert state.potions == ("fire_potion",)
    assert [card.card_id for card in state.master_deck[-2:]] == ["shop_attack", "flash_of_steel"]
    assert [item.purchased for item in state.shop.items] == [True, True, True, True, False]


def test_shop_buying_card_adds_it_to_master_deck() -> None:
    state = _enter_shop()

    next_state = step(state, _action(state, "shop_buy", "shop:0"))

    assert next_state.phase.value == "shop"
    assert next_state.player.gold == state.player.gold - 50
    assert next_state.master_deck[-1].card_id == "shop_attack"
    assert next_state.master_deck[-1].instance_id.startswith("shop_")


def test_shop_bought_card_uses_egg_relic_upgrade() -> None:
    state = _enter_shop(relics=("molten_egg",))

    next_state = step(state, _action(state, "shop_buy", "shop:0"))

    bought = next_state.master_deck[-1]
    assert bought.card_id == "shop_attack"
    assert bought.upgraded is True
    assert bought.effects["sequence"][0]["damage"] == 9
    assert next_state.replay_log[-1].events[-1].kind == "relic_card_upgraded"


def test_shop_buying_colorless_card_adds_it_to_master_deck() -> None:
    state = _enter_shop()

    next_state = step(state, _action(state, "shop_buy", "shop:1"))

    assert next_state.phase.value == "shop"
    assert next_state.player.gold == state.player.gold - 120
    assert next_state.master_deck[-1].card_id == "flash_of_steel"
    assert next_state.shop is not None
    assert next_state.shop.items[1].kind == "colorless_card"


def test_shop_buying_potion_adds_it_to_potions() -> None:
    state = _enter_shop()

    next_state = step(state, _action(state, "shop_buy", "shop:3"))

    assert next_state.phase.value == "shop"
    assert next_state.player.gold == state.player.gold - 30
    assert next_state.potions == ("fire_potion",)


def test_full_potion_slots_block_shop_potion_buy_until_discard() -> None:
    state = _enter_shop(potions=("fire_potion", "skill_potion", "foul_potion"))

    assert not any(action.target_id == "shop:3" for action in legal_actions(state))
    assert any(action.target_id == "potion:1" for action in legal_actions(state))

    discarded = step(state, _action(state, "discard_potion", "potion:1"))

    assert discarded.potions == ("fire_potion", "foul_potion")
    assert discarded.replay_log[-1].events[0].kind == "potion_discarded"
    assert any(action.target_id == "shop:3" for action in legal_actions(discarded))

    bought = step(discarded, _action(discarded, "shop_buy", "shop:3"))

    assert bought.potions == ("fire_potion", "foul_potion", "fire_potion")


def test_potion_belt_allows_two_extra_potion_slots() -> None:
    state = _enter_shop(
        relics=("potion_belt",),
        potions=("fire_potion", "skill_potion", "foul_potion"),
    )

    assert any(action.target_id == "shop:3" for action in legal_actions(state))


def test_ascension_potion_slot_reduction_blocks_third_potion() -> None:
    state = _enter_shop(
        ascension=11,
        potions=("fire_potion", "skill_potion"),
    )

    assert not any(action.target_id == "shop:3" for action in legal_actions(state))


def test_shop_card_removal_removes_target_card() -> None:
    state = _enter_shop()

    next_state = step(state, _action(state, "shop_buy", "remove:defend_1"))

    assert [card.instance_id for card in next_state.master_deck] == ["strike_1"]
    assert next_state.player.gold == state.player.gold - 75
    assert next_state.flags["shop_card_removals_bought"] == 1
    assert next_state.shop is not None
    assert next_state.shop.items[4].purchased is True
    assert next_state.replay_log[-1].events[0].kind == "shop_card_removed"


def test_shop_leave_completes_room() -> None:
    state = _enter_shop()

    next_state = step(state, _action(state, "shop_leave"))

    assert next_state.phase.value == "map"
    assert next_state.shop is None
    assert next_state.map is not None
    assert next_state.map.current_node_id in next_state.map.completed_node_ids
    assert next_state.replay_log[-1].events[0].kind == "shop_left"
