from __future__ import annotations

from sts2sim.mechanics import (
    RelicHook,
    apply_relic_price_modifiers,
    relic_potion_slot_bonus,
    resolve_relic_hook,
    resolve_relic_pickup,
    supported_relic_ids,
    unsupported_relic_handlers,
)


def test_potion_slot_relics_add_source_backed_capacity() -> None:
    assert relic_potion_slot_bonus(("POTION_BELT",)) == 2
    assert relic_potion_slot_bonus(({"id": "ALCHEMICAL_COFFER"},)) == 4
    assert relic_potion_slot_bonus(({"id": "PHIAL_HOLSTER"},)) == 1
    assert relic_potion_slot_bonus(({"id": "BELT_BUCKLE", "name": "Belt Buckle"},)) == 0
    assert relic_potion_slot_bonus(("POTION_BELT", "PHIAL_HOLSTER")) == 3


def test_shop_price_modifiers_stack_and_smiling_mask_fixes_removal_price() -> None:
    membership = apply_relic_price_modifiers(100, "card", ("MEMBERSHIP_CARD",))
    courier = apply_relic_price_modifiers(100, "relic", ("THE_COURIER",))
    stacked = apply_relic_price_modifiers(
        100,
        "potion",
        ("MEMBERSHIP_CARD", "THE_COURIER"),
    )
    removal = apply_relic_price_modifiers(
        125,
        "card_removal",
        ("MEMBERSHIP_CARD", "SMILING_MASK"),
    )

    assert membership.price == 50
    assert membership.applied_relic_ids == ("membership_card",)
    assert courier.price == 80
    assert stacked.price == 40
    assert stacked.multiplier_percent == 40
    assert removal.price == 50
    assert removal.fixed_price == 50
    assert removal.applied_relic_ids == ("smiling_mask",)


def test_old_coin_pickup_reports_gold_delta_and_marker() -> None:
    result = resolve_relic_pickup({"id": "OLD_COIN", "name": "Old Coin"})

    assert result.unsupported is False
    assert result.gold_delta == 300
    assert result.markers[0].kind == "old_coin_gold_gained"
    assert result.markers[0].amount == 300


def test_phial_holster_pickup_reports_slot_and_random_potion_marker() -> None:
    result = resolve_relic_pickup({"id": "PHIAL_HOLSTER", "name": "Phial Holster"})

    assert result.unsupported is False
    assert result.potion_slot_delta == 1
    assert result.markers[0].kind == "potion_slots_gained"
    assert result.markers[0].amount == 1
    assert result.markers[0].metadata == {"fill_random_potions": 2}


def test_meal_ticket_shop_entry_heal_is_capped_by_missing_hp() -> None:
    result = resolve_relic_hook(
        ("MEAL_TICKET",),
        RelicHook.SHOP_ENTER,
        hp=70,
        max_hp=80,
    )

    assert result.hp_delta == 10
    assert result.markers[0].kind == "meal_ticket_healed"
    assert result.markers[0].amount == 10


def test_combat_hook_markers_are_mapping_based() -> None:
    result = resolve_relic_hook(("ANCHOR", "VAJRA"), RelicHook.START_COMBAT)

    assert [marker.kind for marker in result.markers] == ["gain_block", "gain_status"]
    assert result.markers[0].amount == 10
    assert result.markers[1].metadata == {"status": "strength"}


def test_registry_includes_engine_level_reward_and_card_add_relics() -> None:
    supported = supported_relic_ids()

    assert {"black_star", "molten_egg", "toxic_egg", "frozen_egg"} <= supported
    assert "black_star" not in supported_relic_ids(RelicHook.START_COMBAT)


def test_unsupported_relic_handler_reporting_uses_inferred_hooks() -> None:
    unsupported = unsupported_relic_handlers(
        (
            {
                "id": "UNHANDLED_STARTER",
                "name": "Unhandled Starter",
                "description": "At the start of each combat, do a very specific thing.",
            },
        )
    )

    assert len(unsupported) == 1
    assert unsupported[0].relic_id == "unhandled_starter"
    assert unsupported[0].unsupported_hooks == (RelicHook.START_COMBAT,)


def test_empty_potion_slot_fill_relic_is_start_combat_not_capacity() -> None:
    unsupported = unsupported_relic_handlers(
        (
            {
                "id": "DELICATE_FROND",
                "name": "Delicate Frond",
                "description": (
                    "At the start of each combat, fill all empty potion slots "
                    "with random potions."
                ),
            },
        )
    )

    assert len(unsupported) == 1
    assert unsupported[0].unsupported_hooks == (RelicHook.START_COMBAT,)
