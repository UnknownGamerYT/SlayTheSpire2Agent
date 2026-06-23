from __future__ import annotations

import re
from pathlib import Path

from sts2sim.mechanics import (
    RelicHook,
    apply_relic_price_modifiers,
    relic_potion_slot_bonus,
    resolve_relic_hook,
    resolve_relic_pickup,
    supported_relic_ids,
    unsupported_relic_handlers,
)
from sts2sim.mechanics.relic_combat import combat_end, monster_killed, start_of_combat, turn_start


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


def test_pickup_deck_mutation_relics_return_integration_markers() -> None:
    astrolabe = resolve_relic_pickup("ASTROLABE")
    empty_cage = resolve_relic_pickup("EMPTY_CAGE")
    mirror = resolve_relic_pickup("DOLLYS_MIRROR")

    assert astrolabe.unsupported is False
    assert astrolabe.markers[0].kind == "transform_deck_cards"
    assert astrolabe.markers[0].amount == 3
    assert astrolabe.markers[0].metadata == {
        "selection": "chosen",
        "upgrade_transformed": True,
    }
    assert empty_cage.markers[0].kind == "remove_deck_cards"
    assert empty_cage.markers[0].amount == 2
    assert empty_cage.markers[0].metadata == {"selection": "chosen"}
    assert mirror.markers[0].kind == "duplicate_deck_card"
    assert mirror.markers[0].amount == 1
    assert mirror.markers[0].metadata == {"selection": "chosen"}


def test_pickup_reward_and_rule_relics_return_bounded_markers() -> None:
    arcane_scroll = resolve_relic_pickup("ARCANE_SCROLL")
    cauldron = resolve_relic_pickup("CAULDRON")
    darkstone = resolve_relic_pickup("DARKSTONE_PERIAPT")
    lead_paperweight = resolve_relic_pickup("LEAD_PAPERWEIGHT")

    assert arcane_scroll.markers[0].kind == "card_reward"
    assert arcane_scroll.markers[0].amount == 1
    assert arcane_scroll.markers[0].metadata == {"rarity": "rare", "selection": "random"}
    assert cauldron.markers[0].kind == "random_potions_gained"
    assert cauldron.markers[0].amount == 5
    assert darkstone.markers[0].kind == "curse_obtained_max_hp_delta_enabled"
    assert darkstone.markers[0].amount == 6
    assert darkstone.markers[0].metadata == {"trigger": "curse_obtained"}
    assert lead_paperweight.markers[0].kind == "card_reward"
    assert lead_paperweight.markers[0].metadata == {
        "card_color": "colorless",
        "choices": 2,
        "selection": "choose",
    }


def test_fixed_card_pickup_relics_return_deck_add_markers() -> None:
    jewelry_box = resolve_relic_pickup("JEWELRY_BOX")
    neows_torment = resolve_relic_pickup("NEOWS_TORMENT")
    paels_horn = resolve_relic_pickup("PAELS_HORN")
    storybook = resolve_relic_pickup("STORYBOOK")
    whistle = resolve_relic_pickup("TANXS_WHISTLE")

    assert [
        (marker.kind, marker.amount, marker.metadata["card_id"])
        for marker in (
            jewelry_box.markers[0],
            neows_torment.markers[0],
            paels_horn.markers[0],
            storybook.markers[0],
            whistle.markers[0],
        )
    ] == [
        ("add_deck_cards", 1, "apotheosis"),
        ("add_deck_cards", 1, "neows_fury"),
        ("add_deck_cards", 2, "relax"),
        ("add_deck_cards", 1, "brightest_flame"),
        ("add_deck_cards", 1, "whistle"),
    ]


def test_more_fixed_card_and_reward_pickups_return_source_backed_markers() -> None:
    rose = resolve_relic_pickup("BLOOD_SOAKED_ROSE")
    byrdpip = resolve_relic_pickup("BYRDPIP")
    bell = resolve_relic_pickup("CALLING_BELL")
    tablet = resolve_relic_pickup("HEFTY_TABLET")
    massive_scroll = resolve_relic_pickup("MASSIVE_SCROLL")
    orrery = resolve_relic_pickup("ORRERY")

    assert rose.markers[0].metadata == {"card_id": "enthralled"}
    assert byrdpip.markers[0].metadata == {"card_id": "byrd_swoop"}
    assert [(marker.kind, marker.amount) for marker in bell.markers] == [
        ("add_deck_cards", 1),
        ("random_relics_gained", 3),
    ]
    assert bell.markers[0].metadata == {"card_id": "curse_of_the_bell"}
    assert [marker.kind for marker in tablet.markers] == ["card_reward", "add_deck_cards"]
    assert tablet.markers[0].metadata == {"rarity": "rare", "choices": 3, "selection": "choose"}
    assert tablet.markers[1].metadata == {"card_id": "injury"}
    assert massive_scroll.markers[0].metadata == {
        "card_pool": "multiplayer",
        "choices": 3,
        "selection": "choose",
    }
    assert orrery.markers[0].kind == "card_reward"
    assert orrery.markers[0].amount == 5


def test_random_upgrade_and_transform_pickup_relics_return_deck_markers() -> None:
    fragrant = resolve_relic_pickup("FRAGRANT_MUSHROOM")
    sand_castle = resolve_relic_pickup("SAND_CASTLE")
    war_paint = resolve_relic_pickup("WAR_PAINT")
    whetstone = resolve_relic_pickup("WHETSTONE")
    pandoras_box = resolve_relic_pickup("PANDORAS_BOX")
    talisman = resolve_relic_pickup("NEOWS_TALISMAN")

    assert fragrant.hp_delta == -15
    assert [marker.kind for marker in fragrant.markers] == ["hp_delta", "upgrade_deck_cards"]
    assert fragrant.markers[1].amount == 2
    assert sand_castle.markers[0].kind == "upgrade_deck_cards"
    assert sand_castle.markers[0].amount == 6
    assert war_paint.markers[0].metadata == {"selection": "random", "card_type": "skill"}
    assert whetstone.markers[0].metadata == {"selection": "random", "card_type": "attack"}
    assert pandoras_box.markers[0].kind == "transform_deck_cards"
    assert pandoras_box.markers[0].metadata["selection"] == "matching"
    assert pandoras_box.markers[0].metadata["match_card_ids"] == ("strike", "defend")
    assert [marker.metadata["match_card_ids"] for marker in talisman.markers] == [
        ("strike",),
        ("defend",),
    ]


def test_more_transform_upgrade_and_remove_pickup_relics_return_deck_markers() -> None:
    tooth = resolve_relic_pickup("ARCHAIC_TOOTH")
    claws = resolve_relic_pickup("CLAWS")
    leaf = resolve_relic_pickup("NEW_LEAF")
    poultice = resolve_relic_pickup("LEAFY_POULTICE")
    pomander = resolve_relic_pickup("POMANDER")
    cookie = resolve_relic_pickup("YUMMY_COOKIE")
    shears = resolve_relic_pickup("PRECARIOUS_SHEARS")
    scissors = resolve_relic_pickup("PRECISE_SCISSORS")
    fog = resolve_relic_pickup("PRESERVED_FOG")

    assert tooth.markers[0].metadata["transform_pool"] == "ancient"
    assert tooth.markers[0].metadata["match_card_ids"] == ("strike", "defend")
    assert claws.markers[0].metadata == {"selection": "chosen", "target_card_id": "maul"}
    assert leaf.markers[0].metadata == {"selection": "chosen"}
    assert poultice.max_hp_delta == -12
    assert [marker.kind for marker in poultice.markers] == [
        "max_hp_delta",
        "transform_deck_cards",
        "transform_deck_cards",
    ]
    assert pomander.markers[0].amount == 1
    assert cookie.markers[0].amount == 4
    assert shears.hp_delta == -16
    assert [marker.kind for marker in shears.markers] == ["hp_delta", "remove_deck_cards"]
    assert scissors.markers[0].amount == 1
    assert [(marker.kind, marker.amount) for marker in fog.markers] == [
        ("remove_deck_cards", 3),
        ("add_deck_cards", 1),
    ]
    assert fog.markers[1].metadata == {"card_id": "folly"}


def test_pickup_enchant_style_relics_return_modify_deck_markers() -> None:
    bracelet = resolve_relic_pickup("BEAUTIFUL_BRACELET")
    shrymp = resolve_relic_pickup("ELECTRIC_SHRYMP")
    hammer = resolve_relic_pickup("GNARLED_HAMMER")
    kifuda = resolve_relic_pickup("KIFUDA")
    claw = resolve_relic_pickup("PAELS_CLAW")
    growth = resolve_relic_pickup("PAELS_GROWTH")
    dagger = resolve_relic_pickup("PUNCH_DAGGER")
    stamp = resolve_relic_pickup("ROYAL_STAMP")

    first_three_kinds = [
        marker.kind
        for marker in (bracelet.markers[0], shrymp.markers[0], hammer.markers[0])
    ]
    assert first_three_kinds == [
        "modify_deck_cards",
        "modify_deck_cards",
        "modify_deck_cards",
    ]
    assert bracelet.markers[0].amount == 3
    assert bracelet.markers[0].metadata["custom"] == {
        "enchant_keyword": "swift",
        "enchant_amount": 3,
    }
    assert shrymp.markers[0].metadata["card_type"] == "skill"
    assert hammer.markers[0].metadata["custom"] == {
        "enchant_keyword": "sharp",
        "enchant_amount": 3,
    }
    assert kifuda.markers[0].metadata["custom"] == {"enchant_keyword": "adroit"}
    assert claw.markers[0].metadata["match_card_ids"] == ("defend",)
    assert growth.markers[0].metadata["custom"] == {"enchant_keyword": "clone"}
    assert dagger.markers[0].metadata["custom"] == {
        "enchant_keyword": "momentum",
        "enchant_amount": 5,
    }
    assert stamp.markers[0].metadata["match_card_types"] == ("attack", "skill")


def test_misc_pickup_relics_return_reward_relic_and_deck_rule_markers() -> None:
    cape = resolve_relic_pickup("DISTINGUISHED_CAPE")
    dusty_tome = resolve_relic_pickup("DUSTY_TOME")
    ghost_seed = resolve_relic_pickup("GHOST_SEED")
    glass_eye = resolve_relic_pickup("GLASS_EYE")
    large_capsule = resolve_relic_pickup("LARGE_CAPSULE")
    lost_coffer = resolve_relic_pickup("LOST_COFFER")
    small_capsule = resolve_relic_pickup("SMALL_CAPSULE")
    wongo_badge = resolve_relic_pickup("WONGO_CUSTOMER_APPRECIATION_BADGE")

    assert cape.max_hp_delta == -9
    assert [marker.kind for marker in cape.markers] == ["max_hp_delta", "add_deck_cards"]
    assert cape.markers[1].metadata == {"card_id": "apparition"}
    assert dusty_tome.markers[0].metadata == {"card_pool": "ancient", "selection": "random"}
    assert ghost_seed.markers[0].kind == "modify_deck_cards"
    assert ghost_seed.markers[0].metadata["custom"] == {"ethereal": True}
    assert glass_eye.markers[0].metadata["rarities"] == (
        "common",
        "common",
        "uncommon",
        "uncommon",
        "rare",
    )
    assert [marker.kind for marker in large_capsule.markers] == [
        "random_relics_gained",
        "add_deck_cards",
        "add_deck_cards",
    ]
    assert [marker.kind for marker in lost_coffer.markers] == [
        "card_reward",
        "random_potions_gained",
    ]
    assert small_capsule.markers[0].kind == "random_relics_gained"
    assert wongo_badge.markers[0].kind == "no_effect"


def test_pickup_passive_reward_relics_register_no_immediate_effect_markers() -> None:
    dingy_rug = resolve_relic_pickup("DINGY_RUG")
    driftwood = resolve_relic_pickup("DRIFTWOOD")
    glitter = resolve_relic_pickup("GLITTER")
    candy = resolve_relic_pickup("LASTING_CANDY")
    wing = resolve_relic_pickup("PAELS_WING")
    white_star = resolve_relic_pickup("WHITE_STAR")
    wing_charm = resolve_relic_pickup("WING_CHARM")
    crucible = resolve_relic_pickup("SILVER_CRUCIBLE")

    assert [marker.kind for marker in (dingy_rug.markers[0], driftwood.markers[0])] == [
        "no_effect",
        "no_effect",
    ]
    assert dingy_rug.markers[0].metadata["passive"] == "card_rewards_can_include_colorless"
    assert driftwood.markers[0].metadata["passive"] == "card_reward_reroll_once"
    assert glitter.markers[0].metadata == {
        "passive": "card_rewards_enchanted",
        "enchant_keyword": "glam",
    }
    assert candy.markers[0].metadata["passive"] == (
        "card_rewards_gain_additional_power_every_other_combat"
    )
    assert wing.markers[0].metadata["passive"] == "sacrifice_card_rewards_for_relic_counter"
    assert white_star.markers[0].metadata["passive"] == "elite_rare_card_reward_delta"
    assert wing_charm.markers[0].metadata == {
        "passive": "card_reward_random_card_enchanted",
        "enchant_keyword": "swift",
        "enchant_amount": 1,
    }
    assert crucible.markers[0].kind == "relic_counter_changed"
    assert crucible.markers[0].amount == 3
    assert crucible.markers[0].metadata["first_treasure_empty"] is True


def test_remaining_non_combat_pickup_relics_return_source_backed_markers() -> None:
    fishing_rod = resolve_relic_pickup("FISHING_ROD")
    hug = resolve_relic_pickup("BIIIG_HUG")
    kaleidoscope = resolve_relic_pickup("KALEIDOSCOPE")
    neows_bones = resolve_relic_pickup("NEOWS_BONES")
    soup = resolve_relic_pickup("NUTRITIOUS_SOUP")
    scroll_boxes = resolve_relic_pickup("SCROLL_BOXES")
    sea_glass = resolve_relic_pickup("SEA_GLASS")
    sere_talon = resolve_relic_pickup("SERE_TALON")
    silken_tress = resolve_relic_pickup("SILKEN_TRESS")
    toy_box = resolve_relic_pickup("TOY_BOX")
    tri_boomerang = resolve_relic_pickup("TRI_BOOMERANG")
    winged_boots = resolve_relic_pickup("WINGED_BOOTS")
    mystery_ticket = resolve_relic_pickup("WONGOS_MYSTERY_TICKET")

    assert fishing_rod.markers[0].metadata == {
        "passive": "normal_combat_count_random_deck_upgrade",
        "combat_type": "normal",
        "interval": 3,
        "upgrade_count": 1,
        "selection": "random",
        "needed_subsystem": "combat_count_relic_trigger",
    }
    assert [marker.kind for marker in hug.markers] == ["remove_deck_cards", "no_effect"]
    assert hug.markers[0].amount == 4
    assert hug.markers[1].metadata == {
        "passive": "shuffle_add_card_to_draw_pile",
        "card_id": "soot",
        "needed_subsystem": "draw_pile_shuffle_relic_trigger",
    }
    assert [(marker.kind, marker.amount) for marker in neows_bones.markers] == [
        ("random_relics_gained", 2),
        ("random_curses_gained", 1),
    ]
    assert neows_bones.markers[0].metadata == {"relic_pool": "neow"}
    assert kaleidoscope.markers[0].kind == "card_reward"
    assert kaleidoscope.markers[0].amount == 2
    assert kaleidoscope.markers[0].metadata == {
        "card_pool": "other_character",
        "selection": "choose",
        "needed_subsystem": "cross_character_card_reward",
    }
    assert soup.markers[0].kind == "modify_deck_cards"
    assert soup.markers[0].amount == 3
    assert soup.markers[0].metadata["match_card_ids"] == ("strike",)
    assert soup.markers[0].metadata["operation"] == "add_damage"
    assert soup.markers[0].metadata["custom"] == {"damage_bonus": 3}
    assert [(marker.kind, marker.amount) for marker in scroll_boxes.markers] == [
        ("card_pack_reward", 1),
    ]
    assert scroll_boxes.markers[0].metadata["needed_subsystem"] == "card_pack_rewards"
    assert sea_glass.markers[0].metadata == {
        "card_pool": "other_character",
        "selection": "choose_any",
        "needed_subsystem": "cross_character_card_reward",
    }
    assert [(marker.kind, marker.amount) for marker in sere_talon.markers] == [
        ("random_curses_gained", 2),
        ("add_deck_cards", 3),
    ]
    assert sere_talon.markers[1].metadata == {"card_id": "wish"}
    assert [(marker.kind, marker.amount) for marker in silken_tress.markers] == [
        ("set_gold", 0),
        ("no_effect", None),
    ]
    assert silken_tress.markers[1].metadata == {
        "passive": "first_card_reward_enchanted",
        "enchant_keyword": "glam",
        "card_reward_index": 1,
        "needed_subsystem": "card_reward_enchant_trigger",
    }
    assert toy_box.markers[0].metadata == {"relic_pool": "wax"}
    assert toy_box.markers[1].metadata["passive"] == "wax_relic_melts_every_combats"
    assert tri_boomerang.markers[0].metadata["custom"] == {"enchant_keyword": "instinct"}
    assert winged_boots.markers[0].kind == "relic_counter_changed"
    assert winged_boots.markers[0].amount == 3
    assert mystery_ticket.markers[0].metadata["random_relics"] == 3


def test_remaining_non_combat_passive_pickups_document_needed_subsystems() -> None:
    expected_passives = {
        "BING_BONG": "deck_card_added_duplicate_copy",
        "BOOK_OF_FIVE_RINGS": "deck_card_added_count_heal",
        "BOWLER_HAT": "gold_gain_bonus_percent",
        "DRAGON_FRUIT": "gold_gain_max_hp_delta",
        "FRESNEL_LENS": "deck_added_block_card_enchanted",
        "JUZU_BRACELET": "question_rooms_no_regular_enemy_combats",
        "LUCKY_FYSH": "deck_card_added_gold",
        "MAW_BANK": "floor_climb_gold_until_shop_spend",
        "PLANISPHERE": "event_room_enter_heal",
    }

    for relic_id, passive in expected_passives.items():
        marker = resolve_relic_pickup(relic_id).markers[0]

        assert marker.kind == "no_effect"
        assert marker.metadata["passive"] == passive
        assert "needed_subsystem" in marker.metadata

    fur_coat = resolve_relic_pickup("FUR_COAT")
    touch = resolve_relic_pickup("TOUCH_OF_OROBAS")

    assert fur_coat.markers[0].kind == "mark_map_rooms"
    assert fur_coat.markers[0].amount == 7
    assert fur_coat.markers[0].metadata["needed_subsystem"] == "map_room_markers"
    assert touch.markers[0].kind == "replace_starter_relic"
    assert touch.markers[0].metadata["needed_subsystem"] == "starter_relic_upgrade"


def test_max_hp_and_healing_pickup_relics_report_bounded_markers() -> None:
    strawberry = resolve_relic_pickup("STRAWBERRY")
    mango = resolve_relic_pickup("MANGO")
    waffle = resolve_relic_pickup("LEES_WAFFLE")
    fake_waffle = resolve_relic_pickup("FAKE_LEES_WAFFLE")

    assert strawberry.max_hp_delta == 7
    assert strawberry.markers[0].kind == "max_hp_delta"
    assert strawberry.markers[0].amount == 7
    assert mango.max_hp_delta == 14
    assert waffle.max_hp_delta == 7
    assert [marker.kind for marker in waffle.markers] == ["max_hp_delta", "heal_to_full"]
    assert fake_waffle.markers[0].kind == "heal_percent_max_hp"
    assert fake_waffle.markers[0].amount == 10


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


def test_remaining_shop_and_campfire_relics_register_hook_markers() -> None:
    shop_entry = resolve_relic_hook(("LORDS_PARASOL",), RelicHook.SHOP_ENTER)
    shop_purchase = resolve_relic_hook(("MAW_BANK",), RelicHook.SHOP_PURCHASE)
    campfire = resolve_relic_hook(
        (
            "DREAM_CATCHER",
            "ETERNAL_FEATHER",
            "MEAT_CLEAVER",
            "MINIATURE_TENT",
            "REGAL_PILLOW",
            "STONE_HUMIDIFIER",
            "TINY_MAILBOX",
        ),
        RelicHook.CAMPFIRE_ENTER,
    )

    assert shop_entry.markers[0].kind == "lords_parasol_claimed_shop"
    assert shop_entry.markers[0].metadata == {
        "engine_handler": "shop_entry_claim_all_non_service_items",
    }
    assert shop_purchase.markers[0].kind == "maw_bank_disabled"
    assert shop_purchase.markers[0].metadata == {
        "passive": "floor_climb_gold_until_shop_spend",
        "needed_subsystem": "shop_purchase_relic_state",
    }
    assert [marker.kind for marker in campfire.markers] == [
        "no_effect",
        "no_effect",
        "campfire_cook_unlocked",
        "campfire_multi_action_unlocked",
        "no_effect",
        "no_effect",
        "no_effect",
    ]
    assert campfire.markers[0].metadata["passive"] == "rest_add_card_reward"
    assert campfire.markers[1].metadata == {
        "passive": "rest_site_enter_heal_per_deck_cards",
        "cards_per_heal": 5,
        "heal": 3,
        "needed_subsystem": "campfire_enter_deck_size",
    }
    assert campfire.markers[3].metadata["passive"] == "campfire_choose_any_number_of_options"
    assert campfire.markers[4].metadata["passive"] == "rest_heal_bonus"
    assert campfire.markers[5].metadata["passive"] == "rest_max_hp_delta"
    assert campfire.markers[6].metadata["passive"] == "rest_random_potions"


def test_combat_hook_markers_are_mapping_based() -> None:
    result = resolve_relic_hook(("ANCHOR", "VAJRA"), RelicHook.START_COMBAT)

    assert [marker.kind for marker in result.markers] == ["gain_block", "gain_status"]
    assert result.markers[0].amount == 10
    assert result.markers[1].metadata == {"status": "strength"}


def test_new_start_combat_relic_markers_are_registered_for_audits() -> None:
    result = resolve_relic_hook(
        (
            "LANTERN",
            "DIVINE_RIGHT",
            "FENCING_MANUAL",
            "RUNIC_CAPACITOR",
            "INFUSED_CORE",
            "SNECKO_EYE",
            "NINJA_SCROLL",
            "TWISTED_FUNNEL",
        ),
        RelicHook.START_COMBAT,
    )

    assert [marker.kind for marker in result.markers] == [
        "gain_energy",
        "player_resource",
        "player_resource",
        "orb_slot_delta",
        "channel_orb",
        "gain_status",
        "add_card_to_hand",
        "apply_status",
    ]
    assert result.markers[1].metadata == {"resource": "star"}
    assert result.markers[2].metadata == {"resource": "forge"}
    assert result.markers[4].metadata == {"orb": "lightning", "lightning_damage_bonus": 1}
    assert result.markers[5].metadata == {"status": "confused"}
    assert result.markers[6].metadata["card_id"] == "shiv"
    assert result.markers[7].metadata == {"status": "poison"}


def test_new_turn_start_and_combat_end_relic_markers_are_registered_for_audits() -> None:
    turn = resolve_relic_hook(
        ("MERCURY_HOURGLASS", "PAELS_BLOOD", "SAI", "SNECKO_EYE", "PENDULUM"),
        RelicHook.START_TURN,
    )
    end = resolve_relic_hook(("CHOSEN_CHEESE",), RelicHook.END_COMBAT)

    assert [marker.kind for marker in turn.markers] == [
        "all_damage",
        "draw_cards",
        "gain_block",
        "draw_cards",
        "draw_cards",
    ]
    assert end.max_hp_delta == 1
    assert end.markers[0].kind == "max_hp_delta"


def test_registry_includes_engine_level_reward_and_card_add_relics() -> None:
    supported = supported_relic_ids()

    assert {"black_star", "molten_egg", "toxic_egg", "frozen_egg"} <= supported
    assert "black_star" not in supported_relic_ids(RelicHook.START_COMBAT)


def test_registry_includes_new_pickup_deck_mutation_relic_markers() -> None:
    supported = supported_relic_ids(RelicHook.PICKUP)

    assert {
        "arcane_scroll",
        "archaic_tooth",
        "astrolabe",
        "beautiful_bracelet",
        "blood_soaked_rose",
        "byrdpip",
        "calling_bell",
        "cauldron",
        "claws",
        "darkstone_periapt",
        "dingy_rug",
        "dollys_mirror",
        "driftwood",
        "electric_shrymp",
        "empty_cage",
        "lead_paperweight",
        "distinguished_cape",
        "dusty_tome",
        "fragrant_mushroom",
        "ghost_seed",
        "glass_eye",
        "glitter",
        "gnarled_hammer",
        "hefty_tablet",
        "jewelry_box",
        "kifuda",
        "large_capsule",
        "lasting_candy",
        "leafy_poultice",
        "lost_coffer",
        "massive_scroll",
        "new_leaf",
        "neows_talisman",
        "neows_torment",
        "orrery",
        "paels_claw",
        "paels_growth",
        "paels_horn",
        "paels_wing",
        "pandoras_box",
        "pomander",
        "precarious_shears",
        "precise_scissors",
        "preserved_fog",
        "punch_dagger",
        "royal_stamp",
        "sand_castle",
        "silver_crucible",
        "small_capsule",
        "storybook",
        "tanxs_whistle",
        "white_star",
        "wing_charm",
        "war_paint",
        "whetstone",
        "wongo_customer_appreciation_badge",
        "yummy_cookie",
    } <= supported


def test_registry_includes_remaining_non_combat_relic_batch() -> None:
    requested_ids = {
        "biiig_hug",
        "bing_bong",
        "book_of_five_rings",
        "bowler_hat",
        "dragon_fruit",
        "dream_catcher",
        "eternal_feather",
        "fishing_rod",
        "fresnel_lens",
        "fur_coat",
        "juzu_bracelet",
        "kaleidoscope",
        "lords_parasol",
        "lucky_fysh",
        "maw_bank",
        "meat_cleaver",
        "miniature_tent",
        "neows_bones",
        "nutritious_soup",
        "planisphere",
        "regal_pillow",
        "scroll_boxes",
        "sea_glass",
        "sere_talon",
        "silken_tress",
        "stone_humidifier",
        "tiny_mailbox",
        "touch_of_orobas",
        "toy_box",
        "tri_boomerang",
        "winged_boots",
        "wongos_mystery_ticket",
    }

    assert requested_ids <= supported_relic_ids()
    assert {
        "biiig_hug",
        "bing_bong",
        "book_of_five_rings",
        "bowler_hat",
        "dragon_fruit",
        "fresnel_lens",
        "fur_coat",
        "juzu_bracelet",
        "lucky_fysh",
        "maw_bank",
        "neows_bones",
        "nutritious_soup",
        "planisphere",
        "scroll_boxes",
        "sea_glass",
        "sere_talon",
        "touch_of_orobas",
        "toy_box",
        "tri_boomerang",
        "winged_boots",
        "wongos_mystery_ticket",
    } <= supported_relic_ids(RelicHook.PICKUP)
    assert {"lords_parasol"} <= supported_relic_ids(RelicHook.SHOP_ENTER)
    assert {"maw_bank"} <= supported_relic_ids(RelicHook.SHOP_PURCHASE)
    assert {
        "dream_catcher",
        "eternal_feather",
        "meat_cleaver",
        "miniature_tent",
        "regal_pillow",
        "stone_humidifier",
        "tiny_mailbox",
    } <= supported_relic_ids(RelicHook.CAMPFIRE_ENTER)


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


def _central_combat_relic_executor_kinds() -> tuple[set[str], set[str]]:
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "src" / "sts2sim" / "engine" / "transitions.py").read_text()
    match = re.search(
        r"def _apply_combat_relic_marker\(.*?\n(?=def _apply_combat_relic_status\()",
        source,
        flags=re.S,
    )
    assert match is not None
    body = match.group(0)
    exact = set(re.findall(r'marker\.kind\s*==\s*"([^"]+)"', body))
    for group in re.findall(r"marker\.kind\s+in\s+\{([^}]+)\}", body, flags=re.S):
        exact.update(re.findall(r'"([^"]+)"', group))
    prefixes = set(re.findall(r'marker\.kind\.startswith\("([^"]+)"\)', body))
    return exact, prefixes


def test_high_risk_combat_relic_markers_have_runtime_executor_support() -> None:
    emitted = {
        marker.kind
        for resolution in (
            start_of_combat(
                ("gambling_chip", "petrified_toad", "delicate_frond"),
                metadata={"empty_potion_slots": 2},
            ),
            turn_start(
                ("history_course",),
                metadata={
                    "last_played_card_type": "attack",
                    "last_played_card_id": "strike",
                },
            ),
            monster_killed(("war_hammer", "black_star"), encounter_type="elite"),
            combat_end(("paels_tooth",), metadata={"paels_tooth_removed_count": 1}),
        )
        for marker in resolution.markers
    }
    executor_kinds, executor_prefixes = _central_combat_relic_executor_kinds()
    external_runtime_kinds = {
        # Black Star is already applied by combat reward generation, not by
        # _apply_combat_relic_marker.
        "reward_relic_count_delta",
        # These start-combat effects mutate RunState/pending choices around the
        # opening draw instead of going through the central combat marker applier.
        "opening_hand_discard_redraw",
        "play_card_copy",
        "procure_potion",
        "random_potions_gained",
        # These combat-win effects mutate RunState after the fight, then attach
        # their events to combat.last_events.
        "add_deck_cards",
        "upgrade_deck_cards",
    }

    missing = sorted(
        kind
        for kind in emitted - external_runtime_kinds
        if kind not in executor_kinds
        and not any(kind.startswith(prefix) for prefix in executor_prefixes)
    )

    assert missing == []
