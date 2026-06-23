from __future__ import annotations

import ast
from pathlib import Path

import sts2sim.mechanics.relic_combat as relic_combat_module
from sts2sim.mechanics import (
    CombatRelicHook,
    card_played,
    combat_end,
    damage_dealt,
    damage_taken,
    monster_killed,
    resolve_combat_relic_hook,
    start_of_combat,
    supported_combat_relic_ids,
    turn_end,
    turn_start,
    unsupported_combat_relic_handlers,
)


def test_start_of_combat_outputs_markers_and_capped_heal() -> None:
    result = start_of_combat(
        ("ANCHOR", {"name": "Blood Vial"}, "VAJRA"),
        player_hp=79,
        player_max_hp=80,
    )

    assert result.hook is CombatRelicHook.START_OF_COMBAT
    assert result.hp_delta == 1
    assert result.block_delta == 10
    assert result.blockers == ()
    assert [(marker.kind, marker.amount, marker.target_id) for marker in result.markers] == [
        ("gain_block", 10, "player"),
        ("heal_player", 1, "player"),
        ("gain_status", 1, "player"),
    ]
    assert result.markers[2].metadata == {"status": "strength"}
    assert result.markers[2].source_id == "vajra"


def test_start_of_combat_relics_emit_draw_vigor_and_focus_markers() -> None:
    result = start_of_combat(("AKABEKO", "BAG_OF_PREPARATION", "DATA_DISK"))

    assert [(marker.kind, marker.amount, marker.metadata) for marker in result.markers] == [
        ("gain_status", 8, {"status": "vigor"}),
        ("draw_cards", 2, {}),
        ("gain_status", 1, {"status": "focus"}),
    ]


def test_bellows_upgrades_the_opening_hand_draw_count() -> None:
    result = start_of_combat(("BELLOWS",), metadata={"opening_draw_count": 6})

    assert [(marker.kind, marker.amount, marker.metadata) for marker in result.markers] == [
        ("upgrade_draw_pile_cards", 6, {"mode": "opening_hand"})
    ]


def test_start_of_combat_resource_orb_and_damage_relic_markers() -> None:
    result = start_of_combat(
        (
            "LANTERN",
            "GORGET",
            "DIVINE_RIGHT",
            "FENCING_MANUAL",
            "RUNIC_CAPACITOR",
            "INFUSED_CORE",
            "SYMBIOTIC_VIRUS",
            "FESTIVE_POPPER",
            "RING_OF_THE_SNAKE",
            "NINJA_SCROLL",
            "FUNERARY_MASK",
            "TWISTED_FUNNEL",
            "BLESSED_ANTLER",
            "ROYAL_POISON",
            "VERY_HOT_COCOA",
            "JEWELED_MASK",
            "POWER_CELL",
            "RADIANT_PEARL",
            "STONE_CRACKER",
        )
    )

    assert [
        (marker.relic_id, marker.kind, marker.amount, marker.target_id, marker.metadata)
        for marker in result.markers
    ] == [
        ("lantern", "gain_energy", 1, "player", {}),
        ("gorget", "gain_status", 4, "player", {"status": "plated_armor"}),
        ("divine_right", "player_resource", 3, "player", {"resource": "star"}),
        ("fencing_manual", "player_resource", 10, "player", {"resource": "forge"}),
        ("runic_capacitor", "orb_slot_delta", 3, "player", {}),
        (
            "infused_core",
            "channel_orb",
            3,
            "player",
            {"orb": "lightning", "lightning_damage_bonus": 1},
        ),
        ("symbiotic_virus", "channel_orb", 1, "player", {"orb": "dark"}),
        ("festive_popper", "all_damage", 9, "all_enemies", {}),
        ("ring_of_the_snake", "draw_cards", 2, "player", {}),
        (
            "ninja_scroll",
            "add_card_to_hand",
            3,
            "player",
            {"card_id": "shiv", "card_type": "attack", "target": "enemy"},
        ),
        (
            "funerary_mask",
            "add_card_to_draw_pile",
            3,
            "player",
            {"card_id": "soul", "card_type": "skill", "target": "self"},
        ),
        ("twisted_funnel", "apply_status", 4, "all_enemies", {"status": "poison"}),
        (
            "blessed_antler",
            "shuffle_status_into_draw_pile",
            3,
            "player",
            {"card_id": "dazed", "card_type": "status", "target": "self"},
        ),
        ("royal_poison", "lose_hp", 4, "player", {}),
        ("very_hot_cocoa", "gain_energy", 4, "player", {}),
        (
            "jeweled_mask",
            "move_card_type_from_draw_to_hand",
            1,
            "player",
            {"card_type": "power", "free_to_play_this_turn": True},
        ),
        (
            "power_cell",
            "move_zero_cost_cards_to_hand",
            2,
            "player",
            {"free_to_play_this_turn": True},
        ),
        (
            "radiant_pearl",
            "add_card_to_hand",
            1,
            "player",
            {
                "card_id": "luminesce",
                "name": "Luminesce",
                "card_type": "skill",
                "target": "self",
                "cost": 0,
                "exhaust": True,
            },
        ),
        ("stone_cracker", "upgrade_draw_pile_cards", 2, "player", {"mode": "combat_only"}),
    ]


def test_start_of_combat_random_card_relic_markers() -> None:
    result = start_of_combat(
        (
            "BIG_HAT",
            "CHOICES_PARADOX",
            "ORANGE_DOUGH",
            "TOOLBOX",
            "VEXING_PUZZLEBOX",
        )
    )

    assert [
        (marker.relic_id, marker.kind, marker.amount, marker.target_id, marker.metadata)
        for marker in result.markers
    ] == [
        (
            "big_hat",
            "add_card_to_hand",
            2,
            "player",
            {"selection": "random", "keyword": "ethereal", "ethereal": True},
        ),
        (
            "choices_paradox",
            "add_card_to_hand",
            1,
            "player",
            {
                "selection": "choose_one_of_random",
                "choice_count": 5,
                "retain_once": True,
            },
        ),
        (
            "orange_dough",
            "add_card_to_hand",
            2,
            "player",
            {"selection": "random", "card_pool": "colorless"},
        ),
        (
            "toolbox",
            "add_card_to_hand",
            1,
            "player",
            {
                "selection": "choose_one_of_random",
                "choice_count": 3,
                "card_pool": "colorless",
            },
        ),
        (
            "vexing_puzzlebox",
            "add_card_to_hand",
            1,
            "player",
            {"selection": "random", "free_to_play_this_turn": True},
        ),
    ]


def test_brimstone_turn_start_buffs_player_and_enemies() -> None:
    result = turn_start(("BRIMSTONE",))

    assert [(marker.kind, marker.amount, marker.target_id) for marker in result.markers] == [
        ("gain_status", 2, "player"),
        ("apply_status", 1, "all_enemies"),
    ]
    assert result.markers[0].metadata == {"status": "strength"}
    assert result.markers[1].metadata == {"status": "strength"}


def test_more_turn_start_relic_markers_cover_draw_block_damage_and_resources() -> None:
    result = turn_start(
        (
            "MERCURY_HOURGLASS",
            "PAELS_BLOOD",
            "SAI",
            "BOUND_PHYLACTERY",
            "SNECKO_EYE",
            "MR_STRUGGLES",
            "PENDULUM",
            "POLLINOUS_CORE",
            "SEAL_OF_GOLD",
            "TOASTY_MITTENS",
        ),
        turn_number=4,
        metadata={"gold": 10},
    )

    assert [
        (marker.relic_id, marker.kind, marker.amount, marker.target_id, marker.metadata)
        for marker in result.markers
    ] == [
        ("mercury_hourglass", "all_damage", 3, "all_enemies", {}),
        ("paels_blood", "draw_cards", 1, "player", {}),
        ("sai", "gain_block", 7, "player", {}),
        ("bound_phylactery", "player_resource", 1, "player", {"resource": "summon"}),
        ("snecko_eye", "draw_cards", 2, "player", {}),
        ("mr_struggles", "all_damage", 4, "all_enemies", {"turn_number": 4}),
        ("pollinous_core", "draw_cards", 2, "player", {"period": 4, "turn_number": 4}),
        ("seal_of_gold", "gold_delta", -5, "player", {"condition": "has_at_least_5_gold"}),
        ("seal_of_gold", "gain_energy", 1, "player", {"condition": "spent_5_gold"}),
        ("toasty_mittens", "exhaust_top_draw_pile", 1, "player", {}),
        ("toasty_mittens", "gain_status", 1, "player", {"status": "strength"}),
    ]


def test_turn_start_card_generation_and_draw_lock_relic_markers() -> None:
    result = turn_start(("FIDDLE", "CROSSBOW"))

    assert [
        (marker.relic_id, marker.kind, marker.amount, marker.target_id, marker.metadata)
        for marker in result.markers
    ] == [
        ("fiddle", "draw_cards", 2, "player", {}),
        ("fiddle", "gain_status", 1, "player", {"status": "no_draw_this_turn"}),
        (
            "crossbow",
            "add_card_to_hand",
            1,
            "player",
            {
                "selection": "random",
                "card_type": "attack",
                "free_to_play_this_turn": True,
            },
        ),
    ]


def test_turn_timing_relics_apply_on_their_specific_turns() -> None:
    opening = turn_start(("BREAD",), turn_number=1)
    later_bread = turn_start(("BREAD",), turn_number=2)
    second_turn = turn_start(("CANDELABRA", "HORN_CLEAT"), turn_number=2)
    third_turn = turn_start(("CAPTAINS_WHEEL", "CHANDELIER"), turn_number=3)
    quiet = turn_start(("CANDELABRA", "CAPTAINS_WHEEL", "CHANDELIER"), turn_number=4)

    assert [(marker.relic_id, marker.kind, marker.amount) for marker in opening.markers] == [
        ("bread", "gain_energy", -2)
    ]
    assert later_bread.markers[0].amount == 1
    assert [(marker.relic_id, marker.kind, marker.amount) for marker in second_turn.markers] == [
        ("candelabra", "gain_energy", 2),
        ("horn_cleat", "gain_block", 14),
    ]
    assert [(marker.relic_id, marker.kind, marker.amount) for marker in third_turn.markers] == [
        ("captains_wheel", "gain_block", 18),
        ("chandelier", "gain_energy", 3),
    ]
    assert quiet.markers == ()


def test_sts2_turn_timing_relics_apply_on_their_specific_turns() -> None:
    first = turn_start(("RING_OF_THE_DRAKE",), turn_number=1)
    third = turn_start(("SPARKLING_ROUGE", "PAELS_FLESH", "RING_OF_THE_DRAKE"), turn_number=3)
    fourth = turn_start(("SPARKLING_ROUGE", "PAELS_FLESH", "RING_OF_THE_DRAKE"), turn_number=4)

    assert [
        (marker.relic_id, marker.kind, marker.amount, marker.metadata)
        for marker in first.markers
    ] == [
        (
            "ring_of_the_drake",
            "draw_cards",
            2,
            {"condition": "first_3_turns", "turn_number": 1},
        )
    ]
    assert [
        (marker.relic_id, marker.kind, marker.amount, marker.metadata)
        for marker in third.markers
    ] == [
        (
            "sparkling_rouge",
            "gain_status",
            1,
            {"status": "strength", "turn_number": 3},
        ),
        (
            "sparkling_rouge",
            "gain_status",
            1,
            {"status": "dexterity", "turn_number": 3},
        ),
        (
            "paels_flesh",
            "gain_energy",
            1,
            {"condition": "turn_3_and_after", "turn_number": 3},
        ),
        (
            "ring_of_the_drake",
            "draw_cards",
            2,
            {"condition": "first_3_turns", "turn_number": 3},
        ),
    ]
    assert [(marker.relic_id, marker.kind, marker.amount) for marker in fourth.markers] == [
        ("paels_flesh", "gain_energy", 1)
    ]


def test_turn_end_relics_are_conditional() -> None:
    result = turn_end(
        (
            "CLOAK_CLASP",
            "PAELS_TEARS",
            "SCREAMING_FLAGON",
            "STONE_CALENDAR",
            "RUNIC_PYRAMID",
            "ART_OF_WAR",
        ),
        turn_number=7,
        metadata={"hand_size": 3, "energy": 1, "attacks_played_this_turn": 0},
    )
    empty_hand = turn_end(("SCREAMING_FLAGON",), metadata={"hand_size": 0})
    early_calendar = turn_end(("STONE_CALENDAR",), turn_number=6)
    attack_played = turn_end(("ART_OF_WAR",), metadata={"attacks_played_this_turn": 1})

    assert [
        (marker.relic_id, marker.kind, marker.amount, marker.metadata)
        for marker in result.markers
    ] == [
        ("cloak_clasp", "gain_block", 3, {"hand_size": 3}),
        (
            "paels_tears",
            "gain_status",
            2,
            {"status": "next_turn_energy", "condition": "unspent_energy"},
        ),
        ("stone_calendar", "all_damage", 52, {"turn_number": 7}),
        ("runic_pyramid", "retain_hand", 1, {"mode": "retain_full_hand"}),
        (
            "art_of_war",
            "gain_status",
            1,
            {"status": "next_turn_energy", "condition": "no_attacks_played"},
        ),
    ]
    assert empty_hand.markers[0].kind == "all_damage"
    assert empty_hand.markers[0].amount == 20
    assert early_calendar.markers == ()
    assert attack_played.markers == ()


def test_sts2_turn_end_relics_are_conditional() -> None:
    quiet_turn = turn_end(
        ("POCKETWATCH", "RIPPLE_BASIN", "RINGING_TRIANGLE"),
        turn_number=1,
        metadata={"cards_played_this_turn": 3, "attacks_played_this_turn": 0},
    )
    busy_turn = turn_end(
        ("POCKETWATCH", "RIPPLE_BASIN", "RINGING_TRIANGLE"),
        turn_number=2,
        metadata={"cards_played_this_turn": 4, "attacks_played_this_turn": 1},
    )
    missing_count = turn_end(("POCKETWATCH",), turn_number=1)

    assert [
        (marker.relic_id, marker.kind, marker.amount, marker.metadata)
        for marker in quiet_turn.markers
    ] == [
        (
            "pocketwatch",
            "gain_status",
            3,
            {
                "status": "next_turn_draw",
                "condition": "played_3_or_fewer_cards",
                "cards_played_this_turn": 3,
            },
        ),
        ("ripple_basin", "gain_block", 4, {"condition": "no_attacks_played"}),
        (
            "ringing_triangle",
            "retain_hand",
            1,
            {"mode": "retain_full_hand", "condition": "first_turn"},
        ),
    ]
    assert busy_turn.markers == ()
    assert missing_count.markers == ()


def test_sling_of_courage_and_bone_tea_are_contextual_start_relics() -> None:
    elite = start_of_combat(
        ("SLING_OF_COURAGE", "BONE_TEA", "BOOMING_CONCH"),
        encounter_type="elite",
        relic_counters={"bone_tea": 1},
        metadata={"opening_draw_count": 6},
    )
    normal = start_of_combat(
        ("SLING_OF_COURAGE", "BONE_TEA", "BOOMING_CONCH"),
        encounter_type="monster",
        relic_counters={"bone_tea": 0},
        metadata={"opening_draw_count": 6},
    )

    assert [
        (marker.relic_id, marker.kind, marker.amount, marker.metadata)
        for marker in elite.markers
    ] == [
        (
            "sling_of_courage",
            "gain_status",
            2,
            {"status": "strength", "condition": "elite_combat"},
        ),
        (
            "bone_tea",
            "upgrade_draw_pile_cards",
            6,
            {"mode": "opening_hand", "condition": "next_combat_charge", "next_counter": 0},
        ),
        (
            "booming_conch",
            "gain_energy",
            1,
            {"condition": "elite_combat"},
        ),
    ]
    assert normal.markers == ()


def test_more_contextual_start_relics_cover_potions_charges_and_low_hp() -> None:
    result = start_of_combat(
        ("DELICATE_FROND", "EMBER_TEA", "TEA_OF_DISCOURTESY", "RED_SKULL"),
        player_hp=30,
        player_max_hp=80,
        relic_counters={"ember_tea": 5, "tea_of_discourtesy": 1},
        metadata={"empty_potion_slots": 2},
    )
    no_context = start_of_combat(("DELICATE_FROND", "EMBER_TEA"), relic_counters={})

    assert [
        (marker.relic_id, marker.kind, marker.amount, marker.target_id, marker.metadata)
        for marker in result.markers
    ] == [
        (
            "delicate_frond",
            "random_potions_gained",
            2,
            "player",
            {"condition": "empty_potion_slots", "empty_potion_slots": 2},
        ),
        (
            "ember_tea",
            "gain_status",
            2,
            "player",
            {
                "status": "strength",
                "condition": "next_5_combat_charge",
                "next_counter": 4,
            },
        ),
        (
            "tea_of_discourtesy",
            "shuffle_status_into_draw_pile",
            2,
            "player",
            {
                "card_id": "dazed",
                "card_type": "status",
                "target": "self",
                "condition": "next_combat_charge",
                "next_counter": 0,
            },
        ),
        (
            "red_skull",
            "gain_status",
            3,
            "player",
            {"status": "strength", "condition": "hp_at_or_below_50_percent"},
        ),
    ]
    assert no_context.markers == ()


def test_card_played_relics_cover_type_cost_and_threshold_rules() -> None:
    attack = card_played(("DAUGHTER_OF_THE_WIND",), card_type="attack")
    shiv = card_played(("HELICAL_DART",), card_type="attack", card_id="shiv")
    costly = card_played(
        ("INTIMIDATING_HELMET", "IVORY_TILE"),
        card_type="skill",
        metadata={"card_cost": 3},
    )
    third_skill = card_played(
        ("LETTER_OPENER",),
        card_type="skill",
        metadata={"skills_played_this_turn": 2},
    )
    power = card_played(("LOST_WISP", "GAME_PIECE", "MUMMIFIED_HAND"), card_type="power")
    nunchaku = card_played(("NUNCHAKU",), card_type="attack", relic_counters={"nunchaku": 9})
    fourth_card = card_played(
        ("IRON_CLUB",),
        card_type="skill",
        metadata={"cards_played_this_turn": 3},
    )
    fifth_card = card_played(
        ("BRILLIANT_SCARF",),
        card_type="skill",
        metadata={"cards_played_this_turn": 4},
    )

    assert attack.markers[0].kind == "gain_block"
    assert attack.markers[0].amount == 1
    assert [marker.metadata["status"] for marker in shiv.markers] == [
        "dexterity",
        "dexterity_down",
    ]
    assert [(marker.kind, marker.amount) for marker in costly.markers] == [
        ("gain_block", 4),
        ("gain_energy", 1),
    ]
    assert third_skill.markers[0].kind == "all_damage"
    assert third_skill.markers[0].amount == 5
    assert [(marker.relic_id, marker.kind, marker.amount) for marker in power.markers] == [
        ("lost_wisp", "all_damage", 8),
        ("game_piece", "draw_cards", 1),
        ("mummified_hand", "make_random_card_free_this_turn", 1),
    ]
    assert nunchaku.markers[0].kind == "gain_energy"
    assert nunchaku.markers[0].amount == 1
    assert nunchaku.markers[0].metadata["next_counter"] == 0
    assert fourth_card.markers[0].kind == "draw_cards"
    assert fourth_card.markers[0].metadata == {"card_count": 4, "period": 4}
    assert fifth_card.markers[0].kind == "make_random_card_free_this_turn"
    assert fifth_card.markers[0].metadata == {"condition": "fifth_card_played", "card_count": 5}


def test_more_card_played_relics_cover_osty_first_power_and_type_set() -> None:
    osty = card_played(("BONE_FLUTE",), metadata={"card_tags": ("OstyAttack",)})
    permafrost = card_played(("PERMAFROST",), card_type="power")
    music = card_played(
        ("MUSIC_BOX",),
        card_type="attack",
        card_id="slash",
        metadata={"attacks_played_this_turn": 0},
    )
    rainbow = card_played(
        ("RAINBOW_RING",),
        card_type="power",
        metadata={"card_types_played_this_turn": ("attack", "skill")},
    )
    quiet = card_played(
        ("BONE_FLUTE", "PERMAFROST", "MUSIC_BOX", "RAINBOW_RING"),
        card_type="attack",
        metadata={"attacks_played_this_turn": 1, "card_types_played_this_turn": ("attack",)},
    )

    assert [(marker.relic_id, marker.kind, marker.amount) for marker in osty.markers] == [
        ("bone_flute", "gain_block", 2)
    ]
    assert permafrost.markers[0].metadata == {
        "condition": "first_power_this_combat",
        "next_counter": 1,
    }
    assert music.markers[0].metadata == {
        "selection": "copy_played_card",
        "copy_source_card_id": "slash",
        "card_type": "attack",
        "ethereal": True,
        "condition": "first_attack_this_turn",
    }
    assert [
        (marker.kind, marker.amount, marker.metadata["status"])
        for marker in rainbow.markers
    ] == [
        ("gain_status", 1, "strength"),
        ("gain_status", 1, "dexterity"),
    ]
    assert quiet.markers == ()


def test_happy_flower_periodic_turn_start_energy_uses_turn_or_counter() -> None:
    quiet = turn_start(("HAPPY_FLOWER",), turn_number=2)
    triggered = turn_start(("HAPPY_FLOWER",), turn_number=3)
    counter_triggered = turn_start(("HAPPY_FLOWER",), relic_counters={"happy_flower": 2})

    assert quiet.markers == ()
    assert triggered.energy_delta == 1
    assert triggered.markers[0].kind == "gain_energy"
    assert triggered.markers[0].metadata == {"period": 3, "turn_number": 3}
    assert counter_triggered.energy_delta == 1
    assert counter_triggered.markers[0].metadata == {"period": 3, "next_counter": 0}


def test_card_played_attack_counter_relics_emit_only_on_threshold() -> None:
    missed = card_played(
        ("SHURIKEN",),
        card_type="attack",
        metadata={"attacks_played_this_turn": 1},
    )
    triggered = card_played(
        ("SHURIKEN", "KUNAI", "ORNAMENTAL_FAN"),
        card_type="attack",
        metadata={"attacks_played_this_turn": 2},
    )

    assert missed.markers == ()
    assert [(marker.kind, marker.amount) for marker in triggered.markers] == [
        ("gain_status", 1),
        ("gain_status", 1),
        ("gain_block", 4),
    ]
    assert triggered.markers[0].metadata["status"] == "strength"
    assert triggered.markers[1].metadata["status"] == "dexterity"
    assert triggered.block_delta == 4


def test_vulnerable_math_markers_for_odd_mushroom_and_paper_phrog() -> None:
    taken = damage_taken(("ODD_MUSHROOM",), player_statuses={"Vulnerable": 1})
    dealt = damage_dealt(("PAPER_PHROG",), target_statuses={"vulnerable": 1}, target_id="jaw_worm")

    assert taken.markers[0].kind == "modify_vulnerable_damage_taken"
    assert taken.markers[0].amount == 125
    assert taken.markers[0].metadata["normal_multiplier_percent"] == 150
    assert dealt.markers[0].kind == "modify_vulnerable_damage_dealt"
    assert dealt.markers[0].amount == 175
    assert dealt.markers[0].target_id == "jaw_worm"


def test_damage_relic_markers_cover_card_and_hp_loss_modifiers() -> None:
    strike = damage_dealt(
        ("STRIKE_DUMMY", "FAKE_STRIKE_DUMMY"),
        card_type="attack",
        card_id="pommel_strike",
    )
    upgraded = damage_dealt(
        ("MINIATURE_CANNON",),
        card_type="attack",
        card_id="slash",
        metadata={"upgraded": True},
    )
    enchanted = damage_dealt(
        ("MYSTIC_LIGHTER",),
        card_type="attack",
        card_id="slash",
        metadata={"enchanted": True},
    )
    krane = damage_taken(("PAPER_KRANE",), target_statuses={"weak": 1})
    rod = damage_taken(("TUNGSTEN_ROD",))
    remnant = damage_taken(("BEATING_REMNANT",))
    clay = damage_taken(("SELF_FORMING_CLAY",), metadata={"hp_loss": 2})
    diadem = damage_taken(("DIAMOND_DIADEM",), metadata={"cards_played_this_turn": 2})

    assert [(marker.relic_id, marker.kind, marker.amount) for marker in strike.markers] == [
        ("strike_dummy", "modify_card_damage", 3),
        ("fake_strike_dummy", "modify_card_damage", 1),
    ]
    assert upgraded.markers[0].amount == 3
    assert enchanted.markers[0].amount == 9
    assert krane.markers[0].kind == "modify_weak_damage_taken"
    assert krane.markers[0].amount == 60
    assert rod.markers[0].kind == "reduce_hp_loss"
    assert rod.markers[0].amount == 1
    assert remnant.markers[0].kind == "cap_hp_loss_per_turn"
    assert remnant.markers[0].amount == 20
    assert clay.markers[0].kind == "gain_status"
    assert clay.markers[0].metadata["status"] == "next_turn_block"
    assert diadem.markers[0].kind == "modify_damage_taken"
    assert diadem.markers[0].amount == 50


def test_more_damage_relic_markers_cover_thresholds_and_doom() -> None:
    pen_counter = damage_dealt(
        ("PEN_NIB",),
        card_type="attack",
        relic_counters={"pen_nib": 8},
    )
    pen_trigger = damage_dealt(
        ("PEN_NIB",),
        card_type="attack",
        relic_counters={"pen_nib": 9},
    )
    boot = damage_dealt(
        ("THE_BOOT",),
        card_type="attack",
        metadata={"unblocked_damage": 4},
    )
    boot_quiet = damage_dealt(
        ("THE_BOOT",),
        card_type="attack",
        metadata={"unblocked_damage": 5},
    )
    boot_osty = damage_dealt(
        ("THE_BOOT",),
        card_type="skill",
        metadata={"card_tags": ("OstyAttack",), "unblocked_damage": 4},
    )
    demon = damage_taken(
        ("DEMON_TONGUE",),
        player_hp=40,
        player_max_hp=80,
        metadata={"hp_loss": 6, "on_player_turn": True},
    )
    demon_used = damage_taken(
        ("DEMON_TONGUE",),
        relic_counters={"demon_tongue": 1},
        metadata={"hp_loss": 6, "on_player_turn": True},
    )
    demon_off_turn = damage_taken(
        ("DEMON_TONGUE",),
        metadata={"hp_loss": 6, "turn_owner": "monster"},
    )
    undying = damage_taken(
        ("UNDYING_SIGIL",),
        target_statuses={"doom": 12},
        metadata={"attacker_hp": 10},
    )
    low_doom = damage_taken(
        ("UNDYING_SIGIL",),
        target_statuses={"doom": 8},
        metadata={"attacker_hp": 10},
    )

    assert pen_counter.markers[0].kind == "relic_counter_changed"
    assert pen_counter.markers[0].metadata == {"counter": 9, "period": 10}
    assert pen_trigger.markers[0].kind == "modify_card_damage"
    assert pen_trigger.markers[0].amount == 200
    assert pen_trigger.markers[0].metadata == {
        "condition": "tenth_attack",
        "operation": "multiply_percent",
        "multiplier_percent": 200,
        "period": 10,
        "next_counter": 0,
    }
    assert boot.markers[0].kind == "modify_card_damage"
    assert boot.markers[0].metadata == {
        "condition": "unblocked_attack_damage_at_most_4",
        "operation": "minimum",
        "minimum": 5,
        "threshold": 4,
        "unblocked_damage": 4,
    }
    assert boot_osty.markers[0].kind == "modify_card_damage"
    assert boot_osty.markers[0].metadata == {
        "condition": "unblocked_attack_damage_at_most_4",
        "operation": "minimum",
        "minimum": 5,
        "threshold": 4,
        "unblocked_damage": 4,
        "card_tag": "OstyAttack",
    }
    assert boot_quiet.markers == ()
    assert demon.markers[0].kind == "heal_player"
    assert demon.markers[0].amount == 6
    assert demon.markers[0].metadata == {
        "condition": "first_hp_loss_on_player_turn",
        "hp_loss": 6,
        "next_counter": 1,
    }
    assert demon_used.markers == ()
    assert demon_off_turn.markers == ()
    assert undying.markers[0].kind == "modify_damage_taken"
    assert undying.markers[0].amount == 50
    assert undying.markers[0].metadata == {
        "condition": "attacker_doom_at_least_hp",
        "multiplier_percent": 50,
        "attacker_hp": 10,
        "doom": 12,
    }
    assert low_doom.markers == ()


def test_additional_damage_relics_cover_low_hp_fatal_and_minion_cards() -> None:
    minion = damage_dealt(
        ("VITRUVIAN_MINION",),
        card_type="attack",
        card_id="clockwork_minion",
    )
    red_skull = damage_taken(
        ("RED_SKULL",),
        player_hp=45,
        player_max_hp=80,
        metadata={"hp_loss": 8},
    )
    tail = damage_taken(
        ("LIZARD_TAIL",),
        player_hp=3,
        player_max_hp=80,
        metadata={"hp_loss": 5},
    )
    tail_used = damage_taken(
        ("LIZARD_TAIL",),
        player_hp=3,
        player_max_hp=80,
        relic_counters={"lizard_tail": 1},
        metadata={"hp_loss": 5},
    )

    assert minion.markers[0].metadata == {
        "condition": "card_contains_minion",
        "operation": "multiply_percent",
        "multiplier_percent": 200,
    }
    assert red_skull.markers[0].metadata == {
        "status": "strength",
        "condition": "hp_crossed_to_at_or_below_50_percent",
        "hp_after_loss": 37,
    }
    assert tail.markers[0].kind == "heal_player"
    assert tail.markers[0].amount == 40
    assert tail.markers[0].metadata == {
        "condition": "fatal_damage",
        "target_hp": 40,
        "hp_loss": 5,
        "next_counter": 1,
    }
    assert tail_used.markers == ()


def test_centennial_puzzle_draws_once_after_first_hp_loss() -> None:
    first_loss = damage_taken(("CENTENNIAL_PUZZLE",), metadata={"hp_loss": 1})
    blocked = damage_taken(("CENTENNIAL_PUZZLE",), metadata={"hp_loss": 0})
    already_used = damage_taken(
        ("CENTENNIAL_PUZZLE",),
        relic_counters={"centennial_puzzle": 1},
        metadata={"hp_loss": 3},
    )

    assert first_loss.markers[0].kind == "draw_cards"
    assert first_loss.markers[0].amount == 3
    assert first_loss.markers[0].metadata == {
        "condition": "first_hp_loss_this_combat",
        "next_counter": 1,
    }
    assert blocked.markers == ()
    assert already_used.markers == ()


def test_black_star_monster_kill_marker_is_elite_only() -> None:
    elite = combat_end(("BLACK_STAR",), encounter_type="elite")
    normal = combat_end(("BLACK_STAR",), encounter_type="normal")
    killed = damage_dealt(("BLACK_STAR",), encounter_type="elite")

    assert elite.markers == ()
    assert normal.markers == ()
    assert killed.markers == ()

    result = monster_killed(("BLACK_STAR",), encounter_type="elite")
    non_elite = monster_killed(("BLACK_STAR",), encounter_type="monster")

    assert result.markers[0].kind == "reward_relic_count_delta"
    assert result.markers[0].amount == 1
    assert non_elite.markers == ()


def test_gremlin_horn_rewards_energy_and_draw_on_monster_kill() -> None:
    result = monster_killed(("GREMLIN_HORN",), target_id="jaw_worm")

    assert [(marker.kind, marker.amount, marker.target_id) for marker in result.markers] == [
        ("gain_energy", 1, "player"),
        ("draw_cards", 1, "player"),
    ]
    assert all(marker.metadata == {"condition": "enemy_killed"} for marker in result.markers)
    assert result.energy_delta == 1


def test_more_monster_kill_and_combat_end_relic_markers() -> None:
    repair = monster_killed(
        ("BOOK_REPAIR_KNIFE",),
        player_hp=70,
        player_max_hp=80,
        metadata={"death_reason": "doom"},
    )
    minion = monster_killed(
        ("BOOK_REPAIR_KNIFE",),
        player_hp=70,
        player_max_hp=80,
        metadata={"death_reason": "doom", "target_is_minion": True},
    )
    sword = monster_killed(
        ("SWORD_OF_STONE",),
        encounter_type="elite",
        relic_counters={"sword_of_stone": 4},
    )
    hammer = monster_killed(("WAR_HAMMER",), encounter_type="elite")
    tooth = combat_end(("PAELS_TOOTH",), metadata={"paels_tooth_removed_count": 2})

    assert repair.markers[0].kind == "heal_player"
    assert repair.markers[0].amount == 3
    assert minion.markers == ()
    assert sword.markers[0].metadata == {
        "counter": 5,
        "period": 5,
        "condition": "elite_killed",
        "transform_ready": True,
    }
    assert hammer.markers[0].kind == "upgrade_deck_cards"
    assert hammer.markers[0].amount == 4
    assert hammer.markers[0].metadata == {"selection": "random", "condition": "elite_killed"}
    assert tooth.markers[0].kind == "add_deck_cards"
    assert tooth.markers[0].metadata == {
        "selection": "random_removed_by_relic",
        "upgraded": True,
        "condition": "after_combat",
    }


def test_orichalcum_turn_end_and_preserved_insect_elite_markers_are_conditional() -> None:
    orichalcum = turn_end(("ORICHALCUM",), player_block=0)
    blocked = turn_end(("ORICHALCUM",), player_block=4)
    elite = start_of_combat(("PRESERVED_INSECT",), encounter_type="elite")
    normal = start_of_combat(("PRESERVED_INSECT",), encounter_type="normal")
    conch = start_of_combat(("BOOMING_CONCH",), encounter_type="elite")
    quiet_conch = start_of_combat(("BOOMING_CONCH",), encounter_type="normal")

    assert orichalcum.block_delta == 6
    assert orichalcum.markers[0].metadata == {
        "condition": "player_block_is_zero",
        "player_block": 0,
    }
    assert blocked.markers == ()
    assert elite.markers[0].kind == "elite_monster_hp_multiplier"
    assert elite.markers[0].amount == 75
    assert elite.markers[0].metadata["hp_reduction_percent"] == 25
    assert normal.markers == ()
    assert conch.markers[0].kind == "gain_energy"
    assert conch.markers[0].amount == 1
    assert conch.markers[0].metadata == {"condition": "elite_combat"}
    assert quiet_conch.markers == ()


def test_more_turn_end_relics_preserve_energy_and_block() -> None:
    result = turn_end(("ICE_CREAM", "STURDY_CLAMP"), player_block=14, metadata={"energy": 3})
    quiet = turn_end(("ICE_CREAM", "STURDY_CLAMP"), player_block=0, metadata={"energy": 0})

    assert [
        (marker.relic_id, marker.kind, marker.amount, marker.metadata)
        for marker in result.markers
    ] == [
        (
            "ice_cream",
            "gain_status",
            3,
            {
                "status": "next_turn_energy",
                "condition": "unspent_energy_conserved",
                "energy": 3,
            },
        ),
        (
            "sturdy_clamp",
            "gain_status",
            10,
            {
                "status": "next_turn_block",
                "condition": "block_persists",
                "player_block": 14,
                "limit": 10,
            },
        ),
    ]
    assert quiet.markers == ()


def test_combat_end_healing_is_capped() -> None:
    result = combat_end(("BURNING_BLOOD", "BLACK_BLOOD"), player_hp=75, player_max_hp=80)

    assert result.hp_delta == 5
    assert [marker.amount for marker in result.markers] == [5, 0]


def test_pantograph_and_meat_on_the_bone_are_contextual() -> None:
    boss = start_of_combat(("PANTOGRAPH",), encounter_type="boss")
    hallway = start_of_combat(("PANTOGRAPH",), encounter_type="monster")
    low_hp = combat_end(("MEAT_ON_THE_BONE",), player_hp=40, player_max_hp=80)
    high_hp = combat_end(("MEAT_ON_THE_BONE",), player_hp=41, player_max_hp=80)

    assert boss.markers[0].kind == "heal_player"
    assert boss.markers[0].amount == 25
    assert hallway.markers == ()
    assert low_hp.markers[0].amount == 12
    assert high_hp.markers == ()


def test_remaining_unknown_start_turn_and_end_relic_markers() -> None:
    opening = start_of_combat(("GAMBLING_CHIP", "PETRIFIED_TOAD"))
    turn = turn_start(
        ("EMOTION_CHIP", "HISTORY_COURSE"),
        metadata={
            "lost_hp_previous_turn": True,
            "last_played_card_type": "Attack",
            "last_played_card_id": "Twin Strike",
        },
    )
    ending = turn_end(
        ("BOOKMARK", "PAELS_EYE", "PARRYING_SHIELD"),
        player_block=12,
        metadata={"retained_card_count": 1, "cards_played_this_turn": 0, "hand_size": 2},
    )
    reward = combat_end(("LAVA_LAMP",), metadata={"damage_taken_this_combat": 0})
    damaged = combat_end(("LAVA_LAMP",), metadata={"damage_taken_this_combat": 1})

    assert [
        (marker.relic_id, marker.kind, marker.amount, marker.metadata)
        for marker in opening.markers
    ] == [
        (
            "gambling_chip",
            "opening_hand_discard_redraw",
            None,
            {"selection": "any_number", "draw_equal_to_discarded": True},
        ),
        ("petrified_toad", "procure_potion", 1, {"potion_id": "potion_shaped_rock"}),
    ]
    assert [
        (marker.relic_id, marker.kind, marker.metadata)
        for marker in turn.markers
    ] == [
        (
            "emotion_chip",
            "trigger_orb_passive",
            {"selector": "all", "condition": "lost_hp_previous_turn"},
        ),
        (
            "history_course",
            "play_card_copy",
            {
                "selection": "last_played_attack_or_skill",
                "card_type": "attack",
                "condition": "start_of_turn",
                "copy_source_card_id": "twin_strike",
            },
        ),
    ]
    assert [
        (marker.relic_id, marker.kind, marker.amount)
        for marker in ending.markers
    ] == [
        ("bookmark", "reduce_retained_card_cost", 1),
        ("paels_eye", "exhaust_hand", 2),
        ("paels_eye", "take_extra_turn", 1),
        ("parrying_shield", "random_damage", 6),
    ]
    assert reward.markers[0].kind == "upgrade_card_rewards"
    assert reward.markers[0].target_id == "reward"
    assert damaged.markers == ()


def test_remaining_unknown_card_played_relic_markers() -> None:
    result = card_played(
        (
            "CHEMICAL_X",
            "KUSARIGAMA",
            "RAZOR_TOOTH",
            "THROWING_AXE",
            "UNSETTLING_LAMP",
        ),
        card_type="attack",
        card_id="strike",
        target_id="jaw_worm",
        metadata={
            "card_cost": "X",
            "attacks_played_this_turn": 2,
            "cards_played_this_combat": 0,
            "debuffs_enemy": True,
        },
    )

    assert [
        (marker.relic_id, marker.kind, marker.amount, marker.target_id)
        for marker in result.markers
    ] == [
        ("chemical_x", "x_card_effect_bonus", 2, "player"),
        ("kusarigama", "random_damage", 6, "random_enemy"),
        ("razor_tooth", "upgrade_played_card", 1, "player"),
        ("throwing_axe", "play_card_again", 1, "player"),
        ("unsettling_lamp", "modify_debuff_application", 200, "jaw_worm"),
    ]
    assert result.markers[1].metadata == {"attack_count": 3, "period": 3}
    assert result.markers[3].metadata == {
        "selection": "played_card",
        "condition": "first_card_played_this_combat",
        "next_counter": 1,
        "copy_source_card_id": "strike",
        "copy_source_card_type": "attack",
        "copy_source_target_id": "jaw_worm",
    }


def test_remaining_unknown_exhaust_and_discard_relic_markers() -> None:
    exhausted = resolve_combat_relic_hook(
        ("BURNING_STICKS", "CHARONS_ASHES", "FORGOTTEN_SOUL", "JOSS_PAPER"),
        CombatRelicHook.CARD_EXHAUSTED,
        card_type="skill",
        card_id="defend",
        relic_counters={"joss_paper": 4},
    )
    discarded = resolve_combat_relic_hook(
        ("TINGSHA", "TOUGH_BANDAGES"),
        CombatRelicHook.CARD_DISCARDED,
        metadata={"discarded_count": 2, "on_player_turn": True},
    )

    assert [
        (marker.relic_id, marker.kind, marker.amount, marker.target_id)
        for marker in exhausted.markers
    ] == [
        ("burning_sticks", "add_card_to_hand", 1, "player"),
        ("charons_ashes", "all_damage", 3, "all_enemies"),
        ("forgotten_soul", "random_damage", 1, "random_enemy"),
        ("joss_paper", "draw_cards", 1, "player"),
    ]
    assert exhausted.markers[0].metadata == {
        "selection": "copy_exhausted_card",
        "card_type": "skill",
        "condition": "first_skill_exhausted_this_combat",
        "next_counter": 1,
        "copy_source_card_id": "defend",
    }
    assert [
        (marker.relic_id, marker.kind, marker.amount)
        for marker in discarded.markers
    ] == [
        ("tingsha", "random_damage", 6),
        ("tough_bandages", "gain_block", 6),
    ]


def test_remaining_unknown_custom_event_relic_markers() -> None:
    created = resolve_combat_relic_hook(
        ("REGALITE",),
        CombatRelicHook.CARD_CREATED,
        metadata={"created_count": 2},
    )
    resource = resolve_combat_relic_hook(
        ("GALACTIC_DUST", "MINI_REGENT"),
        CombatRelicHook.RESOURCE_SPENT,
        metadata={"resource": "star", "amount_spent": 10},
    )
    orb = resolve_combat_relic_hook(
        ("METRONOME",),
        CombatRelicHook.ORB_CHANNELED,
        relic_counters={"metronome": 6},
    )
    cables = resolve_combat_relic_hook(
        ("GOLD_PLATED_CABLES",),
        CombatRelicHook.ORB_PASSIVE_TRIGGERED,
    )
    broken = resolve_combat_relic_hook(
        ("HAND_DRILL",),
        CombatRelicHook.ENEMY_BLOCK_BROKEN,
        target_id="sentry",
    )
    shuffled = resolve_combat_relic_hook(("THE_ABACUS",), CombatRelicHook.DRAW_PILE_SHUFFLED)
    empty = resolve_combat_relic_hook(
        ("UNCEASING_TOP",),
        CombatRelicHook.HAND_EMPTY,
        metadata={"hand_size": 0, "on_player_turn": True},
    )
    potion = resolve_combat_relic_hook(("REPTILE_TRINKET",), CombatRelicHook.POTION_USED)
    applied = resolve_combat_relic_hook(
        ("SNECKO_SKULL",),
        CombatRelicHook.STATUS_APPLIED,
        target_id="jaw_worm",
        metadata={"status": "poison"},
    )
    gained = resolve_combat_relic_hook(
        ("RUINED_HELMET",),
        CombatRelicHook.STATUS_GAINED,
        metadata={"status": "strength"},
    )
    card_block = resolve_combat_relic_hook(
        ("PAELS_LEGION", "VAMBRACE"),
        CombatRelicHook.CARD_BLOCK_GAINED,
    )

    assert created.markers[0].amount == 4
    assert [(marker.relic_id, marker.kind, marker.amount) for marker in resource.markers] == [
        ("galactic_dust", "gain_block", 10),
        ("mini_regent", "gain_status", 1),
    ]
    assert orb.markers[0].kind == "all_damage"
    assert orb.markers[0].amount == 30
    assert cables.markers[0].metadata == {
        "selector": "rightmost",
        "condition": "rightmost_orb_passive",
    }
    assert broken.markers[0].metadata == {
        "status": "vulnerable",
        "condition": "enemy_block_broken",
    }
    assert shuffled.markers[0].amount == 6
    assert empty.markers[0].kind == "draw_cards"
    assert [marker.metadata["status"] for marker in potion.markers] == [
        "strength",
        "strength_down",
    ]
    assert applied.markers[0].target_id == "jaw_worm"
    assert gained.markers[0].kind == "modify_status_gain"
    assert [
        (marker.relic_id, marker.kind, marker.metadata["next_counter"])
        for marker in card_block.markers
    ] == [
        ("paels_legion", "modify_card_block", 2),
        ("vambrace", "modify_card_block", 1),
    ]


def test_remaining_unknown_relics_are_registered_by_hook() -> None:
    expected = {
        CombatRelicHook.START_OF_COMBAT: {"gambling_chip", "petrified_toad"},
        CombatRelicHook.TURN_START: {"emotion_chip", "history_course"},
        CombatRelicHook.TURN_END: {"bookmark", "paels_eye", "parrying_shield"},
        CombatRelicHook.CARD_PLAYED: {
            "chemical_x",
            "kusarigama",
            "razor_tooth",
            "throwing_axe",
            "unsettling_lamp",
        },
        CombatRelicHook.CARD_EXHAUSTED: {
            "burning_sticks",
            "charons_ashes",
            "forgotten_soul",
            "joss_paper",
        },
        CombatRelicHook.CARD_DISCARDED: {"tingsha", "tough_bandages"},
        CombatRelicHook.CARD_CREATED: {"regalite"},
        CombatRelicHook.CARD_BLOCK_GAINED: {"paels_legion", "vambrace"},
        CombatRelicHook.DRAW_PILE_SHUFFLED: {"the_abacus"},
        CombatRelicHook.ENEMY_BLOCK_BROKEN: {"hand_drill"},
        CombatRelicHook.HAND_EMPTY: {"unceasing_top"},
        CombatRelicHook.ORB_CHANNELED: {"metronome"},
        CombatRelicHook.ORB_PASSIVE_TRIGGERED: {"gold_plated_cables"},
        CombatRelicHook.POTION_USED: {"reptile_trinket"},
        CombatRelicHook.RESOURCE_SPENT: {"galactic_dust", "mini_regent"},
        CombatRelicHook.STATUS_APPLIED: {"snecko_skull"},
        CombatRelicHook.STATUS_GAINED: {"ruined_helmet"},
        CombatRelicHook.COMBAT_END: {"lava_lamp"},
    }
    expected_ids = set().union(*expected.values())

    for hook, relic_ids in expected.items():
        assert relic_ids <= supported_combat_relic_ids(hook)
    assert expected_ids <= supported_combat_relic_ids()


def test_card_copy_and_replay_markers_preserve_runtime_source_context() -> None:
    history = turn_start(
        ("HISTORY_COURSE",),
        metadata={
            "last_played_card_type": "Skill",
            "last_played_card_id": "Zap",
            "last_played_card_instance_id": "zap:7",
            "last_played_target_id": "jaw_worm",
            "last_played_card_cost": 1,
        },
    )
    axe = card_played(
        ("THROWING_AXE",),
        card_type="Attack",
        card_id="Pommel Strike",
        target_id="jaw_worm",
        metadata={
            "cards_played_this_combat": 0,
            "played_card_instance_id": "pommel_strike:2",
        },
    )

    assert history.markers[0].kind == "play_card_copy"
    assert history.markers[0].metadata == {
        "selection": "last_played_attack_or_skill",
        "card_type": "skill",
        "condition": "start_of_turn",
        "copy_source_card_id": "zap",
        "copy_source_card_instance_id": "zap:7",
        "copy_source_target_id": "jaw_worm",
        "copy_source_card_cost": 1,
    }
    assert axe.markers[0].kind == "play_card_again"
    assert axe.markers[0].metadata == {
        "selection": "played_card",
        "condition": "first_card_played_this_combat",
        "next_counter": 1,
        "copy_source_card_id": "pommel_strike",
        "copy_source_card_type": "attack",
        "copy_source_card_instance_id": "pommel_strike:2",
        "copy_source_target_id": "jaw_worm",
    }


def test_modifier_markers_preserve_base_amount_context_for_runtime_execution() -> None:
    strike = damage_dealt(
        ("STRIKE_DUMMY",),
        card_type="attack",
        card_id="strike",
        metadata={"base_damage": 6},
    )
    pen = damage_dealt(
        ("PEN_NIB",),
        card_type="attack",
        relic_counters={"pen_nib": 9},
        metadata={"damage": 8},
    )
    phrog = damage_dealt(
        ("PAPER_PHROG",),
        target_statuses={"vulnerable": 1},
        metadata={"amount": 12},
    )
    rod = damage_taken(("TUNGSTEN_ROD",), metadata={"hp_loss": 9})
    remnant = damage_taken(
        ("BEATING_REMNANT",),
        metadata={"hp_loss": 35, "hp_lost_this_turn": 12},
    )
    diadem = damage_taken(
        ("DIAMOND_DIADEM",),
        metadata={"hp_loss": 7, "cards_played_this_turn": 2},
    )
    block = resolve_combat_relic_hook(
        ("PAELS_LEGION", "VAMBRACE"),
        CombatRelicHook.CARD_BLOCK_GAINED,
        metadata={"amount": 7},
    )
    status = resolve_combat_relic_hook(
        ("RUINED_HELMET",),
        CombatRelicHook.STATUS_GAINED,
        metadata={"status": "strength", "amount": 3},
    )

    assert strike.markers[0].metadata["base_damage"] == 6
    assert pen.markers[0].metadata["base_damage"] == 8
    assert phrog.markers[0].metadata["base_damage"] == 12
    assert rod.markers[0].metadata == {
        "condition": "would_lose_hp",
        "operation": "subtract",
        "reduction": 1,
        "incoming_hp_loss": 9,
    }
    assert remnant.markers[0].metadata == {
        "condition": "hp_loss_cap_per_turn",
        "operation": "cap_per_turn",
        "cap": 20,
        "incoming_hp_loss": 35,
        "hp_lost_this_turn": 12,
    }
    assert diadem.markers[0].metadata == {
        "condition": "played_2_or_fewer_cards_this_turn",
        "incoming_hp_loss": 7,
    }
    assert [marker.metadata["base_block"] for marker in block.markers] == [7, 7]
    assert status.markers[0].metadata["base_status_amount"] == 3


def test_combat_relic_marker_kinds_are_audited_against_runtime_surface() -> None:
    emitted = _emitted_marker_kinds()
    generic_executor = {
        "add_card_to_draw_pile",
        "add_card_to_hand",
        "all_damage",
        "apply_status",
        "channel_orb",
        "conditional_damage_taken_multiplier",
        "draw_cards",
        "elite_monster_hp_multiplier",
        "exhaust_hand",
        "exhaust_top_draw_pile",
        "gain_block",
        "gain_energy",
        "gain_status",
        "gold_delta",
        "heal_player",
        "lose_hp",
        "make_random_card_free_this_turn",
        "max_hp_delta",
        "modify_card_block",
        "modify_card_damage",
        "modify_damage_taken",
        "modify_debuff_application",
        "modify_status_gain",
        "modify_vulnerable_damage_dealt",
        "modify_vulnerable_damage_taken",
        "modify_weak_damage_taken",
        "move_card_type_from_draw_to_hand",
        "move_zero_cost_cards_to_hand",
        "orb_slot_delta",
        "periodic_energy_check",
        "player_resource",
        "random_damage",
        "reduce_hp_loss",
        "reduce_retained_card_cost",
        "relic_counter_changed",
        "retain_hand",
        "shuffle_status_into_draw_pile",
        "take_extra_turn",
        "trigger_orb_passive",
        "turn_card_play_limit",
        "upgrade_card_rewards",
        "upgrade_draw_pile_cards",
        "upgrade_played_card",
        "x_card_effect_bonus",
    }
    direct_engine_or_runtime_hook = {
        "cap_hp_loss_per_turn",
        "play_card_again",
    }
    transition_integration_needed = {
        "add_deck_cards",
        "opening_hand_discard_redraw",
        "play_card_copy",
        "procure_potion",
        "random_potions_gained",
        "reward_relic_count_delta",
        "upgrade_deck_cards",
    }

    assert emitted == (
        generic_executor | direct_engine_or_runtime_hook | transition_integration_needed
    )


def test_unsupported_combat_relics_return_blockers_with_source_ids() -> None:
    relic = {
        "id": "UNHANDLED_STARTER",
        "name": "Unhandled Starter",
        "description": "At the start of each combat, do a very specific thing.",
    }

    result = start_of_combat((relic,))
    blockers = unsupported_combat_relic_handlers((relic,))

    assert len(result.blockers) == 1
    assert result.blockers[0].hook is CombatRelicHook.START_OF_COMBAT
    assert result.blockers[0].source_id == "unhandled_starter"
    assert "No pure combat relic helper" in result.blockers[0].reason
    assert blockers == result.blockers


def test_inventory_conditioned_relics_check_current_potion_count() -> None:
    relic = {
        "id": "BELT_BUCKLE",
        "name": "Belt Buckle",
        "description": (
            "While you have no potions, you have [blue]2[/blue] additional "
            "[gold]Dexterity[/gold]."
        ),
    }

    blockers = unsupported_combat_relic_handlers((relic,))
    active = start_of_combat((relic,), metadata={"potion_count": 0})
    blocked = start_of_combat((relic,), metadata={"potion_count": 1})

    assert blockers == ()
    assert active.markers[0].kind == "gain_status"
    assert active.markers[0].amount == 2
    assert active.markers[0].metadata["status"] == "dexterity"
    assert blocked.markers == ()


def test_supported_combat_relic_ids_are_hook_scoped() -> None:
    assert "anchor" in supported_combat_relic_ids(CombatRelicHook.START_OF_COMBAT)
    assert "bag_of_preparation" in supported_combat_relic_ids(
        CombatRelicHook.START_OF_COMBAT
    )
    assert "brimstone" in supported_combat_relic_ids(CombatRelicHook.TURN_START)
    assert "happy_flower" in supported_combat_relic_ids(CombatRelicHook.TURN_START)
    assert "crossbow" in supported_combat_relic_ids(CombatRelicHook.TURN_START)
    assert "fiddle" in supported_combat_relic_ids(CombatRelicHook.TURN_START)
    assert "ring_of_the_drake" in supported_combat_relic_ids(CombatRelicHook.TURN_START)
    assert "paels_flesh" in supported_combat_relic_ids(CombatRelicHook.TURN_START)
    assert "sparkling_rouge" in supported_combat_relic_ids(CombatRelicHook.TURN_START)
    assert "orichalcum" in supported_combat_relic_ids(CombatRelicHook.TURN_END)
    assert "pocketwatch" in supported_combat_relic_ids(CombatRelicHook.TURN_END)
    assert "ringing_triangle" in supported_combat_relic_ids(CombatRelicHook.TURN_END)
    assert "ripple_basin" in supported_combat_relic_ids(CombatRelicHook.TURN_END)
    assert "sling_of_courage" in supported_combat_relic_ids(
        CombatRelicHook.START_OF_COMBAT
    )
    assert "bone_tea" in supported_combat_relic_ids(CombatRelicHook.START_OF_COMBAT)
    assert "centennial_puzzle" in supported_combat_relic_ids(CombatRelicHook.DAMAGE_TAKEN)
    assert "demon_tongue" in supported_combat_relic_ids(CombatRelicHook.DAMAGE_TAKEN)
    assert "undying_sigil" in supported_combat_relic_ids(CombatRelicHook.DAMAGE_TAKEN)
    assert "pen_nib" in supported_combat_relic_ids(CombatRelicHook.DAMAGE_DEALT)
    assert "the_boot" in supported_combat_relic_ids(CombatRelicHook.DAMAGE_DEALT)
    assert "gremlin_horn" in supported_combat_relic_ids(CombatRelicHook.MONSTER_KILLED)
    assert "big_hat" in supported_combat_relic_ids(CombatRelicHook.START_OF_COMBAT)
    assert "choices_paradox" in supported_combat_relic_ids(CombatRelicHook.START_OF_COMBAT)
    assert "orange_dough" in supported_combat_relic_ids(CombatRelicHook.START_OF_COMBAT)
    assert "toolbox" in supported_combat_relic_ids(CombatRelicHook.START_OF_COMBAT)
    assert "vexing_puzzlebox" in supported_combat_relic_ids(CombatRelicHook.START_OF_COMBAT)
    assert "delicate_frond" in supported_combat_relic_ids(CombatRelicHook.START_OF_COMBAT)
    assert "ember_tea" in supported_combat_relic_ids(CombatRelicHook.START_OF_COMBAT)
    assert "tea_of_discourtesy" in supported_combat_relic_ids(
        CombatRelicHook.START_OF_COMBAT
    )
    assert "bone_flute" in supported_combat_relic_ids(CombatRelicHook.CARD_PLAYED)
    assert "music_box" in supported_combat_relic_ids(CombatRelicHook.CARD_PLAYED)
    assert "permafrost" in supported_combat_relic_ids(CombatRelicHook.CARD_PLAYED)
    assert "rainbow_ring" in supported_combat_relic_ids(CombatRelicHook.CARD_PLAYED)
    assert "vitruvian_minion" in supported_combat_relic_ids(CombatRelicHook.DAMAGE_DEALT)
    assert "lizard_tail" in supported_combat_relic_ids(CombatRelicHook.DAMAGE_TAKEN)
    assert "red_skull" in supported_combat_relic_ids(CombatRelicHook.DAMAGE_TAKEN)
    assert "ice_cream" in supported_combat_relic_ids(CombatRelicHook.TURN_END)
    assert "sturdy_clamp" in supported_combat_relic_ids(CombatRelicHook.TURN_END)
    assert "book_repair_knife" in supported_combat_relic_ids(
        CombatRelicHook.MONSTER_KILLED
    )
    assert "sword_of_stone" in supported_combat_relic_ids(CombatRelicHook.MONSTER_KILLED)
    assert "war_hammer" in supported_combat_relic_ids(CombatRelicHook.MONSTER_KILLED)
    assert "paels_tooth" in supported_combat_relic_ids(CombatRelicHook.COMBAT_END)
    assert "unhandled_starter" not in supported_combat_relic_ids()


def _emitted_marker_kinds() -> set[str]:
    source_path = Path(relic_combat_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    marker_kinds: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if function_name not in {"CombatRelicMarker", "CombatRelicMarkerSpec"}:
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "kind"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                marker_kinds.add(keyword.value.value)
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            marker_kinds.add(node.args[0].value)
    return marker_kinds
