from __future__ import annotations

import json
from random import Random

from helpers import project_root

from sts2sim.mechanics import (
    build_encounter_definitions,
    build_monster_definitions,
    choose_encounter,
    initial_monster_move,
    monster_hp_range,
    monster_move_damage,
    monster_power_amount,
    next_monster_move,
    spawn_monsters,
)
from sts2sim.mechanics.monsters import waterfall_giant_siphon_heal


def _monster_source():
    return (
        {
            "id": "TRAINING_DUMMY",
            "name": "Training Dummy",
            "type": "Normal",
            "min_hp": 10,
            "max_hp": 12,
            "min_hp_ascension": 20,
            "max_hp_ascension": 22,
            "moves": (
                {
                    "id": "JAB",
                    "name": "Jab",
                    "intent": "Attack",
                    "damage": {"normal": 5, "ascension": 7, "hit_count": 2},
                    "block": None,
                    "heal": None,
                    "powers": None,
                },
                {
                    "id": "GUARD",
                    "name": "Guard",
                    "intent": "Defend",
                    "damage": None,
                    "block": 4,
                    "heal": None,
                    "powers": None,
                },
            ),
            "attack_pattern": {
                "initial_move": "JAB",
                "states": (
                    {
                        "id": "JAB_MOVE",
                        "move_id": "JAB",
                        "next": "GUARD_MOVE",
                        "type": "move",
                    },
                    {
                        "id": "GUARD_MOVE",
                        "move_id": "GUARD",
                        "next": "JAB_MOVE",
                        "type": "move",
                    },
                ),
                "type": "cycle",
            },
        },
    )


def test_monster_source_parses_scaling_and_cycle_moves() -> None:
    definitions = build_monster_definitions(_monster_source())
    dummy = definitions["TRAINING_DUMMY"]

    assert monster_hp_range(dummy, ascension_level=0) == (10, 12)
    assert monster_hp_range(dummy, ascension_level=7) == (20, 22)

    move = initial_monster_move(dummy, Random(1))
    assert move is not None
    assert move.move_id == "JAB"
    assert move.hit_count == 2
    assert monster_move_damage(dummy, move, ascension_level=0) == 5
    assert monster_move_damage(dummy, move, ascension_level=2) == 7

    next_move = next_monster_move(dummy, move.move_id, Random(1))
    assert next_move is not None
    assert next_move.move_id == "GUARD"


def test_choose_weak_encounter_prefers_single_early_enemy_when_available() -> None:
    encounters = build_encounter_definitions(
        (
            {
                "id": "WEAK_PAIR",
                "name": "Weak Pair",
                "act": "Act 1 - Overgrowth",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "A"}, {"id": "B"}),
            },
            {
                "id": "WEAK_SINGLE",
                "name": "Weak Single",
                "act": "Act 1 - Overgrowth",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "TRAINING_DUMMY"},),
            },
            {
                "id": "NORMAL_SINGLE",
                "name": "Normal Single",
                "act": "Act 1 - Overgrowth",
                "room_type": "Monster",
                "is_weak": False,
                "monsters": ({"id": "TRAINING_DUMMY"},),
            },
        )
    )

    encounter = choose_encounter(
        encounters,
        Random(3),
        act=1,
        room_type="monster",
        prefer_weak=True,
    )

    assert encounter is not None
    assert encounter.encounter_id == "WEAK_SINGLE"


def test_spawn_monsters_suffixes_duplicate_instance_ids() -> None:
    definitions = build_monster_definitions(_monster_source())
    encounters = build_encounter_definitions(
        (
            {
                "id": "DUMMY_PAIR",
                "name": "Dummy Pair",
                "act": "Act 1 - Overgrowth",
                "room_type": "Monster",
                "monsters": ({"id": "TRAINING_DUMMY"}, {"id": "TRAINING_DUMMY"}),
            },
        )
    )

    spawned = spawn_monsters(
        encounters[0],
        definitions,
        Random(1),
        ascension_level=0,
    )

    assert [monster.instance_id for monster in spawned] == [
        "TRAINING_DUMMY#1",
        "TRAINING_DUMMY#2",
    ]


def test_v0107_cache_enemy_values_and_ids_parse() -> None:
    cache_dir = project_root() / "data" / "cache" / "eng"
    definitions = build_monster_definitions(
        json.loads((cache_dir / "monsters.json").read_text(encoding="utf-8"))
    )

    assert "AEONGLASS" in definitions
    assert "DOORMAKER" not in definitions

    soul_scream = definitions["SOUL_FYSH"].move_by_id["SCREAM"]
    assert monster_move_damage(definitions["SOUL_FYSH"], soul_scream, ascension_level=0) == 13
    assert monster_move_damage(definitions["SOUL_FYSH"], soul_scream, ascension_level=4) == 15

    assassin = definitions["ASSASSIN_RUBY_RAIDER"]
    killshot = assassin.move_by_id["KILLSHOT"]
    assert monster_move_damage(assassin, killshot, ascension_level=0) == 10
    assert monster_move_damage(assassin, killshot, ascension_level=2) == 11

    axebot = definitions["AXEBOT"]
    assert initial_monster_move(axebot, Random(1)).move_id == "BOOT_UP"
    assert axebot.move_by_id["ONE_TWO"].hit_count == 2
    assert {power.power_id for power in axebot.move_by_id["HAMMER_UPPERCUT"].powers} == {
        "WEAK",
        "FRAIL",
    }

    prism = definitions["INFESTED_PRISM"]
    assert prism.move_by_id["PULSATE"].powers[0].power_id == "VITAL_SPARK"


def test_v0107_skulking_colony_patch_normalization() -> None:
    definitions = build_monster_definitions(
        (
            {
                "id": "SKULKING_COLONY",
                "name": "Skulking Colony",
                "type": "Elite",
                "min_hp": 75,
                "min_hp_ascension": 80,
                "moves": (
                    {"id": "ZOOM", "name": "Zoom", "intent": "Attack"},
                    {"id": "ZOOM_MOVE_2", "name": "Zoom Move 2", "intent": "Attack"},
                    {
                        "id": "INERTIA",
                        "name": "Inertia",
                        "intent": "Attack + Buff",
                        "powers": ({"power_id": "STRENGTH", "amount": 2, "target": "self"},),
                    },
                    {"id": "PIERCING_STABS", "name": "Piercing Stabs", "intent": "Attack"},
                ),
                "attack_pattern": {
                    "initial_move": "ZOOM",
                    "states": (
                        {
                            "id": "ZOOM_MOVE",
                            "type": "move",
                            "move_id": "ZOOM",
                            "next": "ZOOM_MOVE_2",
                        },
                        {
                            "id": "ZOOM_MOVE_2",
                            "type": "move",
                            "move_id": "ZOOM_MOVE_2",
                            "next": "INERTIA_MOVE",
                        },
                        {
                            "id": "INERTIA_MOVE",
                            "type": "move",
                            "move_id": "INERTIA",
                            "next": "PIERCING_STABS_MOVE",
                        },
                        {
                            "id": "PIERCING_STABS_MOVE",
                            "type": "move",
                            "move_id": "PIERCING_STABS",
                            "next": "ZOOM_MOVE",
                        },
                    ),
                },
            },
        )
    )
    colony = definitions["SKULKING_COLONY"]

    assert monster_hp_range(colony, ascension_level=0) == (75, 75)
    assert monster_hp_range(colony, ascension_level=8) == (80, 80)
    assert [(power.power_id, power.amount) for power in colony.innate_powers] == [
        ("HARDENED_SHELL", 20)
    ]

    strength = colony.move_by_id["INERTIA"].powers[0]
    assert monster_power_amount(colony, strength, ascension_level=2) == 2
    assert monster_power_amount(colony, strength, ascension_level=3) == 3

    first = initial_monster_move(colony, Random(1))
    second = next_monster_move(colony, first.move_id, Random(1))
    assert (first.move_id, second.move_id) == ("ZOOM", "ZOOM_MOVE_2")


def test_v0107_haunted_ship_and_punch_construct_move_chains() -> None:
    definitions = build_monster_definitions(
        (
            {
                "id": "HAUNTED_SHIP",
                "name": "Haunted Ship",
                "type": "Normal",
                "min_hp": 63,
                "moves": (
                    {"id": "SWIPE", "name": "Swipe", "intent": "Attack"},
                    {"id": "STOMP", "name": "Stomp", "intent": "Attack"},
                    {
                        "id": "HAUNT",
                        "name": "Haunt",
                        "intent": "Debuff + Status",
                        "powers": ({"power_id": "WEAK", "amount": 3, "target": "player"},),
                    },
                ),
                "attack_pattern": {
                    "initial_move": "HAUNT",
                    "states": (
                        {
                            "id": "SWIPE_MOVE",
                            "type": "move",
                            "move_id": "SWIPE",
                            "next": "STOMP_MOVE",
                        },
                        {
                            "id": "STOMP_MOVE",
                            "type": "move",
                            "move_id": "STOMP",
                            "next": "SWIPE_MOVE",
                        },
                        {"id": "HAUNT_MOVE", "type": "move", "move_id": "HAUNT"},
                    ),
                },
            },
            {
                "id": "PUNCH_CONSTRUCT",
                "name": "Punch Construct",
                "type": "Normal",
                "min_hp": 55,
                "moves": (
                    {"id": "READY", "name": "Ready", "intent": "Defend"},
                    {"id": "STRONG_PUNCH", "name": "Strong Punch", "intent": "Attack"},
                    {
                        "id": "FAST_PUNCH",
                        "name": "Fast Punch",
                        "intent": "Attack + Debuff",
                        "powers": ({"power_id": "FRAIL", "amount": 1, "target": "player"},),
                    },
                ),
                "attack_pattern": {
                    "states": (
                        {"id": "READY_MOVE", "type": "move", "move_id": "READY"},
                        {"id": "STRONG_PUNCH_MOVE", "type": "move", "move_id": "STRONG_PUNCH"},
                        {"id": "FAST_PUNCH_MOVE", "type": "move", "move_id": "FAST_PUNCH"},
                    ),
                },
            },
        )
    )

    ship = definitions["HAUNTED_SHIP"]
    haunt = initial_monster_move(ship, Random(1))
    assert haunt.move_id == "HAUNT"
    assert [(power.power_id, power.amount) for power in haunt.powers] == [("WEAK", 3)]
    assert next_monster_move(ship, "HAUNT", Random(1)).move_id == "SWIPE"
    assert next_monster_move(ship, "SWIPE", Random(1)).move_id == "STOMP"

    construct = definitions["PUNCH_CONSTRUCT"]
    ready = initial_monster_move(construct, Random(1))
    fast = next_monster_move(construct, ready.move_id, Random(1))
    strong = next_monster_move(construct, fast.move_id, Random(1))
    assert (ready.move_id, fast.move_id, strong.move_id) == (
        "READY",
        "FAST_PUNCH",
        "STRONG_PUNCH",
    )
    assert [(power.power_id, power.amount) for power in fast.powers] == [("FRAIL", 1)]


def test_v0107_waterfall_giant_siphon_heal_amount() -> None:
    assert waterfall_giant_siphon_heal(ascension_level=0) == 10
    assert waterfall_giant_siphon_heal(ascension_level=4) == 15
