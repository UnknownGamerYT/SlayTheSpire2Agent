from __future__ import annotations

from random import Random

from sts2sim.mechanics import (
    build_encounter_definitions,
    build_monster_definitions,
    choose_encounter,
    initial_monster_move,
    monster_hp_range,
    monster_move_damage,
    next_monster_move,
    spawn_monsters,
)


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
