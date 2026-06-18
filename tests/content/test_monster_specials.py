from __future__ import annotations

import json

from helpers import project_root

from sts2sim.mechanics import (
    BOSS_SCRIPT,
    ELITE_SPECIAL_MECHANIC,
    SUMMON_SPAWN_MOVE,
    build_monster_definitions,
    classify_monster_specials,
    classify_raw_monster_specials,
    monster_special_source_coverage,
)


def _codes(requirements):
    return {requirement.code for requirement in requirements}


def _slot_branch_monster():
    return {
        "id": "SLOT_BRANCHER",
        "name": "Slot Brancher",
        "type": "Normal",
        "min_hp": 10,
        "max_hp": 10,
        "moves": (
            {"id": "LEFT", "name": "Left", "intent": "Attack", "damage": {"normal": 4}},
            {"id": "RIGHT", "name": "Right", "intent": "Attack", "damage": {"normal": 5}},
        ),
        "attack_pattern": {
            "initial_move": "INIT",
            "states": (
                {
                    "id": "INIT",
                    "type": "conditional",
                    "branches": (
                        {
                            "condition": 'base.Creature.SlotName == "first"',
                            "move_id": "LEFT",
                        },
                        {
                            "condition": 'base.Creature.SlotName == "second"',
                            "move_id": "RIGHT",
                        },
                    ),
                },
            ),
        },
    }


def test_slot_condition_branches_are_deterministic_hints() -> None:
    definition = build_monster_definitions((_slot_branch_monster(),))["SLOT_BRANCHER"]

    classification = classify_monster_specials(definition)

    assert not classification.blocked
    assert _codes(classification.hints) == {"slot_condition"}
    assert all(requirement.deterministic_hint for requirement in classification.hints)


def test_runtime_conditions_summons_and_phase_moves_are_blockers() -> None:
    raw_monster = {
        "id": "EGG_PHASE",
        "name": "Egg Phase",
        "type": "Normal",
        "min_hp": 20,
        "max_hp": 20,
        "moves": (
            {"id": "LAY_EGGS", "name": "Lay Eggs", "intent": "Summon"},
            {"id": "PHASE3_LACERATE", "name": "Phase3 Lacerate", "intent": "Attack"},
            {"id": "PECK", "name": "Peck", "intent": "Attack", "damage": {"normal": 6}},
        ),
        "attack_pattern": {
            "initial_move": "BRANCH",
            "states": (
                {
                    "id": "BRANCH",
                    "type": "conditional",
                    "branches": (
                        {
                            "condition": "base.Creature.CurrentHp < base.Creature.MaxHp / 2",
                            "move_id": "PHASE3_LACERATE",
                        },
                        {"condition": "CanLay", "move_id": "LAY_EGGS"},
                    ),
                },
            ),
        },
    }
    definition = build_monster_definitions((raw_monster,))["EGG_PHASE"]

    classification = classify_monster_specials(definition)

    assert classification.blocked
    assert {
        "hp_threshold_condition",
        "phase_or_death_script_move",
        "spawn_capacity_condition",
        "summon_move_requires_spawn_resolution",
    } <= _codes(classification.blockers)


def test_raw_boss_script_surfaces_innate_phase_and_one_shot_blockers() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "SCRIPTED_BOSS",
            "name": "Scripted Boss",
            "type": "Boss",
            "min_hp": 60,
            "max_hp": 60,
            "innate_powers": ({"power_id": "ENRAGE", "amount": 2},),
            "moves": (
                {"id": "RESPAWN", "name": "Respawn", "intent": "Heal + Buff"},
                {"id": "SWIPE", "name": "Swipe", "intent": "Attack", "damage": {"normal": 7}},
            ),
            "attack_pattern": {
                "initial_move": "SWIPE_MOVE",
                "states": (
                    {"id": "SWIPE_MOVE", "type": "move", "move_id": "SWIPE"},
                    {
                        "id": "RESPAWN_MOVE",
                        "type": "move",
                        "move_id": "RESPAWN",
                        "must_perform_once": True,
                    },
                ),
            },
        }
    )

    assert classification.blocked
    assert {
        "boss_script_requires_explicit_integration",
        "must_perform_once_state",
        "phase_or_death_script_move",
        "special_innate_power_requires_hook",
    } <= _codes(classification.blockers)


def test_elite_kind_gets_explicit_blocker_even_with_basic_cycle() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "BASIC_ELITE",
            "name": "Basic Elite",
            "type": "Elite",
            "min_hp": 40,
            "max_hp": 40,
            "moves": (
                {"id": "HIT", "name": "Hit", "intent": "Attack", "damage": {"normal": 9}},
            ),
            "attack_pattern": {
                "initial_move": "HIT_MOVE",
                "states": ({"id": "HIT_MOVE", "type": "move", "move_id": "HIT"},),
            },
        }
    )

    assert classification.blocker_codes == ("elite_requires_explicit_integration",)


def test_source_coverage_summary_counts_blocked_encounters_and_missing_monsters() -> None:
    raw_monsters = (
        _slot_branch_monster(),
        {
            "id": "BASIC_ELITE",
            "name": "Basic Elite",
            "type": "Elite",
            "min_hp": 40,
            "max_hp": 40,
            "moves": (
                {"id": "HIT", "name": "Hit", "intent": "Attack", "damage": {"normal": 9}},
            ),
        },
    )
    raw_encounters = (
        {
            "id": "SAFE",
            "name": "Safe",
            "room_type": "Monster",
            "monsters": ({"id": "SLOT_BRANCHER"},),
        },
        {
            "id": "BLOCKED",
            "name": "Blocked",
            "room_type": "Elite",
            "monsters": ({"id": "BASIC_ELITE"},),
        },
        {
            "id": "MISSING",
            "name": "Missing",
            "room_type": "Monster",
            "monsters": ({"id": "UNKNOWN_MONSTER"},),
        },
    )

    summary = monster_special_source_coverage(raw_monsters, raw_encounters)

    assert summary.monster_count == 2
    assert summary.encounter_count == 3
    assert summary.missing_monster_ids == ("UNKNOWN_MONSTER",)
    assert summary.blocked_monster_ids == ("BASIC_ELITE",)
    assert summary.encounter_ids_with_blockers == ("BLOCKED",)
    assert dict(summary.monsters_by_kind) == {"Elite": 1, "Normal": 1}


def test_cached_monster_special_source_coverage_smoke() -> None:
    cache_dir = project_root() / "data" / "cache" / "eng"
    raw_monsters = json.loads((cache_dir / "monsters.json").read_text(encoding="utf-8"))
    raw_encounters = json.loads((cache_dir / "encounters.json").read_text(encoding="utf-8"))

    summary = monster_special_source_coverage(raw_monsters, raw_encounters)

    assert summary.monster_count >= 100
    assert summary.encounter_count >= 80
    assert summary.missing_monster_ids == ()
    assert "DOORMAKER" in summary.blocked_monster_ids
    assert "TEST_SUBJECT" in summary.blocked_monster_ids
    assert dict(summary.requirement_counts)[SUMMON_SPAWN_MOVE] >= 1
    assert dict(summary.blocker_counts)[BOSS_SCRIPT] >= 10
    assert dict(summary.blocker_counts)[ELITE_SPECIAL_MECHANIC] >= 10
