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


def test_wriggler_named_slot_conditions_are_deterministic_hints() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "WRIGGLER",
            "name": "Wriggler",
            "type": "Normal",
            "min_hp": 20,
            "max_hp": 20,
            "moves": (
                {"id": "BITE", "name": "Bite", "intent": "Attack"},
                {"id": "WRIGGLE", "name": "Wriggle", "intent": "Buff"},
            ),
            "attack_pattern": {
                "initial_move": "INIT",
                "states": (
                    {
                        "id": "INIT",
                        "type": "conditional",
                        "branches": (
                            {
                                "condition": 'base.Creature.SlotName == "wriggler1"',
                                "move_id": "BITE",
                            },
                            {
                                "condition": 'base.Creature.SlotName == "wriggler2"',
                                "move_id": "WRIGGLE",
                            },
                        ),
                    },
                ),
            },
        }
    )

    assert not classification.blocked
    assert "slot_condition" in _codes(classification.hints)


def test_empty_random_selector_uses_move_pool_fallback_hint() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "EMPTY_RANDOM",
            "name": "Empty Random",
            "type": "Normal",
            "min_hp": 20,
            "max_hp": 20,
            "moves": (
                {"id": "HIT", "name": "Hit", "intent": "Attack", "damage": {"normal": 5}},
                {"id": "GROWL", "name": "Growl", "intent": "Debuff"},
            ),
            "attack_pattern": {
                "initial_move": "RAND",
                "states": ({"id": "RAND", "type": "random", "branches": ()},),
            },
        }
    )

    assert not classification.blocked
    assert "empty_random_selector" in _codes(classification.hints)


def test_empty_random_selector_without_moves_stays_blocked() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "EMPTY_RANDOM_NO_MOVES",
            "name": "Empty Random No Moves",
            "type": "Normal",
            "min_hp": 20,
            "max_hp": 20,
            "moves": (),
            "attack_pattern": {
                "initial_move": "RAND",
                "states": ({"id": "RAND", "type": "random", "branches": ()},),
            },
        }
    )

    assert classification.blocker_codes == ("empty_random_selector",)


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
        "phase_or_death_script_move",
        "spawn_capacity_condition",
        "summon_move_requires_spawn_resolution",
    } <= _codes(classification.blockers)
    assert "hp_threshold_condition" in _codes(classification.hints)


def test_supported_summon_moves_and_spawn_capacity_are_hints() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "OVICOPTER",
            "name": "Ovicopter",
            "type": "Normal",
            "min_hp": 120,
            "max_hp": 120,
            "moves": (
                {"id": "LAY_EGGS", "name": "Lay Eggs", "intent": "Summon"},
                {"id": "SMASH", "name": "Smash", "intent": "Attack"},
                {"id": "NUTRITIONAL_PASTE", "name": "Nutritional Paste", "intent": "Buff"},
            ),
            "attack_pattern": {
                "initial_move": "SUMMON_BRANCH",
                "states": (
                    {
                        "id": "SUMMON_BRANCH",
                        "type": "conditional",
                        "branches": (
                            {"condition": "CanLay", "move_id": "LAY_EGGS"},
                            {"condition": "!CanLay", "move_id": "NUTRITIONAL_PASTE"},
                        ),
                    },
                ),
            },
        }
    )

    assert not classification.blocked
    assert {
        "spawn_capacity_condition",
        "summon_move_requires_spawn_resolution",
    } <= _codes(classification.hints)


def test_two_tailed_rat_backup_summon_is_supported_hint() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "TWO_TAILED_RAT",
            "name": "Two-Tailed Rat",
            "type": "Normal",
            "min_hp": 17,
            "max_hp": 21,
            "moves": (
                {"id": "SCRATCH", "name": "Scratch", "intent": "Attack"},
                {"id": "CALL_FOR_BACKUP", "name": "Call for Backup", "intent": "Summon"},
            ),
            "attack_pattern": {
                "initial_move": "CALL_FOR_BACKUP_MOVE",
                "states": (
                    {
                        "id": "CALL_FOR_BACKUP_MOVE",
                        "type": "move",
                        "move_id": "CALL_FOR_BACKUP",
                    },
                ),
            },
        }
    )

    assert not classification.blocked
    assert "summon_move_requires_spawn_resolution" in _codes(classification.hints)


def test_tough_egg_hatch_is_supported_self_transition_hint() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "TOUGH_EGG",
            "name": "Tough Egg",
            "type": "Normal",
            "min_hp": 14,
            "max_hp": 18,
            "moves": (
                {"id": "HATCH", "name": "HATCH", "intent": "Summon"},
                {"id": "NIBBLE", "name": "Nibble", "intent": "Attack"},
            ),
            "attack_pattern": {
                "initial_move": "HATCH_MOVE",
                "states": (
                    {"id": "HATCH_MOVE", "type": "move", "move_id": "HATCH"},
                    {"id": "NIBBLE_MOVE", "type": "move", "move_id": "NIBBLE"},
                ),
            },
        }
    )

    assert not classification.blocked
    assert "self_hatch_move" in _codes(classification.hints)


def test_gas_bomb_explode_is_supported_self_destruct_hint() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "GAS_BOMB",
            "name": "Gas Bomb",
            "type": "Normal",
            "min_hp": 7,
            "max_hp": 7,
            "moves": (
                {
                    "id": "EXPLODE",
                    "name": "Explode",
                    "intent": "Special",
                    "damage": {"normal": 8},
                },
            ),
            "attack_pattern": {
                "initial_move": "EXPLODE_MOVE",
                "states": (
                    {"id": "EXPLODE_MOVE", "type": "move", "move_id": "EXPLODE"},
                ),
            },
        }
    )

    assert not classification.blocked
    assert "self_destruct_move" in _codes(classification.hints)


def test_supported_special_innate_powers_are_hints() -> None:
    for power_id in (
        "PLATING",
        "CURL_UP",
        "SKITTISH",
        "RAVENOUS",
        "SHRIEK",
        "ENRAGE",
        "SLIPPERY",
    ):
        classification = classify_raw_monster_specials(
            {
                "id": f"{power_id}_MONSTER",
                "name": f"{power_id} Monster",
                "type": "Normal",
                "min_hp": 20,
                "max_hp": 20,
                "innate_powers": ({"power_id": power_id, "amount": 4},),
                "moves": (
                    {"id": "WAIT", "name": "Wait", "intent": "Unknown"},
                ),
                "attack_pattern": {
                    "initial_move": "WAIT_MOVE",
                    "states": (
                        {"id": "WAIT_MOVE", "type": "move", "move_id": "WAIT"},
                    ),
                },
            }
        )

        assert not classification.blocked
        assert "special_innate_power_supported" in _codes(classification.hints)


def test_escape_moves_are_supported_explicit_removal_hints() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "RUNNER",
            "name": "Runner",
            "type": "Normal",
            "min_hp": 20,
            "max_hp": 20,
            "moves": ({"id": "FLEE", "name": "Flee", "intent": "Escape"},),
            "attack_pattern": {
                "initial_move": "FLEE_MOVE",
                "states": ({"id": "FLEE_MOVE", "type": "move", "move_id": "FLEE"},),
            },
        }
    )

    assert not classification.blocked
    assert "escape_move_requires_combat_removal" in _codes(classification.hints)


def test_supported_formation_conditions_are_hints() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "FORMATION_BRANCHER",
            "name": "Formation Brancher",
            "type": "Normal",
            "min_hp": 20,
            "max_hp": 20,
            "moves": (
                {"id": "ALONE", "name": "Alone", "intent": "Attack"},
                {"id": "FRONT", "name": "Front", "intent": "Attack"},
                {"id": "BACK", "name": "Back", "intent": "Buff"},
            ),
            "attack_pattern": {
                "initial_move": "INIT",
                "states": (
                    {
                        "id": "INIT",
                        "type": "conditional",
                        "branches": (
                            {"condition": "base.Creature.GetAllyCount() == 0", "move_id": "ALONE"},
                            {
                                "condition": "((Nibbit)base.Creature.Monster).IsFront",
                                "move_id": "FRONT",
                            },
                            {
                                "condition": "!((Nibbit)base.Creature.Monster).IsFront",
                                "move_id": "BACK",
                            },
                        ),
                    },
                ),
            },
        }
    )

    assert not classification.blocked
    assert "formation_condition" in _codes(classification.hints)


def test_supported_hp_threshold_conditions_are_hints() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "FROG_KNIGHT",
            "name": "Frog Knight",
            "type": "Normal",
            "min_hp": 70,
            "max_hp": 70,
            "moves": (
                {"id": "TONGUE_LASH", "name": "Tongue Lash", "intent": "Attack"},
                {"id": "BEETLE_CHARGE", "name": "Beetle Charge", "intent": "Attack"},
            ),
            "attack_pattern": {
                "initial_move": "HALF_HEALTH",
                "states": (
                    {
                        "id": "HALF_HEALTH",
                        "type": "conditional",
                        "branches": (
                            {
                                "condition": "HasBeetleCharged || "
                                "base.Creature.CurrentHp >= base.Creature.MaxHp / 2",
                                "move_id": "TONGUE_LASH",
                            },
                            {
                                "condition": "!HasBeetleCharged && "
                                "base.Creature.CurrentHp < base.Creature.MaxHp / 2",
                                "move_id": "BEETLE_CHARGE",
                            },
                        ),
                    },
                ),
            },
        }
    )

    assert not classification.blocked
    assert "hp_threshold_condition" in _codes(classification.hints)


def test_knowledge_demon_script_counter_condition_is_supported_hint() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "KNOWLEDGE_DEMON",
            "name": "Knowledge Demon",
            "type": "Boss",
            "min_hp": 200,
            "max_hp": 200,
            "moves": (
                {"id": "CURSE_OF_KNOWLEDGE", "name": "Curse", "intent": "Debuff"},
                {"id": "SLAP", "name": "Slap", "intent": "Attack"},
            ),
            "attack_pattern": {
                "initial_move": "CURSE_OF_KNOWLEDGE_MOVE",
                "states": (
                    {
                        "id": "CURSE_OF_KNOWLEDGE_MOVE",
                        "type": "move",
                        "move_id": "CURSE_OF_KNOWLEDGE",
                        "next": "CURSE_BRANCH",
                    },
                    {
                        "id": "CURSE_BRANCH",
                        "type": "conditional",
                        "branches": (
                            {
                                "condition": "_curseOfKnowledgeCounter < 3",
                                "move_id": "CURSE_OF_KNOWLEDGE",
                            },
                            {
                                "condition": "_curseOfKnowledgeCounter >= 3",
                                "move_id": "SLAP",
                            },
                        ),
                    },
                ),
            },
        }
    )

    assert not classification.blocked
    assert "boss_explicit_integration_supported" in _codes(classification.hints)
    assert "script_counter_condition" in _codes(classification.hints)


def test_unreachable_empty_conditional_selector_is_hint() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "UNUSED_CONDITIONAL",
            "name": "Unused Conditional",
            "type": "Normal",
            "min_hp": 20,
            "max_hp": 20,
            "moves": ({"id": "HIT", "name": "Hit", "intent": "Attack"},),
            "attack_pattern": {
                "initial_move": "HIT",
                "states": (
                    {"id": "HIT_MOVE", "type": "move", "move_id": "HIT"},
                    {"id": "UNUSED", "type": "conditional", "branches": ()},
                ),
            },
        }
    )

    assert not classification.blocked
    assert "empty_conditional_selector" in _codes(classification.hints)


def test_referenced_empty_conditional_selector_stays_blocked() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "USED_CONDITIONAL",
            "name": "Used Conditional",
            "type": "Normal",
            "min_hp": 20,
            "max_hp": 20,
            "moves": ({"id": "HIT", "name": "Hit", "intent": "Attack"},),
            "attack_pattern": {
                "initial_move": "BRANCH",
                "states": ({"id": "BRANCH", "type": "conditional", "branches": ()},),
            },
        }
    )

    assert classification.blocker_codes == ("empty_conditional_selector",)


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
    } <= _codes(classification.blockers)
    assert "special_innate_power_supported" in _codes(classification.hints)


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


def test_verified_elite_source_integration_is_a_hint() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "BYRDONIS",
            "name": "Byrdonis",
            "type": "Elite",
            "min_hp": 81,
            "max_hp": 84,
            "moves": (
                {"id": "PECK", "name": "Peck", "intent": "Attack", "damage": {"normal": 3}},
                {"id": "SWOOP", "name": "Swoop", "intent": "Attack", "damage": {"normal": 17}},
            ),
            "attack_pattern": {
                "initial_move": "SWOOP",
                "states": (
                    {"id": "PECK_MOVE", "type": "move", "move_id": "PECK", "next": "SWOOP_MOVE"},
                    {"id": "SWOOP_MOVE", "type": "move", "move_id": "SWOOP", "next": "PECK_MOVE"},
                ),
            },
        }
    )

    assert not classification.blocked
    assert "elite_explicit_integration_supported" in _codes(classification.hints)


def test_verified_boss_source_integration_is_a_hint() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "CRUSHER",
            "name": "Crusher",
            "type": "Boss",
            "min_hp": 199,
            "max_hp": 199,
            "moves": (
                {"id": "THRASH", "name": "Thrash", "intent": "Attack", "damage": {"normal": 12}},
                {
                    "id": "ADAPT",
                    "name": "Adapt",
                    "intent": "Buff",
                    "powers": ({"power_id": "STRENGTH", "amount": 2, "target": "self"},),
                },
            ),
            "attack_pattern": {
                "initial_move": "THRASH",
                "states": (
                    {"id": "THRASH_MOVE", "type": "move", "move_id": "THRASH", "next": "ADAPT"},
                    {"id": "ADAPT", "type": "move", "move_id": "ADAPT", "next": "THRASH_MOVE"},
                ),
            },
        }
    )

    assert not classification.blocked
    assert "boss_explicit_integration_supported" in _codes(classification.hints)


def test_doormaker_is_deprecated_as_an_unverified_boss_source() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "DOORMAKER",
            "name": "Doormaker",
            "type": "Boss",
            "min_hp": 489,
            "max_hp": 489,
            "moves": (
                {"id": "DRAMATIC_OPEN", "name": "Dramatic Open", "intent": "Summon"},
                {"id": "HUNGER", "name": "Hunger", "intent": "Attack"},
            ),
            "attack_pattern": {
                "initial_move": "DRAMATIC_OPEN",
                "states": (
                    {
                        "id": "DRAMATIC_OPEN_MOVE",
                        "type": "move",
                        "move_id": "DRAMATIC_OPEN",
                        "next": "HUNGER_MOVE",
                    },
                    {"id": "HUNGER_MOVE", "type": "move", "move_id": "HUNGER"},
                ),
            },
        }
    )

    assert classification.blocked
    assert {
        "boss_script_requires_explicit_integration",
        "summon_move_requires_spawn_resolution",
    } <= _codes(classification.blockers)


def test_v0107_aeonglass_and_infested_prism_surface_explicit_blockers() -> None:
    aeonglass = classify_raw_monster_specials(
        {
            "id": "AEONGLASS",
            "name": "Aeonglass",
            "type": "Boss",
            "min_hp": 512,
            "min_hp_ascension": 535,
            "moves": (
                {
                    "id": "EBB",
                    "name": "Ebb",
                    "intent": "Attack + Defend",
                    "damage": {"normal": 26, "ascension": 32},
                    "block": 33,
                },
                {
                    "id": "EYE_LASERS",
                    "name": "Eye Lasers",
                    "intent": "Attack",
                    "damage": {"normal": 11, "ascension": 12},
                },
                {
                    "id": "INCREASING_INTENSITY",
                    "name": "Increasing Intensity",
                    "intent": "Status + Buff",
                },
            ),
        }
    )
    prism = classify_raw_monster_specials(
        {
            "id": "INFESTED_PRISM",
            "name": "Infested Prism",
            "type": "Elite",
            "min_hp": 161,
            "moves": (
                {
                    "id": "PULSATE",
                    "name": "Pulsate",
                    "intent": "Attack + Buff + Defend",
                    "powers": ({"power_id": "VITAL_SPARK", "amount": 2, "target": "self"},),
                },
            ),
        }
    )

    assert {
        "aeonglass_increasing_intensity_requires_wither_hook",
        "boss_script_requires_explicit_integration",
        "special_innate_power_requires_hook",
    } <= _codes(aeonglass.blockers)
    assert "vital_spark_requires_tainted_skill_hook" in _codes(prism.blockers)


def test_verified_status_boss_integrations_are_hints() -> None:
    for monster_id in (
        "CEREMONIAL_BEAST",
        "LAGAVULIN_MATRIARCH",
        "SOUL_FYSH",
        "THE_INSATIABLE",
        "VANTOM",
    ):
        classification = classify_raw_monster_specials(
            {
                "id": monster_id,
                "name": monster_id.title(),
                "type": "Boss",
                "min_hp": 120,
                "max_hp": 120,
                "moves": (
                    {
                        "id": "HIT",
                        "name": "Hit",
                        "intent": "Attack",
                        "damage": {"normal": 7},
                    },
                ),
                "attack_pattern": {
                    "initial_move": "HIT_MOVE",
                    "states": (
                        {"id": "HIT_MOVE", "type": "move", "move_id": "HIT"},
                    ),
                },
            }
        )

        assert not classification.blocked
        assert "boss_explicit_integration_supported" in _codes(classification.hints)


def test_use_only_once_branch_is_a_supported_hint() -> None:
    classification = classify_raw_monster_specials(
        {
            "id": "ONE_SHOT_BRANCHER",
            "name": "One Shot Brancher",
            "type": "Normal",
            "min_hp": 20,
            "max_hp": 20,
            "moves": (
                {"id": "ROAR", "name": "Roar", "intent": "Debuff"},
                {"id": "CLAW", "name": "Claw", "intent": "Attack", "damage": {"normal": 5}},
            ),
            "attack_pattern": {
                "initial_move": "RAND",
                "states": (
                    {
                        "id": "RAND",
                        "type": "random",
                        "branches": (
                            {"move_id": "ROAR", "repeat": "UseOnlyOnce"},
                            {"move_id": "CLAW"},
                        ),
                    },
                ),
            },
        }
    )

    assert not classification.blocked
    assert "use_only_once_repeat" in _codes(classification.hints)


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
    assert "DOORMAKER" not in summary.classified_monster_ids
    assert "AEONGLASS" in summary.blocked_monster_ids
    assert "INFESTED_PRISM" in summary.blocked_monster_ids
    assert "SKULKING_COLONY" in summary.blocked_monster_ids
    assert "QUEEN" not in summary.blocked_monster_ids
    assert "TEST_SUBJECT" not in summary.blocked_monster_ids
    assert "BYRDONIS" not in summary.blocked_monster_ids
    assert "CEREMONIAL_BEAST" not in summary.blocked_monster_ids
    assert "MAGI_KNIGHT" not in summary.blocked_monster_ids
    assert "LAGAVULIN_MATRIARCH" not in summary.blocked_monster_ids
    assert "SOUL_FYSH" not in summary.blocked_monster_ids
    assert "THE_INSATIABLE" not in summary.blocked_monster_ids
    assert "VANTOM" not in summary.blocked_monster_ids
    assert "WATERFALL_GIANT" not in summary.blocked_monster_ids
    assert "DECIMILLIPEDE_SEGMENT" not in summary.blocked_monster_ids
    assert "DECIMILLIPEDE_SEGMENT_BACK" not in summary.blocked_monster_ids
    assert "DECIMILLIPEDE_SEGMENT_FRONT" not in summary.blocked_monster_ids
    assert "DECIMILLIPEDE_SEGMENT_MIDDLE" not in summary.blocked_monster_ids
    assert dict(summary.requirement_counts)[SUMMON_SPAWN_MOVE] >= 1
    assert dict(summary.blocker_counts)[BOSS_SCRIPT] >= 1
    assert ELITE_SPECIAL_MECHANIC not in dict(summary.blocker_counts)
