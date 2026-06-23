from __future__ import annotations

from sts2sim.content.combat_coverage import (
    CombatCoverageCategory,
    CombatCoverageStatus,
    audit_combat_coverage_from_sources,
    combat_implementation_catalog,
    default_combat_implementation_catalog,
)


def test_combat_coverage_counts_registry_blockers_and_unknown_samples() -> None:
    catalog = combat_implementation_catalog(
        implemented_ids_by_category={
            "relics": {"akabeko"},
            "potions": {"fire_potion"},
        },
        blocked_ids_by_category={
            "cards": {"BLOCKED_CARD": "Needs bespoke card target selection."},
            "relics": {"SCRIPTED_RELIC": "Needs an end-of-combat relic hook."},
            "potions": {"ASHWATER": "Needs hand-selection exhaust UI."},
            "monsters": {"SCRIPTED_BOSS": "Needs boss phase scripting."},
            "encounters": {"BLOCKED_ENCOUNTER": "Blocked until scripted boss exists."},
        },
        executable_card_effect_keys={"damage", "block", "apply_status", "sequence"},
    )

    report = audit_combat_coverage_from_sources(
        cards=(
            {"id": "STRIKE_TEST", "name": "Strike Test", "type": "Attack", "damage": 6},
            {"id": "BLOCKED_CARD", "name": "Blocked Card", "type": "Skill", "block": 5},
            {
                "id": "MYSTERY_CARD",
                "name": "Mystery Card",
                "type": "Skill",
                "effects": {"summon_golem": 1},
            },
        ),
        relics=(
            {"id": "AKABEKO", "name": "Akabeko"},
            {"id": "SCRIPTED_RELIC", "name": "Scripted Relic"},
            {"id": "UNKNOWN_RELIC", "name": "Unknown Relic"},
        ),
        potions=(
            {"id": "FIRE_POTION", "name": "Fire Potion"},
            {"id": "ASHWATER", "name": "Ashwater"},
            {"id": "WILD_POTION", "name": "Wild Potion"},
        ),
        monsters=(
            _monster("TRAINING_AUTOMATON", move_id="STRIKE"),
            _monster("SCRIPTED_BOSS", move_id="PHASE_SHIFT"),
            {"id": "EMPTY_MONSTER", "name": "Empty Monster", "moves": ()},
        ),
        encounters=(
            _encounter("TRAINING_ENCOUNTER", "TRAINING_AUTOMATON"),
            _encounter("BLOCKED_ENCOUNTER", "SCRIPTED_BOSS"),
            _encounter("MISSING_MONSTER_ENCOUNTER", "DOES_NOT_EXIST"),
        ),
        implementation_catalog=catalog,
        unknown_sample_size=1,
    )

    assert report.counts_by_category == {
        "cards": {"total": 3, "implemented": 1, "blocked": 1, "unknown": 1},
        "relics": {"total": 3, "implemented": 1, "blocked": 1, "unknown": 1},
        "potions": {"total": 3, "implemented": 1, "blocked": 1, "unknown": 1},
        "monsters": {"total": 3, "implemented": 1, "blocked": 1, "unknown": 1},
        "encounters": {"total": 3, "implemented": 1, "blocked": 1, "unknown": 1},
    }
    assert report.sample_unknown_ids == {
        "cards": ["MYSTERY_CARD"],
        "relics": ["UNKNOWN_RELIC"],
        "potions": ["WILD_POTION"],
        "monsters": ["EMPTY_MONSTER"],
        "encounters": ["MISSING_MONSTER_ENCOUNTER"],
    }
    assert report.sample_blocked_ids == {
        "cards": ["BLOCKED_CARD"],
        "relics": ["SCRIPTED_RELIC"],
        "potions": ["ASHWATER"],
        "monsters": ["SCRIPTED_BOSS"],
        "encounters": ["BLOCKED_ENCOUNTER"],
    }


def test_combat_coverage_entries_expose_reasons_and_keys() -> None:
    catalog = combat_implementation_catalog(
        blocked_ids_by_category={"cards": {"BLOCKED_CARD": ("Needs X-cost parity.",)}},
        executable_card_effect_keys={"damage"},
    )

    report = audit_combat_coverage_from_sources(
        cards=(
            {"id": "STRIKE_TEST", "name": "Strike Test", "type": "Attack", "damage": 6},
            {"id": "BLOCKED_CARD", "name": "Blocked Card", "type": "Attack", "damage": 4},
            {
                "id": "MYSTERY_CARD",
                "name": "Mystery Card",
                "type": "Skill",
                "effects": {"summon_golem": 1},
            },
        ),
        implementation_catalog=catalog,
    )

    strike = report.entry_for(CombatCoverageCategory.CARDS, "strike_test")
    blocked = report.entry_for("cards", "BLOCKED_CARD")
    mystery = report.entry_for("cards", "MYSTERY_CARD")

    assert strike.status is CombatCoverageStatus.IMPLEMENTED
    assert strike.implemented_keys == ("damage",)
    assert blocked.status is CombatCoverageStatus.BLOCKED
    assert blocked.blocked_keys == ("blocked_card",)
    assert blocked.reasons == ("Needs X-cost parity.",)
    assert mystery.status is CombatCoverageStatus.UNKNOWN
    assert mystery.unknown_keys == ("summon_golem",)
    assert report.as_dict()["counts_by_category"]["cards"]["unknown"] == 1
    assert report.as_dict()["sample_blocked_ids"]["cards"] == ["BLOCKED_CARD"]


def test_combat_coverage_blocks_monsters_with_unintegrated_special_requirements() -> None:
    report = audit_combat_coverage_from_sources(
        cards=(),
        monsters=(
            {
                "id": "SCRIPTED_BOSS",
                "name": "Scripted Boss",
                "type": "Boss",
                "min_hp": 80,
                "max_hp": 80,
                "moves": (
                    {
                        "id": "RESPAWN",
                        "name": "Respawn",
                        "intent": "Special",
                    },
                ),
            },
            {
                "id": "BASIC_SLIME",
                "name": "Basic Slime",
                "type": "Normal",
                "min_hp": 10,
                "max_hp": 10,
                "moves": (
                    {
                        "id": "TACKLE",
                        "name": "Tackle",
                        "intent": "Attack",
                        "damage": {"normal": 3},
                    },
                ),
            },
        ),
        encounters=(
            _encounter("BOSS_FIGHT", "SCRIPTED_BOSS"),
            _encounter("SAFE_FIGHT", "BASIC_SLIME"),
        ),
        implementation_catalog=combat_implementation_catalog(
            executable_card_effect_keys={"damage"}
        ),
    )

    boss = report.entry_for("monsters", "SCRIPTED_BOSS")
    safe = report.entry_for("monsters", "BASIC_SLIME")
    boss_fight = report.entry_for("encounters", "BOSS_FIGHT")

    assert boss.status is CombatCoverageStatus.BLOCKED
    assert "boss_script_requires_explicit_integration" in boss.blocked_keys
    assert "special_intent_requires_handler" in boss.blocked_keys
    assert safe.status is CombatCoverageStatus.IMPLEMENTED
    assert boss_fight.status is CombatCoverageStatus.BLOCKED
    assert boss_fight.blocked_keys == ("scripted_boss",)


def test_default_combat_catalog_reflects_current_mechanics_registries() -> None:
    catalog = default_combat_implementation_catalog()

    assert "damage" in catalog.executable_card_effect_keys
    assert "fire_potion" in catalog.implemented_ids("potions")
    assert "attack_potion" in catalog.implemented_ids("potions")
    assert "fairy_in_a_bottle" in catalog.implemented_ids("potions")
    assert "star_potion" in catalog.implemented_ids("potions")
    assert "akabeko" in catalog.implemented_ids("relics")
    assert "strike_dummy" in catalog.implemented_ids("relics")
    assert "amethyst_aubergine" in catalog.implemented_ids("relics")
    assert catalog.blocker_reasons("cards", "IGNITION") == ()
    assert catalog.blocker_reasons("cards", "MIMIC") == ()


def _monster(monster_id: str, *, move_id: str) -> dict[str, object]:
    return {
        "id": monster_id,
        "name": monster_id.title(),
        "type": "Normal",
        "min_hp": 10,
        "max_hp": 10,
        "moves": (
            {
                "id": move_id,
                "name": move_id.title(),
                "intent": "Attack",
                "damage": {"normal": 3, "ascension": 4, "hit_count": 1},
            },
        ),
        "attack_pattern": {
            "initial_move": move_id,
            "states": ({"id": f"{move_id}_STATE", "move_id": move_id, "type": "move"},),
        },
    }


def _encounter(encounter_id: str, monster_id: str) -> dict[str, object]:
    return {
        "id": encounter_id,
        "name": encounter_id.title(),
        "act": "Act 1 - Overgrowth",
        "room_type": "Monster",
        "monsters": ({"id": monster_id},),
    }
