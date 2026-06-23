from __future__ import annotations

from sts2sim.content.card_coverage import (
    CardCoverageStatus,
    audit_card_coverage_from_sources,
)


def test_card_coverage_classifies_implemented_partial_and_missing_cards() -> None:
    report = audit_card_coverage_from_sources(
        cards=(
            {
                "id": "STRIKE_TEST",
                "name": "Strike Test",
                "type": "Attack",
                "color": "ironclad",
                "rarity": "Common",
                "cost": 1,
                "damage": 6,
            },
            {
                "id": "SAFETY_TEST",
                "name": "Safety Test",
                "type": "Skill",
                "color": "silent",
                "rarity": "Uncommon",
                "cost": 1,
                "keywords_key": ("Retain",),
            },
            {
                "id": "OSTY_STRIKE",
                "name": "Osty Strike",
                "type": "Attack",
                "color": "necrobinder",
                "rarity": "Uncommon",
                "cost": 1,
                "damage": 7,
                "description": "Osty deals 6 damage to a random enemy.",
            },
            {
                "id": "SUMMON_GOLEM",
                "name": "Summon Golem",
                "type": "Skill",
                "color": "defect",
                "rarity": "Rare",
                "cost": 2,
                "effects": {"summon_golem": 1},
            },
            {
                "id": "FINISHING_BLOW",
                "name": "Finishing Blow",
                "type": "Attack",
                "color": "ironclad",
                "rarity": "Rare",
                "cost": 1,
                "damage": 10,
                "description": "Deal 10 damage. If this kills an enemy, gain [energy:1].",
            },
            {
                "id": "ASCENDERS_BANE",
                "name": "Ascender's Bane",
                "type": "Curse",
                "color": "curse",
                "rarity": "Special",
                "cost": -1,
                "keywords_key": ("Unplayable", "Ethereal"),
            },
        ),
    )

    assert report.counts_by_status == {
        "implemented": 5,
        "partial": 0,
        "missing": 1,
    }
    strike = report.entry_for("strike_test")
    safety = report.entry_for("safety_test")
    osty = report.entry_for("osty_strike")
    golem = report.entry_for("summon_golem")
    finishing_blow = report.entry_for("finishing_blow")
    bane = report.entry_for("ascenders_bane")

    assert strike.status is CardCoverageStatus.IMPLEMENTED
    assert strike.executable_keys == ("damage",)
    assert safety.status is CardCoverageStatus.IMPLEMENTED
    assert safety.executable_keys == ("keyword_retain",)

    assert osty.status is CardCoverageStatus.IMPLEMENTED
    assert "osty_action" in osty.executable_keys
    assert osty.blocker_kinds == ()

    assert golem.status is CardCoverageStatus.MISSING
    assert golem.unknown_keys == ("summon_golem",)
    assert finishing_blow.status is CardCoverageStatus.IMPLEMENTED
    assert "damage" in finishing_blow.executable_keys
    assert "if_kill_resource" in finishing_blow.executable_keys
    assert finishing_blow.blocker_kinds == ()
    assert bane.status is CardCoverageStatus.IMPLEMENTED
    assert "keyword_ethereal" in bane.executable_keys
    assert "keyword_unplayable" in bane.executable_keys
    assert bane.unknown_keys == ()


def test_card_coverage_report_filters_and_summarizes_entries() -> None:
    report = audit_card_coverage_from_sources(
        cards=(
            {"id": "DAMAGE", "name": "Damage", "type": "Attack", "color": "ironclad", "damage": 4},
            {
                "id": "SLY_CARD",
                "name": "Sly Card",
                "type": "Skill",
                "color": "silent",
                "keywords_key": ("Sly",),
            },
            {
                "id": "MYSTERY_CARD",
                "name": "Mystery Card",
                "type": "Skill",
                "color": "silent",
                "effects": {"mystery_effect": 1},
            },
        )
    )

    assert report.total_cards == 3
    assert report.implemented_ratio == 2 / 3
    assert report.executable_ratio == 2 / 3
    assert report.entry_for("sly_card").status is CardCoverageStatus.IMPLEMENTED
    assert report.entries_for(status="missing", color="silent")[0].content_id == "MYSTERY_CARD"
    assert report.sample_missing_ids == ("MYSTERY_CARD",)
    assert report.as_dict()["counts_by_color"]["silent"]["missing"] == 1
