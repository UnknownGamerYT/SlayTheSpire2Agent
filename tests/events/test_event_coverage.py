from __future__ import annotations

from sts2sim.content.event_coverage import (
    UNSUPPORTED_BESPOKE_EVENT_IDS,
    EventCoverageCategory,
    audit_event_coverage,
    load_cached_events,
)


def test_every_cached_event_is_categorized() -> None:
    cached_events = load_cached_events()
    report = audit_event_coverage()

    assert len(report.entries) == len(cached_events)
    assert {entry.event_id for entry in report.entries} == {
        str(event.get("id")) for event in cached_events
    }
    assert sum(report.counts_by_category.values()) == len(cached_events)
    assert all(isinstance(entry.category, EventCoverageCategory) for entry in report.entries)
    assert not report.optional_module_errors
    assert report.counts_by_category == {
        "implemented": 0,
        "primitive": 35,
        "stepwise": 8,
        "special": 14,
        "ancient-only": 9,
        "unsupported/bespoke": 0,
    }


def test_special_events_are_marked_complete_when_bespoke_handlers_exist() -> None:
    report = audit_event_coverage()

    for event_id in (
        "BATTLEWORN_DUMMY",
        "COLORFUL_PHILOSOPHERS",
        "DENSE_VEGETATION",
        "FAKE_MERCHANT",
        "INFESTED_AUTOMATON",
        "POTION_COURIER",
        "PUNCH_OFF",
        "RANWID_THE_ELDER",
        "RELIC_TRADER",
        "ROUND_TEA_PARTY",
        "SELF_HELP_BOOK",
        "THE_LANTERN_KEY",
        "THE_FUTURE_OF_POTIONS",
        "WAR_HISTORIAN_REPY",
    ):
        entry = report.entry_for(event_id)
        assert entry.category is EventCoverageCategory.SPECIAL
        assert entry.missing_option_ids == ()
        assert "sts2sim.mechanics.event_specials" in entry.source_modules


def test_ancient_records_are_marked_ancient_only() -> None:
    report = audit_event_coverage()
    ancient_event_ids = {
        str(event.get("id"))
        for event in load_cached_events()
        if event.get("type") == "Ancient"
    }

    assert ancient_event_ids
    for event_id in ancient_event_ids:
        entry = report.entry_for(event_id)
        assert entry.category is EventCoverageCategory.ANCIENT_ONLY
        assert entry.missing_option_ids == ()


def test_optional_event_flows_are_marked_stepwise_when_present() -> None:
    report = audit_event_coverage()

    for event_id in (
        "ABYSSAL_BATHS",
        "COLOSSAL_FLOWER",
        "ENDLESS_CONVEYOR",
        "SLIPPERY_BRIDGE",
        "TABLET_OF_TRUTH",
        "TINKER_TIME",
        "TRIAL",
        "WELCOME_TO_WONGOS",
    ):
        entry = report.entry_for(event_id)
        assert entry.category is EventCoverageCategory.STEPWISE
        assert entry.missing_option_ids == ()
        assert "sts2sim.mechanics.event_flows" in entry.source_modules


def test_no_cached_events_remain_unsupported() -> None:
    report = audit_event_coverage()

    assert report.unsupported_events == ()
    assert frozenset() == UNSUPPORTED_BESPOKE_EVENT_IDS


def test_every_cached_event_option_has_source_backed_coverage() -> None:
    report = audit_event_coverage()

    missing = {
        entry.event_id: entry.missing_option_ids
        for entry in report.entries
        if entry.missing_option_ids
    }

    assert missing == {}


def test_unsupported_catalog_options_are_covered_not_missing() -> None:
    report = audit_event_coverage()
    entry = report.entry_for("BRAIN_LEECH")

    assert "SHARE_KNOWLEDGE" in entry.cached_option_ids
    assert "share_knowledge" in entry.implemented_option_ids
    assert entry.missing_option_ids == ()
    assert entry.category is EventCoverageCategory.PRIMITIVE


def test_missing_optional_modules_do_not_block_audit() -> None:
    report = audit_event_coverage(
        optional_module_names=("sts2sim.mechanics.definitely_missing_event_catalog",)
    )

    assert report.total_events == len(load_cached_events())
    assert report.optional_module_errors == ()
