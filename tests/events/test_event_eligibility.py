from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    CardInstance,
    MapEdgeState,
    MapNodeState,
    MapState,
    RoomKind,
    RunPhase,
    RunState,
)
from sts2sim.engine.transitions import _event_eligibility_context
from sts2sim.mechanics.event_eligibility import (
    EventEligibilityContext,
    evaluate_event_eligibility,
)

SOURCE = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "event_eligibility_source.json").read_text()
)
EVENTS = {row["id"]: row for row in SOURCE["events"]}
ACTS = SOURCE["acts"]


def _context(**changes: Any) -> EventEligibilityContext:
    return replace(
        EventEligibilityContext(act=1, floor=1, gold=99, hp=80, max_hp=80),
        **changes,
    )


CONDITION_BOUNDARIES: tuple[tuple[str, dict[str, Any], dict[str, Any]], ...] = (
    ("Act 1 only", {"act": 1}, {"act": 2}),
    ("Act 1–2 only", {"act": 2}, {"act": 3}),
    ("Act 2 only", {"act": 2}, {"act": 1}),
    ("Act 2+", {"act": 2}, {"act": 1}),
    ("Floor 7+", {"floor": 7}, {"floor": 6}),
    ("Requires 10+ HP", {"hp": 10}, {"hp": 9}),
    ("Requires 12+ HP", {"hp": 12}, {"hp": 11}),
    ("Requires 19+ HP", {"hp": 19}, {"hp": 18}),
    ("Requires ≤70% HP", {"hp": 56}, {"hp": 57}),
    ("Requires 44+ gold", {"gold": 44}, {"gold": 43}),
    ("Requires 55+ gold", {"gold": 55}, {"gold": 54}),
    ("Requires 100+ gold", {"gold": 100}, {"gold": 99}),
    ("Requires 120+ gold", {"gold": 120}, {"gold": 119}),
    ("Requires 125+ gold", {"gold": 125}, {"gold": 124}),
    ("Requires 150+ gold", {"gold": 150}, {"gold": 149}),
    ("Requires 100-149 gold", {"gold": 149}, {"gold": 150}),
    ("Requires 100+ gold or a Foul Potion", {"gold": 100}, {"gold": 99}),
    ("Requires at least one potion", {"potions": ("fire_potion",)}, {"potions": ()}),
    ("Requires 2+ Strikes in deck", {"strikes": 2}, {"strikes": 1}),
    ("Requires 2+ Defends in deck", {"defends": 2}, {"defends": 1}),
    ("Requires removable cards in deck", {"removable_cards": 1}, {"removable_cards": 0}),
    (
        "Requires a removable basic card",
        {"removable_basic_cards": 1},
        {"removable_basic_cards": 0},
    ),
    ("Single player only", {"player_count": 1}, {"player_count": 2}),
    ("Cannot have an event pet", {"has_event_pet": False}, {"has_event_pet": True}),
    (
        "Requires more than one character unlocked",
        {"unlocked_character_count": 2},
        {"unlocked_character_count": 1},
    ),
    (
        "Requires 5+ tradeable relics",
        {"tradeable_relic_count": 5},
        {"tradeable_relic_count": 4},
    ),
)


def test_source_fixture_has_finite_complete_precondition_vocabulary() -> None:
    assert len(EVENTS) == 66
    actual = {condition for row in EVENTS.values() for condition in row["preconditions"] or ()}
    assert actual == {row[0] for row in CONDITION_BOUNDARIES}


@pytest.mark.parametrize("condition,allowed,blocked", CONDITION_BOUNDARIES)
def test_source_precondition_boundaries(
    condition: str, allowed: dict[str, Any], blocked: dict[str, Any]
) -> None:
    event = {"id": "CUSTOM", "preconditions": [condition]}
    eligible = evaluate_event_eligibility(event, _context(**allowed))
    assert eligible.eligible
    assert eligible.unknown == ()
    assert not evaluate_event_eligibility(event, _context(**blocked)).eligible


def test_gold_range_lower_bound_and_foul_potion_alternative() -> None:
    event = {"id": "CUSTOM", "preconditions": ["Requires 100-149 gold"]}
    assert not evaluate_event_eligibility(event, _context(gold=99)).eligible
    assert evaluate_event_eligibility(event, _context(gold=100)).eligible
    event["preconditions"] = ["Requires 100+ gold or a Foul Potion"]
    assert evaluate_event_eligibility(event, _context(gold=0, potions=("FOUL_POTION",))).eligible
    assert not evaluate_event_eligibility(
        event, _context(gold=0, potions=("fire_potion",))
    ).eligible


@pytest.mark.parametrize(
    "condition",
    [
        "Cannot have an event pet",
        "Requires more than one character unlocked",
        "Requires 5+ tradeable relics",
        "Requires a future source-defined condition",
    ],
)
def test_unavailable_state_is_admitted_with_explicit_unknown_diagnostic(condition: str) -> None:
    result = evaluate_event_eligibility({"preconditions": [condition]}, _context())
    assert result.eligible
    assert result.unknown == (condition,)


def test_unknown_requirement_does_not_override_known_failure() -> None:
    result = evaluate_event_eligibility(
        {"preconditions": ["Requires 100+ gold", "Requires a future source-defined condition"]},
        _context(gold=99),
    )
    assert not result.eligible


@pytest.mark.parametrize("act_id", ["OVERGROWTH", "UNDERDOCKS"])
def test_named_act_membership_matches_cached_act_event_lists(act_id: str) -> None:
    context = _context(act_id=act_id, gold=200)
    named_ids = next(row["events"] for row in ACTS if row["id"] == act_id)
    for event in EVENTS.values():
        if event["type"] != "Event":
            continue
        # Isolate act membership from each event's separate requirements.
        event = {**event, "preconditions": []}
        assert evaluate_event_eligibility(event, context, ACTS).eligible == (
            event["id"] in named_ids
        ), event["id"]


def test_all_nine_underdocks_exclusive_events_require_selected_variant() -> None:
    rows = [row for row in EVENTS.values() if row["act"] == "Underdocks"]
    assert len(rows) == 9
    for event in rows:
        event = {**event, "preconditions": []}
        assert evaluate_event_eligibility(event, _context(act_id="UNDERDOCKS"), ACTS).eligible
        assert not evaluate_event_eligibility(event, _context(act_id="OVERGROWTH"), ACTS).eligible
        assert not evaluate_event_eligibility(event, _context(act=2), ACTS).eligible


def test_named_act_fallback_parses_underdocks_without_numeric_act_prefix() -> None:
    event = {"id": "CUSTOM", "act": "Underdocks"}
    assert evaluate_event_eligibility(event, _context(act_id="UNDERDOCKS")).eligible
    assert not evaluate_event_eligibility(event, _context(act_id="OVERGROWTH")).eligible
    assert not evaluate_event_eligibility(event, _context(act=2)).eligible


def test_default_act_one_is_overgrowth_and_shared_statue_allows_either_variant() -> None:
    assert evaluate_event_eligibility(EVENTS["AROMA_OF_CHAOS"], _context(), ACTS).eligible
    assert not evaluate_event_eligibility(EVENTS["ABYSSAL_BATHS"], _context(), ACTS).eligible
    for act_id in ("OVERGROWTH", "UNDERDOCKS"):
        assert evaluate_event_eligibility(
            EVENTS["SUNKEN_STATUE"], _context(act_id=act_id), ACTS
        ).eligible


def test_ancients_are_not_random_event_candidates() -> None:
    rows = [row for row in EVENTS.values() if row["type"] == "Ancient"]
    assert rows
    for event in rows:
        for act in (1, 2, 3):
            assert not evaluate_event_eligibility(event, _context(act=act), ACTS).eligible


def test_cached_problem_events_have_source_backed_boundaries() -> None:
    assert not evaluate_event_eligibility(EVENTS["DOLL_ROOM"], _context(), ACTS).eligible
    assert evaluate_event_eligibility(EVENTS["DOLL_ROOM"], _context(act=2), ACTS).eligible
    assert not evaluate_event_eligibility(
        EVENTS["CRYSTAL_SPHERE"], _context(act=2, gold=99), ACTS
    ).eligible
    assert evaluate_event_eligibility(
        EVENTS["CRYSTAL_SPHERE"], _context(act=2, gold=100), ACTS
    ).eligible
    assert not evaluate_event_eligibility(
        EVENTS["SLIPPERY_BRIDGE"], _context(floor=6, removable_cards=1), ACTS
    ).eligible
    assert evaluate_event_eligibility(
        EVENTS["SLIPPERY_BRIDGE"], _context(floor=7, removable_cards=1), ACTS
    ).eligible


def _enter_public_event(
    events: list[dict[str, Any]],
    *,
    act: int = 1,
    floor: int = 1,
    gold: int = 99,
    **configuration: Any,
) -> RunState:
    state = new_run(
        seed=105,
        character_id="IRONCLAD",
        ascension=0,
        source_data={
            "act": act,
            "events": events,
            "player": {"hp": 80, "max_hp": 80, "gold": gold},
            "flags": {"resolve_unknown_rooms": False},
            **configuration,
        },
    )
    start = MapNodeState(node_id="start", act=act, floor=floor - 1, lane=0, kind=RoomKind.START)
    target = MapNodeState(node_id="target", act=act, floor=floor, lane=0, kind=RoomKind.EVENT)
    game_map = MapState(
        act=act,
        nodes=(start, target),
        edges=(MapEdgeState(from_id="start", to_id="target"),),
        current_node_id="start",
        completed_node_ids=("start",),
    )
    state = state.model_copy(
        update={"phase": RunPhase.MAP, "ancient": None, "map": game_map, "floor": floor - 1}
    )
    action = next(action for action in legal_actions(state) if action.type == "choose_node")
    return step(state, action)


@pytest.mark.parametrize(
    "event_id,act,floor,gold,expected",
    [
        ("DOLL_ROOM", 1, 1, 99, "UNKNOWN_EVENT"),
        ("DOLL_ROOM", 2, 1, 99, "DOLL_ROOM"),
        ("CRYSTAL_SPHERE", 1, 1, 100, "UNKNOWN_EVENT"),
        ("CRYSTAL_SPHERE", 2, 1, 99, "UNKNOWN_EVENT"),
        ("CRYSTAL_SPHERE", 2, 1, 100, "CRYSTAL_SPHERE"),
        ("SLIPPERY_BRIDGE", 1, 6, 99, "UNKNOWN_EVENT"),
        ("SLIPPERY_BRIDGE", 1, 7, 99, "SLIPPERY_BRIDGE"),
    ],
)
def test_public_event_entry_applies_cached_predicates(
    event_id: str, act: int, floor: int, gold: int, expected: str
) -> None:
    state = _enter_public_event(
        [EVENTS[event_id]], act=act, floor=floor, gold=gold, event_pool=(event_id.lower(),)
    )
    assert state.phase == RunPhase.EVENT
    assert state.event is not None
    assert state.event.event_id == expected


def test_public_explicit_event_override_preserves_fixture_control() -> None:
    state = _enter_public_event([EVENTS["DOLL_ROOM"]], event_id="DOLL_ROOM")
    assert state.event is not None
    assert state.event.event_id == "DOLL_ROOM"


def test_public_empty_custom_pool_does_not_admit_unconfigured_eligible_event() -> None:
    state = _enter_public_event(
        [EVENTS["DOLL_ROOM"], {"id": "ELIGIBLE_BUT_NOT_IN_POOL"}],
        event_pool=("doll_room",),
    )
    assert state.event is not None
    assert state.event.event_id == "UNKNOWN_EVENT"
    assert state.event.metadata["eligibility_diagnostics"]["reason"] == "no_eligible_event"


def test_public_unmodeled_precondition_has_visible_diagnostic() -> None:
    condition = "Requires a future source-defined condition"
    state = _enter_public_event([{"id": "CUSTOM", "preconditions": [condition]}])
    assert state.event is not None
    assert state.event.event_id == "CUSTOM"
    unknown = state.event.metadata["eligibility_diagnostics"]["unknown_requirements"]
    assert tuple(unknown["CUSTOM"]) == (condition,)


def test_public_named_variant_admits_underdocks_event() -> None:
    state = _enter_public_event([EVENTS["ABYSSAL_BATHS"]], act_id="UNDERDOCKS")
    assert state.event is not None
    assert state.event.event_id == "ABYSSAL_BATHS"
    other = _enter_public_event([EVENTS["ABYSSAL_BATHS"]], act_id="OVERGROWTH")
    assert other.event is not None
    assert other.event.event_id == "UNKNOWN_EVENT"


@pytest.mark.parametrize(
    "character,strikes,defends,basic_cards",
    [("IRONCLAD", 5, 4, 10), ("SILENT", 5, 5, 12)],
)
def test_runtime_adapter_counts_actual_starters_and_excludes_pommel_strike(
    character: str, strikes: int, defends: int, basic_cards: int
) -> None:
    state = new_run(81, character, 0)
    pommel = CardInstance(
        instance_id="extra-pommel", card_id="pommel_strike", name="Pommel Strike",
        custom={"source_spec": {"rarity": "Common"}},
    )
    state = state.model_copy(update={"master_deck": state.master_deck + (pommel,)})
    context = _event_eligibility_context(state)
    assert (context.strikes, context.defends) == (strikes, defends)
    assert context.removable_basic_cards == basic_cards
    assert context.removable_cards == basic_cards + 1


def test_runtime_adapter_basic_rarity_eternal_profile_and_allies() -> None:
    state = new_run(
        82, "IRONCLAD", 0,
        source_data={"flags": {
            "act_ids": {"1": "UNDERDOCKS"},
            "has_event_pet": True,
            "unlocked_character_count": 3,
            "tradeable_relic_count": 5,
            "allies": ({"id": "partner", "hp": 40, "max_hp": 70},),
        }},
    )
    bash = next(card for card in state.master_deck if card.card_id == "bash")
    eternal_strike = next(card for card in state.master_deck if card.name == "Strike")
    eternal_strike = eternal_strike.model_copy(update={"tags": ("eternal",)})
    state = state.model_copy(update={"master_deck": (bash, eternal_strike)})
    context = _event_eligibility_context(state)
    assert context.removable_cards == context.removable_basic_cards == 1
    assert context.act_id == "UNDERDOCKS"
    assert context.player_count == 2
    assert context.has_event_pet is True
    assert context.unlocked_character_count == 3
    assert context.tradeable_relic_count == 5


def test_custom_event_keeps_declared_act_when_cached_membership_has_no_row() -> None:
    event = {"id": "CUSTOM_ACT_EVENT", "act": "Act 2"}
    assert evaluate_event_eligibility(event, _context(act=2), ACTS).eligible
    assert not evaluate_event_eligibility(event, _context(act=1), ACTS).eligible


def test_amalgamator_entry_grouping_stays_explicit_when_only_one_option_is_possible() -> None:
    event = EVENTS["AMALGAMATOR"]
    assert not evaluate_event_eligibility(
        event, _context(act=2, strikes=1, defends=1), ACTS
    ).eligible
    result = evaluate_event_eligibility(event, _context(act=2, strikes=2, defends=2), ACTS)
    assert result.eligible
    assert result.unknown == ()
    for strikes, defends in ((2, 1), (1, 2)):
        result = evaluate_event_eligibility(
            event, _context(act=2, strikes=strikes, defends=defends), ACTS
        )
        assert result.eligible
        assert result.unknown == ("Amalgamator entry predicate grouping unverified",)
