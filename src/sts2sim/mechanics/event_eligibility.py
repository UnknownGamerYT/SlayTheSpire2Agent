"""Source-backed event entry predicates, with unknown requirements kept visible."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EventEligibilityContext:
    act: int
    floor: int
    gold: int
    hp: int
    max_hp: int
    act_id: str | None = None
    removable_cards: int = 0
    removable_basic_cards: int = 0
    strikes: int = 0
    defends: int = 0
    potions: tuple[str, ...] = ()
    player_count: int = 1
    has_event_pet: bool | None = None
    unlocked_character_count: int | None = None
    tradeable_relic_count: int | None = None


@dataclass(frozen=True)
class EventEligibility:
    eligible: bool
    unknown: tuple[str, ...] = ()


def evaluate_event_eligibility(
    event: Mapping[str, Any],
    context: EventEligibilityContext,
    act_definitions: Sequence[Mapping[str, Any]] = (),
) -> EventEligibility:
    """Evaluate the cached entry clauses; unspecified profile facts stay unknown.

    An unknown clause does not exclude an event. Callers must surface ``unknown``
    rather than treating its admission as verified eligibility.
    """
    if str(event.get("type", "")).lower() == "ancient":
        return EventEligibility(False)
    act_result = _act_eligibility(event, context, act_definitions)
    if not act_result.eligible:
        return act_result
    unknown = list(act_result.unknown)
    raw_conditions = event.get("preconditions") or ()
    if isinstance(raw_conditions, str):
        raw_conditions = (raw_conditions,)
    # The API lists both option requirements without their entry AND/OR rule.
    # Only exclude when neither option is available; keep mixed cases explicit.
    amalgamator_conditions = {"Requires 2+ Strikes in deck", "Requires 2+ Defends in deck"}
    if str(event.get("id", "")).upper() == "AMALGAMATOR" and (
        amalgamator_conditions <= set(raw_conditions)
    ):
        matches = (context.strikes >= 2, context.defends >= 2)
        if not any(matches):
            return EventEligibility(False)
        if not all(matches):
            unknown.append("Amalgamator entry predicate grouping unverified")
        raw_conditions = tuple(c for c in raw_conditions if c not in amalgamator_conditions)
    for raw in raw_conditions:
        condition = str(raw)
        result = _condition_matches(condition, context)
        if result is False:
            return EventEligibility(False)
        if result is None:
            unknown.append(condition)
    return EventEligibility(True, tuple(unknown))


def _act_eligibility(
    event: Mapping[str, Any],
    context: EventEligibilityContext,
    acts: Sequence[Mapping[str, Any]],
) -> EventEligibility:
    label = str(event.get("act") or "").strip().lower()
    if not label:
        return EventEligibility(True)
    selected = (
        context.act_id or {1: "OVERGROWTH", 2: "HIVE", 3: "GLORY"}.get(context.act, "")
    ).lower()
    for act in acts:
        if str(act.get("id", "")).lower() != selected:
            continue
        # A stale current-act flag must not make Act 1 events eligible in Act 2.
        if int(act.get("index", context.act - 1)) + 1 != context.act:
            return EventEligibility(False)
        event_ids = act.get("events")
        known_ids = {str(value).lower() for row in acts for value in (row.get("events") or ())}
        if (
            isinstance(event_ids, Sequence)
            and not isinstance(event_ids, str)
            and str(event.get("id", "")).lower() in known_ids
        ):
            return EventEligibility(
                str(event.get("id", "")).lower() in {str(value).lower() for value in event_ids}
            )
    named = tuple(name for name in ("overgrowth", "underdocks", "hive", "glory") if name in label)
    if named:
        expected_act = {"overgrowth": 1, "underdocks": 1, "hive": 2, "glory": 3}
        return EventEligibility(selected in named and expected_act[selected] == context.act)
    match = re.fullmatch(r"act (\d+)", label)
    if match:
        return EventEligibility(context.act == int(match[1]))
    return EventEligibility(True, (f"unrecognized act: {event.get('act')}",))


def _condition_matches(condition: str, context: EventEligibilityContext) -> bool | None:
    text = condition.lower().replace("–", "-")
    match = re.fullmatch(r"act (\d+)(?:-(\d+))? only", text)
    if match:
        return int(match[1]) <= context.act <= int(match[2] or match[1])
    match = re.fullmatch(r"act (\d+)\+", text)
    if match:
        return context.act >= int(match[1])
    match = re.fullmatch(r"floor (\d+)\+", text)
    if match:
        return context.floor >= int(match[1])
    match = re.fullmatch(r"requires (\d+)\+ (gold|hp)", text)
    if match:
        return (context.gold if match[2] == "gold" else context.hp) >= int(match[1])
    match = re.fullmatch(r"requires (\d+)-(\d+) gold", text)
    if match:
        return int(match[1]) <= context.gold <= int(match[2])
    if text == "requires ≤70% hp":
        return context.hp * 100 <= context.max_hp * 70
    if text == "requires 100+ gold or a foul potion":
        return context.gold >= 100 or any(
            potion.lower() == "foul_potion" for potion in context.potions
        )
    if text == "requires at least one potion":
        return bool(context.potions)
    if text == "requires 2+ strikes in deck":
        return context.strikes >= 2
    if text == "requires 2+ defends in deck":
        return context.defends >= 2
    if text == "requires removable cards in deck":
        return context.removable_cards > 0
    if text == "requires a removable basic card":
        return context.removable_basic_cards > 0
    if text == "single player only":
        return context.player_count == 1
    if text == "cannot have an event pet":
        return None if context.has_event_pet is None else not context.has_event_pet
    if text == "requires more than one character unlocked":
        return (
            None
            if context.unlocked_character_count is None
            else context.unlocked_character_count > 1
        )
    if text == "requires 5+ tradeable relics":
        return None if context.tradeable_relic_count is None else context.tradeable_relic_count >= 5
    return None
