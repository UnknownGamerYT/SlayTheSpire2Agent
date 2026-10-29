"""Source-backed event entry predicates, with unknown requirements kept visible."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .event_entry_data import SOURCE_EVENT_IDS


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
    total_floor: int | None = None
    deck_size: int | None = None
    transformable_cards: int | None = None
    perfect_fit_cards: int | None = None
    souls_power_cards: int | None = None
    spiral_cards: int | None = None
    has_available_relics: bool | None = None
    unavailable_player_facts: frozenset[str] = frozenset()
    other_players: tuple[EventEligibilityContext, ...] = ()


@dataclass(frozen=True)
class EventEligibility:
    eligible: bool
    unknown: tuple[str, ...] = ()


def evaluate_event_eligibility(
    event: Mapping[str, Any],
    context: EventEligibilityContext,
    act_definitions: Sequence[Mapping[str, Any]] = (),
) -> EventEligibility:
    """Evaluate stable source predicates or custom printed entry clauses.

    An unknown clause does not exclude an event. Callers must surface ``unknown``
    rather than treating its admission as verified eligibility.
    """
    if str(event.get("type", "")).lower() == "ancient":
        return EventEligibility(False)
    act_result = _act_eligibility(event, context, act_definitions)
    if not act_result.eligible:
        return act_result
    event_id = str(event.get("id", "")).upper()
    if event_id in SOURCE_EVENT_IDS:
        source_result = _stable_source_entry(event_id, context)
        if not source_result.eligible:
            return source_result
        unknown = [*act_result.unknown, *source_result.unknown]
        if event_id in _ALL_PLAYER_REQUIREMENTS:
            for index, player in enumerate(context.other_players, start=2):
                other = _stable_source_entry(event_id, player)
                if not other.eligible:
                    return other
                unknown.extend(f"Player {index}: {value}" for value in other.unknown)
            if len(context.other_players) < context.player_count - 1:
                unknown.append("Other players' entry prerequisites not supplied")
        return EventEligibility(True, tuple(unknown))
    unknown = list(act_result.unknown)
    raw_conditions = event.get("preconditions") or ()
    if isinstance(raw_conditions, str):
        raw_conditions = (raw_conditions,)
    for raw in raw_conditions:
        condition = str(raw)
        result = _condition_matches(condition, context)
        if result is False:
            return EventEligibility(False)
        if result is None:
            unknown.append(condition)
    return EventEligibility(True, tuple(unknown))


_ALL_PLAYER_REQUIREMENTS = frozenset(
    {
        "AMALGAMATOR",
        "BYRDONIS_NEST",
        "COLORFUL_PHILOSOPHERS",
        "COLOSSAL_FLOWER",
        "CRYSTAL_SPHERE",
        "DENSE_VEGETATION",
        "ENDLESS_CONVEYOR",
        "FAKE_MERCHANT",
        "FIELD_OF_MAN_SIZED_HOLES",
        "GRAVE_OF_THE_FORGOTTEN",
        "LUMINOUS_CHOIR",
        "MORPHIC_GROVE",
        "RANWID_THE_ELDER",
        "RELIC_TRADER",
        "ROUND_TEA_PARTY",
        "SLIPPERY_BRIDGE",
        "SPIRALING_WHIRLPOOL",
        "STONE_OF_ALL_TIME",
        "TEA_MASTER",
        "THE_FUTURE_OF_POTIONS",
        "THE_LEGENDS_WERE_TRUE",
        "TRASH_HEAP",
        "UNREST_SITE",
        "WATERLOGGED_SCRIPTORIUM",
        "WELCOME_TO_WONGOS",
        "WHISPERING_HOLLOW",
        "WOOD_CARVINGS",
        "ZEN_WEAVER",
    }
)


def _stable_source_entry(event_id: str, c: EventEligibilityContext) -> EventEligibility:
    """Exact v0.107.1 IsAllowed overrides; unspecified facts stay explicit."""
    checks: list[tuple[bool | None, str]] = []

    def fact(name: str, allowed: bool) -> bool | None:
        return None if name in c.unavailable_player_facts else allowed

    def count(name: str, value: int | None, minimum: int = 1) -> None:
        checks.append((None if value is None else value >= minimum, name))

    if event_id in {"BRAIN_LEECH", "ROOM_FULL_OF_CHEESE"}:
        checks.append((c.act < 3, "Act 1 or 2"))
    elif event_id in {"POTION_COURIER", "SYMBIOTE"}:
        checks.append((c.act > 1, "Act 2 or 3"))
    elif event_id == "DOLL_ROOM":
        checks.append((c.act == 2, "Act 2"))
    elif event_id == "WAR_HISTORIAN_REPY":
        checks.append((False, "Disabled in stable source"))
    elif event_id == "AMALGAMATOR":
        checks += [
            (fact("strikes", c.strikes >= 2), "2 removable Basic Strike-tag cards"),
            (fact("defends", c.defends >= 2), "2 removable Basic Defend-tag cards"),
        ]
    elif event_id == "BYRDONIS_NEST":
        checks.append(
            (None if c.has_event_pet is None else not c.has_event_pet, "Cannot have an event pet")
        )
    elif event_id == "COLORFUL_PHILOSOPHERS":
        count("More than one unlocked character card pool", c.unlocked_character_count, 2)
    elif event_id in {"COLOSSAL_FLOWER", "ROUND_TEA_PARTY", "TRASH_HEAP"}:
        minimum = {"COLOSSAL_FLOWER": 19, "ROUND_TEA_PARTY": 12, "TRASH_HEAP": 6}[event_id]
        checks.append((fact("hp", c.hp >= minimum), f"Requires {minimum}+ HP"))
    elif event_id == "DENSE_VEGETATION" and c.player_count > 1:
        checks.append((fact("hp", c.hp > 8), "Multiplayer HP must exceed 8"))
    elif event_id in {
        "ENDLESS_CONVEYOR",
        "WATERLOGGED_SCRIPTORIUM",
        "WHISPERING_HOLLOW",
        "ZEN_WEAVER",
    }:
        minimum = {
            "ENDLESS_CONVEYOR": 120,
            "WATERLOGGED_SCRIPTORIUM": 55,
            "WHISPERING_HOLLOW": 44,
            "ZEN_WEAVER": 125,
        }[event_id]
        checks.append((fact("gold", c.gold >= minimum), f"Requires {minimum}+ gold"))
    elif event_id == "CRYSTAL_SPHERE":
        checks += [(c.act > 1, "Act 2 or 3"), (fact("gold", c.gold >= 100), "Requires 100+ gold")]
    elif event_id == "FAKE_MERCHANT":
        gold = fact("gold", c.gold >= 100)
        foul = fact("potions", any(value.upper() == "FOUL_POTION" for value in c.potions))
        alternatives = (
            True if gold is True or foul is True else None if None in (gold, foul) else False
        )
        checks += [
            (c.act > 1, "Act 2 or 3"),
            (c.player_count == 1, "Single player only"),
            (alternatives, "Requires 100+ gold or Foul Potion"),
        ]
    elif event_id == "FIELD_OF_MAN_SIZED_HOLES":
        count("Requires a Perfect Fit enchantable card", c.perfect_fit_cards)
    elif event_id == "GRAVE_OF_THE_FORGOTTEN":
        count("Requires a Soul's Power enchantable card", c.souls_power_cards)
    elif event_id == "SPIRALING_WHIRLPOOL":
        count("Requires a Spiral enchantable Basic Strike/Defend", c.spiral_cards)
    elif event_id == "LUMINOUS_CHOIR":
        # IsAllowed sees the canonical GoldVar(149). CalculateVars rolls the
        # lower 100..149 option price only after entry, on the mutable event.
        checks += [
            (fact("gold", c.gold >= 149), "Requires 149+ gold at entry"),
            (c.has_available_relics, "Requires an available relic in the player's bag"),
        ]
    elif event_id == "MORPHIC_GROVE":
        checks.append((fact("gold", c.gold >= 100), "Requires 100+ gold"))
        count("Requires 2 transformable deck cards", c.transformable_cards, 2)
    elif event_id in {"RANWID_THE_ELDER", "RELIC_TRADER"}:
        checks.append((c.act > 1, "Act 2 or 3"))
        count(
            "Requires tradeable relics",
            c.tradeable_relic_count,
            5 if event_id == "RELIC_TRADER" else 1,
        )
        if event_id == "RANWID_THE_ELDER":
            checks += [
                (fact("gold", c.gold >= 100), "Requires 100+ gold"),
                (fact("potions", bool(c.potions)), "Requires a potion"),
            ]
    elif event_id in {"PUNCH_OFF", "SLIPPERY_BRIDGE"}:
        minimum = 6 if event_id == "PUNCH_OFF" else 7
        checks.append(
            (
                (c.total_floor if c.total_floor is not None else c.floor) >= minimum,
                f"Requires total floor {minimum}+",
            )
        )
        if event_id == "SLIPPERY_BRIDGE":
            checks.append(
                (fact("removable_cards", c.removable_cards > 0), "Requires removable cards")
            )
    elif event_id == "STONE_OF_ALL_TIME":
        checks += [(c.act == 2, "Act 2"), (fact("potions", bool(c.potions)), "Requires a potion")]
    elif event_id == "TEA_MASTER":
        checks += [(c.act < 3, "Act 1 or 2"), (fact("gold", c.gold >= 150), "Requires 150+ gold")]
    elif event_id == "THE_FUTURE_OF_POTIONS":
        checks.append((fact("potions", len(c.potions) >= 2), "Requires at least 2 potions"))
    elif event_id == "THE_LEGENDS_WERE_TRUE":
        checks += [(c.act == 1, "Act 1"), (fact("hp", c.hp >= 10), "Requires 10+ HP")]
        count("Requires a nonempty deck", c.deck_size)
    elif event_id == "UNREST_SITE":
        checks.append(
            (
                None
                if {"hp", "max_hp"} & c.unavailable_player_facts
                else c.hp * 100 <= c.max_hp * 70,
                "Requires at most 70% HP",
            )
        )
    elif event_id == "WELCOME_TO_WONGOS":
        checks += [(c.act == 2, "Act 2"), (fact("gold", c.gold >= 100), "Requires 100+ gold")]
    elif event_id == "WOOD_CARVINGS":
        checks.append(
            (
                fact("removable_basic_cards", c.removable_basic_cards > 0),
                "Requires a removable Basic card",
            )
        )
    return EventEligibility(
        not any(value is False for value, _ in checks),
        tuple(label for value, label in checks if value is None),
    )


def _act_eligibility(
    event: Mapping[str, Any],
    context: EventEligibilityContext,
    acts: Sequence[Mapping[str, Any]],
) -> EventEligibility:
    label = str(event.get("act") or "").strip().lower()
    if not label or label in {"shared", "any", "all"}:
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
        return EventEligibility(selected in named and expected_act.get(selected) == context.act)
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
