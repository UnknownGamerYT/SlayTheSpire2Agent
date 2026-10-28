"""Source-backed adaptive probabilities for question-mark map rooms.

See https://spire-codex.com/api/mechanics/sections/unknown-rooms. The caller
owns persistent odds and resets them between acts. Blacklisted outcomes neither
roll nor gain pity; negative odds never accumulate.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from random import Random

from .source_rng import float32, is_source_rng, source_float

BASE_UNKNOWN_ROOM_ODDS = {"monster": 0.10, "elite": -1.0, "treasure": 0.02, "shop": 0.03}


@dataclass(frozen=True)
class UnknownRoomResolution:
    outcome: str
    roll: float
    next_odds: Mapping[str, float]


def resolve_unknown_room(
    rng: Random,
    *,
    previous_odds: Mapping[str, float] | None = None,
    base_odds: Mapping[str, float] | None = None,
    blacklisted: Iterable[str] = (),
    allowed_outcomes: Iterable[str] | None = None,
) -> UnknownRoomResolution:
    """Use one roll in monster/elite/treasure/shop order, then update pity."""

    bases = {**BASE_UNKNOWN_ROOM_ODDS, **dict(base_odds or {})}
    odds = {key: float((previous_odds or {}).get(key, value)) for key, value in bases.items()}
    if any(not math.isfinite(value) for value in (*bases.values(), *odds.values())):
        raise ValueError("Unknown-room probabilities must be finite.")
    blocked = set(blacklisted)
    allowed = set(allowed_outcomes) if allowed_outcomes is not None else {
        *BASE_UNKNOWN_ROOM_ODDS, "event"
    }
    source_mode = is_source_rng(rng)
    if source_mode:
        bases = {key: float32(value) for key, value in bases.items()}
        odds = {key: float32(value) for key, value in odds.items()}
    eligible = tuple(key for key in BASE_UNKNOWN_ROOM_ODDS
                     if key not in blocked and key in allowed and odds[key] >= 0)
    roll = source_float(rng) if source_mode else rng.random()
    threshold = 0.0
    outcome = "event" if "event" in allowed else next(iter(eligible), "event")
    for key in eligible:
        threshold = float32(threshold + odds[key]) if source_mode else threshold + odds[key]
        if roll <= threshold if source_mode else roll < threshold:
            outcome = key
            break
    updated = dict(odds)
    for key in BASE_UNKNOWN_ROOM_ODDS if source_mode else eligible:
        if key not in blocked and key in allowed:
            updated[key] = bases[key] if key == outcome else (
                float32(odds[key] + bases[key]) if source_mode
                else odds[key] + max(0.0, bases[key])
            )
    return UnknownRoomResolution(outcome=outcome, roll=roll, next_odds=updated)
