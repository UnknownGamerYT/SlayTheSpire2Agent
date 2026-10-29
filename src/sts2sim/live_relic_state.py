"""Translate observed, history-derived relic state without reading the RNG oracle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# (observed field, source period). Internal periodic counters store progress,
# rather than the number of actions remaining until activation.
_PERIODIC = {
    "nunchaku": ("attacks_played", 10),
    "pen_nib": ("attacks_played", 10),
    "happy_flower": ("turns_seen", 3),
    "fake_happy_flower": ("turns_seen", 5),
    "joss_paper": ("cards_exhausted", 5),
    "galactic_dust": ("stars_spent", 10),
    "pendulum": ("turns_seen", 3),
    "pollinous_core": ("turns_seen", 4),
    "fishing_rod": ("combats_seen", 3),
    "book_of_five_rings": ("cards_added", 5),
}
_USED = {
    "burning_sticks": "used_this_combat",
    "centennial_puzzle": "used_this_combat",
    "permafrost": "used_this_combat",
    "ruined_helmet": "used_this_combat",
    "throwing_axe": "used_this_combat",
    "paels_eye": "used_this_combat",
    "demon_tongue": "triggered_this_turn",
    "mini_regent": "used_this_turn",
    "red_skull": "strength_applied",
    "belt_buckle": "dexterity_applied",
    "vambrace": "block_gained_this_combat",
}


def relic_runtime_projection(
    player: Mapping[str, Any], battle: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return flag/metadata patches for a coherent public decision capture.

    ``runtime_state`` contains a finite bridge allowlist of primitive facts
    determined by the owner's past actions. Missing one-shot history fails
    closed; a generic relic status cannot establish whether a hook ran.
    Animation display values equal to a period represent wrapped progress.
    Unrelated relics are left alone, not assigned guessed counters.
    """
    rows = player.get("relics", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("Live relics must be a list of objects.")
    counters: dict[str, int] = {}
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Live relics must contain objects.")
        relic_id = str(row.get("id", "")).strip().lower()
        tracked = relic_id in _PERIODIC or relic_id in _USED or relic_id in {
            "iron_club", "pumpkin_candle", "metronome", "paels_legion", "unsettling_lamp",
            "wongos_mystery_ticket",
        }
        if not relic_id or (tracked and relic_id in seen):
            raise ValueError("Tracked live relic identities must be present and unique.")
        seen.add(relic_id)
        runtime = row.get("runtime_state", {})
        if not isinstance(runtime, Mapping):
            raise ValueError(f"Invalid runtime state for {relic_id}.")
        if relic_id in _PERIODIC:
            field, period = _PERIODIC[relic_id]
            raw = runtime.get(field, row.get("counter"))
            value = _nonnegative(raw, f"{relic_id}.{field}") % period
            _check_display(row, value, period=period)
            counters[relic_id] = value
        elif relic_id == "iron_club":
            # The engine also exposes the cumulative count; the UI shows only
            # count % 4, so it cannot recover the actor's cumulative feature.
            value = _nonnegative(runtime.get("cards_played"), "iron_club.cards_played")
            _check_display(row, value % 4, period=4)
            counters[relic_id] = value
        elif relic_id == "pumpkin_candle":
            value = _nonnegative(
                runtime.get("kindle_count", row.get("counter")), "pumpkin_candle.kindle_count"
            )
            _check_display(row, value)
            counters[relic_id] = value
        elif battle and relic_id == "metronome":
            value = _nonnegative(runtime.get("orbs_channeled"), "metronome.orbs_channeled")
            _check_display(row, min(7, value))
            counters[relic_id] = value
        elif battle and relic_id == "paels_legion":
            raw_cooldown = runtime.get("cooldown")
            if type(raw_cooldown) is not int:
                raise ValueError("Missing or invalid live paels_legion.cooldown integer.")
            if raw_cooldown > 2:
                raise ValueError("Pael's Legion cooldown exceeds its source duration.")
            # The source keeps decrementing below zero; the engine's readiness
            # representation saturates at zero, with identical trigger behavior.
            value = max(0, raw_cooldown)
            _check_display(row, value)
            counters[relic_id] = value
        elif battle and relic_id == "unsettling_lamp":
            finished = _boolean(runtime, "finished_triggering", relic_id)
            triggering = _boolean(runtime, "triggering_card_set", relic_id)
            counters[relic_id] = int(finished or triggering)
        elif battle and relic_id in _USED:
            field = _USED[relic_id]
            counters[relic_id] = int(_boolean(runtime, field, relic_id))
        elif relic_id == "wongos_mystery_ticket":
            # ShowCounter is false at zero and after activation. The simulator
            # uses the same remaining-combats convention, not CombatsFinished.
            remaining = row.get("counter")
            counters[relic_id] = _nonnegative(0 if remaining is None else remaining, relic_id)
    return {"relic_counters": dict(counters)}, {"relic_counters": dict(counters)}


def _nonnegative(raw: object, field: str) -> int:
    if type(raw) is not int or raw < 0:
        raise ValueError(f"Missing or invalid nonnegative live {field} integer.")
    return raw


def _boolean(runtime: Mapping[str, Any], field: str, relic_id: str) -> bool:
    raw = runtime.get(field)
    if type(raw) is not bool:
        raise ValueError(f"Missing or invalid live {relic_id}.{field} boolean.")
    return raw


def _check_display(row: Mapping[str, Any], value: int, *, period: int | None = None) -> None:
    raw = row.get("counter")
    if raw is None:
        return
    display = _nonnegative(raw, f"{row.get('id')}.counter")
    if display != value and not (period is not None and display == period and value == 0):
        raise ValueError(f"Live {row.get('id')} display disagrees with its observed runtime state.")
