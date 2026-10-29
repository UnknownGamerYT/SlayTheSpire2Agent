"""Physical potion-slot identities, independent of the packed legacy inventory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

POTION_SLOT_INDICES = "potion_slot_indices"


def potion_slot_indices(potions: Sequence[object], flags: Mapping[str, Any]) -> tuple[int, ...]:
    """Old saves have contiguous slots; explicit saved identities are authoritative."""
    raw = flags.get(POTION_SLOT_INDICES)
    if raw is None:
        return tuple(range(len(potions)))
    if (
        not isinstance(raw, list | tuple)
        or len(raw) != len(potions)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in raw)
        or len(set(raw)) != len(raw)
    ):
        raise ValueError(
            "Potion slot indices must be distinct nonnegative integers, one per potion"
        )
    return tuple(raw)


def occupied_potion_slots(
    potions: Sequence[str],
    flags: Mapping[str, Any],
) -> tuple[tuple[int, str], ...]:
    return tuple(sorted(zip(potion_slot_indices(potions, flags), potions, strict=True)))


def potion_slot_contents(
    potions: Sequence[str],
    flags: Mapping[str, Any],
    capacity: int,
) -> tuple[str | None, ...]:
    slots: list[str | None] = [None] * max(0, capacity)
    for index, potion in occupied_potion_slots(potions, flags):
        if index >= capacity:
            raise ValueError("A saved potion slot exceeds the potion capacity")
        slots[index] = potion
    return tuple(slots)
