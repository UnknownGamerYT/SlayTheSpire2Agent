"""Stable source power types used by Rend's distinct-debuff damage scaling."""

from __future__ import annotations

from collections.abc import Mapping

from .source_power_rules import SOURCE_POWER_DEFINITIONS, unique_source_debuff_count

SOURCE_DEBUFF_POWER_IDS = frozenset(
    power_id.upper()
    for power_id, definition in SOURCE_POWER_DEFINITIONS.items()
    if definition.kind == "debuff"
)


def unique_debuff_count(statuses: Mapping[str, int]) -> int:
    return unique_source_debuff_count(statuses)
