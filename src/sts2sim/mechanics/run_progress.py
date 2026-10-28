"""Shared source map-history floor convention for event and relic eligibility."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sts2sim.engine.models import RunState


def source_total_floor(state: RunState, flags: Mapping[str, Any] | None = None) -> int:
    """Count current entered rooms and per-act Ancient entries consistently."""
    facts = state.flags if flags is None else flags
    if "total_floor" in facts:
        try:
            return int(facts["total_floor"])
        except (TypeError, ValueError):
            return 0
    entered = bool(
        state.map is not None and state.map.current_node_id is not None
        and state.map.current_node_id not in state.map.completed_node_ids
    )
    return (
        len(state.room_history) + state.act + int(entered)
        - int(facts.get("generation_started_with_neow") is False)
    )
