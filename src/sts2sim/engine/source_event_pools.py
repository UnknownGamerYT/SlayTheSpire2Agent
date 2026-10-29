"""Propagate the run's revealed epochs into native event factory pools."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sts2sim.content.card_pool_selection import PLAYABLE_CARD_POOLS, source_card_pool_rows
from sts2sim.content.source_card_pools import SOURCE_CARD_POOLS, source_pool_cards

from .models import RunState


def source_event_card_rows(
    state: RunState, rows: tuple[Any, ...], pool_id: str,
) -> tuple[Mapping[str, Any], ...] | None:
    if state.character_id.lower() not in PLAYABLE_CARD_POOLS or state.flags.get('card_pool'):
        return None
    registered = {card for pool in SOURCE_CARD_POOLS.values() for card in pool}
    if not any(isinstance(row, Mapping)
               and str(row.get('id', row.get('card_id', ''))).lower() in registered
               for row in rows):
        return None
    epochs = (tuple(str(epoch) for epoch in state.flags['revealed_epochs'])
              if 'revealed_epochs' in state.flags else None)
    return source_card_pool_rows(rows, pool_id, unlocked_epochs=epochs,
                                 multiplayer=int(state.flags.get('player_count', 1)) > 1)


def source_transform_ids(
    state: RunState, pool_id: str, *, for_combat_generation: bool = False,
    rows: tuple[Any, ...] | None = None,
) -> tuple[str, ...] | None:
    """Return eligible catalog IDs; None denotes an explicit synthetic/custom pool.

    Empty registered eligibility is distinct from the custom-pool sentinel. A
    missing or locked registered pool must never fall back to the full cache.
    """
    if state.character_id.lower() not in PLAYABLE_CARD_POOLS or state.flags.get('card_pool'):
        return None
    available: set[str] | None = None
    if rows is not None:
        available = {str(row.get('id', row.get('card_id', ''))).lower()
                     for row in rows if isinstance(row, Mapping)}
        registered = {card for pool in SOURCE_CARD_POOLS.values() for card in pool}
        if not available.intersection(registered):
            return None
    epochs = (tuple(str(epoch) for epoch in state.flags['revealed_epochs'])
              if 'revealed_epochs' in state.flags else None)
    identifiers = source_pool_cards(pool_id, unlocked_epochs=epochs,
                                    multiplayer=int(state.flags.get('player_count', 1)) > 1,
                                    for_combat_generation=for_combat_generation)
    return tuple(card for card in identifiers if available is None or card in available)
