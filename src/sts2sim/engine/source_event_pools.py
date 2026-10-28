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
    state: RunState, pool_id: str, *, for_combat_generation: bool = False
) -> tuple[str, ...]:
    epochs = (tuple(str(epoch) for epoch in state.flags['revealed_epochs'])
              if 'revealed_epochs' in state.flags else None)
    return source_pool_cards(pool_id, unlocked_epochs=epochs,
                             multiplayer=int(state.flags.get('player_count', 1)) > 1,
                             for_combat_generation=for_combat_generation)
