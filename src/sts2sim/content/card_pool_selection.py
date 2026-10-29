"""Join cached card metadata to the stable game's registered reward pools."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .source_card_pools import source_pool_cards

PLAYABLE_CARD_POOLS = frozenset({"ironclad", "silent", "regent", "necrobinder", "defect"})


def source_card_pool_rows(
    rows: Iterable[Any],
    pool: str,
    *,
    unlocked_epochs: Iterable[str] | None = None,
    multiplayer: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    """Return registered, unlocked rows in registry order, without a fallback pool.

    Reward and merchant generation use GetUnlockedCards, rather than the stricter
    CardFactory.FilterForCombat predicate used for cards created during combat.
    Missing metadata stays missing; unrelated cached cards never fill an empty pool.
    """
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        raw_id = row.get("id", row.get("card_id", row.get("content_id", "")))
        card_id = str(raw_id).strip().lower().replace(" ", "_")
        if card_id:
            by_id.setdefault(card_id, row)
    return tuple(
        by_id[card_id]
        for card_id in source_pool_cards(
            pool, unlocked_epochs=unlocked_epochs, multiplayer=multiplayer
        )
        if card_id in by_id
    )
