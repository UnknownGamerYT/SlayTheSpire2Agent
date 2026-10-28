"""Stable HealRestSiteOption and its source modifier amounts."""

from __future__ import annotations

from collections.abc import Iterable


def source_rest_heal_amount(
    max_hp: int, *, relics: Iterable[str] = (), modifiers: Iterable[str] = (),
) -> int:
    """Source base30%, Regal Pillow+15, then run modifiers; no Ascension penalty."""
    owned = {str(item).lower() for item in relics}
    active = {str(item).lower() for item in modifiers}
    # RunState.IterateHookListeners yields player relics before run modifiers.
    if "night_terrors" in active:
        return max_hp
    return max_hp * 3 // 10 + (15 if "regal_pillow" in owned else 0)
