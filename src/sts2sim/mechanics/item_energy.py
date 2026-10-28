"""Stable 0.107.1 relic ModifyMaxEnergy contracts.

Independent semantics from the twelve relic overrides in the pinned source
zhiyue/sts2-rl-agent@5d874e855ae1d0550d5c30416fadd7dbb0d133b7.
Current-turn Energy gains such as Lantern, Seal of Gold and Candelabra remain
separate from maximum Energy modifiers.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

_ALWAYS_ONE = frozenset({
    "blood_soaked_rose", "blessed_antler", "ectoplasm", "sozu",
    "philosophers_stone", "prismatic_gem", "spiked_gauntlets", "velvet_choker",
    "whispering_earring",
})
SOURCE_MAX_ENERGY_RELIC_IDS = _ALWAYS_ONE | {"bread", "paels_flesh", "pumpkin_candle"}


def source_max_energy_bonus(
    relics: Sequence[str], *, turn: int, counters: Mapping[str, int],
) -> int:
    owned = {str(item).lower() for item in relics}
    bonus = len(owned & _ALWAYS_ONE)
    bonus += int("bread" in owned and turn > 1)
    bonus += int("paels_flesh" in owned and turn >= 3)
    bonus += int("pumpkin_candle" in owned and counters.get("pumpkin_candle", 5) > 0)
    return bonus
