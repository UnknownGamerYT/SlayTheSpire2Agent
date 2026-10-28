"""Stable CardFactory noncombat creation: base odds, ordered exclusion, upgrades."""
from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from random import Random

from .combat_rewards import CombatRewardCard
from .source_rng import float32, named_stream, source_float


def source_noncombat_cards(
    rng: Random, pool: Sequence[CombatRewardCard], count: int, *,
    uniform: bool = False, no_upgrade_roll: bool = True, act: int = 1,
    ascension: int = 0, item_rng: Random | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[bool, ...]]:
    """ForNonCombat* uses base odds without reading or changing the rarity offset.

    The source default-odds factory includes NoUpgradeRoll. Explicit custom
    options (for example Colorful Philosophers) consume upgrades separately.
    Rarity always uses Rewards; an override replaces item/upgrade draws only.
    """
    rarity_rng = named_stream(rng, "rewards")
    selection_rng = rarity_rng if item_rng is None else item_rng
    candidates = list(pool)
    cards: list[str] = []
    rarities: list[str] = []
    upgrades: list[bool] = []
    for _ in range(max(0, count)):
        if not candidates:
            break
        if uniform:
            eligible = [card for card in candidates
                        if card.rarity.value not in {"basic", "ancient"}]
        else:
            sample = source_float(rarity_rng)
            rarity = ("rare" if sample < float32(0.0149 if ascension >= 7 else 0.03)
                      else "uncommon" if sample < float32(0.37) else "common")
            eligible = []
            for _attempt in range(3):
                eligible = [card for card in candidates if card.rarity.value == rarity]
                if eligible:
                    break
                rarity = {"common": "uncommon", "uncommon": "rare", "rare": "common"}[rarity]
        if not eligible:
            break
        card = selection_rng.choice(eligible)
        cards.append(card.card_id)
        rarities.append(card.rarity.value)
        upgraded = False
        if not no_upgrade_roll:
            roll = Decimal(format(source_float(selection_rng), ".7g"))
            threshold = (Decimal(0) if card.rarity.value == "rare" else
                         Decimal(max(0, act - 1)) * Decimal(".125" if ascension >= 7 else ".25"))
            upgraded = roll <= threshold
        upgrades.append(upgraded)
        candidates = [value for value in candidates if value.card_id != card.card_id]
    return tuple(cards), tuple(rarities), tuple(upgrades)
