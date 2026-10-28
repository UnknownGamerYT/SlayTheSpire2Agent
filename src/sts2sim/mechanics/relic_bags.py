"""Stable v0.107.1 relic deque pulls and source eligibility.

Source: RelicGrabBag.cs, RelicModel.IsBeforeAct3TreasureChest and the 19
IsAllowed overrides in the pinned decompilation in docs/run_generation_stable.md.
No draw consumes RNG: deques were shuffled once during run initialization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RelicBagContext:
    total_floor: int = 0
    player_count: int = 1
    first_run_ironclad_present: bool = False


@dataclass(frozen=True, slots=True)
class RelicBagDraw:
    relic_id: str | None
    bag: Mapping[str, tuple[str, ...]]
    multiplayer_fallback: tuple[str, ...] = ()


LATE_EXCLUDED_RELIC_IDS = frozenset(
    {
        "AMETHYST_AUBERGINE",
        "BOOK_OF_FIVE_RINGS",
        "BOWLER_HAT",
        "DRAGON_FRUIT",
        "FROZEN_EGG",
        "GIRYA",
        "JUZU_BRACELET",
        "LASTING_CANDY",
        "LUCKY_FYSH",
        "MEAL_TICKET",
        "MOLTEN_EGG",
        "OLD_COIN",
        "PLANISPHERE",
        "SHOVEL",
        "TOXIC_EGG",
        "WHITE_BEAST_STATUE",
        "WHITE_STAR",
    }
)
SHOP_EXCLUDED_RELIC_IDS = frozenset(
    {
        "AMETHYST_AUBERGINE",
        "BOWLER_HAT",
        "LUCKY_FYSH",
        "OLD_COIN",
        "THE_COURIER",
    }
)


def relic_is_allowed(relic_id: str, context: RelicBagContext) -> bool:
    relic_id = relic_id.upper()
    if relic_id in LATE_EXCLUDED_RELIC_IDS:
        threshold = 38 if context.player_count > 1 else 41
        if context.total_floor >= threshold:
            return False
    if relic_id == "LASTING_CANDY" and context.first_run_ironclad_present:
        return False
    if relic_id == "MASSIVE_SCROLL":
        return context.player_count > 1
    if relic_id in {"SILVER_CRUCIBLE", "WINGED_BOOTS"}:
        return context.player_count == 1
    return True


def remove_bag_relics(
    bag: Mapping[str, Sequence[str]],
    relic_ids: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """RelicGrabBag.Remove: remove every occurrence across all rarities."""
    removed = {value.upper() for value in relic_ids}
    return {
        rarity: tuple(value for value in values if value.upper() not in removed)
        for rarity, values in bag.items()
    }


def pull_relic_from_bag(
    bag: Mapping[str, Sequence[str]],
    rarity: str,
    *,
    context: RelicBagContext,
    from_back: bool = False,
    for_shop: bool = False,
    excluded_ids: Sequence[str] = (),
    refresh_originals: Mapping[str, Sequence[str]] | None = None,
    multiplayer_fallback: Sequence[str] = (),
) -> RelicBagDraw:
    """Pull front for rewards/back for shops, with source rarity fallback.

    IsAllowed permanently removes items from every deque before a pull. Shop
    and caller filters leave rejected items available to other reward sources.
    Pass refresh_originals only for the source's refresh-enabled shared bag.
    The caller decides its source-specific fallback (usually Circlet) on None.
    """
    cleaned = {
        key: [value for value in values if relic_is_allowed(value, context)]
        for key, values in bag.items()
    }
    excluded = {value.upper() for value in excluded_ids}
    if for_shop:
        excluded.update(SHOP_EXCLUDED_RELIC_IDS)

    def passes(value: str) -> bool:
        return value.upper() not in excluded

    rarity = rarity.removesuffix(" Relic").capitalize()
    if not cleaned.get(rarity) and refresh_originals is not None:
        cleaned[rarity] = [
            value for value in refresh_originals.get(rarity, ()) if relic_is_allowed(value, context)
        ]
    selected: str | None = None
    while rarity:
        values = cleaned.get(rarity, [])
        indices = range(len(values) - 1, -1, -1) if from_back else range(len(values))
        index = next((index for index in indices if passes(values[index])), None)
        if index is not None:
            selected = values.pop(index)
            break
        rarity = {"Shop": "Common", "Common": "Uncommon", "Uncommon": "Rare"}.get(rarity, "")
    fallback = [value for value in multiplayer_fallback if relic_is_allowed(value, context)]
    if selected is None:
        indices = range(len(fallback) - 1, -1, -1) if from_back else range(len(fallback))
        index = next((index for index in indices if passes(fallback[index])), None)
        if index is not None:
            selected = fallback.pop(index)
    return RelicBagDraw(
        selected, {key: tuple(values) for key, values in cleaned.items()}, tuple(fallback)
    )
