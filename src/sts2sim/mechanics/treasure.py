"""Treasure chest reward generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import Any

from sts2sim.content.sources import PROVISIONAL_STS2_SOURCE, SourceRef

from .rewards import EncounterType, RelicRarity, RewardContext, roll_relic_rarity

TREASURE_RELIC_RARITIES = (
    RelicRarity.COMMON,
    RelicRarity.UNCOMMON,
    RelicRarity.RARE,
)


PoolItem = str | Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TreasureRelic:
    relic_id: str
    rarity: RelicRarity
    pool: str = "shared"
    name: str = ""
    source_id: str | None = None


@dataclass(frozen=True, slots=True)
class TreasureRules:
    gold_range: tuple[int, int] = (42, 52)
    poverty_ascension_level: int = 3
    poverty_gold_multiplier: float = 0.75
    empty_first_chest_relics: frozenset[str] = field(
        default_factory=lambda: frozenset({"silver_crucible"})
    )
    rarity_order: tuple[RelicRarity, ...] = TREASURE_RELIC_RARITIES
    source: SourceRef = PROVISIONAL_STS2_SOURCE


@dataclass(frozen=True, slots=True)
class TreasureContext:
    character_id: str
    act: int = 1
    floor: int = 1
    ascension_level: int = 0
    owned_relics: tuple[str, ...] = ()
    opened_chests: int = 0


@dataclass(frozen=True, slots=True)
class TreasureReward:
    relic_id: str | None
    relic_rarity: RelicRarity | None
    gold: int
    base_gold: int
    empty: bool = False
    empty_reason: str | None = None
    source_relic_id: str | None = None
    source: SourceRef = PROVISIONAL_STS2_SOURCE


DEFAULT_TREASURE_RULES = TreasureRules()


def build_treasure_relic_pool(
    raw_relics: Sequence[PoolItem],
    *,
    character_id: str,
) -> tuple[TreasureRelic, ...]:
    """Build the source-backed relic pool available to treasure chests."""

    character_pool = _normalized_id(character_id)
    relics: list[TreasureRelic] = []
    for raw_relic in raw_relics:
        relic = _treasure_relic_from_item(raw_relic)
        if relic is None:
            continue
        if _normalized_id(relic.pool) not in {"shared", character_pool}:
            continue
        relics.append(relic)
    return tuple(relics)


def draw_treasure_reward(
    rng: Random,
    relic_pool: Sequence[TreasureRelic],
    context: TreasureContext,
    *,
    rules: TreasureRules = DEFAULT_TREASURE_RULES,
) -> TreasureReward:
    """Generate the visible reward bundle for one treasure chest."""

    empty_source = _empty_chest_source(context, rules)
    if empty_source is not None:
        return TreasureReward(
            relic_id=None,
            relic_rarity=None,
            gold=0,
            base_gold=0,
            empty=True,
            empty_reason="first_chest_empty_relic",
            source_relic_id=empty_source,
            source=rules.source,
        )

    low, high = rules.gold_range
    if low > high:
        raise ValueError(f"Invalid treasure gold range: {rules.gold_range}")
    base_gold = rng.randint(low, high)
    gold = _apply_poverty_gold(base_gold, context, rules)

    rarity_roll = roll_relic_rarity(
        rng,
        RewardContext(
            encounter=EncounterType.CHEST,
            act=context.act,
            floor=context.floor,
            ascension_level=context.ascension_level,
        ),
    )
    relic = _choose_relic_by_rarity(
        rng,
        relic_pool,
        rarity_roll.rarity,
        owned_relics=context.owned_relics,
        rules=rules,
    )
    if relic is None:
        return TreasureReward(
            relic_id=None,
            relic_rarity=None,
            gold=gold,
            base_gold=base_gold,
            empty=False,
            empty_reason="relic_pool_exhausted",
            source=rules.source,
        )
    return TreasureReward(
        relic_id=relic.relic_id,
        relic_rarity=relic.rarity,
        gold=gold,
        base_gold=base_gold,
        source_relic_id=relic.source_id,
        source=rules.source,
    )


def _empty_chest_source(
    context: TreasureContext,
    rules: TreasureRules,
) -> str | None:
    if context.opened_chests > 0:
        return None
    owned = {_normalized_id(relic_id) for relic_id in context.owned_relics}
    for relic_id in rules.empty_first_chest_relics:
        if _normalized_id(relic_id) in owned:
            return relic_id
    return None


def _apply_poverty_gold(
    base_gold: int,
    context: TreasureContext,
    rules: TreasureRules,
) -> int:
    if context.ascension_level >= rules.poverty_ascension_level:
        return int(base_gold * rules.poverty_gold_multiplier)
    return base_gold


def _choose_relic_by_rarity(
    rng: Random,
    relic_pool: Sequence[TreasureRelic],
    rarity: RelicRarity,
    *,
    owned_relics: Sequence[str],
    rules: TreasureRules,
) -> TreasureRelic | None:
    owned = {_normalized_id(relic_id) for relic_id in owned_relics}
    available = [
        relic for relic in relic_pool if _normalized_id(relic.relic_id) not in owned
    ]
    for candidate_rarity in _rarity_fallback_order(rarity, rules.rarity_order):
        candidates = [relic for relic in available if relic.rarity is candidate_rarity]
        if candidates:
            return rng.choice(candidates)
    return None


def _rarity_fallback_order(
    rarity: RelicRarity,
    rarity_order: Sequence[RelicRarity],
) -> tuple[RelicRarity, ...]:
    if not rarity_order:
        return ()
    if rarity not in rarity_order:
        return tuple(rarity_order)
    start_index = tuple(rarity_order).index(rarity)
    ordered = tuple(rarity_order)
    return ordered[start_index:] + ordered[:start_index]


def _treasure_relic_from_item(raw_item: PoolItem) -> TreasureRelic | None:
    if isinstance(raw_item, str):
        return TreasureRelic(
            relic_id=_normalized_id(raw_item),
            rarity=RelicRarity.COMMON,
            source_id=raw_item,
        )

    relic_id = _first_present(raw_item, "id", "relic_id", "content_id")
    if relic_id is None:
        return None
    rarity = _treasure_rarity_from_value(
        _first_present(raw_item, "rarity_key", "rarity", "tier")
    )
    if rarity is None:
        return None

    pool = str(raw_item.get("pool", "shared"))
    name = str(raw_item.get("name", relic_id))
    return TreasureRelic(
        relic_id=_normalized_id(str(relic_id)),
        rarity=rarity,
        pool=pool,
        name=name,
        source_id=str(relic_id),
    )


def _treasure_rarity_from_value(value: Any) -> RelicRarity | None:
    normalized = _normalized_id(str(value or "common"))
    if "uncommon" in normalized:
        return RelicRarity.UNCOMMON
    if "common" in normalized:
        return RelicRarity.COMMON
    if "rare" in normalized:
        return RelicRarity.RARE
    return None


def _first_present(source: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = source.get(key)
        if value is not None:
            return value
    return None


def _normalized_id(value: str) -> str:
    return value.lower().replace("'", "").replace(" ", "_").replace("-", "_")
