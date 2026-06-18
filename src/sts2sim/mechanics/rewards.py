"""Reward rarity, pity, gold, and relic helpers.

The default numbers are intentionally table-driven and source-tagged.  Replace
the model objects with extracted STS2 source values as they become available.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from random import Random

from sts2sim.content.sources import STS1_COMPAT_SOURCE, SourceRef

from .ascension import AscensionFlag, ascension_enabled


class EncounterType(str, Enum):
    NORMAL = "normal"
    ELITE = "elite"
    BOSS = "boss"
    EVENT = "event"
    CHEST = "chest"


class CardRarity(str, Enum):
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"


class PotionRarity(str, Enum):
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"


class RelicRarity(str, Enum):
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    BOSS = "boss"
    SHOP = "shop"
    SPECIAL = "special"


@dataclass(frozen=True, slots=True)
class RewardContext:
    encounter: EncounterType = EncounterType.NORMAL
    act: int = 1
    floor: int = 1
    ascension_level: int = 0


@dataclass(frozen=True, slots=True)
class RewardPityState:
    card_non_rare_count: int = 0
    potion_chance_bonus: int = 0


@dataclass(frozen=True, slots=True)
class RarityRoll[T: Enum]:
    rarity: T
    weights: Mapping[T, int]
    probabilities: Mapping[T, float]
    state: RewardPityState
    source: SourceRef


@dataclass(frozen=True, slots=True)
class PotionDropRoll:
    dropped: bool
    chance_percent: int
    roll: int
    state: RewardPityState
    source: SourceRef


@dataclass(frozen=True, slots=True)
class GoldReward:
    amount: int
    base_amount: int
    source: SourceRef


def _default_card_weights() -> dict[CardRarity, int]:
    return {
        CardRarity.COMMON: 60,
        CardRarity.UNCOMMON: 37,
        CardRarity.RARE: 3,
    }


def _default_relic_weights() -> dict[RelicRarity, int]:
    return {
        RelicRarity.COMMON: 50,
        RelicRarity.UNCOMMON: 33,
        RelicRarity.RARE: 17,
    }


def _default_gold_ranges() -> dict[EncounterType, tuple[int, int]]:
    return {
        EncounterType.NORMAL: (10, 20),
        EncounterType.ELITE: (35, 45),
        EncounterType.BOSS: (100, 100),
        EncounterType.EVENT: (0, 0),
        EncounterType.CHEST: (42, 52),
    }


@dataclass(frozen=True, slots=True)
class CardRarityModel:
    base_weights: Mapping[CardRarity, int] = field(default_factory=_default_card_weights)
    rare_pity_per_non_rare: int = 1
    rare_pity_cap: int = 40
    elite_rare_bonus: int = 10
    boss_rare_bonus: int = 20
    source: SourceRef = STS1_COMPAT_SOURCE


@dataclass(frozen=True, slots=True)
class PotionDropModel:
    base_chance_percent: int = 40
    miss_pity_step: int = 10
    hit_pity_step: int = 10
    min_bonus: int = -20
    max_bonus: int = 60
    source: SourceRef = STS1_COMPAT_SOURCE


@dataclass(frozen=True, slots=True)
class GoldRewardModel:
    ranges: Mapping[EncounterType, tuple[int, int]] = field(default_factory=_default_gold_ranges)
    poorer_boss_reward_multiplier: float = 0.85
    source: SourceRef = STS1_COMPAT_SOURCE


@dataclass(frozen=True, slots=True)
class RelicRarityModel:
    weights: Mapping[RelicRarity, int] = field(default_factory=_default_relic_weights)
    elite_rare_bonus: int = 5
    source: SourceRef = STS1_COMPAT_SOURCE


DEFAULT_CARD_RARITY_MODEL = CardRarityModel()
DEFAULT_POTION_DROP_MODEL = PotionDropModel()
DEFAULT_GOLD_REWARD_MODEL = GoldRewardModel()
DEFAULT_RELIC_RARITY_MODEL = RelicRarityModel()


def normalize_weights[T: Enum](weights: Mapping[T, int]) -> dict[T, float]:
    total = sum(max(0, weight) for weight in weights.values())
    if total <= 0:
        raise ValueError("At least one weight must be positive.")
    return {key: max(0, weight) / total for key, weight in weights.items()}


def weighted_choice[T: Enum](rng: Random, weights: Mapping[T, int]) -> T:
    total = sum(max(0, weight) for weight in weights.values())
    if total <= 0:
        raise ValueError("At least one weight must be positive.")
    roll = rng.randrange(total)
    cursor = 0
    for key, weight in weights.items():
        cursor += max(0, weight)
        if roll < cursor:
            return key
    return next(reversed(tuple(weights)))


def card_rarity_weights(
    state: RewardPityState,
    context: RewardContext,
    *,
    model: CardRarityModel = DEFAULT_CARD_RARITY_MODEL,
) -> dict[CardRarity, int]:
    weights = {rarity: max(0, weight) for rarity, weight in model.base_weights.items()}
    pity_bonus = min(model.rare_pity_cap, state.card_non_rare_count * model.rare_pity_per_non_rare)
    weights[CardRarity.RARE] = weights.get(CardRarity.RARE, 0) + pity_bonus
    if context.encounter is EncounterType.ELITE:
        weights[CardRarity.RARE] += model.elite_rare_bonus
    elif context.encounter is EncounterType.BOSS:
        weights[CardRarity.RARE] += model.boss_rare_bonus
    return weights


def roll_card_rarity(
    rng: Random,
    state: RewardPityState,
    context: RewardContext,
    *,
    model: CardRarityModel = DEFAULT_CARD_RARITY_MODEL,
) -> RarityRoll[CardRarity]:
    weights = card_rarity_weights(state, context, model=model)
    rarity = weighted_choice(rng, weights)
    if rarity is CardRarity.RARE:
        next_state = replace(state, card_non_rare_count=0)
    else:
        next_state = replace(state, card_non_rare_count=state.card_non_rare_count + 1)
    return RarityRoll(
        rarity=rarity,
        weights=weights,
        probabilities=normalize_weights(weights),
        state=next_state,
        source=model.source,
    )


def potion_drop_chance(
    state: RewardPityState,
    *,
    model: PotionDropModel = DEFAULT_POTION_DROP_MODEL,
) -> int:
    chance = model.base_chance_percent + state.potion_chance_bonus
    return min(100, max(0, chance))


def roll_potion_drop(
    rng: Random,
    state: RewardPityState,
    *,
    model: PotionDropModel = DEFAULT_POTION_DROP_MODEL,
) -> PotionDropRoll:
    chance = potion_drop_chance(state, model=model)
    roll = rng.randrange(100)
    dropped = roll < chance
    if dropped:
        bonus = max(model.min_bonus, state.potion_chance_bonus - model.hit_pity_step)
    else:
        bonus = min(model.max_bonus, state.potion_chance_bonus + model.miss_pity_step)
    return PotionDropRoll(
        dropped=dropped,
        chance_percent=chance,
        roll=roll,
        state=replace(state, potion_chance_bonus=bonus),
        source=model.source,
    )


def potion_slots_for_ascension(base_slots: int, ascension_level: int) -> int:
    if ascension_enabled(ascension_level, AscensionFlag.FEWER_POTION_SLOTS):
        return max(0, base_slots - 1)
    return base_slots


def roll_gold_reward(
    rng: Random,
    context: RewardContext,
    *,
    model: GoldRewardModel = DEFAULT_GOLD_REWARD_MODEL,
) -> GoldReward:
    low, high = model.ranges.get(context.encounter, (0, 0))
    if low > high:
        raise ValueError(f"Invalid gold range for {context.encounter.value}: {(low, high)}")
    base_amount = rng.randint(low, high)
    amount = base_amount
    if (
        context.encounter is EncounterType.BOSS
        and ascension_enabled(context.ascension_level, AscensionFlag.POORER_BOSS_REWARDS)
    ):
        amount = int(amount * model.poorer_boss_reward_multiplier)
    return GoldReward(amount=amount, base_amount=base_amount, source=model.source)


def relic_rarity_weights(
    context: RewardContext,
    *,
    model: RelicRarityModel = DEFAULT_RELIC_RARITY_MODEL,
) -> dict[RelicRarity, int]:
    if context.encounter is EncounterType.BOSS:
        return {RelicRarity.BOSS: 1}
    weights = {rarity: max(0, weight) for rarity, weight in model.weights.items()}
    if context.encounter is EncounterType.ELITE:
        weights[RelicRarity.RARE] = weights.get(RelicRarity.RARE, 0) + model.elite_rare_bonus
    return weights


def roll_relic_rarity(
    rng: Random,
    context: RewardContext,
    *,
    model: RelicRarityModel = DEFAULT_RELIC_RARITY_MODEL,
) -> RarityRoll[RelicRarity]:
    weights = relic_rarity_weights(context, model=model)
    rarity = weighted_choice(rng, weights)
    state = RewardPityState()
    return RarityRoll(
        rarity=rarity,
        weights=weights,
        probabilities=normalize_weights(weights),
        state=state,
        source=model.source,
    )
