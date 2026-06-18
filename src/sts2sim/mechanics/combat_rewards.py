"""Source-backed combat reward generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from random import Random
from typing import Any

from sts2sim.content.sources import PROVISIONAL_STS2_SOURCE, SourceRef

from .rewards import (
    CardRarity,
    EncounterType,
    RelicRarity,
    RewardPityState,
    weighted_choice,
)
from .treasure import TreasureRelic

PoolItem = str | Mapping[str, Any]

COMBAT_CARD_RARITY_WEIGHTS: dict[EncounterType, dict[CardRarity, int]] = {
    EncounterType.NORMAL: {
        CardRarity.COMMON: 6000,
        CardRarity.UNCOMMON: 3700,
        CardRarity.RARE: 300,
    },
    EncounterType.ELITE: {
        CardRarity.COMMON: 5000,
        CardRarity.UNCOMMON: 4000,
        CardRarity.RARE: 1000,
    },
}
COMBAT_CARD_RARITY_WEIGHTS_A7: dict[EncounterType, dict[CardRarity, int]] = {
    EncounterType.NORMAL: {
        CardRarity.COMMON: 6151,
        CardRarity.UNCOMMON: 3700,
        CardRarity.RARE: 149,
    },
    EncounterType.ELITE: {
        CardRarity.COMMON: 5500,
        CardRarity.UNCOMMON: 4000,
        CardRarity.RARE: 500,
    },
}
COMBAT_RELIC_RARITY_WEIGHTS: dict[RelicRarity, int] = {
    RelicRarity.COMMON: 50,
    RelicRarity.UNCOMMON: 33,
    RelicRarity.RARE: 17,
}
COMBAT_RELIC_RARITY_ORDER = (
    RelicRarity.COMMON,
    RelicRarity.UNCOMMON,
    RelicRarity.RARE,
)
FAKE_MERCHANT_RUG_ID = "fake_merchants_rug"
FALLBACK_FAKE_MERCHANT_RELIC_IDS = (
    "fake_anchor",
    "fake_blood_vial",
    "fake_happy_flower",
    "fake_lees_waffle",
    "fake_mango",
    "fake_orichalcum",
    "fake_snecko_eye",
    "fake_strike_dummy",
    "fake_venerable_tea_set",
)


@dataclass(frozen=True, slots=True)
class CombatRewardCard:
    card_id: str
    rarity: CardRarity
    color: str = "shared"
    name: str = ""
    source_id: str | None = None


@dataclass(frozen=True, slots=True)
class CombatRewardRelic:
    relic_id: str
    rarity: str
    pool: str = "shared"
    name: str = ""
    source_id: str | None = None


@dataclass(frozen=True, slots=True)
class CombatPotionDrop:
    dropped: bool
    base_chance_percent: int
    effective_chance_tenths: int
    roll_tenths: int
    elite_bonus_tenths: int = 0
    state: RewardPityState = RewardPityState()


@dataclass(frozen=True, slots=True)
class CombatRewardRules:
    gold_ranges: Mapping[EncounterType, tuple[int, int]] = field(
        default_factory=lambda: {
            EncounterType.NORMAL: (10, 20),
            EncounterType.ELITE: (35, 45),
            EncounterType.BOSS: (100, 100),
            EncounterType.EVENT: (0, 0),
        }
    )
    default_card_choices: int = 3
    poverty_ascension_level: int = 3
    poverty_gold_multiplier: float = 0.75
    scarcity_ascension_level: int = 7
    rare_pity_growth_basis: int = 100
    scarcity_rare_pity_growth_basis: int = 50
    max_rare_pity_basis: int = 4000
    elite_potion_bonus_tenths: int = 125
    source: SourceRef = PROVISIONAL_STS2_SOURCE


@dataclass(frozen=True, slots=True)
class CombatRewardContext:
    character_id: str
    encounter: EncounterType = EncounterType.NORMAL
    act: int = 1
    floor: int = 1
    ascension_level: int = 0
    owned_relics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CombatRewardBundle:
    gold: int = 0
    base_gold: int = 0
    card_ids: tuple[str, ...] = ()
    card_rarities: tuple[str, ...] = ()
    relic_ids: tuple[str, ...] = ()
    relic_rarities: tuple[str, ...] = ()
    potion_id: str | None = None
    potion_roll: CombatPotionDrop | None = None
    pity_state: RewardPityState = RewardPityState()
    source: SourceRef = PROVISIONAL_STS2_SOURCE


DEFAULT_COMBAT_REWARD_RULES = CombatRewardRules()


def build_combat_card_pool(
    raw_cards: Sequence[PoolItem],
    *,
    character_id: str,
) -> tuple[CombatRewardCard, ...]:
    """Build the combat card reward pool for the active character."""

    character_pool = _normalized_id(character_id)
    cards: list[CombatRewardCard] = []
    for raw_card in raw_cards:
        card = _combat_card_from_item(raw_card)
        if card is None:
            continue
        cards.append(card)

    filtered = [
        card
        for card in cards
        if card.color in {"shared", character_pool}
        or (card.color == "" and character_pool == "test")
    ]
    return tuple(filtered or cards)


def build_combat_potion_pool(
    raw_potions: Sequence[PoolItem],
    *,
    character_id: str,
) -> tuple[str, ...]:
    """Build the combat potion reward pool for the active character."""

    character_pool = _normalized_id(character_id)
    potions: list[tuple[str, str]] = []
    for raw_potion in raw_potions:
        if isinstance(raw_potion, str):
            potions.append((_normalized_id(raw_potion), "shared"))
            continue
        potion_id = _first_present(raw_potion, "id", "potion_id", "content_id")
        if potion_id is None:
            continue
        pool = _normalized_id(str(raw_potion.get("pool", "shared")))
        potions.append((_normalized_id(str(potion_id)), pool))

    filtered = [
        potion_id
        for potion_id, pool in potions
        if pool in {"shared", character_pool}
    ]
    return tuple(filtered or [potion_id for potion_id, _pool in potions])


def build_boss_relic_pool(
    raw_relics: Sequence[PoolItem],
    *,
    character_id: str,
) -> tuple[CombatRewardRelic, ...]:
    """Build the boss/ancient relic pool used by boss rewards."""

    character_pool = _normalized_id(character_id)
    relics: list[CombatRewardRelic] = []
    for raw_relic in raw_relics:
        relic = _combat_relic_from_item(raw_relic)
        if relic is None:
            continue
        if relic.rarity not in {"boss", "ancient"}:
            continue
        relics.append(relic)

    filtered = [
        relic
        for relic in relics
        if _normalized_id(relic.pool) in {"shared", character_pool}
    ]
    return tuple(filtered or relics)


def fake_merchant_reward_relic_ids(
    raw_relics: Sequence[PoolItem] = (),
    *,
    unsold_relic_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return Fake Merchant win rewards: rug plus all unsold fake relics."""

    if unsold_relic_ids:
        unsold = tuple(
            relic_id
            for relic_id in (_normalized_id(str(item)) for item in unsold_relic_ids)
            if relic_id != FAKE_MERCHANT_RUG_ID
        )
    else:
        unsold = tuple(_fake_relic_ids_from_source(raw_relics))
        if not unsold:
            unsold = FALLBACK_FAKE_MERCHANT_RELIC_IDS
    return tuple(dict.fromkeys((FAKE_MERCHANT_RUG_ID,) + unsold))


def draw_combat_reward(
    rng: Random,
    *,
    card_pool: Sequence[CombatRewardCard],
    potion_pool: Sequence[str],
    relic_pool: Sequence[TreasureRelic],
    boss_relic_pool: Sequence[CombatRewardRelic],
    context: CombatRewardContext,
    pity_state: RewardPityState = RewardPityState(),
    card_count: int | None = None,
    relic_count: int | None = None,
    rules: CombatRewardRules = DEFAULT_COMBAT_REWARD_RULES,
) -> CombatRewardBundle:
    """Generate the default reward bundle for a completed combat."""

    base_gold = _roll_gold(rng, context, rules)
    gold = _apply_poverty_gold(base_gold, context, rules)
    cards, card_rarities, next_pity = _draw_card_options(
        rng,
        card_pool,
        context,
        pity_state,
        count=rules.default_card_choices if card_count is None else card_count,
        rules=rules,
    )
    potion_roll = roll_combat_potion_drop(rng, next_pity, context, rules=rules)
    potion_id = rng.choice(tuple(potion_pool)) if potion_roll.dropped and potion_pool else None

    drawn_relics, relic_rarities = _draw_relics(
        rng,
        relic_pool,
        boss_relic_pool,
        context,
        relic_count=relic_count,
    )
    return CombatRewardBundle(
        gold=gold,
        base_gold=base_gold,
        card_ids=cards,
        card_rarities=card_rarities,
        relic_ids=drawn_relics,
        relic_rarities=relic_rarities,
        potion_id=potion_id,
        potion_roll=potion_roll,
        pity_state=potion_roll.state,
        source=rules.source,
    )


def roll_combat_potion_drop(
    rng: Random,
    state: RewardPityState,
    context: CombatRewardContext,
    *,
    rules: CombatRewardRules = DEFAULT_COMBAT_REWARD_RULES,
) -> CombatPotionDrop:
    """Roll a combat potion reward with elite bonus that does not move pity."""

    base_chance_percent = 40 + state.potion_chance_bonus
    base_chance_tenths = max(0, base_chance_percent * 10)
    elite_bonus_tenths = (
        rules.elite_potion_bonus_tenths
        if context.encounter is EncounterType.ELITE
        else 0
    )
    effective_chance_tenths = base_chance_tenths + elite_bonus_tenths
    roll_tenths = rng.randrange(1000)
    dropped = roll_tenths < effective_chance_tenths

    if roll_tenths < base_chance_tenths:
        next_bonus = state.potion_chance_bonus - 10
    elif dropped:
        next_bonus = state.potion_chance_bonus
    else:
        next_bonus = state.potion_chance_bonus + 10

    return CombatPotionDrop(
        dropped=dropped,
        base_chance_percent=base_chance_percent,
        effective_chance_tenths=effective_chance_tenths,
        roll_tenths=roll_tenths,
        elite_bonus_tenths=elite_bonus_tenths,
        state=replace(state, potion_chance_bonus=next_bonus),
    )


def _draw_card_options(
    rng: Random,
    card_pool: Sequence[CombatRewardCard],
    context: CombatRewardContext,
    pity_state: RewardPityState,
    *,
    count: int,
    rules: CombatRewardRules,
) -> tuple[tuple[str, ...], tuple[str, ...], RewardPityState]:
    selected: list[str] = []
    rarities: list[str] = []
    used: set[str] = set()
    current_pity = pity_state

    for _ in range(max(0, count)):
        rarity = _roll_card_rarity(rng, current_pity, context, rules)
        card = _choose_card_by_rarity(rng, card_pool, rarity=rarity, excluded=used)
        if card is None:
            break
        selected.append(card.card_id)
        rarities.append(card.rarity.value)
        used.add(card.card_id)
        current_pity = replace(
            current_pity,
            card_non_rare_count=0
            if card.rarity is CardRarity.RARE
            else current_pity.card_non_rare_count + 1,
        )

    return tuple(selected), tuple(rarities), current_pity


def _roll_card_rarity(
    rng: Random,
    pity_state: RewardPityState,
    context: CombatRewardContext,
    rules: CombatRewardRules,
) -> CardRarity:
    if context.encounter is EncounterType.BOSS:
        return CardRarity.RARE

    base_by_encounter = (
        COMBAT_CARD_RARITY_WEIGHTS_A7
        if context.ascension_level >= rules.scarcity_ascension_level
        else COMBAT_CARD_RARITY_WEIGHTS
    )
    weights = dict(
        base_by_encounter.get(
            context.encounter,
            base_by_encounter[EncounterType.NORMAL],
        )
    )
    growth = (
        rules.scarcity_rare_pity_growth_basis
        if context.ascension_level >= rules.scarcity_ascension_level
        else rules.rare_pity_growth_basis
    )
    rare_bonus = min(rules.max_rare_pity_basis, pity_state.card_non_rare_count * growth)
    weights[CardRarity.RARE] = weights.get(CardRarity.RARE, 0) + rare_bonus
    return weighted_choice(rng, weights)


def _choose_card_by_rarity(
    rng: Random,
    card_pool: Sequence[CombatRewardCard],
    *,
    rarity: CardRarity,
    excluded: set[str],
) -> CombatRewardCard | None:
    available = [card for card in card_pool if card.card_id not in excluded]
    candidates = [card for card in available if card.rarity is rarity]
    if not candidates:
        candidates = available
    return rng.choice(candidates) if candidates else None


def _draw_relics(
    rng: Random,
    relic_pool: Sequence[TreasureRelic],
    boss_relic_pool: Sequence[CombatRewardRelic],
    context: CombatRewardContext,
    *,
    relic_count: int | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if relic_count is None:
        relic_count = 1 if context.encounter in {EncounterType.ELITE, EncounterType.BOSS} else 0

    relic_ids: list[str] = []
    relic_rarities: list[str] = []
    excluded = {_normalized_id(relic_id) for relic_id in context.owned_relics}

    for _ in range(max(0, relic_count)):
        if context.encounter is EncounterType.BOSS:
            boss_relic = _choose_boss_relic(rng, boss_relic_pool, excluded=excluded)
            if boss_relic is None:
                break
            relic_ids.append(boss_relic.relic_id)
            relic_rarities.append(boss_relic.rarity)
            excluded.add(boss_relic.relic_id)
            continue

        rarity = weighted_choice(rng, COMBAT_RELIC_RARITY_WEIGHTS)
        random_relic = _choose_relic_by_rarity(
            rng,
            relic_pool,
            rarity,
            excluded=excluded,
        )
        if random_relic is None:
            break
        relic_ids.append(random_relic.relic_id)
        relic_rarities.append(random_relic.rarity.value)
        excluded.add(random_relic.relic_id)

    return tuple(relic_ids), tuple(relic_rarities)


def _choose_boss_relic(
    rng: Random,
    relic_pool: Sequence[CombatRewardRelic],
    *,
    excluded: set[str],
) -> CombatRewardRelic | None:
    candidates = [relic for relic in relic_pool if relic.relic_id not in excluded]
    return rng.choice(candidates) if candidates else None


def _choose_relic_by_rarity(
    rng: Random,
    relic_pool: Sequence[TreasureRelic],
    rarity: RelicRarity,
    *,
    excluded: set[str],
) -> TreasureRelic | None:
    available = [relic for relic in relic_pool if relic.relic_id not in excluded]
    for candidate_rarity in _rarity_fallback_order(rarity):
        candidates = [relic for relic in available if relic.rarity is candidate_rarity]
        if candidates:
            return rng.choice(candidates)
    return None


def _rarity_fallback_order(rarity: RelicRarity) -> tuple[RelicRarity, ...]:
    if rarity not in COMBAT_RELIC_RARITY_ORDER:
        return COMBAT_RELIC_RARITY_ORDER
    start_index = COMBAT_RELIC_RARITY_ORDER.index(rarity)
    return COMBAT_RELIC_RARITY_ORDER[start_index:] + COMBAT_RELIC_RARITY_ORDER[:start_index]


def _roll_gold(
    rng: Random,
    context: CombatRewardContext,
    rules: CombatRewardRules,
) -> int:
    low, high = rules.gold_ranges.get(context.encounter, (0, 0))
    if low > high:
        raise ValueError(f"Invalid combat gold range: {(low, high)}")
    return rng.randint(low, high)


def _apply_poverty_gold(
    base_gold: int,
    context: CombatRewardContext,
    rules: CombatRewardRules,
) -> int:
    if context.ascension_level >= rules.poverty_ascension_level:
        return int(base_gold * rules.poverty_gold_multiplier)
    return base_gold


def _combat_card_from_item(raw_item: PoolItem) -> CombatRewardCard | None:
    if isinstance(raw_item, str):
        return CombatRewardCard(
            card_id=_normalized_id(raw_item),
            rarity=CardRarity.COMMON,
            source_id=raw_item,
        )

    card_id = _first_present(raw_item, "id", "card_id", "content_id")
    if card_id is None:
        return None
    rarity = _card_rarity_from_value(_first_present(raw_item, "rarity_key", "rarity"))
    if rarity is None:
        return None
    color = _normalized_id(str(_first_present(raw_item, "color", "pool", "character") or "shared"))
    card_type = _normalized_id(str(_first_present(raw_item, "type_key", "type") or ""))
    if color in {"colorless", "event", "curse", "status", "token", "quest"}:
        return None
    if card_type in {"curse", "status"}:
        return None
    return CombatRewardCard(
        card_id=_normalized_id(str(card_id)),
        rarity=rarity,
        color=color,
        name=str(raw_item.get("name", card_id)),
        source_id=str(card_id),
    )


def _combat_relic_from_item(raw_item: PoolItem) -> CombatRewardRelic | None:
    if isinstance(raw_item, str):
        return CombatRewardRelic(
            relic_id=_normalized_id(raw_item),
            rarity="common",
            source_id=raw_item,
        )

    relic_id = _first_present(raw_item, "id", "relic_id", "content_id")
    if relic_id is None:
        return None
    rarity = _normalized_id(str(_first_present(raw_item, "rarity_key", "rarity") or "common"))
    return CombatRewardRelic(
        relic_id=_normalized_id(str(relic_id)),
        rarity=rarity,
        pool=str(raw_item.get("pool", "shared")),
        name=str(raw_item.get("name", relic_id)),
        source_id=str(relic_id),
    )


def _fake_relic_ids_from_source(raw_relics: Sequence[PoolItem]) -> tuple[str, ...]:
    relics: list[str] = []
    for raw_relic in raw_relics:
        if isinstance(raw_relic, str):
            relic_id = _normalized_id(raw_relic)
            name = relic_id
        else:
            relic_id = _normalized_id(str(_first_present(raw_relic, "id", "relic_id") or ""))
            name = str(raw_relic.get("name", ""))
        if not relic_id.startswith("fake_") or relic_id == FAKE_MERCHANT_RUG_ID:
            continue
        if "???" not in name and relic_id not in FALLBACK_FAKE_MERCHANT_RELIC_IDS:
            continue
        relics.append(relic_id)
    return tuple(dict.fromkeys(relics))


def _card_rarity_from_value(value: Any) -> CardRarity | None:
    normalized = _normalized_id(str(value or "common"))
    if "uncommon" in normalized:
        return CardRarity.UNCOMMON
    if "common" in normalized:
        return CardRarity.COMMON
    if "rare" in normalized:
        return CardRarity.RARE
    return None


def _first_present(source: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = source.get(key)
        if value is not None:
            return value
    return None


def _normalized_id(value: str) -> str:
    return value.lower().replace("'", "").replace(" ", "_").replace("-", "_")
