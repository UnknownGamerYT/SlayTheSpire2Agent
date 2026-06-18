"""Pure reward pool filtering and odds metadata helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sts2sim.content.sources import PROVISIONAL_STS2_SOURCE, STS1_COMPAT_SOURCE, SourceRef

from .combat_rewards import (
    COMBAT_CARD_RARITY_WEIGHTS,
    COMBAT_CARD_RARITY_WEIGHTS_A7,
    COMBAT_RELIC_RARITY_WEIGHTS,
    DEFAULT_COMBAT_REWARD_RULES,
    CombatRewardRules,
)
from .rewards import (
    CardRarity,
    EncounterType,
    PotionRarity,
    RelicRarity,
    RewardPityState,
)
from .shop_rooms import (
    SHOP_POTION_RARITY_WEIGHTS,
    SHOP_RELIC_RARITY_WEIGHTS,
    shop_card_rarity_weights,
)

PoolItem = str | Mapping[str, Any]

CARD_REWARD_EXCLUDED_COLORS = frozenset({"curse", "event", "quest", "status", "token"})
CARD_REWARD_EXCLUDED_TYPES = frozenset({"curse", "status"})
RELIC_REWARD_RARITIES = frozenset(
    {RelicRarity.COMMON, RelicRarity.UNCOMMON, RelicRarity.RARE}
)
BOSS_RELIC_SOURCE_RARITIES = frozenset({"boss", "ancient"})


@dataclass(frozen=True, slots=True)
class RewardPoolCard:
    card_id: str
    rarity: CardRarity
    color: str = "shared"
    card_type: str = ""
    name: str = ""
    source_id: str | None = None


@dataclass(frozen=True, slots=True)
class RewardPoolPotion:
    potion_id: str
    rarity: PotionRarity = PotionRarity.COMMON
    pool: str = "shared"
    name: str = ""
    source_id: str | None = None


@dataclass(frozen=True, slots=True)
class RewardPoolRelic:
    relic_id: str
    rarity: RelicRarity
    pool: str = "shared"
    source_rarity: str = "common"
    name: str = ""
    source_id: str | None = None


@dataclass(frozen=True, slots=True)
class CardRewardPools:
    character_cards: tuple[RewardPoolCard, ...]
    colorless_cards: tuple[RewardPoolCard, ...]


@dataclass(frozen=True, slots=True)
class RelicPoolFilterResult:
    available: tuple[RewardPoolRelic, ...]
    excluded_owned: tuple[str, ...] = ()
    excluded_ids: tuple[str, ...] = ()
    duplicate_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RaritySurface:
    kind: str
    weights: Mapping[str, int]
    probabilities: Mapping[str, float]
    source: SourceRef
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RewardShape:
    encounter: EncounterType
    gold_range: tuple[int, int]
    card_count: int
    relic_count: int
    card_rarity: RaritySurface
    relic_rarity: RaritySurface
    potion_drop_chance_percent: int
    potion_effective_chance_tenths: int
    source: SourceRef


def build_character_card_pool(
    raw_cards: Sequence[PoolItem],
    *,
    character_id: str,
    include_colorless: bool = False,
) -> tuple[RewardPoolCard, ...]:
    """Return cards available to the character card reward pool."""

    character_pool = _normalized_id(character_id)
    cards = tuple(
        card for raw_card in raw_cards if (card := _reward_card_from_item(raw_card)) is not None
    )
    filtered = [
        card
        for card in cards
        if _card_matches_character_pool(
            card,
            character_pool=character_pool,
            include_colorless=include_colorless,
        )
    ]
    return _unique_cards(filtered)


def build_colorless_card_pool(raw_cards: Sequence[PoolItem]) -> tuple[RewardPoolCard, ...]:
    """Return source cards that should be handled by colorless-card surfaces."""

    cards = tuple(
        card for raw_card in raw_cards if (card := _reward_card_from_item(raw_card)) is not None
    )
    return _unique_cards(card for card in cards if _is_colorless_card(card))


def split_card_reward_pools(
    raw_cards: Sequence[PoolItem],
    *,
    character_id: str,
) -> CardRewardPools:
    """Split source cards into character and colorless reward pools."""

    return CardRewardPools(
        character_cards=build_character_card_pool(raw_cards, character_id=character_id),
        colorless_cards=build_colorless_card_pool(raw_cards),
    )


def build_potion_reward_pool(
    raw_potions: Sequence[PoolItem],
    *,
    character_id: str,
) -> tuple[RewardPoolPotion, ...]:
    """Return potions available to the character, preserving source order."""

    character_pool = _normalized_id(character_id)
    potions = tuple(
        potion
        for raw_potion in raw_potions
        if (potion := _reward_potion_from_item(raw_potion)) is not None
    )
    filtered = [
        potion
        for potion in potions
        if _normalized_id(potion.pool) in {"shared", character_pool}
    ]
    return _unique_potions(filtered or potions)


def build_reward_relic_pool(
    raw_relics: Sequence[PoolItem],
    *,
    character_id: str,
    allowed_rarities: frozenset[RelicRarity] = RELIC_REWARD_RARITIES,
) -> tuple[RewardPoolRelic, ...]:
    """Return non-duplicate relics in the requested rarity groups."""

    character_pool = _normalized_id(character_id)
    relics = tuple(
        relic
        for raw_relic in raw_relics
        if (relic := _reward_relic_from_item(raw_relic)) is not None
    )
    filtered = [
        relic
        for relic in relics
        if relic.rarity in allowed_rarities
        and _normalized_id(relic.pool) in {"shared", character_pool}
    ]
    return _unique_relics(filtered)


def build_boss_reward_relic_pool(
    raw_relics: Sequence[PoolItem],
    *,
    character_id: str,
) -> tuple[RewardPoolRelic, ...]:
    """Return boss/ancient relics available to boss rewards."""

    return build_reward_relic_pool(
        raw_relics,
        character_id=character_id,
        allowed_rarities=frozenset({RelicRarity.BOSS}),
    )


def filter_available_relics(
    relic_pool: Sequence[RewardPoolRelic],
    *,
    owned_relics: Sequence[str] = (),
    excluded_relics: Sequence[str] = (),
) -> RelicPoolFilterResult:
    """Filter owned, explicitly excluded, and duplicate relic ids."""

    owned = {_normalized_id(relic_id) for relic_id in owned_relics}
    excluded = {_normalized_id(relic_id) for relic_id in excluded_relics}
    seen: set[str] = set()
    available: list[RewardPoolRelic] = []
    excluded_owned: list[str] = []
    excluded_ids: list[str] = []
    duplicate_ids: list[str] = []

    for relic in relic_pool:
        relic_id = _normalized_id(relic.relic_id)
        if relic_id in seen:
            duplicate_ids.append(relic_id)
            continue
        seen.add(relic_id)
        if relic_id in owned:
            excluded_owned.append(relic_id)
            continue
        if relic_id in excluded:
            excluded_ids.append(relic_id)
            continue
        available.append(relic)

    return RelicPoolFilterResult(
        available=tuple(available),
        excluded_owned=tuple(excluded_owned),
        excluded_ids=tuple(excluded_ids),
        duplicate_ids=tuple(duplicate_ids),
    )


def combat_card_rarity_surface(
    encounter: EncounterType,
    *,
    ascension_level: int = 0,
    pity_state: RewardPityState = RewardPityState(),
    rules: CombatRewardRules = DEFAULT_COMBAT_REWARD_RULES,
) -> RaritySurface:
    """Return combat card rarity weights after encounter, ascension, and pity."""

    if encounter is EncounterType.BOSS:
        return _rarity_surface(
            "combat_card",
            {"rare": 1},
            PROVISIONAL_STS2_SOURCE,
            metadata={
                "encounter": encounter.value,
                "forced_rare": True,
                "ascension_level": ascension_level,
            },
        )

    base_by_encounter = (
        COMBAT_CARD_RARITY_WEIGHTS_A7
        if ascension_level >= rules.scarcity_ascension_level
        else COMBAT_CARD_RARITY_WEIGHTS
    )
    weights = _enum_weight_mapping(
        base_by_encounter.get(encounter, base_by_encounter[EncounterType.NORMAL])
    )
    growth = (
        rules.scarcity_rare_pity_growth_basis
        if ascension_level >= rules.scarcity_ascension_level
        else rules.rare_pity_growth_basis
    )
    rare_bonus = min(rules.max_rare_pity_basis, pity_state.card_non_rare_count * growth)
    weights["rare"] = weights.get("rare", 0) + rare_bonus
    return _rarity_surface(
        "combat_card",
        weights,
        rules.source,
        metadata={
            "encounter": encounter.value,
            "ascension_level": ascension_level,
            "card_non_rare_count": pity_state.card_non_rare_count,
            "rare_pity_bonus_basis": rare_bonus,
            "rare_pity_growth_basis": growth,
        },
    )


def combat_relic_rarity_surface(
    encounter: EncounterType,
    *,
    rules: CombatRewardRules = DEFAULT_COMBAT_REWARD_RULES,
) -> RaritySurface:
    """Return combat relic rarity weights for the reward source."""

    if encounter is EncounterType.BOSS:
        return _rarity_surface(
            "combat_relic",
            {"boss": 1},
            rules.source,
            metadata={"encounter": encounter.value, "forced_boss_relic": True},
        )
    return _rarity_surface(
        "combat_relic",
        _enum_weight_mapping(COMBAT_RELIC_RARITY_WEIGHTS),
        rules.source,
        metadata={"encounter": encounter.value},
    )


def potion_drop_surface(
    encounter: EncounterType,
    *,
    pity_state: RewardPityState = RewardPityState(),
    rules: CombatRewardRules = DEFAULT_COMBAT_REWARD_RULES,
) -> RaritySurface:
    """Return the combat potion drop chance as a one-roll odds surface."""

    base_chance_percent = 40 + pity_state.potion_chance_bonus
    base_chance_tenths = max(0, base_chance_percent * 10)
    elite_bonus_tenths = (
        rules.elite_potion_bonus_tenths
        if encounter is EncounterType.ELITE
        else 0
    )
    effective_chance_tenths = base_chance_tenths + elite_bonus_tenths
    effective_chance_tenths = max(0, effective_chance_tenths)
    return _rarity_surface(
        "combat_potion_drop",
        {
            "drop": min(1000, effective_chance_tenths),
            "miss": max(0, 1000 - min(1000, effective_chance_tenths)),
        },
        rules.source,
        metadata={
            "encounter": encounter.value,
            "base_chance_percent": base_chance_percent,
            "effective_chance_tenths": effective_chance_tenths,
            "elite_bonus_tenths": elite_bonus_tenths,
            "potion_chance_bonus": pity_state.potion_chance_bonus,
        },
    )


def shop_card_rarity_surface(
    *,
    ascension_level: int = 0,
    rare_offset_percent: float = 0.0,
) -> RaritySurface:
    """Return shop card rarity weights, including A7 scarcity tuning."""

    return _rarity_surface(
        "shop_card",
        _enum_weight_mapping(
            shop_card_rarity_weights(
                ascension_level=ascension_level,
                rare_offset_percent=rare_offset_percent,
            )
        ),
        STS1_COMPAT_SOURCE,
        metadata={
            "ascension_level": ascension_level,
            "rare_offset_percent": rare_offset_percent,
        },
    )


def shop_relic_rarity_surface() -> RaritySurface:
    """Return the weighted normal relic rarity surface used by shops."""

    return _rarity_surface(
        "shop_relic",
        _enum_weight_mapping(SHOP_RELIC_RARITY_WEIGHTS),
        STS1_COMPAT_SOURCE,
        metadata={"shop_relics_are_filtered_before_weighting": True},
    )


def shop_potion_rarity_surface() -> RaritySurface:
    """Return the weighted potion rarity surface used by shops."""

    return _rarity_surface(
        "shop_potion",
        _enum_weight_mapping(SHOP_POTION_RARITY_WEIGHTS),
        STS1_COMPAT_SOURCE,
    )


def combat_reward_shape(
    encounter: EncounterType,
    *,
    ascension_level: int = 0,
    pity_state: RewardPityState = RewardPityState(),
    card_count: int | None = None,
    relic_count: int | None = None,
    rules: CombatRewardRules = DEFAULT_COMBAT_REWARD_RULES,
) -> RewardShape:
    """Return the default visible reward shape for a combat encounter."""

    default_card_count = 0 if encounter is EncounterType.EVENT else rules.default_card_choices
    default_relic_count = 1 if encounter in {EncounterType.ELITE, EncounterType.BOSS} else 0
    potion_surface = potion_drop_surface(encounter, pity_state=pity_state, rules=rules)
    return RewardShape(
        encounter=encounter,
        gold_range=rules.gold_ranges.get(encounter, (0, 0)),
        card_count=default_card_count if card_count is None else max(0, card_count),
        relic_count=default_relic_count if relic_count is None else max(0, relic_count),
        card_rarity=combat_card_rarity_surface(
            encounter,
            ascension_level=ascension_level,
            pity_state=pity_state,
            rules=rules,
        ),
        relic_rarity=combat_relic_rarity_surface(encounter, rules=rules),
        potion_drop_chance_percent=int(
            potion_surface.metadata.get("base_chance_percent", 0)
        ),
        potion_effective_chance_tenths=int(
            potion_surface.metadata.get("effective_chance_tenths", 0)
        ),
        source=rules.source,
    )


def boss_reward_shape(
    *,
    ascension_level: int = 0,
    pity_state: RewardPityState = RewardPityState(),
    rules: CombatRewardRules = DEFAULT_COMBAT_REWARD_RULES,
) -> RewardShape:
    """Return the default boss reward shape."""

    return combat_reward_shape(
        EncounterType.BOSS,
        ascension_level=ascension_level,
        pity_state=pity_state,
        rules=rules,
    )


def elite_reward_shape(
    *,
    ascension_level: int = 0,
    pity_state: RewardPityState = RewardPityState(),
    rules: CombatRewardRules = DEFAULT_COMBAT_REWARD_RULES,
) -> RewardShape:
    """Return the default elite reward shape."""

    return combat_reward_shape(
        EncounterType.ELITE,
        ascension_level=ascension_level,
        pity_state=pity_state,
        rules=rules,
    )


def _card_matches_character_pool(
    card: RewardPoolCard,
    *,
    character_pool: str,
    include_colorless: bool,
) -> bool:
    if _is_colorless_card(card):
        return include_colorless
    if card.color in {"shared", character_pool}:
        return True
    return card.color == "" and character_pool == "test"


def _is_colorless_card(card: RewardPoolCard) -> bool:
    return card.color == "colorless" or card.card_type == "colorless_card"


def _reward_card_from_item(raw_item: PoolItem) -> RewardPoolCard | None:
    if isinstance(raw_item, str):
        return RewardPoolCard(
            card_id=_normalized_id(raw_item),
            rarity=CardRarity.COMMON,
            source_id=raw_item,
        )

    card_id = _first_present(raw_item, "id", "card_id", "content_id")
    if card_id is None:
        return None
    rarity = _card_rarity_from_value(_first_present(raw_item, "rarity_key", "rarity", "tier"))
    if rarity is None:
        return None
    color = _normalized_id(str(_first_present(raw_item, "color", "pool", "character") or "shared"))
    card_type = _normalized_id(
        str(_first_present(raw_item, "type_key", "type", "card_type", "kind") or "")
    )
    if color in CARD_REWARD_EXCLUDED_COLORS or card_type in CARD_REWARD_EXCLUDED_TYPES:
        return None
    if card_type == "colorless_card":
        color = "colorless"
    return RewardPoolCard(
        card_id=_normalized_id(str(card_id)),
        rarity=rarity,
        color=color,
        card_type=card_type,
        name=str(raw_item.get("name", card_id)),
        source_id=str(card_id),
    )


def _reward_potion_from_item(raw_item: PoolItem) -> RewardPoolPotion | None:
    if isinstance(raw_item, str):
        return RewardPoolPotion(potion_id=_normalized_id(raw_item), source_id=raw_item)

    potion_id = _first_present(raw_item, "id", "potion_id", "content_id")
    if potion_id is None:
        return None
    rarity = _potion_rarity_from_value(_first_present(raw_item, "rarity_key", "rarity", "tier"))
    return RewardPoolPotion(
        potion_id=_normalized_id(str(potion_id)),
        rarity=rarity,
        pool=str(raw_item.get("pool", "shared")),
        name=str(raw_item.get("name", potion_id)),
        source_id=str(potion_id),
    )


def _reward_relic_from_item(raw_item: PoolItem) -> RewardPoolRelic | None:
    if isinstance(raw_item, str):
        return RewardPoolRelic(
            relic_id=_normalized_id(raw_item),
            rarity=RelicRarity.COMMON,
            source_id=raw_item,
        )

    relic_id = _first_present(raw_item, "id", "relic_id", "content_id")
    if relic_id is None:
        return None
    source_rarity = _normalized_id(
        str(_first_present(raw_item, "rarity_key", "rarity", "tier") or "common")
    )
    rarity = _relic_rarity_from_value(source_rarity)
    if rarity is None:
        return None
    return RewardPoolRelic(
        relic_id=_normalized_id(str(relic_id)),
        rarity=rarity,
        pool=str(raw_item.get("pool", "shared")),
        source_rarity=source_rarity,
        name=str(raw_item.get("name", relic_id)),
        source_id=str(relic_id),
    )


def _card_rarity_from_value(value: Any) -> CardRarity | None:
    normalized = _normalized_id(str(value or "common"))
    if "uncommon" in normalized:
        return CardRarity.UNCOMMON
    if "common" in normalized:
        return CardRarity.COMMON
    if "rare" in normalized:
        return CardRarity.RARE
    return None


def _potion_rarity_from_value(value: Any) -> PotionRarity:
    normalized = _normalized_id(str(value or "common"))
    if "uncommon" in normalized:
        return PotionRarity.UNCOMMON
    if "rare" in normalized:
        return PotionRarity.RARE
    return PotionRarity.COMMON


def _relic_rarity_from_value(value: Any) -> RelicRarity | None:
    normalized = _normalized_id(str(value or "common"))
    if normalized in BOSS_RELIC_SOURCE_RARITIES:
        return RelicRarity.BOSS
    if "shop" in normalized:
        return RelicRarity.SHOP
    if "special" in normalized:
        return RelicRarity.SPECIAL
    if "uncommon" in normalized:
        return RelicRarity.UNCOMMON
    if "common" in normalized:
        return RelicRarity.COMMON
    if "rare" in normalized:
        return RelicRarity.RARE
    return None


def _unique_cards(cards: Iterable[RewardPoolCard]) -> tuple[RewardPoolCard, ...]:
    seen: set[str] = set()
    unique: list[RewardPoolCard] = []
    for card in cards:
        if card.card_id in seen:
            continue
        seen.add(card.card_id)
        unique.append(card)
    return tuple(unique)


def _unique_potions(potions: Iterable[RewardPoolPotion]) -> tuple[RewardPoolPotion, ...]:
    seen: set[str] = set()
    unique: list[RewardPoolPotion] = []
    for potion in potions:
        if potion.potion_id in seen:
            continue
        seen.add(potion.potion_id)
        unique.append(potion)
    return tuple(unique)


def _unique_relics(relics: Iterable[RewardPoolRelic]) -> tuple[RewardPoolRelic, ...]:
    seen: set[str] = set()
    unique: list[RewardPoolRelic] = []
    for relic in relics:
        if relic.relic_id in seen:
            continue
        seen.add(relic.relic_id)
        unique.append(relic)
    return tuple(unique)


def _rarity_surface(
    kind: str,
    weights: Mapping[str, int],
    source: SourceRef,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> RaritySurface:
    cleaned = {key: max(0, int(value)) for key, value in weights.items()}
    total = sum(cleaned.values())
    probabilities = (
        {key: value / total for key, value in cleaned.items()}
        if total > 0
        else {key: 0.0 for key in cleaned}
    )
    return RaritySurface(
        kind=kind,
        weights=cleaned,
        probabilities=probabilities,
        source=source,
        metadata=dict(metadata or {}),
    )


def _enum_weight_mapping(weights: Mapping[Any, int]) -> dict[str, int]:
    return {
        (key.value if hasattr(key, "value") else str(key)): int(value)
        for key, value in weights.items()
    }


def _first_present(source: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = source.get(key)
        if value is not None:
            return value
    return None


def _normalized_id(value: str) -> str:
    return value.lower().replace("'", "").replace(" ", "_").replace("-", "_")
