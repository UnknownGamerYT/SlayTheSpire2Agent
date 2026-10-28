"""Stable v0.107.1 Ascension thresholds from AscensionLevel/AscensionManager.

Legacy enum names remain importable, but only source-supported rules activate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sts2sim.content.sources import STABLE_DECOMPILED_SOURCE, SourceRef

MAX_ASCENSION_LEVEL = 10


class AscensionFlag(str, Enum):
    MORE_ELITES = "more_elites"
    REDUCED_ANCIENT_HEAL = "reduced_ancient_heal"
    REDUCED_GOLD = "reduced_gold"
    DEADLY_NORMALS = "deadly_normals"
    DEADLY_ELITES = "deadly_elites"
    DEADLY_BOSSES = "deadly_bosses"
    REDUCED_REST_HEAL = "reduced_rest_heal"
    DAMAGED_START = "damaged_start"
    TOUGHER_NORMALS = "tougher_normals"
    TOUGHER_ELITES = "tougher_elites"
    TOUGHER_BOSSES = "tougher_bosses"
    ASCENDER_CURSE = "ascender_curse"
    FEWER_POTION_SLOTS = "fewer_potion_slots"
    LOWER_UPGRADE_CHANCE = "lower_upgrade_chance"
    POORER_BOSS_REWARDS = "poorer_boss_rewards"
    LOWER_MAX_HP = "lower_max_hp"
    UNFAVORABLE_EVENTS = "unfavorable_events"
    EXPENSIVE_SHOPS = "expensive_shops"
    ADVANCED_NORMAL_AI = "advanced_normal_ai"
    ADVANCED_ELITE_AI = "advanced_elite_ai"
    ADVANCED_BOSS_AI = "advanced_boss_ai"
    DOUBLE_BOSS = "double_boss"


@dataclass(frozen=True, slots=True)
class AscensionFlagRule:
    level: int
    flag: AscensionFlag
    summary: str
    source: SourceRef = STABLE_DECOMPILED_SOURCE


@dataclass(frozen=True, slots=True)
class AscensionConfig:
    level: int
    flags: frozenset[AscensionFlag]
    rules: tuple[AscensionFlagRule, ...]

    def enabled(self, flag: AscensionFlag) -> bool:
        return flag in self.flags


@dataclass(frozen=True, slots=True)
class AscensionEconomyRules:
    base_potion_slots: int = 3
    fewer_potion_slots_level: int = 4
    card_removal_base_price: int = 75
    card_removal_increment: int = 25
    inflated_card_removal_level: int = 6
    inflated_card_removal_base_price: int = 100
    inflated_card_removal_increment: int = 50
    rest_heal_fraction: float = 0.30
    reduced_rest_heal_level: int = 11
    reduced_rest_heal_fraction: float = 0.20
    poorer_reward_level: int = 3
    poorer_reward_gold_multiplier: float = 0.75
    rarity_scarcity_level: int = 7
    ascender_curse_level: int = 5
    source: SourceRef = STABLE_DECOMPILED_SOURCE


@dataclass(frozen=True, slots=True)
class AscensionEconomyModifiers:
    level: int
    potion_slots: int
    card_removal_base_price: int
    card_removal_increment: int
    rest_heal_fraction: float
    reward_gold_multiplier: float
    rarity_scarcity_enabled: bool
    ascender_curse_enabled: bool
    source: SourceRef = STABLE_DECOMPILED_SOURCE


DEFAULT_ASCENSION_ECONOMY_RULES = AscensionEconomyRules()


ASCENSION_FLAG_RULES: tuple[AscensionFlagRule, ...] = (
    AscensionFlagRule(1, AscensionFlag.MORE_ELITES, "Swarming Elites: more elite rooms."),
    AscensionFlagRule(2, AscensionFlag.REDUCED_ANCIENT_HEAL,
                      "Weary Traveler: Ancients heal 80% of missing HP."),
    AscensionFlagRule(3, AscensionFlag.REDUCED_GOLD,
                      "Poverty: enemies and chests drop 25% less gold."),
    AscensionFlagRule(4, AscensionFlag.FEWER_POTION_SLOTS, "Tight Belt: one fewer potion slot."),
    AscensionFlagRule(5, AscensionFlag.ASCENDER_CURSE, "Add Ascender's Bane at run start."),
    AscensionFlagRule(6, AscensionFlag.EXPENSIVE_SHOPS,
                      "Inflation: Merchant card removal costs more."),
    AscensionFlagRule(7, AscensionFlag.LOWER_UPGRADE_CHANCE,
                      "Scarcity: Rare and upgraded cards appear less often."),
    AscensionFlagRule(8, AscensionFlag.TOUGHER_NORMALS, "Tough Enemies: normal HP tuning."),
    AscensionFlagRule(8, AscensionFlag.TOUGHER_ELITES, "Tough Enemies: elite HP tuning."),
    AscensionFlagRule(8, AscensionFlag.TOUGHER_BOSSES, "Tough Enemies: boss HP tuning."),
    AscensionFlagRule(9, AscensionFlag.DEADLY_NORMALS, "Deadly Enemies: normal attack tuning."),
    AscensionFlagRule(9, AscensionFlag.DEADLY_ELITES, "Deadly Enemies: elite attack tuning."),
    AscensionFlagRule(9, AscensionFlag.DEADLY_BOSSES, "Deadly Enemies: boss attack tuning."),
    AscensionFlagRule(10, AscensionFlag.DOUBLE_BOSS, "Fight two final-act bosses."),
)


def validate_ascension_level(level: int) -> int:
    if not 0 <= level <= MAX_ASCENSION_LEVEL:
        raise ValueError(f"Ascension level must be between 0 and {MAX_ASCENSION_LEVEL}: {level}")
    return level


def ascension_rules_for_level(level: int) -> tuple[AscensionFlagRule, ...]:
    level = validate_ascension_level(level)
    return tuple(rule for rule in ASCENSION_FLAG_RULES if rule.level <= level)


def flags_for_ascension(level: int) -> frozenset[AscensionFlag]:
    return frozenset(rule.flag for rule in ascension_rules_for_level(level))


def ascension_config(level: int) -> AscensionConfig:
    rules = ascension_rules_for_level(level)
    return AscensionConfig(
        level=validate_ascension_level(level),
        flags=frozenset(rule.flag for rule in rules),
        rules=rules,
    )


def ascension_enabled(level: int, flag: AscensionFlag) -> bool:
    return flag in flags_for_ascension(level)


def source_for_flag(flag: AscensionFlag) -> SourceRef | None:
    for rule in ASCENSION_FLAG_RULES:
        if rule.flag is flag:
            return rule.source
    return None


def potion_slots_at_ascension(
    ascension_level: int,
    *,
    base_slots: int = DEFAULT_ASCENSION_ECONOMY_RULES.base_potion_slots,
    rules: AscensionEconomyRules = DEFAULT_ASCENSION_ECONOMY_RULES,
) -> int:
    """Return potion slots before relic modifiers are applied."""

    validate_ascension_level(ascension_level)
    slots = max(0, base_slots)
    if ascension_level >= rules.fewer_potion_slots_level:
        return max(0, slots - 1)
    return slots


def card_removal_price_at_ascension(
    ascension_level: int,
    *,
    card_removals_bought: int = 0,
    rules: AscensionEconomyRules = DEFAULT_ASCENSION_ECONOMY_RULES,
) -> int:
    """Return the shop card-removal price before relic discounts."""

    validate_ascension_level(ascension_level)
    removals = max(0, card_removals_bought)
    if ascension_level >= rules.inflated_card_removal_level:
        return (
            rules.inflated_card_removal_base_price
            + rules.inflated_card_removal_increment * removals
        )
    return rules.card_removal_base_price + rules.card_removal_increment * removals


def rest_site_heal_fraction_at_ascension(
    ascension_level: int,
    *,
    rules: AscensionEconomyRules = DEFAULT_ASCENSION_ECONOMY_RULES,
) -> float:
    """Stable HealRestSiteOption uses30% at every Ascension, before modifiers."""

    validate_ascension_level(ascension_level)
    return rules.rest_heal_fraction


def reward_gold_multiplier_at_ascension(
    ascension_level: int,
    *,
    rules: AscensionEconomyRules = DEFAULT_ASCENSION_ECONOMY_RULES,
) -> float:
    """Return the current non-event reward gold multiplier."""

    validate_ascension_level(ascension_level)
    if ascension_level >= rules.poorer_reward_level:
        return rules.poorer_reward_gold_multiplier
    return 1.0


def rarity_scarcity_at_ascension(
    ascension_level: int,
    *,
    rules: AscensionEconomyRules = DEFAULT_ASCENSION_ECONOMY_RULES,
) -> bool:
    """Return whether combat and shop rare-card surfaces use scarcity tuning."""

    validate_ascension_level(ascension_level)
    return ascension_level >= rules.rarity_scarcity_level


def ascension_economy_modifiers(
    ascension_level: int,
    *,
    base_potion_slots: int = DEFAULT_ASCENSION_ECONOMY_RULES.base_potion_slots,
    rules: AscensionEconomyRules = DEFAULT_ASCENSION_ECONOMY_RULES,
) -> AscensionEconomyModifiers:
    """Return the non-combat-AI modifiers used by reward and shop helpers."""

    level = validate_ascension_level(ascension_level)
    inflated_removal = level >= rules.inflated_card_removal_level
    return AscensionEconomyModifiers(
        level=level,
        potion_slots=potion_slots_at_ascension(level, base_slots=base_potion_slots, rules=rules),
        card_removal_base_price=(
            rules.inflated_card_removal_base_price
            if inflated_removal
            else rules.card_removal_base_price
        ),
        card_removal_increment=(
            rules.inflated_card_removal_increment
            if inflated_removal
            else rules.card_removal_increment
        ),
        rest_heal_fraction=rest_site_heal_fraction_at_ascension(level, rules=rules),
        reward_gold_multiplier=reward_gold_multiplier_at_ascension(level, rules=rules),
        rarity_scarcity_enabled=rarity_scarcity_at_ascension(level, rules=rules),
        ascender_curse_enabled=level >= rules.ascender_curse_level,
        source=rules.source,
    )
