from __future__ import annotations

from sts2sim.mechanics.reward_pools import (
    RewardPoolRelic,
    boss_reward_shape,
    build_boss_reward_relic_pool,
    build_reward_relic_pool,
    combat_card_rarity_surface,
    elite_reward_shape,
    filter_available_relics,
    shop_potion_rarity_surface,
    split_card_reward_pools,
)
from sts2sim.mechanics.rewards import EncounterType, RelicRarity, RewardPityState

CARD_SOURCE = (
    {"id": "STRIKE", "rarity": "Common", "color": "test", "type": "Attack"},
    {"id": "SHARED_SKILL", "rarity": "Uncommon", "color": "shared", "type": "Skill"},
    {"id": "OTHER_ONLY", "rarity": "Rare", "color": "other", "type": "Power"},
    {"id": "FLASH_OF_STEEL", "rarity": "Uncommon", "color": "Colorless", "type": "Attack"},
    {"id": "INJURY", "rarity": "Common", "color": "curse", "type": "Curse"},
    {"id": "STRIKE", "rarity": "Common", "color": "test", "type": "Attack"},
)


RELIC_SOURCE = (
    {"id": "ANCHOR", "rarity_key": "Common", "pool": "shared"},
    {"id": "ANCHOR", "rarity_key": "Common", "pool": "shared"},
    {"id": "AKABEKO", "rarity_key": "Uncommon", "pool": "test"},
    {"id": "BOSS_KEY", "rarity_key": "Ancient", "pool": "shared"},
    {"id": "COURIER", "rarity_key": "Shop", "pool": "shared"},
    {"id": "OTHER_RELIC", "rarity_key": "Rare", "pool": "other"},
)


def test_card_pool_split_filters_character_and_colorless_cards() -> None:
    pools = split_card_reward_pools(CARD_SOURCE, character_id="TEST")

    assert [card.card_id for card in pools.character_cards] == ["strike", "shared_skill"]
    assert [card.card_id for card in pools.colorless_cards] == ["flash_of_steel"]
    assert pools.character_cards[1].rarity.value == "uncommon"


def test_relic_pool_filtering_removes_duplicates_owned_and_exclusions() -> None:
    reward_pool = build_reward_relic_pool(RELIC_SOURCE, character_id="TEST")
    boss_pool = build_boss_reward_relic_pool(RELIC_SOURCE, character_id="TEST")

    assert [relic.relic_id for relic in reward_pool] == ["anchor", "akabeko"]
    assert [relic.relic_id for relic in boss_pool] == ["boss_key"]
    assert boss_pool[0].source_rarity == "ancient"

    duplicate = RewardPoolRelic("akabeko", RelicRarity.UNCOMMON, pool="test")
    result = filter_available_relics(
        reward_pool + (duplicate,),
        owned_relics=("anchor",),
        excluded_relics=("akabeko",),
    )

    assert result.available == ()
    assert result.excluded_owned == ("anchor",)
    assert result.excluded_ids == ("akabeko",)
    assert result.duplicate_ids == ("akabeko",)


def test_boss_and_elite_reward_shapes_expose_expected_odds() -> None:
    boss = boss_reward_shape()
    elite = elite_reward_shape()
    scarce_elite_cards = combat_card_rarity_surface(
        EncounterType.ELITE,
        ascension_level=7,
        pity_state=RewardPityState(card_non_rare_count=2),
    )

    assert boss.gold_range == (100, 100)
    assert boss.card_count == 3
    assert boss.relic_count == 1
    assert boss.card_rarity.weights == {"rare": 1}
    assert boss.relic_rarity.weights == {"boss": 1}

    assert elite.gold_range == (35, 45)
    assert elite.relic_count == 1
    assert elite.potion_effective_chance_tenths == 525
    assert elite.card_rarity.weights["rare"] == 1000
    assert scarce_elite_cards.weights["rare"] == 600
    assert scarce_elite_cards.metadata["rare_pity_bonus_basis"] == 100


def test_shop_potion_rarity_surface_is_deterministic_metadata() -> None:
    first = shop_potion_rarity_surface()
    second = shop_potion_rarity_surface()

    assert first == second
    assert first.weights == {"common": 6500, "uncommon": 2500, "rare": 1000}
    assert first.probabilities["rare"] == 0.1
