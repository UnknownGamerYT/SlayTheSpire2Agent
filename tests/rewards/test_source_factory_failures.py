from __future__ import annotations

import random

import pytest

from sts2sim.engine.rng import STS2_STABLE_DERIVATION, RunRng
from sts2sim.mechanics.combat_rewards import (
    CombatRewardCard,
    CombatRewardContext,
    draw_card_reward_options,
)
from sts2sim.mechanics.rewards import CardRarity, EncounterType, RewardPityState
from sts2sim.mechanics.source_card_factory import source_noncombat_cards
from sts2sim.mechanics.source_factory_errors import SourceCardFactoryError


def _rng() -> RunRng:
    return RunRng("FACTORY-EDGE", derivation=STS2_STABLE_DERIVATION)


def test_missing_common_rarity_cycles_to_uncommon_before_rare_and_preserves_rolled_pity():
    rng = _rng()
    cards, rarities, pity = draw_card_reward_options(
        rng.for_stream("rewards"),
        card_pool=(CombatRewardCard("uncommon", CardRarity.UNCOMMON),
                   *(CombatRewardCard(f"rare-{i}", CardRarity.RARE) for i in range(8))),
        context=CombatRewardContext(character_id="IRONCLAD"),
        pity_state=RewardPityState(card_rarity_offset=-.05), card_count=1,
    )
    assert cards == ("uncommon",)
    assert rarities == ("uncommon",)
    assert pity.card_rarity_offset == pytest.approx(-.04)
    assert rng.snapshot().streams["rewards"].draws == 3


def test_missing_boss_rare_wraps_to_common_before_uncommon_and_keeps_rare_reset():
    rng = _rng()
    cards, rarities, pity = draw_card_reward_options(
        rng.for_stream("rewards"),
        card_pool=(CombatRewardCard("common", CardRarity.COMMON),
                   *(CombatRewardCard(f"uncommon-{i}", CardRarity.UNCOMMON) for i in range(8))),
        context=CombatRewardContext(character_id="IRONCLAD", encounter=EncounterType.BOSS),
        pity_state=RewardPityState(card_rarity_offset=.2), card_count=1,
    )
    assert cards == ("common",)
    assert rarities == ("common",)
    assert pity.card_rarity_offset == pytest.approx(-.05)
    assert rng.snapshot().streams["rewards"].draws == 3


def test_source_blacklist_is_applied_before_finding_next_allowed_rarity():
    rng = _rng()
    cards, rarities, pity = draw_card_reward_options(
        rng.for_stream("rewards"),
        card_pool=(CombatRewardCard("uncommon", CardRarity.UNCOMMON),
                   CombatRewardCard("rare", CardRarity.RARE)),
        context=CombatRewardContext(character_id="IRONCLAD"),
        pity_state=RewardPityState(card_rarity_offset=-.05), card_count=2,
    )
    assert cards == ("uncommon", "rare")
    assert rarities == ("uncommon", "rare")
    # Both rarity rolls were common; selecting a fallback Rare does not reset
    # the source rarity offset, which was already updated by the rolled rarity.
    assert pity.card_rarity_offset == pytest.approx(-.03)
    assert rng.snapshot().streams["rewards"].draws == 6


def test_nonuniform_empty_source_factory_consumes_only_the_rarity_draw_then_faults():
    rng = _rng()
    with pytest.raises(SourceCardFactoryError, match="no_valid_rarity") as error:
        source_noncombat_cards(rng, (), 3)
    assert error.value.source_exception_type == "InvalidOperationException"
    assert error.value.generated_card_ids == ()
    assert rng.snapshot().streams["rewards"].draws == 1


def test_uniform_empty_source_factory_faults_without_a_random_item_draw():
    rng = _rng()
    with pytest.raises(SourceCardFactoryError, match="no_valid_card"):
        source_noncombat_cards(rng, (), 3, uniform=True)
    assert rng.snapshot().streams["rewards"].draws == 0


def test_exhausted_source_factory_faults_instead_of_returning_a_short_reward():
    rng = _rng()
    pool = (CombatRewardCard("uncommon", CardRarity.UNCOMMON),
            CombatRewardCard("rare", CardRarity.RARE))
    with pytest.raises(SourceCardFactoryError) as error:
        source_noncombat_cards(rng, pool, 3)
    assert set(error.value.generated_card_ids) == {"uncommon", "rare"}
    assert rng.snapshot().streams["rewards"].draws == 5


def test_legacy_synthetic_factory_keeps_its_explicit_short_pool_behavior():
    pool = (CombatRewardCard("common", CardRarity.COMMON),)
    assert source_noncombat_cards(random.Random(1), pool, 3)[0] == ("common",)
