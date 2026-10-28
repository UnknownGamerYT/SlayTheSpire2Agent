from sts2sim.engine.rng import STS2_STABLE_DERIVATION, RunRng
from sts2sim.mechanics.combat_rewards import CombatRewardCard
from sts2sim.mechanics.rewards import CardRarity
from sts2sim.mechanics.source_card_factory import source_noncombat_cards


def test_noncombat_default_c_vectors_distinct_base_odds_no_upgrade_roll():
    rng = RunRng('TEST', derivation=STS2_STABLE_DERIVATION)
    pool = tuple(CombatRewardCard(str(i), rarity) for rarity, offset in (
        (CardRarity.COMMON, 0), (CardRarity.UNCOMMON, 4), (CardRarity.RARE, 8))
                 for i in range(offset, offset + 4))
    ids, rarities, upgrades = source_noncombat_cards(rng, pool, 5)
    assert ids == ('0', '3', '1', '2', '5')
    assert len(set(ids)) == 5 and len(rarities) == 5
    assert upgrades == (False,) * 5
    assert rng.snapshot().streams['rewards'].draws == 10


def test_uniform_custom_creation_upgrade_roll_is_explicit_and_override_only_items():
    rng = RunRng('TEST', derivation=STS2_STABLE_DERIVATION)
    pool = tuple(CombatRewardCard(str(i), CardRarity.COMMON) for i in range(3))
    _, _, upgrades = source_noncombat_cards(rng, pool, 3, uniform=True,
        no_upgrade_roll=False, act=5, item_rng=rng.for_stream('MODEL'))
    assert upgrades == (True,) * 3
    snapshot = rng.snapshot()
    assert snapshot.streams['MODEL'].draws == 6
    assert snapshot.streams.get('rewards') is None or snapshot.streams['rewards'].draws == 0
