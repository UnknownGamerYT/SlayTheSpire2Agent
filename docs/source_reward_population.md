# Stable combat reward population order

The source combat pipeline separates generation and population from the order in
which the reward screen displays its entries. For the five playable characters,
`source_combat_rewards.py` follows the stable `RewardsSet` lifecycle:

1. Generate the base reward objects. The potion-odds roll happens here.
2. Populate the base objects in constructor order, then populate the room's
   existing `ExtraRewards` in enqueue order.
3. Run `TryModifyRewards` listeners in relic acquisition order. Populate only
   newly appended objects that are still unpopulated.

This puts recovered Heist gold, The Hunt offers and Royalties gold before relic
hook additions. Lava Rock and Black Star append relic rewards at this hook stage;
they do not increase the initial encounter relic count. Prayer Wheel, White Star,
Amethyst Aubergine and a due Wongo ticket retain their acquired listener positions.
A fixed GoldReward already in the room list still consumes its Populate draw.
Amethyst's fixed GoldReward is appended already populated and consumes no draw.

The tutorial base set bypasses generic factories, preserves the fixed potion and
relic entries, and still receives subsequent room hook additions. Its elite set
populates the fixed relic before the manual card offer. Ordinary final-boss sets
have no base or room extras. A ready Wongo ticket can still add its three relics,
because `GenerateWithoutOffering` invokes reward hooks after the early return in
`WithRewardsFromRoom`; Amethyst explicitly excludes final-boss rewards.

New source combats record mixed extra reward descriptors as they are enqueued.
The source death adapter preserves Heist's BeforeDeath position before a Fatal
card reward for the same victim. Saves made before this queue existed cannot
recover an unrecorded mixed order; their reward metadata labels the fallback
`legacy_descriptor_order`. Reserved `TEST` and legacy MT custom fixtures retain
the earlier custom reward pipeline.

The retained path hashes and finite contracts are in
[`source_reward_population_order.json`](../tests/fixtures/source_reward_population_order.json).
Public action, persistence, RNG-call and acquired-order regressions are in
[`test_source_reward_population_order.py`](../tests/rewards/test_source_reward_population_order.py).
These are source-based simulator checks. Installed-game comparisons remain
necessary to establish runtime equivalence.
