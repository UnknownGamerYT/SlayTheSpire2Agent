# Stable relic queues

The source snapshot at
[`5d874e8`](https://github.com/zhiyue/sts2-rl-agent/tree/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/decompiled)
identifies v0.107.1 / Steam build23811903. `RelicGrabBag.cs`, `RelicFactory.cs`,
`RelicCmd.cs`, `MerchantInventory.cs`, and `TreasureRoomRelicSynchronizer.cs`
define the following implemented contracts:

- Initial per-player and shared queues are shuffled during run initialization.
- Elite relic offers consume the player's queue from the front and remove the
  matching shared copy immediately. Skipping does not put an offered item back.
- Shops consume the player's queue from the back, preserve shop-ineligible
  entries, and remove the shared copy. Two normal rarity rolls use Rewards RNG;
  the third initial slot requests Shop rarity. Courier restocks also consume
  the queue and persist its state.
- Chests consume the shared queue from the front with TreasureRoomRelics RNG.
  Claiming a relic removes its player copy; merely offering it does not.
  Chest gold uses the player's Rewards stream.
- Obtaining a fixed, nonstackable relic removes all queue copies. Exhausted
  eligible queues fall back to Circlet. Eligibility and rarity fallbacks come
  from the source helper, including the late-floor exclusions.
- A default boss reward grants gold, a potion roll, and rare card offers.
  It grants no default relic. Lava Rock adds two normal relic offers after the
  act-one boss, once, and persists that trigger. The previous implementation
  incorrectly supplied an Ancient relic by default and added one for Lava Rock.

Public action regressions in `tests/rewards/test_source_relic_bag_runtime.py`
check queue direction, skipped offers, shared/player removal, and serialization.
They use synthetic named queue sentinels to isolate queue semantics; they do not
establish the behavior of each real relic. The pure helper has separate stable
eligibility and fallback fixtures. These tests are source-derived contracts,
not captures from an installed game.

Event-specific offers, complete source RNG consumption order, multiplayer voting,
and independent game trace comparisons require separate validation. Explicit
custom fixtures without serialized source queues retain the legacy pool API.
