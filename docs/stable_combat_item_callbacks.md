# Stable combat item callback contracts

Target: stable v0.107.1. The independently implemented callback projections use
[sts2-rl-agent at 5d874e8](https://github.com/zhiyue/sts2-rl-agent/tree/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/decompiled).
`tests/fixtures/stable_combat_item_callbacks.json` records source hashes, method
scopes and compact facts. The game source is not bundled.

`tests/combat/test_stable_combat_item_callbacks.py` retains public-action outcomes
and save/reload cases for:

- Pael’s Eye: finished manual play history, first-turn Whispering Earring
  exception, one extra turn per combat, and hand exhaustion before Burn with
  ordinary exhaust callbacks. Cards drawn by exhaust callbacks are outside the
  initial exhaustion snapshot.
- Toasty Mittens: first-turn Innate exclusion, discard shuffle before selection,
  and shuffle/exhaust callbacks before the normal hand draw and Strength gain.
  Its BeforeHandDraw hook follows turn-start powers; Demon Form receives the
  first Helmet doubling before Mittens applies its own Strength.
- Unceasing Top: completed potion and card actions, deferred modal choices and
  nested plays, and the source turn-phase filter.
- Unsettling Lamp: first enemy debuff attempt locks one card through its first
  play, all receivers and debuff commands, Artifact, self debuffs after arming,
  attack riders, and exclusion of potion sources. Snecko Skull adds Poison
  before Lamp multiplies it. Source temporary debuffs are represented by one
  flattened temporary stat application, avoiding doubling an internal stat twice.
- Ruined Helmet: first positive Strength receipt before the storage cap,
  including relic/start callbacks, card/potion effects, Arsenal, Red Skull,
  Rupture, Ritual and a non-owner giver. Consumption survives save/reload.
- Imbued: opening hand draw precedes AutoPrePlay and an Imbued card can inspect
  the drawn hand. Diamond Diadem counts AutoPost plays before its regular end
  hook. Consuming Shadow stays after orb passives and hand flush; Hailstorm
  remains in the earlier regular hook.

`tests/combat/test_stable_osty_modifiers.py` additionally covers the shared
AttackCommand begin/finish API: every hit and receiver in one pet command uses
its claimed Gigantification, later commands consume later entries, missing pets
consume nothing, and pet Vigor does not consume the owner’s Vigor. Pet damage
uses the pet’s Vulnerable modifiers, excluding owner Paper Phrog and Cruelty.

These outcomes establish the listed single-player scopes. They do not establish
complete cross-product equivalence for every item, every temporary-power model,
multiplayer participants, mods, or live-game traces against the installed build.
