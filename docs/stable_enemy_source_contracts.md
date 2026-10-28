# Stable enemy contracts

The enemy follow-up targets v0.107.1 public. Reference source is
[zhiyue/sts2-rl-agent at 5d874e8](https://github.com/zhiyue/sts2-rl-agent/tree/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/decompiled).
Its [version record](https://github.com/zhiyue/sts2-rl-agent/blob/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/GAME_VERSIONS.md)
reports Steam build 23811903, release commit 59260271 and main assembly hash
-1555940892. This is third-party build provenance; it has not been matched to
the user's installed executable. No game binaries or decompiled files are
distributed by this change.

`tests/fixtures/stable_enemy_hp_contracts.json` records independently derived
HP ranges and source file hashes for all 115 registered monsters, including
inherited Decimillipede and Mysterious Knight contracts. Source-derived outcome
tests live in `tests/combat/test_stable_enemy_lifecycle.py`.

Confirmed corrections:

- `AscensionLevel`: all enemy HP scales at A8 and damage at A9. The previous
  room-specific thresholds came from Slay the Spire 1. Innate Plating, Curl Up,
  Skittish, Shriek and Slippery scale at A8; other recorded innate values scale
  at A9. Siphon healing scales at A8. Test Subject forms are 100/200/300 HP,
  or 111/212/313 at A8.
- Numeric property review found 19 omitted move-power ascensions, 12 omitted
  Block ascensions and three omitted multi-hit count ascensions. Skulking
  Colony's A9 Inertia gives 4 Strength. The Lost and The Forgotten still steal
  2 Strength/Dexterity at A9.
- `GenerateMoveStateMachine` review found 49 null or missing follow-up edges
  and five missing or incorrect opening states in the API. Explicit source
  edges now take precedence over move-list cycling. This prevents repeating
  cultist startup buffs, repeated Tracker setup and repeated Tough Egg hatches.
- `Axebot` and `StockPower`: initial Stock 2; initial Hammer Uppercut, then
  One-Two and Uppercut alternate. Death creates a new Axebot in the same slot
  with Stock reduced by one, preserving combat until the last replacement is
  defeated. Replacements start with Boot Up, granting 10 Block and 3/6 Strength
  (15 Block and 4/8 Strength at A9). Each replacement rolls its ordinary HP;
  the beta v0.111.0 cumulative +10 HP change is excluded. Targeted multi-hit
  attacks retain the dead creature identity. AoE and random hits can hit the
  replacement; AoE debuffs resolve after replacement creation. Replacements
  created during the enemy turn wait until the next enemy turn.
- `Aeonglass.IncreasingIntensityMove`: upgrade existing Withers, advance the
  Wither counter, create Withers at that new counter, grant 3/4 Strength plus
  prior uses, then advance the Strength counter. The wiki's 2/3 base Strength
  is inconsistent with the pinned stable source. Withering Presence continues
  to generate its card after each sixth resolved card play.
- `SurroundedPower`: uninitialized direction is enum value Right (zero).
  Targeted cards and potions change facing. A claw's death faces the surviving
  side, and Crab Rage grants 6 Strength and 99 Block before the next attack
  command. Its death trigger is consumed once even when finalization revisits
  the action's events.
- `Fabricator`: Fabricate creates one defensive bot (Guardbot/Noisebot) and one
  aggressive bot (Zapbot/Stabbot); Fabricating Strike creates an aggressive bot.
  It excludes the last spawned bot from the next selection and grants Minion.
- `Entomancer`: Bees, Spear, Pheromone Spit, repeat. Spit grants one Personal
  Hive and one Strength while Hive is below three, then two Strength. The API
  had flattened the mutually exclusive branches into unconditional effects.
- `CombatState.CreateCreature`: HP selection uses the Niche stream and removes
  other current enemies' maximum HP values from its candidate range, falling
  back to the full range when every candidate is occupied. Initial and later
  AI rolls and Fabricator choices use MonsterAi; Hopper theft uses
  CombatCardGeneration. Stream ownership and save/reload checks are in
  `tests/combat/test_stable_enemy_rng.py`. The per-creature model RNG's only
  reference in the registered monster classes is Tough Egg's cosmetic skin.
- `RandomBranchState`: selection rounds through C# float precision, and
  CanRepeatXTimes limits consecutive uses rather than lifetime uses.

The independent HP table and focused outcome cases establish these scopes.
They do not establish complete equivalence for every enemy callback, every
power interaction, multiplayer HP scaling, every encounter's generated roster,
or complete live-game action traces. Those scopes require further verification.
