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

- `AscensionLevel`: enemy HP scales at A8 and attacks generally scale at A9.
  Gremlin Merc's three attack properties explicitly use A8 instead. The previous
  room-specific thresholds came from Slay the Spire 1. Innate Plating, Curl Up,
  Skittish, Shriek and Slippery scale at A8; other recorded innate values scale
  at A9. Siphon healing scales at A8. Test Subject forms are 100/200/300 HP,
  or 111/212/313 at A8.
- Numeric property review found 19 omitted move-power ascensions, 14 omitted
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
- All 87 `GenerateMonsters` implementations now have opening-roster contracts:
  78 fixed rosters and nine generated or setting-dependent rosters. The source
  list's possible participants are no longer used as the opening lineup.
  Random Ruby Raiders and Bowlbugs exclude duplicates; Slimes and Strangler
  companions follow their source composition rules. Per-instance properties
  preserve alternating Chompers, the middle Inklet, the first Kin Follower,
  Punch Off damage/openings, four event Wrigglers and phased Slug/Scroll/Rat/
  Decimillipede starts. Encounter-local randomness uses the source encounter ID
  and TotalFloor; an imported `flags.total_floor` overrides the run-history
  calculation. Battleworn Dummy requires its selected setting.
- Tough Egg hatches into the same creature, clears powers except Minion,
  restores 19–21 HP (20–22 at A8), and repeats Nibble. Enemy-turn summons display
  the source Hatch counter. Hatch does not add a creature and therefore does
  not reapply Fur Coat. Source slot names are retained; dynamic summons reuse
  available named slots and public combat state follows source slot order.
- All 28 random selectors omitted from the content API now use the source's
  73 ordered branches, weights, consecutive limits and cooldowns. Zero-weight
  selections retain the source draw and subtraction behavior. Two-Tailed Rats
  share the three-backup limit and wait two attacks before requesting backup.
- Enemy Block expires for the whole side before its actions. Ally effects
  survive later actors' turns. Ritual skips the side on which it was applied;
  Ritual, Territorial and High Voltage resolve after all enemy actions. These
  phase boundaries are preserved by save/reload.
- Guardbot's Guard grants 15 Block only to Fabricators. Cubex starts with 13
  Block. Source A8 Block values apply to Magi's Prepare and Louse Progenitor's
  Curl and Grow. Toadpole spends two Thorns before Spike Spit; Spiny Toad spends
  five after Spike Explosion. Living Fog creates one Gas Bomb before each
  Bloat attack; the source does not increase that summon count.

- `OstyCmd.Summon`, `BoundPhylactery`, `PhylacteryUnbound` and `DieForYouPower`:
  missing pets have zero HP and cannot attack, block, heal or die again. A
  character ID alone does not create a pet. Bound opens at 1 HP and summons 1
  after later energy resets. Unbound opens at 5, then summons 2 on every turn,
  including the first (7 HP at the first action). A living summon adds the
  full amount to both maximum and current HP. A dead pet stays in combat and
  revives with maximum/current HP equal to the new summon amount.
- Pet attacks preserve selected target identity, refresh AoE/random targets
  between hits, resolve replacement deaths immediately, and use CombatTargets
  even for a singleton random target. Rattle counts attack commands and Bone
  Flute triggers once after each command. Thorns consumes shared owner Block
  and pet HP before each target's damage; a pet death finishes that AoE batch
  and stops later hits. Necro Mastery stacks, bypasses enemy Block, and echoes
  actual pet HP loss. These outcomes include missing pets and save/reload.
- `PlatingPower` grants opening Block, skips its first enemy-round decrement,
  then decrements on later enemy-side starts. It grants Block for the whole
  side before side-end damage. Slumber's side-end wake removes Plating after
  that Block is granted. Sewer Clam starts with Plating 8/9 at A8; Mysterious
  Knight inherits Flail Knight and adds Strength 6 and Plating 6.
- Player Thorns resolves before each enemy hit as blockable, unpowered damage.
  Killing the attacker completes the current hit and stops subsequent hits.
  Hopper steals before that attack and uses any permanent deck version as its
  final fallback. Illusions keep buffs, signed negative buffs and temporary
  debuff wrappers after death, using the source power type declarations.
  Spectral Knight applies Hex 2 to the player on its first enemy action.
- Philosopher's Stone applies one Strength to each newly added enemy, including
  summons, death spawns and every Stock replacement. Same-creature hatches and
  revivals do not invoke the added-creature hook. Required summon definitions
  now fail the transition with a named diagnostic when missing, instead of
  silently omitting enemies from a training trajectory.
- Independent numeric fixtures cover 226 registered attack payloads mapped to
  source methods: 223 constant/ascension damage expressions and 225 hit-count expressions at
  A0/A7/A8/A9. This identified Gremlin Merc's A8 exception and Fake Merchant's
  unconditional Spew Coins 2 damage × 8 hits. Dynamic Dread, Pressure Gun,
  Explode and Multi Claw expressions retain their focused behavior checks.
- Enemy Move Block now uses signed Dexterity, temporary Dexterity, Fasten and
  Frail, retains fractions through Shadowmeld, and applies the creature storage
  cap. Unpowered gains omit Dexterity/Frail. NoBlock exempts gains without a
  card source, and Unmovable exempts monster targets. All 22 source Move Block
  payloads have numeric A0/A7/A8/A9 contracts. The Forgotten's successive Miasma
  gains are 8, 10, 12, before its next Dexterity increase.

The independent HP table and focused outcome cases establish these scopes.
They do not establish complete equivalence for every enemy callback, every
power interaction, multiplayer HP scaling,
or complete live-game action traces. Those scopes require further verification.
