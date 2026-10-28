# Stable v0.107.1 run generation

The standard map, named act selection, room sequences and relic deque helpers
now use the public source snapshot
[zhiyue/sts2-rl-agent at 5d874e8](https://github.com/zhiyue/sts2-rl-agent/tree/5d874e855ae1d0550d5c30416fadd7dbb0d133b7).
Its [GAME_VERSIONS.md](https://github.com/zhiyue/sts2-rl-agent/blob/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/GAME_VERSIONS.md)
identifies v0.107.1, Steam public build 23811903, release commit 59260271,
and assembly hash -1555940892. This is a version-qualified public decompilation;
the local user's installed game has not been fingerprinted or executed here.

## Implemented source rules

- `StandardActMap.cs`, `MapPathPruning.cs`, `MapPostProcessing.cs` and
  `MapCoord.cs`: seven columns and seven generated paths; distinct starts for
  the first two paths; shuffled adjacent steps; crossover rejection; fixed
  first-fight, treasure and last-rest rows; up to three assignment passes;
  duplicate segment pruning and count repair; centering, spreading and path
  straightening. Stable shuffles sort coordinates before Fisher-Yates.
- `SpoilsActMap.cs` and `GoldenPathActMap.cs`: source hourglass construction
  on the independent `spoils_map` RNG, centered treasure convergence, parent/
  child limits, source pruning, and the fixed Golden Compass path in column 3.
  Multiplayer removes the Golden Path's third room. Golden Compass marks the
  act in which it was obtained. Spoils Map marks the resulting map's first
  treasure if another map modifier replaces its hourglass.
- The four `Models.Acts` classes and `MapPointTypeCounts.cs`: 15/15/14/13 room
  rows for Overgrowth/Underdocks/Hive/Glory, one fewer in multiplayer; actual
  Gaussian count formulas and RNG call order; 5 elites at A0 and 8 at A1+;
  3 shops; 10–14 unknowns in Act 1 and 9–13 later; 6–7 rests in Acts 1–2 and
  5–6 in Act 3. The counts are assignment targets. Fixed rows and placement
  constraints can affect final totals. No additional treasure rooms are placed
  by the random assignment queue. Elites/rests start at row 6; generated rests
  cannot occupy the three rows before the boss, apart from the forced last row.
- `ActModel.GetRandomList` and `StartRunLobby.BeginRunLocally`: ordered act
  choices from `act_selection`, unlock gates, first discovery of Underdocks,
  and explicit named act overrides. The act selected for a run governs its
  maps, encounter pools, event lists, bosses and native Ancient candidates.
- `RunManager.GenerateRooms`, `ActModel.GenerateRooms`, `GrabBag.cs` and
  `RoomSet.cs`: initialize all acts up front; shuffle events, then generate
  the complete normal/elite encounter sequences, boss and Ancient. First
  3/3/2/2 actual normal combats use each act's weak pool. Weak/regular/elite
  draws use ordered bags without replacement, reject the previous encounter
  and overlapping source tags when another candidate exists, and refill only
  when empty. Rejections and singleton draws consume RNG. Fifteen elite
  entries are generated per act. Visits consume and cycle the stored plans.
- Source profile discovery rules: first-unseen boss override, first-ever
  Overgrowth combat/elite/event ordering, Neow/Orobas epoch gates and shared
  Darv allocation to at most one later act. Multiplayer skips tutorial
  encounter/discovery overrides. A10 creates and traverses two distinct final
  bosses before completing the run. A profile without Neow starts with the
  source's weak monster room at the starting point before row 1.
- `RoomSet.EnsureNextEventIsValid`: consume the stored shuffled event sequence,
  skip visited or ineligible IDs and serialize the cursor. Unique eligible
  events can repeat after exhaustion. The existing eligibility layer still
  returns `UNKNOWN_EVENT` if no known eligible event exists; it retains unknown
  entry-clause diagnostics rather than selecting an event known to be invalid.
- `RunManager.InitializeNewRun`, `RelicGrabBag.cs` and source relic pools:
  initialize the shared and player rarity deques in source order before room
  generation. Shared bags retain their extra Ancient/Event buckets. The pure
  deque helper pulls front/back without RNG, permanently removes source-
  disallowed relics, preserves shop-filtered items, applies rarity fallback,
  and supports the source shared-bag refresh and multiplayer fallback deque.

`act_room_sets`, per-kind visit counters, `act_ids`, `visited_event_ids`,
`shared_relic_bag`, `shared_relic_bag_originals` and `player_relic_bag` persist
in run flags. Existing explicit encounter/monster/event fixtures still override
normal selection. Synthetic `map_width`, `map_paths` and `map_floors` overrides
remain available; custom geometry is outside the seven-column game contract.

The reserved `TEST` character with MT19937 is a synthetic legacy fixture.
Its historical setup RNG consumption and synthetic Ancient offerings remain
stable so card/relic tests keep their controlled opening hands and avoid
unrelated pickup choices. It records `fixture_generation_mode` accordingly.
It does not represent any playable game character or a game fidelity mode.
The five real characters use the stable run generation even when a caller
explicitly selects MT19937 for a diagnostic run.

## Profile assumptions and evidence limits

Absent profile facts, training uses all unlocked content and records
`generation_profile="all_unlocked"`. Callers can supply `revealed_epochs`,
`number_of_runs`, `seen_encounters`, `discovered_act_ids`, `player_count` and
`unlocked_relic_ids`. The epoch/relic/event projections are version-pinned.
Unknown profile state cannot establish that a displayed seed selects the same
run as a particular player's game profile.

The source implementations require exact RNG and source-content ordering.
Normal run streams are supplied by the companion stable RNG implementation.
Map generation and Ancient option generation construct a fresh local model RNG
each time, as in the source; regenerating a map restarts its sequence without
advancing the persistent run RNG bank.
See [stable RNG runtime](stable_rng_runtime.md) for its scope and replay
compatibility. Ancient identity selection is source-backed here; this change
does not independently certify every Ancient offering effect or option filter.
Ancient offering effects and their profile predicates remain separate from
the source-backed identity allocation implemented here.

## Retained validation

`stable_encounter_source_oracle.json` contains independently compiled C
v0.107.1 encounter results and their consumed random tapes for all four acts,
using [tckmn/sts2-seed-search at cedcb54](https://github.com/tckmn/sts2-seed-search/tree/cedcb540eca0c5393b883cd847d975df5e97bf64).
The independent C normal and elite sequences, boss selection and consumed
samples agree with this implementation.

`stable_map_source_oracle.json` contains twelve complete layouts produced by
the external Python source port with a separately written sample-based random
tape adapter. Expected nodes, edges and draw counts cover all four acts at A0
and A1, plus four Spoils layouts across single-player and multiplayer. The
external Spoils port shuffled assigned nodes, contrary to the pinned C#;
the fixture adapter corrects that candidate filter before comparison. A further
30 standard and 30 Spoils complete stable RNG seeded layouts were compared
with the independent port during development. These are source algorithm cross-checks,
not captured game runs; the external port's obsolete RNG was not used.

Public regressions exercise late-floor first weak combats, elite/normal cursor
separation, Underdocks pool binding, event queue skipping, profile gates,
multiplayer room counts, final double bosses and identical serialized reload
continuations. Separate pure tests cover pruning/repair/coordinate transforms,
bag refill/rejection/overlap fallback, boss discovery and relic deque boundaries.
