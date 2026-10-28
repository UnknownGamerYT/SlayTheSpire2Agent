# Stable v0.107.1 fidelity review

Target selected by the user: **latest stable v0.107.1**. Reviewed 28 October 2026.

**The simulator is not certified identical or nearly identical to the game.**
Its training and replay pipeline works in the tested flows. Passing its own
regressions, importing all source IDs, and deterministic replay do not establish
that those flows match the game. No meaningful overall fidelity percentage is
currently available.

## Version evidence

A live fetch of the [official Steam announcement feed](https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid=2868840&count=50&maxlength=1500&format=json)
still identifies [Major Update #2, v0.107.1](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/1835871199305790)
as the newest announced main-branch patch, published 19 June 2026 UTC. The newest
announced beta is [v0.111.0](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/1840944183778277),
14 August 2026 UTC. A September newsletter describes the next patch as still in
internal testing. These statements verify the publicly announced target, not an
installed executable or Steam depot.

The 17 local datasets are checksum-pinned to the stable channel. Their
`game_version` fields remain `null`: the API payloads are not independently tied
to an exact executable build. Current wiki pages may describe beta mechanics;
they must be checked against stable data, version histories, and other sources.
A fresh read-only fetch of all 17 stable endpoints matched the cached canonical
JSON hashes during this review. That verifies snapshot freshness; it does not
verify the runtime implementation.

## What the previous checks established

At published revision `8f538ec`, 1,901 tests passed. Normal random-policy runs
exercised 1,662 actions and 109 reload comparisons; synthetic-HP runs exercised
3,000 actions and 180 reload comparisons. They tested internal invariants and
repeatability. The synthetic starting HP was changed deliberately, so those
runs did not establish normal-game win rates.

The source registry contained 577 cards, 296 relics, 63 potions, 115 monsters,
87 encounters and 66 events. A registered handler means some execution surface
is present. It does not mean every clause, upgrade, trigger, condition, ordering
rule and interaction has been validated. The earlier finite predicate audit
closed its recorded findings, not every printed rule on all 577 cards.

The parity comparator is available, but its synthetic regression fixtures are
not independent game results. No complete v0.107.1 game action trace has been
validated against the simulator in this workspace.

## Confirmed discrepancies and follow-up corrections

An independent source-contract review reproduced these failures through the
public action API before the follow-up changes:

| Mechanic | Source contract | Previous behavior |
| --- | --- | --- |
| Precise Cut | Damage falls for each other card in hand; upgraded calculation base changes | Static damage |
| Ashen Strike | Additional damage per exhausted card, with upgraded scaling | Static base damage |
| Helix Drill | One hit per Energy spent this turn; zero previous spending means zero hits | One unconditional hit |
| Bully | Extra damage per Vulnerable stack | Static base damage |
| Rend | Extra damage per unique debuff, with stable upgraded base/scaling | Static base damage |
| Radiate | One AoE hit per Star actually gained this turn | One unconditional hit |
| Convergence | Next-turn Energy and Stars; retain the current hand | Immediate Energy, incomplete delayed resources, no retention |
| Expect a Fight | Energy per Attack in hand, then no more Energy gain that turn | Flat Energy and no restriction |
| Panic Button | Card-origin Block restriction over the printed duration | Later Block cards still granted Block |
| Event entry | Source act/floor/gold/deck and other entry clauses | Numeric-act-only selection admitted ineligible events |
| Sea Glass | Fifteen other-character offers: five Common, five Uncommon, five Rare | Uniform selection across rarities |

The follow-up adds retained public-action regressions and exact source
projections for these rules. See `tests/combat/test_source_damage_fidelity.py`,
`tests/combat/test_source_resource_fidelity.py`,
`tests/combat/test_source_debuff_damage.py`,
`tests/combat/test_source_radiate_fidelity.py`,
`tests/events/test_event_eligibility.py`, and
`tests/rewards/test_sea_glass.py`. See also [event eligibility](event_eligibility.md)
and [Sea Glass source qualifications](sea_glass_runtime.md).

Panic Button regressions distinguish direct-card Block from Power triggers,
relics and potions, cover Artifact preventing the source-classified debuffs,
and check future Block timing. Radiate covers actual positive gains from cards,
potions, relics, combat opening and turn start; retained Stars and spending do
not count again.

Stable Rend upgrade values are cross-checked against
[Untapped's stable card page](https://sts2.untapped.gg/en/cards/colorless/rend):
18 base and 8 extra per debuff. The cached upgrade delta agrees, while its
rendered upgrade description retains a stale base value. The current wiki Rend
page shows beta values and must not replace the stable specification.

Event prerequisites with missing profile facts or ambiguous source grouping
remain visible as unknown diagnostics. The named-act selector governs event
eligibility; it does not implement complete variant-specific maps or encounter
pools. Fixed rules and unknown rules must remain distinct.

## Follow-up validation

After these corrections, the full local suite passed **2,060 tests**, with one
expected missing-PyTorch-path skip in an environment where PyTorch is installed.
Ruff, strict mypy (121 source files), source and wheel builds passed. The added
source fixtures and tests cover nine card corrections, event entry predicates
and Sea Glass's rarity split.

A fresh CPU checkpoint smoke resumed from 12 to 24 steps and matched an
independent uninterrupted run exactly, including model, optimizer, RNG, rewards
and seeds. Repeated fixed-seed evaluation preserved checkpoint bytes; all 12
inference actions were legal. A prior checkpoint was correctly rejected after
the game logic changed. These checks establish pipeline consistency only.

Updated normal random-policy QA exercised five characters over 25 runs,
1,738 actions and 113 reload comparisons with zero internal invariant findings.
All runs ended in random-policy deaths; the maximum reached floor was 11.
These are simulator consistency results, not independent game comparisons.

## Remaining barriers to an equivalence claim

- **Normal runs use MT19937.** `new_run_state` calls the default Python RNG, and
  serialized normal run state uses that algorithm. The separate xoshiro256**
  adapter is not wired into normal runs; its seed/stream derivation is explicitly
  simulator-specific. Matching the game's displayed seed cannot currently imply
  matching maps, shuffles, encounters, rewards or random effects.
- Map construction and encounter selection are not established as the game's
  exact generators or distributions. Event weights, history rules and complete
  act-variant selection also remain unverified.
- Axebot replacement waves are not implemented. Public notes describe several
  timings, but the exact stable Stock depletion and encounter-victory rules
  were not established by this review.
- Aeonglass's initial Increasing Intensity counter/update order and Kaiser
  Crab's initial facing remain unverified. Executable handlers are present;
  their existence does not validate the provisional initial assumptions.
- [Fur Coat](fur_coat_runtime.md) normal-room eligibility and exact RNG are
  unverified. Sea Glass's rarity profile has source evidence; class-selection
  weights, display order and exact RNG consumption still need independent checks.
- Supplied ally metadata supports individual effects, not a complete networked
  multiplayer game. This review does not certify co-op behavior.
- Complete base/upgraded card contracts, potions, relics, enemy moves, events and
  cross-effect ordering still need systematic independent validation. The
  source-contract failures found here show why test totals cannot replace it.

## Route to stronger verification

1. Maintain one stable version/build target and distinguish stable rules from
   beta rules and conflicting source renderings.
2. Record each item's base/upgraded behavior, costs, target rules, conditions,
   trigger timing and ordering, with independent expected outcomes and both
   positive and negative cases. Track verified scopes, not a handler percentage.
3. Implement and serialize the actual RNG streams and run generation only from
   verified engine evidence; compare known seeded checkpoints.
4. Capture matching initial states and action sequences from the stable game,
   compare after each action, and retain the first divergence as a regression.
5. Exercise all five characters and three acts, including targeted interaction
   fixtures and full runs. Report exact mismatches and untested scopes.

One candidate oracle is [wuhao21/sts2-cli](https://github.com/wuhao21/sts2-cli):
its code calls the installed game's `RunManager`, `PlayCardAction` and
`PlayerCmd.EndTurn` through a headless adapter. It requires the user's installed
game DLLs and .NET 9. Its README reports testing beta v0.111.0, not this stable
version. Its async patches and fallback paths also need review: some failure
paths force a game-over result or discard a potion. It is a research candidate,
not a validated stable oracle or an integrated backend in this project.
