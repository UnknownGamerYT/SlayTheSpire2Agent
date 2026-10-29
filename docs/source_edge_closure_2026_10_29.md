# Stable source edge closure — 29 October 2026

Target: user-selected stable v0.107.1, singleplayer. The public source reference
is `zhiyue/sts2-rl-agent@5d874e855ae1d0550d5c30416fadd7dbb0d133b7`, labeled
Steam build 23811903. That source label does not authenticate an installed game.

The [machine-readable ledger](source_edge_closure_2026_10_29.json) records a
finite ID for each follow-up edge, its status, source evidence and retained
regressions. A closed known queue means those identified differences have been
implemented and checked. It does not establish that every interaction has been
discovered or that the simulator is identical to the game.

## Follow-up scope

| Area | Identified edges and evidence |
| --- | --- |
| Potion inventory and generation | Physical slot holes through consumption, discard, Fairy, procurement, generated potions, replay and observations; native removal locks; unique factory batches; Coffer's four preferred new slots; White Beast odds/pity. IDs POT-01–03 and RNG-01. |
| Merchant and treasure | Explicit missing-rarity failures, Courier slot/type/rarity/exclusion/draw contracts and sale lifecycle; hidden chest preparation, mandatory explicit opening, immediate gold and each Spoils Map copy. IDs SHOP-01–03 and TRS-01–02. |
| Combat timing and costs | Extra player turns and side-specific expiry; automatic continuation after retention choices; sequential Stampede autoplays; ordered temporary Star and Banshee Energy costs; one-play exhaust; consistent multi-hit totals and one-time live intent modifiers; source DeathBlow attack/self-kill semantics; Royalties, Infused Core, all five orb lifecycles and their awaited callbacks. IDs CBT-01–05, CST-01–02, CRD-01 and INT-01–02. |
| Rewards and events | Manual versus returned physical card rewards; tutorial presets; factory fallback/failure; ordered base, FIFO extra and acquisition-ordered hook rewards; Trial confirm/cancel; serialized native callback awaits and custom reward completion; Wongo run/profile separation. IDs RWD-01–05 and EVT-01–04. |
| Bridge and policy inputs | Pure GET observations, explicit chest commands, observed all-character combat and relic history import, checkpoint-bound comparison cases, canonical observed action encoding, saved-policy selection across supported noncombat controls, repeatable chosen event controls, and the exclusive Trial abandon-confirmation modal. IDs BRG-01–07 and LEARN-01. |

The 38 identified edges are resolved in source revision
`d78cc048596855db2f46f01cadb9490a005987b2`, including the two explicitly modeled
source faults below. The DeathBlow correction recognizes the attack
subtype used by Gas Bomb and Waterfall Giant, applies its damage before the
owner dies, and includes its threat in observed features. A finite source scan
covers 121 monster files and 333 MoveState constructors; their attack lists have
at most one attack intent. Heterogeneous attack-list rejection is a defensive
invalid-input check, rather than a missing shipped move implementation.
The final bridge correction exports Trial's actual abandon-confirmation popup,
provides Confirm/Cancel commands, and blocks underlying room/potion controls.

## Intentional source-fault handling

Kaleidoscope/Driftwood's empty custom reroll and Driftwood/Pael's Wing's excess
alternatives have source-defined exception paths. Oracle inspectors reproduce
their defined prefix and partial state without inventing a normal successor.
The training mask deliberately avoids those faulting selections. Other legal
rewards and applicable reward-set closing actions remain available. See IDs
FLT-01–02 and [source reward faults](source_reward_faults.md).

The `TEST` character and explicit legacy MT19937 saves retain their documented
synthetic compatibility behavior. They do not represent a playable character
or a source-seed equivalence mode.

Older saves can lack the new mixed-reward enqueue order or qualifying damage
history. They use explicit compatibility fallbacks because unrecorded past
state cannot be recovered. New runs record those source facts. See
[reward population](source_reward_population.md) and
[orb lifecycle](stable_orb_lifecycle.md).

## Installed-game work

The remaining external comparison workflow runs on the user's Windows Steam
installation: fingerprint the selected stable build, compile/load the patched
bridge against its assemblies, capture before/action/after decisions, and run
the declared-field/RNG comparisons. Actual UI and task recovery after source
exceptions also needs an installed-game capture. See
[Windows stable validation](windows_stable_validation.md).

No game process has run in this cloud workspace, and the bridge has not been
built against the user's installed assemblies. Complete multiplayer, exhaustive
interactions, every whole-run seed trace and a capable trained agent are outside
the established evidence.

## Validation

Latest source: `d78cc048596855db2f46f01cadb9490a005987b2`.
Frozen engine validation: `e3fbd1d13c5beaaf08d695c8f6fddd3fcc5941c5`.
Its intent integration suite passes 83 tests; Ruff and strict mypy on 193 source
files pass. A fresh CPU PPO smoke ran 12 steps and then 12 resumed steps with
exact model, optimizer, RNG, rewards and seeds against 24 uninterrupted steps.
Two repeated held-out evaluations preserved checkpoint bytes and reports, with
no evaluation errors. Zero target successes are expected for this short
pipeline smoke; these weights have not learned useful gameplay. Actual saved
weights also selected observed map/reward controls without a running game.

The ordinary random-action check ran 25 runs with 1,609 actions and 105 reload
comparisons, with no consistency findings; all 25 runs died. Separate synthetic
progression used artificial HP, Energy and combat-stat boosts: ten runs traversed
all three acts, including each A10 double final boss, over 3,194 actions and
1,096 reload comparisons with no issues.

Additional bounded random-action QA used only an initial HP/maximum-HP boost to
10,000: 25 runs, 9,655 actions and 614 reload comparisons had no consistency
findings. Twenty-two runs reached the 400-action cap and three reached the
20-second wall-time cap; the deepest reached Act 2. These are capped robustness
checks. Neither boosted suite establishes a normal-game win rate.

The UI-only bridge correction at `d78cc04` passes all 227 live-tooling tests,
Ruff and strict mypy. The actual R3-trained checkpoint selects Confirm and
Cancel while underlying actions remain blocked; its bytes and engine logic
checksum remain unchanged. The latest bridge patch passes six preparation and
migration cases with 81 retained source hashes. It has not been compiled or
tested in the game.

Earlier local evidence remains bound to its recorded revisions: 6,419 full-suite
tests at `91e7869`, followed by 206 live-tooling tests at `675da70`. Those results
are not attributed to the later DeathBlow engine change. All-tests CI and rebuilt
distribution verification for the final published head are recorded separately
in [draft PR #2](https://github.com/UnknownGamerYT/SlayTheSpire2Agent/pull/2)
and delivered publication metadata.

The [final validation report](validation/stable_source_training_2026_10_29_final.json)
retains the latest runtime checks. The
[earlier 29 October report](validation/stable_source_training_2026_10_29.json) and
[28 October report](validation/stable_source_training_2026_10_28.json) retain
their actual historical revisions and test-only settings. The final engine
correction changes the logic checksum; start fresh checkpoints after this update.
