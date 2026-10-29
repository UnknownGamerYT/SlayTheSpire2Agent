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
| Combat timing and costs | Extra player turns and side-specific expiry; automatic continuation after retention choices; sequential Stampede autoplays; ordered temporary Star and Banshee Energy costs; one-play exhaust; consistent multi-hit totals and one-time live intent modifiers; Royalties, Infused Core, all five orb lifecycles and their awaited callbacks. IDs CBT-01–05, CST-01–02, CRD-01 and INT-01. |
| Rewards and events | Manual versus returned physical card rewards; tutorial presets; factory fallback/failure; ordered base, FIFO extra and acquisition-ordered hook rewards; Trial confirm/cancel; serialized native callback awaits and custom reward completion; Wongo run/profile separation. IDs RWD-01–05 and EVT-01–04. |
| Bridge and policy inputs | Pure GET observations, explicit chest commands, observed all-character combat and relic history import, checkpoint-bound comparison cases, canonical observed action encoding, saved-policy selection across supported noncombat controls, and repeatable chosen event controls. IDs BRG-01–06 and LEARN-01. |

The 36 identified edges are resolved in source revision
`675da70f6f3726ed2708cf85960821c4deaa487f`, including the two explicitly modeled
source faults below. Core validation uses frozen revision `91e7869`; the final
bridge-only repeatable-control correction has separate checks at `675da70`.
Full-suite, PPO and rollout results from the earlier revision do not validate
the subsequent bridge edits.

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

Frozen core validation revision:
`91e7869b7dcfb1aaf77a5c3d72c64729f37ae066`.

The full suite passed **6,419 tests**, with one expected skip and two Gymnasium
checker warnings, in 711 seconds.

Completed checks at this revision include Ruff, strict mypy on 193 source
files, source/wheel builds, byte-identical packaged source and isolated wheel
CLI startup. The CPU PPO smoke ran 12 fresh steps and 12 resumed steps with
exact model, optimizer, RNG, rewards and seeds against 24 uninterrupted steps.
Two repeated held-out evaluations preserved checkpoint bytes and reports, with
no evaluation errors. Their zero target successes are expected for this short
pipeline smoke; these weights have not learned useful gameplay. An older
checkpoint was rejected for the changed simulator logic.

Normal random-action QA ran 25 ordinary runs with 1,609 actions and 105 reload
comparisons, with no consistency findings; all 25 runs died. Separate synthetic
progression used artificial HP, Energy and combat-stat boosts: ten runs traversed
all three acts, including each A10 double final boss, over 3,194 actions and
1,096 reload comparisons with no issues. Those synthetic results establish
bounded consistency rather than a normal-game win rate.

Additional bounded random-action QA used only an initial HP/maximum-HP boost to
10,000: 25 runs, 9,297 actions and 591 reload comparisons had no consistency
findings. Twenty-one runs reached the 400-action cap and four reached the
20-second wall-time cap; the deepest reached Act 2. These are capped robustness
checks rather than completed ordinary runs.

The later `675da70` follow-up changes bridge control filtering while preserving
the engine logic checksum. Its full live-tooling suite passes 206 tests; Ruff
and strict mypy pass at that revision. A trained
checkpoint can select Trial's repeatable chosen control, ordinary chosen
controls remain blocked, and checkpoint bytes remain unchanged. The patched
bridge source also passes six fresh/repeat/migration preparation cases against
its pin; it has not been built or game-tested. All-tests CI for the final
published head is linked from
[draft PR #2](https://github.com/UnknownGamerYT/SlayTheSpire2Agent/pull/2).

The JSON ledger and
[validation report](validation/stable_source_training_2026_10_29.json) retain
these exact revision scopes. The
[28 October report](validation/stable_source_training_2026_10_28.json) remains
historical evidence for its own b3893e0/4004f77 revisions and test-only settings.
It does not validate later source changes. Fresh checkpoints are required after
this simulator and observation update.
