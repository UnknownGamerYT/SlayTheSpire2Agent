# Project status — 28 October 2026

This is a historical baseline. Its test totals and rollout results apply only
to the revisions recorded below and in its JSON report. See the
[current source edge ledger](source_edge_closure_2026_10_29.md) and
[current fidelity review](stable_game_parity_review.md) for the later update.

The training-readiness baseline below is supplemented by the later
[stable game parity review](stable_game_parity_review.md). That review found
additional rule mismatches and explicitly does not certify game equivalence.

The training pipeline is runnable and now checks source and checkpoint compatibility.
This is a headless simulator for agent development. Simulator scores have not been
validated as live-game win rates, and this update does not supply a trained champion.

## Source snapshot

The simulator targets the public stable v0.107.1 rules. The public beta v0.111.0
is a separate channel. All 17 configured English stable datasets were refreshed on
28 October 2026. The schema-2 manifest records channel, URL, count, timestamp and
SHA-256. The API does not expose a verifiable installed game build, so
`game_version` remains `null`; the target version is not a fabricated build pin.

| Dataset | Count | SHA-256 prefix |
| --- | ---: | --- |
| acts | 4 | `768ccb7ab723` |
| afflictions | 7 | `b55258f2f86d` |
| ascensions | 11 | `f9342d130bf1` |
| cards | 577 | `ae1bf6a2f52d` |
| characters | 5 | `9be5d05020b8` |
| enchantments | 22 | `78289bf6efb9` |
| encounters | 87 | `a24589ba6560` |
| events | 66 | `37be0fc3457e` |
| intents | 14 | `5b2543cc0ba7` |
| keywords | 7 | `1646574b348e` |
| mechanics_constants | 7 | `5bbd658f5b7d` |
| mechanics_sections | 27 | `33ac3898bfec` |
| monsters | 115 | `7be9b2b075e0` |
| orbs | 5 | `dea78e4a9dca` |
| potions | 63 | `b3acfc172172` |
| powers | 257 | `58659b3f69f3` |
| relics | 296 | `ba7c9ce8db46` |

The complete hashes are in [data/manifest.json](../data/manifest.json).
Source files are excluded from Git. A fresh checkout must run:

```sh
uv sync --locked --extra dev --extra rl
uv run sts2sim sync-data
uv run sts2sim training-check
```

Sync stages and validates a complete snapshot before replacement. Stable/beta and
languages use distinct paths; URL parameters are preserved. Training verifies the
actual selected bytes and consistent stable provenance. A beta snapshot may be
inspected, but it is not declared compatible with the stable simulator.

## Training and inference

- Curriculum resumes mutable latest checkpoints, forwards champion controls, and
  promotes only the verified champion into completed stage checkpoints.
- PPO bootstraps truncated runs, isolates interrupted episodes, and excludes
  teacher-selected actions from actor/entropy objectives. Teacher imitation is a
  separate loss. Model/sampling/minibatch RNG state supports exact CPU continuation.
- Default policy observations hide the seed, RNG, replay history, draw order and
  queued enemy moves. Frozen Eye reveals the known draw order. Oracle observations
  are an explicit debugging option. See [policy visibility](policy_visibility.md).
- Checkpoints validate tensor keys, shapes, finite values, architecture, observation
  schema, source bytes and logic checksum before loading. Corrupt files and Git LFS
  pointers fail clearly. Older policies require a fresh experiment after this update.
- `evaluate-ppo` evaluates a fixed seed suite without modifying the checkpoint.
  `MaskedPpoPolicy` provides reusable inference with strict legal action masks.
- A true optional Gymnasium adapter supports numeric observations, seeded resets,
  masked sampling and vectorization. Built-in PPO uses the rich environment.

Use the CPU smoke command in [README.md](../README.md#training), then increase
training/evaluation budgets. Keep final evaluation seeds separate from training
and champion-selection suites. Reports distinguish truncations, deaths and wins;
a successful short smoke run does not establish gameplay skill.

## Source implementation follow-up

The implementation now follows the pinned public stable source snapshot rather
than inferring behavior solely from card descriptions. Normal playable runs use
source RNG primitives/streams, maps, act variants, encounter bags and relic queues.
All 577 card metadata variants, all 22 live enchantments and seven afflictions have
finite source crosswalks; enemy numeric/state/roster contracts and the complete
live relic/potion inventory have retained review records. The
[current fidelity review](stable_game_parity_review.md) links their actual scope.

Corrections include damage rounding/value props, source power classification,
ordered Energy costs, physical-card and Osty lifecycles, card pool profiles,
item pickups and trigger timing, native event flows and reward generation. The
source Ascension table is A0–A10: Tight Belt at A4 and Ascender's Bane at A5.
These records describe source implementation and checks, not full interaction
certification or installed-game parity.

## Validation and limits

The historical frozen source revision passed 6,071 tests, with one expected skip and two
Gymnasium checker warnings. Ruff, strict mypy (175 source files), source/wheel
builds, exact CPU continuation, repeated evaluation and legal inference passed.
The [bounded validation report](validation/stable_source_training_2026_10_28.json)
records the runtime revisions and normal/synthetic rollout settings. Hosted CI
and complete delivered logs are linked from draft PR #2.
Earlier 1901/2060-test baselines predate the source follow-up and must not be used
as validation of the new runtime. Checkpoint smoke and random-policy consistency
checks do not establish a useful agent or normal-game win rate.

Default observations hide RNG, future room/relic queues, draw order and future
enemy moves. Checkpoints bind to source bytes and game logic; use fresh training
outputs after this update. The cloud has no installed game or .NET/game assembly
build evidence. The user's Windows Steam machine can run the
[installation fingerprint and bridge capture steps](windows_stable_validation.md).
Exact whole-run comparisons, exhaustive interactions and full multiplayer remain
outside the established evidence. Unsupported source edge cases are explicit;
the simulator does not silently manufacture a successful game outcome.
