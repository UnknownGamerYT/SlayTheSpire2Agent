# Project status — 28 October 2026

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

## Simulator corrections

All five characters now use their source-defined HP, 99 starting gold, starter
cards and relics. Card identity allocation preserves permanent cards across
rewards, shop purchases, transformations, duplication and serialization.

The expanded runtime covers innate monster powers, source-linked debuffs, movement
and reinforcements, temporary death/respawn behavior, attack-only damage modifiers,
Buffer, physical card replay/autoplay, conditional/Fatal card outcomes, Bound,
correct combat cleanup, and additional card turn/draw/play/status triggers.
Installed Powers leave the combat piles rather than being drawn and installed
again. All 11 stable X-cost cards distinguish payment from effect scaling,
including free autoplay, replay, zero X and Chemical X for Energy or Stars.
Source upgrades apply absolute costs, keyword changes and explicit field aliases;
conditional damage upgrades no longer inflate the immediate base damage.

Run mechanics include adaptive unknown-room probabilities, Winged Boots charges,
Maw Bank/Bowler/Dragon Fruit gold hooks, Planisphere, source-backed campfire
choices, Tea Sets, Pumpkin Candle/Kindle, Fishing Rod, Toy Box, Wongo's ticket,
Scroll Boxes, starter relic replacement, chosen deck mutations and Fur Coat.

Coverage audits describe recognized execution surfaces. They are not proof that
all 577 base/upgraded cards, 296 relics, 63 potions and their interactions exactly
match the game. Bounded audits found parsed effects with missing consumers; those
findings drove explicit runtime fixes and action-based regression tests. The
[bounded evidence inventory](runtime_coverage_scope.md) distinguishes verified
predicates, spot checks and the remaining external verification questions.

## Validation and limits

The test suite includes legal actions, save/reload determinism, source-backed
mechanics, safe policy observations, PPO loss/checkpoint behavior, Gymnasium checks,
and recorded/fake bridge integration. Ruff and strict mypy are required by CI.
Final local results: 1,901 tests passed with one expected missing-dependency-path
skip, Ruff passed, strict mypy passed across 118 source files, and source/wheel
builds succeeded. The CPU smoke resumed from 12 to 24 training steps and matched
an independent uninterrupted two-batch run exactly, including optimizer and RNG
state. Repeated fixed-seed evaluation left the checkpoint unchanged; 12 loaded
policy actions were legal and masked. These runs are deliberately too short to
learn useful gameplay. Publication/tree checks and hosted CI results are recorded
in the pull request and delivered validation log.

A multi-character random-policy check exercised all five characters, 25 runs,
1,662 legal actions and 109 serialized reload comparisons without invariant or
transition-divergence findings after the final generated-ID fix. Random deaths were
expected; this is a correctness check rather than a strength benchmark.

A separate deeper check used one explicit test-only initial HP/max-HP boost to
10,000 per character, seed 1 and a 600-action cap. All five characters completed
3,000 actions and 180 reload comparisons with zero invariant findings; the
Defect run reached Act 3. This modifies the starting state and measures transition
consistency only, never normal-game survival or policy strength.

Remaining verification boundaries:

- Aeonglass's first Increasing Intensity application/update order needs a runtime
  trace. Crusher/Rocket initial facing needs a verified encounter trace.
- The mapping from displayed run seeds to the game's xoshiro streams is unverified.
  Simulator determinism does not imply identical maps, draws or rewards in the game.
- Exact Fur Coat mark eligibility/RNG and unmodeled Axebot replacement waves need
  live/source confirmation; see [the bounded Fur Coat implementation](fur_coat_runtime.md).
- Co-op modeling covers supplied ally state; a complete multi-agent game protocol
  has not been verified. Exact Sea Glass rarity odds and branch interactions need
  source evidence or traces.
- No game executable or running bridge is available in this workspace. Public
  bridge source schemas omit fields needed for lossless checkpoint inference.
  Live inference stops on incomplete states; heuristic fallback is explicit.
  Bridge setup and source-patch validation are distinct from an actual game test.

Do not remove these limits solely because a source ID has a registered handler.
Capture the installed game version/channel and compare recorded traces before
interpreting simulator training as live-game performance.
