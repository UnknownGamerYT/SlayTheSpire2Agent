# sts2sim

Headless Slay the Spire 2 simulator and tooling for deterministic runs,
content validation, parity checks, and training experiments.

This repository focuses on local simulation and developer tools. Graphics and
game assets are out of scope. Implementation notes and current project direction
live in [HANDOFF.md](HANDOFF.md).

The simulator is still undergoing game-fidelity validation. See the
[current stable parity review](docs/stable_game_parity_review.md) for confirmed
source-rule corrections and remaining equivalence gaps.

The [October 2026 status report](docs/project_status_2026_10_28.md) records the
stable data refresh and remaining fidelity gaps. The simulator targets stable
v0.107.1; the v0.111.0 beta is a separate data/game channel.

Normal runs use the source xoshiro256** RNG streams and stable run-generation
rules. [Per-item source reviews](docs/item_source_audit/README.md) and action
regressions record what was checked. For your Windows Steam installation, use
[the stable game validation guide](docs/windows_stable_validation.md).

## Requirements

- Windows or Linux with Python 3.12 or newer.
- `uv` for dependency management.
- Optional: a CUDA-capable PyTorch setup for faster PPO training.

Install `uv` on Windows if it is not already available:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then restart the terminal, or add it to the current session:

```powershell
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
```

## Setup

Basic install:

```powershell
uv sync
```

Install development tools:

```powershell
uv sync --extra dev
```

Install training dependencies:

```powershell
uv sync --extra rl
```

Install everything commonly used while developing and training:

```powershell
uv sync --extra dev --extra rl
```

If `uv` warns that hardlinking failed, the install is still valid. To suppress
that warning:

```powershell
uv sync --extra dev --extra rl --link-mode=copy
```

## Quick Checks

Create the local source cache before testing or training. It is intentionally
excluded from Git, so this step is required after cloning:

```powershell
uv run sts2sim sync-data --data-dir data/cache --manifest data/manifest.json
uv run sts2sim training-check
```

`training-check` works offline and exits with code 1 if RL packages, cached files,
or manifest checksums are missing or inconsistent. A passing check verifies local
inputs; it does not certify that all simulator mechanics match the game.

```powershell
uv run sts2sim --help
uv run pytest -q
uv run ruff check
uv run mypy
```

If you only installed `--extra rl`, `pytest`, `ruff`, and `mypy` will not be
available. Run `uv sync --extra dev --extra rl` before using those checks.

## CLI Overview

Show every command:

```powershell
uv run sts2sim --help
```

Useful command families:

```powershell
uv run sts2sim audit-coverage
uv run sts2sim audit-events
uv run sts2sim audit-combat
uv run sts2sim audit-cards
uv run sts2sim play-run --help
uv run sts2sim replay --help
uv run sts2sim learning-progress-report --help
```

## Simulator API

```python
from sts2sim import legal_actions, load_state, new_run, serialize, step

state = new_run(seed=1, character_id="IRONCLAD", ascension=0)
action = legal_actions(state)[0]
transition = step(state, action)
payload = serialize(transition.state)
restored = load_state(payload)
```

## Map Preview

```powershell
uv run python visualize_map.py
uv run python visualize_map.py 12345
uv run python visualize_map.py --seed 12345 --character DEFECT
uv run python visualize_map.py --seed 12345 --output generated_maps/map.txt
```

## Shop Tester

```powershell
uv run python shop_test.py
uv run python shop_test.py --web
uv run python shop_test.py --seed 12345 --gold 500 --potion foul_potion
uv run python shop_test.py --web --relic the_courier
```

The shop tester uses the same simulator actions as normal runs, including card
removal, cards, colorless cards, relics, potions, Membership Card, Smiling Mask,
The Courier restocks, and Foul Potion throws.

## Combat Tester

```powershell
uv run python combat_test.py
uv run python combat_test.py --web
uv run python combat_test.py --web --seed 12345 --relic anchor --potion fire_potion
uv run python combat_test.py --web --character SILENT --ascension 6
```

The combat tester starts a deterministic one-room fight and drives the same
`legal_actions` and `step` API used by the simulator. The web view includes
hands, piles, deck state, enemy state, orbs, relics, potions, and optional debug
tools.

## Combat Reward Tester

```powershell
uv run python combat_reward_test.py --web
uv run python combat_reward_test.py --encounter elite --seed 12345
uv run python combat_reward_test.py --web --encounter event --event-preset fake_merchant
```

Use this to inspect post-fight reward generation for monster, elite, boss, and
event encounters.

## Parity And Live Capture

```powershell
uv run sts2sim trace-template --output traces/example.parity.json
uv run sts2sim compare-trace traces/example.parity.json
uv run sts2sim compare-trace traces/example.parity.json --mode exact
uv run sts2sim find-run-files "path/to/saves/history"
uv run sts2sim import-run "path/to/run.run" --trace-output traces/from-run.parity.json
uv run sts2sim probe-live
uv run sts2sim capture-live --output traces/live-state.parity.json
uv run sts2sim live-play --max-steps 5 --policy first --output traces/live-play.parity.json
```

Parity traces are sparse by default. A `before`, `after`, or `initial_state`
snapshot only needs the fields you want to verify. Use `--mode exact` when a
snapshot should match the simulator payload without extra fields.

For live captures, the CLI knows the common STS2MCP and STS2-Agent bridge ports.
Use `--base-url http://localhost:15526` or
`--base-url http://127.0.0.1:8080` to force a specific bridge.

## Training

Install RL dependencies first:

```powershell
uv sync --extra rl
```

First verify a short training run on CPU:

```powershell
uv run sts2sim training-check
uv run sts2sim train-masked-ppo --max-batches 1 --train-runs-per-batch 1 --train-max-steps 12 --eval-runs 1 --eval-max-steps 12 --champion-eval-runs 1 --hidden-size 32 --hidden-layers 1 --head-hidden-layers 1 --ppo-epochs 1 --minibatch-size 16 --device cpu --rollout-workers 1 --history-mode off --seed smoke --no-resume --model-output checkpoints/smoke.pt --output reports/smoke.json --progress-output reports/smoke_progress.json --report-output reports/smoke.html
```

Repeat with `--resume` and the same paths to check continuation. This short run
checks rollout collection, optimization, evaluation, checkpointing, and reports;
it is too short to learn useful gameplay. For small CPU experiments, setting
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before launching can reduce thread overhead.

Fresh runs now seed model initialization, action sampling, and minibatch order.
PPO bootstraps time-limited rollouts from their final state and isolates interrupted
episodes. Parallel rollout scheduling and CUDA kernels can still introduce
variation. Teacher-selected actions are excluded from the PPO actor and entropy
losses. `--imitation-coef` controls the separate teacher loss; use `--teacher-mix 0`
for an unassisted baseline (also set `--imitation-coef 0` if you previously
enabled imitation). Policy observations hide RNG state, future draws,
and queued enemy moves; see [the observation contract](docs/policy_visibility.md).

Checkpoints record source-data hashes, model architecture, observation schema,
and simulator logic. Resume/inference rejects incompatible, corrupt, nonfinite,
or Git LFS pointer files before training or sending a game action. Start a fresh
experiment after changing the simulator or its source snapshot.

Evaluate a saved checkpoint on a fixed, repeatable seed suite without training:

```powershell
uv run sts2sim evaluate-ppo --checkpoint checkpoints/smoke.pt --seed heldout-v1 --runs 20 --character IRONCLAD --device cpu --output reports/smoke_eval.json
```

Keep evaluation seeds separate from your training and champion-selection seeds.
The report records each actual episode seed and leaves checkpoint bytes unchanged.

To use saved weights with the game, follow the
[live bridge setup guide](docs/live_bridge_setup.md) to download the pinned bridge,
apply its observation patch, and build/install it against your game. Once a
supported Ironclad or Silent combat turn is visible:

```powershell
uv run sts2sim doctor-live-bridge --base-url http://localhost:15526
uv run sts2sim play-live-agent --base-url http://localhost:15526 --checkpoint checkpoints/smoke.pt --device cpu --character IRONCLAD --ascension 0 --max-steps 20 --output traces/live-checkpoint.json
```

Unsupported observations stop with a diagnostic. The guide explains the explicit
`--heuristic-fallback` option and how reports distinguish mixed policy play.
The smoke checkpoint verifies this workflow; it has not learned useful gameplay.

Single-target Act 1 boss training:

```powershell
uv run sts2sim train-masked-ppo --target act1-boss --until-stopped --train-runs-per-batch 16 --train-max-steps 1000 --eval-runs 8 --eval-max-steps 1000 --seed ppo-act1 --character IRONCLAD --ascension 0 --device auto --no-resume --model-output checkpoints\ppo_act1_boss.pt --output reports\ppo_act1_boss_latest.json --progress-output reports\ppo_act1_boss_progress.json --report-output reports\ppo_act1_boss_latest.html --terminal-progress
```

Curriculum training through the Act 1, Act 2, and final Act 3 boss clears:

```powershell
uv run sts2sim train-ppo-curriculum --stages act1-boss,act2-boss,act3-boss --until-stopped --run-name ppo_curriculum --train-runs-per-batch 16 --eval-runs 20 --champion-eval-runs 64 --champion-patience 3 --target-success-rate 0.95 --target-consecutive-successes 3 --seed ppo-curriculum --character IRONCLAD --ascension 0 --device auto --no-resume --output reports\ppo_curriculum_latest.json --report-output reports\ppo_curriculum_latest.html --terminal-progress
```

PPO now keeps a per-stage `*_champion.pt` checkpoint. It evaluates that policy on the
same fixed holdout seeds every batch, promotes only the champion, and rolls back the
mutable `*_latest.pt` model after three consecutively worse champion-holdout results.
Use `--champion-eval-runs` to choose the fixed holdout size and
`--champion-patience` to adjust the rollback threshold.

Resume a training run by replacing `--no-resume` with `--resume` and keeping
the same checkpoint/report paths.

For curriculum training, `--until-stopped` retries each stage from its latest
checkpoint until it meets the advancement criteria, then moves to the next act.
It exits only after the final game-clear stage succeeds or the process is stopped.

Useful training options:

- `--rollout-workers 0` auto-uses available CPU cores for rollouts.
- `--rollout-inference batched-gpu` centralizes action selection on the trainer
  device.
- `--history-mode highlights` writes compact best/worst evaluation histories.
- `--history-mode all-eval` writes histories for every evaluation run.

Open a generated report:

```powershell
Start-Process reports\ppo_curriculum_latest.html
```

## Gymnasium Integration

The existing `Sts2Env` exposes structured observations for the built-in trainer.
Use the optional numeric adapter with Gymnasium or mask-aware external libraries:

```python
from sts2sim.gymnasium_adapter import GymnasiumSts2Env

env = GymnasiumSts2Env(character_id="IRONCLAD", max_episode_steps=1000)
observation, info = env.reset(seed=42)
action = env.action_space.sample(mask=observation["action_mask"])
observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

Observations contain a float32 `vector` and int8 `action_mask`. Full structured
observations and action descriptors are in `info["rich_observation"]` and
`info["action_space"]`. `action_masks()` supports MaskablePPO-style callers.
Always select actions using the current mask and reset after an episode ends.
The adapter uses learning rewards by default and supports Gymnasium vectorization.

## Source Data

The simulator uses Spire Codex as the primary public reference for current
Slay the Spire 2 data and mechanics:

- <https://spire-codex.com/docs>
- <https://spire-codex.com/developers>
- <https://github.com/ptrlrd/spire-codex>

Runtime simulation should use local cached data, not live network fetches.

## Notes

- The selected latest checkpoint is tracked with Git LFS. Other generated
  checkpoints and reports are local outputs and are ignored by default.
- Full parity depends on content coverage. Missing cards, relics, potions,
  powers, event options, or monster moves should be treated as explicit gaps.
- For internal project state and next engineering steps, read
  [HANDOFF.md](HANDOFF.md).
