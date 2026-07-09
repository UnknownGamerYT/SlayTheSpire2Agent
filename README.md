# sts2sim

Headless Slay the Spire 2 simulator and tooling for deterministic runs,
content validation, parity checks, and training experiments.

This repository focuses on local simulation and developer tools. Graphics and
game assets are out of scope. Implementation notes and current project direction
live in [HANDOFF.md](HANDOFF.md).

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

Single-target Act 1 boss training:

```powershell
uv run sts2sim train-masked-ppo --target act1-boss --until-stopped --train-runs-per-batch 16 --train-max-steps 1000 --eval-runs 8 --eval-max-steps 1000 --seed ppo-act1 --character IRONCLAD --ascension 0 --device auto --no-resume --model-output checkpoints\ppo_act1_boss.pt --output reports\ppo_act1_boss_latest.json --progress-output reports\ppo_act1_boss_progress.json --report-output reports\ppo_act1_boss_latest.html --terminal-progress
```

Curriculum training through Act 1, Act 2, Act 3, and game completion:

```powershell
uv run sts2sim train-ppo-curriculum --stages act1-boss,act2-boss,act3-boss,game-clear --run-name ppo_curriculum --train-runs-per-batch 16 --eval-runs 20 --target-success-rate 0.95 --target-consecutive-successes 3 --seed ppo-curriculum --character IRONCLAD --ascension 0 --device auto --no-resume --output reports\ppo_curriculum_latest.json --report-output reports\ppo_curriculum_latest.html --terminal-progress
```

Resume a training run by replacing `--no-resume` with `--resume` and keeping
the same checkpoint/report paths.

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
