# SlayTheSpire2Agent Handoff

Last updated: 2026-07-05

## Goal

This repo is building a headless Slay the Spire 2 simulator plus learning agents.
The simulator should expose enough deterministic game state for agents to learn
map routing, combat, rewards, shops, events, potions, relics, and long-term deck
strategy without graphics.

## Project Shape

- `src/sts2sim/engine/`: core run/combat state models and transitions.
- `src/sts2sim/mechanics/`: game mechanics, content handlers, rewards, shops,
  events, relics, potions, cards, combat helpers, and semantic feature helpers.
- `src/sts2sim/agent_api.py`: compact/rich observation encoding, legal actions,
  action descriptors, masks, and action semantics for agents.
- `src/sts2sim/gymnasium_env.py`: Gymnasium-style environment wrapper.
- `src/sts2sim/learning/masked_ppo.py`: current PPO trainer and network.
- `src/sts2sim/learning/rewards.py`: outcome-based reward shaping.
- `src/sts2sim/history.py`: JSON/HTML run histories and compact map rendering.
- `src/sts2sim/cli/app.py`: CLI commands.
- `tests/`: split by domain; run `uv run pytest -q`.

## Current Learning Design

- PPO is the active path. Old Q-learning is baseline/reference only.
- The PPO action head scores currently legal action descriptors with action
  masking.
- The model has auxiliary planning heads:
  - `aggression_target`
  - `hp_floor`
  - `hp_spend_budget`
  - `combat_pace`
  - `route_preference`
  - `potion_policy`
  - `reward_pickiness`
  - `expected_hp_loss`
  - `expected_turns_to_kill`
  - `boss_readiness`
- Rich observations include state, action semantics, content identity vocab,
  mechanic tags, reward/path summaries, enemy traits, card/status atoms, and
  previous agent memory.

## Reward Shaping Decisions

The reward system is outcome-based and aggression-aware:

- No generic step penalty.
- No direct death penalty.
- No generic win reward.
- Gold is rewarded directly and capped because it is almost always useful.
- Cards, relics, removals, and skips do not get direct fixed rewards. Their value
  should be learned from later survival, combat strength, and boss progress.
- Combat rewards scale by room and act:
  - normal combat
  - elite combat
  - Act 1 boss
  - Act 2 boss
  - Act 3 boss / game completion
- Enemy HP progress is rewarded once per monster HP pool so healing enemies
  cannot be farmed.
- HP loss penalty and enemy damage reward are weighted by deterministic
  `combat_aggression_pressure`, not by the model's own aggression output. This
  avoids reward hacking.
- Prevented HP loss is rewarded, capped, and weighted more strongly when the
  state pressure favors safe play.
- Potion use scores through outcomes; wasteful discard can be penalized only
  when there is no slot pressure or replacement opportunity.

## History / Debugging

Best and worst PPO evaluation runs write history artifacts beside the report:

- `*_best_run_history.json`
- `*_best_run_history.html`
- `*_best_run_map.txt`
- same for `worst`

The JSON remains full fidelity. The HTML is intentionally compact:

- combat actions are grouped by turn
- all cards played in a turn are listed in order
- enemy turn / next intent is summarized
- normal steps show mostly what changed
- raw engine events and full policy outputs are collapsed in details sections

## Useful Commands

Install/sync:

```powershell
uv sync
```

Run tests:

```powershell
uv run pytest -q
uv run ruff check
uv run mypy
```

Fresh Act 1 boss training:

```powershell
uv run sts2sim train-masked-ppo --target act1-boss --until-stopped --train-runs-per-batch 16 --train-max-steps 1000 --eval-runs 8 --eval-max-steps 1000 --seed ppo-reward-v2-act1 --character IRONCLAD --ascension 0 --device auto --no-resume --model-output checkpoints\ppo_reward_v2_act1_boss.pt --output reports\ppo_reward_v2_act1_boss_latest.json --progress-output reports\ppo_reward_v2_act1_boss_progress.json --report-output reports\ppo_reward_v2_act1_boss_latest.html --terminal-progress
```

Fresh Act 2 boss training:

```powershell
uv run sts2sim train-masked-ppo --target act2-boss --until-stopped --train-runs-per-batch 16 --train-max-steps 1600 --eval-runs 8 --eval-max-steps 1600 --seed ppo-reward-v2-act2 --character IRONCLAD --ascension 0 --device auto --no-resume --model-output checkpoints\ppo_reward_v2_act2_boss.pt --output reports\ppo_reward_v2_act2_boss_latest.json --progress-output reports\ppo_reward_v2_act2_boss_progress.json --report-output reports\ppo_reward_v2_act2_boss_latest.html --terminal-progress
```

Resume the same run later by using the same command and replacing
`--no-resume` with `--resume`.

Open a generated report:

```powershell
Start-Process reports\ppo_reward_v2_act2_boss_latest.html
```

## Notes For A New Codex Session

Start by reading this file, then inspect:

```powershell
git status --short
rg -n "class LearningRewardConfig|train_masked_ppo|run_history_html|PLANNING_HEAD_SCHEMA" src tests
```

Good next work:

- Tune reward coefficients from training reports instead of guessing.
- Improve deck-performance diagnostics, not direct card/relic rewards.
- Audit histories where the agent makes strange choices and convert those into
  simulator bugs, observation gaps, or reward-shaping diagnostics.
- Continue expanding combat/relic/potion/event parity as missing interactions
  show up in histories.
