# sts2sim

Headless Slay the Spire 2 simulator for deterministic agent training.

The project intentionally separates source data, deterministic simulation, game
mechanics, executable content handlers, and CLI tooling. Graphics and game
assets are out of scope.

## Source Data

The simulator uses Spire Codex as the primary public reference for current
Slay the Spire 2 data and mechanics:

- <https://spire-codex.com/docs>
- <https://spire-codex.com/developers>
- <https://github.com/ptrlrd/spire-codex>

Use cached snapshots with attribution and polite API access. Runtime simulation
must use local cached JSON rather than fetching during agent training.

## Development

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run mypy src/sts2sim
```

CLI entrypoint:

```powershell
uv run sts2sim --help
```

Map text preview:

```powershell
uv run python visualize_map.py
uv run python visualize_map.py 12345
uv run python visualize_map.py --seed 12345 --output generated_maps/map.txt
uv run python visualize_map.py --seed 12345 --character DEFECT
```

Interactive shop tester:

```powershell
uv run python shop_test.py
uv run python shop_test.py --web
uv run python shop_test.py --seed 12345 --gold 500 --potion foul_potion
uv run python shop_test.py --web --relic the_courier
```

The tester uses the real simulator shop actions, so buying Membership Card,
Smiling Mask, card removal, cards, colorless cards, relics, potions, and
throwing Foul Potion all update the same state the agent will use. Use
`--relic the_courier` to test Courier restocks, since The Courier is
blacklisted from normal merchant stock.

Interactive combat reward tester:

```powershell
uv run python combat_reward_test.py --web
uv run python combat_reward_test.py --encounter elite --seed 12345
uv run python combat_reward_test.py --web --encounter event --event-preset fake_merchant
```

The combat reward tester generates the same `RewardState` used by the engine
after monster, elite, boss, and event fights. It also lists event options from
the cached Spire Codex data that mention combat or rewards, with mapped presets
for known post-fight bundles such as Fake Merchant and Battleworn Dummy.

Interactive combat tester:

```powershell
uv run python combat_test.py --web
uv run python combat_test.py --web --seed 12345 --relic anchor --potion fire_potion
uv run python combat_test.py --web --character SILENT --ascension 6
uv run python combat_test.py
```

The combat tester starts a deterministic one-room fight from the selected
character's cached starter deck, starting HP, gold, energy, and starter relics.
It drives the same `legal_actions` and `step` API that the agent will use.
The page shows the hand, draw pile, discard pile, exhaust pile, and master deck
by default, plus the active orb slots for characters and relics that use them.
Optional debug controls for infinite energy, healing, drawing cards, adding
cards to specific piles, channeling or evoking orbs, mutating player or enemy
statuses, spawning enemies, adding relics and potions, and inspecting the raw
combat payload are hidden until you click `Show Debug Tools`.

## Public API

The intended stable API is:

```python
from sts2sim import legal_actions, load_state, new_run, replay, serialize, step

state = new_run(seed=1, character_id="IRONCLAD", ascension=0)
action = legal_actions(state)[0]
transition = step(state, action)
payload = serialize(transition.state)
restored = load_state(payload)
```

Full parity is gated by content coverage. A missing card, relic, potion, power,
event option, or monster move should be reported as an explicit blocker rather
than silently approximated.
