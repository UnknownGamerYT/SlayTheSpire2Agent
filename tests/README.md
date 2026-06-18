# Test Layout

Tests are grouped by simulator domain so VS Code and pytest discovery are easier to scan.

- `combat/`: combat engine, cards, monsters, statuses, combat UI payloads, combat potion/relic hooks.
- `content/`: parsers and handlers for cards, relics, potions, monsters, powers, and ascension rules.
- `data/`: data sync and manifest behavior.
- `events/`: event catalog parsing, event effects, event flows, and event-specific logic.
- `interfaces/`: public agent, CLI, and Gymnasium-facing interfaces.
- `rewards/`: reward generation, reward room state, and combat reward tester payloads.
- `rooms/`: shop, campfire, treasure, and room-specific engines.
- `run/`: map generation, replay/RNG, visualizer, and full run progression.

Shared helpers stay in the root of `tests/`.
