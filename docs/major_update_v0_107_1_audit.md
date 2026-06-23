# Slay the Spire 2 v0.107.1 Update Audit

Source inputs:

- Local patch notes attachment: `C:\Users\kyria\.codex\attachments\32af1130-872e-4ae5-9d31-56081cc4a19d\pasted-text.txt`
- Spire Codex sync run on 2026-06-19.
- Steam/SteamDB patch pages checked by the monster/event worker.

## Data Refresh

`uv run sts2sim sync-data --data-dir data/cache --force` refreshed the manifest.

Important count changes:

- Cards: 576 -> 577
- Relics: 293 -> 296
- Powers: 259 -> 257
- Encounters: still 87
- Monsters: still 115
- Events: still 66

The three new relic IDs from coverage are:

- `FISHING_ROD`
- `KALEIDOSCOPE`
- `SILKEN_TRESS`

`AEONGLASS`, `INFESTED_PRISM`, and `SKULKING_COLONY` are present in the source cache and remain explicitly blocked where source text implies special behavior that is not yet fully modeled.

## Implemented This Pass

- Added `xoshiro256**` as an opt-in deterministic RNG adapter, with replayable state and named streams.
- Documented that exact STS2 run-seed-to-stream derivation is still unverified.
- Fixed `audit-combat` and `audit-cards` so `--cache-dir data/cache` resolves to `data/cache/eng`.
- Updated Ancient/Neow offering pools:
  - `Fishing Rod` and `Kaleidoscope` as positive options.
  - `Pumpkin Candle`, `Seal of Gold`, and `Silken Tress` as third-option/cursed pool entries.
  - `Scroll Boxes` no longer removes all gold.
- Added Ancient markers for the new relic effects that need downstream run-state hooks.
- Updated monster/event helpers for the v0.107.1 changes:
  - Skulking Colony HP/shell/move handling.
  - Punch Construct chain and Frail behavior.
  - Haunted Ship opening Weak behavior.
  - Slippery Bridge non-repeating removal offers.
  - Multi-Lantern-Key War Historian/Repy handling.
- Wired Waterfall Giant Siphon runtime heal to `10` below boss ascension scaling and `15` at high ascension.
- Restored full card and relic audit coverage:
  - Cards: 577/577 implemented.
  - Relics: 296/296 implemented.
  - Unknown combat IDs: 0.

## Remaining Blockers

- `AEONGLASS` full boss behavior is source-present but still intentionally blocked.
- `INFESTED_PRISM` has `VITAL_SPARK` behavior marked as requiring special handling.
- `SKULKING_COLONY` still has special move/power behavior marked as a blocker in combat coverage.
- Exact STS2 xoshiro stream derivation needs live-game or source verification.
- Some new relic downstream hooks are marker-level until the broader run-state trigger layer consumes them everywhere:
  - `Fishing Rod`: every 3 normal combats upgrades a random deck card.
  - `Kaleidoscope`: creates two card rewards from other characters.
  - `Silken Tress`: first card reward receives Glam after pickup and loses all gold.
  - `Pumpkin Candle`: extinguish/kindle counter lifecycle.
  - `Seal of Gold`: turn-start gold-for-energy payment behavior.
