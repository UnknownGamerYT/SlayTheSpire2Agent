# Stable bridge observation schema 2

The reviewed STS2MCP source pin remains
`55e064850a68f3b4cde7e5fd525bf9b2dec4e885`. The prepared patch now exports
`observation_schema_version: 2`. It is checked against the public stable source
`5d874e855ae1d0550d5c30416fadd7dbb0d133b7`; source paths/hashes and the finite
contracts are retained in `tests/fixtures/bridge_observation_v2_source.json`.

Cards expose stable opaque object IDs and their permanent-deck object ID,
actual upgrade level, enchantment/affliction ID and amount, base dynamic values,
base/local/final costs, ordered local cost modifiers, temporary Star costs,
exact keyword IDs and transient card flags. Wither's fake upgrade count is
separate from actual upgrade level. IDs correlate observed objects within a
running bridge process; they are not a portable game checkpoint.

Combat exports explicit zero/empty values for Stars, orb slots/orbs and pets,
the source history's finished-play counters and current-turn play identities.
Glass exports its raw accumulator separately from the Focus-modified display.
Power `amount` is the actual stored Amount; `display_amount` retains its UI value.
Relic display counters remain UI observations and must not be interpreted as
arbitrary internal relic counters.

Event data comes from the current mutable event rather than the canonical
model. Current page/body key, values, stable option keys and offered relic IDs
are exported. The run includes current room type and action-running state;
game-over observations include the actual recorded win result.

The `validation` object is oracle evidence for comparisons. It contains native
run/player RNG counters, retained observed event RNG counters, native draw order,
source odds and serialized saved relic properties. It must stay outside actor
observations. Matching these fields does not provide a lossless combat checkpoint
or authenticate an executable.

GET state no longer opens treasure chests or the Fake Merchant UI. An unopened
chest exposes `can_open_chest` and has no visible relic list. The explicit
`open_chest` action clicks an enabled chest; it is separate from claiming a
revealed relic. Shop purchases still open their required UI through the existing
action executor. These changes prevent a read-only capture from consuming chest
reward RNG or selecting a game action.

`prepare-live-ppo-bridge` can upgrade the exact previous reviewed observation
patch to this version. It verifies both checksums, rejects unrelated changes,
and restores the previous patch if applying the new one fails. Repeating a
completed preparation is idempotent. The old reviewed patch is retained only as
a migration baseline.

Patch application and contract tests run offline. Building/loading against the
installed `sts2.dll`, Godot/Harmony assemblies and actual game observations
remain Windows checks. None of these source checks certify game equivalence.
