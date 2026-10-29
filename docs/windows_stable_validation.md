# Validate the Windows stable game

The simulator targets **v0.107.1**, Steam public build **23811903**. The public
[source snapshot](https://github.com/zhiyue/sts2-rl-agent/blob/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/GAME_VERSIONS.md)
reports release commit `59260271` and internal `main_assembly_hash` `-1555940892`.
That integer is the game's build identifier, not a SHA-256 checksum. The source
snapshot's version label still needs comparison with an installed game.

In PowerShell, from the project directory after installing
[uv](https://docs.astral.sh/uv/getting-started/installation/):

```powershell
uv sync --locked --extra dev --extra rl
$GameDir = "C:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2"
uv run sts2sim inspect-game-installation --game-dir "$GameDir" --output reports/windows-game-installation.json
```

Use your actual Steam library path. The command reads `release_info.json`, finds
the library's `appmanifest_2868840.acf`, and records the local assembly's SHA-256
without loading it. `--steam-manifest PATH` overrides the manifest location.
A wrong build, pending different update, missing metadata, or ambiguous assembly
returns a failing exit code. A successful report confirms metadata agreement;
it does not certify the binary or simulator behavior.

Follow [the bridge setup](live_bridge_setup.md) to build the pinned, patched
STS2MCP source against this installation. Then check and capture a combat state:

```powershell
uv run sts2sim doctor-live-bridge --base-url http://localhost:15526
uv run sts2sim capture-live --base-url http://localhost:15526 --output reports/windows-stable-capture.json
```

The schema-2 patch exports all five characters' observed resources, physical
card identities/modifiers, orb/Osty state, separate round/player turns, and
native RNG counters under a separate `validation` object. These oracle fields
are used for comparisons and excluded from policy input. A capture alone is
not a passing action comparison.

To compare one action, prepare a JSON plan with these fields:

```json
{
  "schema_version": 1,
  "case_id": "ironclad-bash-01",
  "game_version": "0.107.1",
  "installation": {},
  "simulator_before": {},
  "simulator_action": {
    "type": "play_card",
    "card_instance_id": "YOUR_SIMULATOR_CARD_INSTANCE",
    "target_id": "YOUR_SIMULATOR_TARGET"
  },
  "external_action": {
    "action": "play_card",
    "card_index": 0,
    "target": "YOUR_OBSERVED_BRIDGE_TARGET"
  },
  "compare_rng": true,
  "fields": []
}
```

Replace `installation` with the complete successful fingerprint report and
`simulator_before` with the complete `sts2sim.api.serialize(state)` result from
the matching simulator run, at the same decision. Match the selected card and
target in both actions. A normalized capture or imported sparse observation is
not a complete simulator checkpoint. Keep the same seed and prior choices when
preparing both runs; a starting-state or RNG mismatch prevents execution.

Optional `fields` add explicit bindings for behavior outside the default
projection. For example, an entry with `live_path` equal to
`player.hand[0].dynamic_vars.Damage`, `simulator_path` pointing to the matching
serialized simulator value, and `numeric: true` compares decimal values to exact
fractions. Paths must exist on both sides; an absent field is unsupported.

Save the plan as `reports/comparison-plan.json`, then:

```powershell
uv run sts2sim capture-live-comparison reports/comparison-plan.json --base-url http://localhost:15526 --execute --output reports/windows-comparison.json
uv run sts2sim compare-live-comparison reports/windows-comparison.json --output reports/windows-comparison-replay.json
```

The first command submits exactly the planned game action after checking the
fingerprint, schema and starting projection. It waits for source effect/enemy
resolution or a visible child choice and records raw before/action/after plus
a replay report. The second command reruns that simulator step without touching
the game. Nonmatching or unsupported cases return a failing exit status.

The default projection compares observed player/monster state, inventories,
true/fake upgrades, local costs and modifiers, class resources, native draw order,
and persistent run/player plus retained observed event RNG counters. Raw dynamic
values and saved-model oracle fields remain available for explicit bindings.
It does not reconstruct private pending callbacks or prove every interaction.
Count unsupported cases separately from matches and mismatches. Keep the
installation fingerprint alongside each trace and never treat a
simulator-generated fixture as independent game evidence.

Only JSON reports and traces are needed for review. Keep the installed game DLLs
and assets on your own machine. Cloud validation currently exercises the tool
with synthetic installation fixtures; a Windows game process has not been run.
