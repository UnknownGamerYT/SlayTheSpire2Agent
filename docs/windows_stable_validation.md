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

The current bridge supports limited combat comparisons. Exact future draw order,
RNG counters, every class resource, and every item interaction are not all
exported. A capture alone is not a passing action comparison. Keep the
installation fingerprint alongside each actual before/action/after trace; count
unsupported comparisons separately from matches and mismatches. Never treat a
simulator-generated trace as independent evidence of game parity.

Only JSON reports and traces are needed for review. Keep the installed game DLLs
and assets on your own machine. Cloud validation currently exercises the tool
with synthetic installation fixtures; a Windows game process has not been run.
