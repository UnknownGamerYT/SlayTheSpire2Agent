# Live Bridge References

This folder tracks notes for third-party Slay the Spire 2 live-state bridges.
The actual cloned repositories live under `external/live_bridges/repos/`, which
is intentionally ignored by git.

These projects were downloaded as shallow clones for local reference only. Do
not run or build them blindly; read their install instructions and use a
throwaway/test run when enabling game-control features.

## Downloaded Repositories

| Name | Local path | Source | Commit | Main use |
| --- | --- | --- | --- | --- |
| STS2MCP | `repos/STS2MCP` | <https://github.com/Gennadiyev/STS2MCP> | `20eadeb` | Full localhost REST bridge and MCP server. Default HTTP server: `http://localhost:15526`. |
| STS2-Agent | `repos/STS2-Agent` | <https://github.com/CharTyr/STS2-Agent> | `2617fb1` | Live state, legal actions, and safe action execution. Health endpoint: `http://127.0.0.1:8080/health`. |
| BoberInSpire | `repos/BoberInSpire` | <https://github.com/S0ul3r/BoberInSpire> | `67263f2` | Assistant/overlay pattern. Exports combat/merchant state to JSON and streams via WebSocket. |
| CLI-Anything | `repos/CLI-Anything` | <https://github.com/HKUDS/CLI-Anything> | `bf3cc39` | CLI bridge pattern around a local `STS2_Bridge` HTTP API at `localhost:15526`. |

## Our Integration Commands

Probe common bridge ports:

```powershell
uv run sts2sim probe-live
```

Capture state from whichever known bridge is currently reachable:

```powershell
uv run sts2sim capture-live --output traces/live-state.parity.json
```

Let a tiny policy take a few live actions and record a trace:

```powershell
uv run sts2sim live-play --max-steps 5 --policy first --output traces/live-play.parity.json
```

If auto-detection fails, provide the bridge explicitly:

```powershell
uv run sts2sim capture-live --base-url http://localhost:15526
uv run sts2sim capture-live --base-url http://127.0.0.1:8080
```

