# Live bridge references

The tracked [registry](bridges.json) pins reviewed public source commits and
published release assets separately. Cloned sources in `repos/`, staged release
files in `downloads/`, and build outputs are ignored by git. No licensed game
files are included.

| Name | Reviewed source | Source pin | Integration |
| --- | --- | --- | --- |
| STS2MCP | <https://github.com/Gennadiyev/STS2MCP> | `55e064850a68f3b4cde7e5fd525bf9b2dec4e885` | Preferred REST protocol at `http://localhost:15526`; source observation patch needed for checkpoint inputs. |
| STS2-Agent | <https://github.com/CharTyr/STS2-Agent> | `89d5fd3d56c25d7d87a09dfbaa283485a6c3e1b1` | Alternative capture API at `http://127.0.0.1:8080`; checkpoint control unsupported. |
| BoberInSpire | <https://github.com/S0ul3r/BoberInSpire> | `67263f2` (older reference) | Read-only JSON export / WebSocket overlay reference. |
| CLI-Anything | <https://github.com/HKUDS/CLI-Anything> | `bf3cc39` (older reference) | Command-line bridge reference. |

Reproducible download and source preparation:

```sh
uv run sts2sim sync-live-bridges
uv run sts2sim doctor-live-bridge --offline
uv run sts2sim prepare-live-ppo-bridge
```

Downloads are checksum verified and staged without installing or executing them.
The source preparation command verifies the exact reviewed commit, checks the
[patch](patches/STS2MCP-55e06485-ppo-observations.patch) checksum, and applies it.
Compilation requires .NET 9 and assemblies from the user's own installed game.
See [the complete setup guide](../../docs/live_bridge_setup.md) for the build,
installation, checkpoint evaluation, and live inference commands and their limits.

Once the modded game is running:

```sh
uv run sts2sim probe-live
uv run sts2sim doctor-live-bridge --base-url http://localhost:15526
uv run sts2sim capture-live --base-url http://localhost:15526 --output traces/live-state.parity.json
```

Both published release downloads and fresh pinned-source patch application were
verified on 2026-10-28. No game process or game assemblies were available in this
environment, so compilation and actual game control remain unverified.
