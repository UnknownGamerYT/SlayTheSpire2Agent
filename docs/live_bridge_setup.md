# Checkpoint inference and the live bridge

Run these commands from the project root with the `rl` dependencies installed
(`uv sync --extra rl --extra dev`). Training and fixed-seed evaluation run in the
simulator without a game process. The live bridge needs a separately installed
copy of Slay the Spire 2.

```sh
uv run sts2sim evaluate-ppo --checkpoint checkpoints/masked_ppo_latest.pt --seed acceptance-1 --runs 20 --max-steps 1200 --device cpu --output reports/checkpoint-evaluation.json
```

This evaluates saved weights without updating them. Use `train-masked-ppo` or
`train-ppo-curriculum` to produce a checkpoint first. Git LFS pointer files,
corrupt weights, and incompatible observation/action architectures are rejected
explicitly. The checkpoint supplies its model dimensions and activation.

## Get verified public bridge packages

```sh
uv run sts2sim sync-live-bridges
uv run sts2sim doctor-live-bridge --offline
```

Downloads stay in `external/live_bridges/downloads/`; no code is executed and no
files are installed into the game. SHA-256 and byte sizes are pinned in
[the registry](../external/live_bridges/bridges.json), using GitHub's published
asset digests. `--bridge all` also downloads the alternative STS2-Agent Windows
archive. Its state/action protocol differs; this project's checkpoint controller
currently targets STS2MCP.

Reviewed on 2026-10-28:

| Bridge | Published release | Reviewed source commit | Scope |
| --- | --- | --- | --- |
| [STS2MCP](https://github.com/Gennadiyev/STS2MCP) | [0.4.0](https://github.com/Gennadiyev/STS2MCP/releases/tag/0.4.0), 2026-05-05 | [`55e064850a68f3b4cde7e5fd525bf9b2dec4e885`](https://github.com/Gennadiyev/STS2MCP/tree/55e064850a68f3b4cde7e5fd525bf9b2dec4e885) | STS2MCP capture/control protocol; source patch below enables precise card identity. |
| [STS2-Agent](https://github.com/CharTyr/STS2-Agent) | [v0.17.0](https://github.com/CharTyr/STS2-Agent/releases/tag/v0.17.0), 2026-10-28 | [`89d5fd3d56c25d7d87a09dfbaa283485a6c3e1b1`](https://github.com/CharTyr/STS2-Agent/tree/89d5fd3d56c25d7d87a09dfbaa283485a6c3e1b1) | Alternative state capture; checkpoint control unsupported. |

The source pins identify reviewed code. They do not identify the source used to
compile the separately published release binaries. STS2MCP's reviewed README
reports testing against game `v0.103.2`; compatibility with another installed
game build remains unverified.

## Prepare the bridge for PPO observations

The unmodified STS2MCP source emits names/descriptions for combat piles and omits
the permanent deck. Its published package therefore lacks fields needed by the
checkpoint adapter. These are explicit unsupported observations, even though
heuristic play and state capture can still work.

```sh
uv run sts2sim prepare-live-ppo-bridge
```

This fetches the exact reviewed source commit into
`external/live_bridges/repos/STS2MCP` if missing, verifies HEAD and the patch
checksum, and applies
[the observation patch](../external/live_bridges/patches/STS2MCP-55e06485-ppo-observations.patch).
It refuses another source revision or unrelated changes. Repeating the command
with exactly this patch applied is safe. `--no-clone` checks an existing source
checkout without downloading anything; `--source-dir PATH` selects another
checkout.

The patch reuses `BuildCardInfo` for pile and permanent-deck cards and exposes
physical identities, modifiers, all five characters' resources and explicit
past-action history. It also removes side effects from state GETs and adds
explicit chest/reward-alternative commands; observation itself selects no game
action. See [schema 2](live_observation_schema_v2.md) for the exact field scope. `Player.Deck.Cards`, `CardModel.CurrentUpgradeLevel`, and the singular
`CardModel.Enchantment` property are also used by the reviewed public
[STS2-Agent run serializer](https://github.com/CharTyr/STS2-Agent/blob/89d5fd3d56c25d7d87a09dfbaa283485a6c3e1b1/STS2AIAgent/Game/GameStateService.Run.cs)
and [card modifier reader](https://github.com/CharTyr/STS2-Agent/blob/89d5fd3d56c25d7d87a09dfbaa283485a6c3e1b1/STS2AIAgent/Game/GameStateService.AgentView.cs).

Build using .NET 9 and your installed game, following the pinned
[upstream build instructions](https://github.com/Gennadiyev/STS2MCP/blob/55e064850a68f3b4cde7e5fd525bf9b2dec4e885/README.md#build--install).
In PowerShell, replace the game path with your actual installation:

```powershell
$GameDir = "C:\path\to\Slay the Spire 2"
dotnet build external/live_bridges/repos/STS2MCP/STS2_MCP.csproj -c Release -o external/live_bridges/build/STS2MCP "-p:STS2GameDir=$GameDir"
New-Item -ItemType Directory -Force "$GameDir/mods"
Copy-Item external/live_bridges/build/STS2MCP/STS2_MCP.dll "$GameDir/mods/STS2_MCP.dll"
Copy-Item external/live_bridges/repos/STS2MCP/mod_manifest.json "$GameDir/mods/STS2_MCP.json"
```

The pinned project references `sts2.dll`, `GodotSharp.dll`, and `0Harmony.dll`
from its `STS2GameDataDir`. Its defaults are:

| Platform | Data directory relative to `STS2GameDir` | Mod directory |
| --- | --- | --- |
| Windows | `data_sts2_windows_x86_64` | `<game>/mods` |
| Linux | `data_sts2_linuxbsd_x86_64` | `<game>/mods` |
| macOS | `SlayTheSpire2.app/Contents/Resources/data_sts2_macos_arm64` | `<game>/SlayTheSpire2.app/Contents/MacOS/mods` |

If your game uses a different data directory, pass
`-p:STS2GameDataDir="ACTUAL_DIRECTORY"` to `dotnet build`. The build command is
source-backed; this environment has no game assemblies or .NET SDK, so compilation
and in-game loading have not been verified here. Replacing a published DLL with
this build is necessary to include the observation patch.

Enable the mod through the game's mod flow and launch the game. Upstream starts
its HTTP server at `http://localhost:15526`. Then run:

```sh
uv run sts2sim doctor-live-bridge --base-url http://localhost:15526
uv run sts2sim capture-live --base-url http://localhost:15526 --output traces/live-state.parity.json
```

The doctor submits no actions and confirms read-only schema 2 through health
before requesting STS2MCP state. `download_ready`, `ready_for_capture`, and
`ready_for_checkpoint_control` are separate checks. Capture readiness means an
observation is available; checkpoint readiness means the current screen has
validated model inputs and mapped commands. A menu or an unhandled overlay can
still be captured without being a supported policy decision.

## Let saved weights select live actions

Enter an active run on a supported screen, then:

```sh
uv run sts2sim play-live-agent --base-url http://localhost:15526 --checkpoint checkpoints/masked_ppo_latest.pt --device cpu --no-start-if-needed --max-steps 20 --output traces/live-checkpoint.json
```

The model scores enabled actions using the training feature vocabulary. Combat
imports support Ironclad, Silent, Defect, Necrobinder/Osty and Regent, including
physical card/potion slots, observed modifiers, local energy/Star costs and
owned relic history. Card plays, end turn, manual potion use and discard map
back to explicit bridge commands. A displayed enemy attack is already modified;
the patch separately exports its current raw base so simulator previews can
apply later status changes without counting existing modifiers twice.

Outside the main combat screen, the controller builds a visible decision view
for map paths, event/ancient options and dialogue, rest sites, shop/Fake Merchant
stock, rewards/card alternatives, treasure/relic choices, card/hand/bundle
selectors and Crystal Sphere controls. It scores only current enabled controls
and observed cards/items. These views are feature containers; they are never
stepped as private simulator checkpoints. Unopened card-reward contents and
unrevealed Sphere item bounds stay hidden. The known source alternative-overflow selector
receives the same narrow training restriction as the simulator;
ordinary SpecialCardReward rows bypass that selector.

Each model decision is labeled `masked_ppo` and records its action ID and
policy scores. Missing identities, required history, invalid modifiers or
unmapped screens stop with `checkpoint_observation_unsupported` and a specific
diagnostic. Invalid checkpoints fail before contacting the game. Sorted visible
draw-pile contents provide no native future draw order to the actor; validation
RNG/odds/draw-order data remains outside model inputs.

To explicitly allow heuristic navigation on an unsupported screen:

```sh
uv run sts2sim play-live-agent --base-url http://localhost:15526 --checkpoint checkpoints/masked_ppo_latest.pt --device cpu --heuristic-fallback --max-steps 100 --output traces/live-mixed-policy.json
```

Every substituted decision is labeled `heuristic_fallback` and includes the
unsupported-observation reason. Without the flag, fallback is disabled. Inspect
the report before treating a run as a checkpoint evaluation. The optional
startup flow selects a run using the supplied `--character`, `--ascension` and
`--seed`; it does not represent learned menu play.

The live agent's lightweight comparison summaries are diagnostic. For an
actual before/action/after comparison with a complete simulator checkpoint,
fingerprinted installation and native persistent RNG, use the commands in
[Windows validation](windows_stable_validation.md).

Offline validation exercises tiny trained checkpoints, real HTTP clients against
schema fixtures, source-backed modifier/history regressions, rejection of sparse
or invalid observations, release checksums and pinned-source patch application.
Building against the installed game assemblies, loading the mod and real-game
action comparisons remain Windows work. These tests do not establish game
win rates or certify exhaustive game parity.
