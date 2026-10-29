# Stable v0.107.1 simulator fidelity

The target is the user-selected **stable v0.107.1**, Steam public build
**23811903**. The simulator is usable for local training experiments. It is not
certified identical or nearly identical to an installed game, and simulator
results are not established live-game win rates.

The [source edge closure ledger](source_edge_closure_2026_10_29.md) tracks the
finite follow-up queue, each implemented contract, retained source evidence and
regression paths. Its closed-queue status applies to those identified edges;
it does not imply that every possible game interaction has been discovered.

## Evidence and version binding

The implementation uses the public [version-qualified source snapshot](https://github.com/zhiyue/sts2-rl-agent/blob/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/GAME_VERSIONS.md)
at commit `5d874e855ae1d0550d5c30416fadd7dbb0d133b7`. It reports release commit
`59260271` and internal `main_assembly_hash` `-1555940892`. That integer is not a
SHA-256 digest. The third-party source label does not authenticate an installed
executable. The repository retains functional implementations, compact source
contracts, paths and hashes; game assemblies, assets and the full decompilation
are not included.

The 17 checksum-pinned stable API datasets contain 577 cards, 296 live relics,
63 live potions, 115 monsters, 87 encounters and 66 events. Their manifest
`game_version` remains null because the API does not establish a binary build.
Source registry placeholders, abstract helpers and test models are distinguished
from live content. Stable and beta caches remain separate.

Three types of evidence have different meanings: source-derived expected
outcomes check a rule, regression and replay tests check the implementation,
and installed-game before/action/after captures check actual game agreement.
A handler count or a passing test total is not a fidelity percentage.

## Implemented source contracts

| Area | Implementation and retained evidence |
| --- | --- |
| RNG | Normal playable characters use xoshiro256**, UTF-16 displayed-seed hashing, named/player/model streams, float32 draws, source sampling and .NET 9 duplicate-card sorting. Independent C vectors and serialized counters are retained. See [RNG](stable_rng_runtime.md). |
| Runs | Source map pruning/post-processing, named act variants, profile gates, encounter bags, opening rosters, event history, Ancient offers and front/back relic queues persist across reloads. See [run generation](run_generation_stable.md) and [relic queues](relic_queues_stable.md). |
| Cards | Base/upgraded metadata crosswalks cover all 577 source rows. Stateful action tests cover targeted card rules, physical-card replay, Forge/Osty/orbs, conditional damage, resource timing and ordered cost modifiers. See [core audit](source_core_card_audit.json) and [class audit](stable_class_card_audit.md). |
| Modifiers | All seven afflictions and the source keyword set have finite lifecycle audits. All 22 live enchantments have source-method scope records and public-action fixtures. Power taxonomy is separate from power behavior. See [modifiers](../tests/fixtures/source_modifier_audit.json) and [enchantments](source_enchantment_audit.json). |
| Enemies | Source HP/ascension values, state edges, initial states, numeric attacks/Block, encounter opening rosters and targeted lifecycle rules include Axebot waves, Aeonglass, Kaiser facing, summons, Osty and side timing. See [enemy contracts](stable_enemy_source_contracts.md). |
| Items | All 296 live relics and 63 live potions have finite source-method review records. Implemented corrections include pickup choices, duration/cost rules, procurement, Energy/Block/HP modifiers, source RNG and reward hooks. Each ledger states the reviewed and tested scope. See [item audit](item_source_audit/README.md) and [runtime contracts](stable_item_contracts.md). |
| Events and rewards | Native startup/model RNG, eligibility, typed choices, generated grids, recurring flows, Crystal Sphere, event fights, per-reward rerolls and saved callback continuations have retained source fixtures. See [native event completion](stable_native_event_completion.md), [reward faults](source_reward_faults.md) and the closure ledger. |
| Training | Legal action masks, hidden-information policy observations, deterministic CPU checkpoint continuation, fixed-seed inference/evaluation and source/logic compatibility checks are retained. See [policy visibility](policy_visibility.md). |

The reserved `TEST` character is a controlled synthetic fixture. It keeps legacy
setup RNG and offering behavior where needed by isolated tests. It does not
represent a playable character or game-equivalence mode. Explicit MT19937 saves
and diagnostic runs remain supported; they do not establish game seed parity.

The follow-up corrections preserve physical potion holes through use,
procurement, generated potions and Fairy activation; source chest opening and
merchant restocks; temporary Star and Energy costs; actual extra player turns;
and end-turn choices that await their child actions. Manual card rewards and
returned physical cards retain different source lifecycles. Native custom
reward screens finish their event after completion, and later event relics are
drawn after earlier pickup callbacks finish. These are source-derived fixes
with finite regression evidence, rather than results measured in the game.

## Validation boundaries

Whole-run RNG consumption and every card/relic/power ordering combination have
not been independently compared to the executable. Source contracts and
arithmetic fixes reduce concrete differences but cannot establish complete
behavioral equivalence. Source helpers, API descriptions and third-party ports
can disagree; retained fixtures identify the rule used by each correction.

Complete multiplayer synchronization, co-op targeting and all player-count
interactions are outside the verified single-player scope. Two identified
source reward failures are modeled explicitly. Kaleidoscope plus Driftwood can
reroll an empty custom pool; Driftwood plus Pael's Wing can exceed the source
limit of two card alternatives. Oracle inspectors reproduce their defined
exception prefixes without inventing a successful successor. The training mask
deliberately avoids those faulting selections while retaining unrelated legal
actions. The installed game's screen/task recovery still needs Windows traces.
The Trial Double Down popup has serialized confirm and cancel actions, including
repeatable cancellation and forced abandonment; its rendered UI sequence also
needs a Windows trace.

Bridge schema v2 separates observed policy inputs from validation-only state.
Its imports cover the retained source contracts for all five characters and
reject contradictory or missing required inputs. GET observation requests do
not open chests or Fake Merchant. An observation is still not a lossless game
checkpoint, and a sparse field match is not whole-state parity. Source patch
preparation and mocked tests are separate from compiling, loading and exercising
the bridge against the selected game build.

The user's final verification machine is **Windows with Steam installed**.
Follow [Windows stable validation](windows_stable_validation.md) to fingerprint
that installation and prepare actual comparison captures. A successful metadata
fingerprint alone is not binary authentication or a behavioral match. No game
process has been run in this cloud workspace.

Latest integrated validation is recorded in the pull request and delivered
validation reports. A short PPO smoke checks the training pipeline; it does not
produce a capable agent. Start fresh checkpoints after this logic update.

The [29 October report](validation/stable_source_training_2026_10_29.json)
records 6,419 passing tests at frozen core revision `91e7869`, followed by 206
passing live-tooling tests and static checks for the bridge-only correction at
`675da70`. PPO continuation and bounded rollout results retain their core
revision identity. Final published-head CI is separate evidence for the final
tree, including its later bridge change.

The [28 October validation report](validation/stable_source_training_2026_10_28.json)
is historical evidence for its recorded runtime revisions. The closure ledger
links the latest frozen validation report separately. Each report identifies its
actual runtime revision and test-only settings; earlier totals do not validate
later source changes.
