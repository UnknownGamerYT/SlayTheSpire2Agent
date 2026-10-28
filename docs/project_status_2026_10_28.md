# Project status: 2026-10-28

The project has a substantial deterministic simulator, legal-action observations,
heuristic policies, and masked PPO/curriculum training. Content registration is
broad, but monster and encounter coverage still limits game fidelity. Training
results measure success in this simulator; live-game success requires parity
validation against the user's installed game and bridge.

## Source data refresh

Fetched all 12 configured English datasets on 2026-10-28 using the existing
`sts2sim.data.sync.sync_all` against `https://spire-codex.com/api` (default stable
channel). The cache is local and ignored under `data/cache/eng`; the tracked
`data/manifest.json` records the refreshed counts, timestamps, source URLs and
SHA-256 checksums. All 12 cached payloads passed count/checksum validation.

Counts match the prior 2026-06-19 manifest. Seven payload hashes changed: cards,
characters, monsters, events, powers, acts, and mechanics sections. Only hashes
were retained for the previous snapshot, so a field-by-field historical diff is
unavailable. A changed hash can include metadata additions and does not by itself
establish a mechanics change. An unchanged count does not establish parity.

| Dataset | Count | Hash vs June manifest | Refreshed SHA-256 |
| --- | ---: | --- | --- |
| acts | 4 | changed | `768ccb7ab7231c8b7584eca5a0a6e4932c67dc671f0f286bb8226f2ec02a51e5` |
| ascensions | 11 | same | `f9342d130bf18981b6bc34df91da6c04379acced3e3e3105405aea1bfc0e43da` |
| cards | 577 | changed | `ae1bf6a2f52d6a80327de818c2c6c93749f97faccbf1f7df0d72164539b5f0ef` |
| characters | 5 | changed | `9be5d05020b896f438eb5356166b2e097343d0c6a842fb749ff411cf9488fe69` |
| encounters | 87 | same | `a24589ba6560dc6c120d75068b46933fd7337699a0a67e7e7254582d25eb0c04` |
| events | 66 | changed | `37be0fc3457e3e0745f098b304b49bc95c067f222a5e0af50680d5ffec69bec0` |
| mechanics_constants | 7 | same | `5bbd658f5b7d10e7c50c883adf2f2cf454757bf69007d55983a0c38f6f1efd35` |
| mechanics_sections | 27 | changed | `33ac3898bfeca90f9061f367db7e58cff4214f92f76326809b32165257dba832` |
| monsters | 115 | changed | `7be9b2b075e09cba1f46cad387581ca2b347d31821fe9e9f22dee4a9dd4372d1` |
| potions | 63 | same | `b3acfc172172201470a3386b5bff37651f4476ddcb4f34b9b10661665ed1afbe` |
| powers | 257 | changed | `58659b3f69f33a846ddfa9f36f4ccb9d2b9e6d7d4aeb45582c34c008af9348be` |
| relics | 296 | same | `ba7c9ce8db46c798437ce8899b9b805e7b1dd7d6d6e977a3ac6635c2f78425da` |

Recreate the cache and verify its manifest from the repository root:

```bash
uv run sts2sim sync-data --data-dir data/cache
uv run sts2sim audit-coverage --data-dir data/cache
uv run sts2sim audit-cards --cache-dir data/cache --summary-only
uv run sts2sim audit-combat --cache-dir data/cache --summary-only
uv run sts2sim audit-events --summary-only
```

The API default is stable. The [Codex developer documentation](https://spire-codex.com/developers)
documents `?channel=beta` as a separate selection. On the review date,
[`/api/beta/version`](https://spire-codex.com/api/beta/version) returned
`{"beta_version":"v0.111.0","render_version":"v0.111.0"}`.
The [official Steam announcements API](https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid=2868840&count=40&maxlength=500&feeds=steam_community_announcements)
listed beta v0.111.0 dated 2026-08-14 UTC and main-branch Major Update #2 v0.107.1
dated 2026-06-19 UTC. These are the latest corresponding patch entries in the
retrieved feed, not an attestation of the installed game build. The stable API
payloads do not include a verified game build ID in this manifest. This refresh
does not claim implementation of the v0.111.0 beta balance changes.

## Coverage against the refreshed stable cache

These are executable-handler audit classifications, not end-to-end parity
certifications. A registered handler can still omit an interaction or differ in
numeric balance. Run histories and live traces are needed to verify behavior.

| Combat category | Total | Implemented | Blocked | Unknown |
| --- | ---: | ---: | ---: | ---: |
| Cards | 577 | 577 | 0 | 0 |
| Relics | 296 | 296 | 0 | 0 |
| Potions | 63 | 63 | 0 | 0 |
| Monsters | 115 | 79 | 36 | 0 |
| Encounters | 87 | 53 | 34 | 0 |

The separate card audit reports 577 implemented, 0 partial, 0 missing.
The refreshed source exposes named powers for Test Subject, Lagavulin Matriarch,
and Decimillipede whose identity-specific scripts were already present. The
classifier now recognizes those existing scripts only on the corresponding
monsters. Byrdonis's `TERRITORIAL` remains blocked because its turn-end Strength
gain is not implemented.
The event audit finds 66 events: 35 primitive, 14 special, 8 stepwise, and
9 ancient-only; no optional-module import errors were reported. Event category
coverage likewise does not verify every branch or live interaction.

Examples of blocked combat content include `AEONGLASS_BOSS`,
`BATTLEWORN_DUMMY_EVENT_ENCOUNTER`, `BOWLBUGS_NORMAL`, `BOWLBUGS_WEAK`, and
`BYGONE_EFFIGY_ELITE`. Monster blockers include `AEONGLASS`, `BATTLE_FRIEND_V1`,
`BATTLE_FRIEND_V2`, `BATTLE_FRIEND_V3`, and `BOWLBUG_ROCK`. Use
`audit-combat --status blocked` for the complete current list. The older
[v0.107.1 audit](major_update_v0_107_1_audit.md) lists selected known issues and
should not be interpreted as the complete blocker inventory.

Exact run-seed/stream derivation remains unverified against STS2. Deterministic
simulator replay establishes internal reproducibility; it does not establish
that a live STS2 run with the same displayed seed has the same map, draws, rewards,
or enemy choices.

## Changes and validation in this update

- Repaired curriculum stage completion, champion option forwarding, retry
  checkpoint selection, and promotion of champion weights into stage milestones.
- Fixed PPO value bootstrapping at time limits and return isolation after
  interrupted episodes. Fresh experiment seeds now control initialization,
  sampling, and minibatch order.
- Added a real optional Gymnasium adapter with numeric observations, current
  action masks, seeded reset sequences, and vectorized-environment support.
- Added offline `training-check` dependency and source-integrity diagnostics.
- Implemented Fishing Rod's persistent three-normal-victory upgrade cycle and
  corrected source classification for already implemented monster scripts.
- Added GitHub Actions for locked dependency installation, lint, type checks,
  stable source caching, readiness checks, and tests. The hosted workflow still
  needs its first run; the equivalent checks below passed locally.

Validation in the managed Linux environment with Python 3.12.14:

- Full suite: **974 passed, 1 skipped**, with two advisory Gymnasium warnings
  about the intentionally unbounded numeric observation Box.
- Ruff passed; mypy passed across 105 source files; `git diff --check` passed.
- Readiness check passed for PyTorch 2.12.1, NumPy 2.4.6, Gymnasium 1.3.0, and all
  12 cached datasets.
- A CPU PPO smoke experiment collected 12 steps, updated the policy, evaluated
  it, and saved model/champion checkpoints and JSON/HTML reports. Resuming the
  compatible checkpoint completed a second batch: 24 steps across two runs,
  with no evaluation errors or failed-to-continue episodes.
- A separate one-stage CPU curriculum experiment trained, evaluated, saved its
  reports, and stopped cleanly when the short experiment missed its target.
  Mocked regression tests also cover successful stage advancement and champion
  promotion. No claim of learned gameplay strength follows from these smoke runs.

Local smoke artifacts are ignored under `reports/training_smoke*`,
`reports/curriculum_smoke*`, and `checkpoints/`. They are diagnostic artifacts,
not pretrained policies. The live game and bridge are not installed in this
environment, so end-to-end live play was not tested.

## Next training work

- Complete the short CPU training/checkpoint/evaluation workflow before a long
  experiment, then use fixed held-out seeds for policy comparisons.
- Inspect evaluation histories for reward pickup, deck growth, illegal or stalled
  actions, unknown content, and approximation-driven wins before tuning rewards.
- Expand blocked encounters and live trace regressions for the character and
  ascension actually being trained.
- Record the installed game build and channel alongside future live parity
  captures; keep stable and beta datasets/checkpoints separate.
- Correct teacher-mixed PPO actor updates before using nonzero `teacher_mix`;
  default training uses zero. Improve corrupt/LFS-pointer checkpoint diagnostics
  and make checkpoint benchmark seeds independent of filenames.
- Finish Pumpkin Candle's lifecycle, Kaleidoscope's cross-character reward pool,
  and Silken Tress's first-card-reward Glam effect. Relic handler registration
  does not currently mean all downstream triggers are complete.
