# Fur Coat stable runtime contract

The stable source marks **seven Monster or Elite map points** in the pickup act.
It includes all grid points of those types, with no completed-point or future-floor
filter. Boss and Unknown points are excluded. It excludes points already marked
by another Fur Coat. Selection enumerates the grid column first, then row, runs
an unstable shuffle using independent `Rng(owner, FUR_COAT)`, and takes seven.
The model RNG consumes no persistent run-bank draws. Legacy imported MT19937 saves
retain their existing RNG adapter; they do not claim game-seed parity.

Selected points are stored by act in `fur_coat_marked_nodes`. Entering a marked
Monster/Elite room sets enemies' **current HP** to one and preserves maximum HP.
The shared monster-creation path applies this to later summons and restocks.
Healing can restore HP above one. Test Subject's same-creature revive sets maximum
HP and heals; it does not emit the creature-added hook, so Fur Coat does not repeat.

The stable source is the publicly available decompilation at
[zhiyue/sts2-rl-agent, 5d874e8](https://github.com/zhiyue/sts2-rl-agent/tree/5d874e855ae1d0550d5c30416fadd7dbb0d133b7).
Its `GAME_VERSIONS.md` identifies public v0.107.1, Steam build 23811903,
release commit `59260271`, assembly hash `-1555940892`. This is repository-author
version evidence; this workspace has not checked an installed DLL.

Reviewed files: `Models.Relics/FurCoat.cs`, `Map/ActMap.cs`, `Map/MapPointType.cs`,
and `Models.Monsters/TestSubject.cs`, under `decompiled/MegaCrit.Sts2.Core.*`.
File SHA-256 values and compact source projections are retained in
`tests/fixtures/stable_item_contracts.json`; decompiled code is not redistributed.
Public-action tests cover eligibility, save/reload, independent RNG, maximum HP,
later summons, and same-creature revival.

The [wiki's update history](https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Fur_Coat)
independently records that v0.102.0 added later-enemy support, included in stable
v0.103.2. The v0.110.0 beta increase from seven to eight is not applied.
Full multiplayer synchronization and an installed-game action trace remain unverified.
