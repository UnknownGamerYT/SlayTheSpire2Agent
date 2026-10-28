# Stable item contract corrections

Target: public v0.107.1. Source pin and file hashes are retained in
`tests/fixtures/stable_item_contracts.json`. The source repository identifies Steam
build 23811903 / release commit `59260271` / assembly hash `-1555940892`; no local
DLL or independent full-game action trace has been checked.

The first reviewed family corrects six item contracts:

| Item | Corrected contract |
| --- | --- |
| Glowwater Potion | Exhaust the original hand, resolve exhaust hooks, then draw ten. |
| Liquid Memories | Recover the selected discarded card free for this turn. |
| Gigantification Potion | Retain each next-Attack charge across turns; a second potion provides another charge. |
| Beetle Juice | Apply four turns of Shrink to one enemy; Artifact can block it; stack duration; multiply attack reduction with Weak before final rounding. |
| Powdered Demise | Apply a stackable, Artifact-blockable Demise debuff to one enemy. At enemy-side end deal unpowered, unblockable damage; HP-loss caps and damage reactions apply. |
| Fur Coat | Include Monster and Elite points, use the independent model RNG, preserve maximum HP, and avoid reapplying on Test Subject's same-creature revive. |

`test_stable_item_contracts.py` and `test_fur_coat.py` exercise positive and negative
cases through public legal actions, action-space exposure, and serialization.
Potions have no upgrades. Existing tests that asserted the incorrect behavior
were corrected. Coverage registration and these six fixes do not certify the
other item clauses or cross-item ordering.

Implementation source:
[pinned repository](https://github.com/zhiyue/sts2-rl-agent/tree/5d874e855ae1d0550d5c30416fadd7dbb0d133b7),
`Models.Potions/{GlowwaterPotion,LiquidMemories,GigantificationPotion,BeetleJuice,PowderedDemise}.cs`,
`Models.Powers/{ShrinkPower,DemisePower,GigantificationPower}.cs`,
plus the Fur Coat files listed in `fur_coat_runtime.md`.

Wiki histories were checked for
[Beetle Juice](https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Beetle_Juice),
[Powdered Demise](https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Powdered_Demise),
[Gigantification](https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Gigantification_Potion),
[Glowwater](https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Glowwater_Potion), and
[Liquid Memories](https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Liquid_Memories).
The Beetle Juice hover tooltip on the wiki still says three turns; its printed
description, history (v0.78.0: three to four), stable API, and pinned executable
implementation agree on four. Liquid Memories' Free-to-Play wording change
predates v0.103.2. Fur Coat's eight-room beta description is excluded.
