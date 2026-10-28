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

A second reviewed family corrects twelve additional item contracts. Block Potion
and Ship in a Bottle give unpowered Block, bypass Dexterity/Frail and card-only
bonuses, preserve card bonus charges, and still trigger powers that react to any
Block gain. Ship's delayed Block also bypasses the card Block restriction.
Flex and Speed Potion expire at the end of the player's side; Artifact can block
the negative stat application at that point. Ruined Helmet's missing runtime
Strength-gain dispatch is connected, including temporary Strength gains.

Attack, Skill, Power, and Colorless Potion's generated offers can be skipped.
Generated free cards preserve their base energy cost, apply the source's combat
generation and supplied unlock/MP eligibility, and use the source registry order
and named card-generation stream. Cosmic Concoction generates distinct upgraded
cards. Orobic Acid uses a distinct draw for each of its three types. Free-to-Play
also removes Stars over its printed duration; Touch of Insanity offers only cards
with a local or globally modified energy/Star cost. The all-card Block helper now
applies powered Dexterity/Frail rules to Attack cards too. Block gains touched by
these paths respect the source maximum of 999,999,999.

The second family is covered by fourteen additional parametrized public-action
cases in `test_stable_item_contracts.py`. The free-energy-until-played cleanup
integration is shared with the core-card source patch; absent profile unlock facts
still mean an explicitly assumed fully unlocked profile.

The third family adds four newly corrected items: Blood Potion, Fruit Juice,
and Entropic Brew expose their source `AnyTime` use in noncombat phases, while
combat-only/manual-unusable potions remain masked. Scroll Boxes rolls each
Defect Claw exception before selecting that bundle, draws common/common/uncommon
sequentially without replacement, preserves source pool order, and uses the
player rewards stream. An extra source-contract test records the RNG call order.
Fortifier and delayed potion Block also obey the source Block cap. Fur Coat's
replay event now records the preserved maximum HP.

Three further relic contracts are corrected: Fiddle distinguishes the normal
opening/turn hand draw from other draws, and permits draws on the enemy side;
Sturdy Clamp retains at most ten Block remaining **after** enemy attacks;
Red Skull tracks direct HP loss and healing across the half-HP threshold, stores
activation state, and applies its real negative Strength delta (including Artifact)
when deactivating. Public tests cover threshold changes between effect steps,
save/reload, normal draw, blocked potion draw, enemy-side Centennial Puzzle draw,
and consumed/remaining Block. Ordering when Sturdy Clamp competes with other
Block-clear preventers still needs an independent game trace.

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
