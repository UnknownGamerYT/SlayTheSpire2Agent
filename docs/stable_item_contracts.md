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

Counter lifecycle review distinguishes thirteen per-combat relic counters from
saved run counters. A new combat resets Burning Sticks, Centennial Puzzle, Demon
Tongue, Metronome, Mini Regent, Pael's Eye/Legion, Permafrost, Red Skull, Ruined
Helmet, Throwing Axe, Unsettling Lamp, and Vambrace. Demon Tongue and Mini Regent
also reset on each owner turn. Joss Paper, Nunchaku, Pen Nib, Galactic Dust and
other saved counts remain intact. Mini Regent consumes its first-Star-spend
activation; Galactic Dust accumulates separate small spends and keeps the remainder
after each ten-Star trigger. Demon Tongue calculates recovery from HP after damage.

Card HP-loss events from the modeled `HpLossVar` damage path now reach Centennial
Puzzle/Demon Tongue and the existing damage-relic dispatcher. Potion, end-hand,
end-turn relic and debuff damage use the same event adapter. This classifies these
unpowered/unblockable damage paths; it does not classify arbitrary payment costs
as attacks. Tests cover source counter reset/preservation, active consumers after
reset, once-per-turn activation, and repeated small resource spends through saves.

Source potion generation now has the exact five class pools, Shared pool order,
twenty-one epoch-gated potions, source rarity metadata, and the three in-combat
generation exclusions (Fairy, Fruit Juice, Regen). Alchemize uses those exclusions;
Entropic Brew deliberately uses the out-of-combat factory even during combat.
Both use the shared float32 rarity-first factory and the combat potion-generation
stream. Profile epochs absent from input still assume all potions unlocked.

Sea Glass honors the character preselected by Orobas. On the stable RNG path it
selects and removes each offer in source registry order. Its factory options
preserve NoUpgradeRoll, so fifteen choices consume fifteen Rewards draws and no
random upgrade or class reroll. This corrects the earlier thirty-draw assumption.
Dusty Tome adds the preselected Ancient card directly, upgraded;
the pickup does not reroll its card or create an ordinary card reward screen.

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

Fairy in a Bottle now consumes and heals at the death check between individual
damage hits and HP-loss effect steps. Its automatic use runs the potion-use
callbacks, including Reptile Trinket and Belt Buckle. Multiple Fairies may be
consumed on separate hits, and later hits can still kill the player. Temporary
Strength granted during the enemy side remains until the player's next side end.
Belt Buckle tracks its actual applied state as potion inventory changes, including
Artifact blocking its negative Dexterity removal. Lethal damage skips the source
AfterDamageReceived callbacks before Fairy restores HP; surviving monster hits
resolve Centennial Puzzle and other damage-received relics before the next hit,
transferring any draw/shuffle RNG changes back to the combat RNG bank.

Gold gains use one source modifier pass before the gain: Bowler Hat multiplies
by 1.25 and truncates once; Ectoplasm reduces the gain to zero. Dragon Fruit
observes one notification per successful gain, including separate Lucky Fysh
gains for separate added cards, and receives none when Ectoplasm prevents a gain.
Direct room/reward/pickup/card/deck/merchant/heist gain paths share the helper;
the legacy event adapter rewrites its gain event to the actual resulting amount.
New Leaf's chosen transformation uses the run Niche RNG stream. Tests retain
source RNG stream changes and reloads alongside positive and prevented gains.

Clarity and Radiant Tincture stack three turns of duration per use, applying one
extra normal hand draw or one extra Energy per future turn. Clarity's draw belongs
to the hand draw and works with Fiddle. Snecko Oil uses CombatEnergyCosts and its
absolute Energy override expires when the physical card is played or the turn
ends. Soldiers' Stew checks CanonicalTags/actual Strike tags and increases base
replay count, without adding an enchantment. The compact CanonicalTags registry
also covers Defend, OstyAttack, Shiv and Minion tags for other consumers.
Regen Potion heals and decays at the owner's side end before enemy attacks. Heart
of Iron's Plating decays at the next owner side start, rather than on damage hits.
Blood Potion truncates one fifth of maximum HP with no minimum heal of one.

Neow's Talisman upgrades the last Basic Strike and the last Basic Defend in deck
order without random draws; an already-upgraded final copy does not redirect the
upgrade to an earlier copy. Nutritious Soup applies Tezcatara's Ember1 to all
eligible Basic Strikes, including its free cost and Eternal property. Pael's Claw
applies Goopy1 to eligible Defend-tag cards. Pael's Growth applies Clone4 to its
chosen card. Shared source enchantment predicates now filter relic card choices
and automatic applications, and canonical source tags survive card creation.

Sozu prevents owner potion procurement through rewards, shops, generated potions,
events and combat relic grants. A forbidden potion reward is still claimed, as in
PotionReward.OnSelect; a forbidden merchant potion is not purchased. Entropic
Brew creates one potion before procurement fails and stops, retaining those RNG
draws. Alchemize's stable factory draw still occurs when procurement is forbidden.
Snecko Eye's existing Confused grant now has its actual draw consumer, using
CombatEnergyCosts and a combat-local cost override. An X card consumes a roll
while retaining its X-cost behavior. Spiked Gauntlets adds one to Power Energy
costs after local free overrides. Pael's Flesh raises maximum Energy from turn
three, retains that maximum across later turns, and restores the run base after
combat. Ordered local Energy modifiers use the shared source_cost_hooks API.

All twelve relic ModifyMaxEnergy overrides now update the combat maximum at
Energy reset. The nine unconditional +1 relics stack, Bread adds its maximum
bonus after turn one while keeping the separate first-turn loss, Pael's Flesh
starts on turn three, and Pumpkin Candle contributes only while kindled. Legacy
current-Energy markers are suppressed for managed maximum modifiers. The adapter
tracks only its own relative bonus, preserves unrelated maximum changes, and
restores the run base after combat. Current-turn gains remain separate effects.

Vambrace and Pael's Legion multiply all Block commands of the qualifying card
before final rounding; charges are consumed after that card finishes. Pael's
Legion decrements its two-turn cooldown on owner turn starts. Unpowered potion
Block and zero modified Block do not consume these relics. Unmovable shares the
path: it counts positive BlockGainedEntry records from other CardPlays this turn,
not distinct cards. Every Block command of the current qualifying play doubles,
while all of those entries count toward later plays' threshold. History includes
Block gained before Unmovable was applied and resets on the next owner turn.


Card reward factory callbacks now keep per-option enchantments and actual upgrade
states in both the offered action descriptors and selected permanent cards. The
shared modifier applies Silver Crucible's first three CardReward batches, Wing
Charm's single eligible Swift target with Niche RNG, Silken Tress's first batch,
Fresnel Lens and the three Egg modifiers in acquisition order. NoModifyHooks is
separate from NoCardModelModifications; the latter permits the source's manual
CardReward upgrade hooks. New relic acquisition rechecks only the new modifier;
ordinary reward actions do not repeat old hooks. Driftwood rerolls explicitly
repopulate their one group and run its factory callbacks again.

The saved item lifecycle fixtures additionally cover Pael's Tooth selecting five
removable, upgradeable deck instances and returning each serialized instance only
once (including duplicate card IDs); Tea of Discourtesy's single default charge
and pre-draw Innate ordering; and Silver Crucible's first treasure entered after
acquisition rather than the run's first treasure. Sea Glass preserves its three
five-card factory callbacks for Eggs/Wing Charm/Fresnel Lens; reward-only hooks
remain excluded from that fixed selection grid. Glitter and Lava Lamp now join
the shared late-hook dispatcher in actual relic acquisition order.
