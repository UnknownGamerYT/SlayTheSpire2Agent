# Stable class-card source audit

Reference: public decompilation of stable 0.107.1, build 23811903, at
[zhiyue/sts2-rl-agent, commit 5d874e8](https://github.com/zhiyue/sts2-rl-agent/tree/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/decompiled).
No decompiled implementation is included in this repository.

The audit compared the cached Defect, Necrobinder, Regent, Event, Token,
Status, Curse and Quest rows against their `OnPlay` methods and inspected
the power callbacks involved in the changed rules. The cached scope contains
338 rows. This is a review scope, not a statement that all 338 cards now have
complete runtime parity.

`tests/fixtures/source_class_rules.json` records independently written
expectations and the corresponding source files. Public-action regressions in
`tests/combat/test_source_class_rules.py` exercise card play, card choices,
reloads, generated-card identity, resource spending, damage, pile movement and
physical/deck card mutation. Existing primitive-only tests do not establish
that an effect is connected to the public simulator.

The changes correct immediate versus deferred resource gains, accidental
draws from generated-card variables, duplicate token generation, transforms,
choice limits and filters, damage-versus-HP-loss semantics, per-hit scaling,
temporary stat ownership, source ordering, card growth, random generation
without replacement, and inherent pile callbacks. Forge now creates and
modifies executable Sovereign Blades, and Blade powers apply to later blades.
Shatter evokes each original orb twice. Tesla Coil directs every Lightning
passive at its selected target. Sic 'Em installs its debuff after its own hit.

Several old synthetic fixtures contradicted the source: Soul cost 1, Furnace
4/6, Sic 'Em summoning on its initial hit, and Beat Into Shape using Energy
instead of base 5 plus 5 per previous hit. Their expectations were corrected
using the referenced constructors and callbacks.

The following limits remain and should not be presented as verified parity:

| Area | Remaining limitation |
| --- | --- |
| Multiplayer | Ally dictionaries are not a complete multi-player combat model. Glimpse Beyond and Legion of Bone still require complete teammate pile/Osty ownership and per-player RNG behavior; targeted ally cards and simultaneous choices need live validation. |
| Trigger ordering | Existing general power/relic dispatch is not an exact implementation of the game's hook priority graph. Nested draws, deaths, generated cards and multiple simultaneous powers require captured traces. |
| Randomness | Selecting the correct source RNG stream and drawing the correct count are separate requirements. Source stream work must be integrated before claiming seeded parity for generation, transforms and random pile insertion. |
| Physical card rebuilding | Combat and permanent growth must be retained by upgrade/downgrade adapters. The shared source-card adapter supplies this integration; test the combined branch, particularly Dampen and generated clones. |
| Noncombat cards | Quest/campfire/event lifecycle behavior is outside these OnPlay fixes. Source-only membership and non-upgradable metadata belong to the source pool registry. |
| Platform validation | These are source-contract regressions. They do not replace Windows game traces for turn timing, selection-screen ordering or save conversion. |

There is no assertion here that unknown hooks, unsupported multiplayer state,
or untested combinations are equivalent to the game.
