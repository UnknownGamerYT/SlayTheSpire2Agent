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

Follow-up source contracts now cover saved Mad Science rider effects and
Innate upgrades, ImprovementPower's distinct combat-end upgrades, Claws upgrade
and compatible-enchantment preservation, and source shuffle/selection streams.
Their source hashes and public-action fixtures are recorded in
`tests/fixtures/source_mad_science.json`, `source_claws.json` and the class-rule
fixtures. The modifier audit records all seven concrete afflictions and all
eight CardKeyword enum values; its empty known-gap lists apply to those stated
single-player contracts, not every combination of cards and powers.

The following validation limits should not be described as known missing
single-player implementations or as verified game parity:

| Area | Remaining limitation |
| --- | --- |
| Multiplayer | Ally dictionaries are not a complete multi-player combat model. Glimpse Beyond and Legion of Bone still require complete teammate pile/Osty ownership and per-player RNG behavior; targeted ally cards and simultaneous choices need live validation. |
| Trigger ordering | Existing general power/relic dispatch is not an exact implementation of the game's hook priority graph. Nested draws, deaths, generated cards and multiple simultaneous powers require captured traces. |
| Randomness | Source named streams, shuffle/selection ordering and draw counts are integrated for the retained source contracts. Independent Windows traces are still needed to certify displayed-seed parity and combined runtime ordering. |
| Physical card rebuilding | Upgrade/downgrade adapters retain the covered growth, cost and enchantment state. Unrecorded historical state from older snapshots and combinations absent from the retained fixtures are not independently certified. |
| Noncombat cards | This document's review scope is OnPlay and related card callbacks. Separate event, rest, item and source-pool fixtures cover their stated lifecycle contracts; this document does not expand its scope to certify them all. |
| Platform validation | These are source-contract regressions. They do not replace Windows game traces for turn timing, selection-screen ordering or save conversion. |

There is no assertion here that unknown hooks, unsupported multiplayer state,
or untested combinations are equivalent to the game.
