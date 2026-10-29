# Stable single-player treasure flow

The version-qualified source snapshot is `5d874e855ae1d0550d5c30416fadd7dbb0d133b7`
for v0.107.1. Paths, hashes and the finite contracts are retained in
`tests/fixtures/source_treasure_edges.json`.

Entering a source treasure room prepares and consumes the shared relic offer.
The actor sees a closed chest with `open_chest` and `proceed` actions. Its hidden
prepared relic is retained in engine saves and omitted from policy observations.
Skipping the unopened chest consumes that prepared offer without granting gold.

Opening grants chest gold immediately using one Rewards draw in the inclusive
42–52 range. Poverty truncates the amount after the roll. Gold modifiers and
positive-gain notifications run once. Chest gold is absent from the subsequent
skippable reward list; the relic is revealed after the grant.

Silver Crucible suppresses generation and normal payouts for the first treasure
room entered after obtaining it. The first-ever tutorial chest instead forces
Gorget when generation is permitted, while still consuming its rarity draw.

Spoils Map targets Act 2. Opening its marked chest completes each held copy after
the base gold grant, preserving separate 600-gold notifications and removals.
Skipping the closed chest or opening a Silver Crucible-suppressed chest does not
complete the quest. The reserved MT fixture behavior remains available for older
synthetic tests.

Retained public-action tests cover save/reload, generation counters, skip/open,
Poverty, modifiers, tutorial forcing, quest copies and hidden-only observation
invariance. Actual Windows bridge capture and installed-game comparisons remain
external validation; these source contracts do not establish exhaustive parity.
