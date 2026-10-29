# Stable single-player merchant refills

The source reference is the v0.107.1 snapshot at
`5d874e855ae1d0550d5c30416fadd7dbb0d133b7`. The finite contracts and source
path hashes are recorded in `tests/fixtures/source_merchant_refills.json`.

Courier refills an existing entry directly. A colored card slot retains its
original Attack, Skill or Power constraint and the player's saved rarity odds.
A colorless slot retains its original fixed rarity. Both exclude the original
card and the other stocked cards. Refilling consumes the model-selection,
upgrade and price calls once; it does not generate another inventory or sale.
The resulting physical card receives the usual merchant creation modifiers.

Potion refills use an empty blacklist, so another stocked potion may be chosen.
Relic refills roll one rarity, pull from the back of the shared bag with the
source exclusions, and price that single result.

An initial card sale remains marked while merchant price hooks run. Source
integer halving occurs before Courier and Membership Card discounts. Refilling
the entry clears that sale flag. Saved inventory retains both the offered card
and sale flag, so reloading does not reroll either.

Public-action regressions cover type and rarity constraints, saved pity,
excluded old models, duplicate potions, relic refill RNG counts, physical card
modifiers, sale truncation and reload determinism. This establishes the listed
source contracts; installed-game comparison remains necessary for game parity.
