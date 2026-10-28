# Sea Glass fixed rarity selection

Sea Glass now offers five distinct Common, five distinct Uncommon, and five
distinct Rare cards from one other character. Each rarity is sampled uniformly
without replacement from sorted card IDs, so duplicate source rows and source
payload order do not change simulator selection. This is simulator RNG
determinism; the game's RNG consumption and exact run-seed mapping are not proven.

All fifteen cards remain optional independent choices. Existing mandatory cards
in the same reward retain their mandatory status. Selected card identities,
remaining choices, and rarity metadata survive JSON save/reload.

Incomplete custom datasets use only available cards of each required rarity.
They never substitute extra Commons for missing Uncommons or Rares. Reward
metadata records `sea_glass_rarity_counts` and
`sea_glass_missing_rarity_counts`; the generation replay event records the same
counts and deficits. An absent other-character pool retains the existing
explicit blocked event.

## Public evidence and version limits

Sources fetched October 28, 2026:

- [Sea Glass wiki](https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Sea_Glass):
  "Upon pickup, you see 15 cards from a class other than your own" and "The
  options always consist of 5 Commons, 5 Uncommons, and 5 Rares."
- [Spire Codex card rarity mechanics](https://spire-codex.com/mechanics/card-rarity):
  lists Sea Glass under "Fixed rarity, uniform within rarity" and states
  "15 cards: 5 Common + 5 Uncommon + 5 Rare, keep as many as you want."
- Stable cached `data/cache/eng/relics.json`, `SEA_GLASS`, confirms fifteen cards
  from another character and any-number selection.

The Codex mechanics table labels the Sea Glass pool "your character", which
contradicts its relic description and the wiki. This implementation keeps the
other-character pool supported by the stable cached relic description and wiki.

The wiki has a general Beta Content Warning. Its Sea Glass update history lists
only prerelease naming/icon changes, and the public sources checked document no
post-v0.107.1 gameplay change to this rarity profile. That supports this bounded
correction, but does not independently prove the implementation of an exact
stable v0.107.1 executable. Class-selection weights, card display order, branch
interactions, and exact game RNG consumption remain unverified.

Tests use an accurate mixed-rarity fixture and cover the five-per-rarity
contract, optional selection of zero through fifteen cards, mandatory-reward
interaction, identity persistence, deficient rarity pools, and immunity to
source order or duplicate rows. The former all-Common fixture was insufficient
to validate the game's rarity distribution.
