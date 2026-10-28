# Stable Ancient offers

Normal xoshiro runs use native offer rules from the pinned v0.107.1
[Models.Events source snapshot](https://github.com/zhiyue/sts2-rl-agent/tree/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/decompiled/MegaCrit.Sts2.Core.Models.Events).
The previous generator used Neow's positive/curse pools for every Ancient.

`mechanics/source_ancients.py` now implements the eight active Ancient classes:

| Ancient | Source rule |
|---|---|
| Neow | One of eight curse relics, three conditional boolean pairs, ordered positive pool and exclusions, then shuffle and take two. Solo excludes Massive Scroll; profile/pool facts gate Kaleidoscope and Scroll Boxes. |
| Orobas | Choose another unlocked character before the Prismatic Gem/Sea Glass roll; choose one offer from each of three pools. Starter relic/card availability gates the third pool. Sea Glass's character is stored before selection. |
| Pael | One offer from each pool. Middle candidates are duplicated before the single Growth entry, preserving its lower weight. Defend enchantment, removable-card and event-pet gates use the actual deck. |
| Tezcatara | One offer from each of three distinct pools; a Basic Strike permits Nutritious Soup. |
| Nonupeipe | Shuffle nine candidates plus Beautiful Bracelet when four Swift-enchantable cards exist; take three. |
| Tanx | Shuffle nine weapons plus Tri Boomerang when three Instinct-enchantable Attacks exist; take three. |
| Vakuu | Shuffle each of three pools separately. Distinguished Cape's offer loses nine Max HP. |
| Darv | Pick from eligible relic groups, shuffle, then optionally replace the third offer with Dusty Tome. Its upgraded Ancient card is chosen on the Rewards stream during offer setup. |

Generation uses the actual deck, relics and profile facts. Event RNG is
independent and recreated from the exact model ID; inspecting or reloading an
already generated offer does not regenerate it. Native choices run the real
relic pickup handler once, instead of also applying the older Neow projection.
Locked Orobas options remain visible in state and are absent from legal actions.
The explicit legacy MT configuration retains its prior synthetic offering path.

Independent C reference vectors cover the exact Neow, Pael, Nonupeipe and Vakuu
sequences for displayed seed `TEST`. Regressions cover real normal-run offers,
profile/deck gates, model stream isolation and a source relic pickup after JSON
reload. This establishes these offer contracts; full-game equivalence still
requires independent stable game traces and validation of the invoked relic
handlers and all event outcomes.
