# Stable event entry eligibility

Normal event selection applies the v0.107.1 `IsAllowed` methods from the
[version-qualified public source snapshot](https://github.com/zhiyue/sts2-rl-agent/tree/5d874e855ae1d0550d5c30416fadd7dbb0d133b7),
then consumes the act's stored shuffled event queue. Named act membership,
previously visited events and current entry requirements govern selection.
Ancients are separate act-start rooms. Explicit `event_id` / `event_room_id`
fixtures continue to override selection and eligibility.

`tests/fixtures/stable_event_entry_source.json` retains the exact 35 source
entry method bodies and all 57 canonical ordinary event IDs. The remaining
ordinary events inherit the base `true` method. Source boundary regressions
cover every override, and public transitions cover actual deck/relic/profile
facts and serialized event continuation. The older
`event_eligibility_source.json` remains a projection of the 66 cached API event
rows, including Ancients. Its 26 published precondition strings continue to
work for custom events; stable canonical IDs use the full game-source rules.

The source resolves several previous API ambiguities and omissions:

- Amalgamator requires both two removable Basic cards tagged Strike and two
  removable Basic cards tagged Defend. A card's title does not establish its
  tag, rarity or removability.
- Punch Off requires `TotalFloor >= 6`; Slippery Bridge requires
  `TotalFloor > 6` and a removable deck card. Total floor counts entered map
  points across acts, including Ancient rooms and the current room. The local
  map coordinate is a different number.
- Luminous Choir tests its canonical 149 Gold entry threshold and an available
  relic in the player's bag. After entry, its mutable price rolls 100–149.
  The API's `100-149` text is not an inclusive player-gold eligibility range.
- Morphic Grove needs two transformable deck cards in addition to 100 Gold.
  Field of Man-Sized Holes needs a Perfect Fit enchantable card; Grave of the
  Forgotten needs a Soul's Power enchantable card; Spiraling Whirlpool needs
  an enchantable Basic Strike/Defend. Existing enchantments, unplayable cards,
  card type and printed Exhaust are checked against source enchantment rules.
- Ranwid additionally needs 100 Gold and at least one tradable relic.
  The Future of Potions needs two potions. Trash Heap requires HP above 5.
  Dense Vegetation permits any single-player HP, while every multiplayer
  participant must have HP above 8. War Historian Repy is source-disabled.

The runtime derives event pets from Byrdpip, Pael's Legion and Byrdonis Egg;
uses profile epochs for unlocked character card pools; and derives relic
tradability from source rarity, pickup/pet effects, used-up state and supplied
melted state. Unknown custom relic metadata leaves tradability unresolved.
Callers can still supply `has_event_pet`, `unlocked_character_count` or
`tradeable_relic_count` for explicit fixtures and imported profile facts.

All-player predicates apply to supplied ally facts. An ally can supply an
`event_eligibility` mapping for deck and profile counts not otherwise present
in its lightweight metadata. Missing ally/profile facts stay visible as
unknown requirements; they are not converted to zero or asserted as verified.
This supports eligibility evaluation without certifying a complete networked
multiplayer game.

`source_data={"act_id": "UNDERDOCKS"}` or per-act `act_ids` overrides the
source-selected act. Normal runs persist the same act identity for maps,
encounters, events, bosses and Ancient candidates; see
[stable run generation](run_generation_stable.md). Custom events absent from
the canonical act lists retain their declared act-label and printed-clause
eligibility.

Unresolved candidate facts appear in
`event.metadata.eligibility_diagnostics.unknown_requirements`, including facts
on unselected candidates. Diagnostics and event-queue cursors survive reload.
Unique eligible events can repeat after exhaustion. If every candidate is
known to be ineligible, the simulator emits `UNKNOWN_EVENT` with
`no_eligible_event`; this deliberately preserves the established eligibility
invariant instead of the source queue's final unchecked fallback.
