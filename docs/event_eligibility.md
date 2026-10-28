# Event entry eligibility

Random event selection applies the entry clauses in the stable English source
snapshot to the current act, floor, gold, HP, deck, potions and supplied profile
facts. Ancient records are not random event rooms. An exhausted or narrowed pool
produces an explicit `UNKNOWN_EVENT` with `no_eligible_event`; it never falls back
to unrelated or ineligible events. Explicit `event_id` / `event_room_id` fixtures
continue to override selection and eligibility.

The fixture `tests/fixtures/event_eligibility_source.json` projects all 66 cached
event rows and four act definitions. Its tests cover all 26 distinct published
precondition strings, including numeric boundaries, the Foul Potion alternative,
and actual map-to-event transitions. Basic-card eligibility uses source rarity,
and deck removability uses the existing Eternal-card rule. These checks establish
the listed predicates, not exact game event distributions.

The runtime previously tracked numeric acts only. Event selection now accepts
`source_data={"act_id": "UNDERDOCKS"}` for Act 1 or a per-act string-keyed
`act_ids` mapping. Named events follow the selected cached act's `events` list;
shared events still apply their own entry clauses. The existing default is
Overgrowth, Hive, Glory. This selector does not implement variant-specific maps,
encounter pools or random act selection. Custom event IDs absent from the cached
act lists retain their declared act-label eligibility.

The runtime does not establish event-pet taxonomy, profile unlock state or relic
tradeability. Callers can supply `has_event_pet`, `unlocked_character_count` and
`tradeable_relic_count`. Without a supplied fact, the corresponding requirement
remains unknown and does not exclude the event. Supplied `player_count` or ally
state excludes single-player-only events in multiplayer contexts. Unknown source
clauses are also retained rather than guessed.

The Amalgamator API lists requirements for two different options without an
entry AND/OR rule. When only one card pair is available, entry remains admitted
with an explicit grouping diagnostic; neither pair excludes it, and both pairs
establish the printed requirements. This needs game-source or trace evidence.

Selection records unresolved candidate requirements in
`event.metadata.eligibility_diagnostics.unknown_requirements`. This includes
requirements on unselected candidates because they affect the sampling pool.
These diagnostics survive serialization. Event weights, no-repeat rules and
other hidden prerequisites remain unverified; this change does not invent them.
