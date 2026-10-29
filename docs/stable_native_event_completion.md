# Stable native event callback completion

Native event callbacks now preserve their awaited source command order. The
saved `source_native_event_continuation` cursor pauses when an obtained relic
opens a reward screen or deck choice. A later relic is pulled only after the
earlier callback finishes. Reloading in the child screen preserves that order,
the event RNG, and the remaining relic bag. The event finishes after the last
callback, rather than while a child screen remains open.

This covers sequential obtains such as Trial's merchant verdict and Ranwid's
relic trade, This or That's curse after its relic callback, and Wongo's points
and badge after a purchased relic finishes. Existing deck-add observers settle
before the next command, without running again on resume.

Wongo records the points earned in the current run separately from the saved
profile total. The event uses that profile total to check the badge threshold;
the game's progress-save manager credits the earned points after the run.

Native custom reward offers open immediately in the reward phase. Resolving
their final entry closes the custom screen and resumes the event in the same
action. Closing or skipping a custom offer also finishes its callback. Colorful
Philosophers waits for its three card rewards. Future of Potions and Ranwid keep
potion use and discard locked throughout their child screens and release their
event-owned lock after completion.

The twelve public-action regression cases include save/reload, immutable input
states, interleaved potion use, sequential RNG consumption, nested deck choices,
reward autoclose, and potion locks. Reviewed source paths and hashes are in
`tests/fixtures/source_native_event_completion.json`. These are finite
source-derived contracts; comparison with the installed game is still required.
