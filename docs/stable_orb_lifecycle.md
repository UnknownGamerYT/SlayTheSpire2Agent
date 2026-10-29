# Stable orb lifecycle

The finite orb audit uses the pinned v0.107.1 C# reference recorded in
`tests/fixtures/source_orb_lifecycle.json`. It covers all five orb types and
the source callbacks that invoke their passives or evokes.

Automatic passives visit the queue from front to back and stop when combat
ends. Plasma gains energy immediately after the next turn starts; it does not
install a delayed energy power during the previous turn. Gold Plated Cables
adds one trigger to the first orb in automatic start/end callbacks. Manual
passives from Loop, Emotion Chip and card effects do not receive that bonus.
Loop and Emotion Chip run after hand draw; Spinner channels after energy reset,
before hand draw. Emotion Chip reads damage history since the previous player
turn started, including enemy damage, and tests `!DamageResult.WasFullyBlocked`.
This source predicate can qualify a zero-HP result: a zero-damage hit without
Block, or damage prevented by Buffer/Tungsten Rod without Block. A zero result
with existing or consumed Block is fully blocked unless it is unblockable.
Actual HP sums stay separate from this history count. Old saves without the
new count retain their prior HP-loss fallback.

Dualcast, Quadcast and Multi-Cast repeatedly evoke the front orb, removing it on
the final evoke. Shatter processes each original orb from front to back with two
evokes per orb. Glass doubles its Focus-adjusted passive value for an evoke and
only decreases its raw passive value when a positive passive fires. Dark breaks
equal-HP ties by encounter order. A zero-damage Lightning evoke still returns its
selected target to Thunder and consumes exactly the original target selection.
Frost awaits the ordinary block callback pipeline after each passive or evoke.
Juggernaut therefore resolves before the next orb, draw phase or retention
choice, including Loop/Emotion Chip manual passives and channel overflow. A
combat-ending block callback prevents later orb callbacks from running.

Metronome activates once on the seventh successful channel and keeps recording
later channels. A non-Defect character with zero orb capacity gains one slot on
its first channel; Defect at zero capacity does not. The explicit zero-slot
synthetic `TEST` fixture retains its compatibility behavior.

Public action tests include save/reload, queue order, fight-ending callbacks,
negative Focus, multi-evoke cards, manual versus automatic triggers, and source
RNG draw counts. These are source-derived checks. Installed-game differential
captures remain required to establish actual runtime parity.
