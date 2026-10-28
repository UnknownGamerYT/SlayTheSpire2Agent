# Fur Coat runtime behavior

The stable cached source describes seven random marked combats. Pickup now
samples uncompleted normal monster nodes ahead of the current floor in the
current act. Selected node IDs are stored by act in `fur_coat_marked_nodes` and
recorded as `fur_coat_room_marked` replay events. Additional pickups select only
previously unmarked nodes. The actual Ancient pickup route shares the Ancient
choice RNG, so sampling advances the persisted stream instead of being skipped
or overwritten.

Entering a marked normal room sets initial enemies' current and maximum HP to
one. The shared monster creation path applies the same limit to modeled summons
and reinforcements. Scripted Test Subject replacement phases are also limited in
marked rooms; Decimillipede reattachment already respects the resulting maximum
HP. Combat metadata and replay events identify the affected room and enemies.
Save/reload tests cover both selection and later summons.

The existing marker's `room_kind: combat` contract is interpreted as normal
monster nodes. Elites, bosses, and unknown event rooms are excluded, including
when an imported flag incorrectly marks one. The public sources below do not
establish the game's precise room eligibility predicate or RNG stream, so this
implementation does not claim exact game-seed parity for room selection.

The wiki records a beta increase from seven to eight in v0.110.0 on July 30,
2026. This implementation retains the stable cached count of seven. Its
v0.102.0 entry specifically confirms that enemies entering a fight are affected,
including restocked Axebots. The simulator has no source-backed Axebot encounter
restock script; all monster creation paths currently modeled receive the Fur
Coat effect, without inventing an encounter wave schedule.

Sources checked October 28, 2026:

- Stable local `data/cache/eng/relics.json`, `FUR_COAT`.
- <https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Fur_Coat>

The wiki identifies Nonupeipe at the beginning of Act 3 as the pickup source.
Single-player runtime behavior is covered here; multiplayer selection and
synchronization are outside the simulator's current scope.
