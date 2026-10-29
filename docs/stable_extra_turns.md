# Stable single-player extra turns

Pael’s Eye now starts an actual extra player turn after the full player end-turn
sequence. It exhausts the source hand snapshot, resets Energy, runs player start
hooks and draws the next hand. Enemy actions, poison, intent selection, Block
clearing, duration ticks, Hatch and Nemesis do not run during that transition.
The relic remains once per combat and counts completed manual plays only.

The player turn number increases on an extra turn; the source combat round does
not. The saved `source_round_number` records this distinction once needed.
First-round enemy Plating therefore grants its opening Block again on an extra
player start, while its first actual enemy turn still does not reduce Plating.
Rampart explicitly skips the extra start. Slow retains its previous count until
the enemy participates; Hardened Shell resets at either side start.

The shared duration adapter now distinguishes owner-side end from enemy-side
end. Temporary owner stats, No Draw and Borrowed Time expire before the extra
turn; Weak, Frail, Vulnerable, Intangible and Colossus do not tick without an
enemy side. Duration-counter powers decrement one stack instead of removing all
remaining duration. Reflect decrements at its owner's start. Temporary Strength
recovery still passes through Ruined Helmet, and its ordering against Ritual
follows the player's stored power application order.

Compact pinned-source paths, hashes and predicate facts are retained in
`tests/fixtures/source_extra_turns.json`; public-action tests include save/reload
successors, actual enemy-turn suppression, resource callbacks and later normal
enemy turns. These are source-derived software checks, not installed-game parity
evidence or a claim to cover every ordered interaction.
