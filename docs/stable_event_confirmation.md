# Stable Trial abandonment confirmation

The source-derived Trial Double Down action opens an abandon-run confirmation.
It does not immediately damage the player or finish the event. Confirm and
Cancel are serialized public event choices in the simulator. Cancel returns to
the original Reject page, where Double Down remains repeatable and Accept can
still draw the Trial case. Neither opening nor canceling consumes model RNG.

While the modal is open, underlying room and potion actions are unavailable.
Confirmation marks the run abandoned and force-kills the player, bypassing Fairy
in a Bottle and damage prevention. The resulting failed state has no actions.

The finite source contracts and file hashes are retained in
`tests/fixtures/source_trial_confirmation.json`; public-action tests cover
cancellation, repeated opening, save/reload, action descriptors and forced death.
The reserved legacy pure-flow fixture retains its former direct lethal action.
This implements the headless decision sequence. Godot input/rendering behavior
and the compiled Windows game's behavior still need installed-game comparison.
