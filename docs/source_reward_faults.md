# Source-defined reward faults

The pinned stable v0.107.1 source contains a Kaleidoscope + Driftwood reroll
whose explicitly empty custom pool cannot produce a card. The source does not
replace that pool with the player's ordinary character pool.

`inspect_reward_reroll_fault(state, group="primary")`, exported from `sts2sim`,
reproduces the defined source prefix on an independent RNG/state view. It
records disabling the reroll, unpicked card-choice history, clearing the group's
underlying cards, and one Rewards rarity draw before `InvalidOperationException`.
The other group, input run, rarity offset and unrelated RNG streams stay intact.
It returns the partial reward model and structured fault, **not a normal run
successor**. No cards, victory, loss or UI recovery are fabricated.

This is an explicit oracle diagnostic: its RNG snapshots must not enter policy
observations. The training action mask deliberately avoids the source-faulting
reroll. With Kaleidoscope and Driftwood alone, original card selection and
skipping remain available in the untouched input state. Older snapshots which called the fault unsupported are
recognized by the diagnostic without rewriting their saved bytes.

The installed build's visible screen and task recovery after this exception
still need a Windows comparison. The source-defined prefix is implemented;
successful recovery cannot be established from the pinned source path.

Related source factory corrections preserve the rarity cycle after exclusion
and fail explicitly when a real source pool is exhausted, instead of silently
returning a short reward. Synthetic legacy RNG fixtures retain their documented
short-pool behavior. Source paths, hashes and finite contracts are retained in
`tests/fixtures/source_factory_edges.json` and `source_kaleidoscope.json`.

## Card alternative-count fault

Driftwood and Pael's Wing together create three source alternatives: Skip,
Reroll, and Sacrifice. `CardRewardAlternative.Generate` throws when more than
two alternatives exist. On ordinary reward selection this occurs before the
card-selection screen opens, before a player-choice ID is reserved, and without
changing cards, history, relic counters or RNG. Both relics have distinct
Ancient sources; the simulator does not pretend that three choices are valid.

`inspect_reward_alternative_fault(state, group="primary")` exposes that oracle
trace. The normal training mask avoids only flat actions that would open the
faulting group's card-selection screen. Other rewards and the ordinary
reward-set Proceed/skip-all path remain available; the inspector does not assert
how the installed UI recovers after a player actually opens the faulting screen.
