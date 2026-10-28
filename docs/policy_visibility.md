# Policy observations and debugging oracles

Policy observation schema version 2 defaults to information available before an
action is taken. The vector dimensions are unchanged, but their semantics have
changed; previous checkpoints must not silently resume under the new contract.

`encode_observation`, `action_space`, `encode_rich_observation`, and
`preview_actions` accept the keyword-only `oracle=False`. Normal training and
Gymnasium clients use this default. An explicit `oracle=True` restores exact
simulator information and bounded next-turn lookahead for debugging.

Without Frozen Eye, the draw pile is a canonical composition, rather than the
actual shuffle order. Compact card slots, position hashes, structured positions,
rich zones, and action card references all follow that boundary. Hidden draw
positions are omitted or zeroed in numeric features; `position_from_top` is
`None`. Draw probabilities use the full unknown pile and expected reshuffle
composition instead of peeking at its first cards. With Frozen Eye, the current
draw order is available; the outcome of a future shuffle remains unknown.

Default action previews retain only deterministic effects of the selected
visible combat action. They use an independent canonical state and a public
constant preview RNG, never the actual RNG snapshot, run seed or hidden state
hash. A transition that consumes randomness or draws/moves hidden draw cards is
suppressed. Noncombat/map outcomes, end-turn results and next-hand lookahead
are not previewed by default. An unavailable preview is
`{"preview_available": 0}`; it is not a prediction of zero damage or reward.
Current card mechanics and visible targets remain in action descriptors even
when the preview is unavailable.

Normal observations omit the run seed, saved RNG, replay hashes and preselected
next monster move ID. Rich observations expose a limited set of public run
counters instead of arbitrary source/debug flags. `include_state` now defaults
to `False`; explicitly included state is sanitized unless `oracle=True` is
also supplied. To obtain an exact engine save for debugging:

```python
snapshot = encode_observation(state, include_state=True, oracle=True)["state"]
```

Training code should not feed that debugging snapshot or oracle action
descriptors to its policy. The dedicated `serialize(state)` API remains the
lossless save/replay interface.

PPO also gives its strategic teacher the independent visibility-safe state
copy. Its search can reason over canonical unknown-card composition and a
public independent RNG, while preserving currently visible hand/action IDs and
Frozen Eye's known draw order. It never receives the actual saved RNG or hidden
shuffle order through the teacher-label path. Calling the strategic agent
directly remains available for simulator debugging.

Regression tests compare complete compact and rich observations, action
descriptors and packed PPO inputs for states differing only in hidden draw
order and saved RNG. They also cover Frozen Eye and oracle exceptions, mask-safe
map previews, preserved deterministic damage previews, random/draw suppression,
and absence of source-state or RNG mutation.
