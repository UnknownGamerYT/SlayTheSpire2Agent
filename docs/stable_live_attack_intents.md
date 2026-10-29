# Current visible attack damage

Source `AttackIntent.GetSingleDamage` applies damage hooks before producing its
display label. Treating that label as a raw attack adds Strength and applies
Weak or Vulnerable a second time during a simulator preview.

Schema-2 attack rows therefore retain `base_damage` from the current
`AttackIntent.DamageCalc()` and `repeats` separately from the displayed label.
The importer checks that source damage hooks reproduce that label and stores
the exact raw amount independently of total visible `intent_damage`. Execution
and subsequent intent refresh use the raw amount, including after the player
applies a new visible power. Rounded labels are never inverted to guess a base.

Powered attacks without raw telemetry fail closed. The source-only, unmodified
legacy fixtures remain readable. Current attack rows do not reconstruct future
monster AI, dynamic damage delegates, or a lossless installed-game checkpoint.
Source paths/hashes and finite contracts are retained in
`tests/fixtures/source_live_attack_intents.json`; public action tests cover
Strength, Weak, Vulnerable, Intangible, fractional amounts, multiple hits,
save/reload and a Weak card played after capture. Windows game comparisons
remain required.
