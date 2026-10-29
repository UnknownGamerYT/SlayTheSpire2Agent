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

`DeathBlowIntent` inherits the same single-attack calculation while exposing a
different display type. The two source uses, Gas Bomb and Waterfall Giant's
Explode moves, attack and then kill their owner. Imported observations retain
the original display type and use attack semantics for damage, potion timing
and the known owner-kill continuation. Native potion timing also reads positive
canonical damage under both DeathBlow and older Special API labels.

The bounded constructor review covered all 121 monster C# files, including the
115 cached monster models: 333 MoveState constructions contain 161 single,
54 multi and two DeathBlow attack-intent constructors. Every MoveState has at
most one attack intent. No shipped move therefore requires a simultaneous
heterogeneous attack sequence. The defensive rejection of that input is not a
remaining source move implementation. Counts, reviewed paths and all corpus
hashes are retained in `tests/fixtures/source_death_blow.json`.
