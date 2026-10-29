# Runtime audit scope

`audit-cards` and `audit-combat` report `registered_execution_surface` coverage.
An `implemented` entry means its normalized effect keys or named handlers are
registered. It does not establish complete printed behavior, timing, or parity
with the game. Both JSON reports expose this scope and its limits.

The [bounded runtime inventory](runtime_audit_inventory.json) records concrete
findings separately. It contains 81 distinct cards from two audits: 41 timing
and condition findings, plus 44 status and dispatch findings, with four cards
in both. The inventory records each finding's source baseline and the actual
fix commits and regression test files. A verified fix covers the recorded
predicate. All 81 recorded findings now have this evidence. It does not certify
the rest of those cards.

The original conditional scan left 82 cards for runtime review. The status and
dispatch audit established concrete findings for 25 of those cards. Subsequent
source predicate reviews checked the other 57, including the final 54 listed
individually as `predicate_spot_checked` in the inventory. Their evidence
combines source clauses, normalized plans, named adapters and retained public
test references. This closes that finite review queue. It does not establish
every interaction or certify the 577 source cards. The conditional scan
covered 140 cards.

Supplementary evidence records actual draw mutations, delayed effects,
retention, other-player Attack traces, and the eleven tested X-cost cards.
The X-cost tests cover Energy versus Stars, zero X, base and upgraded variants,
Chemical X's bonus for both resources, repeat/Evoke behavior, original-X
replays and free autoplays. Source Osty Attack tests cover installed Monarch's
Gaze and Reaper Form with Block, Artifact, upgrades and save/reload. Each group
states its tested scope; these results do not certify full game or co-op parity.

`KNOWN_CARD_RUNTIME_GAPS` keeps proven outstanding gaps explicit in card and
combat reports. Its recorded findings are now resolved. A future fix must have
source-backed runtime regression evidence before its row is removed.

Commit IDs in the inventory are local review provenance. They may not be
reachable in public history when publication combines commits. Primary
reproduction evidence is the retained test paths and source fixtures at the
published checkout, together with source snapshot hashes and URLs. The
publishing workflow records the final public revision and validated tree hash
outside this inventory in its deliverable or PR validation log.

The earlier Aeonglass intensity and Crusher/Rocket initial-facing questions now
have pinned source contracts and targeted runtime fixtures; see
[enemy contracts](stable_enemy_source_contracts.md). Their historical inventory
rows remain as provenance, with follow-up status recorded. Installed-game action
comparisons remain pending. A bounded marker dispatcher sweep describes only its
45 kinds/168 sampled outputs; it is not evidence for complete relic behavior.

The later [item method review](item_source_audit/README.md), card/modifier audits
and source fixtures supersede earlier handler-count completion statements. Each
review covers its listed predicates and methods, not every possible interaction.
