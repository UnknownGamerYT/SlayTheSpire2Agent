# Stable item source review

The item inventory distinguishes 296 live relics and 63 live potions from the
source registry's history placeholders, test models, and the Vakuu selector
helper. Source is pinned to `zhiyue/sts2-rl-agent` commit
`5d874e855ae1d0550d5c30416fadd7dbb0d133b7` (public v0.107.1). Source paths and SHA256
hashes identify the reviewed files without distributing game source or binaries.

Each entry lists actual source overrides, the shared Python entry points traced,
retained tests which explicitly reference the ID, and known remaining gaps.
`reviewed` means a finite static method-body and consumer review; it does not mean
a gameplay-oracle comparison, complete test coverage, or interaction parity.
`fixed` identifies a correction with retained public-action tests. `gap` records a
known missing or partial consumer, including items with earlier scoped fixes.
Retained tests can cover only a subset of an entry's listed methods. Empty test
lists explicitly identify the absence of a retained per-ID contract test.

The fragments partition the live inventory: `relics_a_m.json` (160),
`relics_n_z.json` (136), and `potions.json` (63). Multiplayer targeting, sharing,
scaling and ordered interactions remain separate validation scopes. The source
registry also registers DeprecatedRelic, DeprecatedPotion, and a test-only mock
potion; those are recorded in `registry_exclusions.json` rather than included in
live-item completion counts. Reward pool membership is narrower than ModelDb
registration and is not inferred from the inventory.
