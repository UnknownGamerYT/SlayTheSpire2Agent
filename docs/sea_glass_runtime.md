# Sea Glass source runtime

Source: `SeaGlass.cs`, `Orobas.cs`, `CardCreationOptions.cs` and `CardFactory.cs`
in the [pinned stable source snapshot](https://github.com/zhiyue/sts2-rl-agent/tree/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/decompiled).
Orobas preselects and saves the other character. Sea Glass offers three packs:
five distinct Common, five distinct Uncommon and five distinct Rare cards from
that character. Each card remains an optional selection.

Normal source runs preserve registered pool order and draw sequentially without
replacement on Rewards, with the supplied unlock and player-count constraints.
`ForNonCombatWithUniformOdds` includes `NoUpgradeRoll`; `WithFlags` preserves it.
Consequently the fifteen selections consume fifteen draws and no random upgrade
rolls. Applicable creation modifiers have their own source flags and scope.

Character selection, fixed-card upgrades/enchantments, optional choices and
remaining indices persist across reloads. Incomplete custom catalogs report
per-rarity deficits; they do not replace missing Uncommons or Rares with Commons.
Synthetic legacy fixtures retain their diagnostic RNG behavior.

Tests in `tests/rewards/test_sea_glass.py` and
`tests/rewards/test_source_item_pool_profiles.py` retain selection, persistence,
capacity, source factory flags and profile checks. Source paths/hashes are in
`tests/fixtures/stable_item_contracts.json`. These contracts are not an
installed-game trace or a certificate for every reward-hook interaction.
