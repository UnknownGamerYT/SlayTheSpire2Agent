from sts2sim.content.source_card_pools import SOURCE_CARD_POOLS, source_pool_cards


def test_stable_ordered_card_pool_membership():
    assert {k: len(v) for k, v in SOURCE_CARD_POOLS.items()} == {
        "colorless": 64,
        "curse": 18,
        "defect": 88,
        "deprecated": 1,
        "event": 27,
        "ironclad": 87,
        "necrobinder": 88,
        "quest": 3,
        "regent": 88,
        "silent": 88,
        "status": 12,
        "token": 14,
    }
    assert SOURCE_CARD_POOLS["ironclad"][:3] == ("aggression", "anger", "armaments")
    assert len({c for p in SOURCE_CARD_POOLS.values() for c in p}) == 578


def test_combat_generation_excludes_basics_explicit_forbidden_and_multiplayer():
    pool = source_pool_cards("ironclad", for_combat_generation=True)
    assert "strike_ironclad" not in pool and "defend_ironclad" not in pool
    assert "feed" not in pool and "demonic_shield" not in pool
    assert "anger" in pool
    assert "demonic_shield" in source_pool_cards(
        "ironclad", multiplayer=True, for_combat_generation=True
    )


def test_unrevealed_epoch_cards_filter_without_reordering_remaining_cards():
    pool = source_pool_cards("ironclad", unlocked_epochs=())
    assert "molten_fist" not in pool and "cruelty" not in pool and "dominate" not in pool
    revealed = source_pool_cards("ironclad", unlocked_epochs=("IRONCLAD2_EPOCH",))
    assert all(c in revealed for c in ("molten_fist", "cruelty", "dominate"))
    assert list(pool) == [c for c in revealed if c in pool]
