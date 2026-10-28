from sts2sim.content.source_potion_pools import (
    COMBAT_POTION_EXCLUSIONS,
    SOURCE_POTION_POOLS,
    source_potion_options,
)


def test_source_potion_pool_order_and_epoch_unlocks():
    for character in ("ironclad", "silent", "regent", "necrobinder", "defect"):
        options = source_potion_options(character)
        assert len(options) == 48
        assert options[:3] == SOURCE_POTION_POOLS[character]
        assert options[3:] == SOURCE_POTION_POOLS["shared"]
        locked = source_potion_options(character, unlocked_epochs=())
        assert len(locked) == 39
        assert not set(SOURCE_POTION_POOLS[character]).intersection(locked)
        assert not {"beetle_juice", "powdered_demise"}.intersection(locked)
        combat = source_potion_options(character, in_combat=True)
        assert len(combat) == 45
        assert not COMBAT_POTION_EXCLUSIONS.intersection(combat)
        assert not {"foul_potion", "glowwater_potion", "potion_shaped_rock"}.intersection(options)
