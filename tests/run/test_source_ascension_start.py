"""Stable AscensionManager.ApplyEffectsTo and source threshold boundaries."""
from __future__ import annotations

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize
from sts2sim.engine import RewardState, RunPhase
from sts2sim.mechanics.ascension import (
    MAX_ASCENSION_LEVEL,
    AscensionFlag,
    ascension_enabled,
)


@pytest.mark.parametrize("character", ["IRONCLAD", "SILENT", "REGENT", "NECROBINDER", "DEFECT"])
@pytest.mark.parametrize("ascension", [4, 5, 10])
def test_default_real_character_start_adds_bane_exactly_at_a5(character, ascension):
    baseline = new_run("ASCENSION-START", character, 0)
    state = new_run("ASCENSION-START", character, ascension)
    banes = [card for card in state.master_deck if card.card_id == "ascenders_bane"]
    assert len(banes) == int(ascension >= 5)
    assert len(state.master_deck) == len(baseline.master_deck) + int(ascension >= 5)
    assert state.player.max_hp == baseline.player.max_hp
    assert len({card.instance_id for card in state.master_deck}) == len(state.master_deck)
    if banes:
        assert banes[0].type == "curse"
        assert banes[0].custom["floor_added_to_deck"] == 1
        assert banes[0].cost == -1
    assert serialize(load_state(serialize(state))) == serialize(state)


@pytest.mark.parametrize("ascension,available", [(3, True), (4, False), (10, False)])
def test_tight_belt_restricts_public_reward_pickup_at_a4(ascension, available):
    state = new_run("TIGHT-BELT", "IRONCLAD", ascension)
    state = state.model_copy(update={
        "phase": RunPhase.REWARD,
        "reward": RewardState(reward_id="potion", potion_id="block_potion"),
        "potions": ("fire_potion", "block_potion"),
    })
    assert any(action.type == "take_reward_potion" for action in legal_actions(state)) is available


@pytest.mark.parametrize("flag,boundary", [
    (AscensionFlag.MORE_ELITES, 1), (AscensionFlag.REDUCED_ANCIENT_HEAL, 2),
    (AscensionFlag.REDUCED_GOLD, 3), (AscensionFlag.FEWER_POTION_SLOTS, 4),
    (AscensionFlag.ASCENDER_CURSE, 5), (AscensionFlag.EXPENSIVE_SHOPS, 6),
    (AscensionFlag.LOWER_UPGRADE_CHANCE, 7), (AscensionFlag.TOUGHER_NORMALS, 8),
    (AscensionFlag.TOUGHER_ELITES, 8), (AscensionFlag.TOUGHER_BOSSES, 8),
    (AscensionFlag.DEADLY_NORMALS, 9), (AscensionFlag.DEADLY_ELITES, 9),
    (AscensionFlag.DEADLY_BOSSES, 9), (AscensionFlag.DOUBLE_BOSS, 10),
])
def test_each_supported_source_flag_activates_at_correct_boundary(flag, boundary):
    assert not ascension_enabled(boundary - 1, flag)
    assert ascension_enabled(boundary, flag)


def test_stable_maximum_is_ten_and_old_twenty_level_defaults_are_not_applied():
    assert MAX_ASCENSION_LEVEL == 10
    for flag in (AscensionFlag.REDUCED_REST_HEAL, AscensionFlag.DAMAGED_START,
                 AscensionFlag.LOWER_MAX_HP, AscensionFlag.POORER_BOSS_REWARDS,
                 AscensionFlag.ADVANCED_NORMAL_AI):
        assert not ascension_enabled(10, flag)
    with pytest.raises(ValueError, match="between 0 and 10"):
        new_run("INVALID-ASCENSION", "IRONCLAD", 11)


def test_explicit_initial_deck_remains_a_complete_fixture_override():
    state = new_run("CUSTOM-DECK", "IRONCLAD", 5, source_data={"deck": ["anger"]})
    assert [card.card_id for card in state.master_deck] == ["anger"]
