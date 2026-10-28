"""Claws.CreateMaulFromOriginal stable source upgrade/enchantment contracts."""
from __future__ import annotations

import pytest
from combat.test_source_modifiers import _fight, _play
from combat.test_stored_turn_powers import _source_card
from rewards.test_relic_deck_choices import _choose, _reward_state

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine.rng import RunRng, capture_random_state
from sts2sim.engine.transitions import _enchant_card


@pytest.mark.parametrize("old_id,enchantment,expected_enchant,damage", [
    ("bash", "sharp", True, 16),
    ("defend_ironclad", "nimble", False, 12),
])
def test_claws_preserves_upgrades_and_only_enchantments_that_can_enchant_maul(
    old_id, enchantment, expected_enchant, damage,
):
    original = _source_card(old_id, upgraded=True).model_copy(update={"instance_id": "original"})
    original = _enchant_card(original, enchantment, 2, source_id="earlier")
    state = _reward_state("claws", cards=(original,))
    state = state.model_copy(update={"rng": capture_random_state(RunRng(
        "CLAWS", derivation="sts2.v0.107.1.utf16-djb2-splitmix64",
    ))})
    state = step(state, next(a for a in legal_actions(state) if a.type == "take_reward_relic"))
    state = _choose(load_state(serialize(state)), "original")
    maul = state.master_deck[0]
    assert maul.instance_id == "original" and maul.card_id == "maul" and maul.upgraded
    assert bool(maul.enchantments) is expected_enchant
    if expected_enchant:
        assert maul.enchantments[0].keyword == "sharp"
        assert maul.enchantments[0].amount == 2
        assert maul.enchantments[0].source_id == "earlier"
    assert all(stream.draws == 0 for stream in state.rng.streams.values())
    result = _play(_fight(maul), "original")
    assert result.combat.monsters[0].hp == 500 - damage
