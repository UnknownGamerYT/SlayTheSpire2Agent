from __future__ import annotations

import pytest
from combat.test_combat_engine_monsters import _play_first_card
from combat.test_monster_innate_hooks import _card, _combat, _monster
from combat.test_source_card_conditions import _no_stubs
from combat.test_source_card_timing import _power


@pytest.mark.parametrize("card_type", ["Attack", "Skill", "Power"])
def test_sealed_throne_any_card_condition_matches_every_playable_type(card_type: str) -> None:
    power = _power("the_sealed_throne", "Whenever you play a card, gain [star:1].")
    follower = {**_card("follow", damage=0), "type": card_type}
    state = _combat(_monster("VICTIM", {}), deck=(power, follower))
    state = _play_first_card(state, "the_sealed_throne")
    assert state.combat.player.resources.get("star", 0) == 0
    state = _play_first_card(state, "follow")
    assert state.combat.player.resources["star"] == 1
    _no_stubs(state)
