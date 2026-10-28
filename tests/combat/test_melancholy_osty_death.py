from __future__ import annotations

import pytest
from combat.test_combat_engine_monsters import _end_turn_action, _play_first_card
from combat.test_monster_innate_hooks import _card, _combat, _monster
from combat.test_source_card_conditions import _source

from sts2sim import load_state, serialize, step


@pytest.mark.parametrize("sacrifice", [False, True])
def test_melancholy_reduces_cost_when_osty_dies_to_attack_or_sacrifice(sacrifice: bool) -> None:
    melancholy = _source(
        "melancholy",
        "Gain 13 Block. Reduce this card's cost by [energy:1] whenever ANYONE dies.",
        cost=3,
        block=13,
        energy_gain=1,
    )
    melancholy["tags"] = ("retain",)
    kill = _card("kill_osty", effects={"osty_action": {"action": "die"}})
    state = _combat(_monster("VICTIM", {}, damage=5), deck=(melancholy, kill))
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "metadata": {
                        **state.combat.metadata,
                        "osty": {"hp": 3, "max_hp": 3, "alive": True},
                    },
                }
            )
        }
    )
    state = load_state(serialize(state))
    if sacrifice:
        state = _play_first_card(state, "kill_osty")
    else:
        state = step(state, _end_turn_action(state))
    assert state.combat.metadata["osty"]["alive"] is False
    assert next(card for card in state.combat.hand if card.card_id == "melancholy").cost == 2
    assert next(card for card in state.master_deck if card.card_id == "melancholy").cost == 3


def test_dead_osty_does_not_generate_another_cost_reduction() -> None:
    melancholy = _source(
        "melancholy",
        "Gain 13 Block. Reduce this card's cost by [energy:1] whenever ANYONE dies.",
        cost=3,
        block=13,
        energy_gain=1,
    )
    kill = _card("kill_osty", effects={"osty_action": {"action": "die"}})
    state = _combat(_monster("VICTIM", {}), deck=(melancholy, kill))
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "metadata": {
                        **state.combat.metadata,
                        "osty": {"hp": 0, "max_hp": 3, "alive": False},
                    },
                }
            )
        }
    )
    state = _play_first_card(state, "kill_osty")
    assert next(card for card in state.combat.hand if card.card_id == "melancholy").cost == 3
