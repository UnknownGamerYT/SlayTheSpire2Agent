from __future__ import annotations

import json

import pytest
from combat.test_fishing_rod import _enter_combat, _run, _win
from rooms.test_campfire_relics import _act, _enter_rest

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.agent_api import action_space
from sts2sim.engine import CardInstance, RunPhase


def test_hatch_transforms_every_egg_into_swoop_and_grants_byrdpip() -> None:
    state = _run(relics=("miniature_tent",))
    eggs = tuple(
        CardInstance(instance_id=f"egg:{index}", card_id="BYRDONIS_EGG", cost=-1)
        for index in range(2)
    )
    state = state.model_copy(
        update={
            "master_deck": state.master_deck + eggs,
            "flags": {
                **state.flags,
                "cards": (
                    {
                        "id": "BYRD_SWOOP",
                        "name": "Byrd Swoop",
                        "type": "Attack",
                        "target": "AnyEnemy",
                        "cost": 0,
                        "damage": 14,
                    },
                ),
            },
        }
    )
    original_ids = {card.instance_id for card in state.master_deck}
    state = _act(_enter_rest(state), "hatch")
    assert state.phase == RunPhase.REST
    assert "byrdpip" in state.relics
    swoops = [card for card in state.master_deck if card.card_id == "byrd_swoop"]
    assert len(swoops) == 2
    assert all(card.type == "attack" and card.cost == 0 for card in swoops)
    assert all(card.effects["sequence"] == [{"damage": 14}] for card in swoops)
    assert {card.instance_id for card in state.master_deck} == original_ids
    assert not any(action.type == "hatch" for action in legal_actions(state))
    assert any(
        action.type == "smith" and action.target_id == "egg:0" for action in legal_actions(state)
    )


def test_hatch_is_unavailable_without_an_egg() -> None:
    state = _enter_rest(_run(relics=()))
    assert not any(action.type == "hatch" for action in legal_actions(state))


@pytest.mark.parametrize("source_key", ["allies", "ally_players"])
def test_mend_heals_selected_ally_thirty_percent_capped_and_does_not_rest(source_key: str) -> None:
    state = _run(relics=("regal_pillow", "stone_humidifier", "tiny_mailbox", "dream_catcher"))
    state = state.model_copy(
        update={
            "flags": {
                **state.flags,
                source_key: (
                    {"id": "ally:0", "hp": 10, "max_hp": 51},
                    {"id": "ally:1", "hp": 78, "max_hp": 80},
                ),
            }
        }
    )
    state = _enter_rest(state)
    descriptor = next(action for action in action_space(state) if action["target_id"] == "ally:1")
    assert descriptor["target"] == {"kind": "ally", "target_id": "ally:1", "hp": 78, "max_hp": 80}
    state = step(state, next(a for a in legal_actions(state) if a.target_id == "ally:1"))
    assert state.phase == RunPhase.MAP
    assert state.flags[source_key][0]["hp"] == 10
    assert state.flags[source_key][1]["hp"] == 80
    assert state.player.hp == 80
    assert state.player.max_hp == 80
    assert not state.reward
    assert not state.potions


def test_mend_requires_a_living_injured_ally_and_is_once_per_rest_site() -> None:
    state = _run(relics=("miniature_tent",))
    assert not any(a.type == "mend" for a in legal_actions(_enter_rest(state)))
    state = state.model_copy(
        update={
            "flags": {
                **state.flags,
                "allies": (
                    {"id": "injured", "hp": 1, "max_hp": 51},
                    {"id": "full", "hp": 50, "max_hp": 50},
                    {"id": "dead", "hp": 0, "max_hp": 50},
                ),
            }
        }
    )
    state = _act(_enter_rest(state), "mend")
    assert state.flags["allies"][0]["hp"] == 16
    assert state.phase == RunPhase.REST
    assert not any(a.type == "mend" for a in legal_actions(state))
    restored = load_state(json.loads(json.dumps(serialize(state))))
    assert not any(a.type == "mend" for a in legal_actions(restored))


@pytest.mark.parametrize(
    ("relics", "bonus"),
    [
        (("venerable_tea_set",), 2),
        (("FAKE_VENERABLE_TEA_SET",), 1),
        (("venerable_tea_set", "fake_venerable_tea_set"), 3),
    ],
)
def test_tea_sets_grant_stacking_bonus_only_on_first_turn_of_next_combat(relics, bonus) -> None:
    state = _enter_rest(_run(relics=relics))
    state = _act(state, "smith")
    state = load_state(json.loads(json.dumps(serialize(state))))
    state = _enter_combat(state)
    assert state.player.energy == state.player.max_energy + bonus
    assert "rest_site_next_combat_energy" not in state.flags
    state = _act(state, "end_turn")
    assert state.player.energy == state.player.max_energy
    state = _win(state)
    state = _enter_combat(state)
    assert state.player.energy == state.player.max_energy


def test_visiting_multiple_rest_sites_does_not_stack_one_tea_sets_bonus() -> None:
    state = _enter_rest(_run(relics=("venerable_tea_set",)))
    state = _act(state, "rest")
    state = _enter_rest(state)
    state = _act(state, "rest")
    assert _enter_combat(state).player.energy == 5
