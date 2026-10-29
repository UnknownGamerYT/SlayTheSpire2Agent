from __future__ import annotations

import pytest
from combat.test_combat_engine_monsters import _play_first_card
from combat.test_monster_innate_hooks import _card, _combat, _monster
from combat.test_source_card_conditions import _no_stubs, _source
from combat.test_source_card_timing import _power

from sts2sim import load_state, serialize


def test_source_rupture_installs_a_persistent_hp_loss_listener() -> None:
    power = _power(
        "rupture",
        "Whenever you lose HP on your turn, gain 1 Strength.",
        powers_applied=({"power_key": "Strength", "amount": 1},),
    )
    state = _combat(_monster("VICTIM", {}), deck=(power, _card("pay", effects={"hp_loss": 3})))
    state = _play_first_card(state, "rupture")
    assert state.combat.player.statuses.get("strength", 0) == 0
    assert state.combat.player.statuses["rupture"] == 1
    state = _play_first_card(load_state(serialize(state)), "pay")
    assert state.combat.player.hp == 77
    assert state.combat.player.statuses["strength"] == 1
    _no_stubs(state)


@pytest.mark.parametrize("artifact,expected_block", [(False, 2), (True, 0)])
def test_source_shroud_requires_a_successful_enemy_doom_application(
    artifact: bool,
    expected_block: int,
) -> None:
    power = _power("shroud", "Whenever you apply Doom, gain 2 Block.", block=2)
    doom = _source("doom", "Apply 2 Doom.", powers_applied=({"power_key": "Doom", "amount": 2},))
    state = _combat(_monster("VICTIM", {"ARTIFACT": 1} if artifact else {}), deck=(power, doom))
    state = _play_first_card(state, "shroud")
    assert state.combat.player.block == 0
    state = _play_first_card(state, "doom")
    assert state.combat.player.block == expected_block
    _no_stubs(state)


@pytest.mark.parametrize("artifact,expected_draw", [(False, 1), (True, 0)])
def test_source_vicious_draws_only_after_vulnerable_is_actually_applied(
    artifact: bool,
    expected_draw: int,
) -> None:
    power = _power("vicious", "Whenever you apply Vulnerable, draw 1 card.", cards_draw=1)
    apply = _source(
        "apply", "Apply 2 Vulnerable.", powers_applied=({"power_key": "Vulnerable", "amount": 2},)
    )
    state = _combat(
        _monster("VICTIM", {"ARTIFACT": 1} if artifact else {}),
        deck=(power, apply, _card("filler", damage=0)),
    )
    filler = next(card for card in state.combat.hand if card.card_id == "filler")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": tuple(card for card in state.combat.hand if card.card_id != "filler"),
                    "draw_pile": (filler,),
                }
            )
        }
    )
    state = _play_first_card(state, "vicious")
    assert len(state.combat.draw_pile) == 1
    state = _play_first_card(load_state(serialize(state)), "apply")
    assert len(state.combat.hand) == expected_draw
    assert len(state.combat.draw_pile) == 1 - expected_draw
    _no_stubs(state)


def _creator() -> dict:
    return _card(
        "create",
        effects={
            "add_card_to_hand": {
                "card": {
                    "card_id": "dazed",
                    "type": "status",
                    "cost": -2,
                    "target": "none",
                    "effects": {"noop": True},
                }
            }
        },
    )


def test_source_pillar_of_creation_gains_block_for_actual_created_cards_only() -> None:
    power = _power("pillar_of_creation", "Whenever you create a card, gain 3 Block.", block=3)
    state = _combat(_monster("VICTIM", {}), deck=(power, _creator()))
    state = _play_first_card(state, "pillar_of_creation")
    assert state.combat.player.block == 0
    state = _play_first_card(state, "create")
    assert [card.card_id for card in state.combat.hand] == ["dazed"]
    assert state.combat.player.block == 3
    _no_stubs(state)


def test_source_trash_to_treasure_channels_only_after_creating_a_status() -> None:
    power = _power("trash_to_treasure", "Whenever you create a Status, Channel 1 random Orb.")
    state = _combat(_monster("VICTIM", {}), deck=(power, _creator()))
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"orb_slots": 3})})
    state = _play_first_card(state, "trash_to_treasure")
    assert not state.combat.orbs
    state = _play_first_card(state, "create")
    assert [card.card_id for card in state.combat.hand] == ["dazed"]
    assert len(state.combat.orbs) == 1
    _no_stubs(state)


def test_oblivion_doom_is_recorded_after_card_resolution_for_next_deaths_door() -> None:
    oblivion = _source(
        "oblivion", "Whenever you play a card this turn, apply 3 Doom to the enemy.", target="enemy"
    )
    door = _source(
        "deaths_door",
        "Gain 6 Block. If you applied Doom this turn, gain Block 2 additional times.",
        block=6,
    )
    state = _combat(_monster("VICTIM", {}), deck=(oblivion, door, door))
    state = _play_first_card(state, "oblivion")
    state = _play_first_card(state, "deaths_door")
    assert state.combat.player.block == 6
    assert state.combat.monsters[0].statuses["doom"] == 3
    state = _play_first_card(state, "deaths_door")
    assert state.combat.player.block == 24
    _no_stubs(state)


def test_neurosurge_self_doom_triggers_shroud_and_enables_deaths_door() -> None:
    from combat.test_combat_engine_monsters import _end_turn_action

    from sts2sim import step

    shroud = _power("shroud", "Whenever you apply Doom, gain 2 Block.", block=2)
    surge = _power(
        "neurosurge",
        "Gain [energy:3]. Draw 2 cards. At the start of your turn, apply 3 Doom to yourself.",
        energy_gain=3,
        cards_draw=2,
        powers_applied=({"power_key": "Neurosurge", "amount": 3},),
    )
    door = _source(
        "deaths_door",
        "Gain 6 Block. If you applied Doom this turn, gain Block 2 additional times.",
        block=6,
    )
    state = _combat(_monster("VICTIM", {}), deck=(shroud, surge, door))
    state = _play_first_card(state, "shroud")
    state = _play_first_card(state, "neurosurge")
    state = step(state, _end_turn_action(state))
    assert state.combat.player.statuses["doom"] == 3
    assert state.combat.player.block == 2
    state = _play_first_card(state, "deaths_door")
    assert state.combat.player.block == 20
    _no_stubs(state)
