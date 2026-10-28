from __future__ import annotations

import pytest
from combat.test_combat_engine_monsters import _end_turn_action, _play_first_card
from combat.test_monster_innate_hooks import _card, _combat, _monster
from combat.test_source_card_conditions import _no_stubs, _source

from sts2sim import load_state, serialize, step
from sts2sim.engine import OrbState
from sts2sim.mechanics.card_effects import card_effect_plan


def _power(card_id: str, description: str, **fields) -> dict:
    return _source(card_id, description, type="Power", **fields)


def test_demon_form_grants_strength_on_future_turns_only() -> None:
    power = _power(
        "demon_form",
        "At the start of your turn, gain 2 Strength.",
        powers_applied=({"power_key": "Strength", "amount": 2},),
    )
    state = _combat(_monster("VICTIM", {}), deck=(power,))
    state = _play_first_card(state, "demon_form")
    assert state.combat.player.statuses.get("strength", 0) == 0
    state = step(load_state(serialize(state)), _end_turn_action(state))
    assert state.combat.player.statuses["strength"] == 2
    state = step(state, _end_turn_action(state))
    assert state.combat.player.statuses["strength"] == 4
    _no_stubs(state)


def test_machine_learning_draws_an_additional_card_every_future_turn() -> None:
    power = _power(
        "machine_learning", "At the start of your turn, draw 1 additional card.", cards_draw=1
    )
    state = _combat(_monster("VICTIM", {}), deck=(power, _card("filler", damage=0)))
    filler = next(card for card in state.combat.hand if card.card_id == "filler")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": tuple(card for card in state.combat.hand if card.card_id != "filler"),
                    "draw_pile": tuple(
                        filler.model_copy(update={"instance_id": f"draw:{i}"}) for i in range(4)
                    ),
                    "draw_per_turn": 0,
                }
            )
        }
    )
    state = _play_first_card(state, "machine_learning")
    assert len(state.combat.hand) == 0
    for expected in (1, 2):
        state = step(state, _end_turn_action(state))
        assert len(state.combat.hand) == 1
        assert len(state.combat.draw_pile) == 4 - expected
    _no_stubs(state)


def test_pyre_grants_future_energy_instead_of_immediate_energy() -> None:
    state = _combat(
        _monster("VICTIM", {}),
        deck=(_power("pyre", "Gain [energy:1] at the start of each turn.", energy_gain=1),),
    )
    state = _play_first_card(state, "pyre")
    assert state.combat.player.energy == 3
    for _ in range(2):
        state = step(state, _end_turn_action(state))
        assert state.combat.player.energy == 4
    _no_stubs(state)


def test_infinite_blades_adds_a_shiv_every_future_turn_only() -> None:
    source = {
        "id": "INFINITE_BLADES",
        "type": "Power",
        "cost": 0,
        "description": "At the start of your turn, add 1 [gold]Shiv[/gold] into your Hand.",
        "spawns_cards": ("SHIV",),
    }
    shiv = _card("shiv", damage=4)
    card = dict(card_effect_plan(source, card_library={"shiv": shiv}).card)
    state = _combat(_monster("VICTIM", {}), deck=(card,))
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"draw_per_turn": 0})}
    )
    state = _play_first_card(state, "infinite_blades")
    assert not state.combat.hand
    for index in range(2):
        state = step(state, _end_turn_action(state))
        assert [card.card_id for card in state.combat.hand] == ["shiv"]
        state = _play_first_card(state, "shiv")
        assert state.combat.monsters[0].hp == 100 - (index + 1) * 4
    _no_stubs(state)


def test_toric_toughness_grants_block_now_and_exactly_two_future_turns() -> None:
    card = _source(
        "toric_toughness", "Gain 5 Block. Gain 5 Block at the start of the next 2 turns.", block=5
    )
    state = _combat(_monster("VICTIM", {}, damage=0), deck=(card,))
    state = _play_first_card(state, "toric_toughness")
    assert state.combat.player.block == 5
    for block in (5, 5, 0):
        state = step(state, _end_turn_action(state))
        assert state.combat.player.block == block
    _no_stubs(state)


def test_noxious_fumes_does_not_apply_poison_when_the_power_is_played() -> None:
    power = _power(
        "noxious_fumes",
        "At the start of your turn, apply 2 Poison to ALL enemies.",
        powers_applied=({"power_key": "Poison", "amount": 2},),
        target="all_enemies",
    )
    state = _combat(_monster("VICTIM", {}), deck=(power,))
    state = _play_first_card(state, "noxious_fumes")
    assert state.combat.monsters[0].statuses.get("poison", 0) == 0
    state = step(state, _end_turn_action(state))
    assert state.combat.monsters[0].hp == 100
    assert state.combat.monsters[0].statuses["poison"] == 2
    _no_stubs(state)


def test_corrosive_wave_waits_for_draw_then_expires_at_turn_end() -> None:
    wave = _source(
        "corrosive_wave",
        "Whenever you draw a card this turn, apply 2 Poison to ALL enemies.",
        powers_applied=({"power_key": "Poison", "amount": 2},),
        target="all_enemies",
    )
    draw = _source("draw", "Draw 1 card.", cards_draw=1)
    state = _combat(_monster("VICTIM", {}), deck=(wave, draw, _card("filler", damage=0)))
    filler = next(card for card in state.combat.hand if card.card_id == "filler")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "draw_pile": (filler.model_copy(update={"instance_id": "drawn"}),),
                }
            )
        }
    )
    state = _play_first_card(state, "corrosive_wave")
    assert state.combat.monsters[0].statuses.get("poison", 0) == 0
    state = _play_first_card(state, "draw")
    assert state.combat.monsters[0].statuses["poison"] == 2
    state = step(state, _end_turn_action(state))
    assert state.combat.monsters[0].statuses["poison"] == 1
    _no_stubs(state)


@pytest.mark.parametrize("cost,block", [(1, 0), (2, 4), (3, 4)])
def test_danse_macabre_requires_the_played_card_to_cost_two_or_more(cost: int, block: int) -> None:
    power = _power(
        "danse_macabre",
        "Whenever you play a card that costs [energy:2] or more, gain 4 Block.",
        powers_applied=({"power_key": "DanseMacabre", "amount": 4},),
    )
    attack = {**_card("attack", damage=0), "cost": cost}
    state = _combat(_monster("VICTIM", {}), deck=(power, attack))
    state = _play_first_card(state, "danse_macabre")
    assert state.combat.player.block == 0
    state = _play_first_card(state, "attack")
    assert state.combat.player.block == block
    _no_stubs(state)


def test_loop_triggers_the_rightmost_passive_only_once_per_turn_start() -> None:
    power = _power(
        "loop", "At the start of your turn, trigger the passive ability of your rightmost Orb."
    )
    state = _combat(_monster("VICTIM", {}), deck=(power,))
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "orbs": (OrbState(orb_id="lightning"),),
                    "orb_slots": 3,
                }
            )
        }
    )
    state = _play_first_card(state, "loop")
    state = step(state, _end_turn_action(state))
    # Lightning's normal end-turn passive and exactly one Loop passive.
    assert state.combat.monsters[0].hp == 94
    _no_stubs(state)


def test_call_of_the_void_creates_ethereal_card_without_an_immediate_draw() -> None:
    power = _power(
        "call_of_the_void",
        "At the start of your turn, add 1 random card into your Hand. It gains Ethereal.",
        cards_draw=1,
    )
    state = _combat(_monster("VICTIM", {}), deck=(power,))
    state = state.model_copy(
        update={
            "flags": {**state.flags, "combat_reward_card_pool": (_card("generated", damage=4),)},
            "combat": state.combat.model_copy(update={"draw_per_turn": 0}),
        }
    )
    state = _play_first_card(state, "call_of_the_void")
    assert not state.combat.hand
    state = step(state, _end_turn_action(state))
    assert len(state.combat.hand) == 1
    assert state.combat.hand[0].card_id == "generated"
    assert state.combat.hand[0].custom["ethereal"] is True
    _no_stubs(state)


def test_haunt_only_damages_an_enemy_after_soul_is_played() -> None:
    haunt = _power("haunt", "Whenever you play a Soul, a random enemy loses 6 HP.", hp_loss=6)
    soul = _source("soul", "Draw 1 card.", cards_draw=1)
    state = _combat(_monster("VICTIM", {}), deck=(haunt, soul))
    state = _play_first_card(state, "haunt")
    assert state.combat.player.hp == 80
    assert [card.card_id for card in state.combat.hand] == ["soul"]
    assert state.combat.monsters[0].hp == 100
    state = _play_first_card(state, "soul")
    assert state.combat.monsters[0].hp == 94
    _no_stubs(state)


def test_friendship_loses_strength_now_and_gains_energy_on_future_turns() -> None:
    card = _source(
        "friendship",
        "Lose 2 Strength. Gain [energy:1] at the start of each turn.",
        powers_applied=({"power_key": "Strength", "amount": 2},),
        energy_gain=1,
    )
    state = _combat(_monster("VICTIM", {}), deck=(card,))
    state = _play_first_card(state, "friendship")
    assert state.combat.player.statuses["strength"] == -2
    assert state.combat.player.energy == 3
    for _ in range(2):
        state = step(state, _end_turn_action(state))
        assert state.combat.player.energy == 4
        assert state.combat.player.statuses["strength"] == -2
    _no_stubs(state)


@pytest.mark.parametrize("plays,expected_draw", [(4, 0), (5, 1)])
def test_pale_blue_dot_checks_completed_turn_count_and_draws_next_turn(
    plays: int,
    expected_draw: int,
) -> None:
    power = _power(
        "pale_blue_dot",
        "If you play 5 or more cards in a turn, draw 1 card at the start of your next turn.",
        cards_draw=1,
    )
    fillers = tuple(_card(f"filler_{index}", damage=0) for index in range(4))
    state = _combat(_monster("VICTIM", {}), deck=(power,) + fillers)
    prototype = next(card for card in state.combat.hand if card.card_id == "filler_0")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "draw_per_turn": 0,
                    "draw_pile": (prototype.model_copy(update={"instance_id": "future_draw"}),),
                }
            )
        }
    )
    state = _play_first_card(state, "pale_blue_dot")
    assert len(state.combat.draw_pile) == 1
    for index in range(plays - 1):
        state = _play_first_card(state, f"filler_{index}")
    state = step(load_state(serialize(state)), _end_turn_action(state))
    assert len(state.combat.hand) == expected_draw
    assert len(state.combat.draw_pile) == 1 - expected_draw
    _no_stubs(state)


def test_devour_life_does_not_create_a_soul_when_the_power_is_played() -> None:
    power = _power("devour_life", "Whenever you play a Soul, Summon 1.")
    soul = _source("soul", "Draw 0 cards.")
    state = _combat(_monster("VICTIM", {}), deck=(power, soul))
    state = _play_first_card(state, "devour_life")
    assert [card.card_id for card in state.combat.hand] == ["soul"]
    assert state.combat.player.resources.get("summon", 0) == 0
    state = _play_first_card(state, "soul")
    assert state.combat.player.resources["summon"] == 1
    _no_stubs(state)


def test_entropy_waits_until_turn_start_to_offer_transform_and_does_not_draw() -> None:
    power = _power(
        "entropy", "At the start of your turn, Transform 1 card in your Hand.", cards_draw=1
    )
    keep = {**_card("keep", damage=0), "tags": ("retain",)}
    state = _combat(_monster("VICTIM", {}), deck=(power, keep))
    prototype = next(card for card in state.combat.hand if card.card_id == "keep")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "draw_per_turn": 0,
                    "draw_pile": (prototype.model_copy(update={"instance_id": "future_draw"}),),
                }
            )
        }
    )
    state = _play_first_card(state, "entropy")
    assert len(state.combat.draw_pile) == 1
    assert not state.combat.pending_choices
    state = step(state, _end_turn_action(state))
    assert state.combat.pending_choices[0].kind == "transform"
    assert len(state.combat.draw_pile) == 1
    _no_stubs(state)


def test_spectrum_shift_creates_future_colorless_cards_without_immediate_draw() -> None:
    power = _power(
        "spectrum_shift",
        "At the start of your turn, add 1 random Colorless card into your Hand.",
        cards_draw=1,
    )
    state = _combat(_monster("VICTIM", {}), deck=(power, _card("filler", damage=0)))
    filler = next(card for card in state.combat.hand if card.card_id == "filler")
    state = state.model_copy(
        update={
            "flags": {
                **state.flags,
                "combat_reward_card_pool": (
                    {**_card("colorless", damage=4), "color": "colorless"},
                ),
            },
            "combat": state.combat.model_copy(
                update={
                    "hand": tuple(card for card in state.combat.hand if card.card_id != "filler"),
                    "draw_pile": (filler,),
                    "draw_per_turn": 0,
                }
            ),
        }
    )
    state = _play_first_card(state, "spectrum_shift")
    assert not state.combat.hand
    assert len(state.combat.draw_pile) == 1
    state = step(state, _end_turn_action(state))
    assert [card.card_id for card in state.combat.hand] == ["colorless"]
    assert len(state.combat.draw_pile) == 1
    _no_stubs(state)
