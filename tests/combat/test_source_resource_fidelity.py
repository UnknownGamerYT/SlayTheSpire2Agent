"""Public-step regressions against the October stable Spire Codex source rows."""

import json
from pathlib import Path

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _state
from combat.test_glam import _end_turn, _play

from sts2sim import legal_actions, serialize, step
from sts2sim.engine import CardEnchantment
from sts2sim.engine.transitions import _card_from_spec

_ROWS = json.loads(
    (Path(__file__).parents[1] / "fixtures/source_resource_fidelity.json").read_text()
)


def _source(card_id, upgraded=False, instance_id=None):
    source = next(row for row in _ROWS if row["id"] == card_id)
    return _card_from_spec(
        {**source, "upgraded": upgraded, "instance_id": instance_id or card_id.lower()}, 0
    )


@pytest.mark.parametrize("upgraded,stars", [(False, 1), (True, 2)])
def test_convergence_delays_resources_retains_once_and_survives_reload(upgraded, stars):
    state = _play(_state(_source("CONVERGENCE", upgraded), _simple("keep")), "convergence")
    assert (state.player.energy, state.player.resources["star"]) == (19, 20)
    state = _end_turn(_reload(state))
    assert (state.player.energy, state.player.resources["star"]) == (21, 20 + stars)
    assert [card.instance_id for card in state.combat.hand] == ["keep"]
    state = _end_turn(_reload(state))
    assert not state.combat.hand
    assert (state.player.energy, state.player.resources["star"]) == (20, 20 + stars)


@pytest.mark.parametrize("upgraded,cost", [(False, 2), (True, 1)])
@pytest.mark.parametrize("attacks", [0, 1, 3])
def test_expect_a_fight_counts_attacks_and_prevents_energy_until_next_turn(upgraded, cost, attacks):
    strikes = tuple(_source("STRIKE_IRONCLAD", instance_id=f"strike{i}") for i in range(attacks))
    energy = _source("BLOODLETTING")
    state = _play(_state(_source("EXPECT_A_FIGHT", upgraded), energy, *strikes), "expect_a_fight")
    assert state.player.energy == 20 - cost + attacks
    state = _play(_reload(state), "bloodletting")
    assert state.player.energy == 20 - cost + attacks
    assert state.player.hp == 77  # Other effects still resolve while energy is blocked.
    state = _end_turn(state)
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"hand": (energy,)})})
    state = _play(_reload(state), "bloodletting")
    assert state.player.energy == 22


@pytest.mark.parametrize("source", ["DOUBLE_ENERGY", "BLOODLETTING"])
def test_expect_a_fight_blocks_energy_formulas_and_potions(source):
    state = _play(_state(_source("EXPECT_A_FIGHT"), _source(source)), "expect_a_fight")
    before = state.player.energy
    state = _play(_reload(state), source.lower())
    assert state.player.energy == before - _source(source).cost
    state = state.model_copy(update={"potions": ("energy_potion",)})
    before = state.player.energy
    state = step(state, next(a for a in legal_actions(state) if a.type == "use_potion"))
    assert state.player.energy == before


@pytest.mark.parametrize("upgraded,block", [(False, 30), (True, 40)])
def test_panic_button_prevents_card_block_for_current_and_next_turn_only(upgraded, block):
    defend = _source("DEFEND_IRONCLAD")
    state = _play(_state(_source("PANIC_BUTTON", upgraded), defend), "panic_button")
    assert state.player.block == block
    state = _play(_reload(state), "defend_ironclad")
    assert state.player.block == block
    for expected in (0, 5):
        state = _end_turn(_reload(state))
        state = state.model_copy(
            update={"combat": state.combat.model_copy(update={"hand": (defend,)})}
        )
        state = _play(_reload(state), "defend_ironclad")
        assert state.player.block == expected


def test_panic_button_allows_block_from_potions():
    state = _play(_state(_source("PANIC_BUTTON")), "panic_button")
    state = state.model_copy(update={"potions": ("block_potion",)})
    state = step(state, next(a for a in legal_actions(state) if a.type == "use_potion"))
    assert state.player.block == 42


@pytest.mark.parametrize("card_id", ["EXPECT_A_FIGHT", "CONVERGENCE", "PANIC_BUTTON"])
def test_resource_rule_replays_are_deterministic_after_save_reload(card_id):
    card = _source(card_id).model_copy(
        update={"enchantments": (CardEnchantment(keyword="glam", amount=1),)}
    )
    state = _state(card, _source("STRIKE_IRONCLAD"))
    direct = _play(state, card.instance_id)
    assert serialize(direct) == serialize(_play(_reload(state), card.instance_id))
    assert sum(event.kind == "card_extra_played" for event in direct.combat.last_events) == 1
    if card_id == "CONVERGENCE":
        state = _end_turn(direct)
        assert (state.player.energy, state.player.resources["star"]) == (22, 22)
    elif card_id == "EXPECT_A_FIGHT":
        assert direct.player.energy == 19  # Only the first resolution grants energy.
    else:
        assert direct.player.block == 30  # The lock applies before the second resolution.


def test_expect_a_fight_prevents_relic_energy_and_expiry_precedes_turn_start_gains():
    strike = _source("STRIKE_IRONCLAD")
    state = _state(_source("EXPECT_A_FIGHT"), strike, relics=("nunchaku", "happy_flower"))
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "metadata": {"relic_counters": {"nunchaku": 9, "happy_flower": 2}},
                }
            )
        }
    )
    state = _play(state, "expect_a_fight")
    state = _play(_reload(state), "strike_ironclad")
    assert state.player.energy == 18
    assert state.combat.metadata["relic_counters"]["nunchaku"] == 0
    state = _end_turn(_reload(state))
    assert state.player.energy == 21


def test_expect_a_fight_prevents_plasma_evoke_energy():
    from sts2sim.engine import OrbState

    evoke = _simple("evoke", effects={"evoke_orb": {"selector": "rightmost", "amount": 1}})
    state = _state(_source("EXPECT_A_FIGHT"), evoke)
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "orbs": (OrbState(orb_id="plasma"),),
                }
            )
        }
    )
    state = _play(state, "expect_a_fight")
    before = state.player.energy
    state = _play(_reload(state), "evoke")
    assert state.player.energy == before
    assert not state.combat.orbs


# StS2 wiki Panic Button Notes explicitly exempt Power triggers, including these
# four Powers. Source NO_BLOCK and NO_ENERGY_GAIN rows classify both as Debuffs.
# https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Panic_Button#Notes
# https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Debuffs
@pytest.mark.parametrize(
    "card_id,followup,expected",
    [
        ("FEEL_NO_PAIN", "exhaust", 36),
        ("SHROUD", "doom", 32),
        ("CHILD_OF_THE_STARS", "star", 34),
        ("AFTERIMAGE", "plain", 33),
    ],
)
def test_panic_button_does_not_block_registered_or_stored_power_triggers(
    card_id, followup, expected
):
    actions = {
        "exhaust": _simple("trigger", exhausts=True),
        "doom": _simple(
            "trigger", target="enemy", effects={"apply_status": {"target": "enemy", "doom": 1}}
        ),
        "star": _source("FALLING_STAR", instance_id="trigger"),
        "plain": _simple("trigger"),
    }
    state = _play(
        _state(_source(card_id), _source("PANIC_BUTTON"), actions[followup]), card_id.lower()
    )
    state = _play(_reload(state), "panic_button")
    state = _play(_reload(state), "trigger")
    assert state.player.block == expected


@pytest.mark.parametrize(
    "card_id,followup,status",
    [
        ("PANIC_BUTTON", "DEFEND_IRONCLAD", "no_card_block_gain"),
        ("EXPECT_A_FIGHT", "BLOODLETTING", "no_energy_gain_this_turn"),
    ],
)
def test_artifact_prevents_source_classified_debuffs_from_resource_cards(card_id, followup, status):
    state = _play(
        _state(_source(card_id), _source(followup), statuses={"artifact": 1}), card_id.lower()
    )
    assert "artifact" not in state.player.statuses
    assert status not in state.player.statuses
    before_energy = state.player.energy
    state = _play(_reload(state), followup.lower())
    if card_id == "PANIC_BUTTON":
        assert state.player.block == 35
    else:
        assert state.player.energy == before_energy + 2


@pytest.mark.parametrize("card_id,expected", [("DODGE_AND_ROLL", 4), ("GLITTERSTREAM", 5)])
@pytest.mark.parametrize("before_panic", [False, True])
def test_panic_button_blocks_new_fixed_future_block_but_keeps_existing(
    card_id, expected, before_panic
):
    state = _state(_source(card_id), _source("PANIC_BUTTON"))
    order = (card_id.lower(), "panic_button") if before_panic else ("panic_button", card_id.lower())
    for instance_id in order:
        state = _play(_reload(state), instance_id)
    state = _end_turn(_reload(state))
    assert state.player.block == (expected if before_panic else 0)


def test_prolong_can_schedule_current_block_under_panic_button():
    state = _play(_state(_source("PANIC_BUTTON"), _source("PROLONG")), "panic_button")
    state = _play(_reload(state), "prolong")
    state = _end_turn(_reload(state))
    assert state.player.block == 30
