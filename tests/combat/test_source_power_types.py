"""Artifact and Rend contracts from stable PowerModel and power declarations."""

import json
from pathlib import Path

import pytest
from combat.test_stored_turn_powers import _end, _fight, _play

from sts2sim import load_state, serialize
from sts2sim.mechanics.source_power_rules import (
    SOURCE_POWER_DEFINITIONS,
    power_type_for_amount,
    source_power_definition,
    unique_source_debuff_count,
)

FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures/source_power_metadata.json").read_text()
)
ROWS = FIXTURE["rows"]


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["id"])
def test_all_260_power_type_and_temporary_declarations(row):
    definition = source_power_definition(row["id"])
    assert definition is not None
    assert definition.kind == row["type"]
    assert definition.stack_type == row["stack_type"]
    assert definition.allow_negative == row["allow_negative"]
    assert definition.temporary == row["temporary"]


def _apply_status(status, amount, *, initial=None, artifact=1, player_powers=None):
    state = _fight("zap", statuses=player_powers)
    card = state.combat.hand[0].model_copy(
        update={
            "effects": {
                "apply_status": {"target": "all_enemies", status: amount},
            }
        }
    )
    monster = state.combat.monsters[0].model_copy(
        update={
            "statuses": {
                **({"artifact": artifact} if artifact else {}),
                **(initial or {}),
            }
        }
    )
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"hand": (card,), "monsters": (monster,)})}
    )
    return _play(load_state(serialize(state)), "zap")


@pytest.mark.parametrize("status", ["thorns", "vigor", "plating", "metallicize", "artifact"])
def test_negative_fixed_buff_removal_preserves_artifact(status):
    initial = {status: 4}
    state = _apply_status(status, -2, initial=initial)
    assert state.combat.monsters[0].statuses[status] == (2 if status != "artifact" else 2)
    if status != "artifact":
        assert state.combat.monsters[0].statuses["artifact"] == 1
    assert not any(e.kind == "status_blocked_by_artifact" for e in state.combat.last_events)


@pytest.mark.parametrize(
    "status", ["weak", "vulnerable", "poison", "no_card_block_gain", "no_energy_gain_this_turn"]
)
def test_debuff_removal_is_a_buff_and_does_not_consume_artifact(status):
    state = _apply_status(status, -2, initial={status: 3})
    assert state.combat.monsters[0].statuses == {status: 1, "artifact": 1}


@pytest.mark.parametrize("status", ["strength", "dexterity", "focus"])
@pytest.mark.parametrize("initial", [-2, 0, 5])
def test_negative_signed_stat_application_is_debuff_even_when_result_would_be_positive(
    status, initial
):
    state = _apply_status(status, -2, initial={status: initial})
    assert state.combat.monsters[0].statuses.get(status, 0) == initial
    assert "artifact" not in state.combat.monsters[0].statuses


@pytest.mark.parametrize("status", ["strength", "dexterity", "focus"])
def test_signed_stat_loss_triggers_debuff_damage_even_if_current_amount_stays_positive(status):
    state = _apply_status(
        status, -2, initial={status: 5}, artifact=0, player_powers={"sleightofflesh": 9}
    )
    assert state.combat.monsters[0].statuses[status] == 3
    assert state.combat.monsters[0].hp == 991


@pytest.mark.parametrize("status", ["weak", "thorns", "vigor"])
def test_buff_removal_does_not_trigger_sleight_of_flesh(status):
    state = _apply_status(
        status, -2, initial={status: 5}, artifact=0, player_powers={"sleightofflesh": 9}
    )
    assert state.combat.monsters[0].hp == 1000


@pytest.mark.parametrize(
    "statuses,count",
    [
        ({"strength": -1}, 1),
        ({"dexterity": -1}, 1),
        ({"focus": -1}, 1),
        ({"strength": -1, "dexterity": -2, "focus": -3}, 3),
        ({"dexterity": -2, "temporary_dexterity": 3}, 0),
        ({"focus": 2, "temporary_focus": -3}, 1),
        ({"mangle": 3, "crush_under": 1, "dying_star": 9}, 0),
        ({"strength": -2, "darkshackles": 2, "piercing_wail": 3}, 1),
        ({"strength": -2, "strength_down": 3}, 1),
        ({"shriek": -1, "shrink": -1}, 2),
        ({"weak": -1}, 0),
    ],
)
@pytest.mark.parametrize("upgraded,base,multiplier", [(False, 15, 5), (True, 18, 8)])
def test_rend_counts_signed_current_stats_and_excludes_every_temporary_wrapper(
    statuses,
    count,
    upgraded,
    base,
    multiplier,
):
    assert unique_source_debuff_count(statuses) == count
    state = _fight("rend", upgraded=upgraded)
    monster = state.combat.monsters[0].model_copy(update={"statuses": statuses})
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )
    state = _play(load_state(serialize(state)), "rend")
    assert state.combat.monsters[0].hp == 1000 - base - count * multiplier


def test_power_fixture_scope_and_signed_types_are_complete():
    assert FIXTURE["game_version"] == "0.107.1"
    assert len(SOURCE_POWER_DEFINITIONS) == 260
    assert {p.power_id for p in SOURCE_POWER_DEFINITIONS.values() if p.allow_negative} == {
        "strength",
        "dexterity",
        "focus",
        "shrink",
        "shriek",
    }
    assert power_type_for_amount("ThornsPower", -1) == "buff"
    assert power_type_for_amount("WeakPower", -1) == "buff"


def test_temporary_focus_expiry_keeps_real_focus_when_artifact_blocks_stat_loss():
    state = _play(_fight("hotfix", statuses={"artifact": 1}), "hotfix")
    assert state.player.statuses.get("temporary_focus") == 2
    state = _end(load_state(serialize(state)))
    assert state.player.statuses.get("focus") == 2
    assert "artifact" not in state.player.statuses
    assert "temporary_focus" not in state.player.statuses
