"""Frost must await block listeners before another orb or phase proceeds."""

from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _source
from combat.test_glam import _end_turn, _play
from combat.test_source_orb_lifecycle import _events, _orb_state

from sts2sim.engine import RunPhase
from sts2sim.engine.rng import STS2_STABLE_DERIVATION, RunRng, capture_random_state


def _source_rng(state):
    return state.model_copy(
        update={
            "character_id": "DEFECT",
            "rng": capture_random_state(
                RunRng("frost-block-callbacks", derivation=STS2_STABLE_DERIVATION)
            ),
        }
    )


def test_frost_passive_block_listener_can_end_combat_before_other_orbs_burn_or_retention():
    state = _orb_state(
        _simple("held"),
        _source("BURN"),
        orbs=("frost", "lightning"),
        statuses={"juggernaut": 5, "well_laid_plans": 1},
    )
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": (state.combat.monsters[0].model_copy(update={"hp": 5}),),
                }
            )
        }
    )
    state = _end_turn(_reload(_source_rng(state)))
    assert state.phase == RunPhase.REWARD
    assert state.player.hp == 80
    assert not state.combat.pending_choices
    assert not any(e.source_id == "orb:lightning" for e in _events(state))
    assert [e.source_id for e in _events(state) if e.kind == "monster_damaged"] == ["juggernaut"]
    assert state.rng.streams["combat_targets"].draws == 1


@pytest.mark.parametrize(
    "relics,damage,draws",
    [
        ((), 15, 3),
        (("gold_plated_cables",), 20, 4),
    ],
)
def test_loop_frost_awaits_one_block_listener_per_passive_without_cables_on_manual_calls(
    relics,
    damage,
    draws,
):
    state = _source_rng(
        _orb_state(orbs=("frost",), relics=relics, statuses={"juggernaut": 5, "loop": 2})
    )
    result = _end_turn(_reload(state))
    assert result.combat.monsters[0].hp == 100 - damage
    assert result.player.block == 4
    assert result.rng.streams["combat_targets"].draws == draws
    assert _end_turn(_reload(state)) == result


def test_emotion_chip_frost_awaits_its_start_of_turn_block_listener():
    state = _source_rng(
        _orb_state(orbs=("frost",), relics=("emotion_chip",), statuses={"juggernaut": 5})
    )
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": (
                        state.combat.monsters[0].model_copy(
                            update={"intent": "Attack", "intent_damage": 5}
                        ),
                    ),
                }
            )
        }
    )
    state = _end_turn(_reload(state))
    assert state.player.hp == 77
    assert state.player.block == 2
    assert state.combat.monsters[0].hp == 90
    assert state.rng.streams["combat_targets"].draws == 2


def test_repeated_frost_evokes_resolve_each_block_listener_once_before_the_next_evoke():
    state = _source_rng(
        _orb_state(_source("DUALCAST"), orbs=("frost", "lightning"), statuses={"juggernaut": 5})
    )
    state = _play(_reload(state), "dualcast")
    assert state.player.block == 10
    assert state.combat.monsters[0].hp == 90
    assert [orb.orb_id for orb in state.combat.orbs] == ["lightning"]
    assert state.rng.streams["combat_targets"].draws == 2
    assert [e.kind for e in _events(state) if e.kind in {"player_block", "monster_damaged"}] == [
        "player_block",
        "monster_damaged",
        "player_block",
        "monster_damaged",
    ]


def test_frost_overflow_awaits_block_listener_before_recording_the_new_channel():
    state = _source_rng(_orb_state(_source("ZAP"), orbs=("frost",), statuses={"juggernaut": 5}))
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"orb_slots": 1})})
    state = _play(_reload(state), "zap")
    assert state.player.block == 5
    assert state.combat.monsters[0].hp == 95
    assert state.rng.streams["combat_targets"].draws == 1
    events = _events(state)
    assert next(i for i, e in enumerate(events) if e.source_id == "juggernaut") < next(
        i for i, e in enumerate(events) if e.kind == "orb_channeled"
    )


def test_zero_frost_block_never_triggers_juggernaut_or_draws_a_target():
    state = _source_rng(
        _orb_state(_source("DUALCAST"), orbs=("frost",), statuses={"juggernaut": 5, "focus": -99})
    )
    state = _play(_reload(state), "dualcast")
    assert state.player.block == 0
    assert state.combat.monsters[0].hp == 100
    assert not state.combat.orbs
    assert not any(e.source_id == "juggernaut" for e in _events(state))
    assert "combat_targets" not in state.rng.streams
