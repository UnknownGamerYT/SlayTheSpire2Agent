"""Public regressions for source extra turns and side-specific durations."""
from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _source, _state
from combat.test_glam import _end_turn, _play

from sts2sim.engine import MonsterState
from sts2sim.engine.rng import (
    STS2_STABLE_DERIVATION,
    RunRng,
    capture_random_state,
    random_stream,
)


def _eye_state(*, statuses=None, source=False, relics=("paels_eye",)):
    state = _state(_simple("held"), draw=(_simple("fresh"),), relics=relics,
                   draw_per_turn=1, statuses=statuses)
    monster = state.combat.monsters[0].model_copy(update={
        "intent": "Attack", "intent_damage": 10, "statuses": {"poison": 3},
        "block": 7,
    })
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (monster,),
    })})
    if source:
        rng = RunRng("extra-turn-source", derivation=STS2_STABLE_DERIVATION)
        random_stream(rng, "monster_ai").random()
        state = state.model_copy(update={"character_id": "IRONCLAD",
                                        "rng": capture_random_state(rng)})
    return state


@pytest.mark.parametrize("source", [False, True])
def test_eye_starts_real_extra_player_turn_without_enemy_attack_or_poison(source):
    state = _eye_state(source=source)
    before = _reload(state)
    result = _end_turn(_reload(state))
    assert result.player.hp == 80
    assert result.combat.turn == 2
    assert result.combat.metadata["source_round_number"] == 1
    assert not result.combat.metadata.get("extra_turns")
    assert result.combat.metadata["relic_counters"]["paels_eye"] == 1
    assert [c.card_id for c in result.combat.hand] == ["fresh"]
    assert [c.card_id for c in result.combat.exhaust_pile] == ["held"]
    assert result.combat.monsters[0] == before.combat.monsters[0]
    assert not any(e.kind == "monster_move" for e in result.replay_log[-1].events)
    assert any(e.kind == "extra_turn_started" for e in result.replay_log[-1].events)
    if source:
        assert result.rng.streams["monster_ai"] == before.rng.streams["monster_ai"]
    assert _end_turn(_reload(before)) == result
    # Eye is spent. The following turn permits the enemy's first actual turn.
    result = _end_turn(_reload(result))
    assert result.player.hp < 80
    assert result.combat.monsters[0].hp == 97
    assert result.combat.monsters[0].statuses["poison"] == 2
    assert result.combat.turn == 3
    assert result.combat.metadata["source_round_number"] == 2
    assert not any(e.kind == "extra_turn_started" for e in result.replay_log[-1].events)


def test_extra_turn_repeats_player_end_start_hooks_without_enemy_duration_ticks():
    state = _eye_state(statuses={
        "weak": 2, "frail": 2, "vulnerable": 2, "intangible": 2,
        "source_core_colossus": 2, "source_core_double_damage": 2,
        "borrowed_time": 1, "no_draw_this_turn": 1, "debilitate": 2,
        "regen": 3, "ritual": 2,
    })
    player = state.player.model_copy(update={"hp": 50})
    state = state.model_copy(update={"player": player,
        "combat": state.combat.model_copy(update={"player": player})})
    result = _end_turn(_reload(state))
    assert result.player.hp == 53
    assert result.player.statuses["regen"] == 2
    assert result.player.statuses["strength"] == 2  # Owner-side Ritual still runs.
    for status in ("weak", "frail", "vulnerable", "intangible", "source_core_colossus"):
        assert result.player.statuses[status] == 2
    assert result.player.statuses["source_core_double_damage"] == 1
    assert result.player.statuses["debilitate"] == 1
    assert "borrowed_time" not in result.player.statuses
    assert "no_draw_this_turn" not in result.player.statuses


def test_owner_temporary_stat_expires_before_extra_turn_without_expiring_enemy_stat():
    state = _eye_state(statuses={"temporary_strength": 4, "temporary_dexterity": 3})
    monster = state.combat.monsters[0].model_copy(update={"statuses": {
        "temporary_strength": -6, "weak": 2,
    }})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (monster,),
    })})
    result = _end_turn(_reload(state))
    assert "temporary_strength" not in result.player.statuses
    assert "temporary_dexterity" not in result.player.statuses
    assert result.combat.monsters[0].statuses["temporary_strength"] == -6
    assert result.combat.monsters[0].statuses["weak"] == 2


def test_extra_turn_preserves_enemy_turn_state_and_repeats_first_round_plating():
    state = _eye_state()
    first = state.combat.monsters[0].model_copy(update={
        "statuses": {"plating": 3, "rampart": 4, "nemesis": 1}, "block": 3,
        "metadata": {"initial_plating_given": True, "initial_plating_given_turn": 1,
                     "nemesis_turns": 1, "slow_cards_played": 2,
                     "hardened_shell_hp_lost_this_turn": 7},
    })
    second = MonsterState(monster_id="turret", hp=100, max_hp=100, block=1,
                          metadata={"source_monster_id": "turret_operator"})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (first, second),
    })})
    result = _end_turn(_reload(state))
    first, second = result.combat.monsters
    assert first.block == 6  # Round one BeforeSideTurnStart repeats on an extra turn.
    assert first.statuses["plating"] == 3
    assert first.metadata["nemesis_turns"] == 1
    assert first.metadata["slow_cards_played"] == 2
    assert "hardened_shell_hp_lost_this_turn" not in first.metadata
    assert second.block == 1  # Rampart explicitly excludes extra player turns.
    result = _end_turn(_reload(result))
    assert result.combat.monsters[0].statuses["plating"] == 3
    assert result.combat.monsters[0].metadata["nemesis_turns"] == 2
    assert "slow_cards_played" not in result.combat.monsters[0].metadata
    assert result.combat.monsters[1].block == 4


def test_completed_manual_play_and_whispering_first_turn_allow_enemy_turn():
    manual = _eye_state()
    manual = _play(manual, "held")
    manual = _end_turn(_reload(manual))
    assert manual.player.hp < 80
    assert "source_round_number" not in manual.combat.metadata
    first = _end_turn(_reload(_eye_state(relics=("paels_eye", "whispering_earring"))))
    assert first.player.hp < 80
    assert not any(e.kind == "extra_turn_started" for e in first.replay_log[-1].events)


def test_source_duration_counter_stacks_tick_once_per_relevant_side():
    state = _state(_simple("held"), statuses={
        "source_core_double_damage": 3, "source_core_colossus": 3,
        "conqueror": 3, "retain_hand": 3,
    })
    result = _end_turn(_reload(state))
    for key in ("source_core_double_damage", "source_core_colossus", "conqueror", "retain_hand"):
        assert result.player.statuses[key] == 2
    result = _end_turn(_reload(result))
    for key in ("source_core_double_damage", "source_core_colossus", "conqueror", "retain_hand"):
        assert result.player.statuses[key] == 1


def test_reflect_ticks_on_owner_start_during_extra_turn():
    state = _eye_state(statuses={"reflect": 2})
    monster = state.combat.monsters[0].model_copy(update={"statuses": {"reflect": 2}})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (monster,),
    })})
    result = _end_turn(_reload(state))
    assert result.player.statuses["reflect"] == 1
    assert result.combat.monsters[0].statuses["reflect"] == 2
    result = _end_turn(_reload(result))
    assert "reflect" not in result.player.statuses
    assert result.combat.monsters[0].statuses["reflect"] == 1


def test_extra_turn_runs_installed_source_turn_start_power():
    state = _play(_state(_source("DEMON_FORM"), _simple("held"), relics=("paels_eye",)),
                  "demon_form")
    state = _end_turn(_reload(state))
    assert state.player.statuses["strength"] == 2
    state = _end_turn(_reload(state))
    assert state.player.statuses["strength"] == 4
    assert any(e.kind == "extra_turn_started" for e in state.replay_log[-1].events)


@pytest.mark.parametrize("statuses", [
    {"temporary_strength": -4}, {"strength": -4, "mangle": 4},
])
def test_owner_stat_recovery_uses_first_positive_strength_receiver_hook(statuses):
    result = _end_turn(_reload(_eye_state(
        statuses=statuses, relics=("paels_eye", "ruined_helmet"),
    )))
    assert result.player.statuses["strength"] == 4
    assert result.combat.metadata["relic_counters"]["ruined_helmet"] == 1
    assert "mangle" not in result.player.statuses
    assert "temporary_strength" not in result.player.statuses


@pytest.mark.parametrize("recovery_first,expected", [(True, 6), (False, 4)])
def test_owner_stat_recovery_and_ritual_follow_source_power_application_order(
    recovery_first, expected,
):
    statuses = ({"strength": -4, "mangle": 4, "ritual": 2} if recovery_first else
                {"strength": -4, "ritual": 2, "mangle": 4})
    result = _end_turn(_reload(_eye_state(
        statuses=statuses, relics=("paels_eye", "ruined_helmet"),
    )))
    assert result.player.statuses["strength"] == expected
