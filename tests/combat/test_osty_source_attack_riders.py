from __future__ import annotations

import pytest
from combat.test_stored_turn_powers import _fight, _play

from sts2sim import load_state, serialize


@pytest.mark.parametrize("power", ["monarchs_gaze", "reaper_form"])
@pytest.mark.parametrize("upgraded,damage", [(False, 10), (True, 15)])
@pytest.mark.parametrize("block,artifact", [(0, 0), (100, 0), (0, 1), (100, 1)])
def test_source_osty_attack_respects_each_installed_riders_dealer_predicate(
    power, upgraded, damage, block, artifact,
) -> None:
    state = _play(_fight(power, "sweeping_gaze", upgraded=upgraded), power)
    monster = state.combat.monsters[0].model_copy(update={
        "block": block, "statuses": {"artifact": artifact} if artifact else {},
    })
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (monster,),
        "metadata": {**state.combat.metadata, "osty": {"alive": True, "hp": 20}},
    })})
    state = _play(load_state(serialize(state)), "sweeping_gaze")
    monster = state.combat.monsters[0]
    hp_loss = max(0, damage - block)
    assert monster.hp == 1000 - hp_loss
    expected = damage if power == "reaper_form" and not artifact else 0
    status = "temporary_strength" if power == "monarchs_gaze" else "doom"
    assert monster.statuses.get(status, 0) == expected
    attempts_application = power == "reaper_form"
    assert monster.statuses.get("artifact", 0) == (artifact if not attempts_application else 0)
    hits = [event for event in state.combat.last_events if event.kind == "osty_damaged_enemies"]
    assert len(hits) == 1
    assert hits[0].amount == damage


@pytest.mark.parametrize("power", ["monarchs_gaze", "reaper_form"])
def test_dead_osty_cannot_trigger_installed_attack_riders(power) -> None:
    state = _play(_fight(power, "sweeping_gaze"), power)
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "metadata": {**state.combat.metadata, "osty": {"alive": False, "hp": 0}},
    })})
    state = _play(state, "sweeping_gaze")
    monster = state.combat.monsters[0]
    assert monster.hp == 1000
    assert "temporary_strength" not in monster.statuses
    assert "doom" not in monster.statuses
    assert any(event.kind == "osty_action_blocked" for event in state.combat.last_events)
