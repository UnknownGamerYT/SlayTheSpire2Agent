"""Infused Core is an owner Lightning modifier after the Focus clamp."""
from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _source, _state
from combat.test_glam import _end_turn, _play

from sts2sim.engine import OrbState


def _orbs(*cards, relics=(), focus=0, orbs=("lightning",), channel_count=1):
    state = _state(*cards, relics=relics, statuses={"focus": focus})
    return state.model_copy(update={"combat": state.combat.model_copy(update={
        "orb_slots": 3, "orbs": tuple(OrbState(orb_id=x) for x in orbs),
        "metadata": {**state.combat.metadata,
                     "orbs_channeled_this_combat": {"lightning": channel_count}},
    })})


@pytest.mark.parametrize("relics,focus,damage", [
    ((), 0, 3), (("infused_core",), 0, 4), (("infused_core",), 2, 6),
    ((), -3, 0), (("infused_core",), -3, 1), (("infused_core",), -99, 1),
])
def test_passive_uses_owner_core_after_focus_clamps(relics, focus, damage):
    state = _end_turn(_reload(_orbs(_simple("held"), relics=relics, focus=focus)))
    assert state.combat.monsters[0].hp == 100 - damage


@pytest.mark.parametrize("relics,focus,damage", [
    ((), 0, 16), (("infused_core",), 0, 18), (("infused_core",), 2, 22),
    ((), -8, 0), (("infused_core",), -8, 2), (("infused_core",), -99, 2),
])
def test_dualcast_repeated_evoke_uses_core_after_focus_clamps(relics, focus, damage):
    state = _play(_reload(_orbs(_source("DUALCAST"), relics=relics, focus=focus)), "dualcast")
    assert not state.combat.orbs
    assert state.combat.monsters[0].hp == 100 - damage


@pytest.mark.parametrize("upgraded,expected", [(False, 11), (True, 20)])
def test_tesla_coil_directed_lightning_passives_use_the_same_modifier(upgraded, expected):
    state = _orbs(_source("TESLA_COIL", upgraded=upgraded), relics=("infused_core",),
                  orbs=("lightning", "lightning"))
    state = _play(_reload(state), "tesla_coil")
    assert state.combat.monsters[0].hp == 100 - expected
    assert len(state.combat.orbs) == 2


def test_negative_focus_tesla_passives_retain_core_point_on_each_orb():
    state = _orbs(_source("TESLA_COIL"), relics=("infused_core",), focus=-99,
                  orbs=("lightning", "lightning"))
    state = _play(_reload(state), "tesla_coil")
    assert state.combat.monsters[0].hp == 95  # Card3 plus two Lightning passives of1.


def test_voltaic_channel_overflow_evoke_uses_core_for_each_new_channel():
    state = _orbs(_source("VOLTAIC"), relics=("infused_core",),
                  orbs=("lightning",) * 3, channel_count=3)
    state = _play(_reload(state), "voltaic")
    assert state.combat.monsters[0].hp == 73  # Three overflow evokes at9.
    assert len(state.combat.orbs) == 3
    assert state.combat.metadata["orbs_channeled_this_combat"]["lightning"] == 6


def test_core_does_not_change_other_orb_values_or_use_foreign_relic_metadata():
    state = _orbs(_simple("held"), relics=("infused_core",), orbs=("frost",), focus=-1)
    state = _end_turn(_reload(state))
    block = [e.amount for e in state.replay_log[-1].events if e.source_id == "orb:frost"
             and e.kind == "player_block"]
    assert block == [1]
    assert state.combat.monsters[0].hp == 100
    state = _orbs(_simple("held"))
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "metadata": {**state.combat.metadata, "other_player_relics": ("infused_core",)},
    })})
    assert _end_turn(_reload(state)).combat.monsters[0].hp == 97
