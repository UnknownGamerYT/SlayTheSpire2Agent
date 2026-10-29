"""Source phase outcomes for optional retention and sequential AutoPost plays."""
from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _source, _state
from combat.test_glam import _end_turn, _play

from sts2sim import legal_actions, step
from sts2sim.engine import CardEnchantment, OrbState
from sts2sim.engine.rng import STS2_STABLE_DERIVATION, RunRng, capture_random_state
from sts2sim.engine.transitions import _energy_cost


def _choose(state, kind, identifier=None):
    return step(_reload(state), next(a for a in legal_actions(state)
        if a.type == kind and (identifier is None or a.card_instance_id == identifier)))


def test_source_retention_sees_completed_orb_and_hand_effects_before_its_choice():
    ordinary = _simple("ordinary")
    state = _state(ordinary, _source("BURN"), _source("DAZED"),
                   statuses={"well_laid_plans": 1})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "orbs": (OrbState(orb_id="frost", value=2),), "orb_slots": 3,
    })})
    state = _end_turn(_reload(state))
    assert state.player.hp == 80  # Frost resolves before Burn.
    assert [c.instance_id for c in state.combat.hand] == ["ordinary"]
    assert [c.card_id for c in state.combat.discard_pile] == ["burn"]
    assert [c.card_id for c in state.combat.exhaust_pile] == ["dazed"]
    assert state.combat.pending_choices[0].candidate_ids == ("ordinary",)
    assert state.combat.pending_choices[0].metadata["source_phase"] == "before_flush_late"
    state = _choose(state, "choose_card", "ordinary")
    assert state.combat.turn == 2
    assert not state.combat.pending_choices
    assert [c.instance_id for c in state.combat.hand] == ["ordinary"]
    assert not state.combat.hand[0].custom.get("retain")


def test_eye_exhausts_hand_before_late_retention_and_really_starts_extra_turn():
    state = _state(_simple("held"), statuses={"well_laid_plans": 1},
                   relics=("paels_eye",), draw=(_simple("fresh"),), draw_per_turn=1)
    state = _end_turn(_reload(state))
    assert state.combat.turn == 2
    assert not state.combat.pending_choices
    assert [c.instance_id for c in state.combat.exhaust_pile] == ["held"]
    assert [c.instance_id for c in state.combat.hand] == ["fresh"]
    assert any(e.kind == "extra_turn_started" for e in state.replay_log[-1].events)


@pytest.mark.parametrize("relics,statuses", [
    (("runic_pyramid",), {}), (("ringing_triangle",), {}), ((), {"retain_hand": 1}),
])
def test_retention_power_does_not_prompt_when_source_should_flush_is_false(relics, statuses):
    state = _state(_simple("held"), statuses={"well_laid_plans": 1, **statuses}, relics=relics)
    state = _end_turn(_reload(state))
    assert not state.combat.pending_choices
    assert state.combat.turn == 2
    assert [c.instance_id for c in state.combat.hand] == ["held"]


def test_retention_filter_excludes_already_retained_cards_and_sees_before_flush_discount():
    retained = _simple("retained", custom={"retain": True})
    discounted = _simple("discounted", cost=2,
                         enchantments=(CardEnchantment(keyword="slumbering_essence", amount=1),))
    state = _state(retained, discounted, statuses={"well_laid_plans": 2})
    state = _end_turn(_reload(state))
    assert state.combat.pending_choices[0].candidate_ids == ("discounted",)
    card = next(c for c in state.combat.hand if c.instance_id == "discounted")
    assert _energy_cost(card, state.player.energy, combat=state.combat) == 1
    state = _choose(state, "choose_card", "discounted")
    assert state.combat.turn == 2
    assert {c.instance_id for c in state.combat.hand} == {"retained", "discounted"}
    state = _play(state, "discounted")
    assert state.player.energy == 19


def test_optional_retention_skip_finishes_the_same_end_turn_after_reload():
    state = _play(_state(_source("WELL_LAID_PLANS"), _simple("held")), "well_laid_plans")
    assert state.player.statuses["well_laid_plans"] == 1
    assert not state.combat.metadata.get("combat_triggers")
    pending = _end_turn(_reload(state))
    result = _choose(pending, "proceed")
    assert result.combat.turn == 2
    assert not result.combat.hand
    assert not result.combat.pending_choices
    assert not result.combat.metadata.get("source_end_turn_resume")
    assert _choose(_reload(pending), "proceed") == result


def _stampede_choice():
    state = _state(_source("DAGGER_THROW"), _simple("held"),
                   statuses={"stampede": 2, "well_laid_plans": 1},
                   draw=(_source("HEADBUTT"),))
    return state.model_copy(update={
        "character_id": "IRONCLAD",
        "rng": capture_random_state(RunRng("sequential-auto", derivation=STS2_STABLE_DERIVATION)),
    })


def test_stampede_waits_for_first_discard_before_selecting_another_attack():
    state = _end_turn(_reload(_stampede_choice()))
    assert state.combat.pending_choices[0].kind == "discard"
    assert {c.instance_id for c in state.combat.hand} == {"held", "headbutt"}
    assert state.rng.streams["shuffle"].draws == 1
    assert state.combat.monsters[0].hp == 91
    state = _choose(state, "discard_card", "headbutt")
    assert state.combat.pending_choices[0].kind == "add_retain"
    assert state.combat.pending_choices[0].candidate_ids == ("held",)
    assert state.rng.streams["shuffle"].draws == 1  # Empty second selection consumes nothing.
    assert state.combat.monsters[0].hp == 91
    state = _choose(state, "proceed")
    assert state.combat.turn == 2
    assert not state.combat.pending_choices


def test_stampede_resamples_current_hand_after_resolved_discard_and_waits_for_headbutt():
    state = _end_turn(_reload(_stampede_choice()))
    state = _choose(state, "discard_card", "held")
    assert state.combat.pending_choices[0].kind == "move_to_draw_top"
    assert state.rng.streams["shuffle"].draws == 2
    assert state.combat.monsters[0].hp == 82
    assert not state.combat.hand
    state = _choose(state, "choose_card", "held")
    assert state.combat.turn == 2
    assert not state.combat.pending_choices
    assert state.combat.draw_pile[0].instance_id == "held"


def test_native_stampede_power_precedes_exhaust_pile_auto_post_card_listener():
    state = _state(_source("BASH"), statuses={"stampede": 1})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "exhaust_pile": (_source("HOWL_FROM_BEYOND"),),
    })})
    state = _end_turn(_reload(state))
    assert state.combat.monsters[0].hp == 68  # Bash8 then Vulnerable Howl16*1.5.
    plays = [e.source_id for e in state.replay_log[-1].events if e.kind == "card_played_by_trigger"]
    assert plays == ["power:stampede", "howl_from_beyond"]


def test_lethal_stampede_stops_auto_post_before_later_card_listeners():
    state = _state(_source("BASH"), statuses={"stampede": 1})
    monster = state.combat.monsters[0].model_copy(update={"hp": 5})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (monster,), "exhaust_pile": (_source("HOWL_FROM_BEYOND"),),
    })})
    state = _end_turn(_reload(state))
    assert state.phase.value == "reward"
    assert [c.card_id for c in state.combat.exhaust_pile] == ["howl_from_beyond"]
    assert not any(e.kind == "howl_from_beyond_triggered" for e in state.replay_log[-1].events)
