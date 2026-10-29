"""CardModel.TemporaryStarCost order and its independent expiry scopes."""
from __future__ import annotations

from combat.test_card_play_status_runtime import _reload, _simple, _state
from combat.test_glam import _end_turn, _play
from combat.test_x_cost_source_runtime import _fight

from sts2sim.engine import CardInstance
from sts2sim.engine.source_cost_hooks import (
    expire,
    free_to_play,
    local_star_cost,
    preserve_changes,
    set_local_star_cost,
)
from sts2sim.engine.transitions import _card_resource_costs


def test_temporary_star_cost_uses_last_surviving_entry_and_independent_expiry():
    card = _simple("card", custom={"star_cost": 5})
    card = set_local_star_cost(card, 2, "combat")
    card = set_local_star_cost(card, 1, "played")
    card = set_local_star_cost(card, 0, "turn")
    assert local_star_cost(card) == 0
    card = expire(card, when_played=True)
    assert local_star_cost(card) == 0
    card = expire(card, turn_end=True)
    assert local_star_cost(card) == 2
    assert local_star_cost(expire(card, combat_end=True)) == 5


def test_older_combat_star_override_survives_later_turn_or_played_override():
    card = set_local_star_cost(_simple("card", custom={"star_cost": 4}), 2, "combat")
    card = set_local_star_cost(card, 0, "turn_or_played")
    assert local_star_cost(card) == 0
    assert local_star_cost(expire(card, when_played=True)) == 2


def test_legacy_consumed_free_flag_does_not_keep_stars_free():
    card = _simple("card", cost=2, custom={
        "star_cost": 3, "free_to_play_this_turn": True, "free_energy_consumed": True,
    })
    assert _card_resource_costs(card, available_star=20) == {"star": 3}


def test_source_free_turn_card_returns_to_original_star_cost_after_public_play_and_reload():
    card = free_to_play(_simple("card", cost=2, custom={"star_cost": 3}))
    state = _play(_state(card), "card")
    assert state.player.resources["star"] == 20
    played = _reload(state).combat.discard_pile[0]
    assert _card_resource_costs(played, available_star=20) == {"star": 3}
    # Return the already-played physical card without changing its modifiers.
    combat = state.combat.model_copy(update={"hand": (played,), "discard_pile": ()})
    state = _play(_reload(state.model_copy(update={"combat": combat})), "card")
    assert state.player.resources["star"] == 17
    assert state.player.energy == 18


def test_source_turn_expiry_restores_star_cost_on_a_retained_card():
    card = free_to_play(_simple("card", cost=2, custom={"star_cost": 3, "retain": True}))
    state = _reload(_end_turn(_state(card)))
    assert state.combat.hand[0].instance_id == "card"
    assert _card_resource_costs(state.combat.hand[0], available_star=20) == {"star": 3}
    state = _play(state, "card")
    assert state.player.resources["star"] == 17


def test_source_free_this_combat_stays_free_after_play_and_turn_expiry():
    card = free_to_play(_simple("card", cost=2, custom={"star_cost": 3}), combat=True)
    card = expire(expire(card, when_played=True), turn_end=True)
    assert local_star_cost(card) == 0


def test_negative_star_base_is_not_made_into_a_visible_zero_cost():
    card = set_local_star_cost(_simple("card", custom={"star_cost": -1}), 0, "combat")
    assert local_star_cost(card) == -1
    assert _card_resource_costs(card, available_star=20) == {}


def test_star_x_cost_spends_all_stars_even_when_marked_free():
    state = _fight("STARDUST", energy=7, stars=4)
    card = free_to_play(state.combat.hand[0])
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"hand": (card,)})})
    state = _play(_reload(state), card.instance_id)
    assert state.player.resources["star"] == 0
    assert state.player.energy == 7


def test_star_modifiers_survive_upgrade_and_use_upgraded_star_base_after_expiry():
    card = set_local_star_cost(_simple("card", custom={"star_cost": 3}), 0, "played")
    upgraded: CardInstance = _simple("card", upgraded=True, custom={"star_cost": 2})
    card = preserve_changes(card, upgraded)
    assert local_star_cost(card) == 0
    assert local_star_cost(expire(card, when_played=True)) == 2
