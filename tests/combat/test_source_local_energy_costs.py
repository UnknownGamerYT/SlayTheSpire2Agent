"""Source CardEnergyCost/LocalCostModifier insertion order and lifecycle."""
from combat.test_card_play_status_runtime import _simple

from sts2sim.engine.source_cost_hooks import (
    add_local_energy_cost,
    expire,
    free_to_play,
    local_cost,
    preserve_changes,
    set_local_energy_cost,
)


def test_source_cost_documented_ordered_example_preserves_later_relative_modifiers():
    card = _simple("card", cost=5)
    card = set_local_energy_cost(card, 2, "combat")
    card = add_local_energy_cost(card, 1, "combat")
    card = set_local_energy_cost(card, 0, "turn")
    card = add_local_energy_cost(card, 1, "combat")
    assert local_cost(card) == 1
    card = expire(card, turn_end=True)
    assert local_cost(card) == 4


def test_source_reduce_only_modifier_remains_available_after_an_older_modifier_expires():
    card = set_local_energy_cost(_simple("card", cost=3), 0, "turn_or_played")
    card = set_local_energy_cost(card, 1, "combat", reduce_only=True)
    assert local_cost(card) == 0
    assert local_cost(expire(card, turn_end=True)) == 1


def test_source_free_energy_expires_after_play_but_star_free_marker_lasts_turn():
    card = free_to_play(_simple("card", cost=2, custom={"star_cost": 3}))
    assert local_cost(card) == 0
    card = expire(card, when_played=True)
    assert local_cost(card) == 2
    assert card.custom["free_to_play_this_turn"]
    assert card.custom["free_energy_consumed"]
    card = free_to_play(card)
    assert local_cost(card) == 0
    assert not card.custom.get("free_energy_consumed")
    assert not expire(card, turn_end=True).custom.get("free_to_play_this_turn")


def test_source_upgrade_clamps_absolute_local_costs_and_preserves_ordered_modifiers():
    card = set_local_energy_cost(_simple("card", cost=3), 3, "combat")
    card = add_local_energy_cost(card, 1, "combat")
    upgraded = preserve_changes(card, card.model_copy(update={"cost": 2, "upgraded": True}))
    assert local_cost(upgraded) == 3
