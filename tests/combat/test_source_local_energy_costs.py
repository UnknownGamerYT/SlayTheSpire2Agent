"""Source CardEnergyCost/LocalCostModifier insertion order and lifecycle."""
from combat.test_card_play_status_runtime import _simple

from sts2sim.engine.source_cost_hooks import (
    add_local_energy_cost,
    expire,
    free_to_play,
    local_cost,
    local_star_cost,
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


def test_source_free_costs_expire_after_play_and_allow_later_reapplication():
    card = free_to_play(_simple("card", cost=2, custom={"star_cost": 3}))
    assert local_cost(card) == 0
    card = expire(card, when_played=True)
    assert local_cost(card) == 2
    assert local_star_cost(card) == 3
    assert card.custom["free_to_play_this_turn"]
    assert card.custom["free_energy_consumed"]
    card = free_to_play(card)
    assert local_cost(card) == 0
    assert not card.custom.get("free_energy_consumed")
    assert not expire(card, turn_end=True).custom.get("free_to_play_this_turn")


def test_source_upgrade_clamps_absolute_local_costs_and_preserves_ordered_modifiers():
    card = set_local_energy_cost(_simple("card", cost=3), 3, "combat")
    card = add_local_energy_cost(card, 1, "combat")
    upgraded = preserve_changes(card, _simple("card", cost=2, upgraded=True))
    assert local_cost(upgraded) == 3


def test_public_free_local_energy_still_receives_normal_global_cost_increase():
    from combat.test_card_play_status_runtime import _state
    from combat.test_glam import _play

    card = _simple("card", cost=2, custom={"free_to_play_this_turn": True})
    state = _play(_state(card, statuses={"borrowed_time": 1}), "card")
    assert state.player.energy == 19


def test_public_late_free_skill_power_overrides_normal_borrowed_time_increase():
    from combat.test_card_play_status_runtime import _state
    from combat.test_glam import _play

    state = _play(_state(_simple("card", cost=2), statuses={
        "source_core_free_skill": 1, "borrowed_time": 1,
    }), "card")
    assert state.player.energy == 20


def test_public_enlightenment_reduce_only_survives_earlier_free_modifier_expiration():
    from combat.test_card_play_status_runtime import _reload, _source, _state
    from combat.test_glam import _end_turn, _play

    card = _simple("card", cost=3, custom={"free_to_play_this_turn": True, "retain": True})
    state = _play(_state(_source("ENLIGHTENMENT", upgraded=True), card), "enlightenment")
    assert state.combat.hand[0].cost == 0
    state = _reload(_end_turn(state))
    assert state.combat.hand[0].cost == 1
    before = state.player.energy
    state = _play(state, "card")
    assert state.player.energy == before - 1


def test_public_later_combat_cost_override_survives_snecko_turn_modifier_expiration():
    from combat.test_card_play_status_runtime import _reload, _state
    from combat.test_glam import _end_turn, _play
    from combat.test_stable_item_contracts import _potion

    victim = _simple("victim", cost=4, custom={"retain": True})
    setter = _simple("setter", effects={"set_hand_cost": {"cost": 1, "duration": "combat"}})
    state = _state(victim, setter, draw=tuple(_simple(f"draw{i}") for i in range(6)))
    state = _potion(state, "snecko_oil")
    state = _play(state, "setter")
    state = _reload(_end_turn(state))
    assert next(c for c in state.combat.hand if c.instance_id == "victim").cost == 1
    before = state.player.energy
    state = _play(state, "victim")
    assert state.player.energy == before - 1


def test_public_later_free_modifier_overrides_snecko_then_both_expire_on_play():
    from combat.test_card_play_status_runtime import _reload, _state
    from combat.test_glam import _play
    from combat.test_stable_item_contracts import _potion

    victim = _simple("victim", cost=2)
    setter = _simple("setter", effects={"set_hand_free_to_play_this_turn": True})
    state = _state(victim, setter, draw=tuple(_simple(f"draw{i}") for i in range(6)))
    state = _potion(state, "snecko_oil")
    state = _play(state, "setter")
    before = state.player.energy
    state = _play(_reload(state), "victim")
    assert state.player.energy == before
    assert next(c for c in state.combat.discard_pile if c.instance_id == "victim").cost == 2


def test_public_stomp_cost_reduction_before_later_absolute_set_is_overridden():
    from combat.test_card_play_status_runtime import _source, _state
    from combat.test_glam import _play

    attack = _simple("attack", card_type="attack", target="enemy", effects={"damage": 1})
    setter = _simple("setter", effects={"set_hand_cost": {"cost": 2, "duration": "combat"}})
    state = _play(_state(attack, setter, _source("STOMP")), "attack")
    state = _play(state, "setter")
    before = state.player.energy
    state = _play(state, "stomp")
    assert state.player.energy == before - 2
    assert state.combat.discard_pile[-1].cost == 1  # Own BeforeCardPlayed reduces future cost.


def test_public_pinpoint_cost_reduction_after_absolute_set_preserves_source_order():
    from combat.test_card_play_status_runtime import _source, _state
    from combat.test_glam import _play

    skill = _simple("skill")
    setter = _simple("setter", effects={"set_hand_cost": {"cost": 2, "duration": "combat"}})
    state = _play(_state(skill, setter, _source("PINPOINT")), "skill")
    state = _play(state, "setter")
    before = state.player.energy
    state = _play(state, "pinpoint")
    assert state.player.energy == before - 1
