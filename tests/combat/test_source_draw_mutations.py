from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _source, _state
from combat.test_glam import _end_turn, _play

from sts2sim import legal_actions, step


def _card(state, instance_id):
    return next(
        card
        for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile")
        for card in getattr(state.combat, zone)
        if card.instance_id == instance_id
    )


def test_kingly_kick_reduces_each_draw_floored_at_zero_and_keeps_master_cost() -> None:
    kick = _source("KINGLY_KICK")
    draw = _simple("draw", effects={"draw": 1})
    state = _state(draw, draw=(kick,))
    state = state.model_copy(update={"master_deck": (kick, draw)})
    state = _play(state, "draw")
    assert _card(state, "kingly_kick").cost == 3
    for expected in (2, 1, 0, 0):
        drawn = _card(state, "kingly_kick")
        state = state.model_copy(
            update={
                "combat": state.combat.model_copy(
                    update={
                        "hand": (),
                        "draw_pile": (drawn,),
                        "discard_pile": (draw,),
                        "draw_per_turn": 1,
                    }
                )
            }
        )
        state = _end_turn(_reload(state))
        assert _card(state, "kingly_kick").cost == expected
    assert state.master_deck[0].cost == 4


@pytest.mark.parametrize("upgraded,damage,increase", [(False, 8, 4), (True, 10, 6)])
def test_kingly_punch_increases_its_own_damage_each_draw_this_combat(
    upgraded, damage, increase
) -> None:
    punch = _source("KINGLY_PUNCH", upgraded=upgraded)
    other = punch.model_copy(update={"instance_id": "other-copy"})
    draw = _simple("draw", effects={"draw": 1})
    state = _play(_state(draw, other, draw=(punch,)), "draw")
    drawn = _card(state, "kingly_punch")
    assert drawn.effects["sequence"][0]["damage"] == damage + increase
    assert _card(state, "other-copy").effects["sequence"][0]["damage"] == damage
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (),
                    "draw_pile": (drawn,),
                    "discard_pile": (),
                    "draw_per_turn": 1,
                }
            )
        }
    )
    state = _end_turn(_reload(state))
    state = _play(state, "kingly_punch")
    assert state.combat.monsters[0].hp == 100 - damage - 2 * increase


def test_moving_kingly_cards_to_hand_does_not_count_as_draw() -> None:
    kick = _source("KINGLY_KICK")
    move = _simple(
        "move", effects={"choose_card": {"action": "move_to_hand", "zone": "draw_pile", "count": 1}}
    )
    state = _play(_state(move, draw=(kick,)), "move")
    state = step(
        state, next(action for action in legal_actions(state) if action.type == "choose_card")
    )
    assert _card(state, "kingly_kick").cost == 4
    assert not any(
        event.kind == "card_cost_changed_on_draw" for event in state.replay_log[-1].events
    )


@pytest.mark.parametrize("zone", ["hand", "draw_pile", "discard_pile", "exhaust_pile"])
def test_rocket_punch_discounts_in_every_zone_until_its_next_play(zone) -> None:
    rocket = _source("ROCKET_PUNCH")
    generator = _simple(
        "generator",
        effects={
            "add_card_to_hand": {
                "card": {
                    "card_id": "created_status",
                    "type": "status",
                    "cost": -1,
                    "custom": {"unplayable": True},
                }
            }
        },
    )
    state = _state(generator)
    state = state.model_copy(
        update={
            "master_deck": (rocket, generator),
            "combat": state.combat.model_copy(
                update={zone: getattr(state.combat, zone) + (rocket,)}
            ),
        }
    )
    state = _play(state, "generator")
    assert _card(state, "rocket_punch").cost == 0
    assert state.master_deck[0].cost == 2
    discounted = _card(state, "rocket_punch")
    # The discount survives a turn and save/reload in any combat pile.
    state = _end_turn(_reload(state))
    assert _card(state, "rocket_punch").cost == 0
    updates = {
        name: tuple(
            card for card in getattr(state.combat, name) if card.instance_id != "rocket_punch"
        )
        for name in ("hand", "draw_pile", "discard_pile", "exhaust_pile")
    }
    updates["hand"] = updates["hand"] + (discounted,)
    state = state.model_copy(update={"combat": state.combat.model_copy(update=updates)})
    before = state.player.energy
    state = _play(_reload(state), "rocket_punch")
    assert state.player.energy == before
    assert _card(state, "rocket_punch").cost == 2
    assert "rocket_punch_cost_before_discount" not in _card(state, "rocket_punch").custom


def test_creating_nonstatus_card_does_not_discount_rocket_punch() -> None:
    rocket = _source("ROCKET_PUNCH")
    generator = _simple(
        "generator", effects={"add_card_to_hand": {"card": {"card_id": "skill", "type": "skill"}}}
    )
    state = _play(_state(generator, rocket), "generator")
    assert _card(state, "rocket_punch").cost == 2


def test_upgraded_rocket_punch_draws_two_source_cards() -> None:
    rocket = _source("ROCKET_PUNCH", upgraded=True)
    junk = (_simple("one"), _simple("two"))
    state = _play(_state(rocket, draw=junk), "rocket_punch")
    assert {card.instance_id for card in state.combat.hand} == {"one", "two"}
    assert state.combat.monsters[0].hp == 86


@pytest.mark.parametrize("upgraded,damage", [(False, 40), (True, 50)])
def test_the_bomb_waits_three_turns_then_explodes_only_once(upgraded, damage) -> None:
    state = _play(_state(_source("THE_BOMB", upgraded=upgraded)), "the_bomb")
    for _ in range(2):
        state = _end_turn(_reload(state))
        assert state.combat.monsters[0].hp == 100
    state = _end_turn(_reload(state))
    assert state.combat.monsters[0].hp == 100 - damage
    state = _end_turn(_reload(state))
    assert state.combat.monsters[0].hp == 100 - damage
    assert not state.combat.metadata.get("combat_triggers")


def test_well_laid_plans_allows_skipping_retention() -> None:
    state = _play(_state(_source("WELL_LAID_PLANS"), _simple("unused")), "well_laid_plans")
    state = _end_turn(state)
    assert state.combat.pending_choices[0].min_choices == 0
    state = step(
        _reload(state), next(action for action in legal_actions(state) if action.type == "proceed")
    )
    state = _end_turn(state)
    assert not state.combat.hand
    assert not any(event.kind == "cards_retained" for event in state.replay_log[-1].events)


def test_well_laid_plans_partial_retention_is_one_turn_not_permanent() -> None:
    state = _play(
        _state(_source("WELL_LAID_PLANS", upgraded=True), _simple("one"), _simple("two")),
        "well_laid_plans",
    )
    state = _end_turn(state)
    state = step(
        state,
        next(
            action
            for action in legal_actions(state)
            if action.type == "choose_card" and action.card_instance_id == "one"
        ),
    )
    assert _card(state, "one").custom["retain"] is True
    state = step(
        _reload(state), next(action for action in legal_actions(state) if action.type == "proceed")
    )
    state = _end_turn(state)
    assert [card.instance_id for card in state.combat.hand] == ["one"]
    assert not _card(state, "one").custom.get("retain")
    # Skip the following turn's fresh retention choice and the old selection expires.
    state = _end_turn(_reload(state))
    state = step(state, next(action for action in legal_actions(state) if action.type == "proceed"))
    state = _end_turn(state)
    assert not state.combat.hand


def test_installed_source_power_leaves_combat_piles_without_exhaust_or_reshuffle() -> None:
    power = _source("AFTERIMAGE")
    filler = _simple("filler", effects={"block": 1})
    state = _play(
        _state(power, draw=(filler,), relics=("charons_ashes",), draw_per_turn=5), "afterimage"
    )
    assert state.player.statuses["afterimage"] == 1
    assert not any(
        card.card_id == "afterimage"
        for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile")
        for card in getattr(state.combat, zone)
    )
    assert state.master_deck[0].card_id == "afterimage"
    assert state.combat.monsters[0].hp == 100
    assert not any(event.kind == "card_exhausted" for event in state.replay_log[-1].events)
    for _ in range(3):
        state = _end_turn(_reload(state))
        assert all(card.card_id != "afterimage" for card in state.combat.hand)
        assert state.player.statuses["afterimage"] == 1
        state = _play(state, "filler")


def test_replayed_power_installs_both_plays_without_moving_or_exhausting_a_copy() -> None:
    from sts2sim.engine import CardEnchantment

    power = _source("AFTERIMAGE").model_copy(
        update={
            "enchantments": (CardEnchantment(keyword="glam", amount=1),),
        }
    )
    state = _play(_state(power, relics=("charons_ashes",)), "afterimage")
    assert state.player.statuses["afterimage"] == 2
    assert len(state.combat.cards_played_this_turn) == 2
    assert state.player.energy == 19
    assert all(
        not getattr(state.combat, zone)
        for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile")
    )
    assert state.combat.monsters[0].hp == 100
    assert not any(event.kind == "card_exhausted" for event in state.replay_log[-1].events)
