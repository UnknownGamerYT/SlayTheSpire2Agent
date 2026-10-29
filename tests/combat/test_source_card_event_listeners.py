from __future__ import annotations

import pytest
from combat.test_combat_engine_monsters import _play_card_targeting, _play_first_card
from combat.test_monster_innate_hooks import _card, _combat, _monster
from combat.test_source_card_conditions import _no_stubs, _source
from combat.test_source_card_timing import _power

from sts2sim import load_state, serialize


@pytest.mark.parametrize("power,applied", [("strangle", 2), ("oblivion", 3)])
def test_bound_enemy_card_play_listener_follows_original_target_after_save_reload(
    power: str,
    applied: int,
) -> None:
    description = (
        "Deal 8 damage. Whenever you play a card this turn, the enemy loses 2 HP."
        if power == "strangle"
        else "Whenever you play a card this turn, apply 3 Doom to the enemy."
    )
    card = _source(
        power,
        description,
        damage=8 if power == "strangle" else 0,
        target="enemy",
        powers_applied=({"power_key": power, "amount": applied},),
    )
    state = _combat(
        _monster("OTHER", {}), _monster("VICTIM", {}), deck=(card, _card("filler", damage=0))
    )
    target = state.combat.monsters[1].monster_id
    state = _play_card_targeting(state, power, target)
    assert state.combat.monsters[1].statuses[power] == applied
    assert state.combat.monsters[1].hp == (92 if power == "strangle" else 100)
    assert state.combat.monsters[1].statuses.get("doom", 0) == 0
    state = _play_card_targeting(
        load_state(serialize(state)), "filler", state.combat.monsters[0].monster_id
    )
    first, second = state.combat.monsters
    assert first.hp == 100
    assert first.statuses.get("doom", 0) == 0
    if power == "strangle":
        assert second.hp == 90
    else:
        assert second.statuses["doom"] == 3
    _no_stubs(state)


@pytest.mark.parametrize("power", ["strangle", "oblivion"])
def test_artifact_blocks_bound_listener_upfront_instead_of_one_later_trigger(power: str) -> None:
    description = (
        "Deal 8 damage. Whenever you play a card this turn, the enemy loses 2 HP."
        if power == "strangle"
        else "Whenever you play a card this turn, apply 3 Doom to the enemy."
    )
    card = _source(power, description, damage=8 if power == "strangle" else 0, target="enemy")
    state = _combat(_monster("VICTIM", {"ARTIFACT": 1}), deck=(card, _card("filler", damage=0)))
    state = _play_first_card(state, power)
    assert not state.combat.metadata.get("combat_triggers")
    assert state.combat.monsters[0].statuses.get("artifact", 0) == 0
    state = _play_first_card(state, "filler")
    assert state.combat.monsters[0].hp == (92 if power == "strangle" else 100)
    assert state.combat.monsters[0].statuses.get("doom", 0) == 0
    _no_stubs(state)


def test_page_storm_waits_for_an_ethereal_draw_and_chains_additional_draws() -> None:
    power = _power("pagestorm", "Whenever you draw an Ethereal card, draw 1 card.", cards_draw=1)
    draw = _source("draw", "Draw 1 card.", cards_draw=1)
    state = _combat(_monster("VICTIM", {}), deck=(power, draw, _card("filler", damage=0)))
    filler = next(card for card in state.combat.hand if card.card_id == "filler")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "draw_pile": (
                        filler.model_copy(
                            update={"instance_id": "ethereal", "tags": ("ethereal",)}
                        ),
                        filler.model_copy(update={"instance_id": "ordinary"}),
                    ),
                }
            )
        }
    )
    state = _play_first_card(state, "pagestorm")
    assert len(state.combat.draw_pile) == 2
    state = _play_first_card(state, "draw")
    assert not state.combat.draw_pile
    assert {card.instance_id for card in state.combat.hand} >= {"ethereal", "ordinary"}
    _no_stubs(state)


@pytest.mark.parametrize("zone", ["hand", "draw_pile", "discard_pile", "exhaust_pile"])
def test_melancholy_death_cost_reduction_reaches_all_combat_zones(zone: str) -> None:
    melancholy = _source(
        "melancholy",
        "Gain 13 Block. Reduce this card's cost by [energy:1] whenever ANYONE dies.",
        block=13,
        energy_gain=1,
        cost=3,
    )
    victim = _monster("VICTIM", {})
    victim.update(min_hp=5, max_hp=5)
    state = _combat(victim, _monster("SURVIVOR", {}), deck=(melancholy, _card("kill", damage=10)))
    watched = next(card for card in state.combat.hand if card.card_id == "melancholy")
    updates = {"hand": tuple(card for card in state.combat.hand if card.card_id != "melancholy")}
    updates[zone] = updates.get(zone, ()) + (watched,)
    state = state.model_copy(update={"combat": state.combat.model_copy(update=updates)})
    state = _play_card_targeting(state, "kill", state.combat.monsters[0].monster_id)
    assert (
        next(card for card in getattr(state.combat, zone) if card.card_id == "melancholy").cost == 2
    )
    assert state.player.energy == 3
    assert next(card for card in state.master_deck if card.card_id == "melancholy").cost == 3
    restored = load_state(serialize(state))
    assert (
        next(card for card in getattr(restored.combat, zone) if card.card_id == "melancholy").cost
        == 2
    )
    _no_stubs(state)


def test_void_loses_energy_when_drawn_and_never_gives_energy() -> None:
    void = _source(
        "void",
        "Whenever you draw this card, lose [energy:1].",
        type="Status",
        cost=-2,
        energy_gain=1,
    )
    draw = _source("draw", "Draw 1 card.", cards_draw=1)
    state = _combat(_monster("VICTIM", {}), deck=(void, draw))
    card = next(card for card in state.combat.hand if card.card_id == "void")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": tuple(card for card in state.combat.hand if card.card_id != "void"),
                    "draw_pile": (card,),
                    "player": state.combat.player.model_copy(update={"energy": 3}),
                }
            )
        }
    )
    state = _play_first_card(state, "draw")
    assert state.combat.player.energy == 2
    _no_stubs(state)


@pytest.mark.parametrize(
    "power,expected_hp,expected_doom", [("strangle", 82, 0), ("oblivion", 100, 3)]
)
def test_replaying_same_physical_bound_card_triggers_prior_stacks_only(
    power: str,
    expected_hp: int,
    expected_doom: int,
) -> None:
    description = (
        "Deal 8 damage. Whenever you play a card this turn, the enemy loses 2 HP."
        if power == "strangle"
        else "Whenever you play a card this turn, apply 3 Doom to the enemy."
    )
    card = _source(power, description, damage=8 if power == "strangle" else 0, target="enemy")
    card["enchantments"] = ({"keyword": "replay", "amount": 1},)
    state = _combat(_monster("VICTIM", {}), deck=(card,))
    state = _play_first_card(state, power)
    assert state.combat.monsters[0].hp == expected_hp
    assert state.combat.monsters[0].statuses.get("doom", 0) == expected_doom
    _no_stubs(state)
