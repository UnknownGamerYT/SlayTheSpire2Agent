from __future__ import annotations

import pytest
from test_named_monster_scripts import _combat
from test_remaining_monster_scripts import _end, _play, _with_child_definitions

from sts2sim import legal_actions, step
from sts2sim.engine.transitions import _card_from_spec
from sts2sim.mechanics import classify_raw_monster_specials


def _all_cards(combat):
    return combat.hand + combat.draw_pile + combat.discard_pile + combat.exhaust_pile


@pytest.mark.parametrize("monster_id,move_id,card_id,count", [
    ("CHOMPER", "SCREECH", "dazed", 3),
    ("LEAF_SLIME_M", "STICKY_SHOT", "slimed", 2),
    ("LEAF_SLIME_S", "GOOP", "slimed", 1),
    ("TWIG_SLIME_M", "STICKY_SHOT", "slimed", 1),
    ("SLIMED_BERSERKER", "VOMIT_ICHOR", "slimed", 10),
    ("PHROG_PARASITE", "INFECT", "infection", 3),
    ("WRIGGLER", "WRIGGLE", "infection", 1),
    ("NOISEBOT", "NOISE", "dazed", 4),
    ("HAUNTED_SHIP", "HAUNT", "dazed", 5),
])
def test_omitted_status_card_payloads_create_real_cards(monster_id, move_id, card_id, count):
    state = _combat(monster_id, moves=({"id": move_id, "intent": "Status"},))
    state = _end(state)
    assert sum(c.card_id == card_id for c in _all_cards(state.combat)) == count
    assert sum(e.kind == "card_created" for e in state.combat.last_events) == count


@pytest.mark.parametrize("ascension,count,strength", [(0, 3, 2), (9, 5, 3)])
def test_burning_growl_resolves_status_cards_and_scaled_strength(ascension, count, strength):
    state = _combat("TEST_SUBJECT", ascension=ascension, moves=({
        "id": "BURNING_GROWL", "intent": "Status + Buff",
        "powers": ({"power_id": "STRENGTH", "amount": 2},),
    },))
    state = _end(state)
    assert sum(c.card_id == "burn" for c in _all_cards(state.combat)) == count
    assert state.combat.monsters[0].statuses["strength"] == strength


def test_slimed_is_playable_draws_a_card_and_exhausts() -> None:
    state = _end(_combat("LEAF_SLIME_M", moves=({"id": "STICKY_SHOT", "intent": "Status"},)))
    slimed = next(c for c in state.combat.discard_pile if c.card_id == "slimed")
    draw_card = _card_from_spec({"card_id": "next", "type": "skill", "cost": 0}, 999)
    combat = state.combat.model_copy(update={
        "hand": (slimed,), "draw_pile": (draw_card,), "discard_pile": (),
    })
    state = state.model_copy(update={"combat": combat})
    before = combat.player.energy
    state = step(state, next(a for a in legal_actions(state) if a.type == "play_card"))
    assert state.combat.player.energy == before - 1
    assert state.combat.hand == (draw_card,)
    assert slimed.instance_id in {c.instance_id for c in state.combat.exhaust_pile}


def test_myte_toxic_cards_damage_in_hand_and_can_be_exhausted_by_playing() -> None:
    state = _end(_combat("MYTE", moves=({"id": "TOXIC", "intent": "Status"},)))
    toxic = [c for c in state.combat.hand if c.card_id == "toxic"]
    assert len(toxic) == 2
    assert all(c.cost == 1 and c.exhausts for c in toxic)
    before = state.combat.player.hp
    card = toxic[0]
    state = step(state, next(a for a in legal_actions(state)
                            if a.type == "play_card" and a.card_instance_id == card.instance_id))
    assert state.combat.player.hp == before
    assert card.instance_id in {c.instance_id for c in state.combat.exhaust_pile}
    state = _end(state)
    assert state.combat.player.hp == before - 5


def test_forbidden_incantation_gains_ritual_and_strength_each_own_turn() -> None:
    state = _combat("DEVOTED_SCULPTOR", moves=(
        {"id": "FORBIDDEN_INCANTATION", "intent": "Buff"},
        {"id": "SAVAGE", "intent": "Attack", "damage": {"normal": 12}},
    ))
    state = _end(state)
    assert state.combat.monsters[0].statuses == {"ritual": 9, "strength": 9}
    before = state.combat.player.hp
    state = _end(state)
    assert state.combat.player.hp == before - 21
    assert state.combat.monsters[0].statuses["strength"] == 18


def test_obscura_wail_buff_reaches_living_allies() -> None:
    state = _combat("THE_OBSCURA", moves=({"id": "SAIL", "intent": "Buff"},))
    ally = state.combat.monsters[0].model_copy(update={
        "monster_id": "ally", "statuses": {}, "intent_damage": 0,
    })
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (state.combat.monsters[0], ally),
    })})
    state = _end(state)
    assert all(m.statuses["strength"] >= 3 for m in state.combat.monsters)


def test_shrink_reduces_after_strength_and_expires_immediately_when_applier_dies() -> None:
    state = _combat("SHRINKER_BEETLE", moves=({"id": "SHRINKER", "intent": "Debuff"},))
    state = _end(state)
    assert state.combat.player.statuses["shrink"] == 1
    player = state.combat.player.model_copy(update={"statuses": {
        **state.combat.player.statuses, "strength": 4,
    }})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"player": player})})
    state = _play(state, effects={"damage": 6})
    assert state.combat.monsters[0].hp == 93  # (6 + 4) * .7
    state = _play(state, effects={"damage": 999})
    assert "shrink" not in state.player.statuses


def test_phrog_children_exist_for_later_effects_in_the_same_card() -> None:
    state = _combat("PHROG_PARASITE", statuses=(("INFESTED", 4),))
    state = _with_child_definitions(state, ("WRIGGLER",))
    state = _play(state, effects={"sequence": ({"damage": 999}, {"all_damage": 3})})
    children = [m for m in state.combat.monsters if m.hp > 0]
    assert len(children) == 4
    assert all(m.hp == 7 for m in children)


@pytest.mark.parametrize("move", [
    {"id": "EMPTY_BUFF", "intent": "Buff"},
    {"id": "EMPTY_DEBUFF", "intent": "Debuff"},
    {"id": "SCREECH", "intent": "Status"},
    {"id": "MIXED", "intent": "Attack + Status", "damage": {"normal": 5}},
    {"id": "MIXED", "intent": "Status + Buff",
     "powers": ({"power_id": "STRENGTH", "amount": 2},)},
])
def test_parsed_move_label_or_partial_attack_does_not_claim_full_effect_coverage(move):
    report = classify_raw_monster_specials({
        "id": "UNSCRIPTED", "min_hp": 10, "type": "Normal", "moves": (move,),
    })
    assert "move_effect_requires_payload_or_handler" in report.blocker_codes


def test_move_applied_arbitrary_power_is_not_implemented_merely_by_storing_stacks() -> None:
    report = classify_raw_monster_specials({
        "id": "UNSCRIPTED", "min_hp": 10, "type": "Normal", "moves": ({
            "id": "POWER", "intent": "Attack + Buff", "damage": {"normal": 1},
            "powers": ({"power_id": "MAGIC_UNIMPLEMENTED_POWER", "amount": 2},),
        },),
    })
    assert "move_power_requires_runtime_hook" in report.blocker_codes
