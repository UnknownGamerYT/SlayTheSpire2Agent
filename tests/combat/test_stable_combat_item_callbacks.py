"""Independent stable relic phase and pre-application modifier outcomes."""
from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _state
from combat.test_combat_engine_cards import _enter_combat
from combat.test_glam import _end_turn, _play
from combat.test_stable_item_contracts import _potion

from sts2sim.engine import CardEnchantment
from sts2sim.engine.transitions import _resolve_extra_card_play


def _autoplay(state, card):
    combat = state.combat.model_copy(update={"hand": tuple(
        item for item in state.combat.hand if item.instance_id != card.instance_id
    )})
    combat, rng, events = _resolve_extra_card_play(
        state, combat, state.rng, card, target_id=None, energy_spent=0,
        source_id="stable_auto_fixture", source_metadata={}, move_physical_card=True,
    )
    return state.model_copy(update={"combat": combat.model_copy(update={"last_events": events}),
                                    "player": combat.player, "rng": rng})


def _extra_turn(state):
    return any(e.kind == "extra_turn_added" and e.metadata.get("relic_id") == "paels_eye"
               for e in state.combat.last_events)


def test_paels_eye_ignores_auto_plays_and_persists_once_per_combat():
    state = _state(_simple("automatic"), relics=("paels_eye",))
    state = _autoplay(state, state.combat.hand[0])
    assert state.combat.cards_played_this_turn
    state = _end_turn(_reload(state))
    assert _extra_turn(state)
    state = _end_turn(_reload(state))
    assert not _extra_turn(state)


def test_paels_eye_counts_manual_play_even_if_it_replays():
    card = _simple("manual", enchantments=(CardEnchantment(keyword="replay", amount=1),))
    state = _play(_state(card, relics=("paels_eye",)), "manual")
    assert len(state.combat.cards_played_this_turn) == 2
    assert not _extra_turn(_end_turn(_reload(state)))


def test_paels_eye_whispering_earring_exception_only_on_first_turn():
    state = _state(_simple("held"), relics=("paels_eye", "whispering_earring"))
    state = _end_turn(_reload(state))
    assert not _extra_turn(state)
    state = _end_turn(_reload(state))
    assert _extra_turn(state)


def test_top_checks_empty_hand_after_potion_completion():
    state = _state(_simple("placeholder"), draw=(_simple("replacement"),),
                   relics=("unceasing_top",))
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"hand": ()})})
    state = _potion(_reload(state), "strength_potion")
    assert [c.card_id for c in state.combat.hand] == ["replacement"]


def test_top_does_not_draw_before_potions_own_draw_sequence_finishes():
    state = _state(_simple("placeholder"), draw=tuple(_simple(f"d{i}") for i in range(6)),
                   relics=("unceasing_top",))
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"hand": ()})})
    state = _potion(_reload(state), "swift_potion")
    assert len(state.combat.hand) == 3


@pytest.mark.parametrize("phase", ["start", "end"])
def test_top_skips_actions_outside_source_valid_turn_phases(phase):
    state = _state(_simple("automatic"), draw=(_simple("replacement"),),
                   relics=("unceasing_top",))
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "metadata": {**state.combat.metadata, "player_turn_phase": phase},
    })})
    assert not _autoplay(state, state.combat.hand[0]).combat.hand


def test_mittens_skips_first_turn_innate_before_opening_draw():
    state = _enter_combat((
        {"card_id": "innate", "type": "skill", "target": "self", "cost": 0,
         "keywords": ["innate"]},
        {"card_id": "ordinary", "type": "skill", "target": "self", "cost": 0},
    ), relics=("toasty_mittens",), flags={"draw_per_turn": 1})
    assert [c.card_id for c in state.combat.hand] == ["innate"]
    assert [c.card_id for c in state.combat.exhaust_pile] == ["ordinary"]
    assert state.player.statuses["strength"] == 1


def test_mittens_shuffles_discard_then_exhausts_with_shuffle_and_exhaust_callbacks():
    state = _state(_simple("unused"), relics=("toasty_mittens", "the_abacus", "charons_ashes"),
                   draw_per_turn=1)
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "hand": (), "draw_pile": (), "discard_pile": tuple(_simple(f"d{i}") for i in range(3)),
    })})
    state = _end_turn(_reload(state))
    assert len(state.combat.exhaust_pile) == 1
    assert len(state.combat.hand) == 1
    assert state.player.block == 6
    assert state.combat.monsters[0].hp == 97
    assert any(e.kind == "card_exhausted" for e in state.combat.last_events)


def _debuff_state(card, *, artifact=0, relics=("unsettling_lamp",)):
    state = _state(card, relics=relics)
    monster = state.combat.monsters[0].model_copy(update={"statuses": {"artifact": artifact}})
    other = monster.model_copy(update={"monster_id": "other"})
    return state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (monster, other),
    })})


@pytest.mark.parametrize("artifact", [0, 1])
def test_lamp_doubles_every_debuff_of_first_card_on_every_receiver_even_after_artifact(artifact):
    card = _simple("debuff", target="all_enemies", effects={"sequence": [
        {"apply_status": {"target": "all_enemies", "weak": 2}},
        {"apply_status": {"target": "all_enemies", "vulnerable": 3}},
    ]}, custom={"replay": 1})
    state = _play(_reload(_debuff_state(card, artifact=artifact)), "debuff")
    for monster in state.combat.monsters:
        assert monster.statuses["weak"] == (2 if artifact else 6)
        assert monster.statuses["vulnerable"] == 9
    assert state.combat.metadata["relic_counters"]["unsettling_lamp"] == 1


def test_lamp_only_arms_from_enemy_debuff_but_doubles_later_self_debuff_from_same_card():
    card = _simple("mixed", target="enemy", effects={"sequence": [
        {"apply_status": {"target": "self", "weak": 1}},
        {"apply_status": {"target": "enemy", "vulnerable": 1}},
        {"apply_status": {"target": "self", "frail": 1}},
    ]})
    state = _play(_debuff_state(card), "mixed")
    assert state.player.statuses["weak"] == 1
    assert state.player.statuses["frail"] == 2
    assert state.combat.monsters[0].statuses["vulnerable"] == 2


def test_snecko_addition_precedes_lamp_multiplication_and_artifact():
    card = _simple("toxin", target="enemy", effects={"apply_status": {
        "target": "enemy", "poison": 2,
    }})
    state = _play(_debuff_state(card, relics=("snecko_skull", "unsettling_lamp")), "toxin")
    assert state.combat.monsters[0].statuses["poison"] == 6


def test_lamp_does_not_arm_on_potion_but_does_double_attack_debuff_rider():
    card = _simple("attack", card_type="attack", target="enemy", effects={"damage": 5})
    state = _debuff_state(card)
    state = _potion(state, "weak_potion")
    assert not state.combat.metadata.get("unsettling_lamp_triggering_card")
    player = state.player.model_copy(update={"statuses": {"reaper_form": 1}})
    state = state.model_copy(update={"player": player, "combat": state.combat.model_copy(
        update={"player": player},
    )})
    state = _play(_reload(state), "attack")
    assert state.combat.monsters[0].statuses["doom"] == 10


def test_helmet_doubles_start_relic_strength_before_any_card_and_only_once():
    state = _enter_combat((
        {"card_id": "held", "type": "skill", "target": "self", "cost": 0},
    ), relics=("ruined_helmet", "vajra"), flags={"draw_per_turn": 1})
    assert state.player.statuses["strength"] == 2
    state = _potion(_reload(state), "strength_potion")
    assert state.player.statuses["strength"] == 4


def test_helmet_doubles_ritual_strength_without_a_card_source():
    state = _state(_simple("held"), statuses={"ritual": 3}, relics=("ruined_helmet",))
    state = _end_turn(_reload(state))
    assert state.player.statuses["strength"] == 6
    state = _end_turn(_reload(state))
    assert state.player.statuses["strength"] == 9


def test_imbued_after_draw_can_affect_opening_hand_and_top_sees_finished_action():
    state = _enter_combat((
        {"card_id": "imbued", "type": "skill", "target": "self", "cost": 2,
         "effects": {"block_formula": {"formula": "hand_count", "multiplier": 2}},
         "enchantments": ({"keyword": "imbued", "amount": 1},)},
        {"card_id": "normal", "type": "skill", "target": "self", "cost": 0},
    ), relics=("unceasing_top",), flags={"draw_per_turn": 1})
    assert [c.card_id for c in state.combat.hand] == ["normal"]
    assert state.player.block == 2
    events = state.combat.last_events
    assert next(i for i,e in enumerate(events) if e.kind == "cards_drawn") < next(
        i for i,e in enumerate(events) if e.kind == "imbued_card_triggered")


def test_paels_eye_exhausts_burn_before_hand_effect_and_runs_exhaust_callbacks():
    from combat.test_card_play_status_runtime import _source

    state = _state(_source("BURN"), _simple("ordinary"), relics=("paels_eye", "charons_ashes"))
    state = _end_turn(_reload(state))
    assert state.player.hp == 80
    assert {card.card_id for card in state.combat.exhaust_pile} == {"burn", "ordinary"}
    assert state.combat.monsters[0].hp == 94
    assert _extra_turn(state)


def test_helmet_accepts_enemy_giver_and_ignores_negative_strength():
    from sts2sim.engine.transitions import _apply_monster_power
    from sts2sim.mechanics.monsters import MonsterDefinition, MonsterPower

    state = _state(_simple("held"), relics=("ruined_helmet",))
    monster = state.combat.monsters[0]
    definition = MonsterDefinition("fixture", "Fixture", "normal", 100, 100, None, None, (), ())
    metadata = {**state.combat.metadata, "item_relic_ids": state.relics}
    _apply_monster_power(definition, monster, state.player,
        MonsterPower("strength", -2, target="player"), ascension_level=0, item_metadata=metadata)
    assert not metadata.get("relic_counters", {}).get("ruined_helmet")
    _, player, _ = _apply_monster_power(definition, monster, state.player,
        MonsterPower("strength", 3, target="player"), ascension_level=0, item_metadata=metadata)
    assert player.statuses["strength"] == 6
    assert metadata["relic_counters"]["ruined_helmet"] == 1


def test_diadem_counts_auto_post_plays_before_its_regular_turn_end_check():
    # Each exhaust-pile Howl autoplays during AutoPost, after two manual plays.
    from combat.test_card_play_status_runtime import _source

    first, second = _simple("first"), _simple("second")
    state = _play(_play(_state(first, second, relics=("diamond_diadem",)), "first"), "second")
    howl = _source("HOWL_FROM_BEYOND")
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "exhaust_pile": (howl,),
    })})
    state = _end_turn(_reload(state))
    assert not state.player.statuses.get("diamond_diadem")


def test_top_waits_for_parent_card_after_nested_autoplay_before_its_own_draw():
    parent = _simple("parent", effects={"sequence": [{"play_top_card": 1}, {"draw": 1}]})
    state = _state(parent, draw=(_simple("automatic"), _simple("one"), _simple("two")),
                   relics=("unceasing_top",))
    state = _play(_reload(state), "parent")
    assert [c.card_id for c in state.combat.hand] == ["one"]
    assert [c.card_id for c in state.combat.draw_pile] == ["two"]
    assert not state.combat.metadata.get("item_action_depth")


def test_top_resumes_potion_completion_check_after_skipping_modal_card_choice():
    from sts2sim import legal_actions, step

    state = _state(_simple("held"), draw=(_simple("replacement"),), relics=("unceasing_top",))
    state = state.model_copy(update={"character_id": "IRONCLAD", "flags": {**state.flags,
        "cards": ({"id": "anger", "color": "ironclad", "rarity": "Common", "type": "Attack",
                   "target": "enemy", "cost": 0},),
    }, "combat": state.combat.model_copy(update={"hand": ()})})
    state = _potion(_reload(state), "attack_potion")
    assert state.combat.pending_choices
    assert not state.combat.hand
    action = next(a for a in legal_actions(state) if a.type == "proceed")
    state = step(_reload(state), action)
    assert [c.card_id for c in state.combat.hand] == ["replacement"]


def test_mittens_before_hand_draw_follows_demon_form_and_helmet_receiver_hook():
    from combat.test_card_play_status_runtime import _source

    state = _play(_state(_source("DEMON_FORM"), relics=("toasty_mittens", "ruined_helmet")),
                  "demon_form")
    state = _end_turn(_reload(state))
    assert state.player.statuses["strength"] == 5  # DemonForm2 doubles to4, then Mittens1.
    state = _end_turn(_reload(state))
    assert state.player.statuses["strength"] == 8
