"""Public-action fixtures independently derived from stable C# card models.

Razhuo/sts2-rl-agent@5d874e855ae1d0550d5c30416fadd7dbb0d133b7;
GAME_VERSIONS.md: stable 0.107.1, build 23811903. Each test names the C#
card model whose CanonicalVars/OnUpgrade/OnPlay supplies its expectations.
Input JSON rows are public Codex data, not an implementation oracle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _state
from combat.test_glam import _end_turn, _play

from sts2sim import legal_actions, step
from sts2sim.engine.transitions import _card_from_spec

ROWS = {
    c["id"]: c
    for c in json.loads((Path(__file__).parents[1] / "fixtures/source_core_cards.json").read_text())
}


def source(name, up=False):
    return _card_from_spec({**ROWS[name.upper()], "instance_id": name.lower(), "upgraded": up}, 0)


def choose(state, name):
    return step(state, next(a for a in legal_actions(state) if a.card_instance_id == name))


def attack(name="hit", damage=1):
    return _simple(name, effects={"damage": damage}, card_type="attack", target="enemy")


@pytest.mark.parametrize("up", [False, True])
@pytest.mark.parametrize(
    "name,count",
    [("blade_dance", 3), ("blade_of_ink", 2), ("cloak_and_dagger", 1), ("up_my_sleeve", 3)],
)
def test_shiv_generators_create_real_shivs_and_never_draw(name, count, up):
    state = _play(_state(source(name, up), draw=(attack("draw"),)), name)
    assert len(state.combat.draw_pile) == 1
    assert len(state.combat.hand) == count + up
    assert all(
        c.card_id == "shiv" and c.type.value == "attack" and c.exhausts and c.cost == 0
        for c in state.combat.hand
    )
    if name == "up_my_sleeve":
        assert state.combat.discard_pile[-1].cost == 1
    if name != "blade_of_ink":
        state = _play(_reload(state), state.combat.hand[0].instance_id)
        assert state.combat.monsters[0].hp == 96


@pytest.mark.parametrize("up", [False, True])
def test_accuracy_does_not_create_shiv(up):
    state = _play(_state(source("accuracy", up)), "accuracy")
    assert not state.combat.hand
    assert state.player.statuses["accuracy"] == (6 if up else 4)


@pytest.mark.parametrize("up", [False, True])
def test_fiend_fire_snapshots_entire_hand_and_hits_per_exhaust_with_strength(up):
    state = _play(
        _state(source("fiend_fire", up), _simple("one"), _simple("two"), statuses={"strength": 2}),
        "fiend_fire",
    )
    assert not state.combat.hand
    assert len(state.combat.exhaust_pile) == 3
    assert state.combat.monsters[0].hp == 100 - 2 * ((10 if up else 7) + 2)


@pytest.mark.parametrize("up", [False, True])
def test_fiend_fire_empty_hand_deals_no_damage(up):
    state = _play(_state(source("fiend_fire", up), statuses={"strength": 9}), "fiend_fire")
    assert state.combat.monsters[0].hp == 100


@pytest.mark.parametrize("up", [False, True])
def test_second_wind_exhausts_only_non_attacks_and_gains_separate_block(up):
    state = _play(
        _state(
            source("second_wind", up),
            attack(),
            _simple("one"),
            _simple("two"),
            statuses={"dexterity": 2},
        ),
        "second_wind",
    )
    assert [c.card_id for c in state.combat.hand] == ["hit"]
    assert [c.card_id for c in state.combat.exhaust_pile] == ["one", "two"]
    assert state.player.block == 2 * ((7 if up else 5) + 2)


@pytest.mark.parametrize("up", [False, True])
def test_finisher_counts_finished_attacks_and_flechettes_counts_skills(up):
    state = _state(
        attack(),
        source("finisher", up),
        source("flechettes", up),
        _simple("a"),
        _simple("b"),
        statuses={"strength": 1},
    )
    state = _play(_play(_reload(state), "hit"), "finisher")
    assert state.combat.monsters[0].hp == 100 - 2 - (9 if up else 7)
    state = _play(_reload(state), "flechettes")
    assert state.combat.monsters[0].hp == 100 - 2 - (9 if up else 7) - 2 * (8 if up else 6)


def test_finisher_before_any_attack_has_zero_hits():
    state = _play(_state(source("finisher"), statuses={"strength": 5}), "finisher")
    assert state.combat.monsters[0].hp == 100


@pytest.mark.parametrize("up", [False, True])
def test_perfected_strike_counts_itself_and_strikes_in_every_combat_pile(up):
    state = _state(
        source("perfected_strike", up), source("strike_ironclad"), draw=(source("strike_silent"),)
    )
    state = _play(_reload(state), "perfected_strike")
    assert state.combat.monsters[0].hp == 100 - 6 - 3 * (3 if up else 2)


@pytest.mark.parametrize("up", [False, True])
def test_expertise_draws_until_target_hand_size(up):
    draw = tuple(_simple(f"draw{i}") for i in range(9))
    state = _play(_state(source("expertise", up), _simple("held"), draw=draw), "expertise")
    assert len(state.combat.hand) == (7 if up else 6)


@pytest.mark.parametrize("up", [False, True])
def test_calculated_gamble_discards_and_draws_same_count(up):
    state = _play(
        _state(
            source("calculated_gamble", up),
            _simple("a"),
            _simple("b"),
            draw=(_simple("d1"), _simple("d2"), _simple("d3")),
        ),
        "calculated_gamble",
    )
    assert [c.card_id for c in state.combat.hand] == ["d1", "d2"]
    assert state.combat.draw_pile[0].card_id == "d3"


@pytest.mark.parametrize("up", [False, True])
def test_storm_of_steel_shiv_count_equals_discarded_snapshot(up):
    state = _play(
        _state(source("storm_of_steel", up), _simple("a"), _simple("b")), "storm_of_steel"
    )
    assert len(state.combat.hand) == 2
    state = _play(_reload(state), state.combat.hand[0].instance_id)
    assert state.combat.monsters[0].hp == 100 - (6 if up else 4)


@pytest.mark.parametrize("up", [False, True])
def test_bouncing_flask_repeats_separate_poison_applications(up):
    state = _play(_state(source("bouncing_flask", up)), "bouncing_flask")
    assert state.combat.monsters[0].statuses["poison"] == (12 if up else 9)


@pytest.mark.parametrize("name,hits", [("ricochet", 4), ("sword_boomerang", 3)])
@pytest.mark.parametrize("up", [False, True])
def test_random_attack_repeat_upgrades_increase_hits(name, hits, up):
    state = _play(_state(source(name, up), statuses={"strength": 1}), name)
    assert state.combat.monsters[0].hp == 100 - (hits + up) * 4


@pytest.mark.parametrize("up", [False, True])
def test_rampage_damage_growth_belongs_to_physical_card(up):
    state = _play(_state(source("rampage", up)), "rampage")
    card = state.combat.discard_pile[-1]
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"hand": (card,), "discard_pile": ()})}
    )
    state = _play(_reload(state), "rampage")
    assert state.combat.monsters[0].hp == 100 - 9 - 9 - (9 if up else 5)


@pytest.mark.parametrize("up", [False, True])
def test_memento_mori_observes_discard_choice_and_ignores_play_destination(up):
    state = _state(
        source("prepared"), source("memento_mori", up), _simple("trash"), draw=(_simple("draw"),)
    )
    state = choose(_play(state, "prepared"), "trash")
    state = _play(_reload(state), "memento_mori")
    assert state.combat.monsters[0].hp == 100 - (11 if up else 9) - (5 if up else 4)


def test_murder_counts_actual_draws_across_turns_and_reload():
    state = _state(
        source("murder"), _simple("draw", effects={"draw": 2}), draw=(_simple("a"), _simple("b"))
    )
    state = _play(_play(state, "draw"), "murder")
    assert state.combat.monsters[0].hp == 97


@pytest.mark.parametrize("up", [False, True])
def test_fisticuffs_block_counts_blocked_damage_and_overkill(up):
    state = _state(source("fisticuffs", up))
    mon = state.combat.monsters[0].model_copy(update={"hp": 3, "block": 2})
    other = mon.model_copy(update={"monster_id": "other", "hp": 100, "block": 0})
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (mon, other)})}
    )
    state = _play(state, "fisticuffs")
    assert state.player.block == (9 if up else 7)


@pytest.mark.parametrize("up", [False, True])
def test_echoing_slash_adds_one_repeat_per_enemy_killed(up):
    state = _state(source("echoing_slash", up))
    first = state.combat.monsters[0].model_copy(update={"hp": 1})
    second = first.model_copy(update={"monster_id": "second", "hp": 1})
    third = first.model_copy(update={"monster_id": "third", "hp": 100})
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (first, second, third)})}
    )
    state = _play(state, "echoing_slash")
    assert state.combat.monsters[2].hp == 100 - 3 * (13 if up else 10)


@pytest.mark.parametrize("up", [False, True])
def test_havoc_plays_top_card_for_free_then_exhausts(up):
    state = _play(_state(source("havoc", up), draw=(attack("top", 7), attack("next", 4))), "havoc")
    assert state.combat.monsters[0].hp == 93
    assert state.combat.draw_pile[0].card_id == "next"
    assert state.combat.exhaust_pile[-1].card_id == "top"


@pytest.mark.parametrize("up", [False, True])
def test_armaments_upgrades_selected_or_whole_hand(up):
    state = _play(
        _state(source("armaments", up), source("strike_ironclad"), source("strike_silent")),
        "armaments",
    )
    if not up:
        state = choose(_reload(state), "strike_ironclad")
    assert state.player.block == 5
    assert state.combat.hand[0].upgraded
    assert state.combat.hand[1].upgraded == up


@pytest.mark.parametrize("up", [False, True])
def test_temporary_strength_and_dexterity_expire(up):
    state = _play(
        _play(_state(source("anticipate", up), source("setup_strike", up)), "anticipate"),
        "setup_strike",
    )
    assert state.player.statuses["temporary_dexterity"] == (3 if up else 2)
    assert state.player.statuses["temporary_strength"] == (3 if up else 2)
    state = _end_turn(_reload(state))
    assert not state.player.statuses.get("temporary_strength")
    assert not state.player.statuses.get("temporary_dexterity")


@pytest.mark.parametrize("name,next_type", [("pounce", "skill"), ("unrelenting", "attack")])
def test_next_card_cost_charge_is_consumed_once_and_survives_reload(name, next_type):
    next_card = _simple("next", cost=4, card_type=next_type)
    state = _play(_state(source(name), next_card), name)
    before = state.player.energy
    state = _play(_reload(state), "next")
    assert state.player.energy == before
    assert not state.player.statuses.get(f"source_core_free_{next_type}")


@pytest.mark.parametrize("up", [False, True])
def test_primal_force_transforms_each_attack_without_adding_an_extra_card(up):
    state = _play(
        _state(source("primal_force", up), attack("a"), attack("b"), _simple("skill")),
        "primal_force",
    )
    assert [c.card_id for c in state.combat.hand] == ["giant_rock", "giant_rock", "skill"]
    assert all(c.cost == 1 for c in state.combat.hand[:2])
    state = _play(_reload(state), state.combat.hand[0].instance_id)
    assert state.combat.monsters[0].hp == 100 - (20 if up else 16)


@pytest.mark.parametrize("up", [False, True])
def test_burning_pact_chooses_before_draw_and_still_draws_on_empty_hand(up):
    draws = tuple(_simple(f"d{i}") for i in range(4))
    state = _play(_state(source("burning_pact", up), _simple("old"), draw=draws), "burning_pact")
    assert [c.card_id for c in state.combat.hand] == ["old"]
    state = choose(_reload(state), "old")
    assert len(state.combat.hand) == (3 if up else 2)
    assert state.combat.exhaust_pile[0].card_id == "old"
    state = _play(_state(source("burning_pact", up), draw=draws), "burning_pact")
    assert len(state.combat.hand) == (3 if up else 2)


@pytest.mark.parametrize("up", [False, True])
def test_brand_strength_follows_exhaust_and_still_applies_with_empty_hand(up):
    state = _play(_state(source("brand", up), _simple("old")), "brand")
    assert state.player.hp == 79
    assert not state.player.statuses.get("strength")
    state = choose(_reload(state), "old")
    assert state.player.statuses["strength"] == (2 if up else 1)
    state = _play(_state(source("brand", up)), "brand")
    assert state.player.statuses["strength"] == (2 if up else 1)


@pytest.mark.parametrize("up", [False, True])
def test_hidden_daggers_discards_then_creates_two_shivs_upgrade_changes_damage(up):
    state = _play(
        _state(source("hidden_daggers", up), _simple("a"), _simple("b")), "hidden_daggers"
    )
    assert len(state.combat.hand) == 2
    state = choose(choose(_reload(state), "a"), "b")
    assert len(state.combat.hand) == 2
    state = _play(state, state.combat.hand[0].instance_id)
    assert state.combat.monsters[0].hp == (94 if up else 96)


@pytest.mark.parametrize("up", [False, True])
def test_knife_trap_plays_existing_exhaust_shivs_and_upgrades_them(up):
    from sts2sim.mechanics.source_core_rules import shiv_spec

    shiv = _card_from_spec({**shiv_spec(), "instance_id": "old_shiv"}, 0)
    state = _state(source("knife_trap", up))
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"exhaust_pile": (shiv,)})}
    )
    state = _play(_reload(state), "knife_trap")
    assert state.combat.monsters[0].hp == (94 if up else 96)
    assert [c.instance_id for c in state.combat.exhaust_pile] == ["old_shiv"]
    assert state.combat.exhaust_pile[0].upgraded == up


@pytest.mark.parametrize("up", [False, True])
def test_gold_axe_counts_finished_cards_and_excludes_itself(up):
    state = _play(_state(source("gold_axe", up)), "gold_axe")
    assert state.combat.monsters[0].hp == 100
    state = _play(_play(_state(_simple("a"), source("gold_axe", up)), "a"), "gold_axe")
    assert state.combat.monsters[0].hp == 99


@pytest.mark.parametrize("up", [False, True])
def test_dodge_and_roll_next_turn_stores_actual_modified_block(up):
    state = _play(_state(source("dodge_and_roll", up), statuses={"dexterity": 2}), "dodge_and_roll")
    assert state.player.block == (8 if up else 6)
    state = _end_turn(_reload(state))
    assert state.player.block == (8 if up else 6)


def test_fan_of_knives_changes_generated_and_existing_shivs_to_all_enemy_damage():
    state = _play(_state(source("fan_of_knives")), "fan_of_knives")
    first = state.combat.monsters[0]
    second = first.model_copy(update={"monster_id": "other"})
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (first, second)})}
    )
    state = _play(_reload(state), state.combat.hand[0].instance_id)
    assert [m.hp for m in state.combat.monsters] == [96, 96]


def test_juggling_copies_only_third_attack_and_counts_plays_before_power():
    attacks = tuple(attack(f"a{i}") for i in range(6))
    state = _state(*attacks, source("juggling"))
    state = _play(_play(_play(state, "a0"), "a1"), "juggling")
    state = _play(state, "a2")
    assert len([c for c in state.combat.hand if c.card_id == "a2"]) == 1
    for name in ("a3", "a4", "a5"):
        state = _play(state, name)
    assert [c.card_id for c in state.combat.hand] == ["a2"]


def test_stoke_uses_exhaust_snapshot_and_upgrades_generated_cards():
    state = _state(source("stoke", True), _simple("a"), _simple("b"))
    state = state.model_copy(
        update={"character_id": "IRONCLAD", "flags": {"card_pool": (ROWS["ANGER"],)}}
    )
    state = _play(_reload(state), "stoke")
    assert len(state.combat.hand) == 2
    assert all(c.card_id == "anger" and c.upgraded for c in state.combat.hand)
    assert [c.card_id for c in state.combat.exhaust_pile] == ["a", "b"]


def test_beat_down_plays_selected_discard_attacks_without_drawing_skills():
    state = _state(source("beat_down"), draw=(_simple("untouched"),))
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={"discard_pile": (attack("a", 2), attack("b", 3), _simple("skill"))}
            )
        }
    )
    state = _play(state, "beat_down")
    assert state.combat.monsters[0].hp == 95
    assert [c.card_id for c in state.combat.draw_pile] == ["untouched"]


def test_catastrophe_plays_cards_from_draw_without_adding_them_to_hand():
    state = _play(
        _state(source("catastrophe"), draw=(attack("a", 2), attack("b", 3))), "catastrophe"
    )
    assert state.combat.monsters[0].hp == 95
    assert not state.combat.hand and not state.combat.draw_pile


def test_pillage_stops_on_first_non_attack_and_leaves_following_cards_undrawn():
    state = _play(
        _state(source("pillage"), draw=(attack("a"), _simple("stop"), attack("after"))), "pillage"
    )
    assert [c.card_id for c in state.combat.hand] == ["a", "stop"]
    assert [c.card_id for c in state.combat.draw_pile] == ["after"]


def test_anointed_moves_only_rares_without_counting_as_a_draw():
    state = _state(source("anointed"), draw=(source("murder"), source("anger")))
    state = _play(state, "anointed")
    assert [c.card_id for c in state.combat.hand] == ["murder"]
    assert [c.card_id for c in state.combat.draw_pile] == ["anger"]
    assert not state.combat.metadata.get("source_core_draws")


def test_hand_trick_grants_single_turn_sly_to_one_skill():
    state = _play(_state(source("hand_trick"), _simple("chosen")), "hand_trick")
    state = choose(_reload(state), "chosen")
    assert state.combat.hand[0].custom["sly"]
    state = _end_turn(state)
    assert not any(c.custom.get("source_core_temporary_sly") for c in state.combat.discard_pile)
