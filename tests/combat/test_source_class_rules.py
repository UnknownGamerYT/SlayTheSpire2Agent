"""Public-action regressions derived from stable v0.107.1 OnPlay methods.

Source provenance and independently recorded expectations are in
tests/fixtures/source_class_rules.json; no decompiled game code is vendored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from combat.test_stored_turn_powers import _end, _play, _source_card
from combat.test_stored_turn_powers import _fight as _base_fight

from sts2sim import legal_actions, load_state, serialize, step


def _fight(*args, **kwargs):
    state = _base_fight(*args, **kwargs)
    player = state.player.model_copy(update={"resources": {"star": 20}})
    return state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )


def _reload(state):
    return load_state(serialize(state))


def _zones(state):
    combat = state.combat
    return combat.hand + combat.draw_pile + combat.discard_pile + combat.exhaust_pile


def _choose_all(state):
    while state.combat.pending_choices:
        actions = legal_actions(state)
        choice = next(a for a in actions if a.type == "choose_card")
        state = step(_reload(state), choice)
    return state


def _with_discard(state, *ids):
    return state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "discard_pile": tuple(_source_card(card_id) for card_id in ids),
                }
            )
        }
    )


@pytest.mark.parametrize(
    "card_id,token,destination,count",
    [
        ("reave", "soul", "draw_pile", 1),
        ("grave_warden", "soul", "draw_pile", 1),
        ("capture_spirit", "soul", "draw_pile", 3),
        ("glimpse_beyond", "soul", "draw_pile", 3),
        ("collision_course", "debris", "hand", 1),
        ("boost_away", "dazed", "discard_pile", 1),
        ("fight_through", "wound", "discard_pile", 2),
        ("gunk_up", "slimed", "discard_pile", 1),
        ("turbo", "void", "discard_pile", 1),
    ],
)
@pytest.mark.parametrize("upgraded", [False, True])
def test_source_generated_cards_have_exact_count_and_executable_identity(
    card_id,
    token,
    destination,
    count,
    upgraded,
):
    state = _fight(card_id, upgraded=upgraded)
    original_draw = tuple(c.instance_id for c in state.combat.draw_pile)
    before_hand = len(state.combat.hand)
    state = _play(_reload(state), card_id)
    expected = count + int(upgraded and card_id in {"capture_spirit", "glimpse_beyond"})
    generated = [c for c in getattr(state.combat, destination) if c.card_id.lower() == token]
    assert len(generated) == expected
    assert all(c.type.value in {"skill", "status"} for c in generated)
    assert not any(c.type.value == "unknown" for c in _zones(state))
    # Generated-card variables are not draws. Existing pile order is preserved.
    assert (
        tuple(c.instance_id for c in state.combat.draw_pile if c.card_id != token) == original_draw
    )
    assert len(state.combat.hand) == before_hand - 1 + (expected if destination == "hand" else 0)
    if token == "soul":
        assert all(c.cost == 0 and c.exhausts for c in generated)
        assert all(c.upgraded == (upgraded and card_id == "reave") for c in generated)
    assert not any(e.kind == "effect_stubbed" for e in state.combat.last_events)


@pytest.mark.parametrize("upgraded", [False, True])
def test_severance_generates_exactly_one_base_soul_per_destination(upgraded):
    state = _play(_fight("severance", upgraded=upgraded), "severance")
    for zone in ("hand", "draw_pile", "discard_pile"):
        souls = [c for c in getattr(state.combat, zone) if c.card_id == "soul"]
        assert len(souls) == 1
        assert souls[0].cost == 0 and not souls[0].upgraded


@pytest.mark.parametrize("upgraded", [False, True])
def test_capture_spirit_loses_hp_once_bypassing_block_and_attack_modifiers(upgraded):
    state = _fight("capture_spirit", upgraded=upgraded, statuses={"strength": 100})
    monster = state.combat.monsters[0].model_copy(
        update={"block": 100, "statuses": {"vulnerable": 2}}
    )
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )
    state = _play(state, "capture_spirit")
    assert state.combat.monsters[0].hp == 1000 - (4 if upgraded else 3)
    assert state.combat.monsters[0].block == 100


@pytest.mark.parametrize(
    "card_id,zone,count,token",
    [
        ("charge", "draw_pile", 2, "minion_dive_bomb"),
        ("begone", "hand", 1, "minion_strike"),
        ("guards", "hand", 3, "minion_sacrifice"),
        ("seance", "draw_pile", 1, "soul"),
    ],
)
@pytest.mark.parametrize("upgraded", [False, True])
def test_transform_cards_replace_selected_cards_without_extra_generation(
    card_id,
    zone,
    count,
    token,
    upgraded,
):
    state = _fight(card_id, upgraded=upgraded)
    if zone == "hand":
        state = state.model_copy(
            update={
                "combat": state.combat.model_copy(
                    update={
                        "hand": state.combat.hand + state.combat.draw_pile[:3],
                        "draw_pile": state.combat.draw_pile[3:],
                    }
                )
            }
        )
    before = len(_zones(state))
    state = _play(state, card_id)
    assert state.combat.pending_choices[0].remaining == count
    assert not any(c.card_id == token for c in _zones(state))
    state = _choose_all(state)
    tokens = [c for c in getattr(state.combat, zone) if c.card_id.lower() == token]
    assert len(tokens) == count
    assert all(c.type.value in {"attack", "skill"} for c in tokens)
    assert all(c.upgraded == (upgraded and card_id != "seance") for c in tokens)
    assert len(_zones(state)) == before


@pytest.mark.parametrize(
    "card_id,block,focus",
    [
        ("hotfix", 0, 2),
        ("focused_strike", 0, 1),
        ("hyperbeam", 0, -3),
    ],
)
def test_focus_effects_apply_to_player_with_correct_sign_and_lifetime(card_id, block, focus):
    state = _play(_fight(card_id), card_id)
    assert sum(v for k, v in state.player.statuses.items() if "focus" in k) == focus
    assert not any("focus" in m.statuses for m in state.combat.monsters)
    state = _end(_reload(state))
    assert sum(v for k, v in state.player.statuses.items() if "focus" in k) == (
        -3 if card_id == "hyperbeam" else 0
    )


@pytest.mark.parametrize("card_id,loss", [("crush_under", 1), ("dying_star", 9)])
def test_temporary_enemy_strength_loss_does_not_reduce_player_strength(card_id, loss):
    state = _play(_fight(card_id), card_id)
    assert not state.player.statuses.get("strength")
    assert state.combat.monsters[0].statuses.get("temporary_strength") == -loss


@pytest.mark.parametrize(
    "card_id",
    ["charge_battery", "delay", "hegemony", "refine_blade", "outmaneuver", "invoke", "relax"],
)
def test_next_turn_energy_is_not_gained_on_play(card_id):
    state = _fight(card_id)
    card = state.combat.hand[0]
    state = _play(state, card_id)
    assert state.player.energy == 20 - (card.cost or 0)
    assert state.player.statuses.get("next_turn_energy", 0) > 0 or card_id == "invoke"


def test_neows_fury_optional_two_card_discard_choice_does_not_draw():
    state = _with_discard(_fight("neows_fury"), "zap", "leap", "skim")
    state = _play(state, "neows_fury")
    assert len(state.combat.hand) == 0
    pending = state.combat.pending_choices[0]
    assert pending.min_choices == 0 and pending.max_choices == 2
    assert set(pending.candidate_ids) == {"zap", "leap", "skim"}
    state = _choose_all(state)
    assert len(state.combat.hand) == 2 and len(state.combat.draw_pile) == 15


def test_dredge_retrieves_three_without_drawing():
    state = _with_discard(_fight("dredge"), "zap", "leap", "skim")
    state = _choose_all(_play(state, "dredge"))
    assert {c.card_id for c in state.combat.hand} == {"zap", "leap", "skim"}
    assert len(state.combat.draw_pile) == 15


def test_all_for_one_retrieves_only_zero_energy_playable_cards_automatically():
    state = _with_discard(_fight("all_for_one"), "claw", "soul", "leap", "wound", "tempest")
    state = _play(state, "all_for_one")
    assert {c.card_id for c in state.combat.hand} == {"claw", "soul"}
    assert not state.combat.pending_choices


def test_anointed_respects_rare_filter_and_hand_capacity():
    state = _fight("anointed", "zap", "leap", "skim", "claw", "reave", "sow", "poke", "fear")
    rare = tuple(_source_card(card_id) for card_id in ("buffer", "reboot", "murder"))
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "draw_pile": rare + state.combat.draw_pile,
                }
            )
        }
    )
    state = _play(state, "anointed")
    assert len(state.combat.hand) == 10
    assert len([c for c in state.combat.hand if c.card_id in {"buffer", "reboot", "murder"}]) == 2
    assert not state.combat.pending_choices


@pytest.mark.parametrize(
    "card_id,expected", [("barrage", 15), ("lunar_blast", 12), ("pull_from_below", 15)]
)
def test_calculated_hit_counts_are_independent_hits(card_id, expected):
    state = _fight(card_id, orbs=("lightning", "frost", "dark"), statuses={"strength": 1})
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "metadata": {
                        "skills_played_this_turn": 3,
                        "source_class_ethereal_plays": 3,
                    }
                }
            )
        }
    )
    state = _play(state, card_id)
    base = {"barrage": 5, "lunar_blast": 4, "pull_from_below": 5}[card_id]
    assert state.combat.monsters[0].hp == 1000 - (base + 1) * 3


def test_sic_em_applies_after_osty_hit_and_is_playable_without_osty():
    state = _fight("sic_em")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "metadata": {"osty": {"alive": True, "hp": 10, "max_hp": 10}},
                }
            )
        }
    )
    state = _play(state, "sic_em")
    assert state.combat.metadata["osty"]["max_hp"] == 10
    assert state.combat.monsters[0].statuses.get("sic_em", 0) == 3
    missing = _fight("sic_em")
    missing = missing.model_copy(
        update={
            "combat": missing.combat.model_copy(
                update={
                    "metadata": {"osty": {"alive": False, "hp": 0, "max_hp": 10}},
                }
            )
        }
    )
    missing = _play(missing, "sic_em")
    assert missing.combat.monsters[0].hp == 1000
    assert missing.combat.monsters[0].statuses.get("sic_em", 0) == 3


def test_snap_retains_a_card_even_without_osty_and_excludes_already_retained_cards():
    state = _fight("snap", "zap")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "metadata": {"osty": {"alive": False, "hp": 0}},
                    "hand": state.combat.hand + state.combat.draw_pile[:1],
                }
            )
        }
    )
    state = _play(state, "snap")
    assert state.combat.pending_choices[0].candidate_ids == ("zap",)
    state = _choose_all(state)
    assert next(c for c in state.combat.hand if c.card_id == "zap").custom["retain"]


def test_pacts_end_is_playable_and_deals_no_damage_below_threshold():
    state = _play(_fight("pacts_end"), "pacts_end")
    assert state.combat.monsters[0].hp == 1000
    assert len(state.combat.cards_played_this_turn) == 1


def test_no_escape_applies_scaled_doom_without_damage():
    state = _fight("no_escape")
    monster = state.combat.monsters[0].model_copy(update={"statuses": {"doom": 29}})
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )
    state = _play(state, "no_escape")
    assert state.combat.monsters[0].hp == 1000
    assert state.combat.monsters[0].statuses["doom"] == 49


def test_reboot_shuffles_hand_into_draw_without_touching_discard_or_exhaust():
    state = _with_discard(_fight("reboot", "zap", "leap"), "murder")
    state = _play(state, "reboot")
    assert len(state.combat.hand) == 4
    assert [c.card_id for c in state.combat.discard_pile] == ["murder"]
    assert len(state.combat.draw_pile) == 13


@pytest.mark.parametrize("card_id,expected", [("adaptive_strike", 0), ("undeath", 0)])
def test_self_clones_preserve_executable_rules_and_source_upgrade(card_id, expected):
    state = _play(_fight(card_id, upgraded=True), card_id)
    copies = [c for c in state.combat.discard_pile if c.card_id == card_id]
    assert len(copies) == 2
    assert all(c.upgraded and c.effects for c in copies)
    if card_id == "adaptive_strike":
        assert sorted(c.cost for c in copies) == [0, 2]


def test_growth_applies_to_all_existing_claws_and_survives_reload():
    state = _fight("claw")
    second = _source_card("claw").model_copy(update={"instance_id": "claw:2"})
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": state.combat.hand + (second,),
                }
            )
        }
    )
    state = _play(state, "claw")
    state = _play(_reload(state), "claw")
    assert state.combat.monsters[0].hp == 992
    assert all(c.effects["sequence"][0]["damage"] == 7 for c in state.combat.discard_pile)


def test_source_fixture_pin_and_rules_are_available():
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures/source_class_rules.json").read_text()
    )
    assert fixture["game_version"] == "0.107.1"
    assert fixture["source_commit"] == "5d874e855ae1d0550d5c30416fadd7dbb0d133b7"
    assert len(fixture["rules"]) >= 60
