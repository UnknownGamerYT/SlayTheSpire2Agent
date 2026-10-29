"""Finite keyword contracts and end-in-hand sequencing from stable source."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from combat.test_combat_engine_cards import _enter_combat
from combat.test_source_modifiers import _card, _end, _fight, _play
from combat.test_stored_turn_powers import _fight as _source_fight
from combat.test_stored_turn_powers import _source_card

from sts2sim import load_state, serialize
from sts2sim.engine import MonsterState, OrbState


@pytest.mark.parametrize("pyramid", [False, True])
def test_ethereal_exhausts_before_retain_and_full_hand_retention(pyramid):
    state = _fight(_card("ethereal", ethereal=True), _card("retained"))
    state = state.model_copy(update={"relics": ("runic_pyramid",) if pyramid else ()})
    result = _end(state)
    assert [c.instance_id for c in result.combat.exhaust_pile] == ["ethereal"]
    assert [c.instance_id for c in result.combat.hand] == ["retained"]


@pytest.mark.parametrize(
    "identity,damage",
    [
        ("burn", 2),
        ("decay", 2),
        ("infection", 3),
        ("toxic", 5),
        ("wither", 3),
    ],
)
def test_end_hand_damage_wrappers_discard_despite_retain_or_pyramid(identity, damage):
    state = _fight(_card(identity))
    state = state.model_copy(update={"relics": ("runic_pyramid",)})
    result = _end(state)
    assert result.player.hp == 100 - damage
    assert not result.combat.hand
    assert [c.card_id for c in result.combat.discard_pile] == [identity]


@pytest.mark.parametrize("identity,loss", [("bad_luck", 13), ("beckon", 6), ("regret", 3)])
def test_unblockable_end_hand_damage_uses_buffer_and_preserves_block(identity, loss):
    state = _fight(_card(identity), _card("ordinary"), _card("ethereal", ethereal=True))
    player = state.player.model_copy(update={"block": 50})
    state = state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )
    result = _end(state)
    assert result.player.hp == 100 - loss
    event = next(
        e
        for e in result.replay_log[-1].events
        if e.source_id == identity and e.kind == "player_damaged"
    )
    assert event.metadata["blocked"] == 0
    player = player.model_copy(update={"statuses": {"buffer": 1}})
    state = state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )
    result = _end(load_state(serialize(state)))
    assert result.player.hp == 100
    assert "buffer" not in result.player.statuses


def test_regret_captures_full_hand_before_ordinary_ethereal_exhausts():
    result = _end(_fight(_card("regret"), _card("ethereal", ethereal=True), _card("ordinary")))
    assert result.player.hp == 97


def test_ordinary_ethereal_exhaust_hooks_resolve_before_burn_damage():
    state = _fight(_source_card("feel_no_pain"), _card("ethereal", ethereal=True), _card("burn"))
    state = _play(state, "feel_no_pain")
    result = _end(state)
    assert result.player.hp == 100
    assert (
        next(
            e
            for e in result.replay_log[-1].events
            if e.source_id == "burn" and e.kind == "player_damaged"
        ).metadata["blocked"]
        == 2
    )


def test_ethereal_burn_resolves_damage_before_its_own_exhaust_hook():
    state = _play(
        _fight(_source_card("feel_no_pain"), _card("burn", ethereal=True)), "feel_no_pain"
    )
    result = _end(state)
    assert result.player.hp == 98
    assert [c.card_id for c in result.combat.exhaust_pile] == ["burn"]


def test_frost_passive_resolves_before_burn_hand_effect():
    state = _fight(_card("burn"))
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "orbs": (OrbState(orb_id="frost"),),
                    "orb_slots": 3,
                }
            )
        }
    )
    assert _end(state).player.hp == 100


@pytest.mark.parametrize("identity,power", [("doubt", "weak"), ("shame", "frail")])
@pytest.mark.parametrize("existing", [False, True])
def test_doubt_and_shame_skip_duration_only_when_creating_a_new_debuff(identity, power, existing):
    result = _end(_fight(_card(identity), statuses={power: 2} if existing else {}))
    assert result.player.statuses[power] == (2 if existing else 1)
    assert not result.combat.hand


def test_debt_loses_at_most_ten_gold_and_leaves_hand():
    state = _fight(_card("debt"))
    player = state.player.model_copy(update={"gold": 7})
    state = state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )
    result = _end(state)
    assert result.player.gold == 0
    assert not result.combat.hand


def test_innate_uses_full_deck_shuffle_then_reverses_collected_order():
    deck = tuple({"card_id": f"card_{i}", "type": "skill", "cost": 0} for i in range(12))
    ordinary = _enter_combat(deck, flags={"draw_per_turn": 0})
    innate = _enter_combat(
        tuple({**card, "keywords_key": ("Innate",)} for card in deck), flags={"draw_per_turn": 0}
    )
    assert ordinary.rng == innate.rng
    assert [c.card_id for c in innate.combat.hand + innate.combat.draw_pile] == list(
        reversed([c.card_id for c in ordinary.combat.draw_pile])
    )


def test_imbued_innate_card_is_excluded_from_extra_opening_hand_draw_count():
    # The Imbued callback autoplays before hand draw. Its Innate keyword must
    # not increase the opening draw budget for the other cards.
    deck = tuple(
        {"card_id": f"card_{i}", "type": "skill", "cost": 0, "keywords_key": ("Innate",)}
        for i in range(3)
    )
    deck += (
        {
            "card_id": "imbued",
            "type": "power",
            "cost": 0,
            "keywords_key": ("Innate",),
            "enchantments": [{"keyword": "Imbued"}],
        },
    )
    result = _enter_combat(deck, flags={"draw_per_turn": 0})
    assert len(result.combat.hand) == 3


def test_each_galvanic_source_resolves_a_separate_unpowered_hit():
    state = _fight(
        _card("power", "power", effects={"noop": True}),
        statuses={"buffer": 1},
        enemy_statuses={"galvanic": 2},
    )
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": state.combat.monsters
                    + (
                        MonsterState(
                            monster_id="second", hp=100, max_hp=100, statuses={"galvanic": 3}
                        ),
                    ),
                }
            )
        }
    )
    result = _play(state, "power")
    assert result.player.hp == 97


def test_calculated_gamble_draws_before_discarded_sly_card_autoplays():
    state = _fight(_source_card("calculated_gamble"),
                   _card("sly", sly=True, effects={"exhaust_hand": {"mode": "all"}}),
                   draw=(_card("drawn"),))
    result = _play(state, "calculated_gamble")
    assert [c.instance_id for c in result.combat.exhaust_pile] == ["drawn", "calculated_gamble"]
    assert not result.combat.hand


def test_sly_autoplay_finds_card_drawn_from_the_just_shuffled_discard_pile():
    state = _fight(_source_card("calculated_gamble"), _card("sly", sly=True, effects={"block": 3}))
    result = _play(state, "calculated_gamble")
    assert result.player.block == 3
    assert not result.combat.hand
    assert result.combat.metadata["skills_played_this_turn"] == 2


def test_consuming_shadow_evokes_rightmost_dark_after_every_orb_passive():
    state = _play(_source_fight("consuming_shadow", orbs=("lightning",)), "consuming_shadow")
    result = _end(state)
    assert result.combat.monsters[0].hp == 985
    assert [orb.orb_id for orb in result.combat.orbs] == ["lightning", "dark"]


def test_source_modifier_audit_enumerates_all_afflictions_and_keyword_values():
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures/source_modifier_audit.json").read_text()
    )
    afflictions = {r["id"] for r in fixture["rows"] if r["category"] == "affliction"}
    keywords = {r["id"] for r in fixture["rows"] if r["category"] == "keyword"}
    assert afflictions == {
        "bound",
        "entangled",
        "galvanized",
        "hexed",
        "ringing",
        "smog",
        "tainted",
    }
    assert keywords == {
        "none",
        "exhaust",
        "ethereal",
        "innate",
        "unplayable",
        "retain",
        "sly",
        "eternal",
    }
    for row in fixture["rows"]:
        assert row["runtime_entrypoints"] and row["regressions"]
        assert all(len(source["sha256"]) == 64 for source in row["sources"])
