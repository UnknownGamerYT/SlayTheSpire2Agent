from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _source, _state
from combat.test_glam import _end_turn, _play


def _slots(state):
    return state.model_copy(update={"combat": state.combat.model_copy(update={"orb_slots": 3})})


@pytest.mark.parametrize(
    "card_id,hp_loss,gold_loss,status",
    [
        ("BAD_LUCK", 13, 0, None),
        ("DEBT", 0, 10, None),
        ("DECAY", 2, 0, None),
        ("DOUBT", 0, 0, "weak"),
        ("SHAME", 0, 0, "frail"),
    ],
)
def test_held_source_curse_predicates_resolve_at_end_turn(card_id, hp_loss, gold_loss, status):
    state = _state(_source(card_id))
    player = state.player.model_copy(update={"gold": 50})
    state = state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )
    state = _end_turn(_reload(state))
    assert state.player.hp == 80 - hp_loss
    assert state.player.gold == 50 - gold_loss
    if status:
        assert any(
            event.kind == "status_applied" and event.metadata.get("status") == status
            for event in state.replay_log[-1].events
        )


def test_source_caltrops_retaliates_against_enemy_attack():
    state = _state(_source("CALTROPS"))
    monster = state.combat.monsters[0].model_copy(update={"intent": "attack", "intent_damage": 4})
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )
    state = _end_turn(_play(state, "caltrops"))
    assert state.player.hp == 76
    assert state.combat.monsters[0].hp == 97


def test_source_calamity_creates_only_after_attack():
    attack = _simple("attack", card_type="attack", target="enemy", effects={"damage": 1})
    skill = _simple("skill")
    state = _state(_source("CALAMITY"), attack, skill)
    state = state.model_copy(
        update={
            "flags": {
                **state.flags,
                "card_pool": (
                    {
                        "id": "TOKEN",
                        "type": "Attack",
                        "target": "Enemy",
                        "cost": 0,
                        "color": "test",
                    },
                ),
            }
        }
    )
    state = _play(_play(state, "calamity"), "skill")
    assert not any(card.card_id == "token" for card in state.combat.hand)
    state = _play(_reload(state), "attack")
    assert sum(card.card_id == "token" for card in state.combat.hand) == 1


def test_source_hello_world_waits_for_turn_start_and_uses_common_pool():
    state = _state(_source("HELLO_WORLD"))
    state = state.model_copy(
        update={
            "flags": {
                **state.flags,
                "card_pool": (
                    {
                        "id": "TOKEN",
                        "type": "Skill",
                        "cost": 0,
                        "color": "test",
                        "rarity": "Common",
                    },
                ),
            }
        }
    )
    state = _play(state, "hello_world")
    assert not state.combat.hand
    state = _end_turn(_reload(state))
    assert [card.card_id for card in state.combat.hand] == ["token"]


@pytest.mark.parametrize("upgraded,stars", [(False, 2), (True, 3)])
def test_source_genesis_gains_stars_at_turn_start(upgraded, stars):
    state = _play(_state(_source("GENESIS", upgraded=upgraded)), "genesis")
    assert state.player.resources["star"] == 20
    state = _end_turn(_reload(state))
    assert state.player.resources["star"] == 20 + stars


def test_source_dark_embrace_draws_after_another_card_exhausts():
    exhaust = _simple("exhaust", exhausts=True)
    state = _play(
        _state(_source("DARK_EMBRACE"), exhaust, draw=(_simple("drawn"),)), "dark_embrace"
    )
    assert [card.instance_id for card in state.combat.draw_pile] == ["drawn"]
    state = _play(_reload(state), "exhaust")
    assert [card.instance_id for card in state.combat.hand] == ["drawn"]


def test_source_consuming_shadow_channels_then_evokes_rightmost_after_orb_passives():
    state = _play(_slots(_state(_source("CONSUMING_SHADOW"))), "consuming_shadow")
    assert len(state.combat.orbs) == 2
    assert all(orb.orb_id == "dark" for orb in state.combat.orbs)
    state = _end_turn(_reload(state))
    assert len(state.combat.orbs) == 1
    assert state.combat.monsters[0].hp == 88


def test_source_lightning_rod_channels_at_exactly_next_two_turn_starts():
    state = _play(_slots(_state(_source("LIGHTNING_ROD"))), "lightning_rod")
    assert not state.combat.orbs
    for count in (1, 2, 2):
        state = _end_turn(_reload(state))
        assert len(state.combat.orbs) == count


def test_source_monologue_grants_temporary_strength_on_later_cards_only_this_turn():
    state = _play(_state(_source("MONOLOGUE"), _simple("later")), "monologue")
    state = _play(_reload(state), "later")
    assert state.player.statuses["temporary_strength"] == 1
    state = _end_turn(state)
    assert not state.player.statuses.get("temporary_strength")
    assert not state.combat.metadata.get("combat_triggers")


def test_source_right_hand_hand_returns_from_discard_after_costly_card():
    right = _source("RIGHT_HAND_HAND")
    costly = _simple("costly", cost=2)
    state = _state(costly)
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"discard_pile": (right,)})}
    )
    state = _play(_reload(state), "costly")
    assert [card.card_id for card in state.combat.hand] == ["right_hand_hand"]


def test_source_spinner_channels_glass_at_turn_start():
    state = _play(_slots(_state(_source("SPINNER"))), "spinner")
    assert not state.combat.orbs
    state = _end_turn(_reload(state))
    assert [orb.orb_id for orb in state.combat.orbs] == ["glass"]


def test_source_spirit_of_ash_requires_played_ethereal_keyword():
    ethereal = _simple("ethereal", tags=("ethereal",))
    state = _play(_state(_source("SPIRIT_OF_ASH"), _simple("ordinary"), ethereal), "spirit_of_ash")
    state = _play(state, "ordinary")
    assert state.player.block == 0
    state = _play(_reload(state), "ethereal")
    assert state.player.block == 4


def test_source_storm_requires_another_power_play():
    state = _play(
        _slots(_state(_source("STORM"), _simple("skill"), _simple("power", card_type="power"))),
        "storm",
    )
    state = _play(state, "skill")
    assert not state.combat.orbs
    state = _play(_reload(state), "power")
    assert [orb.orb_id for orb in state.combat.orbs] == ["lightning"]
