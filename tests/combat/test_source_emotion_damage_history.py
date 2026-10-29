"""Emotion Chip queries source damage history, not just positive HP loss."""

from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _source
from combat.test_glam import _end_turn, _play
from combat.test_named_monster_scripts import _combat
from combat.test_source_orb_lifecycle import _orb_state

from sts2sim.engine import OrbState


@pytest.mark.parametrize(
    "damage,block,statuses,relics,expected_hp,history_count",
    [
        (0, 0, {}, (), 80, 1),
        (0, 1, {}, (), 80, 0),
        (4, 0, {"buffer": 1}, (), 80, 1),
        (4, 2, {"buffer": 1}, (), 80, 0),
        (4, 4, {}, (), 80, 0),
        (1, 0, {}, ("tungsten_rod",), 80, 1),
        (4, 0, {}, (), 76, 1),
    ],
)
def test_actual_damage_result_predicate_controls_next_turn_emotion_chip(
    damage,
    block,
    statuses,
    relics,
    expected_hp,
    history_count,
):
    state = _combat(
        "HISTORY_ATTACK",
        moves=(
            {
                "id": "ATTACK",
                "intent": "Attack",
                "damage": {"normal": damage, "hit_count": 1},
            },
        ),
    )
    player = state.player.model_copy(update={"block": block, "statuses": statuses})
    state = state.model_copy(
        update={
            "player": player,
            "relics": ("emotion_chip", *relics),
            "combat": state.combat.model_copy(
                update={
                    "player": player,
                    "orb_slots": 3,
                    "orbs": (OrbState(orb_id="lightning"),),
                }
            ),
        }
    )
    result = _end_turn(_reload(state))
    assert result.player.hp == expected_hp
    assert result.combat.monsters[0].hp == 97 - 3 * history_count
    assert result.combat.metadata["damage_not_fully_blocked_count_previous_turn"] == history_count
    assert result.combat.metadata["damage_not_fully_blocked_count_this_turn"] == 0
    assert result.combat.metadata["lost_hp_previous_turn"] is bool(history_count)
    event = next(e for e in result.replay_log[-1].events if e.kind == "player_damaged")
    assert event.metadata["was_fully_blocked"] is (not bool(history_count))
    assert _end_turn(_reload(state)) == result


def test_unblockable_buffer_prevented_self_damage_still_qualifies_with_existing_block():
    state = _orb_state(
        _source("BLOODLETTING"),
        orbs=("lightning",),
        relics=("emotion_chip",),
        statuses={"buffer": 1},
    )
    player = state.player.model_copy(update={"block": 50})
    state = state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )
    state = _play(_reload(state), "bloodletting")
    assert state.player.hp == 80 and state.player.block == 50
    assert state.combat.metadata.get("hp_lost_this_turn", 0) == 0
    assert state.combat.metadata["damage_not_fully_blocked_count_this_turn"] == 1
    state = _end_turn(_reload(state))
    assert state.combat.monsters[0].hp == 94
    assert state.combat.metadata["hp_lost_previous_turn"] == 0
    assert state.combat.metadata["damage_not_fully_blocked_count_previous_turn"] == 1


def test_new_explicit_damage_history_overrides_legacy_hp_loss_fallback():
    state = _orb_state(orbs=("lightning",), relics=("emotion_chip",))
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "metadata": {
                        **state.combat.metadata,
                        "hp_lost_this_turn": 3,
                        "damage_not_fully_blocked_count_this_turn": 0,
                    },
                }
            )
        }
    )
    assert _end_turn(_reload(state)).combat.monsters[0].hp == 97
    metadata = dict(state.combat.metadata)
    metadata.pop("damage_not_fully_blocked_count_this_turn")
    old_save = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "metadata": metadata,
                }
            )
        }
    )
    assert _end_turn(_reload(old_save)).combat.monsters[0].hp == 94
