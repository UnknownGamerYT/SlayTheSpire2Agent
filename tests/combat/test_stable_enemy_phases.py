"""Independent BeforeDamage/Plating/Illusion outcomes from stable source."""

from __future__ import annotations

import pytest
from combat.test_monster_innate_hooks import _card, _combat, _monster
from combat.test_stable_enemy_lifecycle import _end, _fight, _play

from sts2sim import load_state, serialize
from sts2sim.engine.transitions import _apply_player_thorns_before_attack


@pytest.mark.parametrize("ascension,plating", [(0, 8), (7, 8), (8, 9), (9, 9)])
def test_sewer_clam_starts_with_source_plating_and_skips_first_decrement(ascension, plating):
    state = _fight(("SEWER_CLAM",), ascension=ascension)
    clam = state.combat.monsters[0]
    assert clam.statuses["plating"] == plating
    assert clam.block == plating
    state = _end(state)
    assert state.combat.monsters[0].statuses["plating"] == plating
    assert state.combat.monsters[0].block == plating
    state = _end(load_state(serialize(state)))
    assert state.combat.monsters[0].statuses["plating"] == plating - 1
    assert state.combat.monsters[0].block == plating - 1


def test_mysterious_knight_inherits_flail_and_starts_with_six_strength_and_plating():
    state = _fight(("MYSTERIOUS_KNIGHT",))
    knight = state.combat.monsters[0]
    assert knight.hp == 101
    assert knight.statuses["strength"] == 6
    assert knight.statuses["plating"] == 6
    assert knight.block == 6
    assert knight.move_id == "RAM"


def test_slumber_side_end_wake_keeps_block_already_granted_by_plating():
    state = _fight(("SLUMBERING_BEETLE", "ZAPBOT"))
    state = _end(_end(_end(state)))
    beetle = state.combat.monsters[0]
    assert "slumber" not in beetle.statuses
    assert "plating" not in beetle.statuses
    assert beetle.block == 13  # 15, skip first decrement, then 14 and 13.
    events = state.combat.last_events
    last_attack = max(i for i, e in enumerate(events) if e.kind == "player_damaged")
    plating = next(i for i, e in enumerate(events) if e.kind == "monster_plating_block")
    wake = next(i for i, e in enumerate(events) if e.kind == "monster_awakened")
    assert last_attack < plating < wake


def test_thorns_before_damage_kills_enemy_but_current_hit_completes():
    state = _combat(_monster("FIXTURE", {}, damage=5, hits=2))
    combat = state.combat
    state = state.model_copy(
        update={
            "combat": combat.model_copy(
                update={
                    "monsters": (combat.monsters[0].model_copy(update={"hp": 2}),),
                    "player": combat.player.model_copy(update={"statuses": {"thorns": 3}}),
                }
            )
        }
    )
    state = _end(state)
    events = state.combat.last_events
    assert state.combat.player.hp == 75
    assert sum(e.kind == "player_damaged" for e in events) == 1
    assert next(i for i, e in enumerate(events) if e.kind == "monster_defeated") < next(
        i for i, e in enumerate(events) if e.kind == "player_damaged"
    )


def test_player_thorns_is_unpowered_blockable_damage():
    state = _fight()
    enemy = state.combat.monsters[0].model_copy(update={"block": 4})
    player = state.combat.player.model_copy(update={"statuses": {"thorns": 3}})
    result, events = _apply_player_thorns_before_attack(enemy, player, source_id=enemy.monster_id)
    assert result.hp == enemy.hp
    assert result.block == 1
    assert events[0].metadata["blocked"] == 3


def test_illusion_preserves_negative_buffs_and_temporary_debuff_wrappers():
    state = _combat(
        _monster("PARAFRIGHT", {"ILLUSION": 1}),
        _monster("LEADER", {}),
        deck=(_card("lethal", damage=1000),),
    )
    monsters = list(state.combat.monsters)
    monsters[0] = monsters[0].model_copy(
        update={
            "statuses": {
                **monsters[0].statuses,
                "strength": -3,
                "weak": 2,
                "dark_shackles": 3,
                "poison": 5,
            }
        }
    )
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": tuple(monsters),
                }
            )
        }
    )
    state = _play(state)
    assert state.combat.monsters[0].statuses == {
        "illusion": 1,
        "minion": 1,
        "strength": -3,
        "dark_shackles": 3,
    }


def test_spectral_knight_hex_move_targets_player_after_first_enemy_action():
    state = _fight(("SPECTRAL_KNIGHT",))
    assert "hex" not in state.combat.player.statuses
    assert "hex" not in state.combat.monsters[0].statuses
    state = _end(state)
    assert state.combat.player.statuses["hex"] == 2
    assert "hex" not in state.combat.monsters[0].statuses
