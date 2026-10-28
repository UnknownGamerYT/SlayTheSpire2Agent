"""Pinned stable power/Block boundaries and actual relic RNG call chains.

TEST/CombatTargets indices [1,1,2,1,0,2] were compiled independently in C
from the stable seed hash, SplitMix64 and xoshiro256** primitive contract.
Source file hashes and finite reviewed scopes are retained in the audit ledger.
"""
from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _source, _state
from combat.test_glam import _end_turn, _play

from sts2sim.engine.rng import STS2_STABLE_DERIVATION, RunRng, capture_random_state


def _targets(state, count=3):
    prototype = state.combat.monsters[0]
    monsters = tuple(prototype.model_copy(update={
        "monster_id": f"enemy:{index}", "hp": 100, "max_hp": 100,
        "intent": "unknown", "intent_damage": 0, "statuses": {"vulnerable": 3},
    }) for index in range(count))
    return state.model_copy(update={
        "character_id": "IRONCLAD",
        "rng": capture_random_state(RunRng("TEST", derivation=STS2_STABLE_DERIVATION)),
        "combat": state.combat.model_copy(update={"monsters": monsters}),
    })


def test_player_plating_blocks_burn_before_end_hand_damage_and_decays_at_next_start():
    state = _state(_source("BURN"), statuses={"plated_armor": 3})
    result = _end_turn(_reload(state))
    assert result.player.hp == 80
    assert result.player.statuses["plated_armor"] == 2
    events = result.replay_log[-1].events
    plating_index = next(i for i, e in enumerate(events) if e.kind == "player_block"
                         and e.metadata.get("status") == "plated_armor")
    burn_index = next(i for i, e in enumerate(events) if e.kind == "player_damaged"
                      and e.source_id == "burn")
    assert plating_index < burn_index
    assert events[plating_index].amount == 3
    assert events[burn_index].metadata["blocked"] == 2


def test_early_plating_is_unpowered_but_respects_shadowmeld_and_block_cap():
    state = _state(_source("BURN"), statuses={
        "plated_armor": 1, "shadowmeld": 2, "dexterity": 99, "frail": 2,
    })
    result = _end_turn(_reload(state))
    block = next(e for e in result.replay_log[-1].events if e.kind == "player_block"
                 and e.metadata.get("status") == "plated_armor")
    assert block.amount == 4
    assert result.player.hp == 80
    assert "shadowmeld" not in result.player.statuses


def test_parrying_shield_sees_early_plating_and_uses_one_combat_targets_draw():
    state = _state(_simple("wait"), statuses={
        "plated_armor": 3, "dexterity": 99, "frail": 2, "strength": 99, "weak": 2,
    }, relics=("parrying_shield",))
    player = state.player.model_copy(update={"block": 7})
    state = state.model_copy(update={"player": player, "combat": state.combat.model_copy(update={
        "player": player,
    })})
    result = _end_turn(_reload(_targets(state)))
    assert [m.hp for m in result.combat.monsters] == [100, 94, 100]
    assert result.rng.streams["combat_targets"].draws == 1
    damage = next(e for e in result.replay_log[-1].events if e.kind == "monster_damaged"
                  and e.metadata.get("relic_id") == "parrying_shield")
    assert damage.amount == 6  # Unpowered: Strength, Weak and Vulnerable do not modify it.


def test_tingsha_resolves_each_discard_with_its_own_combat_targets_draw():
    fillers = tuple(_simple(f"filler:{i}") for i in range(6))
    state = _state(_source("CALCULATED_GAMBLE"), *fillers,
                   statuses={"strength": 99, "weak": 2}, relics=("tingsha",))
    result = _play(_reload(_targets(state)), "calculated_gamble")
    assert [m.hp for m in result.combat.monsters] == [97, 91, 94]
    assert result.rng.streams["combat_targets"].draws == 6
    events = [e for e in result.replay_log[-1].events if e.kind == "monster_damaged"
              and e.metadata.get("relic_id") == "tingsha"]
    assert [e.target_id for e in events] == [
        "enemy:1", "enemy:1", "enemy:2", "enemy:1", "enemy:0", "enemy:2",
    ]
    assert [e.amount for e in events] == [3] * 6


def test_tingsha_singleton_target_still_consumes_one_draw_per_discard():
    state = _state(_source("CALCULATED_GAMBLE"), _simple("one"), _simple("two"),
                   relics=("tingsha",))
    result = _play(_reload(_targets(state, 1)), "calculated_gamble")
    assert result.combat.monsters[0].hp == 94
    assert result.rng.streams["combat_targets"].draws == 2


@pytest.mark.parametrize("stacks,strength,expected", [(2, 3, 39), (4, 0, 60)])
def test_hang_multiplies_once_after_additive_strength_and_before_vulnerable(
    stacks, strength, expected,
):
    state = _state(_source("HANG"), statuses={"strength": strength})
    target = state.combat.monsters[0].model_copy(update={
        "statuses": {"hang": stacks, "vulnerable": 2},
    })
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (target,),
    })})
    result = _play(_reload(state), "hang")
    assert result.combat.monsters[0].hp == 100 - expected
    assert result.combat.monsters[0].statuses["hang"] == stacks * 2


@pytest.mark.parametrize("relics,expected", [((), 15), (("vambrace",), 25)])
def test_entrench_is_unpowered_move_block_but_keeps_unmovable_and_relics(relics, expected):
    state = _state(_source("ENTRENCH"), statuses={
        "dexterity": 10, "frail": 2, "no_card_block_gain": 1, "unmovable": 1,
    }, relics=relics)
    player = state.player.model_copy(update={"block": 5})
    state = state.model_copy(update={"player": player, "combat": state.combat.model_copy(update={
        "player": player,
    })})
    result = _play(_reload(state), "entrench")
    assert result.player.block == expected
    gained = next(e for e in result.replay_log[-1].events if e.kind == "player_block")
    assert gained.metadata["item_block_multipliers"] == (
        ("unmovable", "vambrace") if relics else ("unmovable",)
    )
