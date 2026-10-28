"""Source-derived arithmetic at public action boundaries, including save/reload.

Stable source5d874e8: Hook.ModifyDamageInternal accumulates decimal additive
and multiplicative hooks; Creature.DamageBlockInternal/LoseHpInternal convert
to integers only when committing block/HP. TaintedPower is additive before Weak.
"""

from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _state
from combat.test_glam import _end_turn, _play

from sts2sim.mechanics.combat_math import ModifierSnapshot, modified_attack_damage


@pytest.mark.parametrize(
    ("statuses", "target_statuses", "expected"),
    [
        ({"weak": 1}, {"vulnerable": 1}, 10),
        ({"weak": 1, "shrink": 1}, {"vulnerable": 1}, 7),
        ({"weak": 1, "debilitate": 1}, {"vulnerable": 1}, 6),
        ({"weak": 1}, {"vulnerable": 1, "debilitate": 1}, 13),
    ],
)
def test_attack_combines_decimal_modifiers_before_rounding(statuses, target_statuses, expected):
    attack = _simple("attack", effects={"damage": 9}, card_type="attack", target="enemy")
    state = _state(attack, statuses=statuses)
    monster = state.combat.monsters[0].model_copy(update={"statuses": target_statuses})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (monster,),
    })})
    result = _play(_reload(state), "attack")
    assert result.combat.monsters[0].hp == 100 - expected
    _reload(result)  # Fractional arithmetic must not leak into serializable state.


@pytest.mark.parametrize("tainted,expected", [(0, 11), (3, 14)])
def test_enemy_attack_combines_weak_vulnerable_and_tainted_before_rounding(tainted, expected):
    state = _state(_simple("wait"), statuses={"vulnerable": 2, "tainted": tainted})
    monster = state.combat.monsters[0].model_copy(update={
        "intent": "attack", "intent_damage": 10, "statuses": {"weak": 2},
    })
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (monster,),
    })})
    result = _end_turn(_reload(state))
    assert result.player.hp == 80 - expected
    _reload(result)


def test_pure_math_uses_same_final_floor_as_game_damage_commands():
    assert modified_attack_damage(
        9, attacker=ModifierSnapshot(weak=True), defender=ModifierSnapshot(vulnerable=True),
    ) == 10


def test_intangible_caps_damage_before_consuming_enemy_block():
    attack = _simple("attack", effects={"damage": 9}, card_type="attack", target="enemy")
    state = _state(attack)
    monster = state.combat.monsters[0].model_copy(update={
        "block": 5, "statuses": {"intangible": 1},
    })
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (monster,),
    })})
    result = _play(_reload(state), "attack")
    assert result.combat.monsters[0].hp == 100
    assert result.combat.monsters[0].block == 4


def test_damage_result_preserves_actual_hp_loss_blocked_and_overkill():
    attack = _simple("attack", effects={"damage": 9}, card_type="attack", target="enemy")
    state = _state(attack)
    monster = state.combat.monsters[0].model_copy(update={"hp": 3, "block": 2})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (monster,),
    })})
    result = _play(_reload(state), "attack")
    event = next(e for e in result.combat.last_events if e.kind == "monster_damaged")
    assert event.amount == 3
    assert event.metadata["blocked"] == 2
    assert event.metadata["overkill_damage"] == 4
    assert event.metadata["total_damage_including_overkill"] == 9


@pytest.mark.parametrize("effects", [{"damage": 0}, {"damage_formula": "player_block"}])
def test_zero_base_attack_still_receives_strength(effects):
    attack = _simple("attack", effects=effects, card_type="attack", target="enemy")
    result = _play(_state(attack, statuses={"strength": 3}), "attack")
    assert result.combat.monsters[0].hp == 97


def test_signed_calculated_damage_adds_strength_before_clamping():
    attack = _simple("attack", effects={"damage_formula": {
        "formula": "hand_count", "base": 13, "multiplier": -2,
    }}, card_type="attack", target="enemy")
    others = tuple(_simple(f"other:{index}") for index in range(8))
    result = _play(_state(attack, *others, statuses={"strength": 5}), "attack")
    assert result.combat.monsters[0].hp == 98  # (13 - 2*8) + 5 = 2.


@pytest.mark.parametrize("card_id,status,amount,expected", [
    ("sweep", "weak", 3, [73, 91]),
    ("sovereign_blade", "conqueror", 1, [82, 91]),
    ("hang", "hang", 2, [82, 91]),
])
def test_target_specific_damage_multipliers_do_not_spread_across_aoe(
    card_id, status, amount, expected,
):
    attack = _simple(card_id, effects={"all_damage": 9}, card_type="attack", target="all_enemies")
    state = _state(attack, statuses={"tracking": amount} if status == "weak" else {})
    first = state.combat.monsters[0].model_copy(update={"statuses": {status: 2}})
    second = first.model_copy(update={"monster_id": "other", "statuses": {}})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (first, second),
    })})
    result = _play(_reload(state), card_id)
    assert [monster.hp for monster in result.combat.monsters] == expected


def test_reflection_returns_blocked_damage_without_attack_modifiers():
    state = _state(_simple("wait"), statuses={"reflect": 1, "strength": 99})
    player = state.player.model_copy(update={"block": 4})
    monster = state.combat.monsters[0].model_copy(update={
        "intent": "attack", "intent_damage": 8, "statuses": {"weak": 2, "vulnerable": 2},
    })
    state = state.model_copy(update={"player": player, "combat": state.combat.model_copy(update={
        "player": player, "monsters": (monster,),
    })})
    result = _end_turn(_reload(state))
    assert result.player.hp == 78  # 8*.75=6; four Block absorbed.
    assert result.combat.monsters[0].hp == 96
    assert result.player.statuses.get("reflect", 0) == 0


@pytest.mark.parametrize("base,dexterity,expected", [(9, 0, 13), (0, 3, 4)])
def test_block_combines_frail_and_doubling_before_final_floor(base, dexterity, expected):
    block = _simple("block", effects={"block": base})
    result = _play(_state(block, statuses={
        "dexterity": dexterity, "frail": 1, "double_block_gain_this_turn": 1,
    }), "block")
    assert result.player.block == expected
    _reload(result)


def test_defend_bonus_is_additive_before_frail_and_doubling():
    defend = _simple("defend", effects={"block": 5}, tags=("defend",))
    result = _play(_state(defend, statuses={
        "dexterity": 1, "defend_block_bonus": 3, "frail": 1,
        "double_block_gain_this_turn": 1,
    }), "defend")
    assert result.player.block == 13  # (5+1+3)*.75*2 ->13.


def test_block_cap_limits_stored_block_and_preserves_requested_hook_amount():
    block = _simple("block", effects={"block": 5})
    state = _state(block)
    player = state.player.model_copy(update={"block": 999_999_998})
    state = state.model_copy(update={"player": player, "combat": state.combat.model_copy(update={
        "player": player,
    })})
    result = _play(state, "block")
    assert result.player.block == 999_999_999
    event = next(event for event in result.combat.last_events if event.kind == "player_block")
    assert event.amount == 5
