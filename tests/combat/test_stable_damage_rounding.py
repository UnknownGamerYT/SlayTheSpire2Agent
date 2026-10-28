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
