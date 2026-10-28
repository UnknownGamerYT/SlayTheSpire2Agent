"""Source powered attack owner/pet predicates, enchantments and damage results."""
from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _state
from combat.test_glam import _play

from sts2sim.engine import CardEnchantment


def _pet_state(*cards, statuses=None, pet_statuses=None, relics=(), target_statuses=None, block=0):
    state = _state(*cards, statuses=statuses, relics=relics)
    monster = state.combat.monsters[0].model_copy(update={
        "statuses": target_statuses or {}, "block": block,
    })
    return state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (monster,), "metadata": {**state.combat.metadata, "osty": {
            "alive": True, "hp": 20, "max_hp": 20, "statuses": pet_statuses or {},
        }},
    })})


def _attack(identity="pet_attack", damage=10, *, target="enemy", **kwargs):
    return _simple(identity, card_type="attack", target=target,
                   effects={"osty_action": {"action": "damage", "amount": damage,
                                            "target": target}}, **kwargs)


def test_pet_damage_uses_pet_strength_and_weak_but_owner_double_damage_and_tracking():
    state = _pet_state(_attack(), statuses={
        "strength": 99, "weak": 1, "source_core_double_damage": 1, "tracking": 2,
    }, pet_statuses={"strength": 3, "weak": 1}, target_statuses={"weak": 1})
    result = _play(_reload(state), "pet_attack")
    assert result.combat.monsters[0].hp == 61  # (10+3)*.75*2*2 =39.


def test_pet_source_enchantment_and_owner_relic_additions_apply_before_multipliers():
    card = _attack(tags=("strike", "minion"), upgraded=True,
                   enchantments=(CardEnchantment(keyword="sharp", amount=2),))
    state = _pet_state(card, relics=("strike_dummy", "miniature_cannon", "mystic_lighter",
                                    "vitruvian_minion"))
    result = _play(_reload(state), "pet_attack")
    assert result.combat.monsters[0].hp == 46  # (10+2+3+3+9)*2.


def test_pet_tracking_is_evaluated_per_receiver_and_combined_before_floor():
    card = _attack(damage=9, target="all_enemies")
    state = _pet_state(card, statuses={"tracking": 2}, pet_statuses={"weak": 1},
                       target_statuses={"weak": 1})
    first = state.combat.monsters[0]
    second = first.model_copy(update={"monster_id": "other", "statuses": {}})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (first, second),
    })})
    result = _play(_reload(state), "pet_attack")
    assert [monster.hp for monster in result.combat.monsters] == [87, 94]


def test_pet_tainted_addition_recovers_signed_strength_before_clamping():
    state = _pet_state(_attack(damage=5), pet_statuses={"strength": -7},
                       target_statuses={"tainted": 3, "vulnerable": 1})
    assert _play(_reload(state), "pet_attack").combat.monsters[0].hp == 99


def test_owner_lethality_applies_once_to_first_physical_pet_attack_not_replay():
    card = _attack(enchantments=(CardEnchantment(keyword="replay", amount=1),))
    state = _pet_state(card, statuses={"lethality": 100})
    result = _play(_reload(state), "pet_attack")
    assert result.combat.monsters[0].hp == 70


@pytest.mark.parametrize("artifact,doom", [(0, 10), (1, 0)])
def test_pet_rider_source_predicates_and_total_damage_include_block(artifact, doom):
    state = _pet_state(_attack(), statuses={
        "monarchs_gaze": 3, "envenom": 2, "reaper_form": 1,
    }, target_statuses={"artifact": artifact}, block=8)
    result = _play(_reload(state), "pet_attack")
    monster = result.combat.monsters[0]
    assert monster.hp == 98
    assert monster.statuses.get("doom", 0) == doom
    assert monster.statuses.get("temporary_strength", 0) == 0
    assert monster.statuses.get("poison", 0) == 0
    assert monster.statuses.get("artifact", 0) == 0


def test_pet_uses_dealer_vulnerable_scaling_without_owner_paper_phrog_or_cruelty():
    state = _pet_state(_attack(), statuses={"cruelty": 3}, relics=("paper_phrog",),
                       target_statuses={"vulnerable": 1})
    assert _play(_reload(state), "pet_attack").combat.monsters[0].hp == 85


def test_direct_pet_command_boundaries_include_all_hits_but_not_next_command():
    from sts2sim.engine.transitions import _apply_osty_action_effect

    card = _attack(damage=4)
    state = _pet_state(card, statuses={"vigor": 50}, pet_statuses={"vigor": 2})
    combat = state.combat.model_copy(update={"metadata": {**state.combat.metadata,
        "next_card_damage_multiplier": ({"card_type": "attack", "multiplier": 3},),
    }})
    combat = combat.model_copy(update={"metadata": {**combat.metadata,
        "osty_attacks_this_turn": 1,
        "osty": {**combat.metadata["osty"], "attacks_this_turn": 1},
    }})
    payload = {"osty_action": [
        {"action": "damage", "amount": 4, "extra_hits": "attacks_this_turn"},
        {"action": "damage", "amount": 4, "extra_hits": "attacks_this_turn"},
    ]}
    combat, _, _ = _apply_osty_action_effect(combat, state.rng, card, payload)
    assert combat.monsters[0].hp == 52  # (4+2)*3*2, then 4*3.
    assert combat.player.statuses["vigor"] == 50
    assert not combat.metadata["osty"]["statuses"].get("vigor")
    assert not combat.metadata.get("next_card_damage_multiplier")
    assert not combat.metadata.get("source_attack_commands")


def test_missing_pet_does_not_consume_queued_command_or_owner_vigor():
    from sts2sim.engine.transitions import _apply_osty_action_effect

    card = _attack()
    state = _pet_state(card, statuses={"vigor": 8})
    combat = state.combat.model_copy(update={"metadata": {**state.combat.metadata,
        "osty": {"alive": False, "hp": 0, "max_hp": 20},
        "next_card_damage_multiplier": ({"card_type": "attack", "multiplier": 3},),
    }})
    combat, _, _ = _apply_osty_action_effect(combat, state.rng, card, card.effects)
    assert combat.monsters[0].hp == 100
    assert combat.player.statuses["vigor"] == 8
    assert combat.metadata.get("next_card_damage_multiplier")
    assert not combat.metadata.get("source_attack_commands")
