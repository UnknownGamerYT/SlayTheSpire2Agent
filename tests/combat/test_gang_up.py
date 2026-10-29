from __future__ import annotations

import pytest
from combat.test_card_status_runtime import _attack, _end, _fight, _play, _skill

from sts2sim import load_state, serialize
from sts2sim.mechanics.card_effects import card_effect_plan

GANG_UP = {
    "id": "GANG_UP",
    "type": "Attack",
    "target": "AnyEnemy",
    "cost": 1,
    "damage": 5,
    "description": "Deal 5 damage.\nDeals 5 additional damage for each time another player "
    "has attacked the enemy this turn.",
    "upgrade_description": "Deal 5 damage.\nDeals 7 additional damage for each time another player "
    "has attacked the enemy this turn.",
    "upgrade": {"extradamage": "+2"},
    "vars": {"CalculatedDamage": 5, "CalculationBase": 5, "ExtraDamage": 5},
    "multiplayer_only": True,
}


@pytest.mark.parametrize("upgraded,bonus", [(False, 5), (True, 7)])
def test_gang_up_source_formula_uses_other_player_attacks_not_multihit_damage(upgraded, bonus):
    source = {**GANG_UP, "upgraded": upgraded}
    plan = card_effect_plan(source)
    assert plan.steps == (
        {
            "damage_formula": {
                "formula": "other_player_attacks_this_turn",
                "multiplier": bonus,
                "base": 5,
            }
        },
    )
    state = _fight(
        source,
        _attack("own"),
        _attack("friend", effects=[{"damage": 1}] * 3, custom={"owner_id": "friend"}),
        _attack("friend_two", damage=2, custom={"owner_id": "friend"}),
        _attack("other_target", custom={"owner_id": "second_friend"}),
    )
    state = _play(_play(_play(state, "own"), "friend"), "friend_two")
    state = _play(state, "other_target", target="b")
    state = load_state(serialize(state))
    hp = state.combat.monsters[0].hp
    state = _play(state, "gang_up")
    assert hp - state.combat.monsters[0].hp == 5 + bonus * 2
    assert state.combat.metadata["player_attacks_by_target_this_turn"]["a"]["friend"] == 2


def test_fully_blocked_attack_counts_but_other_player_skill_damage_does_not():
    state = _fight(
        GANG_UP,
        _attack("friend", custom={"owner_id": "friend"}),
        _skill("skill", {"damage": 2}, target="Enemy", custom={"owner_id": "friend"}),
        block=10,
    )
    state = _play(_play(state, "friend"), "skill")
    hp = state.combat.monsters[0].hp
    state = _play(state, "gang_up")
    assert hp - state.combat.monsters[0].hp == 10


def test_other_player_attack_counts_expire_at_turn_boundary():
    state = _fight(
        {**GANG_UP, "tags": ["retain"]},
        _attack("friend", custom={"owner_id": "friend"}),
    )
    state = _end(_play(state, "friend"))
    assert "player_attacks_by_target_this_turn" not in state.combat.metadata
    hp = state.combat.monsters[0].hp
    state = _play(state, "gang_up")
    assert hp - state.combat.monsters[0].hp == 5


def test_gang_up_viewpoint_excludes_the_cards_owner_and_does_not_guess_osty_as_player():
    state = _fight(
        {**GANG_UP, "custom": {"owner_id": "friend"}},
        _attack("friend", custom={"owner_id": "friend"}),
        _attack("own"),
        _attack("osty", effects={"osty_action": {"action": "damage", "amount": 7}}),
    )
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "metadata": {
                        **state.combat.metadata,
                        "osty": {"hp": 20, "max_hp": 20, "alive": True},
                    }
                }
            )
        }
    )
    state = _play(_play(_play(state, "friend"), "own"), "osty")
    hp = state.combat.monsters[0].hp
    state = _play(state, "gang_up")
    assert hp - state.combat.monsters[0].hp == 10  # One other-player attack: local player's.
    assert "osty" not in state.combat.metadata["player_attacks_by_target_this_turn"]["a"]


def test_replayed_attacks_count_as_separate_attacks_after_save_reload():
    state = _fight(GANG_UP, _attack("friend", custom={"owner_id": "friend", "replay": 1}))
    state = _play(state, "friend")
    state = load_state(serialize(state))
    hp = state.combat.monsters[0].hp
    state = _play(state, "gang_up")
    assert hp - state.combat.monsters[0].hp == 15
