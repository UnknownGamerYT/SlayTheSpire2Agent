"""Independent v0.107.1 item contracts exercised through public actions.

Source revision, file hashes and compact source projections are retained in
fixtures/stable_item_contracts.json; the game decompilation is not bundled.
"""

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _state
from combat.test_glam import _end_turn, _play

from sts2sim import legal_actions, serialize, step
from sts2sim.agent_api import action_space


def _potion(state, potion_id, target_id=None):
    state = state.model_copy(update={"potions": state.potions + (potion_id,)})
    action = next(a for a in legal_actions(state) if a.type == "use_potion"
                  and a.payload["potion_id"] == potion_id
                  and (target_id is None or a.target_id == target_id))
    assert any(a["type"] == "use_potion" and a["target_id"] == action.target_id
               for a in action_space(state))
    assert serialize(step(state, action)) == serialize(step(_reload(state), action))
    return step(state, action)


def _enemies(state, *, hp=100, damage=0, statuses=None):
    monster = state.combat.monsters[0]
    monsters = tuple(monster.model_copy(update={
        "monster_id": name, "hp": hp, "max_hp": hp, "intent": "attack",
        "intent_damage": damage, "statuses": dict(statuses or {}),
        "move_id": "ATTACK",
        "metadata": {"damage_per_hit": damage, "source_monster_id": "TEST_ATTACKER"},
    }) for name in ("target", "other"))
    return state.model_copy(update={
        "flags": {**state.flags, "monsters": ({"id": "TEST_ATTACKER", "min_hp": hp,
            "max_hp": hp, "moves": ({"id": "ATTACK", "intent": "Attack",
                                      "damage": {"normal": damage}},)},)},
        "combat": state.combat.model_copy(update={"monsters": monsters}),
    })


def test_glowwater_exhausts_original_hand_then_draws_with_exhaust_hooks_and_reload():
    draw = tuple(_simple(f"draw:{i}") for i in range(12))
    state = _state(_simple("held:a"), _simple("held:b"), draw=draw,
                   relics=("charons_ashes",))
    state = _potion(state, "glowwater_potion")
    assert [c.instance_id for c in state.combat.hand] == [f"draw:{i}" for i in range(10)]
    assert [c.instance_id for c in state.combat.exhaust_pile] == ["held:a", "held:b"]
    assert [c.instance_id for c in state.combat.draw_pile] == ["draw:10", "draw:11"]
    assert state.combat.monsters[0].hp == 94  # Both exhaust events reach the relic.


def test_liquid_memories_discount_expires_on_retained_card_after_turn():
    recovered = _simple("recovered", cost=3, custom={"retain": True})
    state = _state(_simple("unused"))
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "hand": (), "discard_pile": (recovered,),
    })})
    state = _potion(state, "liquid_memories")
    action = next(a for a in legal_actions(state) if a.type == "choose_card")
    state = step(_reload(state), action)
    assert state.combat.hand[0].custom["free_to_play_this_turn"]
    assert not state.combat.hand[0].custom.get("free_to_play_this_combat")
    state = _end_turn(_reload(state))
    assert state.combat.hand[0].cost == 3
    assert not state.combat.hand[0].custom.get("free_to_play_this_turn")


def test_gigantification_survives_turn_end_skips_skills_and_counts_next_attacks():
    attack = _simple("attack", effects={"damage": 4}, card_type="attack", target="enemy",
                     custom={"retain": True})
    skill = _simple("skill", effects={"damage": 2}, target="enemy", custom={"retain": True})
    state = _state(attack, skill)
    state = _potion(_potion(state, "gigantification_potion"), "gigantification_potion")
    state = _end_turn(_reload(state))
    state = _play(state, "skill")
    state = _play(_reload(state), "attack")
    assert state.combat.monsters[0].hp == 86  # 2 skill damage + one triple attack.
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "hand": (attack.model_copy(update={"instance_id": "attack:2"}),),
    })})
    state = _play(_reload(state), "attack:2")
    assert state.combat.monsters[0].hp == 74
    assert not state.combat.metadata.get("next_card_damage_multiplier")


@pytest.mark.parametrize("artifact", [0, 1])
def test_beetle_juice_target_artifact_and_four_enemy_sides(artifact):
    state = _enemies(_state(_simple("unused")), damage=10)
    target = state.combat.monsters[0].model_copy(update={"statuses": {"artifact": artifact}})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (target, state.combat.monsters[1]),
    })})
    state = _potion(state, "beetle_juice", "target")
    assert state.combat.monsters[0].statuses.get("shrink", 0) == (0 if artifact else 4)
    assert not state.combat.monsters[1].statuses
    assert state.combat.monsters[0].statuses.get("artifact", 0) == 0
    for remaining in (3, 2, 1, 0):
        before = state.player.hp
        state = _end_turn(_reload(state))
        assert before - state.player.hp == (20 if artifact else 17)
        assert state.combat.monsters[0].statuses.get("shrink", 0) == (0 if artifact else remaining)
        # Keep the independent fixture alive through all five enemy sides.
        player = state.player.model_copy(update={"hp": 80})
        state = state.model_copy(update={"player": player, "combat": state.combat.model_copy(
            update={"player": player},
        )})
    before = state.player.hp
    state = _end_turn(_reload(state))
    assert before - state.player.hp == 20


def test_beetle_juice_stacks_duration_and_multiplies_weak_before_rounding():
    state = _enemies(_state(_simple("unused")), damage=14, statuses={"weak": 9})
    state = _potion(_potion(state, "beetle_juice", "target"), "beetle_juice", "target")
    assert state.combat.monsters[0].statuses["shrink"] == 8
    before = state.player.hp
    state = _end_turn(_reload(state))
    assert before - state.player.hp == 17  # floor(14*.7*.75)=7, other floor(14*.75)=10.
    assert state.combat.monsters[0].statuses["shrink"] == 7


@pytest.mark.parametrize("artifact", [0, 1])
def test_powdered_demise_targets_one_enemy_stacks_and_waits_for_enemy_side_end(artifact):
    state = _enemies(_state(_simple("unused")), damage=3)
    target = state.combat.monsters[0].model_copy(update={
        "statuses": {"artifact": artifact, "vulnerable": 2},
    })
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (target, state.combat.monsters[1]),
    })})
    state = _potion(_potion(state, "powdered_demise", "target"), "powdered_demise", "target")
    assert [m.hp for m in state.combat.monsters] == [100, 100]
    assert state.combat.monsters[0].statuses["demise"] == (9 if artifact else 18)
    for expected in (100 - (9 if artifact else 18), 100 - (18 if artifact else 36)):
        state = _end_turn(_reload(state))
        assert [m.hp for m in state.combat.monsters] == [expected, 100]
        events = state.combat.last_events
        last_attack = max(i for i, e in enumerate(events) if e.kind == "player_damaged")
        demise = next(i for i, e in enumerate(events) if e.source_id == "demise")
        assert demise > last_attack


@pytest.mark.parametrize("defense,expected", [("intangible", 1), ("slippery", 1),
                                             ("hard_to_kill", 2)])
def test_powdered_demise_is_unblockable_unpowered_damage_with_hp_caps(defense, expected):
    state = _enemies(_state(_simple("unused")), statuses={defense: 2})
    state = _potion(state, "powdered_demise", "target")
    state = _end_turn(_reload(state))
    assert state.combat.monsters[0].hp == 100 - expected
    assert state.combat.monsters[1].hp == 100

