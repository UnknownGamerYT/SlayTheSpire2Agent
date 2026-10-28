from __future__ import annotations

import pytest
from combat.test_stored_turn_powers import _end, _fight, _play, _source_card

from sts2sim import load_state, serialize
from sts2sim.engine import CardInstance, CardType


def _resources(state, amount: int):
    player = state.player.model_copy(update={"resources": {"star": amount}})
    return state.model_copy(update={"player": player, "combat": state.combat.model_copy(
        update={"player": player},
    )})


def _hand(state, *cards):
    return state.model_copy(update={"combat": state.combat.model_copy(update={"hand": cards})})


@pytest.mark.parametrize("upgraded,per_star", [(False, 2), (True, 3)])
def test_child_of_the_stars_counts_actual_star_cost_once_after_reload(upgraded, per_star):
    state = _resources(_fight("child_of_the_stars", "falling_star", "cloak_of_stars",
                              upgraded=upgraded), 3)
    state = _play(state, "child_of_the_stars")
    assert state.player.block == 0
    state = _play(load_state(serialize(state)), "falling_star")
    assert state.player.block == per_star * 2
    assert state.player.resources["star"] == 1
    state = _play(state, "cloak_of_stars")
    cloak = 10 if upgraded else 7
    assert state.player.block == per_star * 3 + cloak
    assert state.player.resources["star"] == 0
    blocks = [event for event in state.combat.last_events
              if event.source_id == "child_of_the_stars" and event.kind == "player_block"]
    assert [event.amount for event in blocks] == [per_star]


def test_child_of_the_stars_does_not_reward_gains_or_over_requested_star_loss() -> None:
    state = _resources(_fight("child_of_the_stars"), 0)
    state = _play(state, "child_of_the_stars")
    gain = CardInstance(
        instance_id="gain", card_id="gain", type="skill", cost=0, target="self",
        effects={"player_resource": {"resource": "star", "amount": 2}},
    )
    drain = gain.model_copy(update={
        "instance_id": "drain", "card_id": "drain",
        "effects": {"player_resource": {"resource": "star", "amount": -9}},
    })
    state = _play(_hand(state, gain, drain), "gain")
    assert state.player.block == 0
    state = _play(state, "drain")
    assert state.player.resources["star"] == 0
    assert state.player.block == 4
    delta = next(event.amount for event in state.combat.last_events
                 if event.kind == "player_resource_changed")
    assert delta == -2


@pytest.mark.parametrize("upgraded,block,retaliation", [(False, 12, 4), (True, 16, 6)])
def test_flame_barrier_retaliates_each_blocked_hit_then_expires(upgraded, block, retaliation):
    state = _fight("flame_barrier", upgraded=upgraded, enemy_damage=4)
    source = dict(state.flags["monsters"][0])
    source["moves"] = ({**source["moves"][0], "damage": {"normal": 4, "hit_count": 3}},)
    state = state.model_copy(update={"flags": {**state.flags, "monsters": (source,)}})
    state = _play(state, "flame_barrier")
    assert state.player.block == block
    assert state.player.statuses["flame_barrier"] == retaliation
    assert state.combat.monsters[0].hp == 1000
    state = _end(load_state(serialize(state)))
    assert state.player.hp == 200  # Fully blocked hits still trigger retaliation.
    assert state.combat.monsters[0].hp == 1000 - retaliation * 3
    assert "flame_barrier" not in state.player.statuses
    state = _end(load_state(serialize(state)))
    assert state.player.hp == 188
    assert state.combat.monsters[0].hp == 1000 - retaliation * 3


def test_flame_barrier_retaliates_to_a_zero_damage_attack() -> None:
    state = _fight("flame_barrier")
    source = dict(state.flags["monsters"][0])
    source["moves"] = ({**source["moves"][0], "intent": "Attack", "damage": {"normal": 0}},)
    state = state.model_copy(update={"flags": {**state.flags, "monsters": (source,)}})
    state = _end(_play(state, "flame_barrier"))
    assert state.player.hp == 200
    assert state.combat.monsters[0].hp == 996


@pytest.mark.parametrize("upgraded,cost", [(False, 2), (True, 1)])
def test_monarchs_gaze_weakens_a_fully_blocked_attack_for_one_enemy_turn(upgraded, cost):
    state = _fight("monarchs_gaze", upgraded=upgraded, enemy_damage=4)
    assert state.combat.hand[0].cost == cost
    state = _play(state, "monarchs_gaze")
    monster = state.combat.monsters[0].model_copy(update={"block": 100})
    attack = CardInstance(instance_id="attack", card_id="attack", type="attack", cost=0,
                          target="enemy", effects={"damage": 6})
    state = _hand(state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (monster,),
    })}), attack)
    state = _play(load_state(serialize(state)), "attack")
    assert state.combat.monsters[0].hp == 1000
    assert state.combat.monsters[0].statuses["temporary_strength"] == -1
    state = _end(state)
    assert state.player.hp == 197
    assert "temporary_strength" not in state.combat.monsters[0].statuses
    state = _end(load_state(serialize(state)))
    assert state.player.hp == 193


def test_monarchs_gaze_applies_once_per_attack_hit_and_artifact_blocks_one_hit() -> None:
    state = _play(_fight("monarchs_gaze"), "monarchs_gaze")
    monster = state.combat.monsters[0].model_copy(update={
        "block": 100, "statuses": {"artifact": 1},
    })
    attack = CardInstance(instance_id="attack", card_id="attack", type="attack", cost=0,
                          target="enemy", effects={"sequence": ({"damage": 6}, {"damage": 6})})
    state = _hand(state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (monster,),
    })}), attack)
    state = _play(state, "attack")
    assert state.combat.monsters[0].statuses["temporary_strength"] == -1
    assert "artifact" not in state.combat.monsters[0].statuses


@pytest.mark.parametrize("upgraded", [False, True])
def test_reaper_form_uses_attack_hp_damage_and_ignores_blocked_hits_and_nonattacks(upgraded):
    state = _fight("reaper_form", upgraded=upgraded)
    assert bool(state.combat.hand[0].custom.get("retain")) is upgraded
    state = _play(state, "reaper_form")
    monster = state.combat.monsters[0].model_copy(update={"block": 8})
    attack = CardInstance(instance_id="attack", card_id="attack", type="attack", cost=0,
                          target="enemy", effects={"sequence": ({"damage": 6}, {"damage": 6})})
    skill = attack.model_copy(update={
        "instance_id": "skill", "card_id": "skill", "type": CardType.SKILL,
        "effects": {"damage": 5},
    })
    state = _hand(state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (monster,),
    })}), attack, skill)
    state = _play(load_state(serialize(state)), "attack")
    assert state.combat.monsters[0].hp == 996
    assert state.combat.monsters[0].statuses["doom"] == 4
    state = _play(state, "skill")
    assert state.combat.monsters[0].hp == 991
    assert state.combat.monsters[0].statuses["doom"] == 4


@pytest.mark.parametrize("regrets,fillers,buffer,loss", [(1, 2, 0, 3), (2, 3, 0, 10), (2, 3, 1, 5)])
def test_regret_loses_hp_per_hand_card_for_each_copy_after_reload(regrets, fillers, buffer, loss):
    state = _fight(statuses={"buffer": buffer} if buffer else {})
    cards = tuple(_source_card("regret").model_copy(update={"instance_id": f"regret:{i}"})
                  for i in range(regrets)) + state.combat.draw_pile[:fillers]
    state = _hand(state, *cards)
    assert state.player.hp == 200
    state = _end(load_state(serialize(state)))
    assert state.player.hp == 200 - loss
    losses = [event.amount for event in state.combat.last_events
              if event.kind == "player_damaged" and event.metadata.get("card_id") == "regret"]
    assert sum(losses) == loss
    assert len(losses) == regrets
