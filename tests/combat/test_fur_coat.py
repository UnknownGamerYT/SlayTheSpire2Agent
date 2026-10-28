from __future__ import annotations

import json

import pytest
from combat.test_fishing_rod import _run

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import (
    AncientOptionState,
    AncientState,
    MapEdgeState,
    MapNodeState,
    MapState,
    RewardState,
    RoomKind,
    RunPhase,
)
from sts2sim.engine.rng import RunRng, capture_random_state


def _map(*, normal_count: int = 12, act: int = 3) -> MapState:
    start = MapNodeState(node_id="start", act=act, floor=0, lane=0, kind=RoomKind.START)
    nodes = [start]
    nodes.extend(
        MapNodeState(node_id=f"fight:{i}", act=act, floor=1, lane=i, kind=RoomKind.MONSTER)
        for i in range(normal_count)
    )
    nodes.extend(
        MapNodeState(node_id=kind.value, act=act, floor=1, lane=20 + i, kind=kind)
        for i, kind in enumerate([RoomKind.ELITE, RoomKind.BOSS, RoomKind.EVENT, RoomKind.REST])
    )
    return MapState(
        act=act,
        nodes=tuple(nodes),
        edges=tuple(MapEdgeState(from_id="start", to_id=node.node_id) for node in nodes[1:]),
        current_node_id="start",
        completed_node_ids=("start",),
    )


def _reward_state(*, normal_count: int = 12):
    return _run(relics=()).model_copy(
        update={
            "act": 3,
            "map": _map(normal_count=normal_count),
            "phase": RunPhase.REWARD,
            "reward": RewardState(reward_id="fur-coat", relic_id="FUR_COAT"),
        }
    )


def _pickup(state):
    return step(state, next(a for a in legal_actions(state) if a.type == "take_reward_relic"))


def _enter(state, node_id: str):
    state = state.model_copy(update={"phase": RunPhase.MAP, "reward": None})
    return step(
        state,
        next(a for a in legal_actions(state) if a.type == "choose_node" and a.target_id == node_id),
    )


def test_fur_coat_pickup_marks_seven_unique_normal_or_elite_nodes_with_replayable_rng() -> None:
    before = _reward_state()
    first = _pickup(before)
    second = _pickup(load_state(json.loads(json.dumps(serialize(before)))))

    marks = first.flags["fur_coat_marked_nodes"]["3"]
    assert len(marks) == len(set(marks)) == 7
    assert all(first.map.node_by_id[node_id].kind in {RoomKind.MONSTER, RoomKind.ELITE}
               for node_id in marks)
    assert serialize(first) == serialize(second)
    assert first.rng != before.rng
    assert len([e for e in first.replay_log[-1].events if e.kind == "fur_coat_room_marked"]) == 7
    assert before.flags.get("fur_coat_marked_nodes") is None


def test_fur_coat_ancient_pickup_uses_the_shared_rng_and_does_not_skip_marking() -> None:
    state = _reward_state().model_copy(
        update={
            "phase": RunPhase.ANCIENT,
            "reward": None,
            "ancient": AncientState(
                act=3,
                ancient_id="nonupeipe",
                options=(
                    AncientOptionState(
                        option_id="fur",
                        name="Fur Coat",
                        kind="positive_relic",
                        relic_id="FUR_COAT",
                    ),
                ),
            ),
        }
    )
    action = next(a for a in legal_actions(state) if a.type == "choose_ancient")
    after = step(state, action)

    assert len(after.flags["fur_coat_marked_nodes"]["3"]) == 7
    assert after.rng != state.rng
    assert serialize(after) == serialize(step(load_state(serialize(state)), action))


def test_stable_fur_coat_uses_independent_relic_rng_and_includes_elites() -> None:
    state = _reward_state(normal_count=1).model_copy(update={
        "rng": capture_random_state(RunRng("FURCOAT")),
    })
    after = _pickup(state)
    assert set(after.flags["fur_coat_marked_nodes"]["3"]) == {"fight:0", "elite"}
    assert after.rng == state.rng
    elite = _enter(load_state(serialize(after)), "elite")
    assert all(monster.hp == 1 < monster.max_hp for monster in elite.combat.monsters)


def test_fur_coat_includes_elite_completed_and_past_points_and_handles_small_maps() -> None:
    state = _reward_state(normal_count=3)
    nodes = state.map.nodes + (
        MapNodeState(node_id="past", act=3, floor=0, lane=1, kind=RoomKind.MONSTER),
    )
    state = state.model_copy(
        update={
            "map": state.map.model_copy(
                update={"nodes": nodes, "completed_node_ids": ("start", "fight:0")}
            )
        }
    )
    state = _pickup(state)
    assert set(state.flags["fur_coat_marked_nodes"]["3"]) == {
        "fight:0", "fight:1", "fight:2", "elite", "past",
    }


def test_fur_coat_subsequent_pickup_only_adds_unmarked_rooms() -> None:
    state = _pickup(_reward_state(normal_count=12))
    first = set(state.flags["fur_coat_marked_nodes"]["3"])
    state = state.model_copy(
        update={"reward": RewardState(reward_id="second-fur", relic_id="FUR_COAT")}
    )
    state = _pickup(state)
    marks = set(state.flags["fur_coat_marked_nodes"]["3"])
    assert len(marks) == 13
    assert first < marks
    new_marks = {
        e.target_id for e in state.replay_log[-1].events if e.kind == "fur_coat_room_marked"
    }
    assert len(new_marks) == 6
    assert not first & new_marks


def test_fur_coat_marks_survive_reload_and_apply_to_placeholder_and_source_enemies() -> None:
    state = _pickup(_reward_state())
    state = load_state(json.loads(json.dumps(serialize(state))))
    node_id = state.flags["fur_coat_marked_nodes"]["3"][0]
    first = _enter(state, node_id)
    assert all(monster.hp == 1 < monster.max_hp for monster in first.combat.monsters)
    assert first.combat.metadata["fur_coat_marked_room"] == node_id
    assert any(e.kind == "fur_coat_monster_hp_set" for e in first.replay_log[-1].events)

    state = state.model_copy(
        update={
            "flags": {
                **state.flags,
                "combat_monster_ids": ("TEST_FIGHTER",),
                "monsters": (
                    {
                        "id": "TEST_FIGHTER",
                        "name": "Test Fighter",
                        "type": "Normal",
                        "min_hp": 30,
                        "max_hp": 30,
                        "moves": ({"id": "WAIT", "intent": "Buff"},),
                    },
                ),
            }
        }
    )
    source_fight = _enter(state, node_id)
    assert source_fight.combat.monsters[0].hp == 1
    assert source_fight.combat.monsters[0].max_hp == 30
    assert source_fight.combat.monsters[0].metadata["fur_coat_original_max_hp"] == 30


@pytest.mark.parametrize("node_id", ["boss", "event"])
def test_fur_coat_excludes_boss_and_unknown_points(node_id: str) -> None:
    state = _pickup(_reward_state())
    # Even an invalid imported mark cannot override the node eligibility guard.
    state.flags["fur_coat_marked_nodes"]["3"] += (node_id,)
    state = _enter(state, node_id)
    if state.combat is not None:
        assert all(monster.hp > 1 for monster in state.combat.monsters)


def test_fur_coat_unmarked_rooms_and_same_node_id_in_other_acts_are_unaffected() -> None:
    state = _pickup(_reward_state())
    marks = state.flags["fur_coat_marked_nodes"]["3"]
    unmarked = next(
        node.node_id
        for node in state.map.nodes
        if node.kind == RoomKind.MONSTER and node.node_id not in marks
    )
    assert all(monster.hp > 1 for monster in _enter(state, unmarked).combat.monsters)
    other_act = state.model_copy(update={"act": 2, "map": _map(act=2)})
    assert all(monster.hp > 1 for monster in _enter(other_act, marks[0]).combat.monsters)


def _summoning_fight(*, marked: bool):
    state = _pickup(_reward_state(normal_count=1))
    flags = {
        **state.flags,
        "combat_monster_ids": ("FOGMOG",),
        "monsters": (
            {
                "id": "FOGMOG",
                "name": "Fogmog",
                "type": "Normal",
                "min_hp": 100,
                "max_hp": 100,
                "moves": ({"id": "ILLUSION", "intent": "Summon"},),
                "attack_pattern": {"initial_move": "ILLUSION"},
            },
            {
                "id": "EYE_WITH_TEETH",
                "name": "Eye With Teeth",
                "type": "Normal",
                "min_hp": 16,
                "max_hp": 16,
                "moves": ({"id": "GUARD", "intent": "Defend", "block": 15},),
            },
        ),
    }
    if not marked:
        flags.pop("fur_coat_marked_nodes")
    return _enter(state.model_copy(update={"flags": flags}), "fight:0")


@pytest.mark.parametrize("marked,expected_hp", [(True, 1), (False, 16)])
def test_fur_coat_applies_to_enemies_summoned_during_the_marked_fight(
    marked: bool,
    expected_hp: int,
) -> None:
    state = _summoning_fight(marked=marked)
    state = load_state(json.loads(json.dumps(serialize(state))))
    action = next(a for a in legal_actions(state) if a.type == "end_turn")
    after = step(state, action)
    summoned = next(m for m in after.combat.monsters if m.metadata.get("summoned"))
    assert summoned.hp == expected_hp
    assert summoned.max_hp == 16
    assert serialize(after) == serialize(step(load_state(serialize(state)), action))


def test_fur_coat_does_not_reapply_to_same_creature_reviving_in_a_marked_room() -> None:
    state = _pickup(_reward_state(normal_count=1))
    state = state.model_copy(
        update={
            "flags": {
                **state.flags,
                "combat_monster_ids": ("TEST_SUBJECT",),
                "monsters": (
                    {
                        "id": "TEST_SUBJECT",
                        "name": "Test Subject",
                        "type": "Normal",
                        "min_hp": 100,
                        "max_hp": 100,
                        "moves": ({"id": "BITE", "intent": "Attack"},),
                    },
                ),
            }
        }
    )
    state = _enter(state, "fight:0")
    state = step(
        state,
        next(
            a
            for a in legal_actions(state)
            if a.type == "play_card" and a.card_instance_id == "finisher:1"
        ),
    )
    monster = state.combat.monsters[0]
    assert state.phase == RunPhase.COMBAT
    assert monster.metadata["test_subject_respawns"] == 1
    assert monster.hp == monster.max_hp > 1
