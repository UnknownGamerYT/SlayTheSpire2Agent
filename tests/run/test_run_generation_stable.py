"""Retained stable source examples and public transition regressions."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    Action,
    ActionType,
    MapEdgeState,
    MapNodeState,
    MapState,
    RoomKind,
    RunPhase,
    load_state,
    serialize_state,
)
from sts2sim.engine.rng import RunRng, random_stream
from sts2sim.mechanics.map_generation import generate_standard_map, map_point_counts
from sts2sim.mechanics.run_generation import select_run_act_ids, selected_act_id

FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures/stable_map_source_oracle.json").read_text()
)


class SampleTape(random.Random):
    """Same sample-based bounded/Fisher-Yates calls as the independent fixture adapter."""

    def __init__(self, seed: int):
        super().__init__(seed)
        self.draws = 0

    def random(self) -> float:
        self.draws += 1
        return super().random()

    def _randbelow(self, n: int) -> int:
        return int(self.random() * n)


@pytest.mark.parametrize(
    "case", FIXTURE["cases"], ids=lambda value: value["act_id"] + "-A" + str(value["ascension"])
)
def test_complete_map_matches_independent_source_port(case: dict[str, Any]) -> None:
    rng = SampleTape(case["tape_seed"])
    layout = generate_standard_map(
        rng,
        act=case["act"],
        act_id=case["act_id"],
        ascension_level=case["ascension"],
    )
    nodes = {node.node_id: node for node in layout.nodes}
    actual_nodes = sorted([node.floor, node.lane, node.kind.value] for node in layout.nodes)
    actual_edges = sorted(
        [
            nodes[edge.from_id].floor,
            nodes[edge.from_id].lane,
            nodes[edge.to_id].floor,
            nodes[edge.to_id].lane,
        ]
        for edge in layout.edges
    )
    assert actual_nodes == case["nodes"]
    assert actual_edges == case["edges"]
    assert rng.draws == case["draws"]


def test_per_act_counts_and_ascension_change_elites_without_changing_other_rolls() -> None:
    for act_id in ("OVERGROWTH", "UNDERDOCKS", "HIVE", "GLORY"):
        low = map_point_counts(act_id, RunRng("COUNTS"), ascension_level=0)
        high = map_point_counts(act_id, RunRng("COUNTS"), ascension_level=10)
        assert low == {**high, "elite": 5}
        assert high["elite"] == 8
        assert low["shop"] == 3
        assert low["event"] in (
            range(10, 15) if act_id in {"OVERGROWTH", "UNDERDOCKS"} else range(9, 14)
        )
        assert low["rest"] in (range(5, 7) if act_id == "GLORY" else range(6, 8))


def test_variant_unlock_discovery_and_explicit_selection() -> None:
    assert select_run_act_ids({"revealed_epochs": []}, RunRng("LOCKED"))["1"] == "OVERGROWTH"
    profile = {"revealed_epochs": ["UNDERDOCKS_EPOCH"], "discovered_act_ids": []}
    assert select_run_act_ids(profile, RunRng("NEW"))["1"] == "UNDERDOCKS"
    assert select_run_act_ids({"act_id": "underdocks"}, RunRng("FORCE"))["1"] == "UNDERDOCKS"
    with pytest.raises(ValueError, match="cannot use named act"):
        selected_act_id(2, {"act_ids": {"2": "UNDERDOCKS"}})


def test_all_unlocked_initial_bags_and_shared_ancient_allocation() -> None:
    state = new_run("BAGS", "IRONCLAD", 0)
    shared = state.flags["shared_relic_bag"]
    player = state.flags["player_relic_bag"]
    assert {key: len(value) for key, value in shared.items()} == {
        "Uncommon": 30,
        "Common": 25,
        "Rare": 35,
        "Shop": 25,
        "Ancient": 2,
        "Event": 1,
    }
    assert {key: len(value) for key, value in player.items()} == {
        "Uncommon": 32,
        "Common": 26,
        "Rare": 38,
        "Shop": 26,
    }
    plans = state.flags["act_room_sets"]
    assert plans["1"]["ancient_id"] == "NEOW"
    assert sum(plan["ancient_id"] == "DARV" for plan in plans.values()) <= 1
    assert len(plans["1"]["normal_encounters"]) == 15
    assert len(plans["2"]["normal_encounters"]) == 14
    assert len(plans["3"]["normal_encounters"]) == 13


def _single_next_room(state: Any, kind: RoomKind, *, floor: int = 12) -> Any:
    start = MapNodeState(
        node_id="fixture:start", act=state.act, floor=0, lane=3, kind=RoomKind.START
    )
    target = MapNodeState(node_id="fixture:next", act=state.act, floor=floor, lane=3, kind=kind)
    return state.model_copy(
        update={
            "phase": RunPhase.MAP,
            "ancient": None,
            "map": MapState(
                act=state.act,
                nodes=(start, target),
                edges=(MapEdgeState(from_id=start.node_id, to_id=target.node_id),),
                current_node_id=start.node_id,
                completed_node_ids=(start.node_id,),
            ),
        }
    )


def test_first_actual_hallway_is_weak_even_on_late_map_floor_and_reload_matches() -> None:
    state = _single_next_room(
        new_run(
            "WEAK",
            "IRONCLAD",
            0,
            source_data={
                "act_id": "UNDERDOCKS",
            },
        ),
        RoomKind.MONSTER,
    )
    expected_id = state.flags["act_room_sets"]["1"]["normal_encounters"][0]
    action = next(value for value in legal_actions(state) if value.type == ActionType.CHOOSE_NODE)
    actual = step(state, action)
    restored = step(load_state(serialize_state(state)), action)
    assert serialize_state(actual) == serialize_state(restored)
    assert actual.combat is not None
    assert actual.combat.monsters[0].metadata["encounter_id"] == expected_id
    assert expected_id.endswith("_WEAK")
    assert actual.flags["act_room_sets"]["1"]["normal_encounters_visited"] == 1


def test_elites_do_not_advance_normal_sequence_and_named_act_never_leaks() -> None:
    state = _single_next_room(
        new_run(
            "ELITE",
            "IRONCLAD",
            0,
            source_data={
                "act_id": "UNDERDOCKS",
            },
        ),
        RoomKind.ELITE,
    )
    expected = state.flags["act_room_sets"]["1"]["elite_encounters"][0]
    state = step(state, Action(type=ActionType.CHOOSE_NODE, target_id="fixture:next"))
    assert state.combat is not None
    assert state.combat.monsters[0].metadata["encounter_id"] == expected
    assert expected in {"TERROR_EEL_ELITE", "PHANTASMAL_GARDENERS_ELITE", "SKULKING_COLONY_ELITE"}
    assert state.flags["act_room_sets"]["1"]["normal_encounters_visited"] == 0
    assert state.flags["act_room_sets"]["1"]["elite_encounters_visited"] == 1


def test_event_queue_advances_skipped_and_selected_ids_and_survives_reload() -> None:
    state = new_run("EVENT", "IRONCLAD", 0, source_data={"act_id": "UNDERDOCKS"})
    plans = dict(state.flags["act_room_sets"])
    plans["1"] = {**plans["1"], "events": ["AROMA_OF_CHAOS", "ABYSSAL_BATHS", "TRASH_HEAP"]}
    state = _single_next_room(
        state.model_copy(
            update={
                "flags": {
                    **state.flags,
                    "act_room_sets": plans,
                    "resolve_unknown_rooms": False,
                }
            }
        ),
        RoomKind.EVENT,
    )
    action = Action(type=ActionType.CHOOSE_NODE, target_id="fixture:next")
    actual = step(state, action)
    restored = step(load_state(serialize_state(state)), action)
    assert serialize_state(actual) == serialize_state(restored)
    assert actual.event is not None and actual.event.event_id == "ABYSSAL_BATHS"
    assert actual.flags["act_room_sets"]["1"]["events_visited"] == 2
    assert "ABYSSAL_BATHS" in actual.flags["visited_event_ids"]


def test_locked_neow_skips_ancient_and_multiplayer_removes_one_room() -> None:
    first = new_run(
        "FIRST",
        "IRONCLAD",
        0,
        source_data={
            "revealed_epochs": [],
            "number_of_runs": 0,
        },
    )
    assert first.phase == RunPhase.MAP
    assert first.ancient is None
    assert first.flags["act_ids"]["1"] == "OVERGROWTH"
    assert first.flags["act_room_sets"]["1"]["normal_encounters"][:3] == [
        "NIBBITS_WEAK",
        "SLIMES_WEAK",
        "SHRINKER_BEETLE_WEAK",
    ]
    multi = new_run("MULTI", "IRONCLAD", 0, source_data={"player_count": 2})
    assert multi.map is not None and max(node.floor for node in multi.map.nodes) == 15
    assert len(multi.flags["act_room_sets"]["1"]["normal_encounters"]) == 14


def test_double_boss_map_and_distinct_pregenerated_bosses() -> None:
    state = new_run("DOUBLE", "IRONCLAD", 10, source_data={"act": 3})
    assert state.map is not None
    bosses = [node for node in state.map.nodes if node.kind == RoomKind.BOSS]
    assert len(bosses) == 2
    assert bosses[0].floor + 1 == bosses[1].floor
    assert state.map.boss_node_id == bosses[1].node_id
    plan = state.flags["act_room_sets"]["3"]
    assert plan["boss_encounter"] != plan["second_boss_encounter"]


def test_map_generation_uses_dedicated_act_stream() -> None:
    rng = RunRng("STREAMS")
    before = random_stream(rng, "rewards").random()
    generate_standard_map(random_stream(rng, "act_1_map"), act=1, act_id="OVERGROWTH")
    after = random_stream(rng, "rewards").random()
    baseline = random_stream(RunRng("STREAMS"), "rewards")
    assert (before, after) == (baseline.random(), baseline.random())


def test_public_boss_completion_uses_pregenerated_next_act_and_reload_is_identical() -> None:
    state = new_run(
        "ADVANCE",
        "IRONCLAD",
        0,
        source_data={
            "act_id": "OVERGROWTH",
            "starter_deck": [
                {
                    "id": "source_test_sweep",
                    "name": "Source Test Sweep",
                    "type": "Attack",
                    "cost": 0,
                    "target": "all_enemies",
                    "effects": {"all_damage": 99999},
                }
            ],
            "draw_per_turn": 1,
        },
    )
    expected_ancient = state.flags["act_room_sets"]["2"]["ancient_id"].lower()
    expected_plan = state.flags["act_room_sets"]["2"]
    state = _single_next_room(state, RoomKind.BOSS)
    state = step(state, Action(type=ActionType.CHOOSE_NODE, target_id="fixture:next"))
    attack = next(action for action in legal_actions(state) if action.type == ActionType.PLAY_CARD)
    state = step(state, attack)
    assert state.phase == RunPhase.REWARD
    proceed = next(action for action in legal_actions(state) if action.type == ActionType.PROCEED)
    restored = step(load_state(serialize_state(state)), proceed)
    state = step(state, proceed)
    assert serialize_state(state) == serialize_state(restored)
    assert state.act == 2 and state.phase == RunPhase.ANCIENT
    assert state.ancient is not None and state.ancient.ancient_id == expected_ancient
    assert state.map is not None and state.map.act == 2
    assert state.flags["act_room_sets"]["2"] == expected_plan
