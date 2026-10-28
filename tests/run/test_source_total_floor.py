"""Shared source map history count at room-entry and late relic boundaries."""

from __future__ import annotations

import pytest

from sts2sim import new_run
from sts2sim.engine import MapNodeState, MapState, RoomKind, load_state, serialize_state
from sts2sim.engine.transitions import _source_total_floor
from sts2sim.mechanics.relic_bag_runtime import draw_source_relic
from sts2sim.mechanics.run_progress import source_total_floor


@pytest.mark.parametrize("act,history,entered,neow,expected", [
    (1, 0, False, True, 1), (1, 0, False, False, 0),
    (1, 0, True, False, 1), (1, 1, False, False, 1),
    (2, 15, False, True, 17), (2, 15, True, True, 18),
    (3, 38, False, False, 40), (3, 38, True, False, 41),
])
def test_total_floor_counts_ancients_current_room_and_first_run_start_consistently(
    act, history, entered, neow, expected,
) -> None:
    state = new_run("FLOOR", "IRONCLAD", 0)
    node = MapNodeState(node_id="current", act=act, floor=3, lane=3, kind=RoomKind.MONSTER)
    state = state.model_copy(update={
        "act": act, "room_history": tuple(str(i) for i in range(history)),
        "flags": {**state.flags, "generation_started_with_neow": neow},
        "map": MapState(act=act, nodes=(node,), current_node_id="current",
                        completed_node_ids=() if entered else ("current",)),
    })
    assert source_total_floor(state) == _source_total_floor(state) == expected
    assert source_total_floor(load_state(serialize_state(state))) == expected
    assert source_total_floor(state, {**state.flags, "total_floor": 11}) == 11


@pytest.mark.parametrize("entered,expected", [(False, "amethyst_aubergine"), (True, "anchor")])
def test_bag_and_event_eligibility_agree_at_late_treasure_boundary(entered, expected) -> None:
    state = new_run("FLOOR-BAG", "IRONCLAD", 0)
    node = MapNodeState(node_id="current", act=3, floor=8, lane=3, kind=RoomKind.TREASURE)
    state = state.model_copy(update={
        "act": 3, "room_history": tuple(str(i) for i in range(38)),
        "flags": {**state.flags, "generation_started_with_neow": False,
                  "player_relic_bag": {"Common": ["AMETHYST_AUBERGINE", "ANCHOR"]}},
        "map": MapState(act=3, nodes=(node,), current_node_id="current",
                        completed_node_ids=() if entered else ("current",)),
    })
    flags = dict(state.flags)
    assert draw_source_relic(state, flags, "common") == expected
    if entered:
        assert "AMETHYST_AUBERGINE" not in flags["player_relic_bag"]["Common"]
