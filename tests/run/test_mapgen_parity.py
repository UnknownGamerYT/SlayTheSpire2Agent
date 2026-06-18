from __future__ import annotations

import pytest

from sts2sim.mechanics.mapgen import (
    MapEdge,
    MapLayout,
    MapNode,
    NodeKind,
    assert_map_layout_valid,
    default_act_constraints,
    default_act_floor_count,
    map_layout_from_state,
    validate_act_map_parity,
)


def _valid_sparse_layout() -> MapLayout:
    lanes = (1, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 1)
    kinds = (
        NodeKind.START,
        NodeKind.MONSTER,
        NodeKind.EVENT,
        NodeKind.SHOP,
        NodeKind.MONSTER,
        NodeKind.EVENT,
        NodeKind.ELITE,
        NodeKind.MONSTER,
        NodeKind.SHOP,
        NodeKind.TREASURE,
        NodeKind.MONSTER,
        NodeKind.EVENT,
        NodeKind.ELITE,
        NodeKind.MONSTER,
        NodeKind.SHOP,
        NodeKind.REST,
        NodeKind.BOSS,
    )
    nodes = tuple(
        MapNode(f"a1:{floor}:{lane}", floor, kind, lane=lane)
        for floor, (lane, kind) in enumerate(zip(lanes, kinds, strict=True))
    )
    edges = tuple(
        MapEdge(nodes[floor].node_id, nodes[floor + 1].node_id)
        for floor in range(len(nodes) - 1)
    )
    return MapLayout(nodes=nodes, edges=edges)


def test_default_act_constraints_match_realistic_floor_counts() -> None:
    assert default_act_floor_count(1) == 17
    assert default_act_floor_count(2) == 16
    assert default_act_floor_count(3) == 15
    assert default_act_constraints(1).boss_floor == 16


def test_sparse_valid_layout_satisfies_map_parity() -> None:
    layout = _valid_sparse_layout()
    report = validate_act_map_parity(layout, act=1)

    assert report.ok
    assert_map_layout_valid(layout, default_act_constraints(1))


def test_map_layout_from_state_accepts_engine_like_mappings() -> None:
    payload = {
        "nodes": [
            {"node_id": "a1:0:0", "floor": 0, "lane": 0, "kind": "start"},
            {"node_id": "a1:1:0", "floor": 1, "lane": 0, "kind": "monster"},
        ],
        "edges": [{"from_id": "a1:0:0", "to_id": "a1:1:0"}],
    }

    layout = map_layout_from_state(payload)

    assert layout.nodes[0] == MapNode("a1:0:0", 0, NodeKind.START, lane=0)
    assert layout.edges == (MapEdge("a1:0:0", "a1:1:0"),)


def test_sparse_lane_jump_and_missing_rest_before_boss_are_reported() -> None:
    layout = _valid_sparse_layout()
    nodes = list(layout.nodes)
    nodes[2] = MapNode("a1:2:4", 2, NodeKind.EVENT, lane=4)
    nodes[15] = MapNode("a1:15:1", 15, NodeKind.MONSTER, lane=1)
    edges = list(layout.edges)
    edges[1] = MapEdge(nodes[1].node_id, nodes[2].node_id)
    edges[2] = MapEdge(nodes[2].node_id, nodes[3].node_id)
    edges[14] = MapEdge(nodes[14].node_id, nodes[15].node_id)
    edges[15] = MapEdge(nodes[15].node_id, nodes[16].node_id)

    report = validate_act_map_parity(MapLayout(nodes=tuple(nodes), edges=tuple(edges)), act=1)
    codes = {issue.code for issue in report.issues}

    assert "sparse_path_lane_jump" in codes
    assert "required_floor_kind" in codes
    assert "missing_rest_before_boss" in codes


def test_repeated_elite_shop_rest_and_duplicate_branch_types_are_reported() -> None:
    layout = _valid_sparse_layout()
    nodes = list(layout.nodes)
    nodes[7] = MapNode("a1:7:1", 7, NodeKind.ELITE, lane=1)
    nodes[14] = MapNode("a1:14:0", 14, NodeKind.REST, lane=0)
    nodes.append(MapNode("a1:3:2", 3, NodeKind.SHOP, lane=2))
    edges = list(layout.edges)
    edges.append(MapEdge(nodes[2].node_id, nodes[-1].node_id))

    report = validate_act_map_parity(MapLayout(nodes=tuple(nodes), edges=tuple(edges)), act=1)
    codes = {issue.code for issue in report.issues}

    assert "repeated_special_kind" in codes
    assert "duplicate_outgoing_destination_kind" in codes
    assert "disallowed_node_kind" in codes


def test_assert_map_layout_valid_raises_compact_error() -> None:
    layout = _valid_sparse_layout()
    broken = MapLayout(nodes=layout.nodes[:-1], edges=layout.edges[:-1])

    with pytest.raises(ValueError, match="missing_boss"):
        assert_map_layout_valid(broken, default_act_constraints(1))
