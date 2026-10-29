"""Small graph fixtures for the stable game's map cleanup contracts.

Source: zhiyue/sts2-rl-agent@5d874e855ae1d0550d5c30416fadd7dbb0d133b7,
decompiled/MegaCrit.Sts2.Core.Map/{MapPathPruning,MapPostProcessing}.cs.
These synthetic graphs exercise the source rules, not captured game seeds.
"""

from __future__ import annotations

from random import Random

from sts2sim.mechanics.map_postprocessing import (
    Grid,
    MapPoint,
    all_points,
    center_grid,
    find_all_paths,
    find_matching_segments,
    prune_duplicate_segments,
    repair_point_types,
    spread_adjacent_points,
    stable_shuffle_points,
    straighten_paths,
)


def _diamond(second_kind: str = "monster") -> tuple[Grid, MapPoint, MapPoint, MapPoint]:
    start = MapPoint(3, 0, "start")
    left, right = MapPoint(2, 1, "monster"), MapPoint(4, 1, second_kind)
    boss = MapPoint(3, 2, "boss")
    start.add_child(left)
    start.add_child(right)
    left.add_child(boss)
    right.add_child(boss)
    return {(2, 1): left, (4, 1): right}, start, left, right


def test_same_type_diamond_prunes_one_route_preserving_reciprocal_edges() -> None:
    grid, start, left, right = _diamond()
    starts = {left, right}
    boss = left.children[0]
    assert len(find_matching_segments(start)) == 1

    prune_duplicate_segments(grid, starts, start, Random(0))

    assert len(grid) == len(starts) == 1
    remaining = next(iter(grid.values()))
    removed = right if remaining is left else left
    assert start.children == [remaining]
    assert remaining.parents == [start]
    assert remaining.children == [boss]
    assert boss.parents == [remaining]
    assert removed.children == removed.parents == []
    assert find_all_paths(start) == [[start, remaining, boss]]
    assert find_matching_segments(start) == []


def test_distinct_room_sequences_keep_both_routes() -> None:
    grid, start, left, right = _diamond("event")
    assert find_matching_segments(start) == []
    prune_duplicate_segments(grid, {left, right}, start, Random(0))
    assert len(grid) == 2
    assert len(find_all_paths(start)) == 2


def test_repair_counts_fixed_rooms_but_preserves_protected_monsters() -> None:
    first = MapPoint(0, 1, "monster", can_modify=False)
    fixed_rest = MapPoint(0, 6, "rest", can_modify=False)
    candidate = MapPoint(1, 4, "monster")
    rejected = MapPoint(2, 4, "monster")
    grid = {(p.col, p.row): p for p in (first, fixed_rest, candidate, rejected)}

    changed = repair_point_types(
        grid, Random(4), {"shop": 1, "rest": 1},
        lambda kind, point: kind == "shop" and point is candidate,
    )

    assert changed
    assert candidate.kind == "shop"
    assert first.kind == rejected.kind == "monster"
    assert fixed_rest.kind == "rest"
    assert not repair_point_types(grid, Random(4), {"shop": 1, "rest": 1}, lambda *_: True)


def test_stable_shuffle_normalizes_column_then_row_before_shuffle() -> None:
    points = [MapPoint(2, 1), MapPoint(0, 2), MapPoint(0, 1), MapPoint(1, 2)]
    reordered = list(reversed(points))
    stable_shuffle_points(points, Random(9))
    stable_shuffle_points(reordered, Random(9))
    assert points == reordered
    grid = {(p.col, p.row): p for p in reordered}
    assert [(p.col, p.row) for p in all_points(grid)] == [(0, 1), (0, 2), (1, 2), (2, 1)]


def test_center_grid_shifts_only_when_two_columns_on_one_side_are_empty() -> None:
    point, other = MapPoint(0, 1), MapPoint(1, 2)
    grid = {(0, 1): point, (1, 2): other}
    center_grid(grid, 7, 4)
    assert grid == {(1, 1): point, (2, 2): other}
    assert (point.col, other.col) == (1, 2)
    # A direct second call still shifts once; CenterGrid does not iterate.
    center_grid(grid, 7, 4)
    assert grid == {(2, 1): point, (3, 2): other}
    # Both outer pairs are now empty, so the source stops shifting.
    center_grid(grid, 7, 4)
    assert grid == {(2, 1): point, (3, 2): other}


def test_spreading_maximizes_gap_within_all_parent_and_child_constraints() -> None:
    parent, child = MapPoint(3, 0, "start"), MapPoint(3, 2, "boss")
    left, right = MapPoint(2, 1), MapPoint(3, 1)
    for point in (left, right):
        parent.add_child(point)
        point.add_child(child)
    grid = {(p.col, p.row): p for p in (left, right)}

    spread_adjacent_points(grid, 7, 2)

    assert (left.col, right.col) == (2, 4)
    assert grid == {(2, 1): left, (4, 1): right}


def test_straightening_follows_source_left_to_right_mutation_order() -> None:
    parent, child = MapPoint(3, 0, "start"), MapPoint(3, 2, "boss")
    point = MapPoint(1, 1)
    parent.add_child(point)
    point.add_child(child)
    grid = {(1, 1): point}
    straighten_paths(grid, 7, 2)
    # A move to the right is visited again by the same row's column loop.
    assert point.col == 3
    assert grid == {(3, 1): point}

    point.col = 5
    grid = {(5, 1): point}
    straighten_paths(grid, 7, 2)
    # A move to the left enters an already visited column.
    assert point.col == 4
    assert grid == {(4, 1): point}


def test_straightening_cannot_move_onto_an_occupied_position() -> None:
    parent, child = MapPoint(3, 0, "start"), MapPoint(3, 2, "boss")
    point, obstacle = MapPoint(1, 1), MapPoint(2, 1)
    parent.add_child(point)
    point.add_child(child)
    grid = {(1, 1): point, (2, 1): obstacle}
    straighten_paths(grid, 7, 2)
    assert point.col == 1
