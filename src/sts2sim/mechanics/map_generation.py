"""Stable v0.107.1 standard act map construction.

Ported from StandardActMap, the four ActModel.GetMapPointTypes overrides,
MapPathPruning and MapPostProcessing in the version-qualified source listed in
docs/run_generation_stable.md. The engine retains its own map state models.
"""

from __future__ import annotations

from collections import deque
from random import Random

from .map_postprocessing import (
    Grid,
    MapPoint,
    postprocess,
    prune_and_repair,
    stable_shuffle_points,
)
from .mapgen import MapEdge, MapLayout, MapNode, NodeKind


def map_point_counts(act_id: str, rng: Random, *, ascension_level: int = 0) -> dict[str, int]:
    """Roll the game's counts, including rejection-sampled Gaussian integers."""

    from sts2sim.engine.rng import stable_gaussian_int

    act_id = act_id.upper()
    if act_id == "GLORY":
        rests = rng.randrange(5, 7)
    else:
        rests = stable_gaussian_int(rng, 6 if act_id == "HIVE" else 7, 1, 6, 7)
    unknowns = stable_gaussian_int(rng, 12, 1, 10, 14)
    if act_id in {"HIVE", "GLORY"}:
        unknowns -= 1
    return {"rest": rests, "shop": 3, "elite": 8 if ascension_level >= 1 else 5, "event": unknowns}


def generate_standard_map(
    rng: Random,
    *,
    act: int,
    act_id: str,
    ascension_level: int = 0,
    room_count: int | None = None,
    width: int = 7,
    path_count: int = 7,
    replace_treasure: bool = False,
    has_second_boss: bool = False,
) -> MapLayout:
    """Generate source topology, room assignment, pruning and layout transforms.

    Width/room/path overrides support synthetic fixtures. The source game uses
    seven columns and seven generated paths; only that default is source parity.
    """

    width = max(1, min(7, width))
    rooms = room_count if room_count is not None else {1: 15, 2: 14, 3: 13}.get(act, 13)
    length = max(3, rooms + 1)
    counts = map_point_counts(act_id, rng, ascension_level=ascension_level)
    grid: Grid = {}
    start = MapPoint(width // 2, 0, "start", can_modify=False)
    boss = MapPoint(width // 2, length, "boss", can_modify=False)
    second_boss = (
        MapPoint(width // 2, length + 1, "boss", can_modify=False) if has_second_boss else None
    )
    start_points: set[MapPoint] = set()

    def get_point(col: int, row: int) -> MapPoint:
        if (col, row) not in grid:
            grid[col, row] = MapPoint(col, row)
        return grid[col, row]

    for iteration in range(max(1, path_count)):
        point = get_point(rng.randrange(width), 1)
        if iteration == 1 and width > 1:
            while point in start_points:
                point = get_point(rng.randrange(width), 1)
        start_points.add(point)
        while point.row < length - 1:
            steps = [-1, 0, 1]
            rng.shuffle(steps)
            for step in steps:
                col = max(0, min(width - 1, point.col + step))
                other = grid.get((col, point.row))
                delta = col - point.col
                if (
                    delta
                    and other
                    and any(child.col - other.col == -delta for child in other.children)
                ):
                    continue
                child = get_point(col, point.row + 1)
                point.add_child(child)
                point = child
                break
            else:
                raise ValueError("No valid noncrossing map step")

    for (_col, row), point in sorted(grid.items()):
        if row == length - 1:
            point.add_child(boss)
            point.kind = "rest"
            point.can_modify = False
        elif row == length - 7:
            point.kind = "elite" if replace_treasure else "treasure"
            point.can_modify = False
        elif row == 1:
            start.add_child(point)
            point.kind = "monster"
            point.can_modify = False
    if second_boss is not None:
        boss.add_child(second_boss)

    def is_valid(kind: str, point: MapPoint) -> bool:
        if point.row < 6 and kind in {"rest", "elite"}:
            return False
        if point.row >= length - 3 and kind == "rest":
            return False
        if kind in {"elite", "rest", "treasure", "shop"} and any(
            neighbor.kind == kind for neighbor in (*point.parents, *point.children)
        ):
            return False
        return not (
            kind in {"rest", "monster", "event", "elite", "shop"}
            and any(
                sibling is not point and sibling.kind == kind
                for parent in point.parents
                for sibling in parent.children
            )
        )

    queue = deque(kind for kind in ("rest", "shop", "elite", "event") for _ in range(counts[kind]))
    for _ in range(3):
        if not queue:
            break
        candidates = [point for point in grid.values() if point.kind == "unassigned"]
        stable_shuffle_points(candidates, rng)
        for point in candidates:
            if not queue:
                break
            for _ in range(len(queue)):
                kind = queue.popleft()
                if is_valid(kind, point):
                    point.kind = kind
                    break
                queue.append(kind)
    for point in grid.values():
        if point.kind == "unassigned":
            point.kind = "monster"
    prune_and_repair(grid, start_points, start, rng, counts, is_valid)
    grid = postprocess(grid, width, length)
    points = [start, *grid.values(), boss]
    if second_boss is not None:
        points.append(second_boss)

    def node_id(point: MapPoint) -> str:
        return f"a{act}:{point.row}:{point.col}"

    return MapLayout(
        nodes=tuple(
            MapNode(node_id(point), point.row, NodeKind(point.kind), point.col)
            for point in sorted(points, key=lambda p: (p.row, p.col))
        ),
        edges=tuple(
            MapEdge(node_id(point), node_id(child))
            for point in sorted(points, key=lambda p: (p.row, p.col))
            for child in sorted(point.children, key=lambda p: (p.row, p.col))
        ),
    )
