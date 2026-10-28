"""Stable v0.107.1 map path pruning, count repair, and coordinate cleanup.

Derived from ``MapPathPruning.cs``, ``MapPostProcessing.cs`` and
``MapCoord.cs`` in zhiyue/sts2-rl-agent at
5d874e855ae1d0550d5c30416fadd7dbb0d133b7. Its GAME_VERSIONS.md identifies
Steam public build 23811903, game commit 59260271. Points use identity and
insertion-ordered adjacency, matching the source's point references/HashSets.
Start and boss are outside the grid, as in ActMap.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from random import Random


@dataclass(eq=False, slots=True)
class MapPoint:
    col: int
    row: int
    kind: str = "unassigned"
    parents: list[MapPoint] = field(default_factory=list)
    children: list[MapPoint] = field(default_factory=list)
    can_modify: bool = True

    def add_child(self, child: MapPoint) -> None:
        if child not in self.children:
            self.children.append(child)
        if self not in child.parents:
            child.parents.append(self)

    def remove_child(self, child: MapPoint) -> None:
        if child in self.children:
            self.children.remove(child)
        if self in child.parents:
            child.parents.remove(self)


Grid = dict[tuple[int, int], MapPoint]
PointValidator = Callable[[str, MapPoint], bool]
_KIND_VALUES = {
    "unassigned": 0, "event": 1, "shop": 2, "treasure": 3,
    "rest": 4, "monster": 5, "elite": 6, "boss": 7, "start": 8,
}


def all_points(grid: Grid) -> list[MapPoint]:
    """ActMap.GetAllMapPoints traverses columns first, then rows."""
    return [point for _, point in sorted(grid.items())]


def stable_shuffle_points(points: list[MapPoint], rng: Random) -> None:
    """StableShuffle sorts by MapCoord's (column, row), then Fisher-Yates."""
    points.sort(key=lambda point: (point.col, point.row))
    rng.shuffle(points)


def prune_and_repair(
    grid: Grid,
    start_points: set[MapPoint],
    start: MapPoint,
    rng: Random,
    counts: Mapping[str, int],
    is_valid: PointValidator,
) -> None:
    """Run the source's maximum three prune/repair passes, in place."""
    for _ in range(3):
        prune_duplicate_segments(grid, start_points, start, rng)
        if not repair_point_types(grid, rng, counts, is_valid):
            break


def repair_point_types(
    grid: Grid, rng: Random, counts: Mapping[str, int], is_valid: PointValidator
) -> bool:
    """Restore Shops, Elites, Rests, Unknowns in that exact source order."""
    changed = False
    for kind in ("shop", "elite", "rest", "event"):
        missing = counts.get(kind, 0) - sum(point.kind == kind for point in grid.values())
        if missing <= 0:
            continue
        candidates = [p for p in all_points(grid) if p.kind == "monster" and p.can_modify]
        stable_shuffle_points(candidates, rng)
        for point in candidates:
            if missing == 0:
                break
            if is_valid(kind, point):
                point.kind = kind
                missing -= 1
                changed = True
    return changed


def prune_duplicate_segments(
    grid: Grid, start_points: set[MapPoint], start: MapPoint, rng: Random
) -> None:
    iterations = 0
    while _prune_paths(grid, start_points, find_matching_segments(start), rng):
        iterations += 1
        if iterations > 50:
            raise ValueError(f"Unable to prune matching segments in {iterations} iterations")


def find_all_paths(point: MapPoint) -> list[list[MapPoint]]:
    if point.kind == "boss":
        return [[point]]
    return [[point, *path] for child in point.children for path in find_all_paths(child)]


def find_matching_segments(start: MapPoint) -> list[list[list[MapPoint]]]:
    """Return disjoint interior paths sharing endpoints and point-type sequence."""
    segments: dict[str, list[list[MapPoint]]] = {}
    for path in find_all_paths(start):
        for index in range(len(path) - 1):
            first = path[index]
            if len(first.children) <= 1 and first.row != 0:
                continue
            for offset in range(2, len(path) - index):
                last = path[index + offset]
                if len(last.parents) < 2:
                    continue
                segment = path[index:index + offset + 1]
                prefix = f"{first.row}-" if first.row == 0 else f"{first.col},{first.row}-"
                kinds = ",".join(str(_KIND_VALUES[p.kind]) for p in segment)
                key = f"{prefix}{last.col},{last.row}-{kinds}"
                existing = segments.setdefault(key, [])
                if not any(_overlap(other, segment) for other in existing):
                    existing.append(segment)
    # C# uses an ordinal SortedDictionary; string order also governs RNG use.
    return [segments[key] for key in sorted(segments) if len(segments[key]) > 1]


def _overlap(first: list[MapPoint], second: list[MapPoint]) -> bool:
    if len(first) < 3 or len(second) < 3:
        return False
    return any(first[i] is second[i] for i in range(1, len(first) - 1))


def _prune_paths(
    grid: Grid, start_points: set[MapPoint], matches: list[list[list[MapPoint]]], rng: Random
) -> bool:
    for group in matches:
        rng.shuffle(group)
        removed = 0
        for segment in group:
            if removed == len(group) - 1:
                break
            if _prune_segment(grid, start_points, segment):
                removed += 1
        if removed:
            return True
        for segment in group:
            if _break_parent_child_relationship(segment):
                return True
    return False


def _prune_segment(grid: Grid, start_points: set[MapPoint], segment: list[MapPoint]) -> bool:
    changed = False
    for index, point in enumerate(segment[:-1]):
        if (point.col, point.row) not in grid and point.kind not in {"start", "boss"}:
            return True
        if (
            len(point.children) > 1
            or len(point.parents) > 1
            or any(len(p.children) == 1 and (p.col, p.row) in grid for p in point.parents)
        ):
            continue
        remainder = segment[index:]
        if any(len(p.children) > 1 and len(p.parents) == 1 for p in remainder):
            continue
        if len(segment[-1].parents) == 1:
            return False
        if any(child not in segment and len(child.parents) == 1 for child in point.children):
            continue
        _remove_point(grid, start_points, point)
        changed = True
    return changed


def _remove_point(grid: Grid, start_points: set[MapPoint], point: MapPoint) -> None:
    grid.pop((point.col, point.row), None)
    start_points.discard(point)
    for child in point.children.copy():
        point.remove_child(child)
    for parent in point.parents.copy():
        parent.remove_child(point)


def _break_parent_child_relationship(segment: list[MapPoint]) -> bool:
    changed = False
    for point, child in zip(segment[:-1], segment[1:], strict=True):
        if len(point.children) >= 2 and len(child.parents) != 1:
            point.remove_child(child)
            changed = True
    return changed


def postprocess(grid: Grid, width: int, length: int) -> Grid:
    """Run CenterGrid, SpreadAdjacentMapPoints, StraightenPaths in order."""
    center_grid(grid, width, length)
    spread_adjacent_points(grid, width, length)
    straighten_paths(grid, width, length)
    return grid


def center_grid(grid: Grid, width: int, length: int) -> None:
    if width < 2:
        return
    occupied = {col for col, _ in grid}
    left_empty = 0 not in occupied and 1 not in occupied
    right_empty = width - 1 not in occupied and width - 2 not in occupied
    shift = -1 if left_empty and not right_empty else 1 if right_empty and not left_empty else 0
    if not shift:
        return
    points = list(grid.values())
    grid.clear()
    for point in points:
        target = point.col + shift
        if 0 <= target < width and 0 <= point.row < length:
            point.col = target
            grid[target, point.row] = point


def _allowed_positions(point: MapPoint, width: int) -> list[int]:
    allowed = set(range(width))
    for other in (*point.parents, *point.children):
        allowed.intersection_update(range(max(0, other.col - 1), min(width, other.col + 2)))
    return sorted(allowed)


def _gap(col: int, points: list[MapPoint], current: MapPoint) -> int:
    return min((abs(col - p.col) for p in points if p is not current), default=2**31 - 1)


def spread_adjacent_points(grid: Grid, width: int, length: int) -> None:
    for row in range(length):
        points = [grid[col, row] for col in range(width) if (col, row) in grid]
        changed = True
        while changed:
            changed = False
            for point in points:
                col = point.col
                best_col, best_gap = col, _gap(col, points, point)
                for target in _allowed_positions(point, width):
                    if target == col or ((target, row) in grid and grid[target, row] is not point):
                        continue
                    gap = _gap(target, points, point)
                    if gap > best_gap:
                        best_col, best_gap = target, gap
                if best_col != col:
                    del grid[col, row]
                    grid[best_col, row] = point
                    point.col = best_col
                    changed = True


def straighten_paths(grid: Grid, width: int, length: int) -> None:
    for row in range(length):
        for col in range(width):
            point = grid.get((col, row))
            if point is None or len(point.parents) != 1 or len(point.children) != 1:
                continue
            parent, child = point.parents[0], point.children[0]
            shift = (
                1 if point.col < child.col and point.col < parent.col
                else -1 if point.col > child.col and point.col > parent.col
                else 0
            )
            target = col + shift
            if shift and 0 <= target < width and (target, row) not in grid:
                del grid[col, row]
                grid[target, row] = point
                point.col = target
