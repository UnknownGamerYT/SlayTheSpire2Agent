"""Map generation constraints and validators.

This is not a full map generator.  It provides source-backed constraints and a
layout validator so a future generator can be audited independently.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sts2sim.content.sources import STS1_COMPAT_SOURCE, SourceRef

from .ascension import AscensionFlag, ascension_enabled


class NodeKind(str, Enum):
    START = "start"
    MONSTER = "monster"
    ELITE = "elite"
    EVENT = "event"
    SHOP = "shop"
    REST = "rest"
    TREASURE = "treasure"
    BOSS = "boss"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FloorConstraint:
    floor: int
    allowed_kinds: frozenset[NodeKind]
    min_nodes: int = 1
    max_nodes: int = 7
    required_kind: NodeKind | None = None


@dataclass(frozen=True, slots=True)
class MapConstraints:
    act: int
    floors: tuple[FloorConstraint, ...]
    boss_floor: int
    start_floor: int = 0
    require_rest_before_boss: bool = True
    prevent_same_kind_edges: frozenset[NodeKind] = field(
        default_factory=lambda: frozenset(
            {NodeKind.ELITE, NodeKind.SHOP, NodeKind.REST, NodeKind.TREASURE}
        )
    )
    require_unique_outgoing_destination_kinds: bool = True
    elite_bias: int = 0
    max_sparse_lane_delta: int = 1
    expected_floor_count: int | None = None
    required_boss_count: int = 1
    source: SourceRef = STS1_COMPAT_SOURCE

    @property
    def by_floor(self) -> Mapping[int, FloorConstraint]:
        return {constraint.floor: constraint for constraint in self.floors}


@dataclass(frozen=True, slots=True)
class MapNode:
    node_id: str
    floor: int
    kind: NodeKind
    lane: int | None = None


@dataclass(frozen=True, slots=True)
class MapEdge:
    from_id: str
    to_id: str


@dataclass(frozen=True, slots=True)
class MapLayout:
    nodes: tuple[MapNode, ...]
    edges: tuple[MapEdge, ...]


@dataclass(frozen=True, slots=True)
class MapValidationIssue:
    code: str
    message: str
    node_id: str | None = None


@dataclass(frozen=True, slots=True)
class MapValidationReport:
    issues: tuple[MapValidationIssue, ...]
    source: SourceRef

    @property
    def ok(self) -> bool:
        return not self.issues


DEFAULT_ACT_FLOOR_COUNTS = {
    1: 17,
    2: 16,
    3: 15,
}


def default_act_floor_count(act: int) -> int:
    """Return the generator's default room rows plus start and boss rows."""

    if act <= 1:
        return DEFAULT_ACT_FLOOR_COUNTS[1]
    if act == 2:
        return DEFAULT_ACT_FLOOR_COUNTS[2]
    return DEFAULT_ACT_FLOOR_COUNTS[3]


def default_act_constraints(act: int = 1, *, ascension_level: int = 0) -> MapConstraints:
    """Return conservative act-map constraints.

    The floor shape mirrors common deckbuilder act layouts.  Exact STS2 map
    generation weights should replace this table when extracted.
    """

    elite_bias = 1 if ascension_enabled(ascension_level, AscensionFlag.MORE_ELITES) else 0
    floor_count = default_act_floor_count(act)
    boss_floor = floor_count - 1
    rest_floor = boss_floor - 1
    treasure_floor = max(2, boss_floor - 7)
    floors: list[FloorConstraint] = [
        FloorConstraint(
            0,
            frozenset({NodeKind.START}),
            min_nodes=1,
            max_nodes=1,
            required_kind=NodeKind.START,
        ),
    ]
    for floor in range(1, boss_floor):
        if floor == rest_floor:
            floors.append(
                FloorConstraint(
                    floor,
                    frozenset({NodeKind.REST}),
                    min_nodes=1,
                    max_nodes=7,
                    required_kind=NodeKind.REST,
                )
            )
        elif floor == treasure_floor:
            floors.append(
                FloorConstraint(
                    floor,
                    frozenset({NodeKind.TREASURE}),
                    min_nodes=1,
                    max_nodes=7,
                    required_kind=NodeKind.TREASURE,
                )
            )
        elif floor == 1:
            floors.append(
                FloorConstraint(
                    floor,
                    frozenset({NodeKind.MONSTER}),
                    min_nodes=1,
                    max_nodes=7,
                    required_kind=NodeKind.MONSTER,
                )
            )
        elif floor <= 5:
            floors.append(
                FloorConstraint(
                    floor,
                    frozenset({NodeKind.MONSTER, NodeKind.EVENT, NodeKind.SHOP}),
                    min_nodes=1,
                    max_nodes=7,
                )
            )
        elif floor == rest_floor - 1:
            floors.append(
                FloorConstraint(
                    floor,
                    frozenset(
                        {
                            NodeKind.MONSTER,
                            NodeKind.ELITE,
                            NodeKind.EVENT,
                            NodeKind.SHOP,
                            NodeKind.TREASURE,
                        }
                    ),
                    min_nodes=1,
                    max_nodes=7,
                )
            )
        else:
            floors.append(
                FloorConstraint(
                    floor,
                    frozenset(
                        {
                            NodeKind.MONSTER,
                            NodeKind.ELITE,
                            NodeKind.EVENT,
                            NodeKind.SHOP,
                            NodeKind.REST,
                            NodeKind.TREASURE,
                        }
                    ),
                    min_nodes=1,
                    max_nodes=7,
                )
            )
    floors.append(
        FloorConstraint(
            boss_floor,
            frozenset({NodeKind.BOSS}),
            min_nodes=1,
            max_nodes=1,
            required_kind=NodeKind.BOSS,
        )
    )
    return MapConstraints(
        act=act,
        floors=tuple(floors),
        boss_floor=boss_floor,
        elite_bias=elite_bias,
        expected_floor_count=floor_count,
    )


def allowed_kinds_for_floor(floor: int, constraints: MapConstraints) -> frozenset[NodeKind]:
    floor_constraint = constraints.by_floor.get(floor)
    if floor_constraint is None:
        return frozenset()
    return floor_constraint.allowed_kinds


def map_layout_from_state(map_state: Any) -> MapLayout:
    """Convert engine-like map state objects or mappings into a pure layout."""

    nodes = tuple(_map_node_from_any(node) for node in _field_value(map_state, "nodes", ()))
    edges = tuple(_map_edge_from_any(edge) for edge in _field_value(map_state, "edges", ()))
    return MapLayout(nodes=nodes, edges=edges)


def validate_act_map_parity(
    layout: MapLayout,
    *,
    act: int = 1,
    ascension_level: int = 0,
) -> MapValidationReport:
    """Validate a layout against the current act-map parity constraints."""

    return validate_map_layout(
        layout,
        default_act_constraints(act, ascension_level=ascension_level),
    )


def assert_map_layout_valid(layout: MapLayout, constraints: MapConstraints) -> None:
    """Raise a compact error if a layout violates parity constraints."""

    report = validate_map_layout(layout, constraints)
    if report.ok:
        return
    details = "; ".join(
        f"{issue.code}{f'[{issue.node_id}]' if issue.node_id else ''}: {issue.message}"
        for issue in report.issues
    )
    raise ValueError(details)


def validate_map_layout(layout: MapLayout, constraints: MapConstraints) -> MapValidationReport:
    issues: list[MapValidationIssue] = []
    nodes_by_id: dict[str, MapNode] = {}
    nodes_by_floor: dict[int, list[MapNode]] = defaultdict(list)

    for node in layout.nodes:
        if node.node_id in nodes_by_id:
            issues.append(
                MapValidationIssue(
                    "duplicate_node_id",
                    f"Duplicate node id '{node.node_id}'.",
                    node.node_id,
                )
            )
            continue
        nodes_by_id[node.node_id] = node
        nodes_by_floor[node.floor].append(node)

    constraints_by_floor = constraints.by_floor
    if constraints.expected_floor_count is not None:
        expected_floors = set(range(constraints.start_floor, constraints.boss_floor + 1))
        if len(expected_floors) != constraints.expected_floor_count:
            issues.append(
                MapValidationIssue(
                    "constraint_floor_count_mismatch",
                    "Constraint floor count does not match its boss floor.",
                )
            )
        actual_floors = {node.floor for node in layout.nodes}
        if actual_floors and actual_floors != expected_floors:
            missing = sorted(expected_floors - actual_floors)
            extra = sorted(actual_floors - expected_floors)
            issues.append(
                MapValidationIssue(
                    "floor_count_mismatch",
                    f"Map floors do not match expected rows; missing={missing} extra={extra}.",
                )
            )

    for floor, floor_constraint in constraints_by_floor.items():
        floor_nodes = nodes_by_floor.get(floor, [])
        if len(floor_nodes) < floor_constraint.min_nodes:
            issues.append(MapValidationIssue("too_few_nodes", f"Floor {floor} has too few nodes."))
        if len(floor_nodes) > floor_constraint.max_nodes:
            issues.append(
                MapValidationIssue("too_many_nodes", f"Floor {floor} has too many nodes.")
            )
        for node in floor_nodes:
            if node.kind not in floor_constraint.allowed_kinds:
                issues.append(
                    MapValidationIssue(
                        "disallowed_node_kind",
                        (
                            f"Node '{node.node_id}' kind {node.kind.value} is not allowed "
                            f"on floor {floor}."
                        ),
                        node.node_id,
                    )
                )
        if floor_constraint.required_kind is not None:
            for node in floor_nodes:
                if node.kind is not floor_constraint.required_kind:
                    issues.append(
                        MapValidationIssue(
                            "required_floor_kind",
                            f"Floor {floor} requires {floor_constraint.required_kind.value} nodes.",
                            node.node_id,
                        )
                    )

    for node in layout.nodes:
        if node.floor not in constraints_by_floor:
            issues.append(
                MapValidationIssue(
                    "unknown_floor",
                    f"Node '{node.node_id}' is on an unconstrained floor.",
                    node.node_id,
                )
            )

    outgoing: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, list[str]] = defaultdict(list)
    for edge in layout.edges:
        start = nodes_by_id.get(edge.from_id)
        end = nodes_by_id.get(edge.to_id)
        if start is None:
            issues.append(
                MapValidationIssue(
                    "edge_missing_start",
                    f"Edge start '{edge.from_id}' does not exist.",
                )
            )
            continue
        if end is None:
            issues.append(
                MapValidationIssue(
                    "edge_missing_end",
                    f"Edge end '{edge.to_id}' does not exist.",
                )
            )
            continue
        if end.floor != start.floor + 1:
            issues.append(
                MapValidationIssue(
                    "edge_skips_floor",
                    f"Edge '{edge.from_id}->{edge.to_id}' must connect adjacent floors.",
                    edge.from_id,
                )
            )
        start_lane = _lane_for_node(start)
        end_lane = _lane_for_node(end)
        if (
            start_lane is not None
            and end_lane is not None
            and abs(end_lane - start_lane) > constraints.max_sparse_lane_delta
        ):
            issues.append(
                MapValidationIssue(
                    "sparse_path_lane_jump",
                    (
                        f"Edge '{edge.from_id}->{edge.to_id}' moves from lane "
                        f"{start_lane} to {end_lane}."
                    ),
                    edge.from_id,
                )
            )
        if start.kind in constraints.prevent_same_kind_edges and start.kind is end.kind:
            issues.append(
                MapValidationIssue(
                    "repeated_special_kind",
                    f"Edge repeats special node kind {start.kind.value}.",
                    edge.from_id,
                )
            )
        outgoing[edge.from_id].append(edge.to_id)
        incoming[edge.to_id].append(edge.from_id)

    if constraints.require_unique_outgoing_destination_kinds:
        for node_id, target_ids in outgoing.items():
            if len(target_ids) < 2:
                continue
            start = nodes_by_id[node_id]
            if start.kind is NodeKind.START:
                continue
            seen_kinds: dict[NodeKind, str] = {}
            for target_id in target_ids:
                target = nodes_by_id[target_id]
                previous = seen_kinds.setdefault(target.kind, target_id)
                if previous != target_id:
                    issues.append(
                        MapValidationIssue(
                            "duplicate_outgoing_destination_kind",
                            (
                                f"Node '{node_id}' has multiple outgoing paths to "
                                f"{target.kind.value} rooms."
                            ),
                            node_id,
                        )
                    )
                    break

    start_ids = [node.node_id for node in layout.nodes if node.kind is NodeKind.START]
    boss_ids = [
        node.node_id
        for node in layout.nodes
        if node.kind is NodeKind.BOSS and node.floor == constraints.boss_floor
    ]
    if not start_ids:
        issues.append(MapValidationIssue("missing_start", "Map has no start node."))
    if not boss_ids:
        issues.append(MapValidationIssue("missing_boss", "Map has no boss node on the boss floor."))
    if len(boss_ids) != constraints.required_boss_count:
        issues.append(
            MapValidationIssue(
                "boss_count_mismatch",
                (
                    f"Map has {len(boss_ids)} boss nodes on floor {constraints.boss_floor}; "
                    f"expected {constraints.required_boss_count}."
                ),
            )
        )

    reachable = _reachable_from(starts=start_ids, outgoing=outgoing)
    for node in layout.nodes:
        if node.node_id not in reachable and node.kind is not NodeKind.START:
            issues.append(
                MapValidationIssue(
                    "unreachable_node",
                    f"Node '{node.node_id}' is unreachable from start.",
                    node.node_id,
                )
            )
    for boss_id in boss_ids:
        if boss_id not in reachable:
            issues.append(
                MapValidationIssue(
                    "boss_unreachable",
                    f"Boss node '{boss_id}' is unreachable from start.",
                    boss_id,
                )
            )

    reverse_reachable = _reachable_from(starts=boss_ids, outgoing=incoming)
    for node in layout.nodes:
        if node.node_id not in reverse_reachable and node.kind is not NodeKind.BOSS:
            issues.append(
                MapValidationIssue(
                    "dead_end",
                    f"Node '{node.node_id}' cannot reach the boss.",
                    node.node_id,
                )
            )

    if constraints.require_rest_before_boss:
        boss_inputs = {edge.from_id for edge in layout.edges if edge.to_id in boss_ids}
        for node_id in boss_inputs:
            node = nodes_by_id[node_id]
            if node.kind is not NodeKind.REST:
                issues.append(
                    MapValidationIssue(
                        "missing_rest_before_boss",
                        "Every boss input should come from a rest floor.",
                        node_id,
                    )
                )

    return MapValidationReport(issues=tuple(issues), source=constraints.source)


def _map_node_from_any(raw_node: Any) -> MapNode:
    kind = _node_kind_from_value(_field_value(raw_node, "kind", NodeKind.UNKNOWN))
    return MapNode(
        node_id=str(_field_value(raw_node, "node_id", _field_value(raw_node, "id", ""))),
        floor=int(_field_value(raw_node, "floor", 0)),
        kind=kind,
        lane=_optional_int(_field_value(raw_node, "lane", None)),
    )


def _map_edge_from_any(raw_edge: Any) -> MapEdge:
    return MapEdge(
        from_id=str(_field_value(raw_edge, "from_id", _field_value(raw_edge, "from", ""))),
        to_id=str(_field_value(raw_edge, "to_id", _field_value(raw_edge, "to", ""))),
    )


def _node_kind_from_value(value: Any) -> NodeKind:
    if isinstance(value, NodeKind):
        return value
    raw_value = getattr(value, "value", value)
    try:
        return NodeKind(str(raw_value))
    except ValueError:
        return NodeKind.UNKNOWN


def _field_value(source: Any, key: str, default: Any) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _lane_for_node(node: MapNode) -> int | None:
    if node.lane is not None:
        return node.lane
    for separator in (":", "-"):
        parts = node.node_id.rsplit(separator, maxsplit=1)
        if len(parts) == 2:
            return _optional_int(parts[1])
    return None


def _reachable_from(*, starts: list[str], outgoing: Mapping[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    queue: deque[str] = deque(starts)
    while queue:
        node_id = queue.popleft()
        if node_id in seen:
            continue
        seen.add(node_id)
        for next_id in outgoing.get(node_id, []):
            if next_id not in seen:
                queue.append(next_id)
    return seen
