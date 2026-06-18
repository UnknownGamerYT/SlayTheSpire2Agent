"""Text rendering for deterministic act maps."""

from __future__ import annotations

import argparse
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sts2sim.engine import MapState, RoomKind, new_run_state

LANE_GAP = 4
FLOOR_GAP = 4
DEFAULT_OUTPUT_DIR = Path("generated_maps")
RANDOM_SEED_LIMIT = 2**31

ROOM_SYMBOLS = {
    RoomKind.START: "A",
    RoomKind.MONSTER: "M",
    RoomKind.ELITE: "L",
    RoomKind.EVENT: "E",
    RoomKind.SHOP: "S",
    RoomKind.REST: "F",
    RoomKind.TREASURE: "T",
    RoomKind.BOSS: "B",
}

LEGEND = (
    "A=ancient/start, M=monster, L=elite, E=event, "
    "S=shop, F=fire, T=treasure, B=boss"
)


@dataclass(frozen=True)
class MapRenderResult:
    seed: int
    act: int
    character_id: str
    output_path: Path
    text: str


def render_map_text(
    seed: int,
    *,
    act: int = 1,
    character_id: str = "MAP_PREVIEW",
    map_width: int | None = None,
    map_floors: int | None = None,
    map_paths: int | None = None,
) -> str:
    """Create a text rendering for the actual engine-generated map."""

    source_data: dict[str, int] = {"act": act}
    if map_width is not None:
        source_data["map_width"] = map_width
    if map_floors is not None:
        source_data["map_floors"] = map_floors
    if map_paths is not None:
        source_data["map_paths"] = map_paths

    state = new_run_state(
        seed=seed,
        character_id=_normalized_character_id(character_id),
        ascension=0,
        source_data=source_data,
    )
    if state.map is None:
        raise RuntimeError("new run did not generate a map")
    return render_map_state(state.map, seed=seed, character_id=state.character_id)


def render_map_state(
    map_state: MapState,
    *,
    seed: int | None = None,
    character_id: str | None = None,
) -> str:
    """Render a map with stable node positions and edge directions."""

    if not map_state.nodes:
        return _header(seed=seed, act=map_state.act, character_id=character_id) + "\n<empty map>\n"

    max_floor = max(node.floor for node in map_state.nodes)
    max_lane = max(node.lane for node in map_state.nodes)
    grid_height = max_floor * FLOOR_GAP + 1
    grid_width = max_lane * LANE_GAP + 1
    grid = [[" " for _ in range(grid_width)] for _ in range(grid_height)]
    positions = {
        node.node_id: _node_position(node.floor, node.lane, max_floor)
        for node in map_state.nodes
    }

    for edge in map_state.edges:
        start = positions.get(edge.from_id)
        end = positions.get(edge.to_id)
        if start is None or end is None:
            continue
        _draw_edge(grid, start, end)

    for node in map_state.nodes:
        row, col = positions[node.node_id]
        grid[row][col] = ROOM_SYMBOLS[node.kind]

    body = _format_grid(grid, max_floor=max_floor, max_lane=max_lane)
    return _header(seed=seed, act=map_state.act, character_id=character_id) + "\n" + body + "\n"


def write_map_file(
    seed: int | None = None,
    *,
    act: int = 1,
    character_id: str = "MAP_PREVIEW",
    output_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    map_width: int | None = None,
    map_floors: int | None = None,
    map_paths: int | None = None,
) -> MapRenderResult:
    """Render a map and write it to a text file."""

    actual_seed = random_seed() if seed is None else seed
    text = render_map_text(
        actual_seed,
        act=act,
        character_id=character_id,
        map_width=map_width,
        map_floors=map_floors,
        map_paths=map_paths,
    )
    target = output_path or output_dir / f"map_seed_{actual_seed}_act_{act}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return MapRenderResult(
        seed=actual_seed,
        act=act,
        character_id=_normalized_character_id(character_id),
        output_path=target,
        text=text,
    )


def random_seed() -> int:
    """Return a fresh non-negative seed for map previews."""

    return secrets.randbelow(RANDOM_SEED_LIMIT)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a deterministic Slay the Spire 2 act map as text.",
    )
    parser.add_argument(
        "seed",
        nargs="?",
        type=int,
        help="Seed to render. Omit to generate a random seed.",
    )
    parser.add_argument(
        "--seed",
        dest="seed_option",
        type=int,
        help="Seed to render. Equivalent to the positional seed.",
    )
    parser.add_argument("--act", type=int, default=1, help="Act number to render.")
    parser.add_argument(
        "--character",
        default="MAP_PREVIEW",
        help="Character ID to record as the preview context.",
    )
    parser.add_argument("--width", type=int, default=None, help="Optional map width.")
    parser.add_argument("--floors", type=int, default=None, help="Optional floor count.")
    parser.add_argument("--paths", type=int, default=None, help="Optional path count.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Exact output text file path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used when --output is omitted.",
    )
    args = parser.parse_args(argv)

    if args.seed is not None and args.seed_option is not None:
        parser.error("provide either positional seed or --seed, not both")

    seed = args.seed_option if args.seed_option is not None else args.seed
    result = write_map_file(
        seed,
        act=args.act,
        character_id=args.character,
        output_path=args.output,
        output_dir=args.output_dir,
        map_width=args.width,
        map_floors=args.floors,
        map_paths=args.paths,
    )
    print(f"Wrote {result.output_path}")
    print(f"Seed: {result.seed}")
    print(f"Character: {result.character_id}")
    return 0


def _node_position(floor: int, lane: int, max_floor: int) -> tuple[int, int]:
    return ((max_floor - floor) * FLOOR_GAP, lane * LANE_GAP)


def _draw_edge(
    grid: list[list[str]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    start_row, start_col = start
    end_row, end_col = end
    row_delta = end_row - start_row
    col_delta = end_col - start_col
    steps = max(abs(row_delta), 1)

    for step in range(1, steps):
        row = start_row + round(row_delta * step / steps)
        col = start_col + round(col_delta * step / steps)
        if col_delta == 0:
            char = "|"
        elif col_delta > 0:
            char = "/"
        else:
            char = "\\"
        _put_edge_char(grid, row, col, char)


def _put_edge_char(grid: list[list[str]], row: int, col: int, char: str) -> None:
    if row < 0 or row >= len(grid):
        return
    if col < 0 or col >= len(grid[row]):
        return

    current = grid[row][col]
    if current == " " or current == char:
        grid[row][col] = char
    else:
        grid[row][col] = "+"


def _format_grid(grid: list[list[str]], *, max_floor: int, max_lane: int) -> str:
    lines: list[str] = []
    for row_index, row in enumerate(grid):
        if row_index % FLOOR_GAP == 0:
            floor = max_floor - row_index // FLOOR_GAP
            label = f"{floor:>2} "
        else:
            label = "   "
        lines.append(label + "".join(row).rstrip())

    lane_chars = [" " for _ in range(max_lane * LANE_GAP + 1)]
    for lane in range(max_lane + 1):
        lane_chars[lane * LANE_GAP] = str(lane % 10)
    lines.append("   " + "".join(lane_chars).rstrip())
    return "\n".join(lines)


def _header(*, seed: int | None, act: int, character_id: str | None = None) -> str:
    seed_text = "random" if seed is None else str(seed)
    header = [f"Seed: {seed_text}", f"Act: {act}"]
    if character_id:
        header.append(f"Character: {_normalized_character_id(character_id)}")
    header.append(f"Legend: {LEGEND}")
    return "\n".join(header)


def _normalized_character_id(value: object) -> str:
    character_id = str(value or "MAP_PREVIEW").strip().upper()
    return character_id or "MAP_PREVIEW"
