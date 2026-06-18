from __future__ import annotations

from pathlib import Path

from sts2sim.map_visualizer import render_map_text, write_map_file


def test_map_visualizer_writes_text_file(tmp_path: Path) -> None:
    output = tmp_path / "map.txt"

    result = write_map_file(
        123,
        output_path=output,
        map_width=3,
        map_floors=6,
        map_paths=4,
    )

    assert result.output_path == output
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "Seed: 123" in text
    assert "Character: MAP_PREVIEW" in text
    assert "Legend:" in text
    assert "A" in text
    assert "B" in text
    assert "|" in text
    assert "/" in text or "\\" in text


def test_map_visualizer_is_seed_deterministic() -> None:
    first = render_map_text(456, map_width=5, map_floors=8, map_paths=5)
    second = render_map_text(456, map_width=5, map_floors=8, map_paths=5)

    assert first == second


def test_map_visualizer_records_character_context() -> None:
    text = render_map_text(
        789,
        character_id="defect",
        map_width=3,
        map_floors=6,
        map_paths=4,
    )

    assert "Seed: 789" in text
    assert "Character: DEFECT" in text
