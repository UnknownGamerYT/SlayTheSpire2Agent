from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from sts2sim import legal_actions, new_run, serialize, step
from sts2sim.cli.app import app
from sts2sim.parity import (
    ParityCompareConfig,
    compare_snapshots,
    compare_trace_file,
    trace_template,
)


def test_subset_snapshot_compares_only_captured_fields() -> None:
    mismatches = compare_snapshots(
        {"player": {"hp": 80}},
        {"player": {"hp": 80, "max_hp": 80}, "phase": "ancient"},
    )

    assert mismatches == ()


def test_exact_snapshot_reports_extra_fields() -> None:
    mismatches = compare_snapshots(
        {"player": {"hp": 80}},
        {"player": {"hp": 80, "max_hp": 80}},
        ParityCompareConfig(mode="exact"),
    )

    assert len(mismatches) == 1
    assert mismatches[0].kind == "extra"
    assert mismatches[0].path == "player.max_hp"


def test_snapshot_value_mismatch_reports_nested_path() -> None:
    mismatches = compare_snapshots(
        {"combat": {"monsters": [{"hp": 15}]}},
        {"combat": {"monsters": [{"hp": 12}]}},
    )

    assert len(mismatches) == 1
    assert mismatches[0].kind == "value"
    assert mismatches[0].path == "combat.monsters[0].hp"
    assert mismatches[0].expected == 15
    assert mismatches[0].actual == 12


def test_compare_trace_file_replays_real_simulator_action(tmp_path: Path) -> None:
    state = new_run(seed=17, character_id="IRONCLAD", ascension=0)
    before = serialize(state)
    action = legal_actions(state)[0]
    next_state = step(state, action)
    after = serialize(next_state)
    trace_path = tmp_path / "one-step.parity.json"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trace_id": "one-step",
                "seed": 17,
                "character_id": "IRONCLAD",
                "ascension": 0,
                "initial_state": {
                    "phase": before["phase"],
                    "player": {"hp": before["player"]["hp"]},
                },
                "steps": [
                    {
                        "action": action.model_dump(mode="json", exclude_none=True),
                        "before": {"phase": before["phase"]},
                        "after": {"phase": after["phase"]},
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = compare_trace_file(trace_path)

    assert report.matched is True
    assert report.mismatch_count == 0
    assert report.steps[0].matched is True


def test_compare_trace_file_reports_mismatch(tmp_path: Path) -> None:
    trace_path = tmp_path / "bad.parity.json"
    trace = trace_template()
    trace["initial_state"] = {"player": {"hp": -999}}
    trace["steps"] = []
    trace_path.write_text(json.dumps(trace), encoding="utf-8")

    report = compare_trace_file(trace_path)

    assert report.matched is False
    assert report.mismatch_count == 1
    assert report.initial_mismatches[0].path == "player.hp"


def test_compare_trace_cli_prints_report(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.parity.json"
    trace = trace_template()
    trace["initial_state"] = {"phase": "ancient"}
    trace["steps"] = []
    trace_path.write_text(json.dumps(trace), encoding="utf-8")

    result = CliRunner().invoke(app, ["compare-trace", str(trace_path)])
    payload = _json_output(result)

    assert payload["matched"] is True
    assert payload["trace_id"] == "example"


def test_trace_template_cli_can_write_file(tmp_path: Path) -> None:
    output_path = tmp_path / "template.json"

    result = CliRunner().invoke(app, ["trace-template", "--output", str(output_path)])
    payload = _json_output(result)

    assert output_path.exists()
    assert payload["schema_version"] == 1
    assert json.loads(output_path.read_text(encoding="utf-8"))["trace_id"] == "example"


def _json_output(result: Any) -> dict[str, Any]:
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, dict)
    return payload
