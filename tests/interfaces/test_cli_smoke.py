from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner

from sts2sim.cli.app import app


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    if name in {"sts2sim.content", "sts2sim.data", "sts2sim.api"}:
        cast(Any, module).__path__ = []
    return module


@pytest.fixture()
def fake_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    data_pkg = _module("sts2sim.data")
    data_sync = _module("sts2sim.data.sync")
    data_coverage = _module("sts2sim.data.coverage")
    content_pkg = _module("sts2sim.content")
    card_coverage = _module("sts2sim.content.card_coverage")
    event_coverage = _module("sts2sim.content.event_coverage")
    combat_coverage = _module("sts2sim.content.combat_coverage")
    api = _module("sts2sim.api")

    def sync_data(**kwargs: Any) -> dict[str, Any]:
        return {
            "command": "sync-data",
            "data_dir": str(kwargs["data_dir"]),
            "synced": 2,
        }

    def audit_coverage(**_: Any) -> dict[str, Any]:
        return {"command": "audit-coverage", "coverage": 1.0, "missing": []}

    def audit_event_coverage(**_: Any) -> dict[str, Any]:
        return {
            "entries": [
                {"event_id": "A", "category": "primitive"},
                {"event_id": "B", "category": "unsupported/bespoke"},
            ],
            "optional_module_errors": [],
        }

    def audit_combat_coverage(**_: Any) -> dict[str, Any]:
        return {
            "entries": [
                {"content_id": "Strike", "category": "cards", "status": "implemented"},
                {"content_id": "Boss", "category": "monsters", "status": "blocked"},
                {"content_id": "Relic X", "category": "relics", "status": "unknown"},
            ],
            "counts_by_category": {},
            "sample_unknown_ids": {},
            "total_ids": 3,
        }

    def audit_card_coverage(**_: Any) -> dict[str, Any]:
        return {
            "entries": [
                {
                    "content_id": "Strike",
                    "color": "ironclad",
                    "card_type": "attack",
                    "status": "implemented",
                },
                {
                    "content_id": "Osty Strike",
                    "color": "necrobinder",
                    "card_type": "attack",
                    "status": "partial",
                },
                {
                    "content_id": "Mystery",
                    "color": "silent",
                    "card_type": "skill",
                    "status": "missing",
                },
            ],
            "counts_by_status": {},
            "sample_missing_ids": {},
            "sample_partial_ids": {},
            "total_cards": 3,
        }

    def play_run(**kwargs: Any) -> dict[str, Any]:
        return {
            "command": "play-run",
            "seed": kwargs["seed"],
            "transcript": [{"step": 0, "action": "attack"}],
        }

    def replay(**kwargs: Any) -> dict[str, Any]:
        return {
            "command": "replay",
            "path": str(kwargs["replay_path"]),
            "matched": True,
        }

    def fuzz_run(**kwargs: Any) -> dict[str, Any]:
        seeds = kwargs.get("seeds") or list(
            range(kwargs["start_seed"], kwargs["start_seed"] + kwargs["count"])
        )
        return {"command": "fuzz-run", "seeds": seeds, "failures": []}

    cast(Any, data_sync).sync_data = sync_data
    cast(Any, data_coverage).audit_coverage = audit_coverage
    cast(Any, card_coverage).audit_card_coverage = audit_card_coverage
    cast(Any, event_coverage).audit_event_coverage = audit_event_coverage
    cast(Any, combat_coverage).audit_combat_coverage = audit_combat_coverage
    cast(Any, api).play_run = play_run
    cast(Any, api).replay = replay
    cast(Any, api).fuzz_run = fuzz_run

    cast(Any, data_pkg).sync = data_sync
    cast(Any, data_pkg).coverage = data_coverage
    cast(Any, content_pkg).card_coverage = card_coverage
    cast(Any, content_pkg).event_coverage = event_coverage
    cast(Any, content_pkg).combat_coverage = combat_coverage

    monkeypatch.setitem(sys.modules, "sts2sim.data", data_pkg)
    monkeypatch.setitem(sys.modules, "sts2sim.data.sync", data_sync)
    monkeypatch.setitem(sys.modules, "sts2sim.data.coverage", data_coverage)
    monkeypatch.setitem(sys.modules, "sts2sim.content", content_pkg)
    monkeypatch.setitem(sys.modules, "sts2sim.content.card_coverage", card_coverage)
    monkeypatch.setitem(sys.modules, "sts2sim.content.event_coverage", event_coverage)
    monkeypatch.setitem(sys.modules, "sts2sim.content.combat_coverage", combat_coverage)
    monkeypatch.setitem(sys.modules, "sts2sim.api", api)


def _json_output(result: Any) -> dict[str, Any]:
    assert result.exit_code == 0, result.output
    return cast(dict[str, Any], json.loads(result.output))


def test_cli_commands_smoke(fake_backends: None, tmp_path: Path) -> None:
    runner = CliRunner()
    replay_path = tmp_path / "run.json"

    sync_result = runner.invoke(
        app,
        ["sync-data", "--data-dir", str(tmp_path / "data")],
    )
    assert _json_output(sync_result)["command"] == "sync-data"

    audit_result = runner.invoke(
        app,
        ["audit-coverage", "--data-dir", str(tmp_path / "data"), "--fail-under", "0.9"],
    )
    assert _json_output(audit_result)["coverage"] == 1.0

    event_audit_result = runner.invoke(
        app,
        ["audit-events", "--category", "unsupported", "--summary-only"],
    )
    event_audit_payload = _json_output(event_audit_result)
    assert event_audit_payload["total_events"] == 1
    assert event_audit_payload["counts_by_category"] == {"unsupported/bespoke": 1}
    assert "entries" not in event_audit_payload

    combat_audit_result = runner.invoke(
        app,
        ["audit-combat", "--category", "relic", "--status", "unknown", "--summary-only"],
    )
    combat_audit_payload = _json_output(combat_audit_result)
    assert combat_audit_payload["total_ids"] == 1
    assert combat_audit_payload["counts_by_category"] == {
        "relics": {"total": 1, "implemented": 0, "blocked": 0, "unknown": 1}
    }
    assert "entries" not in combat_audit_payload

    card_audit_result = runner.invoke(
        app,
        ["audit-cards", "--status", "partial", "--type", "attack", "--summary-only"],
    )
    card_audit_payload = _json_output(card_audit_result)
    assert card_audit_payload["total_cards"] == 1
    assert card_audit_payload["counts_by_status"] == {
        "implemented": 0,
        "partial": 1,
        "missing": 0,
    }
    assert "entries" not in card_audit_payload

    play_result = runner.invoke(
        app,
        [
            "play-run",
            "--seed",
            "17",
            "--max-steps",
            "1",
            "--output",
            str(replay_path),
        ],
    )
    play_payload = _json_output(play_result)
    assert play_payload["seed"] == 17
    assert replay_path.exists()

    replay_result = runner.invoke(app, ["replay", str(replay_path)])
    assert _json_output(replay_result)["matched"] is True

    fuzz_result = runner.invoke(
        app,
        ["fuzz-run", "--count", "3", "--start-seed", "10"],
    )
    assert _json_output(fuzz_result)["seeds"] == [10, 11, 12]
