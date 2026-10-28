from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sts2sim.cli.app import app
from sts2sim.data.sync import sync_all
from sts2sim.learning.readiness import check_training_readiness


@pytest.fixture
def sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr("sts2sim.learning.readiness.importlib.util.find_spec", lambda _: True)
    cache_dir = tmp_path / "cache"
    manifest = tmp_path / "manifest.json"
    sync_all(cache_dir, manifest_path=manifest, fetcher=lambda _: [{"id": "test"}])
    return cache_dir, manifest


def test_valid_sources_pass_without_network(sources: tuple[Path, Path]) -> None:
    cache_dir, manifest = sources
    result = check_training_readiness(cache_dir=cache_dir, manifest_path=manifest)
    assert result["ready"] is True
    assert len(result["datasets"]) == 12
    assert all(item["ok"] for item in result["datasets"])


def test_missing_and_changed_cache_fail(sources: tuple[Path, Path]) -> None:
    cache_dir, manifest = sources
    (cache_dir / "eng/cards.json").write_text('[{"id":"changed"}]', encoding="utf-8")
    (cache_dir / "eng/relics.json").unlink()
    result = check_training_readiness(cache_dir=cache_dir, manifest_path=manifest)
    assert result["ready"] is False
    failed = {item["dataset"] for item in result["datasets"] if not item["ok"]}
    assert failed == {"eng/cards", "eng/relics"}


@pytest.mark.parametrize("payload", [None, "{}", "not JSON", "null"])
def test_missing_or_malformed_manifest_fails(tmp_path: Path, payload: str | None) -> None:
    manifest = tmp_path / "manifest.json"
    if payload is not None:
        manifest.write_text(payload, encoding="utf-8")
    result = check_training_readiness(cache_dir=tmp_path, manifest_path=manifest)
    assert result["ready"] is False
    assert result["manifest_error"]
    assert len(result["datasets"]) == 12


def test_absent_dependency_fails(
    sources: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir, manifest = sources
    monkeypatch.setattr(
        "sts2sim.learning.readiness.importlib.util.find_spec",
        lambda name: None if name == "torch" else True,
    )
    result = check_training_readiness(cache_dir=cache_dir, manifest_path=manifest)
    assert result["ready"] is False
    assert "uv sync --extra rl" in result["recommendations"][0]


def test_training_check_cli_exit_status(sources: tuple[Path, Path]) -> None:
    cache_dir, manifest = sources
    args = ["training-check", "--cache-dir", str(cache_dir), "--manifest", str(manifest)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["ready"] is True
    (cache_dir / "eng/monsters.json").unlink()
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert json.loads(result.output)["ready"] is False
