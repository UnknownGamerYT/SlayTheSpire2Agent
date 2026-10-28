from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sts2sim.cli.app import app
from sts2sim.data.sync import DEFAULT_ENDPOINTS, sync_all
from sts2sim.learning.provenance import source_data_snapshot
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
    assert len(result["datasets"]) == len(DEFAULT_ENDPOINTS)
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
    assert len(result["datasets"]) == len(DEFAULT_ENDPOINTS)


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


def test_readiness_reports_provenance_without_inferring_version(sources: tuple[Path, Path]) -> None:
    cache_dir, manifest = sources
    result = check_training_readiness(cache_dir=cache_dir, manifest_path=manifest)
    assert result["channel"] == "stable"
    assert result["manifest_channels"] == ["stable"]
    assert result["snapshot_integrity"] is True
    assert result["stable_training_compatible"] is True
    assert result["game_version"] is None
    assert result["game_version_status"] == "unverified"
    assert all(item["channel"] == "stable" for item in result["datasets"])


def test_beta_snapshot_is_auditable_but_not_ready_for_training(
    sources: tuple[Path, Path],
) -> None:
    cache_dir, _ = sources
    sync_all(
        cache_dir,
        channel="beta",
        game_version="v0.111.0",
        fetcher=lambda _: [{"id": "test"}],
    )
    result = check_training_readiness(cache_dir=cache_dir, channel="beta")
    assert result["snapshot_integrity"] is True
    assert result["channel_consistent"] is True
    assert result["stable_training_compatible"] is False
    assert result["ready"] is False
    assert result["game_version"] == "v0.111.0"
    assert result["manifest_path"] == str(cache_dir / "beta/manifest.json")
    cli = CliRunner().invoke(
        app, ["training-check", "--cache-dir", str(cache_dir), "--channel", "beta"]
    )
    assert cli.exit_code == 1
    assert json.loads(cli.output)["snapshot_integrity"] is True


@pytest.mark.parametrize("disguise", ["explicit-channel", "source-url"])
def test_mixed_channel_manifest_fails_even_if_checksums_pass(
    sources: tuple[Path, Path],
    disguise: str,
) -> None:
    cache_dir, manifest = sources
    payload = json.loads(manifest.read_text())
    if disguise == "explicit-channel":
        payload["sources"]["eng/cards"]["channel"] = "beta"
    else:
        payload["sources"]["eng/cards"]["source_url"] += "&channel=beta"
    manifest.write_text(json.dumps(payload))
    result = check_training_readiness(cache_dir=cache_dir, manifest_path=manifest)
    assert result["snapshot_integrity"] is True
    assert result["channel_consistent"] is False
    assert result["stable_training_compatible"] is False
    assert result["ready"] is False


def test_default_manifest_follows_custom_cache_root(sources: tuple[Path, Path]) -> None:
    cache_dir, manifest = sources
    result = check_training_readiness(cache_dir=cache_dir)
    assert result["ready"] is True
    assert result["manifest_path"] == str(manifest)


def test_recorded_versions_must_agree(sources: tuple[Path, Path]) -> None:
    cache_dir, manifest = sources
    payload = json.loads(manifest.read_text())
    for source in payload["sources"].values():
        source["game_version"] = "verified-build"
    manifest.write_text(json.dumps(payload))
    result = check_training_readiness(cache_dir=cache_dir, manifest_path=manifest)
    assert result["game_version"] == "verified-build"
    assert result["game_version_status"] == "recorded"
    payload["sources"]["eng/cards"]["game_version"] = "other-build"
    manifest.write_text(json.dumps(payload))
    result = check_training_readiness(cache_dir=cache_dir, manifest_path=manifest)
    assert result["game_version"] is None
    assert result["game_versions"] == ["other-build", "verified-build"]
    assert result["ready"] is False
    assert result["version_consistent"] is False
    assert result["game_version_status"] == "mixed"


def test_invalid_readiness_channel_cli_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="channel"):
        check_training_readiness(cache_dir=tmp_path, channel="../beta")
    result = CliRunner().invoke(app, ["training-check", "--channel", "../beta"])
    assert result.exit_code == 2
    assert "Unsupported source channel" in result.output


@pytest.mark.parametrize("mostly_verified", [False, True])
def test_partial_version_provenance_matches_training_rejection(
    sources: tuple[Path, Path], mostly_verified: bool
) -> None:
    cache_dir, manifest = sources
    payload = json.loads(manifest.read_text())
    for source in payload["sources"].values():
        source["game_version"] = "verified-build" if mostly_verified else None
    payload["sources"]["eng/cards"]["game_version"] = (
        None if mostly_verified else "verified-build"
    )
    manifest.write_text(json.dumps(payload))
    result = check_training_readiness(cache_dir=cache_dir, manifest_path=manifest)
    assert result["snapshot_integrity"] is True
    assert result["game_versions"] == ["verified-build"]
    assert result["game_version"] is None
    assert result["version_consistent"] is False
    assert result["game_version_status"] == "mixed"
    assert result["stable_training_compatible"] is False
    assert result["ready"] is False
    assert source_data_snapshot(cache_dir=cache_dir).consistent is False
    cli = CliRunner().invoke(
        app, ["training-check", "--cache-dir", str(cache_dir), "--manifest", str(manifest)]
    )
    assert cli.exit_code == 1
    assert json.loads(cli.output)["ready"] is False
