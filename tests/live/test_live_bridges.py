from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from sts2sim.cli.app import app
from sts2sim.live_bridges import (
    BridgeSetupError,
    doctor_live_bridge,
    prepare_live_ppo_bridge,
    sync_live_bridges,
)
from sts2sim.live_capture import LiveBridgeProbe


def _registry(tmp_path: Path, files: dict[str, bytes] | None = None) -> Path:
    assets = files or {"STS2_MCP.dll": b"test-only-not-a-game-mod", "STS2_MCP.json": b"{}"}
    source = "https://github.com/Gennadiyev/STS2MCP"
    entry = {
        "name": "STS2MCP",
        "source": source,
        "commit": "a" * 40,
        "default_base_url": "http://localhost:15526",
        "release": {
            "tag": "test",
            "url": source + "/releases/tag/test",
            "assets": [
                {
                    "name": name,
                    "url": source + "/releases/download/test/" + name,
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                for name, data in assets.items()
            ],
        },
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"bridges": [entry]}))
    return path


def _edit_registry(path: Path, **updates: Any) -> None:
    data = json.loads(path.read_text())
    data["bridges"][0].update(updates)
    path.write_text(json.dumps(data))


def test_download_is_verified_reusable_and_not_installed(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    files = {"STS2_MCP.dll": b"test-only-not-a-game-mod", "STS2_MCP.json": b"{}"}
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=files[request.url.path.rsplit("/", 1)[-1]])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = sync_live_bridges(
            registry_path=registry, destination=tmp_path / "downloads", client=client
        )
        repeated = sync_live_bridges(
            registry_path=registry, destination=tmp_path / "downloads", client=client
        )
    assert len(requests) == 2
    assert report["game_included"] is False
    assert report["bridges"][0]["installed"] is False
    assert repeated["bridges"][0]["reused"] is True
    receipt = tmp_path / "downloads/STS2MCP/test/receipt.json"
    assert json.loads(receipt.read_text())["installed"] is False
    doctor = doctor_live_bridge(
        registry_path=registry, destination=tmp_path / "downloads", offline=True
    )
    assert doctor["download_ready"] is True
    assert doctor["runtime_checked"] is False
    assert doctor["ready_for_checkpoint_control"] is False


@pytest.mark.parametrize("bad", [b"wrong-checksum-same-size", b"too long" * 100, b"short"])
def test_failed_download_preserves_previous_release(tmp_path: Path, bad: bytes) -> None:
    registry = _registry(tmp_path)
    target = tmp_path / "downloads/STS2MCP/test"
    target.mkdir(parents=True)
    (target / "keep").write_text("previous-package")
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=bad))
    ) as client, pytest.raises(BridgeSetupError, match="download|Download"):
        sync_live_bridges(
            registry_path=registry, destination=tmp_path / "downloads", client=client
        )
    assert (target / "keep").read_text() == "previous-package"
    assert not (target / "STS2_MCP.dll").exists()


def test_publish_failure_rolls_back_existing_release(tmp_path: Path, monkeypatch) -> None:
    registry = _registry(tmp_path)
    target = tmp_path / "downloads/STS2MCP/test"
    target.mkdir(parents=True)
    (target / "keep").write_text("old")
    original = Path.replace

    def fail_stage(path: Path, destination: Path) -> Path:
        if path.name == "release":
            raise OSError("fixture publication failed")
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", fail_stage)
    files = {"STS2_MCP.dll": b"test-only-not-a-game-mod", "STS2_MCP.json": b"{}"}
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=files[request.url.path.rsplit("/", 1)[-1]])
        )
    ) as client, pytest.raises(OSError, match="publication"):
        sync_live_bridges(
            registry_path=registry, destination=tmp_path / "downloads", client=client
        )
    assert (target / "keep").read_text() == "old"


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "..", "unsafe/name"])
def test_registry_cannot_traverse_download_directory(tmp_path: Path, name: str) -> None:
    registry = _registry(tmp_path)
    _edit_registry(registry, name=name)
    with pytest.raises(BridgeSetupError, match="Invalid"):
        sync_live_bridges(bridge=name, registry_path=registry, destination=tmp_path / "out")


def _git(directory: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(directory), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    (source / "state.cs").write_text("before\n")
    _git(source, "add", "state.cs")
    _git(
        source, "-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", "commit", "-m", "pin"
    )
    commit = _git(source, "rev-parse", "HEAD")
    (source / "state.cs").write_text("after\n")
    patch = tmp_path / "reviewed.patch"
    patch.write_text(_git(source, "diff", "--binary", "HEAD") + "\n")
    _git(source, "checkout", "--", "state.cs")
    registry = _registry(tmp_path)
    _edit_registry(
        registry,
        commit=commit,
        ppo_patch=str(patch),
        ppo_patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
    )
    return source, registry


def test_source_patch_checks_pin_and_applies_idempotently(tmp_path: Path) -> None:
    source, registry = _source_fixture(tmp_path)
    first = prepare_live_ppo_bridge(source_dir=source, registry_path=registry, clone=False)
    second = prepare_live_ppo_bridge(source_dir=source, registry_path=registry, clone=False)
    assert (source / "state.cs").read_text() == "after\n"
    assert first["patch_applied"] is True
    assert first["already_applied"] is False
    assert second["already_applied"] is True
    assert first["built"] is False
    assert first["game_tested"] is False


@pytest.mark.parametrize("change", ["wrong_pin", "dirty", "untracked", "bad_patch"])
def test_source_patch_refuses_unreviewed_content(tmp_path: Path, change: str) -> None:
    source, registry = _source_fixture(tmp_path)
    if change == "wrong_pin":
        _edit_registry(registry, commit="b" * 40)
    elif change == "dirty":
        (source / "state.cs").write_text("user change\n")
    elif change == "untracked":
        (source / "untracked").write_text("user content\n")
    else:
        (tmp_path / "reviewed.patch").write_text("tampered\n")
    with pytest.raises(BridgeSetupError):
        prepare_live_ppo_bridge(source_dir=source, registry_path=registry, clone=False)
    assert (source / "state.cs").read_text() != "after\n"


def test_doctor_disconnected_is_explicit_and_submits_no_actions(
    tmp_path: Path, monkeypatch
) -> None:
    registry = _registry(tmp_path)
    calls = []

    def probe(**kwargs):
        calls.append(kwargs)
        return (LiveBridgeProbe(base_url="http://localhost:15526", available=False),)

    monkeypatch.setattr("sts2sim.live_bridges.probe_live_bridges", probe)
    report = doctor_live_bridge(registry_path=registry)
    assert len(calls) == 1
    assert report["ready_for_capture"] is False
    assert report["ready_for_checkpoint_control"] is False
    assert "launch" in report["diagnostic"]


def test_doctor_reuses_probe_and_rejects_sparse_live_observation(
    tmp_path: Path, monkeypatch
) -> None:
    registry = _registry(tmp_path)
    monkeypatch.setattr(
        "sts2sim.live_bridges.probe_live_bridges",
        lambda **kwargs: (
            LiveBridgeProbe(base_url="http://bridge", available=True, state_ok=True),
        ),
    )

    class Client:
        def __init__(self, config):
            pass

        def state(self):
            return {"state_type": "monster", "player": {}}

        def close(self):
            pass

    monkeypatch.setattr("sts2sim.live_bridges.LiveApiClient", Client)
    report = doctor_live_bridge(registry_path=registry)
    assert report["ready_for_capture"] is True
    assert report["ready_for_checkpoint_control"] is False
    assert "unsupported" in report["diagnostic"].lower()


def test_setup_clis_expose_verified_backends(monkeypatch) -> None:
    calls = []

    def backend(**kwargs):
        calls.append(kwargs)
        return {"built": False, "game_tested": False}

    for function in ("sync_live_bridges", "prepare_live_ppo_bridge", "doctor_live_bridge"):
        monkeypatch.setattr(f"sts2sim.live_bridges.{function}", backend)
    runner = CliRunner()
    for command in (
        ["sync-live-bridges", "--bridge", "all"],
        ["prepare-live-ppo-bridge", "--no-clone"],
        ["doctor-live-bridge", "--offline"],
    ):
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["game_tested"] is False
    assert calls[0]["bridge"] == "all"
    assert calls[1]["clone"] is False
    assert calls[2]["offline"] is True
