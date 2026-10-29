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


def _patch_upgrade_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source, registry = _source_fixture(tmp_path)
    old_patch = tmp_path / "reviewed.patch"
    prepare_live_ppo_bridge(source_dir=source, registry_path=registry, clone=False)
    (source / "state.cs").write_text("new observation\n")
    new_patch = tmp_path / "new.patch"
    new_patch.write_text(_git(source, "diff", "--binary", "--unified=1", "HEAD") + "\n")
    (source / "state.cs").write_text("after\n")
    _edit_registry(
        registry,
        ppo_patch=str(new_patch),
        ppo_patch_sha256=hashlib.sha256(new_patch.read_bytes()).hexdigest(),
        previous_ppo_patches=[{
            "path": str(old_patch),
            "sha256": hashlib.sha256(old_patch.read_bytes()).hexdigest(),
        }],
        observation_schema_version=2,
    )
    return source, registry, new_patch


def test_preparation_upgrades_only_an_exact_reviewed_previous_patch(tmp_path: Path) -> None:
    source, registry, _ = _patch_upgrade_fixture(tmp_path)
    old_sha = hashlib.sha256((tmp_path / "reviewed.patch").read_bytes()).hexdigest()
    upgraded = prepare_live_ppo_bridge(source_dir=source, registry_path=registry, clone=False)
    assert (source / "state.cs").read_text() == "new observation\n"
    assert upgraded["upgraded_from_patch"] == old_sha
    assert upgraded["observation_schema_version"] == 2
    assert upgraded["already_applied"] is False
    assert upgraded["built"] is False
    repeated = prepare_live_ppo_bridge(source_dir=source, registry_path=registry, clone=False)
    assert repeated["already_applied"] is True
    assert repeated["upgraded_from_patch"] is None


def test_failed_patch_upgrade_restores_the_previous_source(tmp_path: Path) -> None:
    source, registry, new_patch = _patch_upgrade_fixture(tmp_path)
    new_patch.write_text(new_patch.read_text().replace("-before", "-wrong baseline"))
    _edit_registry(registry, ppo_patch_sha256=hashlib.sha256(new_patch.read_bytes()).hexdigest())
    with pytest.raises(BridgeSetupError, match="git operation failed"):
        prepare_live_ppo_bridge(source_dir=source, registry_path=registry, clone=False)
    assert (source / "state.cs").read_text() == "after\n"


def test_patch_upgrade_refuses_modified_previous_patch_and_user_edits(tmp_path: Path) -> None:
    source, registry, _ = _patch_upgrade_fixture(tmp_path)
    (source / "state.cs").write_text("user adjustment\n")
    with pytest.raises(BridgeSetupError, match="unrelated changes"):
        prepare_live_ppo_bridge(source_dir=source, registry_path=registry, clone=False)
    assert (source / "state.cs").read_text() == "user adjustment\n"
    (source / "state.cs").write_text("after\n")
    (tmp_path / "reviewed.patch").write_text("tampered\n")
    with pytest.raises(BridgeSetupError, match="Previous observation patch SHA-256"):
        prepare_live_ppo_bridge(source_dir=source, registry_path=registry, clone=False)
    assert (source / "state.cs").read_text() == "after\n"


def test_readonly_probe_checks_health_before_touching_an_unpatched_game(monkeypatch) -> None:
    from sts2sim.live_capture import probe_live_bridges

    calls = []
    class LegacyClient:
        def __init__(self, config):
            pass

        def health(self):
            calls.append("health")
            return {"status": "ok", "message": "Hello from STS2 MCP"}

        def state(self):
            raise AssertionError("Unsafe legacy GET must not be requested")

        def actions(self, state):
            raise AssertionError("Unsafe legacy actions GET must not be requested")

        def close(self):
            calls.append("close")

    monkeypatch.setattr("sts2sim.live_capture.LiveApiClient", LegacyClient)
    result = probe_live_bridges(base_urls=("http://bridge",), require_readonly_observer=True)
    assert result[0].health_ok
    assert not result[0].state_ok
    assert "Read-only observation schema2" in result[0].errors[0]
    assert calls == ["health", "close"]


@pytest.mark.parametrize("has_path", [True, False])
def test_doctor_checks_visible_noncombat_policy_controls_without_submitting_actions(
    tmp_path: Path, monkeypatch, has_path: bool,
) -> None:
    registry = _registry(tmp_path)
    monkeypatch.setattr(
        "sts2sim.live_bridges.probe_live_bridges",
        lambda **kwargs: (
            LiveBridgeProbe(base_url="http://bridge", available=True, state_ok=True),
        ),
    )
    monkeypatch.setattr("sts2sim.live_policy_views._card_library", lambda _: {})
    path = {"index": 0, "col": 1, "row": 1, "type": "Monster"}
    raw = {
        "observation_schema_version": 2,
        "state_type": "map",
        "run": {"act": 1, "act_floor": 0, "total_floor": 1, "ascension": 0},
        "player": {"character_id": "IRONCLAD", "hp": 80, "max_hp": 80, "gold": 99,
                   "max_potion_slots": 3, "master_deck": [], "relics": [], "potions": [],
                   "status": [], "can_remove_potions": True},
        "map": {"nodes": [path], "next_options": [path] if has_path else [], "visited": []},
    }
    calls = []
    class Client:
        def __init__(self, config):
            pass

        def state(self):
            calls.append("state")
            return raw

        def close(self):
            calls.append("close")

        def apply_action(self, action):
            raise AssertionError("Doctor must not submit game actions")

    monkeypatch.setattr("sts2sim.live_bridges.LiveApiClient", Client)
    result = doctor_live_bridge(registry_path=registry)
    assert result["ready_for_capture"] is True
    assert result["ready_for_checkpoint_control"] is has_path
    assert result["game_compatibility_verified"] is False
    if has_path:
        assert result["checkpoint_view"] == "visible_decision"
        assert result["checkpoint_action_count"] == 1
    else:
        assert "unsupported" in result["diagnostic"].lower()
    assert calls == ["state", "close"]
