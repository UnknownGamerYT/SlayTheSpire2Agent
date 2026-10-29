"""Pinned third-party bridge downloads, source preparation, and read-only diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from sts2sim.live_capture import LiveApiClient, LiveApiConfig, probe_live_bridges
from sts2sim.live_policy import LivePolicyUnsupported, import_policy_combat

DEFAULT_REGISTRY = Path("external/live_bridges/bridges.json")
DEFAULT_DOWNLOADS = Path("external/live_bridges/downloads")
DEFAULT_PPO_SOURCE = Path("external/live_bridges/repos/STS2MCP")
MAX_ASSET_BYTES = 16 * 1024 * 1024


class BridgeSetupError(ValueError):
    """A pinned bridge package or source tree could not be verified."""


def _registry(path: Path | str) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BridgeSetupError(f"Cannot read bridge registry {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("bridges"), list):
        raise BridgeSetupError("Bridge registry must contain a bridges list.")
    return data


def _bridge(name: str, registry: Path | str) -> dict[str, Any]:
    for candidate in _registry(registry)["bridges"]:
        if isinstance(candidate, dict) and str(candidate.get("name", "")).lower() == name.lower():
            return candidate
    raise BridgeSetupError(f"Unknown bridge {name!r}; choose STS2MCP or STS2-Agent.")


def _component(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise BridgeSetupError(f"Invalid pinned bridge {field}.")
    if value in {".", ".."}:
        raise BridgeSetupError(f"Invalid pinned bridge {field}.")
    return value


def _release_assets(bridge: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    release = bridge.get("release")
    if not isinstance(release, dict) or not isinstance(release.get("assets"), list):
        raise BridgeSetupError(f"No published download is pinned for {bridge.get('name')}.")
    tag = _component(release.get("tag"), "release tag")
    source = str(bridge.get("source", ""))
    parsed = urlparse(source)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or not re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", parsed.path)
    ):
        raise BridgeSetupError("Pinned bridge source must be an HTTPS GitHub repository.")
    assets = []
    for asset in release["assets"]:
        if not isinstance(asset, dict):
            raise BridgeSetupError("Invalid bridge asset entry.")
        filename = _component(asset.get("name"), "asset filename")
        if asset.get("url") != f"{source}/releases/download/{tag}/{filename}":
            raise BridgeSetupError("Bridge asset URL does not match its pinned repository/release.")
        if not re.fullmatch(r"[a-f0-9]{64}", str(asset.get("sha256", ""))):
            raise BridgeSetupError("Bridge asset needs a pinned SHA-256 checksum.")
        size = asset.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= MAX_ASSET_BYTES:
            raise BridgeSetupError("Bridge asset size exceeds the allowed download limit.")
        assets.append(asset)
    if not assets or len({asset["name"] for asset in assets}) != len(assets):
        raise BridgeSetupError("Bridge release needs distinct pinned assets.")
    return tag, assets


def _asset_check(path: Path, asset: Mapping[str, Any]) -> str | None:
    if path.is_symlink() or not path.is_file():
        return "missing or unsafe file"
    if path.stat().st_size != asset["size"]:
        return "size mismatch"
    if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
        return "SHA-256 mismatch"
    return None


def sync_live_bridges(
    *,
    bridge: str = "STS2MCP",
    destination: Path | str = DEFAULT_DOWNLOADS,
    registry_path: Path | str = DEFAULT_REGISTRY,
    timeout_seconds: float = 30.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Stage checksum-pinned release files; never execute or install downloaded code."""
    names = ("STS2MCP", "STS2-Agent") if bridge.lower() == "all" else (bridge,)
    results = []
    active = client or httpx.Client(timeout=timeout_seconds, follow_redirects=True)
    try:
        for name in names:
            entry = _bridge(name, registry_path)
            tag, assets = _release_assets(entry)
            target = Path(destination) / _component(entry["name"], "name") / tag
            if target.is_symlink():
                raise BridgeSetupError(f"Refusing symlinked download destination: {target}")
            reused = target.is_dir() and all(
                _asset_check(target / asset["name"], asset) is None for asset in assets
            )
            if not reused:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix=".bridge-", dir=target.parent) as staging:
                    stage = Path(staging) / "release"
                    stage.mkdir()
                    for asset in assets:
                        output = stage / asset["name"]
                        total = 0
                        try:
                            with active.stream(
                                "GET", asset["url"], follow_redirects=True
                            ) as response:
                                response.raise_for_status()
                                with output.open("wb") as handle:
                                    for chunk in response.iter_bytes():
                                        total += len(chunk)
                                        if total > asset["size"]:
                                            raise BridgeSetupError(
                                                f"Oversized bridge download: {asset['name']}"
                                            )
                                        handle.write(chunk)
                        except httpx.HTTPError as exc:
                            raise BridgeSetupError(f"Bridge download failed: {exc}") from exc
                        if reason := _asset_check(output, asset):
                            raise BridgeSetupError(f"Invalid download {asset['name']}: {reason}")
                    receipt = {
                        "bridge": entry["name"],
                        "release": entry["release"],
                        "reviewed_source_commit": entry.get("commit"),
                        "installed": False,
                    }
                    (stage / "receipt.json").write_text(
                        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
                    )
                    backup = Path(staging) / "previous"
                    if target.exists():
                        target.replace(backup)
                    try:
                        stage.replace(target)
                    except OSError:
                        if backup.exists():
                            backup.replace(target)
                        raise
            results.append(
                {
                    "bridge": entry["name"],
                    "release_tag": tag,
                    "directory": str(target),
                    "verified_files": [asset["name"] for asset in assets],
                    "reused": reused,
                    "installed": False,
                    "ppo_observation_ready": False,
                }
            )
    finally:
        if client is None:
            active.close()
    return {"bridges": results, "game_included": False}


def _git(directory: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise BridgeSetupError(f"Bridge source git operation failed: {detail}") from exc
    return result.stdout.strip()


def prepare_live_ppo_bridge(
    *,
    source_dir: Path | str = DEFAULT_PPO_SOURCE,
    registry_path: Path | str = DEFAULT_REGISTRY,
    clone: bool = True,
) -> dict[str, Any]:
    """Check an exact source pin and apply its reviewed observation-only patch."""
    entry = _bridge("STS2MCP", registry_path)
    _release_assets(entry)  # Validate pinned repository URL before git can contact it.
    commit = str(entry.get("commit", ""))
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise BridgeSetupError("Bridge source needs an exact 40-character Git commit pin.")
    patch = Path(str(entry.get("ppo_patch", "")))
    if not patch.is_file():
        raise BridgeSetupError(f"Missing reviewed bridge patch: {patch}")
    if hashlib.sha256(patch.read_bytes()).hexdigest() != entry.get("ppo_patch_sha256"):
        raise BridgeSetupError("Reviewed bridge patch SHA-256 does not match the registry.")
    directory = Path(source_dir)
    if directory.is_symlink():
        raise BridgeSetupError("Refusing a symlinked bridge source directory.")
    if not directory.exists():
        if not clone:
            raise BridgeSetupError("Pinned bridge source is missing; allow cloning or provide it.")
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".source-", dir=directory.parent) as staging:
            stage = Path(staging)
            _git(stage, "init")
            _git(stage, "fetch", "--depth", "1", str(entry["source"]), commit)
            _git(stage, "checkout", "--detach", "FETCH_HEAD")
            if _git(stage, "rev-parse", "HEAD") != commit:
                raise BridgeSetupError("Downloaded source did not resolve to the pinned commit.")
            stage.replace(directory)
    if _git(directory, "rev-parse", "HEAD") != commit:
        raise BridgeSetupError(
            "Bridge source HEAD differs from the reviewed pin; refusing to patch."
        )
    patch_argument = str(patch.resolve())
    changed = _git(directory, "status", "--porcelain", "--untracked-files=all")
    already_applied = False
    upgraded_from_patch: str | None = None
    if changed:
        # Idempotency requires the *entire* tracked diff to equal this patch.
        expected = subprocess.run(
            ["git", "-C", str(directory), "diff", "--binary", "--unified=1", "HEAD"],
            check=True,
            capture_output=True,
        ).stdout
        if any(line.startswith("??") for line in changed.splitlines()):
            raise BridgeSetupError(
                "Bridge source has unrelated changes; refusing to overwrite them."
            )
        if expected == patch.read_bytes():
            _git(directory, "apply", "--reverse", "--check", patch_argument)
            already_applied = True
        else:
            previous = _previous_observation_patch(entry, expected)
            if previous is None:
                raise BridgeSetupError(
                    "Bridge source has unrelated changes; refusing to overwrite them."
                )
            previous_argument = str(previous.resolve())
            _git(directory, "apply", "--reverse", "--check", previous_argument)
            _git(directory, "apply", "--reverse", previous_argument)
            try:
                _git(directory, "apply", "--check", patch_argument)
                _git(directory, "apply", patch_argument)
            except BridgeSetupError:
                _git(directory, "apply", previous_argument)
                raise
            upgraded_from_patch = hashlib.sha256(previous.read_bytes()).hexdigest()
    else:
        _git(directory, "apply", "--check", patch_argument)
        _git(directory, "apply", patch_argument)
    return {
        "bridge": "STS2MCP",
        "source_directory": str(directory),
        "source_commit": commit,
        "patch": str(patch),
        "patch_sha256": entry["ppo_patch_sha256"],
        "already_applied": already_applied,
        "upgraded_from_patch": upgraded_from_patch,
        "observation_schema_version": entry.get("observation_schema_version", 1),
        "patch_applied": True,
        "built": False,
        "game_tested": False,
        "required_sdk": ".NET 9",
        "required_game_assemblies": ["sts2.dll", "GodotSharp.dll", "0Harmony.dll"],
    }


def _previous_observation_patch(entry: Mapping[str, Any], current_diff: bytes) -> Path | None:
    """Recognize an exact older reviewed patch without replacing user modifications."""
    candidates = entry.get("previous_ppo_patches", ())
    if not isinstance(candidates, list | tuple):
        raise BridgeSetupError("Previous observation patches must be a list.")
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise BridgeSetupError("Invalid previous observation patch entry.")
        path = Path(str(candidate.get("path", "")))
        if not path.is_file():
            raise BridgeSetupError(f"Missing previous reviewed observation patch: {path}")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != candidate.get("sha256"):
            raise BridgeSetupError("Previous observation patch SHA-256 differs from the registry.")
        if content == current_diff:
            return path
    return None


def doctor_live_bridge(
    *,
    bridge: str = "STS2MCP",
    destination: Path | str = DEFAULT_DOWNLOADS,
    registry_path: Path | str = DEFAULT_REGISTRY,
    base_url: str | None = None,
    offline: bool = False,
    timeout_seconds: float = 1.0,
) -> dict[str, Any]:
    """Inspect package integrity and reachable protocol without submitting game actions."""
    entry = _bridge(bridge, registry_path)
    tag, assets = _release_assets(entry)
    directory = Path(destination) / entry["name"] / tag
    checks = {
        asset["name"]: _asset_check(directory / asset["name"], asset) or "verified"
        for asset in assets
    }
    report: dict[str, Any] = {
        "bridge": entry["name"],
        "release_tag": tag,
        "release_url": entry["release"]["url"],
        "reviewed_source_commit": entry.get("commit"),
        "release_files": checks,
        "download_ready": all(value == "verified" for value in checks.values()),
        "game_included": False,
        "dotnet_available": shutil.which("dotnet") is not None,
        "git_available": shutil.which("git") is not None,
        "runtime_checked": not offline,
        "ready_for_capture": False,
        "ready_for_checkpoint_control": False,
        "game_compatibility_verified": False,
        "notes": entry.get("notes", ""),
    }
    if offline:
        report["diagnostic"] = "Runtime was not probed (--offline)."
        return report
    url = base_url or entry["default_base_url"]
    probes = probe_live_bridges(base_urls=(url,), timeout_seconds=timeout_seconds)
    report["runtime"] = [probe.model_dump(mode="json") for probe in probes]
    report["ready_for_capture"] = any(probe.state_ok for probe in probes)
    if not report["ready_for_capture"]:
        report["diagnostic"] = "No bridge game-state endpoint is reachable; launch the modded game."
        return report
    if entry["name"] != "STS2MCP":
        report["diagnostic"] = (
            "State capture only: this bridge's checkpoint action protocol is unsupported."
        )
        return report
    client = LiveApiClient(
        LiveApiConfig(
            base_url=url, state_path="/api/v1/singleplayer", timeout_seconds=timeout_seconds
        )
    )
    try:
        raw = client.state()
        import_policy_combat(raw)
    except (RuntimeError, ValueError, OSError, LivePolicyUnsupported) as exc:
        report["diagnostic"] = f"Checkpoint observation unsupported: {exc}"
    else:
        report["ready_for_checkpoint_control"] = True
        report["diagnostic"] = (
            "Current combat observation is mappable; "
            "model actions are limited to play_card/end_turn."
        )
    finally:
        client.close()
    return report
