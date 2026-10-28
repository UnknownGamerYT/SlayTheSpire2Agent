"""Offline checks for the dependencies and source snapshots used in training."""

from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from sts2sim.data.sync import (
    DEFAULT_ENDPOINTS,
    channel_cache_dir,
    load_manifest,
    resolve_manifest_path,
    source_channel_matches,
)


def check_training_readiness(
    *,
    cache_dir: Path | str = Path("data/cache"),
    manifest_path: Path | str | None = None,
    channel: str = "stable",
) -> dict[str, Any]:
    """Check local inputs without fetching data or starting a training run.

    A passing check establishes installation and snapshot integrity, not parity
    with the live game. Registered mechanics may still be partial or blocked.
    """

    cache_root = Path(cache_dir)
    cache_dir = channel_cache_dir(cache_root, channel)
    manifest_path = resolve_manifest_path(manifest_path, cache_root, channel=channel)
    dependencies = []
    for name in ("torch", "numpy", "gymnasium"):
        available = importlib.util.find_spec(name) is not None
        try:
            installed_version = version(name) if available else None
        except PackageNotFoundError:
            installed_version = None
        dependencies.append({"name": name, "available": available, "version": installed_version})

    manifest_error = None
    try:
        manifests = load_manifest(manifest_path)
        if not manifests:
            manifest_error = "Source manifest is missing or empty."
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        manifests = {}
        manifest_error = f"Cannot read source manifest: {exc}"

    manifest_channels = sorted({source.channel for source in manifests.values()})
    channel_consistent = bool(manifests) and all(
        source_channel_matches(source, channel) for source in manifests.values()
    )
    game_versions = sorted(
        {source.game_version for source in manifests.values() if source.game_version is not None}
    )
    game_version = (
        game_versions[0]
        if len(game_versions) == 1
        and all(source.game_version is not None for source in manifests.values())
        else None
    )
    # A partly verified manifest is inconsistent with the exact provenance
    # contract used by training; all-unverified snapshots remain valid.
    version_consistent = len({source.game_version for source in manifests.values()}) <= 1

    datasets: list[dict[str, Any]] = []
    for endpoint in DEFAULT_ENDPOINTS:
        key = f"eng/{endpoint.key}"
        path = cache_dir / f"{key}.json"
        entry: dict[str, Any] = {"dataset": key, "path": str(path), "ok": False}
        source = manifests.get(key)
        if source is None:
            entry["error"] = "Dataset is absent from the source manifest."
        else:
            entry.update(
                source_url=source.source_url,
                fetched_at=source.fetched_at.isoformat(),
                expected_count=source.count,
                expected_sha256=source.sha256,
                channel=source.channel,
                game_version=source.game_version,
                channel_ok=source_channel_matches(source, channel),
            )
            try:
                payload_bytes = path.read_bytes()
                payload = json.loads(payload_bytes)
                if isinstance(payload, list):
                    count = len(payload)
                elif isinstance(payload, dict):
                    items = next(
                        (
                            payload[k]
                            for k in ("items", "data", "results")
                            if isinstance(payload.get(k), list)
                        ),
                        None,
                    )
                    count = len(items) if items is not None else len(payload)
                else:
                    raise ValueError("Expected a JSON list or object.")
                digest = sha256(payload_bytes).hexdigest()
                entry.update(actual_count=count, actual_sha256=digest)
                entry["ok"] = count == source.count and digest == source.sha256
                if not entry["ok"]:
                    entry["error"] = "Cached count or checksum differs from the manifest."
            except (OSError, ValueError) as exc:
                entry["error"] = f"Cannot validate cached source: {exc}"
        datasets.append(entry)

    recommendations = []
    if not all(item["available"] for item in dependencies):
        recommendations.append("Install training dependencies: uv sync --extra rl")
    snapshot_integrity = not manifest_error and all(item["ok"] for item in datasets)
    stable_training_compatible = channel == "stable" and channel_consistent and version_consistent
    if not snapshot_integrity or not channel_consistent or not version_consistent:
        recommendations.append(
            f"Refresh {channel} source data into an isolated manifest: "
            f"uv run sts2sim sync-data --data-dir {cache_root} --channel {channel} "
            f"--manifest {manifest_path}"
        )
    if not stable_training_compatible:
        recommendations.append(
            "The simulator trains against stable English source data. "
            "Use a stable snapshot with consistent channel provenance; beta is audit-only."
        )
    return {
        "ready": not recommendations,
        "scope": "Dependency availability and stable source snapshot integrity; not game parity.",
        "cache_dir": str(cache_dir),
        "manifest_path": str(manifest_path),
        "channel": channel,
        "manifest_channels": manifest_channels,
        "channel_consistent": channel_consistent,
        "snapshot_integrity": snapshot_integrity,
        "stable_training_compatible": stable_training_compatible,
        "game_version": game_version,
        "game_versions": game_versions,
        "version_consistent": version_consistent,
        "game_version_status": (
            "mixed" if not version_consistent else "recorded" if game_version else "unverified"
        ),
        "manifest_error": manifest_error,
        "dependencies": dependencies,
        "datasets": datasets,
        "recommendations": recommendations,
    }
