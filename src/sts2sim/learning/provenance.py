"""Fingerprint the exact English source snapshots selected by the simulator."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from sts2sim.data.sync import (
    DEFAULT_ENDPOINTS,
    SourceManifest,
    load_manifest,
    resolve_manifest_path,
    source_channel_matches,
)

SOURCE_DATA_PROVENANCE_SCHEMA_VERSION = 1
_PACKAGE_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"
_LOCATION_MODIFIERS = frozenset({"data_cache_dir", "cache_dir", "manifest_path"})


def source_cache_roots(cache_dir: Path | str | None = None) -> tuple[Path, ...]:
    """Match engine resolution: configured cache, cwd cache, package fallback."""
    candidates = (
        *((Path(cache_dir),) if cache_dir is not None else ()),
        Path("data/cache"),
        _PACKAGE_CACHE_DIR,
    )
    return tuple(dict.fromkeys(path.resolve() for path in candidates))


def resolve_english_dataset_path(
    dataset: str,
    cache_dir: Path | str | None = None,
) -> Path | None:
    """Resolve one dataset as the engine does, including per-file fallback."""
    if dataset not in {endpoint.key for endpoint in DEFAULT_ENDPOINTS}:
        raise ValueError(f"Unknown source dataset: {dataset!r}")
    for root in source_cache_roots(cache_dir):
        path = root / "eng" / f"{dataset}.json"
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


@dataclass(frozen=True)
class SourceDataSnapshot:
    """Content identity plus diagnostic provenance; paths stay outside its hash."""

    checksum: str
    datasets: Mapping[str, Mapping[str, Any]]
    roots: tuple[Path, ...]
    channel: str
    game_version: str | None
    complete: bool
    consistent: bool

    def metadata(self) -> dict[str, Any]:
        return {
            "source_data_provenance_schema_version": SOURCE_DATA_PROVENANCE_SCHEMA_VERSION,
            "source_data_checksum": self.checksum,
            "source_data_channel": self.channel,
            "source_data_game_version": self.game_version,
            "source_data_complete": self.complete,
            "source_data_consistent": self.consistent,
            "source_data_datasets": {key: dict(value) for key, value in self.datasets.items()},
        }


def source_data_snapshot(
    *,
    cache_dir: Path | str | None = None,
    modifiers: Mapping[str, Any] | None = None,
) -> SourceDataSnapshot:
    """Hash payload bytes and matching channel/version provenance, never fetch times.

    Optional JSON-valued modifiers describe source overrides such as draw counts.
    Cache/manifest location selectors are excluded because relocating an identical
    snapshot must not invalidate it. The standard PPO trainer uses no overrides.
    """
    manifests: dict[Path, dict[str, SourceManifest]] = {}
    datasets: dict[str, dict[str, Any]] = {}
    roots: list[Path] = []
    for endpoint in DEFAULT_ENDPOINTS:
        key = f"eng/{endpoint.key}"
        path = resolve_english_dataset_path(endpoint.key, cache_dir)
        entry: dict[str, Any] = {
            "sha256": None,
            "valid_json": False,
            "manifest_matches": False,
            "channel": None,
            "game_version": None,
        }
        datasets[key] = entry
        if path is None:
            continue
        root = path.parent.parent
        roots.append(root)
        try:
            payload_bytes = path.read_bytes()
            entry["sha256"] = sha256(payload_bytes).hexdigest()
            payload = json.loads(payload_bytes)
            entry["valid_json"] = isinstance(payload, list | dict)
        except (OSError, ValueError):
            continue
        if root not in manifests:
            try:
                manifests[root] = load_manifest(resolve_manifest_path(None, root))
            except (OSError, ValueError, TypeError, AttributeError):
                manifests[root] = {}
        source = manifests[root].get(key)
        if source is None:
            continue
        entry["manifest_matches"] = source.sha256 == entry["sha256"]
        if entry["manifest_matches"]:
            entry["channel"] = source.channel
            entry["game_version"] = source.game_version
            entry["channel_matches_url"] = source_channel_matches(source, source.channel)

    active_roots = tuple(dict.fromkeys(roots))
    channels = {entry["channel"] for entry in datasets.values()}
    channel = (
        str(next(iter(channels)))
        if len(channels) == 1 and None not in channels
        else ("unknown" if channels == {None} else "mixed")
    )
    versions = {entry["game_version"] for entry in datasets.values()}
    game_version = next(iter(versions)) if len(versions) == 1 else None
    complete = all(entry["valid_json"] for entry in datasets.values())
    consistent = (
        complete
        and len(active_roots) == 1
        and len(channels) == 1
        and None not in channels
        and len(versions) == 1
        and all(entry.get("channel_matches_url", False) for entry in datasets.values())
    )
    payload = {
        "schema_version": SOURCE_DATA_PROVENANCE_SCHEMA_VERSION,
        "datasets": datasets,
        "modifiers": {
            key: value for key, value in (modifiers or {}).items() if key not in _LOCATION_MODIFIERS
        },
    }
    checksum = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SourceDataSnapshot(
        checksum=checksum,
        datasets=datasets,
        roots=active_roots,
        channel=channel,
        game_version=game_version,
        complete=complete,
        consistent=consistent,
    )


def require_training_sources(
    character_id: str,
    snapshot: SourceDataSnapshot | None = None,
) -> SourceDataSnapshot:
    """Reject incomplete/mixed/beta runs before collecting real-character rollouts."""
    active = snapshot or source_data_snapshot()
    if character_id.strip().upper() == "TEST":
        return active
    if not active.complete or not active.consistent or active.channel != "stable":
        raise ValueError(
            "Training requires one complete, consistent stable English source snapshot. "
            "Run `sts2sim sync-data --data-dir data/cache --channel stable` and "
            "`sts2sim training-check`; beta snapshots are audit-only."
        )
    from sts2sim.learning.readiness import check_training_readiness

    readiness = check_training_readiness(cache_dir=active.roots[0])
    if not readiness["ready"]:
        raise ValueError(
            "Training source/dependency checks failed: " + "; ".join(readiness["recommendations"])
        )
    return active
