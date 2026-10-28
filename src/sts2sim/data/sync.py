"""Sync small JSON snapshots from the public Spire Codex API.

The functions here are intentionally transport-light: tests can inject a
``fetcher`` callable or an ``httpx.Client`` so unit tests never need live
network access.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, Field

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - exercised only in lean envs.
    httpx = None  # type: ignore[assignment]

DEFAULT_BASE_URL = "https://spire-codex.com/api"
MANIFEST_SCHEMA_VERSION = 2
DEFAULT_MANIFEST_PATH = Path("data") / "manifest.json"
DEFAULT_CACHE_DIR = Path("data") / "cache"
SUPPORTED_CHANNELS = frozenset({"stable", "beta"})
SUPPORTED_LANGUAGES = frozenset(
    {
        "deu",
        "eng",
        "esp",
        "fra",
        "ita",
        "jpn",
        "kor",
        "pol",
        "ptb",
        "rus",
        "spa",
        "tha",
        "tur",
        "zhs",
        "zht",
    }
)

JsonPayload = Any
Fetcher = Callable[[str], JsonPayload]


class SourceManifest(BaseModel):
    """Metadata for one cached Spire Codex source payload."""

    source_url: str
    fetched_at: datetime
    sha256: str
    count: int = Field(ge=0)
    schema_version: int = MANIFEST_SCHEMA_VERSION
    channel: Literal["stable", "beta"] = "stable"
    game_version: str | None = None


class SourceAudit(BaseModel):
    """Count and checksum status for one cached source."""

    expected_count: int
    actual_count: int
    count_ok: bool
    sha256_ok: bool
    cache_path: str
    channel: Literal["stable", "beta"] = "stable"
    game_version: str | None = None
    channel_ok: bool = True


@dataclass(frozen=True)
class EndpointSpec:
    """A list endpoint and the local cache key used for its payload."""

    key: str
    path: str
    use_lang: bool = True
    use_channel: bool = True


DEFAULT_ENDPOINTS: tuple[EndpointSpec, ...] = (
    EndpointSpec("cards", "/cards"),
    EndpointSpec("characters", "/characters"),
    EndpointSpec("relics", "/relics"),
    EndpointSpec("potions", "/potions"),
    EndpointSpec("monsters", "/monsters"),
    EndpointSpec("encounters", "/encounters"),
    EndpointSpec("events", "/events"),
    EndpointSpec("powers", "/powers"),
    EndpointSpec("acts", "/acts"),
    EndpointSpec("ascensions", "/ascensions"),
    EndpointSpec("enchantments", "/enchantments"),
    EndpointSpec("afflictions", "/afflictions"),
    EndpointSpec("keywords", "/keywords"),
    EndpointSpec("orbs", "/orbs"),
    EndpointSpec("intents", "/intents"),
    EndpointSpec("mechanics_constants", "/mechanics/constants", use_lang=False, use_channel=False),
    EndpointSpec("mechanics_sections", "/mechanics/sections", use_lang=False, use_channel=False),
)

_ENDPOINT_ALIASES = {
    alias: spec.key
    for spec in DEFAULT_ENDPOINTS
    for alias in {
        spec.key,
        spec.key.replace("_", "-"),
        spec.path.strip("/"),
        spec.path.strip("/").replace("/", "_"),
        spec.path.strip("/").replace("/", "-"),
    }
}


def sync_all(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    lang: str = "eng",
    *,
    base_url: str = DEFAULT_BASE_URL,
    channel: Literal["stable", "beta"] = "stable",
    game_version: str | None = None,
    manifest_path: Path | None = None,
    fetcher: Fetcher | None = None,
    client: Any | None = None,
    endpoints: Iterable[EndpointSpec] = DEFAULT_ENDPOINTS,
) -> dict[str, SourceManifest]:
    """Fetch a complete snapshot before publishing it, rolling back write failures.

    ``cache_dir`` is the cache root. Beta snapshots live beneath its ``beta``
    subdirectory and cannot be merged into a stable manifest. ``game_version``
    is optional externally verified provenance; the API default is not a build ID.
    """

    _validate_language(lang)
    _validate_channel(channel)
    root_cache_dir = Path(cache_dir)
    cache_dir = channel_cache_dir(root_cache_dir, channel)
    manifest_file = resolve_manifest_path(manifest_path, root_cache_dir, channel=channel)
    existing_sources = load_manifest(manifest_file)
    if any(not source_channel_matches(source, channel) for source in existing_sources.values()):
        raise ValueError("Cannot merge source channels in one manifest; use an isolated manifest.")
    endpoint_specs = tuple(endpoints)
    writes: dict[Path, bytes] = {}
    destinations: set[Path] = {manifest_file.resolve()}
    for endpoint in endpoint_specs:
        path = _cache_path(cache_dir, endpoint.key, lang)
        if path.resolve() in destinations:
            raise ValueError("Snapshot destinations must be distinct.")
        destinations.add(path.resolve())
        writes[path] = b""
    if not endpoint_specs:
        raise ValueError("At least one source endpoint is required.")

    owns_client = client is None and fetcher is None
    active_client = client
    if active_client is None and fetcher is None:
        if httpx is None:
            raise RuntimeError("httpx is required for network sync; pass a fetcher for tests")
        active_client = httpx.Client(timeout=httpx.Timeout(30.0))
    fetched_at = datetime.now(UTC)
    manifests: dict[str, SourceManifest] = {}

    try:
        for endpoint in endpoint_specs:
            url = _build_url(base_url, endpoint, lang, channel)
            payload = _fetch_payload(url, fetcher=fetcher, client=active_client)
            if not isinstance(payload, (list, dict)):
                raise ValueError(f"Expected a JSON list or object from {url}")
            payload_bytes = _json_bytes(payload)
            digest = sha256(payload_bytes).hexdigest()

            cache_path = _cache_path(cache_dir, endpoint.key, lang)
            writes[cache_path] = payload_bytes

            manifest_key = _manifest_key(endpoint.key, lang)
            manifests[manifest_key] = SourceManifest(
                source_url=url,
                fetched_at=fetched_at,
                sha256=digest,
                count=_count_payload(payload),
                schema_version=MANIFEST_SCHEMA_VERSION,
                channel=channel,
                game_version=game_version,
            )
    finally:
        if owns_client and active_client is not None:
            active_client.close()

    existing_sources.update(manifests)
    recorded_versions = {
        source.game_version
        for source in existing_sources.values()
        if source.game_version is not None
    }
    if len(recorded_versions) > 1:
        raise ValueError("Cannot merge different recorded game versions in one manifest.")
    writes[manifest_file] = _manifest_bytes(existing_sources, generated_at=fetched_at)
    _publish_snapshot(writes)
    return manifests


def load_manifest(path: Path | None = None) -> dict[str, SourceManifest]:
    """Load ``data/manifest.json`` or another manifest path."""

    manifest_file = Path(path or DEFAULT_MANIFEST_PATH)
    if manifest_file.is_dir():
        manifest_file = _resolve_manifest_path(None, manifest_file)
    if not manifest_file.exists():
        return {}

    raw = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Source manifest must be a JSON object.")
    raw_sources = raw.get("sources", raw)
    if not isinstance(raw_sources, dict):
        raise ValueError("Manifest sources must be a JSON object.")
    return {key: _model_validate(SourceManifest, value) for key, value in raw_sources.items()}


def load_cached_json(
    cache_dir: Path, dataset: str, lang: str = "eng", *, channel: str = "stable"
) -> JsonPayload:
    """Load one cached dataset by key, such as ``cards`` or ``mechanics/constants``."""

    resolved_lang, key = _resolve_dataset(dataset, lang)
    path = _cache_path(channel_cache_dir(cache_dir, channel), key, resolved_lang)
    return json.loads(path.read_text(encoding="utf-8"))


def audit_source_counts(
    cache_dir: Path,
    lang: str = "eng",
    *,
    manifest_path: Path | None = None,
    channel: str = "stable",
) -> dict[str, SourceAudit]:
    """Compare cached JSON counts and checksums against the manifest."""

    _validate_language(lang)
    _validate_channel(channel)
    manifests = load_manifest(resolve_manifest_path(manifest_path, cache_dir, channel=channel))
    cache_dir = channel_cache_dir(cache_dir, channel)
    prefix = f"{lang}/"
    audits: dict[str, SourceAudit] = {}

    for manifest_key, manifest in manifests.items():
        if lang and not manifest_key.startswith(prefix):
            continue
        resolved_lang, key = _resolve_dataset(manifest_key, lang)
        path = _cache_path(cache_dir, key, resolved_lang)
        payload_bytes = path.read_bytes()
        payload = json.loads(payload_bytes.decode("utf-8"))
        actual_count = _count_payload(payload)
        audits[manifest_key] = SourceAudit(
            expected_count=manifest.count,
            actual_count=actual_count,
            count_ok=actual_count == manifest.count,
            sha256_ok=sha256(payload_bytes).hexdigest() == manifest.sha256,
            cache_path=str(path),
            channel=manifest.channel,
            game_version=manifest.game_version,
            channel_ok=source_channel_matches(manifest, channel),
        )
    return audits


def _build_url(base_url: str, endpoint: EndpointSpec, lang: str, channel: str = "stable") -> str:
    _validate_language(lang)
    _validate_channel(channel)
    base = urlsplit(base_url)
    suffix = urlsplit(endpoint.path)
    query = [
        (key, value)
        for key, value in (
            *parse_qsl(base.query, keep_blank_values=True),
            *parse_qsl(suffix.query, keep_blank_values=True),
        )
        if key not in {"lang", "channel"}
    ]
    if endpoint.use_lang:
        query.append(("lang", lang))
    if endpoint.use_channel and channel == "beta":
        query.append(("channel", "beta"))
    path = f"{base.path.rstrip('/')}/{suffix.path.lstrip('/')}"
    return urlunsplit((base.scheme, base.netloc, path, urlencode(query), ""))


def _validate_language(lang: str) -> None:
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported source language: {lang!r}")


def _validate_channel(channel: str) -> None:
    if channel not in SUPPORTED_CHANNELS:
        raise ValueError(f"Unsupported source channel: {channel!r}")


def channel_cache_dir(cache_dir: Path | str, channel: str = "stable") -> Path:
    """Resolve the isolated cache directory for a source channel."""
    _validate_channel(channel)
    root = Path(cache_dir)
    return root / "beta" if channel == "beta" else root


def source_channel_matches(source: SourceManifest, channel: str) -> bool:
    """Check explicit provenance and any channel selector in the source URL."""
    selectors = [
        v
        for k, v in parse_qsl(urlsplit(source.source_url).query, keep_blank_values=True)
        if k == "channel"
    ]
    return source.channel == channel and all(value == channel for value in selectors)


def _fetch_payload(
    url: str,
    *,
    fetcher: Fetcher | None,
    client: Any | None,
) -> JsonPayload:
    if fetcher is not None:
        result = fetcher(url)
    else:
        if client is None:
            raise ValueError("client is required when fetcher is not provided")
        response = client.get(url)
        response.raise_for_status()
        result = response

    if hasattr(result, "raise_for_status") and hasattr(result, "json"):
        result.raise_for_status()
        return result.json()
    if isinstance(result, bytes):
        return json.loads(result.decode("utf-8"))
    if isinstance(result, str):
        return json.loads(result)
    return result


def _count_payload(payload: JsonPayload) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("items", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
        return len(payload)
    return 1


def _json_bytes(payload: JsonPayload) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _manifest_key(endpoint_key: str, lang: str) -> str:
    return f"{lang}/{endpoint_key}"


def _cache_path(cache_dir: Path, endpoint_key: str, lang: str) -> Path:
    _validate_language(lang)
    if re.fullmatch(r"[a-z][a-z0-9_]*", endpoint_key) is None:
        raise ValueError(f"Invalid source dataset key: {endpoint_key!r}")
    return cache_dir / lang / f"{endpoint_key}.json"


def _resolve_dataset(dataset: str, lang: str) -> tuple[str, str]:
    _validate_language(lang)
    cleaned = dataset.strip().strip("/")
    if cleaned.startswith("api/"):
        cleaned = cleaned.removeprefix("api/")

    maybe_lang, separator, remainder = cleaned.partition("/")
    if separator and maybe_lang in SUPPORTED_LANGUAGES:
        lang = maybe_lang
        cleaned = remainder

    alias_key = cleaned.lower().replace("\\", "/").replace("-", "_")
    alias_key = alias_key.replace("/", "_")
    key = _ENDPOINT_ALIASES.get(cleaned, _ENDPOINT_ALIASES.get(alias_key))
    if key is None:
        allowed = ", ".join(sorted(spec.key for spec in DEFAULT_ENDPOINTS))
        raise KeyError(f"Unknown cached dataset {dataset!r}; expected one of: {allowed}")
    return lang, key


def _resolve_manifest_path(
    manifest_path: Path | None,
    cache_dir: Path | None = None,
) -> Path:
    if manifest_path is not None:
        return Path(manifest_path)
    if cache_dir is None:
        return DEFAULT_MANIFEST_PATH
    cache_dir = Path(cache_dir)
    if cache_dir.name == "cache":
        return cache_dir.parent / "manifest.json"
    return cache_dir / "manifest.json"


def resolve_manifest_path(
    manifest_path: Path | str | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    channel: str = "stable",
) -> Path:
    """Resolve the stable legacy location or an isolated beta manifest."""
    _validate_channel(channel)
    if manifest_path is not None:
        return Path(manifest_path)
    if channel == "beta":
        return channel_cache_dir(cache_dir, channel) / "manifest.json"
    return _resolve_manifest_path(None, Path(cache_dir))


def _manifest_bytes(sources: dict[str, SourceManifest], *, generated_at: datetime) -> bytes:
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "sources": {key: _model_dump(value) for key, value in sorted(sources.items())},
    }
    return _json_bytes(payload)


def _stage_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _publish_snapshot(writes: dict[Path, bytes]) -> None:
    staged: dict[Path, Path] = {}
    backups: dict[Path, Path | None] = {}
    published: list[Path] = []
    retained_backups: set[Path] = set()
    try:
        for path, data in writes.items():
            staged[path] = _stage_bytes(path, data)
            backups[path] = _stage_bytes(path, path.read_bytes()) if path.exists() else None
        try:
            for path, temporary in staged.items():
                published.append(path)
                temporary.replace(path)
        except BaseException as publish_error:
            rollback_errors: list[BaseException] = []
            for path in reversed(published):
                backup = backups[path]
                try:
                    if backup is None:
                        path.unlink(missing_ok=True)
                    else:
                        os.replace(backup, path)
                except BaseException as rollback_error:
                    if backup is not None:
                        retained_backups.add(backup)
                        rollback_error.add_note(f"Original snapshot backup preserved at {backup}")
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                raise BaseExceptionGroup(
                    "Snapshot publication and rollback failed", [publish_error, *rollback_errors]
                ) from None
            raise
    finally:
        for candidate in (*staged.values(), *backups.values()):
            if candidate is not None and candidate not in retained_backups:
                candidate.unlink(missing_ok=True)


def _model_validate(model_type: type[BaseModel], value: Any) -> Any:
    if hasattr(model_type, "model_validate"):
        return model_type.model_validate(value)
    return model_type.parse_obj(value)


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        dumped = model.model_dump(mode="json")
        return dict(dumped)
    loaded = json.loads(model.json())
    return dict(loaded) if isinstance(loaded, dict) else {}
