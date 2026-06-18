"""Sync small JSON snapshots from the public Spire Codex API.

The functions here are intentionally transport-light: tests can inject a
``fetcher`` callable or an ``httpx.Client`` so unit tests never need live
network access.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from pydantic import BaseModel, Field

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - exercised only in lean envs.
    httpx = None  # type: ignore[assignment]

DEFAULT_BASE_URL = "https://spire-codex.com/api"
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MANIFEST_PATH = Path("data") / "manifest.json"
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


class SourceAudit(BaseModel):
    """Count and checksum status for one cached source."""

    expected_count: int
    actual_count: int
    count_ok: bool
    sha256_ok: bool
    cache_path: str


@dataclass(frozen=True)
class EndpointSpec:
    """A list endpoint and the local cache key used for its payload."""

    key: str
    path: str
    use_lang: bool = True


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
    EndpointSpec("mechanics_constants", "/mechanics/constants", use_lang=False),
    EndpointSpec("mechanics_sections", "/mechanics/sections", use_lang=False),
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
    cache_dir: Path,
    lang: str = "eng",
    *,
    base_url: str = DEFAULT_BASE_URL,
    manifest_path: Path | None = None,
    fetcher: Fetcher | None = None,
    client: Any | None = None,
    endpoints: Iterable[EndpointSpec] = DEFAULT_ENDPOINTS,
) -> dict[str, SourceManifest]:
    """Fetch all configured endpoints into ``cache_dir`` and update a manifest."""

    cache_dir = Path(cache_dir)
    manifest_file = _resolve_manifest_path(manifest_path, cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    owns_client = client is None and fetcher is None
    active_client = client
    if active_client is None and fetcher is None:
        if httpx is None:
            raise RuntimeError("httpx is required for network sync; pass a fetcher for tests")
        active_client = httpx.Client(timeout=httpx.Timeout(30.0))
    fetched_at = datetime.now(UTC)
    manifests: dict[str, SourceManifest] = {}

    try:
        for endpoint in endpoints:
            url = _build_url(base_url, endpoint, lang)
            payload = _fetch_payload(url, fetcher=fetcher, client=active_client)
            payload_bytes = _json_bytes(payload)
            digest = sha256(payload_bytes).hexdigest()

            cache_path = _cache_path(cache_dir, endpoint.key, lang)
            _write_bytes_atomic(cache_path, payload_bytes)

            manifest_key = _manifest_key(endpoint.key, lang)
            manifests[manifest_key] = SourceManifest(
                source_url=url,
                fetched_at=fetched_at,
                sha256=digest,
                count=_count_payload(payload),
                schema_version=MANIFEST_SCHEMA_VERSION,
            )
    finally:
        if owns_client and active_client is not None:
            active_client.close()

    _write_manifest(manifest_file, manifests, generated_at=fetched_at)
    return manifests


def load_manifest(path: Path | None = None) -> dict[str, SourceManifest]:
    """Load ``data/manifest.json`` or another manifest path."""

    manifest_file = Path(path or DEFAULT_MANIFEST_PATH)
    if manifest_file.is_dir():
        manifest_file = _resolve_manifest_path(None, manifest_file)
    if not manifest_file.exists():
        return {}

    raw = json.loads(manifest_file.read_text(encoding="utf-8"))
    raw_sources = raw.get("sources", raw)
    return {
        key: _model_validate(SourceManifest, value)
        for key, value in raw_sources.items()
    }


def load_cached_json(cache_dir: Path, dataset: str, lang: str = "eng") -> JsonPayload:
    """Load one cached dataset by key, such as ``cards`` or ``mechanics/constants``."""

    resolved_lang, key = _resolve_dataset(dataset, lang)
    path = _cache_path(Path(cache_dir), key, resolved_lang)
    return json.loads(path.read_text(encoding="utf-8"))


def audit_source_counts(
    cache_dir: Path,
    lang: str = "eng",
    *,
    manifest_path: Path | None = None,
) -> dict[str, SourceAudit]:
    """Compare cached JSON counts and checksums against the manifest."""

    cache_dir = Path(cache_dir)
    manifests = load_manifest(_resolve_manifest_path(manifest_path, cache_dir))
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
        )
    return audits


def _build_url(base_url: str, endpoint: EndpointSpec, lang: str) -> str:
    url = f"{base_url.rstrip('/')}/{endpoint.path.lstrip('/')}"
    if endpoint.use_lang:
        return f"{url}?{urlencode({'lang': lang})}"
    return url


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
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _manifest_key(endpoint_key: str, lang: str) -> str:
    return f"{lang}/{endpoint_key}"


def _cache_path(cache_dir: Path, endpoint_key: str, lang: str) -> Path:
    return cache_dir / lang / f"{endpoint_key}.json"


def _resolve_dataset(dataset: str, lang: str) -> tuple[str, str]:
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


def _write_manifest(
    manifest_path: Path,
    sources: dict[str, SourceManifest],
    *,
    generated_at: datetime,
) -> None:
    existing_sources = load_manifest(manifest_path)
    existing_sources.update(sources)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "sources": {
            key: _model_dump(value)
            for key, value in sorted(existing_sources.items())
        },
    }
    _write_bytes_atomic(manifest_path, _json_bytes(payload))


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


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
