from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from typer.testing import CliRunner

from sts2sim.cli.app import app
from sts2sim.data import sync as data_sync
from sts2sim.data.sync import (
    DEFAULT_ENDPOINTS,
    EndpointSpec,
    audit_source_counts,
    load_cached_json,
    load_manifest,
    sync_all,
)

ENDPOINTS = (EndpointSpec("cards", "/cards"), EndpointSpec("relics", "/relics"))


def test_channel_snapshots_and_manifests_are_isolated(tmp_path: Path) -> None:
    cache = tmp_path / "data/cache"
    stable = sync_all(cache, fetcher=lambda _: [{"id": "stable"}], endpoints=ENDPOINTS)
    beta = sync_all(
        cache,
        channel="beta",
        game_version="v0.111.0",
        fetcher=lambda _: [{"id": "beta"}],
        endpoints=ENDPOINTS,
    )
    assert (tmp_path / "data/manifest.json").exists()
    assert (cache / "beta/manifest.json").exists()
    assert load_cached_json(cache, "cards") == [{"id": "stable"}]
    assert load_cached_json(cache, "cards", channel="beta") == [{"id": "beta"}]
    assert stable["eng/cards"].channel == "stable"
    assert stable["eng/cards"].game_version is None
    assert beta["eng/cards"].channel == "beta"
    assert beta["eng/cards"].game_version == "v0.111.0"
    assert "channel=beta" in beta["eng/cards"].source_url
    assert all(item.channel_ok for item in audit_source_counts(cache, channel="beta").values())
    assert load_manifest(cache / "beta")["eng/cards"].channel == "beta"


def test_legacy_manifest_without_channel_is_stable(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    sync_all(cache, fetcher=lambda _: [], endpoints=ENDPOINTS)
    manifest = tmp_path / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["schema_version"] = 1
    for item in payload["sources"].values():
        item.pop("channel")
        item.pop("game_version")
        item["schema_version"] = 1
    manifest.write_text(json.dumps(payload))
    loaded = load_manifest(manifest)
    assert loaded["eng/cards"].channel == "stable"
    assert loaded["eng/cards"].game_version is None
    sync_all(cache, fetcher=lambda _: [], endpoints=ENDPOINTS)
    assert json.loads(manifest.read_text())["schema_version"] == 2


def test_url_selectors_preserve_unrelated_query_and_support_zht(tmp_path: Path) -> None:
    seen = []

    def fetcher(url: str) -> list[object]:
        seen.append(url)
        return []

    sync_all(
        tmp_path,
        lang="zht",
        channel="beta",
        base_url="https://example.test/api?token=a%2Fb&lang=jpn&channel=stable&tag=1&tag=2",
        endpoints=(EndpointSpec("cards", "/cards?limit=20"),),
        fetcher=fetcher,
    )
    parsed = urlsplit(seen[0])
    assert parsed.path == "/api/cards"
    assert parse_qs(parsed.query) == {
        "token": ["a/b"],
        "lang": ["zht"],
        "channel": ["beta"],
        "tag": ["1", "2"],
        "limit": ["20"],
    }
    assert (tmp_path / "beta/zht/cards.json").exists()


def test_stable_selection_cannot_inherit_beta_url_selector(tmp_path: Path) -> None:
    seen = []
    sync_all(
        tmp_path,
        base_url="https://example.test/api?channel=beta&lang=zht",
        fetcher=lambda url: seen.append(url) or [],
        endpoints=ENDPOINTS,
    )
    assert all(parse_qs(urlsplit(url).query) == {"lang": ["eng"]} for url in seen)


def test_global_mechanics_endpoints_do_not_receive_entity_channel(tmp_path: Path) -> None:
    seen = []
    sync_all(
        tmp_path,
        channel="beta",
        endpoints=DEFAULT_ENDPOINTS[-2:],
        fetcher=lambda url: seen.append(url) or {},
    )
    assert all(not urlsplit(url).query for url in seen)


@pytest.mark.parametrize("lang", ["../eng", "eng/../../", "", "unknown"])
def test_invalid_language_rejected_before_fetch_or_write(tmp_path: Path, lang: str) -> None:
    seen = []
    with pytest.raises(ValueError, match="language"):
        sync_all(tmp_path / "cache", lang=lang, fetcher=lambda url: seen.append(url))
    assert not seen
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="language"):
        load_cached_json(tmp_path, "cards", lang=lang)


@pytest.mark.parametrize("channel", ["../beta", "BETA", "", "preview"])
def test_invalid_channel_rejected_before_fetch_or_write(tmp_path: Path, channel: str) -> None:
    seen = []
    with pytest.raises(ValueError, match="channel"):
        sync_all(tmp_path / "cache", channel=channel, fetcher=lambda url: seen.append(url))
    assert not seen
    assert not list(tmp_path.iterdir())


def test_mixed_manifest_is_never_merged(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    manifest = tmp_path / "manifest.json"
    sync_all(cache, fetcher=lambda _: [], endpoints=ENDPOINTS)
    original = manifest.read_bytes()
    with pytest.raises(ValueError, match="channels"):
        sync_all(cache, channel="beta", manifest_path=manifest, fetcher=lambda _: [])
    assert manifest.read_bytes() == original
    assert not (cache / "beta").exists()


def test_legacy_beta_url_cannot_masquerade_as_stable(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    manifest = tmp_path / "manifest.json"
    sync_all(cache, fetcher=lambda _: [], endpoints=ENDPOINTS)
    payload = json.loads(manifest.read_text())
    payload["sources"]["eng/cards"]["source_url"] += "&channel=beta"
    manifest.write_text(json.dumps(payload))
    assert audit_source_counts(cache)["eng/cards"].channel_ok is False
    with pytest.raises(ValueError, match="channels"):
        sync_all(cache, fetcher=lambda _: [], endpoints=ENDPOINTS)


def test_fetch_failure_leaves_previous_snapshot_byte_identical(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    sync_all(cache, fetcher=lambda _: [{"id": "original"}], endpoints=ENDPOINTS)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}

    def fail_second(url: str) -> list[dict[str, str]]:
        assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before
        if "/relics?" in url:
            raise RuntimeError("remote failure")
        return [{"id": "replacement"}]

    with pytest.raises(RuntimeError, match="remote failure"):
        sync_all(cache, fetcher=fail_second, endpoints=ENDPOINTS)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before


def test_recorded_version_merge_does_not_publish_mixed_snapshot(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    sync_all(cache, game_version="old-build", fetcher=lambda _: [], endpoints=ENDPOINTS)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    with pytest.raises(ValueError, match="game versions"):
        sync_all(cache, game_version="new-build", fetcher=lambda _: [], endpoints=ENDPOINTS[:1])
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before


@pytest.mark.parametrize(
    "endpoints", [(), (ENDPOINTS[0], ENDPOINTS[0]), (EndpointSpec("../escape", "/cards"),)]
)
def test_invalid_endpoints_cannot_publish(
    tmp_path: Path, endpoints: tuple[EndpointSpec, ...]
) -> None:
    with pytest.raises(ValueError):
        sync_all(tmp_path / "cache", fetcher=lambda _: [], endpoints=endpoints)
    assert not list(tmp_path.iterdir())


def test_manifest_cannot_alias_a_dataset_path(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="distinct"):
        sync_all(
            cache,
            manifest_path=cache / "eng/../eng/cards.json",
            fetcher=lambda _: [],
            endpoints=ENDPOINTS,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("invalid_payload", [None, 42, "not JSON"])
def test_invalid_download_never_replaces_cached_snapshot(
    tmp_path: Path,
    invalid_payload: object,
) -> None:
    cache = tmp_path / "cache"
    sync_all(cache, fetcher=lambda _: [], endpoints=ENDPOINTS)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    with pytest.raises(ValueError):
        sync_all(cache, fetcher=lambda _: invalid_payload, endpoints=ENDPOINTS)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before


@pytest.mark.parametrize("failed_name", ["relics.json", "manifest.json"])
def test_publish_failure_rolls_back_existing_and_removes_new_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_name: str
) -> None:
    cache = tmp_path / "cache"
    sync_all(cache, fetcher=lambda _: [{"id": "original"}], endpoints=ENDPOINTS[:1])
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    original_replace = Path.replace

    def fail_publish(path: Path, target: Path) -> Path:
        if target.name == failed_name:
            raise OSError("disk write failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_publish)
    with pytest.raises(OSError, match="disk write failed"):
        sync_all(cache, fetcher=lambda _: [{"id": "new"}], endpoints=ENDPOINTS)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before
    assert not list(tmp_path.rglob("*.tmp"))


def test_staging_failure_leaves_snapshot_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    sync_all(cache, fetcher=lambda _: [], endpoints=ENDPOINTS)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    original_stage = data_sync._stage_bytes

    def fail_stage(path: Path, data: bytes) -> Path:
        if path.name == "relics.json":
            raise OSError("no disk space")
        return original_stage(path, data)

    monkeypatch.setattr(data_sync, "_stage_bytes", fail_stage)
    with pytest.raises(OSError, match="no disk space"):
        sync_all(cache, fetcher=lambda _: [{"id": "replacement"}], endpoints=ENDPOINTS)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before
    assert not list(tmp_path.rglob("*.tmp"))


def test_interruption_after_replace_still_restores_previous_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    sync_all(cache, fetcher=lambda _: [{"id": "original"}], endpoints=ENDPOINTS)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    original_replace = Path.replace

    def replace_then_interrupt(path: Path, target: Path) -> Path:
        result = original_replace(path, target)
        if target.name == "relics.json":
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(Path, "replace", replace_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        sync_all(cache, fetcher=lambda _: [{"id": "new"}], endpoints=ENDPOINTS)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before
    assert not list(tmp_path.rglob("*.tmp"))


def test_rollback_failure_preserves_original_backup_and_reports_both_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    sync_all(cache, fetcher=lambda _: [{"id": "original"}], endpoints=ENDPOINTS)
    original_bytes = (cache / "eng/cards.json").read_bytes()
    original_replace = Path.replace
    original_os_replace = data_sync.os.replace

    def fail_publish(path: Path, target: Path) -> Path:
        if target.name == "relics.json":
            raise OSError("publish failed")
        return original_replace(path, target)

    def fail_restore(source: Path, target: Path) -> None:
        if target.name == "cards.json":
            raise OSError("rollback failed")
        original_os_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_publish)
    monkeypatch.setattr(data_sync.os, "replace", fail_restore)
    with pytest.raises(ExceptionGroup, match="publication and rollback failed") as caught:
        sync_all(cache, fetcher=lambda _: [{"id": "new"}], endpoints=ENDPOINTS)
    assert len(caught.value.exceptions) == 2
    backups = list(tmp_path.rglob("*.tmp"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original_bytes


def test_cli_sync_default_matches_simulator_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    original_sync = sync_all

    def fake_network(
        cache_dir: Path,
        lang: str = "eng",
        channel: str = "stable",
        manifest_path: Path | None = None,
    ) -> object:
        return original_sync(
            cache_dir,
            lang,
            channel=channel,
            manifest_path=manifest_path,
            fetcher=lambda _: [{"id": "test"}],
        )

    monkeypatch.setattr(data_sync, "sync_all", fake_network)
    result = CliRunner().invoke(app, ["sync-data"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "data/cache/eng/cards.json").exists()
    assert (tmp_path / "data/manifest.json").exists()
    result = CliRunner().invoke(app, ["sync-data", "--channel", "beta"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "data/cache/beta/eng/cards.json").exists()
    assert (tmp_path / "data/cache/beta/manifest.json").exists()
    result = CliRunner().invoke(app, ["audit-coverage", "--channel", "beta", "--fail-under", "1"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["eng/cards"]["channel"] == "beta"
