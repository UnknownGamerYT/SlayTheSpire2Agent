from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

from fakes import local_sync_manifest
from helpers import call_with_supported_kwargs, import_attr, jsonable


def _sync_manifest() -> Any:
    return import_attr(
        ("sts2sim.data.manifest", "sts2sim.data.sync", "sts2sim.data"),
        ("sync_manifest", "sync_data", "sync_all", "sync"),
    ) or local_sync_manifest


def test_manifest_sync_uses_fetcher_and_verifies_sha256(tmp_path: Path) -> None:
    sync = _sync_manifest()
    if getattr(sync, "__name__", "") == "sync_all":
        _assert_sync_all_with_fake_fetcher(sync, tmp_path)
        return

    cards_payload = b'[{"id":"strike","type":"attack"}]\n'
    relics_payload = b'[{"id":"burning_blood"}]\n'
    remote = {
        "fake://cards": cards_payload,
        "fake://relics": relics_payload,
    }
    fetch_calls: list[str] = []

    def fetcher(url: str) -> bytes:
        fetch_calls.append(url)
        return remote[url]

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": "test",
                "files": [
                    {
                        "path": "cards.json",
                        "url": "fake://cards",
                        "sha256": hashlib.sha256(cards_payload).hexdigest(),
                    },
                    {
                        "path": "relics/relics.json",
                        "url": "fake://relics",
                        "sha256": hashlib.sha256(relics_payload).hexdigest(),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    data_dir = tmp_path / "data"
    result = call_with_supported_kwargs(
        _sync_manifest(),
        manifest_path=manifest_path,
        manifest=manifest_path,
        data_dir=data_dir,
        fetcher=fetcher,
        force=False,
    )
    payload = jsonable(result)

    assert (data_dir / "cards.json").read_bytes() == cards_payload
    assert (data_dir / "relics" / "relics.json").read_bytes() == relics_payload
    assert fetch_calls == ["fake://cards", "fake://relics"]
    assert payload["verified"] == 2

    fetch_calls.clear()
    second = call_with_supported_kwargs(
        _sync_manifest(),
        manifest_path=manifest_path,
        manifest=manifest_path,
        data_dir=data_dir,
        fetcher=fetcher,
        force=False,
    )
    second_payload = jsonable(second)

    assert fetch_calls == []
    assert second_payload["skipped"] == 2


def _assert_sync_all_with_fake_fetcher(sync: Any, tmp_path: Path) -> None:
    data_sync = importlib.import_module("sts2sim.data.sync")
    endpoint = data_sync.EndpointSpec
    endpoints = (
        endpoint("cards", "/cards"),
        endpoint("relics", "/relics"),
    )
    remote = {
        "fake://codex/cards?lang=eng": [{"id": "strike", "type": "attack"}],
        "fake://codex/relics?lang=eng": [{"id": "burning_blood"}],
    }
    fetch_calls: list[str] = []

    def fetcher(url: str) -> Any:
        fetch_calls.append(url)
        return remote[url]

    data_dir = tmp_path / "cache"
    manifest_path = tmp_path / "manifest.json"
    result = call_with_supported_kwargs(
        sync,
        cache_dir=data_dir,
        data_dir=data_dir,
        manifest_path=manifest_path,
        lang="eng",
        base_url="fake://codex",
        fetcher=fetcher,
        endpoints=endpoints,
    )
    payload = jsonable(result)

    assert (data_dir / "eng" / "cards.json").exists()
    assert (data_dir / "eng" / "relics.json").exists()
    assert manifest_path.exists()
    assert fetch_calls == [
        "fake://codex/cards?lang=eng",
        "fake://codex/relics?lang=eng",
    ]
    assert sorted(payload) == ["eng/cards", "eng/relics"]
