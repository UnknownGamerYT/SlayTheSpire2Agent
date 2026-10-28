from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

import sts2sim.learning.provenance as provenance
from sts2sim.data.sync import DEFAULT_ENDPOINTS, sync_all
from sts2sim.learning.content_vocab import load_content_vocab
from sts2sim.learning.masked_ppo import (
    _architecture_mismatches,
    _checkpoint_architecture,
    _expected_ppo_architecture,
)


@pytest.fixture
def snapshot_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provenance, "_PACKAGE_CACHE_DIR", tmp_path / "package-cache")
    root = tmp_path / "data/cache"
    sync_all(root, game_version="v0.test", fetcher=lambda _: [{"id": "SAME_ID", "damage": 6}])
    return root


def test_snapshot_hash_changes_when_balance_changes_without_changing_content_ids(
    snapshot_dir: Path,
) -> None:
    before = _expected_ppo_architecture(
        hidden_size=8, hidden_layers=1, head_hidden_layers=1, activation="silu"
    )
    sync_all(
        snapshot_dir, game_version="v0.test", fetcher=lambda _: [{"id": "SAME_ID", "damage": 7}]
    )
    after = _expected_ppo_architecture(
        hidden_size=8, hidden_layers=1, head_hidden_layers=1, activation="silu"
    )
    assert before["content_vocab_checksum"] == after["content_vocab_checksum"]
    assert before["source_data_checksum"] != after["source_data_checksum"]
    assert [row["key"] for row in _architecture_mismatches(before, after)] == [
        "source_data_checksum"
    ]
    saved = _checkpoint_architecture({"metadata": after})
    assert saved["source_data_checksum"] == after["source_data_checksum"]
    assert saved["source_data_channel"] == "stable"
    assert saved["source_data_game_version"] == "v0.test"


def test_snapshot_is_stable_across_refresh_timestamps_and_relocation(
    snapshot_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = provenance.source_data_snapshot()
    sync_all(
        snapshot_dir, game_version="v0.test", fetcher=lambda _: [{"id": "SAME_ID", "damage": 6}]
    )
    assert provenance.source_data_snapshot().checksum == before.checksum
    relocated = tmp_path / "relocated"
    shutil.copytree(snapshot_dir.parent, relocated / "data")
    monkeypatch.chdir(relocated)
    after = provenance.source_data_snapshot()
    assert before.roots != after.roots
    assert before.checksum == after.checksum
    assert str(tmp_path) not in json.dumps(after.metadata())


def test_snapshot_records_only_matching_manifest_provenance(snapshot_dir: Path) -> None:
    before = provenance.source_data_snapshot()
    assert before.channel == "stable"
    assert before.game_version == "v0.test"
    path = snapshot_dir / "eng/cards.json"
    path.write_text('[{"id":"SAME_ID","damage":42}]', encoding="utf-8")
    after = provenance.source_data_snapshot()
    assert after.datasets["eng/cards"]["channel"] is None
    assert after.datasets["eng/cards"]["game_version"] is None
    assert after.channel == "mixed"
    assert after.game_version is None
    with pytest.raises(ValueError, match="stable English source"):
        provenance.require_training_sources("IRONCLAD", after)


def test_unknown_version_stays_unknown_and_mixed_versions_refuse_training(
    snapshot_dir: Path,
) -> None:
    sync_all(snapshot_dir, fetcher=lambda _: [{"id": "SAME_ID", "damage": 6}])
    snapshot = provenance.source_data_snapshot()
    assert snapshot.game_version is None
    assert snapshot.consistent is True
    manifest = snapshot_dir.parent / "manifest.json"
    raw = json.loads(manifest.read_text())
    raw["sources"]["eng/cards"]["game_version"] = "another-build"
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    mixed = provenance.source_data_snapshot()
    assert mixed.game_version is None
    assert mixed.consistent is False
    with pytest.raises(ValueError, match="consistent stable"):
        provenance.require_training_sources("IRONCLAD", mixed)


def test_modifiers_change_snapshot_but_location_selectors_do_not(snapshot_dir: Path) -> None:
    first = provenance.source_data_snapshot(
        modifiers={"draw_per_turn": 5, "data_cache_dir": "/one"}
    )
    moved = provenance.source_data_snapshot(
        modifiers={"draw_per_turn": 5, "data_cache_dir": "/two"}
    )
    changed = provenance.source_data_snapshot(modifiers={"draw_per_turn": 6})
    assert first.checksum == moved.checksum
    assert first.checksum != changed.checksum


def test_missing_source_changes_hash_and_refuses_real_character_training(
    snapshot_dir: Path,
) -> None:
    before = provenance.source_data_snapshot()
    (snapshot_dir / "eng/cards.json").unlink()
    after = provenance.source_data_snapshot()
    assert before.checksum != after.checksum
    assert after.complete is False
    assert after.datasets["eng/cards"]["sha256"] is None
    with pytest.raises(ValueError, match="stable English source"):
        provenance.require_training_sources("IRONCLAD", after)
    assert provenance.require_training_sources("TEST", after) is after


def test_mixed_cache_roots_refuse_training_even_with_valid_individual_snapshots(
    snapshot_dir: Path, tmp_path: Path
) -> None:
    package = tmp_path / "package-cache"
    sync_all(package, game_version="v0.test", fetcher=lambda _: [{"id": "FALLBACK"}])
    (snapshot_dir / "eng/cards.json").unlink()
    snapshot = provenance.source_data_snapshot()
    assert snapshot.complete is True
    assert len(snapshot.roots) == 2
    assert snapshot.consistent is False
    with pytest.raises(ValueError, match="consistent stable"):
        provenance.require_training_sources("DEFECT", snapshot)


def test_beta_manifest_cannot_be_used_for_stable_training(snapshot_dir: Path) -> None:
    manifest = snapshot_dir.parent / "manifest.json"
    raw = json.loads(manifest.read_text())
    for source in raw["sources"].values():
        source["channel"] = "beta"
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    snapshot = provenance.source_data_snapshot()
    assert snapshot.channel == "beta"
    with pytest.raises(ValueError, match="beta snapshots are audit-only"):
        provenance.require_training_sources("SILENT", snapshot)


def test_valid_stable_snapshot_passes_preflight_and_tracks_all_configured_datasets(
    snapshot_dir: Path,
) -> None:
    snapshot = provenance.require_training_sources("REGENT")
    assert snapshot.complete and snapshot.consistent
    assert len(snapshot.datasets) == len(DEFAULT_ENDPOINTS)


def test_vocab_prefers_cwd_and_follows_each_engine_dataset_fallback(
    snapshot_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package-cache"
    sync_all(package, fetcher=lambda _: [{"id": "PACKAGE_ID"}])
    vocab = load_content_vocab()
    assert vocab.id_for_token("card:same_id") != vocab.unknown_id
    assert vocab.id_for_token("card:package_id") == vocab.unknown_id
    (snapshot_dir / "eng/cards.json").unlink()
    fallback_vocab = load_content_vocab()
    assert fallback_vocab.id_for_token("card:package_id") != fallback_vocab.unknown_id
    assert fallback_vocab.id_for_token("relic:same_id") != fallback_vocab.unknown_id
    # A cwd change must not reuse a vocabulary cached under the old default None key.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    moved_vocab = load_content_vocab()
    assert moved_vocab.id_for_token("relic:package_id") != moved_vocab.unknown_id
    assert moved_vocab.id_for_token("relic:same_id") == moved_vocab.unknown_id


def test_vocab_refreshes_replaced_ids_even_when_file_mtime_is_preserved(snapshot_dir: Path) -> None:
    before = load_content_vocab()
    path = snapshot_dir / "eng/cards.json"
    stat = path.stat()
    path.write_bytes(path.read_bytes().replace(b"SAME_ID", b"NEXT_ID"))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = load_content_vocab()
    assert before.id_for_token("card:next_id") == before.unknown_id
    assert after.id_for_token("card:next_id") != after.unknown_id
    assert after.id_for_token("card:same_id") == after.unknown_id


def test_train_masked_ppo_checks_sources_before_initializing_torch(
    snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sts2sim.learning.masked_ppo as ppo

    (snapshot_dir / "eng/cards.json").unlink()
    monkeypatch.setattr(
        ppo, "_load_torch", lambda: pytest.fail("Should fail before model creation")
    )
    with pytest.raises(ValueError, match="stable English source"):
        ppo.train_masked_ppo(character_id="IRONCLAD", max_batches=1)


def test_checkpoint_validation_rejects_changed_stats_with_identical_vocabulary(
    snapshot_dir: Path, tmp_path: Path
) -> None:
    import sts2sim.learning.masked_ppo as ppo

    torch = pytest.importorskip("torch")
    architecture = _expected_ppo_architecture(
        hidden_size=8, hidden_layers=1, head_hidden_layers=1, activation="silu"
    )
    model = ppo._masked_actor_critic_class(torch.nn)(
        **ppo._model_kwargs_from_architecture(architecture)
    )
    path = tmp_path / "policy.pt"
    torch.save({"architecture": architecture, "model_state": model.state_dict()}, path)
    assert (
        ppo.validate_masked_ppo_checkpoint(
            path, hidden_size=8, hidden_layers=1, head_hidden_layers=1
        )["compatible"]
        is True
    )
    sync_all(
        snapshot_dir, game_version="v0.test", fetcher=lambda _: [{"id": "SAME_ID", "damage": 7}]
    )
    changed = ppo.validate_masked_ppo_checkpoint(
        path, hidden_size=8, hidden_layers=1, head_hidden_layers=1
    )
    assert changed["compatible"] is False
    assert [item["key"] for item in changed["mismatches"]] == ["source_data_checksum"]
