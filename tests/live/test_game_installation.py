from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sts2sim.cli.app import app
from sts2sim.game_installation import STABLE_BUILD, inspect_game_installation


def _installation(tmp_path: Path) -> Path:
    directory = tmp_path / "steamapps/common/Slay the Spire 2"
    assembly = directory / "data_sts2_windows_x86_64/sts2.dll"
    assembly.parent.mkdir(parents=True)
    assembly.write_bytes(b"synthetic fingerprint fixture, never executed")
    release = {key: STABLE_BUILD[key] for key in ("version", "commit", "main_assembly_hash")}
    release.update(branch="public", date="2026-06-18T15:43:56-07:00")
    (directory / "release_info.json").write_text(json.dumps(release), encoding="utf-8")
    (directory.parent.parent / "appmanifest_2868840.acf").write_text(
        '"AppState"\n{\n "appid" "2868840"\n "buildid" "23811903"\n}', encoding="utf-8"
    )
    return directory


def test_matching_metadata_records_hash_without_claiming_binary_or_behavior_parity(
    tmp_path: Path,
) -> None:
    directory = _installation(tmp_path)
    report = inspect_game_installation(directory)
    assert report["matches_target_metadata"] is True
    assert report["binary_verified_against_trusted_digest"] is False
    assert report["game_behavior_validated"] is False
    binary = directory / "data_sts2_windows_x86_64/sts2.dll"
    assert report["assemblies"][0]["sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert report["target"]["main_assembly_hash"] == -1555940892


@pytest.mark.parametrize(
    ("field", "value"),
    [("version", "0.111.0"), ("commit", "another-build"), ("main_assembly_hash", 123)],
)
def test_different_release_is_rejected(tmp_path: Path, field: str, value: str | int) -> None:
    directory = _installation(tmp_path)
    path = directory / "release_info.json"
    release = json.loads(path.read_text())
    release[field] = value
    path.write_text(json.dumps(release))
    report = inspect_game_installation(directory)
    assert report["matches_target_metadata"] is False
    assert next(check for check in report["checks"] if check["field"] == field)["matched"] is False


@pytest.mark.parametrize("change", ["manifest", "pending_update", "no_assembly", "ambiguous"])
def test_incomplete_or_changing_installation_is_rejected(tmp_path: Path, change: str) -> None:
    directory = _installation(tmp_path)
    manifest = directory.parent.parent / "appmanifest_2868840.acf"
    if change == "manifest":
        manifest.unlink()
    elif change == "pending_update":
        manifest.write_text(manifest.read_text() + '\n"TargetBuildID" "24724944"\n')
    elif change == "no_assembly":
        (directory / "data_sts2_windows_x86_64/sts2.dll").unlink()
    else:
        second = directory / "data_sts2_linuxbsd_x86_64/sts2.dll"
        second.parent.mkdir()
        second.write_bytes(b"second")
    assert inspect_game_installation(directory)["matches_target_metadata"] is False


def test_cli_writes_machine_readable_fingerprint_and_returns_failure_for_wrong_build(
    tmp_path: Path,
) -> None:
    directory = _installation(tmp_path)
    output = tmp_path / "fingerprint.json"
    result = CliRunner().invoke(app, [
        "inspect-game-installation", "--game-dir", str(directory), "--output", str(output),
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())["matches_target_metadata"] is True
    (directory.parent.parent / "appmanifest_2868840.acf").unlink()
    result = CliRunner().invoke(app, ["inspect-game-installation", "--game-dir", str(directory)])
    assert result.exit_code == 1
    assert json.loads(result.output)["matches_target_metadata"] is False


def test_invalid_release_json_does_not_execute_any_binary(tmp_path: Path) -> None:
    directory = _installation(tmp_path)
    (directory / "release_info.json").write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        inspect_game_installation(directory)
