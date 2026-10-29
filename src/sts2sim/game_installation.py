"""Read-only identification of a locally owned stable Steam installation.

The source pin reports these build identifiers; it does not publish a trusted
SHA-256 of the game binary. Recording a local digest never proves game parity.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

STABLE_SOURCE_COMMIT = "5d874e855ae1d0550d5c30416fadd7dbb0d133b7"
STABLE_SOURCE_URL = (
    "https://github.com/zhiyue/sts2-rl-agent/blob/"
    + STABLE_SOURCE_COMMIT
    + "/GAME_VERSIONS.md"
)
STABLE_BUILD = {
    "version": "0.107.1",
    "steam_appid": "2868840",
    "steam_buildid": "23811903",
    "commit": "59260271",
    "main_assembly_hash": -1555940892,
}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _manifest_value(text: str, key: str) -> str | None:
    match = re.search(r'"' + re.escape(key) + r'"\s+"([^"\r\n]*)"', text)
    return match.group(1) if match else None


def inspect_game_installation(
    game_dir: Path | str, *, steam_manifest: Path | str | None = None
) -> dict[str, Any]:
    """Fingerprint files without loading assemblies, starting Steam, or modifying the game."""
    directory = Path(game_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Game directory does not exist: {directory}")
    release_path = directory / "release_info.json"
    if not release_path.is_file():
        raise ValueError(f"Missing game release metadata: {release_path}")
    try:
        release = json.loads(release_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read game release metadata: {exc}") from exc
    if not isinstance(release, dict):
        raise ValueError("Game release metadata must be a JSON object.")

    checks: list[dict[str, Any]] = []
    for key in ("version", "commit", "main_assembly_hash"):
        actual = release.get(key)
        if key == "version" and isinstance(actual, str):
            actual = actual.removeprefix("v")
        expected = STABLE_BUILD[key]
        checks.append({
            "field": key, "expected": expected, "actual": actual,
            "matched": type(actual) is type(expected) and actual == expected,
        })

    manifest_path = (
        Path(steam_manifest).expanduser().resolve()
        if steam_manifest is not None
        else directory.parent.parent / "appmanifest_2868840.acf"
    )
    manifest: dict[str, str | None] = {}
    if manifest_path.is_file():
        manifest_text = manifest_path.read_text(encoding="utf-8-sig")
        for key, expected_key in (("appid", "steam_appid"), ("buildid", "steam_buildid")):
            actual = _manifest_value(manifest_text, key)
            manifest[key] = actual
            checks.append({
                "field": expected_key, "expected": STABLE_BUILD[expected_key],
                "actual": actual, "matched": actual == STABLE_BUILD[expected_key],
            })
        # A pending update can leave the manifest identifying a different binary.
        manifest["target_buildid"] = _manifest_value(manifest_text, "TargetBuildID")
        target = manifest["target_buildid"]
        if target and target != "0":
            checks.append({
                "field": "pending_steam_buildid", "expected": STABLE_BUILD["steam_buildid"],
                "actual": target, "matched": target == STABLE_BUILD["steam_buildid"],
            })
    else:
        checks.append({
            "field": "steam_manifest", "expected": str(manifest_path),
            "actual": None, "matched": False,
        })

    assemblies = sorted(directory.glob("data_sts2_*/sts2.dll"))
    checks.append({
        "field": "main_assembly_count", "expected": 1, "actual": len(assemblies),
        "matched": len(assemblies) == 1,
    })
    assembly_info = [
        {"path": str(path.relative_to(directory)), "bytes": path.stat().st_size,
         "sha256": _sha256(path)}
        for path in assemblies
    ]
    matches = all(check["matched"] for check in checks)
    return {
        "schema_version": 1,
        "game_directory": str(directory),
        "target": STABLE_BUILD,
        "target_provenance": STABLE_SOURCE_URL,
        "release_info": {
            key: release.get(key)
            for key in ("version", "commit", "date", "branch", "main_assembly_hash")
        },
        "release_info_sha256": _sha256(release_path),
        "steam_manifest": manifest,
        "assemblies": assembly_info,
        "checks": checks,
        "matches_target_metadata": matches,
        "binary_verified_against_trusted_digest": False,
        "game_behavior_validated": False,
        "next_step": (
            "Load the patched bridge and capture real-game action comparisons."
            if matches else "Resolve failed installation checks before recording stable traces."
        ),
    }
