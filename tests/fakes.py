"""Local fixtures that model the planned simulator interfaces.

These keep Worker D tests executable before the core simulator modules land.
When those modules exist, tests prefer the real public interfaces instead.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _stable_seed(seed: int, stream: str) -> int:
    digest = hashlib.sha256(f"{seed}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class LocalGameRng:
    """Minimal deterministic RNG with named streams."""

    def __init__(self, seed: int = 0, stream: str = "root") -> None:
        self.seed = seed
        self.stream = stream
        self._random = random.Random(_stable_seed(seed, stream))

    def random_int(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper must be positive")
        return self._random.randrange(upper)

    def choice(self, options: Sequence[Any]) -> Any:
        if not options:
            raise ValueError("options must not be empty")
        return options[self.random_int(len(options))]

    def fork(self, stream: str) -> LocalGameRng:
        return LocalGameRng(self.seed, f"{self.stream}/{stream}")


def rng_int(rng: Any, upper: int) -> int:
    """Read an integer from either the planned RNG or a common equivalent."""

    for name in ("random_int", "randbelow", "randrange"):
        method = getattr(rng, name, None)
        if callable(method):
            return int(method(upper))
    raise AttributeError("RNG must expose random_int(upper)")


def rng_choice(rng: Any, options: Sequence[Any]) -> Any:
    method = getattr(rng, "choice", None)
    if callable(method):
        return method(options)
    return options[rng_int(rng, len(options))]


def local_play_run(
    seed: int = 0,
    policy: str = "random",
    max_steps: int | None = None,
    output_path: Path | None = None,
    output: Path | None = None,
    replay_path: Path | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Small deterministic episode used when sts2sim.api is unavailable."""

    steps = 10 if max_steps is None else max_steps
    rng = LocalGameRng(seed)
    hp = 80
    transcript: list[dict[str, Any]] = []
    for step in range(steps):
        action = rng.choice(("attack", "defend", "take-left", "take-right"))
        roll = rng.random_int(10)
        if action == "attack" and roll % 3 == 0:
            hp -= 1
        transcript.append(
            {"step": step, "action": action, "roll": roll, "player_hp": hp}
        )

    result = {
        "seed": seed,
        "policy": policy,
        "transcript": transcript,
        "final": {"player_hp": hp},
    }
    path = output_path or output or replay_path
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    return result


def local_replay(
    replay_path: Path,
    strict: bool = True,
    **_: Any,
) -> dict[str, Any]:
    """Replay and verify a local deterministic episode."""

    recorded = json.loads(replay_path.read_text(encoding="utf-8"))
    regenerated = local_play_run(
        seed=int(recorded["seed"]),
        policy=recorded.get("policy", "random"),
        max_steps=len(recorded.get("transcript", [])),
    )
    matched = regenerated["transcript"] == recorded.get("transcript", [])
    if strict and not matched:
        raise AssertionError("replay transcript did not match recorded run")
    return {
        "matched": matched,
        "seed": recorded["seed"],
        "transcript": regenerated["transcript"],
    }


@dataclass(frozen=True)
class DamageResult:
    damage: int
    hp_loss: int
    remaining_block: int


def calculate_attack_damage(
    base: int,
    strength: int = 0,
    weak: bool = False,
    vulnerable: bool = False,
) -> int:
    damage = max(0, base + strength)
    if weak:
        damage = math.floor(damage * 0.75)
    if vulnerable:
        damage = math.floor(damage * 1.5)
    return damage


def apply_block_damage(damage: int, block: int) -> DamageResult:
    blocked = min(max(block, 0), max(damage, 0))
    return DamageResult(
        damage=max(damage, 0),
        hp_loss=max(damage, 0) - blocked,
        remaining_block=max(block, 0) - blocked,
    )


def generate_card_rewards(
    pool: Sequence[Mapping[str, Any]],
    rng: Any,
    count: int = 3,
    **_: Any,
) -> list[Mapping[str, Any]]:
    remaining = list(pool)
    rewards: list[Mapping[str, Any]] = []
    while remaining and len(rewards) < count:
        index = rng_int(rng, len(remaining))
        rewards.append(remaining.pop(index))
    return rewards


def generate_map(rng: Any, floors: int = 5, width: int = 3) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: dict[str, list[str]] = {}
    for floor in range(floors):
        for lane in range(width):
            node_id = f"{floor}-{lane}"
            kind = "boss" if floor == floors - 1 else rng_choice(
                rng, ("monster", "elite", "event", "shop")
            )
            nodes.append(
                {"id": node_id, "floor": floor, "lane": lane, "kind": kind}
            )
            if floor < floors - 1:
                edges[node_id] = [
                    f"{floor + 1}-{next_lane}"
                    for next_lane in range(max(0, lane - 1), min(width, lane + 2))
                ]
    return {
        "nodes": nodes,
        "edges": edges,
        "start_ids": [f"0-{lane}" for lane in range(width)],
    }


def reachable_next_nodes(game_map: Mapping[str, Any], node: Mapping[str, Any] | str) -> list[str]:
    node_id = node if isinstance(node, str) else str(node["id"])
    return list(game_map.get("edges", {}).get(node_id, []))


def local_sync_manifest(
    manifest_path: Path,
    data_dir: Path,
    fetcher: Callable[[str], bytes],
    force: bool = False,
    **_: Any,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fetched = 0
    skipped = 0
    verified = 0
    files: list[str] = []

    for entry in manifest["files"]:
        target = data_dir / entry["path"]
        expected_sha = entry["sha256"]

        if target.exists() and not force:
            current = target.read_bytes()
            if hashlib.sha256(current).hexdigest() == expected_sha:
                skipped += 1
                verified += 1
                files.append(entry["path"])
                continue

        payload = fetcher(entry["url"])
        if hasattr(payload, "content"):
            payload = payload.content
        if not isinstance(payload, bytes):
            payload = bytes(payload)
        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"sha256 mismatch for {entry['path']}")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        fetched += 1
        verified += 1
        files.append(entry["path"])

    return {
        "version": manifest.get("version"),
        "fetched": fetched,
        "skipped": skipped,
        "verified": verified,
        "files": files,
    }
