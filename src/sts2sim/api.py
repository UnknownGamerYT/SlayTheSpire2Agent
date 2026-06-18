"""Stable public API for the headless simulator."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sts2sim.engine import (
    legal_actions as _legal_actions,
)
from sts2sim.engine import (
    load_state as _load_state,
)
from sts2sim.engine import (
    new_run_state,
    replay_actions,
    serialize_state,
    state_digest,
    step_state,
)


def new_run(
    seed: int | str,
    character_id: str,
    ascension: int,
    *,
    act_variant_policy: str = "source_default",
    source_data: Any | None = None,
) -> Any:
    """Create a deterministic run state."""
    return new_run_state(
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        source_data=_with_run_options(source_data, act_variant_policy),
    )


def legal_actions(state: Any) -> list[Any]:
    """Return actions currently legal for a state."""
    return list(_legal_actions(state))


def step(state: Any, action: Any) -> Any:
    """Apply one action and return a transition object."""
    return step_state(state, action)


def serialize(state: Any) -> dict[str, Any]:
    """Serialize a run state to a JSON-compatible payload."""
    payload = serialize_state(state)
    if isinstance(payload, str):
        return dict(json.loads(payload))
    return dict(payload)


def load_state(payload: dict[str, Any]) -> Any:
    """Restore a run state from a JSON-compatible payload."""
    return _load_state(payload)


def replay(
    seed: int | str | None = None,
    character_id: str = "IRONCLAD",
    ascension: int = 0,
    actions: list[Any] | None = None,
    *,
    replay_path: Path | str | None = None,
    path: Path | str | None = None,
    strict: bool = True,
    source_data: Any | None = None,
    data_dir: Path | str | None = None,
) -> Any:
    """Replay a run from actions or verify a replay file."""
    del data_dir
    replay_source = replay_path if replay_path is not None else path
    replay_file = Path(replay_source) if replay_source is not None else None
    if replay_file is not None:
        recorded = json.loads(replay_file.read_text(encoding="utf-8"))
        regenerated = play_run(
            seed=recorded.get("seed", 0),
            character_id=recorded.get("character_id", recorded.get("character", character_id)),
            ascension=int(recorded.get("ascension", ascension)),
            policy=recorded.get("policy", "recorded"),
            max_steps=len(recorded.get("actions", recorded.get("transcript", []))),
            actions=recorded.get("actions") or _actions_from_transcript(recorded),
            source_data=source_data,
        )
        matched = regenerated.get("transcript") == recorded.get("transcript")
        if strict and not matched:
            raise AssertionError("replay transcript did not match recorded run")
        return {
            "matched": matched,
            "seed": recorded.get("seed"),
            "character_id": recorded.get("character_id", recorded.get("character")),
            "ascension": recorded.get("ascension", ascension),
            "transcript": regenerated.get("transcript", []),
        }

    if seed is None:
        raise TypeError("seed is required when replay_path/path is not provided")
    initial = new_run(seed, character_id, ascension, source_data=source_data)
    return replay_actions(initial, actions or [])


def play_run(
    seed: int | str = 0,
    character_id: str = "IRONCLAD",
    ascension: int = 0,
    *,
    policy: str = "random",
    max_steps: int | None = None,
    output_path: Path | str | None = None,
    output: Path | str | None = None,
    replay_path: Path | str | None = None,
    actions: list[Any] | None = None,
    source_data: Any | None = None,
    data_dir: Path | str | None = None,
    act: int | None = None,
) -> dict[str, Any]:
    """Run a small deterministic episode and optionally write a replay JSON file."""
    del data_dir, act
    state = new_run(seed, character_id, ascension, source_data=source_data)
    rng = random.Random(f"{seed}:{policy}")
    transcript: list[dict[str, Any]] = []
    action_payloads: list[dict[str, Any]] = []
    supplied_actions = list(actions or [])
    limit = (
        len(supplied_actions)
        if actions is not None
        else (10 if max_steps is None else max_steps)
    )

    for step_index in range(limit):
        legal = list(legal_actions(state))
        if not legal:
            break
        if supplied_actions:
            action = supplied_actions.pop(0)
        elif policy == "first":
            action = legal[0]
        else:
            action = rng.choice(legal)

        action_payload = _jsonable_action(action)
        before = state_digest(_load_state(state))
        state = step(state, action)
        after = state_digest(_load_state(state))
        action_payloads.append(action_payload)
        transcript.append(
            {
                "step": step_index,
                "action": action_payload,
                "phase": getattr(state.phase, "value", str(state.phase)),
                "state_hash_before": before,
                "state_hash_after": after,
            }
        )

    result = {
        "seed": seed,
        "character_id": character_id,
        "ascension": ascension,
        "policy": policy,
        "actions": action_payloads,
        "transcript": transcript,
        "final": {
            "phase": getattr(state.phase, "value", str(state.phase)),
            "state_hash": state_digest(_load_state(state)),
        },
    }
    target_source = output_path if output_path is not None else output
    target_source = replay_path if target_source is None else target_source
    target = Path(target_source) if target_source is not None else None
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def fuzz_run(
    *,
    count: int = 100,
    start_seed: int = 0,
    seeds: list[int] | None = None,
    max_steps: int | None = None,
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run a deterministic smoke sweep over seeds."""
    del data_dir
    selected = seeds or list(range(start_seed, start_seed + count))
    failures: list[dict[str, Any]] = []
    for seed in selected:
        try:
            play_run(seed=seed, max_steps=max_steps)
        except Exception as exc:  # pragma: no cover - defensive reporting surface
            failures.append({"seed": seed, "error": str(exc)})
            if output_dir is not None:
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                (out / f"seed-{seed}.error.txt").write_text(str(exc), encoding="utf-8")
    return {"seeds": selected, "failures": failures}


def _with_run_options(source_data: Any | None, act_variant_policy: str) -> Any:
    if source_data is None:
        return {"flags": {"act_variant_policy": act_variant_policy}}
    if not isinstance(source_data, Mapping):
        return source_data
    merged = dict(source_data)
    flags = dict(merged.get("flags", {})) if isinstance(merged.get("flags"), Mapping) else {}
    flags.setdefault("act_variant_policy", act_variant_policy)
    merged["flags"] = flags
    return merged


def _jsonable_action(action: Any) -> dict[str, Any]:
    if hasattr(action, "model_dump") and callable(action.model_dump):
        return dict(action.model_dump(mode="json"))
    if isinstance(action, Mapping):
        return dict(action)
    return {"type": str(action)}


def _actions_from_transcript(recorded: Mapping[str, Any]) -> list[dict[str, Any]]:
    transcript = recorded.get("transcript", [])
    if not isinstance(transcript, list):
        return []
    actions: list[dict[str, Any]] = []
    for entry in transcript:
        if isinstance(entry, Mapping) and isinstance(entry.get("action"), Mapping):
            actions.append(dict(entry["action"]))
    return actions
