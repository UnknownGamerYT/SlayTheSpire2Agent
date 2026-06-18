from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .models import RunState

SerializedState = str | bytes | bytearray | Mapping[str, Any] | RunState


def serialize_state(state: RunState, *, indent: int | None = None) -> str:
    return state.model_dump_json(indent=indent)


def load_state(data: SerializedState) -> RunState:
    if isinstance(data, RunState):
        return data
    if isinstance(data, (bytes, bytearray)):
        return RunState.model_validate_json(bytes(data).decode("utf-8"))
    if isinstance(data, str):
        return RunState.model_validate_json(data)
    if isinstance(data, Mapping):
        return RunState.model_validate(dict(data))
    raise TypeError(f"Unsupported state payload type: {type(data)!r}")


def state_digest(state: RunState, *, include_replay: bool = False) -> str:
    payload = state.model_dump(mode="json", exclude_none=False)
    if not include_replay:
        payload = dict(payload)
        payload["replay_log"] = []
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

