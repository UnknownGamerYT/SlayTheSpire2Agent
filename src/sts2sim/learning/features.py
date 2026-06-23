"""State/action signatures for dependency-free self-learning agents."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def state_signature(observation: Mapping[str, Any]) -> str:
    """Return a coarse symbolic state bucket derived only from observations."""

    vector_schema = _sequence(observation.get("vector_schema"))
    vector = _sequence(observation.get("vector"))
    values = {
        str(name): _float(vector[index])
        for index, name in enumerate(vector_schema)
        if index < len(vector)
    }
    phase = str(observation.get("phase", "unknown"))
    return "|".join(
        (
            f"phase:{phase}",
            f"act:{int(values.get('act', 0.0))}",
            f"floor:{_bucket(values.get('floor', 0.0), 5)}",
            f"hp:{_ratio_bucket(values.get('player_hp', 0.0), values.get('player_max_hp', 1.0))}",
            f"energy:{int(values.get('player_energy', 0.0))}",
            f"incoming:{_bucket(values.get('incoming_damage', 0.0), 8)}",
            f"monsters:{int(values.get('alive_monster_count', 0.0))}",
            f"deck:{_bucket(values.get('master_deck_count', 0.0), 5)}",
        )
    )


def action_signature(action_descriptor: Mapping[str, Any]) -> str:
    """Return a portable action signature.

    This avoids using state-local action IDs as the learning key. It uses only
    the observable action shape, not any hand-authored strategic score.
    """

    action = _mapping(action_descriptor.get("action"))
    payload = _mapping(action.get("payload"))
    card = _mapping(action_descriptor.get("card"))
    node = _mapping(action_descriptor.get("node"))
    signature = {
        "type": str(action.get("type", action_descriptor.get("type", "unknown"))),
        "has_card": action.get("card_instance_id") is not None,
        "has_target": action.get("target_id") is not None,
        "payload_keys": sorted(payload.keys()),
    }
    if card:
        signature["card"] = {
            "card_id": str(card.get("card_id", "")),
            "type": str(card.get("type", "")),
            "cost": _int(card.get("cost")),
            "target": str(card.get("target", "")),
            "upgraded": bool(card.get("upgraded", False)),
            "exhausts": bool(card.get("exhausts", False)),
            "effect_keys": sorted(str(key) for key in _sequence(card.get("effect_keys"))),
        }
    if node:
        signature["node"] = {
            "kind": str(node.get("kind", "")),
            "act": _int(node.get("act")),
            "floor_bucket": _bucket(_float(node.get("floor")), 3),
        }
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def state_action_key(observation: Mapping[str, Any], action_descriptor: Mapping[str, Any]) -> str:
    """Return stable key used by the lightweight Q table."""

    return f"{state_signature(observation)}::{action_signature(action_descriptor)}"


def _ratio_bucket(value: float, maximum: float) -> int:
    maximum = max(1.0, maximum)
    return max(0, min(10, int((value / maximum) * 10)))


def _bucket(value: float, size: int) -> int:
    return int(max(0.0, value) // max(1, size))


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _float(value: object) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _int(value: object) -> int:
    return int(_float(value))
