"""Masked random baseline for self-learning experiments."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any


class MaskedRandomAgent:
    """Choose uniformly from currently legal action IDs."""

    def __init__(self, seed: int | str = 0) -> None:
        self._rng = random.Random(str(seed))

    def choose_action_id(
        self,
        observation: Mapping[str, Any],
        info: Mapping[str, Any] | None = None,
    ) -> int | None:
        """Choose a legal action ID from an observation/action mask."""

        del info
        legal_ids = legal_action_ids(observation)
        if not legal_ids:
            return None
        return self._rng.choice(legal_ids)


def legal_action_ids(observation: Mapping[str, Any]) -> tuple[int, ...]:
    """Return legal action IDs from either fixed mask or structured ids."""

    mask = observation.get("action_mask")
    if isinstance(mask, Sequence) and not isinstance(mask, (str, bytes, bytearray)):
        ids = tuple(index for index, value in enumerate(mask) if value == 1)
        if ids:
            return ids
    legal = observation.get("legal_actions")
    if isinstance(legal, Mapping):
        raw_ids = legal.get("ids")
        if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes, bytearray)):
            return tuple(_int(value) for value in raw_ids)
    return ()


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0
