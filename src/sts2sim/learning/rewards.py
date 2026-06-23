"""Sparse reward functions for self-learning agents.

The defaults intentionally avoid tactical advice. They reward external run
progress, terminal victory, death avoidance, and a tiny step penalty.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from sts2sim.api import serialize


class LearningRewardConfig(BaseModel):
    """Sparse reward configuration for self-learning runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    floor_reward: float = 1.0
    act_reward: float = 10.0
    win_reward: float = 100.0
    death_penalty: float = -100.0
    step_penalty: float = -0.01


DEFAULT_REWARD_CONFIG = LearningRewardConfig()


def learning_reward(
    previous_state: Any,
    next_state: Any,
    config: LearningRewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    """Return sparse self-learning reward for one simulator transition."""

    previous = serialize(previous_state)
    current = serialize(next_state)
    reward = config.step_penalty

    reward += max(0, _int(current.get("floor")) - _int(previous.get("floor"))) * config.floor_reward
    reward += max(0, _int(current.get("act")) - _int(previous.get("act"))) * config.act_reward

    previous_phase = str(previous.get("phase", ""))
    current_phase = str(current.get("phase", ""))
    if current_phase == "complete" and previous_phase != "complete":
        reward += config.win_reward
    if current_phase == "failed" and previous_phase != "failed":
        reward += config.death_penalty
    return float(reward)


def terminal_stats(state: Any) -> dict[str, Any]:
    """Return compact terminal/evaluation fields for a simulator state."""

    payload = serialize(state)
    return {
        "phase": str(payload.get("phase", "unknown")),
        "act": _int(payload.get("act")),
        "floor": _int(payload.get("floor")),
        "won": str(payload.get("phase")) == "complete",
        "dead": str(payload.get("phase")) == "failed",
    }


def _int(value: object, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    if isinstance(value, Mapping):
        return default
    return default
