"""Optional Gymnasium environment with fixed-shape numeric observations.

Import this module after installing ``sts2sim[rl]``. The lightweight
``sts2sim.Sts2Env`` interface remains available without Gymnasium or NumPy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

try:
    import gymnasium as gym
    import numpy as np
    from numpy.typing import NDArray
except ImportError as exc:
    raise ImportError(
        "GymnasiumSts2Env requires RL dependencies; install sts2sim[rl] "
        "or run `uv sync --extra rl`."
    ) from exc

from sts2sim.agent_api import DEFAULT_MAX_ACTIONS, OBSERVATION_VECTOR_SCHEMA
from sts2sim.gymnasium_env import RewardFn, Sts2Env
from sts2sim.learning.rewards import learning_reward

NumericObservation = dict[str, NDArray[Any]]


class GymnasiumSts2Env(gym.Env[NumericObservation, int]):
    """Expose the simulator to Gymnasium and action-mask-aware RL libraries.

    Observations contain only ``vector`` (float32) and ``action_mask`` (int8).
    The complete structured observation is retained in ``info['rich_observation']``;
    state-local action descriptors remain in ``info['action_space']``.
    ``action_masks()`` supports libraries such as sb3-contrib MaskablePPO.

    Learning rewards are enabled by default. Actions must be selected with the
    current mask: padded or otherwise invalid actions raise ``ValueError``.
    Call ``reset`` before the first step and after termination or truncation.
    """

    metadata: dict[str, Any] = {"render_modes": []}
    render_mode = None

    def __init__(
        self,
        *,
        seed: int | str = 0,
        character_id: str = "IRONCLAD",
        ascension: int = 0,
        source_data: Any | None = None,
        max_actions: int = DEFAULT_MAX_ACTIONS,
        max_episode_steps: int | None = None,
        reward_fn: RewardFn | None = None,
        include_serialized_state: bool = False,
    ) -> None:
        super().__init__()
        if max_episode_steps is not None and max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self.simulator = Sts2Env(
            seed=seed,
            character_id=character_id,
            ascension=ascension,
            source_data=source_data,
            max_actions=max_actions,
            max_episode_steps=max_episode_steps,
            reward_fn=learning_reward if reward_fn is None else reward_fn,
            include_serialized_state=include_serialized_state,
        )
        self.action_space = gym.spaces.Discrete(max_actions)
        self.observation_space = gym.spaces.Dict(
            {
                "vector": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(OBSERVATION_VECTOR_SCHEMA),),
                    dtype=np.float32,
                ),
                "action_mask": gym.spaces.MultiBinary(max_actions),
            }
        )
        self._initial_seed = seed
        self._has_reset = False
        self._episode_done = False
        self._action_mask = np.zeros(max_actions, dtype=np.bool_)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NumericObservation, dict[str, Any]]:
        """Reset deterministically; unseeded later resets draw new run seeds."""

        first_reset = not self._has_reset
        gym_seed = seed
        if first_reset and gym_seed is None:
            initial = self._initial_seed
            gym_seed = (
                initial
                if isinstance(initial, int)
                else int.from_bytes(hashlib.sha256(initial.encode()).digest()[:8], "big")
            )
        super().reset(seed=gym_seed)
        run_seed = seed
        if seed is None and not first_reset:
            run_seed = int(self.np_random.integers(0, 2**63))
        rich, info = self.simulator.reset(seed=run_seed, options=options)
        self._has_reset = True
        self._episode_done = False
        return self._encode_result(rich, info)

    def step(
        self,
        action: int,
    ) -> tuple[NumericObservation, float, bool, bool, dict[str, Any]]:
        """Apply a currently legal action without silently coercing its ID."""

        if not self._has_reset or self._episode_done:
            raise gym.error.ResetNeeded("Call reset() before starting or continuing an episode.")
        if (
            isinstance(action, bool | np.bool_)
            or not self.action_space.contains(action)
            or not self._action_mask[int(action)]
        ):
            raise ValueError(f"Action {action!r} is not legal for the current state.")
        rich, reward, terminated, truncated, info = self.simulator.step(int(action))
        self._episode_done = terminated or truncated
        observation, result_info = self._encode_result(rich, info)
        return observation, reward, terminated, truncated, result_info

    def action_masks(self) -> NDArray[np.bool_]:
        """Return an independent boolean mask for the current state."""

        if not self._has_reset:
            raise gym.error.ResetNeeded("Call reset() before requesting legal actions.")
        return self._action_mask.copy()

    def close(self) -> None:
        self.simulator.close()

    def _encode_result(
        self,
        rich: Mapping[str, Any],
        info: Mapping[str, Any],
    ) -> tuple[NumericObservation, dict[str, Any]]:
        observation: NumericObservation = {
            "vector": np.asarray(rich["vector"], dtype=np.float32),
            "action_mask": np.asarray(rich["action_mask"], dtype=np.int8),
        }
        self._action_mask = observation["action_mask"].astype(np.bool_)
        return observation, {**info, "rich_observation": dict(rich)}
