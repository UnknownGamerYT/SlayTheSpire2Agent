"""Lightweight Gymnasium-style adapter for the simulator."""

from __future__ import annotations

import importlib
import importlib.util
import random
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any, cast

from sts2sim.agent_api import (
    DEFAULT_MAX_ACTIONS,
    OBSERVATION_VECTOR_SCHEMA,
    action_mask,
    decode_action,
    encode_observation,
)
from sts2sim.agent_api import (
    action_space as agent_action_space,
)
from sts2sim.api import new_run
from sts2sim.api import step as step_state

RewardFn = Callable[[Any, Any], float]


class SimpleDiscrete:
    """Small fallback compatible with the bits of ``gymnasium.spaces.Discrete`` used here."""

    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = n

    def sample(self) -> int:
        return random.randrange(self.n)

    def contains(self, value: object) -> bool:
        if isinstance(value, bool):
            return False
        try:
            integer = int(cast(Any, value))
        except (TypeError, ValueError):
            return False
        return 0 <= integer < self.n


class SimpleBox:
    """Fallback metadata object for fixed-shape numeric observations."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def contains(self, value: object) -> bool:
        if not isinstance(value, list | tuple):
            return False
        return len(value) == self.shape[0]


class SimpleMultiBinary:
    """Fallback metadata object for fixed-width binary masks."""

    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError("n must be non-negative")
        self.n = n

    def contains(self, value: object) -> bool:
        if not isinstance(value, list | tuple) or len(value) != self.n:
            return False
        return all(item in {0, 1} for item in value)


class SimpleDict:
    """Fallback metadata object for dictionary observations."""

    def __init__(self, spaces: Mapping[str, Any]) -> None:
        self.spaces = dict(spaces)

    def contains(self, value: object) -> bool:
        return bool(isinstance(value, Mapping))


class Sts2Env:
    """A small Gymnasium-style environment without a hard Gymnasium dependency."""

    metadata: dict[str, list[str]] = {"render_modes": []}

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
        if max_actions <= 0:
            raise ValueError("max_actions must be positive")
        self.seed = seed
        self.character_id = character_id
        self.ascension = ascension
        self.source_data = source_data
        self.max_actions = max_actions
        self.max_episode_steps = max_episode_steps
        self.reward_fn = reward_fn or _zero_reward
        self.include_serialized_state = include_serialized_state
        self.state: Any | None = None
        self.steps = 0
        self._agent_memory: list[dict[str, Any]] = []
        self._pending_policy_output: dict[str, Any] = {}
        self._reward_tracker: Any | None = None
        self._reward_breakdown_fn: Callable[..., Any] | None = None
        try:
            from sts2sim.learning.rewards import (
                LearningRewardTracker,
                learning_reward,
                learning_reward_breakdown,
            )

            if self.reward_fn is learning_reward:
                self._reward_tracker = LearningRewardTracker()
                self._reward_breakdown_fn = learning_reward_breakdown
        except ImportError:
            self._reward_tracker = None
            self._reward_breakdown_fn = None
        self._last_reward_breakdown: dict[str, Any] = {}

        self._gymnasium = _optional_import("gymnasium")
        self._numpy = _optional_import("numpy") if self._gymnasium is not None else None
        self.using_gymnasium = self._gymnasium is not None and self._numpy is not None
        self.action_space, self.observation_space = self._make_spaces()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the run and return ``(observation, info)``."""

        options = options or {}
        run_seed = self.seed if seed is None else seed
        self.state = new_run(
            seed=run_seed,
            character_id=str(options.get("character_id", self.character_id)),
            ascension=int(options.get("ascension", self.ascension)),
            source_data=options.get("source_data", self.source_data),
        )
        self.steps = 0
        self._agent_memory = []
        self._pending_policy_output = {}
        if self._reward_tracker is not None:
            self._reward_tracker.reset()
        self._last_reward_breakdown = {}
        return self._observation(), self._info()

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply a discrete action ID and return the Gymnasium 5-tuple."""

        if self.state is None:
            self.reset()
        assert self.state is not None

        previous_state = self.state
        engine_action = decode_action(previous_state, action)
        previous_action_space = agent_action_space(previous_state)
        action_descriptor = _descriptor_for_action_id(previous_action_space, action)
        self.state = step_state(previous_state, engine_action)
        self.steps += 1

        terminated = _is_terminal(self.state)
        truncated = (
            self.max_episode_steps is not None
            and self.steps >= self.max_episode_steps
            and not terminated
        )
        reward_breakdown: dict[str, Any]
        if self._reward_tracker is not None and self._reward_breakdown_fn is not None:
            breakdown = self._reward_breakdown_fn(
                previous_state,
                self.state,
                tracker=self._reward_tracker,
                action_descriptor=action_descriptor,
            )
            reward = float(breakdown.total)
            reward_breakdown = breakdown.model_dump(mode="json")
        else:
            reward = float(self.reward_fn(previous_state, self.state))
            reward_breakdown = {"total": reward}
        self._last_reward_breakdown = reward_breakdown
        self._record_agent_memory(
            action_descriptor=action_descriptor,
            action_id=action,
            reward=reward,
            reward_breakdown=reward_breakdown,
            done=terminated or truncated,
        )
        return self._observation(), reward, terminated, truncated, self._info(engine_action)

    def render(self) -> None:
        """No-op render hook for Gymnasium compatibility."""

    def close(self) -> None:
        """No-op close hook for Gymnasium compatibility."""

    def set_pending_policy_output(self, payload: Mapping[str, Any] | None) -> None:
        """Attach policy diagnostics to the next memory entry."""

        self._pending_policy_output = dict(payload or {})

    def _make_spaces(self) -> tuple[Any, Any]:
        if self.using_gymnasium:
            gymnasium = cast(Any, self._gymnasium)
            spaces = gymnasium.spaces
            np = self._numpy
            assert np is not None
            action_space = spaces.Discrete(self.max_actions)
            observation_space = spaces.Dict(
                {
                    "vector": spaces.Box(
                        low=-float("inf"),
                        high=float("inf"),
                        shape=(len(OBSERVATION_VECTOR_SCHEMA),),
                        dtype=np.float32,
                    ),
                    "action_mask": spaces.MultiBinary(self.max_actions),
                }
            )
            return action_space, observation_space

        return (
            SimpleDiscrete(self.max_actions),
            SimpleDict(
                {
                    "vector": SimpleBox((len(OBSERVATION_VECTOR_SCHEMA),)),
                    "action_mask": SimpleMultiBinary(self.max_actions),
                }
            ),
        )

    def _observation(self) -> dict[str, Any]:
        assert self.state is not None
        observation = encode_observation(
            self.state,
            include_state=self.include_serialized_state,
            agent_memory={"entries": self._agent_memory},
        )
        observation["action_mask"] = list(action_mask(self.state, max_actions=self.max_actions))
        return observation

    def _info(self, action: Any | None = None) -> dict[str, Any]:
        assert self.state is not None
        descriptors = agent_action_space(self.state)
        info: dict[str, Any] = {
            "action_mask": list(action_mask(self.state, max_actions=self.max_actions)),
            "action_space": descriptors,
            "legal_action_count": len(descriptors),
            "agent_memory": {"entries": list(self._agent_memory)},
            "gymnasium_available": self.using_gymnasium,
            "reward_breakdown": dict(self._last_reward_breakdown),
        }
        if action is not None:
            info["action"] = (
                action.model_dump(mode="json", exclude_none=True)
                if hasattr(action, "model_dump")
                else action
            )
        return info

    def _record_agent_memory(
        self,
        *,
        action_descriptor: Mapping[str, Any],
        action_id: Any,
        reward: float,
        reward_breakdown: Mapping[str, Any],
        done: bool,
    ) -> None:
        preview = _mapping(action_descriptor.get("preview"))
        action_index = _to_int(action_id)
        policy = dict(self._pending_policy_output)
        self._pending_policy_output = {}
        entry = {
            "age": 0,
            "action_type": str(action_descriptor.get("type", "")),
            "action_type_id": _to_int(action_descriptor.get("action_type_id")),
            "action_id": _to_int(action_descriptor.get("id", action_index)),
            "action_index": _to_int(policy.get("action_index", action_index)),
            "confidence": _to_float(policy.get("confidence")),
            "log_prob": _to_float(policy.get("log_prob")),
            "value": _to_float(policy.get("value")),
            "reward": reward,
            "reward_aggression_pressure": _to_float(
                reward_breakdown.get("aggression_pressure")
            ),
            "reward_hp_loss_penalty": _to_float(reward_breakdown.get("hp_loss_penalty")),
            "reward_enemy_hp_progress": _to_float(
                reward_breakdown.get("enemy_hp_progress_reward")
            ),
            "reward_prevented_hp": _to_float(reward_breakdown.get("prevented_hp_reward")),
            "reward_gold": _to_float(reward_breakdown.get("gold_reward")),
            "reward_combat_win": _to_float(reward_breakdown.get("combat_win_reward"))
            + _to_float(reward_breakdown.get("boss_reward")),
            "hp_delta": _to_int(preview.get("player_hp_delta")),
            "block_delta": _to_int(preview.get("player_block_delta")),
            "energy_delta": _to_int(preview.get("player_energy_delta")),
            "gold_delta": _to_int(preview.get("player_gold_delta")),
            "floor_delta": _to_int(preview.get("floor_delta")),
            "phase_changed": _to_int(preview.get("phase_changed")),
            "target_hp_delta": _to_int(preview.get("target_hp_delta")),
            "monster_hp_total_delta": _to_int(preview.get("monster_hp_total_delta")),
            "kills": _to_int(preview.get("kills")),
            "incoming_damage_delta": _to_int(preview.get("incoming_damage_delta")),
            "preview_error": _to_int(preview.get("preview_error")),
            "done": int(done),
            "plan_aggression_target": _to_float(policy.get("aggression_target")),
            "plan_hp_floor": _to_float(policy.get("hp_floor")),
            "plan_hp_spend_budget": _to_float(policy.get("hp_spend_budget")),
            "plan_combat_pace": _to_float(policy.get("combat_pace")),
            "plan_route_preference": _to_float(policy.get("route_preference")),
            "plan_potion_policy": _to_float(policy.get("potion_policy")),
            "plan_reward_pickiness": _to_float(policy.get("reward_pickiness")),
            "plan_expected_hp_loss": _to_float(policy.get("expected_hp_loss")),
            "plan_expected_turns_to_kill": _to_float(policy.get("expected_turns_to_kill")),
            "plan_boss_readiness": _to_float(policy.get("boss_readiness")),
        }
        aged_entries = [
            {**entry, "age": min(index + 1, 99)}
            for index, entry in enumerate(self._agent_memory)
        ]
        self._agent_memory = [entry, *aged_entries][:4]


SlayTheSpire2Env = Sts2Env


def make_env(**kwargs: Any) -> Sts2Env:
    """Create a Gymnasium-style simulator environment."""

    return Sts2Env(**kwargs)


def gymnasium_available() -> bool:
    """Return whether Gymnasium can be imported in this environment."""

    return _optional_import("gymnasium") is not None


def _optional_import(module_name: str) -> ModuleType | None:
    if importlib.util.find_spec(module_name) is None:
        return None
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


def _zero_reward(_previous_state: Any, _next_state: Any) -> float:
    return 0.0


def _is_terminal(state: Any) -> bool:
    phase = getattr(getattr(state, "phase", None), "value", getattr(state, "phase", ""))
    return str(phase) in {"complete", "failed"}


def _descriptor_for_action_id(
    descriptors: list[dict[str, Any]],
    action_id: Any,
) -> Mapping[str, Any]:
    wanted = _to_int(action_id)
    for descriptor in descriptors:
        if _to_int(descriptor.get("id")) == wanted:
            return descriptor
    return {}


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _to_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _to_float(value: object) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
