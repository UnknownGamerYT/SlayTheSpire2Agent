"""Rollout collection for self-learning agents."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sts2sim.gymnasium_env import Sts2Env
from sts2sim.history import (
    RunHistory,
    append_history_step,
    record_history_step,
    start_run_history,
)
from sts2sim.learning.models import LearningBatchResult, LearningRunResult, LearningStep
from sts2sim.learning.observation import encode_rich_observation
from sts2sim.learning.progress import progress_from_runs
from sts2sim.learning.random_agent import MaskedRandomAgent
from sts2sim.learning.rewards import learning_reward


def collect_random_rollouts(
    *,
    runs: int = 1,
    max_steps: int = 500,
    start_seed: int = 0,
    character_id: str = "TEST",
    ascension: int = 0,
    output_path: Path | str | None = None,
    include_steps: bool = True,
    observation_mode: str = "rich",
    include_history: bool = True,
) -> LearningBatchResult:
    """Collect rollouts from the masked random baseline."""

    observation_mode = _normalize_observation_mode(observation_mode)
    run_results: list[LearningRunResult] = []
    for run_index in range(max(0, runs)):
        run_results.append(
            collect_random_rollout(
                run_index=run_index,
                seed=start_seed + run_index,
                max_steps=max_steps,
                character_id=character_id,
                ascension=ascension,
                include_steps=include_steps,
                observation_mode=observation_mode,
                include_history=include_history,
            )
        )

    result = _batch_result(
        policy="masked_random",
        runs_requested=runs,
        runs=tuple(run_results),
        output_path=str(output_path) if output_path is not None else None,
    )
    if output_path is not None:
        _write_batch_result(result, output_path)
    return result


def collect_random_rollout(
    *,
    run_index: int = 0,
    seed: int | str = 0,
    max_steps: int = 500,
    character_id: str = "TEST",
    ascension: int = 0,
    include_steps: bool = True,
    observation_mode: str = "rich",
    include_history: bool = True,
) -> LearningRunResult:
    """Collect one masked-random simulator rollout."""

    observation_mode = _normalize_observation_mode(observation_mode)
    env = Sts2Env(
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        max_episode_steps=max_steps,
        reward_fn=learning_reward,
        include_serialized_state=False,
    )
    agent = MaskedRandomAgent(seed=seed)
    observation, info = env.reset()
    history: RunHistory | None = (
        start_run_history(env.state, policy="masked_random") if include_history else None
    )
    steps: list[LearningStep] = []
    total_reward = 0.0
    terminated = False
    truncated = False
    steps_taken = 0

    for step_index in range(max_steps):
        action_id = agent.choose_action_id(observation, info)
        if action_id is None:
            break
        action = _action_for_id(info, action_id)
        phase_before = str(observation.get("phase", "unknown"))
        before_state = env.state
        next_observation, reward, terminated, truncated, info = env.step(action_id)
        total_reward += reward
        if history is not None and before_state is not None and env.state is not None:
            history = append_history_step(
                history,
                record_history_step(
                    step_index=step_index,
                    before_state=before_state,
                    action=action,
                    after_state=env.state,
                    reward=reward,
                ),
                env.state,
            )
        if include_steps:
            stored_observation = _stored_observation(
                env.state,
                observation,
                observation_mode,
            )
            steps.append(
                LearningStep(
                    run_index=run_index,
                    step_index=step_index,
                    seed=seed,
                    action_id=action_id,
                    action=action,
                    reward=round(float(reward), 6),
                    terminated=terminated,
                    truncated=truncated,
                    phase_before=phase_before,
                    phase_after=str(next_observation.get("phase", "unknown")),
                    vector=tuple(
                        _float(value) for value in _sequence(observation.get("vector"))
                    ),
                    action_mask=tuple(
                        _int(value) for value in _sequence(observation.get("action_mask"))
                    ),
                    observation_mode=observation_mode,
                    observation=stored_observation,
                )
            )
        observation = next_observation
        steps_taken += 1
        if terminated or truncated:
            break

    env.close()
    return LearningRunResult(
        run_index=run_index,
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        policy="masked_random",
        steps_taken=steps_taken,
        total_reward=round(total_reward, 6),
        terminated=terminated,
        truncated=truncated,
        final_phase=str(observation.get("phase", "unknown")),
        final_act=_int(_lookup_vector(observation, "act")),
        final_floor=_int(_lookup_vector(observation, "floor")),
        history=history.model_dump(mode="json") if history is not None else None,
        steps=tuple(steps),
    )


def _batch_result(
    *,
    policy: str,
    runs_requested: int,
    runs: tuple[LearningRunResult, ...],
    output_path: str | None,
) -> LearningBatchResult:
    total_steps = sum(run.steps_taken for run in runs)
    total_reward = sum(run.total_reward for run in runs)
    completed = len(runs)
    return LearningBatchResult(
        policy=policy,
        runs_requested=runs_requested,
        runs_completed=completed,
        total_steps=total_steps,
        average_reward=round(total_reward / max(1, completed), 6),
        average_floor=round(
            sum(run.final_floor for run in runs) / max(1, completed),
            6,
        ),
        wins=sum(1 for run in runs if run.final_phase == "complete"),
        deaths=sum(1 for run in runs if run.final_phase == "failed"),
        output_path=output_path,
        progress=progress_from_runs(runs, policy=policy),
        runs=runs,
    )


def _write_batch_result(result: LearningBatchResult, output_path: Path | str) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stored_observation(
    state: Any,
    compact_observation: Mapping[str, Any],
    observation_mode: str,
) -> dict[str, Any] | None:
    if observation_mode == "compact":
        return None
    if state is None:
        return dict(compact_observation)
    return encode_rich_observation(state)


def _normalize_observation_mode(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in {"compact", "rich"}:
        raise ValueError("observation_mode must be 'compact' or 'rich'")
    return normalized


def _action_for_id(info: Mapping[str, Any], action_id: int) -> dict[str, Any]:
    action_space = info.get("action_space")
    if isinstance(action_space, list):
        for descriptor in action_space:
            if isinstance(descriptor, Mapping) and descriptor.get("id") == action_id:
                action = descriptor.get("action")
                return dict(action) if isinstance(action, Mapping) else {}
    return {}


def _lookup_vector(observation: Mapping[str, Any], field: str) -> object:
    schema = _sequence(observation.get("vector_schema"))
    vector = _sequence(observation.get("vector"))
    for index, name in enumerate(schema):
        if str(name) == field and index < len(vector):
            return vector[index]
    return 0


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return ()


def _int(value: object) -> int:
    if value is None or isinstance(value, bool):
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
