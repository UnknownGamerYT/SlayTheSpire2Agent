"""Evaluation helpers for self-learning agents."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from sts2sim.gymnasium_env import Sts2Env
from sts2sim.learning.models import LearningBatchResult, LearningRunResult
from sts2sim.learning.progress import progress_from_runs
from sts2sim.learning.q_learning import QLearningAgent, load_q_learning_model
from sts2sim.learning.random_agent import MaskedRandomAgent
from sts2sim.learning.rewards import learning_reward


class _LearningPolicy(Protocol):
    def choose_action_id(
        self,
        observation: dict[str, object],
        info: dict[str, object],
    ) -> int | None:
        """Choose a state-local action id."""


def evaluate_learning_agent(
    *,
    policy: str = "random",
    model_path: Path | str | None = None,
    runs: int = 10,
    max_steps: int = 500,
    start_seed: int = 0,
    character_id: str = "TEST",
    ascension: int = 0,
) -> LearningBatchResult:
    """Evaluate a self-learning policy over fixed simulator seeds."""

    run_results: list[LearningRunResult] = []
    for run_index in range(max(0, runs)):
        run_results.append(
            _evaluate_one(
                policy=policy,
                model_path=model_path,
                run_index=run_index,
                seed=start_seed + run_index,
                max_steps=max_steps,
                character_id=character_id,
                ascension=ascension,
            )
        )
    completed = len(run_results)
    return LearningBatchResult(
        policy=policy,
        runs_requested=runs,
        runs_completed=completed,
        total_steps=sum(run.steps_taken for run in run_results),
        average_reward=round(
            sum(run.total_reward for run in run_results) / max(1, completed),
            6,
        ),
        average_floor=round(
            sum(run.final_floor for run in run_results) / max(1, completed),
            6,
        ),
        wins=sum(1 for run in run_results if run.final_phase == "complete"),
        deaths=sum(1 for run in run_results if run.final_phase == "failed"),
        progress=progress_from_runs(run_results, policy=policy),
        runs=tuple(run_results),
    )


def _evaluate_one(
    *,
    policy: str,
    model_path: Path | str | None,
    run_index: int,
    seed: int,
    max_steps: int,
    character_id: str,
    ascension: int,
) -> LearningRunResult:
    env = Sts2Env(
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        max_episode_steps=max_steps,
        reward_fn=learning_reward,
        include_serialized_state=False,
    )
    if policy == "q_learning":
        if model_path is None:
            raise ValueError("model_path is required for q_learning evaluation")
        agent: _LearningPolicy = QLearningAgent(
            model=load_q_learning_model(model_path),
            seed=seed,
            epsilon=0.0,
        )
    elif policy == "random":
        agent = MaskedRandomAgent(seed=seed)
    else:
        raise ValueError("policy must be 'random' or 'q_learning'")

    observation, info = env.reset()
    total_reward = 0.0
    steps_taken = 0
    terminated = False
    truncated = False
    for _step_index in range(max_steps):
        action_id = agent.choose_action_id(observation, info)
        if action_id is None:
            break
        observation, reward, terminated, truncated, info = env.step(action_id)
        total_reward += reward
        steps_taken += 1
        if terminated or truncated:
            break
    env.close()
    return LearningRunResult(
        run_index=run_index,
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        policy=policy,
        steps_taken=steps_taken,
        total_reward=round(total_reward, 6),
        terminated=terminated,
        truncated=truncated,
        final_phase=str(observation.get("phase", "unknown")),
        final_act=_lookup_vector_int(observation, "act"),
        final_floor=_lookup_vector_int(observation, "floor"),
        steps=(),
    )


def _lookup_vector_int(observation: dict[str, object], field: str) -> int:
    schema = observation.get("vector_schema")
    vector = observation.get("vector")
    if isinstance(schema, list) and isinstance(vector, list):
        for index, name in enumerate(schema):
            if str(name) == field and index < len(vector):
                value = vector[index]
                if isinstance(value, int | float):
                    return int(value)
    return 0
