"""Dependency-free masked Q-learning baseline."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sts2sim.gymnasium_env import Sts2Env
from sts2sim.learning.features import state_action_key
from sts2sim.learning.models import LearningProgressPoint, QLearningModel, TrainingResult
from sts2sim.learning.progress import (
    with_moving_averages,
    write_learning_progress_data,
    write_learning_progress_report,
)
from sts2sim.learning.random_agent import legal_action_ids
from sts2sim.learning.rewards import learning_reward


class QLearningAgent:
    """Small Q-learning agent keyed by observable state/action signatures."""

    def __init__(
        self,
        *,
        model: QLearningModel | None = None,
        seed: int | str = 0,
        epsilon: float = 0.1,
    ) -> None:
        self.model = model or QLearningModel()
        self.epsilon = max(0.0, min(1.0, float(epsilon)))
        self._rng = random.Random(str(seed))

    def choose_action_id(
        self,
        observation: Mapping[str, Any],
        info: Mapping[str, Any],
    ) -> int | None:
        """Choose a legal action using epsilon-greedy Q values."""

        legal_ids = legal_action_ids(observation)
        action_space = _action_space(info)
        if not legal_ids or not action_space:
            return None
        if self._rng.random() < self.epsilon:
            return self._rng.choice(legal_ids)

        candidates = [
            descriptor
            for descriptor in action_space
            if _int(descriptor.get("id")) in legal_ids
        ]
        if not candidates:
            return self._rng.choice(legal_ids)
        best = max(
            candidates,
            key=lambda descriptor: (
                self.model.q_values.get(state_action_key(observation, descriptor), 0.0),
                -_int(descriptor.get("id")),
            ),
        )
        return _int(best.get("id"))

    def update(
        self,
        observation: Mapping[str, Any],
        action_descriptor: Mapping[str, Any],
        reward: float,
        next_observation: Mapping[str, Any],
        next_info: Mapping[str, Any],
        *,
        alpha: float,
        gamma: float,
        terminal: bool,
    ) -> None:
        """Apply one Q-learning update."""

        key = state_action_key(observation, action_descriptor)
        current = self.model.q_values.get(key, 0.0)
        next_best = 0.0 if terminal else self._best_next_value(next_observation, next_info)
        updated = current + alpha * (float(reward) + gamma * next_best - current)
        q_values = dict(self.model.q_values)
        visits = dict(self.model.visits)
        q_values[key] = updated
        visits[key] = visits.get(key, 0) + 1
        self.model = self.model.model_copy(update={"q_values": q_values, "visits": visits})

    def save(self, path: Path | str) -> None:
        """Write the model checkpoint to JSON."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _best_next_value(
        self,
        observation: Mapping[str, Any],
        info: Mapping[str, Any],
    ) -> float:
        values = [
            self.model.q_values.get(state_action_key(observation, descriptor), 0.0)
            for descriptor in _action_space(info)
            if _int(descriptor.get("id")) in set(legal_action_ids(observation))
        ]
        return max(values) if values else 0.0


def train_q_learning(
    *,
    runs: int = 10,
    max_steps: int = 500,
    seed: int | str = 0,
    character_id: str = "TEST",
    ascension: int = 0,
    epsilon: float = 0.2,
    alpha: float = 0.1,
    gamma: float = 0.99,
    output_path: Path | str | None = Path("checkpoints/q_learning_latest.json"),
    progress_output_path: Path | str | None = None,
    report_output_path: Path | str | None = None,
    progress_window: int = 10,
) -> TrainingResult:
    """Train the dependency-free self-learning baseline in simulation."""

    agent = QLearningAgent(seed=seed, epsilon=epsilon)
    total_reward = 0.0
    total_steps = 0
    wins = 0
    deaths = 0
    progress_points: list[LearningProgressPoint] = []

    for run_index in range(max(0, runs)):
        run_seed = _run_seed(seed, run_index)
        env = Sts2Env(
            seed=run_seed,
            character_id=character_id,
            ascension=ascension,
            max_episode_steps=max_steps,
            reward_fn=learning_reward,
            include_serialized_state=False,
        )
        observation, info = env.reset()
        run_reward = 0.0
        run_steps = 0
        run_terminated = False
        run_truncated = False
        for _step_index in range(max_steps):
            action_id = agent.choose_action_id(observation, info)
            if action_id is None:
                break
            descriptor = _descriptor_for_action_id(info, action_id)
            next_observation, reward, terminated, truncated, next_info = env.step(action_id)
            terminal = terminated or truncated
            agent.update(
                observation,
                descriptor,
                reward,
                next_observation,
                next_info,
                alpha=alpha,
                gamma=gamma,
                terminal=terminal,
            )
            run_reward += reward
            run_steps += 1
            run_terminated = terminated
            run_truncated = truncated
            total_steps += 1
            observation = next_observation
            info = next_info
            if terminal:
                break
        total_reward += run_reward
        final_phase = str(observation.get("phase", "unknown"))
        wins += int(final_phase == "complete")
        deaths += int(final_phase == "failed")
        progress_points.append(
            LearningProgressPoint(
                run_index=run_index,
                seed=run_seed,
                policy="q_learning",
                steps_taken=run_steps,
                total_reward=round(run_reward, 6),
                final_phase=final_phase,
                final_act=_lookup_vector_int(observation, "act"),
                final_floor=_lookup_vector_int(observation, "floor"),
                win=final_phase == "complete",
                death=final_phase == "failed",
                truncated=run_truncated and not run_terminated,
            )
        )
        env.close()

    progress = with_moving_averages(progress_points, window=progress_window)
    model_path = str(output_path) if output_path is not None else None
    metadata = {
        "runs": runs,
        "max_steps": max_steps,
        "seed": seed,
        "character_id": character_id,
        "ascension": ascension,
        "epsilon": epsilon,
        "alpha": alpha,
        "gamma": gamma,
        "progress_window": progress_window,
    }
    agent.model = agent.model.model_copy(update={"metadata": metadata})
    if output_path is not None:
        agent.save(output_path)
    if progress_output_path is not None:
        write_learning_progress_data(
            progress,
            progress_output_path,
            title="Q-learning Training Progress",
            window=progress_window,
        )
    if report_output_path is not None:
        write_learning_progress_report(
            progress,
            report_output_path,
            title="Q-learning Training Progress",
            window=progress_window,
        )

    completed_runs = max(1, runs)
    return TrainingResult(
        algorithm="state_action_signature_q_learning",
        runs=runs,
        total_steps=total_steps,
        average_reward=round(total_reward / completed_runs, 6),
        wins=wins,
        deaths=deaths,
        model_path=model_path,
        progress_output_path=(
            str(progress_output_path) if progress_output_path is not None else None
        ),
        report_output_path=str(report_output_path) if report_output_path is not None else None,
        progress=progress,
        evaluation={},
    )


def load_q_learning_model(path: Path | str) -> QLearningModel:
    """Load a Q-learning checkpoint from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return QLearningModel.model_validate(payload)


def _action_space(info: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = info.get("action_space")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return tuple(item for item in raw if isinstance(item, Mapping))
    return ()


def _descriptor_for_action_id(info: Mapping[str, Any], action_id: int) -> Mapping[str, Any]:
    for descriptor in _action_space(info):
        if _int(descriptor.get("id")) == action_id:
            return descriptor
    return {"id": action_id, "action": {"type": "unknown"}}


def _run_seed(seed: int | str, run_index: int) -> int | str:
    if isinstance(seed, int):
        return seed + run_index
    try:
        return int(seed) + run_index
    except ValueError:
        return f"{seed}:{run_index}"


def _lookup_vector_int(observation: Mapping[str, Any], field: str) -> int:
    schema = observation.get("vector_schema")
    vector = observation.get("vector")
    if isinstance(schema, list | tuple) and isinstance(vector, list | tuple):
        for index, name in enumerate(schema):
            if str(name) == field and index < len(vector):
                return _int(vector[index])
    return 0


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
