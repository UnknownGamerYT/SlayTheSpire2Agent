"""Train self-learning agents until a run target is reached."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from random import Random
from typing import Any

from sts2sim.api import new_run
from sts2sim.engine import RoomKind
from sts2sim.gymnasium_env import Sts2Env
from sts2sim.learning.models import (
    LearningProgressPoint,
    LearningRunResult,
    QLearningModel,
    TrainingTargetBatchSummary,
    TrainingUntilTargetResult,
)
from sts2sim.learning.progress import (
    with_moving_averages,
    write_learning_progress_data,
    write_learning_progress_report,
)
from sts2sim.learning.q_learning import QLearningAgent, load_q_learning_model
from sts2sim.learning.rewards import learning_reward


def train_q_learning_until_boss(
    *,
    max_batches: int = 5,
    batch_runs: int = 50,
    train_max_steps: int = 500,
    eval_runs: int = 5,
    eval_max_steps: int = 500,
    seed: int | str = 0,
    eval_start_seed: int = 10_000,
    character_id: str = "TEST",
    ascension: int = 0,
    epsilon: float = 0.2,
    alpha: float = 0.1,
    gamma: float = 0.99,
    target_act: int = 1,
    target_floor: int | None = None,
    target_reward: float = 100.0,
    success_replay_passes: int = 0,
    train_seed_mode: str = "sequential",
    eval_seed_mode: str = "sequential",
    target_eval_successes: int = 1,
    resume: bool = False,
    resume_from_path: Path | str | None = None,
    model_output_path: Path | str | None = Path("checkpoints/q_learning_until_boss.json"),
    output_path: Path | str | None = Path("reports/q_learning_until_boss_latest.json"),
    progress_output_path: Path | str | None = Path("reports/q_learning_until_boss_progress.json"),
    report_output_path: Path | str | None = Path("reports/q_learning_until_boss_latest.html"),
    progress_window: int = 10,
) -> TrainingUntilTargetResult:
    """Train in batches and stop once evaluation reaches the Act boss floor."""

    normalized_max_batches = max(0, int(max_batches))
    normalized_batch_runs = max(0, int(batch_runs))
    normalized_eval_runs = max(0, int(eval_runs))
    normalized_train_seed_mode = _normalize_seed_mode(train_seed_mode)
    normalized_eval_seed_mode = _normalize_seed_mode(eval_seed_mode)
    normalized_target_eval_successes = max(1, int(target_eval_successes))
    resolved_target_floor = (
        int(target_floor)
        if target_floor is not None
        else _boss_floor_for_new_run(
            seed=seed,
            character_id=character_id,
            ascension=ascension,
            target_act=target_act,
        )
    )
    model, resumed_from = _load_resume_model(
        resume=resume,
        resume_from_path=resume_from_path,
        model_output_path=model_output_path,
    )
    agent = QLearningAgent(model=model, seed=seed, epsilon=epsilon)
    train_seed_rng = Random(f"{seed}:train-seeds")
    eval_seed_rng = Random(f"{seed}:eval-seeds")

    training_points: list[LearningProgressPoint] = []
    evaluation_points: list[LearningProgressPoint] = []
    batch_summaries: list[TrainingTargetBatchSummary] = []
    total_reward = 0.0
    total_steps = 0
    runs_trained = 0
    reached_batch: int | None = None

    for batch_index in range(1, normalized_max_batches + 1):
        for _inner_index in range(normalized_batch_runs):
            run_index = runs_trained
            run_seed = _training_seed(
                seed,
                run_index,
                mode=normalized_train_seed_mode,
                rng=train_seed_rng,
            )
            point = _train_one_run(
                agent=agent,
                run_index=run_index,
                run_seed=run_seed,
                max_steps=train_max_steps,
                character_id=character_id,
                ascension=ascension,
                alpha=alpha,
                gamma=gamma,
                target_act=target_act,
                target_floor=resolved_target_floor,
                target_reward=target_reward,
                success_replay_passes=success_replay_passes,
            )
            training_points.append(point)
            total_reward += point.total_reward
            total_steps += point.steps_taken
            runs_trained += 1

        eval_seeds = tuple(
            _evaluation_seed(
                eval_start_seed,
                ((batch_index - 1) * normalized_eval_runs) + eval_index,
                mode=normalized_eval_seed_mode,
                rng=eval_seed_rng,
            )
            for eval_index in range(normalized_eval_runs)
        )
        eval_results = _evaluate_current_model(
            model=agent.model,
            batch_index=batch_index,
            eval_seeds=eval_seeds,
            eval_max_steps=eval_max_steps,
            character_id=character_id,
            ascension=ascension,
            target_act=target_act,
            target_floor=resolved_target_floor,
        )
        evaluation_points.extend(_progress_from_eval_runs(eval_results))
        target_successes = sum(
            _run_reached_target(
                run,
                target_act=target_act,
                target_floor=resolved_target_floor,
            )
            for run in eval_results
        )
        batch_reached = target_successes >= normalized_target_eval_successes
        if batch_reached and reached_batch is None:
            reached_batch = batch_index
        batch_summaries.append(
            _batch_summary(
                batch_index=batch_index,
                trained_runs_total=runs_trained,
                train_total_steps=total_steps,
                eval_results=eval_results,
                reached_target=batch_reached,
                target_successes=target_successes,
            )
        )
        result = _persist_training_target(
            agent=agent,
            target_act=target_act,
            target_floor=resolved_target_floor,
            reached_batch=reached_batch,
            max_batches=normalized_max_batches,
            batch_runs=normalized_batch_runs,
            runs_trained=runs_trained,
            total_steps=total_steps,
            total_reward=total_reward,
            resumed_from=resumed_from,
            model_output_path=model_output_path,
            output_path=output_path,
            progress_output_path=progress_output_path,
            report_output_path=report_output_path,
            progress_window=progress_window,
            training_points=training_points,
            evaluation_points=evaluation_points,
            batch_summaries=batch_summaries,
            metadata={
                "mode": "train_until_boss",
                "max_batches": max_batches,
                "batch_runs": batch_runs,
                "train_max_steps": train_max_steps,
                "eval_runs": eval_runs,
                "eval_max_steps": eval_max_steps,
                "seed": seed,
                "eval_start_seed": eval_start_seed,
                "character_id": character_id,
                "ascension": ascension,
                "epsilon": epsilon,
                "alpha": alpha,
                "gamma": gamma,
                "target_act": target_act,
                "target_floor": resolved_target_floor,
                "target_reward": target_reward,
                "success_replay_passes": success_replay_passes,
                "train_seed_mode": normalized_train_seed_mode,
                "eval_seed_mode": normalized_eval_seed_mode,
                "target_eval_successes": normalized_target_eval_successes,
            },
        )
        if batch_reached:
            return result
    result = _persist_training_target(
        agent=agent,
        target_act=target_act,
        target_floor=resolved_target_floor,
        reached_batch=reached_batch,
        max_batches=normalized_max_batches,
        batch_runs=normalized_batch_runs,
        runs_trained=runs_trained,
        total_steps=total_steps,
        total_reward=total_reward,
        resumed_from=resumed_from,
        model_output_path=model_output_path,
        output_path=output_path,
        progress_output_path=progress_output_path,
        report_output_path=report_output_path,
        progress_window=progress_window,
        training_points=training_points,
        evaluation_points=evaluation_points,
        batch_summaries=batch_summaries,
        metadata={
            "mode": "train_until_boss",
            "max_batches": max_batches,
            "batch_runs": batch_runs,
            "train_max_steps": train_max_steps,
            "eval_runs": eval_runs,
            "eval_max_steps": eval_max_steps,
            "seed": seed,
            "eval_start_seed": eval_start_seed,
            "character_id": character_id,
            "ascension": ascension,
            "epsilon": epsilon,
            "alpha": alpha,
            "gamma": gamma,
            "target_act": target_act,
            "target_floor": resolved_target_floor,
            "target_reward": target_reward,
            "success_replay_passes": success_replay_passes,
            "train_seed_mode": normalized_train_seed_mode,
            "eval_seed_mode": normalized_eval_seed_mode,
            "target_eval_successes": normalized_target_eval_successes,
        },
    )
    return result


def _train_one_run(
    *,
    agent: QLearningAgent,
    run_index: int,
    run_seed: int | str,
    max_steps: int,
    character_id: str,
    ascension: int,
    alpha: float,
    gamma: float,
    target_act: int,
    target_floor: int,
    target_reward: float,
    success_replay_passes: int,
) -> LearningProgressPoint:
    env: Sts2Env | None = None
    observation: dict[str, Any] = {}
    info: dict[str, Any] = {}
    run_reward = 0.0
    run_steps = 0
    terminated = False
    truncated = False
    error: str | None = None
    failed_to_continue = False
    reached_target_run = False
    trajectory: list[
        tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            float,
            Mapping[str, Any],
            Mapping[str, Any],
            bool,
        ]
    ] = []
    try:
        env = Sts2Env(
            seed=run_seed,
            character_id=character_id,
            ascension=ascension,
            max_episode_steps=max_steps,
            reward_fn=learning_reward,
            include_serialized_state=False,
        )
        observation, info = env.reset()
        for _step_index in range(max_steps):
            action_id = agent.choose_action_id(observation, info)
            if action_id is None:
                failed_to_continue = True
                error = "No legal action id was available before the run reached a terminal phase."
                break
            descriptor = _descriptor_for_action_id(info, action_id)
            next_observation, reward, terminated, truncated, next_info = env.step(action_id)
            reached_target = _observation_reached_target(
                next_observation,
                target_act=target_act,
                target_floor=target_floor,
            )
            effective_reward = reward + (float(target_reward) if reached_target else 0.0)
            terminal = terminated or truncated or reached_target
            trajectory.append(
                (
                    observation,
                    descriptor,
                    effective_reward,
                    next_observation,
                    next_info,
                    terminal,
                )
            )
            agent.update(
                observation,
                descriptor,
                effective_reward,
                next_observation,
                next_info,
                alpha=alpha,
                gamma=gamma,
                terminal=terminal,
            )
            run_reward += effective_reward
            run_steps += 1
            observation = next_observation
            info = next_info
            reached_target_run = reached_target_run or reached_target
            if terminal:
                break
        if reached_target_run and success_replay_passes > 0:
            _replay_success_trajectory(
                agent,
                trajectory,
                alpha=alpha,
                gamma=gamma,
                passes=success_replay_passes,
            )
    except Exception as exc:  # pragma: no cover - exercised by integration faults
        failed_to_continue = True
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()

    final_phase = str(observation.get("phase", "unknown"))
    if final_phase in {"complete", "failed"} and error is not None:
        failed_to_continue = False
    return LearningProgressPoint(
        run_index=run_index,
        seed=run_seed,
        policy="q_learning_train",
        steps_taken=run_steps,
        total_reward=round(run_reward, 6),
        final_phase=final_phase,
        final_act=_lookup_vector_int(observation, "act"),
        final_floor=_lookup_vector_int(observation, "floor"),
        win=final_phase == "complete",
        death=final_phase == "failed",
        truncated=truncated and not terminated,
        failed_to_continue=failed_to_continue,
        error=error,
    )


def _replay_success_trajectory(
    agent: QLearningAgent,
    trajectory: Sequence[
        tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            float,
            Mapping[str, Any],
            Mapping[str, Any],
            bool,
        ]
    ],
    *,
    alpha: float,
    gamma: float,
    passes: int,
) -> None:
    """Replay a successful target-reaching run backward to propagate sparse rewards."""

    if not trajectory or passes <= 0:
        return
    for _pass_index in range(passes):
        for (
            observation,
            descriptor,
            reward,
            next_observation,
            next_info,
            terminal,
        ) in reversed(trajectory):
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


def _evaluate_current_model(
    *,
    model: QLearningModel,
    batch_index: int,
    eval_seeds: Sequence[int],
    eval_max_steps: int,
    character_id: str,
    ascension: int,
    target_act: int,
    target_floor: int,
) -> tuple[LearningRunResult, ...]:
    results: list[LearningRunResult] = []
    for eval_index, seed in enumerate(eval_seeds):
        global_index = ((batch_index - 1) * len(eval_seeds)) + eval_index
        results.append(
            _evaluate_one_current_model(
                model=model,
                run_index=global_index,
                seed=seed,
                max_steps=eval_max_steps,
                character_id=character_id,
                ascension=ascension,
                target_act=target_act,
                target_floor=target_floor,
            )
        )
    return tuple(results)


def _evaluate_one_current_model(
    *,
    model: QLearningModel,
    run_index: int,
    seed: int,
    max_steps: int,
    character_id: str,
    ascension: int,
    target_act: int,
    target_floor: int,
) -> LearningRunResult:
    env: Sts2Env | None = None
    observation: dict[str, Any] = {}
    info: dict[str, Any] = {}
    total_reward = 0.0
    steps_taken = 0
    terminated = False
    truncated = False
    error: str | None = None
    failed_to_continue = False
    try:
        env = Sts2Env(
            seed=seed,
            character_id=character_id,
            ascension=ascension,
            max_episode_steps=max_steps,
            reward_fn=learning_reward,
            include_serialized_state=False,
        )
        agent = QLearningAgent(model=model, seed=seed, epsilon=0.0)
        observation, info = env.reset()
        for _step_index in range(max_steps):
            action_id = agent.choose_action_id(observation, info)
            if action_id is None:
                failed_to_continue = True
                error = "No legal action id was available before the run reached a terminal phase."
                break
            observation, reward, terminated, truncated, info = env.step(action_id)
            total_reward += reward
            steps_taken += 1
            if terminated or truncated:
                break
            if _observation_reached_target(
                observation,
                target_act=target_act,
                target_floor=target_floor,
            ):
                break
    except Exception as exc:  # pragma: no cover - exercised by integration faults
        failed_to_continue = True
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()

    final_phase = str(observation.get("phase", "unknown"))
    if final_phase in {"complete", "failed"} and error is not None:
        failed_to_continue = False
    return LearningRunResult(
        run_index=run_index,
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        policy="q_learning_eval",
        steps_taken=steps_taken,
        total_reward=round(total_reward, 6),
        terminated=terminated,
        truncated=truncated,
        final_phase=final_phase,
        final_act=_lookup_vector_int(observation, "act"),
        final_floor=_lookup_vector_int(observation, "floor"),
        error=error,
        failed_to_continue=failed_to_continue,
        steps=(),
    )


def _progress_from_eval_runs(
    runs: Sequence[LearningRunResult],
) -> tuple[LearningProgressPoint, ...]:
    return tuple(
        LearningProgressPoint(
            run_index=run.run_index,
            seed=run.seed,
            policy=run.policy,
            steps_taken=run.steps_taken,
            total_reward=run.total_reward,
            final_phase=run.final_phase,
            final_act=run.final_act,
            final_floor=run.final_floor,
            win=run.final_phase == "complete",
            death=run.final_phase == "failed",
            truncated=run.truncated and not run.terminated,
            failed_to_continue=run.failed_to_continue,
            error=run.error,
        )
        for run in runs
    )


def _batch_summary(
    *,
    batch_index: int,
    trained_runs_total: int,
    train_total_steps: int,
    eval_results: Sequence[LearningRunResult],
    reached_target: bool,
    target_successes: int,
) -> TrainingTargetBatchSummary:
    completed = len(eval_results)
    return TrainingTargetBatchSummary(
        batch_index=batch_index,
        trained_runs_total=trained_runs_total,
        train_total_steps=train_total_steps,
        evaluation_runs=completed,
        evaluation_average_reward=round(
            sum(run.total_reward for run in eval_results) / max(1, completed),
            6,
        ),
        evaluation_average_floor=round(
            sum(run.final_floor for run in eval_results) / max(1, completed),
            6,
        ),
        evaluation_best_floor=max((run.final_floor for run in eval_results), default=0),
        evaluation_best_reward=round(
            max((run.total_reward for run in eval_results), default=0.0),
            6,
        ),
        evaluation_errors=sum(1 for run in eval_results if run.error is not None),
        evaluation_failed_to_continue=sum(
            1 for run in eval_results if run.failed_to_continue
        ),
        reached_target=reached_target,
        evaluation_target_successes=target_successes,
        evaluation_target_success_rate=round(target_successes / max(1, completed), 6),
    )


def _persist_training_target(
    *,
    agent: QLearningAgent,
    target_act: int,
    target_floor: int,
    reached_batch: int | None,
    max_batches: int,
    batch_runs: int,
    runs_trained: int,
    total_steps: int,
    total_reward: float,
    resumed_from: str | None,
    model_output_path: Path | str | None,
    output_path: Path | str | None,
    progress_output_path: Path | str | None,
    report_output_path: Path | str | None,
    progress_window: int,
    training_points: Sequence[LearningProgressPoint],
    evaluation_points: Sequence[LearningProgressPoint],
    batch_summaries: Sequence[TrainingTargetBatchSummary],
    metadata: Mapping[str, Any],
) -> TrainingUntilTargetResult:
    progress = with_moving_averages(training_points, window=progress_window)
    evaluation_progress = with_moving_averages(evaluation_points, window=progress_window)
    enriched_metadata = {
        **dict(metadata),
        "reached_target": reached_batch is not None,
        "reached_batch": reached_batch,
        "runs_trained": runs_trained,
        "total_steps": total_steps,
        "batches_completed": len(batch_summaries),
    }
    agent.model = agent.model.model_copy(update={"metadata": enriched_metadata})
    if model_output_path is not None:
        agent.save(model_output_path)
    if progress_output_path is not None:
        write_learning_progress_data(
            progress,
            progress_output_path,
            title="Q-learning Until Act Boss",
            window=progress_window,
        )
    if report_output_path is not None:
        write_learning_progress_report(
            progress,
            report_output_path,
            title="Q-learning Until Act Boss",
            window=progress_window,
        )

    result = TrainingUntilTargetResult(
        algorithm="state_action_signature_q_learning",
        target_act=target_act,
        target_floor=target_floor,
        reached_target=reached_batch is not None,
        reached_batch=reached_batch,
        batches_completed=len(batch_summaries),
        max_batches=max_batches,
        batch_runs=batch_runs,
        runs_trained=runs_trained,
        total_steps=total_steps,
        average_training_reward=round(total_reward / max(1, runs_trained), 6),
        wins=sum(1 for point in training_points if point.win),
        deaths=sum(1 for point in training_points if point.death),
        resumed_from_path=resumed_from,
        model_path=str(model_output_path) if model_output_path is not None else None,
        output_path=str(output_path) if output_path is not None else None,
        progress_output_path=(
            str(progress_output_path) if progress_output_path is not None else None
        ),
        report_output_path=str(report_output_path) if report_output_path is not None else None,
        batch_summaries=tuple(batch_summaries),
        progress=progress,
        evaluation_progress=evaluation_progress,
    )
    if output_path is not None:
        _write_model(result, output_path)
    return result


def _load_resume_model(
    *,
    resume: bool,
    resume_from_path: Path | str | None,
    model_output_path: Path | str | None,
) -> tuple[QLearningModel | None, str | None]:
    if not resume:
        return None, None
    candidate_paths = [
        Path(resume_from_path) if resume_from_path is not None else None,
        Path(model_output_path) if model_output_path is not None else None,
    ]
    for candidate in candidate_paths:
        if candidate is None or not candidate.exists():
            continue
        return load_q_learning_model(candidate), str(candidate)
    return None, None


def _boss_floor_for_new_run(
    *,
    seed: int | str,
    character_id: str,
    ascension: int,
    target_act: int,
) -> int:
    state = new_run(seed=seed, character_id=character_id, ascension=ascension)
    game_map = getattr(state, "map", None)
    if game_map is None:
        return 16 if target_act <= 1 else 15
    boss_node_id = getattr(game_map, "boss_node_id", None)
    node_by_id = getattr(game_map, "node_by_id", {})
    if isinstance(boss_node_id, str) and boss_node_id in node_by_id:
        return _int(getattr(node_by_id[boss_node_id], "floor", 16))
    boss_floors = [
        _int(getattr(node, "floor", 0))
        for node in getattr(game_map, "nodes", ())
        if getattr(node, "kind", None) == RoomKind.BOSS
    ]
    if boss_floors:
        return max(boss_floors)
    return max(
        (_int(getattr(node, "floor", 0)) for node in getattr(game_map, "nodes", ())),
        default=16,
    )


def _run_reached_target(
    run: LearningRunResult,
    *,
    target_act: int,
    target_floor: int,
) -> bool:
    if run.final_act > target_act:
        return True
    return run.final_act == target_act and run.final_floor >= target_floor


def _observation_reached_target(
    observation: Mapping[str, Any],
    *,
    target_act: int,
    target_floor: int,
) -> bool:
    act = _lookup_vector_int(observation, "act")
    floor = _lookup_vector_int(observation, "floor")
    return act > target_act or (act == target_act and floor >= target_floor)


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


def _training_seed(
    seed: int | str,
    run_index: int,
    *,
    mode: str,
    rng: Random,
) -> int | str:
    if mode == "random":
        return _random_run_seed(rng)
    return _run_seed(seed, run_index)


def _evaluation_seed(
    eval_start_seed: int,
    eval_index: int,
    *,
    mode: str,
    rng: Random,
) -> int:
    if mode == "random":
        return _random_run_seed(rng)
    return eval_start_seed + eval_index


def _random_run_seed(rng: Random) -> int:
    return rng.randrange(0, 2_147_483_647)


def _normalize_seed_mode(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in {"sequential", "random"}:
        raise ValueError("seed mode must be 'sequential' or 'random'")
    return normalized


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


def _write_model(result: TrainingUntilTargetResult, path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
