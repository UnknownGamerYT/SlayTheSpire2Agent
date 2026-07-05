"""Curriculum runner for staged random-seed PPO training."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from html import escape as html_escape
from pathlib import Path
from typing import Any

from sts2sim.learning.masked_ppo import resolve_ppo_target, train_masked_ppo

DEFAULT_CURRICULUM_STAGES: tuple[str, ...] = (
    "act1-boss",
    "act2-boss",
    "act3-boss",
    "game-clear",
)


@dataclass(frozen=True)
class CurriculumStageDefaults:
    """Default work budget for one PPO curriculum stage."""

    max_batches: int
    train_max_steps: int
    eval_max_steps: int
    target_eval_successes: int
    target_consecutive_successes: int


CURRICULUM_STAGE_DEFAULTS: dict[str, CurriculumStageDefaults] = {
    "act1-boss": CurriculumStageDefaults(
        max_batches=50,
        train_max_steps=800,
        eval_max_steps=800,
        target_eval_successes=8,
        target_consecutive_successes=2,
    ),
    "act2-boss": CurriculumStageDefaults(
        max_batches=80,
        train_max_steps=1400,
        eval_max_steps=1400,
        target_eval_successes=6,
        target_consecutive_successes=2,
    ),
    "act3-boss": CurriculumStageDefaults(
        max_batches=120,
        train_max_steps=2200,
        eval_max_steps=2200,
        target_eval_successes=5,
        target_consecutive_successes=2,
    ),
    "game-clear": CurriculumStageDefaults(
        max_batches=160,
        train_max_steps=3000,
        eval_max_steps=3000,
        target_eval_successes=3,
        target_consecutive_successes=2,
    ),
}

PPOTrainer = Callable[..., Mapping[str, Any]]


def train_masked_ppo_curriculum(
    *,
    stages: str | Sequence[str] | None = None,
    run_name: str = "ppo_curriculum",
    max_batches: int | None = None,
    train_runs_per_batch: int = 128,
    eval_runs: int = 32,
    train_max_steps: int | None = None,
    eval_max_steps: int | None = None,
    seed: int | str = "ppo-curriculum",
    character_id: str = "IRONCLAD",
    ascension: int = 0,
    hidden_size: int = 256,
    hidden_layers: int = 3,
    head_hidden_layers: int = 2,
    activation: str = "silu",
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    planning_coef: float = 0.1,
    teacher_mix: float = 0.0,
    imitation_coef: float = 0.0,
    ppo_epochs: int = 4,
    minibatch_size: int = 256,
    target_reward: float = 100.0,
    target_eval_successes: int | None = None,
    target_consecutive_successes: int | None = None,
    target_success_rate: float = 0.0,
    resume: bool = True,
    resume_from_path: Path | str | None = None,
    checkpoint_dir: Path | str = Path("checkpoints"),
    report_dir: Path | str = Path("reports"),
    output_path: Path | str | None = Path("reports/ppo_curriculum_latest.json"),
    report_output_path: Path | str | None = Path("reports/ppo_curriculum_latest.html"),
    progress_window: int = 20,
    device: str = "auto",
    trainer: PPOTrainer | None = None,
) -> dict[str, Any]:
    """Train PPO through staged targets, advancing only after comfort criteria pass."""

    resolved_stages = resolve_curriculum_stages(stages)
    trainer_func = trainer or train_masked_ppo
    checkpoint_root = Path(checkpoint_dir)
    report_root = Path(report_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    previous_model_path = Path(resume_from_path) if resume_from_path is not None else None
    stage_summaries: list[dict[str, Any]] = []
    stopped_reason = "completed"

    for stage_index, stage_name in enumerate(resolved_stages):
        defaults = CURRICULUM_STAGE_DEFAULTS[stage_name]
        slug = stage_name.replace("-", "_")
        model_path = checkpoint_root / f"{run_name}_{slug}.pt"
        stage_output_path = report_root / f"{run_name}_{slug}_latest.json"
        stage_progress_path = report_root / f"{run_name}_{slug}_progress.json"
        stage_report_path = report_root / f"{run_name}_{slug}_latest.html"
        stage_resume_from = previous_model_path if previous_model_path is not None else None
        stage_should_resume = resume or stage_resume_from is not None

        if output_path is not None:
            _persist_curriculum_result(
                _with_batch_metrics(
                    _curriculum_result(
                        resolved_stages=resolved_stages,
                        stage_summaries=[
                            *stage_summaries,
                            _initial_or_running_stage_summary(
                                stage_index=stage_index,
                                stage_name=stage_name,
                                resume_from_path=stage_resume_from,
                                model_path=model_path,
                                output_path=stage_output_path,
                                progress_output_path=stage_progress_path,
                                report_output_path=stage_report_path,
                            ),
                        ],
                        stopped_reason=f"training {stage_name}",
                        status="running",
                        output_path=output_path,
                        report_output_path=report_output_path,
                        final_model_path=previous_model_path,
                        seed=seed,
                        character_id=character_id,
                        ascension=ascension,
                        run_name=run_name,
                        train_runs_per_batch=train_runs_per_batch,
                        eval_runs=eval_runs,
                        target_eval_successes=target_eval_successes,
                        target_consecutive_successes=target_consecutive_successes,
                        target_success_rate=target_success_rate,
                        hidden_size=hidden_size,
                        hidden_layers=hidden_layers,
                        head_hidden_layers=head_hidden_layers,
                        activation=activation,
                        planning_coef=planning_coef,
                        teacher_mix=teacher_mix,
                        imitation_coef=imitation_coef,
                        device=device,
                    )
                ),
                output_path,
                report_output_path,
            )

        def update_active_stage_summary(
            stage_result: Mapping[str, Any],
            *,
            callback_stage_index: int = stage_index,
            callback_stage_name: str = stage_name,
            callback_resume_from: Path | None = stage_resume_from,
            callback_model_path: Path = model_path,
            callback_output_path: Path = stage_output_path,
            callback_progress_path: Path = stage_progress_path,
            callback_report_path: Path = stage_report_path,
            callback_final_model_path: Path | None = previous_model_path,
        ) -> None:
            if output_path is None:
                return
            _persist_curriculum_result(
                _with_batch_metrics(
                    _curriculum_result(
                        resolved_stages=resolved_stages,
                        stage_summaries=[
                            *stage_summaries,
                            _stage_summary(
                                stage_index=callback_stage_index,
                                stage_name=callback_stage_name,
                                stage_result=stage_result,
                                resume_from_path=callback_resume_from,
                                model_path=callback_model_path,
                                output_path=callback_output_path,
                                progress_output_path=callback_progress_path,
                                report_output_path=callback_report_path,
                                status="running",
                            ),
                        ],
                        stopped_reason=f"training {callback_stage_name}",
                        status="running",
                        output_path=output_path,
                        report_output_path=report_output_path,
                        final_model_path=callback_final_model_path,
                        seed=seed,
                        character_id=character_id,
                        ascension=ascension,
                        run_name=run_name,
                        train_runs_per_batch=train_runs_per_batch,
                        eval_runs=eval_runs,
                        target_eval_successes=target_eval_successes,
                        target_consecutive_successes=target_consecutive_successes,
                        target_success_rate=target_success_rate,
                        hidden_size=hidden_size,
                        hidden_layers=hidden_layers,
                        head_hidden_layers=head_hidden_layers,
                        activation=activation,
                        planning_coef=planning_coef,
                        teacher_mix=teacher_mix,
                        imitation_coef=imitation_coef,
                        device=device,
                    )
                ),
                output_path,
                report_output_path,
            )

        stage_result = dict(
            trainer_func(
                target=stage_name,
                max_batches=max_batches or defaults.max_batches,
                train_runs_per_batch=train_runs_per_batch,
                train_max_steps=train_max_steps or defaults.train_max_steps,
                eval_runs=eval_runs,
                eval_max_steps=eval_max_steps or defaults.eval_max_steps,
                seed=f"{seed}:{stage_name}",
                character_id=character_id,
                ascension=ascension,
                hidden_size=hidden_size,
                hidden_layers=hidden_layers,
                head_hidden_layers=head_hidden_layers,
                activation=activation,
                learning_rate=learning_rate,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_ratio=clip_ratio,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                planning_coef=planning_coef,
                teacher_mix=teacher_mix,
                imitation_coef=imitation_coef,
                ppo_epochs=ppo_epochs,
                minibatch_size=minibatch_size,
                target_reward=target_reward,
                target_eval_successes=(
                    target_eval_successes or defaults.target_eval_successes
                ),
                target_consecutive_successes=(
                    target_consecutive_successes
                    or defaults.target_consecutive_successes
                ),
                target_success_rate=target_success_rate,
                resume=stage_should_resume,
                resume_from_path=stage_resume_from,
                model_output_path=model_path,
                output_path=stage_output_path,
                progress_output_path=stage_progress_path,
                report_output_path=stage_report_path,
                progress_window=progress_window,
                device=device,
                progress_callback=update_active_stage_summary,
            )
        )
        stage_summary = _stage_summary(
            stage_index=stage_index,
            stage_name=stage_name,
            stage_result=stage_result,
            resume_from_path=stage_resume_from,
            model_path=model_path,
            output_path=stage_output_path,
            progress_output_path=stage_progress_path,
            report_output_path=stage_report_path,
            status="complete" if bool(stage_result.get("reached_target", False)) else "stopped",
        )
        stage_summaries.append(stage_summary)

        if not bool(stage_result.get("reached_target", False)):
            stopped_reason = f"stage {stage_name} did not meet comfort criteria"
            break
        previous_model_path = model_path

        if output_path is not None:
            _persist_curriculum_result(
                _with_batch_metrics(
                    _curriculum_result(
                        resolved_stages=resolved_stages,
                        stage_summaries=stage_summaries,
                        stopped_reason=f"completed {stage_name}",
                        status="running",
                        output_path=output_path,
                        report_output_path=report_output_path,
                        final_model_path=previous_model_path,
                        seed=seed,
                        character_id=character_id,
                        ascension=ascension,
                        run_name=run_name,
                        train_runs_per_batch=train_runs_per_batch,
                        eval_runs=eval_runs,
                        target_eval_successes=target_eval_successes,
                        target_consecutive_successes=target_consecutive_successes,
                        target_success_rate=target_success_rate,
                        hidden_size=hidden_size,
                        hidden_layers=hidden_layers,
                        head_hidden_layers=head_hidden_layers,
                        activation=activation,
                        planning_coef=planning_coef,
                        teacher_mix=teacher_mix,
                        imitation_coef=imitation_coef,
                        device=device,
                    )
                ),
                output_path,
                report_output_path,
            )

    completed_stages = sum(1 for stage in stage_summaries if stage["reached_target"])
    completed_curriculum = completed_stages == len(resolved_stages)
    if completed_curriculum:
        stopped_reason = "completed"
    result = _curriculum_result(
        resolved_stages=resolved_stages,
        stage_summaries=stage_summaries,
        stopped_reason=stopped_reason,
        status="complete" if completed_curriculum else "stopped",
        output_path=output_path,
        report_output_path=report_output_path,
        final_model_path=previous_model_path,
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        run_name=run_name,
        train_runs_per_batch=train_runs_per_batch,
        eval_runs=eval_runs,
        target_eval_successes=target_eval_successes,
        target_consecutive_successes=target_consecutive_successes,
        target_success_rate=target_success_rate,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        head_hidden_layers=head_hidden_layers,
        activation=activation,
        planning_coef=planning_coef,
        teacher_mix=teacher_mix,
        imitation_coef=imitation_coef,
        device=device,
    )
    result = _with_batch_metrics(result)
    _persist_curriculum_result(result, output_path, report_output_path)
    return result


def resolve_curriculum_stages(stages: str | Sequence[str] | None = None) -> tuple[str, ...]:
    """Normalize and validate curriculum stage names."""

    if stages is None:
        raw_stages = DEFAULT_CURRICULUM_STAGES
    elif isinstance(stages, str):
        raw_stages = tuple(part for part in stages.replace(";", ",").split(",") if part)
    else:
        raw_stages = tuple(stages)
    normalized = tuple(_normalize_stage_name(stage) for stage in raw_stages)
    if not normalized:
        raise ValueError("At least one curriculum stage is required.")
    for stage in normalized:
        resolve_ppo_target(stage)
        if stage not in CURRICULUM_STAGE_DEFAULTS:
            valid = ", ".join(DEFAULT_CURRICULUM_STAGES)
            raise ValueError(f"Unsupported curriculum stage {stage!r}. Valid stages: {valid}.")
    return normalized


def _stage_summary(
    *,
    stage_index: int,
    stage_name: str,
    stage_result: Mapping[str, Any],
    resume_from_path: Path | None,
    model_path: Path,
    output_path: Path,
    progress_output_path: Path,
    report_output_path: Path,
    status: str,
) -> dict[str, Any]:
    batch_summaries = _sequence_of_mappings(stage_result.get("batch_summaries"))
    latest_batch = dict(batch_summaries[-1]) if batch_summaries else {}
    resolved_resume_from_path = _stage_resume_from_path(
        stage_result,
        resume_from_path=resume_from_path,
    )
    return {
        "stage_index": stage_index,
        "stage": stage_name,
        "status": status,
        "reached_target": bool(stage_result.get("reached_target", False)),
        "reached_batch": stage_result.get("reached_batch"),
        "batches_completed": _int(stage_result.get("batches_completed")),
        "runs_trained": _int(stage_result.get("runs_trained")),
        "total_steps": _int(stage_result.get("total_steps")),
        "resume_from_path": resolved_resume_from_path,
        "model_path": str(model_path),
        "output_path": str(output_path),
        "progress_output_path": str(progress_output_path),
        "report_output_path": str(report_output_path),
        "latest_batch": latest_batch,
    }


def _stage_resume_from_path(
    stage_result: Mapping[str, Any],
    *,
    resume_from_path: Path | None,
) -> str | None:
    actual_resume_from_path = stage_result.get("resumed_from_path")
    if actual_resume_from_path:
        return str(actual_resume_from_path)
    return str(resume_from_path) if resume_from_path else None


def _running_stage_summary(
    *,
    stage_index: int,
    stage_name: str,
    resume_from_path: Path | None,
    model_path: Path,
    output_path: Path,
    progress_output_path: Path,
    report_output_path: Path,
) -> dict[str, Any]:
    return {
        "stage_index": stage_index,
        "stage": stage_name,
        "status": "running",
        "reached_target": False,
        "reached_batch": None,
        "batches_completed": 0,
        "runs_trained": 0,
        "total_steps": 0,
        "resume_from_path": str(resume_from_path) if resume_from_path else None,
        "model_path": str(model_path),
        "output_path": str(output_path),
        "progress_output_path": str(progress_output_path),
        "report_output_path": str(report_output_path),
        "latest_batch": {},
    }


def _initial_or_running_stage_summary(
    *,
    stage_index: int,
    stage_name: str,
    resume_from_path: Path | None,
    model_path: Path,
    output_path: Path,
    progress_output_path: Path,
    report_output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, Mapping):
            return _stage_summary(
                stage_index=stage_index,
                stage_name=stage_name,
                stage_result=payload,
                resume_from_path=resume_from_path,
                model_path=model_path,
                output_path=output_path,
                progress_output_path=progress_output_path,
                report_output_path=report_output_path,
                status="running",
            )
    return _running_stage_summary(
        stage_index=stage_index,
        stage_name=stage_name,
        resume_from_path=resume_from_path,
        model_path=model_path,
        output_path=output_path,
        progress_output_path=progress_output_path,
        report_output_path=report_output_path,
    )


def _curriculum_result(
    *,
    resolved_stages: Sequence[str],
    stage_summaries: Sequence[Mapping[str, Any]],
    stopped_reason: str,
    status: str,
    output_path: Path | str | None,
    report_output_path: Path | str | None,
    final_model_path: Path | None,
    seed: int | str,
    character_id: str,
    ascension: int,
    run_name: str,
    train_runs_per_batch: int,
    eval_runs: int,
    target_eval_successes: int | None,
    target_consecutive_successes: int | None,
    target_success_rate: float,
    hidden_size: int,
    hidden_layers: int,
    head_hidden_layers: int,
    activation: str,
    planning_coef: float,
    teacher_mix: float,
    imitation_coef: float,
    device: str,
) -> dict[str, Any]:
    completed_stages = sum(
        1 for stage in stage_summaries if bool(stage.get("reached_target", False))
    )
    completed_curriculum = (
        completed_stages == len(resolved_stages) and status == "complete"
    )
    current_stage = stage_summaries[-1]["stage"] if stage_summaries else None
    return {
        "algorithm": "masked_ppo_curriculum",
        "status": status,
        "stages_requested": list(resolved_stages),
        "stages_started": len(stage_summaries),
        "stages_completed": completed_stages,
        "completed_curriculum": completed_curriculum,
        "stopped_reason": stopped_reason,
        "current_stage": current_stage,
        "final_model_path": str(final_model_path) if final_model_path else None,
        "output_path": str(output_path) if output_path is not None else None,
        "report_output_path": (
            str(report_output_path) if report_output_path is not None else None
        ),
        "stage_summaries": [dict(stage) for stage in stage_summaries],
        "metadata": {
            "seed": seed,
            "character_id": character_id,
            "character_name": _character_display_name(character_id),
            "ascension": ascension,
            "run_name": run_name,
            "train_runs_per_batch": train_runs_per_batch,
            "eval_runs": eval_runs,
            "hidden_size": hidden_size,
            "hidden_layers": max(1, hidden_layers),
            "head_hidden_layers": max(1, head_hidden_layers),
            "activation": activation,
            "planning_coef": planning_coef,
            "teacher_mix": teacher_mix,
            "imitation_coef": imitation_coef,
            "requested_device": device,
            "target_eval_successes_override": target_eval_successes,
            "target_consecutive_successes_override": target_consecutive_successes,
            "target_success_rate": max(0.0, min(1.0, _float(target_success_rate))),
        },
    }


def _with_batch_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    batch_metrics = _collect_curriculum_batch_metrics(payload)
    payload["batch_metrics"] = batch_metrics
    payload["batch_metric_summary"] = _batch_metric_summary(batch_metrics)
    return payload


def _collect_curriculum_batch_metrics(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in _sequence_of_mappings(result.get("stage_summaries")):
        stage_name = str(stage.get("stage", "unknown"))
        stage_batches = _stage_batch_summaries(stage)
        if not stage_batches and isinstance(stage.get("latest_batch"), Mapping):
            stage_batches = (stage["latest_batch"],)
        for batch in stage_batches:
            rows.append(
                {
                    "x": len(rows) + 1,
                    "stage": stage_name,
                    "batch_index": _int(batch.get("batch_index")),
                    "trained_runs_total": _int(batch.get("trained_runs_total")),
                    "train_total_steps": _int(batch.get("train_total_steps")),
                    "evaluation_runs": _int(batch.get("evaluation_runs")),
                    "evaluation_average_reward": _float(
                        batch.get("evaluation_average_reward")
                    ),
                    "evaluation_average_floor": _float(
                        batch.get("evaluation_average_floor")
                    ),
                    "evaluation_best_floor": _float(batch.get("evaluation_best_floor")),
                    "evaluation_best_reward": _float(batch.get("evaluation_best_reward")),
                    "evaluation_target_successes": _int(
                        batch.get("evaluation_target_successes")
                    ),
                    "evaluation_target_success_rate": _float(
                        batch.get("evaluation_target_success_rate")
                    ),
                    "evaluation_max_consecutive_successes": _int(
                        batch.get("evaluation_max_consecutive_successes")
                    ),
                    "reached_target": bool(batch.get("reached_target", False)),
                }
            )
    return rows


def _stage_batch_summaries(stage: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    output_path = stage.get("output_path")
    if not isinstance(output_path, str | Path):
        return ()
    path = Path(output_path)
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(payload, Mapping):
        return ()
    return _sequence_of_mappings(payload.get("batch_summaries"))


def _batch_metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "batches": 0,
            "best_floor": 0,
            "best_reward": 0.0,
            "best_target_success_rate": 0.0,
            "best_consecutive_successes": 0,
            "latest_target_success_rate": 0.0,
            "latest_average_floor": 0.0,
            "latest_average_reward": 0.0,
        }
    latest = rows[-1]
    return {
        "batches": len(rows),
        "best_floor": max(_float(row.get("evaluation_best_floor")) for row in rows),
        "best_reward": max(_float(row.get("evaluation_best_reward")) for row in rows),
        "best_target_success_rate": max(
            _float(row.get("evaluation_target_success_rate")) for row in rows
        ),
        "best_consecutive_successes": max(
            _int(row.get("evaluation_max_consecutive_successes")) for row in rows
        ),
        "latest_target_success_rate": _float(
            latest.get("evaluation_target_success_rate")
        ),
        "latest_average_floor": _float(latest.get("evaluation_average_floor")),
        "latest_average_reward": _float(latest.get("evaluation_average_reward")),
    }


def _persist_curriculum_result(
    result: Mapping[str, Any],
    output_path: Path | str | None,
    report_output_path: Path | str | None,
) -> None:
    if output_path is not None:
        _write_json(result, output_path)
    if report_output_path is not None:
        write_curriculum_progress_report(result, report_output_path)


def write_curriculum_progress_report(
    result: Mapping[str, Any],
    path: Path | str,
) -> None:
    """Write a standalone HTML dashboard for curriculum batch metrics."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(curriculum_progress_html(result), encoding="utf-8")


def curriculum_progress_html(result: Mapping[str, Any]) -> str:
    """Render the combined curriculum learning dashboard."""

    rows = _sequence_of_mappings(result.get("batch_metrics"))
    summary = _mapping(result.get("batch_metric_summary"))
    metadata = _mapping(result.get("metadata"))
    stage_summaries = _sequence_of_mappings(result.get("stage_summaries"))
    current_stage = str(result.get("current_stage", "unknown"))
    status = str(result.get("status", "unknown"))
    character_name = str(metadata.get("character_name", metadata.get("character_id", "")))
    title = f"PPO Curriculum Progress - {character_name}".strip()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #5f6b72;
      --line: #d7dde1;
      --panel: #ffffff;
      --paper: #f5f7f8;
      --blue: #2563eb;
      --green: #16855b;
      --amber: #b7791f;
      --red: #c24130;
      --purple: #7c3aed;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 44px;
    }}
    h1 {{ margin: 0; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }}
    p {{ margin: 4px 0 0; color: var(--muted); }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 10px;
      margin: 20px 0;
    }}
    .kpi, .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .kpi {{ padding: 12px; }}
    .kpi span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .kpi strong {{
      display: block;
      margin-top: 4px;
      font-size: 24px;
      line-height: 1.1;
    }}
    .section {{
      margin-top: 14px;
      padding: 16px;
      overflow: hidden;
    }}
    .chart {{ display: block; width: 100%; height: auto; }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .dot {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 6px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }}
    th {{ color: var(--muted); font-weight: 650; }}
    @media (max-width: 780px) {{
      main {{ width: min(100vw - 20px, 1180px); padding-top: 18px; }}
      h1 {{ font-size: 24px; }}
      .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .section {{ padding: 12px; }}
      table {{ display: block; overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html_escape(title)}</h1>
    <p>Status: {html_escape(status)}. Current stage: {html_escape(current_stage)}.</p>
    <div class="kpis">
      {_kpi("Batches", _format_number(summary.get("batches")))}
      {_kpi("Best Floor", _format_number(summary.get("best_floor")))}
      {_kpi("Best Reward", _format_number(summary.get("best_reward")))}
      {_kpi("Best Success Rate", _format_percent(summary.get("best_target_success_rate")))}
      {_kpi("Best Streak", _format_number(summary.get("best_consecutive_successes")))}
    </div>
    <div class="section">
      <h2>Evaluation Reward</h2>
      {_curriculum_line_chart(
          rows,
          (
              ("Average reward", "evaluation_average_reward", "#2563eb"),
              ("Best reward", "evaluation_best_reward", "#7c3aed"),
          ),
          y_label="Reward",
      )}
    </div>
    <div class="section">
      <h2>Floor Progress</h2>
      {_curriculum_line_chart(
          rows,
          (
              ("Average floor", "evaluation_average_floor", "#16855b"),
              ("Best floor", "evaluation_best_floor", "#b7791f"),
          ),
          y_label="Floor",
          y_floor=0.0,
      )}
    </div>
    <div class="section">
      <h2>Target Progress</h2>
      {_curriculum_line_chart(
          rows,
          (("Success rate", "evaluation_target_success_rate", "#2563eb"),),
          y_label="Success rate",
          y_floor=0.0,
          y_ceiling=1.0,
          percent=True,
      )}
      {_curriculum_line_chart(
          rows,
          (("Consecutive successes", "evaluation_max_consecutive_successes", "#c24130"),),
          y_label="Streak",
          y_floor=0.0,
      )}
    </div>
    <div class="section">
      <h2>Stages</h2>
      {_stage_table(stage_summaries)}
    </div>
    <div class="section">
      <h2>Recent Batches</h2>
      {_batch_table(rows)}
    </div>
  </main>
</body>
</html>
"""


def _curriculum_line_chart(
    rows: Sequence[Mapping[str, Any]],
    series: Sequence[tuple[str, str, str]],
    *,
    y_label: str,
    y_floor: float | None = None,
    y_ceiling: float | None = None,
    percent: bool = False,
) -> str:
    width = 1040
    height = 260
    left = 54
    right = 16
    top = 22
    bottom = 38
    plot_width = width - left - right
    plot_height = height - top - bottom
    if not rows:
        return '<p>No batch metrics have been saved yet.</p>'

    values = [
        _float(row.get(key))
        for _label, key, _color in series
        for row in rows
    ]
    ymin = min(values) if y_floor is None else y_floor
    ymax = max(values) if y_ceiling is None else y_ceiling
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    pad = (ymax - ymin) * 0.08
    if y_floor is None:
        ymin -= pad
    if y_ceiling is None:
        ymax += pad

    def point(index: int, value: float) -> tuple[float, float]:
        x = left + (plot_width * index / max(1, len(rows) - 1))
        y = top + ((ymax - value) / max(0.000001, ymax - ymin)) * plot_height
        return x, y

    polylines: list[str] = []
    legends: list[str] = []
    for label, key, color in series:
        points = " ".join(
            f"{x:.2f},{y:.2f}"
            for index, row in enumerate(rows)
            for x, y in (point(index, _float(row.get(key))),)
        )
        polylines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" />'
        )
        legends.append(
            f'<span><i class="dot" style="background:{color}"></i>{html_escape(label)}</span>'
        )

    y_ticks = []
    for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = ymin + (ymax - ymin) * ratio
        y = top + (1.0 - ratio) * plot_height
        label = _format_percent(value) if percent else _format_number(value)
        y_ticks.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
            'stroke="#d7dde1" stroke-width="1" />'
            f'<text x="8" y="{y + 4:.2f}" fill="#5f6b72" font-size="12">{html_escape(label)}</text>'
        )
    stage_labels = _stage_markers(
        rows,
        left=left,
        top=top,
        bottom=bottom,
        height=height,
        plot_width=plot_width,
    )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html_escape(y_label)} chart">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />'
        f'{"".join(y_ticks)}'
        f'{stage_labels}'
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" '
        f'y2="{height - bottom}" stroke="#172026" />'
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{height - bottom}" stroke="#172026" />'
        f'{"".join(polylines)}'
        f'<text x="{left}" y="{height - 10}" fill="#5f6b72" font-size="12">Batches over time</text>'
        f'</svg><div class="legend">{"".join(legends)}</div>'
    )


def _stage_markers(
    rows: Sequence[Mapping[str, Any]],
    *,
    left: int,
    top: int,
    bottom: int,
    height: int,
    plot_width: int,
) -> str:
    markers: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        stage = str(row.get("stage", ""))
        if not stage or stage in seen:
            continue
        seen.add(stage)
        x = left + (plot_width * index / max(1, len(rows) - 1))
        markers.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}" '
            'stroke="#d7dde1" stroke-dasharray="4 4" />'
            f'<text x="{x + 4:.2f}" y="{top + 12}" fill="#5f6b72" '
            f'font-size="11">{html_escape(stage)}</text>'
        )
    return "".join(markers)


def _stage_table(stages: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for stage in stages:
        latest = _mapping(stage.get("latest_batch"))
        rows.append(
            "<tr>"
            f"<td>{html_escape(str(stage.get('stage', '')))}</td>"
            f"<td>{html_escape(str(stage.get('status', '')))}</td>"
            f"<td>{_format_number(stage.get('batches_completed'))}</td>"
            f"<td>{_format_number(stage.get('runs_trained'))}</td>"
            f"<td>{_format_number(stage.get('total_steps'))}</td>"
            f"<td>{_format_percent(latest.get('evaluation_target_success_rate'))}</td>"
            f"<td>{_format_number(latest.get('evaluation_max_consecutive_successes'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Stage</th><th>Status</th><th>Batches</th>"
        "<th>Runs</th><th>Steps</th><th>Target Rate</th><th>Streak</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _batch_table(rows: Sequence[Mapping[str, Any]]) -> str:
    recent = tuple(rows[-12:])
    table_rows = []
    for row in recent:
        table_rows.append(
            "<tr>"
            f"<td>{html_escape(str(row.get('stage', '')))}</td>"
            f"<td>{_format_number(row.get('batch_index'))}</td>"
            f"<td>{_format_number(row.get('trained_runs_total'))}</td>"
            f"<td>{_format_number(row.get('evaluation_average_floor'))}</td>"
            f"<td>{_format_number(row.get('evaluation_best_floor'))}</td>"
            f"<td>{_format_number(row.get('evaluation_average_reward'))}</td>"
            f"<td>{_format_percent(row.get('evaluation_target_success_rate'))}</td>"
            f"<td>{_format_number(row.get('evaluation_max_consecutive_successes'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Stage</th><th>Batch</th><th>Runs</th>"
        "<th>Avg Floor</th><th>Best Floor</th><th>Avg Reward</th>"
        "<th>Target Rate</th><th>Streak</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody></table>"
    )


def _kpi(label: str, value: str) -> str:
    return (
        f'<div class="kpi"><span>{html_escape(label)}</span>'
        f"<strong>{html_escape(value)}</strong></div>"
    )


def _normalize_stage_name(stage: object) -> str:
    return str(stage).strip().lower().replace("_", "-")


def _character_display_name(character_id: object) -> str:
    normalized = str(character_id or "").strip().upper()
    return {
        "IRONCLAD": "The Ironclad",
        "SILENT": "The Silent",
        "DEFECT": "The Defect",
        "WATCHER": "The Watcher",
        "NECROBINDER": "The Necrobinder",
    }.get(normalized, normalized or "Unknown")


def _sequence_of_mappings(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _write_json(value: Mapping[str, Any], path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _int(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _float(value: object) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _format_number(value: object) -> str:
    number = _float(value)
    if abs(number - int(number)) < 0.000001:
        return str(int(number))
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _format_percent(value: object) -> str:
    return f"{_float(value) * 100.0:.1f}%"
