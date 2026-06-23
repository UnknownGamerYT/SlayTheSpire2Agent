from __future__ import annotations

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from sts2sim.cli.app import app
from sts2sim.learning.curriculum import (
    _stage_summary,
    resolve_curriculum_stages,
    train_masked_ppo_curriculum,
)


def test_resolve_curriculum_stages_defaults_and_commas() -> None:
    assert resolve_curriculum_stages() == (
        "act1-boss",
        "act2-boss",
        "act3-boss",
        "game-clear",
    )
    assert resolve_curriculum_stages("act1_boss, act2-boss") == (
        "act1-boss",
        "act2-boss",
    )


def test_train_masked_ppo_curriculum_advances_until_stage_fails(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_trainer(**kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        target = str(kwargs["target"])
        reached = target == "act1-boss"
        return {
            "reached_target": reached,
            "reached_batch": 1 if reached else None,
            "batches_completed": 1,
            "runs_trained": 2,
            "total_steps": 3,
            "batch_summaries": [
                {
                    "batch_index": 1,
                    "evaluation_target_successes": 2 if reached else 0,
                    "evaluation_max_consecutive_successes": 2 if reached else 0,
                }
            ],
        }

    result = train_masked_ppo_curriculum(
        stages=("act1-boss", "act2-boss", "game-clear"),
        max_batches=1,
        train_runs_per_batch=2,
        eval_runs=2,
        resume=False,
        checkpoint_dir=tmp_path / "checkpoints",
        report_dir=tmp_path / "reports",
        output_path=tmp_path / "curriculum.json",
        report_output_path=tmp_path / "curriculum.html",
        trainer=fake_trainer,
    )

    assert result["completed_curriculum"] is False
    assert result["stages_started"] == 2
    assert result["stages_completed"] == 1
    assert result["current_stage"] == "act2-boss"
    assert "did not meet comfort criteria" in str(result["stopped_reason"])
    assert calls[0]["resume_from_path"] is None
    assert calls[0]["hidden_size"] == 256
    assert calls[0]["hidden_layers"] == 3
    assert calls[0]["head_hidden_layers"] == 2
    assert calls[0]["activation"] == "silu"
    assert calls[0]["planning_coef"] == 0.1
    assert calls[1]["resume_from_path"] == tmp_path / "checkpoints" / (
        "ppo_curriculum_act1_boss.pt"
    )
    assert result["batch_metric_summary"]["batches"] == 2
    assert result["batch_metrics"][-1]["stage"] == "act2-boss"
    assert (tmp_path / "curriculum.json").exists()
    assert (tmp_path / "curriculum.html").exists()


def test_stage_summary_prefers_actual_resume_checkpoint(tmp_path: Path) -> None:
    summary = _stage_summary(
        stage_index=0,
        stage_name="act1-boss",
        stage_result={
            "resumed_from_path": "checkpoints/ppo_curriculum_silu_act1_boss.pt",
            "batches_completed": 1,
            "runs_trained": 128,
            "total_steps": 1000,
            "batch_summaries": [{"batch_index": 1}],
        },
        resume_from_path=None,
        model_path=tmp_path / "checkpoint.pt",
        output_path=tmp_path / "latest.json",
        progress_output_path=tmp_path / "progress.json",
        report_output_path=tmp_path / "latest.html",
        status="running",
    )

    assert summary["resume_from_path"] == "checkpoints/ppo_curriculum_silu_act1_boss.pt"


def test_train_ppo_curriculum_help_lists_stage_and_comfort_controls() -> None:
    result = CliRunner().invoke(app, ["train-ppo-curriculum", "--help"])

    assert result.exit_code == 0
    assert "--stages" in result.output
    assert "comfortable" in result.output.lower()
    assert "--target-consecut" in result.output
    assert "--report-output" in result.output
    assert "--hidden-layers" in result.output
    assert "--head-hidden" in result.output
    assert "--activation" in result.output
    assert "--planning-coef" in result.output
