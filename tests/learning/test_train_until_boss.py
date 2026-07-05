from __future__ import annotations

import json

from typer.testing import CliRunner

from sts2sim.cli.app import app
from sts2sim.learning import train_q_learning_until_boss


def test_train_until_boss_stops_when_eval_reaches_target_floor(tmp_path) -> None:
    model_path = tmp_path / "q_until.json"
    output_path = tmp_path / "summary.json"
    progress_path = tmp_path / "progress.json"
    report_path = tmp_path / "progress.html"

    result = train_q_learning_until_boss(
        max_batches=1,
        batch_runs=1,
        train_max_steps=4,
        eval_runs=1,
        eval_max_steps=4,
        seed=120,
        eval_start_seed=220,
        character_id="TEST",
        target_floor=1,
        success_replay_passes=1,
        model_output_path=model_path,
        output_path=output_path,
        progress_output_path=progress_path,
        report_output_path=report_path,
    )

    assert result.reached_target is True
    assert result.reached_batch == 1
    assert result.batch_summaries[0].reached_target is True
    assert result.batch_summaries[0].evaluation_target_successes >= 1
    assert model_path.exists()
    assert output_path.exists()
    assert progress_path.exists()
    assert report_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["reached_target"] is True
    assert payload["resumed_from_path"] is None


def test_train_until_boss_cli_smoke(tmp_path) -> None:
    model_path = tmp_path / "cli_q_until.json"
    output_path = tmp_path / "cli_summary.json"
    progress_path = tmp_path / "cli_progress.json"
    report_path = tmp_path / "cli_progress.html"

    result = CliRunner().invoke(
        app,
        [
            "train-until-boss",
            "--max-batches",
            "1",
            "--batch-runs",
            "1",
            "--train-max-steps",
            "3",
            "--eval-runs",
            "1",
            "--eval-max-steps",
            "3",
            "--character",
            "TEST",
            "--target-floor",
            "1",
            "--success-replay-passes",
            "1",
            "--train-seed-mode",
            "random",
            "--eval-seed-mode",
            "random",
            "--target-eval-successes",
            "1",
            "--model-output",
            str(model_path),
            "--output",
            str(output_path),
            "--progress-output",
            str(progress_path),
            "--report-output",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert model_path.exists()
    assert output_path.exists()
    assert progress_path.exists()
    assert report_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["resumed_from_path"] is None


def test_train_until_boss_help_says_resume_is_opt_in() -> None:
    result = CliRunner().invoke(app, ["train-until-boss", "--help"])

    assert result.exit_code == 0
    assert "--resume" in result.output
    assert "--no-resume" in result.output
