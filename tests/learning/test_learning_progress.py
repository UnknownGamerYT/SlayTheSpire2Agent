from __future__ import annotations

import json

from typer.testing import CliRunner

from sts2sim.cli.app import app
from sts2sim.learning import (
    build_learning_progress_report,
    collect_random_rollouts,
    load_learning_progress,
    train_q_learning,
)


def test_rollout_output_includes_progress_points(tmp_path) -> None:
    output_path = tmp_path / "rollouts.json"

    result = collect_random_rollouts(
        runs=3,
        max_steps=2,
        start_seed=90,
        character_id="TEST",
        output_path=output_path,
        include_steps=False,
        include_history=False,
    )

    assert len(result.progress) == 3
    assert result.progress[0].moving_average_floor >= 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload["progress"]) == 3


def test_progress_report_builds_from_rollout_json(tmp_path) -> None:
    input_path = tmp_path / "rollouts.json"
    report_path = tmp_path / "progress.html"
    collect_random_rollouts(
        runs=2,
        max_steps=2,
        start_seed=93,
        character_id="TEST",
        output_path=input_path,
        include_steps=False,
        include_history=False,
    )

    result = build_learning_progress_report(
        input_path=input_path,
        output_path=report_path,
        title="Test Learning Progress",
    )

    html = report_path.read_text(encoding="utf-8")
    assert result["points"] == 2
    assert "Test Learning Progress" in html
    assert "Reward By Run" in html
    assert "Floor Progress" in html


def test_q_learning_training_writes_progress_and_report(tmp_path) -> None:
    model_path = tmp_path / "model.json"
    progress_path = tmp_path / "progress.json"
    report_path = tmp_path / "training.html"

    result = train_q_learning(
        runs=2,
        max_steps=3,
        seed=95,
        character_id="TEST",
        output_path=model_path,
        progress_output_path=progress_path,
        report_output_path=report_path,
    )

    assert len(result.progress) == 2
    assert progress_path.exists()
    assert report_path.exists()
    assert load_learning_progress(progress_path)
    assert "Q-learning Training Progress" in report_path.read_text(encoding="utf-8")


def test_learning_progress_report_cli(tmp_path) -> None:
    input_path = tmp_path / "rollouts.json"
    report_path = tmp_path / "cli_progress.html"
    collect_random_rollouts(
        runs=1,
        max_steps=2,
        start_seed=97,
        character_id="TEST",
        output_path=input_path,
        include_steps=False,
        include_history=False,
    )

    result = CliRunner().invoke(
        app,
        [
            "learning-progress-report",
            str(input_path),
            "--output",
            str(report_path),
            "--title",
            "CLI Progress",
        ],
    )

    assert result.exit_code == 0
    assert report_path.exists()
    assert "CLI Progress" in report_path.read_text(encoding="utf-8")
