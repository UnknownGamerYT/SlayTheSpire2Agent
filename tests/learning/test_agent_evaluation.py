from __future__ import annotations

import json

from typer.testing import CliRunner

from sts2sim.cli.app import app
from sts2sim.learning import evaluate_agent_baselines


def test_evaluate_agent_baselines_compares_fixed_seed_policies(tmp_path) -> None:
    output_path = tmp_path / "baselines.json"
    report_path = tmp_path / "baselines.html"

    result = evaluate_agent_baselines(
        runs=2,
        max_steps=3,
        start_seed=100,
        character_id="TEST",
        policies=("random", "q_learning", "strategic"),
        output_path=output_path,
        report_output_path=report_path,
    )

    assert result.policies == ("random", "q_learning", "strategic")
    assert {summary.policy for summary in result.summaries} == set(result.policies)
    assert all(len(result.runs_by_policy[policy]) == 2 for policy in result.policies)
    assert all(len(result.progress_by_policy[policy]) == 2 for policy in result.policies)
    assert all(summary.failed_to_continue == 0 for summary in result.summaries)
    assert output_path.exists()
    assert report_path.exists()
    assert "Agent Baseline Comparison" in report_path.read_text(encoding="utf-8")
    assert json.loads(output_path.read_text(encoding="utf-8"))["policies"] == list(
        result.policies
    )


def test_evaluate_agents_cli_smoke(tmp_path) -> None:
    output_path = tmp_path / "cli_baselines.json"
    report_path = tmp_path / "cli_baselines.html"

    result = CliRunner().invoke(
        app,
        [
            "evaluate-agents",
            "--runs",
            "1",
            "--max-steps",
            "2",
            "--character",
            "TEST",
            "--output",
            str(output_path),
            "--report-output",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert report_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["policies"] == ["random", "q_learning", "strategic"]


def test_q_learning_seed_two_potion_sequence_does_not_fail_to_continue() -> None:
    result = evaluate_agent_baselines(
        runs=1,
        max_steps=7,
        start_seed=2,
        character_id="TEST",
        policies=("q_learning",),
        output_path=None,
        report_output_path=None,
    )

    run = result.runs_by_policy["q_learning"][0]
    assert run.error is None
    assert run.failed_to_continue is False
    assert result.summaries[0].failed_to_continue == 0
