from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from sts2sim.cli.app import app
from sts2sim.learning.curriculum import (
    _migrate_legacy_active_checkpoint,
    _stage_summary,
    _target_passed_by_checkpoint_chain,
    resolve_curriculum_stages,
    train_masked_ppo_curriculum,
)


def test_resolve_curriculum_stages_defaults_and_commas() -> None:
    assert resolve_curriculum_stages() == (
        "act1-boss",
        "act2-boss",
        "act3-boss",
    )
    assert resolve_curriculum_stages("act1_boss, act2-boss") == (
        "act1-boss",
        "act2-boss",
    )


def test_train_masked_ppo_curriculum_advances_until_stage_fails(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    progress_payloads: list[dict[str, Any]] = []

    def progress_reporter(payload: dict[str, Any]) -> None:
        progress_payloads.append(payload)

    def fake_trainer(**kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        Path(kwargs["model_output_path"]).write_bytes(
            str(kwargs["target"]).encode("utf-8")
        )
        reporter = kwargs.get("progress_reporter")
        if callable(reporter):
            reporter({"event": "batch_saved", "target": kwargs["target"]})
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
        stages=("act1-boss", "act2-boss", "act3-boss"),
        max_batches=1,
        train_runs_per_batch=2,
        eval_runs=2,
        checkpoint_dir=tmp_path / "checkpoints",
        report_dir=tmp_path / "reports",
        output_path=tmp_path / "curriculum.json",
        report_output_path=tmp_path / "curriculum.html",
        target_success_rate=0.95,
        rollout_workers=2,
        rollout_inference="batched-gpu",
        history_mode="off",
        envs_per_worker=3,
        policy_server_min_batch=4,
        policy_server_max_wait_ms=7,
        progress_reporter=progress_reporter,
        trainer=fake_trainer,
    )
    assert result["completed_curriculum"] is False
    assert result["stages_started"] == 2
    assert result["stages_completed"] == 1
    assert result["current_stage"] == "act2-boss"
    assert "did not meet comfort criteria" in str(result["stopped_reason"])
    assert calls[0]["resume_from_path"] is None
    assert calls[0]["resume"] is True
    assert calls[0]["hidden_size"] == 256
    assert calls[0]["hidden_layers"] == 3
    assert calls[0]["head_hidden_layers"] == 2
    assert calls[0]["activation"] == "silu"
    assert calls[0]["planning_coef"] == 0.1
    assert calls[0]["teacher_mix"] == 0.0
    assert calls[0]["imitation_coef"] == 0.0
    assert calls[0]["target_success_rate"] == 0.95
    assert calls[0]["device"] == "auto"
    assert calls[0]["rollout_workers"] == 2
    assert calls[0]["rollout_inference"] == "batched-gpu"
    assert calls[0]["history_mode"] == "off"
    assert calls[0]["envs_per_worker"] == 3
    assert calls[0]["policy_server_min_batch"] == 4
    assert calls[0]["policy_server_max_wait_ms"] == 7
    assert calls[0]["progress_reporter"] is progress_reporter
    assert calls[1]["resume_from_path"] == tmp_path / "checkpoints" / (
        "ppo_curriculum_latest.pt"
    )
    assert calls[1]["resume"] is True
    assert calls[1]["target_success_rate"] == 0.95
    assert progress_payloads[0]["event"] == "curriculum_resume_check"
    assert progress_payloads[0]["latest_checkpoint_check"]["decision"] == "missing"
    assert [
        payload for payload in progress_payloads if payload["event"] == "batch_saved"
    ] == [
        {"event": "batch_saved", "target": "act1-boss"},
        {"event": "batch_saved", "target": "act2-boss"},
    ]
    assert result["metadata"]["target_success_rate"] == 0.95
    assert result["metadata"]["rollout_workers"] == 2
    assert result["metadata"]["rollout_inference"] == "batched-gpu"
    assert result["metadata"]["history_mode"] == "off"
    assert result["metadata"]["envs_per_worker"] == 3
    assert result["metadata"]["policy_server_min_batch"] == 4
    assert result["metadata"]["policy_server_max_wait_ms"] == 7
    assert result["batch_metric_summary"]["batches"] == 2
    assert result["batch_metrics"][-1]["stage"] == "act2-boss"
    assert (tmp_path / "checkpoints" / "ppo_curriculum_latest.pt").exists()
    assert (tmp_path / "checkpoints" / "ppo_curriculum_act1_boss.pt").exists()
    assert not (tmp_path / "checkpoints" / "ppo_curriculum_act2_boss.pt").exists()
    assert (tmp_path / "checkpoints" / "ppo_curriculum_act1_boss.pt").read_bytes() == (
        b"act1-boss"
    )
    assert (tmp_path / "checkpoints" / "ppo_curriculum_latest.pt").read_bytes() == (
        b"act2-boss"
    )
    assert (tmp_path / "curriculum.json").exists()
    assert (tmp_path / "curriculum.html").exists()


def test_later_checkpoint_can_verify_an_earlier_curriculum_target() -> None:
    checks = {
        "act1-boss": {
            "target_checks": [{"target": "act1-boss", "passed": False}],
        },
        "act2-boss": {
            "target_checks": [
                {"target": "act1-boss", "passed": True},
                {"target": "act2-boss", "passed": False},
            ],
        },
    }

    assert _target_passed_by_checkpoint_chain(checks, "act1-boss") is True
    assert _target_passed_by_checkpoint_chain(checks, "act2-boss") is False


def test_curriculum_promotes_the_stage_champion_not_mutable_latest(tmp_path: Path) -> None:
    def fake_trainer(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["model_output_path"]).write_bytes(b"latest-regressed")
        Path(kwargs["champion_model_path"]).write_bytes(b"champion-policy")
        return {
            "reached_target": True,
            "reached_batch": 2,
            "batches_completed": 2,
            "runs_trained": 4,
            "total_steps": 5,
            "champion": {
                "checkpoint_path": str(kwargs["champion_model_path"]),
                "best": {"passed": True, "batch_index": 2},
            },
            "batch_summaries": [{"batch_index": 2}],
        }

    train_masked_ppo_curriculum(
        stages=("act1-boss",),
        max_batches=1,
        checkpoint_dir=tmp_path / "checkpoints",
        report_dir=tmp_path / "reports",
        output_path=tmp_path / "curriculum.json",
        report_output_path=tmp_path / "curriculum.html",
        trainer=fake_trainer,
    )

    checkpoints = tmp_path / "checkpoints"
    assert (checkpoints / "ppo_curriculum_act1_boss.pt").read_bytes() == b"champion-policy"
    assert (checkpoints / "ppo_curriculum_latest.pt").read_bytes() == b"champion-policy"


def test_legacy_active_checkpoint_migrates_to_latest_slot(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    legacy_path = checkpoint_root / "ppo_curriculum_act2_boss.pt"
    legacy_path.write_bytes(b"active act 2 checkpoint")
    output_path = tmp_path / "curriculum.json"
    output_path.write_text(
        '{"stages_requested": ["act1-boss", "act2-boss"], '
        '"current_stage": "act2-boss"}',
        encoding="utf-8",
    )
    latest_path = checkpoint_root / "ppo_curriculum_latest.pt"

    migrated = _migrate_legacy_active_checkpoint(
        output_path=output_path,
        resolved_stages=("act1-boss", "act2-boss"),
        checkpoint_root=checkpoint_root,
        run_name="ppo_curriculum",
        latest_model_path=latest_path,
        resume=True,
    )

    assert migrated == legacy_path
    assert not legacy_path.exists()
    assert latest_path.read_bytes() == b"active act 2 checkpoint"


def test_resume_recovers_completed_stage_reports_after_stale_root_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    report_root = tmp_path / "reports"
    checkpoint_root.mkdir()
    report_root.mkdir()
    (checkpoint_root / "ppo_curriculum_act1_boss.pt").write_bytes(b"act 1")
    (checkpoint_root / "ppo_curriculum_act2_boss.pt").write_bytes(b"act 2")
    (checkpoint_root / "ppo_curriculum_latest.pt").write_bytes(b"stale act 1")
    output_path = report_root / "ppo_curriculum_latest.json"
    output_path.write_text(
        json.dumps(
            {
                "stages_requested": ["act1-boss", "act2-boss", "act3-boss"],
                "current_stage": "act1-boss",
                "stage_summaries": [
                    {"stage": "act1-boss", "reached_target": True}
                ],
            }
        ),
        encoding="utf-8",
    )
    (report_root / "ppo_curriculum_act2_boss_latest.json").write_text(
        json.dumps(
            {
                "target": {"name": "act2-boss"},
                "reached_target": True,
                "reached_batch": 4,
                "batches_completed": 4,
                "runs_trained": 10,
                "total_steps": 20,
                "batch_summaries": [],
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, Any]] = []

    def fake_trainer(**kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        Path(kwargs["model_output_path"]).write_bytes(b"act 3")
        return {
            "reached_target": False,
            "reached_batch": None,
            "batches_completed": 1,
            "runs_trained": 1,
            "total_steps": 1,
            "batch_summaries": [],
        }

    monkeypatch.setattr(
        "sts2sim.learning.curriculum.evaluate_masked_ppo_checkpoint",
        lambda _path, **kwargs: {
            "compatible": True,
            "passed": True,
            "target": kwargs["target"],
            "target_success_rate": 1.0,
            "target_successes": kwargs["eval_runs"],
            "eval_runs": kwargs["eval_runs"],
            "max_consecutive_successes": kwargs["eval_runs"],
            "reason": "target criteria met",
        },
    )

    result = train_masked_ppo_curriculum(
        stages=("act1-boss", "act2-boss", "act3-boss"),
        max_batches=1,
        train_runs_per_batch=1,
        eval_runs=1,
        checkpoint_dir=checkpoint_root,
        report_dir=report_root,
        output_path=output_path,
        report_output_path=report_root / "ppo_curriculum_latest.html",
        trainer=fake_trainer,
    )

    assert [call["target"] for call in calls] == ["act3-boss"]
    assert calls[0]["resume_from_path"] == checkpoint_root / "ppo_curriculum_act2_boss.pt"
    assert result["current_stage"] == "act3-boss"


def test_curriculum_resume_skips_completed_stage_and_uses_current_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_trainer(**kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        Path(kwargs["model_output_path"]).touch()
        reached = str(kwargs["target"]) == "act1-boss"
        return {
            "reached_target": reached,
            "reached_batch": 1 if reached else None,
            "batches_completed": 1,
            "runs_trained": 1,
            "total_steps": 1,
            "batch_summaries": [],
        }

    options = {
        "stages": ("act1-boss", "act2-boss"),
        "max_batches": 1,
        "train_runs_per_batch": 1,
        "eval_runs": 1,
        "checkpoint_dir": tmp_path / "checkpoints",
        "report_dir": tmp_path / "reports",
        "output_path": tmp_path / "curriculum.json",
        "report_output_path": tmp_path / "curriculum.html",
        "trainer": fake_trainer,
    }
    monkeypatch.setattr(
        "sts2sim.learning.curriculum.evaluate_masked_ppo_checkpoint",
        lambda _path, **kwargs: {
            "compatible": True,
            "passed": True,
            "target": kwargs["target"],
            "target_success_rate": 1.0,
            "target_successes": kwargs["eval_runs"],
            "eval_runs": kwargs["eval_runs"],
            "max_consecutive_successes": kwargs["eval_runs"],
            "reason": "target criteria met",
        },
    )

    train_masked_ppo_curriculum(**options)
    resumed = train_masked_ppo_curriculum(**options)

    assert [call["target"] for call in calls] == ["act1-boss", "act2-boss", "act2-boss"]
    assert calls[-1]["resume_from_path"] == tmp_path / "checkpoints" / (
        "ppo_curriculum_latest.pt"
    )
    assert resumed["stages_started"] == 2
    assert resumed["stage_summaries"][0]["reached_target"] is True
    assert resumed["current_stage"] == "act2-boss"


def test_curriculum_resume_skips_through_act3_and_stays_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    validated_checkpoints: list[str] = []
    completed_targets = {"act1-boss", "act2-boss", "act3-boss"}

    def fake_trainer(**kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        Path(kwargs["model_output_path"]).touch()
        reached = str(kwargs["target"]) in completed_targets
        return {
            "reached_target": reached,
            "reached_batch": 1 if reached else None,
            "batches_completed": 1,
            "runs_trained": 1,
            "total_steps": 1,
            "batch_summaries": [],
        }

    options = {
        "stages": ("act1-boss", "act2-boss", "act3-boss"),
        "max_batches": 1,
        "train_runs_per_batch": 1,
        "eval_runs": 1,
        "checkpoint_dir": tmp_path / "checkpoints",
        "report_dir": tmp_path / "reports",
        "output_path": tmp_path / "curriculum.json",
        "report_output_path": tmp_path / "curriculum.html",
        "trainer": fake_trainer,
    }
    def evaluate_checkpoint(path: Path, **kwargs: Any) -> dict[str, Any]:
        validated_checkpoints.append(f"{path.name}:{kwargs['target']}")
        return {
            "compatible": True,
            "passed": True,
            "target": kwargs["target"],
            "target_success_rate": 1.0,
            "target_successes": kwargs["eval_runs"],
            "eval_runs": kwargs["eval_runs"],
            "max_consecutive_successes": kwargs["eval_runs"],
            "reason": "target criteria met",
        }

    monkeypatch.setattr(
        "sts2sim.learning.curriculum.evaluate_masked_ppo_checkpoint",
        evaluate_checkpoint,
    )

    first = train_masked_ppo_curriculum(**options)
    resumed = train_masked_ppo_curriculum(**options)

    assert [call["target"] for call in calls] == [
        "act1-boss",
        "act2-boss",
        "act3-boss",
    ]
    assert first["completed_curriculum"] is True
    assert resumed["completed_curriculum"] is True
    assert resumed["stages_started"] == 3
    assert resumed["stages_completed"] == 3
    assert resumed["current_stage"] == "act3-boss"
    assert resumed["stage_summaries"][-1]["reached_target"] is True
    assert validated_checkpoints == [
        "ppo_curriculum_latest.pt:act1-boss",
        "ppo_curriculum_latest.pt:act2-boss",
        "ppo_curriculum_act1_boss.pt:act1-boss",
        "ppo_curriculum_act2_boss.pt:act1-boss",
        "ppo_curriculum_act2_boss.pt:act2-boss",
        "ppo_curriculum_act3_boss.pt:act1-boss",
        "ppo_curriculum_act3_boss.pt:act2-boss",
        "ppo_curriculum_act3_boss.pt:act3-boss",
    ]


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
    assert "consecutive" in result.output.lower()
    assert "--report-output" in result.output
    assert "--hidden-layers" in result.output
    assert "--head-hidden" in result.output
    assert "--activation" in result.output
    assert "--planning-coef" in result.output
    assert "--teacher-mix" in result.output
    assert "--imitation-coef" in result.output
    assert "--target-succes" in result.output
    assert "--device" in result.output
    assert "rollout" in result.output.lower()
    assert "inference" in result.output.lower()
    assert "terminal" in result.output.lower()
    assert "--resume" in result.output
    assert "--no-resume" in result.output
