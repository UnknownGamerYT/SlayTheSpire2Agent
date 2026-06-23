from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sts2sim.cli.app import app
from sts2sim.learning.masked_ppo import (
    ACTION_FEATURE_DIM,
    PLANNING_HEAD_DIM,
    PLANNING_HEAD_SCHEMA,
    TrainingTarget,
    _empty_observation_vector,
    _masked_actor_critic_class,
    max_consecutive_target_successes,
    resolve_ppo_target,
    train_masked_ppo,
)
from sts2sim.learning.models import LearningRunResult


def _run(index: int, *, act: int, floor: int, phase: str = "map") -> LearningRunResult:
    return LearningRunResult(
        run_index=index,
        seed=index,
        character_id="TEST",
        ascension=0,
        policy="test",
        steps_taken=1,
        total_reward=0.0,
        terminated=False,
        truncated=False,
        final_phase=phase,
        final_act=act,
        final_floor=floor,
    )


def test_resolve_ppo_target_presets() -> None:
    act_2 = resolve_ppo_target("act2_boss")
    game_clear = resolve_ppo_target("game-clear")

    assert act_2 == TrainingTarget(name="act2-boss", target_act=2, target_floor=15)
    assert game_clear.target_phase == "complete"


def test_resolve_ppo_target_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="Valid targets"):
        resolve_ppo_target("heart")


def test_max_consecutive_target_successes_counts_holdout_streaks() -> None:
    target = resolve_ppo_target("act2-boss")
    runs = (
        _run(0, act=1, floor=16),
        _run(1, act=2, floor=15),
        _run(2, act=3, floor=0),
        _run(3, act=2, floor=10),
        _run(4, act=2, floor=15),
    )

    assert max_consecutive_target_successes(runs, target) == 2


def test_train_masked_ppo_help_lists_success_streak_controls() -> None:
    result = CliRunner().invoke(app, ["train-masked-ppo", "--help"])

    assert result.exit_code == 0
    assert "--target" in result.output
    assert "--target-consecut" in result.output
    assert "--hidden-layers" in result.output
    assert "--head-hidden" in result.output
    assert "--activation" in result.output
    assert "--planning-coef" in result.output
    assert "Consecutive" in result.output


def test_masked_ppo_architecture_can_scale_up() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment.")

    import torch
    from torch import nn

    model_class = _masked_actor_critic_class(nn)
    observation_dim = len(_empty_observation_vector())
    small = model_class(
        observation_dim=observation_dim,
        action_feature_dim=ACTION_FEATURE_DIM,
        hidden_size=128,
        hidden_layers=1,
        head_hidden_layers=1,
        activation="tanh",
    )
    larger = model_class(
        observation_dim=observation_dim,
        action_feature_dim=ACTION_FEATURE_DIM,
        hidden_size=256,
        hidden_layers=3,
        head_hidden_layers=2,
        activation="silu",
    )

    small_params = sum(parameter.numel() for parameter in small.parameters())
    larger_params = sum(parameter.numel() for parameter in larger.parameters())
    obs = torch.zeros((1, observation_dim), dtype=torch.float32)
    actions = torch.zeros((1, 3, ACTION_FEATURE_DIM), dtype=torch.float32)
    logits, value, planning = larger(obs, actions)

    assert small_params < 1_500_000
    assert larger_params > small_params
    assert larger_params > 2_000_000
    assert tuple(logits.shape) == (1, 3)
    assert tuple(value.shape) == (1,)
    assert tuple(planning.shape) == (1, PLANNING_HEAD_DIM)
    assert len(PLANNING_HEAD_SCHEMA) == PLANNING_HEAD_DIM


def test_train_masked_ppo_resume_continues_batches_and_progress(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment.")

    model_path = tmp_path / "ppo.pt"
    output_path = tmp_path / "ppo.json"
    progress_path = tmp_path / "ppo_progress.json"
    report_path = tmp_path / "ppo.html"

    first = train_masked_ppo(
        max_batches=1,
        train_runs_per_batch=1,
        train_max_steps=5,
        eval_runs=1,
        eval_max_steps=5,
        seed="resume-test",
        resume=False,
        target_eval_successes=99,
        target_consecutive_successes=99,
        model_output_path=model_path,
        output_path=output_path,
        progress_output_path=progress_path,
        report_output_path=report_path,
    )
    second = train_masked_ppo(
        max_batches=2,
        train_runs_per_batch=1,
        train_max_steps=5,
        eval_runs=1,
        eval_max_steps=5,
        seed="resume-test",
        resume=True,
        target_eval_successes=99,
        target_consecutive_successes=99,
        model_output_path=model_path,
        output_path=output_path,
        progress_output_path=progress_path,
        report_output_path=report_path,
    )

    assert first["batches_completed"] == 1
    assert second["resumed_from_path"] == str(model_path)
    assert second["batches_completed"] == 2
    assert second["runs_trained"] == 2
    assert [batch["batch_index"] for batch in second["batch_summaries"]] == [1, 2]
    assert [point["run_index"] for point in second["progress"]] == [0, 1]
    assert second["progress"][0]["seed"] != second["progress"][1]["seed"]
    assert second["metadata"]["network_schema_version"] == 4
    assert second["metadata"]["planning_head_schema"] == list(PLANNING_HEAD_SCHEMA)
    assert "planning_output_averages" in second["batch_summaries"][-1]
    assert "Planning Head Trends" in report_path.read_text(encoding="utf-8")


def test_train_masked_ppo_rejects_incompatible_old_checkpoint(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment.")

    import torch

    model_path = tmp_path / "old_ppo.pt"
    torch.save(
        {
            "architecture": {
                "network_schema_version": 3,
                "observation_dim": 1,
                "action_feature_dim": 1,
                "hidden_size": 1,
                "hidden_layers": 1,
                "head_hidden_layers": 1,
                "activation": "tanh",
            }
        },
        model_path,
    )

    with pytest.raises(RuntimeError, match="incompatible PPO checkpoint"):
        train_masked_ppo(
            max_batches=1,
            train_runs_per_batch=1,
            train_max_steps=1,
            eval_runs=1,
            eval_max_steps=1,
            resume=True,
            model_output_path=model_path,
            output_path=tmp_path / "ppo.json",
            progress_output_path=tmp_path / "ppo_progress.json",
            report_output_path=tmp_path / "ppo.html",
        )


def test_train_masked_ppo_explains_missing_torch(tmp_path) -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("PyTorch is installed in this environment.")

    with pytest.raises(RuntimeError, match="uv sync --extra rl"):
        train_masked_ppo(
            max_batches=1,
            train_runs_per_batch=1,
            train_max_steps=1,
            eval_runs=1,
            eval_max_steps=1,
            resume=False,
            model_output_path=tmp_path / "ppo.pt",
            output_path=tmp_path / "ppo.json",
            progress_output_path=tmp_path / "ppo_progress.json",
            report_output_path=tmp_path / "ppo.html",
        )
