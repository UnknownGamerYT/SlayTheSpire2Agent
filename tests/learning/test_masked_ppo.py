from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sts2sim import Sts2Env
from sts2sim.cli.app import (
    _batch_progress_line,
    _compact_training_result,
    _diagnostic_progress_line,
    _reward_signal_line,
    _throughput_progress_line,
    app,
)
from sts2sim.learning.content_vocab import (
    CONTENT_IDENTITY_EMBED_DIM,
    CONTENT_IDENTITY_SLOTS,
    load_content_vocab,
)
from sts2sim.learning.masked_ppo import (
    ACTION_FEATURE_DIM,
    PLANNING_HEAD_DIM,
    PLANNING_HEAD_SCHEMA,
    TrainingTarget,
    _accumulate_run_diagnostics,
    _action_features,
    _empty_observation_vector,
    _final_run_diagnostics,
    _masked_actor_critic_class,
    _observation_vector,
    _policy_input,
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
    assert "--target-consec" in result.output
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
    assert "--until-stopped" in result.output
    assert "--terminal-prog" in result.output
    assert "Consecutive" in result.output
    assert "--resume" in result.output
    assert "--no-resume" in result.output


def test_training_cli_summary_omits_embedded_histories() -> None:
    compact = _compact_training_result(
        {
            "algorithm": "masked_action_descriptor_ppo",
            "batches_completed": 2,
            "until_stopped": True,
            "runs_trained": 16,
            "total_steps": 1235,
            "output_path": "reports/latest.json",
            "batch_summaries": [
                {
                    "batch_index": 2,
                    "evaluation_average_reward": 1.25,
                    "evaluation_target_success_rate": 0.5,
                }
            ],
            "highlight_run_histories": {
                "schema_version": 2,
                "generated_at": "2026-06-24T00:00:00Z",
                "best": {
                    "seed": 1,
                    "history": {"steps": [{"big": "payload"}]},
                    "json_path": "best.json",
                    "html_path": "best.html",
                    "map_path": "best.txt",
                },
            },
        }
    )

    assert compact["runs_trained"] == 16
    assert compact["until_stopped"] is True
    assert compact["latest_batch"]["batch_index"] == 2
    assert compact["highlight_run_histories"]["generated_at"] == "2026-06-24T00:00:00Z"
    assert "history" not in compact["highlight_run_histories"]["best"]


def test_training_terminal_batch_progress_shows_health_snapshot() -> None:
    line = _batch_progress_line(
        {
            "batch_index": 7,
            "evaluation_target_success_rate": 0.25,
            "target_success_rate_threshold": 0.75,
            "target_successes": 1,
            "eval_runs": 4,
            "evaluation_average_floor": 8.5,
            "evaluation_best_floor": 14,
            "evaluation_average_reward": 12.25,
            "evaluation_best_reward": 33.5,
            "evaluation_max_consecutive_successes": 2,
            "evaluation_errors": 1,
            "evaluation_failed_to_continue": 3,
            "runs_trained": 56,
        }
    )

    assert "Batch 7 saved" in line
    assert "success=0.250/0.750 (1/4 eval)" in line
    assert "avg_floor=8.50" in line
    assert "best_floor=14" in line
    assert "avg_reward=12.25" in line
    assert "best_reward=33.50" in line
    assert "consec=2" in line
    assert "errors=1" in line
    assert "failed=3" in line
    assert "runs_trained=56" in line


def test_training_terminal_reward_and_diagnostic_lines_are_compact() -> None:
    reward_line = _reward_signal_line(
        {
            "reward_component_averages": {
                "total": 9.5,
                "combat_win_reward": 2.0,
                "boss_reward": 0.0,
                "enemy_hp_progress_reward": 4.25,
                "hp_loss_penalty": -1.5,
                "gold_reward": 0.3,
                "reward_skip_penalty": -0.2,
                "deck_capability_reward": 0.1,
            }
        }
    )
    diagnostic_line = _diagnostic_progress_line(
        {
            "diagnostic_averages": {
                "reward_card_picked": 1.0,
                "reward_card_presented": 2.0,
                "reward_card_skipped": 0.5,
                "reward_gold_picked": 4.0,
                "reward_gold_presented": 5.0,
                "reward_gold_unclaimed": 1.0,
                "reward_relic_picked": 0.5,
                "reward_relic_presented": 1.0,
                "reward_relic_unclaimed": 0.4,
                "reward_potion_picked": 0.25,
                "reward_potion_presented": 0.75,
                "reward_potion_skipped": 0.25,
                "final_deck_size": 13.0,
                "final_unknown_card_count": 2.0,
                "final_gold": 88.0,
            }
        }
    )

    assert reward_line.startswith("  reward avg:")
    assert "combat=2.00" in reward_line
    assert "enemy_hp=4.25" in reward_line
    assert "hp_loss=-1.50" in reward_line
    assert "deck=0.10" in reward_line
    assert diagnostic_line.startswith("  deck/items avg:")
    assert "cards=1.00/2.00 missed=0.50" in diagnostic_line
    assert "gold=4.00/5.00 missed=1.00" in diagnostic_line
    assert "relics=0.50/1.00 missed=0.40" in diagnostic_line
    assert "potions=0.25/0.75 missed=0.25" in diagnostic_line
    assert "final_deck=13.0" in diagnostic_line
    assert "final_gold=88.0" in diagnostic_line


def test_training_terminal_throughput_line_is_compact() -> None:
    line = _throughput_progress_line(
        {
            "throughput": {
                "env_steps_per_second": 123.456,
                "runs_per_second": 4.25,
                "active_env_streams": 8,
                "policy_server_min_batch": 4,
                "policy_server_max_wait_ms": 15,
            }
        }
    )

    assert line == (
        "  throughput: steps/s=123.5, runs/s=4.25, "
        "active_envs=8, min_batch=4, wait_ms=15"
    )


def test_reward_pickup_diagnostics_count_presented_items_once() -> None:
    reward_state = {
        "phase": "reward",
        "player": {"gold": 10, "deck": []},
        "reward": {
            "reward_id": "combat:1",
            "source": "combat",
            "gold": 19,
            "card_options": ["strike", "defend", "bash"],
            "relic_ids": ["anchor", "bag_of_marbles"],
            "potion_ids": ["fire_potion", "block_potion"],
        },
    }
    proceed_state = {
        "phase": "reward",
        "player": {"gold": 29, "deck": []},
        "reward": {
            "reward_id": "combat:1",
            "source": "combat",
            "gold": 19,
            "gold_claimed": True,
            "card_options": ["strike", "defend", "bash"],
            "card_claimed": True,
            "relic_ids": ["anchor", "bag_of_marbles"],
            "skipped_relic_ids": ["anchor"],
            "potion_ids": ["fire_potion", "block_potion"],
        },
    }
    closed_state = {"phase": "map", "player": {"gold": 29, "deck": []}, "reward": None}
    diagnostics: dict[str, object] = {}

    _accumulate_run_diagnostics(
        diagnostics,
        before_state=reward_state,
        after_state=reward_state,
        action_descriptor={"type": "take_reward_gold"},
    )
    _accumulate_run_diagnostics(
        diagnostics,
        before_state=reward_state,
        after_state=reward_state,
        action_descriptor={"type": "take_reward_card"},
    )
    _accumulate_run_diagnostics(
        diagnostics,
        before_state=reward_state,
        after_state=reward_state,
        action_descriptor={
            "type": "skip_reward",
            "reward_choice": {"skip_kind": "relic"},
        },
    )
    _accumulate_run_diagnostics(
        diagnostics,
        before_state=proceed_state,
        after_state=closed_state,
        action_descriptor={"type": "proceed"},
    )

    result = _final_run_diagnostics(diagnostics, closed_state)

    assert result["reward_gold_presented"] == 1.0
    assert result["reward_card_presented"] == 1.0
    assert result["reward_relic_presented"] == 2.0
    assert result["reward_potion_presented"] == 2.0
    assert result["reward_gold_picked"] == 1.0
    assert result["reward_card_picked"] == 1.0
    assert result["reward_relic_skipped"] == 1.0
    assert result["reward_relic_unclaimed"] == 1.0
    assert result["reward_potion_unclaimed"] == 2.0
    assert "reward_gold_unclaimed" not in result
    assert "__reward_presentation_seen__" not in result


def test_policy_input_cache_matches_direct_encoding() -> None:
    env = Sts2Env(seed=21, character_id="TEST", ascension=0, max_actions=16)
    observation, info = env.reset()

    packed = _policy_input(observation, info)
    again = _policy_input(observation, info)

    assert packed is again
    assert info["_policy_input"] is packed
    assert packed.observation_vector == _observation_vector(observation)
    assert packed.action_features[0] == _action_features(info["action_space"][0])
    assert packed.action_ids == tuple(descriptor["id"] for descriptor in info["action_space"])


def test_masked_ppo_architecture_can_scale_up() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment.")

    import torch
    from torch import nn

    model_class = _masked_actor_critic_class(nn)
    observation_dim = len(_empty_observation_vector())
    vocab = load_content_vocab()
    small = model_class(
        observation_dim=observation_dim,
        action_feature_dim=ACTION_FEATURE_DIM,
        content_vocab_size=vocab.size,
        hidden_size=128,
        hidden_layers=1,
        head_hidden_layers=1,
        activation="tanh",
    )
    larger = model_class(
        observation_dim=observation_dim,
        action_feature_dim=ACTION_FEATURE_DIM,
        content_vocab_size=vocab.size,
        hidden_size=256,
        hidden_layers=3,
        head_hidden_layers=2,
        activation="silu",
    )

    small_params = sum(parameter.numel() for parameter in small.parameters())
    larger_params = sum(parameter.numel() for parameter in larger.parameters())
    obs = torch.zeros((1, observation_dim), dtype=torch.float32)
    actions = torch.zeros((1, 3, ACTION_FEATURE_DIM), dtype=torch.float32)
    action_ids = torch.zeros((1, 3, CONTENT_IDENTITY_SLOTS), dtype=torch.long)
    logits, value, planning = larger(obs, actions, action_ids)

    assert small_params < 1_500_000
    assert larger_params > small_params
    assert larger_params > 2_000_000
    assert larger.action_encoder[0].in_features == (
        ACTION_FEATURE_DIM + CONTENT_IDENTITY_SLOTS * CONTENT_IDENTITY_EMBED_DIM
    )
    assert "content_embedding.weight" in larger.state_dict()
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
        target_eval_successes=99,
        target_consecutive_successes=99,
        model_output_path=model_path,
        output_path=output_path,
        progress_output_path=progress_path,
        report_output_path=report_path,
    )

    assert first["batches_completed"] == 1
    assert first["resumed_from_path"] is None
    assert first["previous_batches"] == 0
    assert first["requested_new_batches"] == 1
    assert first["batch_limit"] == 1
    assert second["resumed_from_path"] == str(model_path)
    assert second["previous_batches"] == 1
    assert second["requested_new_batches"] == 2
    assert second["batch_limit"] == 3
    assert second["batches_completed"] == 3
    assert second["runs_trained"] == 3
    assert [batch["batch_index"] for batch in second["batch_summaries"]] == [1, 2, 3]
    assert [point["run_index"] for point in second["progress"]] == [0, 1, 2]
    assert len({point["seed"] for point in second["progress"]}) == 3
    assert second["metadata"]["network_schema_version"] == 5
    assert second["metadata"]["content_vocab_size"] == load_content_vocab().size
    assert "content_vocab_checksum" in second["metadata"]
    assert second["metadata"]["planning_head_schema"] == list(PLANNING_HEAD_SCHEMA)
    assert second["metadata"]["target_success_rate"] == 0.0
    assert second["metadata"]["reward_schema_version"] == 5
    assert second["metadata"]["rollout_workers"] == 1
    assert second["metadata"]["rollout_inference"] == "worker"
    assert second["metadata"]["history_mode"] == "highlights"
    assert second["metadata"]["envs_per_worker"] == 1
    assert second["metadata"]["policy_server_min_batch"] == 1
    assert second["metadata"]["policy_server_max_wait_ms"] == 20
    assert "throughput" in second["batch_summaries"][-1]
    assert second["batch_summaries"][-1]["throughput"]["env_steps_per_second"] > 0.0
    assert "planning_output_averages" in second["batch_summaries"][-1]
    assert "reward_component_averages" in second["batch_summaries"][-1]
    assert "diagnostic_averages" in second["batch_summaries"][-1]
    assert "total" in second["batch_summaries"][-1]["reward_component_averages"]
    histories = second["highlight_run_histories"]
    for role in ("best", "worst"):
        entry = histories[role]
        html_path = Path(entry["html_path"])
        json_path = Path(entry["json_path"])
        map_path = Path(entry["map_path"])
        assert html_path.exists()
        assert json_path.exists()
        assert map_path.exists()
        assert entry["generated_at"].endswith("Z")
        html_text = html_path.read_text(encoding="utf-8")
        assert "Map Path" in html_text
        assert "Timeline" in html_text
        assert "Generated At" in html_text
        history_payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert history_payload["generated_at"] == entry["generated_at"]
        assert history_payload["highlight_role"] == role
        assert history_payload["steps"]
        assert "reward_total" in history_payload["steps"][0]["decision"]
        map_text = map_path.read_text(encoding="utf-8")
        assert f"Generated at: {entry['generated_at']}" in map_text
        assert "Legend:" in map_text
    report_text = report_path.read_text(encoding="utf-8")
    assert "Planning Head Trends" in report_text
    assert "Reward Component Trends" in report_text
    assert "Reward And Deck Diagnostics" in report_text
    assert "Throughput" in report_text
    assert "Best And Worst Evaluation Run Histories" in report_text
    assert "Generated" in report_text
    assert "ppo_best_run_history.html" in report_text


def test_train_masked_ppo_history_off_skips_highlight_artifacts(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment.")

    result = train_masked_ppo(
        max_batches=1,
        train_runs_per_batch=0,
        train_max_steps=1,
        eval_runs=1,
        eval_max_steps=1,
        seed="history-off-test",
        resume=False,
        target_eval_successes=99,
        target_consecutive_successes=99,
        history_mode="off",
        model_output_path=tmp_path / "ppo.pt",
        output_path=tmp_path / "ppo.json",
        progress_output_path=tmp_path / "ppo_progress.json",
        report_output_path=tmp_path / "ppo.html",
    )

    assert result["metadata"]["history_mode"] == "off"
    assert result["highlight_run_histories"] == {}
    assert not (tmp_path / "ppo_best_run_history.json").exists()
    assert not (tmp_path / "ppo_worst_run_history.json").exists()


def test_train_masked_ppo_rejects_multi_envs_without_batched_gpu(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment.")

    with pytest.raises(ValueError, match="envs_per_worker"):
        train_masked_ppo(
            max_batches=1,
            train_runs_per_batch=1,
            train_max_steps=1,
            eval_runs=1,
            eval_max_steps=1,
            resume=False,
            envs_per_worker=2,
            rollout_inference="worker",
            model_output_path=tmp_path / "ppo.pt",
            output_path=tmp_path / "ppo.json",
            progress_output_path=tmp_path / "ppo_progress.json",
            report_output_path=tmp_path / "ppo.html",
        )


def test_train_masked_ppo_batched_gpu_multi_env_smoke(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment.")

    result = train_masked_ppo(
        max_batches=1,
        train_runs_per_batch=2,
        train_max_steps=1,
        eval_runs=2,
        eval_max_steps=1,
        seed="batched-gpu-smoke",
        resume=False,
        target_eval_successes=99,
        target_consecutive_successes=99,
        rollout_workers=2,
        rollout_inference="batched-gpu",
        envs_per_worker=2,
        policy_server_min_batch=2,
        policy_server_max_wait_ms=5,
        history_mode="off",
        ppo_epochs=1,
        minibatch_size=4,
        model_output_path=tmp_path / "ppo.pt",
        output_path=tmp_path / "ppo.json",
        progress_output_path=tmp_path / "ppo_progress.json",
        report_output_path=tmp_path / "ppo.html",
    )

    assert result["runs_trained"] == 2
    assert len(result["evaluation_progress"]) == 2
    assert result["metadata"]["envs_per_worker"] == 2
    assert result["metadata"]["active_env_streams"] == 4
    assert result["metadata"]["policy_server_min_batch"] == 2
    assert result["metadata"]["policy_server_max_wait_ms"] == 5
    throughput = result["batch_summaries"][-1]["throughput"]
    assert throughput["active_env_streams"] == 4
    assert throughput["rollout_inference"] == "batched-gpu"


def test_train_masked_ppo_rejects_incompatible_old_checkpoint(tmp_path: Path) -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in this environment.")

    import torch

    model_path = tmp_path / "old_ppo.pt"
    torch.save(
        {
            "architecture": {
                "network_schema_version": 4,
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
