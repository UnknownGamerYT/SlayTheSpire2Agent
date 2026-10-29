from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from sts2sim import Sts2Env, legal_actions
from sts2sim.cli.app import app
from sts2sim.learning.masked_ppo import train_masked_ppo
from sts2sim.learning.policy import (
    CheckpointPolicyError,
    MaskedPpoPolicy,
    PolicyInferenceError,
    evaluate_ppo,
)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("torch")
    checkpoint = tmp_path_factory.mktemp("ppo-policy") / "tiny.pt"
    train_masked_ppo(
        max_batches=1,
        train_runs_per_batch=1,
        train_max_steps=2,
        eval_runs=0,
        eval_max_steps=1,
        champion_eval_runs=1,
        hidden_size=8,
        hidden_layers=1,
        head_hidden_layers=1,
        ppo_epochs=1,
        minibatch_size=2,
        history_mode="off",
        model_output_path=checkpoint,
        output_path=None,
        progress_output_path=None,
        report_output_path=None,
        resume=False,
        device="cpu",
        seed="inference-fixture",
    )
    return checkpoint


def test_trained_checkpoint_selects_legal_actions_on_a_fresh_environment(
    tiny_checkpoint: Path,
) -> None:
    policy = MaskedPpoPolicy(tiny_checkpoint, device="cpu")
    assert policy.configuration["hidden_size"] == 8
    env = Sts2Env(seed=912, max_episode_steps=5)
    observation, info = env.reset()
    for _ in range(5):
        decision = policy.predict(observation, info)
        assert observation["action_mask"][decision.action_id] == 1
        assert decision.action in [
            action.model_dump(mode="json", exclude_none=True) for action in legal_actions(env.state)
        ]
        env.set_pending_policy_output(decision.policy_output)
        observation, _, terminated, truncated, info = env.step(decision.action_id)
        if terminated or truncated:
            break


def test_load_and_deterministic_inference_preserve_global_torch_rng(
    tiny_checkpoint: Path,
) -> None:
    torch = pytest.importorskip("torch")
    before = torch.get_rng_state().clone()
    policy = MaskedPpoPolicy(tiny_checkpoint, device="cpu")
    env = Sts2Env(seed=33)
    observation, info = env.reset()
    first = policy.predict(observation, info)
    second = policy.choose_action(env.state)
    assert first.action_id == second.action_id
    assert first.policy_output == second.policy_output
    assert torch.equal(before, torch.get_rng_state())


def test_masks_restrict_scoring_and_never_create_a_fallback(tiny_checkpoint: Path) -> None:
    policy = MaskedPpoPolicy(tiny_checkpoint, device="cpu")
    observation, info = Sts2Env(seed=25).reset()
    legal_id = info["action_space"][-1]["id"]
    mask = [0] * len(observation["action_mask"])
    mask[legal_id] = 1
    observation = {**observation, "action_mask": mask}
    info = {**info, "action_mask": mask}
    assert policy.choose_action_id(observation, info) == legal_id
    with pytest.raises(PolicyInferenceError, match="No supported legal actions"):
        policy.predict(observation, info, allowed_action_ids=[])
    with pytest.raises(PolicyInferenceError, match="must be integers"):
        policy.predict(observation, info, allowed_action_ids=[True])
    with pytest.raises(PolicyInferenceError, match="disagree"):
        policy.predict(observation, {**info, "action_mask": [0] * len(mask)})


@pytest.mark.parametrize("invalid", ["missing-mask", "shape", "nan", "duplicate-id"])
def test_invalid_observations_fail_explicitly(tiny_checkpoint: Path, invalid: str) -> None:
    policy = MaskedPpoPolicy(tiny_checkpoint, device="cpu")
    observation, info = Sts2Env(seed=23).reset()
    if invalid == "missing-mask":
        observation.pop("action_mask")
        info.pop("action_mask")
    elif invalid == "shape":
        observation["vector"] = [0.0]
    elif invalid == "nan":
        observation["vector"] = [float("nan")] * len(observation["vector"])
    else:
        info["action_space"] = [info["action_space"][0]] * 2
    with pytest.raises(PolicyInferenceError):
        policy.predict(observation, info)


@pytest.mark.parametrize(
    "content", [b"not a checkpoint", b"", b"version https://git-lfs.github.com/spec/v1\n"]
)
def test_invalid_checkpoint_is_not_replaced_with_random_weights(
    tmp_path: Path,
    content: bytes,
) -> None:
    path = tmp_path / "invalid.pt"
    path.write_bytes(content)
    with pytest.raises(CheckpointPolicyError, match="Cannot use PPO checkpoint"):
        MaskedPpoPolicy(path, device="cpu")


@pytest.mark.parametrize("invalid", ["architecture", "missing-weights", "shape", "nan"])
def test_incompatible_checkpoint_is_rejected(
    tiny_checkpoint: Path,
    tmp_path: Path,
    invalid: str,
) -> None:
    torch = pytest.importorskip("torch")
    payload = torch.load(tiny_checkpoint, weights_only=True)
    if invalid == "architecture":
        payload["architecture"]["game_logic_checksum"] = "different"
    elif invalid == "missing-weights":
        payload.pop("model_state")
    else:
        key = next(iter(payload["model_state"]))
        if invalid == "shape":
            payload["model_state"][key] = torch.zeros(1)
        else:
            payload["model_state"][key].view(-1)[0] = float("nan")
    path = tmp_path / "changed.pt"
    torch.save(payload, path)
    with pytest.raises(CheckpointPolicyError):
        MaskedPpoPolicy(path, device="cpu")


def test_all_checkpoint_reads_use_weights_only(
    tiny_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    original = torch.load
    calls = []

    def record_load(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("weights_only"))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", record_load)
    MaskedPpoPolicy(tiny_checkpoint, device="cpu")
    assert calls and all(value is True for value in calls)


def test_evaluation_uses_fixed_seeds_and_preserves_checkpoint(tiny_checkpoint: Path) -> None:
    before = hashlib.sha256(tiny_checkpoint.read_bytes()).hexdigest()
    args = {"seed": "holdout", "eval_runs": 2, "eval_max_steps": 2, "device": "cpu"}
    first = evaluate_ppo(tiny_checkpoint, **args)
    second = evaluate_ppo(tiny_checkpoint, **args)
    assert first == second
    assert len(first["episode_seeds"]) == 2
    assert first["architecture_configuration"]["hidden_size"] == 8
    assert first["evaluation_errors"] == 0
    assert hashlib.sha256(tiny_checkpoint.read_bytes()).hexdigest() == before


def test_evaluate_cli_requires_explicit_seed_and_writes_report(
    tiny_checkpoint: Path,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    missing = runner.invoke(app, ["evaluate-ppo", "--checkpoint", str(tiny_checkpoint)])
    assert missing.exit_code == 2
    report = tmp_path / "evaluation.json"
    result = runner.invoke(
        app,
        [
            "evaluate-ppo",
            "--checkpoint",
            str(tiny_checkpoint),
            "--seed",
            "cli-holdout",
            "--runs",
            "1",
            "--max-steps",
            "2",
            "--device",
            "cpu",
            "--output",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(report.read_text())
    assert payload["seed"] == "cli-holdout"
    assert len(payload["episode_seeds"]) == 1
