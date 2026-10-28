from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import sts2sim.learning.masked_ppo as ppo


@pytest.fixture
def network() -> tuple[Any, Any, Any]:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    return torch, model, torch.optim.Adam(model.parameters())


def _resume(network: tuple[Any, Any, Any], path: Path, fallback: Path | None = None) -> Any:
    torch, model, optimizer = network
    return ppo._load_checkpoint_if_available(
        torch=torch,
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        expected_architecture={},
        resume=True,
        resume_from_path=path,
        model_output_path=fallback,
    )


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (b"", "could not be loaded"),
        (b"not a checkpoint", "could not be loaded"),
        (b"version https://git-lfs.github.com/spec/v1\noid sha256:123\n", "Git LFS pointer"),
    ],
)
def test_corrupt_or_lfs_checkpoint_falls_back_without_mutating_fresh_weights(
    tmp_path: Path, network: tuple[Any, Any, Any], content: bytes, reason: str
) -> None:
    torch, model, _optimizer = network
    before = deepcopy(model.state_dict())
    path = tmp_path / "broken.pt"
    path.write_bytes(content)

    result = _resume(network, path)

    assert result.path is None
    assert reason in result.checkpoint_checks[-1]["reason"]
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


@pytest.mark.parametrize("invalid", ["missing", "shape", "non_tensor", "nan"])
def test_invalid_model_state_cannot_partially_overwrite_weights(
    tmp_path: Path, network: tuple[Any, Any, Any], invalid: str
) -> None:
    torch, model, _optimizer = network
    before = deepcopy(model.state_dict())
    state: Any = {"weight": torch.full((2, 2), 99.0), "bias": torch.zeros(2)}
    if invalid == "missing":
        state = None
    elif invalid == "shape":
        state["bias"] = torch.zeros(3)
    elif invalid == "non_tensor":
        state["bias"] = [0.0, 0.0]
    else:
        state["bias"][0] = float("nan")
    path = tmp_path / "invalid.pt"
    torch.save({"architecture": {}, "model_state": state}, path)

    result = _resume(network, path)

    assert result.path is None
    assert "model state" in result.checkpoint_checks[-1]["reason"]
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_bad_first_candidate_can_resume_from_valid_fallback(
    tmp_path: Path, network: tuple[Any, Any, Any]
) -> None:
    torch, model, optimizer = network
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"not a checkpoint")
    fallback = tmp_path / "valid.pt"
    expected = {"weight": torch.full((2, 2), 7.0), "bias": torch.full((2,), 8.0)}
    torch.save(
        {"architecture": {}, "model_state": expected, "optimizer_state": optimizer.state_dict()},
        fallback,
    )

    result = _resume(network, broken, fallback)

    assert result.path == str(fallback)
    assert [check["decision"] for check in result.checkpoint_checks] == ["fresh", "resume"]
    assert result.checkpoint_checks[-1]["optimizer_restored"] is True
    assert result.checkpoint_checks[-1]["rng_restored"] is False
    assert "no RNG state" in result.checkpoint_checks[-1]["warnings"][0]
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in expected.items())


def test_bad_optimizer_is_reset_without_discarding_valid_model_weights(
    tmp_path: Path, network: tuple[Any, Any, Any]
) -> None:
    torch, model, optimizer = network
    # Initialize valid Adam moment shapes, then corrupt one in the checkpoint.
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    bad_optimizer = deepcopy(optimizer.state_dict())
    first_parameter_id = bad_optimizer["param_groups"][0]["params"][0]
    bad_optimizer["state"][first_parameter_id]["exp_avg"] = torch.zeros(9)
    path = tmp_path / "bad-optimizer.pt"
    expected = deepcopy(model.state_dict())
    torch.save(
        {"architecture": {}, "model_state": expected, "optimizer_state": bad_optimizer}, path
    )
    fresh_optimizer = torch.optim.Adam(model.parameters())

    result = _resume((torch, model, fresh_optimizer), path)

    assert result.path == str(path)
    assert result.checkpoint_checks[-1]["optimizer_restored"] is False
    assert "incompatible shape" in result.checkpoint_checks[-1]["warnings"][0]
    assert fresh_optimizer.state_dict()["state"] == {}
    # It remains usable for the next training update.
    fresh_optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    fresh_optimizer.step()


def test_validation_rejects_matching_metadata_without_model_state(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    architecture = ppo._expected_ppo_architecture(
        hidden_size=8, hidden_layers=1, head_hidden_layers=1, activation="silu"
    )
    path = tmp_path / "metadata-only.pt"
    torch.save({"architecture": architecture}, path)

    result = ppo.validate_masked_ppo_checkpoint(
        path, hidden_size=8, hidden_layers=1, head_hidden_layers=1
    )

    assert result["compatible"] is False
    assert "no model state" in result["reason"]


def test_failed_champion_restore_preserves_current_weights_and_optimizer(
    tmp_path: Path, network: tuple[Any, Any, Any]
) -> None:
    torch, model, optimizer = network
    previous = deepcopy(model.state_dict())
    path = tmp_path / "broken-champion.pt"
    torch.save(
        {"model_state": {name: value + 10 for name, value in previous.items()}}, path
    )

    restored = ppo._restore_ppo_checkpoint(
        torch=torch,
        model=model,
        optimizer=optimizer,
        checkpoint_path=path,
        device=torch.device("cpu"),
    )

    assert restored is False
    assert optimizer.state_dict()["state"] == {}
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in previous.items())


def test_evaluation_uses_the_same_seed_suite_after_a_checkpoint_is_renamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    architecture = ppo._expected_ppo_architecture(
        hidden_size=8, hidden_layers=1, head_hidden_layers=1, activation="silu"
    )
    model = ppo._masked_actor_critic_class(torch.nn)(
        **ppo._model_kwargs_from_architecture(architecture)
    )
    payload = {"architecture": architecture, "model_state": model.state_dict()}
    first_path = tmp_path / "latest.pt"
    renamed_path = tmp_path / "act1_champion.pt"
    torch.save(payload, first_path)
    torch.save(payload, renamed_path)
    jobs = []

    def capture_jobs(**kwargs: Any) -> tuple:
        jobs.append(kwargs["jobs"])
        return ()

    monkeypatch.setattr(ppo, "_collect_evaluation_rollouts", capture_jobs)
    for path in (first_path, renamed_path):
        ppo.evaluate_masked_ppo_checkpoint(
            path,
            target="act1-boss",
            eval_runs=3,
            eval_max_steps=1,
            device="cpu",
            hidden_size=8,
            hidden_layers=1,
            head_hidden_layers=1,
        )

    assert len(jobs[0]) == 3
    assert jobs[0] == jobs[1]
