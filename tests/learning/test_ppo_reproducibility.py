from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

import sts2sim.learning.masked_ppo as ppo


@pytest.fixture
def torch_rng() -> Any:
    torch = pytest.importorskip("torch")
    python_state = random.getstate()
    torch_state = torch.get_rng_state()
    try:
        yield torch
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)


def _assert_state_equal(torch: Any, left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert left.keys() == right.keys()
        for key, value in left.items():
            _assert_state_equal(torch, value, right[key])
    elif isinstance(left, Sequence) and not isinstance(left, str | bytes):
        assert isinstance(right, Sequence)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_state_equal(torch, left_item, right_item)
    else:
        assert left == right


def test_checkpoint_resume_matches_uninterrupted_cpu_training(
    tmp_path: Path, torch_rng: Any
) -> None:
    torch = torch_rng
    options = {
        "character_id": "TEST",
        "seed": "resume-reproducibility",
        "device": "cpu",
        "hidden_size": 8,
        "hidden_layers": 1,
        "head_hidden_layers": 1,
        "train_runs_per_batch": 2,
        "train_max_steps": 1,
        "eval_runs": 1,
        "eval_max_steps": 1,
        "champion_eval_runs": 1,
        "champion_patience": 99,
        "target_eval_successes": 99,
        "ppo_epochs": 2,
        "minibatch_size": 1,
        "history_mode": "off",
        "output_path": None,
        "progress_output_path": None,
        "report_output_path": None,
    }
    uninterrupted_path = tmp_path / "uninterrupted.pt"
    uninterrupted = ppo.train_masked_ppo(
        max_batches=2, resume=False, model_output_path=uninterrupted_path, **options
    )
    split_path = tmp_path / "split.pt"
    first = ppo.train_masked_ppo(
        max_batches=1, resume=False, model_output_path=split_path, **options
    )
    resumed = ppo.train_masked_ppo(
        max_batches=1, resume=True, model_output_path=split_path, **options
    )

    assert first["batches_completed"] == 1
    assert resumed["batches_completed"] == uninterrupted["batches_completed"] == 2
    assert resumed["metadata"]["checkpoint_compatibility_checks"][-1]["rng_restored"] is True
    for key, count in (("progress", 4), ("evaluation_progress", 2)):
        uninterrupted_seeds = [point["seed"] for point in uninterrupted[key]]
        resumed_seeds = [point["seed"] for point in resumed[key]]
        assert uninterrupted_seeds == resumed_seeds
        assert len(set(resumed_seeds)) == count
    assert resumed["total_steps"] == uninterrupted["total_steps"]
    assert resumed["average_training_reward"] == uninterrupted["average_training_reward"]

    uninterrupted_payload = torch.load(uninterrupted_path, map_location="cpu", weights_only=True)
    resumed_payload = torch.load(split_path, map_location="cpu", weights_only=True)
    for key in ("model_state", "optimizer_state", "rng_state"):
        _assert_state_equal(torch, uninterrupted_payload[key], resumed_payload[key])


@pytest.mark.parametrize("request_kind", ["deterministic", "forced", "mixed"])
def test_batched_nonsampling_decisions_preserve_torch_rng(
    torch_rng: Any, request_kind: str
) -> None:
    torch = torch_rng

    class FixedPolicy:
        def __call__(
            self, observations: Any, _features: Any, _identity_ids: Any
        ) -> tuple[Any, Any, Any]:
            batch_size = observations.shape[0]
            return (
                torch.tensor([[0.1, 0.2, 0.8]]).expand(batch_size, -1),
                torch.zeros(batch_size),
                torch.zeros(batch_size, ppo.PLANNING_HEAD_DIM),
            )

    def request(*, deterministic: bool, forced_index: int | None) -> dict[str, Any]:
        return {
            "worker_id": 0,
            "request_id": f"{deterministic}:{forced_index}",
            "observation_vector": (0.0,),
            "action_features": ((0.0,), (1.0,), (2.0,)),
            "action_identity_ids": ((0,), (0,), (0,)),
            "action_ids": (7, 8, 9),
            "deterministic": deterministic,
            "forced_action_index": forced_index,
        }

    requests = []
    expected_actions = []
    if request_kind in {"deterministic", "mixed"}:
        requests.append(request(deterministic=True, forced_index=None))
        expected_actions.append(9)
    if request_kind in {"forced", "mixed"}:
        requests.append(request(deterministic=False, forced_index=1))
        expected_actions.append(8)
    before = torch.get_rng_state().clone()

    responses = ppo._batched_policy_decisions(
        torch=torch, model=FixedPolicy(), requests=requests, device=torch.device("cpu")
    )

    assert torch.equal(before, torch.get_rng_state())
    assert [response["decision"][0] for response in responses] == expected_actions
