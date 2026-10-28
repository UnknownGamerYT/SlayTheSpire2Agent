from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import pytest

import sts2sim.learning.masked_ppo as ppo


def _transition(*, reward: float, value: float, done: bool = False) -> ppo._Transition:
    return ppo._Transition(
        observation_vector=(0.0,),
        action_features=((0.0,),),
        action_identity_ids=((0,),),
        action_index=0,
        old_log_prob=0.0,
        value=value,
        planning_targets=(),
        planning_outputs=(),
        reward=reward,
        done=done,
    )


def test_time_limit_bootstraps_without_mixing_the_following_episode() -> None:
    limited_run = ppo._close_training_rollout(
        [_transition(reward=1.0, value=3.0)], bootstrap_value=7.0
    )
    next_run = _transition(reward=100.0, value=200.0, done=True)

    returns, _advantages = ppo._returns_and_advantages(
        [*limited_run, next_run], gamma=0.9, gae_lambda=0.95
    )

    assert limited_run[-1].done is True
    assert returns == pytest.approx((7.3, 100.0))


def test_gae_propagates_within_a_run_and_stops_at_an_interrupted_run() -> None:
    interrupted = ppo._close_training_rollout(
        [
            _transition(reward=1.0, value=2.0),
            _transition(reward=4.0, value=3.0),
        ],
        bootstrap_value=0.0,
    )
    following = _transition(reward=1000.0, value=500.0, done=True)

    returns, _advantages = ppo._returns_and_advantages(
        [*interrupted, following], gamma=0.9, gae_lambda=0.5
    )

    assert returns == pytest.approx((4.15, 4.0, 1000.0))
    assert ppo._close_training_rollout([], bootstrap_value=0.0) == ()


@pytest.mark.parametrize("policy_server", [False, True])
@pytest.mark.parametrize("ending", ["time_limit", "terminal", "target", "interrupted"])
def test_training_collectors_close_runs_and_bootstrap_only_time_limits(
    monkeypatch: pytest.MonkeyPatch,
    policy_server: bool,
    ending: str,
) -> None:
    decisions: list[bool] = []
    info = {
        "action_space": [{"id": 0, "type": "end_turn"}],
        "reward_breakdown": {"total": 1.0},
    }

    class FakeEnv:
        def __init__(self, **_kwargs: Any) -> None:
            self.state = object()

        def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
            return {"phase": "map", "act": 1, "floor": 1}, info

        def step(self, _action_id: int) -> tuple[dict[str, Any], float, bool, bool, dict]:
            observation = {
                "phase": "failed" if ending == "terminal" else "map",
                "act": 1,
                "floor": 1,
            }
            if ending == "target":
                observation.update(phase="ancient", act=2, floor=0)
            return observation, 1.0, ending == "terminal", ending != "interrupted", info

        def set_pending_policy_output(self, _payload: Mapping[str, Any]) -> None:
            pass

        def close(self) -> None:
            pass

    def choose(*_args: Any, **kwargs: Any) -> Any:
        deterministic = bool(kwargs["deterministic"])
        decisions.append(deterministic)
        if ending == "interrupted" and len(decisions) == 2:
            return None
        value = 7.0 if deterministic else 3.0
        return 0, 0, 0.0, value, ((0.0,),), ((0,),), {}

    monkeypatch.setattr(ppo, "Sts2Env", FakeEnv)
    monkeypatch.setattr(ppo, "_choose_action", choose)
    monkeypatch.setattr(ppo, "_request_policy_server_decision", choose)
    monkeypatch.setattr(ppo, "_observation_vector", lambda _observation: (0.0,))
    monkeypatch.setattr(ppo, "_planning_targets", lambda _observation: ())
    monkeypatch.setattr(ppo, "_accumulate_run_diagnostics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ppo, "_final_run_diagnostics", lambda *_args: {})
    arguments = {
        "target": ppo.TrainingTarget("test", 2, 0, "ancient"),
        "run_index": 0,
        "seed": 1,
        "max_steps": 2 if ending == "interrupted" else 1,
        "character_id": "TEST",
        "ascension": 0,
        "gamma": 0.9,
        "target_reward": 0.0,
    }
    if policy_server:
        run, transitions = ppo._collect_training_run_with_policy_server(
            worker_id=0, request_queue=None, response_queue=None, **arguments
        )
    else:
        run, transitions = ppo._collect_training_run(
            torch=None, model=None, device=None, **arguments
        )

    assert len(transitions) == 1
    assert transitions[-1].done is True
    assert run.failed_to_continue is (ending == "interrupted")
    assert transitions[-1].bootstrap_value == (7.0 if ending == "time_limit" else 0.0)
    if ending == "time_limit":
        assert decisions == [False, True]
    elif ending == "interrupted":
        assert decisions == [False, False]
    else:
        assert decisions == [False]


def test_training_seed_repeats_torch_samples_and_minibatch_order() -> None:
    torch = pytest.importorskip("torch")
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    try:
        ppo._seed_training_rngs(torch, "repeatable-training")
        first_weights = torch.nn.Linear(3, 2).weight.detach().clone()
        first_sample = torch.distributions.Categorical(logits=torch.zeros(3)).sample((8,))
        first_order = list(range(12))
        random.shuffle(first_order)

        ppo._seed_training_rngs(torch, "repeatable-training")
        second_weights = torch.nn.Linear(3, 2).weight.detach().clone()
        second_sample = torch.distributions.Categorical(logits=torch.zeros(3)).sample((8,))
        second_order = list(range(12))
        random.shuffle(second_order)

        assert torch.equal(first_weights, second_weights)
        assert torch.equal(first_sample, second_sample)
        assert first_order == second_order
    finally:
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
