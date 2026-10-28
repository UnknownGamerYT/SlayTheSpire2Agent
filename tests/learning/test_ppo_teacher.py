from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

import sts2sim.learning.masked_ppo as ppo


def _transition(index: int, *, forced: bool = False) -> ppo._Transition:
    return ppo._Transition(
        observation_vector=(float(index),),
        action_features=((0.0,), (1.0,)),
        action_identity_ids=((0,), (0,)),
        action_index=0,
        old_log_prob=-0.6931471805599453,
        value=0.0,
        planning_targets=(1.0,) * ppo.PLANNING_HEAD_DIM,
        planning_outputs=(0.5,) * ppo.PLANNING_HEAD_DIM,
        reward=float(index + 1),
        done=True,
        teacher_action_index=1,
        teacher_forced=forced,
    )


def _model(torch: Any) -> Any:
    class SeparateHeads(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(3, 2))
            self.values = torch.nn.Parameter(torch.zeros(3))
            self.plans = torch.nn.Parameter(torch.zeros(3, ppo.PLANNING_HEAD_DIM))

        def forward(self, observations: Any, *_args: Any) -> tuple[Any, Any, Any]:
            rows = observations[:, 0].long()
            return self.logits[rows], self.values[rows], self.plans[rows].sigmoid()

    return SeparateHeads()


def _update(torch: Any, model: Any, transitions: list[ppo._Transition], **kwargs: Any) -> None:
    ppo._ppo_update(
        torch=torch,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        transitions=transitions,
        gamma=0.9,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=1.0,
        entropy_coef=1.0,
        planning_coef=1.0,
        imitation_coef=kwargs.get("imitation_coef", 0.0),
        ppo_epochs=1,
        minibatch_size=10,
        device=torch.device("cpu"),
    )


def test_forced_samples_train_critic_and_planning_without_actor_gradients() -> None:
    torch = pytest.importorskip("torch")
    model = _model(torch)
    with torch.no_grad():
        model.logits[0, 0] = 1.0
    previous_logits = model.logits.detach().clone()
    _update(torch, model, [_transition(0, forced=True)])

    assert torch.equal(model.logits, previous_logits)
    assert model.values[0] > 0.0
    assert model.plans[0].sum() > 0.0


def test_forced_samples_keep_the_explicit_imitation_objective() -> None:
    torch = pytest.importorskip("torch")
    model = _model(torch)
    _update(torch, model, [_transition(0, forced=True)], imitation_coef=1.0)

    assert model.logits[0, 1] > model.logits[0, 0]
    assert model.values[0] > 0.0


def test_forced_samples_do_not_change_learner_advantage_normalization_or_actor_scale() -> None:
    torch = pytest.importorskip("torch")
    baseline = _model(torch)
    mixed = _model(torch)
    learner_samples = [_transition(0), _transition(1)]
    forced = replace(_transition(2, forced=True), reward=1000.0, old_log_prob=-1.0e9)

    _update(torch, baseline, learner_samples)
    _update(torch, mixed, [*learner_samples, forced])

    assert not torch.equal(baseline.logits[:2], torch.zeros_like(baseline.logits[:2]))
    assert torch.equal(baseline.logits[:2], mixed.logits[:2])
    assert torch.equal(mixed.logits[2], torch.zeros_like(mixed.logits[2]))


@pytest.mark.parametrize("policy_server", [False, True])
@pytest.mark.parametrize("teacher_mix", [0.0, 1.0])
@pytest.mark.parametrize("imitation_enabled", [False, True])
def test_collectors_mark_teacher_interventions_independently_of_imitation(
    monkeypatch: pytest.MonkeyPatch,
    policy_server: bool,
    teacher_mix: float,
    imitation_enabled: bool,
) -> None:
    info = {
        "action_space": [{"id": 0, "type": "end_turn"}, {"id": 1, "type": "end_turn"}],
        "reward_breakdown": {"total": 1.0},
    }

    class FakeEnv:
        def __init__(self, **_kwargs: Any) -> None:
            self.state = object()

        def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
            return {"phase": "map", "act": 1, "floor": 1}, info

        def step(self, action_id: int) -> tuple[dict[str, Any], float, bool, bool, dict]:
            assert action_id == (1 if teacher_mix else 0)
            return {"phase": "failed", "act": 1, "floor": 1}, 1.0, True, False, info

        def set_pending_policy_output(self, payload: Mapping[str, Any]) -> None:
            assert payload.get("teacher_forced", 0.0) == teacher_mix

        def close(self) -> None:
            pass

    def choose(*_args: Any, **kwargs: Any) -> Any:
        forced = kwargs.get("forced_action_index")
        assert forced == (1 if teacher_mix else None)
        index = 0 if forced is None else forced
        return index, index, -0.69, 0.0, ((0.0,), (1.0,)), ((0,), (0,)), {}

    monkeypatch.setattr(ppo, "Sts2Env", FakeEnv)
    monkeypatch.setattr(ppo, "_teacher_action_index", lambda *_args: 1)
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
        "max_steps": 1,
        "character_id": "TEST",
        "ascension": 0,
        "gamma": 0.9,
        "target_reward": 0.0,
        "teacher_agent": object(),
        "teacher_mix": teacher_mix,
        "imitation_enabled": imitation_enabled,
    }
    if policy_server:
        run, transitions = ppo._collect_training_run_with_policy_server(
            worker_id=0, request_queue=None, response_queue=None, **arguments
        )
    else:
        run, transitions = ppo._collect_training_run(
            torch=None, model=None, device=None, **arguments
        )

    assert run.error is None
    assert len(transitions) == 1
    assert transitions[0].teacher_forced is bool(teacher_mix)
    assert transitions[0].teacher_action_index == (1 if imitation_enabled else None)
