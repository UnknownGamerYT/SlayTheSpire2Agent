from __future__ import annotations

import importlib.util

from sts2sim import Sts2Env, action_space
from sts2sim.gymnasium_env import SimpleDiscrete, gymnasium_available


def test_normal_import_and_env_fallback_do_not_require_gymnasium(monkeypatch) -> None:
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name in {"gymnasium", "numpy"}:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    env = Sts2Env(seed=10, character_id="TEST", ascension=0, max_actions=16)

    assert gymnasium_available() is False
    assert env.using_gymnasium is False
    assert isinstance(env.action_space, SimpleDiscrete)

    observation, info = env.reset()

    assert len(observation["action_mask"]) == 16
    assert info["legal_action_count"] == len(action_space(env.state))
    assert info["action_mask"][: info["legal_action_count"]] == [1] * info[
        "legal_action_count"
    ]


def test_env_step_decodes_discrete_action_id(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs: None)
    env = Sts2Env(seed=11, character_id="TEST", ascension=0, max_actions=16)
    _observation, info = env.reset()

    first_action_id = info["action_space"][0]["id"]
    next_observation, reward, terminated, truncated, next_info = env.step(first_action_id)

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert next_observation["phase"] == "map"
    assert next_info["action"]["type"] == "choose_ancient"


def test_env_records_policy_output_and_realized_preview_memory(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args, **kwargs: None)
    env = Sts2Env(seed=12, character_id="TEST", ascension=0, max_actions=16)
    observation, info = env.reset()

    assert observation["agent_memory"]["entries"] == []

    first_action_id = info["action_space"][0]["id"]
    env.set_pending_policy_output(
        {
            "action_index": 0,
            "confidence": 0.5,
            "log_prob": -0.7,
            "value": 1.25,
            "aggression_target": 0.35,
            "route_preference": 0.8,
            "boss_readiness": 0.6,
        }
    )
    next_observation, _reward, _terminated, _truncated, next_info = env.step(first_action_id)
    memory_entry = next_observation["agent_memory"]["entries"][0]

    assert memory_entry["action_type"] == "choose_ancient"
    assert memory_entry["confidence"] == 0.5
    assert memory_entry["log_prob"] == -0.7
    assert memory_entry["value"] == 1.25
    assert memory_entry["plan_aggression_target"] == 0.35
    assert memory_entry["plan_route_preference"] == 0.8
    assert memory_entry["plan_boss_readiness"] == 0.6
    assert "preview_error" in memory_entry
    assert next_info["agent_memory"]["entries"][0]["action_type"] == "choose_ancient"
