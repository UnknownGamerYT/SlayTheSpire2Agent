from __future__ import annotations

import pytest

gym = pytest.importorskip("gymnasium")
np = pytest.importorskip("numpy")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from sts2sim.api import serialize  # noqa: E402
from sts2sim.gymnasium_adapter import GymnasiumSts2Env  # noqa: E402


def test_gymnasium_adapter_numeric_contract_and_rich_context() -> None:
    env = GymnasiumSts2Env(character_id="TEST", max_actions=16)
    observation, info = env.reset(seed=10)

    assert isinstance(env, gym.Env)
    assert set(observation) == {"vector", "action_mask"}
    assert observation["vector"].dtype == np.float32
    assert observation["action_mask"].dtype == np.int8
    assert env.observation_space.contains(observation)
    assert info["rich_observation"]["phase"] == "ancient"
    assert len(info["action_space"]) == np.count_nonzero(env.action_masks())

    masks = env.action_masks()
    masks[:] = False
    assert env.action_masks().any()
    action = int(np.flatnonzero(env.action_masks())[0])
    next_observation, reward, terminated, truncated, next_info = env.step(action)

    assert env.observation_space.contains(next_observation)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert next_info["action"]["type"] == "choose_ancient"
    assert next_info["rich_observation"]["agent_memory"]["entries"]


def test_gymnasium_checker_with_mask_aware_sampling(monkeypatch) -> None:
    env = GymnasiumSts2Env(character_id="TEST", max_actions=16)

    # The generic checker samples the padded Discrete space without a mask.
    # Supply the same mask-aware sampling required by real training clients.
    original_sample = env.action_space.sample
    monkeypatch.setattr(
        env.action_space,
        "sample",
        lambda: original_sample(mask=env.action_masks().astype(np.int8)),
    )
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize("action", [True, np.bool_(True), 0.0, "0", -1, 16, 15])
def test_gymnasium_adapter_rejects_invalid_actions_without_advancing(action) -> None:
    env = GymnasiumSts2Env(character_id="TEST", max_actions=16)
    env.reset(seed=10)
    before = serialize(env.simulator.state)

    with pytest.raises(ValueError, match="not legal"):
        env.step(action)

    assert serialize(env.simulator.state) == before
    assert env.simulator.steps == 0


def test_gymnasium_adapter_requires_reset_and_preserves_truncation_state() -> None:
    env = GymnasiumSts2Env(character_id="TEST", max_actions=16, max_episode_steps=1)
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)
    with pytest.raises(gym.error.ResetNeeded):
        env.action_masks()

    env.reset(seed=12)
    observation, _reward, terminated, truncated, _info = env.step(np.int64(0))

    assert terminated is False
    assert truncated is True
    assert env.observation_space.contains(observation)
    assert observation["action_mask"].any()  # Available for value bootstrapping.
    with pytest.raises(gym.error.ResetNeeded):
        env.step(0)

    env.reset()
    assert env.step(0)[3] is True


def test_gymnasium_adapter_seeded_reset_sequences_are_reproducible() -> None:
    first = GymnasiumSts2Env(character_id="TEST", seed="experiment")
    second = GymnasiumSts2Env(character_id="TEST", seed="experiment")
    seeds = []

    for seed in [None, None, 45, None, 45]:
        first.reset(seed=seed)
        second.reset(seed=seed)
        first_state = serialize(first.simulator.state)
        assert first_state == serialize(second.simulator.state)
        seeds.append(first_state["seed"])

    assert seeds[0] == "experiment"
    assert seeds[0] != seeds[1]
    assert seeds[2] == seeds[4] == 45
    assert seeds[3] != seeds[2]


def test_gymnasium_adapter_vectorizes_with_masked_actions() -> None:
    envs = gym.vector.SyncVectorEnv(
        [lambda: GymnasiumSts2Env(character_id="TEST", max_episode_steps=1)] * 2
    )
    try:
        observation, _info = envs.reset(seed=20)
        actions = np.asarray([np.flatnonzero(mask)[0] for mask in observation["action_mask"]])
        next_observation, rewards, terminated, truncated, _info = envs.step(actions)

        assert envs.observation_space.contains(next_observation)
        assert rewards.shape == (2,)
        assert not terminated.any()
        assert truncated.all()
    finally:
        envs.close()


def test_gymnasium_adapter_validates_time_limit() -> None:
    with pytest.raises(ValueError, match="max_episode_steps must be positive"):
        GymnasiumSts2Env(max_episode_steps=0)
