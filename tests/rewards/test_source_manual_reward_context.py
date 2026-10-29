import pytest

from sts2sim import load_state, new_run, serialize, step
from sts2sim.engine import Action, ActionType, RewardState, RunPhase
from sts2sim.engine.source_card_reward_modifiers import apply_card_reward_modifiers


def _manual_reward(*, relics=(), fixed_relic=None, subscribed=False):
    state = new_run("manual-reward-context", "IRONCLAD", 0)
    reward = RewardState(
        reward_id="manual:source-contract", source="combat",
        card_options=("setup_strike", "tremble", "blood_wall"),
        relic_ids=(fixed_relic,) if fixed_relic else (),
        metadata={"encounter": "normal", "card_reroll_contexts": {"primary": {
            "mode": "combat", "encounter": "normal", "pools": ("ironclad",), "count": 3,
            "initial_is_card_reward": True, "is_card_reward": False,
            "updates_on_relic_obtain": subscribed,
        }}},
    )
    state = state.model_copy(update={"phase": RunPhase.REWARD, "reward": reward,
                                      "relics": relics})
    return apply_card_reward_modifiers(state)[0]


def test_manual_initial_is_reward_hook_differs_from_for_room_reroll_options():
    state = _manual_reward(relics=("silver_crucible", "driftwood"))
    assert state.flags["silver_crucible_times_used"] == 1
    assert state.reward.metadata["card_option_upgraded"] == (True, True, True)
    action = Action(type=ActionType.REROLL_REWARD, target_id="reward:reroll:primary")
    result = step(load_state(serialize(state)), action)
    assert result.flags["silver_crucible_times_used"] == 1
    assert result.reward.metadata["card_option_upgraded"] == (False, False, False)
    assert len(result.reward.card_options) == 3
    assert serialize(result) == serialize(step(state, action))


@pytest.mark.parametrize("subscribed", [False, True])
def test_only_regular_reward_constructor_observes_later_relic_obtains(subscribed):
    state = _manual_reward(fixed_relic="molten_egg", subscribed=subscribed)
    result = step(load_state(serialize(state)), Action(
        type=ActionType.TAKE_REWARD_RELIC, target_id="reward:relic:0",
    ))
    assert "molten_egg" in result.relics
    assert result.reward.metadata["card_option_upgraded"][0] is subscribed
    assert result.reward.metadata["card_option_upgraded"][1:] == (False, False)
