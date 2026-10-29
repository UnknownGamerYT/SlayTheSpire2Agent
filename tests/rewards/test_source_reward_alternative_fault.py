from __future__ import annotations

import json

import pytest

from sts2sim import (
    inspect_reward_alternative_fault,
    legal_actions,
    load_state,
    new_run,
    serialize,
    step,
)
from sts2sim.agent_api import action_space
from sts2sim.engine import Action, ActionType, IllegalActionError, RewardState, RunPhase
from sts2sim.engine.source_reward_reroll import card_reroll_context


def _state(*, relics=("driftwood", "paels_wing")):
    state = new_run("ALTERNATIVE-FAULT", "IRONCLAD", 0)
    return state.model_copy(update={
        "phase": RunPhase.REWARD, "relics": relics,
        "reward": RewardState(reward_id="alternatives", source="other", gold=12,
            card_options=("anger", "shrug_it_off", "bash"),
            card_option_groups=(("anger", "shrug_it_off", "bash"),),
            card_ids=("lantern_key",),
            metadata={"card_reroll_contexts": {
                "primary": card_reroll_context(count=3),
                "group:0": card_reroll_context(count=3)}}),
    })


@pytest.mark.parametrize("group", ["primary", "group:0"])
def test_source_alternative_limit_fault_is_before_screen_choice_rng_and_card_history(group):
    state = _state()
    before = serialize(state)
    trace = inspect_reward_alternative_fault(state, group)
    assert serialize(state) == before
    assert inspect_reward_alternative_fault(load_state(before), group) == trace
    assert json.loads(json.dumps(trace))["source_exception"]["reason"] == (
        "more_than_two_card_reward_alternatives")
    assert trace["alternatives"] == ("Skip", "REROLL", "SACRIFICE")
    assert trace["history_append"] == ()
    assert trace["rng_before"] == trace["rng_after"] == before["rng"]
    assert trace["partial_reward_model"] == before["reward"]
    assert trace["card_selection_screen_opened"] is False
    assert trace["player_choice_id_reserved"] is False
    assert trace["normal_transition"] is False
    assert trace["continuation_requires_live_ui_observation"] is True


def test_training_mask_keeps_set_proceed_and_other_rewards_around_faulting_card_screens():
    state = _state()
    actions = legal_actions(state)
    assert not any(a.type in {ActionType.SACRIFICE_REWARD, ActionType.REROLL_REWARD}
                   for a in actions)
    assert [a.target_id for a in actions if a.type == ActionType.TAKE_REWARD_CARD] == [
        "reward:fixed_card:0"]
    assert not any(a.target_id in {"reward:card_options", "reward:card_group:0"}
                   for a in actions if a.type == ActionType.SKIP_REWARD)
    assert any(a.type == ActionType.TAKE_REWARD_GOLD for a in actions)
    assert any(a.type == ActionType.PROCEED for a in actions)
    assert all(entry["target_id"] != "reward:card:0"
               for entry in action_space(state) if "target_id" in entry)
    with pytest.raises(IllegalActionError):
        step(state, Action(type=ActionType.TAKE_REWARD_CARD, target_id="reward:card:0"))
    restored = load_state(serialize(state))
    proceed = next(a for a in actions if a.type == ActionType.PROCEED)
    assert serialize(step(restored, proceed)) == serialize(step(state, proceed))


@pytest.mark.parametrize("relics", [("driftwood",), ("paels_wing",), ()])
def test_two_or_fewer_alternatives_remain_ordinary_selectable_rewards(relics):
    state = _state(relics=relics)
    assert any(a.target_id == "reward:card:0" for a in legal_actions(state))
    with pytest.raises(ValueError, match="no source alternative-count fault"):
        inspect_reward_alternative_fault(state)


def test_source_can_reroll_is_per_group_and_a_spent_group_no_longer_overflows():
    state = _state()
    state = state.model_copy(update={"reward": state.reward.model_copy(update={
        "metadata": {**state.reward.metadata, "card_reward_rerolled_groups": ("primary",)},
    })})
    actions = legal_actions(state)
    assert any(a.target_id == "reward:card:0" for a in actions)
    assert any(a.target_id == "reward:sacrifice:primary" for a in actions)
    assert not any((a.target_id or "").startswith("reward:card_group:0:") for a in actions)
    with pytest.raises(ValueError, match="no source alternative-count fault"):
        inspect_reward_alternative_fault(state, "primary")
    assert inspect_reward_alternative_fault(state, "group:0")["normal_transition"] is False
