import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import Action, ActionType, CardInstance, RewardState, RunPhase


def _special_reward(ordinary, alternatives=("paels_wing", "driftwood")):
    state = new_run("special-card-reward", "IRONCLAD", 0)
    returned = CardInstance(instance_id="stolen:bash", card_id="bash", type="attack",
                            effects={"damage": 8}, cost=2)
    reward = RewardState(
        reward_id="source-special-card", source="combat", gold=1,
        card_options=("setup_strike", "tremble", "blood_wall") if ordinary else (),
        card_option_groups=(("bash",),),
        metadata={"returned_card_option_specs": {"0": returned.model_dump(mode="json")},
                  "upgrade_card_rewards": True},
    )
    return state.model_copy(update={"phase": RunPhase.REWARD, "reward": reward,
        "relics": ("silver_crucible", "silken_tress", "wing_charm", *alternatives)})


@pytest.mark.parametrize("ordinary", [False, True])
@pytest.mark.parametrize("alternative_relic", ["paels_wing", "driftwood"])
def test_returned_special_card_never_receives_card_reward_modifiers_or_alternatives(
    ordinary, alternative_relic,
):
    state = _special_reward(ordinary, (alternative_relic,))
    result = step(load_state(serialize(state)), Action(
        type=ActionType.TAKE_REWARD_GOLD, target_id="reward:gold",
    ))
    metadata = result.reward.metadata
    assert "group:0" not in metadata["source_modified_card_reward_groups"]
    assert "group:0" not in metadata["card_reward_option_modifiers"]
    assert result.flags.get("silver_crucible_times_used", 0) == int(ordinary)
    assert bool(result.flags.get("silken_tress_reward_used")) is ordinary
    alternatives = [a for a in legal_actions(result) if a.type in {
        ActionType.SACRIFICE_REWARD, ActionType.REROLL_REWARD,
    }]
    assert not any(a.target_id.endswith("group:0") for a in alternatives)
    assert bool(alternatives) is ordinary
    if not ordinary:
        assert result.rng == state.rng
    returned = step(result, Action(type=ActionType.TAKE_REWARD_CARD,
                                   target_id="reward:card_group:0:0"))
    card = next(card for card in returned.master_deck if card.instance_id == "stolen:bash")
    assert card.card_id == "bash" and not card.upgraded and not card.enchantments
    assert serialize(returned) == serialize(step(load_state(serialize(result)), Action(
        type=ActionType.TAKE_REWARD_CARD, target_id="reward:card_group:0:0",
    )))


def test_returned_special_card_bypasses_the_normal_card_alternative_overflow():
    from sts2sim import inspect_reward_alternative_fault

    state = _special_reward(True)
    with pytest.raises(ValueError, match="no source alternative-count fault"):
        inspect_reward_alternative_fault(state, "group:0")
    assert inspect_reward_alternative_fault(state, "primary")["normal_transition"] is False
    assert any(action.target_id == "reward:card_group:0:0" for action in legal_actions(state))
    assert not any(action.target_id == "reward:card:0" for action in legal_actions(state))
