"""Each revealed Sphere gold tile retains its own deferred GoldReward claim."""
from __future__ import annotations

from events.test_source_crystal_sphere import _event

from sts2sim import action_space, legal_actions, load_state, serialize, step
from sts2sim.engine import Action, ActionType
from sts2sim.engine.source_crystal_sphere import _sphere_event
from sts2sim.mechanics.source_crystal_sphere import SphereBoard, SphereItem


def test_two_gold_tiles_defer_gains_then_round_and_trigger_dragon_per_claim():
    state = _event().model_copy(update={'relics': ('bowler_hat', 'dragon_fruit')})
    state = step(state, Action(type=ActionType.CHOOSE_EVENT, target_id='payment_plan'))
    board = SphereBoard((
        SphereItem('gold', x=4, y=4, placed=True, subscriptions=1),
        SphereItem('gold', x=5, y=4, placed=True, subscriptions=1),
    ), (), 2)
    state = state.model_copy(update={'event': _sphere_event(state.event, board)})
    before = state.player
    before_draws = state.rng.streams['CRYSTAL_SPHERE'].draws
    state = step(load_state(serialize(state)), Action(type=ActionType.CHOOSE_EVENT,
                                                     target_id='small_4_4'))
    assert state.reward is None and state.player == before
    assert state.rng.streams['CRYSTAL_SPHERE'].draws == before_draws
    state = step(load_state(serialize(state)), Action(type=ActionType.CHOOSE_EVENT,
                                                     target_id='small_5_4'))
    assert state.reward.gold == 20 and state.player == before
    assert state.reward.metadata['gold_reward_components'] == (
        {'amount': 10, 'source_id': 'crystal_sphere'},
        {'amount': 10, 'source_id': 'crystal_sphere'},
    )
    assert state.rng.streams['CRYSTAL_SPHERE'].draws == before_draws + 2
    descriptors = [action for action in action_space(state) if action['type'] == 'take_reward_gold']
    assert [(action['target_id'], action['reward_choice']['gold']) for action in descriptors] == [
        ('reward:gold:0', 10), ('reward:gold:1', 10),
    ]
    generated_rng = state.rng
    for index in (1, 0):
        action = next(action for action in legal_actions(state)
                      if action.type == ActionType.TAKE_REWARD_GOLD
                      and action.target_id == f'reward:gold:{index}')
        state = step(load_state(serialize(state)), action)
        gained = 1 if index == 1 else 2
        assert state.player.gold == before.gold + 12 * gained
        assert state.player.max_hp == before.max_hp + gained
        assert state.rng == generated_rng
    assert state.reward.gold_claimed
    assert not any(action.type == ActionType.TAKE_REWARD_GOLD for action in legal_actions(state))
