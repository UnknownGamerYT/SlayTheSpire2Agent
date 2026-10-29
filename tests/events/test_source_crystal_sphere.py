from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import Action, ActionType, RoomKind
from sts2sim.engine import transitions as e
from sts2sim.engine.rng import capture_random_state, random_from_state, random_stream
from sts2sim.engine.source_crystal_sphere import _sphere_rewards
from sts2sim.mechanics.source_crystal_sphere import SphereBoard, SphereItem, sphere_click


def _event():
    state = new_run('TEST', 'IRONCLAD', 0)
    state = state.model_copy(update={'flags': {**state.flags, 'event_id': 'CRYSTAL_SPHERE'},
        'player': state.player.model_copy(update={'gold': 200})})
    return e._enter_event_room(state, state.map.nodes[0].model_copy(
        update={'kind': RoomKind.EVENT}))[0]


def test_crystal_sphere_board_positions_match_independent_c_port():
    state = step(_event(), Action(type=ActionType.CHOOSE_EVENT, target_id='uncover_future'))
    board = SphereBoard.restore(state.event.metadata['crystal_sphere_board'])
    assert board.placed_all and len(board.items) == 15
    assert tuple((item.x, item.y) for item in board.items) == (
        (6, 5), (3, 5), (5, 6), (6, 2), (6, 0), (4, 1), (4, 9), (1, 3),
        (2, 6), (8, 9), (4, 5), (7, 10), (4, 6), (2, 8), (7, 4))
    assert board.remaining == 3 and len(board.cleared) == 24
    assert state.player.gold == 137
    assert state.rng.streams['CRYSTAL_SPHERE'].draws == 16
    assert len(legal_actions(state)) == 194


def test_payment_plan_adds_debt_immediately_and_has_six_divinations():
    state = step(_event(), Action(type=ActionType.CHOOSE_EVENT, target_id='payment_plan'))
    assert state.master_deck[-1].card_id == 'debt'
    assert state.player.gold == 200
    assert state.event.metadata['crystal_sphere_board']['remaining'] == 6
    assert state.reward is None


def test_big_divination_clears_square_and_item_completion_once_in_source_order():
    board = SphereBoard((SphereItem('curse', width=2, height=2, x=4, y=4,
                                   placed=True, subscriptions=1),), (), 3)
    board, completed = sphere_click(board, 5, 5, big=True)
    assert len(board.cleared) == 9 and completed == (0,) and board.revealed == (0,)
    board, completed = sphere_click(board, 3, 3, big=False)
    assert completed == () and board.revealed == (0,) and board.remaining == 1


def test_public_divination_survives_reload_and_curse_is_added_before_final_reward():
    state = step(_event(), Action(type=ActionType.CHOOSE_EVENT, target_id='payment_plan'))
    board = SphereBoard((SphereItem('curse', width=2, height=2, x=4, y=4,
                                   placed=True, subscriptions=1),), (), 1)
    from sts2sim.engine.source_crystal_sphere import _sphere_event
    state = state.model_copy(update={'event': _sphere_event(state.event, board)})
    result = step(load_state(serialize(state)), Action(type=ActionType.CHOOSE_EVENT,
                                                     target_id='big_5_5'))
    assert result.master_deck[-1].card_id == 'doubt'
    assert result.event.resolved_option_id == 'finish' and result.reward is None
    assert [a.type for a in legal_actions(result)] == [ActionType.PROCEED]


def test_sphere_rewards_override_card_items_upgrades_potion_and_fixed_gold_stream():
    state = _event()
    rng = random_stream(random_from_state(state.rng), 'CRYSTAL_SPHERE')
    board = SphereBoard((SphereItem('card', 'common'), SphereItem('potion', 'common'),
                         SphereItem('gold')), (), 0, revealed=(0, 1, 2))
    next_state, reward = _sphere_rewards(state, board, rng)
    assert reward.gold == 10 and len(reward.card_options) == 3 and len(reward.potion_ids) == 1
    assert capture_random_state(rng).streams['CRYSTAL_SPHERE'].draws == 9  # startup +8
    assert capture_random_state(rng).streams['rewards'].draws == 0
    assert next_state.flags == state.flags
