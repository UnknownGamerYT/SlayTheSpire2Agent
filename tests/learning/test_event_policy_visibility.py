"""Event observations expose revealed facts while retaining exact engine saves."""
from __future__ import annotations

from dataclasses import replace

from sts2sim import action_space, encode_observation, new_run, serialize, step
from sts2sim.engine import Action, ActionType, RoomKind
from sts2sim.engine import transitions as t
from sts2sim.engine.source_crystal_sphere import _sphere_event
from sts2sim.learning.masked_ppo import _policy_input
from sts2sim.learning.observation import encode_rich_observation
from sts2sim.mechanics.source_crystal_sphere import SphereBoard, SphereItem
from sts2sim.policy_visibility import policy_payload, policy_state


def _event(identifier):
    state = new_run('EVENT-VISIBILITY', 'IRONCLAD', 0)
    state = state.model_copy(update={'relics': ('anchor',),
        'flags': {**state.flags, 'event_id': identifier}})
    return t._enter_event_room(state, state.map.nodes[0].model_copy(
        update={'kind': RoomKind.EVENT}))[0]


def _views(state):
    compact = encode_observation(state, include_state=True)
    actions = action_space(state)
    return (
        policy_payload(serialize(state)), serialize(policy_state(state)), compact, actions,
        encode_rich_observation(state, include_serialized_state=True),
        _policy_input(compact, {'action_space': actions}),
    )


def _sphere_pair():
    state = _event('CRYSTAL_SPHERE')
    board = SphereBoard(items=(
        SphereItem('gold', x=4, y=4, placed=True, subscriptions=1),
        SphereItem('card', 'common', 2, 2, 2, 4, True, 1),
        SphereItem('card', 'uncommon', 2, 2, 6, 6, True, 1),
        SphereItem('card', 'rare', 2, 2, 8, 8, True, 1),
    ), cleared=(26, 48), remaining=3, revealed=(0,), placed_all=True)
    first = state.model_copy(update={'event': _sphere_event(state.event, board)})
    hidden = (*board.items[:2], replace(board.items[3], x=6, y=6),
              replace(board.items[2], x=8, y=8, subscriptions=3))
    second = state.model_copy(update={'event': _sphere_event(state.event, replace(
        board, items=hidden, placed_all=False))})
    return first, second


def test_sphere_hidden_placements_retry_counts_and_item_order_do_not_reach_policy_inputs():
    first, second = _sphere_pair()
    originals = (serialize(first), serialize(second))
    assert _views(first) == _views(second)
    assert (serialize(first), serialize(second)) == originals
    board = policy_payload(serialize(first))['event']['metadata']['crystal_sphere_board']
    assert set(board) == {'remaining', 'cleared', 'visible_cells', 'revealed_items'}
    assert board['visible_cells'] == [
        {'x': 2, 'y': 4, 'kind': 'card', 'rarity': 'common'},
        {'x': 4, 'y': 4, 'kind': 'gold', 'rarity': ''},
    ]
    assert board['revealed_items'] == [
        {'kind': 'gold', 'rarity': '', 'x': 4, 'y': 4, 'width': 1, 'height': 1},
    ]
    assert policy_payload(policy_payload(serialize(first))) == policy_payload(serialize(first))
    assert all('sphere_cell' in descriptor['event_option']['metadata']
               for descriptor in action_space(first))


def test_sphere_policy_observations_change_only_after_cells_reveal_the_different_contents():
    first, second = _sphere_pair()
    action = Action(type=ActionType.CHOOSE_EVENT, target_id='small_6_6')
    first, second = step(first, action), step(second, action)
    assert encode_rich_observation(first)['event'] != encode_rich_observation(second)['event']
    for state, rarity in ((first, 'uncommon'), (second, 'rare')):
        cells = encode_rich_observation(state)['event']['metadata'][
            'crystal_sphere_board']['visible_cells']
        assert next(cell for cell in cells if (cell['x'], cell['y']) == (6, 6))['rarity'] == rarity


def test_oracle_and_engine_save_retain_full_sphere_boards():
    first, second = _sphere_pair()
    assert policy_payload(serialize(first), oracle=True) == serialize(first)
    assert serialize(policy_state(first, oracle=True)) == serialize(first)
    assert encode_rich_observation(first, oracle=True)['event'] != encode_rich_observation(
        second, oracle=True)['event']
    raw = encode_observation(first, include_state=True, oracle=True)['state']
    assert len(raw['event']['metadata']['crystal_sphere_board']['items']) == 4
    assert 'rng' in raw and 'seed' in raw


def test_saved_sphere_events_inside_pickup_frames_use_the_same_visible_projection():
    first, second = _sphere_pair()
    current = _event('RELIC_TRADER')
    states = [current.model_copy(update={'flags': {**current.flags,
        'source_pickup_reward_stack': [{
            'phase': 'event', 'reward': None, 'event': outer.event.model_dump(mode='json'),
            'after': None,
        }],
    }}) for outer in (first, second)]
    assert _views(states[0]) == _views(states[1])
    frame = policy_payload(serialize(states[0]))['flags']['source_pickup_reward_stack'][0]
    assert set(frame['event']['metadata']['crystal_sphere_board']) == {
        'remaining', 'cleared', 'visible_cells', 'revealed_items',
    }


def test_startup_flags_are_private_in_current_and_nested_saved_outer_events():
    first = _event('RELIC_TRADER')
    assert 'player_relic_bag' in first.event.metadata['source_startup_flags']
    original_event = first.event.model_dump(mode='json')
    first = first.model_copy(update={'flags': {**first.flags,
        'source_pickup_reward_stack': [{
            'phase': 'event', 'reward': {'reward_id': 'already-revealed',
                'card_options': ['anger']}, 'event': original_event, 'after': None,
        }]}})
    second_event = first.event.model_copy(update={'metadata': {
        **first.event.metadata, 'source_startup_flags': {
            'act_room_sets': {'1': {'events': ['hidden_future_event']}},
            'player_relic_bag': {'Common': ['hidden_future_relic']},
        },
    }})
    second = first.model_copy(update={'event': second_event, 'flags': {**first.flags,
        'source_pickup_reward_stack': [{
            **first.flags['source_pickup_reward_stack'][0],
            'event': second_event.model_dump(mode='json'),
        }]}})
    assert _views(first) == _views(second)
    public = policy_payload(serialize(first))
    assert 'source_startup_flags' not in public['event']['metadata']
    frame = public['flags']['source_pickup_reward_stack'][0]
    assert 'source_startup_flags' not in frame['event']['metadata']
    # Already revealed outer offers stay public.
    assert frame['reward']['card_options'] == ['anger']
    assert policy_payload(serialize(first), oracle=True) == serialize(first)
