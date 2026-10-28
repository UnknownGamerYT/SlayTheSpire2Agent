"""Public event choices use source Damage hooks, distinct from Max-HP costs."""
from __future__ import annotations

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import ActionType, RoomKind, RunPhase
from sts2sim.engine import transitions as e


def _event(event_id, *, hp=50, max_hp=80, relics=('tungsten_rod',), statuses=None):
    state = new_run('TEST', 'IRONCLAD', 0)
    state = state.model_copy(update={
        'flags': {**state.flags, 'event_id': event_id},
        'relics': relics,
        'player': state.player.model_copy(update={
            'hp': hp, 'max_hp': max_hp, 'gold': 1000, 'block': 40,
            'statuses': statuses or {'weak': 2, 'vulnerable': 2, 'tank': 1},
        }),
    })
    return e._enter_event_room(state, state.map.nodes[0].model_copy(
        update={'kind': RoomKind.EVENT}))[0]


def _choose(state, option_id):
    restored = load_state(serialize(state))
    action = next(action for action in legal_actions(restored)
                  if action.type == ActionType.CHOOSE_EVENT
                  and str(action.target_id).upper() == option_id)
    return step(restored, action)


@pytest.mark.parametrize(('event_id', 'option_id', 'damage'), [
    ('DENSE_VEGETATION', 'TRUDGE_ON', 8),
    ('JUNGLE_MAZE_ADVENTURE', 'SOLO_QUEST', 18),
    ('SUNKEN_STATUE', 'DIVE_INTO_WATER', 7),
    ('THIS_OR_THAT', 'PLAIN', 6),
    ('BRAIN_LEECH', 'RIP', 5),
    ('DOLL_ROOM', 'TAKE_SOME_TIME', 5),
    ('DOLL_ROOM', 'EXAMINE', 15),
    ('ROOM_FULL_OF_CHEESE', 'SEARCH', 14),
    ('STONE_OF_ALL_TIME', 'PUSH', 6),
    ('SPIRIT_GRAFTER', 'REJECTION', 10),
    ('THE_LEGENDS_WERE_TRUE', 'SLOWLY_FIND_AN_EXIT', 8),
    ('TRASH_HEAP', 'DIVE_IN', 8),
    ('WHISPERING_HOLLOW', 'HUG', 9),
    ('SLIPPERY_BRIDGE', 'HOLD_ON_0', 3),
])
def test_native_damage_ignores_block_and_attack_modifiers_but_honors_tungsten(
    event_id, option_id, damage,
):
    result = _choose(_event(event_id), option_id)
    event = next(event for event in result.replay_log[-1].events
                 if event.kind == 'player_damaged')
    assert event.amount == damage - 1
    assert event.metadata['base'] == damage
    assert event.metadata['unblockable']
    assert event.metadata['blocked'] == 0
    assert event.metadata['tungsten_rod_reduction'] == 1
    assert result.player.hp == 50 - (damage - 1)
    assert result.player.block == 40


def test_round_tea_second_page_uses_the_same_damage_callback_after_reload():
    state = _choose(_event('ROUND_TEA_PARTY'), 'PICK_FIGHT')
    result = _choose(state, 'CONTINUE_FIGHT')
    assert result.player.hp == 40
    assert result.player.block == 40


@pytest.mark.parametrize(('statuses', 'relics', 'expected_hp', 'expected_buffer'), [
    ({'buffer': 1}, (), 50, 0),
    ({'intangible': 1}, (), 49, 0),
    ({'intangible': 1, 'buffer': 1}, ('tungsten_rod',), 50, 1),
])
def test_native_event_damage_resolves_intangible_tungsten_then_buffer(
    statuses, relics, expected_hp, expected_buffer,
):
    result = _choose(_event('DENSE_VEGETATION', statuses=statuses, relics=relics), 'TRUDGE_ON')
    assert result.player.hp == expected_hp
    assert result.player.statuses.get('buffer', 0) == expected_buffer
    assert result.player.block == 40


def test_colossal_defense_prevents_false_death_and_keeps_the_next_page_playable():
    result = _choose(_event('COLOSSAL_FLOWER', hp=5), 'REACH_DEEPER_1')
    assert result.player.hp == 1
    assert result.player.block == 40
    assert result.phase == RunPhase.EVENT
    assert result.event.page_id == 'REACH_DEEPER_1'
    assert {action.target_id for action in legal_actions(load_state(serialize(result)))} == {
        'EXTRACT_CURRENT_PRIZE_2', 'REACH_DEEPER_2',
    }
    assert all(event.kind != 'event_player_died' for event in result.replay_log[-1].events)


def test_baths_apply_max_hp_gain_before_damage_and_warning_uses_actual_hp():
    state = _choose(_event('ABYSSAL_BATHS', hp=5), 'IMMERSE')
    assert state.player.hp == 5 and state.player.max_hp == 82
    result = _choose(state, 'LINGER')
    assert result.player.hp == 4 and result.player.max_hp == 84
    assert result.player.block == 40
    assert result.event.page_id == 'LINGER1'
    loss = next(event for event in result.replay_log[-1].events
                if event.kind == 'event_hp_lost')
    assert loss.amount == 3


def test_baths_buffer_prevents_raw_lethal_damage_and_is_consumed_once():
    result = _choose(_event('ABYSSAL_BATHS', hp=1, relics=(), statuses={'buffer': 1}), 'IMMERSE')
    assert result.player.hp == 3 and result.player.max_hp == 82
    assert not result.player.statuses.get('buffer')
    assert result.phase == RunPhase.EVENT
    assert any(action.target_id == 'LINGER' for action in legal_actions(result))


def test_tablet_lethal_max_hp_cost_is_not_damage_and_bypasses_damage_defenses():
    state = _event('TABLET_OF_TRUTH', hp=5, max_hp=5, statuses={'buffer': 1, 'intangible': 1})
    state = _choose(state, 'DECIPHER_1')
    result = _choose(state, 'DECIPHER')
    assert result.player.hp == 0 and result.phase == RunPhase.FAILED
    assert result.player.statuses['buffer'] == 1
    assert all(event.kind != 'player_damaged' for event in result.replay_log[-1].events)
