"""Lava Lamp observes the current CombatRoom across nested reward sets."""
from __future__ import annotations

import pytest
from rewards.test_remaining_relic_reward_interactions import _reward_after_kill

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import RewardState, RoomKind, RunPhase
from sts2sim.engine.source_run_passives import in_combat_room


def _nested_orrery(kind=RoomKind.MONSTER, *, event_fight=False):
    state = _reward_after_kill(seed=9401, relics=('lava_lamp',))
    assert state.map is not None and state.reward is not None
    current = state.map.current_node_id
    game_map = state.map.model_copy(update={'nodes': tuple(
        node.model_copy(update={'kind': kind}) if node.node_id == current else node
        for node in state.map.nodes
    )})
    # This is the serialized native pickup queue frame: parent encounter rewards
    # are suspended while five Source.Other Orrery CardRewards are displayed.
    flags = {**state.flags, 'source_pickup_reward_stack': [{
        'phase': state.phase.value, 'reward': state.reward.model_dump(mode='json'),
        'event': None, 'after': None,
    }]}
    if event_fight:
        flags.update(event_fight_id='punch_off', event_combat_restore_flags={
            'event_fight_id': {'present': False, 'value': None},
        })
    child = RewardState(reward_id='pickup:orrery', source='other', gold=1,
        card_options=('common_attack',), card_option_groups=(('common_attack',),) * 4,
        metadata={'source_relic_id': 'orrery'})
    return load_state(serialize(state.model_copy(update={
        'map': game_map, 'reward': child, 'flags': flags,
    })))


@pytest.mark.parametrize('kind,event_fight', [
    (RoomKind.MONSTER, False), (RoomKind.ELITE, False), (RoomKind.BOSS, False),
    (RoomKind.EVENT, True),
])
def test_lava_upgrades_nested_other_source_rewards_in_actual_combat_room(kind, event_fight):
    state = _nested_orrery(kind, event_fight=event_fight)
    action = next(a for a in legal_actions(state) if a.type == 'take_reward_gold')
    result = step(state, action)
    assert serialize(result) == serialize(step(load_state(serialize(state)), action))
    assert tuple(result.reward.metadata['card_option_upgraded']) == (True,)
    assert all(tuple(upgrades) == (True,) for upgrades in
               result.reward.metadata['card_option_group_upgraded'])
    events = [e for e in result.replay_log[-1].events if e.source_id == 'lava_lamp']
    assert len(events) == 5
    assert all(e.metadata['in_combat_room'] is True for e in events)
    card_action = next(a for a in legal_actions(result) if a.type == 'take_reward_card')
    assert step(load_state(serialize(result)), card_action).master_deck[-1].upgraded


@pytest.mark.parametrize('kind', [RoomKind.SHOP, RoomKind.REST, RoomKind.TREASURE,
                                 RoomKind.EVENT, RoomKind.START])
def test_archived_combat_does_not_make_noncombat_room_eligible(kind):
    state = _nested_orrery(kind)
    assert state.combat is not None  # RunState retains the previous encounter.
    assert not in_combat_room(state)
    result = step(state, next(a for a in legal_actions(state) if a.type == 'take_reward_gold'))
    assert not any(result.reward.metadata['card_option_upgraded'])


def test_archived_combat_and_old_map_node_do_not_qualify_after_room_exit():
    state = _nested_orrery().model_copy(update={'phase': RunPhase.MAP})
    assert not in_combat_room(state, {'in_combat_room': True})


def test_nested_combat_rewards_do_not_upgrade_after_blockable_damage():
    state = _nested_orrery()
    state = state.model_copy(update={'combat': state.combat.model_copy(update={
        'metadata': {**state.combat.metadata, 'lava_lamp_took_damage': True},
    })})
    result = step(state, next(a for a in legal_actions(state) if a.type == 'take_reward_gold'))
    assert not any(result.reward.metadata['card_option_upgraded'])
