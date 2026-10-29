"""Await source native event callbacks and custom rewards before finishing.

The saved cursor advances before each command can open a nested choice. Relic
rarities are pulled when their command starts, after earlier callbacks finish.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sts2sim.mechanics.event_flows import EventFlowMarker, EventFlowMarkerKind
from sts2sim.mechanics.relic_bag_runtime import (
    draw_source_relic,
    has_source_relic_bag,
    source_relic_rarity,
)

from .models import (
    Action,
    ActionType,
    EffectEvent,
    EventOptionState,
    EventState,
    RewardState,
    RunPhase,
    RunState,
)
from .rng import capture_random_state, random_from_state, random_stream

PENDING = 'source_native_event_continuation'
POTION_LOCK_EVENTS = frozenset({'RANWID_THE_ELDER', 'STONE_OF_ALL_TIME', 'THE_FUTURE_OF_POTIONS'})


def has_obtain_commands(rule: Mapping[str, Any]) -> bool:
    return not rule.get('reward_relics') and bool(
        rule.get('add_relics') or rule.get('relic_pool') or rule.get('random_relic'))


def _save(state: RunState, frame: Mapping[str, Any]) -> RunState:
    return state.model_copy(update={'flags': {**state.flags, PENDING: dict(frame)}})


def _frame(
    state: RunState, option: EventOptionState, rule: Mapping[str, Any], steps: list[dict[str, Any]],
) -> dict[str, Any]:
    assert state.event is not None
    return {'event': state.event.model_dump(mode='json'),
            'option': option.model_dump(mode='json'), 'rule': dict(rule),
            'steps': steps, 'cursor': 0,
            'stack_depth': len(state.flags.get('source_pickup_reward_stack', ())),
            'potion_lock_owned': state.event.event_id.upper() in POTION_LOCK_EVENTS,
            'potion_lock_was_present': 'can_remove_potions' in state.flags,
            'potion_lock_previous': state.flags.get('can_remove_potions', True)}


def _start(state: RunState, frame: Mapping[str, Any]) -> RunState:
    state = _save(state, frame)
    if frame['potion_lock_owned']:
        state = state.model_copy(update={'flags': {**state.flags, 'can_remove_potions': False}})
    return state


def _blocked(state: RunState, frame: Mapping[str, Any]) -> bool:
    from . import transitions as t

    return (state.phase is not RunPhase.EVENT or state.reward is not None
            or bool(t._pending_relic_deck_choices(state))
            or bool(state.flags.get('scroll_boxes_pending'))
            or bool(state.flags.get('source_direct_card_factory_choice'))
            or len(state.flags.get('source_pickup_reward_stack', ())) > int(frame['stack_depth'])
            or state.event is None or state.event.event_id != frame['event']['event_id'])


def _settle(
    before: RunState, state: RunState, action: Action, events: tuple[EffectEvent, ...],
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    from . import transitions as t
    from .source_card_reward_modifiers import apply_card_reward_modifiers

    state, events = t._apply_deck_add_relics(before, state, action, events)
    state, events = t._apply_run_progression_relics(before, state, events)
    events = tuple(event.model_copy(update={'metadata': {
        **event.metadata, 'source_deck_observers_applied': True,
    }}) for event in events)
    state, modified = apply_card_reward_modifiers(state)
    return state, (*events, *modified)


def begin_obtains(
    before: RunState, state: RunState, action: Action, option: EventOptionState,
    rule: Mapping[str, Any], events: tuple[EffectEvent, ...], deferred_cards: Sequence[str],
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    steps = [{'kind': 'relic', 'id': identifier} for identifier in rule.get('add_relics', ())]
    if rule.get('relic_pool'):
        steps.append({'kind': 'pool_relic', 'pool': tuple(rule['relic_pool'])})
    if rule.get('random_relic'):
        steps.extend({'kind': 'random_relic', 'rarity': rule.get('relic_rarity'),
                      'for_shop': bool(rule.get('relic_for_shop'))}
                     for _ in range(int(rule.get('random_relic_count', 1))))
    if rule.get('wongo_points'):
        steps.append({'kind': 'wongo_points', 'amount': int(rule['wongo_points'])})
    steps.extend({'kind': 'card', 'id': identifier} for identifier in deferred_cards)
    if rule.get('heal_to_full'):
        steps.append({'kind': 'heal_to_full'})
    if rule.get('extra_rng_draws'):
        steps.append({'kind': 'extra_rng', 'count': int(rule['extra_rng_draws'])})
    steps.append({'kind': 'finish'})
    frame = _frame(before, option, rule, steps)
    state = _start(state, frame)
    state, events = _settle(before, state, action, events)
    state, continued = resume(state)
    return state, (*events, *continued)


def offer_custom(
    state: RunState, event: EventState, option: EventOptionState, rule: Mapping[str, Any],
    reward: RewardState,
) -> RunState:
    from .source_pickup_rewards import push_reward

    parent = state.model_copy(update={'event': event, 'reward': None, 'phase': RunPhase.EVENT})
    frame = _frame(parent, option, rule, [{'kind': 'finish'}])
    # OfferCustom does not call WithSkippingDisallowed, including fixed potion
    # offers that exceed inventory capacity.
    return push_reward(_start(parent, frame), reward.model_copy(update={'forced': False}))


def _obtain(state: RunState, step: Mapping[str, Any], frame: Mapping[str, Any]) -> tuple[
    RunState, tuple[EffectEvent, ...],
]:
    from . import transitions as t
    from .source_pickup_rewards import push_reward

    rng = random_stream(random_from_state(state.rng), frame['event']['event_id'].upper())
    flags = dict(state.flags)
    if step['kind'] == 'relic':
        identifier = str(step['id'])
    elif step['kind'] == 'pool_relic':
        identifier = rng.choice(tuple(step['pool']))
    elif has_source_relic_bag(state):
        rarity = step.get('rarity') or source_relic_rarity(random_stream(rng, 'rewards'))
        identifier = draw_source_relic(state, flags, str(rarity),
                                       for_shop=bool(step.get('for_shop')))
    else:
        drawn = t._draw_event_relic_ids(state, rng, 1)
        identifier = drawn[0] if drawn else 'circlet'
    state = state.model_copy(update={'flags': flags, 'rng': capture_random_state(rng)})
    before = state
    state, events = t._add_event_relic(state, identifier)
    # Older single-screen pickup handlers can return a custom reward in-place.
    # Give it the same saved parent/resume contract as the newer pickup handlers.
    if (state.reward is not None and state.phase is RunPhase.EVENT
            and len(state.flags.get('source_pickup_reward_stack', ())) == int(frame['stack_depth'])
            and state.event is not None and state.event.event_id == frame['event']['event_id']):
        reward = state.reward
        state = push_reward(state.model_copy(update={'reward': None}), reward)
    action = Action(type=ActionType.CHOOSE_EVENT, target_id=frame['option']['option_id'])
    return _settle(before, state, action, events)


def _finish(state: RunState, frame: Mapping[str, Any]) -> tuple[RunState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    event = EventState.model_validate(frame['event'])
    option = EventOptionState.model_validate(frame['option'])
    rule = frame['rule']
    event = event.model_copy(update={'resolved_option_id': option.option_id})
    if rule.get('repeat_shop'):
        purchased = set(t._string_items(rule.get('add_relics')))
        remaining = tuple(row.model_copy(update={'disabled': state.player.gold < int(
            t._mapping_from(row.metadata.get('source_native_rule')).get('required_gold', 0))})
            for row in event.options if row.option_id != option.option_id)
        event = event.model_copy(update={'resolved_option_id': None, 'options': remaining,
            'metadata': {**event.metadata, 'fake_merchant_unsold_relic_ids': tuple(
                relic for relic in t._string_items(
                    event.metadata.get('fake_merchant_unsold_relic_ids'))
                if relic not in purchased)}})
    numeric = t._mapping_from(option.metadata.get('source_numeric_effect'))
    if numeric.get('next_page'):
        event = t._source_event_next_page(state, event, str(numeric['next_page']))
    flags = dict(state.flags)
    flags.pop(PENDING, None)
    if frame['potion_lock_owned']:
        if frame['potion_lock_was_present']:
            flags['can_remove_potions'] = frame['potion_lock_previous']
        else:
            flags.pop('can_remove_potions', None)
    return state.model_copy(update={'event': event, 'flags': flags,
        'phase': RunPhase.FAILED if state.player.hp <= 0 else RunPhase.EVENT}), (EffectEvent(
            kind='native_event_callback_completed', source_id=event.event_id,
            target_id=option.option_id),)


def resume(state: RunState) -> tuple[RunState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    raw = state.flags.get(PENDING)
    if not isinstance(raw, Mapping):
        return state, ()
    frame = dict(raw)
    events: list[EffectEvent] = []
    # Custom reward screens close automatically once all their entries resolve.
    # Do not return from an awaited CardPile.Add while its own deck choice is open.
    while (state.phase is RunPhase.REWARD and state.reward is not None
           and len(state.flags.get('source_pickup_reward_stack', ())) > int(frame['stack_depth'])
           and not t._pending_relic_deck_choices(state)
           and not t._unclaimed_reward_skip_targets(state.reward)):
        from .source_pickup_rewards import resume_reward

        before = state
        completed = resume_reward(state)
        assert completed is not None
        state, completed_events = completed
        state, completed_events = _settle(before, state, Action(
            type=ActionType.CHOOSE_EVENT, target_id=frame['option']['option_id']), completed_events)
        events.extend(completed_events)
    while not _blocked(state, frame):
        cursor = int(frame['cursor'])
        steps = list(frame['steps'])
        if cursor >= len(steps):
            break
        step = steps[cursor]
        frame['cursor'] = cursor + 1
        state = _save(state, frame)
        if step['kind'] in {'relic', 'pool_relic', 'random_relic'}:
            state, obtained = _obtain(state, step, frame)
            events.extend(obtained)
        elif step['kind'] == 'card':
            before = state
            rng = random_stream(random_from_state(state.rng), frame['event']['event_id'].upper())
            state, added = t._apply_event_flow_markers(state, (EventFlowMarker(
                kind=EventFlowMarkerKind.CARD_ADD, item_id=str(step['id'])),), rng)
            state = state.model_copy(update={'rng': capture_random_state(rng)})
            state, added = _settle(before, state, Action(type=ActionType.CHOOSE_EVENT,
                target_id=frame['option']['option_id']), added)
            events.extend(added)
        elif step['kind'] == 'heal_to_full':
            if state.player.hp > 0:
                healed = state.player.max_hp - state.player.hp
                state = state.model_copy(update={'player': state.player.model_copy(
                    update={'hp': state.player.max_hp})})
                if healed:
                    events.append(EffectEvent(kind='event_healed', amount=healed))
        elif step['kind'] == 'wongo_points':
            earned, profile = int(step['amount']), int(state.flags.get('wongo_points', 0))
            state = state.model_copy(update={'flags': {**state.flags,
                'wongo_points_earned': earned}})
            if profile % 2000 + earned >= 2000:
                steps.insert(cursor + 1, {
                    'kind': 'relic', 'id': 'wongo_customer_appreciation_badge'})
                frame['steps'] = steps
                state = _save(state, frame)
        elif step['kind'] == 'extra_rng':
            rng = random_stream(random_from_state(state.rng), frame['event']['event_id'].upper())
            for _ in range(int(step['count'])):
                rng.randrange(100)
            state = state.model_copy(update={'rng': capture_random_state(rng)})
        elif step['kind'] == 'finish':
            state, finished_events = _finish(state, frame)
            return state, (*events, *finished_events)
    return state, tuple(events)
