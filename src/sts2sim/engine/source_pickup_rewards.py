"""Stable custom pickup RewardsSets, nested resume frames and card alternatives."""
from __future__ import annotations

from typing import Any

from sts2sim.mechanics.relic_bag_runtime import (
    draw_source_relic,
    has_source_relic_bag,
    source_relic_rarity,
)
from sts2sim.mechanics.source_ancients import source_unlocked_characters
from sts2sim.mechanics.source_card_factory import source_noncombat_cards
from sts2sim.mechanics.source_rng import draw_source_potion

from .models import Action, ActionType, EffectEvent, EventState, RewardState, RunPhase, RunState
from .rng import capture_random_state, random_from_state, random_stream
from .source_reward_reroll import card_reroll_context

PICKUP_REWARDS = frozenset({'calling_bell', 'cauldron', 'small_capsule', 'neows_bones',
                            'orrery', 'glass_eye', 'lost_coffer'})
_NEOW_ALL_OPTIONS = (
    'cursed_pearl', 'hefty_tablet', 'large_capsule', 'leafy_poultice', 'neows_bones',
    'precarious_shears', 'silken_tress', 'silver_crucible', 'arcane_scroll', 'booming_conch',
    'fishing_rod', 'golden_pearl', 'kaleidoscope', 'lead_paperweight', 'lost_coffer',
    'massive_scroll', 'neows_torment', 'new_leaf', 'phial_holster', 'precise_scissors',
    'scroll_boxes', 'winged_boots', 'lava_rock', 'neows_talisman', 'nutritious_oyster',
    'pomander', 'small_capsule', 'stone_humidifier',
)


def push_reward(state: RunState, reward: RewardState, *, after: str | None = None) -> RunState:
    stack = list(state.flags.get('source_pickup_reward_stack', ()))
    stack.append({'phase': state.phase.value,
                  'reward': state.reward.model_dump(mode='json') if state.reward else None,
                  'event': state.event.model_dump(mode='json') if state.event else None,
                  'after': after})
    return state.model_copy(update={'phase': RunPhase.REWARD, 'reward': reward,
        'flags': {**state.flags, 'source_pickup_reward_stack': stack}})


def resume_reward(state: RunState) -> tuple[RunState, tuple[EffectEvent, ...]] | None:
    from .source_item_selection import add_random_curses

    stack = list(state.flags.get('source_pickup_reward_stack', ()))
    if not stack:
        return None
    frame = stack.pop()
    flags = dict(state.flags)
    if stack:
        flags['source_pickup_reward_stack'] = stack
    else:
        flags.pop('source_pickup_reward_stack', None)
    state = state.model_copy(update={'phase': RunPhase(frame['phase']), 'flags': flags,
        'reward': RewardState.model_validate(frame['reward']) if frame.get('reward') else None,
        'event': EventState.model_validate(frame['event']) if frame.get('event') else None})
    events: tuple[EffectEvent, ...] = (EffectEvent(kind='pickup_rewards_completed'),)
    if frame.get('after') == 'neows_bones_curse':
        rng = random_from_state(state.rng)
        state, curse_events = add_random_curses(state, rng, 1, source_id='neows_bones')
        state = state.model_copy(update={'rng': capture_random_state(rng)})
        events += curse_events
    return state, events


def _relic_offer(state: RunState, flags: dict[str, Any], rng: Any,
                 rarity: str | None = None) -> str:
    from sts2sim.mechanics.event_flows import EventFlowMarker, EventFlowMarkerKind

    from . import transitions as t

    rarity = rarity or source_relic_rarity(random_stream(rng, 'rewards'))
    if has_source_relic_bag(state):
        return draw_source_relic(state, flags, rarity)
    marker = EventFlowMarker(kind=EventFlowMarkerKind.RANDOM_RELIC, count=1, qualifier=rarity)
    values = t._draw_event_flow_relic_ids(state.model_copy(update={'flags': flags}), rng, marker)
    return values[0] if values else 'circlet'


def _neow_candidates(state: RunState) -> list[str]:
    from . import transitions as t

    multiplayer = t._flag_int(state, 'player_count', 1) > 1
    candidates = [value for value in _NEOW_ALL_OPTIONS if value != 'neows_bones']
    candidates.remove('winged_boots' if multiplayer else 'massive_scroll')
    if multiplayer:
        candidates.remove('silver_crucible')
    if len(source_unlocked_characters(state.flags)) != 5:
        candidates.remove('kaleidoscope')
    pool = t._combat_card_pool(state)
    if sum(c.rarity.value == 'common' for c in pool) < 4 or sum(
        c.rarity.value == 'uncommon' for c in pool) < 2:
        candidates.remove('scroll_boxes')
    return candidates


def obtain_pickup_reward(
    state: RunState, relic_id: str,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    from sts2sim.mechanics.relics import RelicEffectMarker

    from . import transitions as t
    from .source_run_passives import modify_reward_pool

    rng = random_from_state(state.rng)
    parent_reward = state.reward
    events: list[EffectEvent] = []
    if relic_id == 'calling_bell':
        state, emitted = t._apply_relic_add_deck_cards_marker(state, relic_id,
            RelicEffectMarker(kind='add_deck_cards', relic_id=relic_id, amount=1,
                              metadata={'card_id': 'curse_of_the_bell'}))
        events.extend(emitted)
    flags = dict(state.flags)
    reward = RewardState(reward_id=f'pickup:{relic_id}', source='other',
                         metadata={'source_relic_id': relic_id, 'relic_rewards_use_indices': True})
    if relic_id in {'calling_bell', 'small_capsule'}:
        relic_rarities = ('common', 'uncommon', 'rare') if relic_id == 'calling_bell' else (None,)
        relics = tuple(_relic_offer(state, flags, rng, rarity) for rarity in relic_rarities)
        reward = reward.model_copy(update={'relic_ids': relics})
    elif relic_id == 'cauldron':
        options, potion_rarities = t._item_potion_generation_pool(state, in_combat=False)
        reward = reward.model_copy(update={'potion_ids': tuple(draw_source_potion(
            random_stream(rng, 'rewards'), options, potion_rarities) for _ in range(5))})
    elif relic_id == 'neows_bones':
        candidates = _neow_candidates(state)
        random_stream(rng, 'rewards').shuffle(candidates)
        reward = reward.model_copy(update={'relic_ids': tuple(candidates[:2]), 'forced': True})
    elif relic_id == 'glass_eye':
        # Generate a fresh nested RewardsSet rather than merging into the parent.
        state, emitted = t._apply_glass_eye_rewards(state.model_copy(update={'reward': None}), rng)
        assert state.reward is not None
        reward = state.reward
        contexts = {key: {**value, 'uniform': True, 'no_upgrade_roll': True,
            'rarity': ('common', 'common', 'uncommon', 'uncommon', 'rare')[index]}
            for index, (key, value) in enumerate(reward.metadata['card_reroll_contexts'].items())}
        reward = reward.model_copy(update={'metadata': {**reward.metadata,
            'card_reroll_contexts': contexts}})
        state = state.model_copy(update={'reward': parent_reward})
        events.extend(emitted)
    elif relic_id in {'orrery', 'lost_coffer'}:
        groups: list[tuple[str, ...]] = []
        card_rarities: list[tuple[str, ...]] = []
        upgrades: list[tuple[bool, ...]] = []
        group_count = 5 if relic_id == 'orrery' else 1
        pool = modify_reward_pool(state, t._combat_card_pool(state))
        count = t._card_reward_choice_count(state)
        for _ in range(group_count):
            ids, rarity_ids, upgraded = source_noncombat_cards(rng, pool, count,
                no_upgrade_roll=False, act=state.act, ascension=state.ascension)
            groups.append(ids)
            card_rarities.append(rarity_ids)
            upgrades.append(upgraded)
        reward = t._reward_with_card_option_groups(None, reward_id=f'pickup:{relic_id}',
            source='other', groups=groups, rarities=card_rarities, metadata={
                'source_relic_id': relic_id, 'card_option_upgraded': upgrades[0],
                'card_option_group_upgraded': tuple(upgrades[1:]),
                'card_reroll_contexts': {'primary' if i == 0 else f'group:{i-1}':
                    card_reroll_context(count=count, no_upgrade_roll=False, is_card_reward=True)
                    for i in range(group_count)}})
        if relic_id == 'lost_coffer':
            options, potion_rarities = t._item_potion_generation_pool(state, in_combat=False)
            reward = reward.model_copy(update={'potion_ids': (draw_source_potion(
                random_stream(rng, 'rewards'), options, potion_rarities),)})
    state = state.model_copy(update={'flags': {**state.flags, **flags},
                                     'rng': capture_random_state(rng)})
    state = push_reward(state, reward, after='neows_bones_curse' if relic_id == 'neows_bones'
                        else None)
    events.append(EffectEvent(kind='relic_custom_rewards_offered', source_id=relic_id,
                              metadata={'reward_id': reward.reward_id}))
    return state, tuple(events)


def legal_sacrifices(state: RunState) -> tuple[Action, ...]:
    from .source_card_reward_modifiers import _groups

    if 'paels_wing' not in state.relics or state.reward is None:
        return ()
    return tuple(Action(type=ActionType.SACRIFICE_REWARD, target_id=f'reward:sacrifice:{key}')
                 for key, _ in _groups(state.reward))


def sacrifice_reward(state: RunState, action: Action) -> tuple[RunState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    reward = state.reward
    if reward is None:
        return state, ()
    key = str(action.target_id).removeprefix('reward:sacrifice:')
    if key == 'primary':
        reward = reward.model_copy(update={'card_claimed': True})
    else:
        index = int(key.split(':')[1])
        reward = reward.model_copy(update={'claimed_card_option_group_indices': (
            *reward.claimed_card_option_group_indices, index)})
    flags = dict(state.flags)
    total = int(flags.get('paels_wing_rewards_sacrificed', 0)) + 1
    flags['paels_wing_rewards_sacrificed'] = total
    flags['relic_counters'] = {**t._mapping_from(flags.get('relic_counters')),
                               'paels_wing': total % 2}
    state = state.model_copy(update={'reward': reward, 'flags': flags})
    events: tuple[EffectEvent, ...] = (EffectEvent(
        kind='card_reward_sacrificed', source_id='paels_wing', amount=1,
                          metadata={'card_reward_group': key, 'rewards_sacrificed': total}),)
    if total % 2 == 0:
        rng = random_from_state(state.rng)
        identifier = _relic_offer(state, flags, rng)
        state = state.model_copy(update={'flags': flags, 'rng': capture_random_state(rng)})
        state, emitted = t._add_event_relic(state, identifier)
        events += emitted
    return state, events
