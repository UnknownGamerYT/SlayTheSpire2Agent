"""Public action adapter for the stable Crystal Sphere spatial minigame."""
from __future__ import annotations

from collections.abc import Mapping
from random import Random

from sts2sim.mechanics.event_flows import EventFlowMarker, EventFlowMarkerKind
from sts2sim.mechanics.relic_bag_runtime import (
    draw_source_relic,
    has_source_relic_bag,
    source_relic_rarity,
)
from sts2sim.mechanics.source_card_factory import source_noncombat_cards
from sts2sim.mechanics.source_crystal_sphere import SphereBoard, new_sphere_board, sphere_click

from .models import EffectEvent, EventOptionState, EventState, RewardState, RunState
from .rng import capture_random_state, random_from_state, random_stream


def _sphere_event(event: EventState, board: SphereBoard) -> EventState:
    options = tuple(EventOptionState(option_id=f'{tool}_{x}_{y}',
        title=f'{tool.title()} divination ({x},{y})', metadata={
            'sphere_cell': {'x': x, 'y': y, 'big': tool == 'big'}})
        for x in range(11) for y in range(11) if x * 11 + y not in board.cleared
        for tool in ('small', 'big')) if board.remaining > 0 else ()
    return event.model_copy(update={'page_id': 'DIVINATION', 'options': options,
        'resolved_option_id': None if board.remaining else 'finish',
        'metadata': {**event.metadata, 'crystal_sphere_board': board.payload()}})


def choose_sphere(
    state: RunState, option: EventOptionState
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    from . import transitions as e

    if state.event is None:
        return state, ()
    rng = random_stream(random_from_state(state.rng), 'CRYSTAL_SPHERE')
    events: list[EffectEvent] = [EffectEvent(kind='event_option_chosen',
        source_id='CRYSTAL_SPHERE', target_id=option.option_id)]
    raw = state.event.metadata.get('crystal_sphere_board')
    next_state = state
    if not isinstance(raw, Mapping):
        if option.option_id.upper() == 'UNCOVER_FUTURE':
            next_state, costs = e._apply_event_option_immediate_effects(state, option, rng)
            events.extend(costs)
            count = 3
        elif option.option_id.upper() == 'PAYMENT_PLAN':
            next_state, added = e._apply_event_flow_markers(state, (EventFlowMarker(
                kind=EventFlowMarkerKind.CARD_ADD, item_id='debt'),), rng)
            events.extend(added)
            count = 6
        else:
            return state, ()
        board = new_sphere_board(rng, count)
    else:
        board = SphereBoard.restore(raw)
        cell = option.metadata.get('sphere_cell', {})
        if not isinstance(cell, Mapping):
            return state, ()
        board, completed = sphere_click(
            board, int(cell['x']), int(cell['y']), big=bool(cell['big']))
        for index in completed:
            if board.items[index].kind == 'curse':
                next_state, added = e._apply_event_flow_markers(next_state, (EventFlowMarker(
                    kind=EventFlowMarkerKind.CARD_ADD, item_id='doubt'),), rng)
                events.extend(added)
        events.append(EffectEvent(kind='crystal_sphere_cells_revealed',
            metadata={'remaining': board.remaining, 'completed_items': completed}))
    event = _sphere_event(state.event, board)
    reward = None
    if board.remaining == 0:
        next_state, reward = _sphere_rewards(next_state, board, rng)
        if reward is not None:
            events.extend(e._reward_generated_events(reward))
    return next_state.model_copy(update={'event': event, 'reward': reward,
        'rng': capture_random_state(rng)}), tuple(events)


def _sphere_rewards(
    state: RunState, board: SphereBoard, rng: Random
) -> tuple[RunState, RewardState | None]:
    from . import transitions as e

    items = [board.items[index] for index in board.revealed]
    potion_pool = e._combat_potion_pool(state)
    potion_rarity = e._combat_potion_rarities(state)
    potions: list[str] = []
    # ToReward creates predetermined potions before RewardsSet calls Populate.
    for item in items:
        if item.kind == 'potion':
            candidates = [potion for potion in potion_pool
                          if potion_rarity.get(potion) == item.rarity]
            if candidates:
                potions.append(rng.choice(candidates))
    gold = 0
    relics: list[str] = []
    card_groups: list[tuple[str, ...]] = []
    rarities: list[tuple[str, ...]] = []
    upgrades: list[tuple[bool, ...]] = []
    flags = dict(state.flags)
    for item in items:
        if item.kind in {'gold', 'big_gold'}:
            amount = 30 if item.kind == 'big_gold' else 10
            gold += amount + rng.randrange(1)  # GoldReward.Populate draws even fixed amounts.
        elif item.kind == 'relic':
            rarity = source_relic_rarity(rng)
            if has_source_relic_bag(state):
                relics.append(draw_source_relic(state, flags, rarity))
            else:
                relics.extend(e._draw_event_relic_ids(state, rng, 1))
        elif item.kind == 'card':
            pool = tuple(card for card in e._event_flow_card_pool(state, EventFlowMarker(
                kind=EventFlowMarkerKind.CARD_REWARD, qualifier='character'))
                         if card.rarity.value == item.rarity)
            cards, card_rarities, card_upgrades = source_noncombat_cards(
                rng, pool, 3, uniform=True, no_upgrade_roll=False,
                item_rng=rng, act=state.act, ascension=state.ascension)
            card_groups.append(cards)
            rarities.append(card_rarities)
            upgrades.append(card_upgrades)
    reward: RewardState | None = None
    if card_groups:
        reward = e._reward_with_card_option_groups(None, reward_id='event:crystal_sphere',
            groups=tuple(card_groups), rarities=tuple(rarities), metadata={
                'card_option_upgraded': upgrades[0],
                'card_option_group_upgraded': tuple(upgrades[1:])})
    if gold or relics or potions:
        reward = (reward or RewardState(
            reward_id='event:crystal_sphere', source='event')).model_copy(
            update={'gold': gold, 'relic_ids': tuple(relics), 'potion_ids': tuple(potions)})
    return state.model_copy(update={'flags': flags}), reward
