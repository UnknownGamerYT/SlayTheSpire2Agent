"""Direct stable card factories that do not construct CardReward instances.

Hefty Tablet and Lead Paperweight use a simple optional card selector. Fried
Eel adds its generated card directly. Factory flags and hashes are retained in
source_card_creation_review.json; callbacks still receive IsCardReward=false.
"""
from __future__ import annotations

from random import Random

from sts2sim.mechanics.event_flows import EventFlowMarker, EventFlowMarkerKind
from sts2sim.mechanics.source_card_factory import source_noncombat_cards

from .models import EffectEvent, EventOptionState, EventState, RunPhase, RunState
from .rng import capture_random_state, random_from_state
from .source_card_reward_modifiers import _current_card, modify_generated_cards

_PENDING = 'source_direct_card_factory_choice'


def pickup(
    state: RunState, relic_id: str, rng: Random | None = None,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    rng = rng or random_from_state(state.rng)
    hefty = relic_id == 'hefty_tablet'
    marker = EventFlowMarker(kind=EventFlowMarkerKind.RANDOM_CARD,
                             qualifier='character' if hefty else 'colorless')
    pool = t._event_flow_card_pool(state, marker)
    if hefty:
        pool = tuple(card for card in pool if card.rarity.value == 'rare')
    ids, _, upgrades = source_noncombat_cards(rng, pool, 3 if hefty else 2,
        uniform=hefty, no_upgrade_roll=hefty, act=state.act, ascension=state.ascension)
    state, upgrades, modifiers, events = modify_generated_cards(
        state, rng, ids, upgrades, is_card_reward=False,
        context={'group_key': f'pickup:{relic_id}', 'no_upgrade_roll': hefty},
    )
    offers = tuple(_current_card(state, identifier, upgrades[index],
        t._mapping_from(modifiers.get(str(index)))) for index, identifier in enumerate(ids))
    pending = {'relic_id': relic_id, 'resume_phase': state.phase.value,
               'resume_event': state.event.model_dump(mode='json') if state.event else None,
               'offers': tuple(card.model_dump(mode='json') for card in offers)}
    options = tuple(EventOptionState(option_id=f'factory:{index}', title=card.name or card.card_id,
        metadata={'source_factory_card': card.model_dump(mode='json'),
                  'source_relic_id': relic_id, 'fixed_card_ids': (card.card_id,)})
        for index, card in enumerate(offers)) + (EventOptionState(
            option_id='factory:skip', title='Skip', metadata={'source_relic_id': relic_id}),)
    return state.model_copy(update={'phase': RunPhase.EVENT,
        'flags': {**state.flags, _PENDING: pending}, 'rng': capture_random_state(rng),
        'event': EventState(event_id=f'pickup:{relic_id}', name=relic_id.replace('_', ' ').title(),
                            options=options)}), events + (EffectEvent(
        kind='direct_factory_card_choice_offered', source_id=relic_id, amount=len(offers),
        metadata={'is_card_reward': False, 'no_upgrade_roll': hefty}),)


def choose(
    state: RunState, option: EventOptionState,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    from . import transitions as t
    from .models import CardInstance

    pending = t._mapping_from(state.flags.get(_PENDING))
    if not pending:
        return state, ()
    relic_id = str(pending['relic_id'])
    flags = dict(state.flags)
    flags.pop(_PENDING, None)
    saved = pending.get('resume_event')
    state = state.model_copy(update={'phase': RunPhase(str(pending['resume_phase'])),
        'event': EventState.model_validate(saved) if saved else None, 'flags': flags})
    cards = []
    raw = option.metadata.get('source_factory_card')
    if raw:
        cards.append(CardInstance.model_validate(raw))
    if relic_id == 'hefty_tablet':
        cards.append(t._card_from_spec(
            t._reward_card_spec(state, 'injury'), len(state.master_deck)))
    events: list[EffectEvent] = []
    for card in cards:
        card = t._with_unique_deck_card_id(card, state.master_deck)
        card, upgraded = t._upgrade_card_for_add_relics(state, card)
        state = state.model_copy(update={'master_deck': state.master_deck + (card,)})
        events.append(EffectEvent(kind='relic_deck_card_added', source_id=relic_id,
            target_id=card.instance_id, amount=1, metadata={'card_id': card.card_id}))
        if upgraded is not None:
            events.append(upgraded)
    return state, (EffectEvent(kind='direct_factory_card_choice_resolved', source_id=relic_id,
                               metadata={'skipped': not bool(raw)}),) + tuple(events)


def fried_eel(
    state: RunState, rng: Random,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    pool = t._event_flow_card_pool(state, EventFlowMarker(
        kind=EventFlowMarkerKind.RANDOM_CARD, qualifier='colorless'))
    ids, _, upgrades = source_noncombat_cards(rng, pool, 1, act=state.act,
                                               ascension=state.ascension)
    state, upgrades, modifiers, events = modify_generated_cards(
        state, rng, ids, upgrades, is_card_reward=False,
        context={'group_key': 'event:endless_conveyor:fried_eel', 'no_upgrade_roll': True},
    )
    if not ids:
        return state, events
    card = _current_card(state, ids[0], upgrades[0], t._mapping_from(modifiers.get('0')))
    card = t._with_unique_deck_card_id(card, state.master_deck)
    card, upgraded = t._upgrade_card_for_add_relics(state, card)
    state = state.model_copy(update={'master_deck': state.master_deck + (card,)})
    return state, events + (EffectEvent(kind='event_random_card_added', source_id='event_flow',
        target_id=card.instance_id, metadata={'card_id': card.card_id, 'qualifier': 'colorless'}),
    ) + ((upgraded,) if upgraded is not None else ())
