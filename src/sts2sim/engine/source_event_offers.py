"""Native event offers whose identities are fixed before any choice is made."""
from __future__ import annotations

from collections.abc import Mapping
from random import Random
from typing import Any

from sts2sim.content.source_relic_traits import PICKUP_OR_PET_RELIC_IDS
from sts2sim.mechanics.relic_bag_runtime import (
    draw_source_relic,
    has_source_relic_bag,
    source_relic_rarity,
)
from sts2sim.mechanics.source_ancients import source_unlocked_characters
from sts2sim.mechanics.source_rng import float32, source_float

from .models import EventOptionState, EventState, RunState
from .rng import random_stream
from .source_enchantment_hooks import _SOURCE_BASIC_CARDS as SOURCE_BASIC_CARDS


def source_tradable_relics(state: RunState, rows: tuple[Mapping[str, Any], ...]) -> tuple[str, ...]:
    rarity = {str(row.get('id', '')).lower(): str(row.get('rarity_key', '')).lower()
              for row in rows}
    used = set(state.flags.get('used_up_relic_ids', ()))
    melted = set(state.flags.get('melted_relic_ids', ()))
    return tuple(relic for relic in state.relics if relic not in PICKUP_OR_PET_RELIC_IDS
                 and relic not in used and relic not in melted
                 and rarity.get(relic, '') not in {'starter', 'event', 'ancient'})


def native_dynamic_offers(
    state: RunState, event: EventState, rng: Random, *,
    relic_rows: tuple[Mapping[str, Any], ...], potion_rarities: Mapping[str, str],
) -> EventState:
    event_id = event.event_id.upper()
    options = list(event.options)
    metadata = dict(event.metadata)
    if event_id == 'FAKE_MERCHANT':
        relics = ['fake_anchor', 'fake_blood_vial', 'fake_happy_flower', 'fake_lees_waffle',
                  'fake_mango', 'fake_orichalcum', 'fake_snecko_eye', 'fake_strike_dummy',
                  'fake_venerable_tea_set']
        rng.shuffle(relics)
        relics = relics[:6]
        options = []
        for relic in relics:
            price = round(float32(50 * source_float(random_stream(rng, 'shops'), .85, 1.15)))
            options.append(EventOptionState(option_id=f'buy_{relic}', title=relic,
                description=f'Pay {price} Gold.', disabled=state.player.gold < price,
                metadata={'source_native_rule': {'selection': '', 'add_relics': (relic,),
                    'repeat_shop': True, 'required_gold': price},
                    'source_numeric_effect': {'gold_delta': -price}}))
        options.append(EventOptionState(option_id='leave', title='Leave', metadata={
            'source_native_rule': {'selection': ''}}))
        metadata['fake_merchant_unsold_relic_ids'] = tuple(relics)
    elif event_id == 'SLIPPERY_BRIDGE':
        previous = str(metadata.get('bridge_card_instance_id', ''))
        skipped = set(metadata.get('bridge_skipped_instance_ids', ()))
        removable = [card for card in state.master_deck
                     if not any(enchantment.keyword.lower() == 'eternal'
                                for enchantment in card.enchantments)
                     and 'eternal' not in card.tags]
        previous_card = next((card for card in removable if card.instance_id == previous), None)
        if previous_card is None:
            candidates = [card for card in removable if
                          card.card_id not in SOURCE_BASIC_CARDS]
        else:
            skipped.add(previous)
            candidates = [card for card in removable if card.card_id != previous_card.card_id
                          and card.instance_id not in skipped]
        chosen = rng.choice(candidates or removable) if removable else None
        holds = int(metadata.get('bridge_hold_count', 0))
        metadata.update({'bridge_card_instance_id': chosen.instance_id if chosen else '',
                         'bridge_skipped_instance_ids': tuple(sorted(skipped))})
        suffix = str(holds) if holds < 7 else 'LOOP'
        options = [EventOptionState(option_id='OVERCOME', title='Overcome',
            description=f'Remove {chosen.name}.' if chosen else 'No removable cards.',
            disabled=chosen is None, metadata={'source_native_rule': {'selection': '',
                'remove_instance_id': chosen.instance_id if chosen else ''}}),
            EventOptionState(option_id=f'HOLD_ON_{suffix}', title='Hold On',
                description=f'Lose {3 + holds} HP.', metadata={
                    'source_native_rule': {'selection': '', 'bridge_hold': True},
                    'source_numeric_effect': {'hp_loss': 3 + holds}})]
    elif event_id == 'WELCOME_TO_WONGOS':
        flags = dict(state.flags)
        if has_source_relic_bag(state):
            featured = draw_source_relic(state, flags, 'rare', for_shop=True)
            for index, option in enumerate(options):
                if option.option_id.upper() == 'FEATURED_ITEM':
                    rule = dict(option.metadata.get('source_native_rule', {}))
                    rule['add_relics'] = (featured,)
                    options[index] = option.model_copy(update={
                        'title': featured, 'metadata': {**option.metadata,
                                                      'source_native_rule': rule}})
            metadata['source_startup_flags'] = flags
    elif event_id == 'COLORFUL_PHILOSOPHERS':
        unlocked = {character.upper() for character in source_unlocked_characters(state.flags)}
        colors = [color for color in ('NECROBINDER', 'IRONCLAD', 'REGENT', 'SILENT', 'DEFECT')
                  if color != state.character_id.upper() and color in unlocked]
        while len(colors) > 3:
            colors.pop(rng.randrange(len(colors)))
        options = [EventOptionState(option_id=color.lower(), title=color.title(), metadata={
            'source_native_rule': {'selection': '', 'generated_color': color.lower(),
                                  'colorful_groups': True}}) for color in colors]
    elif event_id in {'STONE_OF_ALL_TIME', 'RANWID_THE_ELDER'}:
        # Empty NextItem returns null without drawing in the source.
        potion_index = rng.randrange(len(state.potions)) if state.potions else None
        if event_id == 'STONE_OF_ALL_TIME':
            options = [option for option in options if option.option_id.upper() != 'LIFT_LOCKED']
            for i, option in enumerate(options):
                if option.option_id.upper() == 'LIFT':
                    rule = dict(option.metadata.get('source_native_rule', {}))
                    rule['remove_potion_index'] = potion_index
                    options[i] = option.model_copy(update={'disabled': potion_index is None,
                        'metadata': {**option.metadata, 'source_native_rule': rule}})
        else:
            tradable = source_tradable_relics(state, relic_rows)
            relic = rng.choice(tradable) if tradable else ''
            options = [EventOptionState(option_id=key, title=key.title(), disabled=disabled,
                metadata={'source_native_rule': rule}) for key, disabled, rule in (
                ('potion', potion_index is None, {'selection': '',
                                                    'remove_potion_index': potion_index,
                                          'random_relic': True}),
                ('gold', state.player.gold < 100, {'selection': '', 'random_relic': True,
                    'numeric': {'gold_delta': -100}, 'required_gold': 100}),
                ('relic', not relic, {'selection': '', 'remove_relic': relic,
                                     'random_relic': True, 'random_relic_count': 2}))]
            for i, option in enumerate(options):
                rule = option.metadata['source_native_rule']
                if rule.get('numeric'):
                    options[i] = option.model_copy(update={'metadata': {**option.metadata,
                        'source_numeric_effect': rule['numeric']}})
    elif event_id == 'RELIC_TRADER':
        owned = sorted(source_tradable_relics(state, relic_rows))
        rng.shuffle(owned)
        owned = owned[:3]
        flags = dict(state.flags)
        new = []
        if owned and has_source_relic_bag(state):
            for _ in range(3):
                rarity = source_relic_rarity(random_stream(rng, 'rewards'))
                new.append(draw_source_relic(state, flags, rarity))
        options = [EventOptionState(option_id=key, title=f'{old} → {relic}', metadata={
            'source_native_rule': {'selection': '', 'remove_relic': old, 'add_relics': (relic,)}})
            for key, old, relic in zip(('top', 'middle', 'bottom'), owned, new, strict=False)]
        if not options:
            options = [EventOptionState(option_id='proceed', title='Proceed', metadata={
                'source_native_rule': {'selection': ''}})]
        metadata['source_startup_flags'] = flags
    elif event_id == 'THE_FUTURE_OF_POTIONS':
        types = []
        for potion in state.potions:
            rarity = potion_rarities.get(potion, 'common')
            types.append(rng.choice(('attack', 'skill') if rarity in {'common', 'token'}
                                    else ('attack', 'skill', 'power')))
        options = []
        for index, potion in enumerate(state.potions[:3]):
            rarity = potion_rarities.get(potion, 'common')
            rarity = 'rare' if rarity == 'event' else 'common' if rarity == 'token' else rarity
            options.append(EventOptionState(option_id=f'potion_{index}', title=potion, metadata={
                'source_native_rule': {'selection': '', 'remove_potion_index': index,
                    'generated_count': 3, 'generated_reward': True, 'generated_uniform': True,
                    'generated_rarity': rarity, 'card_type': types[index],
                    'force_generated_upgrade': True}}))
    return event.model_copy(update={'options': tuple(options), 'metadata': metadata})
