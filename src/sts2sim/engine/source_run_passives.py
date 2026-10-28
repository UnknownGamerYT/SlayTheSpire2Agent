"""Stable reward pool/normal/late hooks for the reviewed A–M relic families.

Independent rules from source5d874e8, with scopes/hashes in the accompanying
source_run_passives fixture. These helpers do not certify full game parity.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from random import Random
from typing import Any

from sts2sim.mechanics.combat_rewards import CombatRewardCard
from sts2sim.mechanics.event_flows import EventFlowMarker, EventFlowMarkerKind
from sts2sim.mechanics.reward_pools import build_colorless_card_pool
from sts2sim.mechanics.source_ancients import source_unlocked_characters
from sts2sim.mechanics.source_rng import float32, source_float

from .errors import EngineError
from .models import CombatState, EffectEvent, RoomKind, RunPhase, RunState
from .rng import random_stream
from .source_enchantment_hooks import can_enchant


def modify_reward_pool(
    state: RunState, pool: Sequence[CombatRewardCard], *, is_card_reward: bool = True,
    no_pool_modifications: bool = False, custom_pool: bool = False,
    pool_ids: Sequence[str] | None = None,
) -> tuple[CombatRewardCard, ...]:
    """Apply source pool hooks before filtering, preserving pool registry order."""
    from . import transitions as t

    if not is_card_reward or no_pool_modifications or custom_pool:
        return tuple(pool)
    result = tuple(pool)
    names = tuple(t._normalized_id(name) for name in pool_ids) if pool_ids is not None else (
        tuple(dict.fromkeys(t._normalized_id(card.color) for card in pool))
        or (t._normalized_id(state.character_id),)
    )
    for raw_relic in state.relics:
        relic = t._normalized_id(raw_relic)
        if relic == 'dingy_rug' and 'colorless' not in names:
            rows = t._registered_reward_card_rows(state, 'colorless')
            colorless = build_colorless_card_pool(
                t._raw_card_pool_items(state) if rows is None else rows)
            result += tuple(CombatRewardCard(
                card.card_id, card.rarity, card.color, card.name, card.source_id,
            ) for card in colorless)
            names += ('colorless',)
        elif relic == 'prismatic_gem' and any(name not in {
                'colorless', 'event', 'deprecated', 'token'} for name in names):
            unlocked = source_unlocked_characters(state.flags)
            expanded = tuple(card for name in unlocked for card in t._event_flow_card_pool(
                state, EventFlowMarker(kind=EventFlowMarkerKind.CARD_REWARD, qualifier=name)))
            # Source Union prepends unlocked character pools, then existing pools.
            result = expanded + result
            names = tuple(dict.fromkeys((*unlocked, *names)))
        else:
            continue
        unique: dict[str, CombatRewardCard] = {}
        for card in result:
            unique.setdefault(card.card_id, card)
        result = tuple(unique.values())
    return result


def append_lasting_candy_power(
    state: RunState, rng: Random, card_ids: Sequence[str], rarities: Sequence[str],
    upgraded: Sequence[bool], *, pool: Sequence[CombatRewardCard],
    encounter: str = 'normal', creation_source: str = 'encounter',
    no_modify_hooks: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[bool, ...]]:
    """The extra Power is a separate Source.Other factory with unchanged pity."""
    from . import transitions as t

    count = t._relic_counters_from_flags(state).get('lasting_candy', 0)
    if state.combat is not None:
        count = t._combat_relic_counters(state.combat).get('lasting_candy', count)
    if (not t._has_relic(state, 'lasting_candy') or no_modify_hooks
        or creation_source != 'encounter' or count <= 0 or count % 2):
        return tuple(card_ids), tuple(rarities), tuple(upgraded)
    all_powers = [card for card in pool if str(t._reward_card_spec(
        state, card.card_id).get('type', '')).lower() == 'power']
    candidates = [card for card in all_powers if card.card_id not in card_ids] or all_powers
    if not candidates:
        return tuple(card_ids), tuple(rarities), tuple(upgraded)
    rewards_rng = random_stream(rng, 'rewards')
    allowed = {card.rarity.value for card in candidates}
    # Source custom-pool constructor rejects non-Uniform single-rarity pools.
    if len(allowed) == 1:
        raise EngineError("Lasting Candy source custom Power pool requires multiple rarities")
    if len(allowed) > 1:
        sample = source_float(rewards_rng)
        scarce = state.ascension >= 7
        if encounter == 'boss':
            rarity = 'rare'
        else:
            elite = encounter == 'elite'
            rare = (0.05 if scarce else .1) if elite else (.0149 if scarce else .03)
            rarity = ('rare' if sample < float32(rare) else 'uncommon'
                      if sample < float32(.4 if elite else .37)
                      else 'common')
        for _ in range(3):
            if rarity in allowed:
                break
            rarity = {'common': 'uncommon', 'uncommon': 'rare', 'rare': 'common'}[rarity]
        candidates = [card for card in candidates if card.rarity.value == rarity]
    chosen = rewards_rng.choice(candidates)
    roll = Decimal(format(source_float(rewards_rng), '.7g'))
    threshold = (Decimal(0) if chosen.rarity.value == 'rare' else
                 Decimal(max(0, state.act - 1))
                 * Decimal('.125' if state.ascension >= 7 else '.25'))
    return (tuple(card_ids) + (chosen.card_id,), tuple(rarities) + (chosen.rarity.value,),
            tuple(upgraded) + (roll <= threshold,))


def after_combat_end(
    state: RunState, combat: CombatState,
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    if not t._has_relic(state, 'lasting_candy'):
        return combat, ()
    counters = t._combat_relic_counters(combat)
    count = counters.get('lasting_candy', 0) + 1
    counters['lasting_candy'] = count
    updated = combat.model_copy(update={'metadata': {
        **combat.metadata, 'relic_counters': counters,
    }})
    return updated, (
        EffectEvent(kind='relic_counter_changed', source_id='lasting_candy', amount=count,
                    metadata={'combats_seen': count, 'display_amount': count % 2}),
    )


def observe_lava_damage(combat: CombatState, events: Sequence[EffectEvent]) -> CombatState:
    if any(event.kind in {'player_damaged', 'player_hp_lost'} and (event.amount or 0) > 0
           and not event.metadata.get('unblockable') and not event.metadata.get('lethal')
           for event in events):
        return combat.model_copy(update={'metadata': {
            **combat.metadata, 'lava_lamp_took_damage': True,
        }})
    return combat


def in_combat_room(state: RunState, context: Mapping[str, Any] | None = None) -> bool:
    """Identify the current room, independently of a nested reward's source.

    Completed CombatState objects remain archived after leaving their room.
    Event encounters temporarily replace an EventRoom until its reward tail ends.
    """
    from . import transitions as t

    if state.phase not in {RunPhase.COMBAT, RunPhase.REWARD}:
        return False
    node = t._current_map_node(state)
    event_fight = bool(state.flags.get('event_combat_restore_flags')) and bool(
        state.flags.get('event_fight_id'))
    if node is not None:
        return node.kind in {RoomKind.MONSTER, RoomKind.ELITE, RoomKind.BOSS} or (
            node.kind == RoomKind.EVENT and event_fight)
    # Standalone fixtures have no map room. Their active phase or explicit
    # creation descriptor provides room provenance, rather than an old combat.
    return (state.phase == RunPhase.COMBAT or event_fight
            or bool((context or {}).get('in_combat_room'))
            or state.reward is not None and state.reward.source == 'combat')


def apply_late_group_relic(
    state: RunState, relic_id: str, card_ids: Sequence[str], upgraded: Sequence[bool], *,
    group_key: str, option_modifiers: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> tuple[tuple[bool, ...], dict[str, Any]]:
    """Apply one Glitter/Lava Lamp late hook, allowing acquisition-order dispatch.

    option_modifiers is the per-index map for this group, using entries with an
    enchantments sequence. The caller owns persistence and one-time hook dispatch.
    """
    from . import transitions as t

    del group_key
    modifiers = {str(key): dict(value) for key, value in (option_modifiers or {}).items()
                 if isinstance(value, Mapping)}
    upgrades = tuple(upgraded) + (False,) * max(0, len(card_ids) - len(upgraded))
    if context and context.get('no_modify_hooks'):
        return upgrades, modifiers
    if relic_id == 'lava_lamp':
        if (state.combat is None or state.combat.metadata.get('lava_lamp_took_damage')
            or not in_combat_room(state, context)):
            return upgrades, modifiers
        upgrades = tuple(flag or t._card_is_upgradable(t._card_from_spec(
            t._reward_card_spec(state, card_id), i + 1))
            for i, (card_id, flag) in enumerate(zip(card_ids, upgrades, strict=True)))
    elif relic_id == 'glitter':
        for index, card_id in enumerate(card_ids):
            spec = t._reward_card_spec(state, card_id)
            spec['upgraded'] = upgrades[index]
            card = t._card_from_spec(spec, index + 1)
            existing = dict(modifiers.get(str(index), {}))
            enchantments = list(existing.get('enchantments', ()))
            for enchantment in enchantments:
                if isinstance(enchantment, Mapping):
                    card = t._enchant_card(card, str(enchantment.get('keyword', '')),
                        int(enchantment.get('amount', 1)), source_id=str(
                            enchantment.get('source_id', 'reward')))
            if can_enchant(card, 'glam', in_deck=False):
                enchantments.append({'keyword': 'glam', 'amount': 1, 'source_id': 'glitter'})
                modifiers[str(index)] = {**existing, 'enchantments': tuple(enchantments)}
    return upgrades, modifiers
