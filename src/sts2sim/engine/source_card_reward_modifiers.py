"""Stable per-factory card modifications, keeping upgrade/enchantment provenance.

CardCreationFlags.NoModifyHooks and NoCardModelModifications are independent.
The latter does not disable Silver Crucible, Eggs, Wing Charm or Silken Tress.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from random import Random
from typing import Any

from .models import CardInstance, EffectEvent, RewardState, RunState
from .rng import capture_random_state, random_from_state, random_stream
from .source_enchantment_hooks import can_enchant


def _current_card(
    state: RunState, identifier: str, upgraded: bool, modifiers: Mapping[str, Any],
) -> CardInstance:
    from . import transitions as t

    card = t._card_from_spec(t._reward_card_spec(state, identifier), 0)
    if upgraded:
        card = t._upgrade_card_instance(card)
    for raw in modifiers.get('enchantments', ()):
        if isinstance(raw, Mapping):
            card = t._enchant_card(card, str(raw.get('keyword', '')),
                int(raw.get('amount', 1)), source_id=str(raw.get('source_id', 'reward')))
    return card


def modify_generated_cards(
    state: RunState, rng: Random, ids: Sequence[str], upgraded: Sequence[bool], *,
    is_card_reward: bool, context: Mapping[str, Any] | None = None,
    current_modifiers: Mapping[str, Any] | None = None,
    relic_ids: Sequence[str] | None = None,
) -> tuple[RunState, tuple[bool, ...], dict[str, Any], tuple[EffectEvent, ...]]:
    """Run one CreateForReward/Populate late-hook batch, in acquired relic order.

    A group of cards may originate from several factory calls. Callers preserve
    that boundary instead of inferring it from how the options are displayed.
    """
    from . import transitions as t

    options = dict(context or {})
    flags = dict(state.flags)
    upgrades = [bool(upgraded[i]) if i < len(upgraded) else False for i in range(len(ids))]
    modifiers = {str(k): dict(v) for k, v in (current_modifiers or {}).items()
                 if isinstance(v, Mapping)}
    if options.get('no_modify_hooks'):
        return state, tuple(upgrades), modifiers, ()
    events: list[EffectEvent] = []
    group_key = str(options.get('group_key', 'generated'))
    for raw_relic in state.relics if relic_ids is None else relic_ids:
        relic = t._normalized_id(raw_relic)
        changed: list[int] = []
        if relic == 'silver_crucible' and is_card_reward:
            used = int(flags.get('silver_crucible_times_used', 0))
            if used >= 3:
                continue
            for index, identifier in enumerate(ids):
                card = _current_card(
                    state, identifier, upgrades[index], modifiers.get(str(index), {}),
                )
                if t._card_is_upgradable(card):
                    upgrades[index] = True
                    changed.append(index)
            flags['silver_crucible_times_used'] = used + 1
            counters = dict(t._mapping_from(flags.get('relic_counters')))
            counters['silver_crucible'] = 2 - used
            flags['relic_counters'] = counters
        elif relic in {'molten_egg', 'toxic_egg', 'frozen_egg'}:
            if options.get('no_hook_upgrades'):
                continue
            kind = {'molten_egg': 'attack', 'toxic_egg': 'skill', 'frozen_egg': 'power'}[relic]
            for index, identifier in enumerate(ids):
                card = _current_card(
                    state, identifier, upgrades[index], modifiers.get(str(index), {}),
                )
                if card.type.value == kind and t._card_is_upgradable(card):
                    upgrades[index] = True
                    changed.append(index)
        elif relic in {'glitter', 'lava_lamp'}:
            from .source_run_passives import apply_late_group_relic, in_combat_room

            before = (tuple(upgrades), dict(modifiers))
            result_upgrades, modifiers = apply_late_group_relic(
                state.model_copy(update={'flags': flags}), relic, ids, upgrades,
                group_key=group_key, option_modifiers=modifiers, context=options,
            )
            upgrades = list(result_upgrades)
            changed = [i for i in range(len(ids))
                       if upgrades[i] != before[0][i]
                       or modifiers.get(str(i)) != before[1].get(str(i))]
        elif relic in {'wing_charm', 'silken_tress', 'fresnel_lens'}:
            if relic == 'silken_tress' and (
                not is_card_reward or flags.get('silken_tress_reward_used')
            ):
                continue
            keyword = {'wing_charm': 'swift', 'silken_tress': 'glam',
                       'fresnel_lens': 'nimble'}[relic]
            amount = 2 if relic == 'fresnel_lens' else 1
            eligible = [i for i, identifier in enumerate(ids)
                        if can_enchant(_current_card(state, identifier, upgrades[i],
                                                     modifiers.get(str(i), {})), keyword)]
            selected = ([random_stream(rng, 'niche').choice(eligible)]
                        if relic == 'wing_charm' and eligible else eligible)
            for index in selected:
                entry = dict(modifiers.get(str(index), {}))
                entry['enchantments'] = (*entry.get('enchantments', ()),
                                        {'keyword': keyword, 'amount': amount, 'source_id': relic})
                modifiers[str(index)] = entry
                changed.append(index)
            if relic == 'silken_tress':
                # Its hook returns true even when no card can receive Glam.
                flags['silken_tress_reward_used'] = True
        else:
            continue
        events.append(EffectEvent(
            kind='relic_card_reward_modified', source_id=relic, amount=len(changed),
            metadata={'card_reward_group': group_key, 'card_indices': changed,
                      'is_card_reward': is_card_reward,
                      **({'in_combat_room': in_combat_room(state, options)}
                         if relic == 'lava_lamp' else {})},
        ))
    return state.model_copy(update={'flags': flags}), tuple(upgrades), modifiers, tuple(events)


def _groups(reward: RewardState) -> tuple[tuple[str, tuple[str, ...]], ...]:
    groups = []
    if reward.card_options and not reward.card_claimed and not reward.card_skipped:
        groups.append(('primary', reward.card_options))
    excluded = {
        *reward.claimed_card_option_group_indices, *reward.skipped_card_option_group_indices,
    }
    groups.extend((f'group:{i}', ids) for i, ids in enumerate(reward.card_option_groups)
                  if ids and i not in excluded)
    return tuple(groups)


def apply_card_reward_modifiers(
    state: RunState, *, force: bool = False, group_keys: Sequence[str] | None = None,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    """Apply newly populated reward groups once; rerolls explicitly force their group."""
    from . import transitions as t

    reward = state.reward
    if reward is None:
        return state, ()
    metadata = dict(reward.metadata)
    applied = set(t._string_items(metadata.get('source_modified_card_reward_groups')))
    contexts = t._mapping_from(metadata.get('card_reroll_contexts'))
    option_mods = dict(t._mapping_from(metadata.get('card_reward_option_modifiers')))
    prior_relics = dict(t._mapping_from(metadata.get('source_card_reward_modified_relics')))
    rng = random_from_state(state.rng)
    events: list[EffectEvent] = []
    primary_upgrades = tuple(bool(v) for v in metadata.get('card_option_upgraded', ()))
    group_upgrades = [tuple(bool(v) for v in values)
                      for values in metadata.get('card_option_group_upgraded', ())]
    whole_group_enchants = dict(t._mapping_from(metadata.get('card_reward_enchantments')))
    for key, ids in _groups(reward):
        if group_keys is not None and key not in group_keys:
            continue
        previous = set(t._string_items(prior_relics.get(key)))
        newly_obtained = tuple(
            relic for relic in state.relics if t._normalized_id(relic) not in previous
        )
        if key in applied and not force and not newly_obtained:
            continue
        context = dict(t._mapping_from(contexts.get(key)))
        context['group_key'] = key
        if key == 'primary':
            upgrades = primary_upgrades
        else:
            index = int(key.split(':', 1)[1])
            upgrades = group_upgrades[index] if index < len(group_upgrades) else ()
        current_modifiers = {} if force else dict(t._mapping_from(option_mods.get(key)))
        legacy = whole_group_enchants.pop(key, ())
        if legacy and not force:
            for i in range(len(ids)):
                current_modifiers[str(i)] = {'enchantments': tuple(legacy)}
        state, upgrades, updated_mods, modified_events = modify_generated_cards(
            state, rng, ids, upgrades,
            is_card_reward=bool(context.get('is_card_reward', True)),
            context=context, current_modifiers=current_modifiers,
            relic_ids=None if force or key not in applied else newly_obtained,
        )
        option_mods[key] = updated_mods
        if key == 'primary':
            primary_upgrades = upgrades
        else:
            index = int(key.split(':', 1)[1])
            while len(group_upgrades) <= index:
                group_upgrades.append(())
            group_upgrades[index] = upgrades
        applied.add(key)
        prior_relics[key] = tuple(t._normalized_id(relic) for relic in state.relics)
        events.extend(modified_events)
    metadata.update(source_modified_card_reward_groups=tuple(sorted(applied)),
                    source_card_reward_modified_relics=prior_relics,
                    card_reward_option_modifiers=option_mods,
                    card_option_upgraded=primary_upgrades,
                    card_option_group_upgraded=tuple(group_upgrades))
    if whole_group_enchants:
        metadata['card_reward_enchantments'] = whole_group_enchants
    else:
        metadata.pop('card_reward_enchantments', None)
    reward = reward.model_copy(update={'metadata': metadata})
    return state.model_copy(update={
        'reward': reward, 'rng': capture_random_state(rng),
    }), tuple(events)


def apply_option_modifiers(
    card: CardInstance, reward: RewardState, key: str, index: int,
) -> CardInstance:
    from . import transitions as t

    by_group = t._mapping_from(reward.metadata.get('card_reward_option_modifiers'))
    by_index = t._mapping_from(by_group.get(key))
    modifiers = t._mapping_from(by_index.get(str(index)))
    return apply_generated_modifiers(card, modifiers)


def apply_generated_modifiers(
    card: CardInstance, modifiers: Mapping[str, Any],
) -> CardInstance:
    """Apply persisted factory enchantments to an offered or selected instance."""
    from . import transitions as t

    for raw in modifiers.get('enchantments', ()):
        if isinstance(raw, Mapping):
            card = t._enchant_card(card, str(raw.get('keyword', '')),
                int(raw.get('amount', 1)), source_id=str(raw.get('source_id', 'reward')))
    return card
