"""Driftwood/CardReward one-use rerolls with serialized creation contexts."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sts2sim.mechanics.combat_rewards import CombatRewardContext, draw_card_reward_options
from sts2sim.mechanics.event_flows import EventFlowMarker, EventFlowMarkerKind
from sts2sim.mechanics.rewards import EncounterType, RewardPityState
from sts2sim.mechanics.source_card_factory import source_noncombat_cards

from .models import Action, ActionType, EffectEvent, RewardState, RunState
from .rng import capture_random_state, random_from_state, random_stream


def card_reroll_context(
    *, mode: str = 'noncombat', pools: Sequence[str] = ('character',), count: int = 3,
    **options: Any,
) -> dict[str, Any]:
    """Serializable CardCreationOptions subset; explicit manual offers use this too."""
    context = {'mode': mode, 'pools': tuple(pools), 'count': count, **options}
    if options.get('unsupported_reason'):
        context['availability'] = 'unsupported'
    return context


def _group_ids(reward: RewardState, key: str) -> tuple[str, ...]:
    if key == 'primary':
        return reward.card_options if not reward.card_claimed and not reward.card_skipped else ()
    if key.startswith('group:'):
        index = int(key.split(':', 1)[1])
        if (0 <= index < len(reward.card_option_groups)
                and index not in reward.claimed_card_option_group_indices
                and index not in reward.skipped_card_option_group_indices):
            return reward.card_option_groups[index]
    return ()


def reward_reroll_contexts(reward: RewardState) -> dict[str, Mapping[str, Any]]:
    raw = reward.metadata.get('card_reroll_contexts', {})
    contexts = ({str(key): value for key, value in raw.items() if isinstance(value, Mapping)}
                if isinstance(raw, Mapping) else {})
    # Older combat snapshots have sufficient persisted source/encounter information.
    if reward.source == 'combat':
        for key in ('primary', *(f'group:{index}' for index in
                                 range(len(reward.card_option_groups)))):
            if (key not in contexts and _group_ids(reward, key)
                    and not (key.startswith('group:') and key.split(':', 1)[1]
                             in reward.metadata.get('returned_card_option_specs', {}))):
                contexts[key] = card_reroll_context(
                    mode='combat', count=len(_group_ids(reward, key)),
                    encounter=reward.metadata.get('encounter', 'normal'))
    return contexts


def legal_reward_rerolls(state: RunState) -> tuple[Action, ...]:
    reward = state.reward
    if reward is None or 'driftwood' not in state.relics:
        return ()
    used = reward.metadata.get('card_reward_rerolled_groups', ())
    return tuple(Action(type=ActionType.REROLL_REWARD, target_id=f'reward:reroll:{key}')
                 for key, context in reward_reroll_contexts(reward).items()
                 if key not in used and _group_ids(reward, key)
                 and not context.get('unsupported_reason'))


def _pool(state: RunState, context: Mapping[str, Any]) -> tuple[Any, ...]:
    from . import transitions as e

    result = []
    seen: set[str] = set()
    for name in context.get('pools', ('character',)):
        marker = EventFlowMarker(kind=EventFlowMarkerKind.CARD_REWARD, qualifier=str(name))
        for card in e._event_flow_card_pool(state, marker):
            if card.card_id not in seen:
                result.append(card)
                seen.add(card.card_id)
    allowed = context.get('card_ids')
    if allowed is not None:
        library = e._card_library({'cards': e._raw_card_pool_items(state)})
        # Manual custom pools can contain cards outside the regular character registry.
        raw = tuple(library[str(card)] for card in allowed if str(card) in library)
        from sts2sim.mechanics.combat_rewards import build_combat_card_pool
        result = list(build_combat_card_pool(tuple({**row, 'color': 'shared', 'pool': 'shared'}
            for row in raw), character_id=state.character_id))
    from . import source_run_passives
    result = list(source_run_passives.modify_reward_pool(
        state, result, is_card_reward=bool(context.get('is_card_reward', True)),
        no_pool_modifications=bool(context.get('no_pool_modifications')),
        custom_pool=allowed is not None,
    ))
    if context.get('rarity'):
        result = [card for card in result if card.rarity.value == context['rarity']]
    if context.get('card_type'):
        result = [card for card in result if str(e._reward_card_spec(state, card.card_id).get(
            'type', '')).lower() == context['card_type']]
    if context.get('card_cost') is not None:
        result = [card for card in result if e._reward_card_spec(state, card.card_id).get(
            'cost') == context['card_cost']]
    return tuple(result)



def reroll_card_reward(state: RunState, action: Action) -> tuple[RunState, tuple[EffectEvent, ...]]:
    reward = state.reward
    if reward is None or action not in legal_reward_rerolls(state) or action.target_id is None:
        return state, ()
    key = action.target_id.removeprefix('reward:reroll:')
    context = reward_reroll_contexts(reward)[key]
    previous = _group_ids(reward, key)
    count = int(context.get('count', len(previous)))
    rng = random_from_state(state.rng)
    flags = dict(state.flags)
    if context.get('mode') == 'combat':
        encounter = EncounterType(str(context.get('encounter', 'normal')))
        upgrades: list[bool] = []
        cards, rarities, pity = draw_card_reward_options(random_stream(rng, 'rewards'),
            card_pool=_pool(state, context), context=CombatRewardContext(
                character_id=state.character_id, encounter=encounter, act=state.act,
                floor=state.floor, ascension_level=state.ascension, owned_relics=state.relics),
            pity_state=RewardPityState(
                card_non_rare_count=int(flags.get('card_non_rare_count', 0)),
                card_rarity_offset=float(flags.get('card_rarity_offset', -.05))),
            card_count=count, upgraded_flags=upgrades)
        flags['card_non_rare_count'] = pity.card_non_rare_count
        if pity.card_rarity_offset is not None:
            flags['card_rarity_offset'] = pity.card_rarity_offset
        from . import source_run_passives
        cards, rarities, upgraded = source_run_passives.append_lasting_candy_power(
            state, rng, cards, rarities, upgrades, pool=_pool(state, context),
            encounter=encounter.value,
        )
    else:
        cards, rarities, upgraded = source_noncombat_cards(rng, _pool(state, context), count,
            uniform=bool(context.get('uniform')), no_upgrade_roll=bool(context.get(
                'no_upgrade_roll', True)), act=state.act, ascension=state.ascension,
            item_rng=random_stream(rng, str(context.get('rng_stream', 'rewards'))))
    if context.get('force_upgrade'):
        upgraded = (True,) * len(cards)
    metadata = dict(reward.metadata)
    metadata['card_reward_rerolled_groups'] = tuple(dict.fromkeys((
        *metadata.get('card_reward_rerolled_groups', ()), key)))
    contexts = dict(reward_reroll_contexts(reward))
    metadata['card_reroll_contexts'] = contexts
    enchantments = dict(metadata.get('card_reward_enchantments', {}))
    enchantments.pop(key, None)
    metadata['card_reward_enchantments'] = enchantments
    if key == 'primary':
        metadata['card_rarities'] = rarities
        metadata['card_option_upgraded'] = upgraded
        reward = reward.model_copy(update={'card_options': cards, 'metadata': metadata})
    else:
        index = int(key.split(':', 1)[1])
        groups = list(reward.card_option_groups)
        groups[index] = cards
        upgrade_groups = list(metadata.get('card_option_group_upgraded', ()))
        while len(upgrade_groups) <= index:
            upgrade_groups.append(())
        upgrade_groups[index] = upgraded
        metadata['card_option_group_upgraded'] = tuple(upgrade_groups)
        rarity_groups = list(metadata.get('card_group_rarities', ()))
        # Combat groups store extras; _reward_with_card_option_groups stores primary+extras.
        position = index + (1 if len(rarity_groups) > len(groups) else 0)
        while len(rarity_groups) <= position:
            rarity_groups.append(())
        rarity_groups[position] = rarities
        metadata['card_group_rarities'] = tuple(rarity_groups)
        reward = reward.model_copy(update={
            'card_option_groups': tuple(groups), 'metadata': metadata})
    next_state = state.model_copy(update={'reward': reward, 'flags': flags,
                                         'rng': capture_random_state(rng)})
    from .source_card_reward_modifiers import apply_card_reward_modifiers

    next_state, modifier_events = apply_card_reward_modifiers(
        next_state, force=True, group_keys=(key,),
    )
    return next_state, (EffectEvent(kind='card_reward_rerolled', source_id='driftwood',
        target_id=key, amount=len(cards), metadata={'previous_card_ids': previous,
            'card_ids': cards, 'card_rarities': rarities, 'card_upgraded': upgraded}),
    ) + modifier_events
