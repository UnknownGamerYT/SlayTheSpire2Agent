"""Source reward population: base set, ordered room extras, then hook additions.

Reward display grouping is independent of the order in which factories consume
RNG. In particular Lava Rock is not part of the encounter's base RelicReward.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from random import Random
from typing import Any

from sts2sim.mechanics.combat_rewards import (
    DEFAULT_COMBAT_REWARD_RULES,
    CombatRewardBundle,
    CombatRewardContext,
    _draw_relics,
    _roll_gold,
    draw_card_reward_options,
    roll_combat_potion_drop,
)
from sts2sim.mechanics.relic_bag_runtime import draw_source_relic, has_source_relic_bag
from sts2sim.mechanics.rewards import EncounterType, RewardPityState
from sts2sim.mechanics.source_rng import draw_source_potion

from . import source_run_passives
from .models import EffectEvent, RewardState, RunState
from .rng import random_stream
from .source_card_reward_modifiers import modify_generated_cards
from .source_extra_rewards import KEY, extra_rewards
from .source_reward_reroll import card_reroll_context
from .source_tutorial_rewards import (
    draw_tutorial_bundle,
    tutorial_reward_metadata,
    tutorial_reward_spec,
)


def build_source_combat_rewards(
    state: RunState, rng: Random,
) -> tuple[RewardState, tuple[EffectEvent, ...], dict[str, Any]]:
    from . import transitions as t

    rng = random_stream(rng, 'rewards')
    encounter = t._combat_reward_encounter(state)
    flags = dict(state.flags)
    final_boss = t._source_final_act_boss_reward(state, encounter)
    # WithRewardsFromRoom returns before adding the base/extras at a final
    # boss; GenerateWithoutOffering still invokes ModifyRewards afterwards.
    context = CombatRewardContext(character_id=state.character_id, encounter=encounter,
        act=state.act, floor=state.floor, ascension_level=state.ascension,
        owned_relics=state.relics, gold_proportion=t._combat_gold_proportion(state))
    pity = RewardPityState(card_non_rare_count=t._flag_int(state, 'card_non_rare_count', 0),
        potion_chance_bonus=t._flag_int(state, 'potion_chance_bonus', 0),
        card_rarity_offset=t._flag_float(state, 'card_rarity_offset', -.05),
        potion_current_value=t._flag_float(state, 'potion_reward_odds', .4))
    trigger = t._combat_reward_generation_result(state, encounter,
        card_count=t._combat_reward_card_count(state, encounter),
        relic_count=t._combat_reward_relic_count(state, encounter))
    effects = tuple(effect for effect in trigger.effects
                    if t._normalized_id(effect.content_id) != 'white_beast_statue')
    queue = () if final_boss else extra_rewards(state)
    base_gold = 0
    order: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    relics: list[str] = []
    relic_rarities: list[str] = []
    potions: list[str] = []
    base_potion_id: str | None = None
    extra_potions: list[str] = []
    gold_components: list[dict[str, Any]] = []
    modifier_events: list[EffectEvent] = []

    def current() -> RunState:
        return state.model_copy(update={'flags': flags})

    def sync_pity() -> None:
        flags['card_non_rare_count'] = pity.card_non_rare_count
        flags['potion_chance_bonus'] = pity.potion_chance_bonus
        if pity.card_rarity_offset is not None:
            flags['card_rarity_offset'] = pity.card_rarity_offset
        if pity.potion_current_value is not None:
            flags['potion_reward_odds'] = pity.potion_current_value

    def card_group(profile: EncounterType, *, source_id: str | None,
                   count: int = 3, fixed: CombatRewardBundle | None = None) -> None:
        nonlocal pity, flags
        if fixed is None:
            upgrades: list[bool] = []
            pool = source_run_passives.modify_reward_pool(current(), t._combat_card_pool(state))
            ids, rarities, pity = draw_card_reward_options(rng, card_pool=pool,
                context=replace(context, encounter=profile), pity_state=pity,
                card_count=count, upgraded_flags=upgrades)
            ids, rarities, upgraded = source_run_passives.append_lasting_candy_power(
                current(), rng, ids, rarities, upgrades, pool=pool, encounter=profile.value)
            creation = card_reroll_context(mode='combat', pools=(state.character_id.lower(),),
                count=count, encounter=profile.value)
        else:
            ids, rarities, upgraded = fixed.card_ids, fixed.card_rarities, fixed.card_upgraded
            spec = tutorial_reward_spec(state, profile)
            assert spec is not None
            creation = dict(tutorial_reward_metadata(spec)['card_reroll_contexts']['primary'])
        if not ids:
            return
        key = ('primary' if source_id is None else
               f"group:{sum(group['source_id'] is not None for group in groups)}")
        creation['group_key'] = key
        sync_pity()
        modified_state, upgraded, modifiers, events = modify_generated_cards(
            current(), rng, ids, upgraded, is_card_reward=True, context=creation)
        flags = dict(modified_state.flags)
        groups.append({'ids': ids, 'rarities': rarities, 'upgraded': upgraded,
                       'context': creation, 'modifiers': modifiers, 'source_id': source_id,
                       'key': key})
        modifier_events.extend(events)
        order.append({'kind': 'card', 'source_id': source_id, 'group': key})

    def relic_reward(*, source_id: str | None, fixed: str | None = None) -> None:
        ids: tuple[str, ...]
        rarities: tuple[str, ...]
        if fixed is not None:
            ids, rarities = (fixed,), ('fixed',)
        else:
            ids, rarities = _draw_relics(rng, t._treasure_relic_pool(current()),
                t._boss_relic_pool(current()), context, relic_count=1,
                relic_selector=(lambda rarity: draw_source_relic(current(), flags, rarity))
                if has_source_relic_bag(current()) else None)
        relics.extend(ids)
        relic_rarities.extend(rarities)
        order.append({'kind': 'relic', 'source_id': source_id, 'relic_ids': ids})

    def potion_reward(*, source_id: str | None, fixed: str | None = None) -> None:
        nonlocal base_potion_id
        identifier = fixed or draw_source_potion(
            rng, t._combat_potion_pool(state), t._combat_potion_rarities(state))
        potions.append(identifier)
        if source_id is None:
            base_potion_id = identifier
        else:
            extra_potions.append(identifier)
        order.append({'kind': 'potion', 'source_id': source_id, 'potion_id': identifier})

    def fixed_gold(component: Mapping[str, Any], *, populate: bool) -> None:
        amount = int(component.get('amount', 0))
        if populate:
            rng.randrange(amount, amount + 1)
        if amount > 0:
            gold_components.append(dict(component))
        order.append({'kind': 'gold', 'source_id': component.get('source_id'), 'amount': amount,
                      'populate_invoked': populate})

    spec = None if final_boss else tutorial_reward_spec(state, encounter)
    potion_roll = None
    if final_boss:
        pass
    elif spec is None:
        # GenerateRewardsFor decides whether a PotionReward exists before
        # GenerateWithoutOffering populates Gold/Potion/Card/Relic in list order.
        potion_roll = roll_combat_potion_drop(rng, pity, context)
        pity = potion_roll.state
        if ('combat_reward_gold' in state.flags
                and state.flags.get('combat_reward_gold_is_base')):
            amount = t._flag_int(state, 'combat_reward_gold', 0)
            base_gold = rng.randrange(amount, amount + 1)
        else:
            base_gold = _roll_gold(rng, context, DEFAULT_COMBAT_REWARD_RULES)
        displayed_gold = t._flag_int(state, 'combat_reward_gold', base_gold)
        if displayed_gold > 0:
            gold_components.append({'amount': displayed_gold, 'source_id': None})
            order.append({'kind': 'gold', 'source_id': None, 'amount': displayed_gold,
                          'populate_invoked': True})
        if potion_roll.dropped and t._combat_potion_pool(state):
            potion_reward(source_id=None)
        card_group(encounter, source_id=None, count=trigger.card_choice_count)
        # Black Star appends a RelicReward during ModifyRewards; it is not
        # part of GenerateRewardsFor's initial elite reward.
        base_relic_count = t._combat_reward_relic_count(state.model_copy(update={
            'relics': tuple(identifier for identifier in state.relics
                            if t._normalized_id(identifier) != 'black_star')}), encounter)
        if state.flags.get('event_fight_id') == 'punch_off':
            base_relic_count = 0
        if base_relic_count is None:
            base_relic_count = 1 if encounter is EncounterType.ELITE else 0
        for _ in range(base_relic_count):
            relic_reward(source_id=None)
    else:
        fixed = draw_tutorial_bundle(state, rng, spec, pity)
        base_gold = fixed.base_gold
        gold_components.append({'amount': fixed.gold, 'source_id': None})
        order.append({'kind': 'gold', 'source_id': None, 'amount': fixed.gold})
        if fixed.potion_id:
            potion_reward(source_id=None, fixed=fixed.potion_id)
        # Tutorial elite constructors add their fixed relic before the
        # manual CardReward; other tutorial sets have no relic.
        for identifier in fixed.relic_ids:
            relic_reward(source_id=None, fixed=identifier)
        card_group(encounter, source_id=None, fixed=fixed)

    # CombatRoom.ExtraRewards exists before ModifyRewards: event parameters,
    # BeforeDeath gold, The Hunt and combat-end powers preserve their enqueue order.
    for extra in queue:
        kind = extra.get('kind')
        source_id = str(extra.get('source_id', 'extra'))
        if kind == 'gold':
            fixed_gold(extra, populate=True)
        elif kind == 'card':
            raw_profile = str(extra.get('encounter', encounter.value))
            profile = (EncounterType(raw_profile)
                       if raw_profile in {'normal', 'elite', 'boss'} else encounter)
            card_group(profile, source_id=source_id, count=t._card_reward_choice_count(state))
        elif kind == 'relic':
            relic_reward(source_id=source_id, fixed=extra.get('relic_id'))
        elif kind == 'potion':
            potion_reward(source_id=source_id, fixed=extra.get('potion_id'))
        elif kind in {'special_card', 'remove_card'}:
            order.append(dict(extra))

    # TryModifyRewards appends in acquired listener order; every appended
    # unpopulated object is populated afterwards, in that same append order.
    added: list[dict[str, Any]] = []
    for raw_relic in state.relics:
        relic = t._normalized_id(raw_relic)
        if (relic == 'amethyst_aubergine' and not final_boss and not (
                'combat_reward_gold' in state.flags
                and not state.flags.get('combat_reward_gold_is_base'))):
            added.append({'kind': 'gold', 'amount': 15, 'source_id': relic})
        elif relic == 'prayer_wheel' and encounter is EncounterType.NORMAL:
            added.append({'kind': 'card', 'source_id': relic, 'encounter': encounter})
        elif relic == 'black_star' and encounter is EncounterType.ELITE:
            added.append({'kind': 'relic', 'source_id': relic})
        elif relic == 'white_star' and encounter is EncounterType.ELITE:
            added.append({'kind': 'card', 'source_id': relic, 'encounter': EncounterType.BOSS})
        elif (relic == 'lava_rock' and encounter is EncounterType.BOSS and state.act == 1
              and not flags.get('lava_rock_triggered')):
            added.extend({'kind': 'relic', 'source_id': relic} for _ in range(2))
            flags['lava_rock_triggered'] = True
        elif (relic == 'wongos_mystery_ticket' and not flags.get(
                'wongos_mystery_ticket_used_up')
              and t._relic_counters_from_flags(current()).get(relic, 5) <= 0):
            added.extend({'kind': 'relic', 'source_id': relic} for _ in range(3))
    for extra in added:
        if extra['kind'] == 'gold':
            fixed_gold(extra, populate=False)
        elif extra['kind'] == 'card':
            card_group(extra['encounter'], source_id=extra['source_id'],
                       count=t._card_reward_choice_count(state))
        else:
            relic_reward(source_id=extra['source_id'])

    if any(extra['source_id'] == 'wongos_mystery_ticket' for extra in added):
        flags['wongos_mystery_ticket_used_up'] = True
    # Older snapshots represent the ticket using the shared delayed-event
    # queue. Migrate that descriptor here so its factory calls occur in its
    # acquired relic hook position and cannot be repeated by the legacy tail.
    if t._has_relic(state, 'wongos_mystery_ticket'):
        pending = []
        ticket_remaining = max(0, t._relic_counters_from_flags(current()).get(
            'wongos_mystery_ticket', 5))
        for entry in t._delayed_event_rewards_from_flags(flags):
            if entry.get('metadata', {}).get('source_relic_id') != 'wongos_mystery_ticket':
                pending.append(entry)
                continue
            ticked = {**entry, 'remaining_combats': ticket_remaining}
            ready = bool(flags.get('wongos_mystery_ticket_used_up'))
            if not ready:
                pending.append(ticked)
            modifier_events.append(EffectEvent(
                kind='delayed_event_reward_ready' if ready else 'delayed_event_reward_pending',
                amount=3 if ready else ticket_remaining,
                metadata={**t._delayed_reward_event_metadata(ticked),
                          **({'relic_ids': tuple(identifier for record in order
                              if record.get('source_id') == 'wongos_mystery_ticket'
                              for identifier in record.get('relic_ids', ()))} if ready else {})},
            ))
        flags = t._write_delayed_event_rewards(flags, pending)
    if not final_boss:
        sync_pity()
    primary = next((group for group in groups if group['key'] == 'primary'), {})
    additional = [group for group in groups if group['key'] != 'primary']
    metadata: dict[str, Any] = {
        'encounter': encounter.value, 'base_gold': base_gold,
        **({'final_act_boss': True} if final_boss else {}),
        'card_rarities': primary.get('rarities', ()),
        'card_option_upgraded': primary.get('upgraded', ()),
        'card_group_rarities': tuple(group['rarities'] for group in additional),
        'card_option_group_upgraded': tuple(group['upgraded'] for group in additional),
        'card_reroll_contexts': {group['key']: group['context'] for group in groups},
        'card_reward_option_modifiers': {group['key']: group['modifiers'] for group in groups},
        'source_modified_card_reward_groups': tuple(group['key'] for group in groups),
        'source_card_reward_modified_relics': {group['key']: tuple(state.relics)
                                              for group in groups},
        'relic_rarities': tuple(relic_rarities),
        'white_star_card_reward_group_indices': tuple(
            i for i, group in enumerate(additional) if group['source_id'] == 'white_star'),
        'gold_reward_components': tuple(gold_components),
        'extra_potion_ids': tuple(extra_potions),
        'potion_slots': t._potion_capacity(state), 'current_potions': len(state.potions),
        'reward_trigger_effects': t._trigger_effect_metadata(effects),
        'source_reward_population_order': tuple(order),
        'source_extra_reward_order': 'recorded' if state.combat is not None and (
            KEY in state.combat.metadata) else 'legacy_descriptor_order',
    }
    if spec is not None:
        metadata['tutorial_rewards'] = tutorial_reward_metadata(spec)['tutorial_rewards']
    provisional_bundle = CombatRewardBundle(potion_roll=potion_roll)
    potion_id = t._combat_reward_potion_id(state, rng,
        default_potion_id=base_potion_id)
    metadata['potion_drop'] = t._combat_potion_roll_metadata(
        state, provisional_bundle, potion_id=potion_id)
    _, escape = t._gold_reward_after_fat_gremlin_escape(state, 100)
    escape.pop('gold_before_fat_gremlin_escape', None)
    escape.pop('gold_before_encounter_escape', None)
    if escape:
        escape['gold_proportion_applied_to_range'] = True
        metadata.update(escape)
    # Custom fixture overrides are explicit; native event extras were already
    # populated in their recorded source position above.
    explicit_relic_ids = t._combat_reward_relic_ids(state, default_relic_ids=())
    if not t._is_fake_merchant_combat(state):
        relics = list(explicit_relic_ids) + relics
    explicit_potions = t._flag_str_sequence(state, 'combat_reward_potion_ids')
    remaining_potions = tuple(extra_potions)
    if explicit_potions:
        remaining_potions = explicit_potions
    elif state.flags.get('event_fight_id') != 'punch_off':
        remaining_potions += t._reward_potion_ids(state, rng,
            explicit_key='combat_reward_potion_ids', count_key='combat_reward_extra_potion_count',
            pool_key='combat_reward_potion_pool')
    metadata['extra_potion_ids'] = remaining_potions
    reward = RewardState(reward_id=f'combat:{len(state.room_history)}', source='combat',
        forced=t._flag_bool(state, 'combat_reward_forced', False),
        gold=sum(int(component['amount']) for component in gold_components),
        relic_id=t._explicit_reward_id(state, 'combat_reward_relic_id'), relic_ids=tuple(relics),
        card_ids=t._fixed_reward_card_ids(state, 'combat_reward_card_ids'),
        card_options=t._flag_str_sequence(state, 'combat_reward_card_options')
                     or tuple(primary.get('ids', ())),
        card_option_groups=tuple(group['ids'] for group in additional),
        potion_id=potion_id, potion_ids=remaining_potions, metadata=metadata)
    return reward, t._reward_generated_events(reward) + t._trigger_effect_events(effects) + tuple(
        modifier_events), flags
