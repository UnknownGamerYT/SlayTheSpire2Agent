"""Shared stable HealRestSiteOption and PlayerCmd.MimicRestSiteHeal callbacks."""
from __future__ import annotations

from collections.abc import Mapping

from sts2sim.mechanics.combat_rewards import CombatRewardContext, draw_card_reward_options
from sts2sim.mechanics.rewards import EncounterType, RewardPityState
from sts2sim.mechanics.source_rest import source_rest_heal_amount
from sts2sim.mechanics.source_rng import draw_source_potion, is_source_rng

from .models import EffectEvent, RewardState, RunPhase, RunState
from .rng import capture_random_state, random_from_state, random_stream
from .source_reward_reroll import card_reroll_context
from .source_run_passives import append_lasting_candy_power, modify_reward_pool


def active_modifiers(state: RunState) -> tuple[str, ...]:
    raw = state.flags.get('run_modifiers', state.flags.get('modifiers', ()))
    if isinstance(raw, str):
        raw = (raw,)
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(str(item.get('id', '')) if isinstance(item, Mapping) else str(item)
                 for item in raw)


def rest_amount(state: RunState) -> int:
    return source_rest_heal_amount(state.player.max_hp, relics=state.relics,
                                   modifiers=active_modifiers(state))


def execute_rest_heal(
    state: RunState, *, is_mimicked: bool,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    """Heal, run AfterRestSiteHeal, then populate custom rewards in listener order.

    The caller controls room completion; Coffee Dripper only removes the actual
    rest-site option, so it does not gate this shared command.
    """
    from . import transitions as t

    heal_amount = rest_amount(state)
    healed = min(max(0, state.player.max_hp - state.player.hp), heal_amount)
    if t._has_relic(state, 'mark_of_the_bloom'):
        healed = 0
    player = state.player.model_copy(update={'hp': state.player.hp + healed})
    events = [EffectEvent(kind='rest_healed', amount=healed, metadata={
        'heal_amount': heal_amount, 'ascension': state.ascension, 'is_mimicked': is_mimicked,
    })]
    for owned_relic in state.relics:
        relic = owned_relic.lower()
        if relic == 'stone_humidifier':
            hp = player.hp
            player = t._change_player_max_hp(player, 5)
            if t._has_relic(state, 'mark_of_the_bloom'):
                player = player.model_copy(update={'hp': hp})
            events.append(EffectEvent(kind='relic_max_hp_changed', source_id=relic,
                                      amount=5, metadata={'is_mimicked': is_mimicked}))
    if not is_mimicked and 'night_terrors' in {m.lower() for m in active_modifiers(state)}:
        loss = min(5, player.max_hp - 1)
        player = t._change_player_max_hp(player, -loss)
        events.append(EffectEvent(kind='modifier_max_hp_lost', source_id='night_terrors',
                                  amount=loss))
    state = state.model_copy(update={'player': player, 'reward': None})
    if not t._has_relic(state, 'dream_catcher', 'tiny_mailbox'):
        return state, tuple(events)
    rng = random_from_state(state.rng)
    reward_rng = random_stream(rng, 'rewards')
    flags = dict(state.flags)
    reward = RewardState(reward_id=f'rest:{state.act}:{state.floor}', source='other',
                         metadata={'rest_heal_is_mimicked': is_mimicked})
    for owned_relic in state.relics:
        relic = owned_relic.lower()
        if relic == 'dream_catcher':
            pool = modify_reward_pool(state, t._combat_card_pool(state))
            upgrades: list[bool] = []
            cards, rarities, pity = draw_card_reward_options(reward_rng, card_pool=pool,
                context=CombatRewardContext(character_id=state.character_id,
                    encounter=EncounterType.NORMAL, act=state.act, floor=state.floor,
                    ascension_level=state.ascension, owned_relics=state.relics),
                pity_state=RewardPityState(
                    card_non_rare_count=t._flag_int(state, 'card_non_rare_count', 0),
                    card_rarity_offset=t._flag_float(state, 'card_rarity_offset', -.05)),
                card_count=3, upgraded_flags=upgrades)
            cards, rarities, upgraded = append_lasting_candy_power(
                state, rng, cards, rarities, upgrades, pool=pool)
            flags['card_non_rare_count'] = pity.card_non_rare_count
            if pity.card_rarity_offset is not None:
                flags['card_rarity_offset'] = pity.card_rarity_offset
            reward = reward.model_copy(update={'card_options': cards, 'metadata': {
                **reward.metadata, 'card_rarities': rarities,
                'card_option_upgraded': upgraded, 'card_reroll_contexts': {
                    'primary': card_reroll_context(mode='combat', count=3,
                                                   encounter='normal')},
            }})
        elif relic == 'tiny_mailbox':
            potion_pool, rarities_by_id = t._item_potion_generation_pool(state, in_combat=False)
            if not is_source_rng(reward_rng) and state.flags.get('event_reward_potion_pool'):
                potion_pool = t._reward_potion_pool(state, 'event_reward_potion_pool')
            # These are separate PotionReward.Populate calls: duplicates are allowed.
            potions = tuple(draw_source_potion(reward_rng, potion_pool, rarities_by_id)
                            for _ in range(2))
            reward = reward.model_copy(update={'potion_ids': reward.potion_ids + potions})
    if not reward.card_options and not reward.potion_ids:
        reward_state = state.model_copy(update={'flags': flags, 'rng': capture_random_state(rng)})
        return reward_state, tuple(events)
    state = state.model_copy(update={'phase': RunPhase.REWARD, 'reward': reward,
                                     'flags': flags, 'rng': capture_random_state(rng)})
    return state, tuple(events) + t._reward_generated_events(reward)


def mimic_dense_vegetation_rest(state: RunState) -> tuple[RunState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    assert state.event is not None
    event = t._source_event_next_page(state, state.event, 'REST')
    state, events = execute_rest_heal(state.model_copy(update={'event': event}), is_mimicked=True)
    if state.reward is not None:
        state = state.model_copy(update={'reward': state.reward.model_copy(update={'metadata': {
            **state.reward.metadata, 'rest_heal_return_phase': RunPhase.EVENT.value,
        }})})
    return state, (EffectEvent(kind='event_option_chosen', source_id=event.event_id,
                              target_id='rest', metadata={'native_source': True}),) + events
