"""Retained public actions for native stable event callback contracts."""
from __future__ import annotations

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import Action, ActionType, RoomKind
from sts2sim.engine import transitions as e
from sts2sim.mechanics.stable_pool_order import STABLE_CARD_POOL_ORDER


def _event(event_id: str):
    state = new_run('TEST', 'IRONCLAD', 0)
    assert state.map is not None
    state = state.model_copy(update={'flags': {**state.flags, 'event_id': event_id},
        'player': state.player.model_copy(update={'gold': 300, 'hp': 50, 'max_hp': 80})})
    node = state.map.nodes[0].model_copy(update={'kind': RoomKind.EVENT})
    return e._enter_event_room(state, node)[0]


def _choose(state, option_id: str, ids: tuple[str, ...] = ()):
    return step(state, Action(type=ActionType.CHOOSE_EVENT, target_id=option_id,
                             payload={'card_instance_ids': ids} if ids else {}))


def _bag(state):
    return state.model_copy(update={'flags': {**state.flags, 'player_relic_bag': {
        'Common': ('ANCHOR', 'BAG_OF_PREPARATION'),
        'Uncommon': ('KUNAI', 'INK_BOTTLE'), 'Rare': ('DU_VU_DOLL',),
    }}})


def test_amalgamator_removes_chosen_basic_cards_and_adds_ultimate_immediately() -> None:
    state = _event('AMALGAMATOR')
    ids = tuple(card.instance_id for card in state.master_deck
                if card.card_id.startswith('strike'))[:2]
    before = len(state.master_deck)
    actions = [a for a in legal_actions(state) if a.target_id == 'combine_strikes']
    assert actions and all(len(a.payload['card_instance_ids']) == 2 for a in actions)
    result = _choose(load_state(serialize(state)), 'combine_strikes', ids)
    assert len(result.master_deck) == before - 1
    assert all(card.instance_id not in ids for card in result.master_deck)
    assert result.master_deck[-1].card_id == 'ultimate_strike'
    assert result.reward is None


def test_aroma_transform_honors_model_stream_and_drops_old_upgrade() -> None:
    state = _event('AROMA_OF_CHAOS')
    first = e._upgrade_card_instance(state.master_deck[0])
    state = state.model_copy(update={'master_deck': (first, *state.master_deck[1:])})
    result = _choose(state, 'let_go', (first.instance_id,))
    assert result.master_deck[0].card_id != first.card_id
    assert not result.master_deck[0].upgraded
    assert result.rng.streams['AROMA_OF_CHAOS'].draws == 1
    assert 'transformations' not in result.rng.streams
    assert result.reward is None


def test_transform_uses_original_card_pool_including_other_character() -> None:
    state = _event('AROMA_OF_CHAOS')
    spec = e._reward_card_spec(state, 'acrobatics')
    spec['card_id'] = 'acrobatics'
    card = e._card_from_spec(spec, 1)
    state = state.model_copy(update={'master_deck': (card, *state.master_deck[1:])})
    result = _choose(state, 'let_go', (card.instance_id,))
    assert result.master_deck[0].card_id in STABLE_CARD_POOL_ORDER['silent']


def test_wood_carvings_fixed_transform_consumes_no_rng() -> None:
    state = _event('WOOD_CARVINGS')
    card = e._upgrade_card_instance(state.master_deck[0])
    state = state.model_copy(update={'master_deck': (card, *state.master_deck[1:])})
    result = _choose(state, 'bird', (card.instance_id,))
    assert result.master_deck[0].card_id == 'peck'
    assert not result.master_deck[0].upgraded
    assert result.rng.streams['WOOD_CARVINGS'].draws == 0


def test_morphic_grove_chosen_transforms_and_all_gold_loss_are_immediate() -> None:
    state = _event('MORPHIC_GROVE')
    ids = tuple(card.instance_id for card in state.master_deck[:2])
    original = {card.instance_id: card.card_id for card in state.master_deck[:2]}
    result = _choose(state, 'group', ids)
    assert result.player.gold == 0
    assert all(card.card_id != original[card.instance_id] for card in result.master_deck[:2])
    assert result.rng.streams['MORPHIC_GROVE'].draws == 2


def test_event_relic_draw_consumes_front_bag_and_rewards_rarity_once() -> None:
    state = _bag(_event('UNREST_SITE'))
    result = _choose(load_state(serialize(state)), 'kill')
    assert result.player.max_hp == 72
    assert result.player.hp == 50
    assert 'kunai' in result.relics
    assert result.flags['player_relic_bag']['Uncommon'] == ('INK_BOTTLE',)
    assert result.rng.streams['rewards'].draws == 1
    assert result.reward is None


def test_round_tea_fight_is_a_two_page_event_not_combat() -> None:
    state = _bag(_event('ROUND_TEA_PARTY'))
    state = _choose(state, 'pick_fight')
    assert state.combat is None and state.player.hp == 50
    assert state.event is not None and state.event.page_id == 'PICK_FIGHT'
    assert [a.target_id for a in legal_actions(state)] == ['continue_fight']
    result = _choose(state, 'continue_fight')
    assert result.player.hp == 39 and 'kunai' in result.relics
    assert result.combat is None


def test_zen_weaver_selected_removal_charges_exact_cost() -> None:
    state = _event('ZEN_WEAVER')
    card = state.master_deck[0]
    result = _choose(state, 'emotional_awareness', (card.instance_id,))
    assert result.player.gold == 175
    assert all(value.instance_id != card.instance_id for value in result.master_deck)
    assert result.reward is None


def test_reflections_duplicates_original_deck_then_adds_bad_luck() -> None:
    state = _event('REFLECTIONS')
    original = state.master_deck
    result = _choose(state, 'shatter')
    assert len(result.master_deck) == len(original) * 2 + 1
    assert result.master_deck[-1].card_id == 'bad_luck'
    assert len({card.instance_id for card in result.master_deck}) == len(result.master_deck)
    assert [card.card_id for card in result.master_deck[len(original):-1]] == [
        card.card_id for card in original
    ]


def test_fixed_card_and_potion_callbacks_preserve_their_source_delivery() -> None:
    nest = _event('BYRDONIS_NEST')
    result = _choose(nest, 'take')
    assert result.master_deck[-1].card_id == 'byrdonis_egg'
    assert result.reward is None
    courier = _event('POTION_COURIER')
    result = _choose(courier, 'grab_potions')
    assert result.reward is not None
    assert result.reward.potion_ids == ('foul_potion',) * 3


def test_brain_leech_grid_generates_five_once_then_adds_chosen_one():
    state = _choose(_event('BRAIN_LEECH'), 'share_knowledge')
    offers = state.event.metadata['generated_cards']
    assert len(offers) == len(set(offers)) == 5
    assert state.rng.streams['rewards'].draws == 10
    assert len(legal_actions(state)) == 5 and state.reward is None
    original = len(state.master_deck)
    result = _choose(load_state(serialize(state)), 'choose_cards', (offers[3],))
    assert len(result.master_deck) == original + 1
    assert result.master_deck[-1].card_id == offers[3]
    assert result.rng.streams['rewards'].draws == 10


def test_cheese_grid_chooses_two_of_eight_common_cards():
    state = _choose(_event('ROOM_FULL_OF_CHEESE'), 'gorge')
    assert len(legal_actions(state)) == 28
    offers = state.event.metadata['generated_cards']
    assert len(offers) == len(set(offers)) == 8
    assert state.rng.streams['rewards'].draws == 8
    result = _choose(state, 'choose_cards', offers[:2])
    assert tuple(card.card_id for card in result.master_deck[-2:]) == offers[:2]


def test_infested_automaton_adds_one_filtered_power_directly():
    state = _event('INFESTED_AUTOMATON')
    result = _choose(state, 'study')
    assert result.master_deck[-1].type.value == 'power'
    assert result.rng.streams['rewards'].draws == 2
    assert result.reward is None


def test_native_enchantment_can_target_eternal_but_rejects_existing_enchantment():
    state = _event('SYMBIOTE')
    card = state.master_deck[0].model_copy(update={'tags': ('eternal',)})
    already = e._enchant_card(state.master_deck[1], 'Sharp', 2, source_id='fixture')
    state = state.model_copy(update={'master_deck': (card, already, *state.master_deck[2:])})
    actions = [a for a in legal_actions(state) if a.target_id == 'approach']
    assert any(card.instance_id in a.payload['card_instance_ids'] for a in actions)
    assert all(already.instance_id not in a.payload['card_instance_ids'] for a in actions)
    result = _choose(state, 'approach', (card.instance_id,))
    assert result.master_deck[0].enchantments[-1].keyword == 'Corrupted'


def test_waterlogged_sponge_enchants_two_for_exact_gold_without_reward():
    state = _event('WATERLOGGED_SCRIPTORIUM')
    ids = tuple(card.instance_id for card in state.master_deck[:2])
    result = _choose(state, 'prickly_sponge', ids)
    assert result.player.gold == 201 and result.reward is None
    assert all(card.enchantments[-1].keyword == 'Steady' for card in result.master_deck[:2])


def _with_potions_event(event_id, potions):
    state = _event(event_id)
    state = state.model_copy(update={'potions': potions})
    return e._enter_event_room(state, state.map.nodes[0].model_copy(
        update={'kind': RoomKind.EVENT}))[0]


def test_stone_lift_fixes_potion_before_reload_and_consumes_unused_roll():
    state = _with_potions_event('STONE_OF_ALL_TIME', ('fire_potion', 'block_potion'))
    assert state.rng.streams['STONE_OF_ALL_TIME'].draws == 1
    option = next(o for o in state.event.options if o.option_id == 'lift')
    selected_index = option.metadata['source_native_rule']['remove_potion_index']
    selected_potion = state.potions[selected_index]
    result = _choose(load_state(serialize(state)), 'lift')
    assert len(result.potions) == 1 and selected_potion not in result.potions
    assert result.player.max_hp == 90 and result.player.hp == 60
    assert result.rng.streams['STONE_OF_ALL_TIME'].draws == 2


def test_future_of_potions_fixes_types_and_upgrades_all_three_offered_cards():
    state = _with_potions_event('THE_FUTURE_OF_POTIONS',
                               ('fire_potion', 'block_potion', 'skill_potion'))
    assert state.rng.streams['THE_FUTURE_OF_POTIONS'].draws == 3
    result = _choose(load_state(serialize(state)), 'potion_1')
    assert len(result.potions) == 2 and result.reward is not None
    assert len(result.reward.card_options) == 3
    assert result.reward.metadata['card_option_upgraded'] == (True,) * 3
    assert result.rng.streams['rewards'].draws == 3


def test_colorful_philosophers_fixes_three_other_colors_and_rolls_nine_upgrades():
    state = _event('COLORFUL_PHILOSOPHERS')
    assert len(state.event.options) == 3
    assert all(o.option_id != 'ironclad' for o in state.event.options)
    assert state.rng.streams['COLORFUL_PHILOSOPHERS'].draws == 1
    result = _choose(state, state.event.options[0].option_id)
    assert result.reward is not None
    assert len(result.reward.card_options) == 3
    assert len(result.reward.card_option_groups) == 2
    assert result.rng.streams['rewards'].draws == 18


def test_doll_room_take_time_damages_then_offers_two_stably_shuffled_relics():
    state = _choose(_event('DOLL_ROOM'), 'take_some_time')
    assert state.player.hp == 45 and len(legal_actions(state)) == 2
    assert state.rng.streams['DOLL_ROOM'].draws == 2
    assert state.reward is None
    relic = state.event.options[0].option_id
    result = _choose(load_state(serialize(state)), relic)
    assert relic in result.relics and result.event.resolved_option_id == relic


def test_ranwid_potion_offer_and_relic_reward_are_fixed_across_reload():
    state = _with_potions_event('RANWID_THE_ELDER', ('fire_potion', 'block_potion'))
    state = _bag(state.model_copy(update={'relics': ('burning_blood', 'anchor', 'kunai')}))
    state = state.model_copy(update={'flags': {**state.flags, 'player_relic_bag': {
        'Common': ('LANTERN', 'VAJRA'), 'Uncommon': ('NUNCHAKU', 'PAPER_PHROG'),
        'Rare': ('TORII',)}}})
    state = e._enter_event_room(state, state.map.nodes[0].model_copy(
        update={'kind': RoomKind.EVENT}))[0]
    assert state.rng.streams['RANWID_THE_ELDER'].draws == 2
    result = _choose(load_state(serialize(state)), 'potion')
    assert len(result.potions) == 1 and len(result.relics) == 4
    assert result.rng.streams['rewards'].draws == 1
    assert result.reward is None


def test_relic_trader_pulls_three_front_offers_on_entry_even_before_selection():
    state = _bag(_event('RELIC_TRADER').model_copy(update={'relics': (
        'burning_blood', 'anchor', 'kunai', 'sundial', 'ink_bottle', 'bird_faced_urn')}))
    state = state.model_copy(update={'flags': {**state.flags, 'player_relic_bag': {
        'Common': ('LANTERN', 'VAJRA'), 'Uncommon': ('NUNCHAKU', 'PAPER_PHROG'),
        'Rare': ('TORII',)}}})
    state = e._enter_event_room(state, state.map.nodes[0].model_copy(
        update={'kind': RoomKind.EVENT}))[0]
    assert len(state.event.options) == 3
    assert state.rng.streams['RELIC_TRADER'].draws == 4
    assert state.rng.streams['rewards'].draws == 3
    assert sum(len(ids) for ids in state.flags['player_relic_bag'].values()) == 2
    option = state.event.options[0]
    rule = option.metadata['source_native_rule']
    result = _choose(load_state(serialize(state)), option.option_id)
    assert rule['remove_relic'] not in result.relics
    assert rule['add_relics'][0] in result.relics
    assert result.rng.streams['rewards'].draws == 3


def test_wongo_featured_relic_is_consumed_on_entry_and_fixed_before_purchase():
    state = _bag(_event('WELCOME_TO_WONGOS'))
    state = state.model_copy(update={'flags': {**state.flags, 'wongo_points': 1990}})
    state = e._enter_event_room(state, state.map.nodes[0].model_copy(
        update={'kind': RoomKind.EVENT}))[0]
    assert not state.flags['player_relic_bag']['Rare']
    option = next(option for option in state.event.options if option.option_id == 'FEATURED_ITEM')
    assert option.metadata['source_native_rule']['add_relics'] == ('du_vu_doll',)
    result = _choose(load_state(serialize(state)), 'FEATURED_ITEM')
    assert result.player.gold == 100 and 'du_vu_doll' in result.relics
    assert result.flags['wongo_points'] == 2006
    assert 'wongo_customer_appreciation_badge' in result.relics
    assert result.rng.streams.get('rewards') is None or result.rng.streams['rewards'].draws == 0


def test_jungle_solo_consumes_cosmetic_stable_shuffle_of_three_effects():
    state = _event('JUNGLE_MAZE_ADVENTURE')
    assert state.rng.streams['JUNGLE_MAZE_ADVENTURE'].draws == 2
    result = _choose(state, 'solo_quest')
    assert result.rng.streams['JUNGLE_MAZE_ADVENTURE'].draws == 4
