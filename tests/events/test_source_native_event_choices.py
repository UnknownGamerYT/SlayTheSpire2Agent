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
