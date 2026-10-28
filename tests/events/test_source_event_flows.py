from dataclasses import replace

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import Action, ActionType, RoomKind
from sts2sim.engine import transitions as e
from sts2sim.engine.rng import random_from_state, random_stream
from sts2sim.mechanics.event_flows import (
    current_event_flow_page,
    event_flow_state,
    resolve_event_flow_option,
    roll_source_conveyor_dish,
)


def _event(event_id, hp=50, max_hp=80):
    state = new_run('TEST', 'IRONCLAD', 0)
    state = state.model_copy(update={'flags': {**state.flags, 'event_id': event_id},
        'player': state.player.model_copy(update={'gold': 1000, 'hp': hp, 'max_hp': max_hp})})
    return e._enter_event_room(state, state.map.nodes[0].model_copy(
        update={'kind': RoomKind.EVENT}))[0]


def test_baths_damage_keeps_growing_after_ninth_page_and_gains_hp_before_damage():
    state = _event('ABYSSAL_BATHS', 500, 500)
    for n in range(15):
        option = 'IMMERSE' if n == 0 else 'LINGER'
        state = step(load_state(serialize(state)), Action(type=ActionType.CHOOSE_EVENT,
                                                        target_id=option))
    assert state.player.max_hp == 530
    assert state.player.hp == 380
    assert state.event.page_id == 'LINGER9'
    assert state.event.options[0].description == 'Gain 2 Max HP. Take 18 damage.'
    assert state.rng.streams['ABYSSAL_BATHS'].draws == 0


def test_baths_warning_next_net_damage_and_actual_lethal_soak():
    state = _event('ABYSSAL_BATHS', 5, 80)
    state = step(state, Action(type=ActionType.CHOOSE_EVENT, target_id='IMMERSE'))
    assert state.event.page_id == 'IMMERSE' and state.player.hp == 4
    state = step(state, Action(type=ActionType.CHOOSE_EVENT, target_id='LINGER'))
    assert state.event.page_id == 'DEATH_WARNING' and state.player.hp == 2
    state = step(state, Action(type=ActionType.CHOOSE_EVENT, target_id='LINGER'))
    assert state.player.hp == 0 and state.player.max_hp == 86


def test_conveyor_c_port_weighted_vectors_and_fifth_roll_consumes_no_rng():
    # Independent C harness ports NextFloat, float product and source list order.
    state = event_flow_state('ENDLESS_CONVEYOR', hp=50, max_hp=80, gold=1000,
                             data={'source_runtime': True})
    rng = random_stream(random_from_state(new_run('TEST', 'IRONCLAD', 0).rng), 'ENDLESS_CONVEYOR')
    dishes = []
    for index in range(10):
        state = replace(state, counters={'endless_conveyor_dishes': index})
        state = roll_source_conveyor_dish(state, rng, has_open_potion_slot=True)
        dishes.append(state.data['conveyor_dish'])
    assert dishes == ['SUSPICIOUS_CONDIMENT', 'SPICY_SNAPPY', 'JELLY_LIVER', 'CLAM_ROLL',
                      'SEAPUNK_SALAD', 'CLAM_ROLL', 'SPICY_SNAPPY', 'SUSPICIOUS_CONDIMENT',
                      'CLAM_ROLL', 'SEAPUNK_SALAD']
    assert rng.snapshot().streams['ENDLESS_CONVEYOR'].draws == 8


def test_conveyor_public_action_initial_dish_and_new_offer_survive_reload():
    state = _event('ENDLESS_CONVEYOR')
    assert len(legal_actions(state)) == 2
    dish = state.event.metadata['flow_data']['conveyor_dish']
    assert state.rng.streams['ENDLESS_CONVEYOR'].draws == 1
    state = step(load_state(serialize(state)), Action(type=ActionType.CHOOSE_EVENT,
                                                    target_id=dish))
    assert state.event.metadata['flow_data']['conveyor_dish'] != dish
    assert state.event.page_id == 'GRAB_SOMETHING_OFF_THE_BELT'
    assert state.rng.streams['ENDLESS_CONVEYOR'].draws == 2


def test_conveyor_golden_fysh_requires_gold_but_does_not_charge_it():
    flow = event_flow_state('ENDLESS_CONVEYOR', hp=50, max_hp=80, gold=40,
        data={'source_runtime': True, 'conveyor_dish': 'GOLDEN_FYSH'})
    option = current_event_flow_page(flow).options[0]
    assert option.required_gold == 40 and not option.locked
    assert resolve_event_flow_option(flow, option.option_id).state.gold == 115
    assert current_event_flow_page(replace(flow, gold=39)).options[0].locked


def test_bridge_offer_is_an_instance_randomly_rolled_once_then_rerolled_without_type_repeat():
    state = _event('SLIPPERY_BRIDGE', 100, 100)
    first_id = state.event.metadata['bridge_card_instance_id']
    first = next(card for card in state.master_deck if card.instance_id == first_id)
    assert state.rng.streams['SLIPPERY_BRIDGE'].draws == 1
    state = step(load_state(serialize(state)), Action(type=ActionType.CHOOSE_EVENT,
                                                    target_id='HOLD_ON_0'))
    second_id = state.event.metadata['bridge_card_instance_id']
    second = next(card for card in state.master_deck if card.instance_id == second_id)
    assert state.player.hp == 97 and first.card_id != second.card_id
    assert state.rng.streams['SLIPPERY_BRIDGE'].draws == 2
    result = step(state, Action(type=ActionType.CHOOSE_EVENT, target_id='OVERCOME'))
    assert all(card.instance_id != second_id for card in result.master_deck)
    assert any(card.instance_id == first_id for card in result.master_deck)
    assert result.rng.streams['SLIPPERY_BRIDGE'].draws == 2


def test_trial_selects_its_case_randomly_without_player_branch_selection():
    state = _event('TRIAL')
    state = step(state, Action(type=ActionType.CHOOSE_EVENT, target_id='ACCEPT'))
    assert state.event.page_id in {'MERCHANT', 'NOBLE', 'NONDESCRIPT'}
    assert {a.target_id for a in legal_actions(state)} == {'GUILTY', 'INNOCENT'}
    assert state.rng.streams['TRIAL'].draws == 1
    assert all(o.metadata.get('source_native_rule') for o in state.event.options)


def test_tablet_offers_give_up_after_each_decipher_and_kills_when_cost_consumes_max_hp():
    state = _event('TABLET_OF_TRUTH', 5, 5)
    state = step(state, Action(type=ActionType.CHOOSE_EVENT, target_id='DECIPHER_1'))
    assert state.player.max_hp == 2 and state.player.hp == 2
    assert {a.target_id for a in legal_actions(state)} == {'DECIPHER', 'GIVE_UP'}
    result = step(state, Action(type=ActionType.CHOOSE_EVENT, target_id='DECIPHER'))
    assert result.player.max_hp == 1 and result.player.hp == 0
    assert result.rng.streams['TABLET_OF_TRUTH'].draws == 1


def test_tinker_types_and_riders_are_two_distinct_model_rng_choices_once_each():
    state = _event('TINKER_TIME')
    state = step(state, Action(type=ActionType.CHOOSE_EVENT, target_id='CHOOSE_CARD_TYPE'))
    assert len(legal_actions(state)) == 2
    assert state.rng.streams['TINKER_TIME'].draws == 2
    assert [option.option_id for option in state.event.options] == ['ATTACK', 'POWER']
    card_type = state.event.options[0].option_id
    state = step(load_state(serialize(state)), Action(type=ActionType.CHOOSE_EVENT,
                                                    target_id=card_type))
    assert len(legal_actions(state)) == 2
    assert state.rng.streams['TINKER_TIME'].draws == 4
    rider = state.event.options[0].option_id
    result = step(state, Action(type=ActionType.CHOOSE_EVENT, target_id=rider))
    assert result.master_deck[-1].custom['base_card_id'] == 'MAD_SCIENCE'
    assert result.master_deck[-1].custom['card_type'] == card_type.lower()
    assert result.master_deck[-1].custom['rider_id'] == rider.lower()
    assert result.rng.streams['TINKER_TIME'].draws == 4
