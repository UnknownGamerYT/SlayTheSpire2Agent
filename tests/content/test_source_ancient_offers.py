"""Native stable Ancient contracts, including independent C offer vectors."""
from __future__ import annotations

from dataclasses import replace

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import Action, ActionType, AncientState
from sts2sim.engine import transitions as e
from sts2sim.engine.rng import RunRng, capture_random_state, model_random, random_stream
from sts2sim.mechanics.ancients import AncientChoice
from sts2sim.mechanics.source_ancients import AncientFacts, generate_source_ancient_offers


def _facts(name: str) -> AncientFacts:
    return AncientFacts(name, 1 if name == 'neow' else 2, 'ironclad', common_cards=10,
        uncommon_cards=10, enchantable_cards=10, enchantable_attacks=5,
        enchantable_defends=5, removable_cards=10, has_basic_strike=True,
        has_starter_relic=True, has_transcendence_starter=True)


def _offers(facts: AncientFacts, seed: str = 'TEST'):
    bank = RunRng(seed)
    event_rng = model_random(bank, facts.ancient_id.upper())
    return generate_source_ancient_offers(facts, event_rng, random_stream(bank, 'rewards'))


def test_neow_matches_independent_c_reference_and_native_normal_run() -> None:
    assert [offer.relic_id for offer in _offers(_facts('neow'))] == [
        'small_capsule', 'kaleidoscope', 'precarious_shears'
    ]
    state = new_run('TEST', 'IRONCLAD', 0)
    assert state.ancient is not None
    assert [option.relic_id for option in state.ancient.options] == [
        'small_capsule', 'kaleidoscope', 'precarious_shears'
    ]
    assert all(option.metadata['source_relic_pickup'] for option in state.ancient.options)


def test_pael_nonupiepe_and_vakuu_match_independent_c_reference() -> None:
    assert [offer.relic_id for offer in _offers(_facts('pael'))] == [
        'paels_horn', 'paels_wing', 'paels_blood'
    ]
    assert [offer.relic_id for offer in _offers(_facts('nonupeipe'))] == [
        'brilliant_scarf', 'glitter', 'beautiful_bracelet'
    ]
    assert [offer.relic_id for offer in _offers(_facts('vakuu'))] == [
        'fiddle', 'sere_talon', 'choices_paradox'
    ]


def test_native_offer_filters_use_real_player_deck_profile_and_pet_facts() -> None:
    for index in range(75):
        neow = _offers(replace(_facts('neow'), unlocked_characters=('ironclad',),
                              common_cards=3), str(index))
        assert not {'massive_scroll', 'kaleidoscope', 'scroll_boxes'} & {
            offer.relic_id for offer in neow
        }
        nonupeipe = _offers(replace(_facts('nonupeipe'), enchantable_cards=3), str(index))
        assert 'beautiful_bracelet' not in {offer.relic_id for offer in nonupeipe}
        tanx = _offers(replace(_facts('tanx'), enchantable_attacks=2), str(index))
        assert 'tri_boomerang' not in {offer.relic_id for offer in tanx}
        pael = _offers(replace(_facts('pael'), enchantable_defends=2,
                              removable_cards=4, has_event_pet=True), str(index))
        assert not {'paels_claw', 'paels_tooth', 'paels_legion'} & {
            offer.relic_id for offer in pael
        }


def test_orobas_preselects_other_character_and_locks_unavailable_third_offer() -> None:
    for index in range(20):
        offers = _offers(replace(_facts('orobas'), has_starter_relic=False,
                                has_transcendence_starter=False), str(index))
        assert offers[2].locked and not offers[2].relic_id
        for offer in offers:
            if offer.relic_id == 'sea_glass':
                assert offer.flags['sea_glass_character_id'] in {
                    'silent', 'regent', 'necrobinder', 'defect'
                }


def test_source_native_relic_pickup_runs_real_handler_once_after_reload() -> None:
    state = new_run('X', 'IRONCLAD', 0)
    choice = AncientChoice(option_id='gold', ancient_id='neow', relic_id='golden_pearl',
                           metadata={'source_relic_pickup': True})
    option = e._ancient_option_from_choice(choice)
    state = state.model_copy(update={'ancient': AncientState(act=1, ancient_id='neow',
                                                           options=(option,))})
    before = state.player.gold
    loaded = load_state(serialize(state))
    result = step(loaded, Action(type=ActionType.CHOOSE_ANCIENT, target_id='gold'))
    assert result.player.gold == before + 150
    assert result.relics.count('golden_pearl') == 1


def test_source_ancient_uses_no_global_draws_except_dusty_tome_setup() -> None:
    bank = RunRng('TEST')
    before = capture_random_state(bank)
    event_rng = model_random(bank, 'NEOW')
    generate_source_ancient_offers(_facts('neow'), event_rng, random_stream(bank, 'rewards'))
    after = capture_random_state(bank)
    assert after.draws == before.draws == 0
    assert after.streams['rewards'].draws == 0
    state = new_run('X', 'IRONCLAD', 0)
    locked = e._ancient_option_from_choice(AncientChoice(option_id='locked',
        metadata={'source_locked': True}))
    assert state.ancient is not None
    state = state.model_copy(update={'ancient': state.ancient.model_copy(update={
        'options': (*state.ancient.options, locked)
    })})
    assert all(action.target_id != 'locked' for action in legal_actions(state))


def test_character_unlock_epochs_filter_ancient_character_choices_in_registry_order():
    from sts2sim.mechanics.source_ancients import source_unlocked_characters
    assert source_unlocked_characters({'revealed_epochs': ()}) == ('ironclad',)
    assert source_unlocked_characters({'revealed_epochs': (
        'SILENT1_EPOCH', 'DEFECT1_EPOCH')}) == ('ironclad', 'silent', 'defect')
    assert source_unlocked_characters({'unlocked_character_ids': ('DEFECT', 'IRONCLAD')}) == (
        'ironclad', 'defect')
