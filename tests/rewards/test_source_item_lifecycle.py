from __future__ import annotations

from rewards.test_relic_deck_choices import _card, _choose, _pickup, _reward_state
from rewards.test_sea_glass import _before_pickup

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import MapNodeState
from sts2sim.engine.rng import RunRng, capture_random_state, random_from_state
from sts2sim.engine.transitions import (
    _apply_paels_tooth_add_removed_card_marker,
    _start_combat_for_node,
    _treasure_reward_state,
)
from sts2sim.mechanics.relic_combat import CombatRelicHook, CombatRelicMarker


def test_paels_tooth_chooses_only_five_removable_upgradable_cards_and_consumes_duplicates():
    cards = tuple(_card(i) for i in range(6)) + (_card(6, upgraded=True), _card(7, eternal=True))
    state = _pickup('paels_tooth', cards=cards)
    eligible = {a.card_instance_id for a in legal_actions(state)}
    assert eligible == {f'deck:{i}' for i in range(6)}
    for i in (5, 3, 1, 4, 2):
        state = _choose(load_state(serialize(state)), f'deck:{i}')
    saved = state.flags['paels_tooth_serialized_cards']
    assert len(saved) == 5
    assert [s['card_id'] for s in saved] == sorted(s['card_id'] for s in saved)
    assert len({s['card_id'] for s in saved}) == 2
    marker = CombatRelicMarker(kind='add_deck_cards', relic_id='paels_tooth',
        hook=CombatRelicHook.COMBAT_END, amount=1, metadata={'upgraded': True})
    rng = RunRng('TOOTH', derivation='sts2.v0.107.1.utf16-djb2-splitmix64')
    for remaining in range(4, -1, -1):
        state, events = _apply_paels_tooth_add_removed_card_marker(state, rng, marker)
        assert events[0].kind == 'relic_deck_card_added'
        assert state.master_deck[-1].upgraded
        assert len(state.flags['paels_tooth_serialized_cards']) == remaining
    assert capture_random_state(rng).streams['rewards'].draws == 5
    result, events = _apply_paels_tooth_add_removed_card_marker(state, rng, marker)
    assert result.master_deck == state.master_deck
    assert events[0].kind == 'relic_deck_card_add_blocked'


def test_tea_defaults_to_one_combat_and_does_not_displace_innate_cards():
    state = _reward_state('tea_of_discourtesy')
    innate = _card(0).model_copy(update={'keywords': ('innate',)})
    state = state.model_copy(update={'relics': ('tea_of_discourtesy',),
        'master_deck': (innate, *_card_sequence()),
        'flags': {}})
    node = MapNodeState(node_id='tea', kind='monster', act=1, floor=1, lane=0)
    _, combat, _, _ = _start_combat_for_node(state, node)
    cards = (*combat.hand, *combat.draw_pile)
    assert sum(c.card_id == 'dazed' for c in cards) == 2
    assert innate.instance_id in {c.instance_id for c in combat.hand}
    assert combat.metadata['relic_counters']['tea_of_discourtesy'] == 0
    state = state.model_copy(update={'flags': {**state.flags,
        'relic_counters': {'tea_of_discourtesy': 0}}})
    _, combat, _, _ = _start_combat_for_node(state, node)
    assert not any(c.card_id == 'dazed' for c in (*combat.hand, *combat.draw_pile))


def _card_sequence():
    return tuple(_card(i) for i in range(1, 9))


def test_silver_crucible_skips_first_treasure_entered_after_acquisition():
    state = _pickup('silver_crucible')
    state = state.model_copy(update={'flags': {**state.flags, 'treasure_chests_opened': 7}})
    node = MapNodeState(node_id='chest', kind='treasure', act=2, floor=5, lane=0)
    reward, _, flags = _treasure_reward_state(state, node, random_from_state(state.rng))
    assert reward.metadata['empty']
    assert flags['silver_crucible_treasure_rooms_entered'] == 1
    state = state.model_copy(update={'flags': flags})
    reward, _, flags = _treasure_reward_state(state, node, random_from_state(state.rng))
    assert not reward.metadata['empty']
    assert flags['silver_crucible_treasure_rooms_entered'] == 2


def test_sea_glass_runs_three_five_card_factory_hooks_without_reward_only_hooks():
    state = _before_pickup()
    state = state.model_copy(update={'relics': ('silver_crucible', 'silken_tress',
                                               'wing_charm', 'toxic_egg')})
    state = step(state, next(a for a in legal_actions(state) if a.type == 'take_reward_relic'))
    assert all(state.reward.metadata['fixed_card_upgraded'])
    assert len(state.reward.metadata['card_reward_option_modifiers']['fixed']) == 3
    assert not state.flags.get('silver_crucible_times_used')
    assert not state.flags.get('silken_tress_reward_used')
