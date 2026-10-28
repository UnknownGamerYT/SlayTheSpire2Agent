"""Independent constructor flags exercised through persisted public selectors."""
import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.agent_api import action_space
from sts2sim.engine import Action, ActionType, EventOptionState, EventState, RunPhase
from sts2sim.engine import transitions as t


def _state(*, act=3):
    rows = [
        {'id': f'rare_{i}', 'color': 'ironclad', 'rarity': 'Rare', 'type': 'Attack',
         'cost': 1, 'effects': {'damage': 5, 'block': 2}} for i in range(3)
    ] + [
        {'id': f'colorless_{i}', 'color': 'colorless', 'rarity': 'Uncommon', 'type': 'Skill',
         'cost': 1, 'effects': {'block': 6}} for i in range(3)
    ] + [{'id': 'wrong_color', 'color': 'silent', 'rarity': 'Rare', 'type': 'Attack',
          'cost': 1, 'effects': {'damage': 99}}]
    state = new_run('DIRECT-FACTORY', 'IRONCLAD', 0, source_data={
        'cards': rows, 'combat_reward_card_pool': rows,
        'relics': ['wing_charm', 'silver_crucible', 'silken_tress', 'prismatic_gem', 'dingy_rug'],
    })
    return state.model_copy(update={'act': act, 'phase': RunPhase.MAP})


def _draws(state, stream):
    value = state.rng.streams.get(stream)
    return value.draws if value else 0


def _same_reload(state, action):
    actual = step(state, action)
    assert serialize(actual) == serialize(step(load_state(serialize(state)), action))
    return actual


@pytest.mark.parametrize('relic_id,draws,prefix', [
    ('hefty_tablet', 3, 'rare_'), ('lead_paperweight', 6, 'colorless_'),
])
def test_direct_ancient_factories_keep_flags_and_offer_plain_optional_selector(
    relic_id, draws, prefix,
):
    state = _state()
    before = _draws(state, 'rewards')
    state, _ = t._apply_relic_pickup_effects(state, relic_id)
    assert _draws(state, 'rewards') - before == draws
    offers = [option.metadata['source_factory_card'] for option in state.event.options
              if 'source_factory_card' in option.metadata]
    assert len(offers) == (3 if relic_id == 'hefty_tablet' else 2)
    assert all(card['card_id'].startswith(prefix) for card in offers)
    assert sum(bool(card['enchantments']) for card in offers) == 1  # Wing Charm batch.
    assert not state.flags.get('silver_crucible_times_used')
    assert not state.flags.get('silken_tress_reward_used')
    assert {a.type for a in legal_actions(state)} == {ActionType.CHOOSE_EVENT}
    described = action_space(state)
    selected = next(item for item in described if item.get('card', {}).get('enchantments'))
    assert selected['card']['enchantments'][0]['keyword'] == 'swift'
    action = next(a for a in legal_actions(state) if a.target_id == selected['action']['target_id'])
    old_count = len(state.master_deck)
    result = _same_reload(state, action)
    assert result.phase == RunPhase.MAP
    assert len(result.master_deck) == old_count + (2 if relic_id == 'hefty_tablet' else 1)
    assert result.master_deck[old_count].enchantments[0].keyword == 'swift'
    if relic_id == 'hefty_tablet':
        assert not result.master_deck[old_count].upgraded
        assert result.master_deck[-1].card_id == 'injury'


@pytest.mark.parametrize('relic_id,added', [('hefty_tablet', 1), ('lead_paperweight', 0)])
def test_skip_ancient_simple_selector_still_adds_hefty_injury_and_restores_nested_event(
    relic_id, added,
):
    state = _state()
    original = EventState(event_id='parent', name='Parent',
                          options=(EventOptionState(option_id='leave', title='Leave'),))
    state = state.model_copy(update={'phase': RunPhase.EVENT, 'event': original})
    count = len(state.master_deck)
    state, _ = t._apply_relic_pickup_effects(state, relic_id)
    result = _same_reload(state, Action(type=ActionType.CHOOSE_EVENT, target_id='factory:skip'))
    assert result.event == original
    assert len(result.master_deck) == count + added
    if added:
        assert result.master_deck[-1].card_id == 'injury'


def test_fried_eel_direct_factory_runs_plain_card_hooks_and_preserves_event_rng():
    state = _state()
    event = EventState(event_id='ENDLESS_CONVEYOR', name='Endless Conveyor', page_id='ALL',
        metadata={'flow_page_id': 'ALL', 'flow_data': {'source_runtime': False}},
        options=(EventOptionState(option_id='FRIED_EEL', title='Fried Eel'),))
    state = state.model_copy(update={'phase': RunPhase.EVENT, 'event': event,
        'player': state.player.model_copy(update={'gold': 200})})
    count = len(state.master_deck)
    rewards = _draws(state, 'rewards')
    niche = _draws(state, 'niche')
    result = _same_reload(state, Action(type=ActionType.CHOOSE_EVENT, target_id='FRIED_EEL'))
    assert len(result.master_deck) == count + 1
    assert result.master_deck[-1].card_id.startswith('colorless_')
    assert result.master_deck[-1].enchantments[0].keyword == 'swift'
    assert _draws(result, 'rewards') - rewards == 2
    assert _draws(result, 'niche') - niche == 1
    assert not result.flags.get('silver_crucible_times_used')
    assert not result.flags.get('silken_tress_reward_used')


@pytest.mark.parametrize('relic_id', ['hefty_tablet', 'lead_paperweight'])
def test_public_ancient_pickup_enters_direct_factory_and_restores_map(relic_id):
    state = _state()
    choice = t.MechanicsAncientChoice(option_id='source_factory', name='Source factory',
        relic_id=relic_id, ancient_id='neow', act=1,
        metadata={'source_relic_pickup': True})
    state = state.model_copy(update={'phase': RunPhase.ANCIENT,
        'ancient': state.ancient.model_copy(update={
            'options': (t._ancient_option_from_choice(choice),),
        })})
    picked = _same_reload(state, Action(type=ActionType.CHOOSE_ANCIENT, target_id='source_factory'))
    assert picked.phase == RunPhase.EVENT
    assert picked.reward is None
    assert relic_id in picked.relics
    result = _same_reload(picked, Action(type=ActionType.CHOOSE_EVENT, target_id='factory:skip'))
    assert result.phase == RunPhase.MAP
    assert 'source_direct_card_factory_choice' not in result.flags
