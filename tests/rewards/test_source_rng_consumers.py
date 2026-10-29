"""Independent stable-source reward/merchant draw order and binary32 contracts."""
from __future__ import annotations

from sts2sim import load_state, new_run, serialize, step
from sts2sim.engine import Action, ActionType, RewardState, RunPhase
from sts2sim.engine import transitions as e
from sts2sim.engine.rng import RunRng, capture_random_state, random_stream
from sts2sim.mechanics.combat_rewards import (
    CombatRewardCard,
    CombatRewardContext,
    draw_combat_reward,
    roll_combat_potion_drop,
)
from sts2sim.mechanics.rewards import (
    CardRarity,
    EncounterType,
    PotionRarity,
    RelicRarity,
    RewardPityState,
)
from sts2sim.mechanics.shop_rooms import build_basic_shop_inventory
from sts2sim.mechanics.shops import ShopItem, ShopItemKind, roll_base_shop_price
from sts2sim.mechanics.source_rng import float32
from sts2sim.mechanics.unknown_rooms import resolve_unknown_room


def test_reward_order_and_upgrades_match_independent_c_reference() -> None:
    bank = RunRng('TEST')
    rng = random_stream(bank, 'rewards')
    cards = tuple(CombatRewardCard(f'{rarity.value}{index}', rarity)
                  for rarity in CardRarity for index in range(3))
    bundle = draw_combat_reward(rng, card_pool=cards, potion_pool=(), relic_pool=(),
        boss_relic_pool=(), context=CombatRewardContext('IRONCLAD', act=2), relic_count=0)
    # C reference: drop float first; gold; then rarity/item/upgrade per card.
    assert bundle.gold == 10
    assert bundle.card_ids == ('common2', 'uncommon1', 'common0')
    assert bundle.card_upgraded == (False, False, True)
    assert bundle.pity_state.card_rarity_offset == float.fromhex('-0x1.47ae14p-6')
    assert bundle.pity_state.potion_current_value == .5
    assert capture_random_state(bank).streams['rewards'].draws == 11


def test_elite_bonus_potion_hit_always_lowers_float32_pity() -> None:
    class Fixed(RunRng):
        def random(self) -> float:
            return .5
    result = roll_combat_potion_drop(Fixed('X'), RewardPityState(),
        CombatRewardContext('IRONCLAD', encounter=EncounterType.ELITE))
    assert result.dropped
    assert result.state.potion_chance_bonus == -10
    assert result.state.potion_current_value == float32(float32(.4) - float32(.1))


def test_relic_price_binary32_vectors_match_independent_c() -> None:
    rng = random_stream(RunRng('TEST'), 'shops')
    assert [roll_base_shop_price(rng, ShopItemKind.RELIC, RelicRarity.COMMON)
            for _ in range(8)] == [166, 180, 176, 159, 149, 160, 155, 196]


def test_source_merchant_stream_counters_include_sale_reroll_and_upgrade_draws() -> None:
    bank = RunRng('TEST')
    rng = random_stream(bank, 'shops')
    cards = tuple(ShopItem(f'{kind}_{rarity.value}_{index}', ShopItemKind.CARD,
                           rarity, card_type=kind)
                  for kind in ('attack', 'skill', 'power')
                  for rarity in CardRarity for index in range(3))
    colorless = tuple(ShopItem(f'colorless_{rarity.value}', ShopItemKind.COLORLESS_CARD, rarity)
                      for rarity in (CardRarity.UNCOMMON, CardRarity.RARE))
    potions = tuple(ShopItem(f'potion_{rarity.value}_{index}', ShopItemKind.POTION, rarity)
                    for rarity in PotionRarity for index in range(3))
    def relic(rarity: str) -> ShopItem:
        return ShopItem(f'relic_{rarity}', ShopItemKind.RELIC, RelicRarity(rarity))
    inventory = build_basic_shop_inventory(rng, card_pool=cards,
        colorless_card_pool=colorless, potion_pool=potions, relic_selector=relic,
        rare_offset_percent=-5)
    assert len(inventory.items) == 14
    state = capture_random_state(bank)
    assert state.streams['shops'].draws == 28
    assert state.streams['rewards'].draws == 14
    assert [item.item.card_type for item in inventory.items[:5]] == [
        'attack', 'attack', 'skill', 'skill', 'power'
    ]


def test_normal_shop_uses_character_source_pool_and_upgraded_offer_survives_reload() -> None:
    state = new_run('X', 'REGENT', 0)
    pool = e._shop_item_pool(state, 'shop_card_pool', e._default_shop_card_pool())
    assert len(pool) > 50
    assert any(item.item_id == 'gather_light' for item in pool)
    assert all(item.item_id != 'pommel_strike' for item in pool)
    reward = RewardState(reward_id='fixture', source='combat', card_options=('gather_light',),
                         metadata={'card_option_upgraded': (True,)})
    state = state.model_copy(update={'phase': RunPhase.REWARD, 'reward': reward})
    restored = load_state(serialize(state))
    result = step(restored, Action(type=ActionType.TAKE_REWARD_CARD, target_id='reward:card:0'))
    assert result.master_deck[-1].card_id == 'gather_light'
    assert result.master_deck[-1].upgraded


def test_unknown_source_boundary_and_golden_path_keep_unrolled_odds() -> None:
    class Fixed(RunRng):
        def random(self) -> float:
            return .1
    rng = Fixed('X')
    result = resolve_unknown_room(rng)
    assert result.outcome == 'monster'  # source <=, after binary32 conversion
    golden = resolve_unknown_room(rng, allowed_outcomes=('event',))
    assert golden.outcome == 'event'
    assert golden.next_odds['monster'] == float32(.1)


def test_first_run_unknowns_do_not_draw_rng() -> None:
    state = new_run('X', 'IRONCLAD', 0)
    state = state.model_copy(update={'flags': {**state.flags, 'number_of_runs': 0}})
    assert state.map is not None
    original = state.rng
    node = state.map.nodes[0].model_copy(update={'kind': e.RoomKind.EVENT})
    outcomes = []
    for index in range(3):
        state, resolved, _ = e._resolve_unknown_node(state,
            node.model_copy(update={'node_id': f'tutorial_unknown_{index}'}))
        outcomes.append(resolved.kind.value)
        assert state.rng == original
    assert outcomes == ['event', 'event', 'monster']


def test_source_generated_status_insertion_keeps_existing_order_and_uses_shuffle_stream() -> None:
    from sts2sim.engine.models import CombatState
    state = new_run('TEST', 'IRONCLAD', 0)
    original = state.master_deck[:4]
    combat = CombatState(player=state.player, draw_pile=original)
    bank = RunRng('TEST')
    updated, _ = e._add_generated_status_cards_to_draw_pile_shuffled(
        combat, bank, 'dazed', 2, source_id='source_fixture', reason='source_fixture')
    original_ids = {card.instance_id for card in original}
    assert tuple(card for card in updated.draw_pile if card.instance_id in original_ids) == original
    snapshot = capture_random_state(bank)
    assert snapshot.streams['shuffle'].draws == 2
    assert 'combat_card_selection' not in snapshot.streams
