"""Source method contracts for fixed reward groups, free removal and delayed exhausts."""

from __future__ import annotations

from collections import Counter

import pytest

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    Action,
    ActionType,
    CardInstance,
    CombatState,
    EffectEvent,
    MapEdgeState,
    MapNodeState,
    MapState,
    MonsterState,
    RewardState,
    RoomKind,
    RunPhase,
    load_state,
    serialize_state,
    transitions,
)
from sts2sim.engine.rng import capture_random_state, random_from_state


@pytest.mark.parametrize("character", ["IRONCLAD", "SILENT", "DEFECT", "REGENT", "NECROBINDER"])
def test_glass_eye_has_five_independent_source_rarity_groups_and_upgrade_rolls(character) -> None:
    state = new_run("GLASS-EYE", character, 0, source_data={"revealed_epochs": []})
    state = state.model_copy(update={"phase": RunPhase.REWARD, "reward": None})
    rng = random_from_state(state.rng)
    before = capture_random_state(rng).streams.get("rewards")
    actual, _ = transitions._apply_glass_eye_rewards(state, rng)
    assert actual.reward is not None
    groups = (actual.reward.card_options, *actual.reward.card_option_groups)
    assert len(groups) == 5
    assert all(len(group) == 3 and len(set(group)) == 3 for group in groups)
    expected = ("common", "common", "uncommon", "uncommon", "rare")
    pool = {card.card_id: card.rarity.value for card in transitions._combat_card_pool(state)}
    assert tuple(tuple(pool[card] for card in group) for group in groups) == tuple(
        (rarity,) * 3 for rarity in expected
    )
    after = capture_random_state(rng).streams["rewards"]
    assert after.draws - (before.draws if before else 0) == 30
    assert actual.flags.get("card_rarity_offset") == state.flags.get("card_rarity_offset")


def _parasol_shop(*, eternal=False, sozu=False):
    state = new_run("PARASOL", "IRONCLAD", 0, source_data={
        "relics": ["lords_parasol", *(["sozu"] if sozu else [])],
        "player": {"gold": 0, "hp": 20, "max_hp": 80},
        "deck": [{"card_id": "strike_ironclad", "type": "attack", "cost": 1,
                  "custom": {"eternal": eternal}}],
        "shop_plan": {"colored_cards": 0, "colorless_cards": 0, "relics": 1,
                      "potions": 1 if sozu else 0},
        "shop_relic_pool": [{"id": "mango", "rarity": "rare", "kind": "relic"}],
    })
    start = MapNodeState(node_id="start", act=1, floor=0, lane=3, kind=RoomKind.START)
    shop = MapNodeState(node_id="shop", act=1, floor=7, lane=3, kind=RoomKind.SHOP)
    state = state.model_copy(update={"phase": RunPhase.MAP, "ancient": None, "map": MapState(
        act=1, nodes=(start, shop), edges=(MapEdgeState(from_id="start", to_id="shop"),),
        current_node_id="start", completed_node_ids=("start",),
    )})
    state = state.model_copy(update={"flags": {**state.flags, "player_relic_bag": {
        "Common": [], "Uncommon": [], "Rare": ["MANGO"], "Shop": [],
    }}})
    return step(state, Action(type=ActionType.CHOOSE_NODE, target_id="shop"))


def test_lords_parasol_runs_relic_pickup_then_requires_free_card_removal_and_reloads() -> None:
    state = _parasol_shop()
    assert state.player.gold == 0 and state.player.max_hp == 94
    assert "mango" in state.relics
    choices = legal_actions(state)
    assert choices and all(a.type == ActionType.CHOOSE_CARD for a in choices)
    assert state.shop is not None and state.shop.card_removals_bought == 1
    action = choices[0]
    actual = step(state, action)
    restored = step(load_state(serialize_state(state)), action)
    assert serialize_state(actual) == serialize_state(restored)
    assert actual.master_deck == () and actual.player.gold == 0
    assert any(a.type == ActionType.SHOP_LEAVE for a in legal_actions(actual))


def test_lords_parasol_does_not_require_removal_when_every_card_is_eternal() -> None:
    state = _parasol_shop(eternal=True)
    assert any(a.type == ActionType.SHOP_LEAVE for a in legal_actions(state))
    assert state.shop is not None and state.shop.card_removals_bought == 0


def test_lords_parasol_cannot_obtain_potions_through_sozu() -> None:
    state = _parasol_shop(sozu=True)
    assert state.potions == ()


def test_glass_eye_public_pickup_offers_five_rarity_groups_with_reload() -> None:
    state = new_run("GLASS-PICKUP", "SILENT", 0)
    state = state.model_copy(update={"phase": RunPhase.REWARD, "reward": RewardState(
        reward_id="fixture", source="other", relic_id="glass_eye",
    )})
    action = next(a for a in legal_actions(state) if a.type == ActionType.TAKE_REWARD_RELIC)
    actual = step(state, action)
    restored = step(load_state(serialize_state(state)), action)
    assert serialize_state(actual) == serialize_state(restored)
    assert actual.reward is not None and len(actual.reward.card_option_groups) == 5
    assert all(len(group) == 3 for group in actual.reward.card_option_groups)


def test_joss_paper_batches_ethereal_exhausts_until_after_side_end() -> None:
    state = new_run("JOSS-PAPER", "IRONCLAD", 0, source_data={"relics": ["joss_paper"]})
    exhausted = tuple(CardInstance(instance_id=f"e{i}", card_id=f"e{i}",
                                    custom={"ethereal": True}) for i in range(4))
    draw = CardInstance(instance_id="draw", card_id="draw")
    combat = CombatState(player=state.player, exhaust_pile=exhausted, draw_pile=(draw,),
                         monsters=(MonsterState(monster_id="dummy", hp=100, max_hp=100),),
                         metadata={"relic_counters": {"joss_paper": 4}})
    events = tuple(EffectEvent(kind="card_exhausted", target_id=card.instance_id,
                              metadata={"reason": "ethereal_end_turn"}) for card in exhausted)
    combat, rng_state, _ = transitions._apply_card_exhausted_triggers_from_events(
        state, combat, state.rng, events,
    )
    assert combat.hand == ()
    assert combat.metadata["relic_counters"]["joss_paper"] == 4
    assert combat.metadata["joss_paper_ethereal_pending"] == 4
    combat, _, _ = transitions._flush_joss_paper_ethereal(state, combat, rng_state)
    assert [card.card_id for card in combat.hand] == ["draw"]
    assert combat.metadata["relic_counters"]["joss_paper"] == 3
    assert "joss_paper_ethereal_pending" not in combat.metadata
    assert Counter(card.card_id for card in combat.exhaust_pile) == Counter(
        c.card_id for c in exhausted
    )
