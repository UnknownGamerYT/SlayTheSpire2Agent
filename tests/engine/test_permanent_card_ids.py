from __future__ import annotations

from random import Random

import pytest

from sts2sim import action_space, legal_actions, load_state, serialize, step
from sts2sim.engine import (
    AncientOptionState,
    AncientState,
    CardInstance,
    PlayerState,
    RewardState,
    RunPhase,
    RunState,
    ShopItemState,
    ShopState,
)
from sts2sim.engine.rng import new_rng_state
from sts2sim.engine.transitions import (
    _apply_event_flow_markers,
    _apply_paels_tooth_add_removed_card_marker,
    _apply_relic_add_deck_cards_marker,
    _apply_relic_duplicate_deck_card_marker,
    _apply_relic_pickup_effects,
    _clone_at_campfire,
    _enchant_card,
    _instantiate_deck,
    _shop_gain_all_non_service_items,
)
from sts2sim.mechanics.event_flows import EventFlowMarker, EventFlowMarkerKind
from sts2sim.mechanics.relic_combat import CombatRelicHook, CombatRelicMarker
from sts2sim.mechanics.relics import RelicEffectMarker


def _starting_state() -> RunState:
    cards = tuple(
        CardInstance(
            instance_id=f"card_{index:03d}", card_id=f"starter_{index}",
            type="attack", target="enemy", cost=1, effects={"damage": 6},
        )
        for index in range(1, 6)
    )
    pool = tuple(
        {
            "id": f"gift_{index}", "color": "test", "rarity": "Common",
            "type": "Attack", "target": "AnyEnemy", "cost": 1, "damage": 8,
        }
        for index in range(3)
    )
    return RunState(
        seed=16, character_id="TEST", rng=new_rng_state(16), phase=RunPhase.REWARD,
        player=PlayerState(hp=60, max_hp=80, gold=100), master_deck=cards,
        flags={"cards": pool, "card_pool": pool},
        reward=RewardState(
            reward_id="remove_then_gain", card_ids=("gift_0", "gift_1"),
            card_options=("gift_2",), metadata={
                "optional_remove_card_count": 1,
                "optional_remove_card_instance_ids": ("card_001",),
            },
        ),
    )


def _after_removal() -> RunState:
    state = _starting_state()
    state = step(
        state, next(a for a in legal_actions(state) if a.target_id == "reward:remove_card:0"),
    )
    assert [card.instance_id for card in state.master_deck] == [
        "card_002", "card_003", "card_004", "card_005",
    ]
    return state


def _assert_addition_ids(before, after, events, *, count: int) -> None:
    ids = [card.instance_id for card in after.master_deck]
    assert len(ids) == len(set(ids))
    assert after.master_deck[:len(before.master_deck)] == before.master_deck
    added_ids = ids[len(before.master_deck):]
    assert len(added_ids) == count
    assert [event.target_id for event in events if event.kind in {
        "reward_card_taken", "ancient_card_added", "event_card_added",
        "event_random_card_added", "event_custom_card_created",
        "relic_deck_card_added", "relic_deck_card_duplicated", "card_cloned",
    }] == added_ids


def test_removal_then_multiple_reward_gains_keeps_ids_and_agent_targets_distinct() -> None:
    state = _after_removal()
    before = state
    restored = load_state(serialize(state))
    targets = ("reward:fixed_card:0", "reward:card:0", "reward:fixed_card:1")
    for target in targets:
        state = step(state, next(a for a in legal_actions(state) if a.target_id == target))
        restored = step(restored, next(a for a in legal_actions(restored) if a.target_id == target))
    assert serialize(state) == serialize(restored)
    _assert_addition_ids(before, state, tuple(
        event for entry in state.replay_log[1:] for event in entry.events
    ), count=3)
    assert [card.instance_id for card in state.master_deck[-3:]] == [
        "card_006", "card_007", "card_008",
    ]

    state, _ = _apply_relic_pickup_effects(state, "paels_growth")
    descriptors = action_space(state)
    assert {row["card_instance_id"] for row in descriptors} == {
        card.instance_id for card in state.master_deck
    }
    chosen_id = state.master_deck[-2].instance_id
    state = step(state, next(a for a in legal_actions(state) if a.card_instance_id == chosen_id))
    assert [card.instance_id for card in state.master_deck if card.enchantments] == [chosen_id]


def test_ancient_removal_and_multiple_gains_in_one_choice_keep_ids_distinct() -> None:
    state = _starting_state()
    option = AncientOptionState(
        option_id="ancient_gifts", name="Gifts", kind="positive_relic", relic_id="test_relic",
        metadata={"choice": {
            "option_id": "ancient_gifts", "name": "Gifts", "kind": "positive_relic",
            "remove_card_ids": ("starter_1",), "fixed_card_ids": ("gift_0", "gift_1"),
        }},
    )
    state = state.model_copy(update={
        "phase": RunPhase.ANCIENT, "reward": None,
        "ancient": AncientState(act=1, ancient_id="neow", options=(option,)),
    })
    restored = load_state(serialize(state))
    results = [step(candidate, next(a for a in legal_actions(candidate)
                                   if a.type == "choose_ancient"))
               for candidate in (state, restored)]
    assert serialize(results[0]) == serialize(results[1])
    result = results[0]
    assert len({card.instance_id for card in result.master_deck}) == 6
    assert result.master_deck[:4] == state.master_deck[1:]
    assert [event.target_id for event in result.replay_log[-1].events
            if event.kind == "ancient_card_added"] == [
        card.instance_id for card in result.master_deck[-2:]
    ]


@pytest.mark.parametrize("kind", ["fixed", "random", "custom", "tinker"])
def test_event_gains_after_removal_reserve_ids_before_emitting_events(kind: str) -> None:
    state = _after_removal()
    if kind == "fixed":
        markers = tuple(EventFlowMarker(kind=EventFlowMarkerKind.CARD_ADD, item_id=f"gift_{i}")
                        for i in range(3))
        count = 3
    elif kind == "random":
        markers = (EventFlowMarker(kind=EventFlowMarkerKind.RANDOM_CARD, count=3),)
        count = 3
    else:
        metadata = {"source_event_id": "tinker_time", "card_type": "skill", "rider_id": "test"}
        if kind == "tinker":
            # Tinker has an explicit identity prefix, independent of card_###.
            state = state.model_copy(update={"master_deck": state.master_deck + (
                state.master_deck[-1].model_copy(update={"instance_id": "tinker_006"}),
            )})
        markers = (EventFlowMarker(kind=EventFlowMarkerKind.CUSTOM_CARD, qualifier="skill",
                                   metadata=metadata if kind == "tinker" else {}),)
        count = 1
    result, events = _apply_event_flow_markers(state, markers, Random(16))
    restored_result, restored_events = _apply_event_flow_markers(
        load_state(serialize(state)), markers, Random(16),
    )
    assert serialize(result) == serialize(restored_result)
    assert events == restored_events
    _assert_addition_ids(state, result, events, count=count)


def test_relic_multiple_gains_after_removal_do_not_depend_on_deck_add_relics() -> None:
    state = _after_removal()
    result, events = _apply_relic_add_deck_cards_marker(
        state, "test_relic", RelicEffectMarker(
            kind="add_deck_cards", relic_id="test_relic", amount=3, metadata={"card_id": "gift_0"},
        ),
    )
    _assert_addition_ids(state, result, events, count=3)
    assert not result.relics


def test_paels_tooth_preserves_returned_card_properties_while_reserving_new_id() -> None:
    state = _after_removal()
    removed = _enchant_card(_starting_state().master_deck[0], "glam", 1, source_id="test")
    state = state.model_copy(update={"flags": {
        **state.flags, "relic_removed_card_specs": (removed.model_dump(mode="json"),),
    }})
    result, events = _apply_paels_tooth_add_removed_card_marker(
        state, Random(16), CombatRelicMarker(
            kind="add_removed_card", relic_id="paels_tooth", hook=CombatRelicHook.COMBAT_END,
        ),
    )
    _assert_addition_ids(state, result, events, count=1)
    returned = result.master_deck[-1]
    assert returned.instance_id == "card_006"
    assert returned.enchantments == removed.enchantments
    assert returned.effects == removed.effects


def test_duplicate_and_clone_preserve_existing_unique_identity_formats() -> None:
    state = _after_removal()
    card = _enchant_card(state.master_deck[0], "clone", 1, source_id="paels_growth")
    state = state.model_copy(update={"master_deck": (card, *state.master_deck[1:],
        card.model_copy(update={"instance_id": f"{card.card_id}:7"}),
        card.model_copy(update={"instance_id": f"{card.instance_id}:clone:1"}),
    )})
    result, events = _apply_relic_duplicate_deck_card_marker(
        state, "dollys_mirror", RelicEffectMarker(
            kind="duplicate_deck_card", relic_id="dollys_mirror", amount=1,
            metadata={"match_card_instance_ids": (card.instance_id,)},
        ),
    )
    _assert_addition_ids(state, result, events, count=1)
    assert result.master_deck[-1].instance_id == f"{card.card_id}:8"
    state = state.model_copy(update={"phase": RunPhase.REST})
    result, events = _clone_at_campfire(state)
    _assert_addition_ids(state, result, events, count=3)
    assert result.master_deck[len(state.master_deck)].instance_id == f"{card.instance_id}:clone:2"


def test_shop_purchase_and_free_claim_reserve_explicit_ids_on_collision() -> None:
    state = _after_removal()
    preferred_id = f"shop_{len(state.replay_log)}_0"
    state = state.model_copy(update={
        "phase": RunPhase.SHOP, "reward": None,
        "master_deck": state.master_deck + (
            state.master_deck[0].model_copy(update={"instance_id": preferred_id}),
            state.master_deck[0].model_copy(update={
                "instance_id": f"shop_free_{len(state.replay_log)}_0",
            }),
        ),
        "shop": ShopState(node_id="shop", items=(ShopItemState(
            slot_id="shop:0", item_id="gift_0", kind="card", price=10, base_price=10,
        ),)),
    })
    result = step(state, next(a for a in legal_actions(state) if a.type == "shop_buy"))
    assert result.master_deck[:-1] == state.master_deck
    assert len({card.instance_id for card in result.master_deck}) == len(result.master_deck)
    assert next(event.metadata["card_instance_id"] for event in result.replay_log[-1].events
                if event.kind == "shop_item_bought") == result.master_deck[-1].instance_id
    update, _ = _shop_gain_all_non_service_items(state, state.shop)
    deck = update["master_deck"]
    assert deck[:-1] == state.master_deck
    assert len({card.instance_id for card in deck}) == len(deck)


def test_initial_duplicate_ids_are_repaired_without_taking_other_explicit_ids() -> None:
    raw = (
        {"card_id": "first", "instance_id": "card_001"},
        {"card_id": "duplicate", "instance_id": "card_001"},
        {"card_id": "third", "instance_id": "card_002"},
        {"card_id": "named", "instance_id": "special"},
        {"card_id": "named_duplicate", "instance_id": "special"},
        {"card_id": "reserved_suffix", "instance_id": "special:copy:1"},
    )
    deck = _instantiate_deck(raw, {})
    assert [card.instance_id for card in deck] == [
        "card_001", "card_003", "card_002", "special", "special:copy:2", "special:copy:1",
    ]
    assert deck == _instantiate_deck(raw, {})


def test_transform_and_upgrade_retain_the_selected_cards_identity() -> None:
    state = _after_removal()
    chosen = state.master_deck[-1]
    result, events = _apply_event_flow_markers(
        state, (EventFlowMarker(kind=EventFlowMarkerKind.CARD_TRANSFORM),), Random(16),
        selected_card_instance_ids=(chosen.instance_id,),
    )
    assert [card.instance_id for card in result.master_deck] == [
        card.instance_id for card in state.master_deck
    ]
    assert result.master_deck[-1].card_id != chosen.card_id
    assert events[-1].target_id == chosen.instance_id
    upgraded, _ = _apply_event_flow_markers(
        result, (EventFlowMarker(kind=EventFlowMarkerKind.CARD_UPGRADE_ALL),), Random(16),
    )
    assert [card.instance_id for card in upgraded.master_deck] == [
        card.instance_id for card in state.master_deck
    ]
