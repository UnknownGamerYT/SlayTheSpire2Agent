"""Stable registered reward/merchant pools, independent source vectors and public flows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sts2sim import legal_actions, new_run, step
from sts2sim.content.card_pool_selection import source_card_pool_rows
from sts2sim.engine import (
    Action,
    ActionType,
    MapEdgeState,
    MapNodeState,
    MapState,
    RoomKind,
    RunPhase,
    load_state,
    serialize_state,
    transitions,
)
from sts2sim.mechanics.shop_rooms import ShopItem, ShopItemKind

SOURCE = json.loads(
    (Path(__file__).parents[1] / "fixtures/stable_card_pool_source.json").read_text()
)


def _expected(pool: str, epochs: list[str] | None, multiplayer: bool) -> tuple[str, ...]:
    excluded_constraint = "SingleplayerOnly" if multiplayer else "MultiplayerOnly"
    return tuple(
        card_id
        for card_id in SOURCE["pools"][pool]
        if SOURCE["multiplayer_constraints"].get(card_id) != excluded_constraint
        and (epochs is None or card_id not in SOURCE["epochs"]
             or SOURCE["epochs"][card_id] in epochs)
    )


@pytest.mark.parametrize("pool", SOURCE["pools"])
@pytest.mark.parametrize("epochs", [None, [], ["IRONCLAD2_EPOCH", "COLORLESS4_EPOCH"]])
@pytest.mark.parametrize("multiplayer", [False, True])
def test_full_ordered_pool_matches_source_vectors(pool, epochs, multiplayer) -> None:
    ids = SOURCE["pools"][pool]
    rows = [{"id": card_id.upper()} for card_id in reversed(ids)]
    rows += [{"id": "UNREGISTERED_CARD"}, {"id": ids[0].upper()}]
    actual = source_card_pool_rows(rows, pool, unlocked_epochs=epochs, multiplayer=multiplayer)
    assert tuple(row["id"].lower() for row in actual) == _expected(pool, epochs, multiplayer)


def test_nine_class_multiplayer_cards_are_removed_from_singleplayer() -> None:
    expected = {
        "ironclad": {"demonic_shield", "tank"},
        "silent": {"flanking", "sneaky"},
        "regent": {"hammer_time", "largesse"},
        "necrobinder": {"glimpse_beyond", "legion_of_bone"},
        "defect": {"energy_surge"},
    }
    for pool, excluded in expected.items():
        state = new_run("SP-POOL", pool.upper(), 0)
        sp_pool = {card.card_id for card in transitions._combat_card_pool(state)}
        mp_state = state.model_copy(update={"flags": {**state.flags, "player_count": 2}})
        mp_pool = {card.card_id for card in transitions._combat_card_pool(mp_state)}
        assert excluded.isdisjoint(sp_pool)
        assert excluded <= mp_pool
        if pool == "silent":
            assert "well_laid_plans" in sp_pool
            assert "well_laid_plans" not in mp_pool


def test_reward_and_shop_epoch_filters_preserve_eligible_source_order() -> None:
    state = new_run("EPOCH-POOL", "IRONCLAD", 0, source_data={"revealed_epochs": []})
    locked = transitions._combat_card_pool(state)
    unlocked_state = state.model_copy(
        update={"flags": {**state.flags, "revealed_epochs": ["IRONCLAD2_EPOCH"]}}
    )
    unlocked = transitions._combat_card_pool(unlocked_state)
    assert {card.card_id for card in unlocked} - {card.card_id for card in locked} == {
        "cruelty", "dominate", "molten_fist"
    }
    shop = transitions._shop_item_pool(unlocked_state, "shop_card_pool", ())
    assert [card.card_id for card in unlocked] == [card.item_id for card in shop]
    assert all(card.card_id not in {"cinder", "inferno"} for card in unlocked)


def test_reward_pool_keeps_cards_excluded_only_from_in_combat_generation() -> None:
    state = new_run("REWARD-POOL", "DEFECT", 0)
    assert "genetic_algorithm" in {card.card_id for card in transitions._combat_card_pool(state)}
    colorless = transitions._shop_item_pool(
        state, "shop_colorless_card_pool", (), default_kind=ShopItemKind.COLORLESS_CARD
    )
    assert "alchemize" in {card.item_id for card in colorless}


def test_explicit_custom_pools_and_reserved_test_character_keep_fixture_scope() -> None:
    cards = [{"id": "CUSTOM_CARD", "color": "ironclad", "rarity": "Rare"}]
    state = new_run("CUSTOM-POOL", "IRONCLAD", 0, source_data={"cards": cards})
    assert [card.card_id for card in transitions._combat_card_pool(state)] == ["custom_card"]
    assert transitions._registered_reward_card_rows(state, "ironclad") is None
    state = new_run(7, "TEST", 0, source_data={"cards": cards})
    assert transitions._registered_reward_card_rows(state, "colorless") is None
    state = state.model_copy(update={"flags": {**state.flags, "shop_card_pool": ["explicit"]}})
    assert transitions._shop_item_pool(state, "shop_card_pool", ())[0].item_id == "explicit"


def test_empty_registered_metadata_never_falls_back_to_unrelated_cards(monkeypatch) -> None:
    state = new_run("NO-METADATA", "IRONCLAD", 0)
    monkeypatch.setattr(
        transitions, "_cached_source_rows",
        lambda *args: ({"id": "UNREGISTERED", "color": "ironclad", "rarity": "Rare"},),
    )
    assert transitions._combat_card_pool(state) == ()
    fallback = (ShopItem("UNRELATED", ShopItemKind.CARD),)
    assert transitions._shop_item_pool(state, "shop_card_pool", fallback) == ()


def _next_room(state: Any, kind: RoomKind) -> Any:
    start = MapNodeState(node_id="start", act=1, floor=0, lane=3, kind=RoomKind.START)
    target = MapNodeState(node_id="next", act=1, floor=8, lane=3, kind=kind)
    return state.model_copy(update={
        "phase": RunPhase.MAP, "ancient": None,
        "map": MapState(act=1, nodes=(start, target),
                        edges=(MapEdgeState(from_id="start", to_id="next"),),
                        current_node_id="start", completed_node_ids=("start",)),
    })


@pytest.mark.parametrize("kind", [RoomKind.MONSTER, RoomKind.SHOP])
def test_public_normal_generation_uses_unlocked_singleplayer_pool_and_reloads(kind) -> None:
    state = _next_room(new_run("PUBLIC-POOL", "SILENT", 0, source_data={
        "revealed_epochs": [],
        "deck": [{"card_id": "debug_kill", "type": "attack", "cost": 0,
                  "target": "all_enemies", "effects": {"all_damage": 999}}],
    }), kind)
    action = Action(type=ActionType.CHOOSE_NODE, target_id="next")
    entered = step(state, action)
    restored = step(load_state(serialize_state(state)), action)
    assert serialize_state(entered) == serialize_state(restored)
    if kind is RoomKind.MONSTER:
        kill = next(
            action for action in legal_actions(entered) if action.type == ActionType.PLAY_CARD
        )
        entered = step(entered, kill)
        assert entered.reward is not None
        offered = entered.reward.card_options
    else:
        assert entered.shop is not None
        offered = tuple(item.item_id for item in entered.shop.items if item.kind == "card")
    assert offered
    assert set(offered) <= set(_expected("silent", [], False))


@pytest.mark.parametrize("key", ["cards", "card_library"])
def test_full_source_catalog_in_flags_still_respects_player_mode_and_epochs(key):
    state = new_run("LIBRARY-POOL", "IRONCLAD", 0)
    rows = transitions._cached_source_rows(state, "cards")
    state = state.model_copy(update={"flags": {
        **state.flags, key: rows, "revealed_epochs": [],
    }})
    pool = {card.card_id for card in transitions._combat_card_pool(state)}
    assert "strike_ironclad" not in pool
    assert {"demonic_shield", "tank", "cruelty", "dominate", "molten_fist"}.isdisjoint(pool)
    assert "bash" not in pool  # Basic cards never become ordinary rewards.
    assert "anger" in pool
