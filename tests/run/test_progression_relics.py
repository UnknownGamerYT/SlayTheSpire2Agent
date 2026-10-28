from __future__ import annotations

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import MapEdgeState, MapNodeState, MapState, RewardState, RoomKind, RunPhase
from sts2sim.engine.models import ShopItemState, ShopState
from sts2sim.engine.transitions import _combat_reward_encounter
from sts2sim.mechanics.rewards import EncounterType


def _state(relics: tuple[str, ...] = (), kind: RoomKind = RoomKind.REST):
    state = new_run(123, "TEST", 0)
    nodes = (MapNodeState(node_id="start", act=1, floor=0, lane=0, kind=RoomKind.START),) + tuple(
        MapNodeState(node_id=f"room{floor}_{lane}", act=1, floor=floor, lane=lane, kind=kind)
        for floor in range(1, 5)
        for lane in range(2)
    )
    edges = (MapEdgeState(from_id="start", to_id="room1_0"),) + tuple(
        MapEdgeState(from_id=f"room{floor}_{lane}", to_id=f"room{floor + 1}_{lane}")
        for floor in range(1, 4)
        for lane in range(2)
    )
    return state.model_copy(
        update={
            "phase": RunPhase.MAP,
            "ancient": None,
            "relics": relics,
            "player": state.player.model_copy(update={"hp": 40, "max_hp": 80, "gold": 100}),
            "map": MapState(
                nodes=nodes, edges=edges, current_node_id="start", completed_node_ids=("start",)
            ),
        }
    )


def _act(state, kind: str, target: str | None = None):
    return next(
        a
        for a in legal_actions(state)
        if a.type == kind and (target is None or a.target_id == target)
    )


def test_bowler_modifies_one_gain_and_dragon_fruit_observes_one_notification() -> None:
    state = _state(("BOWLER_HAT", "DRAGON_FRUIT"))
    state = state.model_copy(
        update={
            "phase": RunPhase.REWARD,
            "reward": RewardState(reward_id="gold", source="combat", gold=20),
        }
    )
    state = step(state, _act(state, "take_reward_gold"))
    assert state.player.gold == 125
    assert (state.player.max_hp, state.player.hp) == (81, 41)
    assert state.reward.gold == 20  # The offer remains unmodified.
    state = step(state, _act(state, "proceed"))
    assert (state.player.gold, state.player.max_hp) == (125, 81)


def test_maw_bank_floor_gold_is_available_on_room_entry_and_stops_after_spending() -> None:
    state = _state(("maw_bank", "bowler_hat", "dragon_fruit"))
    state = step(state, _act(state, "choose_node"))
    assert state.player.gold == 115
    assert state.player.max_hp == 81
    state = state.model_copy(
        update={
            "phase": RunPhase.SHOP,
            "shop": ShopState(
                node_id="shop",
                items=(
                    ShopItemState(
                        slot_id="slot:0",
                        item_id="fire_potion",
                        kind="potion",
                        price=20,
                        base_price=20,
                    ),
                ),
            ),
        }
    )
    state = step(state, _act(state, "shop_buy"))
    assert state.flags["maw_bank_disabled"] is True
    assert state.player.gold == 95
    state = step(state, _act(state, "shop_leave"))
    state = load_state(serialize(state))
    state = step(state, _act(state, "choose_node"))
    assert state.player.gold == 95
    assert state.player.max_hp == 81


def test_free_shop_purchase_does_not_disable_maw_bank() -> None:
    state = _state(("maw_bank",))
    state = state.model_copy(
        update={
            "phase": RunPhase.SHOP,
            "shop": ShopState(
                node_id="shop",
                items=(
                    ShopItemState(
                        slot_id="slot:0",
                        item_id="fire_potion",
                        kind="potion",
                        price=0,
                        base_price=0,
                    ),
                ),
            ),
        }
    )
    state = step(state, _act(state, "shop_buy"))
    assert not state.flags.get("maw_bank_disabled")


def test_public_reward_pickup_chains_deck_and_gold_relics_once_per_added_copy() -> None:
    state = _state(("bing_bong", "lucky_fysh", "bowler_hat", "dragon_fruit", "book_of_five_rings"))
    state = state.model_copy(
        update={
            "phase": RunPhase.REWARD,
            "reward": RewardState(reward_id="card", source="combat", card_options=("guard",)),
            "flags": {**state.flags, "cards": ({"id": "guard", "type": "skill", "block": 5},)},
        }
    )
    before = len(state.master_deck)
    state = step(state, _act(state, "take_reward_card"))
    assert len(state.master_deck) == before + 2
    assert state.player.gold == 136  # Two 15-gold gains, each modified to 18.
    assert state.player.max_hp == 82  # One Dragon Fruit notification per card gain.
    assert state.flags["relic_counters"]["book_of_five_rings"] == 2
    assert len({card.instance_id for card in state.master_deck}) == len(state.master_deck)


def test_winged_boots_only_spends_on_unconnected_next_floor_paths_and_survives_reload() -> None:
    state = _state(("winged_boots",))
    assert {a.target_id for a in legal_actions(state) if a.type == "choose_node"} == {
        "room1_0",
        "room1_1",
    }
    state = step(state, _act(state, "choose_node", "room1_1"))
    assert state.flags["relic_counters"]["winged_boots"] == 2
    state = step(state, _act(state, "proceed"))
    state = load_state(serialize(state))
    state = step(state, _act(state, "choose_node", "room2_1"))
    assert state.flags["relic_counters"]["winged_boots"] == 2
    state = step(state, _act(state, "proceed"))
    state = step(state, _act(state, "choose_node", "room3_0"))
    assert state.flags["relic_counters"]["winged_boots"] == 1


def test_exhausted_boots_and_missing_relic_do_not_unlock_paths() -> None:
    for relics in ((), ("winged_boots",)):
        state = _state(relics)
        state = state.model_copy(
            update={"flags": {**state.flags, "relic_counters": {"winged_boots": 0}}}
        )
        assert [a.target_id for a in legal_actions(state) if a.type == "choose_node"] == ["room1_0"]


@pytest.mark.parametrize("outcome", ["event", "monster", "treasure", "shop"])
def test_question_room_resolution_preserves_map_icon_and_planisphere_heals_once(
    monkeypatch, outcome
) -> None:
    from sts2sim.mechanics.unknown_rooms import UnknownRoomResolution

    state = _state(("planisphere",), RoomKind.EVENT)
    state = state.model_copy(update={"flags": {**state.flags, "resolve_unknown_rooms": True}})
    monkeypatch.setattr(
        "sts2sim.engine.transitions.resolve_unknown_room",
        lambda *a, **k: UnknownRoomResolution(outcome, 0.5, {"monster": 0.2}),
    )
    state = step(state, _act(state, "choose_node"))
    assert state.player.hp == 45
    assert state.map.node_by_id[state.map.current_node_id].kind is RoomKind.EVENT
    assert state.flags["unknown_room_outcomes"][state.map.current_node_id] == outcome
    if outcome == "monster":
        assert state.phase is RunPhase.COMBAT
        assert _combat_reward_encounter(state) is EncounterType.NORMAL
    assert step(load_state(serialize(_state())), _act(_state(), "choose_node")).player.hp == 40


def test_juzu_blacklists_fights_without_advancing_their_pity(monkeypatch) -> None:
    from sts2sim.mechanics.unknown_rooms import resolve_unknown_room

    state = _state(("juzu_bracelet",), RoomKind.EVENT)
    state = state.model_copy(
        update={
            "flags": {
                **state.flags,
                "resolve_unknown_rooms": True,
                "unknown_room_odds_act": 1,
                "unknown_room_odds": {"monster": 0.9},
            }
        }
    )
    seen = []

    def capture(rng, **kwargs):
        seen.extend(kwargs["blacklisted"])
        return resolve_unknown_room(rng, **kwargs)

    monkeypatch.setattr("sts2sim.engine.transitions.resolve_unknown_room", capture)
    state = step(state, _act(state, "choose_node"))
    assert "monster" in seen
    assert state.flags["unknown_room_odds"]["monster"] == 0.9
    assert state.phase is not RunPhase.COMBAT


@pytest.mark.parametrize("relics", [
    ("ectoplasm", "dragon_fruit"),
    ("bowler_hat", "ectoplasm", "dragon_fruit"),
    ("ectoplasm", "bowler_hat", "dragon_fruit"),
])
def test_ectoplasm_prevents_gain_and_dragon_fruit_notification_through_reload(relics):
    state = _state(relics)
    state = state.model_copy(update={"phase": RunPhase.REWARD,
        "reward": RewardState(reward_id="gold", source="combat", gold=20)})
    state = step(load_state(serialize(state)), _act(state, "take_reward_gold"))
    assert state.player.gold == 100
    assert (state.player.max_hp, state.player.hp) == (80, 40)
    event = next(e for e in state.replay_log[-1].events if e.kind == "reward_gold_taken")
    assert event.amount == 0
    assert event.metadata["base_gain"] == 20
    assert not any(e.source_id == "dragon_fruit" for e in state.replay_log[-1].events)


def test_bowler_truncates_modified_gain_once_and_reward_replay_reports_actual_gain():
    state = _state(("bowler_hat", "dragon_fruit"))
    state = state.model_copy(update={"phase": RunPhase.REWARD,
        "reward": RewardState(reward_id="gold", source="combat", gold=3)})
    state = step(load_state(serialize(state)), _act(state, "take_reward_gold"))
    assert state.player.gold == 103
    assert (state.player.max_hp, state.player.hp) == (81, 41)
    assert [e.amount for e in state.replay_log[-1].events
            if e.kind == "reward_gold_taken"] == [3]


def test_ectoplasm_blocks_maw_bank_room_entry_gain_and_reward_card_gold():
    state = _state(("ectoplasm", "maw_bank", "lucky_fysh", "dragon_fruit"))
    state = step(state, _act(state, "choose_node"))
    assert (state.player.gold, state.player.max_hp) == (100, 80)
    state = state.model_copy(update={"phase": RunPhase.REWARD,
        "reward": RewardState(reward_id="card", source="combat", card_options=("guard",)),
        "flags": {**state.flags, "cards": ({"id": "guard", "type": "skill", "block": 5},)}})
    state = step(load_state(serialize(state)), _act(state, "take_reward_card"))
    assert (state.player.gold, state.player.max_hp) == (100, 80)
