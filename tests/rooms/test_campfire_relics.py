from __future__ import annotations

import json

import pytest
from combat.test_fishing_rod import _run

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import MapEdgeState, MapNodeState, MapState, RoomKind, RunPhase


def _enter_rest(state):
    start = MapNodeState(node_id="start", act=1, floor=state.floor, lane=0, kind=RoomKind.START)
    rest = MapNodeState(node_id="rest", act=1, floor=state.floor + 1, lane=0, kind=RoomKind.REST)
    game_map = MapState(
        act=1,
        nodes=(start, rest),
        edges=(MapEdgeState(from_id="start", to_id="rest"),),
        current_node_id="start",
        completed_node_ids=("start",),
    )
    state = state.model_copy(update={"map": game_map, "phase": RunPhase.MAP})
    return step(state, next(a for a in legal_actions(state) if a.type == "choose_node"))


def _act(state, kind):
    return step(state, next(action for action in legal_actions(state) if action.type == kind))


@pytest.mark.parametrize("hp", [10, 78, 80])
def test_rest_combines_regal_pillow_and_stone_humidifier_with_hp_caps(hp: int) -> None:
    state = _run(relics=("REGAL_PILLOW", "STONE_HUMIDIFIER"))
    state = state.model_copy(update={"player": state.player.model_copy(update={"hp": hp})})
    state = _act(_enter_rest(state), "rest")
    assert state.player.max_hp == 85
    assert state.player.hp == min(80, hp + 24 + 15) + 5


@pytest.mark.parametrize(("cards", "healed"), [(4, 0), (5, 3), (9, 3), (10, 6)])
def test_eternal_feather_heals_once_on_entry_for_each_complete_five_cards(
    cards: int, healed: int
) -> None:
    state = _run(relics=("eternal_feather", "miniature_tent"))
    deck = (state.master_deck * 2)[:cards]
    state = state.model_copy(
        update={
            "player": state.player.model_copy(update={"hp": 10}),
            "master_deck": tuple(
                card.model_copy(update={"instance_id": f"deck:{index}"})
                for index, card in enumerate(deck)
            ),
        }
    )
    state = _enter_rest(state)
    assert state.player.hp == 10 + healed
    state = _act(state, "smith")
    assert state.player.hp == 10 + healed


def test_miniature_tent_allows_each_option_once_and_persists_used_options() -> None:
    state = _run(relics=("miniature_tent", "pumpkin_candle", "shovel", "girya"))
    state = state.model_copy(
        update={"flags": {**state.flags, "campfire_dig_relic_pool": ("anchor",)}}
    )
    state = _enter_rest(state)
    for kind in ("rest", "smith", "kindle", "dig", "lift"):
        state = _act(state, kind)
        assert state.phase == RunPhase.REST
        assert not any(a.type == kind for a in legal_actions(state))
        state = load_state(json.loads(json.dumps(serialize(state))))
        assert not any(a.type == kind for a in legal_actions(state))
    assert state.flags["relic_counters"]["pumpkin_candle"] == 10
    assert state.flags["girya_lift_count"] == 1
    assert state.flags["campfire_dig_count"] == 1
    state = _act(state, "proceed")
    assert state.phase == RunPhase.MAP
    assert state.room_history.count("rest") == 1
    state = _enter_rest(state)
    assert any(a.type == "rest" for a in legal_actions(state))
    assert any(a.type == "kindle" for a in legal_actions(state))


@pytest.mark.parametrize("tent", [False, True])
def test_rest_offers_dream_catcher_card_and_two_mailbox_potions_then_returns(tent: bool) -> None:
    relics = ("dream_catcher", "tiny_mailbox") + (("miniature_tent",) if tent else ())
    state = _run(relics=relics)
    state = state.model_copy(
        update={
            "flags": {
                **state.flags,
                "card_pool": (
                    {"id": "REST_CARD", "rarity": "Common", "color": "test", "type": "Skill"},
                ),
                "event_reward_potion_pool": ("fire_potion",),
            }
        }
    )
    state = _act(_enter_rest(state), "rest")
    assert state.phase == RunPhase.REWARD
    assert state.reward.card_options == ("rest_card",)
    assert state.reward.potion_ids == ("fire_potion", "fire_potion")
    state = _act(state, "take_reward_card")
    state = _act(state, "take_reward_potion")
    state = _act(state, "take_reward_potion")
    assert state.potions == ("fire_potion", "fire_potion")
    state = _act(state, "proceed")
    assert state.phase == (RunPhase.REST if tent else RunPhase.MAP)
    if tent:
        assert not any(a.type == "rest" for a in legal_actions(state))
        state = _act(state, "proceed")
    assert state.room_history.count("rest") == 1


def test_mailbox_potions_stay_available_when_belt_is_full() -> None:
    state = _run(relics=("tiny_mailbox",))
    state = state.model_copy(
        update={
            "potions": ("fire_potion", "fire_potion", "fire_potion"),
            "flags": {**state.flags, "event_reward_potion_pool": ("skill_potion",)},
        }
    )
    state = _act(_enter_rest(state), "rest")
    assert len(state.reward.potion_ids) == 2
    assert not any(a.type == "take_reward_potion" for a in legal_actions(state))
    state = _act(state, "discard_potion")
    state = _act(state, "take_reward_potion")
    assert "skill_potion" in state.potions


def test_smith_does_not_trigger_rest_only_relics() -> None:
    state = _run(relics=("dream_catcher", "tiny_mailbox", "stone_humidifier", "regal_pillow"))
    state = state.model_copy(update={"player": state.player.model_copy(update={"hp": 20})})
    state = _act(_enter_rest(state), "smith")
    assert state.phase == RunPhase.MAP
    assert state.player.hp == 20
    assert state.player.max_hp == 80
    assert state.reward is None
    assert not state.potions


def test_shovel_dig_applies_relic_pickup_effects() -> None:
    state = _run(relics=("shovel",))
    state = state.model_copy(
        update={"flags": {**state.flags, "campfire_dig_relic_pool": ("strawberry",)}}
    )
    state = _act(_enter_rest(state), "dig")
    assert "strawberry" in state.relics
    assert state.player.max_hp == 87
    assert state.player.hp == 87
