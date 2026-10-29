"""Public stable chest preparation/opening, hidden offers and ordered gold gains."""
from __future__ import annotations

import pytest

from sts2sim import (
    action_space,
    encode_observation,
    legal_actions,
    load_state,
    new_run,
    serialize,
    step,
)
from sts2sim.agents.policy import StrategicAgent
from sts2sim.engine import CardInstance, IllegalActionError, MapEdgeState, MapNodeState, MapState
from sts2sim.engine.models import Action, ActionType, RunPhase
from sts2sim.engine.source_treasure_runtime import PENDING
from sts2sim.policy_visibility import policy_payload


def _before(*, ascension=0, relics=(), character="IRONCLAD", first_run=False, quests=0):
    state = new_run("SOURCE_CHEST", character, ascension,
                    source_data={"relics": relics})
    act = 2 if quests else 1
    start = MapNodeState(node_id="start", kind="start", act=act, floor=0, lane=0)
    chest = MapNodeState(node_id="chest", kind="treasure", act=act, floor=1, lane=0)
    world = MapState(act=act, nodes=(start, chest),
        edges=(MapEdgeState(from_id="start", to_id="chest"),), current_node_id="start",
        completed_node_ids=("start",))
    flags = dict(state.flags)
    if first_run:
        flags["number_of_runs"] = 0
    if quests:
        flags.update(spoils_map_target_act=act, spoils_map_target_node_id="chest",
                     spoils_map_reward_gold=600)
    deck = state.master_deck + tuple(CardInstance(instance_id=f"quest:{i}", card_id="spoils_map",
        type="quest", target="self", cost=None) for i in range(quests))
    return state.model_copy(update={"phase": RunPhase.MAP, "act": act, "floor": 0,
        "ancient": None, "map": world, "room_history": (), "replay_log": (),
        "master_deck": deck, "player": state.player.model_copy(update={"gold": 0}),
        "flags": flags})


def _act(state, kind):
    action = next(a for a in legal_actions(state) if a.type == kind)
    result = step(state, action)
    assert serialize(result) == serialize(step(load_state(serialize(state)), action))
    return result


def _draws(state, stream):
    return state.rng.streams[stream].draws if stream in state.rng.streams else 0


def test_source_entry_prepares_shared_relic_but_no_gold_or_visible_reward():
    before = _before()
    state = _act(before, "choose_node")
    assert state.phase == RunPhase.TREASURE and state.reward is None
    assert state.player.gold == 0
    assert _draws(state, "rewards") == _draws(before, "rewards")
    assert _draws(state, "treasure_room_relics") == _draws(before, "treasure_room_relics") + 1
    assert state.flags[PENDING]["relic_id"]
    assert {a.type for a in legal_actions(state)} == {ActionType.OPEN_CHEST}
    assert state.flags["player_relic_bag"] == before.flags["player_relic_bag"]
    assert state.flags["shared_relic_bag"] != before.flags["shared_relic_bag"]
    assert not state.flags.get("treasure_chests_opened")


@pytest.mark.parametrize("relics", [(), ("silver_crucible",)])
def test_source_ui_disables_proceed_until_chest_opens_even_if_rewards_are_empty(relics):
    state = _act(_before(quests=1, relics=relics), "choose_node")
    before = serialize(state)
    for snapshot in (state, load_state(before)):
        assert {a.type for a in legal_actions(snapshot)} == {ActionType.OPEN_CHEST}
        with pytest.raises(IllegalActionError):
            step(snapshot, Action(type=ActionType.PROCEED))
    assert serialize(state) == before
    assert state.player.gold == 0
    assert any(card.card_id == "spoils_map" for card in state.master_deck)
    assert not state.flags.get("spoils_map_completed")
    assert not state.flags.get("treasure_chests_opened")


def test_revealed_relic_can_be_skipped_after_immediate_chest_gold():
    before = _act(_before(), "choose_node")
    offered = before.flags[PENDING]["relic_id"]
    opened = _act(before, "open_chest")
    skipped = _act(opened, "proceed")
    assert skipped.phase == RunPhase.MAP and skipped.rng == opened.rng
    assert skipped.player.gold == opened.player.gold > 0
    assert offered not in skipped.relics and skipped.reward is None
    assert skipped.flags["shared_relic_bag"] == before.flags["shared_relic_bag"]
    assert PENDING not in skipped.flags


@pytest.mark.parametrize("ascension", [0, 3, 10])
def test_open_chest_grants_gold_immediately_once_without_skippable_gold_entry(ascension):
    before = _act(_before(ascension=ascension), "choose_node")
    state = _act(before, "open_chest")
    assert _draws(state, "rewards") == _draws(before, "rewards") + 1
    assert state.reward.gold == 0 and state.reward.relic_id
    assert state.player.gold == state.reward.metadata["chest_gold_already_granted"]
    assert (42 <= state.player.gold <= 52) if ascension < 3 else (31 <= state.player.gold <= 39)
    assert not any(a.type == "take_reward_gold" for a in legal_actions(state))
    assert state.flags["treasure_chests_opened"] == 1
    with pytest.raises(IllegalActionError):
        step(state, Action(type=ActionType.OPEN_CHEST))


@pytest.mark.parametrize("character", ["IRONCLAD", "SILENT"])
def test_first_profile_treasure_forces_gorget_but_still_rolls_rarity(character):
    before = _before(character=character, first_run=True)
    pending = _act(before, "choose_node")
    assert _draws(pending, "treasure_room_relics") == _draws(before, "treasure_room_relics") + 1
    assert pending.flags[PENDING]["relic_id"] == "gorget"
    assert all("GORGET" not in values for values in pending.flags["shared_relic_bag"].values())
    opened = _act(pending, "open_chest")
    assert opened.reward.relic_id == "gorget" and opened.reward.metadata["relic_rarity"] == "common"


def test_silver_crucible_suppresses_first_entry_relic_and_open_gold_and_spoils():
    before = _before(relics=("silver_crucible",), quests=1)
    pending = _act(before, "choose_node")
    assert pending.flags["silver_crucible_treasure_rooms_entered"] == 1
    assert pending.flags[PENDING]["empty"]
    assert pending.rng == before.rng
    opened = _act(pending, "open_chest")
    assert opened.rng == before.rng and opened.player.gold == 0
    assert opened.reward.relic_id is None
    assert any(c.card_id == "spoils_map" for c in opened.master_deck)
    assert not opened.flags.get("spoils_map_completed")


def test_spoils_copies_complete_after_chest_gold_as_separate_source_gain_commands():
    state = _act(_before(relics=("bowler_hat", "dragon_fruit"), quests=2), "choose_node")
    maximum = state.player.max_hp
    assert sum(c.card_id == "spoils_map" for c in state.master_deck) == 2
    opened = _act(state, "open_chest")
    base = opened.reward.metadata["chest_gold_already_granted"]
    assert opened.player.gold == base * 125 // 100 + 2 * 750
    assert opened.player.max_hp == maximum + 3
    assert not any(c.card_id == "spoils_map" for c in opened.master_deck)
    assert opened.flags["spoils_map_completed"]
    events = opened.replay_log[-1].events
    gains = [e for e in events if e.kind in {"treasure_gold_gained", "spoils_map_redeemed"}]
    assert [e.kind for e in gains] == ["treasure_gold_gained", "spoils_map_redeemed",
                                      "spoils_map_redeemed"]
    assert sum(e.kind == "event_card_removed" for e in events) == 2


def test_closed_chest_actor_does_not_leak_prepared_relic_and_strategy_opens():
    first = _act(_before(), "choose_node")
    hidden = {**first.flags[PENDING], "relic_id": "anchor", "relic_rarity": "rare"}
    second = first.model_copy(update={"flags": {**first.flags, PENDING: hidden}})
    assert policy_payload(serialize(first)) == policy_payload(serialize(second))
    assert encode_observation(first) == encode_observation(second)
    assert action_space(first) == action_space(second)
    decision = StrategicAgent().choose_action(first)
    assert decision.chosen.action["type"] == "open_chest"
    opened_first, opened_second = _act(first, "open_chest"), _act(second, "open_chest")
    assert encode_observation(opened_first) != encode_observation(opened_second)


def test_source_spoils_created_after_act_two_does_not_invent_an_act_three_quest():
    from sts2sim.engine import RewardState
    from sts2sim.engine.transitions import _card_from_spec

    state = _before().model_copy(update={"act": 2, "phase": RunPhase.REWARD,
        "reward": RewardState(reward_id="late_spoils", source="event", card_ids=("spoils_map",))})
    state = _act(state, "take_reward_card")
    assert not state.flags.get("spoils_map_pending_act")
    assert state.master_deck[-1].custom["spoils_act_index"] == 1
    card = _card_from_spec({"id": "SPOILS_MAP", "type": "Quest", "rarity": "Quest"}, 0)
    assert card.custom["spoils_act_index"] == 1
