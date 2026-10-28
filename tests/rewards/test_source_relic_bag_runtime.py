"""Public queue persistence contracts from stable RelicFactory/RelicCmd.

The named entries are synthetic queue sentinels, not real item behavior tests.
"""

from __future__ import annotations

from combat.test_card_play_status_runtime import _reload, _simple, _state
from combat.test_glam import _play
from rooms.test_treasure_engine import _enter_treasure, _treasure_state

from sts2sim import legal_actions, serialize, step
from sts2sim.engine import MapEdgeState, MapNodeState, MapState, RoomKind, RunPhase
from sts2sim.mechanics.relic_bag_runtime import remove_obtained_source_relic


def _bags():
    bag = {
        rarity: (f"{rarity.upper()}_FRONT", f"{rarity.upper()}_MIDDLE", f"{rarity.upper()}_BACK")
        for rarity in ("Common", "Uncommon", "Rare", "Shop")
    }
    return {"player_relic_bag": bag, "shared_relic_bag": dict(bag)}


def _all_ids(bag):
    return {value.lower() for values in bag.values() for value in values}


def _act(state, kind):
    return step(state, next(action for action in legal_actions(state) if action.type == kind))


def _elite_reward():
    attack = _simple("kill", effects={"all_damage": 999}, card_type="attack", target="all_enemies")
    state = _state(attack)
    flags = {**state.flags, **_bags(), "combat_reward_encounter": "elite"}
    flags.pop("combat_reward_relic_count", None)
    state = state.model_copy(update={"flags": flags})
    return _play(_reload(state), "kill")


def test_elite_offer_pulls_front_and_removes_shared_copy_before_claim():
    result = _elite_reward()
    offered = result.reward.relic_ids[0]
    assert offered.endswith("_front")
    assert offered not in _all_ids(result.flags["player_relic_bag"])
    assert offered not in _all_ids(result.flags["shared_relic_bag"])
    assert serialize(_reload(result))["flags"]["player_relic_bag"] == (
        serialize(result)["flags"]["player_relic_bag"]
    )


def test_skipping_an_offered_relic_does_not_return_it_to_the_bag():
    result = _elite_reward()
    offered = result.reward.relic_ids[0]
    result = _act(_reload(result), "proceed")
    assert offered not in _all_ids(result.flags["player_relic_bag"])


def test_chest_offer_consumes_shared_bag_and_claim_consumes_player_bag():
    state = _treasure_state(seed=14)
    state = state.model_copy(update={"flags": {**state.flags, **_bags()}})
    result = _enter_treasure(_reload(state))
    offered = result.reward.relic_id
    assert offered.endswith("_front")
    assert offered not in _all_ids(result.flags["shared_relic_bag"])
    assert offered in _all_ids(result.flags["player_relic_bag"])
    result = _act(_reload(result), "take_reward_relic")
    assert offered not in _all_ids(result.flags["player_relic_bag"])


def test_shop_offer_pulls_back_and_persists_before_any_purchase():
    state = _state(_simple("wait"))
    start = MapNodeState(node_id="start", act=1, floor=0, lane=0, kind=RoomKind.START)
    shop = MapNodeState(node_id="shop", act=1, floor=1, lane=0, kind=RoomKind.SHOP)
    state = state.model_copy(update={
        "phase": RunPhase.MAP,
        "map": MapState(
            act=1, nodes=(start, shop), edges=(MapEdgeState(from_id="start", to_id="shop"),),
            current_node_id="start", completed_node_ids=("start",),
        ),
        "flags": {**state.flags, **_bags(), "shop_plan": {
            "colored_cards": 0, "colorless_cards": 0, "potions": 0,
            "relics": 1, "include_card_removal": False,
        }},
    })
    result = _act(_reload(state), "choose_node")
    offered = result.shop.items[0].item_id
    assert offered.endswith("_back")
    assert offered not in _all_ids(result.flags["player_relic_bag"])
    assert offered not in _all_ids(result.flags["shared_relic_bag"])
    assert serialize(_reload(result))["flags"]["player_relic_bag"] == (
        serialize(result)["flags"]["player_relic_bag"]
    )


def test_obtaining_fixed_relic_removes_it_from_every_bag_without_mutating_input():
    state = _state(_simple("wait"))
    state = state.model_copy(update={"flags": {**state.flags, **_bags()}})
    result = remove_obtained_source_relic(state, "common_middle")
    assert "common_middle" in _all_ids(state.flags["player_relic_bag"])
    for key in ("player_relic_bag", "shared_relic_bag"):
        assert "common_middle" not in _all_ids(result.flags[key])
