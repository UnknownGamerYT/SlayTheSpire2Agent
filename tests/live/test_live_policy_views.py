from __future__ import annotations

from copy import deepcopy

import pytest
from live import test_live_ppo_policy as live_fixtures
from live.test_live_ppo_policy import Bridge, _combat, _play

from sts2sim.live_policy_views import import_policy_view

live_checkpoint = live_fixtures.live_checkpoint


def _screen(name, payload):
    raw = _combat()
    raw["state_type"] = name
    raw[name] = payload
    raw["player"]["can_remove_potions"] = True
    return raw


def _card(identity="BASH", index=0):
    return {"id": identity, "index": index, "is_upgraded": False}


SCREENS = [
    (
        "map",
        {
            "nodes": [
                {"col": 2, "row": 0, "type": "Ancient", "children": [[1, 1]]},
                {"col": 1, "row": 1, "type": "Monster", "children": []},
            ],
            "next_options": [{"index": 0, "col": 1, "row": 1, "type": "Monster"}],
            "visited": [],
        },
        "choose_map_node",
    ),
    (
        "event",
        {
            "event_id": "NEOW",
            "is_ancient": True,
            "options": [{"index": 1, "option_id": "GIFT", "relic_id": "VAJRA", "is_enabled": True}],
        },
        "choose_event_option",
    ),
    ("event", {"event_id": "NEOW", "in_dialogue": True}, "advance_dialogue"),
    (
        "event",
        {
            "event_id": "THIEVERY",
            "options": [{"index": 0, "option_id": "STEAL", "is_enabled": True}],
        },
        "choose_event_option",
    ),
    (
        "rest_site",
        {"options": [{"index": 1, "id": "SMITH", "is_enabled": True}]},
        "choose_rest_option",
    ),
    (
        "shop",
        {
            "items": [
                {
                    "index": 3,
                    "category": "card",
                    "price": 35,
                    "card_id": "BASH",
                    "card": _card(),
                    "is_stocked": True,
                    "can_afford": True,
                }
            ]
        },
        "shop_purchase",
    ),
    (
        "fake_merchant",
        {
            "shop": {
                "items": [
                    {
                        "index": 0,
                        "category": "relic",
                        "price": 35,
                        "relic_id": "VAJRA",
                        "is_stocked": True,
                        "can_afford": True,
                    }
                ]
            }
        },
        "shop_purchase",
    ),
    ("rewards", {"items": [{"index": 0, "type": "gold", "gold_amount": 17}]}, "claim_reward"),
    ("rewards", {"items": [{"index": 1, "type": "card"}]}, "claim_reward"),
    ("card_reward", {"cards": [_card()], "can_skip": False}, "select_card_reward"),
    (
        "card_reward",
        {"cards": [], "alternatives": [{"index": 1, "option_id": "REROLL", "is_enabled": True}]},
        "select_card_reward_alternative",
    ),
    (
        "card_reward",
        {
            "cards": [],
            "can_skip": True,
            "alternatives": [{"index": 0, "option_id": "Skip", "is_enabled": True}],
        },
        "skip_card_reward",
    ),
    ("relic_select", {"relics": [{"index": 0, "id": "VAJRA"}]}, "select_relic"),
    ("treasure", {"relics": [], "can_open_chest": True, "can_proceed": False}, "open_chest"),
    ("treasure", {"relics": [{"index": 0, "id": "VAJRA"}]}, "claim_treasure_relic"),
    ("card_select", {"cards": [_card()], "screen_type": "upgrade"}, "select_card"),
    (
        "card_select",
        {"cards": [], "preview_showing": True, "can_confirm": True},
        "confirm_selection",
    ),
    ("hand_select", {"cards": [_card()], "can_confirm": False}, "combat_select_card"),
    ("hand_select", {"cards": [], "can_confirm": True}, "combat_confirm_selection"),
    (
        "bundle_select",
        {"bundles": [{"index": 0, "cards": [_card(), _card("STRIKE_IRONCLAD", 1)]}]},
        "select_bundle",
    ),
    ("bundle_select", {"preview_showing": True, "can_confirm": True}, "confirm_bundle_selection"),
    (
        "crystal_sphere",
        {"tool": "small", "clickable_cells": [{"x": 1, "y": 2}]},
        "crystal_sphere_click_cell",
    ),
    ("crystal_sphere", {"tool": "none", "can_use_small_tool": True}, "crystal_sphere_set_tool"),
    ("rest_site", {"options": [], "can_proceed": True}, "proceed"),
]


@pytest.mark.parametrize(("name", "payload", "command"), SCREENS)
def test_visible_screen_commands_run_the_trained_policy_through_http(
    live_checkpoint, name, payload, command
):
    raw = _screen(name, payload)
    view = import_policy_view(raw)
    observation, info, commands = view.inputs()
    assert observation["counts"]["legal_actions"] == len(info["action_space"]) == 1
    assert commands == {0: {**next(iter(commands.values())), "action": command}}
    bridge = Bridge(raw, {"state_type": "game_over"})
    result = _play(bridge, live_checkpoint)
    assert bridge.actions == [commands[0]]
    assert result.steps_taken == 1
    assert result.steps[0].decision.policy == "masked_ppo"
    assert "policy_output" in result.steps[0].decision.policy_details


def test_disabled_controls_never_become_model_candidates_and_unopened_cards_stay_hidden():
    raw = _screen(
        "event",
        {
            "event_id": "TRIAL",
            "options": [
                {"index": 0, "option_id": "LOCKED", "is_enabled": True, "is_locked": True},
                {"index": 1, "option_id": "CHOSEN", "is_enabled": True, "was_chosen": True},
                {"index": 2, "option_id": "ACCEPT", "is_enabled": False},
                {"index": 3, "option_id": "CANCEL", "is_enabled": True},
            ],
        },
    )
    _, info, commands = import_policy_view(raw).inputs()
    assert commands == {0: {"action": "choose_event_option", "index": 3}}
    assert info["action_space"][0]["event_option"]["option_id"] == "CANCEL"
    raw = _screen("rewards", {"items": [{"index": 0, "type": "card"}]})
    _, info, _ = import_policy_view(raw).inputs()
    assert "card" not in info["action_space"][0]
    assert info["action_space"][0]["reward_choice"]["opens_selection"] is True


def test_visible_item_and_card_features_retain_actual_prices_modifiers_and_gold():
    raw = _screen(
        "shop",
        {
            "items": [
                {
                    "index": 2,
                    "category": "card",
                    "price": 23,
                    "on_sale": True,
                    "card_id": "BASH",
                    "card": {**_card(), "enchantment": {"id": "SHARP", "amount": 3}},
                    "is_stocked": True,
                    "can_afford": True,
                }
            ]
        },
    )
    _, info, _ = import_policy_view(raw).inputs()
    descriptor = info["action_space"][0]
    assert descriptor["item"]["price"] == 23
    assert descriptor["card"]["enchantments"][0]["amount"] == 3
    raw = _screen("rewards", {"items": [{"index": 4, "type": "gold", "gold_amount": 17}]})
    _, info, _ = import_policy_view(raw).inputs()
    assert info["action_space"][0]["reward_choice"]["gold"] == 17


def test_validation_oracle_and_private_future_fields_do_not_change_model_inputs():
    raw = _screen("card_reward", {"cards": [_card()]})
    original = import_policy_view(raw).inputs()
    mutated = deepcopy(raw)
    mutated["validation"] = {"player_rng": {"future": [1, 2, 3]}, "relic_bag": ["MONEY"]}
    mutated["debug"] = {"seed": "different", "future_reward": "rare"}
    assert import_policy_view(mutated).inputs() == original


@pytest.mark.parametrize(
    ("name", "payload", "match"),
    [
        (
            "card_reward",
            {"cards": [{"id": "NO_SUCH_CARD", "is_upgraded": False}]},
            "Unknown visible card",
        ),
        (
            "shop",
            {
                "items": [
                    {
                        "index": 0,
                        "category": "card",
                        "card_id": "BASH",
                        "is_stocked": True,
                        "can_afford": True,
                        "price": 3,
                    }
                ]
            },
            "card payload",
        ),
        (
            "card_reward",
            {
                "cards": [],
                "alternatives": [{"index": 0, "option_id": "UNKNOWN", "is_enabled": True}],
            },
            "Unknown source card reward",
        ),
    ],
)
def test_incomplete_or_unknown_visible_inputs_fail_closed(name, payload, match):
    with pytest.raises(ValueError, match=match):
        import_policy_view(_screen(name, payload))


def test_potion_popup_permission_and_sparse_slots_are_authoritative_for_world_controls():
    raw = _screen("rewards", {"items": [], "can_proceed": True})
    raw["player"]["potions"] = [
        {
            "id": "FRUIT_JUICE",
            "slot": 2,
            "can_discard": True,
            "can_use_outside_combat": True,
            "is_queued": False,
            "passes_custom_usability_check": True,
        }
    ]
    _, _, commands = import_policy_view(raw).inputs()
    assert {str(c) for c in commands.values()} == {
        str({"action": "proceed"}),
        str({"action": "discard_potion", "slot": 2}),
        str({"action": "use_potion", "slot": 2}),
    }
    raw["player"]["can_remove_potions"] = False
    assert import_policy_view(raw).inputs()[2] == {0: {"action": "proceed"}}


def test_known_ordinary_card_selector_fault_is_masked_while_special_card_remains_claimable():
    raw = _screen(
        "rewards",
        {
            "items": [
                {
                    "index": 0,
                    "type": "card",
                    "selector_can_skip": True,
                    "selector_can_reroll": True,
                },
                {"index": 1, "type": "special_card", "card": _card()},
            ]
        },
    )
    raw["player"]["relics"] = [{"id": "PAELS_WING"}, {"id": "DRIFTWOOD"}]
    _, _, commands = import_policy_view(raw).inputs()
    assert commands == {0: {"action": "claim_reward", "index": 1}}


def test_sphere_features_contain_uncovered_cells_without_private_item_bounds():
    raw = _screen(
        "crystal_sphere",
        {
            "tool": "small",
            "divinations_left": 2,
            "cells": [
                {"x": 0, "y": 1, "is_hidden": False, "item_type": "GoldSphereItem"},
                {"x": 3, "y": 3, "is_hidden": True, "item_type": "HiddenRareRelic"},
            ],
            "clickable_cells": [{"x": 3, "y": 3}],
            "revealed_items": [{"x": 0, "y": 1, "width": 3, "height": 3}],
        },
    )
    view = import_policy_view(raw)
    board = view.state.event.metadata["crystal_sphere_board"]
    assert board == {
        "remaining": 2,
        "cleared": [1],
        "visible_cells": [{"x": 0, "y": 1, "kind": "GoldSphereItem"}],
    }
    assert "revealed_items" not in board
