"""Source physical slots remain visible after earlier slots become empty."""
from __future__ import annotations

import json

import pytest

from sts2sim import load_state, new_run, serialize
from sts2sim.agent_api import _first_potion_slot_id, _potion_id_for_slot, encode_observation
from sts2sim.learning.observation import encode_rich_observation
from sts2sim.mechanics.trigger_visibility import potion_slot_summary
from sts2sim.policy_visibility import policy_state, public_run_flags


def _state():
    state = new_run(seed=4, character_id="TEST", ascension=0)
    return state.model_copy(update={
        "potions": ("fire_potion", "block_potion"),
        "flags": {**state.flags, "potion_slots": 4, "potion_slot_indices": (0, 2)},
    })


def test_capacity_summary_exposes_each_hole_at_its_actual_position():
    summary = potion_slot_summary(serialize(_state()))
    assert summary["capacity"] == 4
    assert summary["filled"] == 2
    assert summary["empty"] == 2
    assert summary["potions"] == [
        {"slot_index": 0, "id": "fire_potion", "empty": False},
        {"slot_index": 1, "id": "", "empty": True},
        {"slot_index": 2, "id": "block_potion", "empty": False},
        {"slot_index": 3, "id": "", "empty": True},
    ]


def test_actor_and_rich_inventory_positions_use_physical_indices():
    state = _state()
    actor = encode_observation(state)
    rich = encode_rich_observation(state)
    expected = [
        {"potion_id": "fire_potion", "slot_index": 0, "position": 0},
        {"potion_id": "block_potion", "slot_index": 2, "position": 2},
    ]
    assert actor["positions"]["potions"] == expected
    assert rich["player"]["potion_slots"] == expected
    assert rich["run"]["flags"]["potion_slot_indices"] == [0, 2]


def test_slot_lookup_does_not_alias_a_hole_to_packed_inventory_index():
    state = _state()
    assert _potion_id_for_slot(state, "potion:1") is None
    assert _potion_id_for_slot(state, "potion:2") == "block_potion"
    assert _potion_id_for_slot(state, "potion:-1") is None
    assert _first_potion_slot_id(state, "block_potion") == "potion:2"


def test_policy_projection_and_roundtrip_preserve_public_slot_identities():
    state = _state()
    restored = load_state(json.loads(json.dumps(serialize(state))))
    for value in (state, restored, policy_state(state)):
        assert _potion_id_for_slot(value, "potion:2") == "block_potion"
        assert tuple(public_run_flags(value.flags)["potion_slot_indices"]) == (0, 2)


def test_invalid_authoritative_slot_map_fails_instead_of_repacking():
    payload = serialize(_state())
    payload["flags"]["potion_slot_indices"] = [2, 2]
    with pytest.raises(ValueError, match="distinct"):
        potion_slot_summary(payload)
