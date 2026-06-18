from __future__ import annotations

import json

import pytest

from sts2sim import (
    action_mask,
    action_space,
    decode_action,
    encode_observation,
    legal_actions,
    load_state,
    new_run,
    serialize,
    step,
)


def _action_payload(action):
    return action.model_dump(mode="json", exclude_none=True)


def test_action_space_and_mask_are_deterministic() -> None:
    state = new_run(seed=1, character_id="TEST", ascension=0)

    first_space = action_space(state)
    second_space = action_space(state)

    assert first_space == second_space
    assert [descriptor["id"] for descriptor in first_space] == list(range(len(first_space)))
    assert [descriptor["key"] for descriptor in first_space] == sorted(
        descriptor["key"] for descriptor in first_space
    )
    assert action_mask(state) == tuple(1 for _ in first_space)


def test_fixed_width_action_mask_pads_and_rejects_too_small() -> None:
    state = new_run(seed=2, character_id="TEST", ascension=0)
    action_count = len(action_space(state))

    assert action_mask(state, max_actions=action_count + 2) == (
        *(1 for _ in range(action_count)),
        0,
        0,
    )
    with pytest.raises(ValueError):
        action_mask(state, max_actions=action_count - 1)


def test_decode_action_maps_ids_to_legal_engine_actions() -> None:
    state = new_run(seed=3, character_id="TEST", ascension=0)
    legal_payloads = {_json_key(_action_payload(action)) for action in legal_actions(state)}

    for descriptor in action_space(state):
        action = decode_action(state, descriptor["id"])

        assert _json_key(_action_payload(action)) in legal_payloads
        assert _action_payload(action) == descriptor["action"]
        assert step(state, action).phase.value == "map"

    with pytest.raises(IndexError):
        decode_action(state, len(legal_payloads))


def test_action_ids_survive_state_serialization_roundtrip() -> None:
    state = new_run(seed=4, character_id="TEST", ascension=0)
    restored = load_state(serialize(state))

    assert action_space(restored) == action_space(state)
    assert action_mask(restored) == action_mask(state)


def test_encode_observation_is_json_friendly_and_has_numeric_vector() -> None:
    state = new_run(seed=5, character_id="TEST", ascension=0)

    observation = encode_observation(state)
    compact_observation = encode_observation(state, include_state=False)

    json.dumps(observation, sort_keys=True)
    assert "state" in observation
    assert "state" not in compact_observation
    assert len(observation["vector"]) == len(observation["vector_schema"])
    assert all(isinstance(value, int | float) for value in observation["vector"])
    assert observation["counts"]["legal_actions"] == len(action_space(state))
    assert observation["legal_actions"]["ids"] == [
        descriptor["id"] for descriptor in action_space(state)
    ]


def _json_key(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
