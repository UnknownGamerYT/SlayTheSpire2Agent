from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from fakes import LocalGameRng, local_play_run, local_replay, rng_choice, rng_int
from helpers import call_with_supported_kwargs, canonical_digest, import_attr, jsonable

from sts2sim.engine.rng import (
    PYTHON_RANDOM_ALGORITHM,
    STS2_RUN_SEED_DERIVATION_BLOCKER,
    XOSHIRO256STARSTAR_ALGORITHM,
    XOSHIRO_ADAPTER_DERIVATION,
    GameRng,
    Xoshiro256StarStar,
    Xoshiro256StarStarState,
    capture_game_rng_state,
    game_rng_from_state,
    new_rng_state,
    random_from_seed,
    random_from_state,
    shuffled_with_state,
    xoshiro_state_from_seed,
)


def _rng_class() -> type[Any]:
    return import_attr(
        ("sts2sim.engine.rng", "sts2sim.mechanics.rng", "sts2sim.api.rng"),
        ("GameRng", "DeterministicRng", "Rng", "random_from_seed"),
    ) or LocalGameRng


def _make_rng(seed: int) -> Any:
    cls = _rng_class()
    try:
        return cls(seed=seed)
    except TypeError:
        return cls(seed)


def _play_run() -> Any:
    return import_attr(
        ("sts2sim.api", "sts2sim.api.run", "sts2sim.mechanics.run"),
        ("play_run", "run_episode", "run"),
    ) or local_play_run


def _replay() -> Any:
    return import_attr(
        ("sts2sim.api.replay", "sts2sim.replay"),
        ("replay", "replay_run", "verify_replay"),
    ) or local_replay


def test_rng_sequences_are_seed_deterministic() -> None:
    first = _make_rng(12345)
    second = _make_rng(12345)

    first_sequence = [
        rng_int(first, 17),
        rng_int(first, 17),
        rng_choice(first, ("attack", "skill", "power")),
        rng_int(first, 17),
    ]
    second_sequence = [
        rng_int(second, 17),
        rng_int(second, 17),
        rng_choice(second, ("attack", "skill", "power")),
        rng_int(second, 17),
    ]

    assert first_sequence == second_sequence == [5, 0, "power", 14]


def test_xoshiro256starstar_core_matches_reference_sequence() -> None:
    rng = Xoshiro256StarStar((1, 2, 3, 4))

    assert [rng.next_u64() for _ in range(8)] == [
        11520,
        0,
        1509978240,
        1215971899390074240,
        1216172134540287360,
        607988272756665600,
        16172922978634559625,
        8476171486693032832,
    ]


def test_default_rng_state_api_remains_python_random() -> None:
    rng = random_from_seed(2024)
    state = new_rng_state(2024)

    assert isinstance(rng, random.Random)
    assert state.algorithm == PYTHON_RANDOM_ALGORITHM
    assert random_from_state(state).randrange(1000) == random.Random(2024).randrange(1000)


def test_xoshiro_algorithm_can_be_selected_explicitly() -> None:
    rng = random_from_seed(
        2026,
        algorithm=XOSHIRO256STARSTAR_ALGORITHM,
        stream="deck_draw",
    )
    state = new_rng_state(
        2026,
        algorithm=XOSHIRO256STARSTAR_ALGORITHM,
        stream="deck_draw",
    )

    assert isinstance(rng, GameRng)
    assert isinstance(state, Xoshiro256StarStarState)
    assert state.to_dict() == {
        "algorithm": "xoshiro256**",
        "state": [
            8432963606182265559,
            15360152533090171292,
            3484917923720752021,
            13284436364907318534,
        ],
        "draws": 0,
        "seed": 2026,
        "stream": "deck_draw",
        "derivation": XOSHIRO_ADAPTER_DERIVATION,
    }
    assert [rng.random_int(1000) for _ in range(5)] == [216, 972, 419, 437, 896]


def test_xoshiro_named_streams_are_independent() -> None:
    deck = GameRng(seed=12345, stream="deck_draw")
    rewards = GameRng(seed=12345, stream="combat_rewards")
    events = GameRng(seed=12345, stream="events")

    deck_sequence = [deck.random_int(1000) for _ in range(8)]
    rewards_sequence = [rewards.random_int(1000) for _ in range(8)]
    for _ in range(50):
        events.random_int(1000)

    replayed_deck = GameRng(seed=12345, stream="deck_draw")

    assert deck_sequence == [299, 807, 159, 80, 865, 655, 267, 718]
    assert rewards_sequence == [10, 326, 798, 847, 642, 641, 115, 87]
    assert [replayed_deck.random_int(1000) for _ in range(8)] == deck_sequence


def test_xoshiro_state_serialization_replays_suffix() -> None:
    rng = GameRng(seed="ABC123", stream="events")
    prefix = [rng.next_u64() for _ in range(2)]
    payload = json.loads(json.dumps(capture_game_rng_state(rng).to_dict(), sort_keys=True))
    replay = game_rng_from_state(payload)

    original_suffix = [
        rng.next_u64(),
        rng.random_int(50),
        rng.choice(("event", "shop", "fight")),
    ]
    replayed_suffix = [
        replay.next_u64(),
        replay.random_int(50),
        replay.choice(("event", "shop", "fight")),
    ]

    assert prefix == [7630023243700927709, 7127656906987309852]
    assert payload == {
        "algorithm": "xoshiro256**",
        "state": [
            7564723191619479880,
            14038726065959088412,
            883727734062525318,
            944449525190712561,
        ],
        "draws": 2,
        "seed": "ABC123",
        "stream": "events",
        "derivation": XOSHIRO_ADAPTER_DERIVATION,
    }
    assert original_suffix == replayed_suffix == [10982864855384521119, 28, "event"]


def test_shuffled_with_xoshiro_state_replays_from_serialized_payload() -> None:
    state = new_rng_state(
        2026,
        algorithm=XOSHIRO256STARSTAR_ALGORITHM,
        stream="deck_draw",
    )
    assert isinstance(state, Xoshiro256StarStarState)

    first_shuffle, first_next_state = shuffled_with_state(tuple("abcdef"), state)
    second_shuffle, second_next_state = shuffled_with_state(
        tuple("abcdef"),
        json.loads(json.dumps(state.to_dict())),
    )

    assert first_shuffle == second_shuffle == ("e", "a", "c", "f", "d", "b")
    assert isinstance(first_next_state, Xoshiro256StarStarState)
    assert isinstance(second_next_state, Xoshiro256StarStarState)
    assert first_next_state.to_dict() == second_next_state.to_dict()


def test_sts2_run_seed_derivation_blocker_is_documented() -> None:
    state = xoshiro_state_from_seed(7, stream="deck_draw")

    assert state.algorithm == XOSHIRO256STARSTAR_ALGORITHM
    assert state.derivation == XOSHIRO_ADAPTER_DERIVATION
    assert "exact run-seed-to-stream derivation" in STS2_RUN_SEED_DERIVATION_BLOCKER
    assert "stable simulator derivation only" in STS2_RUN_SEED_DERIVATION_BLOCKER


def test_replay_round_trip_matches_recorded_transcript(tmp_path: Path) -> None:
    replay_path = tmp_path / "seed-2024.replay.json"

    first = call_with_supported_kwargs(
        _play_run(),
        seed=2024,
        policy="random",
        max_steps=12,
        output_path=replay_path,
        output=replay_path,
        replay_path=replay_path,
    )
    second = call_with_supported_kwargs(
        _replay(),
        replay_path=replay_path,
        path=replay_path,
        strict=True,
    )
    third = call_with_supported_kwargs(
        _play_run(),
        seed=2024,
        policy="random",
        max_steps=12,
    )

    first_payload = jsonable(first)
    second_payload = jsonable(second)
    third_payload = jsonable(third)

    assert replay_path.exists()
    assert canonical_digest(first_payload.get("transcript")) == canonical_digest(
        third_payload.get("transcript")
    )
    if "matched" in second_payload:
        assert second_payload["matched"] is True
    else:
        assert canonical_digest(first_payload.get("transcript")) == canonical_digest(
            second_payload.get("transcript")
        )
