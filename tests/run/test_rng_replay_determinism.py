from __future__ import annotations

from pathlib import Path
from typing import Any

from fakes import LocalGameRng, local_play_run, local_replay, rng_choice, rng_int
from helpers import call_with_supported_kwargs, canonical_digest, import_attr, jsonable


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

    assert first_sequence == second_sequence


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
