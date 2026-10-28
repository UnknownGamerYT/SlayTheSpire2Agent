"""Stable RNG primitives and normal-run persistence, independent of replay fixtures.

ASCII vectors were compiled from tckmn/sts2-seed-search@cedcb540's pinned
v0.107.1 C reference using gcc -O0 -fwrapv (64-bit unsigned long). The source
contract also matches zhiyue/sts2-rl-agent@5d874e8's decompiled MegaRandom,
StringHelper, Rng, RunRngSet and PlayerRngSet. See docs/stable_rng_runtime.md.
"""
from __future__ import annotations

import json
import random

import pytest
from pydantic import ValidationError

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import RunPhase, state_digest
from sts2sim.engine.models import RngState
from sts2sim.engine.rng import (
    PYTHON_RANDOM_ALGORITHM,
    STS2_STABLE_DERIVATION,
    XOSHIRO256STARSTAR_ALGORITHM,
    GameRng,
    RunRng,
    capture_random_state,
    model_random,
    random_from_state,
    random_stream,
    stable_gaussian_int,
    stable_seed_hash,
    stable_shuffle,
)


@pytest.mark.parametrize(("seed", "expected"), [
    ("", 757602046), ("A", 3452614621), ("AA", 3405071576),
    ("AAA", 2146173209), ("TEST", 4240313414),
    ("AAAAAAAAAA", 593659736), ("0P2ENNHM", 2147483648),
    ("act_selection", 3533043157), ("up_front", 3060220251),
    ("shuffle", 2308280675), ("monster_ai", 1703902611),
    ("rewards", 3835919258), ("NEOW", 348630327),
    ("unknown_map_point", 3896735940), ("shops", 2419795419),
    ("niche", 497466721), ("combat_targets", 1525931235),
])
def test_stable_hash_matches_unsigned_reference(seed: str, expected: int) -> None:
    assert stable_seed_hash(seed) == expected


def test_stable_hash_uses_utf16_code_units_including_surrogates_and_nul() -> None:
    # C# string chars are UTF-16, so a scalar outside the BMP equals its pair.
    assert stable_seed_hash("😀") == stable_seed_hash("\ud83d\ude00")
    assert stable_seed_hash("A\0B") != stable_seed_hash("A")


@pytest.mark.parametrize(("stream", "words", "outputs", "shuffled"), [
    (
        "rewards",
        (16590733663380428507, 16613422498590817128,
         343075632287685896, 11944545702289089340),
        (10052081551662425664, 782048160108802019, 15825693906367729051,
         17477113873447505972, 13464189201363469327),
        [1, 3, 9, 2, 8, 4, 7, 6, 0, 5],
    ),
    (
        "shuffle",
        (13707551629303515050, 11281776258543258139,
         14745421162853941325, 12150150232608050381),
        (13598621604126089151, 15657304458505398507, 290538478968541444,
         5184728912793045114, 9438136181577464957),
        [6, 4, 8, 2, 5, 3, 1, 0, 9, 7],
    ),
])
def test_stable_stream_initialization_draws_and_shuffle_match_c_reference(
    stream: str, words: tuple[int, ...], outputs: tuple[int, ...], shuffled: list[int]
) -> None:
    core = GameRng("TEST", stream, derivation=STS2_STABLE_DERIVATION)
    assert core.getstate().state == words
    assert tuple(core.next_u64() for _ in range(5)) == outputs

    bank = RunRng("TEST")
    view = random_stream(bank, stream)
    items = list(range(10))
    view.shuffle(items)
    assert items == shuffled
    assert capture_random_state(view).streams[stream].draws == 9
    assert view.randrange(1) == 0
    assert capture_random_state(view).streams[stream].draws == 10


def test_named_streams_reload_independently_including_unused_streams() -> None:
    first = RunRng("TEST")
    rewards = random_stream(first, "rewards")
    assert rewards.randrange(10) == 5
    random_stream(first, "shops").shuffle(list(range(15)))
    for _ in range(50):
        random_stream(first, "combat_targets").randrange(3)
    payload = json.loads(capture_random_state(first).model_dump_json())
    second = random_from_state(RngState.model_validate(payload))
    assert capture_random_state(second) == capture_random_state(first)
    for stream in ("rewards", "shops", "combat_targets", "shuffle"):
        assert [random_stream(first, stream).randrange(100) for _ in range(10)] == [
            random_stream(second, stream).randrange(100) for _ in range(10)
        ]
    # Combat targeting cannot change a previously unused shuffle stream.
    clean = RunRng("TEST")
    assert payload["streams"]["combat_targets"]["draws"] == 50
    fresh = random_from_state(payload)
    assert [random_stream(clean, "shuffle").randrange(100) for _ in range(5)] == [
        random_stream(fresh, "shuffle").randrange(100) for _ in range(5)
    ]


def test_getstate_setstate_preserves_existing_shared_views() -> None:
    bank = RunRng("TEST")
    view = random_stream(bank, "shuffle")
    saved = bank.getstate()
    expected = [view.randrange(1000) for _ in range(6)]
    bank.setstate(saved)
    assert [view.randrange(1000) for _ in range(6)] == expected
    assert capture_random_state(bank).streams["shuffle"].draws == 6


def test_model_rng_is_repeatable_and_does_not_consume_run_streams() -> None:
    bank = RunRng("TEST")
    before = capture_random_state(bank)
    first = model_random(bank, "NEOW")
    second = model_random(bank, "NEOW")
    reference = random_stream(RunRng("TEST"), "NEOW")
    expected = [reference.randrange(1000) for _ in range(10)]
    assert [first.randrange(1000) for _ in range(10)] == expected
    assert [second.randrange(1000) for _ in range(10)] == expected
    assert capture_random_state(bank) == before
    assert model_random(bank, "NEOW", player_slot=1).random() != model_random(bank, "NEOW").random()
    assert model_random(bank, "NEOW", mixin=1).random() == model_random(
        bank, "NEOW", player_slot=1
    ).random()


def test_stable_shuffle_sorts_before_shuffling_and_gaussian_rounds_even() -> None:
    assert stable_shuffle([4, 2, 3, 1], RunRng("TEST")) == stable_shuffle(
        [1, 4, 2, 3], RunRng("TEST")
    )
    class FixedRandom(random.Random):
        def random(self) -> float:
            return 0.0
    rng = FixedRandom()
    assert stable_gaussian_int(rng, 3, 2, 1, 4) == 3
    assert stable_gaussian_int(rng, 4, 0, 1, 4) == 4
    with pytest.raises(ValueError, match="Gaussian"):
        stable_gaussian_int(rng, 4, 0, 1, 3)


def test_normal_seed_normalization_and_explicit_legacy_configuration() -> None:
    first = new_run(123, "IRONCLAD", 0)
    second = new_run("123", "IRONCLAD", 0)
    assert first.rng == second.rng
    assert first.rng.seed == "123"
    assert first.rng.algorithm == XOSHIRO256STARSTAR_ALGORITHM
    assert first.rng.derivation == STS2_STABLE_DERIVATION
    assert first.rng.fidelity == "source_decompiled"
    legacy = new_run(123, "IRONCLAD", 0, rng_algorithm=PYTHON_RANDOM_ALGORITHM)
    fixture = new_run(123, "TEST", 0)
    assert legacy.rng.algorithm == fixture.rng.algorithm == PYTHON_RANDOM_ALGORITHM
    assert new_run(123, "TEST", 0, rng_algorithm="xoshiro256**").rng.algorithm == "xoshiro256**"
    assert new_run(123, "IRONCLAD", 0, source_data={
        "rng_algorithm": PYTHON_RANDOM_ALGORITHM
    }).rng.algorithm == PYTHON_RANDOM_ALGORITHM
    with pytest.raises(ValueError, match="unsupported RNG algorithm"):
        new_run(123, "IRONCLAD", 0, rng_algorithm="fake")


def test_legacy_payload_round_trip_keeps_original_shape_and_sequence() -> None:
    rng = random.Random(2024)
    rng.random()
    legacy = capture_random_state(rng).model_dump(mode="json")
    assert set(legacy) == {"algorithm", "version", "internal_state", "gauss_next"}
    legacy_without_label = {k: v for k, v in legacy.items() if k != "algorithm"}
    restored = RngState.model_validate(legacy_without_label)
    assert restored.model_dump(mode="json") == legacy
    actual = random_from_state(restored)
    assert [rng.randrange(1000) for _ in range(10)] == [actual.randrange(1000) for _ in range(10)]
    state = new_run(2024, "IRONCLAD", 0, rng_algorithm=PYTHON_RANDOM_ALGORITHM)
    payload = serialize(state)
    assert serialize(load_state(payload)) == payload


@pytest.mark.parametrize("character", ["IRONCLAD", "SILENT", "REGENT", "NECROBINDER", "DEFECT"])
def test_normal_public_run_reload_and_following_actions_match(character: str) -> None:
    state = new_run("0P2ENNHM", character, 0)
    policy = random.Random(1051)
    for _ in range(45):
        payload = json.loads(json.dumps(serialize(state)))
        restored = load_state(payload)
        assert state_digest(state) == state_digest(restored)
        assert state.rng.algorithm == XOSHIRO256STARSTAR_ALGORITHM
        actions = legal_actions(state)
        if not actions or state.phase in {RunPhase.FAILED, RunPhase.COMPLETE}:
            break
        action = policy.choice(actions)
        first, second = step(state, action), step(restored, action)
        assert state_digest(first) == state_digest(second)
        state = first


@pytest.mark.parametrize("field,value", [
    ("internal_state", [0, 0, 0, 0]), ("internal_state", [1, 2, 3]),
    ("internal_state", [1, 2, 3, 1 << 64]), ("draws", -1), ("version", 2),
])
def test_corrupt_runtime_states_are_rejected(field: str, value: object) -> None:
    payload = capture_random_state(RunRng("TEST")).model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValidationError):
        RngState.model_validate(payload)
