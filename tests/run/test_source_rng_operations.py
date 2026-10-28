"""C reference vectors for float32 and selection (same pinned core as runtime tests)."""
from __future__ import annotations

import struct

import pytest

from sts2sim.engine.rng import RunRng, capture_random_state, random_stream
from sts2sim.mechanics.source_rng import float32, source_float, source_sample, source_weighted_item


def _bits(value: float) -> int:
    return struct.unpack('<I', struct.pack('<f', value))[0]


def test_next_float_matches_independent_c_binary32_reference() -> None:
    bank = RunRng('TEST')
    rng = random_stream(bank, 'shops')
    assert [_bits(source_float(rng)) for _ in range(8)] == [
        1051187040, 1058478410, 1057404317, 1044710581,
        992202933, 1046692076, 1040195103, 1063696328,
    ]
    assert capture_random_state(bank).streams['shops'].draws == 8
    rng = random_stream(RunRng('TEST'), 'shops')
    assert [_bits(source_float(rng, .95, 1.05)) for _ in range(8)] == [
        1065064337, 1065428906, 1065375201, 1064837150,
        1064518549, 1064886687, 1064724263, 1065689802,
    ]


def test_selection_removes_in_order_and_consumes_singleton_draw() -> None:
    bank = RunRng('TEST')
    rng = random_stream(bank, 'shops')
    assert source_sample(rng, list(range(10)), 4) == [3, 6, 5, 1]
    assert capture_random_state(bank).streams['shops'].draws == 4
    assert source_sample(rng, [42], 1) == [42]
    assert capture_random_state(bank).streams['shops'].draws == 5
    assert source_sample(rng, [], 0) == []
    with pytest.raises(ValueError):
        source_sample(rng, [42], 2)


def test_float_bound_overloads_and_boolean_use_one_draw() -> None:
    bank = RunRng('TEST')
    assert bank.next_float(5) == source_float(RunRng('TEST'), 0, 5)
    assert bank.next_float(4, 4) == 4
    assert capture_random_state(bank).draws == 2
    assert bank.next_bool() in {True, False}
    assert capture_random_state(bank).draws == 3


def test_weighted_selection_uses_float32_and_source_boundary() -> None:
    class Fixed:
        def random(self) -> float:
            return 0.5
    assert source_weighted_item(Fixed(), ['a', 'b'], lambda _: .1) == 'a'
    assert float32(.1 + .2) == float32(.3)
    with pytest.raises(ValueError):
        source_weighted_item(Fixed(), ['a'], lambda _: 0)
