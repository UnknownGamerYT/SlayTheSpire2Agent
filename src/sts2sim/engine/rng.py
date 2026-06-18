from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar

from .models import RngState

T = TypeVar("T")
Seed = int | str


def random_from_seed(seed: Seed) -> random.Random:
    return random.Random(seed)


def capture_random_state(rng: random.Random) -> RngState:
    version, internal_state, gauss_next = rng.getstate()
    return RngState(
        version=version,
        internal_state=tuple(int(value) for value in internal_state),
        gauss_next=gauss_next,
    )


def random_from_state(state: RngState) -> random.Random:
    rng = random.Random()
    rng.setstate((state.version, tuple(state.internal_state), state.gauss_next))
    return rng


def new_rng_state(seed: Seed) -> RngState:
    return capture_random_state(random_from_seed(seed))


def shuffled_with_state(items: Sequence[T], state: RngState) -> tuple[tuple[T, ...], RngState]:
    rng = random_from_state(state)
    shuffled = list(items)
    rng.shuffle(shuffled)
    return tuple(shuffled), capture_random_state(rng)

