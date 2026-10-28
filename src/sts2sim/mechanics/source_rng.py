"""v0.107.1 numeric RNG operations, independent of engine/model imports.

The C# contracts are pinned to zhiyue/sts2-rl-agent@5d874e855ae1d0550d5c30416fadd7dbb0d133b7:
Rng.NextFloat/WeightedNextItem and ListExtensions. Python floats store the
exact result of each explicit binary32 rounding below.
"""
from __future__ import annotations

import math
import struct
from collections.abc import Callable, Mapping, Sequence
from random import Random
from typing import Protocol, TypeVar

T = TypeVar("T")


class FloatRandom(Protocol):
    def random(self) -> float: ...


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]  # type: ignore[no-any-return]


def source_float(rng: FloatRandom, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Rng.NextFloat: binary32 bounds/subtraction, double sample, binary32 cast."""
    lower, upper = float32(minimum), float32(maximum)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("float RNG minimum must not exceed maximum and bounds must be finite")
    width = float32(upper - lower)
    return float32(rng.random() * width + lower)


def is_source_rng(rng: object) -> bool:
    """Explicit legacy MT/adapter configurations retain their old behavior."""
    return getattr(rng, "derivation", None) == "sts2.v0.107.1.utf16-djb2-splitmix64"


def source_weighted_item(
    rng: FloatRandom, items: Sequence[T], weight: Callable[[T], float]
) -> T:
    """Rng.WeightedNextItem, including float32 subtraction and <= boundaries."""
    if not items:
        raise ValueError("weighted items must not be empty")
    weights = [float32(weight(item)) for item in items]
    if any(value < 0 or not math.isfinite(value) for value in weights) or not any(weights):
        raise ValueError("weights must be finite, nonnegative and have a positive total")
    # Enumerable.Sum(float) accumulates in double before returning a float.
    total = float32(sum(weights))
    remaining = float32(source_float(rng) * total)
    for item, amount in zip(items, weights, strict=True):
        remaining = float32(remaining - amount)
        if remaining <= 0:
            return item
    return items[-1]


def source_sample(rng: Random, items: Sequence[T], count: int) -> list[T]:
    """Repeated NextItem+Remove; input order and singleton draws are retained."""
    if count < 0 or count > len(items):
        raise ValueError("sample count must be between zero and population size")
    available = list(items)
    return [available.pop(rng.randrange(len(available))) for _ in range(count)]

def draw_source_potion(
    rng: Random, options: Sequence[str], rarities: Mapping[str, str] | None = None
) -> str:
    if not options:
        raise ValueError("potion options must not be empty")
    if not is_source_rng(rng) or rarities is None:
        return rng.choice(options)
    sample = source_float(rng)
    rarity = "rare" if sample <= float32(0.1) else (
        "uncommon" if sample <= float32(0.35) else "common"
    )
    candidates = [potion for potion in options if rarities.get(potion, "common").lower() == rarity]
    if not candidates:
        raise ValueError(f"no {rarity} potion in the source creation pool")
    return rng.choice(candidates)

