from __future__ import annotations

import hashlib
import math
import operator
import random
from collections.abc import Iterable, Mapping, MutableSequence, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, overload

from sts2sim.mechanics.source_rng import source_float, source_sample

from .models import RngState, RngStreamState

T = TypeVar("T")
Seed = int | str
PythonRandomAlgorithm = Literal["python.random.MT19937"]
XoshiroAlgorithm = Literal["xoshiro256**"]

PYTHON_RANDOM_ALGORITHM: PythonRandomAlgorithm = "python.random.MT19937"
XOSHIRO256STARSTAR_ALGORITHM: XoshiroAlgorithm = "xoshiro256**"
XOSHIRO_ADAPTER_DERIVATION = "sts2sim.sha256-splitmix64.v1"
STS2_SOURCE_PORT_DERIVATION = "sts2.v0.107.1.ascii-djb2-splitmix64.source-port"
STS2_STABLE_DERIVATION = "sts2.v0.107.1.utf16-djb2-splitmix64"
STS2_RNG_SOURCE_URL = (
    "https://github.com/tckmn/sts2-seed-search/blob/"
    "cedcb540eca0c5393b883cd847d975df5e97bf64/cleaned-v0.107.1.c"
)
STS2_RUN_SEED_DERIVATION_BLOCKER = (
    "Version-pinned decompiled v0.107.1 source documents UTF-16 displayed seeds, "
    "uint32 stream mixing and sample-based bounded draws. This exact run-seed-to-stream "
    "derivation has not been checked against an independent installed stable game. "
    "The original SHA256 adapter remains a stable simulator derivation only."
)
STS2_STABLE_SOURCE_ROOT = (
    "https://github.com/zhiyue/sts2-rl-agent/blob/"
    "5d874e855ae1d0550d5c30416fadd7dbb0d133b7/decompiled/"
)

_MASK_64 = (1 << 64) - 1
_SPLITMIX64_INCREMENT = 0x9E3779B97F4A7C15


@dataclass(frozen=True, slots=True)
class Xoshiro256StarStarState:
    """Serializable state for the v0.107.1 xoshiro256** core.

    The PRNG step matches xoshiro256**. The seed-to-stream derivation below is a
    stable sts2sim adapter until captured/local game sources reveal STS2's exact
    run-seed derivation.
    """

    state: tuple[int, int, int, int]
    draws: int = 0
    seed: Seed | None = None
    stream: str = "root"
    algorithm: XoshiroAlgorithm = XOSHIRO256STARSTAR_ALGORITHM
    derivation: str = XOSHIRO_ADAPTER_DERIVATION

    def __post_init__(self) -> None:
        if self.algorithm != XOSHIRO256STARSTAR_ALGORITHM:
            raise ValueError(f"unsupported xoshiro algorithm: {self.algorithm!r}")
        if self.draws < 0:
            raise ValueError("draws must be non-negative")
        object.__setattr__(self, "state", _normalized_xoshiro_state(self.state))

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "state": list(self.state),
            "draws": self.draws,
            "seed": self.seed,
            "stream": self.stream,
            "derivation": self.derivation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Xoshiro256StarStarState:
        raw_state = payload.get("state")
        if not isinstance(raw_state, Sequence) or isinstance(raw_state, (str, bytes)):
            raise TypeError("xoshiro state payload must contain a sequence state")

        raw_seed = payload.get("seed")
        if raw_seed is not None and not isinstance(raw_seed, (int, str)):
            raise TypeError("xoshiro seed must be an int, str, or None")

        raw_algorithm = str(payload.get("algorithm", XOSHIRO256STARSTAR_ALGORITHM))
        if raw_algorithm != XOSHIRO256STARSTAR_ALGORITHM:
            raise ValueError(f"unsupported xoshiro algorithm: {raw_algorithm!r}")

        return cls(
            state=_normalized_xoshiro_state(tuple(_payload_int(value) for value in raw_state)),
            draws=_payload_int(payload.get("draws", 0)),
            seed=raw_seed,
            stream=str(payload.get("stream", "root")),
            algorithm=XOSHIRO256STARSTAR_ALGORITHM,
            derivation=str(payload.get("derivation", XOSHIRO_ADAPTER_DERIVATION)),
        )


class Xoshiro256StarStar:
    """Small, dependency-free xoshiro256** implementation."""

    algorithm: XoshiroAlgorithm = XOSHIRO256STARSTAR_ALGORITHM

    def __init__(self, state: Sequence[int]) -> None:
        self._state = list(_normalized_xoshiro_state(state))

    @classmethod
    def from_seed_material(cls, seed_material: int) -> Xoshiro256StarStar:
        return cls(_splitmix64_words(seed_material, 4))

    @classmethod
    def from_seed(cls, seed: Seed, stream: str = "root") -> Xoshiro256StarStar:
        return cls.from_seed_material(_seed_material(seed, stream))

    def state_tuple(self) -> tuple[int, int, int, int]:
        return tuple(self._state)  # type: ignore[return-value]

    def next_u64(self) -> int:
        s0, s1, s2, s3 = self._state
        result = (_rotl((s1 * 5) & _MASK_64, 7) * 9) & _MASK_64
        t = (s1 << 17) & _MASK_64

        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = _rotl(s3, 45)

        self._state = [s0 & _MASK_64, s1 & _MASK_64, s2 & _MASK_64, s3 & _MASK_64]
        return result


class GameRng:
    """v0.107.1-aware RNG adapter with named streams and stable replay state."""

    algorithm: XoshiroAlgorithm = XOSHIRO256STARSTAR_ALGORITHM

    def __init__(
        self,
        seed: Seed = 0,
        stream: str = "root",
        state: Xoshiro256StarStarState | Mapping[str, object] | Sequence[int] | None = None,
        *,
        derivation: str = XOSHIRO_ADAPTER_DERIVATION,
    ) -> None:
        snapshot = (
            xoshiro_state_from_seed(seed, stream, derivation=derivation)
            if state is None
            else _coerce_xoshiro_state(state)
        )
        self.seed = snapshot.seed
        self.stream = snapshot.stream
        self.derivation = snapshot.derivation
        self._draws = snapshot.draws
        self._core = Xoshiro256StarStar(snapshot.state)

    @classmethod
    def from_state(
        cls, state: Xoshiro256StarStarState | Mapping[str, object] | Sequence[int]
    ) -> GameRng:
        return cls(state=state)

    @property
    def draws(self) -> int:
        return self._draws

    def getstate(self) -> Xoshiro256StarStarState:
        return Xoshiro256StarStarState(
            state=self._core.state_tuple(),
            draws=self._draws,
            seed=self.seed,
            stream=self.stream,
            derivation=self.derivation,
        )

    def setstate(
        self, state: Xoshiro256StarStarState | Mapping[str, object] | Sequence[int]
    ) -> None:
        snapshot = _coerce_xoshiro_state(state)
        self.seed = snapshot.seed
        self.stream = snapshot.stream
        self.derivation = snapshot.derivation
        self._draws = snapshot.draws
        self._core = Xoshiro256StarStar(snapshot.state)

    def fork(self, stream: str) -> GameRng:
        if self.seed is None:
            raise ValueError("cannot fork a xoshiro stream without the original seed")
        child_stream = f"{self.stream}/{stream}" if self.stream else stream
        return type(self)(seed=self.seed, stream=child_stream, derivation=self.derivation)

    def next_u64(self) -> int:
        self._draws += 1
        return self._core.next_u64()

    def getrandbits(self, k: int) -> int:
        bits = operator.index(k)
        if bits < 0:
            raise ValueError("number of bits must be non-negative")
        if bits == 0:
            return 0

        value = 0
        generated = 0
        while generated < bits:
            value = (value << 64) | self.next_u64()
            generated += 64
        return value >> (generated - bits)

    def random(self) -> float:
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def randbelow(self, upper: int) -> int:
        bound = operator.index(upper)
        if bound <= 0:
            raise ValueError("upper must be positive")

        if self.derivation in {STS2_SOURCE_PORT_DERIVATION, STS2_STABLE_DERIVATION}:
            return int(self.random() * bound)

        bits = (bound - 1).bit_length()
        while True:
            candidate = self.getrandbits(bits)
            if candidate < bound:
                return candidate

    def random_int(self, upper: int) -> int:
        return self.randbelow(upper)

    def randrange(self, start: int, stop: int | None = None, step: int = 1) -> int:
        start_value = operator.index(start)
        stop_value = operator.index(stop) if stop is not None else start_value
        if stop is None:
            start_value = 0
        step_value = operator.index(step)
        if step_value == 0:
            raise ValueError("zero step for randrange()")

        choices = range(start_value, stop_value, step_value)
        if len(choices) == 0:
            raise ValueError("empty range for randrange()")
        return choices[self.randbelow(len(choices))]

    def randint(self, a: int, b: int) -> int:
        return self.randrange(a, operator.index(b) + 1)

    def choice(self, options: Sequence[T]) -> T:
        if not options:
            raise ValueError("options must not be empty")
        return options[self.randbelow(len(options))]

    def shuffle(self, items: MutableSequence[T]) -> None:
        for index in range(len(items) - 1, 0, -1):
            swap_index = self.randbelow(index + 1)
            items[index], items[swap_index] = items[swap_index], items[index]

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.random()


DeterministicRng = GameRng
Rng = GameRng


def xoshiro_state_from_seed(
    seed: Seed, stream: str = "root", *, derivation: str = XOSHIRO_ADAPTER_DERIVATION
) -> Xoshiro256StarStarState:
    if derivation in {STS2_SOURCE_PORT_DERIVATION, STS2_STABLE_DERIVATION}:
        if not isinstance(seed, str):
            raise ValueError("stable derivation requires a displayed seed string")
        if derivation == STS2_SOURCE_PORT_DERIVATION and (not seed.isascii() or "\0" in seed):
            raise ValueError("source-port derivation supports ASCII seeds without NUL only")
        material = (
            stable_seed_hash(seed) + (stable_seed_hash(stream) if stream else 0)
        ) & 0xFFFFFFFF
    elif derivation == XOSHIRO_ADAPTER_DERIVATION:
        material = _seed_material(seed, stream)
    else:
        raise ValueError(f"unsupported xoshiro seed derivation: {derivation!r}")
    return Xoshiro256StarStarState(
        state=Xoshiro256StarStar.from_seed_material(material).state_tuple(),
        seed=seed,
        stream=stream,
        derivation=derivation,
    )


def stable_seed_hash(value: str) -> int:
    """Unsigned StringHelper.GetDeterministicHashCode over C# UTF-16 chars."""
    first = second = 352654597
    encoded = value.encode("utf-16-le", errors="surrogatepass")
    for index in range(0, len(encoded), 2):
        char = int.from_bytes(encoded[index : index + 2], "little")
        if index % 4:
            second = ((second * 33) ^ char) & 0xFFFFFFFF
        else:
            first = ((first * 33) ^ char) & 0xFFFFFFFF
    return (first + second * 1566083941) & 0xFFFFFFFF


class RunRng(random.Random):
    """Random-compatible runtime with serialized independent xoshiro streams.

    All views share the stream bank; capturing any view captures the complete
    bank. Stream names and call-site consumption remain separately reviewable.
    Python's sample/choices helpers remain simulator algorithms.
    """

    run_seed: Seed | None
    gauss_next: float | None

    def __init__(self, seed: Seed = 0, *, derivation: str | None = None) -> None:
        super().__init__(0)
        # The game accepts a displayed string seed. This API's integer shorthand
        # denotes its decimal displayed string, e.g. 123 means the seed "123".
        self.run_seed = str(seed)
        self.derivation = derivation or STS2_STABLE_DERIVATION
        self._active_stream = "root"
        self._streams = {"root": GameRng(self.run_seed, "", derivation=self.derivation)}

    @classmethod
    def from_snapshot(cls, state: RngState) -> RunRng:
        if state.algorithm != XOSHIRO256STARSTAR_ALGORITHM:
            raise ValueError("RunRng requires a xoshiro runtime state")
        if state.derivation not in {
            XOSHIRO_ADAPTER_DERIVATION,
            STS2_SOURCE_PORT_DERIVATION,
            STS2_STABLE_DERIVATION,
        }:
            raise ValueError(f"unsupported xoshiro seed derivation: {state.derivation!r}")
        # Restoring words is independent of seed conversion. The seed is needed
        # only if a previously unused stream is requested later.
        rng = cls.__new__(cls)
        random.Random.__init__(rng, 0)
        rng.run_seed = state.seed
        rng.derivation = state.derivation
        rng._active_stream = "root"
        rng._streams = {}
        all_states = {
            "root": RngStreamState(internal_state=state.internal_state, draws=state.draws),  # type: ignore[arg-type]
            **state.streams,
        }
        for name, stream_state in all_states.items():
            rng._streams[name] = GameRng.from_state(
                Xoshiro256StarStarState(
                    state=stream_state.internal_state,
                    draws=stream_state.draws,
                    seed=state.seed,
                    stream="" if name == "root" else name,
                    derivation=state.derivation,
                )
            )
        rng.gauss_next = state.gauss_next
        return rng

    def for_stream(self, stream: str) -> RunRng:
        name = stream or "root"
        if name not in self._streams:
            if self.run_seed is None:
                raise ValueError("cannot create a new stream without the original run seed")
            self._streams[name] = GameRng(self.run_seed, name, derivation=self.derivation)
        view = self.__class__.__new__(self.__class__)
        random.Random.__init__(view, 0)
        view.run_seed = self.run_seed
        view.derivation = self.derivation
        view._streams = self._streams
        view._active_stream = name
        return view

    def snapshot(self) -> RngState:
        root = self._streams["root"].getstate()
        return RngState(
            algorithm=XOSHIRO256STARSTAR_ALGORITHM,
            version=1,
            internal_state=root.state,
            draws=root.draws,
            seed=self.run_seed,
            derivation=self.derivation,
            fidelity=(
                "source_decompiled"
                if self.derivation == STS2_STABLE_DERIVATION
                else "source_port"
                if self.derivation == STS2_SOURCE_PORT_DERIVATION
                else "simulator_seeded"
            ),
            routing="partial_source_singleplayer",
            gauss_next=self.gauss_next,
            streams={
                name: RngStreamState(internal_state=rng.getstate().state, draws=rng.draws)
                for name, rng in self._streams.items()
                if name != "root"
            },
        )

    def random(self) -> float:
        return self._streams[self._active_stream].random()

    def getrandbits(self, k: int) -> int:
        return self._streams[self._active_stream].getrandbits(k)

    def next_float(self, minimum_or_maximum: float = 1.0, maximum: float | None = None) -> float:
        if maximum is None:
            return source_float(self, 0.0, minimum_or_maximum)
        return source_float(self, minimum_or_maximum, maximum)

    def next_bool(self) -> bool:
        return self.randrange(2) == 0

    def sample(
        self, population: Sequence[T], k: int, *, counts: Iterable[int] | None = None
    ) -> list[T]:
        if counts is not None:
            raise ValueError("game source sampling does not support repeated population counts")
        return source_sample(self, population, k)

    def _randbelow(self, n: int) -> int:
        return self._streams[self._active_stream].randbelow(n)

    def getstate(self) -> tuple[Any, ...]:
        return (100, self.snapshot().model_dump(), self._active_stream)

    def setstate(self, state: tuple[Any, ...]) -> None:
        if len(state) != 3 or state[0] != 100:
            raise ValueError("RunRng.setstate requires a complete xoshiro runtime snapshot")
        restored = self.from_snapshot(RngState.model_validate(state[1]))
        active_stream = str(state[2])
        if active_stream not in restored._streams:
            raise ValueError("restored active stream is missing from the stream bank")
        if (restored.run_seed, restored.derivation) != (self.run_seed, self.derivation):
            raise ValueError(
                "cannot replace a shared RNG bank with a different run seed or derivation"
            )
        # Preserve the bank identity for existing stream views.
        self._streams.clear()
        self._streams.update(restored._streams)
        self.run_seed = restored.run_seed
        self.derivation = restored.derivation
        self.gauss_next = restored.gauss_next
        self._active_stream = active_stream


def random_stream(rng: random.Random, stream: str) -> random.Random:
    """Select a shared named stream, retaining single-stream legacy MT behavior."""
    return rng.for_stream(stream) if isinstance(rng, RunRng) else rng


def model_random(
    rng: random.Random, model_id: str, *, player_slot: int = 0, mixin: int = 0
) -> random.Random:
    """Independent Rng(player, ModelId, mixin), without consuming the run bank.

    Pass the exact ModelId.Entry (normally upper snake case). Recreating the
    same model RNG intentionally starts the sequence again, as in the game.
    """
    if not isinstance(rng, RunRng):
        return rng
    if rng.run_seed is None:
        raise ValueError("model RNG requires the original displayed run seed")
    if rng.derivation != STS2_STABLE_DERIVATION:
        raise ValueError("model RNG requires the stable game seed derivation")
    seed = (
        stable_seed_hash(str(rng.run_seed)) + player_slot + stable_seed_hash(model_id) + mixin
    ) & 0xFFFFFFFF
    result = RunRng(str(rng.run_seed))
    result._streams["root"] = GameRng.from_state(
        Xoshiro256StarStarState(
            state=Xoshiro256StarStar.from_seed_material(seed).state_tuple(),
            seed=str(rng.run_seed),
            stream=model_id,
            derivation=STS2_STABLE_DERIVATION,
        )
    )
    return result


def stable_gaussian_int(
    rng: random.Random, mean: int, std_dev: int, minimum: int, maximum: int
) -> int:
    """v0.107.1 Rng.NextGaussianInt: reject outside bounds, ties-to-even round."""
    if minimum > maximum or std_dev < 0 or (std_dev == 0 and not minimum <= mean <= maximum):
        raise ValueError("invalid bounded Gaussian parameters")
    while True:
        first = 1.0 - rng.random()
        second = 1.0 - rng.random()
        sample = math.sqrt(-2.0 * math.log(first)) * math.sin(2.0 * math.pi * second)
        value = round(mean + std_dev * sample)
        if minimum <= value <= maximum:
            return value


def stable_shuffle(items: Sequence[T], rng: random.Random, *, key: Any = None) -> list[T]:
    """Sort before Fisher-Yates, mirroring ListExtensions.StableShuffle."""
    result = sorted(items, key=key)
    rng.shuffle(result)
    return result


def run_random_from_seed(
    seed: Seed, *, algorithm: str = XOSHIRO256STARSTAR_ALGORITHM
) -> random.Random:
    if algorithm == XOSHIRO256STARSTAR_ALGORITHM:
        return RunRng(seed)
    if algorithm == PYTHON_RANDOM_ALGORITHM:
        return random.Random(seed)
    raise ValueError(f"unsupported RNG algorithm: {algorithm!r}")


def capture_game_rng_state(rng: GameRng | Xoshiro256StarStar) -> Xoshiro256StarStarState:
    if isinstance(rng, GameRng):
        return rng.getstate()
    return Xoshiro256StarStarState(state=rng.state_tuple())


def game_rng_from_state(
    state: Xoshiro256StarStarState | Mapping[str, object] | Sequence[int],
) -> GameRng:
    return GameRng.from_state(state)


@overload
def random_from_seed(seed: Seed) -> random.Random: ...


@overload
def random_from_seed(
    seed: Seed,
    *,
    algorithm: PythonRandomAlgorithm,
    stream: str = "root",
) -> random.Random: ...


@overload
def random_from_seed(
    seed: Seed,
    *,
    algorithm: XoshiroAlgorithm,
    stream: str = "root",
) -> GameRng: ...


def random_from_seed(
    seed: Seed,
    *,
    algorithm: str = PYTHON_RANDOM_ALGORITHM,
    stream: str = "root",
) -> random.Random | GameRng:
    if algorithm == PYTHON_RANDOM_ALGORITHM:
        return random.Random(seed)
    if algorithm == XOSHIRO256STARSTAR_ALGORITHM:
        return GameRng(seed=seed, stream=stream)
    raise ValueError(f"unsupported RNG algorithm: {algorithm!r}")


def capture_random_state(rng: random.Random) -> RngState:
    if isinstance(rng, RunRng):
        return rng.snapshot()
    version, internal_state, gauss_next = rng.getstate()
    return RngState(
        version=version,
        internal_state=tuple(int(value) for value in internal_state),
        gauss_next=gauss_next,
    )


def capture_rng_state(rng: random.Random | GameRng) -> RngState | Xoshiro256StarStarState:
    if isinstance(rng, GameRng):
        return capture_game_rng_state(rng)
    return capture_random_state(rng)


@overload
def random_from_state(state: RngState) -> random.Random: ...


@overload
def random_from_state(state: Xoshiro256StarStarState) -> GameRng: ...


@overload
def random_from_state(state: Mapping[str, object]) -> random.Random | GameRng: ...


def random_from_state(
    state: RngState | Xoshiro256StarStarState | Mapping[str, object],
) -> random.Random | GameRng:
    algorithm = _state_algorithm(state)
    if algorithm == XOSHIRO256STARSTAR_ALGORITHM:
        if isinstance(state, RngState):
            return RunRng.from_snapshot(state)
        if isinstance(state, Mapping) and "internal_state" in state:
            return RunRng.from_snapshot(RngState.model_validate(state))
        return game_rng_from_state(state)
    if algorithm != PYTHON_RANDOM_ALGORITHM:
        raise ValueError(f"unsupported RNG algorithm: {algorithm!r}")

    rng = random.Random()
    rng.setstate(
        (
            int(_state_value(state, "version")),
            tuple(int(value) for value in _state_value(state, "internal_state")),
            _state_value(state, "gauss_next"),
        )
    )
    return rng


@overload
def new_rng_state(seed: Seed) -> RngState: ...


@overload
def new_rng_state(
    seed: Seed,
    *,
    algorithm: PythonRandomAlgorithm,
    stream: str = "root",
) -> RngState: ...


@overload
def new_rng_state(
    seed: Seed,
    *,
    algorithm: XoshiroAlgorithm,
    stream: str = "root",
) -> Xoshiro256StarStarState: ...


def new_rng_state(
    seed: Seed,
    *,
    algorithm: str = PYTHON_RANDOM_ALGORITHM,
    stream: str = "root",
) -> RngState | Xoshiro256StarStarState:
    if algorithm == PYTHON_RANDOM_ALGORITHM:
        return capture_random_state(random.Random(seed))
    if algorithm == XOSHIRO256STARSTAR_ALGORITHM:
        return xoshiro_state_from_seed(seed, stream)
    raise ValueError(f"unsupported RNG algorithm: {algorithm!r}")


def shuffled_with_state(
    items: Sequence[T],
    state: RngState | Xoshiro256StarStarState | Mapping[str, object],
) -> tuple[tuple[T, ...], RngState | Xoshiro256StarStarState]:
    rng = random_from_state(state)
    shuffled = list(items)
    rng.shuffle(shuffled)
    return tuple(shuffled), capture_rng_state(rng)


def _rotl(value: int, shift: int) -> int:
    return ((value << shift) & _MASK_64) | (value >> (64 - shift))


def _splitmix64_words(seed_material: int, count: int) -> tuple[int, ...]:
    state = seed_material & _MASK_64
    words: list[int] = []
    for _ in range(count):
        state = (state + _SPLITMIX64_INCREMENT) & _MASK_64
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
        words.append((value ^ (value >> 31)) & _MASK_64)
    return tuple(words)


def _seed_material(seed: Seed, stream: str) -> int:
    seed_kind = "int" if isinstance(seed, int) else "str"
    payload = (f"{XOSHIRO_ADAPTER_DERIVATION}\0seed:{seed_kind}:{seed}\0stream:{stream}").encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _normalized_xoshiro_state(state: Sequence[int]) -> tuple[int, int, int, int]:
    if len(state) != 4:
        raise ValueError("xoshiro256** state must contain exactly four 64-bit words")
    words = tuple(int(value) & _MASK_64 for value in state)
    if not any(words):
        raise ValueError("xoshiro256** all-zero state is invalid")
    return words  # type: ignore[return-value]


def _coerce_xoshiro_state(
    state: Xoshiro256StarStarState | Mapping[str, object] | Sequence[int],
) -> Xoshiro256StarStarState:
    if isinstance(state, Xoshiro256StarStarState):
        return state
    if isinstance(state, Mapping):
        return Xoshiro256StarStarState.from_dict(state)
    return Xoshiro256StarStarState(state=_normalized_xoshiro_state(state))


def _payload_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError("xoshiro state words must be integers")


def _state_algorithm(state: RngState | Xoshiro256StarStarState | Mapping[str, object]) -> str:
    if isinstance(state, Mapping):
        return str(state.get("algorithm", PYTHON_RANDOM_ALGORITHM))
    return str(getattr(state, "algorithm", PYTHON_RANDOM_ALGORITHM))


def _state_value(state: RngState | Xoshiro256StarStarState | Mapping[str, object], key: str) -> Any:
    if isinstance(state, Mapping):
        return state[key]
    return getattr(state, key)
