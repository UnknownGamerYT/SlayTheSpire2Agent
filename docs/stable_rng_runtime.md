# Stable v0.107.1 RNG runtime

Normal runs now use xoshiro256** and serialized named streams. The old
`python.random.MT19937` algorithm remains available explicitly, and saved MT
states continue to load with their original JSON shape and sequence. The custom
`TEST` character keeps MT by default for existing synthetic fixtures.

```python
from sts2sim import new_run

state = new_run("TEST", "IRONCLAD", 0)  # actual displayed string seed
legacy = new_run(123, "IRONCLAD", 0, rng_algorithm="python.random.MT19937")
```

Integer API seeds denote their decimal displayed string (`123` means `"123"`).
String inputs are passed to the run seed hash exactly, matching `RunRngSet`.
The simulator does not silently apply the game's lobby UI canonicalization
(uppercase, O→0, I→1, trim). The separate existing `GameRng` utility retains its
original simulator SHA256 derivation and rejection-based bounded integer draws
unless the stable derivation is explicitly selected; normal runs use `RunRng`.

## Evidence and implemented primitives

The pinned [stable decompilation provenance](https://github.com/zhiyue/sts2-rl-agent/blob/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/GAME_VERSIONS.md)
identifies v0.107.1, public build 23811903, release commit `59260271`, and assembly
hash `-1555940892`. This is a third party's recorded provenance, not an
independently obtained installed DLL in this workspace.

The implementation follows that snapshot's:

- [StringHelper](https://github.com/zhiyue/sts2-rl-agent/blob/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/decompiled/MegaCrit.Sts2.Core.Helpers/StringHelper.cs): alternating 32-bit djb2-xor accumulators over UTF-16 code units.
- [MegaRandom](https://github.com/zhiyue/sts2-rl-agent/blob/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/decompiled/MegaCrit.Sts2.Core.Random/MegaRandom.cs): uint32 seed zero-extended to SplitMix64; four words initialize xoshiro256**; double output is `(word >> 11) / 2**53`; bounded integers are `floor(double * bound)`.
- [Rng](https://github.com/zhiyue/sts2-rl-agent/blob/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/decompiled/MegaCrit.Sts2.Core.Random/Rng.cs): named seed is `uint32(base + hash(name))`; player/model seed is `uint32(base + player_slot + hash(ModelId.Entry) + mixin)`; Gaussian integer sampling uses two fresh doubles, sine Box–Muller, ties-to-even rounding and bounded rejection.
- [RunRngSet](https://github.com/zhiyue/sts2-rl-agent/blob/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/decompiled/MegaCrit.Sts2.Core.Runs/RunRngSet.cs), [PlayerRngSet](https://github.com/zhiyue/sts2-rl-agent/blob/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/decompiled/MegaCrit.Sts2.Core.Random/PlayerRngSet.cs) and [Player.InitializeSeed](https://github.com/zhiyue/sts2-rl-agent/blob/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/decompiled/MegaCrit.Sts2.Core.Entities.Players/Player.cs): run names use snake case; per-player streams add the player slot index before name mixing. The normal simulator currently uses slot zero.
- [ListExtensions](https://github.com/zhiyue/sts2-rl-agent/blob/5d874e855ae1d0550d5c30416fadd7dbb0d133b7/decompiled/MegaCrit.Sts2.Core.Extensions/ListExtensions.cs): descending Fisher–Yates; stable shuffle sorts first. Singleton bounded draws consume one core step, while singleton shuffles consume none.

The independent [v0.107.1 C source port](https://github.com/tckmn/sts2-seed-search/blob/cedcb540eca0c5393b883cd847d975df5e97bf64/cleaned-v0.107.1.c)
agrees for its ASCII input domain. `tests/run/test_runtime_rng.py` retains hash,
seed-word, u64 and shuffle vectors compiled from that C core, including negative
signed hashes and 32-bit overflow. The tests also exercise all five characters
through the public action API with reload after each action.

## Persistence and routing

Runtime `RngState` explicitly records the algorithm, derivation, primitive
fidelity, routing scope, original displayed seed, four root words, root draw
count, and each used named stream's words and count. Root is a simulator
fallback for remaining unclassified random call sites. `random_stream` returns
a shared view: capturing any view captures the entire bank. JSON reload restores
all used streams directly and can derive unused streams later. These are
simulator snapshots, not the game's counter-only save format.

Shared helpers route opening/discard shuffles, unknown-room rolls, shops,
rewards, transformations, random card creation and selection, random costs,
random orbs, and random targets to their source categories. `model_random`
creates the source's independent per-model RNG without consuming the run bank;
callers must use the exact `ModelId.Entry` and avoid replacing the saved run bank
with the ephemeral model RNG. Legacy MT deliberately retains one stream.

`fidelity="source_decompiled"` identifies the implemented primitive/seed contract.
`routing="partial_source_singleplayer"` explicitly qualifies the broader
integration. It does not claim exact whole-run seeded equivalence. Every
source call's collection ordering, eligibility, player/model mixins, float32
rounding, counters and number of draws still need complete independent review.
`source_sample` and `RunRng.sample` model explicit repeated `NextItem`/`Remove`
loops. The separate `source_take_random` models `IEnumerableExtensions.TakeRandom`:
shuffle the entire input, then take the requested prefix. It consumes `n-1`
draws even for a zero-sized result; a singleton shuffle consumes none. Combat
`GetDistinctForCombat` and Tinker offers use the latter. Card reward factories
and Reflections use explicit loops. Independent C vectors distinguish these
contracts and verify both their outputs and counters.

There are no remaining `choices` or `uniform` callers under the simulator source.
MT utility helpers retain their original behavior. An end-to-end v0.107.1 game
checkpoint or action trace remains necessary to establish complete seed parity.
