"""State adapters for source relic deques; callers persist the returned flags."""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from random import Random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sts2sim.engine.models import RunState

from .relic_bags import RelicBagContext, pull_relic_from_bag, remove_bag_relics


def source_relic_rarity(rng: Random) -> str:
    """RelicFactory.RollRarity: one NextFloat, Common <.5, Uncommon <.83."""
    value = struct.unpack("<f", struct.pack("<f", rng.random()))[0]
    threshold = struct.unpack("<f", struct.pack("<f", 0.83))[0]
    return "common" if value < 0.5 else "uncommon" if value < threshold else "rare"


def _bag(flags: Mapping[str, Any], key: str) -> dict[str, tuple[str, ...]]:
    raw = flags.get(key)
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(rarity): tuple(str(value) for value in values)
        for rarity, values in raw.items()
        if isinstance(values, (list, tuple))
    }


def has_source_relic_bag(state: RunState, *, shared: bool = False) -> bool:
    key = "shared_relic_bag" if shared else "player_relic_bag"
    return isinstance(state.flags.get(key), Mapping)


def draw_source_relic(
    state: RunState,
    flags: dict[str, Any],
    rarity: str,
    *,
    shared: bool = False,
    from_back: bool = False,
    for_shop: bool = False,
    excluded_ids: Sequence[str] = (),
) -> str:
    """Offer a deque relic and persist removal even when the offer is skipped."""
    key = "shared_relic_bag" if shared else "player_relic_bag"
    context = RelicBagContext(
        total_floor=int(flags.get("total_floor", len(state.room_history) + state.act + 1)),
        player_count=int(flags.get("player_count", 1)),
        first_run_ironclad_present=(
            int(flags.get("number_of_runs", -1)) == 0 and state.character_id.upper() == "IRONCLAD"
        ),
    )
    result = pull_relic_from_bag(
        _bag(flags, key), rarity, context=context,
        from_back=from_back, for_shop=for_shop, excluded_ids=excluded_ids,
        refresh_originals=_bag(flags, "shared_relic_bag_originals") if shared else None,
        multiplayer_fallback=tuple(flags.get("player_relic_bag_multiplayer_fallback", ())),
    )
    flags[key] = dict(result.bag)
    if not shared:
        flags["player_relic_bag_multiplayer_fallback"] = result.multiplayer_fallback
    relic_id = result.relic_id or "CIRCLET"
    if not shared and "shared_relic_bag" in flags:
        flags["shared_relic_bag"] = remove_bag_relics(
            _bag(flags, "shared_relic_bag"), (relic_id,),
        )
    return relic_id.lower()


def remove_obtained_source_relic(state: RunState, relic_id: str) -> RunState:
    # Circlet is stackable and is the infinite exhausted-bag fallback.
    if relic_id.upper() == "CIRCLET":
        return state
    flags = dict(state.flags)
    for key in ("shared_relic_bag", "player_relic_bag"):
        if key in flags:
            flags[key] = remove_bag_relics(_bag(flags, key), (relic_id,))
    return state.model_copy(update={"flags": flags})
