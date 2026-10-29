"""Source Player potion slots and PotionCmd procurement, including empty holes."""

from __future__ import annotations

from collections.abc import Sequence

from sts2sim.mechanics.potion_inventory import (
    POTION_SLOT_INDICES,
    occupied_potion_slots,
    potion_slot_contents,
    potion_slot_indices,
)
from sts2sim.mechanics.source_rng import is_source_rng

from .models import CombatState, EffectEvent, RunPhase, RunState


def physical_slots_enabled(state: RunState) -> bool:
    return is_source_rng(state.rng) or POTION_SLOT_INDICES in state.flags


def occupied_slots(state: RunState) -> tuple[tuple[int, str], ...]:
    return occupied_potion_slots(state.potions, state.flags)


def inventory_slots(state: RunState) -> tuple[str | None, ...]:
    from .transitions import _potion_capacity

    return potion_slot_contents(state.potions, state.flags, _potion_capacity(state))


def potion_at_slot(state: RunState, slot: int) -> str | None:
    return next((potion for index, potion in occupied_slots(state) if index == slot), None)


def with_inventory(
    state: RunState,
    potions: Sequence[str],
    indices: Sequence[int],
) -> RunState:
    flags = dict(state.flags)
    if physical_slots_enabled(state):
        pairs = sorted(zip(indices, potions, strict=True))
        indices = tuple(index for index, _ in pairs)
        potions = tuple(potion for _, potion in pairs)
        flags[POTION_SLOT_INDICES] = tuple(indices)
        potion_slot_indices(potions, flags)
    return state.model_copy(update={"potions": tuple(potions), "flags": flags})


def remove_at_slot(state: RunState, slot: int) -> tuple[RunState, str | None]:
    pairs = occupied_slots(state)
    potion = next((value for index, value in pairs if index == slot), None)
    if potion is None:
        return state, None
    kept = tuple((index, value) for index, value in pairs if index != slot)
    return with_inventory(
        state, tuple(value for _, value in kept), tuple(index for index, _ in kept)
    ), potion


def add_at_slot(
    state: RunState,
    potion: str,
    *,
    preferred_slot: int | None = None,
) -> tuple[RunState, int | None]:
    slots = inventory_slots(state)
    if preferred_slot is None or preferred_slot < 0:
        slot = next((index for index, value in enumerate(slots) if value is None), None)
    else:
        if preferred_slot >= len(slots):
            raise IndexError("Preferred potion slot exceeds the potion capacity")
        slot = preferred_slot if slots[preferred_slot] is None else None
    if slot is None:
        return state, None
    pairs = (*occupied_slots(state), (slot, potion))
    state = with_inventory(
        state, tuple(value for _, value in pairs), tuple(index for index, _ in pairs)
    )
    return state, slot


def procure_at_slot(
    state: RunState,
    potion: str,
    *,
    preferred_slot: int | None = None,
    source_id: str | None = None,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    """ShouldProcure precedes AddPotionInternal; occupied preferred slots fail."""
    from . import transitions as t

    potion = t._normalized_id(potion)
    if t._has_relic(state, "sozu"):
        return state, (t._sozu_blocked_potion_event(potion, source_id),)
    state, slot = add_at_slot(state, potion, preferred_slot=preferred_slot)
    if slot is None:
        return state, (
            EffectEvent(
                kind="event_potion_skipped_no_slot",
                source_id=source_id,
                target_id=potion,
                metadata={
                    "potion_slots": t._potion_capacity(state),
                    "preferred_slot": preferred_slot,
                },
            ),
        )
    events: tuple[EffectEvent, ...] = (
        EffectEvent(
            kind="event_potion_obtained",
            source_id=source_id,
            target_id=potion,
            metadata={"potion_slots": t._potion_capacity(state), "potion_slot": f"potion:{slot}"},
        ),
    )
    if state.phase == RunPhase.COMBAT and state.combat is not None:
        combat, emitted = t._sync_belt_buckle(state.combat, state.potions, state.relics)
        combat = remember_replay(combat, state)
        state = state.model_copy(update={"combat": combat, "player": combat.player})
        events += emitted
    return state, events


def state_from_replay(state: RunState, combat: CombatState) -> RunState:
    potions = tuple(combat.metadata.get("replay_potions", state.potions))
    indices = tuple(
        combat.metadata.get(
            "replay_potion_slot_indices", potion_slot_indices(state.potions, state.flags)
        )
    )
    return with_inventory(state, potions, indices)


def remember_replay(combat: CombatState, state: RunState) -> CombatState:
    return combat.model_copy(
        update={
            "metadata": {
                **combat.metadata,
                "replay_potions": state.potions,
                "replay_potion_slot_indices": potion_slot_indices(state.potions, state.flags),
            }
        }
    )
