"""Stable single-player combat relic callbacks at their source phase boundaries."""
from __future__ import annotations

from typing import Any

from ..mechanics.source_power_rules import power_application_is_debuff
from ..mechanics.triggers import GameTrigger
from .models import (
    PLAYER_TARGET_ID,
    CardInstance,
    CombatState,
    EffectEvent,
    RngState,
    RunPhase,
    RunState,
)


def power_amount(
    metadata: dict[str, Any], status: str, amount: int, target_id: str, *,
    card: CardInstance | None = None, giver_is_owner: bool = True,
) -> int:
    """Modify one power application and its explicitly supplied local metadata copy."""
    from . import transitions as t

    relics = tuple(metadata.get("item_relic_ids", ()))
    counters = dict(t._mapping_from(metadata.get("relic_counters")))
    if (giver_is_owner and amount != 0 and t._normalized_status_id(status) == "poison"
            and t._has_relic_id(relics, "snecko_skull")):
        amount += 1
    if (t._has_relic_id(relics, "ruined_helmet") and target_id == PLAYER_TARGET_ID
            and t._normalized_status_id(status) == "strength" and amount > 0
            and t._coerce_int(counters.get("ruined_helmet"), default=0) <= 0):
        amount *= 2
        counters["ruined_helmet"] = 1
        metadata["relic_counters"] = counters
    if (not t._has_relic_id(relics, "unsettling_lamp") or card is None
            or card.custom.get("source_kind") == "potion"
            or t._coerce_int(counters.get("unsettling_lamp"), default=0) > 0
            or not power_application_is_debuff(status, amount)):
        return amount
    triggering = metadata.get("unsettling_lamp_triggering_card")
    if triggering is None and giver_is_owner and target_id != PLAYER_TARGET_ID:
        triggering = card.instance_id
        metadata["unsettling_lamp_triggering_card"] = triggering
    if triggering == card.instance_id:
        amount *= 2
    return amount


def finish_card(metadata: dict[str, Any], card: CardInstance) -> None:
    if metadata.get("unsettling_lamp_triggering_card") == card.instance_id:
        counters = dict(metadata.get("relic_counters", {}))
        counters["unsettling_lamp"] = 1
        metadata["relic_counters"] = counters


def exhaust_paels_eye(
    state: RunState, combat: CombatState, rng_state: RngState,
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    if (not t._has_relic_id(state.relics, "paels_eye")
            or t._combat_relic_counters(combat).get("paels_eye", 0) > 0
            or t._turn_power_count(combat, "paels_eye_manual")
            or combat.turn == 1 and t._has_relic_id(state.relics, "whispering_earring")):
        return combat, rng_state, ()
    snapshot = t._mapping_from(combat.metadata.get("paels_eye_exhaust"))
    if snapshot.get("turn") != combat.turn:
        snapshot = {"turn": combat.turn, "remaining": tuple(c.instance_id for c in combat.hand)}
    events: list[EffectEvent] = []
    remaining = list(t._string_items(snapshot.get("remaining")))
    while remaining and not combat.pending_choices:
        identity = remaining.pop(0)
        combat = combat.model_copy(update={"metadata": {**combat.metadata,
            "paels_eye_exhaust": {"turn": combat.turn, "remaining": tuple(remaining)},
        }})
        card = t._find_card(combat.hand, identity)
        if card is None:
            continue
        combat = combat.model_copy(update={
            "hand": tuple(c for c in combat.hand if c.instance_id != identity),
            "exhaust_pile": combat.exhaust_pile + (card,),
        })
        movement = t._card_movement_events(
            GameTrigger.CARD_EXHAUSTED, (card,), from_pile="hand", to_pile="exhaust_pile",
            reason="paels_eye_before_side_turn_end_early",
        )
        events.extend(movement)
        combat, rng_state, callbacks = t._apply_card_exhausted_triggers_from_events(
            state, combat, rng_state, movement,
        )
        events.extend(callbacks)
        combat, rng_state, draws = t._apply_pending_relic_draws(state, combat, rng_state)
        events.extend(draws)
        if t._phase_after_combat(combat) != RunPhase.COMBAT:
            break
    return combat, rng_state, tuple(events)
