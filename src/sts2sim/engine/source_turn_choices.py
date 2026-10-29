"""Source AutoPost and BeforeFlushLate choices with saved continuations."""
from __future__ import annotations

from .models import (
    CardInstance,
    CardType,
    CombatState,
    EffectEvent,
    PendingChoiceState,
    RngState,
    RunState,
    TargetType,
)
from .rng import capture_random_state, random_from_state, random_stream


def stampede(
    state: RunState, combat: CombatState, rng_state: RngState,
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    """Await each selected child play before sampling the current hand again."""
    from . import transitions as t

    record = t._mapping_from(combat.metadata.get("source_stampede_turn"))
    remaining = (t._coerce_int(record.get("remaining"), default=0)
                 if record.get("turn") == combat.turn
                 else t._status_amount(combat.player.statuses, "stampede"))
    events: list[EffectEvent] = []
    while remaining > 0 and not combat.pending_choices:
        remaining -= 1
        combat = combat.model_copy(update={"metadata": {
            **combat.metadata, "source_stampede_turn": {
                "turn": combat.turn, "remaining": remaining,
            },
        }})
        candidates = tuple(card for card in combat.hand
                           if card.type == CardType.ATTACK and not t._card_is_unplayable(card))
        if not candidates or combat.player.hp <= 0 or not t._alive_monsters(combat):
            continue
        rng = random_stream(random_from_state(rng_state), "shuffle")
        card = rng.choice(candidates)
        source = CardInstance(instance_id="power:stampede", card_id="stampede",
                              name="Stampede", type=CardType.POWER, target=TargetType.SELF, cost=0)
        combat, rng_state, emitted = t._play_combat_card_from_zone_by_trigger(
            state, combat, capture_random_state(rng), source, card,
            from_zone="hand", target_mode="random_enemy",
        )
        events.extend(emitted)
    return combat, rng_state, tuple(events)


def retain_before_flush(
    state: RunState, combat: CombatState,
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    """Well-Laid Plans sees the hand after ordinary end-hand and BeforeFlush."""
    from . import transitions as t

    if combat.metadata.get("source_well_laid_plans_turn") == combat.turn:
        return combat, ()
    combat = combat.model_copy(update={"metadata": {
        **combat.metadata, "source_well_laid_plans_turn": combat.turn,
    }})
    amount = t._status_amount(combat.player.statuses, "well_laid_plans")
    if (amount <= 0 or t._status_amount(combat.player.statuses, "retain_hand") > 0
            or t._has_relic_id(state.relics, "runic_pyramid")
            or combat.turn == 1 and t._has_relic_id(state.relics, "ringing_triangle")):
        return combat, ()
    phantom = t._status_amount(combat.player.statuses, "phantom_blades") > 0
    candidates = tuple(card.instance_id for card in combat.hand
                       if not t._card_retains_at_end_of_turn(card)
                       and not (phantom and t._is_shiv_card(card)))
    count = min(amount, len(candidates))
    if not count:
        return combat, ()
    choice = PendingChoiceState(
        choice_id=f"well_laid_plans:{combat.turn}", kind="add_retain",
        source_id="well_laid_plans", prompt=f"Retain up to {count} cards.",
        zone="hand", candidate_ids=candidates, min_choices=0, max_choices=count,
        remaining=count, required=False,
        metadata={"retain_once": True, "source_phase": "before_flush_late"},
    )
    return combat.model_copy(update={"pending_choices": combat.pending_choices + (choice,)}), (
        EffectEvent(kind="card_choice_created", source_id="well_laid_plans", amount=count,
                    metadata={"choice_id": choice.choice_id, "choice_kind": "add_retain",
                              "zone": "hand", "source_phase": "before_flush_late"}),
    )
