"""Card keyword and end-in-hand boundaries from stable 0.107.1 source."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sts2sim.mechanics.triggers import GameTrigger

from . import source_modifier_hooks
from .models import CardInstance, CombatState, EffectEvent, RngState, RunState

END_HAND_CARD_IDS = frozenset(
    {
        "bad_luck",
        "beckon",
        "burn",
        "decay",
        "debt",
        "doubt",
        "infection",
        "regret",
        "shame",
        "toxic",
        "wither",
    }
)


def has_end_hand_effect(card: CardInstance) -> bool:
    from . import transitions as t

    if card.card_id.lower() in END_HAND_CARD_IDS:
        return True
    return any(step.get("end_turn_hand_effect") for step in t._effect_steps(card.effects))


def end_hand_effect(
    combat: CombatState,
    card: CardInstance,
    relics: Sequence[str],
) -> tuple[bool, CombatState, tuple[EffectEvent, ...]]:
    """Numeric source effects use the canonical damage/status pipelines once."""
    from . import transitions as t

    identity = card.card_id.lower()
    if identity not in END_HAND_CARD_IDS:
        return False, combat, ()
    if identity == "debt":
        lost = min(10, combat.player.gold)
        combat = combat.model_copy(
            update={
                "player": combat.player.model_copy(
                    update={
                        "gold": combat.player.gold - lost,
                    }
                )
            }
        )
        return (
            True,
            combat,
            (
                EffectEvent(
                    kind="gold_lost",
                    source_id=card.instance_id,
                    target_id="player",
                    amount=lost,
                    metadata={"card_id": card.card_id, "reason": "end_turn_hand_card"},
                ),
            ),
        )
    if identity in {"doubt", "shame"}:
        power = "weak" if identity == "doubt" else "frail"
        statuses, event = t._apply_status_value(
            combat.player.statuses,
            power,
            1,
            source_id=card.instance_id,
            target_id="player",
        )
        combat = combat.model_copy(update={
            "player": combat.player.model_copy(update={"statuses": statuses}),
        })
        return True, combat, (event,) if event is not None else ()
    amount = {
        "bad_luck": 13,
        "beckon": 6,
        "burn": 2,
        "decay": 2,
        "infection": 3,
        "toxic": 5,
        "wither": 3,
        "regret": 0,
    }[identity]
    if identity == "regret":
        amount = int(combat.metadata.get("source_keyword_regret_hand_size", len(combat.hand)))
    elif identity == "wither":
        amount *= 1 + int(card.custom.get("wither_upgrade_count", int(card.upgraded)))
    combat, events = t._damage_combat_player(
        combat,
        amount,
        card.instance_id,
        relics=relics,
        is_attack=False,
        unblockable=identity in {"bad_luck", "beckon", "regret"},
    )
    return (
        True,
        combat,
        tuple(
            event.model_copy(
                update={
                    "metadata": {
                        **event.metadata,
                        "card_id": card.card_id,
                        "reason": "end_turn_hand_card",
                    }
                }
            )
            for event in events
        ),
    )


def resolve_end_hand(
    state: RunState,
    combat: CombatState,
    rng_state: RngState,
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    """Ordinary Ethereal, then per-card hand effects, all before retention."""
    from . import transitions as t

    if combat.metadata.get("source_keyword_hand_end_turn") == combat.turn:
        return combat, rng_state, ()
    metadata = {**combat.metadata, "source_keyword_regret_hand_size": len(combat.hand)}
    combat = combat.model_copy(update={"metadata": metadata})
    combat, rng_state, passive_events = t._apply_orb_passives(
        combat,
        rng_state,
        relics=state.relics,
        state=state,
    )
    events = list(passive_events)
    hand_effects = tuple(card for card in combat.hand if has_end_hand_effect(card))
    ethereal = tuple(
        card
        for card in combat.hand
        if not has_end_hand_effect(card) and source_modifier_hooks.ethereal(combat, card)
    )
    for card in (*ethereal, *hand_effects):
        if combat.player.hp <= 0 or not t._alive_monsters(combat):
            break
        if card in hand_effects:
            combat, hand_events = t._apply_end_turn_hand_card_effects(
                combat,
                (card,),
                relics=state.relics,
            )
            events.extend(hand_events)
            combat, rng_state, relic_events = t._apply_damage_taken_relics_from_events(
                state,
                combat,
                rng_state,
                hand_events,
                turn_owner="player",
            )
            events.extend(relic_events)
        exhaust = source_modifier_hooks.ethereal(combat, card)
        updates: dict[str, Any] = {
            zone: tuple(
                item for item in getattr(combat, zone) if item.instance_id != card.instance_id
            )
            for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile")
        }
        destination = "exhaust_pile" if exhaust else "discard_pile"
        updates[destination] += (card,)
        combat = combat.model_copy(update=updates)
        if exhaust:
            hexed = source_modifier_hooks.affliction_id(card) == "hexed"
            events.append(
                EffectEvent(
                    kind="hex_ethereal_cards_exhausted" if hexed else "ethereal_cards_exhausted",
                    target_id=card.instance_id,
                    amount=1,
                    metadata={"card_instance_ids": [card.instance_id]},
                )
            )
            movement = t._card_movement_events(
                GameTrigger.CARD_EXHAUSTED,
                (card,),
                from_pile="hand",
                to_pile=destination,
                reason="ethereal_end_turn",
            )
            events.extend(movement)
            combat, rng_state, exhaust_events = t._apply_card_exhausted_triggers_from_events(
                state.model_copy(update={"combat": combat, "player": combat.player}),
                combat,
                rng_state,
                movement,
            )
            events.extend(exhaust_events)
        else:
            events.append(
                EffectEvent(
                    kind="end_turn_hand_card_discarded",
                    target_id=card.instance_id,
                    metadata={"card_id": card.card_id, "reason": "end_turn_hand_effect"},
                )
            )
    return (
        combat.model_copy(
            update={
                "metadata": {
                    **combat.metadata,
                    "source_keyword_hand_end_turn": combat.turn,
                }
            }
        ),
        rng_state,
        tuple(events),
    )
