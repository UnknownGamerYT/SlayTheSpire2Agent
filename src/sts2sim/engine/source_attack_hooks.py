"""Stable BeforeAttack/AfterAttack boundaries shared by player and pet commands.

A command includes its full hit count and all receiver batches. The source
Gigantification and Vigor powers lock their first matching command, rather than
consuming after the enclosing card play. No game implementation is vendored.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .models import CardInstance, CardType, CombatState, EffectEvent

_KEY = "source_attack_commands"
# These source cards use one WithHitCount command; old text normalization
# represented the hits as adjacent damage steps.
_STATIC_HITS = {
    "astral_pulse": 2, "celestial_might": 3, "conflagration": 4,
    "dagger_spray": 2, "exterminate": 4, "fight_me": 2, "gunk_up": 3,
    "maul": 2, "peck": 3, "refract": 2, "seven_stars": 7,
    "twin_strike": 2, "uproar": 2,
}


def _commands(combat: CombatState) -> tuple[Mapping[str, Any], ...]:
    raw = combat.metadata.get(_KEY, ())
    return tuple(item for item in raw if isinstance(item, Mapping)) if isinstance(
        raw, Sequence) and not isinstance(raw, (str, bytes)) else ()


def active(combat: CombatState, card: CardInstance) -> bool:
    commands = _commands(combat)
    return bool(commands and commands[-1].get("card_id") == card.instance_id)


def allows_modifier(combat: CombatState, card: CardInstance, modifier: str) -> bool:
    owner = next((item for item in _commands(combat) if item.get(modifier)), None)
    return owner is None or owner.get("card_id") == card.instance_id


def begin_attack_command(
    combat: CombatState, card: CardInstance, *, dealer_is_pet: bool = False,
) -> CombatState:
    from . import transitions as t

    if combat.player.hp <= 0 or not t._alive_monsters(combat):
        return combat
    commands = _commands(combat)
    claim_queue = (card.type == CardType.ATTACK and not any(c.get("queue") for c in commands)
                   and any(t._queued_card_modifier_matches(item, card)
                           for item in t._next_card_damage_multiplier_queue(combat.metadata)))
    vigor = 0 if dealer_is_pet or any(c.get("vigor") for c in commands) else t._status_amount(
        combat.player.statuses, "vigor")
    entry = {"card_id": card.instance_id, "queue": claim_queue, "vigor": vigor}
    return combat.model_copy(update={"metadata": {**combat.metadata, _KEY: commands + (entry,)}})


def finish_attack_command(
    combat: CombatState, card: CardInstance, *, dealer_is_pet: bool = False,
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    commands = list(_commands(combat))
    if not commands or commands[-1].get("card_id") != card.instance_id:
        return combat, ()
    command = commands.pop()
    metadata = dict(combat.metadata)
    if commands:
        metadata[_KEY] = tuple(commands)
    else:
        metadata.pop(_KEY, None)
    combat = combat.model_copy(update={"metadata": metadata})
    events: tuple[EffectEvent, ...] = ()
    if command.get("queue"):
        combat, events = t._consume_next_card_damage_multiplier_after_play(combat, card)
    vigor = int(command.get("vigor", 0)) if not dealer_is_pet else 0
    if vigor:
        statuses = dict(combat.player.statuses)
        for key in tuple(statuses):
            if t._normalized_id(key).replace("_", "") == "vigor":
                remaining = statuses.pop(key) - vigor
                if remaining > 0:
                    statuses[key] = remaining
        combat = combat.model_copy(update={"player": combat.player.model_copy(update={
            "statuses": statuses,
        })})
        events += (EffectEvent(kind="status_consumed", source_id=card.instance_id,
                               target_id="player", amount=vigor,
                               metadata={"status": "vigor"}),)
    return combat, events


def grouped_steps(
    card: CardInstance, steps: tuple[Mapping[str, Any], ...],
) -> tuple[Mapping[str, Any], ...]:
    count = _STATIC_HITS.get(card.card_id.lower(), 0)
    if count and len(steps) >= count and all(
        set(step) <= {"damage", "all_damage"} and bool(step) for step in steps[:count]
    ):
        return ({"attack_command": steps[:count]},) + steps[count:]
    return steps
