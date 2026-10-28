"""Single-player affliction callbacks from stable 0.107.1.

The source gives a card one affliction slot. These callbacks share that slot
with Vital Spark's Tainted lifecycle; a power cannot claim an occupied card.
See tests/fixtures/source_modifier_audit.json for pinned source references.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import CardInstance, CardType, CombatState, EffectEvent

_KEY = "source_core_affliction"
_ZONES = ("hand", "draw_pile", "discard_pile", "exhaust_pile")
_IDS = ("bound", "entangled", "galvanized", "hexed", "ringing", "smog", "tainted")
_OWNER_POWERS = {"tangled": "entangled", "hex": "hexed", "ringing": "ringing"}


def affliction_id(card: CardInstance) -> str:
    raw = card.custom.get(_KEY, card.custom.get("affliction"))
    if isinstance(raw, Mapping):
        return str(raw.get("id", raw.get("affliction_id", ""))).lower()
    if isinstance(raw, str):
        return raw.lower()
    return next((key for key in _IDS if card.custom.get(key)), "")


def afflict(
    card: CardInstance, identity: str, amount: int = 1, *, stack: bool = False
) -> CardInstance:
    """AfflictionModel.CanAfflict permits Unplayable and rejects occupied slots."""
    current = affliction_id(card)
    if current == identity and stack and identity in {"galvanized", "tainted"}:
        previous = card.custom.get(_KEY, card.custom.get("affliction", {}))
        amount += int(previous.get("amount", 0)) if isinstance(previous, Mapping) else 0
    elif current:
        return card
    return card.model_copy(
        update={
            "custom": {
                **card.custom,
                _KEY: {"id": identity, "amount": amount},
                identity: True,
            }
        }
    )


def _clear(card: CardInstance, identities: set[str]) -> CardInstance:
    if affliction_id(card) not in identities:
        return card
    custom = dict(card.custom)
    for key in (_KEY, "affliction", *_IDS):
        custom.pop(key, None)
    return card.model_copy(update={"custom": custom})


def _cards(combat: CombatState) -> tuple[CardInstance, ...]:
    return tuple(card for zone in _ZONES for card in getattr(combat, zone))


def current_card(combat: CombatState, card: CardInstance) -> CardInstance:
    """Refresh identity only, retaining an autoplay's captured X and effects."""
    physical = next((item for item in _cards(combat) if item.instance_id == card.instance_id), None)
    if physical is None:
        return card
    custom = dict(card.custom)
    for key in (_KEY, "affliction", *_IDS):
        custom.pop(key, None)
        if key in physical.custom:
            custom[key] = physical.custom[key]
    return card.model_copy(update={"custom": custom})


def _started(combat: CombatState, *, skills: bool = False) -> int:
    if combat.metadata.get("turn_owner") == "monster":
        return 0
    key = "source_modifier_skills_started" if skills else "source_modifier_plays_started"
    fallback = len(combat.cards_played_this_turn)
    if skills:
        fallback = int(combat.metadata.get("skills_played_this_turn", 0))
    if combat.metadata.get("source_modifier_play_turn") != combat.turn:
        return fallback
    return max(fallback, int(combat.metadata.get(key, 0)))


def synchronize(combat: CombatState) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    """Run first application, card-entry, and removal callbacks once per entry."""
    from . import transitions as t

    metadata = dict(combat.metadata)
    statuses = dict(combat.player.statuses)
    applier = metadata.get("source_hex_applier")
    events: list[EffectEvent] = []
    if applier and not any(m.monster_id == applier and m.hp > 0 for m in combat.monsters):
        amount = t._status_amount(statuses, "hex")
        statuses = {
            key: value
            for key, value in statuses.items()
            if t._normalized_id(key).replace("_", "") != "hex"
        }
        metadata.pop("source_hex_applier", None)
        if amount:
            events.append(
                EffectEvent(
                    kind="temporary_status_expired",
                    source_id=str(applier),
                    target_id="player",
                    amount=-amount,
                    metadata={"status": "hex", "trigger": "applier_death"},
                )
            )
    active = {power: t._status_amount(statuses, power) for power in _OWNER_POWERS}
    previous = metadata.get("source_modifier_owner_powers", {})
    previous = dict(previous) if isinstance(previous, Mapping) else {}
    initial = "source_modifier_seen_cards" not in metadata
    seen = set(str(item) for item in metadata.get("source_modifier_seen_cards", ()))
    galvanic = {
        monster.monster_id: t._status_amount(monster.statuses, "galvanic")
        for monster in combat.monsters
        if monster.hp > 0 and t._status_amount(monster.statuses, "galvanic") > 0
    }
    cleared = {identity for power, identity in _OWNER_POWERS.items() if not active[power]}
    updates: dict[str, Any] = {}
    for zone in _ZONES:
        cards: list[CardInstance] = []
        for card in getattr(combat, zone):
            before = affliction_id(card)
            card = _clear(card, cleared)
            entering = card.instance_id not in seen
            # AfterApplied visits existing cards once. AfterCardEnteredCombat
            # separately claims a new card, provided its affliction is empty.
            for power, identity in _OWNER_POWERS.items():
                if (
                    active[power]
                    and (entering or not previous.get(power))
                    and (power != "tangled" or card.type == CardType.ATTACK)
                ):
                    card = afflict(card, identity)
            if (
                entering
                and card.type == CardType.SKILL
                and t._status_amount(statuses, "smoggy") > 0
                and _started(combat, skills=True)
            ):
                card = afflict(card, "smog")
            if entering and card.type == CardType.POWER and galvanic:
                # BeforeCombatStart calls the stackable affliction for each
                # source. Card-entry hooks require an empty slot, so only the
                # first source claims later generated Power cards.
                amount = sum(galvanic.values()) if initial else next(iter(galvanic.values()))
                card = afflict(card, "galvanized", amount, stack=initial)
            after = affliction_id(card)
            if before != after:
                if before:
                    events.append(
                        EffectEvent(
                            kind="card_affliction_removed",
                            target_id=card.instance_id,
                            metadata={"affliction": before, "trigger": "power_removed"},
                        )
                    )
                if after:
                    events.append(
                        EffectEvent(
                            kind="card_afflicted",
                            target_id=card.instance_id,
                            amount=int(card.custom[_KEY]["amount"]),
                            metadata={
                                "affliction": after,
                                "trigger": "card_entered_or_power_applied",
                            },
                        )
                    )
            cards.append(card)
            seen.add(card.instance_id)
        updates[zone] = tuple(cards)
    metadata.update(
        {"source_modifier_seen_cards": tuple(sorted(seen)), "source_modifier_owner_powers": active}
    )
    updates["metadata"] = metadata
    updates["player"] = combat.player.model_copy(update={"statuses": statuses})
    return combat.model_copy(update=updates), tuple(events)


def blocks_play(combat: CombatState, card: CardInstance) -> bool:
    from . import transitions as t

    identity = affliction_id(current_card(combat, card))
    if identity == "smog" and t._status_amount(combat.player.statuses, "smoggy") > 0:
        return True
    if identity == "ringing" and t._status_amount(combat.player.statuses, "ringing") > 0:
        return _started(combat) > 0
    if identity == "bound" and t._status_amount(combat.player.statuses, "chains_of_binding") > 0:
        return bool(combat.metadata.get("source_bound_card_played"))
    return False


def before_play(combat: CombatState, card: CardInstance) -> CombatState:
    metadata = dict(combat.metadata)
    metadata.update(
        {
            "source_modifier_play_turn": combat.turn,
            "source_modifier_plays_started": _started(combat) + 1,
            "source_modifier_skills_started": _started(combat, skills=True)
            + (card.type == CardType.SKILL),
        }
    )
    if affliction_id(card) == "bound" and not card.custom.get("is_dupe"):
        metadata["source_bound_card_played"] = True
    return combat.model_copy(update={"metadata": metadata})


def after_play(
    combat: CombatState, card: CardInstance
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    combat, lifecycle = synchronize(combat)
    if card.type != CardType.SKILL or t._status_amount(combat.player.statuses, "smoggy") <= 0:
        return combat, lifecycle
    events = list(lifecycle)
    updates: dict[str, Any] = {}
    for zone in _ZONES:
        cards: list[CardInstance] = []
        for item in getattr(combat, zone):
            afflicted = afflict(item, "smog") if item.type == CardType.SKILL else item
            if afflicted != item:
                events.append(
                    EffectEvent(
                        kind="card_afflicted",
                        source_id="smoggy",
                        target_id=item.instance_id,
                        amount=1,
                        metadata={"affliction": "smog", "trigger": "after_skill_played"},
                    )
                )
            cards.append(afflicted)
        updates[zone] = tuple(cards)
    return combat.model_copy(update=updates), tuple(events)


def after_draw(card: CardInstance, combat: CombatState, successful: int) -> CardInstance:
    from . import transitions as t

    limit = t._status_amount(combat.player.statuses, "chains_of_binding")
    if str(combat.metadata.get("turn_owner", "player")) == "monster" or successful >= limit:
        return card
    return afflict(card, "bound", limit)


def ethereal(combat: CombatState, card: CardInstance) -> bool:
    from . import transitions as t

    return t._card_has_keyword(card, "ethereal") or (
        affliction_id(card) == "hexed" and t._status_amount(combat.player.statuses, "hex") > 0
    )


def before_turn_end(combat: CombatState) -> CombatState:
    from . import transitions as t

    identities = (
        {"bound"} if t._status_amount(combat.player.statuses, "chains_of_binding") else set()
    )
    updates: dict[str, Any] = {
        zone: tuple(_clear(card, identities) for card in getattr(combat, zone)) for zone in _ZONES
    }
    updates["metadata"] = {**combat.metadata, "source_bound_card_played": False}
    return combat.model_copy(update=updates)


def after_turn_end(combat: CombatState) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    """Owner-side expiry is unconditional; Tangled lasts through one owner turn."""
    from . import transitions as t

    statuses = dict(combat.player.statuses)
    events: list[EffectEvent] = []
    for power in ("tangled", "ringing"):
        amount = t._status_amount(statuses, power)
        statuses = {
            key: value
            for key, value in statuses.items()
            if t._normalized_id(key).replace("_", "") != power
        }
        if amount:
            events.append(
                EffectEvent(
                    kind="temporary_status_expired",
                    target_id="player",
                    amount=-amount,
                    metadata={"status": power, "trigger": "owner_turn_end"},
                )
            )
    identities = {"entangled", "ringing"}
    if t._status_amount(statuses, "smoggy"):
        identities.add("smog")
    updates: dict[str, Any] = {
        zone: tuple(_clear(card, identities) for card in getattr(combat, zone)) for zone in _ZONES
    }
    updates["player"] = combat.player.model_copy(update={"statuses": statuses})
    updates["metadata"] = {
        key: value for key, value in combat.metadata.items() if key != "tangled_turns_remaining"
    }
    combat, lifecycle = synchronize(combat.model_copy(update=updates))
    return combat, tuple(events) + lifecycle
