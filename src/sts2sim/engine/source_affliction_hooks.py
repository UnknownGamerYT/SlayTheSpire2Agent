"""VitalSpark/Tainted lifecycle from stable 0.107.1 source.

Razhuo/sts2-rl-agent@5d874e855ae1d0550d5c30416fadd7dbb0d133b7:
VitalSparkPower.BeforeCombatStart/AfterCardEnteredCombat/AfterCardPlayed/
AfterPowerAmountChanged/AfterRemoved; AfflictionModel.CanAfflict;
CreatureCmd.Kill removes powers and awaits their AfterRemoved callbacks.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import CardInstance, CardType, CombatState, EffectEvent

_ZONES = ("hand", "draw_pile", "discard_pile", "exhaust_pile")
_KEY = "source_core_affliction"
_GENERATION = "source_core_affliction_generation"
_VERSION = "source_core_affliction_version"


def affliction_id(card: CardInstance) -> str:
    raw = card.custom.get(_KEY, card.custom.get("affliction"))
    if isinstance(raw, Mapping):
        return str(raw.get("id", raw.get("affliction_id", ""))).lower()
    if isinstance(raw, str):
        return raw.lower()
    return next(
        (
            key
            for key in ("bound", "hexed", "entangled", "ringing", "smog", "galvanized")
            if card.custom.get(key)
        ),
        "",
    )


def synchronize(combat: CombatState) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    """Apply initial/entry hooks and source amount/removal callbacks exactly once."""
    from . import transitions as t

    active = {
        m.monster_id: t._status_amount(m.statuses, "vital_spark")
        for m in combat.monsters
        if m.hp > 0 and t._status_amount(m.statuses, "vital_spark") > 0
    }
    metadata = dict(combat.metadata)
    initial = "source_vital_spark_sources" not in metadata
    previous = metadata.get("source_vital_spark_sources", {})
    previous = dict(previous) if isinstance(previous, Mapping) else {}
    if not initial and not active and not previous:
        return combat, ()
    seen = set(str(i) for i in metadata.get("source_vital_spark_seen_cards", ()))
    if not initial and active and not previous:
        # A source acquired after combat start does not re-run BeforeCombatStart.
        seen.update(c.instance_id for zone in _ZONES for c in getattr(combat, zone))
    generation = int(metadata.get("source_vital_spark_generation", 0))
    removed = any(source not in active for source in previous)
    if removed:
        generation += 1
    changed = [
        (source, amount)
        for source, amount in active.items()
        if not initial and previous.get(source) != amount
    ]
    version = int(metadata.get("source_vital_spark_version", 0)) + bool(changed)
    display_amount = (
        changed[-1][1] if changed else metadata.get("source_vital_spark_display_amount")
    )
    events: list[EffectEvent] = []
    updates: dict[str, Any] = {}
    for zone in _ZONES:
        cards: list[CardInstance] = []
        for card in getattr(combat, zone):
            custom = dict(card.custom)
            current = affliction_id(card)
            stale = current == "tainted" and int(custom.get(_GENERATION, generation)) != generation
            if current == "tainted" and (removed or stale):
                custom.pop(_KEY, None)
                custom.pop(_GENERATION, None)
                custom.pop(_VERSION, None)
                custom.pop("affliction", None)
                current = ""
                events.append(
                    EffectEvent(
                        kind="card_affliction_removed",
                        target_id=card.instance_id,
                        metadata={"affliction": "tainted", "trigger": "vital_spark_removed"},
                    )
                )
            entering = card.instance_id not in seen
            eligible = card.type == CardType.SKILL and not t._card_has_keyword(card, "unplayable")
            if initial and eligible and current in {"", "tainted"}:
                # BeforeCombatStart calls Afflict for each source; Tainted is
                # stackable. Its amount is display metadata, not the debuff.
                existing = custom.get(_KEY, custom.get("affliction", {}))
                old_amount = int(existing.get("amount", 0)) if isinstance(existing, Mapping) else 0
                amount = old_amount + sum(active.values())
                if amount:
                    custom[_KEY] = {"id": "tainted", "amount": amount}
                    custom[_GENERATION] = generation
                    custom[_VERSION] = version
                    events.append(
                        EffectEvent(
                            kind="card_afflicted",
                            target_id=card.instance_id,
                            amount=amount - old_amount,
                            metadata={
                                "affliction": "tainted",
                                "trigger": "vital_spark_combat_start",
                            },
                        )
                    )
            elif entering and eligible and not current and active:
                # AfterCardEnteredCombat explicitly requires Affliction == null,
                # so the first source claims a new Skill, even with two Prisms.
                source, amount = next(iter(active.items()))
                custom[_KEY] = {"id": "tainted", "amount": amount}
                custom[_GENERATION] = generation
                custom[_VERSION] = version
                events.append(
                    EffectEvent(
                        kind="card_afflicted",
                        source_id=source,
                        target_id=card.instance_id,
                        amount=amount,
                        metadata={"affliction": "tainted", "trigger": "vital_spark_card_entered"},
                    )
                )
            if affliction_id(card.model_copy(update={"custom": custom})) == "tainted":
                if int(custom.get(_VERSION, 0)) < version and display_amount is not None:
                    custom[_KEY] = {"id": "tainted", "amount": int(display_amount)}
                    custom[_VERSION] = version
                for source, amount in changed:
                    custom[_KEY] = {"id": "tainted", "amount": amount}
                    custom[_VERSION] = version
                    events.append(
                        EffectEvent(
                            kind="card_affliction_amount_set",
                            source_id=source,
                            target_id=card.instance_id,
                            amount=amount,
                            metadata={
                                "affliction": "tainted",
                                "trigger": "vital_spark_amount_changed",
                            },
                        )
                    )
            cards.append(
                card.model_copy(update={"custom": custom}) if custom != card.custom else card
            )
            seen.add(card.instance_id)
        updates[zone] = tuple(cards)
    metadata.update(
        {
            "source_vital_spark_sources": active,
            "source_vital_spark_generation": generation,
            "source_vital_spark_seen_cards": tuple(sorted(seen)),
            "source_vital_spark_version": version,
            "source_vital_spark_display_amount": display_amount,
        }
    )
    updates["metadata"] = metadata
    return combat.model_copy(update=updates), tuple(events)


def after_play(
    combat: CombatState, card: CardInstance
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    combat, lifecycle = synchronize(combat)
    actual = next(
        (c for zone in _ZONES for c in getattr(combat, zone) if c.instance_id == card.instance_id),
        card,
    )
    generation = int(combat.metadata.get("source_vital_spark_generation", 0))
    if (
        affliction_id(actual) != "tainted"
        or int(actual.custom.get(_GENERATION, generation)) != generation
    ):
        return combat, lifecycle
    events = list(lifecycle)
    for monster in combat.monsters:
        amount = t._status_amount(monster.statuses, "vital_spark")
        if monster.hp <= 0 or amount <= 0:
            continue
        statuses, event = t._apply_status_value(
            combat.player.statuses,
            "tainted",
            amount,
            source_id=monster.monster_id,
            target_id="player",
        )
        combat = combat.model_copy(
            update={"player": combat.player.model_copy(update={"statuses": statuses})}
        )
        if event is not None:
            events.append(
                event.model_copy(
                    update={
                        "metadata": {
                            **event.metadata,
                            "trigger": "vital_spark",
                            "card_id": card.card_id,
                        }
                    }
                )
            )
    return combat, tuple(events)


def copy_affliction(source: CardInstance, target: CardInstance) -> CardInstance:
    """Refresh mutable affliction identity without replacing captured play state."""
    custom = dict(target.custom)
    if affliction_id(target) == "tainted" and affliction_id(source) != "tainted":
        custom.pop("affliction", None)
    for key in (_KEY, _GENERATION, _VERSION):
        custom.pop(key, None)
        if key in source.custom:
            custom[key] = source.custom[key]
    return target.model_copy(update={"custom": custom})


def prepare_play(
    combat: CombatState,
    card: CardInstance,
) -> tuple[CombatState, CardInstance, tuple[EffectEvent, ...]]:
    from . import transitions as t

    actual = next(
        (c for zone in _ZONES for c in getattr(combat, zone) if c.instance_id == card.instance_id),
        None,
    )
    if actual is not None:
        return combat, copy_affliction(actual, card), ()
    seen = set(combat.metadata.get("source_vital_spark_seen_cards", ()))
    sources = combat.metadata.get("source_vital_spark_sources", {})
    if (
        card.instance_id in seen
        or card.type != CardType.SKILL
        or affliction_id(card)
        or t._card_has_keyword(card, "unplayable")
        or not isinstance(sources, Mapping)
        or not sources
    ):
        return combat, card, ()
    source, amount = next(iter(sources.items()))
    custom = {
        **card.custom,
        _KEY: {"id": "tainted", "amount": int(amount)},
        _GENERATION: int(combat.metadata.get("source_vital_spark_generation", 0)),
        _VERSION: int(combat.metadata.get("source_vital_spark_version", 0)),
    }
    seen.add(card.instance_id)
    combat = combat.model_copy(
        update={
            "metadata": {**combat.metadata, "source_vital_spark_seen_cards": tuple(sorted(seen))}
        }
    )
    return (
        combat,
        card.model_copy(update={"custom": custom}),
        (
            EffectEvent(
                kind="card_afflicted",
                source_id=str(source),
                target_id=card.instance_id,
                amount=int(amount),
                metadata={"affliction": "tainted", "trigger": "vital_spark_card_entered"},
            ),
        ),
    )
