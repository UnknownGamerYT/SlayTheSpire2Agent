"""Runtime reactions for source-backed poison, damage and resource powers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .models import (
    CardInstance,
    CardType,
    CombatState,
    EffectEvent,
    MonsterState,
    RngState,
    RunState,
)


def card_actor_id(combat: CombatState, card: CardInstance) -> str:
    return str(card.custom.get("owner_id", card.custom.get("player_id", local_player_id(combat))))


def local_player_id(combat: CombatState) -> str:
    return str(combat.metadata.get("player_id", "player"))


def card_actor_statuses(combat: CombatState, card: CardInstance) -> Mapping[str, int]:
    actor_id = card_actor_id(combat, card)
    if actor_id == local_player_id(combat):
        return combat.player.statuses
    for ally in combat.metadata.get("allies", ()):
        if isinstance(ally, Mapping) and str(ally.get("id", "")) == actor_id:
            return {str(k): int(v) for k, v in ally.get("statuses", {}).items()}
    return {}


def record_player_attack(
    combat: CombatState,
    card: CardInstance,
    events: Sequence[EffectEvent],
) -> CombatState:
    """Record one attack per player and enemy, irrespective of hit count or Block."""
    if card.type != CardType.ATTACK:
        return combat
    actor_id = card_actor_id(combat, card)
    if actor_id == "osty" or card.custom.get("owner_kind", "player") != "player":
        return combat
    targets = {
        event.target_id
        for event in events
        if event.kind == "monster_damaged"
        and event.source_id == card.instance_id
        and event.target_id is not None
        and event.metadata.get("attacking_player_id", actor_id) == actor_id
        and not event.metadata.get("card_play_hooks_resolved")
    }
    if not targets:
        return combat
    counts = {
        str(target): dict(players)
        for target, players in combat.metadata.get("player_attacks_by_target_this_turn", {}).items()
    }
    for target_id in targets:
        players = counts.setdefault(target_id, {})
        players[actor_id] = int(players.get(actor_id, 0)) + 1
    return combat.model_copy(
        update={
            "metadata": {
                **combat.metadata,
                "player_attacks_by_target_this_turn": counts,
            }
        }
    )


def other_player_attacks(combat: CombatState, target_id: str, actor_id: str) -> int:
    counts = combat.metadata.get("player_attacks_by_target_this_turn", {}).get(target_id, {})
    return sum(max(0, int(count)) for player_id, count in counts.items() if player_id != actor_id)


def record_multiplier_owner(
    monster: MonsterState,
    status: str,
    value: int,
    owner_id: str,
) -> MonsterState:
    if status not in {"flanking", "knockdown"} or value <= 0:
        return monster
    key = f"{status}_applications"
    applications = tuple(monster.metadata.get(key, ()))
    multiplier = 2 if status == "flanking" else max(1, value)
    return monster.model_copy(
        update={
            "metadata": {
                **monster.metadata,
                key: (*applications, {"owner_id": owner_id, "multiplier": multiplier}),
            }
        }
    )


def other_player_damage_multiplier(
    monster: MonsterState,
    actor_id: str,
    *,
    is_attack: bool,
) -> float:
    multiplier = 1.0
    for status in ("flanking", "knockdown"):
        if monster.statuses.get(status, 0) <= 0 or (status == "flanking" and not is_attack):
            continue
        for application in monster.metadata.get(f"{status}_applications", ()):
            if isinstance(application, Mapping) and application.get("owner_id") != actor_id:
                multiplier *= max(1, int(application.get("multiplier", 1)))
    return multiplier


def debuff_reactions(
    combat: CombatState,
    source_event: EffectEvent,
    *,
    relics: Sequence[str],
    is_debuff: bool,
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    from . import transitions as e

    target_id = source_event.target_id
    if (
        source_event.kind != "status_applied"
        or target_id is None
        or target_id == "player"
        or not source_event.amount
    ):
        return combat, ()
    status = str(source_event.metadata.get("status", ""))
    events: list[EffectEvent] = []
    target = next((m for m in combat.monsters if m.monster_id == target_id), None)
    if target is None:
        return combat, ()
    if status == "poison" and (source_event.amount or 0) > 0:
        outbreak = e._status_amount(combat.player.statuses, "outbreak")
        if outbreak:
            count = int(combat.metadata.get("outbreak_poison_applications", 0)) + 1
            combat = combat.model_copy(
                update={
                    "metadata": {
                        **combat.metadata,
                        "outbreak_poison_applications": count % 3,
                    }
                }
            )
            events.append(
                EffectEvent(
                    kind="outbreak_counted",
                    source_id="outbreak",
                    target_id=target_id,
                    amount=count % 3,
                )
            )
            if count % 3 == 0:
                combat, damage_events = e._damage_all_monsters(
                    combat,
                    outbreak,
                    "outbreak",
                    relics=relics,
                    metadata={"status": "outbreak", "trigger": "poison_applied"},
                )
                events.extend(damage_events)
    damage = e._status_amount(combat.player.statuses, "sleight_of_flesh")
    if not damage:
        damage = e._status_amount(combat.player.statuses, "sleightofflesh")
    if damage and is_debuff and target.hp > 0:
        monsters, damage_events = e._damage_monsters(
            combat.monsters,
            (target_id,),
            damage,
            "sleight_of_flesh",
            relics=relics,
        )
        combat = combat.model_copy(update={"monsters": tuple(monsters)})
        events.extend(
            event.model_copy(
                update={
                    "metadata": {
                        **event.metadata,
                        "status": "sleight_of_flesh",
                        "debuff": status,
                    }
                }
            )
            for event in damage_events
        )
    return combat, tuple(events)


def black_hole_resource_events(
    combat: CombatState,
    source_events: Sequence[EffectEvent],
    *,
    relics: Sequence[str] = (),
    defer: bool = False,
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    from . import transitions as e

    amount = e._status_amount(combat.player.statuses, "black_hole")
    if amount <= 0:
        return combat, ()
    pending = list(combat.metadata.get("pending_black_hole", ()))
    for event in source_events:
        if event.kind not in {"player_resource_changed", "player_resource_spent"}:
            continue
        if str(event.metadata.get("resource", "")) not in {"star", "stars", "starlight"}:
            continue
        if not event.amount or (event.kind == "player_resource_changed" and event.amount < 0):
            continue
        pending.append({"damage": amount, "source_id": event.source_id})
    if len(pending) == len(combat.metadata.get("pending_black_hole", ())):
        return combat, ()
    combat = combat.model_copy(
        update={"metadata": {**combat.metadata, "pending_black_hole": pending}}
    )
    return (combat, ()) if defer else flush_black_hole(combat, relics=relics)


def flush_black_hole(
    combat: CombatState,
    *,
    relics: Sequence[str] = (),
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    from . import transitions as e

    pending = tuple(combat.metadata.get("pending_black_hole", ()))
    if not pending or combat.pending_choices:
        return combat, ()
    metadata = dict(combat.metadata)
    metadata.pop("pending_black_hole", None)
    combat = combat.model_copy(update={"metadata": metadata})
    events: list[EffectEvent] = []
    if combat.player.hp <= 0:
        return combat, ()
    for entry in pending:
        combat, damage_events = e._damage_all_monsters(
            combat,
            int(entry["damage"]),
            "black_hole",
            relics=relics,
            metadata={"status": "black_hole", "trigger_source": entry.get("source_id")},
        )
        events.extend(damage_events)
    return combat, tuple(events)


def created_card_reactions(
    state: RunState,
    combat: CombatState,
    rng_state: RngState,
    source_event: EffectEvent,
    card: CardInstance,
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    from . import transitions as e

    if source_event.metadata.get("card_play_hooks_resolved"):
        return combat, rng_state, ()
    if source_event.source_id in {m.monster_id for m in combat.monsters}:
        return combat, rng_state, ()
    seen = set(combat.metadata.get("runtime_created_card_ids", ()))
    if card.instance_id in seen:
        return combat, rng_state, ()
    seen.add(card.instance_id)
    combat = combat.model_copy(
        update={
            "metadata": {
                **combat.metadata,
                "runtime_created_card_ids": tuple(sorted(seen)),
            }
        }
    )
    events: list[EffectEvent] = []
    if card.type == CardType.STATUS:
        amount = e._status_amount(combat.player.statuses, "smokestack")
        if amount:
            combat, damage_events = e._damage_all_monsters(
                combat,
                amount,
                "smokestack",
                relics=state.relics,
                metadata={"status": "smokestack", "created_card_id": card.card_id},
            )
            events.extend(damage_events)
    context = {"created_card": card, "card_id": card.card_id, "target_id": card.instance_id}
    for trigger in (
        ("card_created", "status_card_created")
        if card.type == CardType.STATUS
        else ("card_created",)
    ):
        combat, rng_state, trigger_events = e._apply_combat_triggers(
            state.model_copy(update={"combat": combat, "player": combat.player}),
            combat,
            rng_state,
            trigger=trigger,
            context=context,
        )
        events.extend(trigger_events)
    return combat, rng_state, tuple(events)


def sneaky_attack_play(
    state: RunState | None,
    combat: CombatState,
    rng_state: RngState,
    card: CardInstance,
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    from . import transitions as e

    if card.type != CardType.ATTACK or not any(m.hp > 0 for m in combat.monsters):
        return combat, rng_state, ()
    actor_id = card_actor_id(combat, card)
    events: list[EffectEvent] = []
    amount = e._status_amount(combat.player.statuses, "sneaky")
    if actor_id != local_player_id(combat) and amount:
        combat, rng_state, block_events = e._gain_player_block(
            state,
            combat,
            rng_state,
            amount,
            source_id="sneaky",
            source_card=None,
            metadata={"status": "sneaky", "attacking_player_id": actor_id},
        )
        events.extend(block_events)
    allies = [dict(ally) for ally in e._ally_states(combat)]
    for ally in allies:
        ally_id = str(ally.get("id", ""))
        statuses = ally.get("statuses", {})
        amount = e._status_amount(statuses, "sneaky") if isinstance(statuses, Mapping) else 0
        if ally_id != actor_id and amount:
            ally["block"] = int(ally.get("block", 0)) + amount
            events.append(
                EffectEvent(
                    kind="ally_block",
                    source_id="sneaky",
                    target_id=ally_id,
                    amount=amount,
                    metadata={"attacking_player_id": actor_id},
                )
            )
    if allies:
        combat = combat.model_copy(
            update={"metadata": {**combat.metadata, "allies": tuple(allies)}}
        )
    return combat, rng_state, tuple(events)
