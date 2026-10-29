"""Stateful adapters for source-defined class card operations.

These independent rules use the engine's ordinary damage, movement and trigger
primitives so generated cards and replays receive the same lifecycle callbacks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sts2sim.mechanics.triggers import GameTrigger

from . import source_cost_hooks
from .models import (
    CardInstance,
    CardType,
    CombatState,
    EffectEvent,
    RngState,
    RunPhase,
    RunState,
    TargetType,
)
from .rng import capture_random_state, random_from_state, random_stream
from .source_cost_hooks import set_local_energy_cost


def execute(
    state: RunState | None,
    combat: CombatState,
    rng_state: RngState,
    card: CardInstance,
    target_id: str | None,
    energy_spent: int,
    payload: Mapping[str, Any],
    relics: Sequence[str],
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    operation = str(payload.get("operation", ""))
    events: list[EffectEvent] = []

    def apply(effect: Mapping[str, Any], target: str | None = target_id) -> None:
        nonlocal combat, rng_state
        combat, rng_state, resolved = t._apply_effect_step(
            state=state,
            combat=combat,
            rng_state=rng_state,
            card=card,
            target_id=target,
            energy_spent=energy_spent,
            effect=effect,
            relics=relics,
        )
        events.extend(resolved)

    def event(kind: str, amount: int = 0, **metadata: Any) -> None:
        events.append(
            EffectEvent(kind=kind, source_id=card.instance_id, amount=amount, metadata=metadata)
        )

    if operation == "generate":
        count = (
            t._formula_amount(payload["count_formula"], combat=combat)
            if "count_formula" in payload
            else int(payload.get("count", 1))
        )
        card_id = str(payload["card_id"])
        destination = str(payload["destination"])
        spec = t._reward_card_spec(state, card_id) if state is not None else {"card_id": card_id}
        # Soul is also constructed by static generation helpers without a card
        # library. Its stable source constructor is zero energy.
        if card_id == "soul":
            spec = {
                "id": "SOUL",
                "type": "Skill",
                "cost": 0,
                "target": "Self",
                "cards_draw": 2,
                "description": "Draw 2 cards.",
                "keywords_key": ("Exhaust",),
                "exhausts": True,
                "upgrade": {"cards": "+1"},
                "upgrade_description": "Draw 3 cards.",
            }
        if payload.get("upgraded"):
            spec = {**spec, "upgraded": True}
        for _index in range(max(0, count)):
            before = {
                item.instance_id
                for zone in (
                    combat.hand,
                    combat.draw_pile,
                    combat.discard_pile,
                    combat.exhaust_pile,
                )
                for item in zone
            }
            key = {
                "hand": "add_card_to_hand",
                "draw_pile": "add_card_to_draw",
                "discard_pile": "add_card_to_discard",
                "exhaust_pile": "add_card_to_exhaust",
            }[destination]
            apply({key: {"card": spec}})
            if destination == "draw_pile" and payload.get("position") == "random":
                # The generation primitive inserts at the top. Source Add uses
                # one random insertion per card and preserves the other order.
                rng = random_from_state(rng_state)
                pile = [item for item in combat.draw_pile if item.instance_id in before]
                added = [item for item in combat.draw_pile if item.instance_id not in before]
                for item in added:
                    pile.insert(rng.randrange(len(pile) + 1), item)
                combat = combat.model_copy(update={"draw_pile": tuple(pile)})
                rng_state = capture_random_state(rng)
    elif operation == "dirge":
        for _ in range(max(0, energy_spent)):
            apply({"player_resource": {"resource": "summon", "amount": int(payload["summon"])}})
        apply(
            {
                "source_class": {
                    "operation": "generate",
                    "card_id": "soul",
                    "destination": "draw_pile",
                    "position": "random",
                    "count": energy_spent,
                    "upgraded": bool(payload.get("upgraded")),
                }
            }
        )
    elif operation == "target_hp_loss":
        target = next((m for m in combat.monsters if m.monster_id == target_id), None)
        if target is not None and target.hp > 0:
            combat, rng_state, damage_events = t._apply_enemy_hp_loss_effect(
                combat,
                rng_state,
                card,
                {"amount": int(payload["amount"])},
                target_id=target_id,
            )
            events.extend(damage_events)
    elif operation == "sacrifice":
        osty = t._osty_state(combat)
        if osty.get("alive", True):
            amount = int(osty.get("max_hp", 0)) * 2
            apply({"osty_action": {"action": "die"}})
            apply({"block": amount})
    elif operation == "shatter":
        # Shatter invokes EvokeNext twice per original orb, keeping the first
        # in place until its second evoke.
        for _ in range(len(combat.orbs)):
            if t._phase_after_combat(combat) != RunPhase.COMBAT:
                break
            apply({"evoke_orb": {"selector": "leftmost", "amount": 2}})
    elif operation == "target_lightning":
        count = sum(orb.orb_id == "lightning" for orb in combat.orbs)
        amount = t._lightning_orb_amount(3, t._player_focus(combat.player), relics)
        for _ in range(count * int(payload["repeats"])):
            monsters, damage_events = t._damage_monsters(
                combat.monsters,
                (target_id,) if target_id else (),
                amount,
                "orb:lightning",
                relics=relics,
            )
            combat = combat.model_copy(update={"monsters": tuple(monsters)})
            events.extend(damage_events)
    elif operation == "stun":
        combat = combat.model_copy(
            update={
                "monsters": tuple(
                    item.model_copy(
                        update={
                            "statuses": {**item.statuses, "stunned": 1},
                            "intent": "Stun",
                            "intent_damage": 0,
                        }
                    )
                    if item.monster_id == target_id and item.hp > 0
                    else item
                    for item in combat.monsters
                )
            }
        )
        event("monster_stunned", 1, target_id=target_id)
    elif operation == "all_players_energy":
        amount = int(payload["amount"])
        apply({"energy": amount})
        allies = tuple(
            {**ally, "energy": int(ally.get("energy", 0)) + amount}
            if ally.get("alive", True) and ally.get("is_player", True)
            else ally
            for ally in t._ally_states(combat)
        )
        combat = combat.model_copy(update={"metadata": {**combat.metadata, "allies": allies}})
    elif operation == "blade_power":
        apply({"apply_status": {"target": "self", str(payload["power"]): int(payload["amount"])}})
        combat = refresh_blades(combat)
    elif operation == "retrieve":
        zones = (
            ("draw_pile", "discard_pile", "exhaust_pile")
            if payload.get("zone") == "any"
            else (str(payload["zone"]),)
        )
        candidates = [item for zone in zones for item in t._cards_in_combat_zone(combat, zone)]
        if payload.get("rarity"):
            candidates = [
                item
                for item in candidates
                if str(t._card_source_spec(item).get("rarity", "")).lower() == payload["rarity"]
            ]
        if payload.get("card_id"):
            candidates = [item for item in candidates if item.card_id.lower() == payload["card_id"]]
        if payload.get("zero_cost"):
            candidates = [
                item
                for item in candidates
                if item.type.value in {"attack", "skill", "power"}
                and not t._card_is_x_cost(item)
                and t._energy_cost(item, combat.player.energy, combat=combat) == 0
            ]
        rng = random_from_state(rng_state)
        if payload.get("random"):
            rng.shuffle(candidates)
        candidates = candidates[: max(0, t.MAX_HAND_SIZE - len(combat.hand))]
        rng_state = capture_random_state(rng)
        for item in candidates:
            for zone in zones:
                combat = t._remove_card_from_combat_zone(combat, zone, item.instance_id)
            combat = combat.model_copy(update={"hand": combat.hand + (item,)})
            events.append(
                EffectEvent(
                    kind="card_moved_by_effect",
                    source_id=card.instance_id,
                    target_id=item.instance_id,
                    amount=1,
                    metadata={"to_pile": "hand"},
                )
            )
    elif operation == "transform_statuses" and state is not None:
        for item in tuple(combat.hand):
            if item.type.value != "status":
                continue
            spec = {
                **t._reward_card_spec(state, str(payload["target_card_id"])),
                "instance_id": item.instance_id,
                "upgraded": bool(payload.get("upgraded")),
            }
            replacement = t._card_from_spec(spec, 1)
            combat = t._replace_card_in_combat_zone(combat, "hand", replacement)
            events.append(
                EffectEvent(
                    kind="card_transformed_by_effect",
                    source_id=card.instance_id,
                    target_id=item.instance_id,
                    metadata={"new_card_id": replacement.card_id},
                )
            )
    elif operation in {"repeat_attack", "calculated_damage"}:
        basis = str(payload["basis"])
        all_cards = combat.hand + combat.draw_pile + combat.discard_pile + combat.exhaust_pile
        if basis == "orbs":
            value = len(combat.orbs)
        elif basis == "skills_turn":
            value = t._metadata_int(combat.metadata, "skills_played_this_turn", 0)
        elif basis == "ethereal_combat":
            value = t._metadata_int(combat.metadata, "source_class_ethereal_plays", 0)
        elif basis == "created_combat":
            value = len(t._string_items(combat.metadata.get("created_card_hook_ids")))
        elif basis == "star_cards":
            # The currently played spear remains in AllCards in the source.
            cards_by_id = {item.instance_id: item for item in all_cards}
            cards_by_id.setdefault(card.instance_id, card)
            value = sum(
                "star_cost" in item.custom
                or t._card_source_spec(item).get("star_cost") is not None
                or t._card_is_x_star_cost(item)
                for item in cards_by_id.values()
            )
        else:
            value = t._metadata_int(combat.metadata, "source_class_draws_turn", 0)
        if operation == "repeat_attack":
            apply({"repeat_damage": {"fixed_hits": max(0, value),
                                     "amount": int(payload["amount"])}})
        else:
            apply({"damage": int(payload["base"]) + int(payload["extra"]) * value})
    elif operation == "no_escape":
        target = next((m for m in combat.monsters if m.monster_id == target_id), None)
        value = t._status_amount(target.statuses, "doom") if target is not None else 0
        apply(
            {
                "apply_status": {
                    "target": "enemy",
                    "doom": int(payload["base"])
                    + int(payload["extra"]) * (value // int(payload["divisor"])),
                }
            }
        )
    elif operation == "doom_kill":
        for monster in tuple(combat.monsters):
            if monster.hp > 0 and t._status_amount(monster.statuses, "doom") >= monster.hp:
                combat = combat.model_copy(
                    update={
                        "monsters": tuple(
                            item.model_copy(update={"hp": 0})
                            if item.monster_id == monster.monster_id
                            else item
                            for item in combat.monsters
                        )
                    }
                )
                events.append(
                    EffectEvent(
                        kind="monster_defeated",
                        source_id=card.instance_id,
                        target_id=monster.monster_id,
                        amount=monster.hp,
                        metadata={"reason": "doom"},
                    )
                )
    elif operation == "spread_debuffs":
        from sts2sim.mechanics.source_power_rules import source_debuff_copy_payloads

        target = next((m for m in combat.monsters if m.monster_id == target_id), None)
        if target is not None:
            for status, amount in source_debuff_copy_payloads(target.statuses):
                for other in tuple(combat.monsters):
                    if other.hp > 0 and other.monster_id != target_id:
                        apply(
                            {"apply_status": {"target": "enemy", status: amount}}, other.monster_id
                        )
    elif operation in {"hang_damage", "hang_double"}:
        target = next((m for m in combat.monsters if m.monster_id == target_id), None)
        amount = t._status_amount(target.statuses, "hang") if target is not None else 0
        if operation == "hang_damage":
            # HangPower's multiplier belongs in target damage math, after
            # additive Strength, together with Vulnerable and other factors.
            apply({"damage": int(payload["amount"])})
        else:
            hang_added = min(max(2, amount), max(0, 999999999 - amount))
            if hang_added:
                apply({"apply_status": {"target": "enemy", "hang": hang_added}})
    elif operation == "beat_forge":
        hits = t._mapping_from(combat.metadata.get("hits_by_target_this_turn"))
        count = int(hits.get(target_id or "", 0))
        apply(
            {
                "player_resource": {
                    "resource": "forge",
                    "amount": int(payload["base"]) + int(payload["extra"]) * max(0, count - 1),
                }
            }
        )
    elif operation == "flak_cannon":
        statuses = [
            (zone, item)
            for zone in ("hand", "draw_pile", "discard_pile")
            for item in t._cards_in_combat_zone(combat, zone)
            if item.type.value == "status"
        ]
        for zone, item in statuses:
            combat = t._remove_card_from_combat_zone(combat, zone, item.instance_id)
            combat = combat.model_copy(update={"exhaust_pile": combat.exhaust_pile + (item,)})
            movement = t._card_movement_events(
                GameTrigger.CARD_EXHAUSTED,
                (item,),
                from_pile=zone,
                to_pile="exhaust_pile",
                source_id=card.instance_id,
                reason="flak_cannon",
            )
            events.extend(movement)
            if state is not None:
                combat, rng_state, triggered = t._apply_card_exhausted_triggers_from_events(
                    state, combat, rng_state, movement
                )
                events.extend(triggered)
        for _ in statuses:
            rng = random_from_state(rng_state)
            random_target_id = t._random_alive_monster_id(combat, rng)
            rng_state = capture_random_state(rng)
            if random_target_id is None:
                break
            apply({"damage": int(payload["amount"])}, random_target_id)
    elif operation == "reboot":
        rng = random_from_state(rng_state)
        pile = list(combat.draw_pile + combat.hand)
        rng.shuffle(pile)
        combat = combat.model_copy(update={"hand": (), "draw_pile": tuple(pile)})
        rng_state = capture_random_state(rng)
        event("draw_pile_shuffled", len(pile), reason="reboot")
    elif operation == "scrape":
        before = {item.instance_id for item in combat.hand}
        apply({"draw": int(payload["count"])})
        discarded = tuple(
            item
            for item in combat.hand
            if item.instance_id not in before
            and (
                t._card_is_x_cost(item)
                or t._energy_cost(item, combat.player.energy, combat=combat) != 0
            )
        )
        combat = combat.model_copy(
            update={
                "hand": tuple(item for item in combat.hand if item not in discarded),
                "discard_pile": combat.discard_pile + discarded,
            }
        )
        movement = t._card_movement_events(
            GameTrigger.CARD_DISCARDED,
            discarded,
            from_pile="hand",
            to_pile="discard_pile",
            source_id=card.instance_id,
            reason="scrape",
        )
        events.extend(movement)
        if state is not None:
            combat, rng_state, triggered = t._apply_card_discarded_relics_from_events(
                state, combat, rng_state, movement
            )
            events.extend(triggered)
    elif operation == "random_attack" and state is not None:
        rng = random_from_state(rng_state)
        candidates = [item for item in combat.draw_pile if item.type.value == "attack"]
        if candidates:
            selected = rng.choice(candidates)
            combat, rng_state, played = t._play_combat_card_from_zone_by_trigger(
                state,
                combat,
                capture_random_state(rng),
                card,
                selected,
                from_zone="draw_pile",
                target_mode="random_enemy",
            )
            events.extend(played)
    elif operation in {"grow_matching", "permanent_growth"}:
        field = str(payload.get("field", "damage"))
        amount = int(payload["amount"])
        metadata = dict(combat.metadata)
        updates = dict(t._mapping_from(metadata.get("played_card_updates")))
        update = dict(t._mapping_from(updates.get(card.instance_id)))
        key = f"source_class_{field}_bonus"
        update[key] = int(update.get(key, 0)) + amount
        updates[card.instance_id] = update
        metadata["played_card_updates"] = updates
        if operation == "permanent_growth":
            growth = dict(t._mapping_from(metadata.get("source_class_permanent_growth")))
            growth[card.instance_id] = int(growth.get(card.instance_id, 0)) + amount
            metadata["source_class_permanent_growth"] = growth
        combat = combat.model_copy(update={"metadata": metadata})
        if operation == "grow_matching":
            for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
                items = tuple(
                    _increase(item, field, amount) if item.card_id == card.card_id else item
                    for item in t._cards_in_combat_zone(combat, zone)
                )
                combat = t._set_combat_zone_cards(combat, zone, items)
        event(
            "card_growth",
            amount,
            field=field,
            scope="run" if operation == "permanent_growth" else "combat",
        )
    elif operation == "clone_discard":
        clone = card.model_copy(
            update={
                "instance_id": t._unique_generated_card_id(
                    combat, f"{card.instance_id}:clone:{combat.turn}"
                ),
                "custom": {**card.custom, "generated": True},
            }
        )
        if payload.get("zero_cost"):
            clone = set_local_energy_cost(clone, 0, "combat")
        apply({"add_card_to_discard": {"card": clone.model_dump(mode="json")}})
    elif operation == "upgrade_random":
        rng = random_from_state(rng_state)
        zone = str(payload["zone"])
        candidates = [item for item in t._cards_in_combat_zone(combat, zone) if not item.upgraded]
        rng.shuffle(candidates)
        for item in candidates[: int(payload["count"])]:
            combat = t._replace_card_in_combat_zone(combat, zone, t._upgrade_card_instance(item))
            events.append(
                EffectEvent(
                    kind="card_upgraded",
                    source_id=card.instance_id,
                    target_id=item.instance_id,
                    amount=1,
                )
            )
        rng_state = capture_random_state(rng)
    elif operation == "set_self_cost":
        metadata = dict(combat.metadata)
        updates = dict(t._mapping_from(metadata.get("played_card_updates")))
        updates[card.instance_id] = {
            **t._mapping_from(updates.get(card.instance_id)),
            "source_class_set_cost": int(payload["cost"]),
        }
        combat = combat.model_copy(
            update={"metadata": {**metadata, "played_card_updates": updates}}
        )
    elif operation == "install_cost":
        powers = dict(t._mapping_from(combat.metadata.get("source_class_cost_powers")))
        power = str(payload["power"])
        if power == "borrowed_time":
            apply({"apply_status": {"target": "self", "borrowed_time": int(payload["amount"])}})
            return combat, rng_state, tuple(events)
        stored = dict(t._mapping_from(powers.get(power)))
        stored["amount"] = int(stored.get("amount", 0)) + int(payload["amount"])
        if power == "orbit":
            instances = list(t._source_sequence(stored.get("instances")))
            instances.append(
                {
                    "spent": 0,
                    "seen": t._metadata_int(combat.metadata, "energy_spent_this_turn", 0),
                    "turn": combat.turn,
                }
            )
            stored["instances"] = instances
        if payload.get("duration") == "turn":
            stored["turn"] = combat.turn
        powers[power] = stored
        combat = combat.model_copy(
            update={"metadata": {**combat.metadata, "source_class_cost_powers": powers}}
        )
    elif operation == "random_generate" and state is not None:
        # Reuse the generated-card factory, then move only the actual created
        # cards to the requested source pile.
        before = {item.instance_id for zone in (combat.hand, combat.discard_pile) for item in zone}
        combat, rng_state, created = t._add_random_pool_cards_to_hand(
            state, combat, rng_state, card, payload
        )
        events.extend(created)
        generated = tuple(
            item
            for zone in (combat.hand, combat.discard_pile)
            for item in zone
            if item.instance_id not in before
        )
        rng = random_from_state(rng_state)
        pile = list(combat.draw_pile)
        for item in generated:
            combat = t._remove_card_from_combat_zone(combat, "hand", item.instance_id)
            combat = t._remove_card_from_combat_zone(combat, "discard_pile", item.instance_id)
            pile.insert(rng.randrange(len(pile) + 1), item)
        combat = combat.model_copy(update={"draw_pile": tuple(pile)})
        rng_state = capture_random_state(rng)
    return combat, rng_state, tuple(events)


def _increase(card: CardInstance, field: str, amount: int) -> CardInstance:
    from . import transitions as t

    steps = tuple(
        {**step, field: int(step[field]) + amount}
        if isinstance(step.get(field), int)
        else dict(step)
        for step in t._effect_steps(card.effects)
    )
    key = f"source_class_{field}_bonus"
    return card.model_copy(
        update={
            "effects": {"sequence": steps},
            "custom": {**card.custom, key: int(card.custom.get(key, 0)) + amount},
        }
    )


def played_updates(card: CardInstance, update: Mapping[str, Any]) -> CardInstance:
    for field in ("damage", "block"):
        amount = int(update.get(f"source_class_{field}_bonus", 0))
        if amount:
            card = _increase(card, field, amount)
    if "source_class_set_cost" in update:
        card = set_local_energy_cost(card, int(update["source_class_set_cost"]), "combat")
    return card


def persist_growth(state: RunState, combat: CombatState) -> tuple[RunState, CombatState]:
    from . import transitions as t

    growth = t._mapping_from(combat.metadata.get("source_class_permanent_growth"))
    if not growth:
        return state, combat
    deck = tuple(
        _increase(
            item,
            "block" if item.card_id.lower() == "genetic_algorithm" else "damage",
            int(growth[item.instance_id]),
        )
        if item.instance_id in growth
        else item
        for item in state.master_deck
    )
    metadata = dict(combat.metadata)
    metadata.pop("source_class_permanent_growth", None)
    return state.model_copy(update={"master_deck": deck}), combat.model_copy(
        update={"metadata": metadata}
    )


def preserve_changes(before: CardInstance, after: CardInstance) -> CardInstance:
    """Source upgrades retain combat and permanently saved card growth."""
    for field in ("damage", "block"):
        amount = int(before.custom.get(f"source_class_{field}_bonus", 0))
        if amount:
            after = _increase(after, field, amount)
    return after


def after_combat(
    state: RunState,
    combat: CombatState,
    rng_state: RngState,
) -> tuple[RunState, RngState, tuple[EffectEvent, ...]]:
    """Improvement upgrades distinct mutable deck cards via CombatCardSelection."""
    from . import transitions as t

    amount = t._status_amount(combat.player.statuses, "improvement") + t._status_amount(
        combat.player.statuses,
        "end_of_combat_upgrade_random",
    )
    if not amount:
        return state, rng_state, ()
    candidates = [card for card in state.master_deck if t._card_is_upgradable(card)]
    deck = list(state.master_deck)
    rng = random_stream(random_from_state(rng_state), "combat_card_selection")
    events: list[EffectEvent] = []
    for _ in range(amount):
        if not candidates:
            break
        card = rng.choice(candidates)
        candidates.remove(card)
        deck[deck.index(card)] = t._upgrade_card_instance(card)
        events.append(
            EffectEvent(
                kind="card_upgraded",
                source_id="improvement",
                target_id=card.instance_id,
                metadata={"card_id": card.card_id, "trigger": "combat_end"},
            )
        )
    return (
        state.model_copy(update={"master_deck": tuple(deck)}),
        capture_random_state(rng),
        tuple(events),
    )


def prepare_blade(combat: CombatState, card: CardInstance) -> CardInstance:
    from . import transitions as t

    if not t._is_sovereign_blade_card(card):
        return card
    amount = t._sovereign_blade_base_damage(card)
    hits = t._sovereign_blade_hit_count(card)
    custom = dict(card.custom)
    seeking = t._status_amount(combat.player.statuses, "seeking_edge") > 0
    target = TargetType.ALL_ENEMIES if seeking else card.target
    steps = [
        {"all_damage" if target == TargetType.ALL_ENEMIES else "damage": amount}
        for _ in range(hits)
    ]
    parry = t._status_amount(combat.player.statuses, "parry")
    if parry:
        steps.append({"block": parry})
    sage = t._status_amount(combat.player.statuses, "sword_sage")
    applied = int(custom.get("source_class_sage_applied", 0))
    if sage > applied:
        custom["replay"] = int(custom.get("replay", 0)) + sage - applied
        custom["source_class_sage_applied"] = sage
    return card.model_copy(
        update={"target": target, "effects": {"sequence": steps}, "custom": custom}
    )


def refresh_blades(combat: CombatState) -> CombatState:
    from . import transitions as t

    for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        items = tuple(prepare_blade(combat, item) for item in t._cards_in_combat_zone(combat, zone))
        combat = t._set_combat_zone_cards(combat, zone, items)
    return combat


def forge_resource(
    state: RunState,
    combat: CombatState,
    rng_state: RngState,
    event: EffectEvent,
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    if event.kind != "player_resource_changed" or event.metadata.get("resource") != "forge":
        return combat, rng_state, ()
    amount = int(event.amount or 0)
    if amount < 0:
        return combat, rng_state, ()
    events: tuple[EffectEvent, ...] = ()
    blades = tuple(
        item
        for zone in (combat.hand, combat.draw_pile, combat.discard_pile)
        for item in zone
        if t._is_sovereign_blade_card(item)
    )
    if not blades:
        source = CardInstance(
            instance_id=event.source_id or "forge", card_id="forge", type=CardType.SKILL
        )
        spec = t._reward_card_spec(state, "sovereign_blade")
        combat, rng_state, events = t._apply_effect_step(
            state=state,
            combat=combat,
            rng_state=rng_state,
            card=source,
            target_id=None,
            energy_spent=0,
            effect={"add_card_to_hand": {"card": spec}},
            relics=state.relics,
        )
    for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        cards = []
        for item in t._cards_in_combat_zone(combat, zone):
            if t._is_sovereign_blade_card(item):
                damage = t._sovereign_blade_base_damage(item) + amount
                item = item.model_copy(
                    update={"custom": {**item.custom, "sovereign_blade_base_damage": damage}}
                )
                item = prepare_blade(combat, item)
            cards.append(item)
        combat = t._set_combat_zone_cards(combat, zone, cards)
    return combat, rng_state, events


def energy_cost(card: CardInstance, combat: CombatState | None, cost: int) -> int:
    from . import transitions as t

    if combat is None:
        return cost
    if card.type == CardType.POWER and cost > 0:
        cost = max(
            0,
            cost
            - t._status_amount(combat.player.statuses, "curious")
            - t._status_amount(combat.player.statuses, "power_cost_reduction"),
        )
    return cost + t._status_amount(combat.player.statuses, "borrowed_time")


def late_energy_cost(card: CardInstance, combat: CombatState | None, cost: int) -> int:
    from . import transitions as t

    if combat is None:
        return cost
    in_hand_or_play = any(item.instance_id == card.instance_id for item in combat.hand) or not any(
        item.instance_id == card.instance_id
        for zone in (combat.draw_pile, combat.discard_pile, combat.exhaust_pile)
        for item in zone
    )
    powers = t._mapping_from(combat.metadata.get("source_class_cost_powers"))
    for power, stored in powers.items():
        value = t._mapping_from(stored)
        if value.get("turn", combat.turn) != combat.turn or not value.get("amount"):
            continue
        if power == "free_power" and card.type.value == "power" and in_hand_or_play:
            return 0
        if power == "free_ethereal" and t._card_has_keyword(card, "ethereal") and in_hand_or_play:
            return 0
    return cost


def before_play(combat: CombatState, card: CardInstance, energy_spent: int) -> CombatState:
    from . import transitions as t

    metadata = dict(combat.metadata)
    powers = dict(t._mapping_from(metadata.get("source_class_cost_powers")))
    for power, matches in (
        ("free_power", card.type.value == "power"),
        ("free_ethereal", t._card_has_keyword(card, "ethereal")),
    ):
        value = dict(t._mapping_from(powers.get(power)))
        if matches and int(value.get("amount", 0)) > 0:
            value["amount"] = int(value["amount"]) - 1
            powers[power] = value
    orbit = dict(t._mapping_from(powers.get("orbit")))
    refund = 0
    if orbit:
        instances = []
        total = t._metadata_int(combat.metadata, "energy_spent_this_turn", 0)
        for raw in t._source_sequence(orbit.get("instances")):
            instance = dict(t._mapping_from(raw))
            seen = int(instance.get("seen", 0)) if instance.get("turn") == combat.turn else 0
            old_spent = int(instance.get("spent", 0))
            spent = old_spent + max(0, total - seen)
            refund += spent // 4 - old_spent // 4
            instances.append({"seen": total, "spent": spent, "turn": combat.turn})
        orbit["instances"] = instances
        powers["orbit"] = orbit
    metadata["source_class_cost_powers"] = powers
    player = combat.player
    if refund:
        from sts2sim.mechanics.source_resource_rules import permitted_energy_delta

        player = player.model_copy(
            update={"energy": player.energy + permitted_energy_delta(player.statuses, refund)}
        )
    return combat.model_copy(update={"metadata": metadata, "player": player})


def after_play(combat: CombatState, card: CardInstance) -> CombatState:
    """Inherent returns trigger from all zones, even before their first play."""
    from . import transitions as t

    if t._card_has_keyword(card, "ethereal"):
        combat = combat.model_copy(
            update={
                "metadata": {
                    **combat.metadata,
                    "source_class_ethereal_plays": t._metadata_int(
                        combat.metadata, "source_class_ethereal_plays", 0
                    )
                    + 1,
                }
            }
        )
        # Banshee's Cry appends AddThisCombat(-2), preserving its position
        # relative to later absolute and relative energy-cost modifiers.
        combat = source_cost_hooks.synchronize_count_cards(combat)
    skills = t._metadata_int(combat.metadata, "skills_played_this_turn", 0)
    candidates = []
    for item in combat.discard_pile:
        if item.card_id.lower() == "make_it_so" and card.type.value == "skill" and skills % 3 == 2:
            candidates.append(item)
    for item in candidates[: max(0, t.MAX_HAND_SIZE - len(combat.hand))]:
        combat = t._remove_card_from_combat_zone(combat, "discard_pile", item.instance_id)
        combat = combat.model_copy(update={"hand": combat.hand + (item,)})
    return combat
