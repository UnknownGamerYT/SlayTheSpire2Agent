"""Stateful core-card operations derived from the pinned stable C# models.

See mechanics.source_core_rules for provenance. Reuse engine operations so
exhaust/discard/draw/child plays retain their normal hooks and serialization.
"""

from __future__ import annotations

from sts2sim.mechanics.source_core_rules import shiv_spec

from .models import CardType, EffectEvent


def apply_source_core(*, state, combat, rng_state, card, target_id, energy_spent, effect, relics):
    from . import transitions as t

    name = effect.get("source_core")
    if not name:
        return None
    up = card.upgraded
    events = []

    def run(step, *, target=target_id, source=card):
        nonlocal combat, rng_state
        combat, rng_state, resolved = t._apply_effect_step(
            state=state,
            combat=combat,
            rng_state=rng_state,
            card=source,
            target_id=target,
            energy_spent=energy_spent,
            effect=step,
            relics=relics,
        )
        events.extend(resolved)
        return resolved

    def exhaust(cards):
        nonlocal combat, rng_state
        for chosen in tuple(cards):
            if t._find_card(combat.hand, chosen.instance_id) is None:
                continue
            combat = t._remove_card_from_combat_zone(combat, "hand", chosen.instance_id)
            combat = combat.model_copy(update={"exhaust_pile": combat.exhaust_pile + (chosen,)})
            moved = t._card_movement_events(
                t.GameTrigger.CARD_EXHAUSTED,
                (chosen,),
                from_pile="hand",
                to_pile="exhaust_pile",
                source_id=card.instance_id,
                reason="source_card_exhaust",
            )
            events.extend(moved)
            if state is not None:
                combat, rng_state, triggered = t._apply_card_exhausted_triggers_from_events(
                    state,
                    combat,
                    rng_state,
                    moved,
                )
                events.extend(triggered)

    def add_damage_bonus(amount):
        nonlocal combat
        updates = dict(combat.metadata.get("played_card_updates", {}))
        update = dict(updates.get(card.instance_id, {}))
        update["source_core_damage_bonus"] = int(update.get("source_core_damage_bonus", 0)) + amount
        updates[card.instance_id] = update
        combat = combat.model_copy(
            update={"metadata": {**combat.metadata, "played_card_updates": updates}}
        )
        events.append(
            EffectEvent(
                kind="card_damage_increased",
                source_id=card.instance_id,
                target_id=card.instance_id,
                amount=amount,
            )
        )

    def bonus():
        return int(card.custom.get("source_core_damage_bonus", 0)) + int(
            combat.metadata.get("played_card_updates", {})
            .get(card.instance_id, {})
            .get("source_core_damage_bonus", 0)
        )

    def play(chosen, zone, *, force_exhaust=False, fixed_target=None):
        nonlocal combat, rng_state
        if state is None:
            return
        if force_exhaust:
            chosen = chosen.model_copy(update={"exhausts": True})
            combat = t._replace_card_in_combat_zone(combat, zone, chosen)
        if fixed_target is None:
            combat, rng_state, resolved = t._play_combat_card_from_zone_by_trigger(
                state,
                combat,
                rng_state,
                card,
                chosen,
                from_zone=zone,
                target_mode="random_enemy",
            )
        else:
            combat = t._remove_card_from_combat_zone(combat, zone, chosen.instance_id)
            combat, rng_state, resolved = t._resolve_extra_card_play(
                state,
                combat,
                rng_state,
                chosen,
                target_id=fixed_target,
                energy_spent=0,
                source_id=card.instance_id,
                source_metadata={"from_pile": zone},
                move_physical_card=True,
                physical_card=chosen,
                event_kind="card_played_by_trigger",
            )
        events.extend(resolved)

    def select_rng(stream="combat_card_selection"):
        rng = t.random_from_state(rng_state)
        # Older test RNGs keep their established behavior; the source RNG bank
        # routes selection separately when available.
        try:
            from .rng import random_stream
        except ImportError:
            return rng
        return random_stream(rng, stream)

    if name in {"brand", "burning_pact", "hidden_daggers"}:
        if name == "brand":
            run({"hp_loss": 1})
            following = ({"apply_status": {"target": "self", "strength": 2 if up else 1}},)
        elif name == "burning_pact":
            following = ({"draw": 3 if up else 2},)
        else:
            following = ({"add_card_to_hand": {"count": 2, "card": shiv_spec(upgraded=up)}},)
        if combat.hand:
            run(
                {
                    "choose_card": {
                        "action": "discard" if name == "hidden_daggers" else "exhaust",
                        "zone": "hand",
                        "count": 2 if name == "hidden_daggers" else 1,
                        "post_effects": following,
                    }
                }
            )
        else:
            for step in following:
                run(step)
    elif name == "gold_axe":
        run({"damage": max(0, int(combat.metadata.get("cards_played_this_combat", 0)) - 1)})
    elif name == "dodge_and_roll":
        resolved = run({"block": 6 if up else 4})
        amount = sum(e.amount or 0 for e in resolved if e.kind == "player_block")
        run({"next_turn": {"block": amount}})
    elif name == "hidden_gem":
        candidates = [
            c
            for c in combat.draw_pile
            if c.type.value not in {"curse", "status"}
            and not t._card_is_unplayable(c)
            and not t._card_has_keyword(c, "replay")
        ]
        candidates = [
            c for c in candidates if c.type.value in {"attack", "skill", "power"}
        ] or candidates
        if candidates:
            rng = select_rng()
            chosen = rng.choice(candidates)
            rng_state = t.capture_random_state(rng)
            chosen = chosen.model_copy(
                update={"custom": {**chosen.custom, "replay": 3 if up else 2}}
            )
            combat = t._replace_card_in_combat_zone(combat, "draw_pile", chosen)
            events.append(
                EffectEvent(
                    kind="card_keyword_added",
                    source_id=card.instance_id,
                    target_id=chosen.instance_id,
                    amount=3 if up else 2,
                    metadata={"keyword": "replay"},
                )
            )
    elif name == "anger":
        run({"damage": 8 if up else 6})
        copy = card.model_dump(mode="json")
        copy.pop("instance_id", None)
        run({"add_card_to_discard": {"card": copy}})
    elif name == "dominate":
        target = next((m for m in combat.monsters if m.monster_id == target_id), None)
        run(
            {
                "apply_status": {
                    "target": "self",
                    "strength": t._status_amount(target.statuses if target else {}, "vulnerable"),
                }
            }
        )
    elif name in {"finisher", "flechettes"}:
        count = (
            t._metadata_int(combat.metadata, "attacks_played_this_turn", 0)
            if name == "finisher"
            else sum(c.type == CardType.SKILL for c in combat.hand)
        )
        damage = (8 if up else 6) if name == "finisher" else (7 if up else 5)
        for _ in range(count):
            run({"damage": damage})
    elif name in {"memento_mori", "murder", "perfected_strike"}:
        if name == "memento_mori":
            count = (
                int(combat.metadata.get("source_core_discards", 0))
                if combat.metadata.get("source_core_discard_turn") == combat.turn
                else 0
            )
            damage = (11 if up else 9) + count * (5 if up else 4)
        elif name == "murder":
            damage = 1 + int(combat.metadata.get("source_core_draws", 0))
        else:
            all_cards = (
                combat.hand + combat.draw_pile + combat.discard_pile + combat.exhaust_pile + (card,)
            )
            count = sum(
                "strike" in {str(tag).lower() for tag in c.tags} or "strike" in c.card_id
                for c in all_cards
            )
            damage = 6 + count * (3 if up else 2)
        run({"damage": damage})
    elif name == "expertise":
        run({"draw": max(0, (7 if up else 6) - len(combat.hand))})
    elif name in {"fiend_fire", "stoke", "second_wind"}:
        selected = tuple(
            c for c in combat.hand if name != "second_wind" or c.type != CardType.ATTACK
        )
        if name == "second_wind":
            for chosen in selected:
                exhaust((chosen,))
                run({"block": 7 if up else 5})
        else:
            exhaust(selected)
            if name == "fiend_fire":
                for _ in selected:
                    run({"damage": 10 if up else 7})
            elif selected:
                run(
                    {
                        "add_random_card_to_hand": {
                            "count": len(selected),
                            "pool": "character",
                            "upgraded": up,
                        }
                    }
                )
    elif name in {"calculated_gamble", "storm_of_steel"}:
        count = len(combat.hand)
        run({"discard_hand": {"mode": "all"}})
        if name == "calculated_gamble":
            run({"draw": count})
        elif count:
            run({"add_card_to_hand": {"count": count, "card": shiv_spec(upgraded=up)}})
    elif name in {"bouncing_flask", "ricochet", "sword_boomerang"}:
        count = (5 if up else 4) if name == "ricochet" else (4 if up else 3)
        for _ in range(count):
            rng = select_rng("combat_targets")
            selected = t._random_alive_monster_id(combat, rng)
            rng_state = t.capture_random_state(rng)
            if selected is None:
                break
            if name == "bouncing_flask":
                run({"apply_status": {"target": "enemy", "poison": 3}}, target=selected)
            else:
                run({"damage": 3}, target=selected)
    elif name == "echoing_slash":
        remaining = 1
        while remaining and t._alive_monsters(combat):
            remaining -= 1
            resolved = run({"all_damage": 13 if up else 10})
            remaining += sum(e.kind == "monster_defeated" for e in resolved)
    elif name in {"fisticuffs", "omnislice"}:
        resolved = run({"damage": (9 if up else 7) if name == "fisticuffs" else (11 if up else 8)})
        # DamageResult.TotalDamage + OverkillDamage includes Block and all
        # post-modifier damage, even when the selected enemy has little HP.
        amount = sum(
            int(
                e.metadata.get(
                    "total_damage_including_overkill",
                    int(e.metadata.get("blocked", 0)) + int(e.amount or 0)
                    if e.metadata.get("intangible") or e.metadata.get("slippery")
                    else e.metadata.get("incoming", 0),
                )
            )
            for e in resolved
            if e.kind == "monster_damaged"
        )
        if name == "fisticuffs":
            run({"block": amount})
        else:
            others = tuple(
                m.monster_id for m in combat.monsters if m.hp > 0 and m.monster_id != target_id
            )
            monsters, resolved = t._damage_monsters(
                combat.monsters,
                others,
                amount,
                card.instance_id,
                relics=relics,
                source_card_type=None,
            )
            combat = combat.model_copy(update={"monsters": tuple(monsters)})
            events.extend(resolved)
    elif name == "molten_fist":
        run({"damage": 14 if up else 10})
        target = next((m for m in combat.monsters if m.monster_id == target_id), None)
        if target and target.hp > 0:
            run(
                {
                    "apply_status": {
                        "target": "enemy",
                        "vulnerable": t._status_amount(target.statuses, "vulnerable"),
                    }
                }
            )
    elif name == "rampage":
        run({"damage": 9 + bonus()})
        add_damage_bonus(9 if up else 5)
    elif name == "thrash":
        run({"damage": (6 if up else 4) + bonus()})
        run({"damage": (6 if up else 4) + bonus()})
        candidates = tuple(c for c in combat.hand if c.type == CardType.ATTACK)
        if candidates:
            rng = select_rng()
            chosen = rng.choice(candidates)
            rng_state = t.capture_random_state(rng)
            value = next(
                (
                    int(s.get("damage", s.get("all_damage", 0)))
                    for s in t._effect_steps(chosen.effects)
                    if "damage" in s or "all_damage" in s
                ),
                0,
            )
            value = t._modified_card_damage(combat, chosen, value, relics=relics)
            add_damage_bonus(int(value))
            exhaust((chosen,))
    elif name == "pillage":
        run({"damage": 9 if up else 6})
        while len(combat.hand) < t.MAX_HAND_SIZE:
            resolved = run({"draw": 1})
            ids = [e.target_id for e in resolved if e.kind == "card_drawn"]
            if not ids:
                break
            drawn = next((c for c in combat.hand if c.instance_id == ids[0]), None)
            if drawn is None or drawn.type != CardType.ATTACK:
                break
    elif name in {"havoc", "beat_down", "catastrophe", "knife_trap"}:
        if name == "havoc":
            if combat.draw_pile:
                play(combat.draw_pile[0], "draw_pile", force_exhaust=True)
        elif name == "knife_trap":
            for chosen in tuple(c for c in combat.exhaust_pile if t._is_shiv_card(c)):
                if up:
                    chosen = t._upgrade_card_instance(chosen)
                    combat = t._replace_card_in_combat_zone(combat, "exhaust_pile", chosen)
                play(chosen, "exhaust_pile", fixed_target=target_id)
        else:
            zone = "discard_pile" if name == "beat_down" else "draw_pile"
            count = (4 if up else 3) if name == "beat_down" else (3 if up else 2)
            candidates = [
                c
                for c in t._cards_in_combat_zone(combat, zone)
                if not t._card_is_unplayable(c)
                and (name != "beat_down" or c.type == CardType.ATTACK)
            ]
            if name == "beat_down":
                rng = select_rng("shuffle")
                rng.shuffle(candidates)
                rng_state = t.capture_random_state(rng)
                for chosen in candidates[:count]:
                    play(chosen, zone)
            else:
                for _ in range(count):
                    candidates = [
                        c for c in combat.draw_pile if not t._card_is_unplayable(c)
                    ] or list(combat.draw_pile)
                    if not candidates:
                        break
                    rng = select_rng("shuffle")
                    rng.shuffle(candidates)
                    rng_state = t.capture_random_state(rng)
                    play(candidates[0], zone)
    elif name == "armaments":
        run({"block": 5})
        if up:
            combat = combat.model_copy(
                update={"hand": tuple(t._upgrade_card_instance(c) for c in combat.hand)}
            )
        else:
            run({"choose_card": {"action": "upgrade", "zone": "hand", "count": 1}})
    elif name == "anointed":
        candidates = [
            c
            for c in combat.draw_pile
            if str(t._card_source_spec(c).get("rarity", "")).lower() == "rare"
        ]
        space = t.MAX_HAND_SIZE - len(combat.hand)
        if len(candidates) > space:
            rng = select_rng()
            rng.shuffle(candidates)
            rng_state = t.capture_random_state(rng)
        chosen = tuple(candidates[:space])
        selected_ids = {c.instance_id for c in chosen}
        combat = combat.model_copy(
            update={
                "hand": combat.hand + chosen,
                "draw_pile": tuple(
                    c for c in combat.draw_pile if c.instance_id not in selected_ids
                ),
            }
        )
        events.extend(
            EffectEvent(
                kind="card_moved_to_hand",
                source_id=card.instance_id,
                target_id=c.instance_id,
                metadata={"card_id": c.card_id},
            )
            for c in chosen
        )
    elif name == "primal_force":
        rock = {
            "card_id": "giant_rock",
            "name": "Giant Rock",
            "type": "attack",
            "cost": 1,
            "target": "enemy",
            "damage": 16,
            "upgrade": {"damage": 4},
            "upgraded": up,
        }
        hand = []
        for chosen in combat.hand:
            if chosen.type == CardType.ATTACK and not t._card_has_keyword(chosen, "eternal"):
                replacement = t._card_from_spec({**rock, "instance_id": chosen.instance_id}, 0)
                hand.append(replacement)
                events.append(
                    EffectEvent(
                        kind="card_transformed",
                        source_id=card.instance_id,
                        target_id=chosen.instance_id,
                        metadata={"card_id": "giant_rock"},
                    )
                )
            else:
                hand.append(chosen)
        combat = combat.model_copy(update={"hand": tuple(hand)})
    elif name in {"jack_of_all_trades", "jackpot"}:
        if name == "jackpot":
            run({"damage": 30 if up else 25})
        if state is not None:
            raw = t._raw_card_pool_items(state)
            candidates = [
                s
                for s in raw
                if isinstance(s, dict)
                and (
                    (
                        str(s.get("color", "")).lower() == "colorless"
                        and str(s.get("id", s.get("card_id", ""))).lower() != "jack_of_all_trades"
                    )
                    if name == "jack_of_all_trades"
                    else (
                        str(s.get("color", "")).lower() == state.character_id.lower()
                        and s.get("cost") == 0
                        and not s.get("is_x_cost")
                    )
                )
            ]
            rng = select_rng("combat_card_generation")
            count = (2 if up else 1) if name == "jack_of_all_trades" else 3
            if name == "jack_of_all_trades":
                rng.shuffle(candidates)
                selected = candidates[:count]
            else:
                selected = [rng.choice(candidates) for _ in range(count)] if candidates else []
            rng_state = t.capture_random_state(rng)
            for spec in selected:
                run(
                    {
                        "add_card_to_hand": {
                            "card": {**spec, "upgraded": up if name == "jackpot" else False}
                        }
                    }
                )
    return combat, rng_state, tuple(events)


def energy_cost(card, combat, default):
    """Late costs from FreeAttack/FreeSkill and the two count-cost cards."""
    kind = card.type.value
    if combat.player.statuses.get(f"source_core_free_{kind}", 0) > 0:
        return 0
    key = {"stomp": "attacks_played_this_turn", "pinpoint": "skills_played_this_turn"}.get(
        card.card_id
    )
    return max(0, default - int(combat.metadata.get(key, 0))) if key else default


def consume_free(combat, card):
    key = f"source_core_free_{card.type.value}"
    amount = combat.player.statuses.get(key, 0)
    if not amount:
        return combat
    statuses = dict(combat.player.statuses)
    if amount > 1:
        statuses[key] = amount - 1
    else:
        statuses.pop(key, None)
    return combat.model_copy(
        update={"player": combat.player.model_copy(update={"statuses": statuses})}
    )


def record_discards(combat, events):
    count = sum(
        e.kind == "card_discarded"
        and e.metadata.get("reason") not in {"end_turn_discard", "played_card_destination"}
        for e in events
    )
    if not count:
        return combat
    previous = (
        int(combat.metadata.get("source_core_discards", 0))
        if combat.metadata.get("source_core_discard_turn") == combat.turn
        else 0
    )
    return combat.model_copy(
        update={
            "metadata": {
                **combat.metadata,
                "source_core_discard_turn": combat.turn,
                "source_core_discards": previous + count,
            }
        }
    )


def after_play(state, combat, rng_state, card):
    from . import transitions as t

    amount = combat.player.statuses.get("source_core_juggling", 0)
    if (
        not amount
        or card.type != CardType.ATTACK
        or int(combat.metadata.get("attacks_played_this_turn", 0)) != 3
    ):
        return combat, rng_state, ()
    spec = card.model_dump(mode="json")
    spec.pop("instance_id", None)
    spec["custom"] = {
        **card.custom,
        "copied_from_card_id": card.card_id,
        "copied_from_instance_id": card.instance_id,
    }
    combat, rng_state, events = t._apply_effect_step(
        state=state,
        combat=combat,
        rng_state=rng_state,
        card=card,
        target_id=None,
        energy_spent=0,
        effect={"add_card_to_hand": {"count": amount, "card": spec}},
        relics=state.relics,
    )
    return (
        combat,
        rng_state,
        events
        + (
            EffectEvent(
                kind="card_copied_to_hand",
                source_id="juggling",
                target_id=card.instance_id,
                amount=amount,
                metadata={"card_id": card.card_id},
            ),
        ),
    )


def dynamic_card(combat, card):
    from . import transitions as t

    if not t._is_shiv_card(card) or not combat.player.statuses.get("source_core_fan_of_knives"):
        return card
    return card.model_copy(update={"target": t.TargetType.ALL_ENEMIES})
