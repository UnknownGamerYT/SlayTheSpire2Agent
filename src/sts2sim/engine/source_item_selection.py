"""Stable0.107.1 item selection order and RNG consumers.

Independent rules from Razhuo/sts2-rl-agent commit
5d874e855ae1d0550d5c30416fadd7dbb0d133b7, CardModel.CompareTo,
ListExtensions.StableShuffle, and the cited relic model source files.
"""
from __future__ import annotations

from collections.abc import Sequence
from random import Random

from sts2sim.content.source_card_pools import NON_UPGRADABLE_CARD_IDS, source_pool_cards
from sts2sim.mechanics.relic_combat import CombatRelicMarker
from sts2sim.mechanics.source_rng import draw_source_potions, is_source_rng

from .models import CardInstance, CombatState, EffectEvent, RunState
from .rng import random_stream, stable_shuffle


def upgradable(card: CardInstance) -> bool:
    return not card.upgraded and card.card_id.lower() not in NON_UPGRADABLE_CARD_IDS


def select_stable_cards(
    cards: Sequence[CardInstance], rng: Random, count: int, *, stream: str,
) -> tuple[CardInstance, ...]:
    # C# CompareTo compares ModelId, then CurrentUpgradeLevel; the full list
    # shuffles before Take(count), including when count covers the whole list.
    shuffled = stable_shuffle(
        cards, random_stream(rng, stream), key=lambda card: (card.card_id.lower(), card.upgraded),
    )
    return tuple(shuffled[:max(0, count)])


def add_random_curses(
    state: RunState, rng: Random, count: int, *, source_id: str,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    """SereTalon/NeowsBones: sorted unlocked modifier curses; NextItem+Remove."""
    from sts2sim.mechanics.relics import RelicEffectMarker

    from . import transitions as t

    epochs = state.flags.get("revealed_epochs", state.flags.get("unlocked_epochs"))
    candidates = sorted(source_pool_cards(
        "curse", unlocked_epochs=None if epochs is None else t._string_items(epochs),
        multiplayer=t._flag_int(state, "player_count", 1) > 1,
    ))
    # Eight curse overrides disable modifier generation; combat-generation
    # eligibility is a separate property and does not substitute for this.
    forbidden = {"ascenders_bane", "bad_luck", "curse_of_the_bell", "enthralled",
                 "folly", "greed", "poor_sleep", "spore_mind"}
    candidates = [identifier for identifier in candidates if identifier not in forbidden]
    rng = random_stream(rng, "niche")
    events: list[EffectEvent] = []
    for _ in range(min(max(0, count), len(candidates))):
        identifier = rng.choice(candidates)
        candidates.remove(identifier)
        state, emitted = t._apply_relic_add_deck_cards_marker(
            state, source_id, RelicEffectMarker(
                kind="add_deck_cards", relic_id=source_id, amount=1,
                metadata={"card_id": identifier},
            ),
        )
        events.extend(emitted)
    return state, tuple(events)


def phial_potions(
    state: RunState, rng: Random,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    """Generate both out-of-combat potions before procurement/Sozu checks."""
    from . import transitions as t

    options, rarities = t._item_potion_generation_pool(state, in_combat=False)
    generation = random_stream(rng, "combat_potion_generation")
    identifiers = draw_source_potions(generation, options, 2, rarities)
    state, events = t._add_event_potions(state, identifiers)
    return state, tuple(event.model_copy(update={
        "source_id": "phial_holster",
        "kind": "relic_potion_obtained" if event.kind == "event_potion_obtained" else event.kind,
    }) for event in events)


def coffer_potions(
    state: RunState, rng: Random,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    """Generate four models, then procure only into the four new physical slots."""
    from . import transitions as t
    from .potion_inventory import procure_at_slot

    original_slots = t._potion_capacity(state) - 4
    options, rarities = t._item_potion_generation_pool(state, in_combat=False)
    identifiers = draw_source_potions(
        random_stream(rng, "combat_potion_generation"), options, 4, rarities,
    )
    events: list[EffectEvent] = []
    for offset, identifier in enumerate(identifiers):
        state, emitted = procure_at_slot(
            state, identifier, preferred_slot=original_slots + offset,
            source_id="alchemical_coffer",
        )
        events.extend(event.model_copy(update={"kind": "relic_potion_obtained"})
                      if event.kind == "event_potion_obtained" else event
                      for event in emitted)
    return state, tuple(events)


def combat_marker(
    combat: CombatState, marker: CombatRelicMarker, rng: Random | None,
) -> tuple[CombatState, tuple[EffectEvent, ...]] | None:
    from . import source_cost_hooks
    from . import transitions as t

    if rng is None or not is_source_rng(rng):
        return None
    if marker.relic_id == "stone_cracker" and marker.kind == "upgrade_draw_pile_cards":
        chosen = select_stable_cards(
            tuple(card for card in combat.draw_pile if upgradable(card)), rng, marker.amount or 0,
            stream="combat_card_selection",
        )
        identifiers = {card.instance_id for card in chosen}
        combat = combat.model_copy(update={"draw_pile": tuple(
            t._upgrade_card_instance(card) if card.instance_id in identifiers else card
            for card in combat.draw_pile
        )})
        if not chosen:
            return combat, ()
        event = t._combat_relic_event(marker, "draw_pile_cards_upgraded", len(chosen))
        return combat, (event.model_copy(update={"metadata": {
            **event.metadata, "card_instance_ids": [c.instance_id for c in chosen],
        }}),)
    if marker.relic_id == "power_cell" and marker.kind == "move_zero_cost_cards_to_hand":
        candidates = tuple(
            card for card in combat.draw_pile
            if not t._card_is_x_cost(card) and (source_cost_hooks.base_cost(card) or 0) >= 0
            and source_cost_hooks.local_cost(card) <= 0
        )
        chosen = select_stable_cards(candidates, rng, marker.amount or 0,
                                     stream="combat_card_selection")
    elif marker.relic_id == "jeweled_mask" and marker.kind == "move_card_type_from_draw_to_hand":
        candidates = tuple(card for card in combat.draw_pile if card.type.value == "power")
        chosen = (
            (random_stream(rng, "combat_card_selection").choice(candidates),) if candidates else ()
        )
        chosen = tuple(source_cost_hooks.free_to_play(card) for card in chosen)
    else:
        return None
    if not chosen:
        return combat, ()
    identifiers = {card.instance_id for card in chosen}
    free_slots = max(0, t.MAX_HAND_SIZE - len(combat.hand))
    hand, overflow = chosen[:free_slots], chosen[free_slots:]
    combat = combat.model_copy(update={
        "hand": combat.hand + hand, "discard_pile": combat.discard_pile + overflow,
        "draw_pile": tuple(c for c in combat.draw_pile if c.instance_id not in identifiers),
    })
    event = t._combat_relic_event(marker, "draw_pile_cards_moved_to_hand", len(hand))
    events = [event.model_copy(update={"metadata": {
        **event.metadata, "card_instance_ids": [c.instance_id for c in hand],
        "overflow_instance_ids": [c.instance_id for c in overflow],
    }})]
    for cards, destination in ((hand, "hand"), (overflow, "discard_pile")):
        events.extend(EffectEvent(
            kind="card_moved", source_id=marker.relic_id, target_id=card.instance_id,
            amount=1, metadata={"card_id": card.card_id, "from_pile": "draw_pile",
                                "to_pile": destination, "reason": f"relic_{marker.kind}"},
        ) for card in cards)
    return combat, tuple(events)
