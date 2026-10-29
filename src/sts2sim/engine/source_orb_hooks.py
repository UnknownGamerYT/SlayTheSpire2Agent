"""Stable orb-only owner callbacks, separate from text-derived turn triggers."""

from __future__ import annotations

from collections.abc import Sequence

from .models import CardInstance, CombatState, EffectEvent, RngState, RunPhase, RunState
from .rng import capture_random_state, random_from_state


def after_channel(
    combat: CombatState,
    relics: Sequence[str],
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    if not t._has_relic_id(relics, "metronome"):
        return combat, ()
    counters = t._combat_relic_counters(combat)
    counters["metronome"] = counters.get("metronome", 0) + 1
    combat = combat.model_copy(update={"metadata": {**combat.metadata, "relic_counters": counters}})
    if counters["metronome"] != 7:
        return combat, ()
    return t._damage_all_monsters(
        combat,
        30,
        "metronome",
        relics=relics,
        metadata={"relic_id": "metronome", "trigger": "orb_channeled"},
    )


def after_energy_reset(
    combat: CombatState,
    rng_state: RngState,
    relics: Sequence[str],
    *,
    state: RunState | None = None,
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    rng = random_from_state(rng_state)
    events: list[EffectEvent] = []
    for _ in range(t._status_amount(combat.player.statuses, "spinner")):
        combat, resolved = t._channel_orb(
            combat,
            rng,
            "spinner",
            "glass",
            target_id=None,
            relics=relics,
            state=state,
        )
        events.extend(resolved)
    return combat, capture_random_state(rng), tuple(events)


def after_player_turn_start(
    combat: CombatState,
    rng_state: RngState,
    relics: Sequence[str],
    *,
    state: RunState | None = None,
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    events: list[EffectEvent] = []
    # Powers precede relics in IterateHookListeners. These manual calls do not
    # receive GoldPlatedCables, which modifies only the automatic queue loops.
    loop = t._status_amount(combat.player.statuses, "loop")
    if loop and combat.orbs:
        combat, rng_state, resolved = t._trigger_orb_passive_from_effect(
            combat,
            rng_state,
            CardInstance(instance_id="loop", card_id="loop"),
            {"trigger_orb_passive": {"selector": "leftmost", "amount": loop}},
            relics=relics,
            state=state,
        )
        events.extend(resolved)
    if (
        t._has_relic_id(relics, "emotion_chip")
        and combat.metadata.get(
            "damage_not_fully_blocked_count_previous_turn",
            combat.metadata.get("lost_hp_previous_turn"),
        )
        and t._phase_after_combat(combat) == RunPhase.COMBAT
    ):
        combat, rng_state, resolved = t._trigger_orb_passive_from_effect(
            combat,
            rng_state,
            CardInstance(instance_id="emotion_chip", card_id="emotion_chip"),
            {"trigger_orb_passive": {"selector": "all"}},
            relics=relics,
            state=state,
        )
        events.extend(resolved)
    combat, rng_state, resolved = t._apply_orb_passives(
        combat,
        rng_state,
        relics=relics,
        at_turn_start=True,
        state=state,
    )
    return combat, rng_state, tuple(events) + resolved
