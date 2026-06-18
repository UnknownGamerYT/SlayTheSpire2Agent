"""Unified trigger dispatch primitives.

This module gives cards, relics, potions, powers, and room systems a shared
vocabulary for "something happened" moments.  Existing bounded relic helpers
are adapted here so future systems can call one trigger API without knowing
which older helper module owns the effect.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sts2sim.mechanics.relic_combat import (
    CombatRelicBlocker,
    CombatRelicContext,
    CombatRelicHook,
    CombatRelicMarker,
    CombatRelicResolution,
    resolve_combat_relic_hook,
)
from sts2sim.mechanics.relics import (
    RelicEffectMarker,
    RelicHook,
    RelicHookResolution,
    RelicInput,
    resolve_relic_hook,
)


class GameTrigger(str, Enum):
    """Shared event names that gameplay content can react to."""

    RELIC_PICKUP = "relic_pickup"
    COMBAT_START = "combat_start"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    CARD_DRAWN = "card_drawn"
    CARD_PLAYED = "card_played"
    CARD_DISCARDED = "card_discarded"
    CARD_EXHAUSTED = "card_exhausted"
    DRAW_PILE_SHUFFLED = "draw_pile_shuffled"
    DAMAGE_DEALT = "damage_dealt"
    DAMAGE_TAKEN = "damage_taken"
    MONSTER_KILLED = "monster_killed"
    COMBAT_END = "combat_end"
    COMBAT_REWARD_GENERATED = "combat_reward_generated"
    CARD_REWARD_TAKEN = "card_reward_taken"
    POTION_USED = "potion_used"
    SHOP_ENTERED = "shop_entered"
    SHOP_PURCHASED = "shop_purchased"
    CAMPFIRE_ENTERED = "campfire_entered"
    CAMPFIRE_RESTED = "campfire_rested"
    CAMPFIRE_SMITHED = "campfire_smithed"


@dataclass(frozen=True, slots=True)
class TriggerContext:
    """Minimal data available to handlers for one trigger dispatch."""

    trigger: GameTrigger | str
    turn_number: int | None = None
    player_hp: int | None = None
    player_max_hp: int | None = None
    player_block: int | None = None
    encounter_type: str | None = None
    card_type: str | None = None
    card_id: str | None = None
    target_id: str | None = None
    player_statuses: Mapping[str, object] = field(default_factory=dict)
    target_statuses: Mapping[str, object] = field(default_factory=dict)
    relic_counters: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trigger", game_trigger(self.trigger))
        object.__setattr__(self, "player_statuses", dict(self.player_statuses))
        object.__setattr__(self, "target_statuses", dict(self.target_statuses))
        object.__setattr__(
            self,
            "relic_counters",
            {str(key): int(value) for key, value in self.relic_counters.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def combat_relic_context(self) -> CombatRelicContext:
        return CombatRelicContext(
            turn_number=self.turn_number,
            player_hp=self.player_hp,
            player_max_hp=self.player_max_hp,
            player_block=self.player_block,
            encounter_type=self.encounter_type,
            card_type=self.card_type,
            card_id=self.card_id,
            target_id=self.target_id,
            player_statuses=self.player_statuses,
            target_statuses=self.target_statuses,
            relic_counters=self.relic_counters,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class TriggerEffect:
    """Normalized effect marker emitted by one trigger handler."""

    kind: str
    trigger: GameTrigger
    source_kind: str
    content_id: str
    amount: int | None = None
    target_id: str | None = None
    source_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "trigger", game_trigger(self.trigger))
        object.__setattr__(self, "source_kind", _normalized_id(self.source_kind))
        object.__setattr__(self, "content_id", _normalized_id(self.content_id))
        if self.source_id is not None:
            object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class TriggerBlocker:
    """A content item looked relevant to a trigger but has no executable handler."""

    trigger: GameTrigger
    source_kind: str
    content_id: str
    reason: str
    source_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trigger", game_trigger(self.trigger))
        object.__setattr__(self, "source_kind", _normalized_id(self.source_kind))
        object.__setattr__(self, "content_id", _normalized_id(self.content_id))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class TriggerHandlerResult:
    """Result returned by custom trigger handlers."""

    effects: tuple[TriggerEffect, ...] = ()
    blockers: tuple[TriggerBlocker, ...] = ()
    gold_delta: int = 0
    hp_delta: int = 0
    max_hp_delta: int = 0
    block_delta: int = 0
    energy_delta: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "effects", tuple(self.effects))
        object.__setattr__(self, "blockers", tuple(self.blockers))


TriggerHandlerCallback = Callable[[TriggerContext], TriggerHandlerResult | None]


@dataclass(frozen=True, slots=True)
class TriggerHandler:
    """A custom handler registration for cards, powers, potions, or relics."""

    handler_id: str
    trigger: GameTrigger | str
    callback: TriggerHandlerCallback
    source_kind: str = "custom"
    content_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "handler_id", str(self.handler_id))
        object.__setattr__(self, "trigger", game_trigger(self.trigger))
        object.__setattr__(self, "source_kind", _normalized_id(self.source_kind))
        if self.content_id is not None:
            object.__setattr__(self, "content_id", _normalized_id(self.content_id))


@dataclass(frozen=True, slots=True)
class TriggerResolution:
    """Combined result for a trigger dispatch."""

    trigger: GameTrigger
    effects: tuple[TriggerEffect, ...] = ()
    blockers: tuple[TriggerBlocker, ...] = ()
    gold_delta: int = 0
    hp_delta: int = 0
    max_hp_delta: int = 0
    block_delta: int = 0
    energy_delta: int = 0
    combat_relic_resolution: CombatRelicResolution | None = None
    relic_hook_resolution: RelicHookResolution | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "trigger", game_trigger(self.trigger))
        object.__setattr__(self, "effects", tuple(self.effects))
        object.__setattr__(self, "blockers", tuple(self.blockers))


class TriggerDispatcher:
    """Small immutable-ish registry for future content trigger handlers."""

    def __init__(self, handlers: Iterable[TriggerHandler] = ()) -> None:
        self._handlers_by_trigger: dict[GameTrigger, list[TriggerHandler]] = {}
        for handler in handlers:
            self.register(handler)

    def register(self, handler: TriggerHandler) -> None:
        self._handlers_by_trigger.setdefault(game_trigger(handler.trigger), []).append(handler)

    def handlers_for(self, trigger: GameTrigger | str) -> tuple[TriggerHandler, ...]:
        return tuple(self._handlers_by_trigger.get(game_trigger(trigger), ()))

    def resolve(self, context: TriggerContext) -> TriggerHandlerResult:
        effects: list[TriggerEffect] = []
        blockers: list[TriggerBlocker] = []
        gold_delta = 0
        hp_delta = 0
        max_hp_delta = 0
        block_delta = 0
        energy_delta = 0

        for handler in self.handlers_for(context.trigger):
            result = handler.callback(context)
            if result is None:
                continue
            effects.extend(result.effects)
            blockers.extend(result.blockers)
            gold_delta += result.gold_delta
            hp_delta += result.hp_delta
            max_hp_delta += result.max_hp_delta
            block_delta += result.block_delta
            energy_delta += result.energy_delta

        return TriggerHandlerResult(
            effects=tuple(effects),
            blockers=tuple(blockers),
            gold_delta=gold_delta,
            hp_delta=hp_delta,
            max_hp_delta=max_hp_delta,
            block_delta=block_delta,
            energy_delta=energy_delta,
        )


def resolve_game_trigger(
    trigger: GameTrigger | str,
    *,
    relics: Sequence[RelicInput] = (),
    context: TriggerContext | None = None,
    dispatcher: TriggerDispatcher | None = None,
    include_blockers: bool = True,
    **context_values: Any,
) -> TriggerResolution:
    """Resolve built-in adapters and custom handlers for one game trigger."""

    normalized_trigger = game_trigger(trigger)
    trigger_context = context or TriggerContext(normalized_trigger, **context_values)
    if trigger_context.trigger is not normalized_trigger:
        trigger_context = TriggerContext(
            normalized_trigger,
            turn_number=trigger_context.turn_number,
            player_hp=trigger_context.player_hp,
            player_max_hp=trigger_context.player_max_hp,
            player_block=trigger_context.player_block,
            encounter_type=trigger_context.encounter_type,
            card_type=trigger_context.card_type,
            card_id=trigger_context.card_id,
            target_id=trigger_context.target_id,
            player_statuses=trigger_context.player_statuses,
            target_statuses=trigger_context.target_statuses,
            relic_counters=trigger_context.relic_counters,
            metadata=trigger_context.metadata,
        )

    combat_resolution = _resolve_combat_relic_adapter(
        normalized_trigger,
        relics,
        trigger_context,
        include_blockers=include_blockers,
    )
    relic_resolution = _resolve_relic_adapter(normalized_trigger, relics, trigger_context)
    custom = (dispatcher or TriggerDispatcher()).resolve(trigger_context)

    effects = list(custom.effects)
    blockers = list(custom.blockers)
    gold_delta = custom.gold_delta
    hp_delta = custom.hp_delta
    max_hp_delta = custom.max_hp_delta
    block_delta = custom.block_delta
    energy_delta = custom.energy_delta

    if combat_resolution is not None:
        effects.extend(_effects_from_combat_relics(normalized_trigger, combat_resolution))
        blockers.extend(_blockers_from_combat_relics(normalized_trigger, combat_resolution))
        hp_delta += combat_resolution.hp_delta
        block_delta += combat_resolution.block_delta
        energy_delta += combat_resolution.energy_delta

    if relic_resolution is not None:
        effects.extend(_effects_from_relic_hook(normalized_trigger, relic_resolution))
        gold_delta += relic_resolution.gold_delta
        hp_delta += relic_resolution.hp_delta
        max_hp_delta += relic_resolution.max_hp_delta

    return TriggerResolution(
        trigger=normalized_trigger,
        effects=tuple(effects),
        blockers=tuple(blockers),
        gold_delta=gold_delta,
        hp_delta=hp_delta,
        max_hp_delta=max_hp_delta,
        block_delta=block_delta,
        energy_delta=energy_delta,
        combat_relic_resolution=combat_resolution,
        relic_hook_resolution=relic_resolution,
    )


def game_trigger(value: GameTrigger | str) -> GameTrigger:
    if isinstance(value, GameTrigger):
        return value
    normalized = _normalized_id(value)
    aliases = {
        "start_combat": GameTrigger.COMBAT_START,
        "start_of_combat": GameTrigger.COMBAT_START,
        "end_combat": GameTrigger.COMBAT_END,
        "card_play": GameTrigger.CARD_PLAYED,
        "shop_enter": GameTrigger.SHOP_ENTERED,
        "shop_purchase": GameTrigger.SHOP_PURCHASED,
        "campfire_enter": GameTrigger.CAMPFIRE_ENTERED,
    }
    if normalized in aliases:
        return aliases[normalized]
    return GameTrigger(normalized)


_COMBAT_RELIC_TRIGGERS: dict[GameTrigger, CombatRelicHook] = {
    GameTrigger.COMBAT_START: CombatRelicHook.START_OF_COMBAT,
    GameTrigger.TURN_START: CombatRelicHook.TURN_START,
    GameTrigger.TURN_END: CombatRelicHook.TURN_END,
    GameTrigger.CARD_PLAYED: CombatRelicHook.CARD_PLAYED,
    GameTrigger.DAMAGE_DEALT: CombatRelicHook.DAMAGE_DEALT,
    GameTrigger.DAMAGE_TAKEN: CombatRelicHook.DAMAGE_TAKEN,
    GameTrigger.MONSTER_KILLED: CombatRelicHook.MONSTER_KILLED,
    GameTrigger.COMBAT_END: CombatRelicHook.COMBAT_END,
}

_RELIC_TRIGGERS: dict[GameTrigger, RelicHook] = {
    GameTrigger.SHOP_ENTERED: RelicHook.SHOP_ENTER,
    GameTrigger.SHOP_PURCHASED: RelicHook.SHOP_PURCHASE,
    GameTrigger.CAMPFIRE_ENTERED: RelicHook.CAMPFIRE_ENTER,
}


def _resolve_combat_relic_adapter(
    trigger: GameTrigger,
    relics: Sequence[RelicInput],
    context: TriggerContext,
    *,
    include_blockers: bool,
) -> CombatRelicResolution | None:
    hook = _COMBAT_RELIC_TRIGGERS.get(trigger)
    if hook is None:
        return None
    return resolve_combat_relic_hook(
        relics,
        hook,
        context=context.combat_relic_context(),
        include_blockers=include_blockers,
    )


def _resolve_relic_adapter(
    trigger: GameTrigger,
    relics: Sequence[RelicInput],
    context: TriggerContext,
) -> RelicHookResolution | None:
    hook = _RELIC_TRIGGERS.get(trigger)
    if hook is None:
        return None
    return resolve_relic_hook(
        relics,
        hook,
        hp=context.player_hp,
        max_hp=context.player_max_hp,
    )


def _effects_from_combat_relics(
    trigger: GameTrigger,
    resolution: CombatRelicResolution,
) -> tuple[TriggerEffect, ...]:
    return tuple(_effect_from_combat_relic_marker(trigger, marker) for marker in resolution.markers)


def _effect_from_combat_relic_marker(
    trigger: GameTrigger,
    marker: CombatRelicMarker,
) -> TriggerEffect:
    metadata = {"hook": marker.hook.value, **dict(marker.metadata)}
    return TriggerEffect(
        kind=marker.kind,
        trigger=trigger,
        source_kind="relic",
        content_id=marker.relic_id,
        amount=marker.amount,
        target_id=marker.target_id,
        source_id=marker.source_id,
        metadata=metadata,
    )


def _blockers_from_combat_relics(
    trigger: GameTrigger,
    resolution: CombatRelicResolution,
) -> tuple[TriggerBlocker, ...]:
    return tuple(_blocker_from_combat_relic(trigger, blocker) for blocker in resolution.blockers)


def _blocker_from_combat_relic(
    trigger: GameTrigger,
    blocker: CombatRelicBlocker,
) -> TriggerBlocker:
    return TriggerBlocker(
        trigger=trigger,
        source_kind="relic",
        content_id=blocker.relic_id,
        reason=blocker.reason,
        source_id=blocker.source_id,
        metadata={
            "hook": blocker.hook.value,
            "name": blocker.name,
            "description": blocker.description,
        },
    )


def _effects_from_relic_hook(
    trigger: GameTrigger,
    resolution: RelicHookResolution,
) -> tuple[TriggerEffect, ...]:
    return tuple(_effect_from_relic_marker(trigger, marker) for marker in resolution.markers)


def _effect_from_relic_marker(
    trigger: GameTrigger,
    marker: RelicEffectMarker,
) -> TriggerEffect:
    return TriggerEffect(
        kind=marker.kind,
        trigger=trigger,
        source_kind="relic",
        content_id=marker.relic_id,
        amount=marker.amount,
        target_id=marker.target_id,
        source_id=marker.relic_id,
        metadata=marker.metadata,
    )


def _normalized_id(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


__all__ = [
    "GameTrigger",
    "TriggerBlocker",
    "TriggerContext",
    "TriggerDispatcher",
    "TriggerEffect",
    "TriggerHandler",
    "TriggerHandlerResult",
    "TriggerResolution",
    "game_trigger",
    "resolve_game_trigger",
]
