"""Pure potion-use trigger helpers.

The engine owns potion consumption and effect application.  This module only
normalizes the potion-use trigger surface and returns markers/deltas that can be
applied by a future integration point.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .potions import PotionInput, potion_content_id
from .relics import RelicInput, relic_content_id
from .triggers import (
    GameTrigger,
    TriggerBlocker,
    TriggerContext,
    TriggerDispatcher,
    TriggerEffect,
    TriggerResolution,
    resolve_game_trigger,
)


@dataclass(frozen=True, slots=True)
class PotionUseTriggerContext:
    """Data available to content reacting to a potion being used."""

    potion: PotionInput
    slot_index: int | None = None
    slot_id: str | None = None
    target_id: str | None = None
    use_mode: str = "combat"
    consumes_potion: bool = True
    player_hp: int | None = None
    player_max_hp: int | None = None
    player_block: int | None = None
    player_statuses: Mapping[str, object] = field(default_factory=dict)
    target_statuses: Mapping[str, object] = field(default_factory=dict)
    owned_relics: Sequence[RelicInput] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "potion", potion_content_id(self.potion))
        if self.slot_index is not None:
            object.__setattr__(self, "slot_index", int(self.slot_index))
        if self.slot_id is not None:
            object.__setattr__(self, "slot_id", str(self.slot_id))
        if self.target_id is not None:
            object.__setattr__(self, "target_id", str(self.target_id))
        object.__setattr__(self, "use_mode", _normalized_id(self.use_mode))
        object.__setattr__(self, "consumes_potion", bool(self.consumes_potion))
        object.__setattr__(self, "player_statuses", dict(self.player_statuses))
        object.__setattr__(self, "target_statuses", dict(self.target_statuses))
        object.__setattr__(self, "owned_relics", tuple(self.owned_relics))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def potion_id(self) -> str:
        return str(self.potion)

    @property
    def potion_slot(self) -> str | None:
        if self.slot_id is not None:
            return self.slot_id
        if self.slot_index is not None:
            return f"potion:{self.slot_index}"
        return None

    @property
    def owned_relic_ids(self) -> tuple[str, ...]:
        seen: set[str] = set()
        relic_ids: list[str] = []
        for relic in self.owned_relics:
            relic_id = relic_content_id(relic)
            if relic_id in seen:
                continue
            seen.add(relic_id)
            relic_ids.append(relic_id)
        return tuple(relic_ids)

    def trigger_context(self) -> TriggerContext:
        """Build the shared trigger context used by custom dispatchers."""

        metadata: dict[str, object] = {
            **dict(self.metadata),
            "potion_id": self.potion_id,
            "use_mode": self.use_mode,
            "consumes_potion": self.consumes_potion,
        }
        if self.slot_index is not None:
            metadata["slot_index"] = self.slot_index
        if self.potion_slot is not None:
            metadata["potion_slot"] = self.potion_slot
        return TriggerContext(
            GameTrigger.POTION_USED,
            player_hp=self.player_hp,
            player_max_hp=self.player_max_hp,
            player_block=self.player_block,
            target_id=self.target_id,
            player_statuses=self.player_statuses,
            target_statuses=self.target_statuses,
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class PotionUseModifierSpec:
    """A small, table-driven potion-use modifier marker and delta."""

    content_id: str
    kind: str
    source_kind: str = "relic"
    amount: int | None = None
    hp_delta: int = 0
    max_hp_delta: int = 0
    block_delta: int = 0
    energy_delta: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_id", _normalized_id(self.content_id))
        object.__setattr__(self, "kind", _normalized_id(self.kind))
        object.__setattr__(self, "source_kind", _normalized_id(self.source_kind))
        if self.amount is not None:
            object.__setattr__(self, "amount", int(self.amount))
        object.__setattr__(self, "hp_delta", int(self.hp_delta))
        object.__setattr__(self, "max_hp_delta", int(self.max_hp_delta))
        object.__setattr__(self, "block_delta", int(self.block_delta))
        object.__setattr__(self, "energy_delta", int(self.energy_delta))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class PotionUseTriggerResult:
    """Potion-use trigger markers plus pure deltas for later engine application."""

    context: PotionUseTriggerContext
    trigger_resolution: TriggerResolution
    effects: tuple[TriggerEffect, ...] = ()
    blockers: tuple[TriggerBlocker, ...] = ()
    hp_delta: int = 0
    max_hp_delta: int = 0
    block_delta: int = 0
    energy_delta: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "effects", tuple(self.effects))
        object.__setattr__(self, "blockers", tuple(self.blockers))


def resolve_potion_use_triggers(
    context: PotionUseTriggerContext,
    *,
    modifiers: Sequence[PotionUseModifierSpec] = (),
    dispatcher: TriggerDispatcher | None = None,
    include_blockers: bool = True,
) -> PotionUseTriggerResult:
    """Resolve pure potion-use modifiers and custom trigger handlers."""

    dispatch = resolve_game_trigger(
        GameTrigger.POTION_USED,
        relics=context.owned_relics,
        context=context.trigger_context(),
        dispatcher=dispatcher,
        include_blockers=include_blockers,
    )
    active_modifiers = _active_default_modifiers(context) + tuple(modifiers)

    hp_delta = dispatch.hp_delta
    max_hp_delta = dispatch.max_hp_delta
    block_delta = dispatch.block_delta
    energy_delta = dispatch.energy_delta
    effects: list[TriggerEffect] = []

    current_hp = context.player_hp
    current_max_hp = context.player_max_hp
    for modifier in active_modifiers:
        modifier_hp_delta = _capped_hp_delta(
            modifier.hp_delta,
            hp=current_hp,
            max_hp=current_max_hp,
        )
        if current_hp is not None:
            current_hp += modifier_hp_delta
        hp_delta += modifier_hp_delta
        max_hp_delta += modifier.max_hp_delta
        block_delta += modifier.block_delta
        energy_delta += modifier.energy_delta
        effects.append(_effect_from_modifier(modifier, context, hp_delta=modifier_hp_delta))

    effects.extend(dispatch.effects)
    combined_resolution = TriggerResolution(
        trigger=GameTrigger.POTION_USED,
        effects=tuple(effects),
        blockers=dispatch.blockers,
        gold_delta=dispatch.gold_delta,
        hp_delta=hp_delta,
        max_hp_delta=max_hp_delta,
        block_delta=block_delta,
        energy_delta=energy_delta,
        combat_relic_resolution=dispatch.combat_relic_resolution,
        relic_hook_resolution=dispatch.relic_hook_resolution,
    )
    return PotionUseTriggerResult(
        context=context,
        trigger_resolution=combined_resolution,
        effects=combined_resolution.effects,
        blockers=combined_resolution.blockers,
        hp_delta=hp_delta,
        max_hp_delta=max_hp_delta,
        block_delta=block_delta,
        energy_delta=energy_delta,
    )


def _active_default_modifiers(
    context: PotionUseTriggerContext,
) -> tuple[PotionUseModifierSpec, ...]:
    owned_relic_ids = set(context.owned_relic_ids)
    return tuple(
        modifier
        for modifier in DEFAULT_POTION_USE_MODIFIERS
        if modifier.content_id in owned_relic_ids
    )


def _effect_from_modifier(
    modifier: PotionUseModifierSpec,
    context: PotionUseTriggerContext,
    *,
    hp_delta: int,
) -> TriggerEffect:
    metadata: dict[str, object] = {
        **dict(modifier.metadata),
        "potion_id": context.potion_id,
        "use_mode": context.use_mode,
        "consumes_potion": context.consumes_potion,
    }
    if context.potion_slot is not None:
        metadata["potion_slot"] = context.potion_slot
    if context.slot_index is not None:
        metadata["slot_index"] = context.slot_index
    if modifier.hp_delta:
        metadata["hp_delta"] = hp_delta
    if modifier.max_hp_delta:
        metadata["max_hp_delta"] = modifier.max_hp_delta
    if modifier.block_delta:
        metadata["block_delta"] = modifier.block_delta
    if modifier.energy_delta:
        metadata["energy_delta"] = modifier.energy_delta

    return TriggerEffect(
        kind=modifier.kind,
        trigger=GameTrigger.POTION_USED,
        source_kind=modifier.source_kind,
        content_id=modifier.content_id,
        amount=modifier.amount
        if modifier.amount is not None
        else _modifier_amount(modifier, hp_delta),
        target_id="player",
        source_id=modifier.content_id,
        metadata=metadata,
    )


def _modifier_amount(modifier: PotionUseModifierSpec, hp_delta: int) -> int | None:
    for amount in (
        hp_delta if modifier.hp_delta else 0,
        modifier.max_hp_delta,
        modifier.block_delta,
        modifier.energy_delta,
    ):
        if amount:
            return amount
    return None


def _capped_hp_delta(delta: int, *, hp: int | None, max_hp: int | None) -> int:
    if delta <= 0 or hp is None or max_hp is None:
        return delta
    return max(0, min(delta, max_hp - hp))


def _normalized_id(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("'", "")
        .replace(" ", "_")
        .replace("-", "_")
    )


DEFAULT_POTION_USE_MODIFIERS: tuple[PotionUseModifierSpec, ...] = (
    PotionUseModifierSpec(
        content_id="toy_ornithopter",
        kind="potion_use_heal",
        hp_delta=5,
        metadata={"modifier": "toy_ornithopter"},
    ),
)


__all__ = [
    "DEFAULT_POTION_USE_MODIFIERS",
    "PotionUseModifierSpec",
    "PotionUseTriggerContext",
    "PotionUseTriggerResult",
    "resolve_potion_use_triggers",
]
