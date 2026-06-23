"""Pure Forge and Sovereign Blade helpers.

The combat engine owns mutation and zone objects. This module keeps the Regent
Forge resource and Sovereign Blade mechanics as immutable records plus plain
mapping helpers so integration code can apply them from any state model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum

from sts2sim.content.sources import PROVISIONAL_STS2_SOURCE, SourceRef

SOVEREIGN_BLADE_CARD_ID = "sovereign_blade"
SOVEREIGN_BLADE_NAME = "Sovereign Blade"
DEFAULT_SOVEREIGN_BLADE_DAMAGE = 10
DEFAULT_SOVEREIGN_BLADE_COST = 2

BEAT_INTO_SHAPE_FORGE = 5
BEAT_INTO_SHAPE_UPGRADED_FORGE = 7
CONQUEROR_FORGE = 3
CONQUEROR_UPGRADED_FORGE = 5
FURNACE_FORGE = 4
FURNACE_UPGRADED_FORGE = 6
SEEKING_EDGE_FORGE = 7
SEEKING_EDGE_UPGRADED_FORGE = 11
SUMMON_FORTH_FORGE = 8
SUMMON_FORTH_UPGRADED_FORGE = 11
PARRY_BLOCK = 10
PARRY_UPGRADED_BLOCK = 14


class ForgeTrigger(str, Enum):
    """Trigger vocabulary for Forge descriptors."""

    IMMEDIATE = "immediate"
    TURN_START = "turn_start"
    FORGE_GAINED = "forge_gained"
    CARD_PLAYED = "card_played"


class SovereignBladeTarget(str, Enum):
    """Targets supported by Sovereign Blade itself."""

    ENEMY = "enemy"
    ALL_ENEMIES = "all_enemies"


@dataclass(frozen=True, slots=True)
class ForgeState:
    """Counters for the player's Forge resource in combat."""

    amount: int = 0
    gained_this_turn: int = 0
    gained_this_combat: int = 0
    times_forged_this_turn: int = 0
    times_forged_this_combat: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount", max(0, int(self.amount)))
        object.__setattr__(self, "gained_this_turn", max(0, int(self.gained_this_turn)))
        object.__setattr__(self, "gained_this_combat", max(0, int(self.gained_this_combat)))
        object.__setattr__(
            self,
            "times_forged_this_turn",
            max(0, int(self.times_forged_this_turn)),
        )
        object.__setattr__(
            self,
            "times_forged_this_combat",
            max(0, int(self.times_forged_this_combat)),
        )

    def add(self, amount: int) -> ForgeState:
        gained = max(0, int(amount))
        if gained == 0:
            return self
        return ForgeState(
            amount=self.amount + gained,
            gained_this_turn=self.gained_this_turn + gained,
            gained_this_combat=self.gained_this_combat + gained,
            times_forged_this_turn=self.times_forged_this_turn + 1,
            times_forged_this_combat=self.times_forged_this_combat + 1,
        )

    def reset_turn_counters(self) -> ForgeState:
        return ForgeState(
            amount=self.amount,
            gained_this_turn=0,
            gained_this_combat=self.gained_this_combat,
            times_forged_this_turn=0,
            times_forged_this_combat=self.times_forged_this_combat,
        )


@dataclass(frozen=True, slots=True)
class ForgeContext:
    """Dynamic values available when resolving a Forge descriptor."""

    energy_spent: int = 0
    previous_hits_on_target_this_turn: int = 0
    source_forge_amount: int = 0
    base_amount: int | None = None
    bonus_per_previous_hit: int | None = None
    upgraded: bool = False
    ally_ids: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy_spent", max(0, int(self.energy_spent)))
        object.__setattr__(
            self,
            "previous_hits_on_target_this_turn",
            max(0, int(self.previous_hits_on_target_this_turn)),
        )
        object.__setattr__(self, "source_forge_amount", max(0, int(self.source_forge_amount)))
        if self.base_amount is not None:
            object.__setattr__(self, "base_amount", max(0, int(self.base_amount)))
        if self.bonus_per_previous_hit is not None:
            object.__setattr__(
                self,
                "bonus_per_previous_hit",
                max(0, int(self.bonus_per_previous_hit)),
            )
        object.__setattr__(self, "ally_ids", tuple(str(ally_id) for ally_id in self.ally_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class ForgeEvent:
    """Mapping-friendly marker for one Forge grant."""

    kind: str = "forge_gained"
    amount: int = 0
    source_id: str | None = None
    target_id: str | None = "player"
    trigger: ForgeTrigger | str = ForgeTrigger.IMMEDIATE
    metadata: Mapping[str, object] = field(default_factory=dict)
    source: SourceRef = PROVISIONAL_STS2_SOURCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _normalized_id(self.kind) or "forge_gained")
        object.__setattr__(self, "amount", max(0, int(self.amount)))
        if self.source_id is not None:
            object.__setattr__(self, "source_id", _normalized_id(self.source_id))
        if self.target_id is not None:
            object.__setattr__(self, "target_id", str(self.target_id))
        object.__setattr__(self, "trigger", forge_trigger(self.trigger))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class ForgeDescriptor:
    """Timed or triggered Forge rule that an engine can register."""

    source_id: str
    trigger: ForgeTrigger | str
    amount: int | None = None
    amount_formula: str | None = None
    target_id: str = "player"
    repeat: bool = False
    duration: str = "instant"
    metadata: Mapping[str, object] = field(default_factory=dict)
    source: SourceRef = PROVISIONAL_STS2_SOURCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _normalized_id(self.source_id))
        object.__setattr__(self, "trigger", forge_trigger(self.trigger))
        if self.amount is not None:
            object.__setattr__(self, "amount", max(0, int(self.amount)))
        if self.amount_formula is not None:
            object.__setattr__(self, "amount_formula", _normalized_id(self.amount_formula))
        object.__setattr__(self, "target_id", _normalized_id(self.target_id) or "player")
        object.__setattr__(self, "duration", _normalized_id(self.duration) or "instant")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class ForgeResolution:
    """Result of applying Forge to resource counters."""

    state: ForgeState
    resources: Mapping[str, int] = field(default_factory=dict)
    events: tuple[ForgeEvent, ...] = ()
    resource_delta: int = 0
    source: SourceRef = PROVISIONAL_STS2_SOURCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resources",
            {str(key): int(value) for key, value in self.resources.items()},
        )
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "resource_delta", max(0, int(self.resource_delta)))


@dataclass(frozen=True, slots=True)
class ConquerorMark:
    """Target-specific Sovereign Blade damage multiplier."""

    target_id: str
    turns_remaining: int = 1
    damage_multiplier: int = 2
    source_id: str = "conqueror"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", str(self.target_id))
        object.__setattr__(self, "turns_remaining", max(0, int(self.turns_remaining)))
        object.__setattr__(self, "damage_multiplier", max(1, int(self.damage_multiplier)))
        object.__setattr__(self, "source_id", _normalized_id(self.source_id))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class SovereignBladeState:
    """Standalone state for the unique Sovereign Blade card."""

    card_id: str = SOVEREIGN_BLADE_CARD_ID
    instance_id: str | None = None
    name: str = SOVEREIGN_BLADE_NAME
    cost: int = DEFAULT_SOVEREIGN_BLADE_COST
    base_damage: int = DEFAULT_SOVEREIGN_BLADE_DAMAGE
    hits: int = 1
    block: int = 0
    replay: int = 0
    target: SovereignBladeTarget = SovereignBladeTarget.ENEMY
    zone: str | None = None
    upgraded: bool = False
    conqueror_marks: tuple[ConquerorMark, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    source: SourceRef = PROVISIONAL_STS2_SOURCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "card_id", _normalized_id(self.card_id) or SOVEREIGN_BLADE_CARD_ID)
        if self.instance_id is not None:
            object.__setattr__(self, "instance_id", str(self.instance_id))
        object.__setattr__(self, "name", str(self.name or SOVEREIGN_BLADE_NAME))
        object.__setattr__(self, "cost", max(0, int(self.cost)))
        object.__setattr__(self, "base_damage", max(0, int(self.base_damage)))
        object.__setattr__(self, "hits", max(1, int(self.hits)))
        object.__setattr__(self, "block", max(0, int(self.block)))
        object.__setattr__(self, "replay", max(0, int(self.replay)))
        object.__setattr__(self, "target", sovereign_blade_target(self.target))
        if self.zone is not None:
            object.__setattr__(self, "zone", normalized_blade_zone(self.zone))
        object.__setattr__(
            self,
            "conqueror_marks",
            tuple(mark for mark in self.conqueror_marks if mark.turns_remaining > 0),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class BladeEvent:
    """Mapping-friendly marker for a Sovereign Blade operation."""

    kind: str
    source_id: str | None = None
    target_id: str | None = None
    amount: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    source: SourceRef = PROVISIONAL_STS2_SOURCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _normalized_id(self.kind))
        if self.source_id is not None:
            object.__setattr__(self, "source_id", _normalized_id(self.source_id))
        if self.target_id is not None:
            object.__setattr__(self, "target_id", str(self.target_id))
        if self.amount is not None:
            object.__setattr__(self, "amount", int(self.amount))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class SovereignBladeOperation:
    """State plus events emitted by a blade operation."""

    blade: SovereignBladeState
    events: tuple[BladeEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))


@dataclass(frozen=True, slots=True)
class SovereignBladeLocation:
    """Location of a Sovereign Blade card inside plain card-zone mappings."""

    zone: str
    index: int
    card: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "zone", normalized_blade_zone(self.zone))
        object.__setattr__(self, "index", max(0, int(self.index)))
        object.__setattr__(self, "card", _clone_mapping(self.card))


@dataclass(frozen=True, slots=True)
class SovereignBladeZoneResult:
    """Result of moving or creating Sovereign Blade in plain zone mappings."""

    zones: Mapping[str, tuple[Mapping[str, object], ...]]
    blade: SovereignBladeState
    previous_zone: str | None = None
    current_zone: str = "hand"
    created: bool = False
    events: tuple[BladeEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "zones", _freeze_zones(self.zones))
        if self.previous_zone is not None:
            object.__setattr__(self, "previous_zone", normalized_blade_zone(self.previous_zone))
        object.__setattr__(self, "current_zone", normalized_blade_zone(self.current_zone))
        object.__setattr__(self, "events", tuple(self.events))


@dataclass(frozen=True, slots=True)
class ParryResult:
    """Block granted by Parry after a card-play check."""

    block: int = 0
    events: tuple[BladeEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "block", max(0, int(self.block)))
        object.__setattr__(self, "events", tuple(self.events))


def forge_trigger(value: ForgeTrigger | str) -> ForgeTrigger:
    """Normalize Forge trigger aliases."""

    if isinstance(value, ForgeTrigger):
        return value
    normalized = _normalized_id(value)
    aliases = {
        "forge": ForgeTrigger.FORGE_GAINED,
        "forged": ForgeTrigger.FORGE_GAINED,
        "on_forge": ForgeTrigger.FORGE_GAINED,
        "start_turn": ForgeTrigger.TURN_START,
        "start_of_turn": ForgeTrigger.TURN_START,
        "turn_start": ForgeTrigger.TURN_START,
        "play_card": ForgeTrigger.CARD_PLAYED,
    }
    return aliases.get(normalized, ForgeTrigger(normalized))


def forge_state_from_resources(
    resources: Mapping[str, object],
    *,
    gained_this_turn: int = 0,
    gained_this_combat: int = 0,
    times_forged_this_turn: int = 0,
    times_forged_this_combat: int = 0,
) -> ForgeState:
    """Build Forge counters from a player resource mapping."""

    return ForgeState(
        amount=forge_resource_amount(resources),
        gained_this_turn=gained_this_turn,
        gained_this_combat=gained_this_combat,
        times_forged_this_turn=times_forged_this_turn,
        times_forged_this_combat=times_forged_this_combat,
    )


def forge_resource_amount(resources: Mapping[str, object]) -> int:
    """Return the current Forge resource amount from a player resource mapping."""

    for key, value in resources.items():
        if _normalized_id(key) == "forge":
            return max(0, _coerce_int(value))
    return 0


def resources_with_forge(resources: Mapping[str, object], forge_amount: int) -> dict[str, int]:
    """Return integer resources with the Forge value replaced."""

    normalized: dict[str, int] = {}
    for key, value in resources.items():
        if isinstance(value, bool) or _normalized_id(key) == "forge":
            continue
        normalized[str(key)] = max(0, _coerce_int(value))
    normalized["forge"] = max(0, int(forge_amount))
    return normalized


def apply_forge(
    amount: int,
    *,
    state: ForgeState | None = None,
    resources: Mapping[str, object] | None = None,
    source_id: str | None = None,
    target_id: str | None = "player",
    trigger: ForgeTrigger | str = ForgeTrigger.IMMEDIATE,
    metadata: Mapping[str, object] | None = None,
) -> ForgeResolution:
    """Apply a Forge gain to counters and an optional resource mapping."""

    base_state = state or forge_state_from_resources(resources or {})
    gained = max(0, int(amount))
    next_state = base_state.add(gained)
    next_resources = resources_with_forge(resources or {}, next_state.amount)
    events: tuple[ForgeEvent, ...] = ()
    if gained > 0:
        events = (
            ForgeEvent(
                amount=gained,
                source_id=source_id,
                target_id=target_id,
                trigger=trigger,
                metadata=metadata or {},
            ),
        )
    return ForgeResolution(
        state=next_state,
        resources=next_resources,
        events=events,
        resource_delta=gained,
    )


def reset_forge_turn_counters(state: ForgeState) -> ForgeState:
    """Clear per-turn Forge counters while preserving combat totals."""

    return state.reset_turn_counters()


def beat_into_shape_forge_amount(
    *,
    previous_hits_on_target_this_turn: int,
    upgraded: bool = False,
    base_amount: int | None = None,
    bonus_per_previous_hit: int | None = None,
) -> int:
    """Return Beat into Shape's dynamic Forge amount."""

    base = _upgraded_amount(
        upgraded=upgraded,
        base=BEAT_INTO_SHAPE_FORGE,
        upgraded_base=BEAT_INTO_SHAPE_UPGRADED_FORGE,
        override=base_amount,
    )
    bonus = base if bonus_per_previous_hit is None else max(0, int(bonus_per_previous_hit))
    previous_hits = max(0, int(previous_hits_on_target_this_turn))
    return base + bonus * previous_hits


def x_cost_forge_amount(energy_spent: int, *, multiplier: int = 1, base: int = 0) -> int:
    """Return a generic X-cost Forge amount."""

    return max(0, int(base)) + max(0, int(energy_spent)) * max(0, int(multiplier))


def resolve_dynamic_forge_amount(
    formula: str,
    context: ForgeContext | None = None,
) -> int:
    """Resolve a named dynamic Forge formula from a context."""

    forge_context = context or ForgeContext()
    normalized = _normalized_id(formula)
    if normalized in {"beat_into_shape", "beat_into_shape_forge"}:
        return beat_into_shape_forge_amount(
            previous_hits_on_target_this_turn=(
                forge_context.previous_hits_on_target_this_turn
            ),
            upgraded=forge_context.upgraded,
            base_amount=forge_context.base_amount,
            bonus_per_previous_hit=forge_context.bonus_per_previous_hit,
        )
    if normalized in {"x", "x_cost", "x_spent", "energy_spent"}:
        return x_cost_forge_amount(forge_context.energy_spent)
    if normalized in {"mirrored", "source_forge_amount", "hammer_time"}:
        return max(0, forge_context.source_forge_amount)
    raise ValueError(f"Unknown dynamic Forge formula: {formula!r}")


def furnace_forge_amount(*, upgraded: bool = False) -> int:
    return FURNACE_UPGRADED_FORGE if upgraded else FURNACE_FORGE


def conqueror_forge_amount(*, upgraded: bool = False) -> int:
    return CONQUEROR_UPGRADED_FORGE if upgraded else CONQUEROR_FORGE


def seeking_edge_forge_amount(*, upgraded: bool = False) -> int:
    return SEEKING_EDGE_UPGRADED_FORGE if upgraded else SEEKING_EDGE_FORGE


def summon_forth_forge_amount(*, upgraded: bool = False) -> int:
    return SUMMON_FORTH_UPGRADED_FORGE if upgraded else SUMMON_FORTH_FORGE


def furnace_forge_descriptor(*, upgraded: bool = False) -> ForgeDescriptor:
    """Descriptor for Furnace's start-of-turn Forge."""

    return ForgeDescriptor(
        source_id="furnace",
        trigger=ForgeTrigger.TURN_START,
        amount=furnace_forge_amount(upgraded=upgraded),
        repeat=True,
        duration="combat",
    )


def hammer_time_forge_descriptor() -> ForgeDescriptor:
    """Descriptor for Hammer Time's mirrored ally Forge trigger."""

    return ForgeDescriptor(
        source_id="hammer_time",
        trigger=ForgeTrigger.FORGE_GAINED,
        amount_formula="source_forge_amount",
        target_id="all_allies",
        repeat=True,
        duration="combat",
    )


def timed_forge_descriptor(
    *,
    source_id: str,
    trigger: ForgeTrigger | str,
    amount: int,
    repeat: bool = True,
    duration: str = "combat",
    target_id: str = "player",
    metadata: Mapping[str, object] | None = None,
) -> ForgeDescriptor:
    """Build a static timed Forge descriptor."""

    return ForgeDescriptor(
        source_id=source_id,
        trigger=trigger,
        amount=amount,
        target_id=target_id,
        repeat=repeat,
        duration=duration,
        metadata=metadata or {},
    )


def triggered_forge_descriptor(
    *,
    source_id: str,
    trigger: ForgeTrigger | str,
    amount: int | None = None,
    amount_formula: str | None = None,
    target_id: str = "player",
    duration: str = "combat",
    metadata: Mapping[str, object] | None = None,
) -> ForgeDescriptor:
    """Build a triggered Forge descriptor, optionally with a dynamic formula."""

    return ForgeDescriptor(
        source_id=source_id,
        trigger=trigger,
        amount=amount,
        amount_formula=amount_formula,
        target_id=target_id,
        repeat=True,
        duration=duration,
        metadata=metadata or {},
    )


def resolve_forge_descriptor(
    descriptor: ForgeDescriptor,
    trigger: ForgeTrigger | str,
    *,
    context: ForgeContext | None = None,
) -> tuple[ForgeEvent, ...]:
    """Return Forge events emitted when a descriptor sees a trigger."""

    normalized_trigger = forge_trigger(trigger)
    if descriptor.trigger is not normalized_trigger:
        return ()
    forge_context = context or ForgeContext()
    if descriptor.amount is not None:
        amount = descriptor.amount
    elif descriptor.amount_formula is not None:
        amount = resolve_dynamic_forge_amount(descriptor.amount_formula, forge_context)
    else:
        amount = 0
    if amount <= 0:
        return ()
    target_ids = _descriptor_target_ids(descriptor, forge_context)
    metadata = {
        "duration": descriptor.duration,
        "repeat": descriptor.repeat,
        **dict(descriptor.metadata),
    }
    if descriptor.amount_formula is not None:
        metadata["amount_formula"] = descriptor.amount_formula
    return tuple(
        ForgeEvent(
            amount=amount,
            source_id=descriptor.source_id,
            target_id=target_id,
            trigger=normalized_trigger,
            metadata=metadata,
            source=descriptor.source,
        )
        for target_id in target_ids
    )


def hammer_time_ally_forge_events(
    source_forge_amount: int,
    ally_ids: Sequence[str],
) -> tuple[ForgeEvent, ...]:
    """Return mirrored ally Forge events for Hammer Time."""

    return resolve_forge_descriptor(
        hammer_time_forge_descriptor(),
        ForgeTrigger.FORGE_GAINED,
        context=ForgeContext(
            source_forge_amount=source_forge_amount,
            ally_ids=tuple(ally_ids),
        ),
    )


def create_sovereign_blade_state(
    *,
    instance_id: str | None = None,
    zone: str | None = None,
    base_damage: int = DEFAULT_SOVEREIGN_BLADE_DAMAGE,
    hits: int = 1,
    block: int = 0,
    replay: int = 0,
    target: SovereignBladeTarget | str = SovereignBladeTarget.ENEMY,
    upgraded: bool = False,
) -> SovereignBladeState:
    """Return default Sovereign Blade state with optional overrides."""

    return SovereignBladeState(
        instance_id=instance_id,
        zone=zone,
        base_damage=base_damage,
        hits=hits,
        block=block,
        replay=replay,
        target=sovereign_blade_target(target),
        upgraded=upgraded,
    )


def sovereign_blade_target(value: SovereignBladeTarget | str) -> SovereignBladeTarget:
    """Normalize target aliases for Sovereign Blade."""

    if isinstance(value, SovereignBladeTarget):
        return value
    normalized = _normalized_id(value)
    aliases = {
        "all": SovereignBladeTarget.ALL_ENEMIES,
        "all_enemies": SovereignBladeTarget.ALL_ENEMIES,
        "allenemies": SovereignBladeTarget.ALL_ENEMIES,
        "aoe": SovereignBladeTarget.ALL_ENEMIES,
        "any_enemy": SovereignBladeTarget.ENEMY,
        "anyenemy": SovereignBladeTarget.ENEMY,
        "enemy": SovereignBladeTarget.ENEMY,
    }
    return aliases.get(normalized, SovereignBladeTarget.ENEMY)


def apply_conqueror(
    blade: SovereignBladeState,
    *,
    target_id: str,
    turns: int = 1,
    damage_multiplier: int = 2,
) -> SovereignBladeState:
    """Mark a target to take multiplied Sovereign Blade damage."""

    mark = ConquerorMark(
        target_id=target_id,
        turns_remaining=turns,
        damage_multiplier=damage_multiplier,
    )
    marks = tuple(
        existing
        for existing in blade.conqueror_marks
        if existing.target_id != mark.target_id and existing.turns_remaining > 0
    )
    return replace(blade, conqueror_marks=marks + (mark,))


def conqueror_operation(
    blade: SovereignBladeState,
    *,
    target_id: str,
    turns: int = 1,
) -> SovereignBladeOperation:
    """Apply Conqueror and return an operation marker."""

    next_blade = apply_conqueror(blade, target_id=target_id, turns=turns)
    return SovereignBladeOperation(
        blade=next_blade,
        events=(
            BladeEvent(
                kind="sovereign_blade_conqueror_mark",
                source_id="conqueror",
                target_id=target_id,
                amount=turns,
                metadata={"damage_multiplier": 2},
            ),
        ),
    )


def apply_seeking_edge(blade: SovereignBladeState) -> SovereignBladeState:
    """Make Sovereign Blade hit all enemies."""

    return replace(blade, target=SovereignBladeTarget.ALL_ENEMIES)


def seeking_edge_operation(blade: SovereignBladeState) -> SovereignBladeOperation:
    next_blade = apply_seeking_edge(blade)
    return SovereignBladeOperation(
        blade=next_blade,
        events=(
            BladeEvent(
                kind="sovereign_blade_target_changed",
                source_id="seeking_edge",
                target_id=SovereignBladeTarget.ALL_ENEMIES.value,
            ),
        ),
    )


def apply_parry(
    blade: SovereignBladeState,
    *,
    amount: int = PARRY_BLOCK,
) -> SovereignBladeState:
    """Make Sovereign Blade gain block directly when played."""

    return replace(blade, block=blade.block + max(0, int(amount)))


def parry_operation(
    blade: SovereignBladeState,
    *,
    amount: int = PARRY_BLOCK,
) -> SovereignBladeOperation:
    next_blade = apply_parry(blade, amount=amount)
    return SovereignBladeOperation(
        blade=next_blade,
        events=(
            BladeEvent(
                kind="sovereign_blade_block_changed",
                source_id="parry",
                amount=max(0, int(amount)),
                metadata={"block": next_blade.block},
            ),
        ),
    )


def apply_sword_sage(blade: SovereignBladeState, *, amount: int = 1) -> SovereignBladeState:
    """Give Sovereign Blade Replay."""

    return replace(blade, replay=blade.replay + max(0, int(amount)))


def sword_sage_operation(
    blade: SovereignBladeState,
    *,
    amount: int = 1,
) -> SovereignBladeOperation:
    next_blade = apply_sword_sage(blade, amount=amount)
    return SovereignBladeOperation(
        blade=next_blade,
        events=(
            BladeEvent(
                kind="sovereign_blade_replay_changed",
                source_id="sword_sage",
                amount=max(0, int(amount)),
                metadata={"replay": next_blade.replay},
            ),
        ),
    )


def summon_forth_blade_state(blade: SovereignBladeState | None = None) -> SovereignBladeState:
    """Return a blade state located in hand."""

    return replace(blade or create_sovereign_blade_state(), zone="hand")


def tick_sovereign_blade_turn(blade: SovereignBladeState) -> SovereignBladeState:
    """Decrement temporary Conqueror marks at turn boundary."""

    marks = tuple(
        replace(mark, turns_remaining=mark.turns_remaining - 1)
        for mark in blade.conqueror_marks
        if mark.turns_remaining > 1
    )
    return replace(blade, conqueror_marks=marks)


def sovereign_blade_damage(
    blade: SovereignBladeState,
    *,
    target_id: str | None = None,
) -> int:
    """Return Sovereign Blade's per-hit damage against a target."""

    return blade.base_damage * conqueror_multiplier(blade, target_id=target_id)


def conqueror_multiplier(
    blade: SovereignBladeState,
    *,
    target_id: str | None,
) -> int:
    """Return the active Conqueror multiplier for a target."""

    if target_id is None:
        return 1
    multipliers = (
        mark.damage_multiplier
        for mark in blade.conqueror_marks
        if mark.target_id == target_id and mark.turns_remaining > 0
    )
    return max(multipliers, default=1)


def sovereign_blade_hit_sequence(
    blade: SovereignBladeState,
    *,
    target_id: str | None = None,
) -> tuple[int, ...]:
    """Return per-hit damage values for one Sovereign Blade play."""

    damage = sovereign_blade_damage(blade, target_id=target_id)
    return tuple(damage for _ in range(blade.hits))


def parry_block_amount(*, upgraded: bool = False) -> int:
    return PARRY_UPGRADED_BLOCK if upgraded else PARRY_BLOCK


def resolve_parry_on_blade_play(
    card: str | Mapping[str, object],
    *,
    upgraded: bool = False,
    amount: int | None = None,
) -> ParryResult:
    """Return Parry block if the played card is Sovereign Blade."""

    if not is_sovereign_blade_card(card):
        return ParryResult()
    block = parry_block_amount(upgraded=upgraded) if amount is None else max(0, int(amount))
    return ParryResult(
        block=block,
        events=(
            BladeEvent(
                kind="gain_block",
                source_id="parry",
                target_id="player",
                amount=block,
                metadata={"card_id": SOVEREIGN_BLADE_CARD_ID},
            ),
        ),
    )


def sovereign_blade_state_mapping(blade: SovereignBladeState) -> dict[str, object]:
    """Serialize blade state into a plain mapping."""

    payload: dict[str, object] = {
        "card_id": blade.card_id,
        "name": blade.name,
        "cost": blade.cost,
        "base_damage": blade.base_damage,
        "hits": blade.hits,
        "block": blade.block,
        "replay": blade.replay,
        "target": blade.target.value,
        "upgraded": blade.upgraded,
        "conqueror_marks": tuple(conqueror_mark_mapping(mark) for mark in blade.conqueror_marks),
        "metadata": dict(blade.metadata),
    }
    if blade.instance_id is not None:
        payload["instance_id"] = blade.instance_id
    if blade.zone is not None:
        payload["zone"] = blade.zone
    return payload


def conqueror_mark_mapping(mark: ConquerorMark) -> dict[str, object]:
    """Serialize a Conqueror mark into a plain mapping."""

    return {
        "target_id": mark.target_id,
        "turns_remaining": mark.turns_remaining,
        "damage_multiplier": mark.damage_multiplier,
        "source_id": mark.source_id,
        "metadata": dict(mark.metadata),
    }


def sovereign_blade_card(
    blade: SovereignBladeState | None = None,
    *,
    instance_id: str | None = None,
    zone: str | None = None,
) -> dict[str, object]:
    """Return an engine-shaped mapping for Sovereign Blade."""

    blade_state = blade or create_sovereign_blade_state()
    if instance_id is not None:
        blade_state = replace(blade_state, instance_id=instance_id)
    if zone is not None:
        blade_state = replace(blade_state, zone=zone)
    damage_key = (
        "all_damage"
        if blade_state.target is SovereignBladeTarget.ALL_ENEMIES
        else "damage"
    )
    sequence: list[dict[str, object]] = [
        {damage_key: blade_state.base_damage} for _ in range(blade_state.hits)
    ]
    if blade_state.block:
        sequence.append({"block": blade_state.block})
    custom: dict[str, object] = {
        "sovereign_blade": sovereign_blade_state_mapping(blade_state)
    }
    if blade_state.replay:
        custom["replay"] = blade_state.replay

    card: dict[str, object] = {
        "card_id": SOVEREIGN_BLADE_CARD_ID,
        "name": SOVEREIGN_BLADE_NAME,
        "type": "attack",
        "target": blade_state.target.value,
        "cost": blade_state.cost,
        "damage": blade_state.base_damage,
        "hit_count": blade_state.hits,
        "effects": {"sequence": sequence},
        "upgraded": blade_state.upgraded,
        "custom": custom,
    }
    if blade_state.block:
        card["block"] = blade_state.block
    if blade_state.instance_id is not None:
        card["instance_id"] = blade_state.instance_id
    return card


def sovereign_blade_from_card(
    card: Mapping[str, object],
    *,
    zone: str | None = None,
) -> SovereignBladeState:
    """Recover blade state from a plain card mapping."""

    custom = card.get("custom")
    payload = _mapping_child(custom, "sovereign_blade")
    source = payload or card
    instance_id = _optional_str(source.get("instance_id", card.get("instance_id")))
    target_value = source.get("target", card.get("target", SovereignBladeTarget.ENEMY.value))
    return SovereignBladeState(
        card_id=_content_id(card),
        instance_id=instance_id,
        name=str(source.get("name", card.get("name", SOVEREIGN_BLADE_NAME))),
        cost=_mapping_int(source, "cost", default=_mapping_int(card, "cost", default=2)),
        base_damage=_mapping_int(
            source,
            "base_damage",
            default=_mapping_int(card, "damage", default=_damage_from_effects(card) or 10),
        ),
        hits=_mapping_int(
            source,
            "hits",
            default=_mapping_int(card, "hit_count", default=_hit_count_from_effects(card) or 1),
        ),
        block=_mapping_int(
            source,
            "block",
            default=_mapping_int(card, "block", default=_block_from_effects(card) or 0),
        ),
        replay=_mapping_int(
            source,
            "replay",
            default=_mapping_int(
                _mapping_child(card.get("custom"), None) or {},
                "replay",
                default=0,
            ),
        ),
        target=sovereign_blade_target(str(target_value)),
        zone=zone or _optional_str(source.get("zone")),
        upgraded=bool(source.get("upgraded", card.get("upgraded", False))),
        conqueror_marks=_marks_from_payload(source.get("conqueror_marks")),
        metadata=_mapping_child(source.get("metadata"), None) or {},
    )


def with_sovereign_blade_state(
    card: Mapping[str, object],
    blade: SovereignBladeState,
) -> dict[str, object]:
    """Return a card mapping updated to match a blade state."""

    updated = _clone_mapping(card)
    blade_card = sovereign_blade_card(blade)
    updated.update(
        {
            "card_id": SOVEREIGN_BLADE_CARD_ID,
            "name": SOVEREIGN_BLADE_NAME,
            "type": blade_card["type"],
            "target": blade_card["target"],
            "cost": blade_card["cost"],
            "damage": blade_card["damage"],
            "hit_count": blade_card["hit_count"],
            "effects": blade_card["effects"],
            "upgraded": blade_card["upgraded"],
        }
    )
    if "block" in blade_card:
        updated["block"] = blade_card["block"]
    else:
        updated.pop("block", None)
    if blade.instance_id is not None:
        updated["instance_id"] = blade.instance_id
    custom = _clone_mapping(card.get("custom")) if isinstance(card.get("custom"), Mapping) else {}
    custom["sovereign_blade"] = sovereign_blade_state_mapping(blade)
    if blade.replay:
        custom["replay"] = blade.replay
    else:
        custom.pop("replay", None)
    updated["custom"] = custom
    return updated


def is_sovereign_blade_card(card: str | Mapping[str, object]) -> bool:
    """Return whether a raw id or card mapping identifies Sovereign Blade."""

    if isinstance(card, str):
        return _normalized_id(card) == SOVEREIGN_BLADE_CARD_ID
    return _content_id(card) == SOVEREIGN_BLADE_CARD_ID


def find_sovereign_blade(
    zones: Mapping[str, Sequence[Mapping[str, object]]],
) -> SovereignBladeLocation | None:
    """Find the first Sovereign Blade card in normalized card-zone mappings."""

    for raw_zone, cards in zones.items():
        zone = normalized_blade_zone(raw_zone)
        for index, card in enumerate(cards):
            if is_sovereign_blade_card(card):
                return SovereignBladeLocation(zone=zone, index=index, card=card)
    return None


def summon_forth_zones(
    zones: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    blade: SovereignBladeState | None = None,
    create_if_missing: bool = True,
) -> SovereignBladeZoneResult:
    """Move Sovereign Blade into hand from any plain card-zone mapping."""

    return move_sovereign_blade_to_hand(
        zones,
        blade=blade,
        create_if_missing=create_if_missing,
        source_id="summon_forth",
    )


def move_sovereign_blade_to_hand(
    zones: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    blade: SovereignBladeState | None = None,
    create_if_missing: bool = True,
    source_id: str = "sovereign_blade_move",
) -> SovereignBladeZoneResult:
    """Move or create Sovereign Blade in the hand zone."""

    mutable_zones = _clone_zone_lists(zones)
    mutable_zones.setdefault("hand", [])
    location = _find_in_zone_lists(mutable_zones)
    previous_zone: str | None = None
    created = False

    if location is None:
        if not create_if_missing:
            blade_state = summon_forth_blade_state(blade)
            return SovereignBladeZoneResult(
                zones=_zone_tuples(mutable_zones),
                blade=blade_state,
                previous_zone=None,
                current_zone="hand",
                created=False,
                events=(),
            )
        blade_state = summon_forth_blade_state(blade)
        card = sovereign_blade_card(blade_state)
        mutable_zones["hand"].append(card)
        created = True
    else:
        previous_zone = location.zone
        card = mutable_zones[location.zone][location.index]
        blade_state = summon_forth_blade_state(blade or sovereign_blade_from_card(card))
        updated_card = with_sovereign_blade_state(card, blade_state)
        if location.zone == "hand":
            mutable_zones["hand"][location.index] = updated_card
        else:
            del mutable_zones[location.zone][location.index]
            mutable_zones["hand"].append(updated_card)

    event_kind = "sovereign_blade_created" if created else "sovereign_blade_moved"
    return SovereignBladeZoneResult(
        zones=_zone_tuples(mutable_zones),
        blade=blade_state,
        previous_zone=previous_zone,
        current_zone="hand",
        created=created,
        events=(
            BladeEvent(
                kind=event_kind,
                source_id=source_id,
                target_id="hand",
                metadata={"from_zone": previous_zone, "created": created},
            ),
        ),
    )


def replace_sovereign_blade_in_zones(
    zones: Mapping[str, Sequence[Mapping[str, object]]],
    blade: SovereignBladeState,
    *,
    create_if_missing: bool = True,
) -> SovereignBladeZoneResult:
    """Replace the first blade card in zones with a card built from state."""

    mutable_zones = _clone_zone_lists(zones)
    location = _find_in_zone_lists(mutable_zones)
    destination = normalized_blade_zone(blade.zone or (location.zone if location else "hand"))
    mutable_zones.setdefault(destination, [])
    previous_zone = location.zone if location else None
    blade_state = replace(blade, zone=destination)

    if location is None:
        if create_if_missing:
            mutable_zones[destination].append(sovereign_blade_card(blade_state))
        return SovereignBladeZoneResult(
            zones=_zone_tuples(mutable_zones),
            blade=blade_state,
            previous_zone=None,
            current_zone=destination,
            created=create_if_missing,
            events=(),
        )

    card = mutable_zones[location.zone][location.index]
    updated_card = with_sovereign_blade_state(card, blade_state)
    if location.zone == destination:
        mutable_zones[location.zone][location.index] = updated_card
    else:
        del mutable_zones[location.zone][location.index]
        mutable_zones[destination].append(updated_card)
    return SovereignBladeZoneResult(
        zones=_zone_tuples(mutable_zones),
        blade=blade_state,
        previous_zone=previous_zone,
        current_zone=destination,
        created=False,
        events=(
            BladeEvent(
                kind="sovereign_blade_replaced",
                source_id="sovereign_blade_state",
                target_id=destination,
                metadata={"from_zone": previous_zone},
            ),
        ),
    )


def normalized_blade_zone(value: object) -> str:
    """Normalize common engine/card-pile zone names."""

    normalized = _normalized_id(value)
    aliases = {
        "discard": "discard_pile",
        "discard_pile": "discard_pile",
        "draw": "draw_pile",
        "draw_pile": "draw_pile",
        "draw_pile_top": "draw_pile",
        "exhaust": "exhaust_pile",
        "exhaust_pile": "exhaust_pile",
        "hand": "hand",
        "master_deck": "master_deck",
    }
    return aliases.get(normalized, normalized or "unknown")


def _descriptor_target_ids(
    descriptor: ForgeDescriptor,
    context: ForgeContext,
) -> tuple[str | None, ...]:
    if descriptor.target_id == "all_allies":
        return tuple(context.ally_ids)
    return (descriptor.target_id,)


def _upgraded_amount(
    *,
    upgraded: bool,
    base: int,
    upgraded_base: int,
    override: int | None,
) -> int:
    if override is not None:
        return max(0, int(override))
    return upgraded_base if upgraded else base


def _content_id(card: Mapping[str, object]) -> str:
    for key in ("card_id", "id", "content_id", "name"):
        value = card.get(key)
        if value is not None:
            return _normalized_id(value)
    return ""


def _mapping_child(value: object, key: str | None) -> Mapping[str, object] | None:
    if key is None:
        return value if isinstance(value, Mapping) else None
    if not isinstance(value, Mapping):
        return None
    child = value.get(key)
    return child if isinstance(child, Mapping) else None


def _mapping_int(mapping: Mapping[str, object], key: str, *, default: int) -> int:
    return max(0, _coerce_int(mapping.get(key), default=default))


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _damage_from_effects(card: Mapping[str, object]) -> int | None:
    for step in _effect_sequence(card):
        if not isinstance(step, Mapping):
            continue
        for key in ("damage", "all_damage"):
            if key in step:
                return max(0, _coerce_int(step.get(key)))
    return None


def _hit_count_from_effects(card: Mapping[str, object]) -> int | None:
    count = 0
    for step in _effect_sequence(card):
        if isinstance(step, Mapping) and ("damage" in step or "all_damage" in step):
            count += 1
    return count or None


def _block_from_effects(card: Mapping[str, object]) -> int | None:
    total = 0
    for step in _effect_sequence(card):
        if isinstance(step, Mapping) and "block" in step:
            total += max(0, _coerce_int(step.get("block")))
    return total or None


def _effect_sequence(card: Mapping[str, object]) -> tuple[object, ...]:
    effects = card.get("effects")
    if not isinstance(effects, Mapping):
        return ()
    sequence = effects.get("sequence")
    if isinstance(sequence, Sequence) and not isinstance(sequence, (str, bytes, bytearray)):
        return tuple(sequence)
    return ()


def _marks_from_payload(value: object) -> tuple[ConquerorMark, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    marks: list[ConquerorMark] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        target_id = item.get("target_id")
        if target_id is None:
            continue
        metadata = _mapping_child(item.get("metadata"), None) or {}
        marks.append(
            ConquerorMark(
                target_id=str(target_id),
                turns_remaining=_mapping_int(item, "turns_remaining", default=1),
                damage_multiplier=_mapping_int(item, "damage_multiplier", default=2),
                source_id=str(item.get("source_id", "conqueror")),
                metadata=metadata,
            )
        )
    return tuple(marks)


def _clone_zone_lists(
    zones: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    cloned: dict[str, list[dict[str, object]]] = {}
    for raw_zone, cards in zones.items():
        zone = normalized_blade_zone(raw_zone)
        cloned.setdefault(zone, []).extend(_clone_mapping(card) for card in cards)
    return cloned


def _find_in_zone_lists(
    zones: Mapping[str, Sequence[Mapping[str, object]]],
) -> SovereignBladeLocation | None:
    for zone, cards in zones.items():
        for index, card in enumerate(cards):
            if is_sovereign_blade_card(card):
                return SovereignBladeLocation(zone=zone, index=index, card=card)
    return None


def _zone_tuples(
    zones: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, tuple[Mapping[str, object], ...]]:
    return {zone: tuple(_clone_mapping(card) for card in cards) for zone, cards in zones.items()}


def _freeze_zones(
    zones: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, tuple[Mapping[str, object], ...]]:
    return {
        normalized_blade_zone(zone): tuple(_clone_mapping(card) for card in cards)
        for zone, cards in zones.items()
    }


def _clone_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _clone_jsonish(item) for key, item in value.items()}


def _clone_jsonish(value: object) -> object:
    if isinstance(value, Mapping):
        return _clone_mapping(value)
    if isinstance(value, tuple):
        return tuple(_clone_jsonish(item) for item in value)
    if isinstance(value, list):
        return [_clone_jsonish(item) for item in value]
    return value


def _coerce_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _normalized_id(value: object) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("'", "")
        .replace(".", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


__all__ = [
    "BEAT_INTO_SHAPE_FORGE",
    "BEAT_INTO_SHAPE_UPGRADED_FORGE",
    "CONQUEROR_FORGE",
    "CONQUEROR_UPGRADED_FORGE",
    "DEFAULT_SOVEREIGN_BLADE_COST",
    "DEFAULT_SOVEREIGN_BLADE_DAMAGE",
    "FURNACE_FORGE",
    "FURNACE_UPGRADED_FORGE",
    "PARRY_BLOCK",
    "PARRY_UPGRADED_BLOCK",
    "SEEKING_EDGE_FORGE",
    "SEEKING_EDGE_UPGRADED_FORGE",
    "SOVEREIGN_BLADE_CARD_ID",
    "SOVEREIGN_BLADE_NAME",
    "SUMMON_FORTH_FORGE",
    "SUMMON_FORTH_UPGRADED_FORGE",
    "BladeEvent",
    "ConquerorMark",
    "ForgeContext",
    "ForgeDescriptor",
    "ForgeEvent",
    "ForgeResolution",
    "ForgeState",
    "ForgeTrigger",
    "ParryResult",
    "SovereignBladeLocation",
    "SovereignBladeOperation",
    "SovereignBladeState",
    "SovereignBladeTarget",
    "SovereignBladeZoneResult",
    "apply_conqueror",
    "apply_forge",
    "apply_parry",
    "apply_seeking_edge",
    "apply_sword_sage",
    "beat_into_shape_forge_amount",
    "conqueror_forge_amount",
    "conqueror_mark_mapping",
    "conqueror_multiplier",
    "conqueror_operation",
    "create_sovereign_blade_state",
    "find_sovereign_blade",
    "forge_resource_amount",
    "forge_state_from_resources",
    "forge_trigger",
    "furnace_forge_amount",
    "furnace_forge_descriptor",
    "hammer_time_ally_forge_events",
    "hammer_time_forge_descriptor",
    "is_sovereign_blade_card",
    "move_sovereign_blade_to_hand",
    "normalized_blade_zone",
    "parry_operation",
    "parry_block_amount",
    "replace_sovereign_blade_in_zones",
    "reset_forge_turn_counters",
    "resolve_dynamic_forge_amount",
    "resolve_forge_descriptor",
    "resolve_parry_on_blade_play",
    "resources_with_forge",
    "seeking_edge_forge_amount",
    "seeking_edge_operation",
    "sovereign_blade_card",
    "sovereign_blade_damage",
    "sovereign_blade_from_card",
    "sovereign_blade_hit_sequence",
    "sovereign_blade_state_mapping",
    "sovereign_blade_target",
    "summon_forth_blade_state",
    "summon_forth_forge_amount",
    "summon_forth_zones",
    "sword_sage_operation",
    "tick_sovereign_blade_turn",
    "timed_forge_descriptor",
    "triggered_forge_descriptor",
    "with_sovereign_blade_state",
    "x_cost_forge_amount",
]
