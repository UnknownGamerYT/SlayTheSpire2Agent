"""Pure helpers for Defect-style orb card effects.

The helpers in this module intentionally accept plain dictionaries, strings,
and tuples instead of engine model classes.  They are meant to be a small bridge
between source-card normalization and the combat engine's eventual timed orb
integration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

type OrbInput = Mapping[str, object] | str
type EffectStep = Mapping[str, object]
type TriggerDuration = Literal["combat", "turn", "once", "uses"]
type PassiveSelector = Literal["leftmost", "rightmost", "all", "matching"]

ORB_TYPES = frozenset({"lightning", "frost", "dark", "plasma", "glass"})


@dataclass(frozen=True, slots=True)
class OrbSnapshot:
    """Normalized, engine-free view of an orb."""

    orb_id: str
    value: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "orb_id", normalize_orb_id(self.orb_id))
        object.__setattr__(self, "value", max(0, _coerce_int(self.value)))

    @classmethod
    def from_raw(cls, raw: OrbInput) -> OrbSnapshot:
        if isinstance(raw, str):
            return cls(raw)
        orb_id = raw.get("orb_id") or raw.get("orb") or raw.get("id") or raw.get("type")
        return cls(str(orb_id or "lightning"), _coerce_int(raw.get("value")))

    def as_mapping(self) -> dict[str, object]:
        return {"orb_id": self.orb_id, "value": self.value}


@dataclass(frozen=True, slots=True)
class PassiveOrbTriggerDescriptor:
    """Descriptor for manually triggering one or more orb passives."""

    selector: str = "rightmost"
    amount: int = 1
    orb_id: str | None = None
    direction: str | None = None
    index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "selector", _normalized_selector(self.selector))
        object.__setattr__(self, "amount", max(0, _coerce_int(self.amount, default=1)))
        if self.orb_id is not None:
            object.__setattr__(self, "orb_id", normalize_orb_id(self.orb_id))
        if self.direction is not None:
            object.__setattr__(self, "direction", _normalized_id(self.direction))
        if self.index is not None:
            object.__setattr__(self, "index", max(0, _coerce_int(self.index)))

    def as_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "selector": self.selector,
            "amount": self.amount,
        }
        if self.orb_id is not None:
            payload["orb"] = self.orb_id
        if self.direction is not None:
            payload["direction"] = self.direction
        if self.index is not None:
            payload["index"] = self.index
        return payload

    def as_effect(self) -> dict[str, object]:
        return {"trigger_orb_passive": self.as_payload()}

    def as_repeated_effects(self) -> tuple[dict[str, object], ...]:
        if self.amount <= 1 or self.selector not in {"leftmost", "rightmost"}:
            return (self.as_effect(),)
        single = PassiveOrbTriggerDescriptor(
            selector=self.selector,
            amount=1,
            orb_id=self.orb_id,
            direction=self.direction,
            index=self.index,
        )
        return tuple(single.as_effect() for _ in range(self.amount))


@dataclass(frozen=True, slots=True)
class TimedOrbTriggerDescriptor:
    """Descriptor for registering an orb-related combat trigger."""

    trigger: str
    effects: tuple[EffectStep, ...]
    duration: TriggerDuration | str = "combat"
    condition: Mapping[str, object] = field(default_factory=dict)
    delay: int = 0
    uses: int | None = None
    counter_scope: str | None = None
    every: int | None = None
    text: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trigger", _normalized_id(self.trigger))
        object.__setattr__(self, "duration", _normalized_id(self.duration) or "combat")
        object.__setattr__(
            self,
            "effects",
            tuple(_copy_mapping(effect) for effect in self.effects),
        )
        object.__setattr__(self, "condition", _copy_mapping(self.condition))
        object.__setattr__(self, "delay", max(0, _coerce_int(self.delay)))
        if self.uses is not None:
            object.__setattr__(self, "uses", max(0, _coerce_int(self.uses)))
        if self.counter_scope is not None:
            object.__setattr__(self, "counter_scope", _normalized_id(self.counter_scope))
        if self.every is not None:
            object.__setattr__(self, "every", max(0, _coerce_int(self.every)))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    def as_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "trigger": self.trigger,
            "duration": self.duration,
            "effects": tuple(_copy_mapping(effect) for effect in self.effects),
        }
        if self.condition:
            payload["condition"] = _copy_mapping(self.condition)
        if self.delay:
            payload["delay"] = self.delay
        if self.uses is not None:
            payload["uses"] = self.uses
        if self.counter_scope:
            payload["counter_scope"] = self.counter_scope
        if self.every is not None:
            payload["every"] = self.every
        if self.text:
            payload["text"] = self.text
        if self.metadata:
            payload["metadata"] = _copy_mapping(self.metadata)
        return payload

    def as_effect(self) -> dict[str, object]:
        return {"combat_trigger": self.as_payload()}


def normalize_orb_id(value: object, *, default: str = "lightning") -> str:
    normalized = _normalized_id(value)
    if normalized == "random":
        return "random_orb"
    if normalized in ORB_TYPES or normalized == "random_orb":
        return normalized
    return default if default in ORB_TYPES or default == "random_orb" else "lightning"


def orb_snapshots(orbs: Sequence[OrbInput] = ()) -> tuple[OrbSnapshot, ...]:
    return tuple(OrbSnapshot.from_raw(orb) for orb in orbs)


def orb_ids(orbs: Sequence[OrbInput] = ()) -> tuple[str, ...]:
    return tuple(orb.orb_id for orb in orb_snapshots(orbs))


def unique_orb_types(orbs: Sequence[OrbInput] = ()) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for orb_id in orb_ids(orbs):
        if orb_id in seen:
            continue
        seen.add(orb_id)
        unique.append(orb_id)
    return tuple(unique)


def unique_orb_count(orbs: Sequence[OrbInput] = ()) -> int:
    return len(unique_orb_types(orbs))


def count_orbs(orbs: Sequence[OrbInput] = (), orb_id: str | None = None) -> int:
    normalized = normalize_orb_id(orb_id) if orb_id is not None else None
    return sum(
        1
        for snapshot in orb_snapshots(orbs)
        if normalized is None or snapshot.orb_id == normalized
    )


def open_orb_slots(orbs: Sequence[OrbInput] = (), orb_slots: int = 0) -> int:
    return max(0, _coerce_int(orb_slots) - len(orbs))


def scale_by_unique_orbs(
    orbs: Sequence[OrbInput] = (),
    *,
    amount_per_unique: int = 1,
    base: int = 0,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    amount = _coerce_int(base) + unique_orb_count(orbs) * _coerce_int(amount_per_unique)
    return _clamp(amount, minimum=minimum, maximum=maximum)


def scale_by_orb_count(
    orbs: Sequence[OrbInput] = (),
    *,
    amount_per_orb: int = 1,
    orb_id: str | None = None,
    base: int = 0,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    amount = _coerce_int(base) + count_orbs(orbs, orb_id) * _coerce_int(amount_per_orb)
    return _clamp(amount, minimum=minimum, maximum=maximum)


def scaled_amount_from_orbs(
    orbs: Sequence[OrbInput] = (),
    *,
    base: int = 0,
    per_orb: int = 0,
    per_unique: int = 0,
    matching_orb: str | None = None,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    matching_count = count_orbs(orbs, matching_orb)
    matching_unique = unique_orb_count(orbs) if matching_orb is None else int(bool(matching_count))
    amount = (
        _coerce_int(base)
        + matching_count * _coerce_int(per_orb)
        + matching_unique * _coerce_int(per_unique)
    )
    return _clamp(amount, minimum=minimum, maximum=maximum)


def compile_driver_draw(orbs: Sequence[OrbInput] = ()) -> int:
    return scale_by_unique_orbs(orbs)


def coolant_block(orbs: Sequence[OrbInput] = (), *, amount_per_unique: int = 2) -> int:
    return scale_by_unique_orbs(orbs, amount_per_unique=amount_per_unique)


def synchronize_focus(orbs: Sequence[OrbInput] = (), *, amount_per_unique: int = 2) -> int:
    return scale_by_unique_orbs(orbs, amount_per_unique=amount_per_unique)


def alive_enemy_count(enemies: Sequence[Mapping[str, object]] = ()) -> int:
    return sum(1 for enemy in enemies if _enemy_is_alive(enemy))


def chill_channel_count(enemies: Sequence[Mapping[str, object]] = ()) -> int:
    return alive_enemy_count(enemies)


def tempest_channel_count(
    energy_spent: int,
    *,
    upgraded: bool = False,
    bonus_channels: int = 0,
) -> int:
    upgrade_bonus = 1 if upgraded else 0
    return max(0, _coerce_int(energy_spent) + upgrade_bonus + _coerce_int(bonus_channels))


def orb_slot_channel_count(
    orb_slots: int,
    *,
    orbs: Sequence[OrbInput] = (),
    open_slots_only: bool = False,
) -> int:
    if open_slots_only:
        return open_orb_slots(orbs, orb_slots)
    return max(0, _coerce_int(orb_slots))


def channeled_orb_count(
    events: Sequence[Mapping[str, object]] = (),
    *,
    orb_id: str | None = None,
) -> int:
    normalized_orb = normalize_orb_id(orb_id) if orb_id is not None else None
    total = 0
    for event in events:
        if _normalized_id(event.get("kind")) != "orb_channeled":
            continue
        metadata = _mapping_from(event.get("metadata"))
        event_orb = normalize_orb_id(metadata.get("orb") or event.get("orb"))
        if normalized_orb is not None and event_orb != normalized_orb:
            continue
        total += max(1, _coerce_int(event.get("amount"), default=1))
    return total


def metadata_orb_channel_count(
    metadata: Mapping[str, object] | None = None,
    *,
    orb_id: str = "lightning",
) -> int:
    metadata = metadata or {}
    normalized = normalize_orb_id(orb_id)
    channel_counts = {
        _normalized_id(key): _coerce_int(value)
        for key, value in _mapping_from(metadata.get("orb_channel_counts")).items()
    }
    if channel_counts:
        return max(0, channel_counts.get(normalized, 0))

    keys = (
        f"{normalized}_channeled",
        f"{normalized}_channeled_this_combat",
        f"{normalized}_orbs_channeled",
        f"{normalized}_orbs_channeled_this_combat",
    )
    for key in keys:
        if key in metadata:
            return max(0, _coerce_int(metadata.get(key)))
    return 0


def voltaic_channel_count(
    *,
    channel_count: int | None = None,
    events: Sequence[Mapping[str, object]] = (),
    metadata: Mapping[str, object] | None = None,
    orb_id: str = "lightning",
) -> int:
    if channel_count is not None:
        return max(0, _coerce_int(channel_count))
    event_count = channeled_orb_count(events, orb_id=orb_id)
    if event_count:
        return event_count
    return metadata_orb_channel_count(metadata, orb_id=orb_id)


def channel_orb_effect(orb_id: str, amount: object = 1) -> dict[str, object]:
    return {
        "channel_orb": {
            "orb": normalize_orb_id(orb_id),
            "amount": _effect_amount_value(amount),
        }
    }


def evoke_orb_effect(selector: str = "leftmost", amount: object = 1) -> dict[str, object]:
    return {
        "evoke_orb": {
            "selector": _normalized_selector(selector),
            "amount": _effect_amount_value(amount),
        }
    }


def trigger_orb_passive_effect(
    selector: str = "rightmost",
    *,
    amount: int = 1,
    orb_id: str | None = None,
    direction: str | None = None,
    index: int | None = None,
) -> dict[str, object]:
    return PassiveOrbTriggerDescriptor(
        selector=selector,
        amount=amount,
        orb_id=orb_id,
        direction=direction,
        index=index,
    ).as_effect()


def orb_evoke_damage_effect(amount: int, *, target: str = "enemies_hit") -> dict[str, object]:
    return {
        "orb_evoke_damage": {
            "amount": max(0, _coerce_int(amount)),
            "target": _normalized_id(target) or "enemies_hit",
        }
    }


def timed_orb_trigger(
    trigger: str,
    effects: Sequence[EffectStep],
    *,
    duration: TriggerDuration | str = "combat",
    condition: Mapping[str, object] | None = None,
    delay: int = 0,
    uses: int | None = None,
    counter_scope: str | None = None,
    every: int | None = None,
    text: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> TimedOrbTriggerDescriptor:
    return TimedOrbTriggerDescriptor(
        trigger=trigger,
        effects=tuple(effects),
        duration=duration,
        condition=condition or {},
        delay=delay,
        uses=uses,
        counter_scope=counter_scope,
        every=every,
        text=text,
        metadata=metadata or {},
    )


def timed_channel_orb_trigger(
    trigger: str,
    orb_id: str,
    amount: object = 1,
    *,
    duration: TriggerDuration | str = "combat",
    condition: Mapping[str, object] | None = None,
    uses: int | None = None,
    text: str | None = None,
) -> TimedOrbTriggerDescriptor:
    return timed_orb_trigger(
        trigger,
        (channel_orb_effect(orb_id, amount),),
        duration=duration,
        condition=condition or {},
        uses=uses,
        text=text,
    )


def timed_evoke_orb_trigger(
    trigger: str,
    selector: str = "leftmost",
    amount: object = 1,
    *,
    duration: TriggerDuration | str = "combat",
    uses: int | None = None,
    text: str | None = None,
) -> TimedOrbTriggerDescriptor:
    return timed_orb_trigger(
        trigger,
        (evoke_orb_effect(selector, amount),),
        duration=duration,
        uses=uses,
        text=text,
    )


def timed_passive_orb_trigger(
    trigger: str,
    selector: str = "rightmost",
    *,
    amount: int = 1,
    orb_id: str | None = None,
    duration: TriggerDuration | str = "combat",
    uses: int | None = None,
    text: str | None = None,
) -> TimedOrbTriggerDescriptor:
    return timed_orb_trigger(
        trigger,
        (trigger_orb_passive_effect(selector, amount=amount, orb_id=orb_id),),
        duration=duration,
        uses=uses,
        text=text,
    )


def storm_trigger(amount: int = 1) -> TimedOrbTriggerDescriptor:
    return timed_channel_orb_trigger(
        "card_played",
        "lightning",
        amount,
        condition={"card_type": "power"},
    )


def spinner_trigger(amount: int = 1) -> TimedOrbTriggerDescriptor:
    return timed_channel_orb_trigger("turn_start", "glass", amount)


def spinner_effects(*, upgraded: bool = False, amount: int = 1) -> tuple[dict[str, object], ...]:
    effects: list[dict[str, object]] = []
    if upgraded:
        effects.append(channel_orb_effect("glass", amount))
    effects.append(spinner_trigger(amount).as_effect())
    return tuple(effects)


def thunder_trigger(amount: int = 6) -> TimedOrbTriggerDescriptor:
    return timed_orb_trigger(
        "orb_evoked",
        (orb_evoke_damage_effect(amount),),
        condition={"orb": "lightning"},
    )


def lightning_rod_trigger(turns: int = 2, amount: int = 1) -> TimedOrbTriggerDescriptor:
    return timed_channel_orb_trigger(
        "turn_start",
        "lightning",
        amount,
        duration="uses",
        uses=max(0, _coerce_int(turns)),
    )


def consuming_shadow_trigger(amount: int = 1) -> TimedOrbTriggerDescriptor:
    return timed_evoke_orb_trigger("turn_end", "leftmost", amount)


def trash_to_treasure_trigger(amount: int = 1) -> TimedOrbTriggerDescriptor:
    return timed_channel_orb_trigger("status_created", "random_orb", amount)


def loop_passive_trigger(times: int = 1) -> PassiveOrbTriggerDescriptor:
    return PassiveOrbTriggerDescriptor(selector="rightmost", amount=times)


def darkness_passive_trigger(times: int = 2) -> PassiveOrbTriggerDescriptor:
    return PassiveOrbTriggerDescriptor(
        selector="matching",
        amount=times,
        orb_id="dark",
        direction="left_to_right",
    )


def passive_trigger_targets(
    orbs: Sequence[OrbInput] = (),
    *,
    selector: str = "rightmost",
    orb_id: str | None = None,
    direction: str = "left_to_right",
) -> tuple[int, ...]:
    snapshots = orb_snapshots(orbs)
    normalized_selector = _normalized_selector(selector)
    normalized_orb = normalize_orb_id(orb_id) if orb_id is not None else None
    candidates = tuple(
        index
        for index, snapshot in enumerate(snapshots)
        if normalized_orb is None or snapshot.orb_id == normalized_orb
    )
    if not candidates:
        return ()
    if normalized_selector == "leftmost":
        return (candidates[0],)
    if normalized_selector == "rightmost":
        return (candidates[-1],)
    if _normalized_id(direction) == "right_to_left":
        return tuple(reversed(candidates))
    return candidates


def _enemy_is_alive(enemy: Mapping[str, object]) -> bool:
    if "alive" in enemy:
        return _truthy(enemy.get("alive"))
    if "hp" in enemy:
        return _coerce_int(enemy.get("hp")) > 0
    return True


def _mapping_from(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _copy_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _copy_jsonish(item) for key, item in value.items()}


def _copy_jsonish(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _copy_jsonish(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_copy_jsonish(item) for item in value)
    return value


def _effect_amount_value(value: object) -> object:
    if isinstance(value, str):
        return _normalized_id(value) or value
    return max(0, _coerce_int(value, default=1))


def _normalized_selector(value: object) -> PassiveSelector:
    normalized = _normalized_id(value)
    aliases = {
        "left": "leftmost",
        "right": "rightmost",
        "all_dark": "matching",
        "dark": "matching",
        "match": "matching",
        "matches": "matching",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {"leftmost", "rightmost", "all", "matching"}:
        return cast(PassiveSelector, normalized)
    return "rightmost"


def _normalized_id(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _coerce_int(value: object = None, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = _normalized_id(value)
    return normalized not in {"", "0", "false", "none", "no", "off"}


def _clamp(amount: int, *, minimum: int, maximum: int | None) -> int:
    clamped = max(minimum, amount)
    if maximum is not None:
        clamped = min(maximum, clamped)
    return clamped


__all__ = [
    "ORB_TYPES",
    "EffectStep",
    "OrbInput",
    "OrbSnapshot",
    "PassiveOrbTriggerDescriptor",
    "PassiveSelector",
    "TimedOrbTriggerDescriptor",
    "TriggerDuration",
    "alive_enemy_count",
    "channel_orb_effect",
    "channeled_orb_count",
    "chill_channel_count",
    "compile_driver_draw",
    "consuming_shadow_trigger",
    "coolant_block",
    "count_orbs",
    "darkness_passive_trigger",
    "evoke_orb_effect",
    "lightning_rod_trigger",
    "loop_passive_trigger",
    "metadata_orb_channel_count",
    "normalize_orb_id",
    "open_orb_slots",
    "orb_evoke_damage_effect",
    "orb_ids",
    "orb_slot_channel_count",
    "orb_snapshots",
    "passive_trigger_targets",
    "scale_by_orb_count",
    "scale_by_unique_orbs",
    "scaled_amount_from_orbs",
    "spinner_effects",
    "spinner_trigger",
    "storm_trigger",
    "synchronize_focus",
    "tempest_channel_count",
    "thunder_trigger",
    "timed_channel_orb_trigger",
    "timed_evoke_orb_trigger",
    "timed_orb_trigger",
    "timed_passive_orb_trigger",
    "trash_to_treasure_trigger",
    "trigger_orb_passive_effect",
    "unique_orb_count",
    "unique_orb_types",
    "voltaic_channel_count",
]
