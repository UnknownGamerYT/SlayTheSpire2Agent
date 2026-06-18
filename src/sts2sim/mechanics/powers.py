"""Pure power and status helpers for combat effect execution.

The engine stores powers/statuses as simple ``dict[str, int]`` payloads.  This
module keeps the deterministic translation layer in one place: source-data
power names become stable status ids, status ids become combat math modifier
snapshots, and basic card effects can be adjusted without knowing about
``RunState`` or engine transition internals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .combat_math import (
    DEFAULT_COMBAT_RULES,
    CombatMathRules,
    ModifierSnapshot,
    modified_attack_damage,
)
from .combat_math import (
    block_gain as combat_block_gain,
)

StatusMap = Mapping[str, Any] | Sequence[Mapping[str, Any]] | None


@dataclass(frozen=True, slots=True)
class StatusApplicationResult:
    """Result of applying a deterministic status delta."""

    statuses: Mapping[str, int]
    events: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "statuses", dict(self.statuses))
        object.__setattr__(self, "events", tuple(dict(event) for event in self.events))


@dataclass(frozen=True, slots=True)
class PowerModifiedEffect:
    """A normalized effect step after power/status modifiers are applied."""

    effect: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect", dict(self.effect))
        object.__setattr__(self, "events", tuple(dict(event) for event in self.events))


def normalize_power_id(value: object) -> str:
    """Return a stable snake-case status id from a power/status source value."""

    if isinstance(value, Mapping):
        for key in ("power_key", "power", "status", "name", "id"):
            raw = value.get(key)
            if raw not in (None, ""):
                return _canonical_status_id(raw)
        return ""
    return _canonical_status_id(value)


def normalize_statuses(statuses: StatusMap) -> dict[str, int]:
    """Normalize status/power dictionaries or lists into ``dict[str, int]``."""

    normalized: dict[str, int] = {}
    if statuses is None:
        return normalized

    if isinstance(statuses, Sequence) and not isinstance(statuses, (str, bytes, bytearray)):
        for item in statuses:
            for status_id, amount in normalize_statuses(item).items():
                normalized[status_id] = normalized.get(status_id, 0) + amount
        return {key: value for key, value in normalized.items() if value != 0}

    if not isinstance(statuses, Mapping):
        return normalized

    nested = statuses.get("statuses")
    if isinstance(nested, Mapping | Sequence) and not isinstance(nested, (str, bytes, bytearray)):
        for status_id, amount in normalize_statuses(nested).items():
            normalized[status_id] = normalized.get(status_id, 0) + amount

    if _looks_like_single_power(statuses):
        status_id = normalize_power_id(statuses)
        if status_id:
            amount = _coerce_int(
                statuses.get("amount", statuses.get("value", statuses.get("stacks", 1))),
                default=1,
            )
            normalized[status_id] = normalized.get(status_id, 0) + amount
        return {key: value for key, value in normalized.items() if value != 0}

    for raw_key, raw_value in statuses.items():
        key = str(raw_key)
        if key in _STATUS_METADATA_KEYS:
            continue
        status_amount = _coerce_status_amount(raw_value)
        if status_amount is None:
            continue
        status_id = normalize_power_id(key)
        if status_id:
            normalized[status_id] = normalized.get(status_id, 0) + status_amount

    return {key: value for key, value in normalized.items() if value != 0}


def modifier_snapshot_from_statuses(statuses: StatusMap) -> ModifierSnapshot:
    """Build the combat math modifier snapshot represented by status stacks."""

    normalized = normalize_statuses(statuses)
    return ModifierSnapshot(
        strength=normalized.get("strength", 0) + normalized.get("temporary_strength", 0),
        dexterity=normalized.get("dexterity", 0) + normalized.get("temporary_dexterity", 0),
        weak=normalized.get("weak", 0) > 0,
        vulnerable=normalized.get("vulnerable", 0) > 0,
        frail=normalized.get("frail", 0) > 0,
        intangible=normalized.get("intangible", 0) > 0,
    )


def modified_card_cost(
    card_or_cost: Mapping[str, Any] | int | str | None,
    statuses: StatusMap = None,
    *,
    card_type: str | None = None,
    available_energy: int | None = None,
) -> int:
    """Return a card's executable energy cost after simple cost modifiers."""

    base_cost = _base_card_cost(card_or_cost)
    if base_cost is None:
        return 0
    if base_cost < 0:
        return max(0, int(available_energy)) if available_energy is not None else -1

    normalized = normalize_statuses(statuses)
    effective_type = _card_type_from(card_or_cost, card_type)
    if normalized.get("next_card_free", 0) > 0:
        return 0

    reduction = (
        normalized.get("cost_reduction", 0)
        + normalized.get("card_cost_reduction", 0)
        + normalized.get("all_cost_reduction", 0)
        + normalized.get("temporary_cost_reduction", 0)
    )
    if effective_type:
        reduction += normalized.get(f"{effective_type}_cost_reduction", 0)
    return max(0, base_cost - max(0, reduction))


def apply_power_modifiers_to_effect(
    effect: Mapping[str, Any],
    *,
    actor_statuses: StatusMap = None,
    defender_statuses: StatusMap = None,
    card_type: str = "attack",
    source_id: str | None = None,
    rules: CombatMathRules = DEFAULT_COMBAT_RULES,
) -> PowerModifiedEffect:
    """Apply strength/dexterity-like status modifiers to one effect step."""

    updated = dict(effect)
    events: list[Mapping[str, Any]] = []
    attacker = modifier_snapshot_from_statuses(actor_statuses)
    defender = modifier_snapshot_from_statuses(defender_statuses)
    is_attack = _normalized_id(card_type) == "attack"

    for key in ("damage", "all_damage"):
        base = _numeric_effect_amount(updated.get(key))
        if base is None:
            continue
        modified = modified_attack_damage(
            base,
            attacker=attacker,
            defender=defender,
            is_attack=is_attack,
            rules=rules,
        )
        updated[key] = modified
        if modified != base:
            events.append(_modifier_event(key, base, modified, source_id))

    base_block = _numeric_effect_amount(updated.get("block"))
    if base_block is not None:
        modified_block = combat_block_gain(base_block, actor=attacker, rules=rules).block
        updated["block"] = modified_block
        if modified_block != base_block:
            events.append(_modifier_event("block", base_block, modified_block, source_id))

    return PowerModifiedEffect(effect=updated, events=tuple(events))


def apply_status_delta(
    statuses: StatusMap,
    delta: StatusMap,
    *,
    source_id: str | None = None,
    target_id: str | None = None,
) -> StatusApplicationResult:
    """Apply normalized status deltas and emit deterministic status events."""

    next_statuses = normalize_statuses(statuses)
    deltas = normalize_statuses(delta)
    events: list[Mapping[str, Any]] = []
    for status_id in sorted(deltas):
        amount = deltas[status_id]
        if amount == 0:
            continue
        next_amount = next_statuses.get(status_id, 0) + amount
        if next_amount == 0:
            next_statuses.pop(status_id, None)
        else:
            next_statuses[status_id] = next_amount
        events.append(
            {
                "kind": "status_applied",
                "source_id": source_id,
                "target_id": target_id,
                "amount": amount,
                "metadata": {"status": status_id},
            }
        )
    return StatusApplicationResult(
        statuses=dict(sorted(next_statuses.items())),
        events=tuple(events),
    )


def power_application_effect(
    application: Mapping[str, Any],
    *,
    default_target: str = "self",
) -> dict[str, Any]:
    """Convert a source-data power application into an engine status step."""

    status_values = normalize_statuses(application)
    target = _normalize_target(application.get("target", default_target), default_target)
    if not status_values:
        return {}
    return {"apply_status": {"target": target, **status_values}}


def end_of_combat_status_events(
    statuses: StatusMap,
    *,
    source_id: str | None = None,
    target_id: str | None = "player",
) -> tuple[Mapping[str, Any], ...]:
    """Return deferred end-of-combat marker events from status stacks."""

    normalized = normalize_statuses(statuses)
    events: list[Mapping[str, Any]] = []
    for status_id, event_kind in _END_OF_COMBAT_MARKERS.items():
        amount = normalized.get(status_id, 0)
        if amount <= 0:
            continue
        events.append(
            {
                "kind": event_kind,
                "source_id": source_id,
                "target_id": target_id,
                "amount": amount,
                "metadata": {"status": status_id},
            }
        )
    return tuple(events)


def _looks_like_single_power(statuses: Mapping[str, Any]) -> bool:
    return any(key in statuses for key in ("power_key", "power", "status")) or (
        "amount" in statuses and any(key in statuses for key in ("name", "id"))
    ) or (
        "name" in statuses and "type" in statuses and "target" not in statuses
    )


def _canonical_status_id(value: object) -> str:
    key = _normalized_id(value)
    if key.endswith("_power"):
        key = key[: -len("_power")]
    elif key.endswith("power") and len(key) > len("power"):
        key = key[: -len("power")]
    return _STATUS_ALIASES.get(key, key)


def _normalized_id(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("'", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _normalize_target(value: object, default: str) -> str:
    key = _normalized_id(value)
    return _TARGET_ALIASES.get(key, _TARGET_ALIASES.get(_normalized_id(default), "self"))


def _coerce_int(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_status_amount(value: Any) -> int | None:
    if isinstance(value, Mapping):
        return _coerce_int(value.get("amount", value.get("value", value.get("stacks", 1))))
    if isinstance(value, bool | int | float | str):
        return _coerce_int(value)
    return None


def _base_card_cost(card_or_cost: Mapping[str, Any] | int | str | None) -> int | None:
    if card_or_cost is None:
        return 0
    if isinstance(card_or_cost, Mapping):
        if card_or_cost.get("is_x_cost") or card_or_cost.get("is_x_star_cost"):
            return -1
        value = card_or_cost.get("cost", card_or_cost.get("energy_cost", 1))
    else:
        value = card_or_cost

    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "x":
        return -1
    return _coerce_int(value, default=0)


def _card_type_from(
    card_or_cost: Mapping[str, Any] | int | str | None,
    card_type: str | None,
) -> str:
    if card_type is not None:
        return _CARD_TYPE_ALIASES.get(_normalized_id(card_type), _normalized_id(card_type))
    if isinstance(card_or_cost, Mapping):
        raw = card_or_cost.get("type", card_or_cost.get("card_type", ""))
        return _CARD_TYPE_ALIASES.get(_normalized_id(raw), _normalized_id(raw))
    return ""


def _numeric_effect_amount(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    return None


def _modifier_event(
    field: str,
    base: int,
    modified: int,
    source_id: str | None,
) -> Mapping[str, Any]:
    return {
        "kind": "effect_power_modified",
        "source_id": source_id,
        "target_id": None,
        "amount": modified - base,
        "metadata": {"field": field, "base": base, "modified": modified},
    }


_STATUS_METADATA_KEYS = frozenset(
    {
        "amount",
        "description",
        "id",
        "name",
        "power",
        "power_key",
        "source",
        "status",
        "target",
        "type",
        "value",
    }
)

_STATUS_ALIASES = {
    "str": "strength",
    "strengthpower": "strength",
    "strength_power": "strength",
    "dex": "dexterity",
    "dexteritypower": "dexterity",
    "dexterity_power": "dexterity",
    "temporary_strength_power": "temporary_strength",
    "temporary_strengthpower": "temporary_strength",
    "temp_strength": "temporary_strength",
    "temporary_dexterity_power": "temporary_dexterity",
    "temporary_dexteritypower": "temporary_dexterity",
    "temp_dexterity": "temporary_dexterity",
    "weakpower": "weak",
    "weak_power": "weak",
    "vuln": "vulnerable",
    "vulnerablepower": "vulnerable",
    "vulnerable_power": "vulnerable",
    "frailpower": "frail",
    "frail_power": "frail",
    "intangiblepower": "intangible",
    "intangible_power": "intangible",
    "thorns_power": "thorns",
    "thornspower": "thorns",
    "powers_cost_reduction": "power_cost_reduction",
    "power_discount": "power_cost_reduction",
    "card_discount": "card_cost_reduction",
    "upgrade_random_end_of_combat": "end_of_combat_upgrade_random",
    "end_of_combat_random_upgrade": "end_of_combat_upgrade_random",
}

_TARGET_ALIASES = {
    "all_enemies": "all_enemies",
    "allenemies": "all_enemies",
    "any_enemy": "enemy",
    "anyenemy": "enemy",
    "enemy": "enemy",
    "random_enemy": "enemy",
    "randomenemy": "enemy",
    "self": "self",
    "player": "self",
}

_CARD_TYPE_ALIASES = {
    "attack": "attack",
    "skill": "skill",
    "power": "power",
    "status": "status",
    "curse": "curse",
}

_END_OF_COMBAT_MARKERS = {
    "end_of_combat_upgrade_random": "card_upgrade_random_pending",
    "end_of_combat_upgrade_all": "card_upgrade_all_pending",
}


__all__ = [
    "PowerModifiedEffect",
    "StatusApplicationResult",
    "apply_power_modifiers_to_effect",
    "apply_status_delta",
    "end_of_combat_status_events",
    "modified_card_cost",
    "modifier_snapshot_from_statuses",
    "normalize_power_id",
    "normalize_statuses",
    "power_application_effect",
]
