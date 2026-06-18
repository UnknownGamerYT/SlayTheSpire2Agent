"""Pure card effect normalization helpers.

These helpers turn source-data card dictionaries into bounded, deterministic
effect steps.  The emitted steps are ordinary mappings so the engine can either
execute the currently-supported primitives directly or add handling for the
small extension markers later.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .card_specials import card_special_plan
from .powers import normalize_power_id, power_application_effect

ENGINE_EFFECT_KEYS = frozenset(
    {
        "add_random_card_to_hand",
        "all_damage",
        "apply_status",
        "block",
        "damage",
        "destination",
        "draw",
        "energy",
        "exhaust_on_play",
        "hp_loss",
        "heal",
        "next_turn",
        "player_resource",
        "retain_hand",
        "sequence",
        "status",
    }
)
EXTENDED_EFFECT_KEYS = frozenset(
    {
        "add_card_to_discard",
        "add_card_to_draw",
        "add_card_to_exhaust",
        "add_card_to_hand",
        "channel_orb",
        "discard_choice",
        "discard_hand",
        "evoke_orb",
        "discard_random",
        "exhaust_random",
        "orb_slot_delta",
    }
)
EXECUTABLE_EFFECT_KEYS = ENGINE_EFFECT_KEYS | EXTENDED_EFFECT_KEYS


@dataclass(frozen=True, slots=True)
class CardEffectPlan:
    """A normalized card plus the effect steps/events used to build it."""

    card_id: str
    card: Mapping[str, Any]
    steps: tuple[Mapping[str, Any], ...]
    events: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "card_id", str(self.card_id))
        object.__setattr__(self, "card", _clone_jsonish_mapping(self.card))
        object.__setattr__(
            self,
            "steps",
            tuple(_clone_jsonish_mapping(step) for step in self.steps),
        )
        object.__setattr__(
            self,
            "events",
            tuple(_clone_jsonish_mapping(event) for event in self.events),
        )


def card_effect_plan(
    card_spec: Mapping[str, Any],
    *,
    card_library: Mapping[str, Mapping[str, Any]] | None = None,
) -> CardEffectPlan:
    """Normalize one card source mapping into executable effect steps."""

    card_id = _card_id(card_spec)
    card_type = _normalize_card_type(card_spec.get("type", card_spec.get("card_type")))
    target = _normalize_target(card_spec.get("target"), card_type=card_type)
    steps = _steps_from_card_spec(
        card_spec,
        card_id=card_id,
        card_type=card_type,
        target=target,
        card_library=card_library,
    )
    special_plan = card_special_plan(card_spec)
    steps = _merged_special_steps(steps, special_plan.steps)
    card = {
        "card_id": card_id,
        "name": str(card_spec.get("name", card_id)),
        "type": card_type,
        "cost": _card_cost(card_spec),
        "target": target,
        "effects": effect_sequence_mapping(steps),
        "tags": _string_tuple(card_spec.get("tags", ())),
        "exhausts": bool(card_spec.get("exhausts", card_spec.get("exhaust", False))),
        "upgraded": bool(card_spec.get("upgraded", False)),
        "custom": dict(card_spec.get("custom", {}))
        if isinstance(card_spec.get("custom"), Mapping)
        else {},
    }
    events = (
        {
            "kind": "card_effects_normalized",
            "source_id": card_id,
            "target_id": None,
            "amount": len(steps),
            "metadata": {"card_type": card_type, "target": target},
        },
    ) + special_plan.events
    return CardEffectPlan(card_id=card_id, card=card, steps=steps, events=events)


def normalize_card_spec(
    card_spec: Mapping[str, Any],
    *,
    card_library: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return only the normalized card mapping for a source card."""

    return dict(card_effect_plan(card_spec, card_library=card_library).card)


def normalize_card_effect_steps(
    source: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    card_type: str = "unknown",
    target: str | None = None,
    card_id: str | None = None,
    card_library: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Normalize a card or raw effects payload into effect-step mappings."""

    if isinstance(source, Mapping) and _looks_like_card_spec(source):
        return card_effect_plan(source, card_library=card_library).steps
    normalized_type = _normalize_card_type(card_type)
    normalized_target = _normalize_target(target, card_type=normalized_type)
    return tuple(
        _steps_from_source(
            source,
            card_id=card_id,
            card_type=normalized_type,
            target=normalized_target,
            card_library=card_library,
        )
    )


def effect_sequence_mapping(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Wrap effect steps in the engine-supported ``sequence`` shape."""

    return {"sequence": [_clone_jsonish_mapping(step) for step in steps]}


def temporary_card_spec(
    card_spec: Mapping[str, Any] | str,
    *,
    card_library: Mapping[str, Mapping[str, Any]] | None = None,
    cost_for_turn: int | None = 0,
    destination: str = "hand",
) -> dict[str, Any]:
    """Return a normalized generated/temporary card descriptor."""

    source = _generated_card_source(card_spec, card_library)
    card = normalize_card_spec(source, card_library=card_library)
    if cost_for_turn is not None:
        card["cost"] = max(0, int(cost_for_turn))
    custom = dict(card.get("custom", {}))
    custom.update(
        {
            "generated": True,
            "temporary": True,
            "destination": _normalize_destination(destination),
        }
    )
    card["custom"] = custom
    return card


def _steps_from_card_spec(
    card_spec: Mapping[str, Any],
    *,
    card_id: str,
    card_type: str,
    target: str,
    card_library: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[Mapping[str, Any], ...]:
    steps: list[Mapping[str, Any]] = []
    explicit = card_spec.get("effects", card_spec.get("effect"))
    source_fields_handled = False
    if isinstance(explicit, str):
        named_steps = _steps_from_named_effect(
            explicit,
            card_spec,
            card_id=card_id,
            card_type=card_type,
            target=target,
            card_library=card_library,
        )
        source_fields_handled = bool(named_steps)
        steps.extend(named_steps)
    elif explicit is not None:
        steps.extend(
            _steps_from_source(
                explicit,
                card_id=card_id,
                card_type=card_type,
                target=target,
                card_library=card_library,
            )
        )
    if not source_fields_handled:
        steps.extend(
            _steps_from_source_fields(
                card_spec,
                card_id=card_id,
                card_type=card_type,
                target=target,
                card_library=card_library,
            )
        )
    if bool(card_spec.get("exhaust_on_play")):
        steps.append({"exhaust_on_play": True})
    destination = card_spec.get("destination")
    if destination is not None:
        steps.append({"destination": _normalize_destination(destination)})
    return tuple(steps)


def _merged_special_steps(
    steps: Sequence[Mapping[str, Any]],
    special_steps: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    merged = list(steps)
    for step in special_steps:
        if any(_equivalent_effect_step(step, existing) for existing in merged):
            continue
        merged.append(step)
    return tuple(merged)


def _equivalent_effect_step(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    if left == right:
        return True
    left_resource = _resource_step_signature(left)
    return left_resource is not None and left_resource == _resource_step_signature(right)


def _resource_step_signature(step: Mapping[str, Any]) -> tuple[str, int] | None:
    payload = step.get("player_resource")
    if not isinstance(payload, Mapping):
        return None
    resource = _normalized_id(payload.get("resource", payload.get("name", "")))
    amount = _amount_from(payload, "amount")
    if not resource or amount is None:
        return None
    return resource, int(amount)


def _steps_from_source(
    source: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    card_id: str | None,
    card_type: str,
    target: str,
    card_library: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray, Mapping)):
        steps: list[Mapping[str, Any]] = []
        for item in source:
            if isinstance(item, Mapping):
                steps.extend(
                    _steps_from_source(
                        item,
                        card_id=card_id,
                        card_type=card_type,
                        target=target,
                        card_library=card_library,
                    )
                )
        return tuple(steps)

    if not isinstance(source, Mapping):
        return ()

    sequence = source.get("sequence", source.get("effects"))
    if isinstance(sequence, Sequence) and not isinstance(sequence, (str, bytes, bytearray)):
        steps = []
        for item in sequence:
            if isinstance(item, Mapping):
                steps.extend(
                    _steps_from_source(
                        item,
                        card_id=card_id,
                        card_type=card_type,
                        target=target,
                        card_library=card_library,
                    )
                )
        return tuple(steps)
    if isinstance(sequence, Mapping):
        return _steps_from_source(
            sequence,
            card_id=card_id,
            card_type=card_type,
            target=target,
            card_library=card_library,
        )

    named_effect = source.get("effect")
    if isinstance(named_effect, str):
        named_steps = _steps_from_named_effect(
            named_effect,
            source,
            card_id=card_id,
            card_type=card_type,
            target=target,
            card_library=card_library,
        )
        if named_steps:
            return named_steps

    return _steps_from_source_fields(
        source,
        card_id=card_id,
        card_type=card_type,
        target=target,
        card_library=card_library,
    )


def _steps_from_source_fields(
    source: Mapping[str, Any],
    *,
    card_id: str | None,
    card_type: str,
    target: str,
    card_library: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[Mapping[str, Any], ...]:
    steps: list[Mapping[str, Any]] = []
    damage_key = "all_damage" if target == "all_enemies" or "all_damage" in source else "damage"
    damage_amount = _amount_from(source, "all_damage" if "all_damage" in source else "damage")
    if damage_amount not in (None, 0):
        for _ in range(_hit_count(source)):
            steps.append({damage_key: damage_amount})

    for source_key, effect_key in (
        ("block", "block"),
        ("energy", "energy"),
        ("energy_gain", "energy"),
        ("draw", "draw"),
        ("cards_draw", "draw"),
        ("heal", "heal"),
        ("hp_loss", "hp_loss"),
    ):
        amount = _amount_from(source, source_key)
        if amount not in (None, 0):
            steps.append({effect_key: amount})

    random_cards = _amount_from(source, "add_random_card_to_hand")
    if random_cards in (None, 0):
        random_cards = _amount_from(source, "random_card_count")
    if random_cards not in (None, 0):
        steps.append({"add_random_card_to_hand": random_cards})

    steps.extend(_status_steps_from_source(source, card_type=card_type, target=target))
    steps.extend(_description_steps(source, target=target))
    steps.extend(
        _generated_card_steps(
            source,
            card_id=card_id,
            card_library=card_library,
        )
    )
    steps.extend(_passthrough_effect_steps(source))
    return tuple(steps)


def _steps_from_named_effect(
    effect: str,
    source: Mapping[str, Any],
    *,
    card_id: str | None,
    card_type: str,
    target: str,
    card_library: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[Mapping[str, Any], ...]:
    key = _normalized_id(effect)
    if key == "apply_debuffs":
        return (
            {
                "apply_status": {
                    "target": "enemy",
                    "weak": _amount_from(source, "weak", default=2),
                    "vulnerable": _amount_from(source, "vulnerable", default=2),
                }
            },
        )
    if key == "gain_strength_and_dexterity":
        return (
            {
                "apply_status": {
                    "target": "self",
                    "strength": _amount_from(source, "strength", default=2),
                    "dexterity": _amount_from(source, "dexterity", default=2),
                }
            },
        )
    if key == "gain_energy":
        return ({"energy": _amount_from(source, "energy", default=1)},)
    if key == "draw_cards":
        return ({"draw": _amount_from(source, "draw", default=1)},)
    if key == "add_random_free_card_to_hand":
        return (
            {"add_random_card_to_hand": _amount_from(source, "random_card_count", default=1)},
        )
    if key == "power_cost_reduction":
        return (
            {
                "apply_status": {
                    "target": "self",
                    "power_cost_reduction": _amount_from(source, "amount", default=1),
                }
            },
        )
    if key == "end_of_combat_upgrade_random":
        return (
            {
                "apply_status": {
                    "target": "self",
                    "end_of_combat_upgrade_random": _amount_from(
                        source,
                        "upgrade_random_count",
                        default=_amount_from(source, "amount", default=1),
                    ),
                }
            },
        )
    if key in {"create_card", "add_card_to_hand"}:
        return _generated_card_steps(source, card_id=card_id, card_library=card_library)
    if key == "additional_hits":
        return _steps_from_source_fields(
            source,
            card_id=card_id,
            card_type=card_type,
            target=target,
            card_library=card_library,
        )
    return ()


def _status_steps_from_source(
    source: Mapping[str, Any],
    *,
    card_type: str,
    target: str,
) -> tuple[Mapping[str, Any], ...]:
    steps: list[Mapping[str, Any]] = []
    for status_key in ("apply_status", "status", "statuses"):
        payload = source.get(status_key)
        if payload is None:
            continue
        steps.extend(_status_payload_steps(payload, default_target=_default_status_target(target)))

    powers = source.get("powers_applied")
    if isinstance(powers, Sequence) and not isinstance(powers, (str, bytes, bytearray)):
        for power in powers:
            if not isinstance(power, Mapping):
                continue
            status_id = normalize_power_id(power)
            default_target = _default_power_target(status_id, card_type=card_type, target=target)
            effect = power_application_effect(power, default_target=default_target)
            if effect:
                steps.append(effect)

    direct_statuses: dict[str, Any] = {}
    for raw_key, value in source.items():
        status_id = normalize_power_id(raw_key)
        if status_id in _DIRECT_STATUS_IDS and value not in (None, 0):
            direct_statuses[status_id] = value
    if direct_statuses:
        default_target = "enemy" if _ENEMY_STATUS_IDS & set(direct_statuses) else "self"
        effect = power_application_effect(
            {"target": default_target, **direct_statuses},
            default_target=default_target,
        )
        if effect:
            steps.append(effect)
    return tuple(steps)


def _status_payload_steps(
    payload: Any,
    *,
    default_target: str,
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray, Mapping)):
        steps: list[Mapping[str, Any]] = []
        for item in payload:
            if isinstance(item, Mapping):
                steps.extend(_status_payload_steps(item, default_target=default_target))
        return tuple(steps)
    if not isinstance(payload, Mapping):
        return ()
    effect = power_application_effect(payload, default_target=default_target)
    return (effect,) if effect else ()


def _generated_card_steps(
    source: Mapping[str, Any],
    *,
    card_id: str | None,
    card_library: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[Mapping[str, Any], ...]:
    generated = source.get(
        "spawns_cards",
        source.get("generated_cards", source.get("created_cards", source.get("cards"))),
    )
    if generated is None and ("card" in source or "card_id" in source):
        generated = source.get("card", source.get("card_id"))
    if generated is None:
        return ()
    if isinstance(generated, (str, Mapping)):
        generated_items: Sequence[Any] = (generated,)
    elif isinstance(generated, Sequence):
        generated_items = generated
    else:
        return ()

    steps: list[Mapping[str, Any]] = []
    destination = _generated_card_destination(source)
    count = _generated_card_count(source)
    for item in generated_items:
        if item in (None, ""):
            continue
        for _ in range(max(1, count)):
            card = temporary_card_spec(item, card_library=card_library, destination=destination)
            steps.append(
                {
                    f"add_card_to_{destination}": {
                        "card": card,
                        "destination": destination,
                        "temporary": True,
                        "source_id": card_id,
                    }
                }
            )
    return tuple(steps)


def _passthrough_effect_steps(source: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {effect_key: _clone_jsonish(source[effect_key])}
        for effect_key in ("channel_orb", "evoke_orb", "orb_slot_delta")
        if effect_key in source
    )


def _description_steps(source: Mapping[str, Any], *, target: str) -> tuple[Mapping[str, Any], ...]:
    description = str(source.get("description", "") or "")
    if not description:
        return ()
    normalized = _normalized_description(description)
    steps: list[Mapping[str, Any]] = []

    for resource_name, amount in re.findall(r"gain\s+\[(star|energy):(\d+)\]", normalized):
        if resource_name == "energy":
            if not _has_amount_step(source, "energy"):
                steps.append({"energy": int(amount)})
        else:
            steps.append({"player_resource": {"resource": resource_name, "amount": int(amount)}})

    next_turn: dict[str, int] = {}
    for amount in re.findall(r"next turn,?\s*gain\s+\[energy:(\d+)\]", normalized):
        next_turn["energy"] = next_turn.get("energy", 0) + int(amount)
    for amount in re.findall(r"next turn,?\s*gain\s+\[star:(\d+)\]", normalized):
        next_turn["star"] = next_turn.get("star", 0) + int(amount)
    for amount in re.findall(r"next turn,?\s*draw\s+(\d+)", normalized):
        next_turn["draw"] = next_turn.get("draw", 0) + int(amount)
    for amount in re.findall(r"next turn,?\s*gain\s+(\d+)\s+\[gold\]block", normalized):
        next_turn["block"] = next_turn.get("block", 0) + int(amount)
    if next_turn:
        steps.append({"next_turn": next_turn})

    if "retain your [gold]hand" in normalized:
        steps.append({"retain_hand": True})

    if "discard your [gold]hand" in normalized or "discard your hand" in normalized:
        steps.append({"discard_hand": {"mode": "all"}})
    else:
        random_discard_match = re.search(
            r"discard\s+(?:(\d+)|a|an|one)?\s*random\s+cards?",
            normalized,
        )
        if random_discard_match:
            steps.append({"discard_random": int(random_discard_match.group(1) or 1)})
        discard_match = re.search(r"discard\s+(?:(\d+)|a|an|one)\s+cards?", normalized)
        if discard_match:
            steps.append({"discard_choice": int(discard_match.group(1) or 1)})

    exhaust_match = re.search(r"exhaust\s+(\d+)\s+card", normalized)
    if exhaust_match:
        steps.append({"exhaust_random": int(exhaust_match.group(1))})

    for amount in re.findall(r"\[gold\]forge\[/gold\]\s+(\d+)", normalized):
        steps.append({"player_resource": {"resource": "forge", "amount": int(amount)}})
    for amount in re.findall(r"\[gold\]summon\[/gold\]\s+(\d+)", normalized):
        steps.append({"player_resource": {"resource": "summon", "amount": int(amount)}})

    if "enemy loses" in normalized and target in {"enemy", "all_enemies"}:
        loss_match = re.search(r"enemy loses\s+(\d+)\s+hp", normalized)
        if loss_match:
            key = "all_damage" if target == "all_enemies" else "damage"
            steps.append({key: int(loss_match.group(1))})

    return tuple(steps)


def _generated_card_destination(source: Mapping[str, Any]) -> str:
    explicit = source.get("destination")
    if explicit is not None:
        return _normalize_destination(explicit)
    description = _normalized_description(str(source.get("description", "") or ""))
    if "discard pile" in description:
        return "discard"
    if "draw pile" in description:
        return "draw"
    return "hand"


def _generated_card_count(source: Mapping[str, Any]) -> int:
    description = _normalized_description(str(source.get("description", "") or ""))
    match = re.search(r"add\s+(\d+)\s+", description)
    if match:
        return max(1, int(match.group(1)))
    if "add a " in description or "add an " in description or "add 1 " in description:
        return 1
    return 1


def _has_amount_step(source: Mapping[str, Any], key: str) -> bool:
    direct = _amount_from(source, key)
    gain = _amount_from(source, f"{key}_gain")
    return direct not in (None, 0) or gain not in (None, 0)


def _normalized_description(description: str) -> str:
    return " ".join(description.replace("\n", " ").lower().split())


def _generated_card_source(
    card_spec: Mapping[str, Any] | str,
    card_library: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    if isinstance(card_spec, Mapping):
        if "card" in card_spec and isinstance(card_spec["card"], Mapping):
            return card_spec["card"]
        card_id = str(card_spec.get("card_id", card_spec.get("id", "")))
        merged = dict(_lookup_card(card_id, card_library))
        merged.update(card_spec)
        return merged
    card_id = str(card_spec)
    return _lookup_card(card_id, card_library) or {"id": card_id, "name": card_id, "cost": 0}


def _lookup_card(
    card_id: str,
    card_library: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    if not card_library:
        return {}
    candidates = (card_id, _normalized_id(card_id), str(card_id).upper())
    for candidate in candidates:
        found = card_library.get(candidate)
        if isinstance(found, Mapping):
            return found
    normalized = _normalized_id(card_id)
    for key, value in card_library.items():
        if _normalized_id(key) == normalized and isinstance(value, Mapping):
            return value
    return {}


def _card_id(card_spec: Mapping[str, Any]) -> str:
    return _normalized_id(card_spec.get("card_id", card_spec.get("id", "unknown_card")))


def _card_cost(card_spec: Mapping[str, Any]) -> int | None:
    if card_spec.get("is_x_cost") or card_spec.get("is_x_star_cost"):
        return -1
    if "cost" not in card_spec:
        return 1
    value = card_spec.get("cost")
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "x":
        return -1
    return int(value)


def _normalize_card_type(value: object) -> str:
    return _CARD_TYPE_ALIASES.get(_normalized_id(value), "unknown")


def _normalize_target(value: object, *, card_type: str) -> str:
    key = _normalized_id(value)
    if not key:
        if card_type in {"skill", "power"}:
            return "self"
        if card_type == "attack":
            return "enemy"
        return "none"
    return _TARGET_ALIASES.get(key, "enemy" if card_type == "attack" else "none")


def _normalize_destination(value: object) -> str:
    key = _normalized_id(value)
    return _DESTINATION_ALIASES.get(key, "hand")


def _amount_from(source: Mapping[str, Any], key: str, *, default: int | None = None) -> int | None:
    if key not in source:
        return default
    value = source.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, Mapping):
        raw = value.get("amount", value.get("value", value.get("base", default)))
        return int(raw) if raw is not None else default
    return int(value)


def _hit_count(source: Mapping[str, Any]) -> int:
    for key in ("hit_count", "hits", "times", "repeat", "count"):
        amount = _amount_from(source, key)
        if amount is not None:
            return max(1, amount)
    return 1


def _default_status_target(target: str) -> str:
    if target == "all_enemies":
        return "all_enemies"
    if target == "enemy":
        return "enemy"
    return "self"


def _default_power_target(status_id: str, *, card_type: str, target: str) -> str:
    if status_id in _SELF_STATUS_IDS:
        return "self"
    if status_id in _ENEMY_STATUS_IDS:
        return _default_status_target(target)
    if card_type == "power":
        return "self"
    return _default_status_target(target)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return ()


def _looks_like_card_spec(source: Mapping[str, Any]) -> bool:
    return any(key in source for key in ("id", "card_id", "name", "type", "card_type")) and any(
        key in source
        for key in (
            "block",
            "cards_draw",
            "damage",
            "effects",
            "energy_gain",
            "hit_count",
            "powers_applied",
            "spawns_cards",
        )
    )


def _clone_jsonish_mapping(source: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _clone_jsonish(value) for key, value in source.items()}


def _clone_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _clone_jsonish_mapping(value)
    if isinstance(value, tuple):
        return tuple(_clone_jsonish(item) for item in value)
    if isinstance(value, list):
        return [_clone_jsonish(item) for item in value]
    return value


def _normalized_id(value: object) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("'", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


_CARD_TYPE_ALIASES = {
    "attack": "attack",
    "curse": "curse",
    "power": "power",
    "skill": "skill",
    "status": "status",
}

_TARGET_ALIASES = {
    "all_enemies": "all_enemies",
    "allenemies": "all_enemies",
    "any": "any",
    "any_ally": "self",
    "anyally": "self",
    "any_enemy": "enemy",
    "anyenemy": "enemy",
    "enemy": "enemy",
    "none": "none",
    "random_enemy": "enemy",
    "randomenemy": "enemy",
    "self": "self",
}

_DESTINATION_ALIASES = {
    "discard": "discard",
    "discard_pile": "discard",
    "draw": "draw",
    "draw_pile": "draw",
    "exhaust": "exhaust",
    "exhaust_pile": "exhaust",
    "hand": "hand",
    "none": "none",
}

_SELF_STATUS_IDS = frozenset(
    {
        "accuracy",
        "afterimage",
        "artifact",
        "dexterity",
        "end_of_combat_upgrade_all",
        "end_of_combat_upgrade_random",
        "intangible",
        "metallicize",
        "plated_armor",
        "power_cost_reduction",
        "strength",
        "temporary_dexterity",
        "temporary_strength",
        "thorns",
    }
)

_ENEMY_STATUS_IDS = frozenset(
    {
        "choking",
        "frail",
        "poison",
        "slow",
        "vulnerable",
        "weak",
    }
)

_DIRECT_STATUS_IDS = _SELF_STATUS_IDS | _ENEMY_STATUS_IDS | frozenset(
    {
        "all_cost_reduction",
        "attack_cost_reduction",
        "card_cost_reduction",
        "cost_reduction",
        "skill_cost_reduction",
    }
)


__all__ = [
    "ENGINE_EFFECT_KEYS",
    "EXECUTABLE_EFFECT_KEYS",
    "EXTENDED_EFFECT_KEYS",
    "CardEffectPlan",
    "card_effect_plan",
    "effect_sequence_mapping",
    "normalize_card_effect_steps",
    "normalize_card_spec",
    "temporary_card_spec",
]
