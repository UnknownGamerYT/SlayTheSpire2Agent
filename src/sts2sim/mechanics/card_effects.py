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
        "add_random_potion",
        "all_damage",
        "apply_status",
        "block",
        "block_formula",
        "damage",
        "damage_formula",
        "destination",
        "draw",
        "draw_formula",
        "energy",
        "energy_formula",
        "end_turn_hand_effect",
        "exhaust_on_play",
        "force_play_priority",
        "hp_loss",
        "heal",
        "next_card_extra_play",
        "next_turn",
        "noop",
        "player_resource",
        "retain_hand",
        "remove_block",
        "remove_status",
        "sequence",
        "set_hand_free_to_play_this_turn",
        "set_hand_cost",
        "self_cost_delta",
        "status",
        "status_formula",
        "upgrade_all_combat_cards",
    }
)
EXTENDED_EFFECT_KEYS = frozenset(
    {
        "add_card_to_discard",
        "add_card_to_draw",
        "add_card_to_exhaust",
        "add_card_to_hand",
        "add_keyword_to_matching_cards",
        "add_keyword_to_random_card",
        "ally_channel_orb",
        "channel_orb",
        "combat_trigger",
        "choose_card",
        "discard_choice",
        "discard_hand",
        "evoke_orb",
        "block_from_ally",
        "discard_random",
        "dynamic_channel_orb",
        "exhaust_choice",
        "exhaust_random",
        "if_kill_resource",
        "orb_slot_delta",
        "osty_action",
        "play_top_card",
        "sovereign_blade",
        "timed_choice",
        "trigger_orb_passive",
    }
)
EXECUTABLE_EFFECT_KEYS = ENGINE_EFFECT_KEYS | EXTENDED_EFFECT_KEYS
_AUTHORITATIVE_EFFECT_KEYS = EXECUTABLE_EFFECT_KEYS | frozenset(
    {
        "description",
        "description_raw",
        "effect",
        "magic_number",
        "text",
    }
)


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

    card_spec = _merge_known_card_spec(
        card_spec,
        _lookup_card(_card_id(card_spec), card_library),
    )
    card_spec = _card_spec_with_upgraded_description(card_spec)
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
        "exhausts": _card_exhausts(card_spec),
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


def _card_spec_with_upgraded_description(card_spec: Mapping[str, Any]) -> Mapping[str, Any]:
    if not bool(card_spec.get("upgraded", False)):
        return card_spec
    upgrade_description = card_spec.get("upgrade_description")
    if not isinstance(upgrade_description, str) or not upgrade_description.strip():
        return card_spec
    updated = dict(card_spec)
    updated["description"] = upgrade_description
    updated["description_raw"] = upgrade_description
    return updated


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
        if _merge_if_kill_resource_step(merged, step):
            continue
        if _merge_damage_formula_step(merged, step):
            continue
        if any(_equivalent_effect_step(step, existing) for existing in merged):
            continue
        merged.append(step)
    return tuple(merged)


def _merge_if_kill_resource_step(
    merged: list[Mapping[str, Any]],
    step: Mapping[str, Any],
) -> bool:
    payload = step.get("if_kill_resource")
    if not isinstance(payload, Mapping):
        return False
    for index in range(len(merged) - 1, -1, -1):
        existing = merged[index]
        if not any(key in existing for key in ("damage", "all_damage", "damage_formula")):
            continue
        merged_step = dict(existing)
        existing_payload = merged_step.get("if_kill_resource")
        if isinstance(existing_payload, Sequence) and not isinstance(
            existing_payload,
            (str, bytes, bytearray),
        ):
            payloads: Any = tuple(existing_payload) + (dict(payload),)
        elif isinstance(existing_payload, Mapping):
            payloads = (dict(existing_payload), dict(payload))
        elif existing_payload:
            payloads = (existing_payload, dict(payload))
        else:
            payloads = dict(payload)
        merged_step["if_kill_resource"] = payloads
        merged[index] = merged_step
        return True
    return False


def _merge_damage_formula_step(
    merged: list[Mapping[str, Any]],
    step: Mapping[str, Any],
) -> bool:
    payload = step.get("damage_formula")
    if not isinstance(payload, Mapping):
        return False
    for index in range(len(merged) - 1, -1, -1):
        existing = merged[index]
        if "damage" not in existing or not isinstance(existing.get("damage"), int):
            continue
        merged_step = dict(existing)
        formula_payload = dict(payload)
        formula_payload["base"] = int(existing["damage"])
        merged_step.pop("damage", None)
        merged_step["damage_formula"] = formula_payload
        merged[index] = merged_step
        return True
    return False


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
    if damage_amount not in (None, 0) and not _is_osty_attack_source(source):
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
        if source_key == "block" and _description_has_osty_alive_condition(source):
            continue
        if source_key == "energy_gain" and _description_has_triggered_energy_gain(source):
            continue
        amount = _amount_from(source, source_key)
        if amount not in (None, 0):
            steps.append({effect_key: amount})

    random_cards = _amount_from(source, "add_random_card_to_hand")
    if random_cards in (None, 0):
        random_cards = _amount_from(source, "random_card_count")
    if random_cards not in (None, 0):
        steps.append({"add_random_card_to_hand": random_cards})

    skip_stale_sovereign_blade_fields = _description_modifies_sovereign_blade_directly(source)
    structured_status_steps: tuple[Mapping[str, Any], ...] = ()
    if not skip_stale_sovereign_blade_fields:
        structured_status_steps = _status_steps_from_source(
            source,
            card_type=card_type,
            target=target,
        )
        steps.extend(structured_status_steps)
    if structured_status_steps:
        steps.extend(_description_steps_without_statuses(source, target=target))
    else:
        steps.extend(_description_steps(source, target=target))
    if not skip_stale_sovereign_blade_fields:
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
    if generated is None and "card" in source:
        generated = source.get("card")
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
        for _ in range(max(0, count)):
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
    raw_card_type = _normalized_id(source.get("type", source.get("card_type")))
    card_type = _normalize_card_type(source.get("type", source.get("card_type")))
    if not description:
        if card_type in {"curse", "status"}:
            return ({"noop": True},)
        return ()
    normalized = _normalized_description(description)
    target = _normalize_target(source.get("target", target), card_type=card_type)
    steps: list[Mapping[str, Any]] = []

    if raw_card_type == "quest" and (
        "can be hatched at a [gold]rest site" in normalized
        or "unlocks a special event in the next act" in normalized
        or "marks a site of" in normalized
    ):
        steps.append({"noop": {"reason": "non_combat_quest_card"}})

    if "at the end of your turn" in normalized and (
        "if this is in your [gold]hand" in normalized or "if this is in your hand" in normalized
    ):
        steps.append({"end_turn_hand_effect": True})
    if "at the end of your turn" in normalized and (
        "if this is in your [gold]exhaust pile" in normalized
        or "if this is in your exhaust pile" in normalized
    ):
        steps.append(
            {
                "combat_trigger": {
                    "trigger": "turn_end",
                    "duration": "combat",
                    "condition": {"zone": "exhaust_pile", "card_id": _card_id(source)},
                    "effects": (
                        {
                            "play_named_card": {
                                "card_id": _card_id(source),
                                "from_zone": "exhaust_pile",
                            }
                        },
                    ),
                    "text": "at the end of your turn, if this is in your exhaust pile, play it",
                }
            }
        )
    if "must be played before other cards" in normalized:
        steps.append({"force_play_priority": True})

    for match in re.finditer(r"gain\s+\[(star|energy):(\d+)\]", normalized):
        if _match_in_timed_or_triggered_sentence(normalized, match.start()):
            continue
        resource_name, amount = match.groups()
        if resource_name == "energy":
            if not _has_amount_step(source, "energy"):
                steps.append({"energy": int(amount)})
        else:
            steps.append({"player_resource": {"resource": resource_name, "amount": int(amount)}})
    for match in re.finditer(
        r"gain\s+(\d+)\s+\[(star|energy):\d+\]",
        normalized,
    ):
        if _match_in_timed_or_triggered_sentence(normalized, match.start()):
            continue
        amount, resource_name = match.groups()
        if resource_name == "energy":
            if not _has_amount_step(source, "energy"):
                steps.append({"energy": int(amount)})
        else:
            steps.append({"player_resource": {"resource": resource_name, "amount": int(amount)}})

    next_turn: dict[str, Any] = {}
    for amount in re.findall(r"next turn,?\s*gain\s+\[energy:(\d+)\]", normalized):
        next_turn["energy"] = next_turn.get("energy", 0) + int(amount)
    for amount in re.findall(r"next turn,?\s*gain\s+\[star:(\d+)\]", normalized):
        next_turn["star"] = next_turn.get("star", 0) + int(amount)
    for amount in re.findall(r"next turn,?\s*draw\s+(\d+)", normalized):
        next_turn["draw"] = next_turn.get("draw", 0) + int(amount)
    for amount in re.findall(r"next turn,?\s*gain\s+(\d+)\s+\[gold\]block", normalized):
        next_turn["block"] = next_turn.get("block", 0) + int(amount)
    if "next turn, gain [gold]block[/gold] equal to your current [gold]block" in normalized:
        next_turn["block"] = {"formula": "player_block"}
    if next_turn:
        steps.append({"next_turn": next_turn})

    turn_start_resource_match = re.search(
        r"at the start of your turn,?\s*gain\s+\[(star|energy):(\d+)\]",
        normalized,
    )
    if turn_start_resource_match:
        resource_name, amount = turn_start_resource_match.groups()
        effect: Mapping[str, Any]
        if resource_name == "energy":
            effect = {"energy": int(amount)}
        else:
            effect = {"player_resource": {"resource": resource_name, "amount": int(amount)}}
        steps.append(
            {
                "combat_trigger": {
                    "trigger": "turn_start",
                    "duration": "combat",
                    "effects": (effect,),
                }
            }
        )

    played_resource_match = re.search(
        r"whenever you play a\s+(card|attack|skill|power),?\s*gain\s+\[(star|energy):(\d+)\]",
        normalized,
    )
    if played_resource_match:
        card_type_filter, resource_name, amount = played_resource_match.groups()
        effect = (
            {"energy": int(amount)}
            if resource_name == "energy"
            else {"player_resource": {"resource": resource_name, "amount": int(amount)}}
        )
        condition = (
            {"card_type": card_type_filter}
            if card_type_filter != "card"
            else {"card_type": "any"}
        )
        steps.append(
            {
                "combat_trigger": {
                    "trigger": "card_played",
                    "duration": "combat",
                    "condition": condition,
                    "effects": (effect,),
                }
            }
        )

    if "retain your [gold]hand" in normalized:
        steps.append({"retain_hand": True})

    if "you cannot draw additional cards this turn" in normalized:
        steps.append({"apply_status": {"target": "self", "no_draw_this_turn": 1}})
    if "all cards in your [gold]hand[/gold] are free to play this turn" in normalized:
        steps.append({"set_hand_free_to_play_this_turn": True})
    if "reduce the cost of all cards in your [gold]hand[/gold] to 1" in normalized:
        steps.append(
            {
                "set_hand_cost": {
                    "cost": 1,
                    "max_cost_only": True,
                    "duration": "combat" if "this combat" in normalized else "turn",
                }
            }
        )
    if "[gold]upgrade[/gold] all your cards" in normalized:
        steps.append({"upgrade_all_combat_cards": True})
    if "procure a random potion" in normalized:
        steps.append({"add_random_potion": {"count": 1}})
    if "double your energy" in normalized:
        steps.append({"energy_formula": {"formula": "current_energy"}})
    if "the first card you play each turn is played an extra time" in normalized:
        steps.append(
            {
                "combat_trigger": {
                    "trigger": "turn_start",
                    "effects": (
                        {
                            "next_card_extra_play": {
                                "card_type": "card",
                                "amount": 1,
                                "duration": "turn",
                            }
                        },
                    ),
                }
            }
        )
    if "draw cards until your [gold]hand[/gold] is full" in normalized:
        steps.append({"draw_formula": {"formula": "hand_space"}})

    random_free_match = re.search(
        r"add a random\s+(attack|skill|power)\s+into your \[gold\]hand\[/gold\]"
        r"\.?\s+it'?s free to play this turn",
        normalized,
    )
    if random_free_match:
        steps.append(
            {
                "add_random_card_to_hand": {
                    "count": 1,
                    "card_types": (random_free_match.group(1),),
                    "free_to_play_this_turn": True,
                }
            }
        )

    if "skills cost 0 [energy:1]" in normalized or "skills cost 0" in normalized:
        steps.append({"apply_status": {"target": "self", "skill_cost_zero": 1}})
    if "whenever you play a skill" in normalized and (
        "[gold]exhaust[/gold] it" in normalized or "exhaust it" in normalized
    ):
        steps.append({"apply_status": {"target": "self", "skill_exhaust_on_play": 1}})

    next_extra_match = re.search(
        r"\b(?:(this turn),?\s+)?your next\s+(card|skill|attack|power)\s+"
        r"is played an extra time\b",
        normalized,
    )
    if next_extra_match:
        steps.append(
            {
                "next_card_extra_play": {
                    "card_type": next_extra_match.group(2),
                    "amount": 1,
                    "duration": "turn" if next_extra_match.group(1) else "combat",
                }
            }
        )

    if (
        "deal damage equal to your [gold]block" in normalized
        or "deal damage equal to your block" in normalized
    ):
        steps.append({"damage_formula": {"formula": "player_block"}})
    if "deal damage equal to the number of cards played this combat" in normalized:
        steps.append({"damage_formula": {"formula": "cards_played_this_combat"}})
    if "deal damage equal to the number of cards in your [gold]draw pile" in normalized:
        steps.append({"damage_formula": {"formula": "draw_pile_count"}})
    if (
        "deal damage equal to the enemys [gold]doom" in normalized
        or "deal damage equal to the enemy's [gold]doom" in normalized
    ):
        steps.append({"damage_formula": {"formula": "target_doom"}})
    if "double your [gold]block" in normalized or "double your block" in normalized:
        steps.append({"block_formula": {"formula": "player_block"}})
    if (
        "gain [gold]block[/gold] equal to the number of cards in your [gold]discard pile"
        in normalized
        or "gain block equal to the number of cards in your discard pile" in normalized
    ):
        steps.append({"block_formula": {"formula": "discard_pile_count"}})
    if "gain [gold]block[/gold] equal to [gold]poison[/gold] on all enemies" in normalized:
        steps.append({"block_formula": {"formula": "all_enemy_poison"}})

    heal_match = re.search(r"\bheal\s+(\d+)\s+hp\b", normalized)
    if heal_match and not _has_amount_step(source, "heal"):
        steps.append({"heal": int(heal_match.group(1))})

    if "remove all [gold]artifact[/gold] and [gold]block[/gold] from the enemy" in normalized:
        steps.append({"remove_status": {"target": "enemy", "statuses": ("artifact",)}})
        steps.append({"remove_block": {"target": "enemy"}})

    strength_loss_target = "all_enemies" if "all enemies lose" in normalized else "enemy"
    strength_loss_status = "temporary_strength" if "this turn" in normalized else "strength"
    strength_loss_match = re.search(
        r"(?:all enemies|enemy)\s+loses?\s+(\d+)\s+\[gold\]strength\[/gold\]",
        normalized,
    )
    if strength_loss_match:
        steps.append(
            {
                "apply_status": {
                    "target": strength_loss_target,
                    strength_loss_status: -int(strength_loss_match.group(1)),
                }
            }
        )
    player_strength_loss_match = re.search(
        r"\blose\s+(\d+)\s+\[gold\]strength\[/gold\]",
        normalized,
    )
    if player_strength_loss_match:
        steps.append(
            {
                "apply_status": {
                    "target": "self",
                    "strength": -int(player_strength_loss_match.group(1)),
                }
            }
        )
    if "enemy loses x" in normalized and "[gold]strength[/gold]" in normalized:
        status_payload: dict[str, Any] = {
            "target": "enemy",
            "strength": {"amount": 0, "per_energy": -1},
        }
        if "apply x" in normalized and "[gold]weak[/gold]" in normalized:
            status_payload["weak"] = {"amount": 0, "per_energy": 1}
        steps.append({"apply_status": status_payload})

    for status_match in re.finditer(
        r"apply\s+(\d+)\s+\[gold\](weak|vulnerable|frail|poison)\[/gold\]"
        r"(?:\s+and\s+\[gold\](weak|vulnerable|frail|poison)\[/gold\])?",
        normalized,
    ):
        amount = int(status_match.group(1))
        status_ids = [status_match.group(2)]
        if status_match.group(3):
            status_ids.append(status_match.group(3))
        status_target = (
            "all_enemies"
            if "all enemies" in normalized[status_match.start() : status_match.end() + 32]
            else _default_status_target(target)
        )
        steps.append(
            {
                "apply_status": {
                    "target": status_target,
                    **{status_id: amount for status_id in status_ids},
                }
            }
        )

    if re.search(r"play the top x(?:\+1)? cards? of your \[gold\]draw pile", normalized):
        steps.append({"play_top_card": {"amount": {"amount": 0, "per_energy": 1}}})

    if "discard your [gold]hand" in normalized or "discard your hand" in normalized:
        steps.append({"discard_hand": {"mode": "all"}})
    else:
        random_discard_match = re.search(
            r"discard\s+(?:(\d+)|a|an|one)?\s*random\s+cards?",
            normalized,
        )
        if random_discard_match and not _match_in_timed_or_triggered_sentence(
            normalized,
            random_discard_match.start(),
        ):
            steps.append({"discard_random": int(random_discard_match.group(1) or 1)})
        discard_match = re.search(r"discard\s+(?:(\d+)|a|an|one)\s+cards?", normalized)
        if discard_match and not _match_in_timed_or_triggered_sentence(
            normalized,
            discard_match.start(),
        ):
            steps.append({"discard_choice": int(discard_match.group(1) or 1)})

    random_exhaust_match = re.search(
        r"exhaust\s+(?:(\d+)|a|an|one)?\s*random\s+cards?",
        normalized,
    )
    if random_exhaust_match and not _match_in_timed_or_triggered_sentence(
        normalized,
        random_exhaust_match.start(),
    ):
        steps.append({"exhaust_random": int(random_exhaust_match.group(1) or 1)})
    else:
        exhaust_match = re.search(r"exhaust\s+(?:(\d+)|a|an|one)\s+cards?", normalized)
        if exhaust_match and not _match_in_timed_or_triggered_sentence(
            normalized,
            exhaust_match.start(),
        ):
            steps.append({"exhaust_choice": int(exhaust_match.group(1) or 1)})

    for forge_match in re.finditer(r"\[gold\]forge\[/gold\]\s+(\d+)", normalized):
        if _match_in_timed_or_triggered_sentence(normalized, forge_match.start()):
            continue
        steps.append(
            {"player_resource": {"resource": "forge", "amount": int(forge_match.group(1))}}
        )
    if (
        not _description_has_triggered_summon(source)
        and not _description_has_dynamic_summon(source)
    ):
        for amount in re.findall(r"\[gold\]summon\[/gold\]\s+(\d+)", normalized):
            steps.append({"player_resource": {"resource": "summon", "amount": int(amount)}})

    if "enemy loses" in normalized and target in {"enemy", "all_enemies"}:
        loss_match = re.search(r"enemy loses\s+(\d+)\s+hp", normalized)
        if loss_match:
            key = "all_damage" if target == "all_enemies" else "damage"
            steps.append({key: int(loss_match.group(1))})

    return tuple(steps)


def _description_steps_without_statuses(
    source: Mapping[str, Any],
    *,
    target: str,
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        step
        for step in _description_steps(source, target=target)
        if "apply_status" not in step
    )


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
    if re.search(r"\badd\s+x\s+", description):
        return 0
    match = re.search(r"add\s+(\d+)\s+", description)
    if match:
        return max(1, int(match.group(1)))
    if "add a " in description or "add an " in description or "add 1 " in description:
        return 1
    return 1


def _is_osty_attack_source(source: Mapping[str, Any]) -> bool:
    tags = {_normalized_id(tag) for tag in _string_tuple(source.get("tags", ()))}
    if "ostyattack" in tags or "osty_attack" in tags:
        return True
    description = _normalized_description(str(source.get("description", "") or ""))
    return "osty" in description and "deals" in description and "damage" in description


def _description_has_osty_alive_condition(source: Mapping[str, Any]) -> bool:
    description = _normalized_description(str(source.get("description", "") or ""))
    return "if [gold]osty[/gold] is alive" in description or "if osty is alive" in description


def _description_has_triggered_energy_gain(source: Mapping[str, Any]) -> bool:
    description = _normalized_description(str(source.get("description", "") or ""))
    return (
        "whenever" in description
        and "costs [energy:" in description
        or "when this card is [gold]exhausted" in description
        or "when this card is exhausted" in description
        or "when this is [gold]exhausted" in description
        or "when this is exhausted" in description
    )


def _card_exhausts(card_spec: Mapping[str, Any]) -> bool:
    if _source_removes_exhaust_on_upgrade(card_spec):
        return False
    explicit = card_spec.get("exhausts", card_spec.get("exhaust"))
    if explicit is not None:
        return bool(explicit)
    keywords = {
        _normalized_id(keyword)
        for keyword in (
            _string_tuple(card_spec.get("keywords", ()))
            + _string_tuple(card_spec.get("keywords_key", ()))
        )
    }
    return "exhaust" in keywords


def _source_removes_exhaust_on_upgrade(card_spec: Mapping[str, Any]) -> bool:
    if not bool(card_spec.get("upgraded", False)):
        return False
    upgrade = card_spec.get("upgrade")
    if not isinstance(upgrade, Mapping):
        return False
    return bool(upgrade.get("remove_exhaust")) or upgrade.get("exhaust") is False


def _description_has_triggered_summon(source: Mapping[str, Any]) -> bool:
    description = _normalized_description(str(source.get("description", "") or ""))
    return "whenever" in description and "[gold]summon[/gold]" in description


def _description_modifies_sovereign_blade_directly(source: Mapping[str, Any]) -> bool:
    description = _normalized_description(str(source.get("description", "") or ""))
    description = re.sub(r"\[[^\]]+\]", "", description)
    return "sovereign blade now gains" in description or "sovereign blade gains" in description


def _description_has_dynamic_summon(source: Mapping[str, Any]) -> bool:
    description = _normalized_description(str(source.get("description", "") or ""))
    return bool(re.search(r"\[gold\]summon\[/gold\]\s+(?:\d+\s+)?x\b", description))


def _match_in_timed_or_triggered_sentence(description: str, match_start: int) -> bool:
    sentence_start = max(description.rfind(".", 0, match_start) + 1, 0)
    sentence = description[sentence_start:].lstrip()
    return sentence.startswith(
        (
            "at the start",
            "at the end",
            "if ",
            "when ",
            "whenever ",
            "next turn",
        )
    )


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
        return _merge_known_card_spec(card_spec, _lookup_card(card_id, card_library))
    card_id = str(card_spec)
    return _lookup_card(card_id, card_library) or {"id": card_id, "name": card_id, "cost": 0}


def _lookup_card(
    card_id: str,
    card_library: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    normalized = _normalized_id(card_id)
    library_card: Mapping[str, Any] = {}
    if not card_library:
        return _KNOWN_CARD_SPECS.get(normalized, {})
    candidates = (card_id, _normalized_id(card_id), str(card_id).upper())
    for candidate in candidates:
        found = card_library.get(candidate)
        if isinstance(found, Mapping):
            library_card = found
            break
    if not library_card:
        for key, value in card_library.items():
            if _normalized_id(key) == normalized and isinstance(value, Mapping):
                library_card = value
                break
    known_card = _KNOWN_CARD_SPECS.get(normalized, {})
    if known_card and library_card:
        merged = dict(known_card)
        merged.update(library_card)
        return merged
    return library_card or known_card or {}


def _card_id(card_spec: Mapping[str, Any]) -> str:
    return _normalized_id(card_spec.get("card_id", card_spec.get("id", "unknown_card")))


def _merge_known_card_spec(
    card_spec: Mapping[str, Any],
    known_card: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not known_card:
        return card_spec
    merged = dict(known_card)
    merged.update(card_spec)
    if _card_spec_has_authoritative_effect_text(card_spec):
        for key in ("effect", "effects"):
            if key not in card_spec:
                merged.pop(key, None)
    source_type = _normalize_card_type(card_spec.get("type", card_spec.get("card_type")))
    known_type = known_card.get("type", known_card.get("card_type"))
    if source_type == "unknown" and _normalize_card_type(known_type) != "unknown":
        merged["type"] = known_type
        if "card_type" in merged or "card_type" in known_card:
            merged["card_type"] = known_type
    if source_type == "unknown" and _normalize_card_type(known_type) in {"curse", "status"}:
        for key in ("cost", "target", "effects", "tags", "exhausts"):
            if key in known_card:
                merged[key] = known_card[key]
        known_custom = known_card.get("custom")
        source_custom = card_spec.get("custom")
        if isinstance(known_custom, Mapping):
            merged["custom"] = {
                **dict(known_custom),
                **(dict(source_custom) if isinstance(source_custom, Mapping) else {}),
            }
    return merged


def _card_spec_has_authoritative_effect_text(card_spec: Mapping[str, Any]) -> bool:
    if any(key in card_spec for key in _AUTHORITATIVE_EFFECT_KEYS):
        return True
    description = card_spec.get("description", card_spec.get("description_raw", ""))
    return isinstance(description, str) and bool(description.strip())


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

_KNOWN_CARD_SPECS: dict[str, Mapping[str, Any]] = {
    "anger": {
        "card_id": "anger",
        "name": "Anger",
        "type": "attack",
        "cost": 0,
        "target": "enemy",
        "effects": {"damage": 6, "add_card_to_discard": {"card_id": "anger"}},
    },
    "armaments": {
        "card_id": "armaments",
        "name": "Armaments",
        "type": "skill",
        "cost": 1,
        "target": "self",
        "effects": {"block": 5, "upgrade_all_combat_cards": 1},
    },
    "battle_trance": {
        "card_id": "battle_trance",
        "name": "Battle Trance",
        "type": "skill",
        "cost": 0,
        "target": "self",
        "effects": {"draw": 3, "apply_status": {"target": "self", "no_draw": 1}},
    },
    "body_slam": {
        "card_id": "body_slam",
        "name": "Body Slam",
        "type": "attack",
        "cost": 1,
        "target": "enemy",
        "effects": {"damage_formula": {"formula": "player_block"}},
    },
    "carnage": {
        "card_id": "carnage",
        "name": "Carnage",
        "type": "attack",
        "cost": 2,
        "target": "enemy",
        "effects": {"damage": 20},
    },
    "cleave": {
        "card_id": "cleave",
        "name": "Cleave",
        "type": "attack",
        "cost": 1,
        "target": "all_enemies",
        "effects": {"all_damage": 8},
    },
    "clumsy": {
        "card_id": "clumsy",
        "name": "Clumsy",
        "type": "curse",
        "cost": -1,
        "target": "none",
        "tags": ("curse", "unplayable", "deck_burden"),
        "custom": {"unplayable": True},
        "effects": {"noop": {"reason": "curse_burden"}},
    },
    "clothesline": {
        "card_id": "clothesline",
        "name": "Clothesline",
        "type": "attack",
        "cost": 2,
        "target": "enemy",
        "effects": {"damage": 12, "apply_status": {"target": "enemy", "weak": 2}},
    },
    "curse_of_the_bell": {
        "card_id": "curse_of_the_bell",
        "name": "Curse of the Bell",
        "type": "curse",
        "cost": -1,
        "target": "none",
        "tags": ("curse", "eternal", "unplayable", "deck_burden"),
        "custom": {"eternal": True, "unplayable": True, "source_relic": "calling_bell"},
        "effects": {"noop": {"reason": "calling_bell_curse"}},
    },
    "decay": {
        "card_id": "decay",
        "name": "Decay",
        "type": "curse",
        "cost": -1,
        "target": "none",
        "tags": ("curse", "unplayable", "deck_burden"),
        "custom": {"unplayable": True},
        "effects": {"noop": {"reason": "curse_burden"}},
    },
    "disarm": {
        "card_id": "disarm",
        "name": "Disarm",
        "type": "skill",
        "cost": 1,
        "target": "enemy",
        "effects": {"apply_status": {"target": "enemy", "strength": -2}},
        "exhausts": True,
    },
    "doubt": {
        "card_id": "doubt",
        "name": "Doubt",
        "type": "curse",
        "cost": -1,
        "target": "none",
        "tags": ("curse", "unplayable", "deck_burden"),
        "custom": {"unplayable": True},
        "effects": {"noop": {"reason": "curse_burden"}},
    },
    "flame_barrier": {
        "card_id": "flame_barrier",
        "name": "Flame Barrier",
        "type": "skill",
        "cost": 2,
        "target": "self",
        "effects": {"block": 12, "apply_status": {"target": "self", "flame_barrier": 4}},
    },
    "hemokinesis": {
        "card_id": "hemokinesis",
        "name": "Hemokinesis",
        "type": "attack",
        "cost": 1,
        "target": "enemy",
        "effects": {"hp_loss": 2, "damage": 15},
    },
    "inflame": {
        "card_id": "inflame",
        "name": "Inflame",
        "type": "power",
        "cost": 1,
        "target": "self",
        "effects": {"apply_status": {"target": "self", "strength": 2}},
    },
    "iron_wave": {
        "card_id": "iron_wave",
        "name": "Iron Wave",
        "type": "attack",
        "cost": 1,
        "target": "enemy",
        "effects": {"damage": 5, "block": 5},
    },
    "offering": {
        "card_id": "offering",
        "name": "Offering",
        "type": "skill",
        "cost": 0,
        "target": "self",
        "effects": {"hp_loss": 6, "energy": 2, "draw": 3},
        "exhausts": True,
    },
    "pommel_strike": {
        "card_id": "pommel_strike",
        "name": "Pommel Strike",
        "type": "attack",
        "cost": 1,
        "target": "enemy",
        "effects": {"damage": 9, "draw": 1},
    },
    "rampage": {
        "card_id": "rampage",
        "name": "Rampage",
        "type": "attack",
        "cost": 1,
        "target": "enemy",
        "effects": {"damage": 8, "apply_status": {"target": "self", "rampage": 5}},
    },
    "shrug_it_off": {
        "card_id": "shrug_it_off",
        "name": "Shrug It Off",
        "type": "skill",
        "cost": 1,
        "target": "self",
        "effects": {"block": 8, "draw": 1},
    },
    "sword_boomerang": {
        "card_id": "sword_boomerang",
        "name": "Sword Boomerang",
        "type": "attack",
        "cost": 1,
        "target": "enemy",
        "effects": {"sequence": [{"damage": 3}, {"damage": 3}, {"damage": 3}]},
    },
    "thunderclap": {
        "card_id": "thunderclap",
        "name": "Thunderclap",
        "type": "attack",
        "cost": 1,
        "target": "all_enemies",
        "effects": {"all_damage": 4, "apply_status": {"target": "all_enemies", "vulnerable": 1}},
    },
    "true_grit": {
        "card_id": "true_grit",
        "name": "True Grit",
        "type": "skill",
        "cost": 1,
        "target": "self",
        "effects": {"block": 7, "exhaust_choice": 1},
    },
    "uppercut": {
        "card_id": "uppercut",
        "name": "Uppercut",
        "type": "attack",
        "cost": 2,
        "target": "enemy",
        "effects": {
            "damage": 13,
            "apply_status": {"target": "enemy", "weak": 1, "vulnerable": 1},
        },
    },
    "blood_wall": {
        "card_id": "blood_wall",
        "name": "Blood Wall",
        "type": "skill",
        "cost": 1,
        "target": "self",
        "effects": {"block": 10, "hp_loss": 2},
    },
    "bully": {
        "card_id": "bully",
        "name": "Bully",
        "type": "attack",
        "cost": 1,
        "target": "enemy",
        "effects": {"damage": 8, "apply_status": {"target": "enemy", "vulnerable": 1}},
    },
    "cinder": {
        "card_id": "cinder",
        "name": "Cinder",
        "type": "attack",
        "cost": 1,
        "target": "enemy",
        "effects": {"damage": 7, "apply_status": {"target": "enemy", "burn": 1}},
    },
    "cruelty": {
        "card_id": "cruelty",
        "name": "Cruelty",
        "type": "attack",
        "cost": 1,
        "target": "enemy",
        "effects": {"damage": 9, "apply_status": {"target": "enemy", "weak": 1}},
    },
    "demonic_shield": {
        "card_id": "demonic_shield",
        "name": "Demonic Shield",
        "type": "skill",
        "cost": 2,
        "target": "self",
        "effects": {"block": 14, "apply_status": {"target": "self", "strength": 1}},
    },
    "dismantle": {
        "card_id": "dismantle",
        "name": "Dismantle",
        "type": "attack",
        "cost": 1,
        "target": "enemy",
        "effects": {"damage": 8, "remove_block": 1},
    },
    "dominate": {
        "card_id": "dominate",
        "name": "Dominate",
        "type": "attack",
        "cost": 1,
        "target": "enemy",
        "effects": {"damage": 10},
    },
    "forgotten_ritual": {
        "card_id": "forgotten_ritual",
        "name": "Forgotten Ritual",
        "type": "power",
        "cost": 1,
        "target": "self",
        "effects": {"apply_status": {"target": "self", "strength": 1, "ritual": 1}},
    },
    "greed": {
        "card_id": "greed",
        "name": "Greed",
        "type": "curse",
        "cost": -1,
        "target": "none",
        "tags": ("curse", "eternal", "unplayable", "gold_frontload", "deck_burden"),
        "custom": {
            "eternal": True,
            "unplayable": True,
            "frontloaded_gold": 333,
            "source_relic": "cursed_pearl",
        },
        "effects": {"noop": {"reason": "frontloaded_gold_curse"}},
    },
    "guilty": {
        "card_id": "guilty",
        "name": "Guilty",
        "type": "curse",
        "cost": -1,
        "target": "none",
        "tags": ("curse", "temporary", "unplayable", "deck_burden"),
        "custom": {"temporary": True, "unplayable": True, "combats_until_remove": 5},
        "effects": {"noop": {"reason": "temporary_curse_burden"}},
    },
    "howl_from_beyond": {
        "card_id": "howl_from_beyond",
        "name": "Howl From Beyond",
        "type": "power",
        "cost": 1,
        "target": "self",
        "effects": {"apply_status": {"target": "self", "strength": 1}},
    },
    "injury": {
        "card_id": "injury",
        "name": "Injury",
        "type": "curse",
        "cost": -1,
        "target": "none",
        "tags": ("curse", "unplayable", "deck_burden"),
        "custom": {"unplayable": True},
        "effects": {"noop": {"reason": "curse_burden"}},
    },
    "inferno": {
        "card_id": "inferno",
        "name": "Inferno",
        "type": "attack",
        "cost": 2,
        "target": "all_enemies",
        "effects": {"all_damage": 14},
    },
    "molten_fist": {
        "card_id": "molten_fist",
        "name": "Molten Fist",
        "type": "attack",
        "cost": 1,
        "target": "enemy",
        "effects": {"damage": 9},
    },
    "normality": {
        "card_id": "normality",
        "name": "Normality",
        "type": "curse",
        "cost": -1,
        "target": "none",
        "tags": ("curse", "unplayable", "deck_burden"),
        "custom": {"unplayable": True},
        "effects": {"noop": {"reason": "curse_burden"}},
    },
    "pain": {
        "card_id": "pain",
        "name": "Pain",
        "type": "curse",
        "cost": -1,
        "target": "none",
        "tags": ("curse", "unplayable", "deck_burden"),
        "custom": {"unplayable": True},
        "effects": {"noop": {"reason": "curse_burden"}},
    },
    "parasite": {
        "card_id": "parasite",
        "name": "Parasite",
        "type": "curse",
        "cost": -1,
        "target": "none",
        "tags": ("curse", "unplayable", "deck_burden"),
        "custom": {"unplayable": True},
        "effects": {"noop": {"reason": "curse_burden"}},
    },
    "regret": {
        "card_id": "regret",
        "name": "Regret",
        "type": "curse",
        "cost": -1,
        "target": "none",
        "tags": ("curse", "unplayable", "deck_burden"),
        "custom": {"unplayable": True},
        "effects": {"noop": {"reason": "curse_burden"}},
    },
    "shame": {
        "card_id": "shame",
        "name": "Shame",
        "type": "curse",
        "cost": -1,
        "target": "none",
        "tags": ("curse", "unplayable", "deck_burden"),
        "custom": {"unplayable": True},
        "effects": {"noop": {"reason": "curse_burden"}},
    },
    "taunt": {
        "card_id": "taunt",
        "name": "Taunt",
        "type": "skill",
        "cost": 1,
        "target": "enemy",
        "effects": {"apply_status": {"target": "enemy", "weak": 2}},
    },
    "vicious": {
        "card_id": "vicious",
        "name": "Vicious",
        "type": "attack",
        "cost": 1,
        "target": "enemy",
        "effects": {"damage": 9},
    },
    "writhe": {
        "card_id": "writhe",
        "name": "Writhe",
        "type": "curse",
        "cost": -1,
        "target": "none",
        "tags": ("curse", "unplayable", "deck_burden"),
        "custom": {"unplayable": True},
        "effects": {"noop": {"reason": "curse_burden"}},
    },
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
