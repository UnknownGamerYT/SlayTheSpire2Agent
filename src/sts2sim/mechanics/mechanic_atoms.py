"""Fixed mechanic atom encoders for visible cards and statuses.

The helpers in this module expose deterministic, JSON-friendly facts only.
They intentionally do not score whether an atom is strategically desirable.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

STATUS_ATOM_KEYS: tuple[str, ...] = (
    "strength",
    "dexterity",
    "focus",
    "poison",
    "weak",
    "vulnerable",
    "frail",
    "intangible",
    "thorns",
    "regen",
    "buffer",
    "artifact",
    "confused",
    "no_draw",
    "retain_block",
    "block_return",
    "barricade",
    "ritual",
    "plated_armor",
    "lock_on",
    "mark",
)

CARD_ATOM_KEYS: tuple[str, ...] = (
    "damage",
    "aoe_damage",
    "block",
    "draw_now",
    "draw_next_turn",
    "energy_now",
    "energy_next_turn",
    "heal",
    "hp_loss",
    "max_hp_gain",
    "gold_gain",
    "apply_poison",
    "apply_weak",
    "apply_vulnerable",
    "apply_frail",
    "apply_strength",
    "apply_dexterity",
    "apply_focus",
    "chosen_discard",
    "random_discard",
    "self_discard_payoff",
    "chosen_exhaust",
    "random_exhaust",
    "exhaust_self",
    "exhaust_payoff",
    "chosen_transform",
    "random_transform",
    "create_card",
    "discover_card",
    "temporary_card",
    "card_upgrade",
    "card_remove",
    "retain",
    "innate",
    "eternal",
    "unplayable",
    "ethereal",
    "cost_change_temp",
    "cost_change_combat",
    "cost_change_permanent",
    "on_play",
    "on_discard",
    "on_exhaust",
    "on_kill",
    "start_turn",
    "end_turn",
    "start_combat",
    "end_combat",
    "delayed",
    "repeating",
    "periodic",
    "randomness",
    "summon",
    "orb_channel",
    "orb_evoke",
    "forge",
    "star",
    "potion_generation",
    "relic_interaction",
)

CARD_SLOT_KEYS: tuple[str, ...] = (
    "card_identity_bucket",
    "card_type_id",
    "cost",
    "upgraded",
    "playable",
    "zone_id",
    "position",
    *CARD_ATOM_KEYS,
)

_CARD_TYPE_IDS = {
    "": 0,
    "unknown": 0,
    "attack": 1,
    "skill": 2,
    "power": 3,
    "status": 4,
    "curse": 5,
}
_ZONE_IDS = {
    "": 0,
    "master_deck": 1,
    "hand": 2,
    "draw_pile": 3,
    "discard_pile": 4,
    "exhaust_pile": 5,
    "reward": 6,
    "shop": 7,
}
_CARD_IDENTITY_BUCKETS = 256
_STATUS_ALIASES = {
    "plate_armor": "plated_armor",
    "plated_armor": "plated_armor",
    "lockon": "lock_on",
    "lock_on": "lock_on",
    "nodraw": "no_draw",
    "no_draw": "no_draw",
}
_STATUS_TO_CARD_ATOM = {
    "poison": "apply_poison",
    "weak": "apply_weak",
    "vulnerable": "apply_vulnerable",
    "frail": "apply_frail",
    "strength": "apply_strength",
    "dexterity": "apply_dexterity",
    "focus": "apply_focus",
}


def status_atom_summary(statuses: Mapping[str, Any]) -> dict[str, float]:
    """Return fixed status atom values from a visible status mapping."""

    summary = {key: 0.0 for key in STATUS_ATOM_KEYS}
    for raw_key, raw_value in statuses.items():
        key = _status_key(str(raw_key))
        if key in summary:
            summary[key] += _amount(raw_value, default=1.0)
    return summary


def status_atom_vector(summary: Mapping[str, Any]) -> tuple[float, ...]:
    """Return status atoms in ``STATUS_ATOM_KEYS`` order."""

    return tuple(_float(summary.get(key)) for key in STATUS_ATOM_KEYS)


def card_atom_summary(card: Mapping[str, Any]) -> dict[str, float]:
    """Return fixed card mechanic atoms from a visible card descriptor."""

    summary = {key: 0.0 for key in CARD_ATOM_KEYS}
    custom = _mapping(card.get("custom"))
    tags = tuple(_normalized_id(str(tag)) for tag in _sequence(card.get("tags")))
    effect_keys = tuple(_normalized_id(str(key)) for key in _sequence(card.get("effect_keys")))
    effects = card.get("effects", {})
    if not isinstance(effects, Mapping):
        effects = {}

    _apply_card_flags(summary, card)
    _apply_card_flags(summary, custom)
    for tag in tags + effect_keys:
        _apply_token(summary, tag, 1.0)

    amounts = _mapping(card.get("effect_amounts"))
    if not effects:
        _add(summary, "damage", _float(amounts.get("damage")))
        _add(summary, "block", _float(amounts.get("block")))
        _add(summary, "draw_now", _float(amounts.get("draw")))
        _add(summary, "energy_now", _float(amounts.get("energy")))
        _add(summary, "heal", _float(amounts.get("heal")))

    _walk_effects(summary, effects)
    if bool(card.get("exhausts")):
        summary["exhaust_self"] = max(summary["exhaust_self"], 1.0)
    return summary


def card_slot_summary(
    card: Mapping[str, Any],
    *,
    zone: str = "",
    position: int = 0,
) -> dict[str, Any]:
    """Return card slot metadata plus fixed card atom values."""

    card_id = str(card.get("card_id", card.get("id", "")))
    card_type = _normalized_id(str(card.get("type", "")))
    atoms = card_atom_summary(card)
    slot: dict[str, Any] = {
        "card_identity_bucket": float(_hash_bucket(card_id, _CARD_IDENTITY_BUCKETS)),
        "card_type_id": float(_CARD_TYPE_IDS.get(card_type, 0)),
        "cost": _float(card.get("cost")),
        "upgraded": _bool_float(card.get("upgraded")),
        "playable": _playable_value(card, atoms),
        "zone_id": float(_ZONE_IDS.get(_normalized_id(zone or str(card.get("zone", ""))), 0)),
        "position": float(position),
    }
    slot.update(atoms)
    return slot


def card_slot_vector(slot: Mapping[str, Any]) -> tuple[float, ...]:
    """Return card slot values in ``CARD_SLOT_KEYS`` order."""

    return tuple(_float(slot.get(key)) for key in CARD_SLOT_KEYS)


def card_slots_from_payload(
    payload: Mapping[str, Any],
    *,
    zone: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Build card slots from serialized master-deck or combat card zones."""

    zone_key = _normalized_id(zone)
    if limit <= 0:
        return []
    if zone_key == "master_deck":
        cards = _sequence(payload.get("master_deck"))
        return [
            card_slot_summary(_mapping(card), zone="master_deck", position=index)
            for index, card in enumerate(cards[:limit])
            if isinstance(card, Mapping)
        ]

    combat = _mapping(payload.get("combat"))
    zones = (
        ("hand", "draw_pile", "discard_pile", "exhaust_pile")
        if zone_key == "combat"
        else (zone_key,)
    )
    slots: list[dict[str, Any]] = []
    for combat_zone in zones:
        for index, card in enumerate(_sequence(combat.get(combat_zone))):
            if len(slots) >= limit:
                return slots
            if isinstance(card, Mapping):
                slots.append(card_slot_summary(card, zone=combat_zone, position=index))
    return slots


def _walk_effects(summary: dict[str, float], value: object, parent_key: str = "") -> None:
    if isinstance(value, Mapping):
        normalized_items = [(_normalized_id(str(key)), item) for key, item in value.items()]
        keys = {key for key, _item in normalized_items}
        if "apply_status" in {parent_key, *keys} or parent_key in {"status", "statuses"}:
            _apply_status_payload(summary, value)
        if _is_next_turn_payload(keys, value):
            _apply_next_turn_payload(summary, value)
        if _is_timing_payload(keys, value):
            _apply_timing_payload(summary, value)
        for key, item in normalized_items:
            amount = _amount(item, default=1.0)
            _apply_token(summary, key, amount, item=item, parent_key=parent_key)
            _walk_effects(summary, item, key)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _walk_effects(summary, item, parent_key)


def _apply_token(
    summary: dict[str, float],
    token: str,
    amount: float,
    *,
    item: object = None,
    parent_key: str = "",
) -> None:
    if not token:
        return
    if token in {"damage", "hit", "attack_damage"}:
        _add(summary, "damage", amount)
    elif token in {"all_damage", "aoe_damage"}:
        _add(summary, "aoe_damage", amount)
    elif token in {"block", "gain_block"}:
        _add(summary, "block", amount)
    elif token in {"draw", "draw_cards"}:
        _add(summary, "draw_next_turn" if parent_key == "next_turn" else "draw_now", amount)
    elif token in {"energy", "gain_energy"}:
        _add(summary, "energy_next_turn" if parent_key == "next_turn" else "energy_now", amount)
    elif token in {"heal", "heal_player"}:
        _add(summary, "heal", amount)
    elif token in {"hp_loss", "lose_hp"}:
        _add(summary, "hp_loss", abs(amount))
    elif token in {"max_hp", "max_hp_gain", "max_hp_delta"}:
        _add(summary, "max_hp_gain", max(0.0, amount))
    elif token in {"gold", "gain_gold", "gold_gain"}:
        _add(summary, "gold_gain", max(0.0, amount))
    elif token in {"discard", "discard_choice", "chosen_discard", "choose_discard"}:
        _add(summary, "chosen_discard", amount)
    elif token in {"random_discard", "discard_random"} or (
        "discard" in token and "random" in token
    ):
        _add(summary, "random_discard", amount)
        summary["randomness"] = max(summary["randomness"], 1.0)
    elif token in {"self_discard_payoff", "discard_payoff"}:
        _add(summary, "self_discard_payoff", amount)
    elif token in {"exhaust", "exhaust_choice", "chosen_exhaust", "choose_exhaust"}:
        _add(summary, "chosen_exhaust", amount)
    elif token in {"random_exhaust", "exhaust_random"} or (
        "exhaust" in token and "random" in token
    ):
        _add(summary, "random_exhaust", amount)
        summary["randomness"] = max(summary["randomness"], 1.0)
    elif token in {"exhaust_self", "exhaust_on_play"}:
        summary["exhaust_self"] = max(summary["exhaust_self"], 1.0)
    elif token in {"exhaust_payoff", "on_exhaust"}:
        _add(summary, "exhaust_payoff" if token == "exhaust_payoff" else "on_exhaust", amount)
    elif token.startswith("keyword_"):
        keyword = token.removeprefix("keyword_")
        if keyword in {"retain", "retained"}:
            summary["retain"] = max(summary["retain"], 1.0)
        elif keyword in {"innate", "eternal", "unplayable", "ethereal"}:
            summary[keyword] = max(summary[keyword], 1.0)
    elif token in {"transform", "transform_choice", "chosen_transform", "choose_transform"}:
        _add(summary, "chosen_transform", amount)
    elif token in {"random_transform", "transform_random"} or (
        "transform" in token and "random" in token
    ):
        _add(summary, "random_transform", amount)
        summary["randomness"] = max(summary["randomness"], 1.0)
    elif token in {"create_card", "add_card_to_hand", "add_card_to_draw", "add_card_to_discard"}:
        _add(summary, "create_card", amount)
    elif token in {"discover_card", "choose_card", "choice_card", "choose_one_of_random"}:
        _add(summary, "discover_card", amount)
    elif token in {"temporary_card", "temp_card"}:
        _add(summary, "temporary_card", amount)
    elif token in {"card_upgrade", "upgrade_card", "upgrade_hand", "upgrade_all"}:
        _add(summary, "card_upgrade", amount)
    elif token in {"card_remove", "remove_card", "remove_deck_cards"}:
        _add(summary, "card_remove", amount)
    elif token in {"cost_change_temp", "self_cost_delta", "set_hand_free_to_play_this_turn"}:
        _add(summary, "cost_change_temp", amount)
    elif token in {"cost_change_combat", "set_hand_cost"}:
        _add(summary, "cost_change_combat", amount)
    elif token in {"cost_change_permanent", "permanent_cost_delta"}:
        _add(summary, "cost_change_permanent", amount)
    elif token in CARD_ATOM_KEYS:
        _add(summary, token, amount)
    elif token in {"channel_orb", "dynamic_channel_orb", "ally_channel_orb"}:
        _add(summary, "orb_channel", amount)
    elif token == "evoke_orb":
        _add(summary, "orb_evoke", amount)
    elif token in {"potion_generation", "random_potion", "add_random_potion"}:
        _add(summary, "potion_generation", amount)
        if "random" in token:
            summary["randomness"] = max(summary["randomness"], 1.0)
    elif "relic" in token:
        _add(summary, "relic_interaction", amount)
    elif "random" in token or _mapping(item).get("selection") == "random":
        summary["randomness"] = max(summary["randomness"], 1.0)


def _apply_status_payload(summary: dict[str, float], payload: Mapping[str, Any]) -> None:
    status = _status_key(str(payload.get("status", payload.get("power", ""))))
    amount = _amount(payload.get("amount"), default=1.0)
    if status in _STATUS_TO_CARD_ATOM:
        _add(summary, _STATUS_TO_CARD_ATOM[status], amount)
    for raw_key, raw_value in payload.items():
        key = _status_key(str(raw_key))
        if key in _STATUS_TO_CARD_ATOM:
            _add(summary, _STATUS_TO_CARD_ATOM[key], _amount(raw_value, default=1.0))


def _apply_next_turn_payload(summary: dict[str, float], payload: Mapping[str, Any]) -> None:
    _add(summary, "draw_next_turn", _float(payload.get("draw", payload.get("draw_cards"))))
    _add(summary, "energy_next_turn", _float(payload.get("energy", payload.get("gain_energy"))))
    if bool(payload.get("retain_hand")):
        summary["retain"] = max(summary["retain"], 1.0)


def _apply_timing_payload(summary: dict[str, float], payload: Mapping[str, Any]) -> None:
    trigger = _normalized_id(str(payload.get("trigger", "")))
    duration = _normalized_id(str(payload.get("duration", "")))
    if trigger in {"turn_start", "start_turn"}:
        summary["start_turn"] = max(summary["start_turn"], 1.0)
    elif trigger in {"turn_end", "end_turn"}:
        summary["end_turn"] = max(summary["end_turn"], 1.0)
    elif trigger in {"combat_start", "start_combat", "start_of_combat"}:
        summary["start_combat"] = max(summary["start_combat"], 1.0)
    elif trigger in {"combat_end", "end_combat"}:
        summary["end_combat"] = max(summary["end_combat"], 1.0)
    if _float(payload.get("delay")):
        summary["delayed"] = max(summary["delayed"], _float(payload.get("delay")))
    if duration and duration not in {"once", "turn"}:
        summary["repeating"] = max(summary["repeating"], 1.0)
    if _float(payload.get("period", payload.get("every"))):
        period = _float(payload.get("period", payload.get("every")))
        summary["periodic"] = max(summary["periodic"], period)
    if bool(payload.get("repeat")):
        summary["repeating"] = max(summary["repeating"], 1.0)


def _apply_card_flags(summary: dict[str, float], values: Mapping[str, Any]) -> None:
    for key in ("retain", "retained", "retain_once", "temporary_retain"):
        if _truthy(values.get(key)):
            summary["retain"] = max(summary["retain"], 1.0)
    for key in ("innate", "eternal", "unplayable", "ethereal"):
        if _truthy(values.get(key)):
            summary[key] = max(summary[key], 1.0)
    keywords = tuple(_normalized_id(str(keyword)) for keyword in _sequence(values.get("keywords")))
    for keyword in keywords:
        if keyword in {"retain", "retained"}:
            summary["retain"] = max(summary["retain"], 1.0)
        elif keyword in {"innate", "eternal", "unplayable", "ethereal"}:
            summary[keyword] = max(summary[keyword], 1.0)


def _is_next_turn_payload(keys: set[str], payload: Mapping[str, Any]) -> bool:
    if "next_turn" in keys:
        return False
    return _normalized_id(str(payload.get("trigger", ""))) in {"turn_start", "next_turn"} or bool(
        payload.get("next_turn")
    )


def _is_timing_payload(keys: set[str], payload: Mapping[str, Any]) -> bool:
    return bool(keys & {"trigger", "duration", "delay", "period", "every", "repeat"}) or bool(
        payload.get("combat_trigger")
    )


def _playable_value(card: Mapping[str, Any], atoms: Mapping[str, float]) -> float:
    if "playable" in card:
        return _bool_float(card.get("playable"))
    return 0.0 if _float(atoms.get("unplayable")) else 1.0


def _add(summary: dict[str, float], key: str, amount: float) -> None:
    if key in summary and amount:
        summary[key] += amount


def _status_key(value: str) -> str:
    normalized = _normalized_id(value)
    return _STATUS_ALIASES.get(normalized, normalized)


def _amount(value: object, *, default: float) -> float:
    if isinstance(value, Mapping):
        for key in ("amount", "value", "count"):
            if key in value:
                return _float(value.get(key))
        return default
    if isinstance(value, bool):
        return default if value else 0.0
    number = _float(value)
    return number if number else default


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _float(value: object) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _bool_float(value: object) -> float:
    return 1.0 if _truthy(value) else 0.0


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "retain"}
    return bool(value)


def _normalized_id(value: str) -> str:
    normalized = value.strip().lower().replace("'", "")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def _hash_bucket(value: str, bucket_count: int) -> int:
    if bucket_count <= 1:
        return 0
    total = 0
    for character in value:
        total = ((total * 33) + ord(character)) % bucket_count
    return total
