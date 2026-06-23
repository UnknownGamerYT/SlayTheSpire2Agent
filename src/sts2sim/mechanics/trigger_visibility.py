"""Deterministic visibility features for live triggers and counters.

The helpers in this module expose observable trigger state for learning
features.  They intentionally summarize timing, counters, and capacity without
ranking whether any mechanic is strategically good.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from sts2sim.mechanics.potions import potion_capacity as compute_potion_capacity
from sts2sim.mechanics.relics import (
    DEFAULT_RELIC_HOOK_RULES,
    DEFAULT_RELIC_POTION_SLOT_MODIFIERS,
    DEFAULT_RELIC_PRICE_MODIFIERS,
    RelicHook,
)
from sts2sim.mechanics.reward_triggers import DEFAULT_REWARD_MODIFIERS

TRIGGER_VISIBILITY_KEYS: tuple[str, ...] = (
    "trigger_count",
    "start_combat_count",
    "start_turn_count",
    "end_turn_count",
    "end_combat_count",
    "on_play_count",
    "on_attack_count",
    "on_skill_count",
    "on_power_count",
    "on_discard_count",
    "on_exhaust_count",
    "on_damage_taken_count",
    "on_kill_count",
    "shop_modifier_count",
    "reward_modifier_count",
    "potion_modifier_count",
    "counter_total",
    "counter_progress_max",
    "turns_until_next_effect_min",
    "remaining_uses_total",
    "already_triggered_count",
    "once_per_combat_count",
    "repeating_count",
    "periodic_count",
    "delayed_count",
    "next_shop_modifier",
    "next_reward_modifier",
    "next_combat_modifier",
    "potion_slots",
    "potion_slots_filled",
    "potion_slots_empty",
    "potion_capacity_pressure",
)

_ZERO_SUMMARY: dict[str, float] = {key: 0.0 for key in TRIGGER_VISIBILITY_KEYS}
_START_COMBAT = {"combat_start", "start_combat", "start_of_combat", "at_combat_start"}
_START_TURN = {"turn_start", "start_turn", "start_of_turn", "at_turn_start"}
_END_TURN = {"turn_end", "end_turn", "end_of_turn", "at_turn_end"}
_END_COMBAT = {"combat_end", "end_combat", "end_of_combat", "at_combat_end"}
_ON_PLAY = {"card_played", "play_card", "on_play", "on_card_played", "card_play"}
_ON_ATTACK = {"attack_played", "on_attack", "attack", "play_attack"}
_ON_SKILL = {"skill_played", "on_skill", "skill", "play_skill"}
_ON_POWER = {"power_played", "on_power", "power", "play_power"}
_ON_DISCARD = {"card_discarded", "discard", "on_discard"}
_ON_EXHAUST = {"card_exhausted", "exhaust", "on_exhaust"}
_ON_DAMAGE_TAKEN = {"damage_taken", "on_damage_taken", "hp_loss", "lose_hp"}
_ON_KILL = {"enemy_killed", "kill", "on_kill", "monster_killed"}
_REPEATING_DURATIONS = {"combat", "battle", "forever", "persistent", "turns", "encounter"}
_ONCE_DURATIONS = {"once", "once_per_combat", "combat_once", "first"}
_SHOP_TERMS = ("shop", "merchant", "price", "discount", "sale", "courier")
_REWARD_TERMS = ("reward", "card_choice", "card_reward", "relic_reward", "extra_card")
_POTION_TERMS = ("potion", "alchemy", "alchemical", "slot", "brew")
_COMBAT_TERMS = ("combat", "battle", "fight", "turn", "attack", "skill", "power")
_TRIGGER_TERMS = (
    "trigger",
    "start",
    "end",
    "turn",
    "combat",
    "play",
    "attack",
    "skill",
    "power",
    "discard",
    "exhaust",
    "damage",
    "kill",
    "every",
    "period",
    "delay",
    "remaining",
    "counter",
)


def trigger_visibility_summary(payload: Mapping[str, Any]) -> dict[str, float]:
    """Return fixed-key numeric visibility for triggers, counters, and slots."""

    summary = dict(_ZERO_SUMMARY)
    if not isinstance(payload, Mapping):
        return summary

    slots = potion_slot_summary(payload)
    summary["potion_slots"] = _float(slots.get("capacity"))
    summary["potion_slots_filled"] = _float(slots.get("filled"))
    summary["potion_slots_empty"] = _float(slots.get("empty"))
    capacity = _float(slots.get("capacity"))
    summary["potion_capacity_pressure"] = (
        round(_float(slots.get("filled")) / capacity, 4) if capacity > 0 else 0.0
    )

    _apply_known_relic_visibility(summary, _relic_ids(payload))
    _apply_reward_modifier_visibility(summary, _relic_ids(payload))

    for trigger in _visible_triggers(payload):
        _add_trigger(summary, trigger)

    for item in _keyword_items(payload):
        _add_keyword_visibility(summary, item)

    for counters in _counter_mappings(payload):
        for value in counters.values():
            amount = _float(value)
            if amount:
                summary["counter_total"] += amount
                summary["counter_progress_max"] = max(summary["counter_progress_max"], amount)
                summary["already_triggered_count"] += 1.0

    if summary["shop_modifier_count"]:
        summary["next_shop_modifier"] = max(
            summary["next_shop_modifier"],
            summary["shop_modifier_count"],
        )
    if summary["reward_modifier_count"]:
        summary["next_reward_modifier"] = max(
            summary["next_reward_modifier"],
            summary["reward_modifier_count"],
        )
    if any(summary[key] for key in _COMBAT_COUNT_KEYS):
        summary["next_combat_modifier"] = max(summary["next_combat_modifier"], 1.0)

    return {key: float(summary.get(key, 0.0)) for key in TRIGGER_VISIBILITY_KEYS}


def trigger_visibility_vector(summary: Mapping[str, Any]) -> tuple[float, ...]:
    """Return fixed-order trigger visibility values."""

    return tuple(_float(summary.get(key)) for key in TRIGGER_VISIBILITY_KEYS)


def potion_slot_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return visible potion slot occupancy with deterministic empty slots."""

    if not isinstance(payload, Mapping):
        return {"capacity": 0, "filled": 0, "empty": 0, "potions": []}

    potions = _potion_ids(payload)
    capacity = _potion_capacity(payload, filled=len(potions))
    filled = len(potions)
    empty = max(0, capacity - filled)
    slots: list[dict[str, Any]] = [
        {"slot_index": index, "id": potion_id, "empty": False}
        for index, potion_id in enumerate(potions)
    ]
    slots.extend(
        {"slot_index": index, "id": "", "empty": True}
        for index in range(filled, capacity)
    )
    return {"capacity": capacity, "filled": filled, "empty": empty, "potions": slots}


_COMBAT_COUNT_KEYS = (
    "start_combat_count",
    "start_turn_count",
    "end_turn_count",
    "end_combat_count",
    "on_play_count",
    "on_attack_count",
    "on_skill_count",
    "on_power_count",
    "on_discard_count",
    "on_exhaust_count",
    "on_damage_taken_count",
    "on_kill_count",
)


def _apply_known_relic_visibility(summary: dict[str, float], relic_ids: tuple[str, ...]) -> None:
    relic_set = set(relic_ids)
    hook_keys = {
        RelicHook.START_COMBAT: "start_combat_count",
        RelicHook.START_TURN: "start_turn_count",
        RelicHook.END_TURN: "end_turn_count",
        RelicHook.END_COMBAT: "end_combat_count",
    }
    for hook, key in hook_keys.items():
        count = sum(
            1
            for relic_id in relic_set
            if relic_id in DEFAULT_RELIC_HOOK_RULES.get(hook, {})
        )
        summary[key] += float(count)
        summary["trigger_count"] += float(count)

    shop_count = sum(1 for relic_id in relic_set if relic_id in DEFAULT_RELIC_PRICE_MODIFIERS)
    potion_count = sum(
        1 for relic_id in relic_set if relic_id in DEFAULT_RELIC_POTION_SLOT_MODIFIERS
    )
    summary["shop_modifier_count"] += float(shop_count)
    summary["potion_modifier_count"] += float(potion_count)


def _apply_reward_modifier_visibility(
    summary: dict[str, float],
    relic_ids: tuple[str, ...],
) -> None:
    relic_set = set(relic_ids)
    count = sum(1 for modifier in DEFAULT_REWARD_MODIFIERS if modifier.content_id in relic_set)
    summary["reward_modifier_count"] += float(count)


def _add_trigger(summary: dict[str, float], trigger: Mapping[str, Any]) -> None:
    summary["trigger_count"] += 1.0
    text = _terms_for_mapping(trigger)
    trigger_name = _normalized(_first_present(trigger, "trigger", "hook", "event", "kind", "type"))
    _add_timing(summary, trigger_name, text)

    delay = _first_number(trigger, "delay", "turn_delay", "turns_until_effect", "turns_until")
    if delay > 0:
        summary["delayed_count"] += 1.0
        _set_min(summary, "turns_until_next_effect_min", delay)
    target_turn = _first_number(trigger, "turn_number", "target_turn", "absolute_turn")
    current_turn = _first_number(trigger, "current_turn", "turn", "registered_turn")
    if target_turn > 0:
        _set_min(summary, "turns_until_next_effect_min", max(0.0, target_turn - current_turn))

    remaining_uses = _first_number(trigger, "remaining_uses", "uses", "charges")
    summary["remaining_uses_total"] += remaining_uses
    counter = _first_number(trigger, "counter", "next_counter", "count", "progress")
    period = _first_number(trigger, "period", "every", "interval", "threshold", "required")
    if counter:
        summary["counter_total"] += counter
        summary["counter_progress_max"] = max(summary["counter_progress_max"], counter)
    if period:
        summary["periodic_count"] += 1.0
        summary["counter_progress_max"] = max(summary["counter_progress_max"], period)

    duration = _normalized(trigger.get("duration"))
    if duration in _ONCE_DURATIONS or bool(trigger.get("once_per_combat")):
        summary["once_per_combat_count"] += 1.0
    if bool(trigger.get("already_triggered")) or bool(trigger.get("triggered")):
        summary["already_triggered_count"] += 1.0
    if bool(trigger.get("repeat")) or duration in _REPEATING_DURATIONS:
        summary["repeating_count"] += 1.0

    if any(term in text for term in _SHOP_TERMS):
        summary["shop_modifier_count"] += 1.0
    if any(term in text for term in _REWARD_TERMS):
        summary["reward_modifier_count"] += 1.0
    if any(term in text for term in _POTION_TERMS):
        summary["potion_modifier_count"] += 1.0


def _add_timing(summary: dict[str, float], trigger_name: str, text: str) -> None:
    terms = {trigger_name, *text.split()}
    if (
        terms & _START_COMBAT
        or _contains_any(text, _START_COMBAT)
        or ("start" in text and "combat" in text)
    ):
        summary["start_combat_count"] += 1.0
    if (
        terms & _START_TURN
        or _contains_any(text, _START_TURN)
        or "next_turn" in text
        or ("start" in text and "turn" in text)
    ):
        summary["start_turn_count"] += 1.0
    if terms & _END_TURN or _contains_any(text, _END_TURN) or ("end" in text and "turn" in text):
        summary["end_turn_count"] += 1.0
    if (
        terms & _END_COMBAT
        or _contains_any(text, _END_COMBAT)
        or ("end" in text and "combat" in text)
    ):
        summary["end_combat_count"] += 1.0
    if terms & _ON_PLAY or _contains_any(text, _ON_PLAY):
        summary["on_play_count"] += 1.0
    if terms & _ON_ATTACK or _contains_any(text, _ON_ATTACK):
        summary["on_attack_count"] += 1.0
    if terms & _ON_SKILL or _contains_any(text, _ON_SKILL):
        summary["on_skill_count"] += 1.0
    if terms & _ON_POWER or _contains_any(text, _ON_POWER):
        summary["on_power_count"] += 1.0
    if terms & _ON_DISCARD or _contains_any(text, _ON_DISCARD):
        summary["on_discard_count"] += 1.0
    if terms & _ON_EXHAUST or _contains_any(text, _ON_EXHAUST):
        summary["on_exhaust_count"] += 1.0
    if terms & _ON_DAMAGE_TAKEN or _contains_any(text, _ON_DAMAGE_TAKEN):
        summary["on_damage_taken_count"] += 1.0
    if terms & _ON_KILL or _contains_any(text, _ON_KILL):
        summary["on_kill_count"] += 1.0


def _add_keyword_visibility(summary: dict[str, float], item: Mapping[str, Any] | str) -> None:
    text = _terms_for_mapping(item) if isinstance(item, Mapping) else _normalized(item)
    if not text:
        return
    if any(term in text for term in _SHOP_TERMS):
        summary["shop_modifier_count"] += 1.0
    if any(term in text for term in _REWARD_TERMS):
        summary["reward_modifier_count"] += 1.0
    if any(term in text for term in _POTION_TERMS):
        summary["potion_modifier_count"] += 1.0
    if any(term in text for term in _COMBAT_TERMS):
        summary["next_combat_modifier"] = max(summary["next_combat_modifier"], 1.0)
    if any(term in text for term in _TRIGGER_TERMS):
        _add_timing(summary, "", text)
    counter = _first_number(_mapping(item), "counter", "count", "progress", "amount")
    if counter:
        summary["counter_total"] += counter
        summary["counter_progress_max"] = max(summary["counter_progress_max"], counter)


def _visible_triggers(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    combat = _mapping(payload.get("combat"))
    metadata = _mapping(combat.get("metadata"))
    top_metadata = _mapping(payload.get("metadata"))
    sources = (
        payload.get("timed_card_triggers"),
        payload.get("combat_triggers"),
        payload.get("triggers"),
        metadata.get("timed_card_triggers"),
        metadata.get("combat_triggers"),
        metadata.get("triggers"),
        top_metadata.get("timed_card_triggers"),
        top_metadata.get("combat_triggers"),
        top_metadata.get("triggers"),
    )
    triggers: list[Mapping[str, Any]] = []
    for source in sources:
        for item in _sequence(source):
            item_map = _mapping(item)
            if item_map:
                triggers.append(item_map)
    return tuple(triggers)


def _keyword_items(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any] | str, ...]:
    player = _mapping(payload.get("player"))
    combat = _mapping(payload.get("combat"))
    metadata = _mapping(combat.get("metadata"))
    shop = _mapping(payload.get("shop"))
    reward = _mapping(payload.get("reward"))
    items: list[Mapping[str, Any] | str] = []
    items.extend(_sequence(payload.get("relics")))
    items.extend(_sequence(payload.get("potions")))
    items.extend(_mapping(player.get("statuses")).keys())
    items.extend(_mapping(player.get("resources")).keys())
    items.extend(_sequence(player.get("powers")))
    items.extend(_mapping(combat.get("statuses")).keys())
    items.extend(_sequence(combat.get("powers")))
    items.extend(_sequence(combat.get("active_powers")))
    items.extend(_sequence(metadata.get("active_powers")))
    items.extend(_mapping(metadata.get("statuses")).keys())
    items.extend(_sequence(shop.get("modifiers")))
    items.extend(_sequence(reward.get("modifiers")))
    items.extend(_sequence(reward.get("reward_modifiers")))
    if shop:
        items.append(shop)
    if reward:
        items.append(reward)
    return tuple(items)


def _counter_mappings(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    combat = _mapping(payload.get("combat"))
    metadata = _mapping(combat.get("metadata"))
    flags = _mapping(payload.get("flags"))
    return tuple(
        mapping
        for mapping in (
            _mapping(payload.get("relic_counters")),
            _mapping(payload.get("counters")),
            _mapping(flags.get("relic_counters")),
            _mapping(metadata.get("relic_counters")),
            _mapping(metadata.get("combat_trigger_counters")),
            _mapping(metadata.get("combat_trigger_turn_counters")),
        )
        if mapping
    )


def _potion_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    player = _mapping(payload.get("player"))
    source = _sequence(payload.get("potions")) or _sequence(player.get("potions"))
    ids: list[str] = []
    for item in source:
        if item is None:
            continue
        if isinstance(item, Mapping) and (item.get("empty") is True or item.get("id") is None):
            continue
        potion_id = _content_id(item)
        if potion_id:
            ids.append(potion_id)
    return tuple(ids)


def _potion_capacity(payload: Mapping[str, Any], *, filled: int) -> int:
    player = _mapping(payload.get("player"))
    flags = _mapping(payload.get("flags"))
    for source in (payload, player, flags):
        value = _first_present(
            source,
            "potion_slots",
            "max_potion_slots",
            "max_potions",
            "potion_capacity",
        )
        capacity = _int(value)
        if capacity > 0:
            return max(capacity, filled)
    base_slots = _first_positive_int(
        payload,
        player,
        flags,
        keys=("base_potion_slots", "base_potion_capacity"),
    )
    ascension_level = _first_int(
        payload,
        player,
        flags,
        keys=("ascension", "ascension_level"),
    )
    bonus_slots = _first_int(
        payload,
        player,
        flags,
        keys=("bonus_potion_slots", "potion_slot_bonus"),
    )
    result = compute_potion_capacity(
        base_slots=base_slots or 3,
        ascension_level=ascension_level,
        relics=_relic_ids(payload),
        bonus_slots=bonus_slots,
        current_potions=tuple(potions["id"] for potions in _potion_slot_items(payload)),
    )
    return max(result.capacity, filled)


def _potion_slot_items(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple({"id": potion_id} for potion_id in _potion_ids(payload))


def _first_positive_int(
    *sources: Mapping[str, Any],
    keys: Sequence[str],
) -> int:
    for source in sources:
        for key in keys:
            value = _int(source.get(key))
            if value > 0:
                return value
    return 0


def _first_int(
    *sources: Mapping[str, Any],
    keys: Sequence[str],
) -> int:
    for source in sources:
        for key in keys:
            if key in source:
                return _int(source.get(key))
    return 0


def _relic_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    ids: list[str] = []
    seen: set[str] = set()
    for item in _sequence(payload.get("relics")):
        relic_id = _content_id(item)
        if relic_id and relic_id not in seen:
            seen.add(relic_id)
            ids.append(relic_id)
    return tuple(ids)


def _content_id(item: Any) -> str:
    if isinstance(item, Mapping):
        return _normalized(
            _first_present(
                item,
                "id",
                "relic_id",
                "potion_id",
                "content_id",
                "item_id",
            )
        )
    return _normalized(item)


def _terms_for_mapping(value: Mapping[str, Any]) -> str:
    terms: list[str] = []

    def walk(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                terms.append(_normalized(key))
                walk(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                walk(child)
        elif isinstance(item, str):
            terms.append(_normalized(item))

    walk(value)
    return " ".join(term for term in terms if term)


def _set_min(summary: dict[str, float], key: str, value: float) -> None:
    if value < 0:
        return
    current = summary.get(key, 0.0)
    summary[key] = value if current == 0.0 else min(current, value)


def _contains_any(text: str, needles: set[str]) -> bool:
    return any(needle and needle in text for needle in needles)


def _first_number(mapping: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        value = _float(mapping.get(key))
        if value:
            return value
    return 0.0


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _float(value: Any) -> float:
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


def _normalized(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    normalized = normalized.replace("'", "")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")
