"""Deck-context semantic helpers for agent action descriptors.

This module deliberately exposes features, not decisions.  The values here are
inputs a policy can learn from: current need, duplicate pressure, combo overlap,
and opportunity costs around shops, potions, and removals.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from sts2sim.mechanics.semantics import (
    action_mechanic_profile,
    card_mechanic_profile,
    potion_mechanic_profile,
    relic_mechanic_profile,
)

SYNERGY_VALUE_KEYS: tuple[str, ...] = (
    "frontload",
    "scaling",
    "block",
    "draw",
    "poison",
    "exhaust",
    "retain",
    "discard_payoff",
    "summon",
    "orb",
    "forge",
    "max_hp_payoff",
    "potion_synergy",
    "relic_synergy",
    "improves_current_need",
    "duplicates_existing_engine",
    "adds_bloat",
    "enables_combo",
    "conflicts_with_plan",
    "gold_pressure",
    "potion_slot_pressure",
    "card_remove_value",
    "future_shop_usefulness",
)

_ENGINE_VALUE_KEYS: tuple[str, ...] = (
    "frontload",
    "scaling",
    "block",
    "draw",
    "poison",
    "exhaust",
    "retain",
    "discard_payoff",
    "summon",
    "orb",
    "forge",
    "max_hp_payoff",
    "potion_synergy",
    "relic_synergy",
)

_BASIC_CARD_IDS = {"strike", "defend"}
_REMOVE_VALUE_IDS = _BASIC_CARD_IDS | {
    "curse",
    "curse_of_the_bell",
    "doubt",
    "injury",
    "shame",
    "wound",
}
_PLAN_KEYS = ("plan", "strategy_plan", "agent_plan", "policy_plan")


def action_synergy_profile(
    state_payload: Mapping[str, Any],
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    """Return reusable synergy/context features for one action descriptor.

    The returned shape is JSON-friendly and deterministic:
    ``values`` carries all numeric values in ``SYNERGY_VALUE_KEYS`` order,
    ``tags`` explains activated semantic buckets, and grouped dictionaries keep
    deck-context and opportunity-cost fields easy to inspect.
    """

    candidate = _candidate_profile(descriptor)
    candidate_values = _mapping(candidate.get("values"))
    candidate_tags = tuple(str(tag) for tag in _sequence(candidate.get("tags")))

    deck_profiles = [_card_profile(_mapping(card)) for card in _deck_cards(state_payload)]
    deck_engine = _sum_engine_values(deck_profiles)
    candidate_engine = _engine_values(candidate_values, candidate_tags, descriptor)

    potion = _mapping(descriptor.get("potion"))

    content_id = _descriptor_content_id(descriptor)
    card_gain = _float(candidate_values.get("card_gain"))
    relic_gain = _float(candidate_values.get("relic_gain"))
    potion_gain = _float(candidate_values.get("potion_gain"))
    price = _descriptor_price(descriptor)
    gold = _player_gold(state_payload)
    deck_cards = tuple(_deck_cards(state_payload))
    deck_ids = Counter(_normalized_id(_card_id(card_map)) for card_map in deck_cards)
    duplicate_count = deck_ids.get(_normalized_id(content_id), 0)

    improves_current_need = _improves_current_need(state_payload, candidate_engine)
    duplicates_existing_engine = _duplicates_existing_engine(
        candidate_engine,
        deck_engine,
        duplicate_count=duplicate_count,
        card_gain=card_gain,
        relic_gain=relic_gain,
        potion_gain=potion_gain,
    )
    enables_combo = _enables_combo(candidate_engine, deck_engine)
    adds_bloat = _adds_bloat(
        candidate_engine,
        improves_current_need=improves_current_need,
        duplicates_existing_engine=duplicates_existing_engine,
        enables_combo=enables_combo,
        card_gain=card_gain,
        deck_size=len(deck_cards),
    )
    conflicts_with_plan = _conflicts_with_plan(
        state_payload,
        descriptor,
        candidate_engine,
        adds_bloat=adds_bloat,
        gold=gold,
        price=price,
    )
    gold_pressure = _gold_pressure(gold=gold, price=price, descriptor=descriptor)
    potion_slot_pressure = _potion_slot_pressure(
        state_payload,
        descriptor,
        potion_gain=potion_gain,
        potion=potion,
    )
    card_remove_value = _card_remove_value(state_payload, descriptor, candidate_values)
    future_shop_usefulness = _future_shop_usefulness(state_payload, descriptor, price=price)

    values = {
        **candidate_engine,
        "improves_current_need": improves_current_need,
        "duplicates_existing_engine": duplicates_existing_engine,
        "adds_bloat": adds_bloat,
        "enables_combo": enables_combo,
        "conflicts_with_plan": conflicts_with_plan,
        "gold_pressure": gold_pressure,
        "potion_slot_pressure": potion_slot_pressure,
        "card_remove_value": card_remove_value,
        "future_shop_usefulness": future_shop_usefulness,
    }
    values = {key: _clamp_nonnegative(_float(values.get(key))) for key in SYNERGY_VALUE_KEYS}

    tags = _synergy_tags(
        descriptor,
        candidate_tags,
        candidate_values,
        values,
        content_id=content_id,
        duplicate_count=duplicate_count,
    )
    deck_context = {
        "improves_current_need": values["improves_current_need"],
        "duplicates_existing_engine": values["duplicates_existing_engine"],
        "adds_bloat": values["adds_bloat"],
        "enables_combo": values["enables_combo"],
        "conflicts_with_plan": values["conflicts_with_plan"],
    }
    opportunity_cost = {
        "gold_pressure": values["gold_pressure"],
        "potion_slot_pressure": values["potion_slot_pressure"],
        "card_remove_value": values["card_remove_value"],
        "future_shop_usefulness": values["future_shop_usefulness"],
    }
    return {
        "values": values,
        "tags": tags,
        "deck_context": deck_context,
        "opportunity_cost": opportunity_cost,
        "content_id": content_id,
    }


def profile_value_vector(profile: Mapping[str, Any]) -> list[float]:
    """Return fixed-order numeric synergy values."""

    values = _mapping(profile.get("values"))
    return [_float(values.get(key)) for key in SYNERGY_VALUE_KEYS]


def _candidate_profile(descriptor: Mapping[str, Any]) -> Mapping[str, Any]:
    profile = action_mechanic_profile(descriptor)
    values = dict(_mapping(profile.get("values")))
    tags = list(str(tag) for tag in _sequence(profile.get("tags")))

    card = _mapping(descriptor.get("card"))
    if card and not values:
        card_profile = card_mechanic_profile(card).as_dict()
        values.update(_mapping(card_profile.get("values")))
        tags.extend(str(tag) for tag in _sequence(card_profile.get("tags")))

    item = _mapping(descriptor.get("item"))
    item_id = str(item.get("item_id", ""))
    kind = str(item.get("kind", ""))
    if item_id and kind in {"card", "colorless_card"}:
        card_profile = card_mechanic_profile({"card_id": item_id}).as_dict()
        values = _sum_values(values, _mapping(card_profile.get("values")))
        tags.extend(str(tag) for tag in _sequence(card_profile.get("tags")))
    elif item_id and kind == "relic":
        relic_profile = relic_mechanic_profile(item_id).as_dict()
        values = _sum_values(values, _mapping(relic_profile.get("values")))
        tags.extend(str(tag) for tag in _sequence(relic_profile.get("tags")))
    elif item_id and kind == "potion":
        potion_profile = potion_mechanic_profile(item_id).as_dict()
        values = _sum_values(values, _mapping(potion_profile.get("values")))
        tags.extend(str(tag) for tag in _sequence(potion_profile.get("tags")))

    reward = _mapping(descriptor.get("reward_choice"))
    reward_id = str(reward.get("content_id", ""))
    reward_kind = str(reward.get("kind", ""))
    if reward_id and reward_kind == "card":
        card_profile = card_mechanic_profile({"card_id": reward_id}).as_dict()
        values = _sum_values(values, _mapping(card_profile.get("values")))
        tags.extend(str(tag) for tag in _sequence(card_profile.get("tags")))
    elif reward_id and reward_kind == "relic":
        relic_profile = relic_mechanic_profile(reward_id).as_dict()
        values = _sum_values(values, _mapping(relic_profile.get("values")))
        tags.extend(str(tag) for tag in _sequence(relic_profile.get("tags")))
    elif reward_id and reward_kind == "potion":
        potion_profile = potion_mechanic_profile(reward_id).as_dict()
        values = _sum_values(values, _mapping(potion_profile.get("values")))
        tags.extend(str(tag) for tag in _sequence(potion_profile.get("tags")))

    return {"values": values, "tags": _unique(tags)}


def _card_profile(card: Mapping[str, Any]) -> Mapping[str, Any]:
    return card_mechanic_profile(card).as_dict()


def _engine_values(
    values: Mapping[str, Any],
    tags: Iterable[str],
    descriptor: Mapping[str, Any],
) -> dict[str, float]:
    tag_set = {_normalized_id(tag) for tag in tags}
    damage = _float(values.get("damage")) + _float(values.get("aoe_damage"))
    scaling = (
        _float(values.get("strength"))
        + _float(values.get("dexterity"))
        + _float(values.get("focus"))
        + _float(values.get("orb_slot_delta"))
        + _float(values.get("repeating_effect"))
        + _float(values.get("periodic_effect"))
    )
    orb = (
        _float(values.get("orb_channel"))
        + _float(values.get("orb_evoke"))
        + _float(values.get("orb_slot_delta"))
        + _float(values.get("focus"))
    )
    potion_synergy = (
        _float(values.get("potion_gain"))
        + max(0.0, _float(values.get("potion_slot_delta")))
        + _float(values.get("potion_loss")) * -0.25
    )
    if "potion" in tag_set or any(tag.startswith("potion") for tag in tag_set):
        potion_synergy = max(potion_synergy, 1.0)

    relic_synergy = _float(values.get("relic_gain"))
    if "relic" in tag_set or any(tag.startswith("relic") for tag in tag_set):
        relic_synergy = max(relic_synergy, 1.0)

    descriptor_text = _descriptor_text(descriptor)
    return {
        "frontload": damage + _float(values.get("block")),
        "scaling": scaling,
        "block": _float(values.get("block")),
        "draw": _float(values.get("draw")),
        "poison": _float(values.get("poison")),
        "exhaust": _float(values.get("exhaust")) + float("exhaust" in tag_set),
        "retain": _float(values.get("retain")) + float("retain" in tag_set),
        "discard_payoff": _float(values.get("discard")) + _text_signal(descriptor_text, "discard"),
        "summon": _float(values.get("summon")),
        "orb": orb,
        "forge": _float(values.get("forge")),
        "max_hp_payoff": max(0.0, _float(values.get("max_hp_delta")))
        + _text_signal(descriptor_text, "max_hp"),
        "potion_synergy": max(0.0, potion_synergy),
        "relic_synergy": max(0.0, relic_synergy),
    }


def _sum_engine_values(profiles: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    totals = {key: 0.0 for key in _ENGINE_VALUE_KEYS}
    for profile in profiles:
        values = _mapping(profile.get("values"))
        tags = tuple(str(tag) for tag in _sequence(profile.get("tags")))
        engine = _engine_values(values, tags, {})
        for key, value in engine.items():
            totals[key] += value
    return totals


def _improves_current_need(
    state_payload: Mapping[str, Any],
    candidate_engine: Mapping[str, float],
) -> float:
    incoming = _incoming_damage(state_payload)
    player = _active_player(state_payload)
    current_block = _float(player.get("block"))
    hp = _float(_first_present(player, "hp", "current_hp"))
    max_hp = max(1.0, _float(player.get("max_hp")))
    threatened = max(0.0, incoming - current_block)
    if threatened > 0 and _float(candidate_engine.get("block")) > 0:
        return min(1.0, _float(candidate_engine.get("block")) / threatened)
    if hp / max_hp < 0.4 and _float(candidate_engine.get("block")) > 0:
        return 0.5
    alive_hp = _alive_monster_hp(state_payload)
    if alive_hp > 0 and _float(candidate_engine.get("frontload")) > 0:
        return min(1.0, _float(candidate_engine.get("frontload")) / alive_hp)
    return 0.0


def _duplicates_existing_engine(
    candidate_engine: Mapping[str, float],
    deck_engine: Mapping[str, float],
    *,
    duplicate_count: int,
    card_gain: float,
    relic_gain: float,
    potion_gain: float,
) -> float:
    if duplicate_count > 0 and card_gain > 0:
        return float(min(3, duplicate_count))
    if card_gain <= 0 and relic_gain <= 0 and potion_gain <= 0:
        return 0.0
    overlap = 0.0
    for key in _ENGINE_VALUE_KEYS:
        if _float(candidate_engine.get(key)) > 0 and _float(deck_engine.get(key)) > 0:
            overlap += 1.0
    return min(3.0, overlap)


def _adds_bloat(
    candidate_engine: Mapping[str, float],
    *,
    improves_current_need: float,
    duplicates_existing_engine: float,
    enables_combo: float,
    card_gain: float,
    deck_size: int,
) -> float:
    if card_gain <= 0:
        return 0.0
    pressure = 1.0 if deck_size >= 15 else 0.5 if deck_size >= 10 else 0.25
    if improves_current_need > 0 or enables_combo > 0:
        pressure *= 0.5
    if duplicates_existing_engine > 0:
        pressure += min(1.0, duplicates_existing_engine / 3.0)
    if _float(candidate_engine.get("draw")) > 0:
        pressure *= 0.75
    return min(2.0, pressure)


def _enables_combo(
    candidate_engine: Mapping[str, float],
    deck_engine: Mapping[str, float],
) -> float:
    pairings = (
        ("exhaust", "draw"),
        ("draw", "discard_payoff"),
        ("discard_payoff", "draw"),
        ("poison", "scaling"),
        ("orb", "scaling"),
        ("summon", "scaling"),
        ("forge", "frontload"),
        ("max_hp_payoff", "frontload"),
        ("potion_synergy", "relic_synergy"),
    )
    enabled = 0.0
    for candidate_key, deck_key in pairings:
        candidate_present = _float(candidate_engine.get(candidate_key)) > 0
        deck_present = _float(deck_engine.get(deck_key)) > 0
        if candidate_present and deck_present:
            enabled += 1.0
    return min(3.0, enabled)


def _conflicts_with_plan(
    state_payload: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    candidate_engine: Mapping[str, float],
    *,
    adds_bloat: float,
    gold: float,
    price: float,
) -> float:
    plan_terms = _plan_terms(state_payload)
    conflicts = 0.0
    if "avoid_deck_bloat" in plan_terms or "deck_bloat" in plan_terms:
        conflicts += float(adds_bloat > 0)
    if "save_gold" in plan_terms or "future_shop" in plan_terms:
        conflicts += float(price > 0 and price > gold * 0.5)
    if "need_block" in plan_terms:
        conflicts += float(
            _float(candidate_engine.get("block")) <= 0 and _card_gain(descriptor) > 0
        )
    if "need_damage" in plan_terms or "frontload_damage" in plan_terms:
        conflicts += float(
            _float(candidate_engine.get("frontload")) <= _float(candidate_engine.get("block"))
            and _card_gain(descriptor) > 0
        )
    return min(3.0, conflicts)


def _gold_pressure(*, gold: float, price: float, descriptor: Mapping[str, Any]) -> float:
    if price <= 0:
        return 0.0
    if gold <= 0:
        return 1.0
    paid_fraction = price / max(1.0, gold)
    unaffordable = float(price > gold)
    return min(2.0, paid_fraction + unaffordable)


def _potion_slot_pressure(
    state_payload: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    *,
    potion_gain: float,
    potion: Mapping[str, Any],
) -> float:
    if potion_gain <= 0 and not potion:
        return 0.0
    potions = [str(item) for item in _sequence(state_payload.get("potions")) if str(item)]
    capacity = _potion_capacity(state_payload)
    if capacity <= 0:
        return 0.0
    if potion_gain > 0:
        return min(1.0, max(0.0, len(potions) + potion_gain - capacity))
    if str(descriptor.get("type", "")) in {"use_potion", "discard_potion"}:
        return 0.0
    return 0.0


def _card_remove_value(
    state_payload: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    candidate_values: Mapping[str, Any],
) -> float:
    if _float(candidate_values.get("card_remove")) <= 0:
        return 0.0
    removable_count = 0
    for card in _deck_cards(state_payload):
        card_id = _normalized_id(_card_id(card))
        card_type = _normalized_id(str(card.get("type", "")))
        if card_id in _REMOVE_VALUE_IDS or card_type == "curse":
            removable_count += 1
    return float(min(3, removable_count))


def _future_shop_usefulness(
    state_payload: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    *,
    price: float,
) -> float:
    gold_after = max(0.0, _player_gold(state_payload) - price)
    path = _mapping(_mapping(descriptor.get("node")).get("path"))
    future_shops = max(
        _float(path.get("future_shops_max")),
        _float(path.get("max_shops")),
        _float(_mapping(state_payload.get("path")).get("future_shops_max")),
    )
    if future_shops <= 0:
        future_shops = 1.0 if gold_after >= 150 else 0.0
    return min(2.0, future_shops * min(1.0, gold_after / 150.0))


def _synergy_tags(
    descriptor: Mapping[str, Any],
    candidate_tags: Iterable[str],
    candidate_values: Mapping[str, Any],
    values: Mapping[str, float],
    *,
    content_id: str,
    duplicate_count: int,
) -> list[str]:
    tags = [f"synergy:{key}" for key in _ENGINE_VALUE_KEYS if _float(values.get(key)) > 0]
    tags.extend(
        f"context:{key}"
        for key in (
            "improves_current_need",
            "duplicates_existing_engine",
            "adds_bloat",
            "enables_combo",
            "conflicts_with_plan",
        )
        if _float(values.get(key)) > 0
    )
    tags.extend(
        f"opportunity:{key}"
        for key in (
            "gold_pressure",
            "potion_slot_pressure",
            "card_remove_value",
            "future_shop_usefulness",
        )
        if _float(values.get(key)) > 0
    )
    if content_id:
        tags.append(f"content:{_normalized_id(content_id)}")
    if duplicate_count:
        tags.append("deck:duplicate_content")
    if _card_gain(descriptor) > 0:
        tags.append("deck:adds_card")
    if _float(candidate_values.get("gold_delta")) < 0:
        tags.append("cost:spends_gold")
    tags.extend(str(tag) for tag in candidate_tags)
    return _unique(tags)


def _deck_cards(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(_mapping(card) for card in _sequence(payload.get("master_deck")) if _mapping(card))


def _active_player(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    combat = _mapping(payload.get("combat"))
    combat_player = _mapping(combat.get("player"))
    if combat_player:
        return combat_player
    return _mapping(payload.get("player"))


def _incoming_damage(payload: Mapping[str, Any]) -> float:
    combat = _mapping(payload.get("combat"))
    incoming = _float(combat.get("incoming_damage"))
    if incoming:
        return incoming
    total = 0.0
    for monster in _sequence(combat.get("monsters")):
        monster_map = _mapping(monster)
        hp = _float(_first_present(monster_map, "hp", "current_hp"))
        if hp <= 0:
            continue
        multiplier = max(1.0, _float(monster_map.get("intent_hits")))
        total += _float(monster_map.get("intent_damage")) * multiplier
    return total


def _alive_monster_hp(payload: Mapping[str, Any]) -> float:
    total = 0.0
    for monster in _sequence(_mapping(payload.get("combat")).get("monsters")):
        monster_map = _mapping(monster)
        total += max(0.0, _float(_first_present(monster_map, "hp", "current_hp")))
    return total


def _descriptor_content_id(descriptor: Mapping[str, Any]) -> str:
    for container_key, id_keys in (
        ("card", ("card_id", "id")),
        ("relic", ("relic_id", "id")),
        ("potion", ("potion_id", "id")),
        ("item", ("item_id", "content_id", "id")),
        ("reward_choice", ("content_id", "id")),
    ):
        container = _mapping(descriptor.get(container_key))
        for key in id_keys:
            value = str(container.get(key, ""))
            if value:
                return value
    return ""


def _descriptor_price(descriptor: Mapping[str, Any]) -> float:
    item = _mapping(descriptor.get("item"))
    if item:
        return max(0.0, _float(item.get("price")))
    values = _mapping(_mapping(descriptor.get("mechanics")).get("values"))
    return max(0.0, -_float(values.get("gold_delta")))


def _card_gain(descriptor: Mapping[str, Any]) -> float:
    values = _mapping(_mapping(descriptor.get("mechanics")).get("values"))
    if _float(values.get("card_gain")) > 0:
        return _float(values.get("card_gain"))
    reward = _mapping(descriptor.get("reward_choice"))
    item = _mapping(descriptor.get("item"))
    if str(reward.get("kind", "")) == "card" or str(item.get("kind", "")) in {
        "card",
        "colorless_card",
    }:
        return 1.0
    return 0.0


def _player_gold(payload: Mapping[str, Any]) -> float:
    return _float(_mapping(payload.get("player")).get("gold"))


def _potion_capacity(payload: Mapping[str, Any]) -> float:
    explicit = _float(_first_present(payload, "potion_slots", "max_potions"))
    if explicit:
        return explicit
    player = _mapping(payload.get("player"))
    explicit = _float(_first_present(player, "potion_slots", "max_potions"))
    if explicit:
        return explicit
    relic_ids = {_normalized_id(str(relic)) for relic in _sequence(payload.get("relics"))}
    return 2.0 + (2.0 if "potion_belt" in relic_ids else 0.0)


def _plan_terms(payload: Mapping[str, Any]) -> set[str]:
    terms: set[str] = set()
    for key in _PLAN_KEYS:
        terms.update(_flatten_terms(payload.get(key)))
    return terms


def _flatten_terms(value: object) -> set[str]:
    if isinstance(value, str):
        return {_normalized_id(value)}
    if isinstance(value, Mapping):
        terms: set[str] = set()
        for key, item in value.items():
            if isinstance(item, bool) and item:
                terms.add(_normalized_id(str(key)))
            else:
                terms.update(_flatten_terms(item))
        return terms
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sequence_terms: set[str] = set()
        for item in value:
            sequence_terms.update(_flatten_terms(item))
        return sequence_terms
    return set()


def _descriptor_text(descriptor: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for value in descriptor.values():
        if isinstance(value, Mapping):
            parts.extend(str(item) for item in value.values())
    return " ".join(parts).lower()


def _text_signal(text: str, key: str) -> float:
    if key == "discard":
        return float("discard" in text and ("whenever" in text or "gain" in text or "draw" in text))
    if key == "max_hp":
        return float("max hp" in text or "max_hp" in text)
    return 0.0


def _card_id(card: Mapping[str, Any]) -> str:
    return str(_first_present(card, "card_id", "id") or "")


def _sum_values(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(left)
    for key, value in right.items():
        values[str(key)] = _float(values.get(str(key))) + _float(value)
    return values


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _sequence(value: object) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _first_present(mapping: Mapping[str, Any], *keys: str) -> object | None:
    for key in keys:
        value: object | None = mapping.get(key)
        if value is not None:
            return value
    return None


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


def _clamp_nonnegative(value: float) -> float:
    return max(0.0, value)


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _normalized_id(value: str) -> str:
    normalized = value.strip().lower().replace("'", "")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")
