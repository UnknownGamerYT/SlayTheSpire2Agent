"""Shared mechanic semantics for agent-facing observations and actions.

This module turns simulator content into a common vocabulary.  It does not tell
the agent whether something is strategically good; it exposes comparable facts
such as "damage", "block", "poison", "spend gold", or "gain relic" across
cards, relics, potions, events, shops, and rewards.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, cast

from sts2sim.mechanics.card_effects import normalize_card_spec
from sts2sim.mechanics.potions import normalize_potion_use
from sts2sim.mechanics.relic_combat import (
    CombatRelicHook,
    resolve_combat_relic_hook,
    supported_combat_relic_ids,
)
from sts2sim.mechanics.relics import (
    DEFAULT_RELIC_HOOK_RULES,
    DEFAULT_RELIC_POTION_SLOT_MODIFIERS,
    DEFAULT_RELIC_PRICE_MODIFIERS,
    RelicHook,
)

MECHANIC_VALUE_KEYS: tuple[str, ...] = (
    "damage",
    "aoe_damage",
    "block",
    "draw",
    "energy",
    "heal",
    "hp_delta",
    "max_hp_delta",
    "gold_delta",
    "card_gain",
    "card_remove",
    "card_upgrade",
    "card_transform",
    "relic_gain",
    "relic_loss",
    "potion_gain",
    "potion_loss",
    "potion_slot_delta",
    "status_self",
    "status_enemy",
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
    "orb_channel",
    "orb_evoke",
    "orb_slot_delta",
    "summon",
    "star",
    "forge",
    "discard",
    "exhaust",
    "retain",
    "cost_reduction",
    "shop_discount",
    "shop_restock",
    "randomness",
    "risk",
    "curse",
    "forced",
    "delayed_reward",
    "turn_delay",
    "turns_until_effect",
    "absolute_turn",
    "combat_delay",
    "remaining_uses",
    "start_turn_timing",
    "end_turn_timing",
    "start_combat_timing",
    "end_combat_timing",
    "current_turn_effect",
    "next_turn_effect",
    "repeating_effect",
    "periodic_effect",
)

MECHANIC_TAG_BUCKETS = 64

_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "eng"
_STATUS_VALUE_KEYS = {
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
}


@dataclass(frozen=True, slots=True)
class MechanicProfile:
    """A normalized, JSON-friendly summary of mechanical consequences."""

    values: Mapping[str, float] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    summary: tuple[str, ...] = ()
    content_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _clean_values(self.values))
        object.__setattr__(self, "tags", _unique_text(self.tags))
        object.__setattr__(self, "summary", _unique_text(self.summary))
        object.__setattr__(self, "content_ids", _unique_text(self.content_ids))

    def as_dict(self) -> dict[str, object]:
        return {
            "values": dict(self.values),
            "tags": list(self.tags),
            "summary": list(self.summary),
            "content_ids": list(self.content_ids),
        }


def profile_value_vector(profile: Mapping[str, Any] | MechanicProfile) -> tuple[float, ...]:
    """Return fixed-order mechanic values for observations and action features."""

    values: Mapping[str, Any]
    if isinstance(profile, MechanicProfile):
        values = profile.values
    else:
        values = _mapping(profile.get("values"))
    return tuple(_float(values.get(key)) for key in MECHANIC_VALUE_KEYS)


def action_mechanic_profile(descriptor: Mapping[str, Any]) -> dict[str, object]:
    """Return mechanics for one action descriptor."""

    action_type = str(descriptor.get("type", ""))
    card = _mapping(descriptor.get("card"))
    item = _mapping(descriptor.get("item"))
    potion = _mapping(descriptor.get("potion"))
    relic = _mapping(descriptor.get("relic"))
    reward = _mapping(descriptor.get("reward_choice"))
    event_option = _mapping(descriptor.get("event_option"))
    ancient_option = _mapping(descriptor.get("ancient_option"))

    profiles: list[MechanicProfile] = []

    if card:
        profiles.append(card_mechanic_profile(card))
    if potion:
        profiles.append(potion_mechanic_profile(str(potion.get("potion_id", ""))))
    if relic:
        profiles.append(relic_mechanic_profile(str(relic.get("relic_id", ""))))

    if reward:
        profiles.append(_reward_profile(reward))
    if item:
        profiles.append(_shop_item_profile(item))
    if event_option:
        profiles.append(_event_option_profile(event_option))
    if ancient_option:
        profiles.append(_ancient_option_profile(ancient_option))

    if action_type == "discard_potion" and potion:
        profiles.append(
            _profile(
                {"potion_loss": -1.0},
                tags=("discard_potion", "potion_slot_opens"),
                summary=(
                    f"Discard potion "
                    f"{_content_name('potions', str(potion.get('potion_id', '')))}",
                ),
            )
        )
    elif action_type == "use_potion" and potion:
        profiles.append(
            _profile({"potion_loss": -1.0}, tags=("consume_potion",), summary=("Consume potion",))
        )
    elif action_type == "throw_potion_at_merchant":
        profiles.append(
            _profile(
                {"gold_delta": 100.0, "potion_loss": -1.0},
                tags=("merchant_throw", "consume_potion", "gain_gold"),
                summary=("Throw Foul Potion at merchant for gold",),
            )
        )
    elif action_type == "smith":
        profiles.append(
            _profile(
                {"card_upgrade": 1.0},
                tags=("campfire", "upgrade_card"),
                summary=("Upgrade selected card",),
            )
        )
    elif action_type == "toke":
        profiles.append(
            _profile(
                {"card_remove": 1.0},
                tags=("campfire", "remove_card"),
                summary=("Remove selected card",),
            )
        )
    elif action_type == "rest":
        profiles.append(_profile({"heal": 1.0}, tags=("campfire", "heal"), summary=("Rest",)))
    elif action_type == "dig":
        profiles.append(
            _profile(
                {"relic_gain": 1.0, "randomness": 1.0},
                tags=("campfire", "gain_relic", "random_relic"),
                summary=("Dig for a relic",),
            )
        )
    elif action_type == "lift":
        profiles.append(
            _profile(
                {"strength": 1.0},
                tags=("campfire", "strength", "permanent_scaling"),
                summary=("Lift for Strength scaling",),
            )
        )
    elif action_type == "recall":
        profiles.append(
            _profile(tags=("campfire", "recall_key"), summary=("Recall the key",))
        )
    elif action_type == "choose_node":
        node = _mapping(descriptor.get("node"))
        kind = str(node.get("kind", ""))
        if kind:
            profiles.append(
                _profile(tags=(f"map:{kind}", kind), summary=(f"Choose {kind} room",))
            )

    combined = combine_profiles(profiles)
    return combined.as_dict()


def state_mechanic_profile(payload: Mapping[str, Any]) -> dict[str, object]:
    """Aggregate mechanic context from the current serialized state."""

    profiles: list[MechanicProfile] = []
    for relic_id in _sequence(payload.get("relics")):
        profiles.append(relic_mechanic_profile(str(relic_id)))
    for potion_id in _sequence(payload.get("potions")):
        profiles.append(potion_mechanic_profile(str(potion_id)))
    for card in _sequence(payload.get("master_deck")):
        profiles.append(card_mechanic_profile(_mapping(card)))
    profiles.extend(_active_timing_profiles(payload))

    player = _mapping(payload.get("player"))
    statuses = _mapping(player.get("statuses"))
    for status_id, amount in statuses.items():
        profiles.append(_status_timing_profile(str(status_id), amount))
        profiles.append(
            _profile(
                {_status_value_key(str(status_id)): _float(amount), "status_self": 1.0},
                tags=("owned_status", f"status:{_normalized_id(str(status_id))}"),
                summary=(f"Player status {status_id}: {_float(amount):g}",),
            )
        )
    combat = _mapping(payload.get("combat"))
    for monster in _sequence(combat.get("monsters")):
        monster_map = _mapping(monster)
        monster_id = str(monster_map.get("monster_id", ""))
        for status_id, amount in _mapping(monster_map.get("statuses")).items():
            profiles.append(_enemy_status_profile(str(status_id), amount, monster_id))

    return combine_profiles(profiles).as_dict()


def card_mechanic_profile(card: Mapping[str, Any]) -> MechanicProfile:
    """Return semantic mechanics for a visible or source-backed card."""

    card_id = str(card.get("card_id", card.get("id", "")))
    candidate = dict(card)
    source = _card_source(card_id)
    if not _mapping(candidate.get("effects")) and source is not None:
        with suppress(Exception):
            candidate = normalize_card_spec(source, card_library=_cached_cards())
    effects = _mapping(candidate.get("effects"))
    tags = ["card", f"card:{_normalized_id(card_id)}"] if card_id else ["card"]
    if card.get("type"):
        tags.append(f"card_type:{_normalized_id(str(card.get('type')))}")
    if card.get("exhausts"):
        tags.append("exhaust_on_play")
    if card.get("upgraded"):
        tags.append("upgraded")
    profile = _effects_profile(effects, target=str(card.get("target", candidate.get("target", ""))))
    return combine_profiles(
        (
            profile,
            _profile(
                tags=tuple(tags),
                summary=(f"Card {_content_name('cards', card_id)}",) if card_id else ("Card",),
                content_ids=(card_id,) if card_id else (),
            ),
        )
    )


def potion_mechanic_profile(potion_id: str) -> MechanicProfile:
    """Return semantic mechanics for a potion id."""

    normalized = _normalized_id(potion_id)
    if not normalized:
        return MechanicProfile()
    normalization = normalize_potion_use(normalized)
    profiles = [
        _profile(
            tags=("potion", f"potion:{normalized}"),
            summary=(f"Potion {_content_name('potions', normalized)}",),
            content_ids=(normalized,),
        )
    ]
    for effect in normalization.effects:
        profiles.append(
            _rule_profile(
                kind=effect.kind,
                amount=effect.amount,
                target=effect.target,
                status=effect.status,
                metadata=effect.metadata,
                source=f"potion:{normalized}",
            )
        )
    if normalization.unsupported:
        profiles.append(_profile(tags=("unsupported_potion_effect",)))
    return combine_profiles(profiles)


def relic_mechanic_profile(relic_id: str) -> MechanicProfile:
    """Return semantic mechanics for a relic id using pickup/shop/combat hooks."""

    normalized = _normalized_id(relic_id)
    if not normalized:
        return MechanicProfile()

    profiles: list[MechanicProfile] = [
        _profile(
            tags=("relic", f"relic:{normalized}"),
            summary=(f"Relic {_content_name('relics', normalized)}",),
            content_ids=(normalized,),
        )
    ]
    for hook, rules in DEFAULT_RELIC_HOOK_RULES.items():
        rule = rules.get(normalized)
        if rule is None:
            continue
        profiles.append(_relic_hook_rule_profile(normalized, hook, rule))

    price_modifier = DEFAULT_RELIC_PRICE_MODIFIERS.get(normalized)
    if price_modifier is not None:
        values: dict[str, float] = {}
        tags = ["shop_price_modifier", "shop_discount"]
        summary: list[str] = ["Modifies shop prices"]
        if price_modifier.multiplier_percent is not None:
            values["shop_discount"] = 100.0 - float(price_modifier.multiplier_percent)
            summary.append(f"Shop price multiplier {price_modifier.multiplier_percent}%")
        if price_modifier.fixed_price is not None:
            values["shop_discount"] = max(values.get("shop_discount", 0.0), 1.0)
            summary.append(f"Sets shop price to {price_modifier.fixed_price} gold")
        profiles.append(_profile(values, tags=tuple(tags), summary=tuple(summary)))

    potion_slots = DEFAULT_RELIC_POTION_SLOT_MODIFIERS.get(normalized, 0)
    if potion_slots:
        profiles.append(
            _profile(
                {"potion_slot_delta": float(potion_slots)},
                tags=("potion_capacity",),
                summary=(f"Gain {potion_slots} potion slots",),
            )
        )

    for combat_hook in CombatRelicHook:
        if normalized not in supported_combat_relic_ids(hook=combat_hook):
            continue
        profiles.append(_profile(tags=("combat_relic", f"timing:{combat_hook.value}")))
        resolution = resolve_combat_relic_hook(
            (normalized,),
            combat_hook,
            include_blockers=False,
        )
        if resolution.hp_delta:
            profiles.append(_profile({"hp_delta": float(resolution.hp_delta)}))
        if resolution.block_delta:
            profiles.append(_profile({"block": float(resolution.block_delta)}))
        if resolution.energy_delta:
            profiles.append(_profile({"energy": float(resolution.energy_delta)}))
        for marker in resolution.markers:
            profiles.append(
                _marker_profile(
                    kind=marker.kind,
                    amount=marker.amount,
                    target=marker.target_id,
                    metadata=marker.metadata,
                    source=f"relic:{normalized}:{combat_hook.value}",
                )
            )
        profiles.extend(_future_relic_timing_profiles(normalized, combat_hook))

    return combine_profiles(profiles)


def combine_profiles(profiles: Iterable[MechanicProfile]) -> MechanicProfile:
    """Merge profiles by summing values and preserving unique tags/summaries."""

    values: dict[str, float] = {}
    tags: list[str] = []
    summary: list[str] = []
    content_ids: list[str] = []
    for profile in profiles:
        for key, value in profile.values.items():
            values[key] = values.get(key, 0.0) + float(value)
        tags.extend(profile.tags)
        summary.extend(profile.summary)
        content_ids.extend(profile.content_ids)
    return MechanicProfile(
        values=values,
        tags=tuple(tags),
        summary=tuple(summary),
        content_ids=tuple(content_ids),
    )


def _reward_profile(reward: Mapping[str, Any]) -> MechanicProfile:
    kind = str(reward.get("kind", ""))
    content_id = str(reward.get("content_id", ""))
    profiles: list[MechanicProfile] = [
        _profile(tags=("reward", f"reward:{kind}") if kind else ("reward",))
    ]
    if reward.get("forced"):
        profiles.append(_profile({"forced": 1.0}, tags=("forced",)))
    if kind == "gold":
        gold = _float(reward.get("gold"))
        profiles.append(
            _profile(
                {"gold_delta": gold},
                tags=("gain_gold",),
                summary=(f"Gain {gold:g} gold",),
            )
        )
    elif kind == "card":
        profiles.append(_profile({"card_gain": 1.0}, tags=("gain_card",)))
    elif kind in {"card_removal", "remove_card"}:
        profiles.append(_profile({"card_remove": 1.0}, tags=("remove_card",)))
    elif kind == "relic":
        profiles.append(_profile({"relic_gain": 1.0}, tags=("gain_relic",)))
    elif kind == "potion":
        profiles.append(_profile({"potion_gain": 1.0}, tags=("gain_potion",)))
    if content_id and content_id != "gold":
        profiles.append(_profile(content_ids=(content_id,)))
    return combine_profiles(profiles)


def _shop_item_profile(item: Mapping[str, Any]) -> MechanicProfile:
    kind = str(item.get("kind", ""))
    item_id = str(item.get("item_id", ""))
    price = _float(item.get("price"))
    values: dict[str, float] = {"gold_delta": -price} if price else {}
    tags = ["shop_purchase", f"shop_item:{_normalized_id(kind)}"] if kind else ["shop_purchase"]
    summary = [f"Spend {price:g} gold"] if price else []
    if kind in {"card", "colorless_card"}:
        values["card_gain"] = values.get("card_gain", 0.0) + 1.0
        tags.append("gain_card")
    elif kind == "relic":
        values["relic_gain"] = values.get("relic_gain", 0.0) + 1.0
        tags.append("gain_relic")
    elif kind == "potion":
        values["potion_gain"] = values.get("potion_gain", 0.0) + 1.0
        tags.append("gain_potion")
    elif kind == "card_removal":
        values["card_remove"] = values.get("card_remove", 0.0) + 1.0
        tags.append("remove_card")
    return _profile(
        values,
        tags=tuple(tags),
        summary=tuple(summary),
        content_ids=(item_id,) if item_id else (),
    )


def _event_option_profile(event_option: Mapping[str, Any]) -> MechanicProfile:
    metadata = _mapping(event_option.get("metadata"))
    values, tags, summary = _metadata_profile_parts(metadata)
    description = str(event_option.get("description", ""))
    if description:
        text_profile = _text_effect_profile(description)
        values = _sum_values(values, text_profile.values)
        tags.extend(text_profile.tags)
        summary.extend(text_profile.summary)
    event_id = str(event_option.get("event_id", ""))
    option_id = str(event_option.get("option_id", ""))
    tags.extend(
        (
            "event_option",
            f"event:{_normalized_id(event_id)}",
            f"event_option:{_normalized_id(option_id)}",
        )
    )
    if event_option.get("disabled"):
        tags.append("disabled")
        values["risk"] = values.get("risk", 0.0) + 1.0
    if event_option.get("skip_action") or metadata.get("skip_event"):
        tags.extend(("skip_action", "skip_event"))
    transform_count = _float(metadata.get("transform_card_count"))
    random_transform_count = _float(metadata.get("transform_random_card_count"))
    if random_transform_count:
        tags.append("random_transform")
    if transform_count > random_transform_count:
        tags.append("chosen_transform")
    title = str(event_option.get("title", ""))
    if title:
        summary.append(title)
    return _profile(values, tags=tuple(tags), summary=tuple(summary))


def _ancient_option_profile(ancient_option: Mapping[str, Any]) -> MechanicProfile:
    metadata = _mapping(ancient_option.get("metadata"))
    values, tags, summary = _metadata_profile_parts(metadata)
    numeric_fields = (
        ("gold_delta", "gold_delta"),
        ("hp_delta", "hp_delta"),
        ("heal_amount", "heal"),
        ("max_hp_delta", "max_hp_delta"),
        ("potion_slot_delta", "potion_slot_delta"),
        ("card_reward_count", "card_gain"),
        ("random_relic_count", "relic_gain"),
        ("random_potion_count", "potion_gain"),
        ("upgrade_random_count", "card_upgrade"),
        ("transform_random_count", "card_transform"),
        ("remove_random_count", "card_remove"),
    )
    for source_key, value_key in numeric_fields:
        raw = _float(ancient_option.get(source_key))
        if raw:
            values[value_key] = values.get(value_key, 0.0) + raw
            tags.append(value_key)
    relic_id = str(ancient_option.get("relic_id", ""))
    if relic_id:
        values["relic_gain"] = values.get("relic_gain", 0.0) + 1.0
        tags.append("gain_relic")
        summary.append(f"Gain relic {_content_name('relics', relic_id)}")
    if ancient_option.get("kind"):
        tags.append(f"ancient_kind:{_normalized_id(str(ancient_option.get('kind')))}")
    tags.append("ancient_option")
    return _profile(
        values,
        tags=tuple(tags),
        summary=tuple(summary),
        content_ids=(relic_id,) if relic_id else (),
    )


def _relic_hook_rule_profile(relic_id: str, hook: RelicHook, rule: Any) -> MechanicProfile:
    values: dict[str, float] = {}
    tags = ["relic_hook", f"timing:{hook.value}"]
    summary: list[str] = []
    if rule.gold_delta:
        values["gold_delta"] = float(rule.gold_delta)
        summary.append(f"Gold {rule.gold_delta:+d} at {hook.value}")
    if rule.hp_delta:
        values["hp_delta"] = float(rule.hp_delta)
        summary.append(f"HP {rule.hp_delta:+d} at {hook.value}")
    if rule.max_hp_delta:
        values["max_hp_delta"] = float(rule.max_hp_delta)
        summary.append(f"Max HP {rule.max_hp_delta:+d} at {hook.value}")
    if rule.potion_slot_delta:
        values["potion_slot_delta"] = float(rule.potion_slot_delta)
        summary.append(f"Potion slots {rule.potion_slot_delta:+d}")
    profiles = [_profile(values, tags=tuple(tags), summary=tuple(summary))]
    for marker in rule.markers:
        profiles.append(
            _marker_profile(
                kind=marker.kind,
                amount=marker.amount,
                target=marker.target_id,
                metadata=marker.metadata,
                source=f"relic:{relic_id}:{hook.value}",
            )
        )
    return combine_profiles(profiles)


def _future_relic_timing_profiles(
    relic_id: str,
    combat_hook: CombatRelicHook,
) -> tuple[MechanicProfile, ...]:
    if combat_hook not in {CombatRelicHook.TURN_START, CombatRelicHook.TURN_END}:
        return ()

    profiles: list[MechanicProfile] = []
    seen_patterns: set[str] = set()
    for turn_number in range(1, 11):
        resolution = resolve_combat_relic_hook(
            (relic_id,),
            combat_hook,
            include_blockers=False,
            turn_number=turn_number,
        )
        for marker in resolution.markers:
            metadata = dict(marker.metadata)
            metadata.setdefault("trigger", combat_hook.value)
            metadata.setdefault("turn_number", turn_number)
            pattern = _timed_marker_pattern(
                marker.kind,
                marker.amount,
                marker.target_id,
                metadata,
            )
            if pattern in seen_patterns:
                continue
            seen_patterns.add(pattern)
            profiles.append(
                _marker_profile(
                    kind=marker.kind,
                    amount=marker.amount,
                    target=marker.target_id,
                    metadata=metadata,
                    source=f"relic:{relic_id}:{combat_hook.value}:future",
                )
            )
    return tuple(profiles)


def _timed_marker_pattern(
    kind: object,
    amount: object,
    target: object,
    metadata: Mapping[str, Any],
) -> str:
    stable_metadata = {
        key: value for key, value in metadata.items() if _normalized_id(str(key)) != "turn_number"
    }
    return json.dumps(
        {
            "kind": str(kind),
            "amount": amount,
            "target": target,
            "metadata": stable_metadata,
        },
        default=str,
        sort_keys=True,
        separators=(",", ":"),
    )


def _metadata_profile_parts(
    metadata: Mapping[str, Any],
) -> tuple[dict[str, float], list[str], list[str]]:
    values: dict[str, float] = {}
    tags: list[str] = []
    summary: list[str] = []
    for key, raw_value in metadata.items():
        normalized_key = _normalized_id(str(key))
        if normalized_key in MECHANIC_VALUE_KEYS:
            values[normalized_key] = values.get(normalized_key, 0.0) + _float(raw_value)
            tags.append(normalized_key)
        elif normalized_key.endswith("_count") and normalized_key not in {
            "fixed_card_count",
            "custom_card_count",
            "card_reward_count",
            "random_card_count",
            "remove_card_count",
            "remove_random_card_count",
            "transform_card_count",
            "transform_random_card_count",
            "upgrade_random_card_count",
            "downgrade_random_card_count",
            "fixed_relic_count",
            "fixed_potion_count",
            "random_relic_count",
            "random_potion_count",
            "event_reward_relic_count",
            "event_reward_potion_count",
        }:
            count_key = normalized_key.removesuffix("_count")
            if count_key in MECHANIC_VALUE_KEYS:
                values[count_key] = values.get(count_key, 0.0) + _float(raw_value)
                tags.append(count_key)
        elif normalized_key == "gold_cost" or normalized_key == "required_gold":
            values["gold_delta"] = values.get("gold_delta", 0.0) - _float(raw_value)
            tags.append("spend_gold")
        elif normalized_key == "potion_cost_count":
            values["potion_loss"] = values.get("potion_loss", 0.0) - _float(raw_value)
            tags.append("spend_potion")
        elif normalized_key == "relic_cost_count":
            values["relic_loss"] = values.get("relic_loss", 0.0) - _float(raw_value)
            tags.append("spend_relic")
        elif normalized_key == "post_combat_reward" and isinstance(raw_value, Mapping):
            post_values, post_tags, post_summary = _metadata_profile_parts(raw_value)
            values = _sum_values(values, post_values)
            tags.extend(("post_combat_reward", *post_tags))
            summary.extend(post_summary)
        elif normalized_key in {"remaining_combats", "delay_combat_count"}:
            values["combat_delay"] = values.get("combat_delay", 0.0) + _float(raw_value)
            values["delayed_reward"] = values.get("delayed_reward", 0.0) + 1.0
            tags.extend(("delayed_reward", "combat_delay"))
        elif normalized_key in {"delay", "turn_delay"}:
            values["turn_delay"] = values.get("turn_delay", 0.0) + _float(raw_value)
            values["turns_until_effect"] = values.get("turns_until_effect", 0.0) + _float(raw_value)
            tags.append("turn_delay")
        elif normalized_key in {"turn_number", "registered_turn", "absolute_turn"}:
            values["absolute_turn"] = values.get("absolute_turn", 0.0) + _float(raw_value)
            tags.append(f"turn:{_optional_int(raw_value) or 0}")
        elif normalized_key in {"remaining_uses", "uses"}:
            values["remaining_uses"] = values.get("remaining_uses", 0.0) + _float(raw_value)
            tags.append("limited_uses")
        elif normalized_key in {"every", "period"}:
            values["periodic_effect"] = values.get("periodic_effect", 0.0) + _float(raw_value)
            tags.append("periodic_effect")
        elif normalized_key == "fixed_relic_ids":
            count = float(len(_sequence(raw_value)))
            values["relic_gain"] = values.get("relic_gain", 0.0) + count
            tags.append("gain_relic")
        elif normalized_key == "fixed_relic_count":
            values["relic_gain"] = values.get("relic_gain", 0.0) + _float(raw_value)
            tags.append("gain_relic")
        elif normalized_key == "fixed_potion_ids":
            count = float(len(_sequence(raw_value)))
            values["potion_gain"] = values.get("potion_gain", 0.0) + count
            tags.append("gain_potion")
        elif normalized_key == "fixed_potion_count":
            values["potion_gain"] = values.get("potion_gain", 0.0) + _float(raw_value)
            tags.append("gain_potion")
        elif normalized_key in {"fixed_card_ids", "custom_card_ids"}:
            count = float(len(_sequence(raw_value)))
            values["card_gain"] = values.get("card_gain", 0.0) + count
            tags.append("gain_card")
        elif normalized_key in {"fixed_card_count", "custom_card_count", "card_reward_count"}:
            values["card_gain"] = values.get("card_gain", 0.0) + _float(raw_value)
            tags.append("gain_card")
        elif normalized_key == "random_card_count":
            values["card_gain"] = values.get("card_gain", 0.0) + _float(raw_value)
            values["randomness"] = values.get("randomness", 0.0) + 1.0
            tags.extend(("gain_card", "random_card"))
        elif normalized_key == "remove_card_ids":
            count = float(len(_sequence(raw_value)))
            values["card_remove"] = values.get("card_remove", 0.0) + count
            tags.append("remove_card")
        elif normalized_key == "remove_card_count":
            values["card_remove"] = values.get("card_remove", 0.0) + _float(raw_value)
            tags.append("remove_card")
        elif normalized_key == "remove_random_card_count":
            values["card_remove"] = values.get("card_remove", 0.0) + _float(raw_value)
            values["randomness"] = values.get("randomness", 0.0) + 1.0
            tags.extend(("remove_card", "random_remove_card"))
        elif normalized_key == "transform_card_count":
            values["card_transform"] = values.get("card_transform", 0.0) + _float(raw_value)
            tags.append("transform_card")
        elif normalized_key == "transform_random_card_count":
            values["randomness"] = values.get("randomness", 0.0) + 1.0
            tags.append("random_transform")
        elif normalized_key == "upgrade_random_card_count":
            values["card_upgrade"] = values.get("card_upgrade", 0.0) + _float(raw_value)
            values["randomness"] = values.get("randomness", 0.0) + 1.0
            tags.extend(("upgrade_card", "random_upgrade"))
        elif normalized_key == "downgrade_random_card_count":
            values["risk"] = values.get("risk", 0.0) + 1.0
            values["randomness"] = values.get("randomness", 0.0) + 1.0
            tags.extend(("downgrade_card", "random_downgrade"))
        elif normalized_key in {"random_relic_count", "event_reward_relic_count"}:
            values["relic_gain"] = values.get("relic_gain", 0.0) + _float(raw_value)
            values["randomness"] = values.get("randomness", 0.0) + 1.0
            tags.extend(("gain_relic", "random_relic"))
        elif normalized_key in {"random_potion_count", "event_reward_potion_count"}:
            values["potion_gain"] = values.get("potion_gain", 0.0) + _float(raw_value)
            values["randomness"] = values.get("randomness", 0.0) + 1.0
            tags.extend(("gain_potion", "random_potion"))
        elif normalized_key == "combat_encounter" or normalized_key == "monster_id":
            tags.append("event_combat")
            values["risk"] = values.get("risk", 0.0) + 1.0
        elif normalized_key == "risk":
            tags.append(f"risk:{_normalized_id(str(raw_value))}")
            values["risk"] = values.get("risk", 0.0) + 1.0
        else:
            tags.append(f"meta:{normalized_key}")
    return values, tags, summary


def _text_effect_profile(text: str) -> MechanicProfile:
    values: dict[str, float] = {}
    tags: list[str] = []
    summary: list[str] = []
    cleaned = re.sub(r"\[[^\]]+\]", "", text)
    if match := re.search(r"\bGain\s+(\d+)\s+Gold\b", cleaned, re.IGNORECASE):
        amount = float(match.group(1))
        values["gold_delta"] = values.get("gold_delta", 0.0) + amount
        tags.append("gain_gold")
        summary.append(f"Gain {amount:g} gold")
    if match := re.search(r"\bLose\s+(\d+)\s+Gold\b", cleaned, re.IGNORECASE):
        amount = float(match.group(1))
        values["gold_delta"] = values.get("gold_delta", 0.0) - amount
        tags.append("spend_gold")
        summary.append(f"Lose {amount:g} gold")
    if match := re.search(r"\b(?:Lose|Take)\s+(\d+)\s+(?:HP|Damage)\b", cleaned, re.IGNORECASE):
        amount = float(match.group(1))
        values["hp_delta"] = values.get("hp_delta", 0.0) - amount
        values["risk"] = values.get("risk", 0.0) + 1.0
        tags.append("hp_cost")
        summary.append(f"Lose {amount:g} HP")
    if re.search(r"\bRelic\b", cleaned, re.IGNORECASE):
        values["relic_gain"] = values.get("relic_gain", 0.0) + 1.0
        tags.append("gain_relic")
    if re.search(r"\bPotion\b", cleaned, re.IGNORECASE):
        values["potion_gain"] = values.get("potion_gain", 0.0) + 1.0
        tags.append("gain_potion")
    if re.search(r"\bTransform\b", cleaned, re.IGNORECASE):
        values["card_transform"] = values.get("card_transform", 0.0) + 1.0
        tags.append("transform_card")
    if re.search(r"\bRemove\b.*\bCard\b", cleaned, re.IGNORECASE):
        values["card_remove"] = values.get("card_remove", 0.0) + 1.0
        tags.append("remove_card")
    if re.search(r"\bUpgrade\b", cleaned, re.IGNORECASE):
        values["card_upgrade"] = values.get("card_upgrade", 0.0) + 1.0
        tags.append("upgrade_card")
    if match := re.search(r"\bnext\s+(\d+)\s+turns?\b", cleaned, re.IGNORECASE):
        turns = float(match.group(1))
        values["next_turn_effect"] = values.get("next_turn_effect", 0.0) + 1.0
        values["remaining_uses"] = values.get("remaining_uses", 0.0) + turns
        values["turn_delay"] = values.get("turn_delay", 0.0) + 1.0
        tags.extend(("next_turn_effect", "limited_uses"))
    return _profile(values, tags=tuple(tags), summary=tuple(summary))


def _effects_profile(effects: Mapping[str, Any], *, target: str = "") -> MechanicProfile:
    profiles: list[MechanicProfile] = []

    def walk(value: object, parent_key: str = "") -> None:
        if isinstance(value, Mapping):
            value_map = cast(Mapping[str, Any], value)
            kind = _normalized_id(str(value_map.get("kind", value_map.get("type", parent_key))))
            if kind:
                metadata_payload = {
                    **dict(_mapping(value_map.get("metadata"))),
                    **{
                        key: value_map[key]
                        for key in (
                            "trigger",
                            "duration",
                            "delay",
                            "remaining_uses",
                            "uses",
                            "every",
                            "repeat",
                            "registered_turn",
                            "turn_number",
                            "period",
                            "counter_scope",
                        )
                        if key in value_map
                    },
                }
                profiles.append(
                    _rule_profile(
                        kind=kind,
                        amount=_optional_int(value_map.get("amount")),
                        target=str(value_map.get("target", target)),
                        status=_optional_str(value_map.get("status")),
                        metadata=metadata_payload,
                    )
                )
            for key, item in value_map.items():
                normalized = _normalized_id(str(key))
                if normalized in {
                    "damage",
                    "all_damage",
                    "block",
                    "draw",
                    "energy",
                    "heal",
                    "hp_loss",
                    "max_hp_delta",
                    "discard",
                    "exhaust",
                } and isinstance(item, int | float):
                    profiles.append(
                        _rule_profile(
                            kind=normalized,
                            amount=int(item),
                            target=target,
                        )
                    )
                    continue
                walk(item, normalized)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                walk(item, parent_key)

    walk(effects)
    return combine_profiles(profiles)


def _rule_profile(
    *,
    kind: str,
    amount: int | None = None,
    target: str | None = None,
    status: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    source: str = "",
) -> MechanicProfile:
    return _marker_profile(
        kind=kind,
        amount=amount,
        target=target,
        metadata={} if metadata is None else metadata,
        source=source,
        status=status,
    )


def _marker_profile(
    *,
    kind: str,
    amount: int | None = None,
    target: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    source: str = "",
    status: str | None = None,
) -> MechanicProfile:
    normalized_kind = _normalized_id(kind)
    metadata_map = {} if metadata is None else dict(metadata)
    normalized_target = _normalized_id(str(target or ""))
    value = float(1 if amount is None else amount)
    values: dict[str, float] = {}
    tags = [normalized_kind]
    if normalized_target:
        tags.append(f"target:{normalized_target}")
    if source:
        tags.append(source)
    summary: list[str] = []

    value_key = _kind_value_key(normalized_kind, normalized_target)
    if value_key is not None:
        values[value_key] = values.get(value_key, 0.0) + value
        summary.append(f"{normalized_kind} {value:g}")
    if normalized_kind in {"hp_loss", "lose_hp"}:
        values["hp_delta"] = values.get("hp_delta", 0.0) - abs(value)
        values["risk"] = values.get("risk", 0.0) + 1.0
    if normalized_kind in {"merchant_gold", "gain_gold"}:
        values["gold_delta"] = values.get("gold_delta", 0.0) + value
    if normalized_kind in {"random_relics_gained", "dig_relic", "relic_reward"}:
        values["relic_gain"] = values.get("relic_gain", 0.0) + value
        values["randomness"] = values.get("randomness", 0.0) + 1.0
        tags.append("random_relic")
    if normalized_kind in {"add_random_potion", "random_potion"}:
        values["potion_gain"] = values.get("potion_gain", 0.0) + value
        values["randomness"] = values.get("randomness", 0.0) + 1.0
    if normalized_kind in {
        "add_deck_cards",
        "add_card_to_hand",
        "add_card_to_draw",
        "add_card_to_discard",
    }:
        values["card_gain"] = values.get("card_gain", 0.0) + value
    if normalized_kind in {"remove_deck_cards", "remove_card"}:
        values["card_remove"] = values.get("card_remove", 0.0) + value
    if normalized_kind in {"transform_deck_cards", "transform_card"}:
        values["card_transform"] = values.get("card_transform", 0.0) + value
    if normalized_kind in {"upgrade_random_cards", "upgrade_all_combat_cards", "upgrade_hand"}:
        values["card_upgrade"] = values.get("card_upgrade", 0.0) + value
    if normalized_kind in {"discard_choice", "discard_random", "discard_hand"}:
        values["discard"] = values.get("discard", 0.0) + value
    if normalized_kind in {"exhaust_choice", "exhaust_random", "exhaust_hand"}:
        values["exhaust"] = values.get("exhaust", 0.0) + value
    if normalized_kind in {"retain_hand"}:
        values["retain"] = values.get("retain", 0.0) + value
    if normalized_kind in {"next_turn"}:
        values["next_turn_effect"] = values.get("next_turn_effect", 0.0) + 1.0
        values["turn_delay"] = values.get("turn_delay", 0.0) + 1.0
        values["turns_until_effect"] = values.get("turns_until_effect", 0.0) + 1.0
        values["start_turn_timing"] = values.get("start_turn_timing", 0.0) + 1.0
        tags.extend(("next_turn_effect", "timing:turn_start"))
    if normalized_kind in {"next_combat_energy"}:
        values["energy"] = values.get("energy", 0.0) + value
        values["start_combat_timing"] = values.get("start_combat_timing", 0.0) + 1.0
        tags.extend(("next_combat_effect", "timing:start_of_combat"))

    resource = _optional_str(metadata_map.get("resource"))
    if normalized_kind == "player_resource" and resource:
        key = _resource_value_key(resource)
        values[key] = values.get(key, 0.0) + value
        tags.append(f"resource:{_normalized_id(resource)}")
    effect_status = status or _optional_str(metadata_map.get("status"))
    if (
        normalized_kind in {"apply_status", "status", "gain_status", "temporary_status"}
        and effect_status
    ):
        status_key = _status_value_key(effect_status)
        values[status_key] = values.get(status_key, 0.0) + value
        if normalized_target in {"enemy", "all_enemies"}:
            values["status_enemy"] = values.get("status_enemy", 0.0) + 1.0
        else:
            values["status_self"] = values.get("status_self", 0.0) + 1.0
        tags.extend((f"status:{_normalized_id(effect_status)}",))

    if normalized_target in {"all_enemies", "all_combatants"}:
        tags.append("aoe")
    if normalized_kind.startswith("random") or metadata_map.get("selection") == "random":
        values["randomness"] = values.get("randomness", 0.0) + 1.0
        tags.append("random")
    if metadata_map.get("selection") == "chosen":
        tags.append("chosen")
    if normalized_kind == "no_effect":
        tags.append("conditional_or_unimplemented")
    timing_profile = _timing_profile_from_payload(metadata_map, kind=normalized_kind)
    values = _sum_values(values, timing_profile.values)
    tags.extend(timing_profile.tags)
    summary.extend(timing_profile.summary)
    return _profile(values, tags=tuple(tags), summary=tuple(summary))


def _active_timing_profiles(payload: Mapping[str, Any]) -> tuple[MechanicProfile, ...]:
    profiles: list[MechanicProfile] = []
    combat = _mapping(payload.get("combat"))
    turn = _optional_int(combat.get("turn")) or 0
    metadata = _mapping(combat.get("metadata"))
    for trigger in _sequence(metadata.get("combat_triggers")):
        trigger_map = _mapping(trigger)
        profiles.append(_trigger_profile(trigger_map, current_turn=turn, source="combat_trigger"))
    for trigger in _sequence(metadata.get("timed_card_triggers")):
        trigger_map = _mapping(trigger)
        profiles.append(
            _trigger_profile(trigger_map, current_turn=turn, source="timed_card_trigger")
        )
    for queued in _sequence(metadata.get("next_card_extra_play")):
        queued_map = _mapping(queued)
        profiles.append(
            combine_profiles(
                (
                    _profile(
                        {"current_turn_effect": 1.0},
                        tags=("active_next_card_extra_play",),
                        content_ids=(
                            str(queued_map.get("source_card_id", "")),
                        ),
                    ),
                    _timing_profile_from_payload(queued_map),
                )
            )
        )

    flags = _mapping(payload.get("flags"))
    for delayed in _sequence(flags.get("delayed_event_rewards")):
        delayed_map = _mapping(delayed)
        remaining = _float(delayed_map.get("remaining_combats"))
        profiles.append(
            _profile(
                {"combat_delay": remaining, "delayed_reward": 1.0},
                tags=(
                    "delayed_reward",
                    f"reward:{_normalized_id(str(delayed_map.get('reward_kind', '')))}",
                ),
                summary=(f"Delayed reward in {remaining:g} combats",),
                content_ids=(
                    str(delayed_map.get("item_id", "")),
                ),
            )
        )
    return tuple(profiles)


def _trigger_profile(
    trigger: Mapping[str, Any],
    *,
    current_turn: int,
    source: str,
) -> MechanicProfile:
    profiles: list[MechanicProfile] = [
        _profile(
            {"repeating_effect": 1.0}
            if _normalized_id(str(trigger.get("duration", "combat"))) != "once"
            else {},
            tags=(source, "active_trigger"),
            content_ids=(str(trigger.get("source_card_id", "")),),
        ),
        _timing_profile_from_payload(trigger, current_turn=current_turn),
    ]
    effects = _sequence(trigger.get("effects")) or _sequence(trigger.get("pre_effects"))
    if effects:
        profiles.append(_effects_profile({"sequence": effects}))
    if trigger.get("choose_card"):
        profiles.append(_profile(tags=("pending_timed_choice",), summary=("Timed card choice",)))
    if trigger.get("add_copies_of_card"):
        profiles.append(
            _profile(
                {"card_gain": _float(trigger.get("copy_count")) or 1.0},
                tags=("timed_card_copy",),
            )
        )
    return combine_profiles(profiles)


def _status_timing_profile(status_id: str, amount: object) -> MechanicProfile:
    normalized = _normalized_id(status_id)
    value = _float(amount)
    if normalized == "next_turn_energy":
        return _profile(
            {"energy": value, "next_turn_effect": 1.0, "turn_delay": 1.0, "start_turn_timing": 1.0},
            tags=("next_turn_effect", "timing:turn_start"),
        )
    if normalized == "next_turn_draw":
        return _profile(
            {"draw": value, "next_turn_effect": 1.0, "turn_delay": 1.0, "start_turn_timing": 1.0},
            tags=("next_turn_effect", "timing:turn_start"),
        )
    if normalized == "next_turn_block":
        return _profile(
            {"block": value, "next_turn_effect": 1.0, "turn_delay": 1.0, "start_turn_timing": 1.0},
            tags=("next_turn_effect", "timing:turn_start"),
        )
    if normalized == "next_turn_star":
        return _profile(
            {"star": value, "next_turn_effect": 1.0, "turn_delay": 1.0, "start_turn_timing": 1.0},
            tags=("next_turn_effect", "timing:turn_start"),
        )
    if normalized.startswith("temporary_"):
        return _profile(
            {"current_turn_effect": 1.0},
            tags=("current_turn_effect", "temporary_status"),
        )
    return MechanicProfile()


def _enemy_status_profile(status_id: str, amount: object, monster_id: str) -> MechanicProfile:
    normalized = _normalized_id(status_id)
    value = _float(amount)
    values = {"status_enemy": 1.0}
    status_key = _status_value_key(status_id)
    if status_key == "status_self":
        values["status_enemy"] += value
    else:
        values[status_key] = value
    return _profile(
        values,
        tags=("enemy_status", f"status:{normalized}", f"monster:{_normalized_id(monster_id)}"),
        summary=(f"Enemy {monster_id} status {status_id}: {value:g}",),
    )


def _timing_profile_from_payload(
    payload: Mapping[str, Any],
    *,
    kind: str = "",
    current_turn: int | None = None,
) -> MechanicProfile:
    values: dict[str, float] = {}
    tags: list[str] = []
    summary: list[str] = []
    trigger = _normalized_id(str(payload.get("trigger", "")))
    duration = _normalized_id(str(payload.get("duration", "")))
    target_turn = _float(payload.get("turn_number"))
    if trigger:
        tags.append(f"timing:{trigger}")
        if trigger == "turn_start":
            values["start_turn_timing"] = values.get("start_turn_timing", 0.0) + 1.0
            if current_turn is not None:
                values["turns_until_effect"] = values.get("turns_until_effect", 0.0) + 1.0
        elif trigger == "turn_end":
            values["end_turn_timing"] = values.get("end_turn_timing", 0.0) + 1.0
            values["current_turn_effect"] = values.get("current_turn_effect", 0.0) + 1.0
        elif trigger in {"combat_start", "start_combat", "start_of_combat"}:
            values["start_combat_timing"] = values.get("start_combat_timing", 0.0) + 1.0
        elif trigger in {"combat_end", "end_combat"}:
            values["end_combat_timing"] = values.get("end_combat_timing", 0.0) + 1.0
        summary.append(f"Triggers at {trigger}")
    if duration:
        tags.append(f"duration:{duration}")
        if duration not in {"once", "turn"}:
            values["repeating_effect"] = values.get("repeating_effect", 0.0) + 1.0
    delay = _float(payload.get("delay"))
    if delay:
        values["turn_delay"] = values.get("turn_delay", 0.0) + delay
        values["turns_until_effect"] = values.get("turns_until_effect", 0.0) + delay
        tags.append("turn_delay")
        summary.append(f"Delayed {delay:g} turns")
    remaining_uses = _float(payload.get("remaining_uses", payload.get("uses")))
    if remaining_uses:
        values["remaining_uses"] = values.get("remaining_uses", 0.0) + remaining_uses
        tags.append("limited_uses")
    every = _float(payload.get("every", payload.get("period")))
    if every:
        values["periodic_effect"] = values.get("periodic_effect", 0.0) + every
        tags.append("periodic_effect")
        summary.append(f"Every {every:g} turns")
    if target_turn:
        values["absolute_turn"] = values.get("absolute_turn", 0.0) + target_turn
        tags.append(f"turn:{int(target_turn)}")
        summary.append(f"Turn {target_turn:g}")
    if target_turn and current_turn is not None:
        values["turns_until_effect"] = values.get("turns_until_effect", 0.0) + max(
            0.0,
            target_turn - float(current_turn),
        )
    if bool(payload.get("repeat")):
        values["repeating_effect"] = values.get("repeating_effect", 0.0) + 1.0
        tags.append("repeating_effect")
    if kind in {"next_turn", "next_turn_effect"}:
        values["next_turn_effect"] = values.get("next_turn_effect", 0.0) + 1.0
        values["turn_delay"] = values.get("turn_delay", 0.0) + 1.0
        tags.append("next_turn_effect")
    return _profile(values, tags=tuple(tags), summary=tuple(summary))


def _kind_value_key(kind: str, target: str) -> str | None:
    if kind in {"damage", "hit", "attack_damage"}:
        return "aoe_damage" if target in {"all_enemies", "all_combatants"} else "damage"
    if kind in {"all_damage"}:
        return "aoe_damage"
    if kind in {"block", "gain_block"}:
        return "block"
    if kind in {"draw", "draw_cards"}:
        return "draw"
    if kind in {"energy", "gain_energy", "next_combat_energy", "start_turn_energy"}:
        return "energy"
    if kind in {"heal", "heal_player"}:
        return "heal"
    if kind in {"max_hp", "max_hp_delta"}:
        return "max_hp_delta"
    if kind in {"orb_slot_delta"}:
        return "orb_slot_delta"
    if kind in {"channel_orb", "dynamic_channel_orb", "ally_channel_orb"}:
        return "orb_channel"
    if kind in {"evoke_orb"}:
        return "orb_evoke"
    if kind in {"self_cost_delta", "set_hand_cost", "set_hand_free_to_play_this_turn"}:
        return "cost_reduction"
    if kind in {"shop_restock", "restock_shop"}:
        return "shop_restock"
    return kind if kind in MECHANIC_VALUE_KEYS else None


def _resource_value_key(resource: str) -> str:
    normalized = _normalized_id(resource)
    if normalized in {"summon", "soul"}:
        return "summon"
    if normalized in {"star", "stars"}:
        return "star"
    if normalized == "forge":
        return "forge"
    return "summon"


def _status_value_key(status: str) -> str:
    normalized = _normalized_id(status)
    return normalized if normalized in _STATUS_VALUE_KEYS else "status_self"


def _card_source(card_id: str) -> Mapping[str, Any] | None:
    if not card_id:
        return None
    return _cached_cards().get(_normalized_id(card_id))


@cache
def _cached_cards() -> dict[str, Mapping[str, Any]]:
    return _cached_rows_by_id("cards")


@cache
def _cached_rows_by_id(dataset: str) -> dict[str, Mapping[str, Any]]:
    path = _CACHE_DIR / f"{dataset}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}
    rows: dict[str, Mapping[str, Any]] = {}
    for row in payload:
        if not isinstance(row, Mapping):
            continue
        row_map = cast(Mapping[str, Any], row)
        content_id = _optional_str(
            _first_present(row_map, "id", "card_id", "relic_id", "potion_id")
        )
        if content_id is None:
            continue
        normalized = _normalized_id(content_id)
        rows[normalized] = row_map
    return rows


def _content_name(dataset: str, content_id: str) -> str:
    normalized = _normalized_id(content_id)
    row = _cached_rows_by_id(dataset).get(normalized)
    if row is None:
        return content_id.replace("_", " ").title()
    name = _optional_str(_first_present(row, "name", "title"))
    return name or content_id.replace("_", " ").title()


def _profile(
    values: Mapping[str, float] | None = None,
    *,
    tags: Iterable[str] = (),
    summary: Iterable[str] = (),
    content_ids: Iterable[str] = (),
) -> MechanicProfile:
    return MechanicProfile(
        values={} if values is None else values,
        tags=tuple(tags),
        summary=tuple(summary),
        content_ids=tuple(content_ids),
    )


def _sum_values(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    values = dict(left)
    for key, value in right.items():
        values[key] = values.get(key, 0.0) + float(value)
    return values


def _clean_values(values: Mapping[str, float]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in values.items():
        normalized = _normalized_id(str(key))
        if normalized not in MECHANIC_VALUE_KEYS:
            continue
        amount = float(value)
        if amount:
            cleaned[normalized] = amount
    return cleaned


def _unique_text(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


def _normalized_id(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace("'", "")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
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


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
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
