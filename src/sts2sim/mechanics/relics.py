"""Pure relic handler helpers.

The engine owns state mutation; this module only resolves table-driven relic
effects into small deltas and markers that callers can apply later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from sts2sim.content.sources import PROVISIONAL_STS2_SOURCE, SourceRef

RelicInput = str | Mapping[str, object]


class RelicHook(str, Enum):
    PICKUP = "pickup"
    SHOP_ENTER = "shop_enter"
    SHOP_PURCHASE = "shop_purchase"
    SHOP_PRICE = "shop_price"
    POTION_CAPACITY = "potion_capacity"
    CAMPFIRE_ENTER = "campfire_enter"
    START_COMBAT = "start_combat"
    START_TURN = "start_turn"
    END_COMBAT = "end_combat"


@dataclass(frozen=True, slots=True)
class RelicMarkerSpec:
    kind: str
    amount: int | None = None
    target_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RelicEffectMarker:
    kind: str
    relic_id: str
    amount: int | None = None
    target_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RelicHookRule:
    gold_delta: int = 0
    hp_delta: int = 0
    max_hp_delta: int = 0
    potion_slot_delta: int = 0
    markers: tuple[RelicMarkerSpec, ...] = ()
    source: SourceRef = PROVISIONAL_STS2_SOURCE


@dataclass(frozen=True, slots=True)
class RelicHookResult:
    hook: RelicHook
    relic_id: str
    gold_delta: int = 0
    hp_delta: int = 0
    max_hp_delta: int = 0
    potion_slot_delta: int = 0
    markers: tuple[RelicEffectMarker, ...] = ()
    unsupported: bool = False
    source: SourceRef = PROVISIONAL_STS2_SOURCE


@dataclass(frozen=True, slots=True)
class RelicHookResolution:
    hook: RelicHook
    results: tuple[RelicHookResult, ...] = ()
    gold_delta: int = 0
    hp_delta: int = 0
    max_hp_delta: int = 0
    potion_slot_delta: int = 0
    markers: tuple[RelicEffectMarker, ...] = ()
    source: SourceRef = PROVISIONAL_STS2_SOURCE


@dataclass(frozen=True, slots=True)
class RelicPriceModifier:
    relic_id: str
    multiplier_percent: int | None = None
    fixed_price: int | None = None
    item_kinds: frozenset[str] = field(default_factory=frozenset)
    source: SourceRef = PROVISIONAL_STS2_SOURCE


@dataclass(frozen=True, slots=True)
class RelicPriceResult:
    item_kind: str
    base_price: int
    price: int
    applied_relic_ids: tuple[str, ...] = ()
    multiplier_percent: int = 100
    fixed_price: int | None = None
    source: SourceRef = PROVISIONAL_STS2_SOURCE


@dataclass(frozen=True, slots=True)
class UnsupportedRelicHandler:
    relic_id: str
    name: str | None = None
    unsupported_hooks: tuple[RelicHook, ...] = ()
    description: str | None = None


def _default_relic_hook_rules() -> dict[RelicHook, dict[str, RelicHookRule]]:
    return {
        RelicHook.PICKUP: {
            "old_coin": RelicHookRule(
                gold_delta=300,
                markers=(RelicMarkerSpec("old_coin_gold_gained"),),
            ),
            "potion_belt": RelicHookRule(
                potion_slot_delta=2,
                markers=(RelicMarkerSpec("potion_slots_gained"),),
            ),
            "alchemical_coffer": RelicHookRule(
                potion_slot_delta=4,
                markers=(
                    RelicMarkerSpec(
                        "potion_slots_gained",
                        metadata={"fill_random_potions": 4},
                    ),
                ),
            ),
            "phial_holster": RelicHookRule(
                potion_slot_delta=1,
                markers=(
                    RelicMarkerSpec(
                        "potion_slots_gained",
                        metadata={"fill_random_potions": 2},
                    ),
                ),
            ),
        },
        RelicHook.SHOP_ENTER: {
            "meal_ticket": RelicHookRule(
                hp_delta=15,
                markers=(RelicMarkerSpec("meal_ticket_healed"),),
            ),
            "the_courier": RelicHookRule(
                markers=(RelicMarkerSpec("shop_restock_enabled"),),
            ),
            "courier": RelicHookRule(
                markers=(RelicMarkerSpec("shop_restock_enabled"),),
            ),
        },
        RelicHook.SHOP_PURCHASE: {
            "the_courier": RelicHookRule(
                markers=(RelicMarkerSpec("shop_restock_purchased_item"),),
            ),
            "courier": RelicHookRule(
                markers=(RelicMarkerSpec("shop_restock_purchased_item"),),
            ),
        },
        RelicHook.CAMPFIRE_ENTER: {
            "venerable_tea_set": RelicHookRule(
                markers=(RelicMarkerSpec("next_combat_energy", amount=2, target_id="player"),),
            ),
            "fake_venerable_tea_set": RelicHookRule(
                markers=(RelicMarkerSpec("next_combat_energy", amount=1, target_id="player"),),
            ),
        },
        RelicHook.START_COMBAT: {
            "akabeko": RelicHookRule(
                markers=(RelicMarkerSpec("gain_status", amount=8, metadata={"status": "vigor"}),),
            ),
            "anchor": RelicHookRule(
                markers=(RelicMarkerSpec("gain_block", amount=10, target_id="player"),),
            ),
            "fake_anchor": RelicHookRule(
                markers=(RelicMarkerSpec("gain_block", amount=4, target_id="player"),),
            ),
            "bag_of_marbles": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "apply_status",
                        amount=1,
                        target_id="all_enemies",
                        metadata={"status": "vulnerable"},
                    ),
                ),
            ),
            "bag_of_preparation": RelicHookRule(
                markers=(RelicMarkerSpec("draw_cards", amount=2, target_id="player"),),
            ),
            "big_mushroom": RelicHookRule(
                markers=(RelicMarkerSpec("draw_cards", amount=-2, target_id="player"),),
            ),
            "blessed_antler": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "shuffle_status_into_draw_pile",
                        amount=3,
                        target_id="player",
                        metadata={"card_id": "dazed"},
                    ),
                ),
            ),
            "blood_vial": RelicHookRule(
                hp_delta=2,
                markers=(RelicMarkerSpec("blood_vial_healed"),),
            ),
            "bronze_scales": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "gain_status",
                        amount=3,
                        target_id="player",
                        metadata={"status": "thorns"},
                    ),
                ),
            ),
            "cracked_core": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "channel_orb",
                        amount=1,
                        target_id="player",
                        metadata={"orb": "lightning"},
                    ),
                ),
            ),
            "data_disk": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "gain_status",
                        amount=1,
                        target_id="player",
                        metadata={"status": "focus"},
                    ),
                ),
            ),
            "fake_blood_vial": RelicHookRule(
                hp_delta=1,
                markers=(RelicMarkerSpec("blood_vial_healed"),),
            ),
            "vajra": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "gain_status",
                        amount=1,
                        target_id="player",
                        metadata={"status": "strength"},
                    ),
                ),
            ),
            "very_hot_cocoa": RelicHookRule(
                markers=(RelicMarkerSpec("gain_energy", amount=4, target_id="player"),),
            ),
        },
        RelicHook.START_TURN: {
            "blessed_antler": RelicHookRule(
                markers=(RelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
            ),
            "blood_soaked_rose": RelicHookRule(
                markers=(RelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
            ),
            "brimstone": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "gain_status",
                        amount=2,
                        target_id="player",
                        metadata={"status": "strength"},
                    ),
                    RelicMarkerSpec(
                        "apply_status",
                        amount=1,
                        target_id="all_enemies",
                        metadata={"status": "strength"},
                    ),
                ),
            ),
            "velvet_choker": RelicHookRule(
                markers=(
                    RelicMarkerSpec("gain_energy", amount=1, target_id="player"),
                    RelicMarkerSpec(
                        "turn_card_play_limit",
                        amount=6,
                        target_id="player",
                    ),
                ),
            ),
        },
        RelicHook.END_COMBAT: {
            "burning_blood": RelicHookRule(
                hp_delta=6,
                markers=(RelicMarkerSpec("burning_blood_healed"),),
            ),
            "black_blood": RelicHookRule(
                hp_delta=12,
                markers=(RelicMarkerSpec("black_blood_healed"),),
            ),
        },
    }


def _default_price_modifiers() -> dict[str, RelicPriceModifier]:
    return {
        "membership_card": RelicPriceModifier("membership_card", multiplier_percent=50),
        "the_courier": RelicPriceModifier("the_courier", multiplier_percent=80),
        "courier": RelicPriceModifier("courier", multiplier_percent=80),
        "smiling_mask": RelicPriceModifier(
            "smiling_mask",
            fixed_price=50,
            item_kinds=frozenset({"card_removal"}),
        ),
    }


DEFAULT_RELIC_HOOK_RULES = _default_relic_hook_rules()
DEFAULT_RELIC_PRICE_MODIFIERS = _default_price_modifiers()
DEFAULT_ENGINE_RELIC_IDS = frozenset(
    {
        "black_star",
        "frozen_egg",
        "molten_egg",
        "toxic_egg",
    }
)
DEFAULT_RELIC_POTION_SLOT_MODIFIERS = {
    "potion_belt": 2,
    "alchemical_coffer": 4,
    "phial_holster": 1,
}


def resolve_relic_pickup(
    relic: RelicInput,
    *,
    hp: int | None = None,
    max_hp: int | None = None,
    rules: Mapping[RelicHook, Mapping[str, RelicHookRule]] = DEFAULT_RELIC_HOOK_RULES,
) -> RelicHookResult:
    """Resolve a single relic pickup into deterministic deltas and markers."""

    relic_id = relic_content_id(relic)
    rule = rules.get(RelicHook.PICKUP, {}).get(relic_id)
    if rule is None:
        return RelicHookResult(
            hook=RelicHook.PICKUP,
            relic_id=relic_id,
            unsupported=True,
        )
    return _result_from_rule(
        relic_id,
        RelicHook.PICKUP,
        rule,
        hp=hp,
        max_hp=max_hp,
    )


def resolve_relic_hook(
    relics: Sequence[RelicInput],
    hook: RelicHook,
    *,
    hp: int | None = None,
    max_hp: int | None = None,
    rules: Mapping[RelicHook, Mapping[str, RelicHookRule]] = DEFAULT_RELIC_HOOK_RULES,
) -> RelicHookResolution:
    """Resolve all relics that have a handler for ``hook``."""

    hook_rules = rules.get(hook, {})
    results: list[RelicHookResult] = []
    current_hp = hp
    for relic_id in _unique_relic_ids(relics):
        rule = hook_rules.get(relic_id)
        if rule is None:
            continue
        result = _result_from_rule(
            relic_id,
            hook,
            rule,
            hp=current_hp,
            max_hp=max_hp,
        )
        results.append(result)
        if current_hp is not None:
            current_hp += result.hp_delta
    return _combine_hook_results(hook, tuple(results))


def apply_relic_price_modifiers(
    base_price: int,
    item_kind: str | Enum,
    relics: Sequence[RelicInput],
    *,
    min_price: int = 0,
    modifiers: Mapping[str, RelicPriceModifier] = DEFAULT_RELIC_PRICE_MODIFIERS,
) -> RelicPriceResult:
    """Apply fixed and percentage shop-price relic modifiers."""

    normalized_kind = _normalized_id(_enum_value(item_kind))
    normalized_base = max(0, int(base_price))
    applied: list[str] = []
    fixed_prices: list[tuple[str, int]] = []
    multiplier_percent = 100

    for relic_id in _unique_relic_ids(relics):
        modifier = modifiers.get(relic_id)
        if modifier is None or not _modifier_applies(modifier, normalized_kind):
            continue
        if modifier.fixed_price is not None:
            fixed_prices.append((relic_id, modifier.fixed_price))
            continue
        if modifier.multiplier_percent is None:
            continue
        applied.append(relic_id)
        multiplier_percent *= max(0, modifier.multiplier_percent)
        multiplier_percent //= 100

    if fixed_prices:
        relic_id, fixed_price = min(fixed_prices, key=lambda item: item[1])
        return RelicPriceResult(
            item_kind=normalized_kind,
            base_price=normalized_base,
            price=max(min_price, fixed_price),
            applied_relic_ids=(relic_id,),
            fixed_price=fixed_price,
        )

    price = normalized_base * multiplier_percent // 100
    return RelicPriceResult(
        item_kind=normalized_kind,
        base_price=normalized_base,
        price=max(min_price, price),
        applied_relic_ids=tuple(applied),
        multiplier_percent=multiplier_percent,
    )


def relic_potion_slot_bonus(
    relics: Sequence[RelicInput],
    *,
    modifiers: Mapping[str, int] = DEFAULT_RELIC_POTION_SLOT_MODIFIERS,
) -> int:
    """Return additional potion slots granted by owned relics."""

    return sum(max(0, int(modifiers.get(relic_id, 0))) for relic_id in _unique_relic_ids(relics))


def supported_relic_ids(
    hook: RelicHook | None = None,
    *,
    rules: Mapping[RelicHook, Mapping[str, RelicHookRule]] = DEFAULT_RELIC_HOOK_RULES,
    price_modifiers: Mapping[str, RelicPriceModifier] = DEFAULT_RELIC_PRICE_MODIFIERS,
    potion_slot_modifiers: Mapping[str, int] = DEFAULT_RELIC_POTION_SLOT_MODIFIERS,
    engine_relic_ids: frozenset[str] = DEFAULT_ENGINE_RELIC_IDS,
) -> frozenset[str]:
    """Return relic ids with at least one explicit helper."""

    if hook is RelicHook.SHOP_PRICE:
        return frozenset(price_modifiers)
    if hook is RelicHook.POTION_CAPACITY:
        return frozenset(potion_slot_modifiers)
    if hook is not None:
        return frozenset(rules.get(hook, {}))

    supported = set(price_modifiers) | set(potion_slot_modifiers) | set(engine_relic_ids)
    for hook_rules in rules.values():
        supported.update(hook_rules)
    return frozenset(supported)


def unsupported_relic_handlers(
    relics: Sequence[RelicInput],
    *,
    hooks: Sequence[RelicHook] | None = None,
    rules: Mapping[RelicHook, Mapping[str, RelicHookRule]] = DEFAULT_RELIC_HOOK_RULES,
    price_modifiers: Mapping[str, RelicPriceModifier] = DEFAULT_RELIC_PRICE_MODIFIERS,
    potion_slot_modifiers: Mapping[str, int] = DEFAULT_RELIC_POTION_SLOT_MODIFIERS,
) -> tuple[UnsupportedRelicHandler, ...]:
    """Report inferred relic hooks without a bounded helper implementation."""

    unsupported: list[UnsupportedRelicHandler] = []
    supported_by_hook = {
        hook: supported_relic_ids(
            hook,
            rules=rules,
            price_modifiers=price_modifiers,
            potion_slot_modifiers=potion_slot_modifiers,
        )
        for hook in RelicHook
    }
    for relic in relics:
        relic_id = relic_content_id(relic)
        needed_hooks = tuple(hooks) if hooks is not None else _inferred_hooks(relic)
        missing = tuple(
            hook
            for hook in needed_hooks
            if relic_id not in supported_by_hook.get(hook, frozenset())
        )
        if missing:
            unsupported.append(
                UnsupportedRelicHandler(
                    relic_id=relic_id,
                    name=_content_str(relic, "name"),
                    unsupported_hooks=missing,
                    description=_content_str(relic, "description", "description_raw"),
                )
            )
    return tuple(unsupported)


def relic_content_id(relic: RelicInput) -> str:
    """Return a normalized relic id from a raw id or Codex-style mapping."""

    if isinstance(relic, str):
        return _normalized_id(relic)
    value = _first_present(relic, "id", "relic_id", "content_id", "item_id")
    if value is None:
        raise ValueError(f"Relic input is missing an id: {relic!r}")
    return _normalized_id(str(value))


def _result_from_rule(
    relic_id: str,
    hook: RelicHook,
    rule: RelicHookRule,
    *,
    hp: int | None,
    max_hp: int | None,
) -> RelicHookResult:
    hp_delta = rule.hp_delta
    if hp_delta > 0 and hp is not None and max_hp is not None:
        hp_delta = max(0, min(hp_delta, max_hp - hp))
    markers = tuple(
        RelicEffectMarker(
            kind=marker.kind,
            relic_id=relic_id,
            amount=_marker_amount(
                marker,
                gold_delta=rule.gold_delta,
                hp_delta=hp_delta,
                max_hp_delta=rule.max_hp_delta,
                potion_slot_delta=rule.potion_slot_delta,
            ),
            target_id=marker.target_id,
            metadata=marker.metadata,
        )
        for marker in rule.markers
    )
    return RelicHookResult(
        hook=hook,
        relic_id=relic_id,
        gold_delta=rule.gold_delta,
        hp_delta=hp_delta,
        max_hp_delta=rule.max_hp_delta,
        potion_slot_delta=rule.potion_slot_delta,
        markers=markers,
        source=rule.source,
    )


def _combine_hook_results(
    hook: RelicHook,
    results: tuple[RelicHookResult, ...],
) -> RelicHookResolution:
    return RelicHookResolution(
        hook=hook,
        results=results,
        gold_delta=sum(result.gold_delta for result in results),
        hp_delta=sum(result.hp_delta for result in results),
        max_hp_delta=sum(result.max_hp_delta for result in results),
        potion_slot_delta=sum(result.potion_slot_delta for result in results),
        markers=tuple(marker for result in results for marker in result.markers),
    )


def _marker_amount(
    marker: RelicMarkerSpec,
    *,
    gold_delta: int,
    hp_delta: int,
    max_hp_delta: int,
    potion_slot_delta: int,
) -> int | None:
    if marker.amount is not None:
        return marker.amount
    for amount in (gold_delta, hp_delta, max_hp_delta, potion_slot_delta):
        if amount:
            return amount
    return None


def _modifier_applies(modifier: RelicPriceModifier, item_kind: str) -> bool:
    return not modifier.item_kinds or item_kind in modifier.item_kinds


def _unique_relic_ids(relics: Sequence[RelicInput]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for relic in relics:
        relic_id = relic_content_id(relic)
        if relic_id in seen:
            continue
        seen.add(relic_id)
        normalized.append(relic_id)
    return tuple(normalized)


def _inferred_hooks(relic: RelicInput) -> tuple[RelicHook, ...]:
    description = _content_str(relic, "description", "description_raw")
    if not description:
        return ()
    text = description.lower()
    hooks: list[RelicHook] = []
    if "upon pickup" in text:
        hooks.append(RelicHook.PICKUP)
    if "enter a shop" in text or "enter a shop room" in text:
        hooks.append(RelicHook.SHOP_ENTER)
    if "merchant" in text or "prices" in text or "discount" in text:
        hooks.append(RelicHook.SHOP_PRICE)
    if (
        "potion slot" in text
        and ("upon pickup" in text or "gain" in text)
        and "empty potion slots" not in text
    ):
        hooks.append(RelicHook.POTION_CAPACITY)
    if "enter a rest site" in text:
        hooks.append(RelicHook.CAMPFIRE_ENTER)
    if "start of each combat" in text or "start each combat" in text:
        hooks.append(RelicHook.START_COMBAT)
    if "start of each turn" in text:
        hooks.append(RelicHook.START_TURN)
    if "end of combat" in text:
        hooks.append(RelicHook.END_COMBAT)
    return tuple(dict.fromkeys(hooks))


def _content_str(item: RelicInput, *keys: str) -> str | None:
    if isinstance(item, str):
        return None
    value = _first_present(item, *keys)
    return None if value is None else str(value)


def _first_present(item: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        value = item.get(key)
        if value is not None:
            return value
    return None


def _enum_value(value: str | Enum) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _normalized_id(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("'", "")
        .replace(" ", "_")
        .replace("-", "_")
    )
