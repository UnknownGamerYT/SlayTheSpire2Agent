"""Pure potion capacity, pickup, discard, and use normalization helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from sts2sim.content.sources import PROVISIONAL_STS2_SOURCE, SourceRef

from .relics import RelicInput, relic_potion_slot_bonus
from .rewards import potion_slots_for_ascension

PotionInput = str | Mapping[str, object]

FOUL_POTION_ID = "foul_potion"


@dataclass(frozen=True, slots=True)
class PotionCapacityResult:
    capacity: int
    base_slots: int
    ascension_slots: int
    relic_bonus_slots: int = 0
    bonus_slots: int = 0
    open_slots: int = 0
    source: SourceRef = PROVISIONAL_STS2_SOURCE


@dataclass(frozen=True, slots=True)
class PotionPickupDecision:
    potion_id: str
    can_pick_up: bool
    requires_discard: bool
    capacity: int
    potions: tuple[str, ...]
    next_potions: tuple[str, ...]
    source: SourceRef = PROVISIONAL_STS2_SOURCE


@dataclass(frozen=True, slots=True)
class PotionDiscardDecision:
    slot_index: int
    can_discard: bool
    discarded_potion_id: str | None
    potions: tuple[str, ...]
    next_potions: tuple[str, ...]
    source: SourceRef = PROVISIONAL_STS2_SOURCE


@dataclass(frozen=True, slots=True)
class PotionEffectRule:
    kind: str
    amount: int | None = None
    target: str | None = None
    status: str | None = None
    duration: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PotionEffect:
    kind: str
    amount: int | None = None
    target: str | None = None
    status: str | None = None
    duration: int | None = None
    target_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PotionUseNormalization:
    potion_id: str
    effects: tuple[PotionEffect, ...]
    mode: str = "combat"
    consumes_potion: bool = True
    unsupported: bool = False
    source: SourceRef = PROVISIONAL_STS2_SOURCE


def _default_potion_effects() -> dict[str, tuple[PotionEffectRule, ...]]:
    return {
        "block_potion": (PotionEffectRule("block", amount=12, target="self"),),
        "cure_all": (
            PotionEffectRule("energy", amount=1, target="self"),
            PotionEffectRule("draw", amount=2, target="self"),
        ),
        "dexterity_potion": (
            PotionEffectRule("status", amount=2, target="self", status="dexterity"),
        ),
        "energy_potion": (PotionEffectRule("energy", amount=2, target="self"),),
        "essence_of_darkness": (
            PotionEffectRule(
                "channel_orb",
                target="self",
                metadata={"orb": "dark", "amount": "orb_slots"},
            ),
        ),
        "explosive_ampoule": (
            PotionEffectRule("damage", amount=10, target="all_enemies"),
        ),
        "fire_potion": (PotionEffectRule("damage", amount=20, target="enemy"),),
        "flex_potion": (
            PotionEffectRule(
                "temporary_status",
                amount=5,
                target="self",
                status="strength",
                duration=1,
            ),
        ),
        "focus_potion": (
            PotionEffectRule("status", amount=2, target="self", status="focus"),
        ),
        "foul_potion": (
            PotionEffectRule(
                "damage",
                amount=12,
                target="all_combatants",
                metadata={"hits_player": True},
            ),
        ),
        "fysh_oil": (
            PotionEffectRule("status", amount=1, target="self", status="strength"),
            PotionEffectRule("status", amount=1, target="self", status="dexterity"),
        ),
        "liquid_bronze": (
            PotionEffectRule("status", amount=3, target="self", status="thorns"),
        ),
        "poison_potion": (
            PotionEffectRule("status", amount=6, target="enemy", status="poison"),
        ),
        "potion_of_binding": (
            PotionEffectRule("status", amount=1, target="all_enemies", status="weak"),
            PotionEffectRule(
                "status",
                amount=1,
                target="all_enemies",
                status="vulnerable",
            ),
        ),
        "potion_of_capacity": (
            PotionEffectRule("orb_slot_delta", amount=2, target="self"),
        ),
        "potion_of_doom": (
            PotionEffectRule("status", amount=33, target="enemy", status="doom"),
        ),
        "potion_shaped_rock": (
            PotionEffectRule("damage", amount=15, target="enemy"),
        ),
        "radiant_tincture": (
            PotionEffectRule("energy", amount=1, target="self"),
            PotionEffectRule(
                "start_turn_energy",
                amount=1,
                target="self",
                duration=3,
            ),
        ),
        "regen_potion": (
            PotionEffectRule("status", amount=5, target="self", status="regen"),
        ),
        "ship_in_a_bottle": (
            PotionEffectRule("block", amount=10, target="self"),
            PotionEffectRule("next_turn_block", amount=10, target="self", duration=1),
        ),
        "speed_potion": (
            PotionEffectRule(
                "temporary_status",
                amount=5,
                target="self",
                status="dexterity",
                duration=1,
            ),
        ),
        "strength_potion": (
            PotionEffectRule("status", amount=2, target="self", status="strength"),
        ),
        "swift_potion": (PotionEffectRule("draw", amount=3, target="self"),),
        "vulnerable_potion": (
            PotionEffectRule("status", amount=3, target="enemy", status="vulnerable"),
        ),
        "weak_potion": (
            PotionEffectRule("status", amount=3, target="enemy", status="weak"),
        ),
    }


DEFAULT_POTION_EFFECTS = _default_potion_effects()
FOUL_POTION_MERCHANT_EFFECT = PotionEffectRule(
    "merchant_gold",
    amount=100,
    target="merchant",
)


def potion_capacity(
    *,
    base_slots: int = 3,
    ascension_level: int = 0,
    relics: Sequence[RelicInput] = (),
    bonus_slots: int = 0,
    current_potions: Sequence[PotionInput] = (),
) -> PotionCapacityResult:
    """Return potion capacity after ascension, relic, and explicit slot modifiers."""

    normalized_base = max(0, int(base_slots))
    ascension_slots = potion_slots_for_ascension(normalized_base, ascension_level)
    relic_bonus = relic_potion_slot_bonus(relics)
    explicit_bonus = max(0, int(bonus_slots))
    capacity = max(0, ascension_slots + relic_bonus + explicit_bonus)
    open_slots = max(0, capacity - len(current_potions))
    return PotionCapacityResult(
        capacity=capacity,
        base_slots=normalized_base,
        ascension_slots=ascension_slots,
        relic_bonus_slots=relic_bonus,
        bonus_slots=explicit_bonus,
        open_slots=open_slots,
    )


def has_open_potion_slot(
    potions: Sequence[PotionInput],
    *,
    base_slots: int = 3,
    ascension_level: int = 0,
    relics: Sequence[RelicInput] = (),
    bonus_slots: int = 0,
) -> bool:
    """Return whether the potion inventory has at least one open slot."""

    return (
        potion_capacity(
            base_slots=base_slots,
            ascension_level=ascension_level,
            relics=relics,
            bonus_slots=bonus_slots,
            current_potions=potions,
        ).open_slots
        > 0
    )


def potion_pickup_decision(
    potions: Sequence[PotionInput],
    potion: PotionInput,
    *,
    base_slots: int = 3,
    ascension_level: int = 0,
    relics: Sequence[RelicInput] = (),
    bonus_slots: int = 0,
) -> PotionPickupDecision:
    """Return whether a potion can be picked up without discarding first."""

    normalized_potions = tuple(potion_content_id(item) for item in potions)
    potion_id = potion_content_id(potion)
    capacity = potion_capacity(
        base_slots=base_slots,
        ascension_level=ascension_level,
        relics=relics,
        bonus_slots=bonus_slots,
        current_potions=normalized_potions,
    )
    can_pick_up = capacity.open_slots > 0
    return PotionPickupDecision(
        potion_id=potion_id,
        can_pick_up=can_pick_up,
        requires_discard=not can_pick_up,
        capacity=capacity.capacity,
        potions=normalized_potions,
        next_potions=normalized_potions + ((potion_id,) if can_pick_up else ()),
    )


def potion_discard_decision(
    potions: Sequence[PotionInput],
    slot_index: int,
) -> PotionDiscardDecision:
    """Return the deterministic result of discarding a potion slot."""

    normalized_potions = tuple(potion_content_id(item) for item in potions)
    if slot_index < 0 or slot_index >= len(normalized_potions):
        return PotionDiscardDecision(
            slot_index=slot_index,
            can_discard=False,
            discarded_potion_id=None,
            potions=normalized_potions,
            next_potions=normalized_potions,
        )
    discarded = normalized_potions[slot_index]
    return PotionDiscardDecision(
        slot_index=slot_index,
        can_discard=True,
        discarded_potion_id=discarded,
        potions=normalized_potions,
        next_potions=tuple(
            potion_id for index, potion_id in enumerate(normalized_potions) if index != slot_index
        ),
    )


def normalize_potion_use(
    potion: PotionInput,
    *,
    target_id: str | None = None,
    merchant_throw: bool = False,
    effects: Mapping[str, tuple[PotionEffectRule, ...]] = DEFAULT_POTION_EFFECTS,
) -> PotionUseNormalization:
    """Normalize a potion's use effect into deterministic primitive markers."""

    potion_id = potion_content_id(potion)
    if potion_id == FOUL_POTION_ID and (
        merchant_throw or (target_id is not None and _normalized_id(target_id) == "merchant")
    ):
        return PotionUseNormalization(
            potion_id=potion_id,
            effects=(_effect_from_rule(FOUL_POTION_MERCHANT_EFFECT, target_id="merchant"),),
            mode="merchant_throw",
        )

    rules = effects.get(potion_id)
    if rules is None and isinstance(potion, Mapping):
        rules = _infer_effect_rules_from_spec(potion)
    if rules is None:
        return PotionUseNormalization(potion_id=potion_id, effects=(), unsupported=True)

    return PotionUseNormalization(
        potion_id=potion_id,
        effects=tuple(_effect_from_rule(rule, target_id=target_id) for rule in rules),
    )


def supported_potion_ids(
    *,
    effects: Mapping[str, tuple[PotionEffectRule, ...]] = DEFAULT_POTION_EFFECTS,
) -> frozenset[str]:
    """Return potion ids with explicit effect normalization mappings."""

    return frozenset(effects)


def potion_content_id(potion: PotionInput) -> str:
    """Return a normalized potion id from a raw id or Codex-style mapping."""

    if isinstance(potion, str):
        return _normalized_id(potion)
    value = _first_present(potion, "id", "potion_id", "content_id", "item_id")
    if value is None:
        raise ValueError(f"Potion input is missing an id: {potion!r}")
    return _normalized_id(str(value))


def _effect_from_rule(rule: PotionEffectRule, *, target_id: str | None) -> PotionEffect:
    concrete_target_id = target_id if rule.target in {"enemy", "any"} else None
    if rule.target == "merchant":
        concrete_target_id = "merchant"
    return PotionEffect(
        kind=rule.kind,
        amount=rule.amount,
        target=rule.target,
        status=rule.status,
        duration=rule.duration,
        target_id=concrete_target_id,
        metadata=rule.metadata,
    )


def _infer_effect_rules_from_spec(
    potion: Mapping[str, object],
) -> tuple[PotionEffectRule, ...] | None:
    description_value = _first_present(potion, "description", "description_raw")
    if description_value is None:
        return None

    description = str(description_value)
    text = description.lower()
    rules: list[PotionEffectRule] = []

    damage = _first_int_after(description, "Deal")
    if damage is not None and "damage" in text:
        if "all players and enemies" in text:
            target = "all_combatants"
        elif "all enemies" in text:
            target = "all_enemies"
        else:
            target = "enemy"
        rules.append(PotionEffectRule("damage", amount=damage, target=target))

    block = _first_int_before_word(description, "Block")
    if block is not None and "block" in text:
        rules.append(PotionEffectRule("block", amount=block, target="self"))
        if "next turn" in text:
            rules.append(PotionEffectRule("next_turn_block", amount=block, target="self"))

    energy = _first_energy_amount(description)
    if energy is not None:
        rules.append(PotionEffectRule("energy", amount=energy, target="self"))

    draw = _first_int_after(description, "Draw")
    if draw is not None and "draw" in text:
        rules.append(PotionEffectRule("draw", amount=draw, target="self"))

    rules.extend(_status_rules_from_description(description))
    rules.extend(_orb_effect_rules_from_description(description))

    return tuple(rules) if rules else None


def _status_rules_from_description(description: str) -> tuple[PotionEffectRule, ...]:
    text = description.lower()
    rules: list[PotionEffectRule] = []
    status_names = (
        "Strength",
        "Dexterity",
        "Focus",
        "Thorns",
        "Buffer",
        "Ritual",
        "Plating",
        "Intangible",
        "Regen",
        "Weak",
        "Vulnerable",
        "Poison",
        "Doom",
    )
    for status in status_names:
        pattern = rf"\[blue\](\d+)\[/blue\]\s+\[gold\]{re.escape(status)}\[/gold\]"
        match = re.search(pattern, description, flags=re.IGNORECASE)
        if match is None:
            continue
        target = "self" if f"gain {match.group(0).lower()}" in text else "enemy"
        if "all enemies" in text and target == "enemy":
            target = "all_enemies"
        rules.append(
            PotionEffectRule(
                "status",
                amount=int(match.group(1)),
                target=target,
                status=_normalized_id(status),
            )
        )
    return tuple(rules)


def _orb_effect_rules_from_description(description: str) -> tuple[PotionEffectRule, ...]:
    normalized = _plain_normalized_text(description)
    rules: list[PotionEffectRule] = []

    slot_match = re.search(r"\bgain\s+(\d+)\s+orb slots?\b", normalized)
    if slot_match is not None:
        rules.append(
            PotionEffectRule(
                "orb_slot_delta",
                amount=int(slot_match.group(1)),
                target="self",
            )
        )

    channel_match = re.search(
        r"\bchannel\s+(?:(\d+)|a|an)?\s*(lightning|frost|dark|plasma|glass)\b",
        normalized,
    )
    if channel_match is not None:
        amount: int | None = int(channel_match.group(1) or 1)
        metadata: dict[str, object] = {"orb": channel_match.group(2)}
        if "for each of your orb slots" in normalized:
            amount = None
            metadata["amount"] = "orb_slots"
        rules.append(
            PotionEffectRule(
                "channel_orb",
                amount=amount,
                target="self",
                metadata=metadata,
            )
        )

    return tuple(rules)


def _plain_normalized_text(description: str) -> str:
    without_tags = re.sub(r"\[/?(?:blue|gold|green|red|white)]", "", description)
    return " ".join(without_tags.lower().split())


def _first_int_after(text: str, lead_word: str) -> int | None:
    match = re.search(
        rf"{re.escape(lead_word)}[^\d]*(?:\[blue\])?(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def _first_int_before_word(text: str, word: str) -> int | None:
    match = re.search(
        rf"(?:\[blue\])?(\d+)(?:\[/blue\])?[^\n.]*{re.escape(word)}",
        text,
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def _first_energy_amount(text: str) -> int | None:
    match = re.search(r"\[energy:(\d+)\]", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _first_present(item: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        value = item.get(key)
        if value is not None:
            return value
    return None


def _normalized_id(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("'", "")
        .replace(" ", "_")
        .replace("-", "_")
    )
