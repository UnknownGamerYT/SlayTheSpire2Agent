"""Pure Soul and Summon helper descriptors for Necrobinder mechanics.

The combat engine owns mutation.  This module only identifies Soul cards,
counts them across combat piles, resolves small dynamic X-cost amounts, and
emits mapping-friendly trigger descriptors that integration code can apply at
the right combat timing hook.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from sts2sim.engine.models import CardInstance, CombatState

type CardLike = CardInstance | Mapping[str, object] | str
type EffectStep = Mapping[str, object]

SOUL_CARD_ID = "soul"
SUMMON_RESOURCE_ID = "summon"


class SoulPileZone(str, Enum):
    """Combat piles that may contain Soul cards."""

    HAND = "hand"
    DRAW_PILE = "draw_pile"
    DISCARD_PILE = "discard_pile"
    EXHAUST_PILE = "exhaust_pile"
    ALL = "all"


@dataclass(frozen=True, slots=True)
class SoulPileCounts:
    """Soul counts split by combat pile."""

    hand: int = 0
    draw_pile: int = 0
    discard_pile: int = 0
    exhaust_pile: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "hand", max(0, int(self.hand)))
        object.__setattr__(self, "draw_pile", max(0, int(self.draw_pile)))
        object.__setattr__(self, "discard_pile", max(0, int(self.discard_pile)))
        object.__setattr__(self, "exhaust_pile", max(0, int(self.exhaust_pile)))

    @property
    def total(self) -> int:
        return self.hand + self.draw_pile + self.discard_pile + self.exhaust_pile

    def for_zone(self, zone: SoulPileZone | str) -> int:
        normalized = soul_pile_zone(zone)
        if normalized is SoulPileZone.HAND:
            return self.hand
        if normalized is SoulPileZone.DRAW_PILE:
            return self.draw_pile
        if normalized is SoulPileZone.DISCARD_PILE:
            return self.discard_pile
        if normalized is SoulPileZone.EXHAUST_PILE:
            return self.exhaust_pile
        return self.total

    def as_mapping(self) -> dict[str, int]:
        return {
            "hand": self.hand,
            "draw_pile": self.draw_pile,
            "discard_pile": self.discard_pile,
            "exhaust_pile": self.exhaust_pile,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class SoulScalingResult:
    """Resolved amount for effects that scale per Soul in a pile."""

    zone: SoulPileZone
    soul_count: int
    amount_per_soul: int
    base_amount: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "soul_count", max(0, int(self.soul_count)))
        object.__setattr__(self, "amount_per_soul", int(self.amount_per_soul))
        object.__setattr__(self, "base_amount", int(self.base_amount))

    @property
    def bonus_amount(self) -> int:
        return self.soul_count * self.amount_per_soul

    @property
    def total_amount(self) -> int:
        return self.base_amount + self.bonus_amount

    def as_mapping(self) -> dict[str, object]:
        return {
            "zone": self.zone.value,
            "soul_count": self.soul_count,
            "amount_per_soul": self.amount_per_soul,
            "base_amount": self.base_amount,
            "bonus_amount": self.bonus_amount,
            "total_amount": self.total_amount,
        }


@dataclass(frozen=True, slots=True)
class DynamicAmountResult:
    """Resolved X-cost amount for Summon or Soul creation."""

    kind: str
    energy_spent: int
    amount_per_energy: int = 1
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _normalized_id(self.kind))
        object.__setattr__(self, "energy_spent", max(0, int(self.energy_spent)))
        object.__setattr__(self, "amount_per_energy", max(0, int(self.amount_per_energy)))
        object.__setattr__(self, "metadata", _clone_mapping(self.metadata))

    @property
    def amount(self) -> int:
        return self.energy_spent * self.amount_per_energy

    def as_mapping(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "energy_spent": self.energy_spent,
            "amount_per_energy": self.amount_per_energy,
            "amount": self.amount,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SoulPlayTriggerDescriptor:
    """Descriptor for powers that trigger when a Soul card is played."""

    source_id: str
    effects: tuple[EffectStep, ...]
    duration: str = "combat"
    text: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _normalized_id(self.source_id))
        object.__setattr__(self, "duration", _normalized_id(self.duration) or "combat")
        object.__setattr__(
            self,
            "effects",
            tuple(_clone_mapping(effect) for effect in self.effects),
        )
        object.__setattr__(self, "metadata", _clone_mapping(self.metadata))

    @property
    def condition(self) -> Mapping[str, object]:
        return {"card_id": SOUL_CARD_ID, "is_soul": True}

    def combat_trigger_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "trigger": "card_played",
            "duration": self.duration,
            "condition": dict(self.condition),
            "effects": self.effects,
            "source_id": self.source_id,
            "source_card_id": self.source_id,
        }
        if self.text:
            payload["text"] = self.text
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_mapping(self) -> dict[str, object]:
        return {"combat_trigger": self.combat_trigger_payload()}


@dataclass(frozen=True, slots=True)
class TimedSummonTriggerDescriptor:
    """Descriptor for delayed or repeated Summon grants."""

    source_id: str
    amount: int
    trigger: str = "turn_start"
    duration: str = "once"
    delay: int = 0
    text: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _normalized_id(self.source_id))
        object.__setattr__(self, "amount", max(0, int(self.amount)))
        object.__setattr__(self, "trigger", _normalized_id(self.trigger) or "turn_start")
        object.__setattr__(self, "duration", _normalized_id(self.duration) or "once")
        object.__setattr__(self, "delay", max(0, int(self.delay)))
        object.__setattr__(self, "metadata", _clone_mapping(self.metadata))

    @property
    def effects(self) -> tuple[EffectStep, ...]:
        return (summon_effect_step(self.amount, source=self.source_id),)

    def combat_trigger_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "trigger": self.trigger,
            "duration": self.duration,
            "effects": self.effects,
            "source_id": self.source_id,
            "source_card_id": self.source_id,
        }
        if self.delay:
            payload["delay"] = self.delay
        if self.text:
            payload["text"] = self.text
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_mapping(self) -> dict[str, object]:
        return {"combat_trigger": self.combat_trigger_payload()}


@dataclass(frozen=True, slots=True)
class DynamicSoulSummonResult:
    """Combined helper for Dirge-style X-cost Summon plus Soul creation."""

    summon: DynamicAmountResult
    souls: DynamicAmountResult
    source_id: str = "dirge"
    destination: str = "draw"
    upgraded_souls: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _normalized_id(self.source_id))
        object.__setattr__(self, "destination", _card_destination(self.destination))

    @property
    def summon_amount(self) -> int:
        return self.summon.amount

    @property
    def soul_count(self) -> int:
        return self.souls.amount

    def effect_steps(self) -> tuple[EffectStep, ...]:
        steps: list[EffectStep] = []
        if self.summon_amount > 0:
            steps.append(summon_effect_step(self.summon_amount, source=self.source_id))
        if self.soul_count > 0:
            steps.append(
                add_soul_effect_step(
                    self.soul_count,
                    destination=self.destination,
                    upgraded=self.upgraded_souls,
                )
            )
        return tuple(steps)

    def as_mapping(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "destination": self.destination,
            "upgraded_souls": self.upgraded_souls,
            "summon": self.summon.as_mapping(),
            "souls": self.souls.as_mapping(),
            "effect_steps": self.effect_steps(),
        }


def soul_pile_zone(zone: SoulPileZone | str) -> SoulPileZone:
    if isinstance(zone, SoulPileZone):
        return zone
    normalized = _normalized_id(zone)
    alias = _ZONE_ALIASES.get(normalized, normalized)
    return SoulPileZone(alias)


def is_soul_card(card: CardLike) -> bool:
    """Return whether a card-like value is the generated Soul card."""

    card_id, name = _card_id_and_name(card)
    if _normalized_card_token(card_id) == SOUL_CARD_ID:
        return True
    return card_id == "" and _normalized_card_token(name) == SOUL_CARD_ID


def count_souls(cards: Iterable[CardLike]) -> int:
    """Count generated Soul cards in a card sequence."""

    return sum(1 for card in cards if is_soul_card(card))


def soul_pile_counts(
    *,
    hand: Iterable[CardLike] = (),
    draw_pile: Iterable[CardLike] = (),
    discard_pile: Iterable[CardLike] = (),
    exhaust_pile: Iterable[CardLike] = (),
) -> SoulPileCounts:
    """Count Souls across the four combat card piles."""

    return SoulPileCounts(
        hand=count_souls(hand),
        draw_pile=count_souls(draw_pile),
        discard_pile=count_souls(discard_pile),
        exhaust_pile=count_souls(exhaust_pile),
    )


def soul_pile_counts_from_combat(combat: CombatState | Mapping[str, object]) -> SoulPileCounts:
    """Count Souls from a ``CombatState`` or a combat-like mapping."""

    if isinstance(combat, CombatState):
        return soul_pile_counts(
            hand=combat.hand,
            draw_pile=combat.draw_pile,
            discard_pile=combat.discard_pile,
            exhaust_pile=combat.exhaust_pile,
        )
    return soul_pile_counts(
        hand=_card_items(combat.get("hand", ())),
        draw_pile=_card_items(combat.get("draw_pile", combat.get("draw", ()))),
        discard_pile=_card_items(combat.get("discard_pile", combat.get("discard", ()))),
        exhaust_pile=_card_items(combat.get("exhaust_pile", combat.get("exhaust", ()))),
    )


def resolve_soul_count_scaling(
    counts: SoulPileCounts,
    *,
    zone: SoulPileZone | str,
    amount_per_soul: int,
    base_amount: int = 0,
) -> SoulScalingResult:
    """Resolve effects such as Soul Storm's per-Soul exhaust-pile bonus."""

    normalized_zone = soul_pile_zone(zone)
    return SoulScalingResult(
        zone=normalized_zone,
        soul_count=counts.for_zone(normalized_zone),
        amount_per_soul=amount_per_soul,
        base_amount=base_amount,
    )


def soul_count_scaled_amount(
    counts: SoulPileCounts,
    *,
    zone: SoulPileZone | str,
    amount_per_soul: int,
    base_amount: int = 0,
) -> int:
    return resolve_soul_count_scaling(
        counts,
        zone=zone,
        amount_per_soul=amount_per_soul,
        base_amount=base_amount,
    ).total_amount


def summon_effect_step(amount: int, *, source: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "resource": SUMMON_RESOURCE_ID,
        "amount": max(0, int(amount)),
    }
    if source:
        payload["source"] = _normalized_id(source)
    return {"player_resource": payload}


def enemy_hp_loss_effect_step(
    amount: int,
    *,
    target: str = "random_enemy",
    source: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "target": _normalized_id(target) or "random_enemy",
        "amount": max(0, int(amount)),
    }
    if source:
        payload["source"] = _normalized_id(source)
    return {"enemy_hp_loss": payload}


def soul_card_payload(*, upgraded: bool = False) -> dict[str, object]:
    """Return the generated temporary Soul card payload used by add-card effects."""

    draw = 3 if upgraded else 2
    return {
        "card": {
            "id": "SOUL",
            "name": "Soul+" if upgraded else "Soul",
            "type": "Skill",
            "target": "Self",
            "cost": 1,
            "draw": draw,
            "description": f"Draw {draw} cards.",
            "keywords_key": ("Exhaust",),
            "exhausts": True,
            "upgraded": upgraded,
        },
        "temporary": True,
    }


def add_soul_effect_step(
    count: int,
    *,
    destination: str = "draw",
    upgraded: bool = False,
) -> dict[str, object]:
    normalized_destination = _card_destination(destination)
    return {
        f"add_card_to_{normalized_destination}": tuple(
            soul_card_payload(upgraded=upgraded) for _ in range(max(0, int(count)))
        )
    }


def soul_play_trigger(
    *,
    source_id: str,
    effects: Sequence[EffectStep],
    duration: str = "combat",
    text: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SoulPlayTriggerDescriptor:
    return SoulPlayTriggerDescriptor(
        source_id=source_id,
        effects=tuple(effects),
        duration=duration,
        text=text,
        metadata=metadata or {},
    )


def devour_life_soul_play_trigger(
    amount: int,
    *,
    source_id: str = "devour_life",
    duration: str = "combat",
) -> SoulPlayTriggerDescriptor:
    return soul_play_trigger(
        source_id=source_id,
        duration=duration,
        effects=(summon_effect_step(amount, source=source_id),),
        text=f"Whenever you play a Soul, Summon {max(0, int(amount))}.",
        metadata={"style": "devour_life"},
    )


def haunt_soul_play_trigger(
    amount: int,
    *,
    source_id: str = "haunt",
    duration: str = "combat",
) -> SoulPlayTriggerDescriptor:
    return soul_play_trigger(
        source_id=source_id,
        duration=duration,
        effects=(enemy_hp_loss_effect_step(amount, target="random_enemy", source=source_id),),
        text=f"Whenever you play a Soul, a random enemy loses {max(0, int(amount))} HP.",
        metadata={"style": "haunt"},
    )


def dynamic_summon_amount(
    energy_spent: int,
    *,
    amount_per_energy: int = 1,
    metadata: Mapping[str, object] | None = None,
) -> DynamicAmountResult:
    return DynamicAmountResult(
        kind="dynamic_summon_amount",
        energy_spent=energy_spent,
        amount_per_energy=amount_per_energy,
        metadata=metadata or {},
    )


def dynamic_soul_creation_count(
    energy_spent: int,
    *,
    souls_per_energy: int = 1,
    metadata: Mapping[str, object] | None = None,
) -> DynamicAmountResult:
    return DynamicAmountResult(
        kind="dynamic_soul_creation_count",
        energy_spent=energy_spent,
        amount_per_energy=souls_per_energy,
        metadata=metadata or {},
    )


def resolve_dirge_x_cost(
    energy_spent: int,
    *,
    summon_per_energy: int = 3,
    souls_per_energy: int = 1,
    upgraded: bool = False,
    destination: str = "draw",
    source_id: str = "dirge",
) -> DynamicSoulSummonResult:
    return DynamicSoulSummonResult(
        source_id=source_id,
        destination=destination,
        upgraded_souls=upgraded,
        summon=dynamic_summon_amount(
            energy_spent,
            amount_per_energy=summon_per_energy,
            metadata={"card": "dirge"},
        ),
        souls=dynamic_soul_creation_count(
            energy_spent,
            souls_per_energy=souls_per_energy,
            metadata={"card": "dirge"},
        ),
    )


def timed_summon_trigger(
    amount: int,
    *,
    source_id: str,
    trigger: str = "turn_start",
    duration: str = "once",
    delay: int = 0,
    text: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> TimedSummonTriggerDescriptor:
    return TimedSummonTriggerDescriptor(
        source_id=source_id,
        amount=amount,
        trigger=trigger,
        duration=duration,
        delay=delay,
        text=text,
        metadata=metadata or {},
    )


def next_turn_summon_trigger(
    amount: int,
    *,
    source_id: str = "summon_next_turn",
) -> TimedSummonTriggerDescriptor:
    return timed_summon_trigger(
        amount,
        source_id=source_id,
        trigger="turn_start",
        duration="once",
        text=f"At the start of your next turn, Summon {max(0, int(amount))}.",
        metadata={"style": "next_turn_summon"},
    )


def _card_id_and_name(card: CardLike) -> tuple[str, str]:
    if isinstance(card, CardInstance):
        return card.card_id, card.name
    if isinstance(card, str):
        return card, ""
    source = _unwrap_card_payload(card)
    card_id = _first_string(source, "card_id", "id", "key")
    return card_id, _first_string(source, "name")


def _unwrap_card_payload(card: Mapping[str, object]) -> Mapping[str, object]:
    nested = card.get("card")
    if any(key in card for key in ("card_id", "id", "key", "name")):
        return card
    if isinstance(nested, Mapping):
        return nested
    return card


def _first_string(source: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = source.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _card_items(value: object) -> tuple[CardLike, ...]:
    if value is None:
        return ()
    if isinstance(value, CardInstance | Mapping | str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items: list[CardLike] = []
        for item in value:
            if isinstance(item, CardInstance | Mapping | str):
                items.append(item)
        return tuple(items)
    return ()


def _normalized_card_token(value: str) -> str:
    return _normalized_id(value).replace("+", "").removesuffix("_plus")


def _card_destination(value: object) -> str:
    normalized = _normalized_id(value)
    alias = _DESTINATION_ALIASES.get(normalized, normalized)
    if alias not in {"hand", "draw", "discard", "exhaust"}:
        raise ValueError(f"Unsupported Soul card destination: {value!r}")
    return alias


def _normalized_id(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("'", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _clone_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _clone_value(item) for key, item in value.items()}


def _clone_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _clone_mapping(value)
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_clone_value(item) for item in value)
    return value


_ZONE_ALIASES = {
    "draw": SoulPileZone.DRAW_PILE.value,
    "discard": SoulPileZone.DISCARD_PILE.value,
    "exhaust": SoulPileZone.EXHAUST_PILE.value,
    "drawpile": SoulPileZone.DRAW_PILE.value,
    "discardpile": SoulPileZone.DISCARD_PILE.value,
    "exhaustpile": SoulPileZone.EXHAUST_PILE.value,
    "total": SoulPileZone.ALL.value,
}

_DESTINATION_ALIASES = {
    "draw_pile": "draw",
    "discard_pile": "discard",
    "exhaust_pile": "exhaust",
    "drawpile": "draw",
    "discardpile": "discard",
    "exhaustpile": "exhaust",
}

__all__ = [
    "SOUL_CARD_ID",
    "SUMMON_RESOURCE_ID",
    "DynamicAmountResult",
    "DynamicSoulSummonResult",
    "EffectStep",
    "SoulPileCounts",
    "SoulPileZone",
    "SoulPlayTriggerDescriptor",
    "SoulScalingResult",
    "TimedSummonTriggerDescriptor",
    "add_soul_effect_step",
    "count_souls",
    "devour_life_soul_play_trigger",
    "dynamic_soul_creation_count",
    "dynamic_summon_amount",
    "enemy_hp_loss_effect_step",
    "haunt_soul_play_trigger",
    "is_soul_card",
    "next_turn_summon_trigger",
    "resolve_dirge_x_cost",
    "resolve_soul_count_scaling",
    "soul_card_payload",
    "soul_count_scaled_amount",
    "soul_pile_counts",
    "soul_pile_counts_from_combat",
    "soul_pile_zone",
    "soul_play_trigger",
    "summon_effect_step",
    "timed_summon_trigger",
]
