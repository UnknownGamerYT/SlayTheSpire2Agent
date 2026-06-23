"""Pure reward-generation trigger helpers.

The engine owns reward drawing and state mutation.  This module only turns
reward-generation trigger inputs into small deltas and normalized trigger
markers that integration code can apply later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from .relics import RelicInput, relic_content_id
from .triggers import (
    GameTrigger,
    TriggerBlocker,
    TriggerContext,
    TriggerDispatcher,
    TriggerEffect,
    TriggerResolution,
    resolve_game_trigger,
)


@dataclass(frozen=True, slots=True)
class RewardGenerationContext:
    """Data available while preparing a reward bundle."""

    source: str = "combat"
    encounter_type: str | Enum | None = None
    card_choice_count: int | None = 3
    card_option_count: int = 0
    card_option_group_count: int = 0
    relic_count: int = 0
    potion_count: int = 0
    gold: int = 0
    owned_relics: Sequence[RelicInput] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _normalized_id(self.source))
        if self.encounter_type is not None:
            object.__setattr__(
                self,
                "encounter_type",
                _normalized_id(_enum_value(self.encounter_type)),
            )
        if self.card_choice_count is not None:
            object.__setattr__(self, "card_choice_count", max(0, int(self.card_choice_count)))
        object.__setattr__(self, "card_option_count", max(0, int(self.card_option_count)))
        object.__setattr__(
            self,
            "card_option_group_count",
            max(0, int(self.card_option_group_count)),
        )
        object.__setattr__(self, "relic_count", max(0, int(self.relic_count)))
        object.__setattr__(self, "potion_count", max(0, int(self.potion_count)))
        object.__setattr__(self, "gold", max(0, int(self.gold)))
        object.__setattr__(self, "owned_relics", tuple(self.owned_relics))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def base_card_choice_count(self) -> int:
        """Return the count card-choice modifiers should adjust."""

        if self.card_choice_count is not None:
            return self.card_choice_count
        return self.card_option_count

    @property
    def has_card_reward_choices(self) -> bool:
        """Return whether this generation pass is expected to create card choices."""

        return (
            self.base_card_choice_count > 0
            or self.card_option_count > 0
            or self.card_option_group_count > 0
            or bool(self.metadata.get("generates_card_reward"))
        )

    @property
    def normalized_encounter_type(self) -> str | None:
        return str(self.encounter_type) if self.encounter_type is not None else None

    @property
    def owned_relic_ids(self) -> tuple[str, ...]:
        seen: set[str] = set()
        relic_ids: list[str] = []
        for relic in self.owned_relics:
            relic_id = relic_content_id(relic)
            if relic_id in seen:
                continue
            seen.add(relic_id)
            relic_ids.append(relic_id)
        return tuple(relic_ids)

    def trigger_context(self) -> TriggerContext:
        """Build the shared trigger context used by custom dispatchers."""

        metadata = {
            **dict(self.metadata),
            "reward_source": self.source,
            "card_choice_count": self.base_card_choice_count,
            "card_option_count": self.card_option_count,
            "card_option_group_count": self.card_option_group_count,
            "relic_count": self.relic_count,
            "potion_count": self.potion_count,
            "gold": self.gold,
        }
        if self.normalized_encounter_type is not None:
            metadata["encounter_type"] = self.normalized_encounter_type
        return TriggerContext(
            GameTrigger.COMBAT_REWARD_GENERATED,
            encounter_type=self.normalized_encounter_type,
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class RewardModifierSpec:
    """A small, table-driven reward modifier marker and delta."""

    content_id: str
    kind: str
    source_kind: str = "relic"
    amount: int | None = None
    card_choice_delta: int = 0
    card_reward_group_delta: int = 0
    relic_count_delta: int = 0
    potion_count_delta: int = 0
    gold_delta: int = 0
    apply_to_sources: frozenset[str] = field(default_factory=frozenset)
    apply_to_encounters: frozenset[str] = field(default_factory=frozenset)
    metadata_equals: Mapping[str, object] = field(default_factory=dict)
    requires_card_reward: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_id", _normalized_id(self.content_id))
        object.__setattr__(self, "kind", _normalized_id(self.kind))
        object.__setattr__(self, "source_kind", _normalized_id(self.source_kind))
        if self.amount is not None:
            object.__setattr__(self, "amount", int(self.amount))
        object.__setattr__(self, "card_choice_delta", int(self.card_choice_delta))
        object.__setattr__(self, "card_reward_group_delta", int(self.card_reward_group_delta))
        object.__setattr__(self, "relic_count_delta", int(self.relic_count_delta))
        object.__setattr__(self, "potion_count_delta", int(self.potion_count_delta))
        object.__setattr__(self, "gold_delta", int(self.gold_delta))
        object.__setattr__(
            self,
            "apply_to_sources",
            frozenset(_normalized_id(source) for source in self.apply_to_sources),
        )
        object.__setattr__(
            self,
            "apply_to_encounters",
            frozenset(_normalized_id(encounter) for encounter in self.apply_to_encounters),
        )
        object.__setattr__(self, "metadata_equals", dict(self.metadata_equals))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class RewardGenerationResult:
    """Reward trigger markers plus pure deltas for later engine application."""

    context: RewardGenerationContext
    trigger_resolution: TriggerResolution
    effects: tuple[TriggerEffect, ...] = ()
    blockers: tuple[TriggerBlocker, ...] = ()
    card_choice_delta: int = 0
    card_reward_group_delta: int = 0
    relic_count_delta: int = 0
    potion_count_delta: int = 0
    gold_delta: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "effects", tuple(self.effects))
        object.__setattr__(self, "blockers", tuple(self.blockers))

    @property
    def card_choice_count(self) -> int:
        return max(0, self.context.base_card_choice_count + self.card_choice_delta)

    @property
    def card_reward_group_count(self) -> int:
        return max(0, self.context.card_option_group_count + self.card_reward_group_delta)

    @property
    def relic_count(self) -> int:
        return max(0, self.context.relic_count + self.relic_count_delta)

    @property
    def potion_count(self) -> int:
        return max(0, self.context.potion_count + self.potion_count_delta)

    @property
    def gold(self) -> int:
        return max(0, self.context.gold + self.gold_delta)


def resolve_reward_generation_triggers(
    context: RewardGenerationContext,
    *,
    modifiers: Sequence[RewardModifierSpec] = (),
    dispatcher: TriggerDispatcher | None = None,
    include_blockers: bool = True,
) -> RewardGenerationResult:
    """Resolve pure reward-generation modifiers and custom trigger handlers."""

    dispatch = resolve_game_trigger(
        GameTrigger.COMBAT_REWARD_GENERATED,
        relics=context.owned_relics,
        context=context.trigger_context(),
        dispatcher=dispatcher,
        include_blockers=include_blockers,
    )
    active_modifiers = _active_default_modifiers(context) + tuple(
        modifier for modifier in modifiers if _modifier_applies(modifier, context)
    )

    card_choice_delta = 0
    card_reward_group_delta = 0
    relic_count_delta = 0
    potion_count_delta = 0
    gold_delta = dispatch.gold_delta
    effects: list[TriggerEffect] = []

    for modifier in active_modifiers:
        card_choice_delta += modifier.card_choice_delta
        card_reward_group_delta += modifier.card_reward_group_delta
        relic_count_delta += modifier.relic_count_delta
        potion_count_delta += modifier.potion_count_delta
        gold_delta += modifier.gold_delta
        effects.append(
            _effect_from_modifier(
                modifier,
                context,
                card_choice_delta=card_choice_delta,
            )
        )

    effects.extend(dispatch.effects)
    combined_resolution = TriggerResolution(
        trigger=GameTrigger.COMBAT_REWARD_GENERATED,
        effects=tuple(effects),
        blockers=dispatch.blockers,
        gold_delta=gold_delta,
        hp_delta=dispatch.hp_delta,
        max_hp_delta=dispatch.max_hp_delta,
        block_delta=dispatch.block_delta,
        energy_delta=dispatch.energy_delta,
        combat_relic_resolution=dispatch.combat_relic_resolution,
        relic_hook_resolution=dispatch.relic_hook_resolution,
    )
    return RewardGenerationResult(
        context=context,
        trigger_resolution=combined_resolution,
        effects=combined_resolution.effects,
        blockers=combined_resolution.blockers,
        card_choice_delta=card_choice_delta,
        card_reward_group_delta=card_reward_group_delta,
        relic_count_delta=relic_count_delta,
        potion_count_delta=potion_count_delta,
        gold_delta=gold_delta,
    )


def _active_default_modifiers(context: RewardGenerationContext) -> tuple[RewardModifierSpec, ...]:
    owned_relic_ids = set(context.owned_relic_ids)
    return tuple(
        modifier
        for modifier in DEFAULT_REWARD_MODIFIERS
        if modifier.content_id in owned_relic_ids and _modifier_applies(modifier, context)
    )


def _modifier_applies(
    modifier: RewardModifierSpec,
    context: RewardGenerationContext,
) -> bool:
    if modifier.apply_to_sources and context.source not in modifier.apply_to_sources:
        return False
    encounter = context.normalized_encounter_type
    if modifier.apply_to_encounters and encounter not in modifier.apply_to_encounters:
        return False
    for key, expected in modifier.metadata_equals.items():
        if context.metadata.get(key) != expected:
            return False
    return not modifier.requires_card_reward or context.has_card_reward_choices


def _effect_from_modifier(
    modifier: RewardModifierSpec,
    context: RewardGenerationContext,
    *,
    card_choice_delta: int,
) -> TriggerEffect:
    metadata: dict[str, object] = {
        **dict(modifier.metadata),
        "reward_source": context.source,
        "card_choice_count": max(0, context.base_card_choice_count + card_choice_delta),
    }
    if context.normalized_encounter_type is not None:
        metadata["encounter_type"] = context.normalized_encounter_type
    if modifier.card_choice_delta:
        metadata["card_choice_delta"] = modifier.card_choice_delta
    if modifier.card_reward_group_delta:
        metadata["card_reward_group_delta"] = modifier.card_reward_group_delta
    if modifier.relic_count_delta:
        metadata["relic_count_delta"] = modifier.relic_count_delta
    if modifier.potion_count_delta:
        metadata["potion_count_delta"] = modifier.potion_count_delta
    if modifier.gold_delta:
        metadata["gold_delta"] = modifier.gold_delta
    if modifier.metadata_equals:
        metadata["metadata_equals"] = dict(modifier.metadata_equals)

    return TriggerEffect(
        kind=modifier.kind,
        trigger=GameTrigger.COMBAT_REWARD_GENERATED,
        source_kind=modifier.source_kind,
        content_id=modifier.content_id,
        amount=modifier.amount if modifier.amount is not None else _modifier_amount(modifier),
        target_id="reward",
        source_id=modifier.content_id,
        metadata=metadata,
    )


def _modifier_amount(modifier: RewardModifierSpec) -> int | None:
    for amount in (
        modifier.card_choice_delta,
        modifier.card_reward_group_delta,
        modifier.relic_count_delta,
        modifier.potion_count_delta,
        modifier.gold_delta,
    ):
        if amount:
            return amount
    return None


def _enum_value(value: str | Enum) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _normalized_id(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("'", "")
        .replace(" ", "_")
        .replace("-", "_")
    )


DEFAULT_REWARD_MODIFIERS: tuple[RewardModifierSpec, ...] = (
    RewardModifierSpec(
        content_id="amethyst_aubergine",
        kind="reward_gold_delta",
        amount=15,
        gold_delta=15,
        apply_to_sources=frozenset({"combat"}),
        metadata={"modifier": "amethyst_aubergine"},
    ),
    RewardModifierSpec(
        content_id="question_card",
        kind="reward_card_choice_delta",
        card_choice_delta=1,
        requires_card_reward=True,
        metadata={"modifier": "question_card"},
    ),
    RewardModifierSpec(
        content_id="busted_crown",
        kind="reward_card_choice_delta",
        card_choice_delta=-2,
        requires_card_reward=True,
        metadata={"modifier": "busted_crown"},
    ),
    RewardModifierSpec(
        content_id="prayer_wheel",
        kind="reward_extra_card_group",
        amount=1,
        card_reward_group_delta=1,
        apply_to_sources=frozenset({"combat"}),
        apply_to_encounters=frozenset({"normal"}),
        requires_card_reward=True,
        metadata={"modifier": "prayer_wheel"},
    ),
    RewardModifierSpec(
        content_id="lava_rock",
        kind="reward_extra_relic",
        amount=1,
        relic_count_delta=1,
        apply_to_sources=frozenset({"combat"}),
        apply_to_encounters=frozenset({"boss"}),
        metadata_equals={"act": 1},
        metadata={"modifier": "lava_rock", "condition": "act_1_boss"},
    ),
    RewardModifierSpec(
        content_id="white_beast_statue",
        kind="reward_extra_potion",
        amount=1,
        potion_count_delta=1,
        apply_to_sources=frozenset({"combat"}),
        metadata={"modifier": "white_beast_statue"},
    ),
)


__all__ = [
    "DEFAULT_REWARD_MODIFIERS",
    "RewardGenerationContext",
    "RewardGenerationResult",
    "RewardModifierSpec",
    "resolve_reward_generation_triggers",
]
