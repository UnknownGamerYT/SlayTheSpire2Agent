"""Campfire action availability and deterministic action results."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from sts2sim.content.sources import STS1_COMPAT_SOURCE, SourceRef

from .ascension import AscensionFlag, ascension_enabled


class CampfireAction(str, Enum):
    REST = "rest"
    SMITH = "smith"
    RECALL = "recall"
    DIG = "dig"
    LIFT = "lift"
    TOKE = "toke"


@dataclass(frozen=True, slots=True)
class CampfireRules:
    rest_heal_fraction: float = 0.30
    ascension_rest_heal_fraction: float = 0.20
    min_rest_heal: int = 1
    max_lift_count: int = 3
    source: SourceRef = STS1_COMPAT_SOURCE


@dataclass(frozen=True, slots=True)
class CampfireState:
    current_hp: int
    max_hp: int
    upgradeable_card_ids: frozenset[str] = frozenset()
    removable_card_ids: frozenset[str] = frozenset()
    has_ruby_key: bool = False
    can_recall: bool = True
    has_shovel: bool = False
    has_girya: bool = False
    lift_count: int = 0
    has_peace_pipe: bool = False


@dataclass(frozen=True, slots=True)
class CampfireChoice:
    action: CampfireAction
    target_id: str | None = None


@dataclass(frozen=True, slots=True)
class CampfireResult:
    choice: CampfireChoice
    state: CampfireState
    hp_delta: int = 0
    upgraded_card_id: str | None = None
    removed_card_id: str | None = None
    grants_relic: bool = False
    source: SourceRef = STS1_COMPAT_SOURCE


DEFAULT_CAMPFIRE_RULES = CampfireRules()


def rest_heal_amount(
    max_hp: int,
    *,
    ascension_level: int = 0,
    rules: CampfireRules = DEFAULT_CAMPFIRE_RULES,
) -> int:
    fraction = (
        rules.ascension_rest_heal_fraction
        if ascension_enabled(ascension_level, AscensionFlag.REDUCED_REST_HEAL)
        else rules.rest_heal_fraction
    )
    return max(rules.min_rest_heal, int(max_hp * fraction))


def available_campfire_actions(
    state: CampfireState,
    *,
    ascension_level: int = 0,
    rules: CampfireRules = DEFAULT_CAMPFIRE_RULES,
) -> frozenset[CampfireAction]:
    actions = {CampfireAction.REST}
    if state.upgradeable_card_ids:
        actions.add(CampfireAction.SMITH)
    if state.can_recall and not state.has_ruby_key:
        actions.add(CampfireAction.RECALL)
    if state.has_shovel:
        actions.add(CampfireAction.DIG)
    if state.has_girya and state.lift_count < rules.max_lift_count:
        actions.add(CampfireAction.LIFT)
    if state.has_peace_pipe and state.removable_card_ids:
        actions.add(CampfireAction.TOKE)
    return frozenset(actions)


def resolve_campfire_action(
    choice: CampfireChoice,
    state: CampfireState,
    *,
    ascension_level: int = 0,
    rules: CampfireRules = DEFAULT_CAMPFIRE_RULES,
) -> CampfireResult:
    available = available_campfire_actions(
        state,
        ascension_level=ascension_level,
        rules=rules,
    )
    if choice.action not in available:
        raise ValueError(f"Campfire action is not available: {choice.action.value}")

    if choice.action is CampfireAction.REST:
        heal = min(
            state.max_hp - state.current_hp,
            rest_heal_amount(state.max_hp, ascension_level=ascension_level, rules=rules),
        )
        next_state = replace(state, current_hp=state.current_hp + max(0, heal))
        return CampfireResult(
            choice=choice,
            state=next_state,
            hp_delta=max(0, heal),
            source=rules.source,
        )

    if choice.action is CampfireAction.SMITH:
        if choice.target_id is None or choice.target_id not in state.upgradeable_card_ids:
            raise ValueError("Smith requires a target id from upgradeable_card_ids.")
        next_upgradeable = frozenset(
            card_id
            for card_id in state.upgradeable_card_ids
            if card_id != choice.target_id
        )
        next_state = replace(state, upgradeable_card_ids=next_upgradeable)
        return CampfireResult(
            choice=choice,
            state=next_state,
            upgraded_card_id=choice.target_id,
            source=rules.source,
        )

    if choice.action is CampfireAction.RECALL:
        next_state = replace(state, has_ruby_key=True)
        return CampfireResult(choice=choice, state=next_state, source=rules.source)

    if choice.action is CampfireAction.DIG:
        return CampfireResult(choice=choice, state=state, grants_relic=True, source=rules.source)

    if choice.action is CampfireAction.LIFT:
        next_state = replace(state, lift_count=state.lift_count + 1)
        return CampfireResult(choice=choice, state=next_state, source=rules.source)

    if choice.action is CampfireAction.TOKE:
        if choice.target_id is None or choice.target_id not in state.removable_card_ids:
            raise ValueError("Toke requires a target id from removable_card_ids.")
        next_removable = frozenset(
            card_id
            for card_id in state.removable_card_ids
            if card_id != choice.target_id
        )
        next_state = replace(state, removable_card_ids=next_removable)
        return CampfireResult(
            choice=choice,
            state=next_state,
            removed_card_id=choice.target_id,
            source=rules.source,
        )

    raise ValueError(f"Unsupported campfire action: {choice.action.value}")
