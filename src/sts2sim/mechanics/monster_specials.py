"""Pure classifiers for source-data monster special handling.

The combat engine can already replay basic source-data moves, powers, and
weighted move selectors. This module keeps the unsupported or risky surfaces
visible: dynamic spawns, phase/death scripts, context-sensitive branch
conditions, and boss/elite scripts that need explicit integration.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .monsters import (
    EncounterDefinition,
    MonsterDefinition,
    MonsterMove,
    MonsterPower,
    build_encounter_definitions,
    build_monster_definitions,
)
from .powers import normalize_power_id

CONDITIONAL_BRANCH_DEPENDENCY = "conditional_branch_dependency"
SUMMON_SPAWN_MOVE = "summon_spawn_move"
PHASE_CHANGE = "phase_change"
SPECIAL_INNATE_POWER = "special_innate_power"
ADVANCED_AI = "advanced_ai"
BOSS_SCRIPT = "boss_script"
ELITE_SPECIAL_MECHANIC = "elite_special_mechanic"

MONSTER_SPECIAL_HINT = "hint"
MONSTER_SPECIAL_BLOCKER = "blocker"

_SUPPORTED_SLOT_NAMES = frozenset(("first", "second", "third", "fourth", "fifth", "sixth"))
_BASIC_STATUS_POWERS = frozenset(
    (
        "strength",
        "temporary_strength",
        "weak",
        "vulnerable",
        "frail",
        "intangible",
    )
)
_SUMMON_MOVE_TERMS = frozenset(
    (
        "summon",
        "fabricate",
        "hatch",
        "lay_eggs",
        "call_for_backup",
        "illusion",
        "dramatic_open",
        "bloat",
    )
)
_SPAWN_MARKER_TERMS = frozenset(("spawned",))
_PHASE_MOVE_TERMS = frozenset(
    (
        "phase",
        "respawn",
        "revive",
        "reattach",
        "dead",
        "about_to_blow",
        "explode",
    )
)
_ESCAPE_MOVE_TERMS = frozenset(("escape", "flee"))


@dataclass(frozen=True, slots=True)
class MonsterSpecialRequirement:
    """One special AI or mechanics surface found in a monster source definition."""

    category: str
    code: str
    severity: str
    monster_id: str
    source_type: str
    source_id: str
    detail: str
    deterministic_hint: str | None = None
    blocker: str | None = None
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MonsterSpecialClassification:
    """Special handling report for a single monster source definition."""

    monster_id: str
    name: str
    kind: str
    requirements: tuple[MonsterSpecialRequirement, ...] = ()

    @property
    def hints(self) -> tuple[MonsterSpecialRequirement, ...]:
        return tuple(
            requirement
            for requirement in self.requirements
            if requirement.severity == MONSTER_SPECIAL_HINT
        )

    @property
    def blockers(self) -> tuple[MonsterSpecialRequirement, ...]:
        return tuple(
            requirement
            for requirement in self.requirements
            if requirement.severity == MONSTER_SPECIAL_BLOCKER
        )

    @property
    def blocked(self) -> bool:
        return bool(self.blockers)

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({requirement.category for requirement in self.requirements}))

    @property
    def blocker_codes(self) -> tuple[str, ...]:
        return tuple(sorted({requirement.code for requirement in self.blockers}))


@dataclass(frozen=True, slots=True)
class MonsterSpecialCoverageSummary:
    """Aggregate source coverage for monster special handling and encounters."""

    monster_count: int
    encounter_count: int
    monsters_by_kind: Mapping[str, int]
    encounters_by_room_type: Mapping[str, int]
    classified_monster_ids: tuple[str, ...]
    monster_ids_with_requirements: tuple[str, ...]
    blocked_monster_ids: tuple[str, ...]
    encounter_ids_with_blockers: tuple[str, ...]
    missing_monster_ids: tuple[str, ...]
    requirement_counts: Mapping[str, int]
    blocker_counts: Mapping[str, int]


def classify_monster_specials(
    definition: MonsterDefinition,
    *,
    raw_source: Mapping[str, Any] | None = None,
) -> MonsterSpecialClassification:
    """Classify special AI and mechanics requirements for one monster definition."""

    requirements: list[MonsterSpecialRequirement] = []
    requirements.extend(_kind_requirements(definition))
    requirements.extend(_innate_power_requirements(definition))
    requirements.extend(_move_requirements(definition))
    requirements.extend(_state_requirements(definition))
    if raw_source is not None:
        requirements.extend(_raw_attack_pattern_requirements(definition, raw_source))

    return MonsterSpecialClassification(
        monster_id=definition.monster_id,
        name=definition.name,
        kind=definition.kind,
        requirements=_dedupe_requirements(requirements),
    )


def classify_raw_monster_specials(raw_monster: Mapping[str, Any]) -> MonsterSpecialClassification:
    """Build and classify a monster directly from one raw source-data row."""

    definitions = build_monster_definitions((raw_monster,))
    monster_id = _text(raw_monster.get("id"))
    definition = definitions.get(monster_id)
    if definition is None:
        definition = MonsterDefinition(
            monster_id=monster_id,
            name=_text(raw_monster.get("name"), monster_id),
            kind=_text(raw_monster.get("type"), "Normal"),
            min_hp=1,
            max_hp=1,
            min_hp_ascension=None,
            max_hp_ascension=None,
            moves=(),
            states=(),
        )
    return classify_monster_specials(definition, raw_source=raw_monster)


def classify_all_monster_specials(
    monster_definitions: Mapping[str, MonsterDefinition],
    *,
    raw_sources: Mapping[str, Mapping[str, Any]] | None = None,
) -> Mapping[str, MonsterSpecialClassification]:
    """Classify all monster definitions by monster id."""

    classifications = {
        monster_id: classify_monster_specials(
            definition,
            raw_source=(raw_sources or {}).get(monster_id),
        )
        for monster_id, definition in monster_definitions.items()
    }
    return MappingProxyType(classifications)


def monster_special_coverage_summary(
    monster_definitions: Mapping[str, MonsterDefinition],
    encounters: Sequence[EncounterDefinition],
    *,
    classifications: Mapping[str, MonsterSpecialClassification] | None = None,
) -> MonsterSpecialCoverageSummary:
    """Summarize monster source coverage and blocked encounter exposure."""

    classifications = classifications or classify_all_monster_specials(monster_definitions)
    missing_monster_ids = sorted(
        {
            monster_id
            for encounter in encounters
            for monster_id in encounter.monster_ids
            if monster_id not in monster_definitions
        }
    )
    blocked_monster_ids = tuple(
        sorted(
            monster_id
            for monster_id, classification in classifications.items()
            if classification.blocked
        )
    )
    blocked_monster_id_set = set(blocked_monster_ids)
    encounter_ids_with_blockers = tuple(
        sorted(
            encounter.encounter_id
            for encounter in encounters
            if any(monster_id in blocked_monster_id_set for monster_id in encounter.monster_ids)
        )
    )
    monster_ids_with_requirements = tuple(
        sorted(
            monster_id
            for monster_id, classification in classifications.items()
            if classification.requirements
        )
    )

    requirement_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    for classification in classifications.values():
        for requirement in classification.requirements:
            requirement_counts[requirement.category] += 1
            if requirement.severity == MONSTER_SPECIAL_BLOCKER:
                blocker_counts[requirement.category] += 1

    return MonsterSpecialCoverageSummary(
        monster_count=len(monster_definitions),
        encounter_count=len(encounters),
        monsters_by_kind=MappingProxyType(
            dict(
                sorted(
                    Counter(
                        definition.kind for definition in monster_definitions.values()
                    ).items()
                )
            )
        ),
        encounters_by_room_type=MappingProxyType(
            dict(sorted(Counter(encounter.room_type for encounter in encounters).items()))
        ),
        classified_monster_ids=tuple(sorted(monster_definitions)),
        monster_ids_with_requirements=monster_ids_with_requirements,
        blocked_monster_ids=blocked_monster_ids,
        encounter_ids_with_blockers=encounter_ids_with_blockers,
        missing_monster_ids=tuple(missing_monster_ids),
        requirement_counts=MappingProxyType(dict(sorted(requirement_counts.items()))),
        blocker_counts=MappingProxyType(dict(sorted(blocker_counts.items()))),
    )


def monster_special_source_coverage(
    raw_monsters: Sequence[Any],
    raw_encounters: Sequence[Any],
) -> MonsterSpecialCoverageSummary:
    """Build source definitions and summarize monster special coverage."""

    monster_definitions = build_monster_definitions(raw_monsters)
    encounters = build_encounter_definitions(raw_encounters)
    raw_sources = {
        monster_id: row
        for row in (_mapping(raw_monster) for raw_monster in raw_monsters)
        if (monster_id := _text(row.get("id")))
    }
    classifications = classify_all_monster_specials(
        monster_definitions,
        raw_sources=raw_sources,
    )
    return monster_special_coverage_summary(
        monster_definitions,
        encounters,
        classifications=classifications,
    )


def _kind_requirements(definition: MonsterDefinition) -> tuple[MonsterSpecialRequirement, ...]:
    kind = definition.kind.strip().lower()
    if kind == "boss":
        return (
            _requirement(
                definition,
                category=BOSS_SCRIPT,
                code="boss_script_requires_explicit_integration",
                severity=MONSTER_SPECIAL_BLOCKER,
                source_type="monster",
                source_id=definition.monster_id,
                detail=f"{definition.name} is a boss source definition.",
                blocker="Boss encounters need explicit script approval before source-data combat "
                "is considered faithful.",
                deterministic_hint="Basic move cycles may be replayable after boss-specific hooks "
                "are modeled.",
            ),
        )
    if kind == "elite":
        return (
            _requirement(
                definition,
                category=ELITE_SPECIAL_MECHANIC,
                code="elite_requires_explicit_integration",
                severity=MONSTER_SPECIAL_BLOCKER,
                source_type="monster",
                source_id=definition.monster_id,
                detail=f"{definition.name} is an elite source definition.",
                blocker="Elite encounters need explicit mechanics approval before source-data "
                "combat is considered faithful.",
                deterministic_hint="Basic move cycles may be replayable after elite-specific hooks "
                "are modeled.",
            ),
        )
    return ()


def _innate_power_requirements(
    definition: MonsterDefinition,
) -> tuple[MonsterSpecialRequirement, ...]:
    requirements: list[MonsterSpecialRequirement] = []
    for power in definition.innate_powers:
        if not power.power_id:
            continue
        normalized = normalize_power_id(power.power_id)
        if normalized in _BASIC_STATUS_POWERS:
            requirements.append(
                _requirement(
                    definition,
                    category=SPECIAL_INNATE_POWER,
                    code="innate_status_can_be_applied",
                    severity=MONSTER_SPECIAL_HINT,
                    source_type="power",
                    source_id=power.power_id,
                    detail=f"Innate power {power.power_id} is representable as a status.",
                    deterministic_hint="Apply the innate power during monster spawn using the "
                    "ascension-scaled amount.",
                    evidence=_power_evidence(power),
                )
            )
            continue
        requirements.append(
            _requirement(
                definition,
                category=SPECIAL_INNATE_POWER,
                code="special_innate_power_requires_hook",
                severity=MONSTER_SPECIAL_BLOCKER,
                source_type="power",
                source_id=power.power_id,
                detail=f"Innate power {power.power_id} has behavior beyond basic status math.",
                blocker="The power can be stored as a status, but its named combat behavior "
                "requires an explicit hook.",
                deterministic_hint="Keep the source power id and ascension amount in monster "
                "metadata until the hook is implemented.",
                evidence=_power_evidence(power),
            )
        )
    return tuple(requirements)


def _move_requirements(definition: MonsterDefinition) -> tuple[MonsterSpecialRequirement, ...]:
    requirements: list[MonsterSpecialRequirement] = []
    for move in definition.moves:
        normalized_text = _move_text(move)
        if _is_summon_move(move, normalized_text):
            requirements.append(
                _requirement(
                    definition,
                    category=SUMMON_SPAWN_MOVE,
                    code="summon_move_requires_spawn_resolution",
                    severity=MONSTER_SPECIAL_BLOCKER,
                    source_type="move",
                    source_id=move.move_id,
                    detail=f"{move.name} can add or create monsters.",
                    blocker="Dynamic monster creation is not handled by basic move execution.",
                    deterministic_hint="Resolve the summoned monster ids and slots before enabling "
                    "this move in combat.",
                    evidence=(move.intent, move.name),
                )
            )
        elif _is_spawn_marker(move, normalized_text):
            requirements.append(
                _requirement(
                    definition,
                    category=SUMMON_SPAWN_MOVE,
                    code="spawn_marker_move",
                    severity=MONSTER_SPECIAL_HINT,
                    source_type="move",
                    source_id=move.move_id,
                    detail=f"{move.name} marks a monster that has already entered combat.",
                    deterministic_hint="The marker can be replayed as a no-op/stun once another "
                    "source has created the monster.",
                    evidence=(move.intent, move.name),
                )
            )

        if _has_any_term(normalized_text, _PHASE_MOVE_TERMS):
            requirements.append(
                _requirement(
                    definition,
                    category=PHASE_CHANGE,
                    code="phase_or_death_script_move",
                    severity=MONSTER_SPECIAL_BLOCKER,
                    source_type="move",
                    source_id=move.move_id,
                    detail=(
                        f"{move.name} appears to participate in a phase, revive, or "
                        "death script."
                    ),
                    blocker="Phase/death transitions need explicit runtime triggers and state.",
                    deterministic_hint="Track the script variable that selects this move before "
                    "allowing automatic move resolution.",
                    evidence=(move.intent, move.name),
                )
            )

        if _has_any_term(normalized_text, _ESCAPE_MOVE_TERMS):
            requirements.append(
                _requirement(
                    definition,
                    category=ADVANCED_AI,
                    code="escape_move_requires_combat_removal",
                    severity=MONSTER_SPECIAL_BLOCKER,
                    source_type="move",
                    source_id=move.move_id,
                    detail=f"{move.name} can remove the monster from combat.",
                    blocker="Escape/flee semantics require monster removal and reward handling.",
                    deterministic_hint="Represent escape as an explicit combat event before "
                    "advancing the move script.",
                    evidence=(move.intent, move.name),
                )
            )

        if _normalized_id(move.intent) == "special":
            requirements.append(
                _requirement(
                    definition,
                    category=ADVANCED_AI,
                    code="special_intent_requires_handler",
                    severity=MONSTER_SPECIAL_BLOCKER,
                    source_type="move",
                    source_id=move.move_id,
                    detail=f"{move.name} has a Special intent.",
                    blocker="Special intents need a move-specific effect handler.",
                    evidence=(move.intent, move.name),
                )
            )
    return tuple(requirements)


def _state_requirements(definition: MonsterDefinition) -> tuple[MonsterSpecialRequirement, ...]:
    requirements: list[MonsterSpecialRequirement] = []
    seen_random_hints: set[str] = set()
    for state in definition.states:
        normalized_state_type = _normalized_id(state.state_type)
        if normalized_state_type == "random":
            if state.branches:
                if state.state_id not in seen_random_hints:
                    seen_random_hints.add(state.state_id)
                    requirements.append(
                        _requirement(
                            definition,
                            category=ADVANCED_AI,
                            code="weighted_random_selector",
                            severity=MONSTER_SPECIAL_HINT,
                            source_type="state",
                            source_id=state.state_id,
                            detail=f"{state.state_id} chooses from weighted random branches.",
                            deterministic_hint="Use the combat RNG plus previous move and move "
                            "counts for deterministic replay.",
                        )
                    )
            else:
                requirements.append(
                    _requirement(
                        definition,
                        category=ADVANCED_AI,
                        code="empty_random_selector",
                        severity=MONSTER_SPECIAL_BLOCKER,
                        source_type="state",
                        source_id=state.state_id,
                        detail=f"{state.state_id} is a random selector with no source branches.",
                        blocker="The source pattern omits the branch list; fallback selection is "
                        "not source-faithful.",
                    )
                )
        elif normalized_state_type == "conditional" and not state.branches:
            requirements.append(
                _requirement(
                    definition,
                    category=CONDITIONAL_BRANCH_DEPENDENCY,
                    code="empty_conditional_selector",
                    severity=MONSTER_SPECIAL_BLOCKER,
                    source_type="state",
                    source_id=state.state_id,
                    detail=f"{state.state_id} is a conditional selector with no source branches.",
                    blocker="The condition source is missing, so the branch cannot be resolved "
                    "faithfully.",
                )
            )
        elif normalized_state_type not in {"move", "conditional"}:
            requirements.append(
                _requirement(
                    definition,
                    category=ADVANCED_AI,
                    code="unknown_state_type",
                    severity=MONSTER_SPECIAL_BLOCKER,
                    source_type="state",
                    source_id=state.state_id,
                    detail=f"{state.state_id} has unsupported state type {state.state_type}.",
                    blocker="Unknown selector state types require a source-specific resolver.",
                )
            )

        for branch in state.branches:
            if branch.condition:
                requirements.append(
                    _condition_requirement(
                        definition,
                        state_id=state.state_id,
                        condition=branch.condition,
                        move_id=branch.move_id,
                    )
                )
            if branch.repeat or branch.max_times is not None:
                requirements.append(
                    _repeat_requirement(
                        definition,
                        state_id=state.state_id,
                        move_id=branch.move_id,
                        repeat=branch.repeat,
                        max_times=branch.max_times,
                    )
                )
    return tuple(requirements)


def _raw_attack_pattern_requirements(
    definition: MonsterDefinition,
    raw_source: Mapping[str, Any],
) -> tuple[MonsterSpecialRequirement, ...]:
    pattern = _mapping(raw_source.get("attack_pattern"))
    requirements: list[MonsterSpecialRequirement] = []
    for raw_state in _sequence(pattern.get("states")):
        state = _mapping(raw_state)
        if state.get("must_perform_once") is True:
            state_id = _text(state.get("id"), "unknown_state")
            requirements.append(
                _requirement(
                    definition,
                    category=ADVANCED_AI,
                    code="must_perform_once_state",
                    severity=MONSTER_SPECIAL_BLOCKER,
                    source_type="state",
                    source_id=state_id,
                    detail=f"{state_id} must be performed once according to source data.",
                    blocker="One-shot forced states require per-state execution tracking or an "
                    "explicit script trigger.",
                    deterministic_hint="Preserve the raw must_perform_once flag when integrating "
                    "scripted phase/death moves.",
                    evidence=("must_perform_once",),
                )
            )
    return tuple(requirements)


def _condition_requirement(
    definition: MonsterDefinition,
    *,
    state_id: str,
    condition: str,
    move_id: str | None,
) -> MonsterSpecialRequirement:
    dependency = _condition_dependency(condition)
    source_id = f"{state_id}:{move_id or 'branch'}"
    if dependency == "slot_index":
        return _requirement(
            definition,
            category=CONDITIONAL_BRANCH_DEPENDENCY,
            code="slot_condition",
            severity=MONSTER_SPECIAL_HINT,
            source_type="branch",
            source_id=source_id,
            detail=f"{state_id} branches on monster slot position.",
            deterministic_hint="Resolve from encounter slot_index before choosing the move.",
            evidence=(condition,),
        )
    return _requirement(
        definition,
        category=CONDITIONAL_BRANCH_DEPENDENCY,
        code=f"{dependency}_condition",
        severity=MONSTER_SPECIAL_BLOCKER,
        source_type="branch",
        source_id=source_id,
        detail=f"{state_id} depends on runtime condition: {condition}",
        blocker="The branch needs combat state that the basic selector cannot infer safely.",
        deterministic_hint=_condition_hint(dependency),
        evidence=(condition,),
    )


def _repeat_requirement(
    definition: MonsterDefinition,
    *,
    state_id: str,
    move_id: str | None,
    repeat: str | None,
    max_times: int | None,
) -> MonsterSpecialRequirement:
    source_id = f"{state_id}:{move_id or 'branch'}"
    repeat_id = _normalized_id(repeat or "")
    if repeat_id == "useonlyonce":
        return _requirement(
            definition,
            category=ADVANCED_AI,
            code="use_only_once_repeat",
            severity=MONSTER_SPECIAL_BLOCKER,
            source_type="branch",
            source_id=source_id,
            detail=f"{state_id} has a UseOnlyOnce branch constraint.",
            blocker="UseOnlyOnce requires move-count exclusion before the branch is enabled.",
            deterministic_hint="Use move_counts to suppress this branch after it has "
            "resolved once.",
            evidence=tuple(item for item in (repeat, str(max_times) if max_times else "") if item),
        )
    return _requirement(
        definition,
        category=ADVANCED_AI,
        code="repeat_constraint",
        severity=MONSTER_SPECIAL_HINT,
        source_type="branch",
        source_id=source_id,
        detail=f"{state_id} has repeat constraint {repeat or max_times}.",
        deterministic_hint="Use previous_move_id and move_counts when resolving this branch.",
        evidence=tuple(item for item in (repeat, str(max_times) if max_times else "") if item),
    )


def _condition_dependency(condition: str) -> str:
    normalized = _normalized_id(condition)
    if "slotname" in normalized:
        slot_names = set(_quoted_values(condition))
        if slot_names and slot_names <= _SUPPORTED_SLOT_NAMES:
            return "slot_index"
        return "named_slot"
    if "currenthp" in normalized or "maxhp" in normalized:
        return "hp_threshold"
    if "allycount" in normalized or "isalone" in normalized or "isfront" in normalized:
        return "formation"
    if "respawns" in normalized:
        return "respawn_counter"
    if "counter" in normalized:
        return "script_counter"
    if "canlay" in normalized or "canfabricate" in normalized:
        return "spawn_capacity"
    if "hasamalgamdied" in normalized:
        return "ally_death"
    if normalized.startswith("!"):
        return "negated_runtime"
    return "runtime"


def _condition_hint(dependency: str) -> str:
    return {
        "ally_death": "Track the referenced ally death flag in combat metadata.",
        "formation": "Track live ally count/front-or-alone formation before selecting moves.",
        "hp_threshold": "Evaluate the branch against current monster hp and max hp.",
        "named_slot": "Map encounter-specific slot names to spawned monster metadata.",
        "negated_runtime": "Evaluate the negated runtime predicate before branch selection.",
        "respawn_counter": "Track the monster respawn counter in combat metadata.",
        "script_counter": "Track the source script counter in combat metadata.",
        "spawn_capacity": "Track available summon slots before branch selection.",
    }.get(dependency, "Add a source-specific predicate before branch selection.")


def _is_summon_move(move: MonsterMove, normalized_text: str) -> bool:
    return "summon" in _normalized_id(move.intent) or _has_any_term(
        normalized_text,
        _SUMMON_MOVE_TERMS,
    )


def _is_spawn_marker(move: MonsterMove, normalized_text: str) -> bool:
    return (
        _has_any_term(normalized_text, _SPAWN_MARKER_TERMS)
        and "summon" not in _normalized_id(move.intent)
    )


def _move_text(move: MonsterMove) -> str:
    return _normalized_id(f"{move.move_id} {move.name} {move.intent}")


def _has_any_term(text: str, terms: frozenset[str]) -> bool:
    return any(term in text for term in terms)


def _power_evidence(power: MonsterPower) -> tuple[str, ...]:
    evidence = [f"amount={power.amount}", f"target={power.target}"]
    if power.amount_ascension is not None:
        evidence.append(f"amount_ascension={power.amount_ascension}")
    return tuple(evidence)


def _requirement(
    definition: MonsterDefinition,
    *,
    category: str,
    code: str,
    severity: str,
    source_type: str,
    source_id: str,
    detail: str,
    deterministic_hint: str | None = None,
    blocker: str | None = None,
    evidence: tuple[str, ...] = (),
) -> MonsterSpecialRequirement:
    return MonsterSpecialRequirement(
        category=category,
        code=code,
        severity=severity,
        monster_id=definition.monster_id,
        source_type=source_type,
        source_id=source_id,
        detail=detail,
        deterministic_hint=deterministic_hint,
        blocker=blocker,
        evidence=tuple(item for item in evidence if item),
    )


def _dedupe_requirements(
    requirements: Sequence[MonsterSpecialRequirement],
) -> tuple[MonsterSpecialRequirement, ...]:
    seen: set[tuple[str, str, str, str, str]] = set()
    deduped: list[MonsterSpecialRequirement] = []
    for requirement in requirements:
        key = (
            requirement.category,
            requirement.code,
            requirement.source_type,
            requirement.source_id,
            requirement.detail,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(requirement)
    return tuple(deduped)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _normalized_id(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("'", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _quoted_values(value: str) -> tuple[str, ...]:
    values: list[str] = []
    current_quote = ""
    current: list[str] = []
    for char in value:
        if current_quote:
            if char == current_quote:
                values.append("".join(current))
                current = []
                current_quote = ""
            else:
                current.append(char)
        elif char in {"'", '"'}:
            current_quote = char
    return tuple(_normalized_id(item) for item in values)


__all__ = [
    "ADVANCED_AI",
    "BOSS_SCRIPT",
    "CONDITIONAL_BRANCH_DEPENDENCY",
    "ELITE_SPECIAL_MECHANIC",
    "MONSTER_SPECIAL_BLOCKER",
    "MONSTER_SPECIAL_HINT",
    "PHASE_CHANGE",
    "SPECIAL_INNATE_POWER",
    "SUMMON_SPAWN_MOVE",
    "MonsterSpecialClassification",
    "MonsterSpecialCoverageSummary",
    "MonsterSpecialRequirement",
    "classify_all_monster_specials",
    "classify_monster_specials",
    "classify_raw_monster_specials",
    "monster_special_coverage_summary",
    "monster_special_source_coverage",
]
