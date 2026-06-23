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
    monster_summon_plan,
    monster_summon_plans_for,
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
_SUPPORTED_SPECIAL_INNATE_POWERS = frozenset(
    ("plating", "curl_up", "skittish", "ravenous", "shriek", "enrage", "slippery")
)
_VERIFIED_ELITE_SOURCE_INTEGRATIONS = frozenset(
    (
        "bygone_effigy",
        "byrdonis",
        "decimillipede_segment",
        "decimillipede_segment_back",
        "decimillipede_segment_front",
        "decimillipede_segment_middle",
        "entomancer",
        "flail_knight",
        "infested_prism",
        "magi_knight",
        "mecha_knight",
        "phantasmal_gardener",
        "phrog_parasite",
        "skulking_colony",
        "soul_nexus",
        "spectral_knight",
        "terror_eel",
        "wriggler",
    )
)
_VERIFIED_BOSS_SOURCE_INTEGRATIONS = frozenset(
    (
        "ceremonial_beast",
        "crusher",
        "kin_follower",
        "kin_priest",
        "knowledge_demon",
        "lagavulin_matriarch",
        "queen",
        "rocket",
        "soul_fysh",
        "test_subject",
        "the_insatiable",
        "torch_head_amalgam",
        "vantom",
        "waterfall_giant",
    )
)
_VERIFIED_MUST_PERFORM_ONCE_STATE_INTEGRATIONS = MappingProxyType(
    {
        "ceremonial_beast": frozenset(("stun_move",)),
        "decimillipede_segment": frozenset(("reattach_move",)),
        "decimillipede_segment_back": frozenset(("reattach_move",)),
        "decimillipede_segment_front": frozenset(("reattach_move",)),
        "decimillipede_segment_middle": frozenset(("reattach_move",)),
        "test_subject": frozenset(("respawn_move",)),
        "waterfall_giant": frozenset(("about_to_blow_move",)),
    }
)
_SUMMON_MOVE_TERMS = frozenset(
    (
        "summon",
        "fabricate",
        "hatch",
        "lay_eggs",
        "call_for_backup",
        "illusion",
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
    normalized_monster_id = _normalized_id(definition.monster_id)
    if kind == "boss":
        if normalized_monster_id in _VERIFIED_BOSS_SOURCE_INTEGRATIONS:
            return (
                _requirement(
                    definition,
                    category=BOSS_SCRIPT,
                    code="boss_explicit_integration_supported",
                    severity=MONSTER_SPECIAL_HINT,
                    source_type="monster",
                    source_id=definition.monster_id,
                    detail=f"{definition.name} has an explicit source-data boss integration.",
                    deterministic_hint="Replay the source attack pattern with supported boss "
                    "powers and move effects.",
                ),
            )
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
        if normalized_monster_id in _VERIFIED_ELITE_SOURCE_INTEGRATIONS:
            return (
                _requirement(
                    definition,
                    category=ELITE_SPECIAL_MECHANIC,
                    code="elite_explicit_integration_supported",
                    severity=MONSTER_SPECIAL_HINT,
                    source_type="monster",
                    source_id=definition.monster_id,
                    detail=f"{definition.name} has an explicit source-data elite integration.",
                    deterministic_hint="Replay the source attack pattern with supported elite "
                    "powers, conditions, and move effects.",
                ),
            )
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
        if normalized in _SUPPORTED_SPECIAL_INNATE_POWERS:
            requirements.append(
                _requirement(
                    definition,
                    category=SPECIAL_INNATE_POWER,
                    code="special_innate_power_supported",
                    severity=MONSTER_SPECIAL_HINT,
                    source_type="power",
                    source_id=power.power_id,
                    detail=f"Innate power {power.power_id} has an explicit runtime hook.",
                    deterministic_hint="Apply the innate power status at spawn and resolve its "
                    "combat trigger in the monster runtime.",
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
        if _is_self_hatch_move(definition, move):
            requirements.append(
                _requirement(
                    definition,
                    category=PHASE_CHANGE,
                    code="self_hatch_move",
                    severity=MONSTER_SPECIAL_HINT,
                    source_type="move",
                    source_id=move.move_id,
                    detail=f"{move.name} transitions this monster into its attack cycle.",
                    deterministic_hint="Emit a hatch event and advance to the next source move.",
                    evidence=(move.intent, move.name),
                )
            )
        elif _is_summon_move(move, normalized_text):
            plan = monster_summon_plan(definition.monster_id, move.move_id)
            requirements.append(
                _requirement(
                    definition,
                    category=SUMMON_SPAWN_MOVE,
                    code="summon_move_requires_spawn_resolution",
                    severity=(
                        MONSTER_SPECIAL_HINT
                        if plan is not None
                        else MONSTER_SPECIAL_BLOCKER
                    ),
                    source_type="move",
                    source_id=move.move_id,
                    detail=f"{move.name} can add or create monsters.",
                    blocker=(
                        None
                        if plan is not None
                        else "Dynamic monster creation needs a source-specific summon plan."
                    ),
                    deterministic_hint=(
                        f"Summon {', '.join(plan.summon_monster_ids)} using the "
                        f"{plan.count_policy} count policy."
                        if plan is not None
                        else "Resolve the summoned monster ids and slots before enabling "
                        "this move in combat."
                    ),
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

        if _is_aeonglass_increasing_intensity(definition, move):
            requirements.append(
                _requirement(
                    definition,
                    category=ADVANCED_AI,
                    code="aeonglass_increasing_intensity_requires_wither_hook",
                    severity=MONSTER_SPECIAL_BLOCKER,
                    source_type="move",
                    source_id=move.move_id,
                    detail="Increasing Intensity has no concrete source payload for its "
                    "Wither/Withering Presence behavior.",
                    blocker="Aeonglass Wither card creation and upgrade behavior needs an "
                    "explicit boss hook before combat replay is faithful.",
                    deterministic_hint="Keep the source move as a blocker until Wither status "
                    "card generation and scaling are modeled.",
                    evidence=(move.intent, move.name),
                )
            )

        requirements.extend(_move_power_requirements(definition, move))

        if _has_any_term(normalized_text, _PHASE_MOVE_TERMS):
            if _is_self_destruct_move(definition, move):
                requirements.append(
                    _requirement(
                        definition,
                        category=PHASE_CHANGE,
                        code="self_destruct_move",
                        severity=MONSTER_SPECIAL_HINT,
                        source_type="move",
                        source_id=move.move_id,
                        detail=f"{move.name} removes this monster after resolving.",
                        deterministic_hint="Resolve damage, then set the monster to inactive "
                        "with a self-destruct event.",
                        evidence=(move.intent, move.name),
                    )
                )
                continue
            if _is_decimillipede_segment(definition.monster_id) and _normalized_id(
                move.move_id
            ) in {"dead", "reattach"}:
                requirements.append(
                    _requirement(
                        definition,
                        category=PHASE_CHANGE,
                        code="decimillipede_reattach_move",
                        severity=MONSTER_SPECIAL_HINT,
                        source_type="move",
                        source_id=move.move_id,
                        detail=f"{move.name} participates in the Decimillipede revive script.",
                        deterministic_hint="Defeated segments count down while another segment "
                        "is alive, then revive with 25 HP and cleared statuses.",
                        evidence=(move.intent, move.name),
                    )
                )
                continue
            if _is_test_subject_respawn_move(definition, move):
                requirements.append(
                    _requirement(
                        definition,
                        category=PHASE_CHANGE,
                        code="test_subject_respawn_move",
                        severity=MONSTER_SPECIAL_HINT,
                        source_type="move",
                        source_id=move.move_id,
                        detail="Respawn transitions Test Subject into its next phase.",
                        deterministic_hint="Revive phase one at 200 HP and phase two at 300 HP, "
                        "clear statuses, then select the next source move from Respawns.",
                        evidence=(move.intent, move.name),
                    )
                )
                continue
            if _is_test_subject_phase_attack_move(definition, move):
                requirements.append(
                    _requirement(
                        definition,
                        category=PHASE_CHANGE,
                        code="test_subject_phase_attack_move",
                        severity=MONSTER_SPECIAL_HINT,
                        source_type="move",
                        source_id=move.move_id,
                        detail=f"{move.name} is part of the handled Test Subject phase cycle.",
                        deterministic_hint="Use the Test Subject respawn counter to enter this "
                        "move, then continue through the source attack pattern.",
                        evidence=(move.intent, move.name),
                    )
                )
                continue
            if _is_waterfall_giant_death_move(definition, move):
                requirements.append(
                    _requirement(
                        definition,
                        category=PHASE_CHANGE,
                        code="waterfall_giant_death_countdown_move",
                        severity=MONSTER_SPECIAL_HINT,
                        source_type="move",
                        source_id=move.move_id,
                        detail=f"{move.name} is part of the handled Steam Eruption death script.",
                        deterministic_hint="On death with Steam Eruption, enter About To Blow, "
                        "then Explode for the stored steam amount.",
                        evidence=(move.intent, move.name),
                    )
                )
                continue
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
                    severity=MONSTER_SPECIAL_HINT,
                    source_type="move",
                    source_id=move.move_id,
                    detail=f"{move.name} can remove the monster from combat.",
                    deterministic_hint="Represent escape as an explicit combat event before "
                    "advancing the move script.",
                    evidence=(move.intent, move.name),
                )
            )

        if _normalized_id(move.intent) == "special":
            if _is_waterfall_giant_death_move(definition, move):
                requirements.append(
                    _requirement(
                        definition,
                        category=ADVANCED_AI,
                        code="waterfall_giant_explode_special",
                        severity=MONSTER_SPECIAL_HINT,
                        source_type="move",
                        source_id=move.move_id,
                        detail="Explode is handled by the Waterfall Giant Steam Eruption hook.",
                        deterministic_hint="Deal damage equal to Steam Eruption, then end combat "
                        "if the player survives.",
                        evidence=(move.intent, move.name),
                    )
                )
                continue
            if _is_self_destruct_move(definition, move):
                requirements.append(
                    _requirement(
                        definition,
                        category=ADVANCED_AI,
                        code="special_intent_self_destruct",
                        severity=MONSTER_SPECIAL_HINT,
                        source_type="move",
                        source_id=move.move_id,
                        detail=f"{move.name} has a handled Special self-destruct intent.",
                        deterministic_hint="Use the Gas Bomb self-destruct runtime handler.",
                        evidence=(move.intent, move.name),
                    )
                )
                continue
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


def _move_power_requirements(
    definition: MonsterDefinition,
    move: MonsterMove,
) -> tuple[MonsterSpecialRequirement, ...]:
    requirements: list[MonsterSpecialRequirement] = []
    for power in move.powers:
        normalized = normalize_power_id(power.power_id)
        if normalized != "vital_spark":
            continue
        requirements.append(
            _requirement(
                definition,
                category=SPECIAL_INNATE_POWER,
                code="vital_spark_requires_tainted_skill_hook",
                severity=MONSTER_SPECIAL_BLOCKER,
                source_type="power",
                source_id=power.power_id,
                detail="Vital Spark changes Skill cards into Tainted sources.",
                blocker="Infested Prism's Tainted Skill and per-hit damage race behavior needs "
                "an explicit combat hook.",
                deterministic_hint="Track Tainted application from Skill plays and scale the "
                "player debuff before replaying this move as supported.",
                evidence=(move.move_id, f"amount={power.amount}"),
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
                severity = (
                    MONSTER_SPECIAL_HINT
                    if definition.moves
                    else MONSTER_SPECIAL_BLOCKER
                )
                requirements.append(
                    _requirement(
                        definition,
                        category=ADVANCED_AI,
                        code="empty_random_selector",
                        severity=severity,
                        source_type="state",
                        source_id=state.state_id,
                        detail=f"{state.state_id} is a random selector with no source branches.",
                        blocker=(
                            None
                            if definition.moves
                            else "The source pattern omits both the branch list and a move pool."
                        ),
                        deterministic_hint=(
                            "Use the engine fallback: choose deterministically from source moves, "
                            "excluding the previous move when possible."
                            if definition.moves
                            else None
                        ),
                    )
                )
        elif normalized_state_type == "conditional" and not state.branches:
            if not _state_selector_referenced(definition, state.state_id):
                requirements.append(
                    _requirement(
                        definition,
                        category=CONDITIONAL_BRANCH_DEPENDENCY,
                        code="empty_conditional_selector",
                        severity=MONSTER_SPECIAL_HINT,
                        source_type="state",
                        source_id=state.state_id,
                        detail=(
                            f"{state.state_id} is an unreachable conditional selector "
                            "with no source branches."
                        ),
                        deterministic_hint="Ignore unreachable empty selector states during "
                        "runtime move selection.",
                    )
                )
                continue
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


def _state_selector_referenced(definition: MonsterDefinition, state_id: str) -> bool:
    if definition.initial_selector == state_id:
        return True
    return any(state.next_selector == state_id for state in definition.states)


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
            verified_states = _VERIFIED_MUST_PERFORM_ONCE_STATE_INTEGRATIONS.get(
                _normalized_id(definition.monster_id),
                frozenset(),
            )
            supported = _normalized_id(state_id) in verified_states
            requirements.append(
                _requirement(
                    definition,
                    category=ADVANCED_AI,
                    code="must_perform_once_state",
                    severity=MONSTER_SPECIAL_HINT if supported else MONSTER_SPECIAL_BLOCKER,
                    source_type="state",
                    source_id=state_id,
                    detail=f"{state_id} must be performed once according to source data.",
                    blocker=None
                    if supported
                    else "One-shot forced states require per-state execution tracking or an "
                    "explicit script trigger.",
                    deterministic_hint="Preserve the raw must_perform_once flag when integrating "
                    "scripted phase/death moves."
                    if not supported
                    else "This one-shot state is covered by an explicit runtime script.",
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
    if dependency == "formation" and _supported_formation_condition(condition):
        return _requirement(
            definition,
            category=CONDITIONAL_BRANCH_DEPENDENCY,
            code="formation_condition",
            severity=MONSTER_SPECIAL_HINT,
            source_type="branch",
            source_id=source_id,
            detail=f"{state_id} branches on live monster formation.",
            deterministic_hint="Resolve from alive ally count and front position before "
            "choosing the move.",
            evidence=(condition,),
        )
    if dependency == "spawn_capacity" and monster_summon_plans_for(definition.monster_id):
        return _requirement(
            definition,
            category=CONDITIONAL_BRANCH_DEPENDENCY,
            code="spawn_capacity_condition",
            severity=MONSTER_SPECIAL_HINT,
            source_type="branch",
            source_id=source_id,
            detail=f"{state_id} branches on available summon capacity.",
            deterministic_hint="Resolve CanLay/CanFabricate from alive ally count and open "
            "monster slots.",
            evidence=(condition,),
        )
    if dependency == "hp_threshold" and _supported_hp_threshold_condition(condition):
        return _requirement(
            definition,
            category=CONDITIONAL_BRANCH_DEPENDENCY,
            code="hp_threshold_condition",
            severity=MONSTER_SPECIAL_HINT,
            source_type="branch",
            source_id=source_id,
            detail=f"{state_id} branches on monster hp threshold.",
            deterministic_hint="Resolve from current monster hp, max hp, and any tracked "
            "threshold move counts before choosing the move.",
            evidence=(condition,),
        )
    if dependency == "script_counter" and _supported_script_counter_condition(
        definition,
        condition,
    ):
        return _requirement(
            definition,
            category=CONDITIONAL_BRANCH_DEPENDENCY,
            code="script_counter_condition",
            severity=MONSTER_SPECIAL_HINT,
            source_type="branch",
            source_id=source_id,
            detail=f"{state_id} branches on an integrated script counter.",
            deterministic_hint="Resolve from tracked source move counts before choosing the move.",
            evidence=(condition,),
        )
    if dependency == "ally_death" and _supported_ally_death_condition(
        definition,
        condition,
    ):
        return _requirement(
            definition,
            category=CONDITIONAL_BRANCH_DEPENDENCY,
            code="ally_death_condition",
            severity=MONSTER_SPECIAL_HINT,
            source_type="branch",
            source_id=source_id,
            detail=f"{state_id} branches on tracked ally death state.",
            deterministic_hint="Resolve HasAmalgamDied from the Torch Head Amalgam combat state.",
            evidence=(condition,),
        )
    if dependency == "respawn_counter" and _supported_respawn_counter_condition(
        definition,
        condition,
    ):
        return _requirement(
            definition,
            category=CONDITIONAL_BRANCH_DEPENDENCY,
            code="respawn_counter_condition",
            severity=MONSTER_SPECIAL_HINT,
            source_type="branch",
            source_id=source_id,
            detail=f"{state_id} branches on Test Subject respawn count.",
            deterministic_hint="Resolve Respawns from Test Subject combat metadata before "
            "choosing the move.",
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
            severity=MONSTER_SPECIAL_HINT,
            source_type="branch",
            source_id=source_id,
            detail=f"{state_id} has a UseOnlyOnce branch constraint.",
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
        if slot_names and (
            slot_names <= _SUPPORTED_SLOT_NAMES or _supported_named_slot_values(slot_names)
        ):
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


def _supported_formation_condition(condition: str) -> bool:
    normalized = "".join(
        char for char in _normalized_id(condition) if char.isalnum() or char in "<>=!"
    )
    return any(
        marker in normalized
        for marker in (
            "getallycount>0",
            "getallycount==0",
            "isalone",
            "isfront",
        )
    )


def _supported_hp_threshold_condition(condition: str) -> bool:
    normalized = "".join(
        char for char in _normalized_id(condition) if char.isalnum() or char in "<>=!|&/"
    )
    return any(
        marker in normalized
        for marker in (
            "currenthp>=basecreaturemaxhp/2",
            "currenthp<basecreaturemaxhp/2",
            "hasbeetlecharged||currenthp>=basecreaturemaxhp/2",
            "!hasbeetlecharged&&currenthp<basecreaturemaxhp/2",
        )
    )


def _supported_script_counter_condition(
    definition: MonsterDefinition,
    condition: str,
) -> bool:
    normalized = _normalized_id(condition)
    return (
        _normalized_id(definition.monster_id) == "knowledge_demon"
        and "curseofknowledgecounter" in normalized
    )


def _supported_ally_death_condition(
    definition: MonsterDefinition,
    condition: str,
) -> bool:
    return (
        _normalized_id(definition.monster_id) == "queen"
        and "hasamalgamdied" in _normalized_id(condition)
    )


def _supported_respawn_counter_condition(
    definition: MonsterDefinition,
    condition: str,
) -> bool:
    normalized = "".join(
        char for char in _normalized_id(condition) if char.isalnum() or char in "<>=!"
    )
    return (
        _normalized_id(definition.monster_id) == "test_subject"
        and "respawns" in normalized
        and ("<2" in normalized or ">=2" in normalized)
    )


def _supported_named_slot_values(slot_names: set[str]) -> bool:
    return all(
        name.startswith("wriggler") and name.removeprefix("wriggler").isdigit()
        for name in slot_names
    )


def _is_summon_move(move: MonsterMove, normalized_text: str) -> bool:
    return "summon" in _normalized_id(move.intent) or _has_any_term(
        normalized_text,
        _SUMMON_MOVE_TERMS,
    )


def _is_self_hatch_move(definition: MonsterDefinition, move: MonsterMove) -> bool:
    return (
        _normalized_id(definition.monster_id) == "tough_egg"
        and _normalized_id(move.move_id) == "hatch"
    )


def _is_self_destruct_move(definition: MonsterDefinition, move: MonsterMove) -> bool:
    return (
        _normalized_id(definition.monster_id) == "gas_bomb"
        and _normalized_id(move.move_id) == "explode"
    )


def _is_test_subject_respawn_move(definition: MonsterDefinition, move: MonsterMove) -> bool:
    return (
        _normalized_id(definition.monster_id) == "test_subject"
        and _normalized_id(move.move_id) == "respawn"
    )


def _is_test_subject_phase_attack_move(definition: MonsterDefinition, move: MonsterMove) -> bool:
    return (
        _normalized_id(definition.monster_id) == "test_subject"
        and _normalized_id(move.move_id) == "phase3_lacerate"
    )


def _is_waterfall_giant_death_move(definition: MonsterDefinition, move: MonsterMove) -> bool:
    return (
        _normalized_id(definition.monster_id) == "waterfall_giant"
        and _normalized_id(move.move_id) in {"about_to_blow", "explode"}
    )


def _is_decimillipede_segment(monster_id: str) -> bool:
    return _normalized_id(monster_id).startswith("decimillipede_segment")


def _is_aeonglass_increasing_intensity(
    definition: MonsterDefinition,
    move: MonsterMove,
) -> bool:
    return (
        _normalized_id(definition.monster_id) == "aeonglass"
        and _normalized_id(move.move_id) == "increasing_intensity"
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
