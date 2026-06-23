"""Audit cached combat content against executable combat helper coverage."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

CachedCombatRow = Mapping[str, Any]

DEFAULT_COMBAT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "eng"
DEFAULT_CARDS_PATH = DEFAULT_COMBAT_CACHE_DIR / "cards.json"
DEFAULT_RELICS_PATH = DEFAULT_COMBAT_CACHE_DIR / "relics.json"
DEFAULT_POTIONS_PATH = DEFAULT_COMBAT_CACHE_DIR / "potions.json"
DEFAULT_MONSTERS_PATH = DEFAULT_COMBAT_CACHE_DIR / "monsters.json"
DEFAULT_ENCOUNTERS_PATH = DEFAULT_COMBAT_CACHE_DIR / "encounters.json"


class CombatCoverageCategory(str, Enum):
    CARDS = "cards"
    RELICS = "relics"
    POTIONS = "potions"
    MONSTERS = "monsters"
    ENCOUNTERS = "encounters"


class CombatCoverageStatus(str, Enum):
    IMPLEMENTED = "implemented"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CombatSourceContent:
    cards: tuple[CachedCombatRow, ...] = ()
    relics: tuple[CachedCombatRow, ...] = ()
    potions: tuple[CachedCombatRow, ...] = ()
    monsters: tuple[CachedCombatRow, ...] = ()
    encounters: tuple[CachedCombatRow, ...] = ()


@dataclass(frozen=True, slots=True)
class CombatImplementationCatalog:
    implemented_ids_by_category: Mapping[str, frozenset[str]] = field(default_factory=dict)
    blocked_ids_by_category: Mapping[str, Mapping[str, tuple[str, ...]]] = field(
        default_factory=dict
    )
    executable_card_effect_keys: frozenset[str] = frozenset()

    def implemented_ids(
        self,
        category: CombatCoverageCategory | str,
    ) -> frozenset[str]:
        return self.implemented_ids_by_category.get(_category_value(category), frozenset())

    def blocker_reasons(
        self,
        category: CombatCoverageCategory | str,
        content_id: str,
    ) -> tuple[str, ...]:
        blockers = self.blocked_ids_by_category.get(_category_value(category), {})
        return blockers.get(_normalized_id(content_id), ())


@dataclass(frozen=True, slots=True)
class CombatCoverageEntry:
    category: CombatCoverageCategory
    content_id: str
    normalized_id: str
    name: str
    status: CombatCoverageStatus
    implemented_keys: tuple[str, ...] = ()
    blocked_keys: tuple[str, ...] = ()
    unknown_keys: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def is_implemented(self) -> bool:
        return self.status is CombatCoverageStatus.IMPLEMENTED

    @property
    def is_blocked(self) -> bool:
        return self.status is CombatCoverageStatus.BLOCKED

    @property
    def is_unknown(self) -> bool:
        return self.status is CombatCoverageStatus.UNKNOWN

    def as_dict(self) -> dict[str, object]:
        return {
            "category": self.category.value,
            "content_id": self.content_id,
            "normalized_id": self.normalized_id,
            "name": self.name,
            "status": self.status.value,
            "implemented_keys": list(self.implemented_keys),
            "blocked_keys": list(self.blocked_keys),
            "unknown_keys": list(self.unknown_keys),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class CombatCoverageSummary:
    category: CombatCoverageCategory
    total: int
    implemented: int
    blocked: int
    unknown: int
    sample_blocked_ids: tuple[str, ...] = ()
    sample_unknown_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            CombatCoverageStatus.IMPLEMENTED.value: self.implemented,
            CombatCoverageStatus.BLOCKED.value: self.blocked,
            CombatCoverageStatus.UNKNOWN.value: self.unknown,
            "sample_blocked_ids": list(self.sample_blocked_ids),
            "sample_unknown_ids": list(self.sample_unknown_ids),
        }


@dataclass(frozen=True, slots=True)
class CombatCoverageReport:
    entries: tuple[CombatCoverageEntry, ...]
    unknown_sample_size: int = 5

    @property
    def total_ids(self) -> int:
        return len(self.entries)

    @property
    def summaries(self) -> tuple[CombatCoverageSummary, ...]:
        return tuple(self.summary_for(category) for category in CombatCoverageCategory)

    @property
    def counts_by_category(self) -> dict[str, dict[str, int]]:
        return {
            summary.category.value: {
                "total": summary.total,
                CombatCoverageStatus.IMPLEMENTED.value: summary.implemented,
                CombatCoverageStatus.BLOCKED.value: summary.blocked,
                CombatCoverageStatus.UNKNOWN.value: summary.unknown,
            }
            for summary in self.summaries
        }

    @property
    def sample_unknown_ids(self) -> dict[str, list[str]]:
        return {
            summary.category.value: list(summary.sample_unknown_ids)
            for summary in self.summaries
        }

    @property
    def sample_blocked_ids(self) -> dict[str, list[str]]:
        return {
            summary.category.value: list(summary.sample_blocked_ids)
            for summary in self.summaries
        }

    @property
    def unknown_entries(self) -> tuple[CombatCoverageEntry, ...]:
        return self.entries_for(status=CombatCoverageStatus.UNKNOWN)

    @property
    def blocked_entries(self) -> tuple[CombatCoverageEntry, ...]:
        return self.entries_for(status=CombatCoverageStatus.BLOCKED)

    def summary_for(
        self,
        category: CombatCoverageCategory | str,
    ) -> CombatCoverageSummary:
        enum_category = _category_enum(category)
        entries = self.entries_for(category=enum_category)
        unknown_entries = tuple(
            entry for entry in entries if entry.status is CombatCoverageStatus.UNKNOWN
        )
        blocked_entries = tuple(
            entry for entry in entries if entry.status is CombatCoverageStatus.BLOCKED
        )
        return CombatCoverageSummary(
            category=enum_category,
            total=len(entries),
            implemented=sum(
                1 for entry in entries if entry.status is CombatCoverageStatus.IMPLEMENTED
            ),
            blocked=sum(1 for entry in entries if entry.status is CombatCoverageStatus.BLOCKED),
            unknown=len(unknown_entries),
            sample_blocked_ids=tuple(
                entry.content_id for entry in blocked_entries[: self.unknown_sample_size]
            ),
            sample_unknown_ids=tuple(
                entry.content_id for entry in unknown_entries[: self.unknown_sample_size]
            ),
        )

    def entries_for(
        self,
        *,
        category: CombatCoverageCategory | str | None = None,
        status: CombatCoverageStatus | str | None = None,
    ) -> tuple[CombatCoverageEntry, ...]:
        enum_category = _category_enum(category) if category is not None else None
        enum_status = _status_enum(status) if status is not None else None
        return tuple(
            entry
            for entry in self.entries
            if (enum_category is None or entry.category is enum_category)
            and (enum_status is None or entry.status is enum_status)
        )

    def entry_for(
        self,
        category: CombatCoverageCategory | str,
        content_id: str,
    ) -> CombatCoverageEntry:
        enum_category = _category_enum(category)
        normalized = _normalized_id(content_id)
        for entry in self.entries:
            if entry.category is enum_category and entry.normalized_id == normalized:
                return entry
        raise KeyError(f"Unknown {enum_category.value} id in combat coverage: {content_id}")

    def as_dict(self) -> dict[str, object]:
        return {
            "total_ids": self.total_ids,
            "counts_by_category": self.counts_by_category,
            "sample_blocked_ids": self.sample_blocked_ids,
            "sample_unknown_ids": self.sample_unknown_ids,
            "entries": [entry.as_dict() for entry in self.entries],
        }


def combat_implementation_catalog(
    *,
    implemented_ids_by_category: Mapping[object, Iterable[object]] | None = None,
    blocked_ids_by_category: Mapping[object, Iterable[object] | Mapping[object, object]]
    | None = None,
    executable_card_effect_keys: Iterable[object] = (),
) -> CombatImplementationCatalog:
    return CombatImplementationCatalog(
        implemented_ids_by_category=_normalize_implemented_ids(
            implemented_ids_by_category or {}
        ),
        blocked_ids_by_category=_normalize_blocked_ids(blocked_ids_by_category or {}),
        executable_card_effect_keys=frozenset(
            _normalized_id(key) for key in executable_card_effect_keys
        ),
    )


def default_combat_implementation_catalog() -> CombatImplementationCatalog:
    from sts2sim.mechanics.card_effects import EXECUTABLE_EFFECT_KEYS
    from sts2sim.mechanics.potions import supported_combat_potion_ids
    from sts2sim.mechanics.relic_combat import supported_combat_relic_ids
    from sts2sim.mechanics.relics import supported_relic_ids
    from sts2sim.mechanics.reward_triggers import DEFAULT_REWARD_MODIFIERS

    reward_relic_ids = frozenset(modifier.content_id for modifier in DEFAULT_REWARD_MODIFIERS)
    blocked_card_ids: Mapping[str, str] = {}

    return combat_implementation_catalog(
        implemented_ids_by_category={
            CombatCoverageCategory.CARDS: {"guilty"},
            CombatCoverageCategory.POTIONS: supported_combat_potion_ids(),
            CombatCoverageCategory.RELICS: (
                supported_relic_ids() | supported_combat_relic_ids() | reward_relic_ids
            ),
        },
        blocked_ids_by_category={
            CombatCoverageCategory.CARDS: blocked_card_ids,
        },
        executable_card_effect_keys=EXECUTABLE_EFFECT_KEYS,
    )


def load_cached_combat_content(
    cache_dir: str | Path | None = None,
    *,
    cards_path: str | Path | None = None,
    relics_path: str | Path | None = None,
    potions_path: str | Path | None = None,
    monsters_path: str | Path | None = None,
    encounters_path: str | Path | None = None,
) -> CombatSourceContent:
    base = Path(cache_dir) if cache_dir is not None else DEFAULT_COMBAT_CACHE_DIR
    return CombatSourceContent(
        cards=_load_cached_rows(cards_path or base / "cards.json", label="cards"),
        relics=_load_cached_rows(relics_path or base / "relics.json", label="relics"),
        potions=_load_cached_rows(potions_path or base / "potions.json", label="potions"),
        monsters=_load_cached_rows(monsters_path or base / "monsters.json", label="monsters"),
        encounters=_load_cached_rows(
            encounters_path or base / "encounters.json",
            label="encounters",
        ),
    )


def audit_combat_coverage(
    cache_dir: str | Path | None = None,
    *,
    cards_path: str | Path | None = None,
    relics_path: str | Path | None = None,
    potions_path: str | Path | None = None,
    monsters_path: str | Path | None = None,
    encounters_path: str | Path | None = None,
    implementation_catalog: CombatImplementationCatalog | None = None,
    unknown_sample_size: int = 5,
) -> CombatCoverageReport:
    content = load_cached_combat_content(
        cache_dir,
        cards_path=cards_path,
        relics_path=relics_path,
        potions_path=potions_path,
        monsters_path=monsters_path,
        encounters_path=encounters_path,
    )
    return audit_combat_coverage_from_sources(
        cards=content.cards,
        relics=content.relics,
        potions=content.potions,
        monsters=content.monsters,
        encounters=content.encounters,
        implementation_catalog=implementation_catalog,
        unknown_sample_size=unknown_sample_size,
    )


def audit_combat_coverage_from_sources(
    *,
    cards: Sequence[CachedCombatRow] = (),
    relics: Sequence[CachedCombatRow] = (),
    potions: Sequence[CachedCombatRow] = (),
    monsters: Sequence[CachedCombatRow] = (),
    encounters: Sequence[CachedCombatRow] = (),
    implementation_catalog: CombatImplementationCatalog | None = None,
    unknown_sample_size: int = 5,
) -> CombatCoverageReport:
    catalog = implementation_catalog or default_combat_implementation_catalog()
    monster_entries = _monster_entries(monsters, catalog)
    monster_statuses = {entry.normalized_id: entry.status for entry in monster_entries}
    entries = (
        _card_entries(cards, catalog)
        + _id_registry_entries(CombatCoverageCategory.RELICS, relics, catalog)
        + _id_registry_entries(CombatCoverageCategory.POTIONS, potions, catalog)
        + monster_entries
        + _encounter_entries(encounters, catalog, monster_statuses)
    )
    return CombatCoverageReport(
        entries=entries,
        unknown_sample_size=max(0, int(unknown_sample_size)),
    )


def _card_entries(
    cards: Sequence[CachedCombatRow],
    catalog: CombatImplementationCatalog,
) -> tuple[CombatCoverageEntry, ...]:
    from sts2sim.mechanics.card_effects import card_effect_plan

    card_library = _card_library(cards)
    entries: list[CombatCoverageEntry] = []
    for index, card in enumerate(cards):
        content_id = _source_id(card, CombatCoverageCategory.CARDS, index)
        normalized = _normalized_id(content_id)
        name = _source_name(card, content_id)
        blocker_reasons = catalog.blocker_reasons(CombatCoverageCategory.CARDS, content_id)
        if blocker_reasons:
            entries.append(
                _entry(
                    CombatCoverageCategory.CARDS,
                    content_id,
                    name,
                    CombatCoverageStatus.BLOCKED,
                    blocked_keys=(normalized,),
                    reasons=blocker_reasons,
                )
            )
            continue

        explicit_handler = normalized in catalog.implemented_ids(CombatCoverageCategory.CARDS)
        effect_keys: tuple[str, ...] = ()
        unknown_keys: tuple[str, ...] = ()
        reasons: tuple[str, ...] = ()
        try:
            plan = card_effect_plan(card, card_library=card_library)
            effect_keys = _effect_keys_from_steps(plan.steps)
            raw_effect_keys = _raw_explicit_effect_keys(card)
            unknown_keys = tuple(
                key
                for key in _unique_ids((*effect_keys, *raw_effect_keys))
                if key not in catalog.executable_card_effect_keys
            )
        except Exception as exc:
            reasons = (f"Card effect normalization failed: {exc}",)

        if explicit_handler:
            status = CombatCoverageStatus.IMPLEMENTED
            if not reasons:
                reasons = ("Explicit card handler registry match.",)
        elif reasons:
            status = CombatCoverageStatus.UNKNOWN
        elif unknown_keys:
            status = CombatCoverageStatus.UNKNOWN
            reasons = ("Card emits effect keys outside the executable effect key set.",)
        elif not effect_keys:
            status = CombatCoverageStatus.UNKNOWN
            reasons = ("No executable card effect keys were discovered.",)
        else:
            status = CombatCoverageStatus.IMPLEMENTED
            reasons = ("Source card normalized to executable effect keys.",)

        entries.append(
            _entry(
                CombatCoverageCategory.CARDS,
                content_id,
                name,
                status,
                implemented_keys=effect_keys if status is CombatCoverageStatus.IMPLEMENTED else (),
                unknown_keys=unknown_keys,
                reasons=reasons,
            )
        )
    return tuple(entries)


def _id_registry_entries(
    category: CombatCoverageCategory,
    rows: Sequence[CachedCombatRow],
    catalog: CombatImplementationCatalog,
) -> tuple[CombatCoverageEntry, ...]:
    entries: list[CombatCoverageEntry] = []
    implemented_ids = catalog.implemented_ids(category)
    for index, row in enumerate(rows):
        content_id = _source_id(row, category, index)
        normalized = _normalized_id(content_id)
        name = _source_name(row, content_id)
        blocker_reasons = catalog.blocker_reasons(category, content_id)
        implemented_keys: tuple[str, ...]
        blocked_keys: tuple[str, ...]
        unknown_keys: tuple[str, ...]
        reasons: tuple[str, ...]
        if blocker_reasons:
            status = CombatCoverageStatus.BLOCKED
            reasons = blocker_reasons
            implemented_keys = ()
            blocked_keys = (normalized,)
            unknown_keys = ()
        elif normalized in implemented_ids:
            status = CombatCoverageStatus.IMPLEMENTED
            reasons = ("Handler registry match.",)
            implemented_keys = (normalized,)
            blocked_keys = ()
            unknown_keys = ()
        else:
            status = CombatCoverageStatus.UNKNOWN
            reasons = (f"No matching {category.value} handler registry entry.",)
            implemented_keys = ()
            blocked_keys = ()
            unknown_keys = (normalized,)

        entries.append(
            _entry(
                category,
                content_id,
                name,
                status,
                implemented_keys=implemented_keys,
                blocked_keys=blocked_keys,
                unknown_keys=unknown_keys,
                reasons=reasons,
            )
        )
    return tuple(entries)


def _monster_entries(
    monsters: Sequence[CachedCombatRow],
    catalog: CombatImplementationCatalog,
) -> tuple[CombatCoverageEntry, ...]:
    from sts2sim.mechanics.monster_specials import classify_all_monster_specials
    from sts2sim.mechanics.monsters import build_monster_definitions

    definitions = build_monster_definitions(monsters)
    raw_sources = {
        _source_id(monster, CombatCoverageCategory.MONSTERS, index): monster
        for index, monster in enumerate(monsters)
    }
    classifications = classify_all_monster_specials(
        definitions,
        raw_sources=raw_sources,
    )
    entries: list[CombatCoverageEntry] = []
    implemented_ids = catalog.implemented_ids(CombatCoverageCategory.MONSTERS)
    for index, monster in enumerate(monsters):
        content_id = _source_id(monster, CombatCoverageCategory.MONSTERS, index)
        normalized = _normalized_id(content_id)
        name = _source_name(monster, content_id)
        blocker_reasons = catalog.blocker_reasons(CombatCoverageCategory.MONSTERS, content_id)
        definition = _definition_for_id(definitions, content_id)
        classification = classifications.get(content_id)
        implemented_keys: tuple[str, ...]
        blocked_keys: tuple[str, ...]
        unknown_keys: tuple[str, ...]
        reasons: tuple[str, ...]
        if blocker_reasons:
            status = CombatCoverageStatus.BLOCKED
            reasons = blocker_reasons
            implemented_keys = ()
            blocked_keys = (normalized,)
            unknown_keys = ()
        elif normalized in implemented_ids:
            status = CombatCoverageStatus.IMPLEMENTED
            reasons = ("Explicit monster handler registry match.",)
            implemented_keys = (normalized,)
            blocked_keys = ()
            unknown_keys = ()
        elif definition is None:
            status = CombatCoverageStatus.UNKNOWN
            reasons = ("Monster source row could not be parsed into a definition.",)
            implemented_keys = ()
            blocked_keys = ()
            unknown_keys = (normalized,)
        elif not definition.moves:
            status = CombatCoverageStatus.UNKNOWN
            reasons = ("Monster definition has no executable moves.",)
            implemented_keys = ()
            blocked_keys = ()
            unknown_keys = (normalized,)
        elif classification is not None and classification.blocked:
            status = CombatCoverageStatus.BLOCKED
            reasons = tuple(
                blocker.blocker or blocker.detail for blocker in classification.blockers
            )
            implemented_keys = ()
            blocked_keys = _unique_ids(blocker.code for blocker in classification.blockers)
            unknown_keys = ()
        else:
            status = CombatCoverageStatus.IMPLEMENTED
            if classification is not None and classification.hints:
                reasons = (
                    "Source monster parsed into executable move definitions with "
                    "deterministic special-handling hints.",
                )
            else:
                reasons = ("Source monster parsed into executable move definitions.",)
            implemented_keys = _unique_ids(move.move_id for move in definition.moves)
            blocked_keys = ()
            unknown_keys = ()

        entries.append(
            _entry(
                CombatCoverageCategory.MONSTERS,
                content_id,
                name,
                status,
                implemented_keys=implemented_keys,
                blocked_keys=blocked_keys,
                unknown_keys=unknown_keys,
                reasons=reasons,
            )
        )
    return tuple(entries)


def _encounter_entries(
    encounters: Sequence[CachedCombatRow],
    catalog: CombatImplementationCatalog,
    monster_statuses: Mapping[str, CombatCoverageStatus],
) -> tuple[CombatCoverageEntry, ...]:
    from sts2sim.mechanics.monsters import build_encounter_definitions

    definitions = build_encounter_definitions(encounters)
    entries: list[CombatCoverageEntry] = []
    implemented_ids = catalog.implemented_ids(CombatCoverageCategory.ENCOUNTERS)
    for index, encounter in enumerate(encounters):
        content_id = _source_id(encounter, CombatCoverageCategory.ENCOUNTERS, index)
        normalized = _normalized_id(content_id)
        name = _source_name(encounter, content_id)
        blocker_reasons = catalog.blocker_reasons(CombatCoverageCategory.ENCOUNTERS, content_id)
        definition = _definition_for_id(definitions, content_id)
        implemented_keys: tuple[str, ...]
        blocked_keys: tuple[str, ...]
        unknown_keys: tuple[str, ...]
        reasons: tuple[str, ...]

        if blocker_reasons:
            status = CombatCoverageStatus.BLOCKED
            reasons = blocker_reasons
            implemented_keys = ()
            blocked_keys = (normalized,)
            unknown_keys = ()
        elif normalized in implemented_ids:
            status = CombatCoverageStatus.IMPLEMENTED
            reasons = ("Explicit encounter handler registry match.",)
            implemented_keys = (normalized,)
            blocked_keys = ()
            unknown_keys = ()
        elif definition is None:
            status = CombatCoverageStatus.UNKNOWN
            reasons = ("Encounter source row could not be parsed into a definition.",)
            implemented_keys = ()
            blocked_keys = ()
            unknown_keys = (normalized,)
        else:
            blocked_monster_ids = tuple(
                monster_id
                for monster_id in definition.monster_ids
                if monster_statuses.get(_normalized_id(monster_id))
                is CombatCoverageStatus.BLOCKED
            )
            unknown_monster_ids = tuple(
                monster_id
                for monster_id in definition.monster_ids
                if monster_statuses.get(_normalized_id(monster_id))
                in {None, CombatCoverageStatus.UNKNOWN}
            )
            if blocked_monster_ids:
                status = CombatCoverageStatus.BLOCKED
                reasons = ("Encounter references blocked monster ids.",)
                implemented_keys = ()
                blocked_keys = _unique_ids(blocked_monster_ids)
                unknown_keys = ()
            elif unknown_monster_ids:
                status = CombatCoverageStatus.UNKNOWN
                reasons = ("Encounter references unknown monster ids.",)
                implemented_keys = ()
                blocked_keys = ()
                unknown_keys = _unique_ids(unknown_monster_ids)
            else:
                status = CombatCoverageStatus.IMPLEMENTED
                reasons = ("Source encounter parsed and all monster ids are executable.",)
                implemented_keys = _unique_ids(definition.monster_ids)
                blocked_keys = ()
                unknown_keys = ()

        entries.append(
            _entry(
                CombatCoverageCategory.ENCOUNTERS,
                content_id,
                name,
                status,
                implemented_keys=implemented_keys,
                blocked_keys=blocked_keys,
                unknown_keys=unknown_keys,
                reasons=reasons,
            )
        )
    return tuple(entries)


def _entry(
    category: CombatCoverageCategory,
    content_id: str,
    name: str,
    status: CombatCoverageStatus,
    *,
    implemented_keys: Iterable[object] = (),
    blocked_keys: Iterable[object] = (),
    unknown_keys: Iterable[object] = (),
    reasons: Iterable[object] = (),
) -> CombatCoverageEntry:
    return CombatCoverageEntry(
        category=category,
        content_id=content_id,
        normalized_id=_normalized_id(content_id),
        name=name,
        status=status,
        implemented_keys=_unique_ids(implemented_keys),
        blocked_keys=_unique_ids(blocked_keys),
        unknown_keys=_unique_ids(unknown_keys),
        reasons=_unique_text(reasons),
    )


def _load_cached_rows(path: str | Path, *, label: str) -> tuple[CachedCombatRow, ...]:
    resolved_path = Path(path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Cached {label} payload must be a list: {resolved_path}")

    rows: list[CachedCombatRow] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"Cached {label} item at index {index} is not an object.")
        rows.append(cast(CachedCombatRow, item))
    return tuple(rows)


def _normalize_implemented_ids(
    raw_registry: Mapping[object, Iterable[object]],
) -> dict[str, frozenset[str]]:
    normalized: dict[str, frozenset[str]] = {}
    for raw_category, raw_ids in raw_registry.items():
        category = _category_value(raw_category)
        normalized[category] = frozenset(_normalized_id(raw_id) for raw_id in raw_ids)
    for category in CombatCoverageCategory:
        normalized.setdefault(category.value, frozenset())
    return normalized


def _normalize_blocked_ids(
    raw_registry: Mapping[object, Iterable[object] | Mapping[object, object]],
) -> dict[str, dict[str, tuple[str, ...]]]:
    normalized: dict[str, dict[str, tuple[str, ...]]] = {
        category.value: {} for category in CombatCoverageCategory
    }
    for raw_category, raw_blockers in raw_registry.items():
        category = _category_value(raw_category)
        if isinstance(raw_blockers, Mapping):
            for raw_id, raw_reasons in raw_blockers.items():
                normalized[category][_normalized_id(raw_id)] = _reason_tuple(raw_reasons)
            continue
        for raw_id in raw_blockers:
            normalized[category][_normalized_id(raw_id)] = ("Blocked by coverage registry.",)
    return normalized


def _card_library(cards: Sequence[CachedCombatRow]) -> dict[str, Mapping[str, Any]]:
    library: dict[str, Mapping[str, Any]] = {}
    for index, card in enumerate(cards):
        content_id = _source_id(card, CombatCoverageCategory.CARDS, index)
        library[content_id] = card
        library[_normalized_id(content_id)] = card
        library[content_id.upper()] = card
    return library


def _effect_keys_from_steps(steps: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return _unique_ids(key for step in steps for key in _effect_keys_from_mapping(step))


def _effect_keys_from_mapping(value: Mapping[str, Any]) -> tuple[str, ...]:
    keys: list[str] = []
    for raw_key, raw_value in value.items():
        key = _normalized_id(raw_key)
        if key in {"sequence", "effects"}:
            keys.extend(_effect_keys_from_value(raw_value))
        elif key:
            keys.append(key)
    return _unique_ids(keys)


def _effect_keys_from_value(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return _effect_keys_from_mapping(cast(Mapping[str, Any], value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _unique_ids(
            key
            for item in value
            for key in _effect_keys_from_value(item)
        )
    return ()


def _raw_explicit_effect_keys(card: CachedCombatRow) -> tuple[str, ...]:
    explicit = card.get("effects", card.get("effect"))
    if isinstance(explicit, Mapping):
        return _effect_keys_from_mapping(cast(Mapping[str, Any], explicit))
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes, bytearray)):
        return _unique_ids(
            key
            for item in explicit
            for key in _effect_keys_from_value(item)
        )
    return ()


def _definition_for_id(definitions: object, content_id: str) -> Any | None:
    normalized = _normalized_id(content_id)
    if isinstance(definitions, Mapping):
        for raw_id, definition in definitions.items():
            if _normalized_id(raw_id) == normalized:
                return definition
        return None
    if isinstance(definitions, Sequence) and not isinstance(definitions, (str, bytes, bytearray)):
        for definition in definitions:
            for attr_name in ("monster_id", "encounter_id", "content_id", "id"):
                raw_id = getattr(definition, attr_name, None)
                if raw_id is not None and _normalized_id(raw_id) == normalized:
                    return definition
    return None


def _source_id(
    row: CachedCombatRow,
    category: CombatCoverageCategory,
    index: int,
) -> str:
    for key in ("id", "card_id", "relic_id", "potion_id", "monster_id", "encounter_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return f"<missing-{category.value}-{index}>"


def _source_name(row: CachedCombatRow, fallback: str) -> str:
    value = row.get("name")
    return fallback if value in (None, "") else str(value)


def _category_enum(category: CombatCoverageCategory | str) -> CombatCoverageCategory:
    return CombatCoverageCategory(_category_value(category))


def _category_value(category: object) -> str:
    if isinstance(category, CombatCoverageCategory):
        return category.value
    normalized = str(category).strip().lower().replace("_", "-")
    aliases = {
        "card": CombatCoverageCategory.CARDS.value,
        "cards": CombatCoverageCategory.CARDS.value,
        "relic": CombatCoverageCategory.RELICS.value,
        "relics": CombatCoverageCategory.RELICS.value,
        "potion": CombatCoverageCategory.POTIONS.value,
        "potions": CombatCoverageCategory.POTIONS.value,
        "monster": CombatCoverageCategory.MONSTERS.value,
        "monsters": CombatCoverageCategory.MONSTERS.value,
        "encounter": CombatCoverageCategory.ENCOUNTERS.value,
        "encounters": CombatCoverageCategory.ENCOUNTERS.value,
    }
    return aliases.get(normalized, normalized)


def _status_enum(status: CombatCoverageStatus | str) -> CombatCoverageStatus:
    if isinstance(status, CombatCoverageStatus):
        return status
    return CombatCoverageStatus(str(status).strip().lower().replace("_", "-"))


def _reason_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ("Blocked by coverage registry.",)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        reasons = tuple(str(reason) for reason in value if str(reason))
        return reasons or ("Blocked by coverage registry.",)
    return (str(value),)


def _unique_ids(values: Iterable[object]) -> tuple[str, ...]:
    seen: set[str] = set()
    results: list[str] = []
    for value in values:
        text = _normalized_id(value)
        if not text or text in seen:
            continue
        seen.add(text)
        results.append(text)
    return tuple(results)


def _unique_text(values: Iterable[object]) -> tuple[str, ...]:
    seen: set[str] = set()
    results: list[str] = []
    for value in values:
        text = str(value)
        key = _normalized_id(text)
        if not key or key in seen:
            continue
        seen.add(key)
        results.append(text)
    return tuple(results)


def _normalized_id(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("'", "")
        .replace(" ", "_")
        .replace("-", "_")
    )


__all__ = [
    "DEFAULT_CARDS_PATH",
    "DEFAULT_COMBAT_CACHE_DIR",
    "DEFAULT_ENCOUNTERS_PATH",
    "DEFAULT_MONSTERS_PATH",
    "DEFAULT_POTIONS_PATH",
    "DEFAULT_RELICS_PATH",
    "CachedCombatRow",
    "CombatCoverageCategory",
    "CombatCoverageEntry",
    "CombatCoverageReport",
    "CombatCoverageStatus",
    "CombatCoverageSummary",
    "CombatImplementationCatalog",
    "CombatSourceContent",
    "audit_combat_coverage",
    "audit_combat_coverage_from_sources",
    "combat_implementation_catalog",
    "default_combat_implementation_catalog",
    "load_cached_combat_content",
]
