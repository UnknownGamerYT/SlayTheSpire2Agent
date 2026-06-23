"""Audit cached event data against implemented event mechanics catalogs."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, cast

CachedEvent = Mapping[str, Any]

DEFAULT_EVENTS_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "cache" / "eng" / "events.json"
)

OPTIONAL_EVENT_MODULES = (
    "sts2sim.mechanics.event_catalog",
    "sts2sim.mechanics.event_specials",
    "sts2sim.mechanics.event_flows",
    "sts2sim.content.event_catalog",
    "sts2sim.content.event_specials",
    "sts2sim.content.event_flows",
)

_CATALOG_ATTRIBUTE_NAMES = (
    "_EVENT_CATALOG",
    "_EVENT_IMPLEMENTATIONS",
    "_EVENT_OPTIONS",
    "_IMPLEMENTED_EVENTS",
    "_KNOWN_EVENT_OPTIONS",
    "_EVENT_SPECIALS",
    "_SPECIAL_EVENTS",
    "_EVENT_FLOWS",
    "_EVENT_FLOW_BUILDERS",
    "_EVENT_PAGE_BUILDERS",
    "_STEPWISE_EVENTS",
    "EVENT_CATALOG",
    "EVENT_IMPLEMENTATIONS",
    "EVENT_OPTIONS",
    "IMPLEMENTED_EVENTS",
    "KNOWN_EVENT_OPTIONS",
    "CATALOG",
    "EVENT_SPECIALS",
    "SPECIAL_EVENTS",
    "SPECIALS",
    "EVENT_FLOWS",
    "EVENT_FLOW_BUILDERS",
    "EVENT_PAGE_BUILDERS",
    "STEPWISE_EVENTS",
    "FLOWS",
)

_CALLABLE_CATALOG_NAMES = (
    "event_catalog_coverage",
    "event_coverage_entries",
    "event_implementations",
    "known_event_implementations",
    "special_event_implementations",
)

_ENTRY_METADATA_KEYS = {
    "act",
    "category",
    "complete",
    "coverage",
    "coverage_category",
    "covers_all",
    "covers_all_options",
    "event_id",
    "id",
    "name",
    "note",
    "notes",
    "option_ids",
    "options",
    "status",
    "type",
}

UNSUPPORTED_BESPOKE_EVENT_IDS = frozenset[str]()


class EventCoverageCategory(str, Enum):
    IMPLEMENTED = "implemented"
    PRIMITIVE = "primitive"
    STEPWISE = "stepwise"
    SPECIAL = "special"
    ANCIENT_ONLY = "ancient-only"
    UNSUPPORTED_BESPOKE = "unsupported/bespoke"


@dataclass(frozen=True, slots=True)
class EventImplementation:
    event_id: str
    category: EventCoverageCategory
    option_ids: tuple[str, ...] = ()
    source_module: str = ""
    covers_all_options: bool = False
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class EventCoverageEntry:
    event_id: str
    name: str
    act: str | None
    event_type: str | None
    category: EventCoverageCategory
    cached_option_ids: tuple[str, ...]
    implemented_option_ids: tuple[str, ...]
    missing_option_ids: tuple[str, ...]
    source_modules: tuple[str, ...] = ()
    explicitly_marked: bool = False
    notes: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return not self.missing_option_ids

    def as_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "name": self.name,
            "act": self.act,
            "event_type": self.event_type,
            "category": self.category.value,
            "cached_option_ids": list(self.cached_option_ids),
            "implemented_option_ids": list(self.implemented_option_ids),
            "missing_option_ids": list(self.missing_option_ids),
            "source_modules": list(self.source_modules),
            "explicitly_marked": self.explicitly_marked,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class EventCoverageReport:
    entries: tuple[EventCoverageEntry, ...]
    optional_module_errors: tuple[str, ...] = ()

    @property
    def total_events(self) -> int:
        return len(self.entries)

    @property
    def counts_by_category(self) -> dict[str, int]:
        counts = {category.value: 0 for category in EventCoverageCategory}
        for entry in self.entries:
            counts[entry.category.value] += 1
        return counts

    @property
    def unsupported_events(self) -> tuple[EventCoverageEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.category is EventCoverageCategory.UNSUPPORTED_BESPOKE
        )

    @property
    def by_event_id(self) -> dict[str, EventCoverageEntry]:
        return {entry.event_id: entry for entry in self.entries}

    def entry_for(self, event_id: str) -> EventCoverageEntry:
        key = _normalized_id(event_id)
        for entry in self.entries:
            if _normalized_id(entry.event_id) == key:
                return entry
        raise KeyError(f"Unknown event id in coverage report: {event_id}")

    def as_dict(self) -> dict[str, object]:
        return {
            "total_events": self.total_events,
            "counts_by_category": self.counts_by_category,
            "optional_module_errors": list(self.optional_module_errors),
            "events": [entry.as_dict() for entry in self.entries],
        }


def load_cached_events(events_path: str | Path | None = None) -> tuple[CachedEvent, ...]:
    path = Path(events_path) if events_path is not None else DEFAULT_EVENTS_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Cached events payload must be a list: {path}")

    events: list[CachedEvent] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"Cached event at index {index} is not an object: {path}")
        events.append(cast(CachedEvent, item))
    return tuple(events)


def audit_event_coverage(
    events_path: str | Path | None = None,
    *,
    optional_module_names: Sequence[str] = OPTIONAL_EVENT_MODULES,
) -> EventCoverageReport:
    events = load_cached_events(events_path)
    implementations, errors = discover_event_implementations(optional_module_names)
    implementations_by_id = _implementations_by_event_id(implementations)

    entries = tuple(
        _coverage_entry(event, implementations_by_id.get(_normalized_id(event.get("id", "")), ()))
        for event in events
    )
    return EventCoverageReport(entries=entries, optional_module_errors=errors)


def discover_event_implementations(
    optional_module_names: Sequence[str] = OPTIONAL_EVENT_MODULES,
) -> tuple[tuple[EventImplementation, ...], tuple[str, ...]]:
    implementations: list[EventImplementation] = []
    errors: list[str] = []

    room_implementations, room_errors = _event_room_implementations()
    implementations.extend(room_implementations)
    errors.extend(room_errors)

    optional_implementations, optional_errors = _optional_module_implementations(
        optional_module_names
    )
    implementations.extend(optional_implementations)
    errors.extend(optional_errors)

    return tuple(implementations), tuple(errors)


def _coverage_entry(
    event: CachedEvent,
    implementations: tuple[EventImplementation, ...],
) -> EventCoverageEntry:
    event_id = str(event.get("id", ""))
    event_type = _optional_str(event.get("type"))
    cached_option_ids = _cached_option_ids(event)

    if implementations:
        category = _best_category(implementation.category for implementation in implementations)
        covers_all = any(implementation.covers_all_options for implementation in implementations)
        source_modules = _unique_ids(
            implementation.source_module
            for implementation in implementations
            if implementation.source_module
        )
        notes = tuple(
            note for note in (implementation.notes for implementation in implementations) if note
        )
        implemented_option_ids = (
            cached_option_ids
            if covers_all
            else _unique_ids(
                option_id
                for implementation in implementations
                for option_id in implementation.option_ids
            )
        )
        missing_option_ids = (
            ()
            if covers_all
            else _missing_option_ids(cached_option_ids, implemented_option_ids)
        )
        explicitly_marked = False
    elif event_type == "Ancient":
        category = EventCoverageCategory.ANCIENT_ONLY
        source_modules = ()
        notes = ("Ancient encounter record; not a normal event-room option catalog.",)
        implemented_option_ids = ()
        missing_option_ids = _missing_option_ids(cached_option_ids, implemented_option_ids)
        explicitly_marked = False
    else:
        category = EventCoverageCategory.UNSUPPORTED_BESPOKE
        source_modules = ()
        implemented_option_ids = ()
        missing_option_ids = cached_option_ids
        explicitly_marked = _normalized_id(event_id) in _normalized_unsupported_ids()
        notes = (
            (
                "Explicit unsupported/bespoke marker; no matching event mechanics "
                "catalog entry is present."
            ),
        ) if explicitly_marked else ("No matching event mechanics catalog entry is present.",)

    return EventCoverageEntry(
        event_id=event_id,
        name=str(event.get("name", event_id)),
        act=_optional_str(event.get("act")),
        event_type=event_type,
        category=category,
        cached_option_ids=cached_option_ids,
        implemented_option_ids=implemented_option_ids,
        missing_option_ids=missing_option_ids,
        source_modules=source_modules,
        explicitly_marked=explicitly_marked,
        notes=notes,
    )


def _event_room_implementations() -> tuple[tuple[EventImplementation, ...], tuple[str, ...]]:
    module_name = "sts2sim.mechanics.event_rooms"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - this is a required local module.
        return (), (f"{module_name}: {exc}",)

    raw_catalog = getattr(module, "_KNOWN_EVENT_OPTIONS", None)
    if not isinstance(raw_catalog, Mapping):
        return (), (f"{module_name}: _KNOWN_EVENT_OPTIONS is not available.",)

    implementations = [
        EventImplementation(
            event_id=str(event_id),
            category=EventCoverageCategory.PRIMITIVE,
            option_ids=_option_ids_from_value(options),
            source_module=module_name,
            covers_all_options=False,
        )
        for event_id, options in raw_catalog.items()
    ]
    return tuple(implementations), ()


def _optional_module_implementations(
    module_names: Sequence[str],
) -> tuple[tuple[EventImplementation, ...], tuple[str, ...]]:
    implementations: list[EventImplementation] = []
    errors: list[str] = []

    for module_name in module_names:
        module = _import_optional_module(module_name, errors)
        if module is None:
            continue
        implementations.extend(
            _implementations_from_module(module, _default_category_for_module(module_name))
        )

    return tuple(implementations), tuple(errors)


def _import_optional_module(module_name: str, errors: list[str]) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        errors.append(f"{module_name}: {exc}")
    except Exception as exc:
        errors.append(f"{module_name}: {exc}")
    return None


def _implementations_from_module(
    module: ModuleType,
    default_category: EventCoverageCategory,
) -> tuple[EventImplementation, ...]:
    implementations: list[EventImplementation] = []

    for attr_name in _CATALOG_ATTRIBUTE_NAMES:
        if not hasattr(module, attr_name):
            continue
        value = getattr(module, attr_name)
        implementations.extend(
            _implementations_from_value(
                value,
                default_category,
                module.__name__,
                attr_name,
            )
        )

    for attr_name in _CALLABLE_CATALOG_NAMES:
        value = getattr(module, attr_name, None)
        if value is None or not callable(value):
            continue
        provider = cast(Callable[[], object], value)
        implementations.extend(
            _implementations_from_value(
                provider(),
                default_category,
                module.__name__,
                attr_name,
            )
        )

    return tuple(implementations)


def _implementations_from_value(
    value: object,
    default_category: EventCoverageCategory,
    source_module: str,
    source_attr: str,
) -> tuple[EventImplementation, ...]:
    if value is None:
        return ()
    if _should_skip_entry(value, source_attr):
        return ()
    if isinstance(value, str):
        return (
            EventImplementation(
                event_id=value,
                category=default_category,
                source_module=source_module,
                covers_all_options=True,
                notes=f"Discovered from {source_attr}.",
            ),
        )
    if isinstance(value, Mapping):
        return _implementations_from_mapping(value, default_category, source_module, source_attr)
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        implementations: list[EventImplementation] = []
        for item in value:
            implementations.extend(
                _implementations_from_value(
                    item,
                    default_category,
                    source_module,
                    source_attr,
                )
            )
        return tuple(implementations)

    event_id = _event_id_from_object(value)
    if event_id is None:
        return ()
    option_ids = _option_ids_from_value(value)
    return (
        EventImplementation(
            event_id=event_id,
            category=_category_from_entry(value, default_category),
            option_ids=option_ids,
            source_module=source_module,
            covers_all_options=_covers_all_options(value, default=not option_ids),
            notes=_notes_from_entry(value) or f"Discovered from {source_attr}.",
        ),
    )


def _implementations_from_mapping(
    value: Mapping[object, object],
    default_category: EventCoverageCategory,
    source_module: str,
    source_attr: str,
) -> tuple[EventImplementation, ...]:
    event_id = _event_id_from_mapping(value)
    if event_id is not None:
        option_ids = _option_ids_from_entry(value)
        return (
            EventImplementation(
                event_id=event_id,
                category=_category_from_entry(value, default_category),
                option_ids=option_ids,
                source_module=source_module,
                covers_all_options=_covers_all_options(value, default=not option_ids),
                notes=_notes_from_entry(value) or f"Discovered from {source_attr}.",
            ),
        )

    implementations: list[EventImplementation] = []
    for raw_key, raw_entry in value.items():
        if isinstance(raw_key, str) and raw_key.lower() in {"events", "catalog", "items"}:
            implementations.extend(
                _implementations_from_value(
                    raw_entry,
                    default_category,
                    source_module,
                    source_attr,
                )
            )
            continue
        if not isinstance(raw_key, str) or raw_key.lower() in _ENTRY_METADATA_KEYS:
            continue
        option_ids = _option_ids_from_entry(raw_entry)
        implementations.append(
            EventImplementation(
                event_id=raw_key,
                category=_category_from_entry(raw_entry, default_category),
                option_ids=option_ids,
                source_module=source_module,
                covers_all_options=_covers_all_options(raw_entry, default=not option_ids),
                notes=_notes_from_entry(raw_entry) or f"Discovered from {source_attr}.",
            )
        )
    return tuple(implementations)


def _implementations_by_event_id(
    implementations: Iterable[EventImplementation],
) -> dict[str, tuple[EventImplementation, ...]]:
    grouped: dict[str, list[EventImplementation]] = {}
    for implementation in implementations:
        grouped.setdefault(_normalized_id(implementation.event_id), []).append(implementation)
    return {event_id: tuple(values) for event_id, values in grouped.items()}


def _cached_option_ids(event: CachedEvent) -> tuple[str, ...]:
    option_ids: list[str] = []

    def add_options(raw_options: object) -> None:
        if isinstance(raw_options, str) or not isinstance(raw_options, Sequence):
            return
        for option in raw_options:
            if not isinstance(option, Mapping):
                continue
            option_id = option.get("id")
            if option_id is not None:
                option_ids.append(str(option_id))

    add_options(event.get("options"))
    pages = event.get("pages")
    if isinstance(pages, Sequence) and not isinstance(pages, str):
        for page in pages:
            if isinstance(page, Mapping):
                add_options(page.get("options"))

    return _unique_ids(option_ids)


def _option_ids_from_entry(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        for key in ("implemented_option_ids", "option_ids"):
            raw_option_ids = value.get(key)
            if raw_option_ids is not None:
                return _option_ids_from_value(raw_option_ids)
        raw_option_id = value.get("option_id")
        if raw_option_id is not None:
            return (str(raw_option_id),)
        raw_options = value.get("options")
        if raw_options is not None:
            return _option_ids_from_value(raw_options)
        option_like_keys = [
            str(raw_key)
            for raw_key in value
            if isinstance(raw_key, str) and raw_key.lower() not in _ENTRY_METADATA_KEYS
        ]
        return _unique_ids(option_like_keys)
    return _option_ids_from_value(value)


def _option_ids_from_value(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    option_id = _option_id_from_option(value)
    if option_id is not None:
        return (option_id,)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        raw_options = value.get("options")
        if raw_options is not None:
            return _option_ids_from_value(raw_options)
        return ()
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return _unique_ids(
            option_id
            for item in value
            for option_id in _option_ids_from_value(item)
        )
    return ()


def _option_id_from_option(value: object) -> str | None:
    if isinstance(value, Mapping):
        raw_option_id = value.get("option_id")
        if raw_option_id is not None:
            return str(raw_option_id)
        raw_id = value.get("id")
        if raw_id is not None and "event_id" not in value:
            return str(raw_id)
        return None
    for attr_name in ("option_id",):
        raw_option_id = getattr(value, attr_name, None)
        if raw_option_id is not None:
            return str(raw_option_id)
    return None


def _event_id_from_mapping(value: Mapping[object, object]) -> str | None:
    for key in ("event_id", "id"):
        raw_event_id = value.get(key)
        if raw_event_id is not None:
            return str(raw_event_id)
    return None


def _event_id_from_object(value: object) -> str | None:
    for attr_name in ("event_id", "id"):
        raw_event_id = getattr(value, attr_name, None)
        if raw_event_id is not None:
            return str(raw_event_id)
    return None


def _category_from_entry(
    value: object,
    default_category: EventCoverageCategory,
) -> EventCoverageCategory:
    raw_category: object | None = None
    if isinstance(value, Mapping):
        for key in ("coverage_category", "category", "coverage", "status"):
            raw_category = value.get(key)
            if raw_category is not None:
                break
    else:
        for attr_name in ("coverage_category", "category", "coverage", "status"):
            raw_category = getattr(value, attr_name, None)
            if raw_category is not None:
                break

    if raw_category is None:
        return default_category
    return _category_from_value(raw_category, default_category)


def _category_from_value(
    value: object,
    default_category: EventCoverageCategory,
) -> EventCoverageCategory:
    if isinstance(value, EventCoverageCategory):
        return value

    raw_value = getattr(value, "value", value)
    normalized = _normalized_id(raw_value)
    aliases = {
        "implemented": EventCoverageCategory.IMPLEMENTED,
        "primitive": EventCoverageCategory.PRIMITIVE,
        "stepwise": EventCoverageCategory.STEPWISE,
        "flow": EventCoverageCategory.STEPWISE,
        "flows": EventCoverageCategory.STEPWISE,
        "special": EventCoverageCategory.SPECIAL,
        "specials": EventCoverageCategory.SPECIAL,
        "ancient_only": EventCoverageCategory.ANCIENT_ONLY,
        "unsupported": EventCoverageCategory.UNSUPPORTED_BESPOKE,
        "unsupported_bespoke": EventCoverageCategory.UNSUPPORTED_BESPOKE,
        "bespoke": EventCoverageCategory.UNSUPPORTED_BESPOKE,
    }
    return aliases.get(normalized, default_category)


def _covers_all_options(value: object, *, default: bool) -> bool:
    raw_value: object | None = None
    if isinstance(value, Mapping):
        for key in ("covers_all_options", "covers_all", "complete"):
            raw_value = value.get(key)
            if raw_value is not None:
                break
    else:
        for attr_name in ("covers_all_options", "covers_all", "complete"):
            raw_value = getattr(value, attr_name, None)
            if raw_value is not None:
                break
    if raw_value is None:
        return default
    return bool(raw_value)


def _should_skip_entry(value: object, source_attr: str) -> bool:
    if source_attr != "event_catalog_coverage":
        return False
    # Catalog coverage rows are source-backed option records.  A row marked
    # unsupported still means the option was discovered and classified, even if
    # the primitive event-room model cannot execute it directly.
    return False


def _notes_from_entry(value: object) -> str | None:
    raw_notes: object | None = None
    if isinstance(value, Mapping):
        raw_notes = value.get("notes", value.get("note"))
    else:
        raw_notes = getattr(value, "notes", getattr(value, "note", None))
    if raw_notes is None:
        return None
    if isinstance(raw_notes, str):
        return raw_notes
    if isinstance(raw_notes, Sequence):
        return " ".join(str(note) for note in raw_notes)
    return str(raw_notes)


def _best_category(categories: Iterable[EventCoverageCategory]) -> EventCoverageCategory:
    rank = {
        EventCoverageCategory.IMPLEMENTED: 0,
        EventCoverageCategory.SPECIAL: 1,
        EventCoverageCategory.STEPWISE: 2,
        EventCoverageCategory.PRIMITIVE: 3,
        EventCoverageCategory.ANCIENT_ONLY: 4,
        EventCoverageCategory.UNSUPPORTED_BESPOKE: 5,
    }
    return min(categories, key=lambda category: rank[category])


def _default_category_for_module(module_name: str) -> EventCoverageCategory:
    if module_name.endswith("event_catalog"):
        return EventCoverageCategory.PRIMITIVE
    if module_name.endswith("event_specials"):
        return EventCoverageCategory.SPECIAL
    if module_name.endswith("event_flows"):
        return EventCoverageCategory.STEPWISE
    return EventCoverageCategory.IMPLEMENTED


def _missing_option_ids(
    cached_option_ids: Sequence[str],
    implemented_option_ids: Sequence[str],
) -> tuple[str, ...]:
    implemented = {_normalized_id(option_id) for option_id in implemented_option_ids}
    return tuple(
        option_id
        for option_id in cached_option_ids
        if _normalized_id(option_id) not in implemented
    )


def _unique_ids(values: Iterable[object]) -> tuple[str, ...]:
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


def _normalized_unsupported_ids() -> frozenset[str]:
    return frozenset(_normalized_id(event_id) for event_id in UNSUPPORTED_BESPOKE_EVENT_IDS)


def _normalized_id(value: object) -> str:
    return str(value).lower().replace("'", "").replace(" ", "_").replace("-", "_")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = [
    "DEFAULT_EVENTS_PATH",
    "OPTIONAL_EVENT_MODULES",
    "UNSUPPORTED_BESPOKE_EVENT_IDS",
    "EventCoverageCategory",
    "EventCoverageEntry",
    "EventCoverageReport",
    "EventImplementation",
    "audit_event_coverage",
    "discover_event_implementations",
    "load_cached_events",
]
