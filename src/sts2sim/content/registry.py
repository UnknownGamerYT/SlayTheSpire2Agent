"""Content registry and implementation coverage markers.

This module deliberately has no engine imports.  Engine code can register cards,
relics, potions, monsters, or events and then attach callable handlers by kind.
Missing required handlers are reported as audit blockers instead of silently
falling through.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .sources import PROVISIONAL_STS2_SOURCE, SourceRef, merge_sources


class ContentKind(str, Enum):
    CARD = "card"
    RELIC = "relic"
    POTION = "potion"
    MONSTER = "monster"
    EVENT = "event"
    STATUS = "status"
    SYSTEM = "system"


class HandlerKind(str, Enum):
    CARD_PLAY = "card_play"
    CARD_UPGRADE = "card_upgrade"
    RELIC_HOOK = "relic_hook"
    POTION_USE = "potion_use"
    MONSTER_ACTION = "monster_action"
    EVENT_RESOLVE = "event_resolve"
    STATUS_TICK = "status_tick"


class CoverageStatus(str, Enum):
    IMPLEMENTED = "implemented"
    SCAFFOLDED = "scaffolded"
    PLACEHOLDER = "placeholder"
    BLOCKED = "blocked"


class CoverageSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCKER = "blocker"


ContentKey = tuple[ContentKind, str]
HandlerKey = tuple[ContentKind, str, HandlerKind]
ContentHandler = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class ContentDefinition:
    """Source-backed content row registered by a manifest loader."""

    content_id: str
    kind: ContentKind
    name: str
    source: SourceRef = PROVISIONAL_STS2_SOURCE
    handler_requirements: frozenset[HandlerKind] = field(default_factory=frozenset)
    coverage: CoverageStatus = CoverageStatus.PLACEHOLDER
    tags: frozenset[str] = field(default_factory=frozenset)
    data: Mapping[str, Any] = field(default_factory=dict)
    notes: str | None = None

    @property
    def key(self) -> ContentKey:
        return (self.kind, self.content_id)


@dataclass(frozen=True, slots=True)
class HandlerRegistration:
    key: HandlerKey
    handler: ContentHandler
    source: SourceRef = PROVISIONAL_STS2_SOURCE
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class CoverageMarker:
    content_id: str
    kind: ContentKind
    status: CoverageStatus
    severity: CoverageSeverity
    message: str
    source: SourceRef = PROVISIONAL_STS2_SOURCE
    handler_kind: HandlerKind | None = None

    @property
    def key(self) -> ContentKey:
        return (self.kind, self.content_id)


@dataclass(frozen=True, slots=True)
class CoverageReport:
    total_definitions: int
    implemented_definitions: int
    scaffolded_definitions: int
    placeholder_definitions: int
    blockers: tuple[CoverageMarker, ...]
    warnings: tuple[CoverageMarker, ...]
    sources: tuple[SourceRef, ...]

    @property
    def has_blockers(self) -> bool:
        return bool(self.blockers)

    @property
    def missing_handler_count(self) -> int:
        return sum(1 for marker in self.blockers if marker.handler_kind is not None)


class MissingHandlerError(LookupError):
    """Raised when engine code requires a handler that has not been registered."""

    def __init__(self, kind: ContentKind, content_id: str, handler_kind: HandlerKind) -> None:
        self.kind = kind
        self.content_id = content_id
        self.handler_kind = handler_kind
        super().__init__(
            f"Missing {handler_kind.value} handler for {kind.value} '{content_id}'. "
            "Register a handler or leave it as an audit blocker."
        )


class ContentRegistry:
    """Registry for content definitions, handlers, and audit coverage."""

    def __init__(self) -> None:
        self._definitions: dict[ContentKey, ContentDefinition] = {}
        self._handlers: dict[HandlerKey, HandlerRegistration] = {}
        self._markers: list[CoverageMarker] = []

    def register_definition(
        self,
        definition: ContentDefinition,
        *,
        replace: bool = False,
    ) -> ContentDefinition:
        key = definition.key
        if key in self._definitions and not replace:
            raise ValueError(f"Content definition already registered: {key}")
        self._definitions[key] = definition
        return definition

    def register_many(
        self,
        definitions: Iterable[ContentDefinition],
        *,
        replace: bool = False,
    ) -> None:
        for definition in definitions:
            self.register_definition(definition, replace=replace)

    def register_handler(
        self,
        kind: ContentKind,
        content_id: str,
        handler_kind: HandlerKind,
        handler: ContentHandler,
        *,
        source: SourceRef = PROVISIONAL_STS2_SOURCE,
        notes: str | None = None,
        replace: bool = False,
    ) -> HandlerRegistration:
        key = (kind, content_id, handler_kind)
        if key in self._handlers and not replace:
            raise ValueError(f"Handler already registered: {key}")
        registration = HandlerRegistration(key=key, handler=handler, source=source, notes=notes)
        self._handlers[key] = registration
        return registration

    def get_definition(self, kind: ContentKind, content_id: str) -> ContentDefinition | None:
        return self._definitions.get((kind, content_id))

    def require_definition(self, kind: ContentKind, content_id: str) -> ContentDefinition:
        definition = self.get_definition(kind, content_id)
        if definition is None:
            raise KeyError(f"Unknown {kind.value} content id: {content_id}")
        return definition

    def iter_definitions(self, kind: ContentKind | None = None) -> Iterator[ContentDefinition]:
        for definition in self._definitions.values():
            if kind is None or definition.kind is kind:
                yield definition

    def get_handler(
        self,
        kind: ContentKind,
        content_id: str,
        handler_kind: HandlerKind,
    ) -> ContentHandler | None:
        registration = self._handlers.get((kind, content_id, handler_kind))
        if registration is None:
            return None
        return registration.handler

    def require_handler(
        self,
        kind: ContentKind,
        content_id: str,
        handler_kind: HandlerKind,
    ) -> ContentHandler:
        handler = self.get_handler(kind, content_id, handler_kind)
        if handler is None:
            raise MissingHandlerError(kind, content_id, handler_kind)
        return handler

    def mark_coverage(self, marker: CoverageMarker) -> CoverageMarker:
        self._markers.append(marker)
        return marker

    def mark_blocker(
        self,
        kind: ContentKind,
        content_id: str,
        message: str,
        *,
        handler_kind: HandlerKind | None = None,
        source: SourceRef = PROVISIONAL_STS2_SOURCE,
    ) -> CoverageMarker:
        marker = CoverageMarker(
            content_id=content_id,
            kind=kind,
            status=CoverageStatus.BLOCKED,
            severity=CoverageSeverity.BLOCKER,
            message=message,
            source=source,
            handler_kind=handler_kind,
        )
        return self.mark_coverage(marker)

    def audit(self) -> CoverageReport:
        markers = list(self._markers)
        implemented = 0
        scaffolded = 0
        placeholders = 0
        sources: list[SourceRef] = []

        for definition in self._definitions.values():
            sources.extend(merge_sources(definition.source))
            if definition.coverage is CoverageStatus.IMPLEMENTED:
                implemented += 1
            elif definition.coverage is CoverageStatus.SCAFFOLDED:
                scaffolded += 1
            elif definition.coverage is CoverageStatus.PLACEHOLDER:
                placeholders += 1
                markers.append(
                    CoverageMarker(
                        content_id=definition.content_id,
                        kind=definition.kind,
                        status=CoverageStatus.PLACEHOLDER,
                        severity=CoverageSeverity.WARNING,
                        message="Content definition is present but marked as placeholder.",
                        source=definition.source,
                    )
                )
            elif definition.coverage is CoverageStatus.BLOCKED:
                markers.append(
                    CoverageMarker(
                        content_id=definition.content_id,
                        kind=definition.kind,
                        status=CoverageStatus.BLOCKED,
                        severity=CoverageSeverity.BLOCKER,
                        message=definition.notes or "Content definition is blocked.",
                        source=definition.source,
                    )
                )

            for handler_kind in definition.handler_requirements:
                key = (definition.kind, definition.content_id, handler_kind)
                if key not in self._handlers:
                    markers.append(
                        CoverageMarker(
                            content_id=definition.content_id,
                            kind=definition.kind,
                            status=CoverageStatus.BLOCKED,
                            severity=CoverageSeverity.BLOCKER,
                            message=f"Required {handler_kind.value} handler is not registered.",
                            source=definition.source,
                            handler_kind=handler_kind,
                        )
                    )

        for registration in self._handlers.values():
            sources.extend(merge_sources(registration.source))

        merged_sources: tuple[SourceRef, ...] = ()
        for source in sources:
            merged_sources = merge_sources(*merged_sources, source)

        blockers = tuple(
            marker for marker in markers if marker.severity is CoverageSeverity.BLOCKER
        )
        warnings = tuple(
            marker for marker in markers if marker.severity is CoverageSeverity.WARNING
        )
        return CoverageReport(
            total_definitions=len(self._definitions),
            implemented_definitions=implemented,
            scaffolded_definitions=scaffolded,
            placeholder_definitions=placeholders,
            blockers=blockers,
            warnings=warnings,
            sources=merged_sources,
        )


def card_definition(
    content_id: str,
    name: str,
    *,
    source: SourceRef = PROVISIONAL_STS2_SOURCE,
    coverage: CoverageStatus = CoverageStatus.PLACEHOLDER,
    tags: Iterable[str] = (),
    data: Mapping[str, Any] | None = None,
    requires_upgrade_handler: bool = False,
    notes: str | None = None,
) -> ContentDefinition:
    requirements = {HandlerKind.CARD_PLAY}
    if requires_upgrade_handler:
        requirements.add(HandlerKind.CARD_UPGRADE)
    return ContentDefinition(
        content_id=content_id,
        kind=ContentKind.CARD,
        name=name,
        source=source,
        handler_requirements=frozenset(requirements),
        coverage=coverage,
        tags=frozenset(tags),
        data=data or {},
        notes=notes,
    )


def relic_definition(
    content_id: str,
    name: str,
    *,
    source: SourceRef = PROVISIONAL_STS2_SOURCE,
    coverage: CoverageStatus = CoverageStatus.PLACEHOLDER,
    tags: Iterable[str] = (),
    data: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> ContentDefinition:
    return ContentDefinition(
        content_id=content_id,
        kind=ContentKind.RELIC,
        name=name,
        source=source,
        handler_requirements=frozenset({HandlerKind.RELIC_HOOK}),
        coverage=coverage,
        tags=frozenset(tags),
        data=data or {},
        notes=notes,
    )


def potion_definition(
    content_id: str,
    name: str,
    *,
    source: SourceRef = PROVISIONAL_STS2_SOURCE,
    coverage: CoverageStatus = CoverageStatus.PLACEHOLDER,
    tags: Iterable[str] = (),
    data: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> ContentDefinition:
    return ContentDefinition(
        content_id=content_id,
        kind=ContentKind.POTION,
        name=name,
        source=source,
        handler_requirements=frozenset({HandlerKind.POTION_USE}),
        coverage=coverage,
        tags=frozenset(tags),
        data=data or {},
        notes=notes,
    )


def create_registry(definitions: Iterable[ContentDefinition] = ()) -> ContentRegistry:
    registry = ContentRegistry()
    registry.register_many(definitions)
    return registry


def scaffold_registry() -> ContentRegistry:
    """Return an empty registry with explicit manifest blockers.

    This is useful for CLI audits before a source manifest loader exists.
    """

    registry = ContentRegistry()
    registry.mark_blocker(
        ContentKind.SYSTEM,
        "sts2-content-manifest",
        "No extracted STS2 content manifest has been registered yet.",
    )
    registry.mark_blocker(
        ContentKind.CARD,
        "sts2-card-handlers",
        "Card definitions and play handlers must be loaded from source data.",
    )
    registry.mark_blocker(
        ContentKind.RELIC,
        "sts2-relic-handlers",
        "Relic definitions and hook handlers must be loaded from source data.",
    )
    registry.mark_blocker(
        ContentKind.POTION,
        "sts2-potion-handlers",
        "Potion definitions and use handlers must be loaded from source data.",
    )
    return registry
