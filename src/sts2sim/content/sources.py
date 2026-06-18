"""Small source metadata helpers used by content and mechanics tables.

The simulator should be driven by extracted game data where possible.  These
objects keep provisional mechanics explicit until a real source manifest is
available.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Reference for a mechanic, content entry, or coverage marker."""

    label: str
    version: str | None = None
    url: str | None = None
    captured_at: str | None = None
    notes: str | None = None

    def describe(self) -> str:
        parts = [self.label]
        if self.version:
            parts.append(f"version={self.version}")
        if self.captured_at:
            parts.append(f"captured_at={self.captured_at}")
        return " ".join(parts)


PROVISIONAL_STS2_SOURCE = SourceRef(
    label="sts2-provisional-mechanics",
    version="scaffold",
    notes=(
        "Placeholder values used until Slay the Spire 2 runtime data, "
        "official patch notes, or curated extracted manifests are registered."
    ),
)

STS1_COMPAT_SOURCE = SourceRef(
    label="sts1-compatible-reference",
    version="scaffold",
    notes=(
        "A conservative Slay the Spire compatible default used only as a "
        "mechanics shape reference. Treat as a blocker for exact STS2 tuning."
    ),
)


def merge_sources(*sources: SourceRef | None) -> tuple[SourceRef, ...]:
    """Return sources without duplicates while preserving order."""

    seen: set[tuple[str, str | None, str | None]] = set()
    merged: list[SourceRef] = []
    for source in sources:
        if source is None:
            continue
        key = (source.label, source.version, source.captured_at)
        if key in seen:
            continue
        seen.add(key)
        merged.append(source)
    return tuple(merged)
