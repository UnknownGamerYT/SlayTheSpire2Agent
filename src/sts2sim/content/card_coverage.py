"""Card-specific executable mechanics audit.

The broad combat coverage audit answers whether a card has any executable combat
surface. This module is narrower: it classifies each source card as fully
implemented, partially implemented, or missing, and exposes the exact effect
keys, keyword rules, and blocker kinds still needed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

from sts2sim.mechanics.card_effects import EXECUTABLE_EFFECT_KEYS, card_effect_plan
from sts2sim.mechanics.card_specials import card_special_plan

DEFAULT_CARD_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "eng"
DEFAULT_CARDS_PATH = DEFAULT_CARD_CACHE_DIR / "cards.json"

CachedCardRow = Mapping[str, Any]


class CardCoverageStatus(str, Enum):
    IMPLEMENTED = "implemented"
    PARTIAL = "partial"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class CardCoverageEntry:
    content_id: str
    normalized_id: str
    name: str
    status: CardCoverageStatus
    color: str = "unknown"
    card_type: str = "unknown"
    rarity: str = "unknown"
    cost: int | str | None = None
    executable_keys: tuple[str, ...] = ()
    blocker_kinds: tuple[str, ...] = ()
    unknown_keys: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def is_implemented(self) -> bool:
        return self.status is CardCoverageStatus.IMPLEMENTED

    @property
    def is_partial(self) -> bool:
        return self.status is CardCoverageStatus.PARTIAL

    @property
    def is_missing(self) -> bool:
        return self.status is CardCoverageStatus.MISSING

    def as_dict(self) -> dict[str, object]:
        return {
            "content_id": self.content_id,
            "normalized_id": self.normalized_id,
            "name": self.name,
            "status": self.status.value,
            "color": self.color,
            "card_type": self.card_type,
            "rarity": self.rarity,
            "cost": self.cost,
            "executable_keys": list(self.executable_keys),
            "blocker_kinds": list(self.blocker_kinds),
            "unknown_keys": list(self.unknown_keys),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class CardCoverageReport:
    entries: tuple[CardCoverageEntry, ...]
    sample_size: int = 10

    @property
    def total_cards(self) -> int:
        return len(self.entries)

    @property
    def counts_by_status(self) -> dict[str, int]:
        return _status_counts(self.entries)

    @property
    def counts_by_color(self) -> dict[str, dict[str, int]]:
        return _bucketed_counts(self.entries, key="color")

    @property
    def counts_by_type(self) -> dict[str, dict[str, int]]:
        return _bucketed_counts(self.entries, key="card_type")

    @property
    def implemented_ratio(self) -> float:
        if not self.entries:
            return 1.0
        return self.counts_by_status[CardCoverageStatus.IMPLEMENTED.value] / len(self.entries)

    @property
    def executable_ratio(self) -> float:
        if not self.entries:
            return 1.0
        executable = (
            self.counts_by_status[CardCoverageStatus.IMPLEMENTED.value]
            + self.counts_by_status[CardCoverageStatus.PARTIAL.value]
        )
        return executable / len(self.entries)

    @property
    def sample_partial_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.content_id
            for entry in self.entries_for(status=CardCoverageStatus.PARTIAL)[: self.sample_size]
        )

    @property
    def sample_missing_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.content_id
            for entry in self.entries_for(status=CardCoverageStatus.MISSING)[: self.sample_size]
        )

    def entries_for(
        self,
        *,
        status: CardCoverageStatus | str | None = None,
        color: str | None = None,
        card_type: str | None = None,
    ) -> tuple[CardCoverageEntry, ...]:
        wanted_status = _status_enum(status) if status is not None else None
        wanted_color = _normalized_id(color) if color is not None else None
        wanted_type = _normalized_id(card_type) if card_type is not None else None
        return tuple(
            entry
            for entry in self.entries
            if (wanted_status is None or entry.status is wanted_status)
            and (wanted_color is None or entry.color == wanted_color)
            and (wanted_type is None or entry.card_type == wanted_type)
        )

    def entry_for(self, content_id: str) -> CardCoverageEntry:
        normalized = _normalized_id(content_id)
        for entry in self.entries:
            if entry.normalized_id == normalized:
                return entry
        raise KeyError(f"Unknown card id in card coverage: {content_id}")

    def as_dict(self) -> dict[str, object]:
        return {
            "total_cards": self.total_cards,
            "counts_by_status": self.counts_by_status,
            "counts_by_color": self.counts_by_color,
            "counts_by_type": self.counts_by_type,
            "implemented_ratio": self.implemented_ratio,
            "executable_ratio": self.executable_ratio,
            "sample_partial_ids": list(self.sample_partial_ids),
            "sample_missing_ids": list(self.sample_missing_ids),
            "entries": [entry.as_dict() for entry in self.entries],
        }


def load_cached_cards(cards_path: str | Path | None = None) -> tuple[CachedCardRow, ...]:
    resolved_path = Path(cards_path) if cards_path is not None else DEFAULT_CARDS_PATH
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Cached cards payload must be a list: {resolved_path}")

    rows: list[CachedCardRow] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"Cached cards item at index {index} is not an object.")
        rows.append(cast(CachedCardRow, item))
    return tuple(rows)


def audit_card_coverage(
    cache_dir: str | Path | None = None,
    *,
    cards_path: str | Path | None = None,
    executable_effect_keys: Iterable[object] | None = None,
    sample_size: int = 10,
) -> CardCoverageReport:
    base = Path(cache_dir) if cache_dir is not None else DEFAULT_CARD_CACHE_DIR
    cards = load_cached_cards(cards_path or base / "cards.json")
    return audit_card_coverage_from_sources(
        cards=cards,
        executable_effect_keys=executable_effect_keys,
        sample_size=sample_size,
    )


def audit_card_coverage_from_sources(
    *,
    cards: Sequence[CachedCardRow],
    executable_effect_keys: Iterable[object] | None = None,
    sample_size: int = 10,
) -> CardCoverageReport:
    raw_executable_keys = (
        EXECUTABLE_EFFECT_KEYS if executable_effect_keys is None else executable_effect_keys
    )
    executable_keys = frozenset(
        _normalized_id(key) for key in raw_executable_keys
    )
    card_library = _card_library(cards)
    entries = tuple(
        _card_entry(
            card,
            index=index,
            card_library=card_library,
            executable_effect_keys=executable_keys,
        )
        for index, card in enumerate(cards)
    )
    return CardCoverageReport(entries=entries, sample_size=max(0, int(sample_size)))


def _card_entry(
    card: CachedCardRow,
    *,
    index: int,
    card_library: Mapping[str, Mapping[str, Any]],
    executable_effect_keys: frozenset[str],
) -> CardCoverageEntry:
    content_id = _source_id(card, index)
    normalized = _normalized_id(content_id)
    name = _source_name(card, content_id)
    color = _source_text(card, "color", default="unknown")
    card_type = _source_text(card, "type", "card_type", default="unknown")
    rarity = _source_text(card, "rarity", "rarity_key", default="unknown")
    cost = _source_cost(card)

    executable_keys: tuple[str, ...] = ()
    unknown_keys: tuple[str, ...] = ()
    blocker_kinds: tuple[str, ...] = ()
    reasons: list[str] = []
    normalization_failed = False

    try:
        effect_plan = card_effect_plan(card, card_library=card_library)
        effect_keys = _effect_keys_from_steps(effect_plan.steps)
        raw_effect_keys = _raw_explicit_effect_keys(card)
        keyword_keys, keyword_unknown_keys, keyword_reasons = _keyword_audit(card)
        executable_keys = _unique_ids(
            key
            for key in (*effect_keys, *keyword_keys)
            if key in executable_effect_keys or key in _SUPPORTED_CARD_MECHANIC_KEYS
        )
        unknown_keys = _unique_ids(
            (
                key
                for key in (*effect_keys, *raw_effect_keys)
                if key not in executable_effect_keys
            ),
            keyword_unknown_keys,
        )
        reasons.extend(keyword_reasons)

        special_plan = card_special_plan(card)
        blocker_kinds = _blocker_kinds(special_plan.blockers)
        reasons.extend(special_plan.reasons)
    except Exception as exc:
        normalization_failed = True
        reasons.append(f"Card effect normalization failed: {exc}")

    if normalization_failed:
        status = CardCoverageStatus.MISSING
    elif blocker_kinds or unknown_keys:
        status = CardCoverageStatus.PARTIAL if executable_keys else CardCoverageStatus.MISSING
        if unknown_keys:
            reasons.append("Card emits effect or keyword keys outside the executable set.")
    elif executable_keys:
        status = CardCoverageStatus.IMPLEMENTED
        reasons.append("Card normalizes to executable mechanics.")
    else:
        status = CardCoverageStatus.MISSING
        reasons.append("No executable card mechanics were discovered.")

    return CardCoverageEntry(
        content_id=content_id,
        normalized_id=normalized,
        name=name,
        status=status,
        color=color,
        card_type=card_type,
        rarity=rarity,
        cost=cost,
        executable_keys=executable_keys,
        blocker_kinds=blocker_kinds,
        unknown_keys=unknown_keys,
        reasons=_unique_text(reasons),
    )


def _keyword_audit(card: CachedCardRow) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    executable_keys: list[str] = []
    unknown_keys: list[str] = []
    reasons: list[str] = []

    if _is_x_cost(card):
        executable_keys.append("x_cost")

    for keyword in _keywords(card):
        if keyword in {
            "eternal",
            "exhaust",
            "ethereal",
            "innate",
            "retain",
            "sly",
            "unplayable",
        }:
            executable_keys.append(f"keyword_{keyword}")
            continue
        if keyword:
            unknown_keys.append(f"keyword_{keyword}")
            reasons.append(f"Unsupported card keyword: {keyword}.")

    if _description_has_standalone_retain(card):
        executable_keys.append("keyword_retain")

    return _unique_ids(executable_keys), _unique_ids(unknown_keys), _unique_text(reasons)


def _keywords(card: CachedCardRow) -> tuple[str, ...]:
    raw_keywords = card.get("keywords_key", card.get("keywords", ()))
    if isinstance(raw_keywords, str):
        return (_normalized_id(raw_keywords),)
    if isinstance(raw_keywords, Sequence):
        return _unique_ids(_normalized_id(keyword) for keyword in raw_keywords)
    return ()


def _is_x_cost(card: CachedCardRow) -> bool:
    if bool(card.get("is_x_cost", card.get("x_cost", False))):
        return True
    cost = card.get("cost")
    return isinstance(cost, int) and cost < 0 and "unplayable" not in set(_keywords(card))


def _description_has_standalone_retain(card: CachedCardRow) -> bool:
    description = str(card.get("description", card.get("description_raw", "")) or "")
    sentences = re.split(r"(?:\n|(?<=[.!?])\s+)", description.replace("\n", ". "))
    return any(sentence.strip().strip(".!?[] ").lower() == "retain" for sentence in sentences)


def _blocker_kinds(blockers: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    kinds: list[str] = []
    for blocker in blockers:
        payload = blocker.get("explicit_blocker")
        if not isinstance(payload, Mapping):
            continue
        kind = payload.get("kind")
        if kind is not None:
            kinds.append(_normalized_id(kind))
    return _unique_ids(kinds)


def _card_library(cards: Sequence[CachedCardRow]) -> dict[str, Mapping[str, Any]]:
    library: dict[str, Mapping[str, Any]] = {}
    for index, card in enumerate(cards):
        content_id = _source_id(card, index)
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
        return _unique_ids(key for item in value for key in _effect_keys_from_value(item))
    return ()


def _raw_explicit_effect_keys(card: CachedCardRow) -> tuple[str, ...]:
    explicit = card.get("effects", card.get("effect"))
    if isinstance(explicit, Mapping):
        return _effect_keys_from_mapping(cast(Mapping[str, Any], explicit))
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes, bytearray)):
        return _unique_ids(key for item in explicit for key in _effect_keys_from_value(item))
    return ()


def _source_id(row: CachedCardRow, index: int) -> str:
    for key in ("id", "card_id", "content_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return f"<missing-card-{index}>"


def _source_name(row: CachedCardRow, fallback: str) -> str:
    value = row.get("name")
    return fallback if value in (None, "") else str(value)


def _source_text(row: CachedCardRow, *keys: str, default: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return _normalized_id(value)
    return default


def _source_cost(row: CachedCardRow) -> int | str | None:
    value = row.get("cost")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | str):
        return value
    return None


def _status_counts(entries: Sequence[CardCoverageEntry]) -> dict[str, int]:
    counts = {status.value: 0 for status in CardCoverageStatus}
    for entry in entries:
        counts[entry.status.value] += 1
    return counts


def _bucketed_counts(
    entries: Sequence[CardCoverageEntry],
    *,
    key: str,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for entry in entries:
        bucket_name = str(getattr(entry, key))
        bucket = counts.setdefault(bucket_name, _status_counts(()))
        bucket[entry.status.value] += 1
    return counts


def _status_enum(status: CardCoverageStatus | str) -> CardCoverageStatus:
    if isinstance(status, CardCoverageStatus):
        return status
    normalized = _normalized_id(status)
    aliases = {
        "done": CardCoverageStatus.IMPLEMENTED.value,
        "implemented": CardCoverageStatus.IMPLEMENTED.value,
        "partial": CardCoverageStatus.PARTIAL.value,
        "blocked": CardCoverageStatus.PARTIAL.value,
        "missing": CardCoverageStatus.MISSING.value,
        "unknown": CardCoverageStatus.MISSING.value,
    }
    return CardCoverageStatus(aliases.get(normalized, normalized))


def _unique_ids(*values: Iterable[object]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for iterable in values:
        for value in iterable:
            normalized = _normalized_id(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
    return tuple(result)


def _unique_text(values: Iterable[object]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _normalized_id(value: object) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("'", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


_SUPPORTED_CARD_MECHANIC_KEYS = frozenset(
    {
        "keyword_ethereal",
        "keyword_exhaust",
        "keyword_eternal",
        "keyword_innate",
        "keyword_retain",
        "keyword_sly",
        "keyword_unplayable",
        "x_cost",
    }
)


__all__ = [
    "CachedCardRow",
    "CardCoverageEntry",
    "CardCoverageReport",
    "CardCoverageStatus",
    "DEFAULT_CARD_CACHE_DIR",
    "DEFAULT_CARDS_PATH",
    "audit_card_coverage",
    "audit_card_coverage_from_sources",
    "load_cached_cards",
]
