"""Parse coarse event option effects from cached Spire Codex text.

This module is intentionally standalone: it reads only the option description
string and does not import engine state, transition models, or content catalogs.
The output is meant for audits and scaffolding, not authoritative resolution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from re import Match


class EventTextEffectKind(str, Enum):
    GOLD_GAIN = "gold_gain"
    GOLD_LOSS = "gold_loss"
    HP_LOSS = "hp_loss"
    HP_HEAL = "hp_heal"
    MAX_HP_GAIN = "max_hp_gain"
    MAX_HP_LOSS = "max_hp_loss"
    MAX_HP_SET = "max_hp_set"
    RELIC_OBTAIN = "relic_obtain"
    POTION_PROCURE = "potion_procure"
    POTION_LOSS = "potion_loss"
    CARD_ADD = "card_add"
    CARD_REMOVE = "card_remove"
    CARD_UPGRADE = "card_upgrade"
    CARD_TRANSFORM = "card_transform"
    FIGHT = "fight"


@dataclass(frozen=True, slots=True)
class EventTextAmount:
    """Numeric value parsed from event option text."""

    value: int | None = None
    minimum: int | None = None
    maximum: int | None = None
    percent: int | None = None
    all: bool = False
    full: bool = False
    or_more: bool = False
    raw: str = ""


@dataclass(frozen=True, slots=True)
class EventTextEffect:
    """One parsed primitive from an event option description."""

    kind: EventTextEffectKind
    amount: EventTextAmount | None = None
    count: int | None = None
    item_kind: str | None = None
    item_name: str | None = None
    random: bool = False
    qualifier: str | None = None
    target: str | None = None
    source_text: str = ""


@dataclass(frozen=True, slots=True)
class _EffectMatch:
    start: int
    end: int
    effect: EventTextEffect


_TAG_RE = re.compile(r"\[/?[A-Za-z][A-Za-z0-9_ -]*\]")
_AMOUNT_TOKEN_RE = r"(?:ALL|\d+-\d+|\d+\+?|\d+%\s+Max)"
_COUNT_TOKEN_RE = r"(?:\d+|one|an|a)"
_NAME_RE = r"[A-Z][A-Za-z0-9' -]*"

_GOLD_GAIN_RE = re.compile(
    rf"\b(?:Gain|Obtain)\s+(?P<amount>{_AMOUNT_TOKEN_RE})\s+Gold\b",
    re.IGNORECASE,
)
_GOLD_LOSS_RE = re.compile(
    rf"\b(?P<verb>Lose|Pay)\s+(?P<amount>{_AMOUNT_TOKEN_RE})(?:\s+of your)?\s+Gold\b",
    re.IGNORECASE,
)
_HP_LOSS_RE = re.compile(
    rf"\b(?:Lose|Take)\s+(?P<amount>{_AMOUNT_TOKEN_RE})\s+(?:HP|damage)\b",
    re.IGNORECASE,
)
_HP_HEAL_RE = re.compile(
    rf"\bHeal\s+(?:(?P<full>to full HP)|(?P<amount>{_AMOUNT_TOKEN_RE})\s+HP)\b",
    re.IGNORECASE,
)
_MAX_HP_GAIN_RE = re.compile(
    rf"\b(?:Gain\s+(?P<gain>{_AMOUNT_TOKEN_RE})\s+Max HP|"
    rf"raise your Max HP by\s+(?P<raise>{_AMOUNT_TOKEN_RE}))\b",
    re.IGNORECASE,
)
_MAX_HP_LOSS_RE = re.compile(
    rf"\bLose\s+(?P<amount>{_AMOUNT_TOKEN_RE})\s+Max HP\b",
    re.IGNORECASE,
)
_MAX_HP_SET_RE = re.compile(
    rf"\bSet Max HP to\s+(?P<amount>{_AMOUNT_TOKEN_RE})\b",
    re.IGNORECASE,
)
_RANDOM_RELIC_RE = re.compile(
    rf"\b(?P<verb>Obtain|Receive|Procure)\s+(?:(?P<count>{_COUNT_TOKEN_RE})\s+)?"
    r"random\s+(?:(?P<qualifier>Common|Uncommon|Rare|Doll|forgotten)\s+)?Relics?\b"
    r"(?:\s+from the past)?",
    re.IGNORECASE,
)
_FIXED_RELIC_RE = re.compile(
    rf"\b(?P<verb>Obtain|Receive)\s+(?:the\s+|an?\s+)?(?P<name>{_NAME_RE})\b",
    re.IGNORECASE,
)
_RANDOM_POTION_RE = re.compile(
    rf"\b(?P<verb>Procure|Obtain|Receive)\s+(?:(?P<count>{_COUNT_TOKEN_RE})\s+)?"
    r"random\s+(?:(?P<qualifier>Common|Uncommon|Rare)\s+)?Potions?\b",
    re.IGNORECASE,
)
_FIXED_POTION_RE = re.compile(
    rf"\bProcure\s+(?:(?P<count>\d+)\s+)?(?P<name>{_NAME_RE}\s+Potions?)\b",
    re.IGNORECASE,
)
_POTION_LOSS_RE = re.compile(
    r"\bLose\s+(?P<random>a random\s+)?(?:a\s+)?Potion\b",
    re.IGNORECASE,
)
_RANDOM_CARD_ADD_RE = re.compile(
    r"\bAdd\s+a random\s+(?P<qualifier>[A-Za-z]+)\s+Card\s+to your Deck\b",
    re.IGNORECASE,
)
_RANDOM_CARD_CHOICE_ADD_RE = re.compile(
    rf"\bChoose\s+(?P<count>{_COUNT_TOKEN_RE})\s+of\s+\d+\s+random\s+"
    r"(?:(?P<qualifier>Common|Uncommon|Rare)\s+)?cards\s+to add to your Deck\b",
    re.IGNORECASE,
)
_FIXED_CARD_ADD_RE = re.compile(
    rf"\bAdd\s+(?:(?P<count>{_COUNT_TOKEN_RE})\s+)?(?P<name>{_NAME_RE})\s+to your Deck\b",
    re.IGNORECASE,
)
_FIXED_CARD_REMOVE_RE = re.compile(
    rf"\bRemove\s+(?P<count>{_COUNT_TOKEN_RE})\s+(?P<name>{_NAME_RE})\b",
    re.IGNORECASE,
)
_GENERIC_CARD_REMOVE_RE = re.compile(
    rf"\bRemove\s+(?P<count>{_COUNT_TOKEN_RE})\s+cards?\s+from your Deck\b",
    re.IGNORECASE,
)
_RANDOM_CARD_REMOVE_RE = re.compile(
    r"\ba random Card is removed from your Deck\b",
    re.IGNORECASE,
)
_FIXED_CARD_LOSS_RE = re.compile(
    rf"\bLose\s+(?P<name>{_NAME_RE})\b",
    re.IGNORECASE,
)
_CARD_UPGRADE_RE = re.compile(
    rf"\bUpgrade\s+(?P<count>ALL|{_COUNT_TOKEN_RE})\s+"
    r"(?P<random>random\s+)?cards?(?:\s+in your Deck)?\b",
    re.IGNORECASE,
)
_CARD_TRANSFORM_DIRECT_RE = re.compile(
    rf"\bTransform\s+(?P<count>{_COUNT_TOKEN_RE})\s+cards?(?:\s+in your Deck)?\b",
    re.IGNORECASE,
)
_CARD_TRANSFORM_CHOOSE_RE = re.compile(
    rf"\bChoose\s+(?P<count>{_COUNT_TOKEN_RE})\s+(?P<qualifier>starter\s+)?"
    rf"card\s+to\s+Transform(?:\s+into\s+(?P<target>{_NAME_RE}))?\b",
    re.IGNORECASE,
)
_CARD_TRANSFORM_CHOOSE_UNCOUNTED_RE = re.compile(
    rf"\bChoose\s+(?P<qualifier>starter\s+)?card\s+to\s+Transform"
    rf"(?:\s+into\s+(?P<target>{_NAME_RE}))?\b",
    re.IGNORECASE,
)
_FIGHT_RE = re.compile(r"\bFight(?:\s+(?P<detail>[^.]+))?", re.IGNORECASE)


def normalize_event_option_description(description: str) -> str:
    """Return event option text with Spire Codex markup and extra whitespace removed."""

    untagged = _TAG_RE.sub("", description)
    return " ".join(untagged.split())


def parse_event_option_effects(description: str) -> tuple[EventTextEffect, ...]:
    """Parse common event option effect primitives from a Codex description string."""

    text = normalize_event_option_description(description)
    if not text:
        return ()

    matches: list[_EffectMatch] = []
    matches.extend(_parse_gold(text))
    matches.extend(_parse_hp(text))
    matches.extend(_parse_relics(text))
    matches.extend(_parse_potions(text))
    matches.extend(_parse_cards(text))
    matches.extend(_parse_fights(text))

    selected: list[EventTextEffect] = []
    occupied: list[tuple[int, int]] = []
    for candidate in sorted(matches, key=lambda match: (match.start, match.end)):
        if _overlaps(candidate.start, candidate.end, occupied):
            continue
        occupied.append((candidate.start, candidate.end))
        selected.append(candidate.effect)
    return tuple(selected)


def _parse_gold(text: str) -> list[_EffectMatch]:
    matches: list[_EffectMatch] = []
    for match in _GOLD_GAIN_RE.finditer(text):
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.GOLD_GAIN,
                    amount=_amount_from_match(match),
                    item_kind="gold",
                    source_text=match.group(0),
                ),
            )
        )
    for match in _GOLD_LOSS_RE.finditer(text):
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.GOLD_LOSS,
                    amount=_amount_from_match(match),
                    item_kind="gold",
                    source_text=match.group(0),
                ),
            )
        )
    return matches


def _parse_hp(text: str) -> list[_EffectMatch]:
    matches: list[_EffectMatch] = []
    for match in _HP_LOSS_RE.finditer(text):
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.HP_LOSS,
                    amount=_amount_from_match(match),
                    item_kind="hp",
                    source_text=match.group(0),
                ),
            )
        )
    for match in _HP_HEAL_RE.finditer(text):
        amount = (
            EventTextAmount(full=True, raw="full")
            if match.groupdict().get("full")
            else _amount_from_match(match)
        )
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.HP_HEAL,
                    amount=amount,
                    item_kind="hp",
                    source_text=match.group(0),
                ),
            )
        )
    for match in _MAX_HP_GAIN_RE.finditer(text):
        raw = match.groupdict().get("gain") or match.groupdict().get("raise")
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.MAX_HP_GAIN,
                    amount=_amount_from_token(raw or ""),
                    item_kind="max_hp",
                    source_text=match.group(0),
                ),
            )
        )
    for match in _MAX_HP_LOSS_RE.finditer(text):
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.MAX_HP_LOSS,
                    amount=_amount_from_match(match),
                    item_kind="max_hp",
                    source_text=match.group(0),
                ),
            )
        )
    for match in _MAX_HP_SET_RE.finditer(text):
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.MAX_HP_SET,
                    amount=_amount_from_match(match),
                    item_kind="max_hp",
                    source_text=match.group(0),
                ),
            )
        )
    return matches


def _parse_relics(text: str) -> list[_EffectMatch]:
    matches: list[_EffectMatch] = []
    for match in _RANDOM_RELIC_RE.finditer(text):
        qualifier = _optional_lower(match.groupdict().get("qualifier"))
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.RELIC_OBTAIN,
                    count=_count_from_match(match),
                    item_kind="relic",
                    random=True,
                    qualifier=qualifier,
                    source_text=match.group(0),
                ),
            )
        )
    for match in _FIXED_RELIC_RE.finditer(text):
        name = _clean_name(match.group("name"))
        if not _looks_like_fixed_relic_name(name):
            continue
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.RELIC_OBTAIN,
                    count=1,
                    item_kind="relic",
                    item_name=name,
                    source_text=match.group(0),
                ),
            )
        )
    return matches


def _parse_potions(text: str) -> list[_EffectMatch]:
    matches: list[_EffectMatch] = []
    for match in _RANDOM_POTION_RE.finditer(text):
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.POTION_PROCURE,
                    count=_count_from_match(match),
                    item_kind="potion",
                    random=True,
                    qualifier=_optional_lower(match.groupdict().get("qualifier")),
                    source_text=match.group(0),
                ),
            )
        )
    for match in _FIXED_POTION_RE.finditer(text):
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.POTION_PROCURE,
                    count=_count_from_match(match),
                    item_kind="potion",
                    item_name=_singular_potion_name(_clean_name(match.group("name"))),
                    source_text=match.group(0),
                ),
            )
        )
    for match in _POTION_LOSS_RE.finditer(text):
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.POTION_LOSS,
                    count=1,
                    item_kind="potion",
                    random=bool(match.groupdict().get("random")),
                    source_text=match.group(0),
                ),
            )
        )
    return matches


def _parse_cards(text: str) -> list[_EffectMatch]:
    matches: list[_EffectMatch] = []
    matches.extend(_parse_card_adds(text))
    matches.extend(_parse_card_removals(text))
    matches.extend(_parse_card_upgrades(text))
    matches.extend(_parse_card_transforms(text))
    return matches


def _parse_card_adds(text: str) -> list[_EffectMatch]:
    matches: list[_EffectMatch] = []
    for match in _RANDOM_CARD_ADD_RE.finditer(text):
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.CARD_ADD,
                    count=1,
                    item_kind="card",
                    random=True,
                    qualifier=_optional_lower(match.group("qualifier")),
                    target="deck",
                    source_text=match.group(0),
                ),
            )
        )
    for match in _RANDOM_CARD_CHOICE_ADD_RE.finditer(text):
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.CARD_ADD,
                    count=_count_from_match(match),
                    item_kind="card",
                    random=True,
                    qualifier=_optional_lower(match.groupdict().get("qualifier")),
                    target="deck",
                    source_text=match.group(0),
                ),
            )
        )
    for match in _FIXED_CARD_ADD_RE.finditer(text):
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.CARD_ADD,
                    count=_count_from_match(match),
                    item_kind="card",
                    item_name=_clean_name(match.group("name")),
                    target="deck",
                    source_text=match.group(0),
                ),
            )
        )
    return matches


def _parse_card_removals(text: str) -> list[_EffectMatch]:
    matches: list[_EffectMatch] = []
    for match in _FIXED_CARD_REMOVE_RE.finditer(text):
        name = _clean_name(match.group("name"))
        if name.lower() == "card" or name.lower() == "cards":
            continue
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.CARD_REMOVE,
                    count=_count_from_match(match),
                    item_kind="card",
                    item_name=_singular_card_name(name),
                    target="deck",
                    source_text=match.group(0),
                ),
            )
        )
    for match in _GENERIC_CARD_REMOVE_RE.finditer(text):
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.CARD_REMOVE,
                    count=_count_from_match(match),
                    item_kind="card",
                    target="deck",
                    source_text=match.group(0),
                ),
            )
        )
    for match in _RANDOM_CARD_REMOVE_RE.finditer(text):
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.CARD_REMOVE,
                    count=1,
                    item_kind="card",
                    random=True,
                    target="deck",
                    source_text=match.group(0),
                ),
            )
        )
    for match in _FIXED_CARD_LOSS_RE.finditer(text):
        name = _clean_name(match.group("name"))
        if not _looks_like_fixed_card_loss_name(name):
            continue
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.CARD_REMOVE,
                    count=1,
                    item_kind="card",
                    item_name=name,
                    source_text=match.group(0),
                ),
            )
        )
    return matches


def _parse_card_upgrades(text: str) -> list[_EffectMatch]:
    matches: list[_EffectMatch] = []
    for match in _CARD_UPGRADE_RE.finditer(text):
        count_text = match.group("count")
        all_cards = count_text.upper() == "ALL"
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.CARD_UPGRADE,
                    amount=EventTextAmount(all=True, raw=count_text) if all_cards else None,
                    count=None if all_cards else _count_from_token(count_text),
                    item_kind="card",
                    random=bool(match.groupdict().get("random")),
                    target="deck",
                    source_text=match.group(0),
                ),
            )
        )
    return matches


def _parse_card_transforms(text: str) -> list[_EffectMatch]:
    matches: list[_EffectMatch] = []
    for match in _CARD_TRANSFORM_DIRECT_RE.finditer(text):
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.CARD_TRANSFORM,
                    count=_count_from_match(match),
                    item_kind="card",
                    target="deck",
                    source_text=match.group(0),
                ),
            )
        )
    for match in _CARD_TRANSFORM_CHOOSE_RE.finditer(text):
        matches.append(_card_transform_choice_match(match))
    for match in _CARD_TRANSFORM_CHOOSE_UNCOUNTED_RE.finditer(text):
        matches.append(_card_transform_choice_match(match, default_count=1))
    return matches


def _parse_fights(text: str) -> list[_EffectMatch]:
    matches: list[_EffectMatch] = []
    for match in _FIGHT_RE.finditer(text):
        amount, target, qualifier = _fight_details(match.groupdict().get("detail"))
        matches.append(
            _effect_match(
                match,
                EventTextEffect(
                    kind=EventTextEffectKind.FIGHT,
                    amount=amount,
                    count=1,
                    item_kind="combat",
                    qualifier=qualifier,
                    target=target,
                    source_text=match.group(0),
                ),
            )
        )
    return matches


def _card_transform_choice_match(
    match: Match[str], *, default_count: int | None = None
) -> _EffectMatch:
    count = default_count if default_count is not None else _count_from_match(match)
    target = match.groupdict().get("target")
    qualifier = match.groupdict().get("qualifier")
    return _effect_match(
        match,
        EventTextEffect(
            kind=EventTextEffectKind.CARD_TRANSFORM,
            count=count,
            item_kind="card",
            qualifier=_optional_lower(qualifier),
            target=_clean_name(target) if target else None,
            source_text=match.group(0),
        ),
    )


def _fight_details(detail: str | None) -> tuple[EventTextAmount | None, str | None, str | None]:
    if detail is None:
        return None, None, None

    clean = detail.strip()
    hp_match = re.fullmatch(r"(?:a|an)?\s*(?P<hp>\d+)\s+HP\s+(?P<target>.+)", clean)
    if hp_match:
        return (
            _amount_from_token(hp_match.group("hp")),
            _clean_fight_target(hp_match.group("target")),
            None,
        )

    rewards_match = re.fullmatch(r"them\s+for\s+(?P<qualifier>.+)", clean, re.IGNORECASE)
    if rewards_match:
        return None, "them", _clean_name(rewards_match.group("qualifier"))

    enemies_match = re.fullmatch(r"some\s+(?P<target>.+)", clean, re.IGNORECASE)
    if enemies_match:
        return None, _clean_fight_target(enemies_match.group("target")), None

    if clean.lower().startswith("to "):
        return None, None, clean

    return None, _clean_fight_target(clean), None


def _amount_from_match(match: Match[str]) -> EventTextAmount:
    return _amount_from_token(match.group("amount"))


def _amount_from_token(raw: str) -> EventTextAmount:
    token = raw.strip()
    lower = token.lower()
    if lower == "all":
        return EventTextAmount(all=True, raw=token)

    percent_match = re.fullmatch(r"(?P<percent>\d+)%\s+max", lower)
    if percent_match:
        return EventTextAmount(percent=int(percent_match.group("percent")), raw=token)

    range_match = re.fullmatch(r"(?P<minimum>\d+)-(?P<maximum>\d+)", lower)
    if range_match:
        return EventTextAmount(
            minimum=int(range_match.group("minimum")),
            maximum=int(range_match.group("maximum")),
            raw=token,
        )

    or_more_match = re.fullmatch(r"(?P<minimum>\d+)\+", lower)
    if or_more_match:
        return EventTextAmount(
            minimum=int(or_more_match.group("minimum")),
            or_more=True,
            raw=token,
        )

    return EventTextAmount(value=int(lower), raw=token)


def _count_from_match(match: Match[str]) -> int:
    return _count_from_token(match.groupdict().get("count"))


def _count_from_token(raw: str | None) -> int:
    if raw is None:
        return 1

    token = raw.strip().lower()
    if token in {"a", "an", "one"}:
        return 1
    return int(token)


def _effect_match(match: Match[str], effect: EventTextEffect) -> _EffectMatch:
    return _EffectMatch(start=match.start(), end=match.end(), effect=effect)


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < span_end and end > span_start for span_start, span_end in spans)


def _clean_name(name: str) -> str:
    return name.strip().rstrip(".").strip()


def _clean_fight_target(target: str) -> str:
    clean = _clean_name(target)
    lower = clean.lower()
    if lower.startswith("a "):
        return clean[2:].strip()
    if lower.startswith("an "):
        return clean[3:].strip()
    return clean


def _optional_lower(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower()


def _looks_like_fixed_relic_name(name: str) -> bool:
    lower = name.lower()
    excluded_fragments = (
        "card",
        "common skill",
        "power",
        "relic",
        "gold",
        "reward",
        "potion",
    )
    return bool(name) and not any(fragment in lower for fragment in excluded_fragments)


def _looks_like_fixed_card_loss_name(name: str) -> bool:
    lower = name.lower()
    excluded_fragments = ("hp", "max hp", "gold", "potion", "relic")
    return bool(name) and not any(fragment in lower for fragment in excluded_fragments)


def _singular_card_name(name: str) -> str:
    special_cases = {
        "Strikes": "Strike",
        "Defends": "Defend",
    }
    return special_cases.get(name, name)


def _singular_potion_name(name: str) -> str:
    if name.endswith(" Potions"):
        return f"{name[:-8]} Potion"
    return name


__all__ = [
    "EventTextAmount",
    "EventTextEffect",
    "EventTextEffectKind",
    "normalize_event_option_description",
    "parse_event_option_effects",
]
