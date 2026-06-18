"""Pure source-row classification for potion combat specials.

The engine currently handles potion use through a small hand-written table.
These helpers provide a source-backed bridge: each potion row is classified into
engine-shaped effect steps that can be executed deterministically, plus explicit
blockers for effects that need choice UI, random pools, or transition control.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .potions import potion_content_id
from .powers import normalize_power_id

PotionSpecialStatus = Literal["executable", "partial", "blocked", "unsupported"]

POTION_EXECUTABLE_EFFECT_KEYS = frozenset(
    {
        "all_damage",
        "apply_status",
        "block",
        "channel_orb",
        "damage",
        "draw",
        "energy",
        "evoke_orb",
        "exhaust_hand",
        "heal",
        "hp_loss",
        "max_hp",
        "next_turn",
        "orb_slot_delta",
        "player_resource",
        "retain_hand",
    }
)


@dataclass(frozen=True, slots=True)
class PotionEffectBlocker:
    """A deliberate blocker for potion behavior that is not a pure effect step."""

    category: str
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", _normalized_id(self.category))
        object.__setattr__(self, "metadata", _clone_jsonish_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class PotionEffectClassification:
    """Executable steps and blockers inferred from one potion source row."""

    potion_id: str
    name: str
    status: PotionSpecialStatus
    executable_steps: tuple[Mapping[str, Any], ...] = ()
    blockers: tuple[PotionEffectBlocker, ...] = ()
    description: str = ""
    rarity: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "potion_id", _normalized_id(self.potion_id))
        object.__setattr__(
            self,
            "executable_steps",
            tuple(_clone_jsonish_mapping(step) for step in self.executable_steps),
        )
        object.__setattr__(self, "blockers", tuple(self.blockers))

    @property
    def is_fully_executable(self) -> bool:
        """Return whether the row has steps and no blockers."""

        return self.status == "executable"


@dataclass(frozen=True, slots=True)
class PotionSpecialCoverageSummary:
    """Aggregate coverage for a collection of source potion rows."""

    total_rows: int
    executable_rows: int
    partial_rows: int
    blocked_rows: int
    unsupported_rows: int
    classifications: tuple[PotionEffectClassification, ...]
    blocker_categories: tuple[str, ...] = ()
    executable_effect_keys: tuple[str, ...] = ()

    @property
    def covered_rows(self) -> int:
        """Rows with at least one executable step or explicit blocker."""

        return self.executable_rows + self.partial_rows + self.blocked_rows


_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "eng"

_CATEGORY_MESSAGES = {
    "card_choice": "Requires choosing one or more cards at potion-use time.",
    "card_duplication": "Requires a card-play hook for duplicating a future card play.",
    "card_generation": "Creates fixed cards and needs generated-card descriptors.",
    "card_play_from_draw_pile": "Requires playing cards from the runtime draw pile.",
    "card_play_modifier": "Requires a future card-play damage/cost modifier hook.",
    "cost_randomization": "Requires randomizing runtime hand card costs.",
    "deck_reorder": "Requires moving or shuffling cards across runtime piles.",
    "escape_combat": "Requires ending or leaving combat through transition logic.",
    "hand_upgrade": "Requires upgrading runtime cards in hand for the combat.",
    "passive_trigger": "Triggers passively and is not a manual potion-use effect.",
    "potion_generation": "Requires random potion generation and open-slot handling.",
    "random_card_choice": "Requires a random card pool and choice UI.",
    "random_card_generation": "Requires random card generation from a card pool.",
    "runtime_block_multiplier": "Requires current block to calculate a block multiplier.",
    "runtime_hp_percent": "Requires runtime max HP to calculate the exact amount.",
    "summon_or_channel": "Requires summon/orb-channel mechanics outside potion steps.",
    "temporary_enemy_debuff": "Requires temporary enemy debuff expiry semantics.",
    "timed_damage_modifier": "Requires timed damage-modifier semantics.",
    "timed_turn_start_effect": "Requires repeated start-of-turn effect semantics.",
    "unparsed_text": "No deterministic potion effect step could be inferred.",
}

_SELF_STATUS_NAMES = {
    "artifact",
    "buffer",
    "dexterity",
    "focus",
    "intangible",
    "plated_armor",
    "plating",
    "regen",
    "ritual",
    "strength",
    "thorns",
}

_ENEMY_STATUS_NAMES = {
    "doom",
    "poison",
    "vulnerable",
    "weak",
}

_RESOURCE_NAMES = {
    "forge",
    "star",
    "summon",
}

_MANUAL_USE_BLOCKER_IDS = {
    "fairy_in_a_bottle": "passive_trigger",
}

_ESCAPE_POTION_IDS = {
    "smoke_bomb",
    "smoke_bomb_potion",
    "smoke_potion",
}


def classify_potion_effect(potion: Mapping[str, Any] | str) -> PotionEffectClassification:
    """Classify a potion row into executable steps plus explicit blockers."""

    row = _row_from_potion(potion)
    potion_id = potion_content_id(row)
    name = str(row.get("name", potion_id))
    description = str(row.get("description", row.get("description_raw", "")) or "")
    plain = _plain_text(description)
    normalized = _normalized_text(plain)

    steps = list(_direct_steps_from_description(description, plain))
    blockers = list(_blockers_from_description(potion_id, name, normalized))

    if not steps and not blockers:
        blockers.append(_blocker("unparsed_text", potion_id=potion_id))

    if steps and blockers:
        status: PotionSpecialStatus = "partial"
    elif steps:
        status = "executable"
    elif any(blocker.category == "unparsed_text" for blocker in blockers):
        status = "unsupported"
    else:
        status = "blocked"

    return PotionEffectClassification(
        potion_id=potion_id,
        name=name,
        status=status,
        executable_steps=tuple(steps),
        blockers=tuple(blockers),
        description=description,
        rarity=str(row["rarity"]) if row.get("rarity") is not None else None,
    )


def classify_potion_effects(
    potions: Sequence[Mapping[str, Any] | str],
) -> tuple[PotionEffectClassification, ...]:
    """Classify a sequence of potion rows."""

    return tuple(classify_potion_effect(potion) for potion in potions)


def potion_special_coverage(
    potions: Sequence[Mapping[str, Any] | str] | None = None,
) -> PotionSpecialCoverageSummary:
    """Return coverage counts and categories for source potion rows."""

    rows = cached_potion_source_rows() if potions is None else tuple(potions)
    classifications = classify_potion_effects(rows)
    statuses = [classification.status for classification in classifications]
    blocker_categories = sorted(
        {
            blocker.category
            for classification in classifications
            for blocker in classification.blockers
        }
    )
    executable_effect_keys = sorted(
        {
            str(key)
            for classification in classifications
            for step in classification.executable_steps
            for key in step
            if key in POTION_EXECUTABLE_EFFECT_KEYS
        }
    )
    return PotionSpecialCoverageSummary(
        total_rows=len(classifications),
        executable_rows=statuses.count("executable"),
        partial_rows=statuses.count("partial"),
        blocked_rows=statuses.count("blocked"),
        unsupported_rows=statuses.count("unsupported"),
        classifications=classifications,
        blocker_categories=tuple(blocker_categories),
        executable_effect_keys=tuple(executable_effect_keys),
    )


def cached_potion_source_rows() -> tuple[Mapping[str, Any], ...]:
    """Return cached English potion rows if the data cache is available."""

    path = _CACHE_DIR / "potions.json"
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in payload if isinstance(item, Mapping))


def _row_from_potion(potion: Mapping[str, Any] | str) -> Mapping[str, Any]:
    if isinstance(potion, str):
        return {"id": potion}
    return potion


def _direct_steps_from_description(
    description: str,
    plain: str,
) -> tuple[Mapping[str, Any], ...]:
    normalized = _normalized_text(plain)
    steps: list[Mapping[str, Any]] = []
    steps.extend(_damage_steps(plain, normalized))
    steps.extend(_block_steps(plain, normalized))
    steps.extend(_energy_steps(description, normalized))
    steps.extend(_draw_steps(plain, normalized))
    steps.extend(_resource_steps(description, plain))
    steps.extend(_max_hp_steps(normalized))
    steps.extend(_orb_steps(normalized))
    steps.extend(_status_steps(plain, normalized))
    steps.extend(_hand_steps(normalized))
    return tuple(_dedupe_steps(steps))


def _damage_steps(plain: str, normalized: str) -> tuple[Mapping[str, Any], ...]:
    steps: list[Mapping[str, Any]] = []
    match = re.search(r"\bdeal\s+(\d+)\s+damage\b", normalized)
    if match is None:
        return ()
    amount = int(match.group(1))
    if "all players and enemies" in normalized:
        steps.append({"hp_loss": amount})
        steps.append({"all_damage": amount})
    elif "all enemies" in normalized:
        steps.append({"all_damage": amount})
    else:
        steps.append({"damage": amount})
    return tuple(steps)


def _block_steps(plain: str, normalized: str) -> tuple[Mapping[str, Any], ...]:
    steps: list[Mapping[str, Any]] = []
    for match in re.finditer(r"\bgain\s+(\d+)\s+block\b", normalized):
        amount = int(match.group(1))
        prefix = normalized[max(0, match.start() - 24) : match.start()]
        if "next turn" in prefix:
            steps.append({"next_turn": {"block": amount}})
        else:
            steps.append({"block": amount})
    return tuple(steps)


def _energy_steps(description: str, normalized: str) -> tuple[Mapping[str, Any], ...]:
    steps: list[Mapping[str, Any]] = []
    for match in re.finditer(r"\[energy:(\d+)\]", description, flags=re.IGNORECASE):
        amount = int(match.group(1))
        plain_prefix = _normalized_text(_plain_text(description[: match.start()]))
        if "start of your next" in plain_prefix[-48:]:
            continue
        steps.append({"energy": amount})
    return tuple(steps)


def _draw_steps(plain: str, normalized: str) -> tuple[Mapping[str, Any], ...]:
    steps: list[Mapping[str, Any]] = []
    for match in re.finditer(r"\bdraw\s+(\d+)\s+cards?\b", normalized):
        prefix = normalized[max(0, match.start() - 48) : match.start()]
        if "start of your next" in prefix:
            continue
        steps.append({"draw": int(match.group(1))})
    return tuple(steps)


def _resource_steps(description: str, plain: str) -> tuple[Mapping[str, Any], ...]:
    steps: list[Mapping[str, Any]] = []
    for resource_name, amount in re.findall(
        r"\[(star):(\d+)\]",
        description,
        flags=re.IGNORECASE,
    ):
        steps.append(
            {"player_resource": {"resource": _normalized_id(resource_name), "amount": int(amount)}}
        )

    normalized = _normalized_text(plain)
    for resource_name in ("forge", "summon"):
        for amount in re.findall(rf"\b{resource_name}\s+(\d+)\b", normalized):
            steps.append({"player_resource": {"resource": resource_name, "amount": int(amount)}})
    return tuple(steps)


def _max_hp_steps(normalized: str) -> tuple[Mapping[str, Any], ...]:
    match = re.search(r"\bgain\s+(\d+)\s+max hp\b", normalized)
    if match is None:
        return ()
    return ({"max_hp": int(match.group(1))},)


def _orb_steps(normalized: str) -> tuple[Mapping[str, Any], ...]:
    steps: list[Mapping[str, Any]] = []

    for match in re.finditer(r"\bgain\s+(\d+)\s+orb slots?\b", normalized):
        steps.append({"orb_slot_delta": int(match.group(1))})

    for match in re.finditer(
        r"\bchannel\s+(?:(\d+)|a|an)?\s*(lightning|frost|dark|plasma|glass)\b",
        normalized,
    ):
        amount: int | str = int(match.group(1) or 1)
        if "for each of your orb slots" in normalized:
            amount = "orb_slots"
        steps.append({"channel_orb": {"orb": match.group(2), "amount": amount}})

    return tuple(steps)


def _status_steps(plain: str, normalized: str) -> tuple[Mapping[str, Any], ...]:
    statuses: dict[str, dict[str, int]] = {}
    for match in re.finditer(
        r"\b(\d+)\s+(artifact|buffer|dexterity|doom|focus|intangible|"
        r"orb slots?|plated armor|plating|poison|regen|ritual|strength|"
        r"thorns|vulnerable|weak)\b",
        normalized,
    ):
        amount = int(match.group(1))
        status = _status_id(match.group(2))
        if status == "orb_slots":
            continue
        context = normalized[max(0, match.start() - 64) : match.end() + 64]
        prefix = normalized[max(0, match.start() - 24) : match.start()]
        is_loss_context = "lose" in prefix or "loses" in prefix
        if status in _RESOURCE_NAMES:
            continue
        if "enemy loses" in context or "all enemies lose" in context:
            target = "all_enemies" if "all enemies" in context else "enemy"
            _add_status(statuses, target, status, -amount)
            if "this turn" in context:
                suffix = "up" if amount > 0 else "down"
                _add_status(statuses, target, f"{status}_{suffix}", amount)
            continue
        if _is_temporary_self_status(status, amount, normalized) and not is_loss_context:
            _add_status(statuses, "self", status, amount)
            _add_status(statuses, "self", f"{status}_down", amount)
            continue
        if "end of your turn" in context and is_loss_context:
            continue
        if _enemy_status_target(status, context, normalized) is not None:
            target = _enemy_status_target(status, context, normalized) or "enemy"
            _add_status(statuses, target, status, amount)
            continue
        if status in _SELF_STATUS_NAMES or "gain" in context:
            _add_status(statuses, "self", status, amount)

    return tuple(
        {"apply_status": {"target": target, **payload}}
        for target, payload in statuses.items()
        if payload
    )


def _hand_steps(normalized: str) -> tuple[Mapping[str, Any], ...]:
    steps: list[Mapping[str, Any]] = []
    if "exhaust your hand" in normalized:
        steps.append({"exhaust_hand": {"mode": "all"}})
    if "retain your hand" in normalized:
        duration_match = re.search(r"retain your hand for (\d+) turns?", normalized)
        if duration_match:
            steps.append(
                {
                    "apply_status": {
                        "target": "self",
                        "retain_hand": int(duration_match.group(1)),
                    }
                }
            )
        else:
            steps.append({"retain_hand": True})
    return tuple(steps)


def _blockers_from_description(
    potion_id: str,
    name: str,
    normalized: str,
) -> tuple[PotionEffectBlocker, ...]:
    blockers: list[PotionEffectBlocker] = []
    blocker_id_category = _MANUAL_USE_BLOCKER_IDS.get(potion_id)
    if blocker_id_category is not None:
        blockers.append(_blocker(blocker_id_category, potion_id=potion_id))
    if potion_id in _ESCAPE_POTION_IDS or _looks_like_escape(name, normalized):
        blockers.append(_blocker("escape_combat", potion_id=potion_id))

    if "choose" in normalized and "random" in normalized and "card" in normalized:
        blockers.append(_blocker("random_card_choice", potion_id=potion_id))
    elif "randomize" not in normalized and "random" in normalized and (
        "card" in normalized or _mentions_random_card_types(normalized)
    ):
        blockers.append(_blocker("random_card_generation", potion_id=potion_id))

    if "random potion" in normalized or "empty potion slots" in normalized:
        blockers.append(_blocker("potion_generation", potion_id=potion_id))
    if "next card is played an extra time" in normalized:
        blockers.append(_blocker("card_duplication", potion_id=potion_id))
    if "next attack you play deals triple damage" in normalized:
        blockers.append(_blocker("card_play_modifier", potion_id=potion_id))
    if "play the top" in normalized and "draw pile" in normalized:
        blockers.append(_blocker("card_play_from_draw_pile", potion_id=potion_id))
    if "randomize the cost" in normalized or "randomize costs" in normalized:
        blockers.append(_blocker("cost_randomization", potion_id=potion_id))
    if "shuffle all your cards" in normalized or (
        "draw pile" in normalized and "choose a card" in normalized
    ):
        blockers.append(_blocker("deck_reorder", potion_id=potion_id))
    if _requires_card_choice(normalized):
        blockers.append(_blocker("card_choice", potion_id=potion_id))
    if "upgrade all cards in your hand" in normalized:
        blockers.append(_blocker("hand_upgrade", potion_id=potion_id))
    if "free to play this combat" in normalized:
        blockers.append(_blocker("card_play_modifier", potion_id=potion_id))
    if "replay" in normalized and "this combat" in normalized:
        blockers.append(_blocker("card_play_modifier", potion_id=potion_id))
    if "triple your block" in normalized:
        blockers.append(_blocker("runtime_block_multiplier", potion_id=potion_id))
    if "heal for" in normalized and "max hp" in normalized and "%" in normalized:
        blockers.append(_blocker("runtime_hp_percent", potion_id=potion_id))
    if "channel" in normalized and not _has_executable_channel_orb(normalized):
        blockers.append(_blocker("summon_or_channel", potion_id=potion_id))
    if (
        "orb slots" in normalized
        and "gain" not in normalized
        and "for each of your orb slots" not in normalized
    ):
        blockers.append(_blocker("summon_or_channel", potion_id=potion_id))
    if "start of your next" in normalized:
        blockers.append(_blocker("timed_turn_start_effect", potion_id=potion_id))
    if "attacks deal" in normalized and "less damage" in normalized:
        blockers.append(_blocker("timed_damage_modifier", potion_id=potion_id))
    if "loses" in normalized and "hp at the end" in normalized:
        blockers.append(_blocker("timed_damage_modifier", potion_id=potion_id))
    if "all enemies lose" in normalized and "this turn" in normalized:
        blockers.append(_blocker("temporary_enemy_debuff", potion_id=potion_id))
    if (
        "add" in normalized
        and _mentions_fixed_cards(normalized)
        and not ("choose" in normalized and "random" in normalized)
    ):
        blockers.append(_blocker("card_generation", potion_id=potion_id))

    return tuple(_dedupe_blockers(blockers))


def _requires_card_choice(normalized: str) -> bool:
    return any(
        phrase in normalized
        for phrase in (
            "any number of cards",
            "choose a card",
            "discard any number of cards",
            "exhaust any number of cards",
            "put a card from",
        )
    )


def _mentions_fixed_cards(normalized: str) -> bool:
    return any(
        phrase in normalized
        for phrase in (
            "colorless cards",
            "shivs",
            "souls",
        )
    )


def _mentions_random_card_types(normalized: str) -> bool:
    return any(
        phrase in normalized
        for phrase in (
            "random attack",
            "random skill",
            "random power",
            "attack, skill, and power",
        )
    )


def _has_executable_channel_orb(normalized: str) -> bool:
    return (
        re.search(
            r"\bchannel\s+(?:(\d+)|a|an)?\s*(lightning|frost|dark|plasma|glass)\b",
            normalized,
        )
        is not None
    )


def _looks_like_escape(name: str, normalized: str) -> bool:
    key = _normalized_id(name)
    return (
        "smoke" in key
        and any(word in normalized for word in ("escape", "flee", "leave combat"))
    ) or "escape combat" in normalized


def _enemy_status_target(
    status: str,
    context: str,
    normalized: str,
) -> str | None:
    if status not in _ENEMY_STATUS_NAMES:
        return None
    if "all enemies" in context or "all enemies" in normalized and "apply" in normalized:
        return "all_enemies"
    if "apply" in context or normalized.startswith("apply "):
        return "enemy"
    return None


def _is_temporary_self_status(status: str, amount: int, normalized: str) -> bool:
    return (
        status in {"strength", "dexterity"}
        and f"gain {amount} {status}" in normalized
        and f"lose {amount} {status}" in normalized
        and "end of your turn" in normalized
    )


def _status_id(value: str) -> str:
    normalized = _normalized_id(value)
    if normalized in {"orb_slot", "orb_slots"}:
        return "orb_slots"
    if normalized == "plating":
        return "plated_armor"
    return normalize_power_id(normalized)


def _add_status(
    statuses: dict[str, dict[str, int]],
    target: str,
    status: str,
    amount: int,
) -> None:
    payload = statuses.setdefault(target, {})
    payload[status] = payload.get(status, 0) + amount


def _blocker(category: str, **metadata: Any) -> PotionEffectBlocker:
    normalized = _normalized_id(category)
    return PotionEffectBlocker(
        category=normalized,
        reason=_CATEGORY_MESSAGES.get(normalized, normalized),
        metadata={key: value for key, value in metadata.items() if value is not None},
    )


def _dedupe_steps(steps: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    deduped: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for step in steps:
        key = json.dumps(step, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(step)
    return tuple(deduped)


def _dedupe_blockers(blockers: Sequence[PotionEffectBlocker]) -> tuple[PotionEffectBlocker, ...]:
    deduped: list[PotionEffectBlocker] = []
    seen: set[tuple[str, str]] = set()
    for blocker in blockers:
        key = (blocker.category, json.dumps(blocker.metadata, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(blocker)
    return tuple(deduped)


def _plain_text(description: str) -> str:
    without_color_tags = re.sub(r"\[/?(?:blue|gold|green|red|white)]", "", description)
    return " ".join(without_color_tags.replace("\n", " ").split())


def _normalized_text(text: str) -> str:
    return " ".join(text.lower().split())


def _normalized_id(value: object) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("'", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _clone_jsonish_mapping(source: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _clone_jsonish(value) for key, value in source.items()}


def _clone_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _clone_jsonish_mapping(value)
    if isinstance(value, tuple):
        return tuple(_clone_jsonish(item) for item in value)
    if isinstance(value, list):
        return [_clone_jsonish(item) for item in value]
    return value


__all__ = [
    "POTION_EXECUTABLE_EFFECT_KEYS",
    "PotionEffectBlocker",
    "PotionEffectClassification",
    "PotionSpecialCoverageSummary",
    "PotionSpecialStatus",
    "cached_potion_source_rows",
    "classify_potion_effect",
    "classify_potion_effects",
    "potion_special_coverage",
]
