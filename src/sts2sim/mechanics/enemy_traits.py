"""Generic enemy behavior trait encoding for learning observations."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

ENEMY_TRAIT_KEYS: tuple[str, ...] = (
    "alive",
    "hp_fraction",
    "block",
    "intent_attack",
    "intent_defend",
    "intent_buff",
    "intent_debuff",
    "intent_escape",
    "intent_sleep",
    "intent_stun",
    "multi_hit",
    "incoming_damage",
    "single_hit_damage",
    "hit_count",
    "current_strength",
    "current_poison",
    "current_weak",
    "current_vulnerable",
    "has_thorns",
    "has_artifact",
    "has_buffer",
    "has_intangible",
    "scaling_strength",
    "scaling_block",
    "scaling_debuff",
    "scaling_summon",
    "scaling_poison",
    "behavior_attack_frequency",
    "behavior_defend_frequency",
    "behavior_buff_frequency",
    "behavior_debuff_frequency",
    "behavior_summon_frequency",
    "behavior_scaling_speed",
    "urgency_score",
    "lethal_to_player",
    "dies_to_poison",
    "retaliates_on_attack",
    "split_or_phase",
    "explode_or_escape",
    "unknown_behavior",
)

ENEMY_TRAIT_AGGREGATE_KEYS: tuple[str, ...] = (
    "enemy_count",
    "alive_count",
    *(f"total_{key}" for key in ENEMY_TRAIT_KEYS),
    *(f"average_{key}" for key in ENEMY_TRAIT_KEYS),
    "max_urgency_score",
    "max_incoming_damage",
)

_ATTACK_TOKENS = (
    "attack",
    "attacking",
    "strike",
    "slash",
    "lacerate",
    "bite",
    "claw",
    "slam",
    "punch",
    "stab",
    "shoot",
    "hit",
    "damage",
)
_DEFEND_TOKENS = ("defend", "defense", "block", "shield", "guard", "armor", "protect")
_BUFF_TOKENS = (
    "buff",
    "power",
    "strength",
    "ritual",
    "rage",
    "enrage",
    "grow",
    "growth",
    "intensity",
    "dexterity",
    "metallicize",
)
_DEBUFF_TOKENS = (
    "debuff",
    "weak",
    "vulnerable",
    "frail",
    "poison",
    "hex",
    "curse",
    "confuse",
    "shackle",
)
_SUMMON_TOKENS = (
    "summon",
    "spawn",
    "minion",
    "backup",
    "fabricate",
    "illusion",
    "egg",
    "bees",
    "respawn",
)
_ESCAPE_TOKENS = ("escape", "flee", "run", "leave")
_SLEEP_TOKENS = ("sleep", "sleeping")
_STUN_TOKENS = ("stun", "stunned")
_SPLIT_TOKENS = ("split", "phase", "transform", "hatch", "metamorph")
_EXPLODE_TOKENS = ("explode", "explosion", "detonate", "blow", "escape", "flee")
_RETALIATE_TOKENS = ("thorns", "sharp_hide", "spikes", "retaliate", "reactive", "shell")
_SCALING_TOKENS = (
    "scale",
    "scaling",
    "ritual",
    "strength",
    "grow",
    "growth",
    "enrage",
    "intensity",
    "inertia",
)

_STATUS_ALIASES: dict[str, tuple[str, ...]] = {
    "strength": ("strength", "str"),
    "poison": ("poison",),
    "weak": ("weak",),
    "vulnerable": ("vulnerable", "vuln"),
    "thorns": ("thorns", "sharp_hide", "spikes", "flame_barrier"),
    "artifact": ("artifact",),
    "buffer": ("buffer",),
    "intangible": ("intangible",),
    "ritual": ("ritual",),
    "metallicize": ("metallicize", "plated_armor"),
}


def enemy_trait_summary(monster: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic reusable behavior traits for one serialized enemy."""

    statuses = _mapping(monster.get("statuses"))
    metadata = _mapping(monster.get("metadata"))
    hp = max(0, _int(monster.get("hp")))
    max_hp = max(0, _int(monster.get("max_hp")))
    block = max(0, _int(monster.get("block")))
    hit_count = max(1, _int(monster.get("hit_count"), 1))
    single_hit_damage = max(0, _int(monster.get("intent_damage")))
    incoming_damage = _incoming_damage(monster, single_hit_damage, hit_count)
    intent = _normalized(monster.get("intent"))
    token_text = _token_text(monster, metadata)

    behavior = _behavior_profile(monster, metadata, token_text)
    intent_attack = incoming_damage > 0 or _has_any(intent, _ATTACK_TOKENS)
    intent_defend = _int(monster.get("intent_block")) > 0 or _has_any(intent, _DEFEND_TOKENS)
    intent_buff = _has_any(intent, _BUFF_TOKENS) or _metadata_has_power(metadata, target="self")
    intent_debuff = _has_any(intent, _DEBUFF_TOKENS) or _metadata_has_power(
        metadata, target="player"
    )
    intent_escape = _has_any(intent, _ESCAPE_TOKENS) or _has_any(token_text, _ESCAPE_TOKENS)
    intent_sleep = _has_any(intent, _SLEEP_TOKENS) or _has_any(token_text, _SLEEP_TOKENS)
    intent_stun = _has_any(intent, _STUN_TOKENS) or _has_any(token_text, _STUN_TOKENS)

    scaling_strength = (
        _status_value(statuses, "ritual") > 0
        or _has_any(token_text, _SCALING_TOKENS)
        or _metadata_power_matches(metadata, ("strength", "ritual"), target="self")
    )
    scaling_block = (
        _status_value(statuses, "metallicize") > 0
        or _has_any(token_text, ("metallicize", "plated_armor", "hardened", "shell"))
        or _metadata_power_matches(metadata, ("metallicize", "plated_armor"), target="self")
    )
    scaling_debuff = _has_any(token_text, ("curse", "hex", "frail", "weak", "vulnerable"))
    scaling_summon = _has_any(token_text, _SUMMON_TOKENS)
    scaling_poison = _has_any(token_text, ("poison", "venom", "toxin"))

    hp_fraction = (hp / max_hp) if max_hp > 0 else 0.0
    dies_to_poison = hp > 0 and _status_value(statuses, "poison") >= hp
    explode_or_escape = _has_any(token_text, _EXPLODE_TOKENS)
    urgency_score = _urgency_score(
        hp_fraction=hp_fraction,
        incoming_damage=incoming_damage,
        scaling_speed=behavior["behavior_scaling_speed"],
        explode_or_escape=explode_or_escape,
        dies_to_poison=dies_to_poison,
    )

    known_behavior = any(
        (
            intent_attack,
            intent_defend,
            intent_buff,
            intent_debuff,
            intent_escape,
            intent_sleep,
            intent_stun,
            behavior["behavior_summon_frequency"] > 0,
            scaling_strength,
            scaling_block,
            scaling_debuff,
            scaling_summon,
            scaling_poison,
            _has_any(token_text, _SPLIT_TOKENS),
            explode_or_escape,
        )
    )

    summary: dict[str, Any] = {
        "monster_id": str(monster.get("monster_id", "")),
        "move_id": str(monster.get("move_id", "")),
        "next_move_id": str(monster.get("next_move_id", "")),
        "alive": 1.0 if hp > 0 else 0.0,
        "hp_fraction": round(_clamp(hp_fraction), 4),
        "block": float(block),
        "intent_attack": _flag(intent_attack),
        "intent_defend": _flag(intent_defend),
        "intent_buff": _flag(intent_buff),
        "intent_debuff": _flag(intent_debuff),
        "intent_escape": _flag(intent_escape),
        "intent_sleep": _flag(intent_sleep),
        "intent_stun": _flag(intent_stun),
        "multi_hit": _flag(incoming_damage > 0 and hit_count > 1),
        "incoming_damage": float(incoming_damage),
        "single_hit_damage": float(single_hit_damage),
        "hit_count": float(hit_count),
        "current_strength": float(_status_value(statuses, "strength")),
        "current_poison": float(_status_value(statuses, "poison")),
        "current_weak": float(_status_value(statuses, "weak")),
        "current_vulnerable": float(_status_value(statuses, "vulnerable")),
        "has_thorns": _flag(
            _status_value(statuses, "thorns") > 0
            or _has_any(token_text, _RETALIATE_TOKENS)
        ),
        "has_artifact": _flag(_status_value(statuses, "artifact") > 0),
        "has_buffer": _flag(_status_value(statuses, "buffer") > 0),
        "has_intangible": _flag(_status_value(statuses, "intangible") > 0),
        "scaling_strength": _flag(scaling_strength),
        "scaling_block": _flag(scaling_block),
        "scaling_debuff": _flag(scaling_debuff),
        "scaling_summon": _flag(scaling_summon),
        "scaling_poison": _flag(scaling_poison),
        **behavior,
        "urgency_score": round(urgency_score, 4),
        "lethal_to_player": _flag(bool(metadata.get("lethal_to_player"))),
        "dies_to_poison": _flag(dies_to_poison),
        "retaliates_on_attack": _flag(
            _status_value(statuses, "thorns") > 0 or _has_any(token_text, _RETALIATE_TOKENS)
        ),
        "split_or_phase": _flag(_has_any(token_text, _SPLIT_TOKENS)),
        "explode_or_escape": _flag(explode_or_escape),
        "unknown_behavior": _flag(not known_behavior),
    }
    return summary


def enemy_trait_vector(summary: Mapping[str, Any]) -> tuple[float, ...]:
    """Return fixed-order numeric enemy trait features."""

    return tuple(_float(summary.get(key)) for key in ENEMY_TRAIT_KEYS)


def enemy_slots_from_payload(
    payload: Mapping[str, Any],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return per-enemy trait summaries from a serialized payload, up to ``limit``."""

    combat = _mapping(payload.get("combat"))
    monsters = _sequence(combat.get("monsters"))
    return [enemy_trait_summary(_mapping(monster)) for monster in monsters[: max(0, limit)]]


def enemy_trait_aggregate(payload: Mapping[str, Any]) -> dict[str, float]:
    """Return total and average trait features for all included enemies."""

    summaries = enemy_slots_from_payload(payload)
    count = len(summaries)
    denominator = max(1, count)
    totals = {
        key: sum(_float(summary.get(key)) for summary in summaries)
        for key in ENEMY_TRAIT_KEYS
    }
    aggregate: dict[str, float] = {
        "enemy_count": float(count),
        "alive_count": totals["alive"],
    }
    aggregate.update((f"total_{key}", value) for key, value in totals.items())
    aggregate.update((f"average_{key}", value / denominator) for key, value in totals.items())
    aggregate["max_urgency_score"] = max(
        (_float(summary.get("urgency_score")) for summary in summaries),
        default=0.0,
    )
    aggregate["max_incoming_damage"] = max(
        (_float(summary.get("incoming_damage")) for summary in summaries),
        default=0.0,
    )
    return aggregate


def enemy_trait_aggregate_vector(payload: Mapping[str, Any]) -> tuple[float, ...]:
    """Return fixed-order numeric aggregate enemy trait features."""

    aggregate = enemy_trait_aggregate(payload)
    return tuple(_float(aggregate.get(key)) for key in ENEMY_TRAIT_AGGREGATE_KEYS)


def _behavior_profile(
    monster: Mapping[str, Any],
    metadata: Mapping[str, Any],
    token_text: str,
) -> dict[str, float]:
    moves = _metadata_moves(metadata)
    if moves:
        denominator = max(1, len(moves))
        attack = sum(1 for move in moves if _move_has(move, _ATTACK_TOKENS))
        defend = sum(1 for move in moves if _move_has(move, _DEFEND_TOKENS))
        buff = sum(1 for move in moves if _move_has(move, _BUFF_TOKENS))
        debuff = sum(1 for move in moves if _move_has(move, _DEBUFF_TOKENS))
        summon = sum(1 for move in moves if _move_has(move, _SUMMON_TOKENS))
        scaling = sum(1 for move in moves if _move_has(move, _SCALING_TOKENS))
        return {
            "behavior_attack_frequency": round(attack / denominator, 4),
            "behavior_defend_frequency": round(defend / denominator, 4),
            "behavior_buff_frequency": round(buff / denominator, 4),
            "behavior_debuff_frequency": round(debuff / denominator, 4),
            "behavior_summon_frequency": round(summon / denominator, 4),
            "behavior_scaling_speed": round(scaling / denominator, 4),
        }

    intent = _normalized(monster.get("intent"))
    incoming_damage = _incoming_damage(
        monster,
        max(0, _int(monster.get("intent_damage"))),
        max(1, _int(monster.get("hit_count"), 1)),
    )
    return {
        "behavior_attack_frequency": _flag(incoming_damage > 0 or _has_any(intent, _ATTACK_TOKENS)),
        "behavior_defend_frequency": _flag(
            _int(monster.get("intent_block")) > 0 or _has_any(intent, _DEFEND_TOKENS)
        ),
        "behavior_buff_frequency": _flag(
            _has_any(intent, _BUFF_TOKENS) or _metadata_has_power(metadata, target="self")
        ),
        "behavior_debuff_frequency": _flag(
            _has_any(intent, _DEBUFF_TOKENS)
            or _metadata_has_power(metadata, target="player")
        ),
        "behavior_summon_frequency": _flag(_has_any(token_text, _SUMMON_TOKENS)),
        "behavior_scaling_speed": _flag(
            _has_any(token_text, _SCALING_TOKENS)
            or _metadata_power_matches(metadata, ("strength", "ritual"), target="self")
        ),
    }


def _metadata_moves(metadata: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    for key in ("moves", "move_profiles", "known_moves", "monster_moves"):
        moves = tuple(_mapping(item) for item in _sequence(metadata.get(key)))
        moves = tuple(move for move in moves if move)
        if moves:
            return moves
    return ()


def _move_has(move: Mapping[str, Any], tokens: Sequence[str]) -> bool:
    text = _flatten_text(
        {
            "id": move.get("id", move.get("move_id")),
            "name": move.get("name"),
            "intent": move.get("intent"),
            "powers": move.get("powers", move.get("move_powers")),
        }
    )
    damage = _mapping(move.get("damage"))
    if tokens is _ATTACK_TOKENS and (
        _int(damage.get("normal")) > 0 or _int(move.get("intent_damage")) > 0
    ):
        return True
    if tokens is _DEFEND_TOKENS and (
        _int(move.get("block")) > 0 or _int(move.get("intent_block")) > 0
    ):
        return True
    return _has_any(text, tokens)


def _incoming_damage(
    monster: Mapping[str, Any],
    single_hit_damage: int,
    hit_count: int,
) -> int:
    metadata = _mapping(monster.get("metadata"))
    if metadata.get("intent_damage_total") is not None:
        return max(0, _int(metadata.get("intent_damage_total")))
    if metadata.get("incoming_damage") is not None:
        return max(0, _int(metadata.get("incoming_damage")))
    return single_hit_damage * hit_count


def _metadata_has_power(metadata: Mapping[str, Any], *, target: str) -> bool:
    return any(
        _metadata_power_targets(power, target=target)
        for power in _metadata_powers(metadata)
    )


def _metadata_power_matches(
    metadata: Mapping[str, Any],
    tokens: Sequence[str],
    *,
    target: str,
) -> bool:
    return any(
        _metadata_power_targets(power, target=target)
        and _has_any(_normalized(power.get("power_id", power.get("status", ""))), tokens)
        for power in _metadata_powers(metadata)
    )


def _metadata_power_targets(power: Mapping[str, Any], *, target: str) -> bool:
    raw_target = _normalized(power.get("target", "self"))
    if target == "player":
        return raw_target in {"player", "enemy", "opponent", "hero"}
    return raw_target in {"", "self", "monster"}


def _metadata_powers(metadata: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    powers: list[Mapping[str, Any]] = []
    for key in ("move_powers", "powers", "applied_powers", "statuses"):
        for raw in _sequence(metadata.get(key)):
            power = _mapping(raw)
            if power:
                powers.append(power)
    return tuple(powers)


def _urgency_score(
    *,
    hp_fraction: float,
    incoming_damage: int,
    scaling_speed: float,
    explode_or_escape: bool,
    dies_to_poison: bool,
) -> float:
    score = min(0.45, incoming_damage / 80.0)
    score += min(0.3, scaling_speed * 0.3)
    if explode_or_escape:
        score += 0.25
    if hp_fraction <= 0.25:
        score += 0.1
    if dies_to_poison:
        score -= 0.2
    return _clamp(score)


def _token_text(monster: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    parts = (
        monster.get("intent"),
        monster.get("move_id"),
        monster.get("next_move_id"),
        _metadata_without_identity(metadata),
    )
    return _flatten_text(parts)


def _metadata_without_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    identity_keys = {"monster_id", "source_monster_id", "target_id"}
    return {
        str(key): value
        for key, value in metadata.items()
        if _normalized(key) not in identity_keys
    }


def _flatten_text(value: object) -> str:
    if isinstance(value, Mapping):
        return " ".join(f"{_normalized(key)} {_flatten_text(item)}" for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(_flatten_text(item) for item in value)
    return _normalized(value)


def _status_value(statuses: Mapping[str, Any], status_name: str) -> int:
    aliases = _STATUS_ALIASES.get(status_name, (status_name,))
    values = [
        _int(value)
        for key, value in statuses.items()
        if _normalized(key) in aliases
    ]
    return sum(values)


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _sequence(value: object) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _int(value: object, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _float(value: object, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _flag(value: bool) -> float:
    return 1.0 if value else 0.0


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _has_any(text: str, tokens: Sequence[str]) -> bool:
    return any(token in text for token in tokens)


def _normalized(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")
