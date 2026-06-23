"""Aggression and HP-spend signals shared by planners and learning encoders."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

AGGRESSION_OBSERVATION_KEYS: tuple[str, ...] = (
    "target",
    "hp_floor",
    "hp_spend_budget",
    "block_priority",
    "combat_pace_pressure",
    "allow_chip_damage",
    "scaling_pressure",
    "enemy_attack_pressure",
    "elite_pressure",
    "future_elite_count",
    "future_rest_count",
    "nearest_elite_distance",
    "nearest_rest_distance",
    "boss_distance",
    "known_elite",
    "unknown_elite_count",
)


def aggression_summary(state_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact aggression inputs derived from the visible run state."""

    player = _active_player(state_summary)
    hp = _int(player.get("hp"))
    max_hp = max(1, _int(player.get("max_hp"), 1))
    hp_ratio = _clamp(hp / max_hp)
    combat = _mapping(state_summary.get("combat"))
    incoming_damage = _incoming_damage(combat)
    scaling_pressure = _scaling_pressure(combat)
    enemy_attack_pressure = _clamp(incoming_damage / max(1, hp))
    relic_count = len(_sequence(state_summary.get("relics")))
    potion_count = len(_sequence(state_summary.get("potions")))
    route = _route_summary(state_summary)
    elites = _elite_summary(state_summary)

    rest_soon_bonus = 0.0
    nearest_rest = _int(route.get("nearest_rest_distance"), 0)
    if nearest_rest:
        rest_soon_bonus = max(0.0, 0.12 - nearest_rest * 0.03)

    elite_pressure = _clamp(
        _scaled(_int(route.get("future_elite_count")), 4)
        + (0.18 if _int(route.get("nearest_elite_distance")) in {1, 2} else 0.0)
        + (0.08 if elites.get("known_elite_id") else 0.0)
    )

    target = 0.5
    if hp_ratio >= 0.85:
        target += 0.12
    elif hp_ratio <= 0.35:
        target -= 0.32
    elif hp_ratio <= 0.55:
        target -= 0.18
    target += min(0.18, relic_count / 40)
    target += 0.07 if potion_count else 0.0
    target += scaling_pressure * 0.24
    target += rest_soon_bonus
    target += elite_pressure * 0.12 if hp_ratio >= 0.62 else -elite_pressure * 0.10
    target -= enemy_attack_pressure * 0.22
    target = _clamp(target)

    hp_floor = 0.92 - target * 0.54
    hp_floor += enemy_attack_pressure * 0.16
    hp_floor += elite_pressure * 0.08
    hp_floor -= rest_soon_bonus
    hp_floor = _clamp(hp_floor, minimum=0.25, maximum=0.95)
    hp_spend_budget = max(0, int(max_hp * max(0.0, hp_ratio - hp_floor)))

    block_priority = _clamp(
        (1.0 - target) * 0.75
        + enemy_attack_pressure * 0.42
        + (0.20 if hp_ratio < hp_floor else 0.0)
    )
    combat_pace_pressure = _clamp(scaling_pressure * 0.75 + target * 0.25)
    combat_pace = "balanced"
    if combat_pace_pressure >= 0.62:
        combat_pace = "rush"
    elif target <= 0.28 and scaling_pressure < 0.25:
        combat_pace = "stall"

    return {
        "target": round(target, 4),
        "hp_floor": round(hp_floor, 4),
        "hp_spend_budget": hp_spend_budget,
        "block_priority": round(block_priority, 4),
        "combat_pace": combat_pace,
        "combat_pace_pressure": round(combat_pace_pressure, 4),
        "allow_chip_damage": hp_spend_budget > 0 and target >= 0.35,
        "scaling_pressure": round(scaling_pressure, 4),
        "enemy_attack_pressure": round(enemy_attack_pressure, 4),
        "elite_pressure": round(elite_pressure, 4),
        "future_elite_count": _int(route.get("future_elite_count")),
        "future_rest_count": _int(route.get("future_rest_count")),
        "nearest_elite_distance": _int(route.get("nearest_elite_distance")),
        "nearest_rest_distance": _int(route.get("nearest_rest_distance")),
        "boss_distance": _int(route.get("boss_distance")),
        "known_elite_id": elites.get("known_elite_id"),
        "possible_elite_ids": tuple(elites.get("possible_elite_ids", ())),
        "unknown_elite_count": _int(elites.get("unknown_elite_count")),
    }


def aggression_vector(summary: Mapping[str, Any]) -> list[float]:
    """Return fixed-length normalized aggression features."""

    return [
        _clamp(_float(summary.get("target"))),
        _clamp(_float(summary.get("hp_floor"))),
        _scaled(_int(summary.get("hp_spend_budget")), 120),
        _clamp(_float(summary.get("block_priority"))),
        _clamp(_float(summary.get("combat_pace_pressure"))),
        1.0 if summary.get("allow_chip_damage") else 0.0,
        _clamp(_float(summary.get("scaling_pressure"))),
        _clamp(_float(summary.get("enemy_attack_pressure"))),
        _clamp(_float(summary.get("elite_pressure"))),
        _scaled(_int(summary.get("future_elite_count")), 5),
        _scaled(_int(summary.get("future_rest_count")), 5),
        _distance_feature(summary.get("nearest_elite_distance")),
        _distance_feature(summary.get("nearest_rest_distance")),
        _distance_feature(summary.get("boss_distance")),
        1.0 if summary.get("known_elite_id") else 0.0,
        _scaled(_int(summary.get("unknown_elite_count")), 4),
    ]


def _active_player(payload: Mapping[str, Any]) -> dict[str, Any]:
    combat_player = _mapping(_mapping(payload.get("combat")).get("player"))
    if combat_player:
        return combat_player
    return _mapping(payload.get("player"))


def _incoming_damage(combat: Mapping[str, Any]) -> int:
    return sum(
        _int(_mapping(monster).get("intent_damage"))
        for monster in _sequence(combat.get("monsters"))
        if _int(_mapping(monster).get("hp")) > 0
    )


def _scaling_pressure(combat: Mapping[str, Any]) -> float:
    pressure = 0.0
    for raw_monster in _sequence(combat.get("monsters")):
        monster = _mapping(raw_monster)
        if _int(monster.get("hp")) <= 0:
            continue
        intent = _normalized(monster.get("intent"))
        move_id = _normalized(monster.get("move_id"))
        next_move_id = _normalized(monster.get("next_move_id"))
        statuses = _mapping(monster.get("statuses"))
        metadata = _mapping(monster.get("metadata"))

        for token in (intent, move_id, next_move_id):
            if "buff" in token:
                pressure += 0.22
            if "debuff" in token:
                pressure += 0.10
            if any(term in token for term in ("split", "summon", "grow", "strength", "ritual")):
                pressure += 0.18
        for key, value in statuses.items():
            status = _normalized(key)
            amount = abs(_int(value))
            if status in {"ritual", "strength", "metallicize", "plated_armor", "regen"}:
                pressure += min(0.22, amount / 20)
            elif status in {"thorns", "buffer", "intangible", "mode_shift", "asleep"}:
                pressure += min(0.16, amount / 25)
        for raw_power in _sequence(metadata.get("move_powers")):
            power = _mapping(raw_power)
            power_id = _normalized(power.get("power_id"))
            if power_id:
                pressure += 0.12
    return _clamp(pressure)


def _route_summary(payload: Mapping[str, Any]) -> dict[str, int]:
    game_map = _mapping(payload.get("map"))
    nodes = {
        str(_mapping(raw_node).get("node_id")): _mapping(raw_node)
        for raw_node in _sequence(game_map.get("nodes"))
    }
    current_id = str(game_map.get("current_node_id") or "")
    if not nodes or current_id not in nodes:
        return {
            "future_elite_count": 0,
            "future_rest_count": 0,
            "nearest_elite_distance": 0,
            "nearest_rest_distance": 0,
            "boss_distance": 0,
        }
    completed = {str(item) for item in _sequence(game_map.get("completed_node_ids"))}
    outgoing: dict[str, list[str]] = {}
    for raw_edge in _sequence(game_map.get("edges")):
        edge = _mapping(raw_edge)
        outgoing.setdefault(str(edge.get("from_id")), []).append(str(edge.get("to_id")))

    visited = {current_id}
    queue: deque[tuple[str, int]] = deque((node_id, 1) for node_id in outgoing.get(current_id, ()))
    future_elites: set[str] = set()
    future_rests: set[str] = set()
    nearest_elite = 0
    nearest_rest = 0
    boss_distance = 0
    while queue:
        node_id, distance = queue.popleft()
        if node_id in visited or node_id in completed:
            continue
        visited.add(node_id)
        node = nodes.get(node_id, {})
        kind = _normalized(node.get("kind"))
        if kind == "elite":
            future_elites.add(node_id)
            nearest_elite = nearest_elite or distance
        elif kind == "rest":
            future_rests.add(node_id)
            nearest_rest = nearest_rest or distance
        elif kind == "boss":
            boss_distance = boss_distance or distance
        for next_node_id in outgoing.get(node_id, ()):
            if next_node_id not in visited:
                queue.append((next_node_id, distance + 1))
    return {
        "future_elite_count": len(future_elites),
        "future_rest_count": len(future_rests),
        "nearest_elite_distance": nearest_elite,
        "nearest_rest_distance": nearest_rest,
        "boss_distance": boss_distance,
    }


def _elite_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    flags = _mapping(payload.get("flags"))
    act = _int(payload.get("act"), 1)
    possible = _ids_from_flag_keys(
        flags,
        (
            "possible_elite_ids",
            "elite_ids",
            "elite_pool",
            "elite_encounter_ids",
            "act_elite_pool",
            f"act_{act}_elite_pool",
            f"act_{act}_elite_ids",
        ),
    )
    seen = set(
        _ids_from_flag_keys(
            flags,
            (
                "seen_elite_ids",
                "defeated_elite_ids",
                "act_elites_seen",
                f"act_{act}_seen_elite_ids",
                f"act_{act}_defeated_elite_ids",
            ),
        )
    )
    combat = _mapping(payload.get("combat"))
    combat_ids = _combat_source_ids(combat)
    if _normalized(payload.get("phase")) == "combat" and combat_ids:
        seen.update(combat_ids)
    remaining = tuple(item for item in possible if item not in seen)
    known = remaining[0] if len(remaining) == 1 else None
    return {
        "known_elite_id": known,
        "possible_elite_ids": remaining,
        "unknown_elite_count": len(remaining),
    }


def _ids_from_flag_keys(flags: Mapping[str, Any], keys: Iterable[str]) -> tuple[str, ...]:
    ids: list[str] = []
    for key in keys:
        ids.extend(_ids_from_value(flags.get(key)))
    return tuple(dict.fromkeys(item for item in ids if item))


def _ids_from_value(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        mapped = _mapping(value)
        for key in ("id", "monster_id", "encounter_id", "source_monster_id"):
            if mapped.get(key) is not None:
                return (_normalized(mapped.get(key)),)
        return tuple(_normalized(key) for key in mapped if _normalized(key))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        ids: list[str] = []
        for item in value:
            ids.extend(_ids_from_value(item))
        return tuple(ids)
    raw = _normalized(value)
    if raw in {"", "normal", "monster", "elite", "boss", "event"}:
        return ()
    return (raw,)


def _combat_source_ids(combat: Mapping[str, Any]) -> tuple[str, ...]:
    ids: list[str] = []
    for raw_monster in _sequence(combat.get("monsters")):
        monster = _mapping(raw_monster)
        metadata = _mapping(monster.get("metadata"))
        source_id = _normalized(metadata.get("source_monster_id", monster.get("monster_id")))
        if source_id:
            ids.append(source_id)
        encounter_id = _normalized(metadata.get("encounter_id"))
        if encounter_id:
            ids.append(encounter_id)
    return tuple(dict.fromkeys(ids))


def _distance_feature(value: object) -> float:
    distance = _int(value)
    if distance <= 0:
        return 0.0
    return _clamp(1.0 / distance)


def _scaled(value: int, maximum: int) -> float:
    if maximum <= 0:
        return 0.0
    return _clamp(value / maximum)


def _clamp(value: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _sequence(value: object) -> tuple[object, ...]:
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
            return int(value)
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


def _normalized(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
