"""Outcome-based reward functions for self-learning agents.

The rewarder deliberately avoids direct advice like "this card is good" or
"always take this relic".  It rewards measurable outcomes instead: survival,
combat progress, useful gold, node progression, and combat victories.  Cards,
relics, removals, and skips earn value only through later outcomes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sts2sim.api import serialize


class LearningRewardConfig(BaseModel):
    """Configurable reward shaping weights for self-learning runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_penalty: float = 0.0
    floor_reward: float = 0.0
    act_reward: float = 0.0
    win_reward: float = 0.0
    death_penalty: float = 0.0

    node_progress_reward: float = 0.10
    normal_combat_win_reward: float = 2.0
    elite_combat_win_reward: float = 10.0
    act1_boss_reward: float = 50.0
    act2_boss_reward: float = 150.0
    act3_boss_reward: float = 400.0

    hp_loss_safe_weight: float = 0.09
    hp_loss_aggressive_weight: float = 0.01
    enemy_damage_safe_weight: float = 0.01
    enemy_damage_aggressive_weight: float = 0.09
    prevented_hp_safe_weight: float = 0.05
    prevented_hp_aggressive_weight: float = 0.015
    prevented_hp_cap: float = 1.5

    pace_turn_cutoff: int = 10
    normal_pace_cap: float = 1.0
    elite_pace_cap: float = 3.0
    boss_pace_cap: float = 6.0

    gold_reward_per_gold: float = 0.01
    gold_reward_cap: float = 0.5
    wasteful_potion_discard_penalty: float = -0.5


class LearningRewardBreakdown(BaseModel):
    """Explainable component rewards for one transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total: float = 0.0
    node_progress_reward: float = 0.0
    combat_win_reward: float = 0.0
    boss_reward: float = 0.0
    combat_pace_reward: float = 0.0
    aggression_pressure: float = 0.5
    hp_loss_penalty: float = 0.0
    enemy_hp_progress_reward: float = 0.0
    prevented_hp_reward: float = 0.0
    gold_reward: float = 0.0
    potion_waste_penalty: float = 0.0
    target_reached_reward: float = 0.0


class LearningRewardTracker(BaseModel):
    """Mutable run-scoped reward bookkeeping."""

    model_config = ConfigDict(extra="forbid")

    rewarded_node_ids: set[str] = Field(default_factory=set)
    active_combat_id: str | None = None
    combat_start_turn: int = 1
    combat_room_kind: str = ""
    max_enemy_hp_progress: dict[str, int] = Field(default_factory=dict)
    previous_player_hp: int | None = None
    previous_projected_damage: int = 0
    potion_discard_context: dict[str, Any] = Field(default_factory=dict)
    component_totals: dict[str, float] = Field(default_factory=dict)

    def reset(self) -> None:
        """Clear all run-scoped reward state."""

        self.rewarded_node_ids.clear()
        self.active_combat_id = None
        self.combat_start_turn = 1
        self.combat_room_kind = ""
        self.max_enemy_hp_progress.clear()
        self.previous_player_hp = None
        self.previous_projected_damage = 0
        self.potion_discard_context.clear()
        self.component_totals.clear()

    def record(self, breakdown: LearningRewardBreakdown) -> None:
        """Accumulate component totals for reporting."""

        for key, value in breakdown.model_dump(mode="json").items():
            if key == "aggression_pressure":
                continue
            self.component_totals[key] = self.component_totals.get(key, 0.0) + _float(value)


DEFAULT_REWARD_CONFIG = LearningRewardConfig()
BREAKDOWN_FIELDS: tuple[str, ...] = tuple(LearningRewardBreakdown.model_fields)


def learning_reward(
    previous_state: Any,
    next_state: Any,
    config: LearningRewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    """Return the outcome-based reward total for one simulator transition."""

    return learning_reward_breakdown(previous_state, next_state, config=config).total


def learning_reward_breakdown(
    previous_state: Any,
    next_state: Any,
    *,
    tracker: LearningRewardTracker | None = None,
    config: LearningRewardConfig = DEFAULT_REWARD_CONFIG,
    action_descriptor: Mapping[str, Any] | None = None,
    target_reached_reward: float = 0.0,
) -> LearningRewardBreakdown:
    """Return an explainable reward breakdown for one transition."""

    previous = _payload(previous_state)
    current = _payload(next_state)
    tracker = tracker or LearningRewardTracker()
    descriptor = _mapping(action_descriptor)
    aggression_pressure = _combat_aggression_pressure(previous, descriptor)
    hp_loss = _hp_loss(previous, current)
    hp_loss_weight = _lerp(
        config.hp_loss_safe_weight,
        config.hp_loss_aggressive_weight,
        aggression_pressure,
    )
    enemy_damage_weight = _lerp(
        config.enemy_damage_safe_weight,
        config.enemy_damage_aggressive_weight,
        aggression_pressure,
    )
    prevented_hp_weight = _lerp(
        config.prevented_hp_safe_weight,
        config.prevented_hp_aggressive_weight,
        aggression_pressure,
    )

    node_progress = _node_progress_reward(previous, current, tracker, config)
    combat_win, boss_reward, pace_reward = _combat_outcome_rewards(
        previous,
        current,
        tracker,
        config,
    )
    enemy_progress = _enemy_hp_progress(previous, current, tracker)
    prevented_hp = _prevented_hp(previous, current, descriptor)
    gold = max(0, _player_gold(current) - _player_gold(previous))
    gold_reward = min(gold * config.gold_reward_per_gold, config.gold_reward_cap)
    potion_penalty = _potion_waste_penalty(previous, descriptor, config)

    breakdown = LearningRewardBreakdown(
        node_progress_reward=round(node_progress, 6),
        combat_win_reward=round(combat_win, 6),
        boss_reward=round(boss_reward, 6),
        combat_pace_reward=round(pace_reward, 6),
        aggression_pressure=round(aggression_pressure, 6),
        hp_loss_penalty=round(-hp_loss * hp_loss_weight, 6),
        enemy_hp_progress_reward=round(enemy_progress * enemy_damage_weight, 6),
        prevented_hp_reward=round(
            min(config.prevented_hp_cap, prevented_hp * prevented_hp_weight),
            6,
        ),
        gold_reward=round(gold_reward, 6),
        potion_waste_penalty=round(potion_penalty, 6),
        target_reached_reward=round(float(target_reached_reward), 6),
    )
    total = sum(
        _float(value)
        for key, value in breakdown.model_dump(mode="json").items()
        if key not in {"total", "aggression_pressure"}
    )
    breakdown = breakdown.model_copy(update={"total": round(total, 6)})
    tracker.record(breakdown)
    _refresh_tracker_after_transition(current, tracker)
    return breakdown


def terminal_stats(state: Any) -> dict[str, Any]:
    """Return compact terminal/evaluation fields for a simulator state."""

    payload = _payload(state)
    return {
        "phase": str(payload.get("phase", "unknown")),
        "act": _int(payload.get("act")),
        "floor": _int(payload.get("floor")),
        "won": str(payload.get("phase")) == "complete",
        "dead": str(payload.get("phase")) == "failed",
    }


def aggression_weights(
    aggression_pressure: float,
    config: LearningRewardConfig = DEFAULT_REWARD_CONFIG,
) -> dict[str, float]:
    """Return the active HP, damage, and prevention weights for diagnostics/tests."""

    pressure = _clamp(float(aggression_pressure))
    return {
        "hp_loss_weight": round(
            _lerp(config.hp_loss_safe_weight, config.hp_loss_aggressive_weight, pressure),
            6,
        ),
        "enemy_damage_weight": round(
            _lerp(
                config.enemy_damage_safe_weight,
                config.enemy_damage_aggressive_weight,
                pressure,
            ),
            6,
        ),
        "prevented_hp_weight": round(
            _lerp(
                config.prevented_hp_safe_weight,
                config.prevented_hp_aggressive_weight,
                pressure,
            ),
            6,
        ),
    }


def _payload(state: Any) -> Mapping[str, Any]:
    if isinstance(state, Mapping):
        return state
    return serialize(state)


def _node_progress_reward(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    tracker: LearningRewardTracker,
    config: LearningRewardConfig,
) -> float:
    previous_history = set(str(item) for item in _sequence(previous.get("room_history")))
    current_history = tuple(str(item) for item in _sequence(current.get("room_history")))
    reward = 0.0
    for node_id in current_history:
        if not node_id or node_id in previous_history or node_id in tracker.rewarded_node_ids:
            continue
        tracker.rewarded_node_ids.add(node_id)
        reward += config.node_progress_reward
    previous_completed = set(_completed_node_ids(previous))
    for node_id in _completed_node_ids(current):
        if node_id in previous_completed or node_id in tracker.rewarded_node_ids:
            continue
        tracker.rewarded_node_ids.add(node_id)
        reward += config.node_progress_reward
    if _int(current.get("floor")) > _int(previous.get("floor")):
        node_id = str(_mapping(current.get("map")).get("current_node_id", "") or "")
        if node_id and node_id not in tracker.rewarded_node_ids:
            tracker.rewarded_node_ids.add(node_id)
            reward += config.node_progress_reward
    return reward


def _combat_outcome_rewards(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    tracker: LearningRewardTracker,
    config: LearningRewardConfig,
) -> tuple[float, float, float]:
    previous_combat = _combat(previous)
    if not previous_combat:
        return 0.0, 0.0, 0.0
    if _phase(previous) != "combat" or _phase(current) == "combat" or _phase(current) == "failed":
        return 0.0, 0.0, 0.0

    kind = tracker.combat_room_kind or _room_kind(previous) or _room_kind(current)
    kind = kind.lower()
    turn = max(1, _int(previous_combat.get("turn"), 1))
    if "boss" in kind:
        boss_reward = _boss_reward(_int(previous.get("act"), _int(current.get("act"), 1)), config)
        pace_cap = config.boss_pace_cap
        return 0.0, boss_reward, _pace_bonus(turn, pace_cap, config)
    if "elite" in kind:
        return (
            config.elite_combat_win_reward,
            0.0,
            _pace_bonus(turn, config.elite_pace_cap, config),
        )
    return (
        config.normal_combat_win_reward,
        0.0,
        _pace_bonus(turn, config.normal_pace_cap, config),
    )


def _boss_reward(act: int, config: LearningRewardConfig) -> float:
    if act <= 1:
        return config.act1_boss_reward
    if act == 2:
        return config.act2_boss_reward
    return config.act3_boss_reward


def _pace_bonus(turn: int, cap: float, config: LearningRewardConfig) -> float:
    cutoff = max(1, int(config.pace_turn_cutoff))
    return cap * max(0, cutoff - max(1, turn)) / cutoff


def _combat_aggression_pressure(
    previous: Mapping[str, Any],
    action_descriptor: Mapping[str, Any],
) -> float:
    combat = _combat(previous)
    if not combat:
        return 0.5
    player = _player(previous)
    monsters = _alive_monsters(combat)
    if not monsters:
        return 0.7

    pressure = 0.45
    kind = _room_kind(previous).lower()
    if "elite" in kind:
        pressure += 0.10
    if "boss" in kind:
        pressure += 0.18
    pressure += min(0.20, max(0, _int(combat.get("turn"), 1) - 1) * 0.025)

    incoming = _incoming_hp_damage(previous)
    hp = _int(player.get("hp"))
    max_hp = max(1, _int(player.get("max_hp"), hp or 1))
    hp_fraction = hp / max_hp
    if hp_fraction < 0.35:
        pressure -= (0.35 - hp_fraction) * 0.75
    if incoming >= max(8, hp * 0.35):
        pressure -= 0.12

    total_hp = sum(max(0, _int(monster.get("hp"))) for monster in monsters)
    total_max = max(1, sum(max(1, _int(monster.get("max_hp"), 1)) for monster in monsters))
    pressure += (1.0 - total_hp / total_max) * 0.16
    pressure += min(0.14, _scaling_pressure(monsters))

    preview = _mapping(action_descriptor.get("preview"))
    if _int(preview.get("kills")) > 0 or bool(preview.get("combat_ended")):
        pressure += 0.08
    if _int(preview.get("projected_damage_taken_after_end")) <= 0 and incoming > 0:
        pressure += 0.04
    return _clamp(pressure)


def _scaling_pressure(monsters: Sequence[Mapping[str, Any]]) -> float:
    pressure = 0.0
    scaling_statuses = {
        "strength",
        "ritual",
        "mode_shift",
        "enrage",
        "malleable",
        "regeneration",
        "regen",
        "metallicize",
        "plated_armor",
    }
    for monster in monsters:
        statuses = _mapping(monster.get("statuses"))
        for key, value in statuses.items():
            normalized = str(key).lower()
            if normalized in scaling_statuses or "strength" in normalized:
                pressure += min(0.08, max(0, _int(value)) * 0.02)
        metadata = _mapping(monster.get("metadata"))
        if _bool(metadata.get("scales")) or _bool(metadata.get("scaling")):
            pressure += 0.05
    return pressure


def _enemy_hp_progress(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    tracker: LearningRewardTracker,
) -> int:
    previous_combat = _combat(previous)
    if not previous_combat:
        return 0
    current_combat = _combat(current)
    combat_finished = bool(
        previous_combat and _phase(previous) == "combat" and _phase(current) != "combat"
    )
    current_by_key = {
        key: monster
        for key, monster in _monster_progress_items(current_combat)
    }
    progress_delta = 0
    for key, previous_monster in _monster_progress_items(previous_combat):
        max_hp = max(1, _int(previous_monster.get("max_hp"), 1))
        if key in current_by_key:
            hp = _int(current_by_key[key].get("hp"), max_hp)
        elif combat_finished and _phase(current) != "failed":
            hp = 0
        else:
            hp = _int(previous_monster.get("hp"), max_hp)
        progress = max(0, min(max_hp, max_hp - hp))
        already_rewarded = tracker.max_enemy_hp_progress.get(key, 0)
        if progress > already_rewarded:
            progress_delta += progress - already_rewarded
            tracker.max_enemy_hp_progress[key] = progress
    return progress_delta


def _prevented_hp(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    action_descriptor: Mapping[str, Any],
) -> int:
    previous_projected = _incoming_hp_damage(previous)
    if previous_projected <= 0:
        return 0
    preview = _mapping(action_descriptor.get("preview"))
    if "projected_damage_taken_after_end" in preview:
        projected_after = max(0, _int(preview.get("projected_damage_taken_after_end")))
    elif _phase(current) != "combat":
        projected_after = 0
    else:
        projected_after = _incoming_hp_damage(current)
    return max(0, previous_projected - projected_after)


def _potion_waste_penalty(
    previous: Mapping[str, Any],
    action_descriptor: Mapping[str, Any],
    config: LearningRewardConfig,
) -> float:
    if str(action_descriptor.get("type", "")) != "discard_potion":
        return 0.0
    strategy = _mapping(action_descriptor.get("potion_strategy"))
    if _float(strategy.get("slot_pressure")) > 0 or _bool(strategy.get("belt_full")):
        return 0.0
    if _has_available_potion_replacement(previous):
        return 0.0
    return config.wasteful_potion_discard_penalty


def _has_available_potion_replacement(payload: Mapping[str, Any]) -> bool:
    reward = _mapping(payload.get("reward"))
    if reward:
        if reward.get("potion_id") and not _bool(reward.get("potion_claimed")):
            return True
        potion_ids = _sequence(reward.get("potion_ids"))
        claimed = set(_sequence(reward.get("claimed_potion_indices")))
        skipped = set(_sequence(reward.get("skipped_potion_indices")))
        if any(index not in claimed and index not in skipped for index in range(len(potion_ids))):
            return True
    shop = _mapping(payload.get("shop"))
    for item in _sequence(shop.get("items")):
        item_map = _mapping(item)
        if str(item_map.get("kind", "")) == "potion" and not _bool(item_map.get("purchased")):
            return True
    return False


def _refresh_tracker_after_transition(
    current: Mapping[str, Any],
    tracker: LearningRewardTracker,
) -> None:
    combat = _combat(current)
    if not combat:
        tracker.active_combat_id = None
        tracker.combat_start_turn = 1
        tracker.combat_room_kind = ""
        tracker.max_enemy_hp_progress.clear()
        tracker.previous_projected_damage = 0
        tracker.previous_player_hp = _int(_player(current).get("hp"))
        return
    combat_id = _combat_id(current)
    if tracker.active_combat_id != combat_id:
        tracker.active_combat_id = combat_id
        tracker.combat_start_turn = _int(combat.get("turn"), 1)
        tracker.combat_room_kind = _room_kind(current)
        tracker.max_enemy_hp_progress.clear()
        for key, monster in _monster_progress_items(combat):
            max_hp = max(1, _int(monster.get("max_hp"), 1))
            tracker.max_enemy_hp_progress[key] = max(0, max_hp - _int(monster.get("hp"), max_hp))
    tracker.previous_projected_damage = _incoming_hp_damage(current)
    tracker.previous_player_hp = _int(_player(current).get("hp"))


def _combat_id(payload: Mapping[str, Any]) -> str:
    combat = _combat(payload)
    metadata = _mapping(combat.get("metadata"))
    for key in ("combat_id", "encounter_id", "encounter"):
        value = metadata.get(key)
        if value:
            return str(value)
    return f"a{_int(payload.get('act'))}:f{_int(payload.get('floor'))}:{_room_kind(payload)}"


def _completed_node_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    game_map = _mapping(payload.get("map"))
    return tuple(str(item) for item in _sequence(game_map.get("completed_node_ids")) if str(item))


def _room_kind(payload: Mapping[str, Any]) -> str:
    for source in (
        _mapping(payload.get("flags")),
        _mapping(_combat(payload).get("metadata")),
        _mapping(_mapping(payload.get("reward")).get("metadata")),
    ):
        for key in ("room_kind", "combat_room_kind", "current_room_kind", "node_kind"):
            value = source.get(key)
            if value:
                return str(value)
    game_map = _mapping(payload.get("map"))
    current_id = str(game_map.get("current_node_id", "") or "")
    if current_id:
        for node in _sequence(game_map.get("nodes")):
            node_map = _mapping(node)
            if str(node_map.get("node_id", "")) == current_id:
                return str(node_map.get("kind", ""))
    history = tuple(str(item) for item in _sequence(payload.get("room_history")) if str(item))
    if history:
        last_id = history[-1]
        for node in _sequence(game_map.get("nodes")):
            node_map = _mapping(node)
            if str(node_map.get("node_id", "")) == last_id:
                return str(node_map.get("kind", ""))
    return ""


def _incoming_hp_damage(payload: Mapping[str, Any]) -> int:
    combat = _combat(payload)
    if not combat:
        return 0
    total = 0
    for monster in _alive_monsters(combat):
        intent = str(monster.get("intent", "") or "").lower()
        if "attack" in intent or _int(monster.get("intent_damage")) > 0:
            total += _int(monster.get("intent_damage")) * max(1, _int(monster.get("hit_count"), 1))
    block = _int(_player(payload).get("block"))
    return max(0, total - block)


def _hp_loss(previous: Mapping[str, Any], current: Mapping[str, Any]) -> int:
    return max(0, _int(_player(previous).get("hp")) - _int(_player(current).get("hp")))


def _player_gold(payload: Mapping[str, Any]) -> int:
    return _int(_player(payload).get("gold"))


def _player(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    combat = _combat(payload)
    combat_player = _mapping(combat.get("player"))
    if combat_player:
        return combat_player
    return _mapping(payload.get("player"))


def _combat(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(payload.get("combat"))


def _phase(payload: Mapping[str, Any]) -> str:
    return str(payload.get("phase", ""))


def _alive_monsters(combat: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        monster
        for monster in (_mapping(item) for item in _sequence(combat.get("monsters")))
        if _int(monster.get("hp")) > 0
    )


def _monster_progress_items(combat: Mapping[str, Any]) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    if not combat:
        return ()
    items: list[tuple[str, Mapping[str, Any]]] = []
    for index, raw_monster in enumerate(_sequence(combat.get("monsters"))):
        monster = _mapping(raw_monster)
        monster_id = str(
            monster.get("instance_id")
            or monster.get("monster_instance_id")
            or monster.get("monster_id")
            or index
        )
        items.append((f"{index}:{monster_id}", monster))
    return tuple(items)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[Any]:
    if value is None or isinstance(value, str | bytes):
        return ()
    if isinstance(value, Sequence):
        return value
    return ()


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


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


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(maximum, max(minimum, value))


def _lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * _clamp(amount)
