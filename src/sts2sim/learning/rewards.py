"""Reward shaping functions for self-learning agents.

The rewarder avoids direct advice like "this card is good" or "always take this
relic".  Most value comes from measurable outcomes: survival, combat progress,
useful gold, node progression, and combat victories.  Reward screens also get
small conservative pickup/skip nudges so obvious resources are not treated as
neutral, while card and relic quality is still learned mostly through later run
outcomes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sts2sim.api import serialize
from sts2sim.mechanics.semantics import card_mechanic_profile, relic_mechanic_profile


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
    card_pickup_reward: float = 0.02
    relic_pickup_reward: float = 0.08
    potion_pickup_reward: float = 0.03
    card_remove_reward: float = 0.05
    skip_gold_penalty: float = -0.5
    skip_relic_penalty: float = -0.25
    relic_skip_penalty_decay: float = 0.60
    exclusive_relic_choice_skip_penalty: float = -0.05
    skip_potion_penalty: float = 0.0
    shop_affordable_relic_leave_penalty: float = -0.08
    shop_affordable_relic_leave_decay: float = 0.55
    shop_unaffordable_relic_shortfall_weight: float = -0.001
    shop_unaffordable_relic_penalty_cap: float = 0.12
    shop_leave_opportunity_penalty_cap: float = 0.35
    shop_restock_opportunity_penalty_cap: float = 0.12
    choice_opportunity_weight: float = 0.05
    choice_opportunity_penalty_cap: float = 0.30
    early_card_skip_penalty: float = -0.01
    early_card_skip_deck_size: int = 14
    deck_capability_reward_weight: float = 0.01
    deck_capability_reward_cap: float = 0.15
    deck_problem_relief_weight: float = 0.45
    deck_pressure_penalty_weight: float = 0.70
    deck_synergy_reward_weight: float = 0.30
    deck_context_weight_floor: float = 0.55
    deck_context_weight_cap: float = 1.80
    deck_growth_soft_cap: int = 18
    deck_growth_penalty_weight: float = 0.35
    deck_large_growth_min_score: float = 1.25
    curse_pickup_penalty: float = -0.35
    eternal_curse_extra_penalty: float = -0.10
    curse_burden_reference_deck_size: int = 10
    curse_burden_min_deck_factor: float = 0.30
    curse_burden_max_deck_factor: float = 1.35
    curse_burden_density_weight: float = 0.60
    curse_burden_pressure_weight: float = 0.30
    curse_burden_support_mitigation_cap: float = 0.75
    curse_burden_penalty_cap: float = 0.70
    starter_deck_similarity_penalty: float = -0.30
    starter_deck_similarity_floor: int = 6
    starter_deck_similarity_threshold: float = 0.50
    starter_deck_retention_weight: float = 0.55
    starter_deck_share_weight: float = 0.45
    starter_deck_plain_card_weight: float = 1.0
    starter_deck_upgraded_card_weight: float = 0.55
    starter_deck_improved_card_weight: float = 0.35
    starter_deck_duplicate_card_weight: float = 1.20
    starter_deck_problem_threshold: float = 0.08
    starter_deck_problem_floor: float = 0.45
    starter_deck_complete_penalty_scale: float = 0.0
    starter_deck_act1_penalty_scale: float = 0.35
    starter_deck_act2_penalty_scale: float = 0.70
    starter_deck_act3_penalty_scale: float = 1.0
    starter_deck_capability_mitigation: float = 0.08
    starter_deck_relic_mitigation: float = 0.12


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
    resource_pickup_reward: float = 0.0
    reward_skip_penalty: float = 0.0
    opportunity_cost_penalty: float = 0.0
    deck_capability_reward: float = 0.0
    deck_burden_penalty: float = 0.0
    starter_deck_similarity_penalty: float = 0.0
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
    starter_deck_counts: dict[str, int] = Field(default_factory=dict)
    starter_deck_size: int = 0
    starter_deck_capability_score: float = 0.0

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
        self.starter_deck_counts.clear()
        self.starter_deck_size = 0
        self.starter_deck_capability_score = 0.0

    def record(self, breakdown: LearningRewardBreakdown) -> None:
        """Accumulate component totals for reporting."""

        for key, value in breakdown.model_dump(mode="json").items():
            if key == "aggression_pressure":
                continue
            self.component_totals[key] = self.component_totals.get(key, 0.0) + _float(value)


DEFAULT_REWARD_CONFIG = LearningRewardConfig()
BREAKDOWN_FIELDS: tuple[str, ...] = tuple(LearningRewardBreakdown.model_fields)
_STARTER_DECK_COUNTS = {"strike": 5, "defend": 4, "bash": 1}
_STARTER_DECK_SIZE = sum(_STARTER_DECK_COUNTS.values())
_STARTER_DECK_CAPABILITY_SCORE = 5.9
_DECK_CAPABILITY_WEIGHTS = {
    "frontload": 0.10,
    "block": 0.08,
    "draw": 0.55,
    "energy": 0.70,
    "scaling": 0.65,
    "exhaust": 0.35,
    "retain": 0.25,
    "status_enemy": 0.25,
}
_DECK_PROBLEM_WEIGHTS = {
    "low_frontload": 1.00,
    "low_block": 0.90,
    "low_draw": 1.15,
    "energy_heavy": 1.20,
    "missing_scaling": 0.95,
    "too_big": 0.85,
    "curse_burden": 1.30,
    "starter_density": 0.55,
}
_CURSE_SOURCE_COMPENSATION_SUPPORT = {
    "calling_bell": 0.34,
    "cursed_pearl": 0.28,
    "hefty_tablet": 0.20,
    "neows_bones": 0.30,
    "sere_talon": 0.42,
}


def learning_reward(
    previous_state: Any,
    next_state: Any,
    config: LearningRewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    """Return the outcome-based reward total for one simulator transition."""

    return learning_reward_breakdown(previous_state, next_state, config=config).total


def deck_delta_summary(
    previous_state: Any,
    next_state: Any,
    config: LearningRewardConfig = DEFAULT_REWARD_CONFIG,
) -> dict[str, Any]:
    """Return diagnostics for how a deck transition changed fit and pressure."""

    return _deck_delta_summary(_payload(previous_state), _payload(next_state), config)


def starter_dependency_summary(
    state: Any,
    *,
    tracker: LearningRewardTracker | None = None,
    config: LearningRewardConfig = DEFAULT_REWARD_CONFIG,
) -> dict[str, Any]:
    """Return diagnostics for terminal starter-deck dependency shaping."""

    payload = _payload(state)
    return _starter_dependency_summary(payload, config, tracker=tracker)


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
    _capture_starter_deck_baseline(previous, tracker)
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
    resource_pickup = _resource_pickup_reward(previous, current, descriptor, config)
    reward_skip = _reward_skip_penalty(previous, descriptor, config)
    opportunity_cost = _opportunity_cost_penalty(previous, descriptor, config)
    deck_capability = _deck_capability_reward(previous, current, config)
    deck_burden = _deck_burden_penalty(previous, current, descriptor, config)
    starter_similarity = _starter_deck_similarity_penalty(current, config, tracker)

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
        resource_pickup_reward=round(resource_pickup, 6),
        reward_skip_penalty=round(reward_skip, 6),
        opportunity_cost_penalty=round(opportunity_cost, 6),
        deck_capability_reward=round(deck_capability, 6),
        deck_burden_penalty=round(deck_burden, 6),
        starter_deck_similarity_penalty=round(starter_similarity, 6),
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


def _resource_pickup_reward(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    action_descriptor: Mapping[str, Any],
    config: LearningRewardConfig,
) -> float:
    action_type = str(action_descriptor.get("type", ""))
    if action_type == "take_reward_card":
        if _master_deck_count(current) > _master_deck_count(previous):
            return 0.0
        return config.card_pickup_reward
    if action_type == "take_reward_relic":
        return config.relic_pickup_reward
    if action_type == "take_reward_potion":
        return config.potion_pickup_reward
    if action_type in {"shop_buy", "take_reward_card_removal"}:
        item = _mapping(action_descriptor.get("item"))
        kind = str(item.get("kind", ""))
        if kind in {"card", "colorless_card"}:
            return (
                0.0
                if _master_deck_count(current) > _master_deck_count(previous)
                else config.card_pickup_reward
            )
        if kind == "relic":
            return config.relic_pickup_reward
        if kind == "potion":
            return config.potion_pickup_reward
        if kind == "card_removal" or _master_deck_count(current) < _master_deck_count(previous):
            return config.card_remove_reward
    if _master_deck_count(current) < _master_deck_count(previous):
        return config.card_remove_reward
    return 0.0


def _reward_skip_penalty(
    previous: Mapping[str, Any],
    action_descriptor: Mapping[str, Any],
    config: LearningRewardConfig,
) -> float:
    action_type = str(action_descriptor.get("type", ""))
    if action_type == "proceed":
        return (
            (config.skip_gold_penalty if _unclaimed_gold_amount(previous) > 0 else 0.0)
            + _unclaimed_relic_skip_penalty(previous, config)
        )
    if action_type != "skip_reward":
        return 0.0
    reward_choice = _mapping(action_descriptor.get("reward_choice"))
    skip_kind = str(reward_choice.get("skip_kind", reward_choice.get("kind", "")))
    if skip_kind == "gold":
        return config.skip_gold_penalty
    if skip_kind == "relic":
        return _unclaimed_relic_skip_penalty(previous, config, minimum_choices=1)
    if skip_kind == "potion":
        return config.skip_potion_penalty if _potion_slots_available(previous) else 0.0
    if (
        skip_kind in {"card_options", "card_group", "fixed_card", "card"}
        and _master_deck_count(previous) <= max(0, config.early_card_skip_deck_size)
    ):
        return config.early_card_skip_penalty
    return 0.0


def _opportunity_cost_penalty(
    previous: Mapping[str, Any],
    action_descriptor: Mapping[str, Any],
    config: LearningRewardConfig,
) -> float:
    return round(
        _shop_leave_opportunity_cost(previous, action_descriptor, config)
        + _choice_opportunity_cost(previous, action_descriptor, config),
        6,
    )


def _shop_leave_opportunity_cost(
    previous: Mapping[str, Any],
    action_descriptor: Mapping[str, Any],
    config: LearningRewardConfig,
) -> float:
    action_type = str(action_descriptor.get("type", ""))
    if action_type not in {"shop_leave", "proceed"}:
        return 0.0
    if _phase(previous) not in {"shop"}:
        return 0.0
    shop = _mapping(previous.get("shop"))
    if not shop:
        return 0.0
    gold = _player_gold(previous)
    relic_prices: list[int] = []
    for raw_item in _sequence(shop.get("items")):
        item = _mapping(raw_item)
        if str(item.get("kind", "")).lower() != "relic" or _bool(item.get("purchased")):
            continue
        price = max(0, _int(item.get("price")))
        if price > 0:
            relic_prices.append(price)
    if not relic_prices:
        return 0.0

    affordable_count = _affordable_shop_purchase_count(gold, relic_prices)
    penalty = _diminishing_penalty(
        config.shop_affordable_relic_leave_penalty,
        affordable_count,
        config.shop_affordable_relic_leave_decay,
    )
    unaffordable_shortfalls = [price - gold for price in relic_prices if price > gold]
    if unaffordable_shortfalls:
        penalty += max(
            -config.shop_unaffordable_relic_penalty_cap,
            min(unaffordable_shortfalls) * config.shop_unaffordable_relic_shortfall_weight,
        )
    cap = (
        config.shop_restock_opportunity_penalty_cap
        if _shop_restock_enabled(previous)
        else config.shop_leave_opportunity_penalty_cap
    )
    return max(-cap, penalty)


def _affordable_shop_purchase_count(gold: int, prices: Sequence[int]) -> int:
    remaining = max(0, gold)
    count = 0
    for price in sorted(price for price in prices if price > 0 and price <= remaining):
        if price > remaining:
            continue
        remaining -= price
        count += 1
    return count


def _shop_restock_enabled(payload: Mapping[str, Any]) -> bool:
    if any(relic_id in {"the_courier", "courier"} for relic_id in _relic_ids(payload)):
        return True
    flags = _mapping(payload.get("flags"))
    if _bool(flags.get("shop_restock_enabled")):
        return True
    shop = _mapping(payload.get("shop"))
    return _bool(shop.get("restock_enabled"))


def _choice_opportunity_cost(
    previous: Mapping[str, Any],
    action_descriptor: Mapping[str, Any],
    config: LearningRewardConfig,
) -> float:
    action_type = str(action_descriptor.get("type", ""))
    if action_type == "choose_ancient":
        return _option_choice_opportunity_cost(
            previous,
            action_descriptor,
            option_container="ancient",
            selected_option=_mapping(action_descriptor.get("ancient_option")),
            config=config,
        )
    if action_type == "choose_event":
        return _option_choice_opportunity_cost(
            previous,
            action_descriptor,
            option_container="event",
            selected_option=_mapping(action_descriptor.get("event_option")),
            config=config,
        )
    return 0.0


def _option_choice_opportunity_cost(
    previous: Mapping[str, Any],
    action_descriptor: Mapping[str, Any],
    *,
    option_container: str,
    selected_option: Mapping[str, Any],
    config: LearningRewardConfig,
) -> float:
    if not selected_option:
        return 0.0
    container = _mapping(previous.get(option_container))
    options = tuple(_mapping(item) for item in _sequence(container.get("options")))
    if len(options) <= 1:
        return 0.0
    selected_id = str(
        selected_option.get("option_id")
        or _mapping(action_descriptor.get("action")).get("target_id")
        or ""
    )
    chosen_score = _generic_option_value_score(selected_option)
    other_scores = [
        _generic_option_value_score(option)
        for option in options
        if str(option.get("option_id", "")) != selected_id
        and not _bool(option.get("disabled"))
    ]
    if not other_scores:
        return 0.0
    visible_gap = max(other_scores) - chosen_score
    if visible_gap <= 0:
        return 0.0
    return -min(
        config.choice_opportunity_penalty_cap,
        visible_gap * config.choice_opportunity_weight,
    )


def _generic_option_value_score(option: Mapping[str, Any]) -> float:
    metadata = _mapping(option.get("metadata"))
    fixed_relic_count = _int(metadata.get("fixed_relic_count"))
    if option.get("relic_id"):
        fixed_relic_count += 1
    if _sequence(metadata.get("fixed_relic_ids")):
        fixed_relic_count += len(_sequence(metadata.get("fixed_relic_ids")))
    relic_count = (
        fixed_relic_count
        + _int(option.get("random_relic_count"))
        + _int(metadata.get("random_relic_count"))
        + _int(option.get("event_reward_relic_count"))
        + _int(metadata.get("event_reward_relic_count"))
    )
    upgrade_count = (
        _int(option.get("upgrade_random_count"))
        + _int(metadata.get("upgrade_random_count"))
        + _int(metadata.get("upgrade_count"))
        + _int(metadata.get("card_upgrade_count"))
    )
    card_reward_count = _int(option.get("card_reward_count")) + _int(
        metadata.get("card_reward_count")
    )
    remove_count = (
        _int(option.get("remove_random_count"))
        + _int(metadata.get("remove_random_count"))
        + _int(metadata.get("remove_count"))
    )
    transform_count = (
        _int(option.get("transform_random_count"))
        + _int(metadata.get("transform_random_count"))
        + _int(metadata.get("transform_count"))
    )
    potion_count = (
        _int(option.get("random_potion_count"))
        + _int(metadata.get("random_potion_count"))
        + _int(metadata.get("potion_count"))
    )
    max_hp_delta = _int(option.get("max_hp_delta")) + _int(metadata.get("max_hp_delta"))
    hp_delta = _int(option.get("hp_delta")) + _int(metadata.get("hp_delta"))
    gold_delta = _int(option.get("gold_delta")) + _int(metadata.get("gold_delta"))
    curse_count = _int(metadata.get("curse_count")) + _int(metadata.get("random_curse_count"))
    if metadata.get("card_id") and "curse" in _normalized_id(metadata.get("card_id")):
        curse_count += 1
    risk = _float(metadata.get("risk"))
    return (
        relic_count * 1.0
        + upgrade_count * 0.35
        + remove_count * 0.30
        + transform_count * 0.22
        + card_reward_count * 0.16
        + potion_count * 0.12
        + max(0, max_hp_delta) * 0.04
        + min(max(0, gold_delta) * 0.01, 0.5)
        + max(0, hp_delta) * 0.02
        + min(0, hp_delta) * 0.03
        - curse_count * 0.45
        - risk * 0.20
    )


def _unclaimed_gold_amount(payload: Mapping[str, Any]) -> int:
    reward = _mapping(payload.get("reward"))
    if not reward:
        return 0
    if _bool(reward.get("gold_claimed")) or _bool(reward.get("gold_skipped")):
        return 0
    return max(0, _int(reward.get("gold")))


def _unclaimed_relic_skip_penalty(
    payload: Mapping[str, Any],
    config: LearningRewardConfig,
    *,
    minimum_choices: int = 0,
) -> float:
    reward = _mapping(payload.get("reward"))
    unclaimed = _unclaimed_relic_count(payload)
    if unclaimed <= 0 and minimum_choices <= 0:
        return 0.0
    choice_limit = _reward_relic_choice_limit(reward)
    if choice_limit > 0:
        claimed = _claimed_relic_count(reward)
        missed_choices = max(0, min(choice_limit, _available_relic_count(reward)) - claimed)
        return _diminishing_penalty(
            config.exclusive_relic_choice_skip_penalty,
            max(minimum_choices, missed_choices),
            config.relic_skip_penalty_decay,
        )
    return _diminishing_penalty(
        config.skip_relic_penalty,
        max(minimum_choices, unclaimed),
        config.relic_skip_penalty_decay,
    )


def _unclaimed_relic_count(payload: Mapping[str, Any]) -> int:
    reward = _mapping(payload.get("reward"))
    if not reward:
        return 0
    claimed = {_normalized_id(item) for item in _sequence(reward.get("claimed_relic_ids"))}
    skipped = {_normalized_id(item) for item in _sequence(reward.get("skipped_relic_ids"))}
    relic_ids = tuple(
        _normalized_id(item)
        for item in _sequence(reward.get("relic_ids"))
        if _normalized_id(item)
    )
    if relic_ids:
        return sum(
            1 for relic_id in relic_ids if relic_id not in claimed and relic_id not in skipped
        )
    relic_id = _normalized_id(reward.get("relic_id"))
    if not relic_id:
        return 0
    if _bool(reward.get("relic_claimed")) or _bool(reward.get("relic_skipped")):
        return 0
    if relic_id in claimed or relic_id in skipped:
        return 0
    return 1


def _diminishing_penalty(base_penalty: float, count: int, decay: float) -> float:
    if count <= 0 or base_penalty == 0.0:
        return 0.0
    clamped_decay = _clamp(float(decay), 0.0, 1.0)
    return sum(base_penalty * (clamped_decay**index) for index in range(count))


def _reward_relic_choice_limit(reward: Mapping[str, Any]) -> int:
    if not reward:
        return 0
    metadata = _mapping(reward.get("metadata"))
    for key in (
        "max_relic_choices",
        "relic_max_choices",
        "relic_choice_count",
        "choose_relic_count",
        "relic_pick_count",
    ):
        value = _int(metadata.get(key))
        if value > 0:
            return value
    if _bool(metadata.get("exclusive_relic_choices")) or _bool(
        reward.get("exclusive_relic_choices")
    ):
        return 1
    return 0


def _available_relic_count(reward: Mapping[str, Any]) -> int:
    relic_ids = tuple(_normalized_id(item) for item in _sequence(reward.get("relic_ids")))
    return len(tuple(item for item in relic_ids if item)) + int(
        bool(_normalized_id(reward.get("relic_id")))
    )


def _claimed_relic_count(reward: Mapping[str, Any]) -> int:
    return len(_sequence(reward.get("claimed_relic_ids"))) + int(_bool(reward.get("relic_claimed")))


def _deck_capability_reward(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    config: LearningRewardConfig,
) -> float:
    if config.deck_capability_reward_weight <= 0:
        return 0.0
    summary = _deck_delta_summary(previous, current, config)
    net_score = _float(summary.get("net_score"))
    if net_score <= 0:
        return 0.0
    return min(
        config.deck_capability_reward_cap,
        net_score * config.deck_capability_reward_weight,
    )


def _deck_delta_summary(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    config: LearningRewardConfig,
) -> dict[str, Any]:
    before = _deck_metrics(previous, config)
    after = _deck_metrics(current, config)
    before_categories = _mapping(before.get("categories"))
    after_categories = _mapping(after.get("categories"))
    category_deltas = {
        key: _float(after_categories.get(key)) - _float(before_categories.get(key))
        for key in _DECK_CAPABILITY_WEIGHTS
    }
    context_weights = _deck_context_weights(previous, before, config)
    category_delta_score = sum(
        delta
        * _DECK_CAPABILITY_WEIGHTS[key]
        * _float(context_weights.get(key), 1.0)
        for key, delta in category_deltas.items()
    )

    before_problems = _deck_problem_scores(previous, before, config)
    after_problems = _deck_problem_scores(current, after, config)
    problem_relief = {
        key: max(0.0, _float(before_problems.get(key)) - _float(after_problems.get(key)))
        for key in _DECK_PROBLEM_WEIGHTS
    }
    problems_worsened = {
        key: max(0.0, _float(after_problems.get(key)) - _float(before_problems.get(key)))
        for key in _DECK_PROBLEM_WEIGHTS
    }
    problem_relief_score = sum(
        value * _DECK_PROBLEM_WEIGHTS[key] for key, value in problem_relief.items()
    )
    pressure_cost = sum(
        value * _DECK_PROBLEM_WEIGHTS[key] for key, value in problems_worsened.items()
    )
    synergy_delta = _float(after.get("synergy_score")) - _float(before.get("synergy_score"))

    added_cards = max(0, int(_float(after.get("deck_size")) - _float(before.get("deck_size"))))
    removed_cards = max(0, int(_float(before.get("deck_size")) - _float(after.get("deck_size"))))
    growth_cost = _deck_growth_cost(added_cards, _float(after.get("deck_size")), config)
    net_score = (
        category_delta_score
        + problem_relief_score * config.deck_problem_relief_weight
        + max(0.0, synergy_delta) * config.deck_synergy_reward_weight
        - pressure_cost * config.deck_pressure_penalty_weight
        - growth_cost
    )
    growth_blocked = False
    if (
        added_cards > 0
        and _float(after.get("deck_size")) > max(0, config.deck_growth_soft_cap)
        and net_score < config.deck_large_growth_min_score
    ):
        growth_blocked = True
        net_score = 0.0

    return {
        "deck_size_before": int(_float(before.get("deck_size"))),
        "deck_size_after": int(_float(after.get("deck_size"))),
        "cards_added": added_cards,
        "cards_removed": removed_cards,
        "capability_before": round(_float(before.get("score")), 6),
        "capability_after": round(_float(after.get("score")), 6),
        "capability_delta": round(_float(after.get("score")) - _float(before.get("score")), 6),
        "category_delta_score": round(category_delta_score, 6),
        "problem_relief_score": round(problem_relief_score, 6),
        "pressure_cost": round(pressure_cost, 6),
        "synergy_before": round(_float(before.get("synergy_score")), 6),
        "synergy_after": round(_float(after.get("synergy_score")), 6),
        "synergy_delta": round(synergy_delta, 6),
        "growth_cost": round(growth_cost, 6),
        "growth_blocked": growth_blocked,
        "net_score": round(max(0.0, net_score), 6),
        "category_deltas": _rounded_float_dict(category_deltas),
        "context_weights": _rounded_float_dict(context_weights),
        "problems_before": _rounded_float_dict(before_problems),
        "problems_after": _rounded_float_dict(after_problems),
        "problem_relief": _rounded_float_dict(problem_relief),
        "problems_worsened": _rounded_float_dict(problems_worsened),
    }


def _deck_metrics(payload: Mapping[str, Any], config: LearningRewardConfig) -> dict[str, Any]:
    cards = _master_deck_cards(payload)
    categories = {key: 0.0 for key in _DECK_CAPABILITY_WEIGHTS}
    attack_count = 0
    skill_count = 0
    power_count = 0
    expensive_count = 0
    low_cost_count = 0
    playable_cost_total = 0.0
    playable_count = 0
    burden_count = 0

    for card in cards:
        profile = card_mechanic_profile(card)
        values = _mapping(profile.values)
        damage = _float(values.get("damage")) + _float(values.get("aoe_damage"))
        categories["frontload"] += min(16.0, damage)
        categories["block"] += min(16.0, _float(values.get("block")))
        categories["draw"] += min(5.0, _float(values.get("draw")))
        categories["energy"] += min(
            3.0,
            _float(values.get("energy")) + _float(values.get("cost_reduction")) * 0.5,
        )
        categories["scaling"] += min(
            6.0,
            _float(values.get("strength"))
            + _float(values.get("dexterity"))
            + _float(values.get("focus"))
            + _float(values.get("repeating_effect"))
            + _float(values.get("periodic_effect")),
        )
        categories["exhaust"] += min(3.0, _float(values.get("exhaust")))
        categories["retain"] += min(3.0, _float(values.get("retain")))
        categories["status_enemy"] += min(
            5.0,
            _float(values.get("weak"))
            + _float(values.get("vulnerable"))
            + _float(values.get("poison"))
            + _float(values.get("status_enemy")),
        )

        card_type = _normalized_id(card.get("type", card.get("card_type")))
        if card_type == "attack":
            attack_count += 1
        elif card_type == "skill":
            skill_count += 1
        elif card_type == "power":
            power_count += 1
        if _card_is_burden(card):
            burden_count += 1
            continue
        cost = _card_play_cost(card)
        if cost is None:
            continue
        playable_count += 1
        playable_cost_total += cost
        if cost >= 2.0:
            expensive_count += 1
        if cost <= 1.0:
            low_cost_count += 1

    deck_size = float(len(cards))
    average_cost = playable_cost_total / max(1.0, float(playable_count))
    starter_weight = _weighted_starter_card_count(cards, config) if cards else 0.0
    metrics: dict[str, Any] = {
        "categories": categories,
        "deck_size": deck_size,
        "average_cost": average_cost,
        "attack_count": float(attack_count),
        "skill_count": float(skill_count),
        "power_count": float(power_count),
        "expensive_count": float(expensive_count),
        "low_cost_count": float(low_cost_count),
        "playable_count": float(playable_count),
        "burden_count": float(burden_count),
        "starter_weight": starter_weight,
    }
    metrics["score"] = _deck_metric_score(metrics, config)
    metrics["synergy_score"] = _deck_synergy_score(payload, metrics)
    return metrics


def _deck_metric_score(metrics: Mapping[str, Any], config: LearningRewardConfig) -> float:
    categories = _mapping(metrics.get("categories"))
    deck_size = _float(metrics.get("deck_size"))
    bloat_penalty = max(0.0, deck_size - max(0, config.deck_growth_soft_cap)) * 0.25
    return (
        sum(
            _float(categories.get(key)) * weight
            for key, weight in _DECK_CAPABILITY_WEIGHTS.items()
        )
        - bloat_penalty
    )


def _deck_problem_scores(
    payload: Mapping[str, Any],
    metrics: Mapping[str, Any],
    config: LearningRewardConfig,
) -> dict[str, float]:
    categories = _mapping(metrics.get("categories"))
    deck_size = max(1.0, _float(metrics.get("deck_size")))
    act = max(1, _int(payload.get("act"), 1))
    hp_fraction = _hp_fraction(payload)
    relic_values = _relic_mechanic_values(payload)
    energy_support = _float(categories.get("energy")) + _float(relic_values.get("energy"))
    draw = _float(categories.get("draw"))

    frontload_target = 24.0 + max(0, act - 1) * 10.0
    block_target = 18.0 + max(0, act - 1) * 6.0
    if hp_fraction < 0.45:
        block_target *= 1.25
        frontload_target *= 0.90
    draw_target = 1.0 + max(0.0, deck_size - 10.0) * 0.10 + max(0, act - 1) * 0.35
    scaling_target = 0.8 if act <= 1 else 2.2 if act == 2 else 3.4
    average_cost_limit = 1.18 + min(0.45, energy_support * 0.08) + min(0.20, draw * 0.025)
    starter_density = _float(metrics.get("starter_weight")) / max(
        deck_size,
        float(_STARTER_DECK_SIZE),
    )

    return {
        "low_frontload": _shortfall(_float(categories.get("frontload")), frontload_target),
        "low_block": _shortfall(_float(categories.get("block")), block_target),
        "low_draw": _shortfall(draw, draw_target),
        "energy_heavy": max(0.0, _float(metrics.get("average_cost")) - average_cost_limit),
        "missing_scaling": _shortfall(_float(categories.get("scaling")), scaling_target),
        "too_big": max(0.0, deck_size - max(0, config.deck_growth_soft_cap))
        / max(6.0, float(max(1, config.deck_growth_soft_cap))),
        "curse_burden": min(1.5, _float(metrics.get("burden_count")) / deck_size * 3.0),
        "starter_density": max(0.0, starter_density - 0.45),
    }


def _deck_context_weights(
    payload: Mapping[str, Any],
    metrics: Mapping[str, Any],
    config: LearningRewardConfig,
) -> dict[str, float]:
    act = max(1, _int(payload.get("act"), 1))
    hp_fraction = _hp_fraction(payload)
    deck_size = _float(metrics.get("deck_size"))
    categories = _mapping(metrics.get("categories"))
    weights = {key: 1.0 for key in _DECK_CAPABILITY_WEIGHTS}

    if act <= 1:
        weights["frontload"] += 0.25
        weights["block"] += 0.10
        weights["scaling"] -= 0.20
    elif act == 2:
        weights["draw"] += 0.20
        weights["scaling"] += 0.25
        weights["status_enemy"] += 0.10
    else:
        weights["draw"] += 0.30
        weights["energy"] += 0.20
        weights["scaling"] += 0.40

    if hp_fraction < 0.45:
        weights["block"] += 0.45
        weights["retain"] += 0.10
        weights["frontload"] -= 0.10
    if deck_size > max(0, config.deck_growth_soft_cap):
        pressure = min(0.60, (deck_size - max(0, config.deck_growth_soft_cap)) * 0.06)
        weights["draw"] += pressure
        weights["exhaust"] += pressure
        weights["frontload"] -= pressure * 0.35
    if _float(metrics.get("average_cost")) > 1.35:
        weights["energy"] += 0.35
        weights["draw"] += 0.10
    if _float(categories.get("scaling")) > 3.0:
        weights["frontload"] += 0.10
        weights["draw"] += 0.10

    return {
        key: _clamp(
            value,
            config.deck_context_weight_floor,
            config.deck_context_weight_cap,
        )
        for key, value in weights.items()
    }


def _deck_synergy_score(payload: Mapping[str, Any], metrics: Mapping[str, Any]) -> float:
    categories = _mapping(metrics.get("categories"))
    relic_values = _relic_mechanic_values(payload)
    deck_size = _float(metrics.get("deck_size"))
    size_pressure = max(0.0, deck_size - 14.0) / 8.0
    attack_count = _float(metrics.get("attack_count"))
    low_cost_count = _float(metrics.get("low_cost_count"))
    expensive_count = _float(metrics.get("expensive_count"))
    burden_count = _float(metrics.get("burden_count"))
    starter_weight = _float(metrics.get("starter_weight"))
    strength_support = _float(categories.get("scaling")) + _float(relic_values.get("strength"))
    energy_support = _float(categories.get("energy")) + _float(relic_values.get("energy"))
    draw = _float(categories.get("draw"))
    exhaust = _float(categories.get("exhaust"))

    score = 0.0
    score += min(3.0, strength_support * 0.25) * min(3.0, attack_count / 4.0)
    score += min(3.0, draw * 0.35) * min(3.0, low_cost_count / 5.0)
    score += min(3.0, energy_support * 0.70) * min(3.0, expensive_count / 2.0)
    score += min(3.0, exhaust * 0.80) * min(
        3.0,
        size_pressure + burden_count + starter_weight / 6.0,
    )
    score += min(2.0, _float(categories.get("status_enemy")) * 0.35) * min(
        2.0,
        attack_count / 5.0,
    )
    return score


def _deck_growth_cost(
    added_cards: int,
    after_deck_size: float,
    config: LearningRewardConfig,
) -> float:
    if added_cards <= 0:
        return 0.0
    over_soft_cap = max(0.0, after_deck_size - max(0, config.deck_growth_soft_cap))
    if over_soft_cap <= 0:
        return 0.0
    return (
        float(added_cards)
        * over_soft_cap
        * config.deck_growth_penalty_weight
        / max(6.0, float(max(1, config.deck_growth_soft_cap)))
    )


def _deck_burden_penalty(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    action_descriptor: Mapping[str, Any],
    config: LearningRewardConfig,
) -> float:
    if config.curse_pickup_penalty >= 0:
        return 0.0
    penalty = 0.0
    before = _deck_metrics(previous, config)
    after = _deck_metrics(current, config)
    after_problems = _deck_problem_scores(current, after, config)
    added_burdens = tuple(
        card for card in _added_deck_cards(previous, current) if _card_is_burden(card)
    )
    source_support = _curse_transition_compensation_support(
        previous,
        current,
        action_descriptor,
        config,
    )
    per_burden_source_support = source_support / max(1.0, float(len(added_burdens)))
    for card in added_burdens:
        penalty -= _card_burden_severity(
            card,
            current,
            before,
            after,
            after_problems,
            config,
            source_compensation_support=per_burden_source_support,
        )
    return penalty


def _card_burden_severity(
    card: Mapping[str, Any],
    current: Mapping[str, Any],
    before_metrics: Mapping[str, Any],
    after_metrics: Mapping[str, Any],
    after_problems: Mapping[str, Any],
    config: LearningRewardConfig,
    *,
    source_compensation_support: float = 0.0,
) -> float:
    base = abs(config.curse_pickup_penalty)
    if _card_has_marker(card, "eternal"):
        base += abs(config.eternal_curse_extra_penalty)
    if _card_has_marker(card, "unplayable"):
        base += abs(config.curse_pickup_penalty) * 0.15
    if _normalized_id(card.get("type", card.get("card_type"))) == "status":
        base *= 0.75

    deck_factor = _curse_deck_size_factor(_float(before_metrics.get("deck_size")), config)
    after_deck_size = max(1.0, _float(after_metrics.get("deck_size")))
    burden_density = _float(after_metrics.get("burden_count")) / after_deck_size
    density_factor = 1.0 + min(
        config.curse_burden_density_weight,
        burden_density * 3.0 * config.curse_burden_density_weight,
    )
    pressure_factor = 1.0 + min(
        config.curse_burden_pressure_weight,
        (
            _float(after_problems.get("low_draw")) * 0.20
            + _float(after_problems.get("energy_heavy")) * 0.20
            + _float(after_problems.get("too_big")) * 0.35
        )
        * config.curse_burden_pressure_weight,
    )
    support = _curse_burden_support_score(
        card,
        current,
        after_metrics,
        config,
        source_compensation_support=source_compensation_support,
    )
    severity = base * deck_factor * density_factor * pressure_factor * (1.0 - support)
    return min(max(0.0, config.curse_burden_penalty_cap), severity)


def _curse_deck_size_factor(deck_size: float, config: LearningRewardConfig) -> float:
    reference = max(1.0, float(config.curse_burden_reference_deck_size))
    effective_size = max(1.0, deck_size)
    return _clamp(
        reference / effective_size,
        config.curse_burden_min_deck_factor,
        config.curse_burden_max_deck_factor,
    )


def _curse_burden_support_score(
    card: Mapping[str, Any],
    current: Mapping[str, Any],
    after_metrics: Mapping[str, Any],
    config: LearningRewardConfig,
    *,
    source_compensation_support: float = 0.0,
) -> float:
    categories = _mapping(after_metrics.get("categories"))
    relic_values = _relic_mechanic_values(current)
    support = 0.0
    support += min(0.20, _float(categories.get("draw")) * 0.035)
    support += min(0.22, _float(categories.get("exhaust")) * 0.08)
    support += min(0.10, _float(categories.get("retain")) * 0.04)
    support += min(0.12, _float(relic_values.get("draw")) * 0.035)
    support += min(0.12, _float(relic_values.get("exhaust")) * 0.08)
    support += min(0.10, _float(relic_values.get("card_remove")) * 0.10)
    support += min(0.08, _float(relic_values.get("energy")) * 0.05)
    support += _curse_payoff_support(current)
    support += max(
        _card_frontloaded_compensation_support(card),
        max(0.0, source_compensation_support),
    )
    return _clamp(support, 0.0, config.curse_burden_support_mitigation_cap)


def _curse_payoff_support(payload: Mapping[str, Any]) -> float:
    support = 0.0
    for relic_id in _relic_ids(payload):
        normalized = _normalized_id(relic_id)
        if normalized in {"blue_candle", "darkstone_periapt", "du_vu_doll", "omamori"}:
            support += 0.18
        if "curse" in normalized or "cursed" in normalized:
            support += 0.08
    return min(0.30, support)


def _curse_transition_compensation_support(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    action_descriptor: Mapping[str, Any],
    config: LearningRewardConfig,
) -> float:
    inferred = _state_delta_compensation_support(previous, current)
    reward_screen = _reward_screen_compensation_support(current)
    descriptor = _descriptor_compensation_support(action_descriptor)
    source_hint = max(
        _source_compensation_support(relic_id)
        for relic_id in ("", *_added_relic_ids(previous, current))
    )
    support = max(inferred, reward_screen, descriptor, source_hint)
    return _clamp(support, 0.0, config.curse_burden_support_mitigation_cap)


def _state_delta_compensation_support(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> float:
    support = 0.0
    support += min(0.30, max(0, _player_gold(current) - _player_gold(previous)) / 1200.0)
    support += min(0.16, max(0, _player_max_hp(current) - _player_max_hp(previous)) / 90.0)
    support += min(0.10, max(0, _player_hp(current) - _player_hp(previous)) / 100.0)
    support += min(0.08, max(0, _potion_count(current) - _potion_count(previous)) * 0.04)
    support += min(0.30, len(_added_relic_ids(previous, current)) * 0.10)
    return support


def _reward_screen_compensation_support(payload: Mapping[str, Any]) -> float:
    reward = _mapping(payload.get("reward"))
    if not reward:
        return 0.0
    support = 0.0
    card_group_count = 0
    if _sequence(reward.get("card_options")) and not _bool(reward.get("card_claimed")):
        card_group_count += 1
    claimed_card_groups = {
        _int(index)
        for index in _sequence(reward.get("claimed_card_option_group_indices"))
    }
    skipped_card_groups = {
        _int(index)
        for index in _sequence(reward.get("skipped_card_option_group_indices"))
    }
    for index, group in enumerate(_sequence(reward.get("card_option_groups"))):
        if (
            _sequence(group)
            and index not in claimed_card_groups
            and index not in skipped_card_groups
        ):
            card_group_count += 1
    support += min(0.18, card_group_count * 0.08)

    relic_count = 0
    if reward.get("relic_id") and not _bool(reward.get("relic_claimed")):
        relic_count += 1
    claimed_relics = {
        _normalized_id(relic_id)
        for relic_id in _sequence(reward.get("claimed_relic_ids"))
    }
    for relic_id in _sequence(reward.get("relic_ids")):
        if _normalized_id(relic_id) not in claimed_relics:
            relic_count += 1
    support += min(0.25, relic_count * 0.10)

    potion_count = 0
    if reward.get("potion_id") and not _bool(reward.get("potion_claimed")):
        potion_count += 1
    claimed_potions = {
        _int(index) for index in _sequence(reward.get("claimed_potion_indices"))
    }
    skipped_potions = {
        _int(index) for index in _sequence(reward.get("skipped_potion_indices"))
    }
    for index, potion_id in enumerate(_sequence(reward.get("potion_ids"))):
        if potion_id and index not in claimed_potions and index not in skipped_potions:
            potion_count += 1
    support += min(0.08, potion_count * 0.04)

    if not _bool(reward.get("gold_claimed")):
        support += min(0.22, max(0, _int(reward.get("gold"))) / 1400.0)
    return support


def _descriptor_compensation_support(action_descriptor: Mapping[str, Any]) -> float:
    descriptor_sources = (
        action_descriptor,
        _mapping(action_descriptor.get("event_option")),
        _mapping(action_descriptor.get("ancient_option")),
        _mapping(action_descriptor.get("reward_choice")),
        _mapping(action_descriptor.get("reward_bundle")),
        _mapping(action_descriptor.get("option_slot")),
    )
    support = 0.0
    for source in descriptor_sources:
        support = max(support, _compensation_support_from_mapping(source))
        support = max(
            support,
            _compensation_support_from_mapping(_mapping(source.get("metadata"))),
        )
    return support


def _compensation_support_from_mapping(source: Mapping[str, Any]) -> float:
    if not source:
        return 0.0
    support = 0.0
    support += min(
        0.30,
        max(0, _int(source.get("gold_delta"), _int(source.get("gold")))) / 1200.0,
    )
    support += min(0.16, max(0, _int(source.get("max_hp_delta"))) / 90.0)
    support += min(
        0.10,
        max(0, _int(source.get("heal_amount")) + _int(source.get("hp_delta"))) / 100.0,
    )
    support += min(
        0.18,
        (
            _int(source.get("card_reward_count"))
            + _int(source.get("card_reward_group_count"))
        )
        * 0.08,
    )
    support += min(
        0.25,
        (
            _int(source.get("random_relic_count"))
            + _int(source.get("fixed_relic_count"))
            + len(_sequence(source.get("fixed_relic_ids")))
            + len(_sequence(source.get("relic_ids")))
        )
        * 0.10,
    )
    support += min(
        0.08,
        (
            _int(source.get("random_potion_count"))
            + _int(source.get("fixed_potion_count"))
            + len(_sequence(source.get("fixed_potion_ids")))
            + len(_sequence(source.get("potion_ids")))
        )
        * 0.04,
    )
    support += min(
        0.14,
        (
            _int(source.get("upgrade_random_count"))
            + _int(source.get("upgrade_random_card_count"))
            + _int(source.get("upgrade_card_count"))
        )
        * 0.05,
    )
    support += min(
        0.16,
        (
            _int(source.get("transform_random_count"))
            + _int(source.get("transform_random_card_count"))
            + _int(source.get("transform_card_count"))
        )
        * 0.06,
    )
    support += min(
        0.16,
        (
            _int(source.get("remove_random_count"))
            + _int(source.get("remove_random_card_count"))
            + _int(source.get("remove_card_count"))
            + len(_sequence(source.get("remove_card_ids")))
        )
        * 0.08,
    )
    for key in ("relic_id", "source_relic", "source_relic_id", "source_id"):
        support = max(support, _source_compensation_support(source.get(key)))
    return support


def _card_frontloaded_compensation_support(card: Mapping[str, Any]) -> float:
    custom = _mapping(card.get("custom"))
    support = 0.0
    support += min(0.35, _frontloaded_amount(card, "gold") / 1200.0)
    support += min(0.16, _frontloaded_amount(card, "max_hp") / 90.0)
    support += min(0.10, _frontloaded_amount(card, "heal") / 100.0)
    support += min(0.25, _frontloaded_amount(card, "relic_count") * 0.10)
    support += min(0.18, _frontloaded_amount(card, "card_reward_count") * 0.08)
    support += min(0.16, _frontloaded_amount(card, "transform_count") * 0.06)
    support += min(0.16, _frontloaded_amount(card, "remove_count") * 0.08)
    support += min(0.08, _frontloaded_amount(card, "potion_count") * 0.04)
    source_hint = max(
        _source_compensation_support(custom.get(key, card.get(key)))
        for key in (
            "source_relic",
            "source_relic_id",
            "source_event",
            "source_event_id",
            "source_id",
        )
    )
    return max(support, source_hint)


def _frontloaded_amount(card: Mapping[str, Any], key: str) -> float:
    custom = _mapping(card.get("custom"))
    aliases = (
        f"frontloaded_{key}",
        f"frontload_{key}",
        key if key.startswith("frontloaded_") else "",
    )
    return max(
        _float(custom.get(alias, card.get(alias)))
        for alias in aliases
        if alias
    )


def _source_compensation_support(source_id: object) -> float:
    return _CURSE_SOURCE_COMPENSATION_SUPPORT.get(_normalized_id(source_id), 0.0)


def _starter_deck_similarity_penalty(
    payload: Mapping[str, Any],
    config: LearningRewardConfig,
    tracker: LearningRewardTracker | None = None,
) -> float:
    if config.starter_deck_similarity_penalty >= 0:
        return 0.0
    summary = _starter_dependency_summary(payload, config, tracker=tracker)
    return _float(summary.get("penalty"))


def _starter_dependency_summary(
    payload: Mapping[str, Any],
    config: LearningRewardConfig,
    *,
    tracker: LearningRewardTracker | None = None,
) -> dict[str, Any]:
    cards = _master_deck_cards(payload)
    baseline_counts = _starter_baseline_counts(tracker)
    baseline_size = sum(baseline_counts.values())
    if not cards or baseline_size <= 0:
        return _starter_dependency_empty_summary(payload, baseline_counts)

    retained_weight, total_weight, duplicate_count = _starter_dependency_weights(
        cards,
        baseline_counts,
        config,
    )
    baseline_weight = max(1.0, float(baseline_size))
    deck_size = max(1.0, float(len(cards)))
    starter_retention = _clamp(retained_weight / baseline_weight)
    starter_share = _clamp(total_weight / deck_size)
    dependency = _clamp(
        _normalized_mix(
            (
                (starter_retention, config.starter_deck_retention_weight),
                (starter_share, config.starter_deck_share_weight),
            )
        )
    )

    threshold = _clamp(config.starter_deck_similarity_threshold, 0.0, 0.95)
    raw_severity = 0.0
    if dependency > threshold:
        raw_severity = (dependency - threshold) / max(0.01, 1.0 - threshold)

    deck_weakness = _starter_deck_weakness(payload, config)
    problem_factor = _starter_problem_factor(deck_weakness, config)
    mitigation = _starter_dependency_mitigation(cards, payload, config, tracker)
    outcome_scale = _starter_outcome_scale(payload, config)
    act_scale = _starter_act_scale(payload, config)
    floor_scale = _starter_floor_scale(payload, config)
    severity = max(0.0, raw_severity * problem_factor - mitigation)
    penalty = (
        config.starter_deck_similarity_penalty
        * severity
        * outcome_scale
        * act_scale
        * floor_scale
    )

    return {
        "baseline_counts": dict(baseline_counts),
        "baseline_size": baseline_size,
        "deck_size": len(cards),
        "retained_starter_weight": round(retained_weight, 6),
        "total_starter_weight": round(total_weight, 6),
        "duplicate_starter_count": duplicate_count,
        "starter_retention": round(starter_retention, 6),
        "starter_share": round(starter_share, 6),
        "dependency_score": round(dependency, 6),
        "threshold": round(threshold, 6),
        "raw_severity": round(raw_severity, 6),
        "deck_weakness": round(deck_weakness, 6),
        "problem_factor": round(problem_factor, 6),
        "mitigation": round(mitigation, 6),
        "outcome_scale": round(outcome_scale, 6),
        "act_scale": round(act_scale, 6),
        "floor_scale": round(floor_scale, 6),
        "severity": round(severity, 6),
        "penalty": round(penalty, 6),
    }


def _starter_dependency_empty_summary(
    payload: Mapping[str, Any],
    baseline_counts: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "baseline_counts": dict(baseline_counts),
        "baseline_size": sum(baseline_counts.values()),
        "deck_size": len(_master_deck_cards(payload)),
        "retained_starter_weight": 0.0,
        "total_starter_weight": 0.0,
        "duplicate_starter_count": 0,
        "starter_retention": 0.0,
        "starter_share": 0.0,
        "dependency_score": 0.0,
        "threshold": 0.0,
        "raw_severity": 0.0,
        "deck_weakness": 0.0,
        "problem_factor": 0.0,
        "mitigation": 0.0,
        "outcome_scale": 0.0,
        "act_scale": 0.0,
        "floor_scale": 0.0,
        "severity": 0.0,
        "penalty": 0.0,
    }


def _weighted_starter_card_count(
    cards: Sequence[Mapping[str, Any]],
    config: LearningRewardConfig,
) -> float:
    _retained_weight, total_weight, _duplicate_count = _starter_dependency_weights(
        cards,
        _STARTER_DECK_COUNTS,
        config,
    )
    return total_weight


def _starter_dependency_weights(
    cards: Sequence[Mapping[str, Any]],
    baseline_counts: Mapping[str, int],
    config: LearningRewardConfig,
) -> tuple[float, float, int]:
    seen: dict[str, int] = {}
    retained_weight = 0.0
    total_weight = 0.0
    duplicate_count = 0
    for card in cards:
        card_id = _normalized_id(card.get("card_id", card.get("id", "")))
        allowed = max(0, int(baseline_counts.get(card_id, 0)))
        if allowed <= 0:
            continue
        count = seen.get(card_id, 0)
        seen[card_id] = count + 1
        if count >= allowed:
            duplicate_count += 1
            total_weight += max(0.0, config.starter_deck_duplicate_card_weight)
            continue
        card_weight = _starter_card_dependency_weight(card, config)
        retained_weight += card_weight
        total_weight += card_weight
    return retained_weight, total_weight, duplicate_count


def _starter_card_dependency_weight(
    card: Mapping[str, Any],
    config: LearningRewardConfig,
) -> float:
    weight = max(0.0, config.starter_deck_plain_card_weight)
    if _starter_card_is_upgraded(card):
        weight = min(weight, _clamp(config.starter_deck_upgraded_card_weight, 0.0, 1.5))
    if _starter_card_is_enhanced(card):
        weight = min(weight, _clamp(config.starter_deck_improved_card_weight, 0.0, 1.5))
    return weight


def _starter_card_is_upgraded(card: Mapping[str, Any]) -> bool:
    return _bool(card.get("upgraded")) or any(
        _normalized_id(tag).replace(":", "_") == "upgraded"
        for tag in _sequence(card.get("tags"))
    )


def _starter_card_is_improved(card: Mapping[str, Any]) -> bool:
    return _starter_card_is_upgraded(card) or _starter_card_is_enhanced(card)


def _starter_card_is_enhanced(card: Mapping[str, Any]) -> bool:
    if _sequence(card.get("enchantments")):
        return True
    custom = _mapping(card.get("custom"))
    if any(
        key in custom
        for key in (
            "damage_bonus",
            "block_bonus",
            "cost_reduction",
            "free_to_play_this_combat",
            "free_to_play_this_turn",
            "innate",
            "retain",
            "strength_bonus",
        )
    ):
        return True
    return any(
        _normalized_id(tag).replace(":", "_")
        in {"enchanted", "innate", "retain"}
        for tag in _sequence(card.get("tags"))
    )


def _starter_dependency_mitigation(
    cards: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    config: LearningRewardConfig,
    tracker: LearningRewardTracker | None,
) -> float:
    return min(
        1.0,
        _starter_deck_capability_gain(cards, tracker) * config.starter_deck_capability_mitigation
        + _starter_relic_support_score(payload) * config.starter_deck_relic_mitigation,
    )


def _starter_deck_capability_gain(
    cards: Sequence[Mapping[str, Any]],
    tracker: LearningRewardTracker | None = None,
) -> float:
    baseline_score = _STARTER_DECK_CAPABILITY_SCORE
    if tracker is not None and tracker.starter_deck_counts:
        baseline_score = tracker.starter_deck_capability_score
    return max(
        0.0,
        _deck_capability_score({"master_deck": tuple(cards)})
        - baseline_score,
    )


def _starter_relic_support_score(payload: Mapping[str, Any]) -> float:
    score = 0.0
    for relic_id in _relic_ids(payload):
        profile = relic_mechanic_profile(relic_id)
        values = _mapping(profile.values)
        score += min(4.0, _float(values.get("block")) * 0.05)
        score += min(4.0, _float(values.get("damage")) * 0.05)
        score += min(4.0, _float(values.get("aoe_damage")) * 0.04)
        score += min(3.0, _float(values.get("strength")) * 0.6)
        score += min(3.0, _float(values.get("dexterity")) * 0.6)
        score += min(3.0, _float(values.get("energy")) * 0.6)
        score += min(3.0, _float(values.get("card_upgrade")) * 0.4)
        score += min(2.0, _float(values.get("repeating_effect")) * 0.2)
        score += min(2.0, _float(values.get("periodic_effect")) * 0.2)
    return score


def _starter_baseline_counts(tracker: LearningRewardTracker | None) -> Mapping[str, int]:
    if tracker is not None and tracker.starter_deck_counts:
        return tracker.starter_deck_counts
    return _STARTER_DECK_COUNTS


def _capture_starter_deck_baseline(
    payload: Mapping[str, Any],
    tracker: LearningRewardTracker,
) -> None:
    if tracker.starter_deck_counts:
        return
    if _int(payload.get("floor")) > 1:
        return
    cards = _master_deck_cards(payload)
    if not cards:
        return
    counts = _card_id_counts(cards)
    if not counts:
        return
    tracker.starter_deck_counts = counts
    tracker.starter_deck_size = sum(counts.values())
    tracker.starter_deck_capability_score = _deck_capability_score(
        {"master_deck": tuple(cards)}
    )


def _card_id_counts(cards: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for card in cards:
        card_id = _normalized_id(card.get("card_id", card.get("id", "")))
        if card_id:
            counts[card_id] = counts.get(card_id, 0) + 1
    return counts


def _starter_deck_weakness(
    payload: Mapping[str, Any],
    config: LearningRewardConfig,
) -> float:
    metrics = _deck_metrics(payload, config)
    problems = _deck_problem_scores(payload, metrics, config)
    relevant_weights = {
        key: weight
        for key, weight in _DECK_PROBLEM_WEIGHTS.items()
        if key != "starter_density"
    }
    total_weight = sum(relevant_weights.values())
    if total_weight <= 0:
        return 0.0
    weighted = sum(
        _float(problems.get(key)) * weight
        for key, weight in relevant_weights.items()
    )
    return _clamp(weighted / total_weight)


def _starter_problem_factor(
    deck_weakness: float,
    config: LearningRewardConfig,
) -> float:
    threshold = max(0.0, config.starter_deck_problem_threshold)
    if deck_weakness < threshold:
        return 0.0
    return _clamp(max(config.starter_deck_problem_floor, deck_weakness))


def _starter_outcome_scale(
    payload: Mapping[str, Any],
    config: LearningRewardConfig,
) -> float:
    phase = _phase(payload)
    if phase == "failed":
        return 1.0
    if phase == "complete":
        return _clamp(config.starter_deck_complete_penalty_scale, 0.0, 1.0)
    return 0.0


def _starter_act_scale(payload: Mapping[str, Any], config: LearningRewardConfig) -> float:
    act = max(1, _int(payload.get("act"), 1))
    if act <= 1:
        return _clamp(config.starter_deck_act1_penalty_scale, 0.0, 1.0)
    if act == 2:
        return _clamp(config.starter_deck_act2_penalty_scale, 0.0, 1.0)
    return _clamp(config.starter_deck_act3_penalty_scale, 0.0, 1.0)


def _starter_floor_scale(payload: Mapping[str, Any], config: LearningRewardConfig) -> float:
    floor = _int(payload.get("floor"))
    floor_gate = max(0, config.starter_deck_similarity_floor)
    if floor < floor_gate:
        return 0.0
    if floor_gate <= 0:
        return 1.0
    return _clamp((floor - floor_gate + 1) / 6.0)


def _added_deck_cards(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    previous_counts: dict[str, int] = {}
    for card in _master_deck_cards(previous):
        key = _card_identity_key(card)
        previous_counts[key] = previous_counts.get(key, 0) + 1

    added: list[Mapping[str, Any]] = []
    for card in _master_deck_cards(current):
        key = _card_identity_key(card)
        available = previous_counts.get(key, 0)
        if available > 0:
            previous_counts[key] = available - 1
        else:
            added.append(card)
    return tuple(added)


def _card_identity_key(card: Mapping[str, Any]) -> str:
    instance_id = str(card.get("instance_id", "") or "")
    if instance_id:
        return f"instance:{instance_id}"
    return f"card:{_normalized_id(card.get('card_id', card.get('id', '')))}"


def _card_is_burden(card: Mapping[str, Any]) -> bool:
    return (
        _normalized_id(card.get("type", card.get("card_type"))) == "curse"
        or _card_has_marker(card, "curse")
        or _card_has_marker(card, "unplayable")
    )


def _card_has_marker(card: Mapping[str, Any], marker: str) -> bool:
    normalized_marker = _normalized_id(marker)
    if _bool(card.get(normalized_marker)):
        return True
    custom = _mapping(card.get("custom"))
    if _bool(custom.get(normalized_marker)):
        return True
    for tag in _sequence(card.get("tags")):
        if _normalized_id(tag).replace(":", "_") == normalized_marker:
            return True
    return False


def _deck_capability_score(payload: Mapping[str, Any]) -> float:
    return _float(_deck_metrics(payload, DEFAULT_REWARD_CONFIG).get("score"))


def _relic_mechanic_values(payload: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for relic_id in _relic_ids(payload):
        profile = relic_mechanic_profile(relic_id)
        for key, value in profile.values.items():
            values[str(key)] = values.get(str(key), 0.0) + _float(value)
    return values


def _card_play_cost(card: Mapping[str, Any]) -> float | None:
    if _card_is_burden(card):
        return None
    raw = card.get("cost")
    if raw is None:
        return 1.0
    if isinstance(raw, str):
        stripped = raw.strip().lower()
        if stripped == "x":
            return 1.5
        try:
            raw = float(stripped)
        except ValueError:
            return 1.0
    if isinstance(raw, bool):
        return 1.0
    if isinstance(raw, int | float):
        value = float(raw)
        if value < 0:
            return 1.5
        return value
    return 1.0


def _hp_fraction(payload: Mapping[str, Any]) -> float:
    player = _player(payload)
    hp = _int(player.get("hp"))
    max_hp = max(1, _int(player.get("max_hp"), hp or 1))
    return _clamp(hp / max_hp)


def _shortfall(value: float, target: float) -> float:
    if target <= 0:
        return 0.0
    return max(0.0, target - value) / target


def _rounded_float_dict(values: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): round(_float(value), 6) for key, value in values.items()}


def _normalized_mix(values: Sequence[tuple[float, float]]) -> float:
    total_weight = sum(max(0.0, weight) for _value, weight in values)
    if total_weight <= 0:
        return 0.0
    return sum(value * max(0.0, weight) for value, weight in values) / total_weight


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


def _potion_slots_available(payload: Mapping[str, Any]) -> bool:
    player = _player(payload)
    potion_count = len(_sequence(player.get("potions", payload.get("potions"))))
    reward = _mapping(payload.get("reward"))
    slots = _int(_mapping(reward.get("metadata")).get("potion_slots"), 3)
    return potion_count < max(0, slots)


def _refresh_tracker_after_transition(
    current: Mapping[str, Any],
    tracker: LearningRewardTracker,
) -> None:
    _capture_starter_deck_baseline(current, tracker)
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


def _player_hp(payload: Mapping[str, Any]) -> int:
    return _int(_player(payload).get("hp"))


def _player_max_hp(payload: Mapping[str, Any]) -> int:
    return _int(_player(payload).get("max_hp"))


def _potion_count(payload: Mapping[str, Any]) -> int:
    player = _player(payload)
    potions = _sequence(player.get("potions", payload.get("potions")))
    return len(potions)


def _master_deck_cards(payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    player = _player(payload)
    cards = _sequence(payload.get("master_deck"))
    if not cards:
        cards = _sequence(player.get("deck"))
    return tuple(_mapping(card) for card in cards)


def _master_deck_count(payload: Mapping[str, Any]) -> int:
    cards = _master_deck_cards(payload)
    if cards:
        return len(cards)
    return _int(_player(payload).get("deck_count"))


def _relic_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    player = _player(payload)
    relics = _sequence(player.get("relics", payload.get("relics")))
    ids: list[str] = []
    for relic in relics:
        relic_id = relic.get("relic_id", relic.get("id")) if isinstance(relic, Mapping) else relic
        normalized = _normalized_id(relic_id)
        if normalized:
            ids.append(normalized)
    return tuple(ids)


def _added_relic_ids(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> tuple[str, ...]:
    previous_counts: dict[str, int] = {}
    for relic_id in _relic_ids(previous):
        previous_counts[relic_id] = previous_counts.get(relic_id, 0) + 1

    added: list[str] = []
    for relic_id in _relic_ids(current):
        available = previous_counts.get(relic_id, 0)
        if available > 0:
            previous_counts[relic_id] = available - 1
        else:
            added.append(relic_id)
    return tuple(added)


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


def _normalized_id(value: object) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("'", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


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
