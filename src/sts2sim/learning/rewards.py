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
    card_pickup_reward: float = 0.15
    relic_pickup_reward: float = 1.0
    potion_pickup_reward: float = 0.25
    card_remove_reward: float = 0.35
    skip_gold_penalty: float = -0.2
    skip_relic_penalty: float = -1.0
    skip_potion_penalty: float = -0.25
    early_card_skip_penalty: float = -0.2
    early_card_skip_deck_size: int = 14
    deck_capability_reward_weight: float = 0.04
    deck_capability_reward_cap: float = 0.8
    curse_pickup_penalty: float = -1.2
    eternal_curse_extra_penalty: float = -0.3
    curse_burden_reference_deck_size: int = 10
    starter_deck_similarity_penalty: float = -1.5
    starter_deck_similarity_floor: int = 3
    starter_deck_similarity_threshold: float = 0.55
    starter_deck_improved_card_weight: float = 0.35
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
_STARTER_DECK_COUNTS = {"strike": 5, "defend": 4, "bash": 1}
_STARTER_DECK_SIZE = sum(_STARTER_DECK_COUNTS.values())
_STARTER_DECK_CAPABILITY_SCORE = 5.9


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
    resource_pickup = _resource_pickup_reward(previous, current, descriptor, config)
    reward_skip = _reward_skip_penalty(previous, descriptor, config)
    deck_capability = _deck_capability_reward(previous, current, config)
    deck_burden = _deck_burden_penalty(previous, current, config)
    starter_similarity = _starter_deck_similarity_penalty(current, config)

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
        return config.card_pickup_reward
    if action_type == "take_reward_relic":
        return config.relic_pickup_reward
    if action_type == "take_reward_potion":
        return config.potion_pickup_reward
    if action_type in {"shop_buy", "take_reward_card_removal"}:
        item = _mapping(action_descriptor.get("item"))
        kind = str(item.get("kind", ""))
        if kind in {"card", "colorless_card"}:
            return config.card_pickup_reward
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
    if str(action_descriptor.get("type", "")) != "skip_reward":
        return 0.0
    reward_choice = _mapping(action_descriptor.get("reward_choice"))
    skip_kind = str(reward_choice.get("skip_kind", reward_choice.get("kind", "")))
    if skip_kind == "gold":
        return config.skip_gold_penalty
    if skip_kind == "relic":
        return config.skip_relic_penalty
    if skip_kind == "potion":
        return config.skip_potion_penalty if _potion_slots_available(previous) else 0.0
    if skip_kind in {"card_options", "card_group", "fixed_card", "card"}:
        if _master_deck_count(previous) <= max(0, config.early_card_skip_deck_size):
            return config.early_card_skip_penalty
    return 0.0


def _deck_capability_reward(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    config: LearningRewardConfig,
) -> float:
    if config.deck_capability_reward_weight <= 0:
        return 0.0
    delta = _deck_capability_score(current) - _deck_capability_score(previous)
    if delta <= 0:
        return 0.0
    return min(
        config.deck_capability_reward_cap,
        delta * config.deck_capability_reward_weight,
    )


def _deck_burden_penalty(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    config: LearningRewardConfig,
) -> float:
    if config.curse_pickup_penalty >= 0:
        return 0.0
    before_count = max(1.0, float(_master_deck_count(previous)))
    density = min(
        1.0,
        max(1.0, float(config.curse_burden_reference_deck_size)) / before_count,
    )
    penalty = 0.0
    for card in _added_deck_cards(previous, current):
        if not _card_is_burden(card):
            continue
        card_penalty = config.curse_pickup_penalty
        if _card_has_marker(card, "eternal"):
            card_penalty += config.eternal_curse_extra_penalty
        penalty += card_penalty * density
    return penalty


def _starter_deck_similarity_penalty(
    payload: Mapping[str, Any],
    config: LearningRewardConfig,
) -> float:
    if config.starter_deck_similarity_penalty >= 0:
        return 0.0
    if _phase(payload) not in {"failed", "complete"}:
        return 0.0
    if _int(payload.get("floor")) < max(0, config.starter_deck_similarity_floor):
        return 0.0

    cards = _master_deck_cards(payload)
    if not cards:
        return 0.0

    weighted_starter = _weighted_starter_card_count(cards, config)
    similarity = weighted_starter / max(float(len(cards)), float(_STARTER_DECK_SIZE))
    threshold = _clamp(config.starter_deck_similarity_threshold, 0.0, 0.95)
    if similarity <= threshold:
        return 0.0

    severity = (similarity - threshold) / max(0.01, 1.0 - threshold)
    mitigation = min(
        1.0,
        _starter_deck_capability_gain(cards) * config.starter_deck_capability_mitigation
        + _starter_relic_support_score(payload) * config.starter_deck_relic_mitigation,
    )
    severity = max(0.0, severity - mitigation)
    return config.starter_deck_similarity_penalty * severity


def _weighted_starter_card_count(
    cards: Sequence[Mapping[str, Any]],
    config: LearningRewardConfig,
) -> float:
    seen: dict[str, int] = {}
    weighted = 0.0
    for card in cards:
        card_id = _normalized_id(card.get("card_id", card.get("id", "")))
        allowed = _STARTER_DECK_COUNTS.get(card_id, 0)
        if allowed <= 0:
            continue
        count = seen.get(card_id, 0)
        if count >= allowed:
            continue
        seen[card_id] = count + 1
        weighted += (
            _clamp(config.starter_deck_improved_card_weight, 0.0, 1.0)
            if _starter_card_is_improved(card)
            else 1.0
        )
    return weighted


def _starter_card_is_improved(card: Mapping[str, Any]) -> bool:
    if _bool(card.get("upgraded")):
        return True
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
        in {"enchanted", "innate", "retain", "upgraded"}
        for tag in _sequence(card.get("tags"))
    )


def _starter_deck_capability_gain(cards: Sequence[Mapping[str, Any]]) -> float:
    return max(
        0.0,
        _deck_capability_score({"master_deck": tuple(cards)})
        - _STARTER_DECK_CAPABILITY_SCORE,
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
    cards = _master_deck_cards(payload)
    if not cards:
        return 0.0
    totals = {
        "frontload": 0.0,
        "block": 0.0,
        "draw": 0.0,
        "energy": 0.0,
        "scaling": 0.0,
        "exhaust": 0.0,
        "retain": 0.0,
        "status_enemy": 0.0,
    }
    for card in cards:
        profile = card_mechanic_profile(card)
        values = _mapping(profile.values)
        damage = _float(values.get("damage")) + _float(values.get("aoe_damage"))
        totals["frontload"] += min(16.0, damage)
        totals["block"] += min(16.0, _float(values.get("block")))
        totals["draw"] += min(5.0, _float(values.get("draw")))
        totals["energy"] += min(3.0, _float(values.get("energy")))
        totals["scaling"] += min(
            6.0,
            _float(values.get("strength"))
            + _float(values.get("dexterity"))
            + _float(values.get("focus"))
            + _float(values.get("repeating_effect"))
            + _float(values.get("periodic_effect")),
        )
        totals["exhaust"] += min(3.0, _float(values.get("exhaust")))
        totals["retain"] += min(3.0, _float(values.get("retain")))
        totals["status_enemy"] += min(
            5.0,
            _float(values.get("weak"))
            + _float(values.get("vulnerable"))
            + _float(values.get("poison"))
            + _float(values.get("status_enemy")),
        )
    deck_size = max(1.0, float(len(cards)))
    bloat_penalty = max(0.0, deck_size - 18.0) * 0.25
    return (
        totals["frontload"] * 0.10
        + totals["block"] * 0.08
        + totals["draw"] * 0.55
        + totals["energy"] * 0.70
        + totals["scaling"] * 0.65
        + totals["exhaust"] * 0.35
        + totals["retain"] * 0.25
        + totals["status_enemy"] * 0.25
        - bloat_penalty
    )


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
        if isinstance(relic, Mapping):
            relic_id = relic.get("relic_id", relic.get("id"))
        else:
            relic_id = relic
        normalized = _normalized_id(relic_id)
        if normalized:
            ids.append(normalized)
    return tuple(ids)


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
