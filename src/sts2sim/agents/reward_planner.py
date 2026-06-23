"""Reward action scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .evaluators import action_payload, action_type, make_score, mapping, normalized
from .models import ActionDescriptor, DecisionContext, ScoredAction


class RewardPlanner:
    """Score combat/event/treasure reward decisions."""

    def score(self, context: DecisionContext, descriptor: ActionDescriptor) -> ScoredAction:
        kind = action_type(descriptor)
        if kind == "take_reward_gold":
            return make_score(
                descriptor,
                score=24.0,
                category="reward",
                reasons=("gold_is_low_risk_flexible_value",),
            )
        if kind == "take_reward_relic":
            return make_score(
                descriptor,
                score=40.0,
                category="reward",
                reasons=("relics_are_high_value_and_do_not_bloat_deck",),
            )
        if kind == "take_reward_potion":
            return self._score_potion(context, descriptor)
        if kind == "take_reward_card":
            return self._score_card(context, descriptor)
        if kind == "skip_reward":
            return self._score_skip_reward(context, descriptor)
        if kind == "proceed":
            return self._score_proceed(context, descriptor)
        return make_score(
            descriptor,
            score=0.0,
            category="reward",
            reasons=("reward_action_without_special_case",),
        )

    def _score_potion(
        self,
        context: DecisionContext,
        descriptor: ActionDescriptor,
    ) -> ScoredAction:
        score = 16.0
        reasons = ["potion_improves_future_combat_lines"]
        capacity = context.plan.economy.potion_capacity
        if capacity is not None and context.plan.economy.potion_count >= capacity:
            score -= 20.0
            reasons.append("potion_slots_are_full")
        if context.plan.risk_level in {"high", "critical"}:
            score += 6.0
            reasons.append("high_risk_values_potion")
        return make_score(
            descriptor,
            score=score,
            category="reward",
            reasons=tuple(reasons),
        )

    def _score_card(self, context: DecisionContext, descriptor: ActionDescriptor) -> ScoredAction:
        card_id = _reward_card_id(
            context.state_summary,
            action_payload(descriptor).get("target_id"),
        )
        normalized_card = normalized(card_id)
        score = 11.0
        reasons = ["card_reward_can_improve_deck"]
        if any(term in normalized_card for term in ("strike", "defend", "basic")):
            score -= 8.0
            reasons.append("basic_like_card_is_low_priority")
        if "frontload_damage" in context.plan.must_find and _looks_like_damage(normalized_card):
            score += 8.0
            reasons.append("matches_frontload_damage_need")
        if "block" in context.plan.must_find and _looks_like_block(normalized_card):
            score += 8.0
            reasons.append("matches_block_need")
        if "card_draw" in context.plan.must_find and _looks_like_draw(normalized_card):
            score += 7.0
            reasons.append("matches_draw_need")
        if "deck_bloat" in context.plan.avoid:
            score -= 4.0
            reasons.append("large_deck_penalizes_optional_cards")
        return make_score(
            descriptor,
            score=score,
            category="reward",
            reasons=tuple(reasons),
        )

    def _score_proceed(
        self,
        context: DecisionContext,
        descriptor: ActionDescriptor,
    ) -> ScoredAction:
        reward = mapping(context.state_summary.get("reward"))
        forced = bool(reward.get("forced"))
        score = -20.0 if forced else 4.0
        reasons = ["skip_optional_remaining_reward"]
        if forced:
            reasons.append("forced_reward_should_be_resolved")
        if "deck_bloat" in context.plan.avoid:
            score += 4.0
            reasons.append("skipping_can_protect_deck_quality")
        return make_score(
            descriptor,
            score=score,
            category="reward",
            reasons=tuple(reasons),
        )

    def _score_skip_reward(
        self,
        context: DecisionContext,
        descriptor: ActionDescriptor,
    ) -> ScoredAction:
        reward_choice = mapping(descriptor.get("reward_choice"))
        skip_kind = str(reward_choice.get("skip_kind", ""))
        score = 3.0
        reasons = ["skip_one_optional_reward_entry"]
        if skip_kind in {"card_options", "card_group", "fixed_card"}:
            score += 2.0
            reasons.append("targeted_skip_can_protect_deck_quality")
            if "deck_bloat" in context.plan.avoid:
                score += 5.0
                reasons.append("deck_bloat_plan_prefers_skipping_low_value_cards")
        elif skip_kind == "potion":
            score -= 4.0
            reasons.append("potions_are_often_useful_if_slots_allow")
        elif skip_kind == "relic":
            score -= 18.0
            reasons.append("relic_skip_is_usually_bad")
        elif skip_kind == "gold":
            score -= 12.0
            reasons.append("gold_skip_is_usually_bad")
        return make_score(
            descriptor,
            score=score,
            category="reward",
            reasons=tuple(reasons),
        )


def _reward_card_id(state_summary: Mapping[str, Any], target_id: object) -> str:
    reward = mapping(state_summary.get("reward"))
    target = str(target_id or "")
    parts = target.split(":")
    if len(parts) >= 3 and parts[1] == "card":
        return _index_lookup(_sequence(reward.get("card_options")), parts[2])
    if len(parts) >= 4 and parts[1] == "card_group":
        group_index = _optional_int(parts[2])
        card_index = _optional_int(parts[3])
        groups = _sequence(reward.get("card_option_groups"))
        if group_index is not None and 0 <= group_index < len(groups):
            return _index_lookup(_sequence(groups[group_index]), str(card_index))
    if len(parts) >= 3 and parts[1] == "fixed_card":
        return _index_lookup(_sequence(reward.get("card_ids")), parts[2])
    return target


def _index_lookup(values: Sequence[object], raw_index: str) -> str:
    index = _optional_int(raw_index)
    if index is None or index < 0 or index >= len(values):
        return ""
    return str(values[index])


def _looks_like_damage(card_id: str) -> bool:
    return any(term in card_id for term in ("strike", "slam", "bash", "slash", "blow", "attack"))


def _looks_like_block(card_id: str) -> bool:
    return any(term in card_id for term in ("defend", "guard", "block", "shrug", "wall"))


def _looks_like_draw(card_id: str) -> bool:
    return any(term in card_id for term in ("draw", "pommel", "skim", "shrug", "acrobatics"))


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()
