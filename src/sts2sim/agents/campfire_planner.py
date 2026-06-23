"""Campfire action scoring."""

from __future__ import annotations

from .evaluators import action_payload, action_type, make_score, mapping, normalized
from .models import ActionDescriptor, DecisionContext, ScoredAction


class CampfirePlanner:
    """Score rest-site choices."""

    def score(self, context: DecisionContext, descriptor: ActionDescriptor) -> ScoredAction:
        kind = action_type(descriptor)
        if kind in {"rest", "proceed"}:
            score = 12.0
            reasons = ["rest_restores_hp"]
            if context.plan.hp_ratio < context.plan.aggression.hp_floor:
                score += 24.0
                reasons.append("below_aggression_hp_floor_prioritizes_rest")
            elif context.plan.aggression.allow_chip_damage:
                score -= 6.0
                reasons.append("hp_spend_budget_makes_upgrade_more_attractive")
            elif context.plan.hp_ratio > 0.8:
                score -= 5.0
                reasons.append("high_hp_makes_rest_less_valuable")
            return make_score(descriptor, score=score, category="campfire", reasons=tuple(reasons))
        if kind == "smith":
            target_id = str(action_payload(descriptor).get("target_id", ""))
            card_id = _master_card_id(context, target_id)
            score = 18.0
            reasons = ["upgrade_improves_future_combat"]
            if normalized(card_id) in {normalized(item) for item in context.plan.upgrade_targets}:
                score += 10.0
                reasons.append("matches_upgrade_priority")
            if context.plan.aggression.target >= 0.65 and context.plan.aggression.allow_chip_damage:
                score += 5.0
                reasons.append("aggressive_hp_budget_favors_power_gain")
            if context.plan.hp_ratio < context.plan.aggression.hp_floor:
                score -= 14.0
                reasons.append("below_hp_floor_may_need_rest_instead")
            return make_score(descriptor, score=score, category="campfire", reasons=tuple(reasons))
        if kind == "toke":
            score = 20.0 if context.plan.remove_targets else 8.0
            return make_score(
                descriptor,
                score=score,
                category="campfire",
                reasons=("campfire_remove_improves_deck",),
            )
        if kind in {"dig", "lift", "recall"}:
            return make_score(
                descriptor,
                score=14.0,
                category="campfire",
                reasons=(f"{kind}_special_campfire_option",),
            )
        return make_score(
            descriptor,
            score=0.0,
            category="campfire",
            reasons=("campfire_action_without_special_case",),
        )


def _master_card_id(context: DecisionContext, instance_id: str) -> str:
    for raw_card in context.state_summary.get("master_deck", ()):
        card = mapping(raw_card)
        if str(card.get("instance_id", "")) == instance_id:
            return str(card.get("card_id", card.get("name", "")))
    return instance_id
