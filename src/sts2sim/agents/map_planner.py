"""Map path action scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .evaluators import action_payload, make_score, mapping, normalized
from .models import ActionDescriptor, DecisionContext, ScoredAction


class MapPlanner:
    """Score reachable map nodes from the current run plan."""

    def score(self, context: DecisionContext, descriptor: ActionDescriptor) -> ScoredAction:
        payload = action_payload(descriptor)
        node = _map_node(context.state_summary, payload.get("target_id"))
        kind = normalized(node.get("kind"))
        score = 5.0
        reasons: list[str] = []

        if kind == "boss":
            score += 50.0
            reasons.append("boss_is_required_progress")
        elif kind == "rest":
            if context.plan.hp_ratio < context.plan.aggression.hp_floor:
                score += 31.0
                reasons.append("below_aggression_hp_floor_values_rest_site")
            elif context.plan.aggression.target >= 0.7 and context.plan.aggression.hp_spend_budget:
                score += 7.0
                reasons.append("aggressive_plan_uses_rest_as_future_safety")
            else:
                score += 11.0
                reasons.append("rest_site_enables_upgrade_or_rest")
        elif kind == "elite":
            if context.plan.elite_budget <= 0:
                score -= 15.0
                reasons.append("plan_has_no_elite_budget")
            elif context.plan.hp_ratio < context.plan.aggression.hp_floor:
                score -= 8.0
                reasons.append("elite_risk_exceeds_current_hp_floor")
            else:
                score += (
                    18.0
                    + context.plan.elite_budget * 4.0
                    + context.plan.aggression.target * 8.0
                )
                reasons.append("aggression_budget_supports_relic_hunt")
                if context.plan.aggression.known_elite_id:
                    reasons.append(f"known_elite_{context.plan.aggression.known_elite_id}")
        elif kind == "treasure":
            score += 22.0
            reasons.append("treasure_improves_relic_count")
        elif kind == "shop":
            score += min(18.0, context.plan.economy.gold / 15.0)
            reasons.append("shop_value_scales_with_gold")
            if context.plan.economy.gold < 75:
                score -= 8.0
                reasons.append("insufficient_gold_for_shop")
        elif kind == "event":
            score += 10.0
            reasons.append("event_has_flexible_upside")
            if "card_remove" in context.plan.must_find:
                score += 2.0
                reasons.append("events_can_solve_deck_problems")
        elif kind == "monster":
            score += 12.0
            reasons.append("monster_reward_builds_deck")
            if context.plan.risk_level in {"high", "critical"}:
                score -= 6.0
                reasons.append("danger_state_penalizes_extra_fights")
            if context.plan.aggression.target >= 0.7:
                score += 4.0
                reasons.append("aggressive_plan_accepts_reward_fights")
            if context.plan.aggression.nearest_rest_distance in {1, 2}:
                score += 2.0
                reasons.append("nearby_rest_allows_hp_spend")
        else:
            reasons.append("unknown_node_kind_uses_progress_score")

        return make_score(
            descriptor,
            score=score,
            category="map",
            reasons=tuple(reasons),
        )


def _map_node(state_summary: Mapping[str, Any], node_id: object) -> dict[str, Any]:
    wanted = str(node_id or "")
    map_state = mapping(state_summary.get("map"))
    for raw_node in _sequence(map_state.get("nodes")):
        node = mapping(raw_node)
        if str(node.get("node_id", "")) == wanted:
            return node
    return {}


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()
