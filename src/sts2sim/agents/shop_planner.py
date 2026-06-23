"""Shop action scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .evaluators import action_payload, action_type, make_score, mapping, number
from .models import ActionDescriptor, DecisionContext, ScoredAction


class ShopPlanner:
    """Score shop purchases and exits."""

    def score(self, context: DecisionContext, descriptor: ActionDescriptor) -> ScoredAction:
        kind = action_type(descriptor)
        if kind == "shop_buy":
            return self._score_buy(context, descriptor)
        if kind in {"shop_leave", "proceed"}:
            return make_score(
                descriptor,
                score=5.0,
                category="shop",
                reasons=("leave_shop_preserves_gold",),
            )
        if kind == "throw_potion_at_merchant":
            return make_score(
                descriptor,
                score=30.0,
                category="shop",
                reasons=("special_merchant_interaction_available",),
            )
        return make_score(
            descriptor,
            score=0.0,
            category="shop",
            reasons=("shop_action_without_special_case",),
        )

    def _score_buy(
        self,
        context: DecisionContext,
        descriptor: ActionDescriptor,
    ) -> ScoredAction:
        target_id = action_payload(descriptor).get("target_id")
        item = _shop_item(context.state_summary, target_id)
        kind = str(item.get("kind", "unknown"))
        price = number(item.get("price"))
        gold = context.plan.economy.gold
        score = 0.0
        reasons: list[str] = []

        if price > gold:
            score -= 100.0
            reasons.append("unaffordable_purchase")
        elif kind == "card_removal":
            score += 28.0 if context.plan.remove_targets else 10.0
            reasons.append("card_remove_improves_deck_quality")
        elif kind in {"relic", "shop_relic"}:
            score += 34.0
            reasons.append("relic_purchase_has_no_deck_bloat")
        elif kind in {"potion"}:
            score += 15.0
            reasons.append("potion_purchase_improves_combat_options")
        elif kind in {"card", "colorless_card"}:
            score += 11.0
            reasons.append("card_purchase_can_patch_plan_need")
            if "deck_bloat" in context.plan.avoid:
                score -= 6.0
                reasons.append("large_deck_penalizes_shop_cards")
        else:
            score += 4.0
            reasons.append("generic_shop_item")

        score -= price / 35.0
        if gold - price < 75:
            score -= 2.0
            reasons.append("purchase_reduces_future_flexibility")
        return make_score(
            descriptor,
            score=score,
            category="shop",
            reasons=tuple(reasons),
        )


def _shop_item(state_summary: Mapping[str, Any], target_id: object) -> dict[str, Any]:
    shop = mapping(state_summary.get("shop"))
    items = _sequence(shop.get("items"))
    index = _shop_target_index(str(target_id or ""))
    if index is not None and 0 <= index < len(items):
        return mapping(items[index])
    return {}


def _shop_target_index(target_id: str) -> int | None:
    for piece in target_id.split(":"):
        try:
            return int(piece)
        except ValueError:
            continue
    return None


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()
