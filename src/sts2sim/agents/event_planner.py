"""Event and ancient choice scoring."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .evaluators import action_payload, action_type, make_score, mapping, normalized
from .models import ActionDescriptor, DecisionContext, ScoredAction


class EventPlanner:
    """Score event-like choices including Ancient starting choices."""

    def score(self, context: DecisionContext, descriptor: ActionDescriptor) -> ScoredAction:
        kind = action_type(descriptor)
        if kind == "choose_ancient":
            return self._score_ancient(context, descriptor)
        if kind == "choose_event":
            return self._score_event(context, descriptor)
        if kind == "proceed":
            return make_score(
                descriptor,
                score=4.0,
                category="event",
                reasons=("event_progress_action",),
            )
        return make_score(
            descriptor,
            score=0.0,
            category="event",
            reasons=("event_action_without_special_case",),
        )

    def _score_ancient(
        self,
        context: DecisionContext,
        descriptor: ActionDescriptor,
    ) -> ScoredAction:
        option = _ancient_option(context, action_payload(descriptor).get("target_id"))
        relic_id = normalized(option.get("relic_id"))
        kind = normalized(option.get("kind"))
        score = 20.0
        reasons = ["ancient_choice_is_forced_starting_plan"]
        if kind == "positive_relic":
            score += 10.0
            reasons.append("positive_relic_has_low_downside")
        if kind == "curse_relic":
            score += 4.0
            reasons.append("curse_relic_may_be_stronger_but_costly")
            if context.plan.deck.total_cards <= 12:
                score -= 4.0
                reasons.append("thin_deck_dislikes_early_curse")
        if any(term in relic_id for term in ("anchor", "blood", "vial", "lantern")):
            score += 3.0
            reasons.append("early_survival_relic")
        return make_score(
            descriptor,
            score=score,
            category="ancient",
            reasons=tuple(reasons),
        )

    def _score_event(
        self,
        context: DecisionContext,
        descriptor: ActionDescriptor,
    ) -> ScoredAction:
        option = _event_option(context, action_payload(descriptor).get("target_id"))
        text = normalized(" ".join(str(value) for value in option.values()))
        score = 8.0
        reasons = ["event_choice_progress"]
        if "remove" in text and context.plan.remove_targets:
            score += 12.0
            reasons.append("event_removal_matches_plan")
        if "upgrade" in text:
            score += 8.0
            reasons.append("upgrade_event_improves_deck")
        if "lose" in text and "hp" in text and context.plan.risk_level in {"high", "critical"}:
            score -= 20.0
            reasons.append("low_hp_avoids_hp_loss_event")
        if "fight" in text and context.plan.risk_level in {"high", "critical"}:
            score -= 8.0
            reasons.append("risk_state_avoids_optional_fight")
        return make_score(
            descriptor,
            score=score,
            category="event",
            reasons=tuple(reasons),
        )


def _ancient_option(context: DecisionContext, target_id: object) -> dict[str, Any]:
    ancient = mapping(context.state_summary.get("ancient"))
    return _option_by_id(_sequence(ancient.get("options")), target_id, "option_id")


def _event_option(context: DecisionContext, target_id: object) -> dict[str, Any]:
    event = mapping(context.state_summary.get("event"))
    return _option_by_id(_sequence(event.get("options")), target_id, "option_id")


def _option_by_id(options: Sequence[object], target_id: object, key: str) -> dict[str, Any]:
    wanted = str(target_id or "")
    for raw_option in options:
        option = mapping(raw_option)
        if str(option.get(key, "")) == wanted:
            return option
    return {}


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()
