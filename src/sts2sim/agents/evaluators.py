"""Evaluator dispatch and shared scoring helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .models import ActionDescriptor, DecisionContext, ScoredAction


class ActionEvaluator(Protocol):
    """Protocol implemented by phase-specific action evaluators."""

    def score(self, context: DecisionContext, descriptor: ActionDescriptor) -> ScoredAction:
        """Score one legal action."""


def action_id(descriptor: Mapping[str, Any]) -> int:
    """Return the state-local action id from an action descriptor."""

    value = descriptor.get("id", 0)
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def action_type(descriptor: Mapping[str, Any]) -> str:
    """Return the engine action type from an action descriptor."""

    value = descriptor.get("type")
    if value is not None:
        return str(value)
    action = descriptor.get("action")
    if isinstance(action, Mapping):
        return str(action.get("type", "unknown"))
    return "unknown"


def action_payload(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Return the JSON action payload from an action descriptor."""

    action = descriptor.get("action")
    if isinstance(action, Mapping):
        return {str(key): value for key, value in action.items()}
    return {}


def make_score(
    descriptor: ActionDescriptor,
    *,
    score: float,
    category: str,
    reasons: tuple[str, ...],
) -> ScoredAction:
    """Build a scored action using standard descriptor fields."""

    return ScoredAction(
        action_id=action_id(descriptor),
        action_type=action_type(descriptor),
        action=action_payload(descriptor),
        score=round(float(score), 4),
        category=category,
        reasons=reasons,
    )


def generic_score(context: DecisionContext, descriptor: ActionDescriptor) -> ScoredAction:
    """Fallback score for legal actions that do not yet have a specialist planner."""

    del context
    kind = action_type(descriptor)
    if kind == "proceed":
        return make_score(
            descriptor,
            score=1.0,
            category="generic",
            reasons=("legal_progress_action",),
        )
    if kind.startswith("choose_"):
        return make_score(
            descriptor,
            score=5.0,
            category="generic",
            reasons=("required_choice_progress",),
        )
    return make_score(
        descriptor,
        score=0.0,
        category="generic",
        reasons=("legal_action_no_specialized_score",),
    )


def mapping(value: object) -> dict[str, Any]:
    """Normalize an arbitrary mapping-like object to a string-keyed dict."""

    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def normalized(value: object) -> str:
    """Normalize ids/names for heuristic matching."""

    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def number(value: object, default: float = 0.0) -> float:
    """Parse a number for heuristic scoring."""

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
