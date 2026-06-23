"""Strategic agent policy orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sts2sim.agent_api import action_space, encode_observation
from sts2sim.api import serialize

from .campfire_planner import CampfirePlanner
from .combat_search import CombatSearchConfig, CombatSearchPlanner
from .evaluators import action_type, generic_score
from .event_planner import EventPlanner
from .map_planner import MapPlanner
from .models import ActionDescriptor, AgentDecision, DecisionContext, ScoredAction
from .reward_planner import RewardPlanner
from .run_analyzer import analyze_serialized_state
from .shop_planner import ShopPlanner


class StrategicAgent:
    """Explainable heuristic agent that coordinates all specialist planners."""

    def __init__(self, *, combat_search_config: CombatSearchConfig | None = None) -> None:
        self._combat = CombatSearchPlanner(combat_search_config)
        self._map = MapPlanner()
        self._reward = RewardPlanner()
        self._shop = ShopPlanner()
        self._campfire = CampfirePlanner()
        self._event = EventPlanner()

    def choose_action(self, state: Any) -> AgentDecision:
        """Choose one legal action for the current simulator state."""

        legal = tuple(action_space(state))
        state_summary = serialize(state)
        observation = encode_observation(state, include_state=False)
        plan = analyze_serialized_state(state_summary, observation=observation)
        context = DecisionContext(
            observation=observation,
            state_summary=state_summary,
            plan=plan,
            legal_actions=legal,
        )
        if not legal:
            return AgentDecision(
                plan=plan,
                candidates=(),
                stopped_reason="no_legal_actions",
            )

        if str(context.state_summary.get("phase")) == "combat":
            scored = self._sorted_scores(self._combat.score_all(context))
        else:
            scored = self._sorted_scores(
                self.score_action(context, descriptor) for descriptor in legal
            )
        return AgentDecision(
            plan=plan,
            chosen=scored[0],
            candidates=scored,
        )

    def _sorted_scores(self, scores: Iterable[ScoredAction]) -> tuple[ScoredAction, ...]:
        """Sort scored actions by score and stable action identity."""

        return tuple(
            sorted(
                scores,
                key=lambda candidate: (
                    -candidate.score,
                    candidate.action_type,
                    candidate.action_id,
                ),
            )
        )

    def score_action(
        self,
        context: DecisionContext,
        descriptor: ActionDescriptor,
    ) -> ScoredAction:
        """Score a single descriptor using the relevant planner."""

        kind = action_type(descriptor)
        phase = str(context.state_summary.get("phase", "unknown"))
        if kind in {
            "play_card",
            "end_turn",
            "use_potion",
            "choose_card",
            "discard_card",
            "exhaust_card",
        }:
            return self._combat.score(context, descriptor)
        if kind == "choose_node":
            return self._map.score(context, descriptor)
        if kind.startswith("take_reward") or (
            phase in {"reward", "treasure"} and kind == "proceed"
        ):
            return self._reward.score(context, descriptor)
        if kind in {"shop_buy", "shop_leave", "throw_potion_at_merchant"} or (
            phase == "shop" and kind == "proceed"
        ):
            return self._shop.score(context, descriptor)
        if kind in {"rest", "smith", "recall", "dig", "lift", "toke"} or (
            phase == "rest" and kind == "proceed"
        ):
            return self._campfire.score(context, descriptor)
        if kind in {"choose_ancient", "choose_event"} or (phase == "event" and kind == "proceed"):
            return self._event.score(context, descriptor)
        return generic_score(context, descriptor)
