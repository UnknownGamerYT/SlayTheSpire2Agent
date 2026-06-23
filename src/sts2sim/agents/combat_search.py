"""Multi-turn combat search over the real simulator transition model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sts2sim.agent_api import action_space
from sts2sim.api import load_state, serialize, step

from .combat_objectives import CombatProjectionScore, evaluate_combat_projection
from .combat_planner import CombatPlanner
from .evaluators import action_payload, action_type, make_score, mapping, normalized, number
from .models import ActionDescriptor, DecisionContext, ScoredAction


@dataclass(frozen=True)
class CombatSearchConfig:
    """Limits for multi-turn combat search."""

    max_depth: int = 18
    max_turns: int = 6
    beam_width: int = 36
    branch_width: int = 8


@dataclass(frozen=True)
class CombatSearchLine:
    """Best projected line found for one first action."""

    first_action_key: str
    first_action: dict[str, Any]
    path: tuple[dict[str, Any], ...]
    projection: CombatProjectionScore


@dataclass(frozen=True)
class _SearchNode:
    state: Any
    first_action_key: str
    first_action: dict[str, Any]
    path: tuple[dict[str, Any], ...]
    event_kinds: tuple[str, ...]


class CombatSearchPlanner:
    """Search-backed combat planner that scores full projected combat lines."""

    def __init__(self, config: CombatSearchConfig | None = None) -> None:
        self.config = config or CombatSearchConfig()
        self._fallback = CombatPlanner()

    def score_all(self, context: DecisionContext) -> tuple[ScoredAction, ...]:
        """Score all current combat actions using shared multi-turn search results."""

        if str(context.state_summary.get("phase")) != "combat":
            return tuple(
                self._fallback.score(context, descriptor)
                for descriptor in context.legal_actions
            )

        lines = self.search(context)
        scored: list[ScoredAction] = []
        for descriptor in context.legal_actions:
            key = str(descriptor.get("key", ""))
            line = lines.get(key)
            if line is None:
                scored.append(self._fallback.score(context, descriptor))
                continue
            scored.append(
                make_score(
                    descriptor,
                    score=line.projection.score,
                    category="combat_search",
                    reasons=_line_reasons(line),
                )
            )
        return tuple(scored)

    def score(self, context: DecisionContext, descriptor: ActionDescriptor) -> ScoredAction:
        """Score one action, falling back to the old heuristic outside batch calls."""

        lines = self.search(context)
        key = str(descriptor.get("key", ""))
        line = lines.get(key)
        if line is None:
            return self._fallback.score(context, descriptor)
        return make_score(
            descriptor,
            score=line.projection.score,
            category="combat_search",
            reasons=_line_reasons(line),
        )

    def search(self, context: DecisionContext) -> dict[str, CombatSearchLine]:
        """Run a bounded full-combat projection search from the current state."""

        initial_state = load_state(context.state_summary)
        initial_payload = serialize(initial_state)
        best: dict[str, CombatSearchLine] = {}
        frontier = self._initial_frontier(initial_state, context.legal_actions)

        for node in frontier:
            _record_best(best, initial_payload, node)

        for _depth in range(1, self.config.max_depth):
            next_frontier: list[_SearchNode] = []
            for node in frontier:
                payload = serialize(node.state)
                if _is_cutoff(initial_payload, payload, node.path, self.config):
                    _record_best(best, initial_payload, node)
                    continue
                for descriptor in _branch_actions(payload, action_space(node.state), self.config):
                    action = action_payload(descriptor)
                    try:
                        next_state = step(node.state, action)
                    except Exception:
                        continue
                    next_node = _SearchNode(
                        state=next_state,
                        first_action_key=node.first_action_key,
                        first_action=node.first_action,
                        path=node.path + (action,),
                        event_kinds=node.event_kinds + _last_event_kinds(next_state),
                    )
                    _record_best(best, initial_payload, next_node)
                    next_frontier.append(next_node)

            if not next_frontier:
                break
            frontier = tuple(
                sorted(
                    next_frontier,
                    key=lambda candidate: _beam_sort_score(initial_payload, candidate),
                    reverse=True,
                )[: self.config.beam_width]
            )

        for node in frontier:
            _record_best(best, initial_payload, node)
        return best

    def _initial_frontier(
        self,
        state: Any,
        descriptors: Sequence[ActionDescriptor],
    ) -> tuple[_SearchNode, ...]:
        nodes: list[_SearchNode] = []
        for descriptor in descriptors:
            action = action_payload(descriptor)
            try:
                next_state = step(state, action)
            except Exception:
                continue
            nodes.append(
                _SearchNode(
                    state=next_state,
                    first_action_key=str(descriptor.get("key", "")),
                    first_action=action,
                    path=(action,),
                    event_kinds=_last_event_kinds(next_state),
                )
            )
        return tuple(nodes)


def _record_best(
    best: dict[str, CombatSearchLine],
    initial_payload: Mapping[str, Any],
    node: _SearchNode,
) -> None:
    payload = serialize(node.state)
    projection = evaluate_combat_projection(
        initial_payload,
        payload,
        path_length=len(node.path),
        event_kinds=node.event_kinds,
    )
    existing = best.get(node.first_action_key)
    if existing is None or projection.score > existing.projection.score:
        best[node.first_action_key] = CombatSearchLine(
            first_action_key=node.first_action_key,
            first_action=node.first_action,
            path=node.path,
            projection=projection,
        )


def _branch_actions(
    state_payload: Mapping[str, Any],
    descriptors: Sequence[ActionDescriptor],
    config: CombatSearchConfig,
) -> tuple[ActionDescriptor, ...]:
    if not descriptors:
        return ()
    sorted_descriptors = sorted(
        descriptors,
        key=lambda descriptor: _quick_action_score(state_payload, descriptor),
        reverse=True,
    )
    selected = list(sorted_descriptors[: config.branch_width])
    end_turn = next(
        (descriptor for descriptor in sorted_descriptors if action_type(descriptor) == "end_turn"),
        None,
    )
    if end_turn is not None and all(action_type(item) != "end_turn" for item in selected):
        selected.append(end_turn)
    return tuple(selected)


def _quick_action_score(
    state_payload: Mapping[str, Any],
    descriptor: ActionDescriptor,
) -> float:
    kind = action_type(descriptor)
    if kind == "end_turn":
        return 1.0
    if kind == "use_potion":
        return 12.0
    if kind in {"choose_card", "discard_card", "exhaust_card"}:
        return 20.0
    if kind != "play_card":
        return 0.0

    payload = action_payload(descriptor)
    combat = mapping(state_payload.get("combat"))
    card = _card_by_instance_id(_sequence(combat.get("hand")), payload.get("card_instance_id"))
    card_type = normalized(card.get("type"))
    effects = mapping(card.get("effects"))
    damage = _effect_amount(effects, "damage") + _effect_amount(effects, "all_damage")
    block = _effect_amount(effects, "block")
    draw = _effect_amount(effects, "draw")
    cost = number(card.get("cost"), 1.0)
    if card_type == "attack":
        return 30.0 + damage * 2.0 - cost
    if card_type == "skill":
        return 18.0 + block * 1.5 + draw * 3.0 - cost
    if card_type == "power":
        return 22.0 - cost
    return 8.0 - cost


def _beam_sort_score(initial_payload: Mapping[str, Any], node: _SearchNode) -> float:
    return evaluate_combat_projection(
        initial_payload,
        serialize(node.state),
        path_length=len(node.path),
        event_kinds=node.event_kinds,
    ).score


def _line_reasons(line: CombatSearchLine) -> tuple[str, ...]:
    first_type = str(line.first_action.get("type", "unknown"))
    reasons = [
        "search_projects_multiple_turns",
        f"first_action:{first_type}",
        f"projected_phase:{line.projection.reached_phase}",
        f"projected_path_length:{len(line.path)}",
    ]
    reasons.extend(line.projection.reasons)
    if line.projection.projected_unblocked_damage == 0:
        reasons.append("line_blocks_or_avoids_projected_incoming_damage")
    if line.projection.max_hp_delta > 0 or line.projection.resource_delta > 0:
        reasons.append("line_values_card_specific_payoff")
    return tuple(dict.fromkeys(reasons))


def _is_cutoff(
    initial_payload: Mapping[str, Any],
    payload: Mapping[str, Any],
    path: Sequence[Mapping[str, Any]],
    config: CombatSearchConfig,
) -> bool:
    del initial_payload
    if _phase(payload) != "combat":
        return True
    player = mapping(mapping(payload.get("combat")).get("player"))
    if number(player.get("hp")) <= 0:
        return True
    if len(path) >= config.max_depth:
        return True
    combat = mapping(payload.get("combat"))
    turn = int(number(combat.get("turn"), 1.0))
    return turn > config.max_turns


def _last_event_kinds(state: Any) -> tuple[str, ...]:
    replay_log = getattr(state, "replay_log", ())
    if not replay_log:
        return ()
    last_entry = replay_log[-1]
    events = getattr(last_entry, "events", ())
    return tuple(str(getattr(event, "kind", "")) for event in events if getattr(event, "kind", ""))


def _card_by_instance_id(cards: Sequence[object], instance_id: object) -> dict[str, Any]:
    wanted = str(instance_id or "")
    for raw_card in cards:
        card = mapping(raw_card)
        if str(card.get("instance_id", "")) == wanted:
            return card
    return {}


def _effect_amount(effects: Mapping[str, Any], key: str) -> float:
    value = effects.get(key)
    if isinstance(value, Mapping):
        return number(value.get("amount"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return float(len(value))
    return number(value)


def _phase(payload: Mapping[str, Any]) -> str:
    return str(payload.get("phase", "unknown"))


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()
