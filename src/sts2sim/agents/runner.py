"""Run the strategic agent inside the simulator."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sts2sim.agent_api import decode_action
from sts2sim.api import new_run, serialize, step
from sts2sim.engine.serialization import state_digest
from sts2sim.history import (
    append_history_step,
    record_history_step,
    start_run_history,
    write_run_history,
)

from .models import AgentDecision
from .policy import StrategicAgent


def play_strategic_run(
    *,
    seed: int | str = 0,
    character_id: str = "IRONCLAD",
    ascension: int = 0,
    max_steps: int = 100,
    output_path: Path | str | None = None,
    history_path: Path | str | None = None,
    source_data: Mapping[str, Any] | None = None,
    agent: StrategicAgent | None = None,
) -> dict[str, Any]:
    """Run a simulator episode using the strategic skeleton agent."""

    active_agent = agent or StrategicAgent()
    state = new_run(
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        source_data=source_data,
    )
    history = start_run_history(state, policy="strategic_v0")
    decisions: list[dict[str, Any]] = []
    stopped_reason = "max_steps"

    for step_index in range(max(0, max_steps)):
        decision = active_agent.choose_action(state)
        if decision.chosen is None:
            stopped_reason = decision.stopped_reason or "no_action_chosen"
            decisions.append(_decision_payload(step_index, state, decision, after_hash=None))
            break

        before_state = state
        before_hash = state_digest(state)
        action = decode_action(state, decision.chosen.action_id)
        state = step(state, action)
        after_hash = state_digest(state)
        decision_payload = _decision_payload(
            step_index,
            before_state,
            decision,
            before_hash=before_hash,
            after_hash=after_hash,
        )
        decisions.append(decision_payload)
        history = append_history_step(
            history,
            record_history_step(
                step_index=step_index,
                before_state=before_state,
                action=action,
                after_state=state,
                decision=_history_decision(decision_payload),
            ),
            state,
        )
        if str(getattr(state.phase, "value", state.phase)) in {"complete", "failed"}:
            stopped_reason = str(getattr(state.phase, "value", state.phase))
            break

    result = {
        "seed": seed,
        "character_id": character_id,
        "ascension": ascension,
        "policy": "strategic_v0",
        "steps_taken": len([item for item in decisions if item.get("chosen") is not None]),
        "stopped_reason": stopped_reason,
        "decisions": decisions,
        "history": history.model_dump(mode="json"),
        "final": {
            "phase": str(getattr(state.phase, "value", state.phase)),
            "act": state.act,
            "floor": state.floor,
            "state_hash": state_digest(state),
        },
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if history_path is not None:
        write_run_history(history, history_path)
    return result


def _decision_payload(
    step_index: int,
    state: Any,
    decision: AgentDecision,
    *,
    before_hash: str | None = None,
    after_hash: str | None,
) -> dict[str, Any]:
    chosen = decision.chosen.model_dump(mode="json") if decision.chosen is not None else None
    return {
        "step": step_index,
        "phase": str(getattr(state.phase, "value", state.phase)),
        "state_hash_before": before_hash,
        "state_hash_after": after_hash,
        "plan": decision.plan.model_dump(mode="json"),
        "chosen": chosen,
        "candidate_count": len(decision.candidates),
        "top_candidates": [
            candidate.model_dump(mode="json") for candidate in decision.candidates[:5]
        ],
        "stopped_reason": decision.stopped_reason,
        "state_summary": _compact_state_summary(serialize(state)),
    }


def _compact_state_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    player = payload.get("player")
    relics = payload.get("relics")
    potions = payload.get("potions")
    return {
        "phase": payload.get("phase"),
        "act": payload.get("act"),
        "floor": payload.get("floor"),
        "player": player if isinstance(player, Mapping) else {},
        "relic_count": len(relics) if isinstance(relics, list) else 0,
        "potion_count": len(potions) if isinstance(potions, list) else 0,
    }


def _history_decision(decision_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "chosen": decision_payload.get("chosen"),
        "candidate_count": decision_payload.get("candidate_count"),
        "top_candidates": decision_payload.get("top_candidates"),
        "plan": decision_payload.get("plan"),
    }
