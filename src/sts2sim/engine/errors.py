from __future__ import annotations

from collections.abc import Sequence

from .models import Action


class EngineError(Exception):
    """Base exception for deterministic engine failures."""


class IllegalActionError(EngineError):
    def __init__(self, action: Action, legal_actions: Sequence[Action]) -> None:
        self.action = action
        self.legal_actions = tuple(legal_actions)
        legal_summary = ", ".join(
            action.model_dump_json(exclude_none=True) for action in self.legal_actions
        )
        super().__init__(
            f"Illegal action {action.model_dump_json(exclude_none=True)}. "
            f"Legal actions: [{legal_summary}]"
        )

