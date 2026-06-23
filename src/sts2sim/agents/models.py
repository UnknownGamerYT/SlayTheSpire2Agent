"""Shared models for strategic simulator agents."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

type StrategyMode = Literal["safe", "balanced", "greedy", "scaling"]
type RiskLevel = Literal["low", "medium", "high", "critical"]
type CombatPace = Literal["stall", "balanced", "rush"]

type ActionDescriptor = dict[str, Any]


class AgentModel(BaseModel):
    """Base model used by the agent layer."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )


class DeckProfile(AgentModel):
    """Compact deck summary used by strategic evaluators."""

    total_cards: int = 0
    attacks: int = 0
    skills: int = 0
    powers: int = 0
    statuses: int = 0
    curses: int = 0
    upgraded: int = 0
    strike_like: int = 0
    defend_like: int = 0
    draw_cards: int = 0
    block_cards: int = 0
    damage_cards: int = 0


class ThreatProfile(AgentModel):
    """Current short-term danger signals."""

    phase: str = "unknown"
    act: int = 1
    floor: int = 0
    incoming_damage: int = 0
    alive_monsters: int = 0
    monster_hp_total: int = 0
    scaling_pressure: float = 0.0
    enemy_attack_pressure: float = 0.0
    next_boss_id: str | None = None
    known_elite_id: str | None = None
    possible_elite_ids: tuple[str, ...] = ()
    unknown_elite_count: int = 0


class AggressionProfile(AgentModel):
    """How willing the agent should be to spend HP for speed and future value."""

    target: float = 0.5
    hp_floor: float = 0.6
    hp_spend_budget: int = 0
    block_priority: float = 0.5
    combat_pace: CombatPace = "balanced"
    combat_pace_pressure: float = 0.0
    allow_chip_damage: bool = False
    scaling_pressure: float = 0.0
    enemy_attack_pressure: float = 0.0
    elite_pressure: float = 0.0
    future_elite_count: int = 0
    future_rest_count: int = 0
    nearest_elite_distance: int = 0
    nearest_rest_distance: int = 0
    boss_distance: int = 0
    known_elite_id: str | None = None
    possible_elite_ids: tuple[str, ...] = ()
    unknown_elite_count: int = 0


class EconomyProfile(AgentModel):
    """Gold, potion, and shop-related state."""

    gold: int = 0
    potion_count: int = 0
    potion_capacity: int | None = None
    relic_count: int = 0
    removable_cards: int = 0


class RunPlan(AgentModel):
    """Persistent strategic plan rebuilt from the latest state each decision."""

    strategy: StrategyMode = "balanced"
    risk_level: RiskLevel = "medium"
    elite_budget: int = 1
    hp_ratio: float = 1.0
    deck: DeckProfile = Field(default_factory=DeckProfile)
    threat: ThreatProfile = Field(default_factory=ThreatProfile)
    economy: EconomyProfile = Field(default_factory=EconomyProfile)
    aggression: AggressionProfile = Field(default_factory=AggressionProfile)
    must_find: tuple[str, ...] = ()
    avoid: tuple[str, ...] = ()
    upgrade_targets: tuple[str, ...] = ()
    remove_targets: tuple[str, ...] = ()
    potion_policy: str = "use_when_life_saving_or_elite"
    notes: tuple[str, ...] = ()


class DecisionContext(AgentModel):
    """Inputs available to evaluators for one decision."""

    observation: dict[str, Any]
    state_summary: dict[str, Any]
    plan: RunPlan
    legal_actions: tuple[ActionDescriptor, ...]


class ScoredAction(AgentModel):
    """One legal action with an explainable score."""

    action_id: int
    action_type: str
    action: dict[str, Any]
    score: float
    category: str
    reasons: tuple[str, ...] = ()


class AgentDecision(AgentModel):
    """Final action choice and the alternatives considered."""

    plan: RunPlan
    chosen: ScoredAction | None = None
    candidates: tuple[ScoredAction, ...] = ()
    stopped_reason: str | None = None

    @property
    def action_id(self) -> int | None:
        """State-local action id for the chosen action."""

        if self.chosen is None:
            return None
        return self.chosen.action_id
