"""Shared models for self-learning agents."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LearningModel(BaseModel):
    """Base pydantic model for learning outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class LearningStep(LearningModel):
    """One transition collected from a simulator rollout."""

    run_index: int
    step_index: int
    seed: int | str
    action_id: int
    action: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    phase_before: str
    phase_after: str
    vector: tuple[float, ...]
    action_mask: tuple[int, ...]
    observation_mode: str = "rich"
    observation: dict[str, Any] | None = None


class LearningRunResult(LearningModel):
    """Result for one self-learning rollout."""

    run_index: int
    seed: int | str
    character_id: str
    ascension: int
    policy: str
    steps_taken: int
    total_reward: float
    terminated: bool
    truncated: bool
    final_phase: str
    final_act: int
    final_floor: int
    error: str | None = None
    failed_to_continue: bool = False
    reward_breakdown_totals: dict[str, float] = Field(default_factory=dict)
    diagnostics: dict[str, float] = Field(default_factory=dict)
    history: dict[str, Any] | None = None
    steps: tuple[LearningStep, ...] = ()


class LearningProgressPoint(LearningModel):
    """One run-level point for learning progress charts."""

    run_index: int
    seed: int | str
    policy: str
    steps_taken: int
    total_reward: float
    final_phase: str
    final_act: int
    final_floor: int
    win: bool = False
    death: bool = False
    truncated: bool = False
    failed_to_continue: bool = False
    error: str | None = None
    moving_average_reward: float = 0.0
    moving_average_floor: float = 0.0
    moving_win_rate: float = 0.0


class LearningBatchResult(LearningModel):
    """Result for a batch of learning rollouts."""

    policy: str
    runs_requested: int
    runs_completed: int
    total_steps: int
    average_reward: float
    average_floor: float
    wins: int
    deaths: int
    output_path: str | None = None
    progress: tuple[LearningProgressPoint, ...] = ()
    runs: tuple[LearningRunResult, ...] = ()


class AgentEvaluationSummary(LearningModel):
    """Aggregate metrics for one evaluated policy."""

    policy: str
    runs: int
    average_reward: float
    average_floor: float
    average_steps: float
    best_floor: int
    best_reward: float
    wins: int
    deaths: int
    errors: int
    failed_to_continue: int
    win_rate: float


class AgentEvaluationResult(LearningModel):
    """Side-by-side baseline evaluation for multiple agent policies."""

    character_id: str
    ascension: int
    start_seed: int
    runs_requested: int
    max_steps: int
    policies: tuple[str, ...]
    q_learning_model_path: str | None = None
    output_path: str | None = None
    report_output_path: str | None = None
    summaries: tuple[AgentEvaluationSummary, ...] = ()
    progress_by_policy: dict[str, tuple[LearningProgressPoint, ...]] = Field(
        default_factory=dict
    )
    runs_by_policy: dict[str, tuple[LearningRunResult, ...]] = Field(default_factory=dict)


class QLearningModel(LearningModel):
    """Serializable dependency-free Q-learning checkpoint."""

    algorithm: Literal["state_action_signature_q_learning"] = "state_action_signature_q_learning"
    version: int = 1
    q_values: dict[str, float] = Field(default_factory=dict)
    visits: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingResult(LearningModel):
    """Summary of a learning run that produced a model checkpoint."""

    algorithm: str
    runs: int
    total_steps: int
    average_reward: float
    wins: int
    deaths: int
    model_path: str | None = None
    progress_output_path: str | None = None
    report_output_path: str | None = None
    progress: tuple[LearningProgressPoint, ...] = ()
    evaluation: dict[str, Any] = Field(default_factory=dict)


class TrainingTargetBatchSummary(LearningModel):
    """One train/evaluate batch while training toward a run target."""

    batch_index: int
    trained_runs_total: int
    train_total_steps: int
    evaluation_runs: int
    evaluation_average_reward: float
    evaluation_average_floor: float
    evaluation_best_floor: int
    evaluation_best_reward: float
    evaluation_errors: int
    evaluation_failed_to_continue: int
    reached_target: bool
    evaluation_target_successes: int = 0
    evaluation_target_success_rate: float = 0.0


class TrainingUntilTargetResult(LearningModel):
    """Summary for bounded training that stops when a target is reached."""

    algorithm: str
    target_act: int
    target_floor: int
    reached_target: bool
    reached_batch: int | None = None
    batches_completed: int
    max_batches: int
    batch_runs: int
    runs_trained: int
    total_steps: int
    average_training_reward: float
    wins: int
    deaths: int
    resumed_from_path: str | None = None
    model_path: str | None = None
    output_path: str | None = None
    progress_output_path: str | None = None
    report_output_path: str | None = None
    batch_summaries: tuple[TrainingTargetBatchSummary, ...] = ()
    progress: tuple[LearningProgressPoint, ...] = ()
    evaluation_progress: tuple[LearningProgressPoint, ...] = ()
