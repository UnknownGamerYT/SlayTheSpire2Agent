"""Self-learning agents and training helpers."""

from sts2sim.learning.agent_eval import evaluate_agent_baselines
from sts2sim.learning.curriculum import (
    CURRICULUM_STAGE_DEFAULTS,
    DEFAULT_CURRICULUM_STAGES,
    resolve_curriculum_stages,
    train_masked_ppo_curriculum,
)
from sts2sim.learning.evaluate import evaluate_learning_agent
from sts2sim.learning.masked_ppo import (
    TrainingTarget,
    max_consecutive_target_successes,
    resolve_ppo_target,
    train_masked_ppo,
)
from sts2sim.learning.models import (
    AgentEvaluationResult,
    AgentEvaluationSummary,
    LearningBatchResult,
    LearningProgressPoint,
    LearningRunResult,
    LearningStep,
    QLearningModel,
    TrainingResult,
    TrainingTargetBatchSummary,
    TrainingUntilTargetResult,
)
from sts2sim.learning.observation import encode_rich_observation
from sts2sim.learning.progress import (
    build_learning_progress_report,
    load_learning_progress,
    progress_from_runs,
    write_learning_progress_report,
)
from sts2sim.learning.q_learning import QLearningAgent, load_q_learning_model, train_q_learning
from sts2sim.learning.random_agent import MaskedRandomAgent
from sts2sim.learning.rewards import LearningRewardConfig, learning_reward
from sts2sim.learning.rollout import collect_random_rollout, collect_random_rollouts
from sts2sim.learning.train_until import train_q_learning_until_boss

__all__ = [
    "AgentEvaluationResult",
    "AgentEvaluationSummary",
    "CURRICULUM_STAGE_DEFAULTS",
    "DEFAULT_CURRICULUM_STAGES",
    "LearningBatchResult",
    "LearningProgressPoint",
    "LearningRewardConfig",
    "LearningRunResult",
    "LearningStep",
    "MaskedRandomAgent",
    "QLearningAgent",
    "QLearningModel",
    "TrainingTargetBatchSummary",
    "TrainingResult",
    "TrainingTarget",
    "TrainingUntilTargetResult",
    "collect_random_rollout",
    "collect_random_rollouts",
    "build_learning_progress_report",
    "encode_rich_observation",
    "evaluate_agent_baselines",
    "evaluate_learning_agent",
    "learning_reward",
    "load_learning_progress",
    "load_q_learning_model",
    "max_consecutive_target_successes",
    "progress_from_runs",
    "resolve_curriculum_stages",
    "resolve_ppo_target",
    "train_q_learning",
    "train_masked_ppo_curriculum",
    "train_masked_ppo",
    "train_q_learning_until_boss",
    "write_learning_progress_report",
]
