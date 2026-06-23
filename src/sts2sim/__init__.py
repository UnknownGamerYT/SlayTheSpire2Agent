"""Headless Slay the Spire 2 simulator."""

from sts2sim.agent_api import action_mask, action_space, decode_action, encode_observation
from sts2sim.agents import StrategicAgent, analyze_run, play_strategic_run
from sts2sim.api import legal_actions, load_state, new_run, replay, serialize, step
from sts2sim.gymnasium_env import SlayTheSpire2Env, Sts2Env, make_env
from sts2sim.history import (
    RunHistory,
    RunHistoryStep,
    append_history_step,
    record_history_step,
    start_run_history,
    write_run_history,
)
from sts2sim.learning import (
    build_learning_progress_report,
    collect_random_rollouts,
    encode_rich_observation,
    evaluate_agent_baselines,
    evaluate_learning_agent,
    load_learning_progress,
    progress_from_runs,
    train_q_learning,
    write_learning_progress_report,
)
from sts2sim.live_agent import play_live_agent
from sts2sim.live_capture import capture_live_state, live_play
from sts2sim.live_parity import compare_live_step_to_simulator
from sts2sim.parity import compare_snapshots, compare_trace, compare_trace_file, load_trace
from sts2sim.run_files import import_run_file

__all__ = [
    "SlayTheSpire2Env",
    "Sts2Env",
    "StrategicAgent",
    "RunHistory",
    "RunHistoryStep",
    "action_mask",
    "action_space",
    "analyze_run",
    "append_history_step",
    "build_learning_progress_report",
    "compare_snapshots",
    "compare_live_step_to_simulator",
    "compare_trace",
    "compare_trace_file",
    "capture_live_state",
    "collect_random_rollouts",
    "decode_action",
    "encode_observation",
    "encode_rich_observation",
    "evaluate_agent_baselines",
    "evaluate_learning_agent",
    "import_run_file",
    "legal_actions",
    "live_play",
    "load_learning_progress",
    "load_state",
    "load_trace",
    "make_env",
    "new_run",
    "play_live_agent",
    "play_strategic_run",
    "progress_from_runs",
    "record_history_step",
    "replay",
    "serialize",
    "start_run_history",
    "step",
    "train_q_learning",
    "write_learning_progress_report",
    "write_run_history",
]
