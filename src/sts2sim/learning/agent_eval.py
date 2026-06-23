"""Side-by-side baseline evaluation for simulator agents."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from html import escape as html_escape
from pathlib import Path
from typing import Any, Protocol

from sts2sim.agent_api import decode_action
from sts2sim.agents import StrategicAgent
from sts2sim.agents.combat_search import CombatSearchConfig
from sts2sim.api import new_run
from sts2sim.api import step as step_state
from sts2sim.gymnasium_env import Sts2Env
from sts2sim.learning.models import (
    AgentEvaluationResult,
    AgentEvaluationSummary,
    LearningProgressPoint,
    LearningRunResult,
    QLearningModel,
)
from sts2sim.learning.progress import progress_from_runs
from sts2sim.learning.q_learning import QLearningAgent, load_q_learning_model
from sts2sim.learning.random_agent import MaskedRandomAgent
from sts2sim.learning.rewards import learning_reward

DEFAULT_BASELINE_POLICIES = ("random", "q_learning", "strategic")
_POLICY_LABELS = {
    "random": "Random",
    "q_learning": "Q-learning",
    "strategic": "Strategic",
}
_POLICY_COLORS = {
    "random": "#5f6b72",
    "q_learning": "#2563eb",
    "strategic": "#16855b",
}
_FAST_STRATEGIC_COMBAT_SEARCH = CombatSearchConfig(
    max_depth=8,
    max_turns=3,
    beam_width=10,
    branch_width=4,
)


class _LearningPolicy(Protocol):
    def choose_action_id(
        self,
        observation: Mapping[str, Any],
        info: Mapping[str, Any],
    ) -> int | None:
        """Choose a state-local action id."""


def evaluate_agent_baselines(
    *,
    runs: int = 10,
    max_steps: int = 500,
    start_seed: int = 0,
    character_id: str = "TEST",
    ascension: int = 0,
    policies: Sequence[str] | None = None,
    model_path: Path | str | None = None,
    output_path: Path | str | None = Path("reports/agent_baselines.json"),
    report_output_path: Path | str | None = Path("reports/agent_baselines.html"),
    progress_window: int = 10,
) -> AgentEvaluationResult:
    """Evaluate baseline agents over identical seeds and write optional reports."""

    normalized_policies = _normalize_policies(policies)
    q_model = load_q_learning_model(model_path) if model_path is not None else None
    runs_by_policy: dict[str, tuple[LearningRunResult, ...]] = {}
    progress_by_policy: dict[str, tuple[LearningProgressPoint, ...]] = {}
    summaries: list[AgentEvaluationSummary] = []

    for policy in normalized_policies:
        policy_runs = tuple(
            _evaluate_one_policy(
                policy=policy,
                run_index=run_index,
                seed=start_seed + run_index,
                max_steps=max_steps,
                character_id=character_id,
                ascension=ascension,
                q_model=q_model,
            )
            for run_index in range(max(0, runs))
        )
        progress = progress_from_runs(policy_runs, policy=policy, window=progress_window)
        runs_by_policy[policy] = policy_runs
        progress_by_policy[policy] = progress
        summaries.append(_policy_summary(policy, policy_runs))

    result = AgentEvaluationResult(
        character_id=character_id,
        ascension=ascension,
        start_seed=start_seed,
        runs_requested=runs,
        max_steps=max_steps,
        policies=normalized_policies,
        q_learning_model_path=str(model_path) if model_path is not None else None,
        output_path=str(output_path) if output_path is not None else None,
        report_output_path=str(report_output_path) if report_output_path is not None else None,
        summaries=tuple(summaries),
        progress_by_policy=progress_by_policy,
        runs_by_policy=runs_by_policy,
    )
    if output_path is not None:
        _write_json(result, output_path)
    if report_output_path is not None:
        write_agent_comparison_report(result, report_output_path)
    return result


def write_agent_comparison_report(
    result: AgentEvaluationResult,
    path: Path | str,
) -> None:
    """Write a standalone HTML baseline comparison report."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(agent_comparison_html(result), encoding="utf-8")


def agent_comparison_html(result: AgentEvaluationResult) -> str:
    """Render the baseline comparison as a static HTML report."""

    title = "Agent Baseline Comparison"
    best_summary = _best_summary(result.summaries)
    failed_runs = sum(summary.failed_to_continue for summary in result.summaries)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #5f6b72;
      --line: #d7dde1;
      --panel: #ffffff;
      --paper: #f5f7f8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 44px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 10px;
      margin: 20px 0;
    }}
    .kpi, .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .kpi {{ padding: 12px; }}
    .kpi span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .kpi strong {{ display: block; margin-top: 4px; font-size: 22px; }}
    .section {{ margin-top: 14px; padding: 16px; overflow: hidden; }}
    .chart {{ width: 100%; height: auto; display: block; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }}
    th {{ color: var(--muted); font-weight: 650; }}
    @media (max-width: 780px) {{
      main {{ width: min(100vw - 20px, 1180px); padding-top: 18px; }}
      h1 {{ font-size: 24px; }}
      .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .section {{ padding: 12px; }}
      table {{ display: block; overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>{title}</h1>
    <p>{html_escape(_comparison_sentence(result, best_summary))}</p>
    <div class="kpis">
      {_kpi("Policies", str(len(result.policies)))}
      {_kpi("Seeds", str(result.runs_requested))}
      {_kpi("Best Avg Floor", _format_number(best_summary.average_floor if best_summary else 0))}
      {_kpi("Best Policy", _policy_label(best_summary.policy if best_summary else "none"))}
      {_kpi("Failed Runs", str(failed_runs))}
    </div>
    <div class="section">
      <h2>Policy Summary</h2>
      {_summary_table(result.summaries)}
    </div>
    <div class="section">
      <h2>Average Reward, Floor, And Win Rate</h2>
      {_summary_bars(result.summaries)}
    </div>
    <div class="section">
      <h2>Floor By Seed</h2>
      {_seed_table(result)}
    </div>
  </main>
</body>
</html>
"""


def _evaluate_one_policy(
    *,
    policy: str,
    run_index: int,
    seed: int,
    max_steps: int,
    character_id: str,
    ascension: int,
    q_model: QLearningModel | None,
) -> LearningRunResult:
    if policy == "strategic":
        return _evaluate_strategic_run(
            run_index=run_index,
            seed=seed,
            max_steps=max_steps,
            character_id=character_id,
            ascension=ascension,
        )
    if policy == "random":
        agent: _LearningPolicy = MaskedRandomAgent(seed=seed)
    elif policy == "q_learning":
        agent = QLearningAgent(model=q_model, seed=seed, epsilon=0.0)
    else:  # pragma: no cover - guarded by normalization
        raise ValueError(f"Unsupported policy: {policy}")
    return _evaluate_gym_policy(
        policy=policy,
        agent=agent,
        run_index=run_index,
        seed=seed,
        max_steps=max_steps,
        character_id=character_id,
        ascension=ascension,
    )


def _evaluate_gym_policy(
    *,
    policy: str,
    agent: _LearningPolicy,
    run_index: int,
    seed: int,
    max_steps: int,
    character_id: str,
    ascension: int,
) -> LearningRunResult:
    env = Sts2Env(
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        max_episode_steps=max_steps,
        reward_fn=learning_reward,
        include_serialized_state=False,
    )
    observation, info = env.reset()
    total_reward = 0.0
    steps_taken = 0
    terminated = False
    truncated = False
    error: str | None = None
    for _step_index in range(max_steps):
        action_id = agent.choose_action_id(observation, info)
        if action_id is None:
            break
        if action_id not in _current_action_ids(info):
            error = f"Policy chose action id {action_id}, which is not in current action_space"
            break
        try:
            observation, reward, terminated, truncated, info = env.step(action_id)
        except Exception as exc:  # pragma: no cover - defensive readiness reporting
            error = f"{type(exc).__name__}: {exc}"
            break
        total_reward += reward
        steps_taken += 1
        if terminated or truncated:
            break
    env.close()
    return LearningRunResult(
        run_index=run_index,
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        policy=policy,
        steps_taken=steps_taken,
        total_reward=round(total_reward, 6),
        terminated=terminated,
        truncated=truncated,
        final_phase="error" if error is not None else str(observation.get("phase", "unknown")),
        final_act=_lookup_vector_int(observation, "act"),
        final_floor=_lookup_vector_int(observation, "floor"),
        error=error,
        failed_to_continue=error is not None,
    )


def _evaluate_strategic_run(
    *,
    run_index: int,
    seed: int,
    max_steps: int,
    character_id: str,
    ascension: int,
) -> LearningRunResult:
    state = new_run(seed=seed, character_id=character_id, ascension=ascension)
    agent = StrategicAgent(combat_search_config=_FAST_STRATEGIC_COMBAT_SEARCH)
    total_reward = 0.0
    steps_taken = 0
    terminated = False
    error: str | None = None
    for _step_index in range(max_steps):
        decision = agent.choose_action(state)
        if decision.chosen is None:
            break
        before_state = state
        try:
            action = decode_action(state, decision.chosen.action_id)
            state = step_state(state, action)
        except Exception as exc:  # pragma: no cover - defensive readiness reporting
            error = f"{type(exc).__name__}: {exc}"
            break
        total_reward += learning_reward(before_state, state)
        steps_taken += 1
        if _is_terminal_phase(state):
            terminated = True
            break
    return LearningRunResult(
        run_index=run_index,
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        policy="strategic",
        steps_taken=steps_taken,
        total_reward=round(total_reward, 6),
        terminated=terminated,
        truncated=steps_taken >= max_steps and not terminated,
        final_phase=(
            "error"
            if error is not None
            else str(getattr(getattr(state, "phase", None), "value", "unknown"))
        ),
        final_act=int(getattr(state, "act", 0)),
        final_floor=int(getattr(state, "floor", 0)),
        error=error,
        failed_to_continue=error is not None,
    )


def _policy_summary(
    policy: str,
    runs: Sequence[LearningRunResult],
) -> AgentEvaluationSummary:
    count = len(runs)
    wins = sum(1 for run in runs if run.final_phase == "complete")
    deaths = sum(1 for run in runs if run.final_phase == "failed")
    errors = sum(1 for run in runs if run.error is not None)
    failed_to_continue = sum(1 for run in runs if run.failed_to_continue)
    return AgentEvaluationSummary(
        policy=policy,
        runs=count,
        average_reward=round(sum(run.total_reward for run in runs) / max(1, count), 6),
        average_floor=round(sum(run.final_floor for run in runs) / max(1, count), 6),
        average_steps=round(sum(run.steps_taken for run in runs) / max(1, count), 6),
        best_floor=max((run.final_floor for run in runs), default=0),
        best_reward=round(max((run.total_reward for run in runs), default=0.0), 6),
        wins=wins,
        deaths=deaths,
        errors=errors,
        failed_to_continue=failed_to_continue,
        win_rate=round(wins / max(1, count), 6),
    )


def _normalize_policies(policies: Sequence[str] | None) -> tuple[str, ...]:
    raw = tuple(policies or DEFAULT_BASELINE_POLICIES)
    normalized: list[str] = []
    valid = set(DEFAULT_BASELINE_POLICIES)
    for policy in raw:
        item = str(policy).strip().lower().replace("-", "_")
        if item not in valid:
            raise ValueError(
                f"Unsupported policy '{policy}'. Expected one of: {', '.join(sorted(valid))}"
            )
        if item not in normalized:
            normalized.append(item)
    return tuple(normalized)


def _write_json(result: AgentEvaluationResult, path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _best_summary(
    summaries: Sequence[AgentEvaluationSummary],
) -> AgentEvaluationSummary | None:
    if not summaries:
        return None
    return max(
        summaries,
        key=lambda item: (
            item.average_floor,
            item.average_reward,
            item.win_rate,
            -item.average_steps,
        ),
    )


def _summary_table(summaries: Sequence[AgentEvaluationSummary]) -> str:
    rows = []
    for summary in summaries:
        rows.append(
            "<tr>"
            f"<td>{html_escape(_policy_label(summary.policy))}</td>"
            f"<td>{summary.runs}</td>"
            f"<td>{_format_number(summary.average_reward)}</td>"
            f"<td>{_format_number(summary.average_floor)}</td>"
            f"<td>{summary.best_floor}</td>"
            f"<td>{_format_percent(summary.win_rate)}</td>"
            f"<td>{summary.deaths}</td>"
            f"<td>{summary.failed_to_continue}</td>"
            f"<td>{summary.errors}</td>"
            f"<td>{_format_number(summary.average_steps)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Policy</th><th>Runs</th><th>Avg Reward</th><th>Avg Floor</th>"
        "<th>Best Floor</th><th>Win Rate</th><th>Deaths</th>"
        "<th>Failed</th><th>Errors</th><th>Avg Steps</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _summary_bars(summaries: Sequence[AgentEvaluationSummary]) -> str:
    if not summaries:
        return "<p>No evaluation data yet.</p>"
    width = 920
    height = 320
    left = 72
    top = 28
    bottom = 52
    right = 24
    plot_width = width - left - right
    metrics = (
        ("Avg reward", [summary.average_reward for summary in summaries]),
        ("Avg floor", [summary.average_floor for summary in summaries]),
        ("Win rate", [summary.win_rate * 100.0 for summary in summaries]),
    )
    max_value = max((value for _label, values in metrics for value in values), default=1.0)
    max_value = max(1.0, max_value)
    group_width = plot_width / len(metrics)
    bar_width = min(44.0, group_width / (len(summaries) + 1.2))
    bars: list[str] = []
    labels: list[str] = []
    legend: list[str] = []
    for metric_index, (metric_label, values) in enumerate(metrics):
        group_x = left + metric_index * group_width
        labels.append(
            f'<text x="{group_x + group_width / 2:.2f}" y="{height - 16}" '
            f'text-anchor="middle" font-size="12" fill="#5f6b72">{metric_label}</text>'
        )
        for policy_index, summary in enumerate(summaries):
            value = float(values[policy_index])
            bar_height = (height - top - bottom) * value / max_value
            x = group_x + 14 + policy_index * (bar_width + 8)
            y = height - bottom - bar_height
            color = _POLICY_COLORS.get(summary.policy, "#5f6b72")
            bars.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{bar_height:.2f}" fill="{color}" rx="2" />'
            )
            bars.append(
                f'<text x="{x + bar_width / 2:.2f}" y="{max(14, y - 5):.2f}" '
                f'text-anchor="middle" font-size="10" fill="{color}">'
                f'{html_escape(_format_number(value))}</text>'
            )
    for summary in summaries:
        color = _POLICY_COLORS.get(summary.policy, "#5f6b72")
        legend.append(
            f'<span><i style="background:{color}"></i>'
            f"{html_escape(_policy_label(summary.policy))}</span>"
        )
    return f"""
<div aria-label="Policy summary bars" role="img">
  <svg class="chart" viewBox="0 0 {width} {height}" aria-hidden="true">
    <line x1="{left}" y1="{height - bottom}" x2="{width - right}"
      y2="{height - bottom}" stroke="#9aa5ad" />
    {"".join(bars)}
    {"".join(labels)}
  </svg>
  <p class="legend">{" ".join(legend)}</p>
</div>
"""


def _seed_table(result: AgentEvaluationResult) -> str:
    rows: list[str] = []
    for run_index in range(result.runs_requested):
        cells = [
            f"<td>{result.start_seed + run_index}</td>",
        ]
        for policy in result.policies:
            runs = result.runs_by_policy.get(policy, ())
            run = next((item for item in runs if item.run_index == run_index), None)
            if run is None:
                cells.append("<td>-</td>")
            else:
                detail = html_escape(run.error) if run.error else html_escape(run.final_phase)
                cells.append(
                    "<td>"
                    f"floor {run.final_floor}, reward {_format_number(run.total_reward)}, "
                    f"{detail}"
                    "</td>"
                )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "<th>Seed</th>" + "".join(
        f"<th>{html_escape(_policy_label(policy))}</th>" for policy in result.policies
    )
    return (
        "<table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _comparison_sentence(
    result: AgentEvaluationResult,
    best_summary: AgentEvaluationSummary | None,
) -> str:
    if best_summary is None:
        return "No agent runs were evaluated."
    model_note = (
        f" Q-learning model: {result.q_learning_model_path}."
        if result.q_learning_model_path is not None
        else " Q-learning is evaluated with an untrained table."
    )
    return (
        f"{len(result.policies)} policies over {result.runs_requested} fixed seeds. "
        f"Best average floor: {_policy_label(best_summary.policy)} "
        f"({_format_number(best_summary.average_floor)})."
        + model_note
    )


def _kpi(label: str, value: str) -> str:
    return (
        '<div class="kpi">'
        f"<span>{html_escape(label)}</span>"
        f"<strong>{html_escape(value)}</strong>"
        "</div>"
    )


def _policy_label(policy: str) -> str:
    return _POLICY_LABELS.get(policy, policy.replace("_", " ").title())


def _format_number(value: object) -> str:
    number = _float(value)
    if number == int(number):
        return str(int(number))
    return f"{number:.2f}"


def _format_percent(value: object) -> str:
    return f"{_float(value) * 100.0:.0f}%"


def _lookup_vector_int(observation: Mapping[str, Any], field: str) -> int:
    schema = observation.get("vector_schema")
    vector = observation.get("vector")
    if isinstance(schema, list | tuple) and isinstance(vector, list | tuple):
        for index, name in enumerate(schema):
            if str(name) == field and index < len(vector):
                return _int(vector[index])
    return 0


def _current_action_ids(info: Mapping[str, Any]) -> set[int]:
    action_space = info.get("action_space")
    if not isinstance(action_space, Sequence) or isinstance(
        action_space,
        (str, bytes, bytearray),
    ):
        return set()
    ids: set[int] = set()
    for descriptor in action_space:
        if isinstance(descriptor, Mapping):
            ids.add(_int(descriptor.get("id")))
    return ids


def _is_terminal_phase(state: Any) -> bool:
    phase = getattr(getattr(state, "phase", None), "value", getattr(state, "phase", ""))
    return str(phase) in {"complete", "failed"}


def _int(value: object) -> int:
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _float(value: object) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
