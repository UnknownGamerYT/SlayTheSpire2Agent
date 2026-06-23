"""Learning progress metrics and static HTML reports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from html import escape as html_escape
from pathlib import Path
from typing import Any

from sts2sim.learning.models import LearningProgressPoint, LearningRunResult

DEFAULT_PROGRESS_WINDOW = 10


def progress_from_runs(
    runs: Sequence[LearningRunResult | Mapping[str, Any]],
    *,
    policy: str | None = None,
    window: int = DEFAULT_PROGRESS_WINDOW,
) -> tuple[LearningProgressPoint, ...]:
    """Build run-level progress points from rollout/evaluation results."""

    points: list[LearningProgressPoint] = []
    for index, run in enumerate(runs):
        payload = _model_payload(run)
        final_phase = str(payload.get("final_phase", "unknown"))
        points.append(
            LearningProgressPoint(
                run_index=_int(payload.get("run_index", index)),
                seed=_seed(payload.get("seed", index)),
                policy=str(payload.get("policy", policy or "unknown")),
                steps_taken=_int(payload.get("steps_taken")),
                total_reward=round(_float(payload.get("total_reward")), 6),
                final_phase=final_phase,
                final_act=_int(payload.get("final_act")),
                final_floor=_int(payload.get("final_floor")),
                win=final_phase == "complete",
                death=final_phase == "failed",
                truncated=bool(payload.get("truncated", False)),
                failed_to_continue=bool(
                    payload.get("failed_to_continue", bool(payload.get("error")))
                ),
                error=_optional_str(payload.get("error")),
            )
        )
    return with_moving_averages(points, window=window)


def with_moving_averages(
    points: Sequence[LearningProgressPoint],
    *,
    window: int = DEFAULT_PROGRESS_WINDOW,
) -> tuple[LearningProgressPoint, ...]:
    """Return points with rolling reward, floor, and win-rate fields populated."""

    normalized_window = max(1, int(window))
    updated: list[LearningProgressPoint] = []
    for index, point in enumerate(points):
        chunk = tuple(points[max(0, index - normalized_window + 1) : index + 1])
        updated.append(
            point.model_copy(
                update={
                    "moving_average_reward": round(
                        sum(item.total_reward for item in chunk) / max(1, len(chunk)),
                        6,
                    ),
                    "moving_average_floor": round(
                        sum(item.final_floor for item in chunk) / max(1, len(chunk)),
                        6,
                    ),
                    "moving_win_rate": round(
                        sum(1 for item in chunk if item.win) / max(1, len(chunk)),
                        6,
                    ),
                }
            )
        )
    return tuple(updated)


def progress_from_payload(
    payload: Mapping[str, Any],
    *,
    window: int = DEFAULT_PROGRESS_WINDOW,
) -> tuple[LearningProgressPoint, ...]:
    """Extract progress points from a learning JSON payload."""

    progress = payload.get("progress")
    if isinstance(progress, Sequence) and not isinstance(progress, str | bytes | bytearray):
        points = [
            LearningProgressPoint.model_validate(item)
            for item in progress
            if isinstance(item, Mapping)
        ]
        return with_moving_averages(points, window=window)

    runs = payload.get("runs")
    if isinstance(runs, Sequence) and not isinstance(runs, str | bytes | bytearray):
        return progress_from_runs(
            tuple(item for item in runs if isinstance(item, Mapping)),
            policy=str(payload.get("policy", "unknown")),
            window=window,
        )
    return ()


def load_learning_progress(
    path: Path | str,
    *,
    window: int = DEFAULT_PROGRESS_WINDOW,
) -> tuple[LearningProgressPoint, ...]:
    """Load progress points from a training, rollout, or evaluation JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return ()
    return progress_from_payload(payload, window=window)


def progress_payload(
    points: Sequence[LearningProgressPoint],
    *,
    title: str = "Learning Progress",
    window: int = DEFAULT_PROGRESS_WINDOW,
) -> dict[str, Any]:
    """Build a JSON-friendly progress payload."""

    normalized_points = with_moving_averages(points, window=window)
    return {
        "title": title,
        "window": max(1, int(window)),
        "summary": progress_summary(normalized_points, window=window),
        "progress": [point.model_dump(mode="json") for point in normalized_points],
    }


def write_learning_progress_data(
    points: Sequence[LearningProgressPoint],
    path: Path | str,
    *,
    title: str = "Learning Progress",
    window: int = DEFAULT_PROGRESS_WINDOW,
) -> None:
    """Write progress points and summary to JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(progress_payload(points, title=title, window=window), indent=2) + "\n",
        encoding="utf-8",
    )


def build_learning_progress_report(
    *,
    input_path: Path | str,
    output_path: Path | str = Path("reports/learning_progress.html"),
    title: str = "Learning Progress",
    window: int = DEFAULT_PROGRESS_WINDOW,
) -> dict[str, Any]:
    """Create a static HTML progress report from a learning JSON file."""

    points = load_learning_progress(input_path, window=window)
    write_learning_progress_report(points, output_path, title=title, window=window)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "points": len(points),
        "summary": progress_summary(points, window=window),
    }


def write_learning_progress_report(
    points: Sequence[LearningProgressPoint],
    path: Path | str,
    *,
    title: str = "Learning Progress",
    window: int = DEFAULT_PROGRESS_WINDOW,
) -> None:
    """Write a standalone HTML report with inline SVG charts."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        learning_progress_html(points, title=title, window=window),
        encoding="utf-8",
    )


def learning_progress_html(
    points: Sequence[LearningProgressPoint],
    *,
    title: str = "Learning Progress",
    window: int = DEFAULT_PROGRESS_WINDOW,
) -> str:
    """Render a standalone static learning progress report."""

    normalized_points = with_moving_averages(points, window=window)
    summary = progress_summary(normalized_points, window=window)
    safe_title = html_escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #5f6b72;
      --line: #d7dde1;
      --panel: #ffffff;
      --paper: #f5f7f8;
      --blue: #2563eb;
      --green: #16855b;
      --amber: #b7791f;
      --red: #c24130;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1160px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 44px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 30px;
      font-weight: 760;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    p {{ margin: 0; color: var(--muted); }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 10px;
      margin: 20px 0;
    }}
    .kpi {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .kpi span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .kpi strong {{
      display: block;
      margin-top: 4px;
      font-size: 24px;
      line-height: 1.1;
    }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-top: 14px;
      padding: 16px;
      overflow: hidden;
    }}
    .chart {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .status {{
      margin-top: 8px;
      color: {("var(--red)" if summary["stuck_signal"] else "var(--green)")};
      font-weight: 700;
    }}
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
      main {{ width: min(100vw - 20px, 1160px); padding-top: 18px; }}
      h1 {{ font-size: 24px; }}
      .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .section {{ padding: 12px; }}
      table {{ display: block; overflow-x: auto; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>{safe_title}</h1>
    <p>{html_escape(_summary_sentence(summary, window=window))}</p>
    <div class="kpis">
      {_kpi("Runs", str(summary["runs"]))}
      {_kpi("Best Floor", str(summary["best_floor"]))}
      {_kpi("Best Reward", _format_number(summary["best_reward"]))}
      {_kpi(f"Last {summary["window"]} Avg Floor", _format_number(summary["recent_average_floor"]))}
      {_kpi(f"Last {summary["window"]} Win Rate", _format_percent(summary["recent_win_rate"]))}
    </div>
    <div class="section">
      <h2>Reward By Run</h2>
      {_line_chart(
          "Reward by run",
          "Reward",
          (
              ("Reward", [point.total_reward for point in normalized_points], "#2563eb"),
              (
                  f"{summary["window"]}-run average",
                  [point.moving_average_reward for point in normalized_points],
                  "#c24130",
              ),
          ),
      )}
    </div>
    <div class="section">
      <h2>Floor Progress</h2>
      {_line_chart(
          "Floor progress",
          "Floor",
          (
              ("Final floor", [float(point.final_floor) for point in normalized_points], "#16855b"),
              (
                  f"{summary["window"]}-run average",
                  [point.moving_average_floor for point in normalized_points],
                  "#b7791f",
              ),
          ),
          y_floor=0.0,
      )}
    </div>
    <div class="section">
      <h2>Win Rate And Run Length</h2>
      {_line_chart(
          "Rolling win rate",
          "Win rate",
          (
              (
                  f"{summary["window"]}-run win rate",
                  [point.moving_win_rate * 100.0 for point in normalized_points],
                  "#2563eb",
              ),
          ),
          y_floor=0.0,
          y_ceiling=100.0,
          suffix="%",
      )}
      {_line_chart(
          "Steps taken",
          "Steps",
          (("Steps", [float(point.steps_taken) for point in normalized_points], "#5f6b72"),),
          y_floor=0.0,
      )}
    </div>
    <div class="section">
      <h2>Progress Signal</h2>
      <p>{html_escape(str(summary["stuck_reason"]))}</p>
      <p class="status">{html_escape(str(summary["status"]))}</p>
    </div>
    <div class="section">
      <h2>Recent Runs</h2>
      {_recent_table(normalized_points)}
    </div>
  </main>
</body>
</html>
"""


def progress_summary(
    points: Sequence[LearningProgressPoint],
    *,
    window: int = DEFAULT_PROGRESS_WINDOW,
) -> dict[str, Any]:
    """Summarize learning progress and a simple stuck signal."""

    normalized_window = max(1, min(max(1, len(points)), int(window)))
    if not points:
        return {
            "runs": 0,
            "window": normalized_window,
            "best_floor": 0,
            "best_reward": 0.0,
            "average_reward": 0.0,
            "average_floor": 0.0,
            "recent_average_reward": 0.0,
            "recent_average_floor": 0.0,
            "recent_win_rate": 0.0,
            "wins": 0,
            "deaths": 0,
            "failed_to_continue": 0,
            "errors": 0,
            "stuck_signal": False,
            "stuck_reason": "No runs have been recorded yet.",
            "status": "Waiting for training data.",
        }

    recent = tuple(points[-normalized_window:])
    previous = tuple(points[-2 * normalized_window : -normalized_window])
    average_reward = sum(point.total_reward for point in points) / len(points)
    average_floor = sum(point.final_floor for point in points) / len(points)
    recent_reward = sum(point.total_reward for point in recent) / len(recent)
    recent_floor = sum(point.final_floor for point in recent) / len(recent)
    recent_win_rate = sum(1 for point in recent if point.win) / len(recent)
    stuck_signal = False
    stuck_reason = (
        "Not enough history for a stuck signal yet."
        if len(points) < normalized_window * 2
        else "Recent runs improved or stayed healthy."
    )
    if previous:
        previous_reward = sum(point.total_reward for point in previous) / len(previous)
        previous_floor = sum(point.final_floor for point in previous) / len(previous)
        previous_win_rate = sum(1 for point in previous if point.win) / len(previous)
        floor_flat = recent_floor <= previous_floor + 0.05
        reward_flat = recent_reward <= previous_reward + 0.05
        win_flat = recent_win_rate <= previous_win_rate + 0.01
        stuck_signal = floor_flat and reward_flat and win_flat
        if stuck_signal:
            stuck_reason = (
                "The recent window did not improve floor, reward, or win rate "
                "compared with the previous window."
            )

    return {
        "runs": len(points),
        "window": normalized_window,
        "best_floor": max(point.final_floor for point in points),
        "best_reward": round(max(point.total_reward for point in points), 6),
        "average_reward": round(average_reward, 6),
        "average_floor": round(average_floor, 6),
        "recent_average_reward": round(recent_reward, 6),
        "recent_average_floor": round(recent_floor, 6),
        "recent_win_rate": round(recent_win_rate, 6),
        "wins": sum(1 for point in points if point.win),
        "deaths": sum(1 for point in points if point.death),
        "failed_to_continue": sum(1 for point in points if point.failed_to_continue),
        "errors": sum(1 for point in points if point.error is not None),
        "stuck_signal": stuck_signal,
        "stuck_reason": stuck_reason,
        "status": "Possibly stuck" if stuck_signal else "Learning signal looks active",
    }


def _line_chart(
    title: str,
    y_label: str,
    series: Sequence[tuple[str, Sequence[float], str]],
    *,
    y_floor: float | None = None,
    y_ceiling: float | None = None,
    suffix: str = "",
) -> str:
    width = 920
    height = 260
    left = 58
    right = 24
    top = 28
    bottom = 42
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [float(value) for _label, data, _color in series for value in data]
    if not values:
        return '<p>No data yet.</p>'
    y_min = min(values) if y_floor is None else min(y_floor, min(values))
    y_max = max(values) if y_ceiling is None else max(y_ceiling, max(values))
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    max_count = max(len(data) for _label, data, _color in series)

    def x_at(index: int) -> float:
        if max_count <= 1:
            return left + plot_width / 2
        return left + (plot_width * index / (max_count - 1))

    def y_at(value: float) -> float:
        return top + (plot_height * (y_max - value) / (y_max - y_min))

    grid = []
    for tick in range(5):
        value = y_min + ((y_max - y_min) * tick / 4)
        y = y_at(value)
        label = f"{_format_number(value)}{suffix}"
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
            'stroke="#e6eaed" />'
        )
        grid.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="11" fill="#5f6b72">{html_escape(label)}</text>'
        )

    lines = []
    legend = []
    for label, data, color in series:
        if not data:
            continue
        points_attr = " ".join(
            f"{x_at(index):.2f},{y_at(float(value)):.2f}"
            for index, value in enumerate(data)
        )
        lines.append(
            f'<polyline points="{points_attr}" fill="none" stroke="{color}" '
            'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" />'
        )
        last_index = len(data) - 1
        last_value = float(data[-1])
        lines.append(
            f'<circle cx="{x_at(last_index):.2f}" cy="{y_at(last_value):.2f}" '
            f'r="3.5" fill="{color}" />'
        )
        lines.append(
            f'<text x="{min(width - right - 140, x_at(last_index) + 7):.2f}" '
            f'y="{y_at(last_value) - 6:.2f}" font-size="11" fill="{color}">'
            f'{html_escape(label)}</text>'
        )
        legend.append(
            f'<span><i style="background:{color}"></i>{html_escape(label)}</span>'
        )

    x_end_label = str(max(0, max_count - 1))
    return f"""
<div aria-label="{html_escape(title)}" role="img">
  <svg class="chart" viewBox="0 0 {width} {height}" aria-hidden="true">
    <text x="{left}" y="18" font-size="13" fill="#172026">{html_escape(y_label)}</text>
    {"".join(grid)}
    <line x1="{left}" y1="{height - bottom}" x2="{width - right}"
      y2="{height - bottom}" stroke="#9aa5ad" />
    <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"
      stroke="#9aa5ad" />
    {"".join(lines)}
    <text x="{left}" y="{height - 13}" font-size="11" fill="#5f6b72">run 0</text>
    <text x="{width - right}" y="{height - 13}" font-size="11"
      text-anchor="end" fill="#5f6b72">run {html_escape(x_end_label)}</text>
  </svg>
  <p class="legend">{" ".join(legend)}</p>
</div>
"""


def _recent_table(points: Sequence[LearningProgressPoint]) -> str:
    rows = []
    for point in points[-20:]:
        rows.append(
            "<tr>"
            f"<td>{point.run_index}</td>"
            f"<td>{html_escape(str(point.seed))}</td>"
            f"<td>{html_escape(point.final_phase)}</td>"
            f"<td>{point.final_act}</td>"
            f"<td>{point.final_floor}</td>"
            f"<td>{point.steps_taken}</td>"
            f"<td>{_format_number(point.total_reward)}</td>"
            f"<td>{_format_number(point.moving_average_floor)}</td>"
            f"<td>{_format_percent(point.moving_win_rate)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Run</th><th>Seed</th><th>Phase</th><th>Act</th><th>Floor</th>"
        "<th>Steps</th><th>Reward</th><th>Moving Floor</th><th>Moving Win</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _kpi(label: str, value: str) -> str:
    return (
        '<div class="kpi">'
        f"<span>{html_escape(label)}</span>"
        f"<strong>{html_escape(value)}</strong>"
        "</div>"
    )


def _summary_sentence(summary: Mapping[str, Any], *, window: int) -> str:
    return (
        f"{summary['runs']} runs tracked. Recent window uses up to {window} runs. "
        f"Best floor {summary['best_floor']}, wins {summary['wins']}, deaths {summary['deaths']}."
    )


def _format_number(value: object) -> str:
    number = _float(value)
    if number == int(number):
        return str(int(number))
    return f"{number:.2f}"


def _format_percent(value: object) -> str:
    return f"{_float(value) * 100.0:.0f}%"


def _model_payload(value: LearningRunResult | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return value.model_dump(mode="json")


def _seed(value: object) -> int | str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return value
    return str(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


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
