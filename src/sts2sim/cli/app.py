"""Typer commands for simulator data, runs, replay, and fuzzing.

The simulator core is owned by sibling workers. This module intentionally keeps
the CLI thin: each command resolves a planned public callable at runtime, passes
through structured arguments, and renders the returned object as JSON.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, cast

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Headless Slay the Spire 2 simulator tools.",
)


class BackendUnavailable(RuntimeError):
    """Raised when a CLI command cannot find its simulator backend."""


def _resolve_backend(
    command: str,
    module_names: Sequence[str],
    callable_names: Sequence[str],
) -> Callable[..., Any]:
    """Find the first available backend callable for a command."""

    import_errors: list[str] = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name and (
                exc.name == module_name or module_name.startswith(f"{exc.name}.")
            ):
                continue
            import_errors.append(f"{module_name}: {exc}")
            continue
        except Exception as exc:  # pragma: no cover - defensive import surface
            import_errors.append(f"{module_name}: {exc}")
            continue

        for callable_name in callable_names:
            candidate = getattr(module, callable_name, None)
            if callable(candidate):
                return cast(Callable[..., Any], candidate)

    searched = ", ".join(
        f"{module}.{name}" for module in module_names for name in callable_names
    )
    detail = f" Searched: {searched}."
    if import_errors:
        detail += f" Import errors: {'; '.join(import_errors)}."
    raise BackendUnavailable(
        f"{command} backend is not available yet.{detail}"
    )


def _call_backend(func: Callable[..., Any], **kwargs: Any) -> Any:
    """Call a backend with only the keyword arguments its signature accepts."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - builtin or opaque callables
        return func(**kwargs)

    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return func(**kwargs)

    accepted = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return func(**accepted)


def _jsonable(value: Any) -> Any:
    """Convert common result objects into JSON-serializable values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _jsonable(value.as_dict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(cast(Any, value)))
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_asdict"):
        return _jsonable(value._asdict())
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if hasattr(value, "__dict__"):
        return {
            key: _jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return repr(value)


def _emit(value: Any) -> Any:
    """Print a stable JSON representation and return the normalized payload."""

    payload = _jsonable(value)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _write_result_if_missing(output_path: Path | None, payload: Any) -> None:
    """Persist CLI output when the backend did not create an output file."""

    if output_path is None or output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _emit_training_result(result: Any, output_path: Path | None) -> None:
    """Print a compact training summary while preserving the full output file."""

    payload = _jsonable(result)
    _emit(_compact_training_result(payload))
    _write_result_if_missing(output_path, payload)


def _compact_training_result(payload: Any) -> Any:
    if not isinstance(payload, Mapping):
        return payload

    summary_keys = (
        "algorithm",
        "target",
        "reached_target",
        "reached_batch",
        "batches_completed",
        "max_batches",
        "until_stopped",
        "previous_batches",
        "requested_new_batches",
        "batch_limit",
        "runs_trained",
        "total_steps",
        "average_training_reward",
        "wins",
        "deaths",
        "resumed_from_path",
        "checkpoint_decision",
        "checkpoint_compatibility_checks",
        "model_path",
        "output_path",
        "progress_output_path",
        "report_output_path",
    )
    compact: dict[str, Any] = {
        key: payload.get(key) for key in summary_keys if key in payload
    }

    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        compact["metadata"] = {
            key: metadata.get(key)
            for key in (
                "character_id",
                "character_name",
                "ascension",
                "network_schema_version",
                "reward_schema_version",
                "network_contract_checksum",
                "reward_config_checksum",
                "game_logic_checksum",
                "checkpoint_compatibility_checks",
                "parameter_count",
                "device",
                "cuda_available",
                "cuda_device_name",
            )
            if key in metadata
        }

    batches = payload.get("batch_summaries")
    if isinstance(batches, Sequence) and not isinstance(batches, (str, bytes, bytearray)):
        latest_batch = next(
            (batch for batch in reversed(tuple(batches)) if isinstance(batch, Mapping)),
            None,
        )
        if latest_batch is not None:
            compact["latest_batch"] = {
                key: latest_batch.get(key)
                for key in (
                    "batch_index",
                    "trained_runs_total",
                    "train_total_steps",
                    "evaluation_runs",
                    "evaluation_average_reward",
                    "evaluation_average_floor",
                    "evaluation_target_success_rate",
                    "evaluation_errors",
                    "evaluation_failed_to_continue",
                    "reached_target",
                )
                if key in latest_batch
            }

    highlights = payload.get("highlight_run_histories")
    if isinstance(highlights, Mapping):
        compact["highlight_run_histories"] = _compact_highlight_histories(highlights)

    return compact


def _compact_highlight_histories(histories: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        key: histories.get(key)
        for key in ("schema_version", "generated_at")
        if key in histories
    }
    for role in ("best", "worst"):
        entry = histories.get(role)
        if not isinstance(entry, Mapping):
            continue
        compact[role] = {
            key: entry.get(key)
            for key in (
                "role",
                "generated_at",
                "run_index",
                "seed",
                "character_id",
                "ascension",
                "steps_taken",
                "total_reward",
                "final_phase",
                "final_act",
                "final_floor",
                "target_reached",
                "failed_to_continue",
                "error",
                "json_path",
                "html_path",
                "map_path",
                "summary_json_path",
                "summary_txt_path",
                "summary_html_path",
            )
            if key in entry
        }
    return compact


class _TrainingTerminalProgress:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._progress: Any | None = None
        self._train_task: Any | None = None
        self._eval_task: Any | None = None
        self._train_started_at: float | None = None
        self._eval_started_at: float | None = None

    def __enter__(self) -> _TrainingTerminalProgress:
        if not self.enabled:
            return self
        try:
            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TextColumn("{task.fields[eta]}"),
                transient=False,
            )
            progress.__enter__()
            self._progress = progress
        except Exception:
            self._progress = None
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._progress is not None:
            self._progress.__exit__(exc_type, exc, traceback)

    def __call__(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        if self._progress is None:
            self._print_plain(payload)
            return
        self._update_rich(payload)

    def _update_rich(self, payload: Mapping[str, Any]) -> None:
        event = str(payload.get("event", ""))
        progress = self._progress
        if progress is None:
            return

        if event == "trainer_start":
            progress.console.log(
                "PPO training started: "
                f"target={_target_name(payload.get('target'))}, "
                f"start_batch={payload.get('start_batch')}, "
                f"device={payload.get('device')}, "
                f"workers={payload.get('rollout_workers')}, "
                f"inference={payload.get('rollout_inference')}, "
                f"history={payload.get('history_mode')}, "
                f"active_envs={payload.get('active_env_streams')}, "
                f"until_stopped={payload.get('until_stopped')}"
            )
            for line in _checkpoint_check_lines(payload):
                progress.console.log(line)
            return
        if event == "batch_start":
            self._remove_task(self._train_task)
            self._remove_task(self._eval_task)
            self._eval_task = None
            self._train_started_at = time.perf_counter()
            self._eval_started_at = None
            total = _progress_int(payload.get("train_runs_per_batch"))
            self._train_task = progress.add_task(
                (
                    f"Batch {payload.get('batch_index')} "
                    f"{_progress_target_name(payload)} train runs"
                ),
                total=max(1, total),
                eta=_eta_progress_text(self._train_started_at, 0, total),
            )
            progress.console.log(
                f"Batch {payload.get('batch_index')} started "
                f"stage={_progress_target_name(payload)} "
                f"({total} train runs, {payload.get('eval_runs')} eval runs)"
            )
            return
        if event == "train_run_end":
            if self._train_task is not None:
                completed = _progress_int(payload.get("run_position"))
                total = _progress_int(payload.get("run_total"))
                progress.update(
                    self._train_task,
                    completed=completed,
                    description=(
                        f"Batch {payload.get('batch_index')} train "
                        f"{_progress_target_name(payload)} "
                        f"{_act_floor_text(payload)}"
                    ),
                    eta=_eta_progress_text(self._train_started_at, completed, total),
                )
            if payload.get("failed_to_continue") or payload.get("error"):
                progress.console.log(f"Train run issue: {_run_progress_line(payload)}")
            return
        if event == "ppo_update_start":
            progress.console.log(
                f"Batch {payload.get('batch_index')} PPO update "
                f"({payload.get('transition_count')} transitions)"
            )
            return
        if event == "ppo_update_end":
            progress.console.log(f"Batch {payload.get('batch_index')} PPO update complete")
            return
        if event == "eval_start":
            self._remove_task(self._eval_task)
            self._eval_started_at = time.perf_counter()
            total = _progress_int(payload.get("eval_runs"))
            self._eval_task = progress.add_task(
                (
                    f"Batch {payload.get('batch_index')} "
                    f"{_progress_target_name(payload)} eval runs"
                ),
                total=max(1, total),
                eta=_eta_progress_text(self._eval_started_at, 0, total),
            )
            return
        if event == "eval_run_end":
            if self._eval_task is not None:
                completed = _progress_int(payload.get("run_position"))
                total = _progress_int(payload.get("run_total"))
                progress.update(
                    self._eval_task,
                    completed=completed,
                    description=(
                        f"Batch {payload.get('batch_index')} eval "
                        f"{_progress_target_name(payload)} "
                        f"{_act_floor_text(payload)}"
                    ),
                    eta=_eta_progress_text(self._eval_started_at, completed, total),
                )
            # Evaluation is small enough to show every completed run.  Printing only
            # successes made the final success count look inconsistent with the log.
            progress.console.log(f"Eval run: {_run_progress_line(payload)}")
            return
        if event == "batch_saved":
            progress.console.log(_batch_progress_line(payload))
            reward_line = _reward_signal_line(payload)
            if reward_line:
                progress.console.log(reward_line)
            diagnostic_line = _diagnostic_progress_line(payload)
            if diagnostic_line:
                progress.console.log(diagnostic_line)
            throughput_line = _throughput_progress_line(payload)
            if throughput_line:
                progress.console.log(throughput_line)

    def _print_plain(self, payload: Mapping[str, Any]) -> None:
        event = str(payload.get("event", ""))
        if event == "trainer_start":
            typer.echo(
                "PPO training started: "
                f"target={_target_name(payload.get('target'))}, "
                f"start_batch={payload.get('start_batch')}, "
                f"device={payload.get('device')}, "
                f"workers={payload.get('rollout_workers')}, "
                f"inference={payload.get('rollout_inference')}, "
                f"history={payload.get('history_mode')}, "
                f"active_envs={payload.get('active_env_streams')}, "
                f"until_stopped={payload.get('until_stopped')}"
            )
            for line in _checkpoint_check_lines(payload):
                typer.echo(line)
        elif event == "batch_start":
            typer.echo(
                f"Batch {payload.get('batch_index')} started: "
                f"stage={_progress_target_name(payload)}, "
                f"{payload.get('train_runs_per_batch')} train runs, "
                f"{payload.get('eval_runs')} eval runs"
            )
        elif event in {"train_run_end", "eval_run_end"}:
            typer.echo(f"{event}: {_run_progress_line(payload)}")
        elif event == "ppo_update_start":
            typer.echo(
                f"Batch {payload.get('batch_index')} PPO update "
                f"({payload.get('transition_count')} transitions)"
            )
        elif event == "ppo_update_end":
            typer.echo(f"Batch {payload.get('batch_index')} PPO update complete")
        elif event == "batch_saved":
            typer.echo(_batch_progress_line(payload))
            reward_line = _reward_signal_line(payload)
            if reward_line:
                typer.echo(reward_line)
            diagnostic_line = _diagnostic_progress_line(payload)
            if diagnostic_line:
                typer.echo(diagnostic_line)
            throughput_line = _throughput_progress_line(payload)
            if throughput_line:
                typer.echo(throughput_line)

    def _remove_task(self, task_id: Any | None) -> None:
        if self._progress is None or task_id is None:
            return
        try:
            self._progress.remove_task(task_id)
        except KeyError:
            return


def _target_name(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    if isinstance(value, Mapping):
        return str(value.get("name", "unknown"))
    return str(value)


def _progress_target_name(payload: Mapping[str, Any]) -> str:
    target_name = payload.get("target_name")
    if target_name is not None and target_name != "":
        return str(target_name)
    return _target_name(payload.get("target"))


def _checkpoint_check_lines(payload: Mapping[str, Any]) -> list[str]:
    checks = [
        check
        for check in payload.get("checkpoint_checks", [])
        if isinstance(check, Mapping)
    ]
    decision = str(payload.get("checkpoint_decision", "fresh"))
    if not checks:
        return [f"Checkpoint: {decision}; no checkpoint checks were needed."]
    lines: list[str] = []
    for check in checks:
        path = str(check.get("checkpoint_path", "") or "")
        check_decision = str(check.get("decision", "fresh"))
        reason = str(check.get("reason", ""))
        prefix = f"Checkpoint: {check_decision}"
        if path:
            prefix += f" {path}"
        if reason:
            prefix += f" ({reason})"
        lines.append(prefix)
        mismatches = [
            mismatch
            for mismatch in check.get("mismatches", [])
            if isinstance(mismatch, Mapping)
        ]
        if mismatches:
            keys = ", ".join(str(mismatch.get("key", "")) for mismatch in mismatches[:8])
            if len(mismatches) > 8:
                keys += f", +{len(mismatches) - 8} more"
            lines.append(f"  changed: {keys}")
    return lines


def _run_progress_line(payload: Mapping[str, Any]) -> str:
    return (
        f"{payload.get('run_position')}/{payload.get('run_total')} "
        f"stage={_progress_target_name(payload)} "
        f"seed={payload.get('seed')} "
        f"reward={_progress_float(payload.get('total_reward')):.2f} "
        f"{_act_floor_text(payload)} "
        f"phase={payload.get('final_phase')} "
        f"steps={payload.get('steps_taken')} "
        f"target={payload.get('reached_target')}"
    )


def _act_floor_text(payload: Mapping[str, Any]) -> str:
    return f"act={payload.get('final_act')} floor={payload.get('final_floor')}"


def _eta_progress_text(
    started_at: float | None,
    completed: int,
    total: int,
    *,
    now: float | None = None,
) -> str:
    """Return stable ETA text for terminal progress bars."""

    if total <= 0:
        return "eta -"
    if completed >= total:
        return "eta 0:00"
    if started_at is None or completed <= 0:
        return "eta calculating"
    current_time = time.perf_counter() if now is None else now
    elapsed = max(0.0, current_time - started_at)
    if elapsed <= 0.0:
        return "eta calculating"
    remaining = (elapsed / max(1, completed)) * max(0, total - completed)
    return f"eta {_format_progress_duration(remaining)}"


def _format_progress_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def _batch_progress_line(payload: Mapping[str, Any]) -> str:
    success_rate = _progress_float(payload.get("evaluation_target_success_rate"))
    success_target = _progress_float(payload.get("target_success_rate_threshold"))
    return (
        f"Batch {payload.get('batch_index')} saved: "
        f"stage={_progress_target_name(payload)}, "
        f"success={success_rate:.3f}/{success_target:.3f} "
        f"({payload.get('target_successes')}/{payload.get('eval_runs')} eval), "
        f"avg_floor={_progress_float(payload.get('evaluation_average_floor')):.2f}, "
        f"best_floor={_progress_float(payload.get('evaluation_best_floor')):.0f}, "
        f"avg_reward={_progress_float(payload.get('evaluation_average_reward')):.2f}, "
        f"best_reward={_progress_float(payload.get('evaluation_best_reward')):.2f}, "
        f"consec={payload.get('evaluation_max_consecutive_successes')}, "
        f"errors={_progress_int(payload.get('evaluation_errors'))}, "
        f"failed={_progress_int(payload.get('evaluation_failed_to_continue'))}, "
        f"runs_trained={payload.get('runs_trained')}"
    )


def _reward_signal_line(payload: Mapping[str, Any]) -> str:
    rewards = _progress_mapping(payload.get("reward_component_averages"))
    if not rewards:
        return ""
    parts = [
        f"total={_progress_float(rewards.get('total')):.2f}",
        f"combat={_progress_float(rewards.get('combat_win_reward')):.2f}",
        f"boss={_progress_float(rewards.get('boss_reward')):.2f}",
        f"enemy_hp={_progress_float(rewards.get('enemy_hp_progress_reward')):.2f}",
        f"hp_loss={_progress_float(rewards.get('hp_loss_penalty')):.2f}",
        f"gold={_progress_float(rewards.get('gold_reward')):.2f}",
        f"skip={_progress_float(rewards.get('reward_skip_penalty')):.2f}",
        f"opp={_progress_float(rewards.get('opportunity_cost_penalty')):.2f}",
        f"deck={_progress_float(rewards.get('deck_capability_reward')):.2f}",
    ]
    return "  reward avg: " + ", ".join(parts)


def _diagnostic_progress_line(payload: Mapping[str, Any]) -> str:
    diagnostics = _progress_mapping(payload.get("diagnostic_averages"))
    if not diagnostics:
        return ""
    parts = [
        _reward_pickup_fragment(diagnostics, "card", "cards", "take_reward_card"),
        _reward_pickup_fragment(diagnostics, "gold", "gold", "take_reward_gold"),
        _reward_pickup_fragment(diagnostics, "relic", "relics", "take_reward_relic"),
        _reward_pickup_fragment(diagnostics, "potion", "potions", "take_reward_potion"),
        _reward_pickup_fragment(diagnostics, "card_removal", "removes", ""),
        f"final_deck={_progress_float(diagnostics.get('final_deck_size')):.1f}",
        f"unknown_cards={_progress_float(diagnostics.get('final_unknown_card_count')):.1f}",
        f"final_gold={_progress_float(diagnostics.get('final_gold')):.1f}",
    ]
    return "  deck/items avg: " + ", ".join(parts)


def _throughput_progress_line(payload: Mapping[str, Any]) -> str:
    throughput = _progress_mapping(payload.get("throughput"))
    if not throughput:
        return ""
    parts = [
        f"steps/s={_progress_float(throughput.get('env_steps_per_second')):.1f}",
        f"runs/s={_progress_float(throughput.get('runs_per_second')):.2f}",
        f"active_envs={_progress_int(throughput.get('active_env_streams'))}",
        f"min_batch={_progress_int(throughput.get('policy_server_min_batch'))}",
        f"wait_ms={_progress_int(throughput.get('policy_server_max_wait_ms'))}",
    ]
    return "  throughput: " + ", ".join(parts)


def _reward_pickup_fragment(
    diagnostics: Mapping[str, Any],
    kind: str,
    label: str,
    legacy_pick_key: str,
) -> str:
    picked = _progress_float(diagnostics.get(f"reward_{kind}_picked"))
    if picked == 0.0 and legacy_pick_key:
        picked = _progress_float(diagnostics.get(legacy_pick_key))
    presented = _progress_float(diagnostics.get(f"reward_{kind}_presented"))
    missed = _progress_float(diagnostics.get(f"reward_{kind}_skipped")) + _progress_float(
        diagnostics.get(f"reward_{kind}_unclaimed")
    )
    if presented > 0.0 or missed > 0.0:
        return f"{label}={picked:.2f}/{presented:.2f} missed={missed:.2f}"
    return f"{label}={picked:.2f}"


def _progress_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _progress_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def _progress_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _coverage_value(payload: Any) -> float | None:
    """Extract a coverage ratio from common audit result shapes."""

    if not isinstance(payload, Mapping):
        return None

    for key in ("coverage", "coverage_ratio", "ratio", "overall"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value) / 100.0 if value > 1 else float(value)

    summary = payload.get("summary")
    if isinstance(summary, Mapping):
        return _coverage_value(summary)

    audit_values = list(payload.values())
    if audit_values and all(isinstance(item, Mapping) for item in audit_values):
        checked = 0
        passed = 0
        for item in audit_values:
            if "count_ok" not in item and "sha256_ok" not in item:
                continue
            checked += 1
            if item.get("count_ok", True) and item.get("sha256_ok", True):
                passed += 1
        if checked:
            return passed / checked
    return None


def _event_audit_payload(
    result: Any,
    *,
    category: str | None,
    summary_only: bool,
) -> Any:
    """Normalize event audit output and optionally filter entries by category."""

    payload = _jsonable(result)
    if not isinstance(payload, Mapping):
        return payload

    normalized_payload: dict[str, Any] = {
        str(key): value for key, value in payload.items()
    }
    entries_key = "entries" if "entries" in normalized_payload else "events"
    entries_value = normalized_payload.get(entries_key, [])
    entries = list(entries_value) if isinstance(entries_value, list) else []
    entries = [_normalized_event_entry(entry) for entry in entries]

    if category is not None:
        wanted = _normalize_event_category(category)
        entries = [
            entry
            for entry in entries
            if isinstance(entry, Mapping)
            and _normalize_event_category(str(entry.get("category", ""))) == wanted
        ]

    counts: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        category_value = str(entry.get("category", "unknown"))
        counts[category_value] = counts.get(category_value, 0) + 1

    normalized_payload["total_events"] = len(entries)
    normalized_payload["counts_by_category"] = counts
    if summary_only:
        normalized_payload.pop("entries", None)
        normalized_payload.pop("events", None)
    else:
        normalized_payload[entries_key] = entries
    return normalized_payload


def _combat_audit_payload(
    result: Any,
    *,
    category: str | None,
    status: str | None,
    summary_only: bool,
) -> Any:
    """Normalize combat audit output and optionally filter content entries."""

    payload = _jsonable(result)
    if not isinstance(payload, Mapping):
        return payload

    normalized_payload: dict[str, Any] = {str(key): value for key, value in payload.items()}
    entries_value = normalized_payload.get("entries", [])
    entries = list(entries_value) if isinstance(entries_value, list) else []
    entries = [
        {str(key): value for key, value in entry.items()}
        for entry in entries
        if isinstance(entry, Mapping)
    ]

    if category is not None:
        wanted_category = _normalize_combat_category(category)
        entries = [
            entry
            for entry in entries
            if _normalize_combat_category(str(entry.get("category", ""))) == wanted_category
        ]
    if status is not None:
        wanted_status = _normalize_combat_status(status)
        entries = [
            entry
            for entry in entries
            if _normalize_combat_status(str(entry.get("status", ""))) == wanted_status
        ]

    normalized_payload["total_ids"] = len(entries)
    normalized_payload["counts_by_category"] = _combat_counts_by_category(entries)
    normalized_payload["sample_unknown_ids"] = _combat_unknown_samples(entries)
    if summary_only:
        normalized_payload.pop("entries", None)
    else:
        normalized_payload["entries"] = entries
    return normalized_payload


def _card_audit_payload(
    result: Any,
    *,
    status: str | None,
    color: str | None,
    card_type: str | None,
    summary_only: bool,
) -> Any:
    """Normalize card audit output and optionally filter card entries."""

    payload = _jsonable(result)
    if not isinstance(payload, Mapping):
        return payload

    normalized_payload: dict[str, Any] = {str(key): value for key, value in payload.items()}
    entries_value = normalized_payload.get("entries", [])
    entries = list(entries_value) if isinstance(entries_value, list) else []
    entries = [
        {str(key): value for key, value in entry.items()}
        for entry in entries
        if isinstance(entry, Mapping)
    ]

    if status is not None:
        wanted_status = _normalize_card_status(status)
        entries = [
            entry
            for entry in entries
            if _normalize_card_status(str(entry.get("status", ""))) == wanted_status
        ]
    if color is not None:
        wanted_color = _normalize_card_bucket(color)
        entries = [
            entry
            for entry in entries
            if _normalize_card_bucket(str(entry.get("color", ""))) == wanted_color
        ]
    if card_type is not None:
        wanted_type = _normalize_card_bucket(card_type)
        entries = [
            entry
            for entry in entries
            if _normalize_card_bucket(str(entry.get("card_type", ""))) == wanted_type
        ]

    normalized_payload["total_cards"] = len(entries)
    normalized_payload["counts_by_status"] = _card_counts_by_status(entries)
    normalized_payload["counts_by_color"] = _card_counts_by_bucket(entries, "color")
    normalized_payload["counts_by_type"] = _card_counts_by_bucket(entries, "card_type")
    normalized_payload["sample_partial_ids"] = _card_sample_ids(entries, "partial")
    normalized_payload["sample_missing_ids"] = _card_sample_ids(entries, "missing")
    if summary_only:
        normalized_payload.pop("entries", None)
    else:
        normalized_payload["entries"] = entries
    return normalized_payload


def _card_counts_by_status(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"implemented": 0, "partial": 0, "missing": 0}
    for entry in entries:
        status = _normalize_card_status(str(entry.get("status", "")))
        if status in counts:
            counts[status] += 1
    return counts


def _card_counts_by_bucket(
    entries: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for entry in entries:
        bucket_name = _normalize_card_bucket(str(entry.get(key, "unknown")))
        bucket = counts.setdefault(
            bucket_name,
            {"implemented": 0, "partial": 0, "missing": 0},
        )
        status = _normalize_card_status(str(entry.get("status", "")))
        if status in bucket:
            bucket[status] += 1
    return counts


def _card_sample_ids(
    entries: Sequence[Mapping[str, Any]],
    status: str,
) -> list[str]:
    wanted_status = _normalize_card_status(status)
    result: list[str] = []
    for entry in entries:
        if _normalize_card_status(str(entry.get("status", ""))) != wanted_status:
            continue
        result.append(str(entry.get("content_id", entry.get("normalized_id", ""))))
        if len(result) >= 10:
            break
    return result


def _combat_counts_by_category(entries: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for entry in entries:
        category = str(entry.get("category", "unknown"))
        status = str(entry.get("status", "unknown"))
        bucket = counts.setdefault(
            category,
            {"total": 0, "implemented": 0, "blocked": 0, "unknown": 0},
        )
        bucket["total"] += 1
        if status in bucket:
            bucket[status] += 1
    return counts


def _combat_unknown_samples(entries: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    samples: dict[str, list[str]] = {}
    for entry in entries:
        if str(entry.get("status")) != "unknown":
            continue
        category = str(entry.get("category", "unknown"))
        if len(samples.setdefault(category, [])) < 5:
            samples[category].append(str(entry.get("content_id", entry.get("normalized_id", ""))))
    return samples


def _normalized_event_entry(entry: Any) -> Any:
    if not isinstance(entry, Mapping):
        return entry
    normalized = {str(key): value for key, value in entry.items()}
    if "category" in normalized:
        normalized["category"] = _event_category_value(normalized["category"])
    return normalized


def _event_category_value(value: object) -> str:
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    return str(value)


def _normalize_event_category(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "ancient": "ancient-only",
        "ancient-only": "ancient-only",
        "bespoke": "unsupported/bespoke",
        "flow": "stepwise",
        "flows": "stepwise",
        "unsupported": "unsupported/bespoke",
        "unsupported-bespoke": "unsupported/bespoke",
    }
    return aliases.get(normalized, normalized)


def _normalize_combat_category(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "card": "cards",
        "cards": "cards",
        "relic": "relics",
        "relics": "relics",
        "potion": "potions",
        "potions": "potions",
        "monster": "monsters",
        "monsters": "monsters",
        "encounter": "encounters",
        "encounters": "encounters",
    }
    return aliases.get(normalized, normalized)


def _normalize_combat_status(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "done": "implemented",
        "implemented": "implemented",
        "blocked": "blocked",
        "unsupported": "blocked",
        "missing": "unknown",
        "unknown": "unknown",
    }
    return aliases.get(normalized, normalized)


def _normalize_card_status(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "blocked": "partial",
        "done": "implemented",
        "implemented": "implemented",
        "missing": "missing",
        "partial": "partial",
        "unknown": "missing",
    }
    return aliases.get(normalized, normalized)


def _normalize_card_bucket(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_lang_cache_dir(cache_dir: Path | None, lang: str = "eng") -> Path | None:
    """Resolve a synced cache root to the language leaf used by content audits."""

    if cache_dir is None:
        return None
    resolved = Path(cache_dir)
    lang_dir = resolved / lang
    if lang_dir.is_dir():
        return lang_dir
    return resolved


def _backend_error(exc: Exception) -> None:
    typer.secho(str(exc), fg=typer.colors.RED, err=True)
    raise typer.Exit(2)


def _call_replay_backend(
    backend: Callable[..., Any],
    replay_path: Path,
    data_dir: Path | None,
    strict: bool,
) -> Any:
    """Call either a path-based replay backend or the seed/actions public API."""

    try:
        signature = inspect.signature(backend)
    except (TypeError, ValueError):  # pragma: no cover - opaque callables
        return backend(replay_path=replay_path, data_dir=data_dir, strict=strict)

    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_kwargs or "replay_path" in parameters or "path" in parameters:
        return _call_backend(
            backend,
            replay_path=replay_path,
            path=replay_path,
            data_dir=data_dir,
            strict=strict,
        )

    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    actions = payload.get("actions")
    if actions is None and isinstance(payload.get("transcript"), list):
        actions = [
            entry.get("action")
            for entry in payload["transcript"]
            if isinstance(entry, Mapping) and "action" in entry
        ]

    return _call_backend(
        backend,
        seed=payload.get("seed"),
        character_id=payload.get("character_id", payload.get("character")),
        ascension=payload.get("ascension", 0),
        actions=actions or [],
        data_dir=data_dir,
        strict=strict,
    )


@app.command("sync-data")
def sync_data(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            "-d",
            help="Directory where normalized game data should be stored.",
        ),
    ] = Path("data"),
    manifest_path: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            "-m",
            help="Path to a local data manifest.",
        ),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option(
            "--source",
            help="Remote manifest or data source identifier.",
        ),
    ] = None,
    lang: Annotated[
        str,
        typer.Option(
            "--lang",
            help="Language code for localized source data.",
        ),
    ] = "eng",
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Fetch and verify files even when local copies already exist.",
        ),
    ] = False,
) -> None:
    """Synchronize local data files from the configured manifest/source."""

    try:
        backend = _resolve_backend(
            "sync-data",
            ("sts2sim.data.sync", "sts2sim.data.manifest", "sts2sim.data"),
            ("sync_data", "sync_manifest", "sync_all", "sync"),
        )
        result = _call_backend(
            backend,
            data_dir=data_dir,
            cache_dir=data_dir,
            manifest_path=manifest_path,
            manifest=manifest_path,
            source=source,
            base_url=source or "https://spire-codex.com/api",
            lang=lang,
            force=force,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)
    _emit(result)


@app.command("audit-coverage")
def audit_coverage(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            "-d",
            help="Directory containing normalized game data.",
        ),
    ] = Path("data"),
    manifest_path: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            "-m",
            help="Path to the expected data manifest.",
        ),
    ] = None,
    lang: Annotated[
        str,
        typer.Option(
            "--lang",
            help="Language code for localized source data.",
        ),
    ] = "eng",
    fail_under: Annotated[
        float | None,
        typer.Option(
            "--fail-under",
            min=0.0,
            max=1.0,
            help="Exit non-zero if coverage is below this ratio.",
        ),
    ] = None,
) -> None:
    """Audit how much required game data is available locally."""

    try:
        backend = _resolve_backend(
            "audit-coverage",
            ("sts2sim.data.coverage", "sts2sim.data.audit", "sts2sim.data"),
            ("audit_coverage", "audit_data_coverage", "audit_source_counts", "audit"),
        )
        result = _call_backend(
            backend,
            data_dir=data_dir,
            cache_dir=data_dir,
            manifest_path=manifest_path,
            manifest=manifest_path,
            lang=lang,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)

    payload = _emit(result)
    coverage = _coverage_value(payload)
    if fail_under is not None:
        if coverage is None:
            typer.secho(
                "audit-coverage result did not include a coverage ratio.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(2)
        if coverage < fail_under:
            typer.secho(
                f"coverage {coverage:.3f} is below required {fail_under:.3f}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)


@app.command("audit-events")
def audit_events(
    events_path: Annotated[
        Path | None,
        typer.Option(
            "--events-path",
            "-e",
            help="Path to cached events.json. Defaults to data/cache/eng/events.json.",
        ),
    ] = None,
    category: Annotated[
        str | None,
        typer.Option(
            "--category",
            "-c",
            help=(
                "Filter by category, e.g. primitive, stepwise, ancient-only, "
                "or unsupported."
            ),
        ),
    ] = None,
    summary_only: Annotated[
        bool,
        typer.Option(
            "--summary-only",
            help="Print counts and optional import errors without event entries.",
        ),
    ] = False,
) -> None:
    """Audit cached events against available event logic implementations."""

    try:
        backend = _resolve_backend(
            "audit-events",
            ("sts2sim.content.event_coverage", "sts2sim.content"),
            ("audit_event_coverage",),
        )
        result = _call_backend(backend, events_path=events_path)
    except BackendUnavailable as exc:
        _backend_error(exc)

    _emit(
        _event_audit_payload(
            result,
            category=category,
            summary_only=summary_only,
        )
    )


@app.command("audit-combat")
def audit_combat(
    cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--cache-dir",
            "-d",
            help="Directory containing cached combat source JSON files.",
        ),
    ] = None,
    category: Annotated[
        str | None,
        typer.Option(
            "--category",
            "-c",
            help="Filter by cards, relics, potions, monsters, or encounters.",
        ),
    ] = None,
    status: Annotated[
        str | None,
        typer.Option(
            "--status",
            "-s",
            help="Filter by implemented, blocked, or unknown.",
        ),
    ] = None,
    summary_only: Annotated[
        bool,
        typer.Option(
            "--summary-only",
            help="Print counts and unknown samples without per-id entries.",
        ),
    ] = False,
    unknown_sample_size: Annotated[
        int,
        typer.Option(
            "--unknown-sample-size",
            min=0,
            help="Number of unknown ids to keep per category before filtering.",
        ),
    ] = 5,
) -> None:
    """Audit cached combat content against executable combat mechanics."""

    try:
        backend = _resolve_backend(
            "audit-combat",
            ("sts2sim.content.combat_coverage", "sts2sim.content"),
            ("audit_combat_coverage",),
        )
        result = _call_backend(
            backend,
            cache_dir=_resolve_lang_cache_dir(cache_dir),
            unknown_sample_size=unknown_sample_size,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)

    _emit(
        _combat_audit_payload(
            result,
            category=category,
            status=status,
            summary_only=summary_only,
        )
    )


@app.command("audit-cards")
def audit_cards(
    cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--cache-dir",
            "-d",
            help="Directory containing cached source JSON files.",
        ),
    ] = None,
    cards_path: Annotated[
        Path | None,
        typer.Option(
            "--cards-path",
            help="Path to cached cards.json. Defaults to data/cache/eng/cards.json.",
        ),
    ] = None,
    status: Annotated[
        str | None,
        typer.Option(
            "--status",
            "-s",
            help="Filter by implemented, partial, or missing.",
        ),
    ] = None,
    color: Annotated[
        str | None,
        typer.Option(
            "--color",
            "-c",
            help="Filter by card color/character, e.g. ironclad or colorless.",
        ),
    ] = None,
    card_type: Annotated[
        str | None,
        typer.Option(
            "--type",
            "-t",
            help="Filter by attack, skill, power, status, curse, or quest.",
        ),
    ] = None,
    summary_only: Annotated[
        bool,
        typer.Option(
            "--summary-only",
            help="Print counts and samples without per-card entries.",
        ),
    ] = False,
    sample_size: Annotated[
        int,
        typer.Option(
            "--sample-size",
            min=0,
            help="Number of partial/missing sample ids to keep before filtering.",
        ),
    ] = 10,
) -> None:
    """Audit cached cards as implemented, partial, or missing mechanics."""

    try:
        backend = _resolve_backend(
            "audit-cards",
            ("sts2sim.content.card_coverage", "sts2sim.content"),
            ("audit_card_coverage",),
        )
        result = _call_backend(
            backend,
            cache_dir=_resolve_lang_cache_dir(cache_dir),
            cards_path=cards_path,
            sample_size=sample_size,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)

    _emit(
        _card_audit_payload(
            result,
            status=status,
            color=color,
            card_type=card_type,
            summary_only=summary_only,
        )
    )


@app.command("play-run")
def play_run(
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            "-s",
            help="Deterministic run seed.",
        ),
    ] = 0,
    policy: Annotated[
        str,
        typer.Option(
            "--policy",
            "-p",
            help="Policy identifier used by the run driver.",
        ),
    ] = "random",
    act: Annotated[
        int,
        typer.Option(
            "--act",
            help="Starting act.",
        ),
    ] = 1,
    max_steps: Annotated[
        int | None,
        typer.Option(
            "--max-steps",
            help="Stop after this many simulator decisions.",
        ),
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            "-d",
            help="Directory containing normalized game data.",
        ),
    ] = None,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional replay/result JSON path.",
        ),
    ] = None,
) -> None:
    """Run one deterministic simulator episode."""

    try:
        backend = _resolve_backend(
            "play-run",
            ("sts2sim.api", "sts2sim.api.run", "sts2sim.mechanics.run"),
            ("play_run", "run_episode", "run"),
        )
        result = _call_backend(
            backend,
            seed=seed,
            policy=policy,
            act=act,
            max_steps=max_steps,
            data_dir=data_dir,
            output_path=output_path,
            output=output_path,
            replay_path=output_path,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)

    payload = _emit(result)
    _write_result_if_missing(output_path, payload)


@app.command("play-strategic-run")
def play_strategic_run(
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            "-s",
            help="Deterministic run seed.",
        ),
    ] = 0,
    character_id: Annotated[
        str,
        typer.Option(
            "--character",
            "-c",
            help="Character id for the simulator run.",
        ),
    ] = "IRONCLAD",
    ascension: Annotated[
        int,
        typer.Option(
            "--ascension",
            "-a",
            min=0,
            help="Ascension level.",
        ),
    ] = 0,
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps",
            min=1,
            help="Stop after this many strategic decisions.",
        ),
    ] = 100,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional JSON path for the strategic decision trace.",
        ),
    ] = None,
    history_path: Annotated[
        Path | None,
        typer.Option(
            "--history-output",
            help="Optional JSON path for the readable run history only.",
        ),
    ] = None,
) -> None:
    """Run one simulator episode with the explainable strategic skeleton agent."""

    try:
        backend = _resolve_backend(
            "play-strategic-run",
            ("sts2sim.agents.runner", "sts2sim.agents"),
            ("play_strategic_run",),
        )
        result = _call_backend(
            backend,
            seed=seed,
            character_id=character_id,
            ascension=ascension,
            max_steps=max_steps,
            output_path=output_path,
            history_path=history_path,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)

    payload = _emit(result)
    _write_result_if_missing(output_path, payload)


@app.command("rollout-random")
def rollout_random(
    runs: Annotated[
        int,
        typer.Option("--runs", "-n", min=1, help="Number of random rollouts."),
    ] = 1,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", min=1, help="Maximum steps per rollout."),
    ] = 500,
    start_seed: Annotated[
        int,
        typer.Option("--start-seed", help="First simulator seed."),
    ] = 0,
    character_id: Annotated[
        str,
        typer.Option("--character", "-c", help="Character id for simulator runs."),
    ] = "IRONCLAD",
    ascension: Annotated[
        int,
        typer.Option("--ascension", "-a", min=0, help="Ascension level."),
    ] = 0,
    include_steps: Annotated[
        bool,
        typer.Option(
            "--include-steps/--summary-only",
            help="Include per-step observations/action masks in output.",
        ),
    ] = True,
    include_history: Annotated[
        bool,
        typer.Option(
            "--include-history/--no-history",
            help="Include the readable run history in rollout output.",
        ),
    ] = True,
    observation_mode: Annotated[
        str,
        typer.Option(
            "--observation-mode",
            help="Per-step observation payload: compact or rich.",
        ),
    ] = "rich",
    output_path: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional JSON path for rollout data."),
    ] = None,
) -> None:
    """Collect self-learning rollouts from a masked random policy."""

    normalized_observation_mode = observation_mode.strip().lower().replace("-", "_")
    if normalized_observation_mode not in {"compact", "rich"}:
        raise typer.BadParameter("observation-mode must be 'compact' or 'rich'")
    try:
        backend = _resolve_backend(
            "rollout-random",
            ("sts2sim.learning.rollout", "sts2sim.learning"),
            ("collect_random_rollouts",),
        )
        result = _call_backend(
            backend,
            runs=runs,
            max_steps=max_steps,
            start_seed=start_seed,
            character_id=character_id,
            ascension=ascension,
            include_steps=include_steps,
            include_history=include_history,
            observation_mode=normalized_observation_mode,
            output_path=output_path,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)

    payload = _emit(result)
    _write_result_if_missing(output_path, payload)


@app.command("train-learning-agent")
def train_learning_agent(
    runs: Annotated[
        int,
        typer.Option("--runs", "-n", min=1, help="Number of simulator runs to train on."),
    ] = 10,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", min=1, help="Maximum steps per training run."),
    ] = 500,
    seed: Annotated[
        str,
        typer.Option("--seed", help="Training seed."),
    ] = "0",
    character_id: Annotated[
        str,
        typer.Option("--character", "-c", help="Character id for simulator runs."),
    ] = "IRONCLAD",
    ascension: Annotated[
        int,
        typer.Option("--ascension", "-a", min=0, help="Ascension level."),
    ] = 0,
    epsilon: Annotated[
        float,
        typer.Option("--epsilon", min=0.0, max=1.0, help="Exploration rate."),
    ] = 0.2,
    alpha: Annotated[
        float,
        typer.Option("--alpha", min=0.0, max=1.0, help="Q-learning update rate."),
    ] = 0.1,
    gamma: Annotated[
        float,
        typer.Option("--gamma", min=0.0, max=1.0, help="Future reward discount."),
    ] = 0.99,
    output_path: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Q-learning checkpoint JSON path."),
    ] = Path("checkpoints/q_learning_latest.json"),
    progress_output_path: Annotated[
        Path | None,
        typer.Option(
            "--progress-output",
            help="Optional JSON path for per-run training progress metrics.",
        ),
    ] = None,
    report_output_path: Annotated[
        Path | None,
        typer.Option(
            "--report-output",
            help="Optional standalone HTML learning-progress report path.",
        ),
    ] = None,
    progress_window: Annotated[
        int,
        typer.Option(
            "--progress-window",
            min=1,
            help="Rolling window size for progress charts.",
        ),
    ] = 10,
) -> None:
    """Train the dependency-free self-learning Q baseline."""

    try:
        backend = _resolve_backend(
            "train-learning-agent",
            ("sts2sim.learning.q_learning", "sts2sim.learning"),
            ("train_q_learning",),
        )
        result = _call_backend(
            backend,
            runs=runs,
            max_steps=max_steps,
            seed=seed,
            character_id=character_id,
            ascension=ascension,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            output_path=output_path,
            progress_output_path=progress_output_path,
            report_output_path=report_output_path,
            progress_window=progress_window,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)

    payload = _emit(result)
    _write_result_if_missing(output_path, payload)


@app.command("train-until-boss")
def train_until_boss(
    max_batches: Annotated[
        int,
        typer.Option(
            "--max-batches",
            min=1,
            help="Maximum train/evaluate batches before stopping.",
        ),
    ] = 5,
    batch_runs: Annotated[
        int,
        typer.Option("--batch-runs", min=1, help="Training runs per batch."),
    ] = 50,
    train_max_steps: Annotated[
        int,
        typer.Option("--train-max-steps", min=1, help="Maximum steps per training run."),
    ] = 500,
    eval_runs: Annotated[
        int,
        typer.Option("--eval-runs", min=1, help="Fixed-seed evaluation runs per batch."),
    ] = 5,
    eval_max_steps: Annotated[
        int,
        typer.Option("--eval-max-steps", min=1, help="Maximum steps per evaluation run."),
    ] = 500,
    seed: Annotated[
        str,
        typer.Option("--seed", help="Training seed."),
    ] = "0",
    eval_start_seed: Annotated[
        int,
        typer.Option("--eval-start-seed", help="First fixed evaluation seed."),
    ] = 10_000,
    character_id: Annotated[
        str,
        typer.Option("--character", "-c", help="Character id for simulator runs."),
    ] = "IRONCLAD",
    ascension: Annotated[
        int,
        typer.Option("--ascension", "-a", min=0, help="Ascension level."),
    ] = 0,
    epsilon: Annotated[
        float,
        typer.Option("--epsilon", min=0.0, max=1.0, help="Exploration rate."),
    ] = 0.2,
    alpha: Annotated[
        float,
        typer.Option("--alpha", min=0.0, max=1.0, help="Q-learning update rate."),
    ] = 0.1,
    gamma: Annotated[
        float,
        typer.Option("--gamma", min=0.0, max=1.0, help="Future reward discount."),
    ] = 0.99,
    target_act: Annotated[
        int,
        typer.Option("--target-act", min=1, help="Target act to reach."),
    ] = 1,
    target_floor: Annotated[
        int | None,
        typer.Option(
            "--target-floor",
            min=0,
            help="Target floor; use 0 for the start of an act.",
        ),
    ] = None,
    target_reward: Annotated[
        float,
        typer.Option("--target-reward", help="Extra training reward when target is reached."),
    ] = 100.0,
    success_replay_passes: Annotated[
        int,
        typer.Option(
            "--success-replay-passes",
            min=0,
            help="Replay target-reaching training trajectories this many times.",
        ),
    ] = 0,
    train_seed_mode: Annotated[
        str,
        typer.Option(
            "--train-seed-mode",
            help="Training seed schedule: sequential or random.",
        ),
    ] = "sequential",
    eval_seed_mode: Annotated[
        str,
        typer.Option(
            "--eval-seed-mode",
            help="Evaluation seed schedule: sequential or random holdout.",
        ),
    ] = "sequential",
    target_eval_successes: Annotated[
        int,
        typer.Option(
            "--target-eval-successes",
            min=1,
            help="Evaluation target successes required before stopping.",
        ),
    ] = 1,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help="Opt in to resuming from --resume-from or existing --model-output checkpoint.",
        ),
    ] = False,
    resume_from_path: Annotated[
        Path | None,
        typer.Option("--resume-from", help="Optional checkpoint to resume from."),
    ] = None,
    model_output_path: Annotated[
        Path | None,
        typer.Option("--model-output", help="Q-learning checkpoint JSON path."),
    ] = Path("checkpoints/q_learning_until_boss.json"),
    output_path: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Training target summary JSON path."),
    ] = Path("reports/q_learning_until_boss_latest.json"),
    progress_output_path: Annotated[
        Path | None,
        typer.Option("--progress-output", help="Per-training-run progress JSON path."),
    ] = Path("reports/q_learning_until_boss_progress.json"),
    report_output_path: Annotated[
        Path | None,
        typer.Option("--report-output", help="Standalone HTML progress report path."),
    ] = Path("reports/q_learning_until_boss_latest.html"),
    progress_window: Annotated[
        int,
        typer.Option("--progress-window", min=1, help="Rolling window size."),
    ] = 10,
) -> None:
    """Train Q-learning in batches until evaluation reaches the Act boss."""

    try:
        backend = _resolve_backend(
            "train-until-boss",
            ("sts2sim.learning.train_until", "sts2sim.learning"),
            ("train_q_learning_until_boss",),
        )
        result = _call_backend(
            backend,
            max_batches=max_batches,
            batch_runs=batch_runs,
            train_max_steps=train_max_steps,
            eval_runs=eval_runs,
            eval_max_steps=eval_max_steps,
            seed=seed,
            eval_start_seed=eval_start_seed,
            character_id=character_id,
            ascension=ascension,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            target_act=target_act,
            target_floor=target_floor,
            target_reward=target_reward,
            success_replay_passes=success_replay_passes,
            train_seed_mode=train_seed_mode,
            eval_seed_mode=eval_seed_mode,
            target_eval_successes=target_eval_successes,
            resume=resume,
            resume_from_path=resume_from_path,
            model_output_path=model_output_path,
            output_path=output_path,
            progress_output_path=progress_output_path,
            report_output_path=report_output_path,
            progress_window=progress_window,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)

    payload = _emit(result)
    _write_result_if_missing(output_path, payload)


@app.command("train-masked-ppo")
def train_masked_ppo(
    target: Annotated[
        str,
        typer.Option(
            "--target",
            help="Curriculum target: act1-boss, act2-boss, act3-boss, or game-clear.",
        ),
    ] = "act1-boss",
    max_batches: Annotated[
        int,
        typer.Option(
            "--max-batches",
            min=1,
            help="Maximum train/evaluate batches before stopping unless --until-stopped is set.",
        ),
    ] = 20,
    until_stopped: Annotated[
        bool,
        typer.Option(
            "--until-stopped/--stop-on-target",
            help=(
                "Keep training indefinitely until the process is stopped. "
                "Checkpoint, reports, and latest-batch best/worst histories are "
                "saved after every completed batch."
            ),
        ),
    ] = False,
    train_runs_per_batch: Annotated[
        int,
        typer.Option(
            "--train-runs-per-batch",
            min=1,
            help="Random-seed simulator runs collected before each PPO update.",
        ),
    ] = 64,
    train_max_steps: Annotated[
        int,
        typer.Option("--train-max-steps", min=1, help="Maximum steps per training run."),
    ] = 1200,
    eval_runs: Annotated[
        int,
        typer.Option("--eval-runs", min=1, help="Random holdout evaluation runs per batch."),
    ] = 32,
    eval_max_steps: Annotated[
        int,
        typer.Option("--eval-max-steps", min=1, help="Maximum steps per evaluation run."),
    ] = 1200,
    seed: Annotated[
        str,
        typer.Option("--seed", help="Top-level trainer seed used to sample run seeds."),
    ] = "ppo",
    character_id: Annotated[
        str,
        typer.Option("--character", "-c", help="Character id for simulator runs."),
    ] = "IRONCLAD",
    ascension: Annotated[
        int,
        typer.Option("--ascension", "-a", min=0, help="Ascension level."),
    ] = 0,
    hidden_size: Annotated[
        int,
        typer.Option("--hidden-size", min=16, help="Neural policy hidden layer size."),
    ] = 256,
    hidden_layers: Annotated[
        int,
        typer.Option(
            "--hidden-layers",
            min=1,
            help="Hidden layers in both state and legal-action encoders.",
        ),
    ] = 3,
    head_hidden_layers: Annotated[
        int,
        typer.Option(
            "--head-hidden-layers",
            min=1,
            help="Hidden layers in the policy and value heads.",
        ),
    ] = 2,
    activation: Annotated[
        str,
        typer.Option(
            "--activation",
            help="MLP activation: silu, gelu, relu, elu, or tanh.",
        ),
    ] = "silu",
    learning_rate: Annotated[
        float,
        typer.Option("--learning-rate", min=0.0, help="Adam optimizer learning rate."),
    ] = 0.0003,
    gamma: Annotated[
        float,
        typer.Option("--gamma", min=0.0, max=1.0, help="Future reward discount."),
    ] = 0.99,
    gae_lambda: Annotated[
        float,
        typer.Option("--gae-lambda", min=0.0, max=1.0, help="GAE lambda."),
    ] = 0.95,
    clip_ratio: Annotated[
        float,
        typer.Option("--clip-ratio", min=0.0, help="PPO clipped objective ratio."),
    ] = 0.2,
    value_coef: Annotated[
        float,
        typer.Option("--value-coef", min=0.0, help="Value loss coefficient."),
    ] = 0.5,
    entropy_coef: Annotated[
        float,
        typer.Option("--entropy-coef", min=0.0, help="Entropy bonus coefficient."),
    ] = 0.01,
    planning_coef: Annotated[
        float,
        typer.Option(
            "--planning-coef",
            min=0.0,
            help="Auxiliary planning-head loss coefficient.",
        ),
    ] = 0.1,
    teacher_mix: Annotated[
        float,
        typer.Option(
            "--teacher-mix",
            min=0.0,
            max=1.0,
            help="Fraction of training actions to take from the strategic teacher.",
        ),
    ] = 0.0,
    imitation_coef: Annotated[
        float,
        typer.Option(
            "--imitation-coef",
            min=0.0,
            help="Supervised loss weight toward the strategic teacher action.",
        ),
    ] = 0.0,
    ppo_epochs: Annotated[
        int,
        typer.Option("--ppo-epochs", min=1, help="PPO optimization epochs per batch."),
    ] = 4,
    minibatch_size: Annotated[
        int,
        typer.Option("--minibatch-size", min=1, help="Transitions sampled per PPO update chunk."),
    ] = 256,
    target_reward: Annotated[
        float,
        typer.Option("--target-reward", help="Extra training reward when target is reached."),
    ] = 100.0,
    target_eval_successes: Annotated[
        int,
        typer.Option(
            "--target-eval-successes",
            min=1,
            help="Holdout target successes required before stopping.",
        ),
    ] = 1,
    target_consecutive_successes: Annotated[
        int,
        typer.Option(
            "--target-consecutive-successes",
            min=1,
            help="Consecutive holdout target successes required before stopping.",
        ),
    ] = 1,
    target_success_rate: Annotated[
        float,
        typer.Option(
            "--target-success-rate",
            min=0.0,
            max=1.0,
            help="Minimum holdout target success rate required before stopping.",
        ),
    ] = 0.0,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help=(
                "Resume PPO weights from --resume-from or existing --model-output "
                "checkpoint. Use --no-resume for a fresh model."
            ),
        ),
    ] = True,
    resume_from_path: Annotated[
        Path | None,
        typer.Option("--resume-from", help="Optional PPO checkpoint to resume from."),
    ] = None,
    model_output_path: Annotated[
        Path | None,
        typer.Option("--model-output", help="PPO checkpoint path."),
    ] = Path("checkpoints/masked_ppo_latest.pt"),
    output_path: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Training target summary JSON path."),
    ] = Path("reports/masked_ppo_latest.json"),
    progress_output_path: Annotated[
        Path | None,
        typer.Option("--progress-output", help="Per-training-run progress JSON path."),
    ] = Path("reports/masked_ppo_progress.json"),
    report_output_path: Annotated[
        Path | None,
        typer.Option("--report-output", help="Standalone HTML progress report path."),
    ] = Path("reports/masked_ppo_latest.html"),
    progress_window: Annotated[
        int,
        typer.Option("--progress-window", min=1, help="Rolling window size."),
    ] = 20,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Torch device for PPO tensors: auto prefers CUDA/GPU when available.",
        ),
    ] = "auto",
    rollout_workers: Annotated[
        int,
        typer.Option(
            "--rollout-workers",
            min=0,
            help=(
                "Parallel simulator rollout workers for train/eval collection. "
                "Use 1 for old sequential collection or 0 to auto-use available CPU cores."
            ),
        ),
    ] = 1,
    rollout_inference: Annotated[
        str,
        typer.Option(
            "--rollout-inference",
            help=(
                "Rollout policy inference mode: worker keeps model copies in each "
                "worker, batched-gpu centralizes action selection on the trainer device."
            ),
        ),
    ] = "worker",
    history_mode: Annotated[
        str,
        typer.Option(
            "--history-mode",
            help="History capture mode: off, highlights, or all-eval.",
        ),
    ] = "highlights",
    envs_per_worker: Annotated[
        int,
        typer.Option(
            "--envs-per-worker",
            min=1,
            help="Active environment streams per rollout worker for batched-gpu inference.",
        ),
    ] = 1,
    policy_server_min_batch: Annotated[
        int,
        typer.Option(
            "--policy-server-min-batch",
            min=1,
            help="Minimum decision requests to batch before GPU policy inference.",
        ),
    ] = 1,
    policy_server_max_wait_ms: Annotated[
        int,
        typer.Option(
            "--policy-server-max-wait-ms",
            min=0,
            help="Maximum milliseconds to wait for a larger GPU policy batch.",
        ),
    ] = 20,
    terminal_progress: Annotated[
        bool,
        typer.Option(
            "--terminal-progress/--no-terminal-progress",
            help="Show live terminal progress bars for train/eval runs.",
        ),
    ] = True,
) -> None:
    """Train the random-seed masked PPO agent toward a run target."""

    try:
        backend = _resolve_backend(
            "train-masked-ppo",
            ("sts2sim.learning.masked_ppo", "sts2sim.learning"),
            ("train_masked_ppo",),
        )
        terminal_reporter = _TrainingTerminalProgress(terminal_progress)
        with terminal_reporter:
            result = _call_backend(
                backend,
                target=target,
                max_batches=max_batches,
                until_stopped=until_stopped,
                train_runs_per_batch=train_runs_per_batch,
                train_max_steps=train_max_steps,
                eval_runs=eval_runs,
                eval_max_steps=eval_max_steps,
                seed=seed,
                character_id=character_id,
                ascension=ascension,
                hidden_size=hidden_size,
                hidden_layers=hidden_layers,
                head_hidden_layers=head_hidden_layers,
                activation=activation,
                learning_rate=learning_rate,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_ratio=clip_ratio,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                planning_coef=planning_coef,
                teacher_mix=teacher_mix,
                imitation_coef=imitation_coef,
                ppo_epochs=ppo_epochs,
                minibatch_size=minibatch_size,
                target_reward=target_reward,
                target_eval_successes=target_eval_successes,
                target_consecutive_successes=target_consecutive_successes,
                target_success_rate=target_success_rate,
                resume=resume,
                resume_from_path=resume_from_path,
                model_output_path=model_output_path,
                output_path=output_path,
                progress_output_path=progress_output_path,
                report_output_path=report_output_path,
                progress_window=progress_window,
                device=device,
                rollout_workers=rollout_workers,
                rollout_inference=rollout_inference,
                history_mode=history_mode,
                envs_per_worker=envs_per_worker,
                policy_server_min_batch=policy_server_min_batch,
                policy_server_max_wait_ms=policy_server_max_wait_ms,
                progress_reporter=terminal_reporter,
            )
    except BackendUnavailable as exc:
        _backend_error(exc)

    _emit_training_result(result, output_path)


@app.command("train-ppo-curriculum")
def train_ppo_curriculum(
    stages: Annotated[
        str,
        typer.Option(
            "--stages",
            help=(
                "Comma-separated stage targets. Default: "
                "act1-boss,act2-boss,act3-boss,game-clear."
            ),
        ),
    ] = "act1-boss,act2-boss,act3-boss,game-clear",
    run_name: Annotated[
        str,
        typer.Option("--run-name", help="Prefix for stage checkpoints and reports."),
    ] = "ppo_curriculum",
    max_batches: Annotated[
        int | None,
        typer.Option(
            "--max-batches",
            min=1,
            help="Override max train/evaluate batches for every stage.",
        ),
    ] = None,
    train_runs_per_batch: Annotated[
        int,
        typer.Option(
            "--train-runs-per-batch",
            min=1,
            help="Random-seed simulator runs collected before each PPO update.",
        ),
    ] = 128,
    eval_runs: Annotated[
        int,
        typer.Option("--eval-runs", min=1, help="Random holdout evaluation runs per batch."),
    ] = 32,
    train_max_steps: Annotated[
        int | None,
        typer.Option(
            "--train-max-steps",
            min=1,
            help="Override maximum steps per training run for every stage.",
        ),
    ] = None,
    eval_max_steps: Annotated[
        int | None,
        typer.Option(
            "--eval-max-steps",
            min=1,
            help="Override maximum steps per evaluation run for every stage.",
        ),
    ] = None,
    seed: Annotated[
        str,
        typer.Option("--seed", help="Top-level curriculum seed."),
    ] = "ppo-curriculum",
    character_id: Annotated[
        str,
        typer.Option("--character", "-c", help="Character id for simulator runs."),
    ] = "IRONCLAD",
    ascension: Annotated[
        int,
        typer.Option("--ascension", "-a", min=0, help="Ascension level."),
    ] = 0,
    hidden_size: Annotated[
        int,
        typer.Option("--hidden-size", min=16, help="Neural policy hidden layer size."),
    ] = 256,
    hidden_layers: Annotated[
        int,
        typer.Option(
            "--hidden-layers",
            min=1,
            help="Hidden layers in both state and legal-action encoders.",
        ),
    ] = 3,
    head_hidden_layers: Annotated[
        int,
        typer.Option(
            "--head-hidden-layers",
            min=1,
            help="Hidden layers in the policy and value heads.",
        ),
    ] = 2,
    activation: Annotated[
        str,
        typer.Option(
            "--activation",
            help="MLP activation: silu, gelu, relu, elu, or tanh.",
        ),
    ] = "silu",
    learning_rate: Annotated[
        float,
        typer.Option("--learning-rate", min=0.0, help="Adam optimizer learning rate."),
    ] = 0.0003,
    gamma: Annotated[
        float,
        typer.Option("--gamma", min=0.0, max=1.0, help="Future reward discount."),
    ] = 0.99,
    gae_lambda: Annotated[
        float,
        typer.Option("--gae-lambda", min=0.0, max=1.0, help="GAE lambda."),
    ] = 0.95,
    clip_ratio: Annotated[
        float,
        typer.Option("--clip-ratio", min=0.0, help="PPO clipped objective ratio."),
    ] = 0.2,
    value_coef: Annotated[
        float,
        typer.Option("--value-coef", min=0.0, help="Value loss coefficient."),
    ] = 0.5,
    entropy_coef: Annotated[
        float,
        typer.Option("--entropy-coef", min=0.0, help="Entropy bonus coefficient."),
    ] = 0.01,
    planning_coef: Annotated[
        float,
        typer.Option(
            "--planning-coef",
            min=0.0,
            help="Auxiliary planning-head loss coefficient.",
        ),
    ] = 0.1,
    teacher_mix: Annotated[
        float,
        typer.Option(
            "--teacher-mix",
            min=0.0,
            max=1.0,
            help="Fraction of training actions to take from the strategic teacher.",
        ),
    ] = 0.0,
    imitation_coef: Annotated[
        float,
        typer.Option(
            "--imitation-coef",
            min=0.0,
            help="Supervised loss weight toward the strategic teacher action.",
        ),
    ] = 0.0,
    ppo_epochs: Annotated[
        int,
        typer.Option("--ppo-epochs", min=1, help="PPO optimization epochs per batch."),
    ] = 4,
    minibatch_size: Annotated[
        int,
        typer.Option("--minibatch-size", min=1, help="Transitions per PPO update chunk."),
    ] = 256,
    target_reward: Annotated[
        float,
        typer.Option("--target-reward", help="Extra training reward when target is reached."),
    ] = 100.0,
    target_eval_successes: Annotated[
        int | None,
        typer.Option(
            "--target-eval-successes",
            min=1,
            help="Override holdout target successes required before stage advancement.",
        ),
    ] = None,
    target_consecutive_successes: Annotated[
        int | None,
        typer.Option(
            "--target-consecutive-successes",
            min=1,
            help="Override consecutive holdout successes required before stage advancement.",
        ),
    ] = None,
    target_success_rate: Annotated[
        float,
        typer.Option(
            "--target-success-rate",
            min=0.0,
            max=1.0,
            help="Minimum holdout target success rate required before stage advancement.",
        ),
    ] = 0.0,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help=(
                "Resume PPO weights for each stage from existing checkpoint files "
                "when available. Use --no-resume for a fresh curriculum start."
            ),
        ),
    ] = True,
    resume_from_path: Annotated[
        Path | None,
        typer.Option("--resume-from", help="Optional initial PPO checkpoint."),
    ] = None,
    checkpoint_dir: Annotated[
        Path,
        typer.Option("--checkpoint-dir", help="Directory for stage PPO checkpoints."),
    ] = Path("checkpoints"),
    report_dir: Annotated[
        Path,
        typer.Option("--report-dir", help="Directory for stage JSON/HTML reports."),
    ] = Path("reports"),
    output_path: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Curriculum summary JSON path."),
    ] = Path("reports/ppo_curriculum_latest.json"),
    report_output_path: Annotated[
        Path | None,
        typer.Option("--report-output", help="Combined curriculum HTML dashboard path."),
    ] = Path("reports/ppo_curriculum_latest.html"),
    progress_window: Annotated[
        int,
        typer.Option("--progress-window", min=1, help="Rolling window size."),
    ] = 20,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Torch device for PPO tensors: auto prefers CUDA/GPU when available.",
        ),
    ] = "auto",
    rollout_workers: Annotated[
        int,
        typer.Option(
            "--rollout-workers",
            min=0,
            help=(
                "Parallel simulator rollout workers for each stage. Use 1 for old "
                "sequential collection or 0 to auto-use available CPU cores."
            ),
        ),
    ] = 1,
    rollout_inference: Annotated[
        str,
        typer.Option(
            "--rollout-inference",
            help=(
                "Rollout policy inference mode: worker keeps model copies in each "
                "worker, batched-gpu centralizes action selection on the trainer device."
            ),
        ),
    ] = "worker",
    history_mode: Annotated[
        str,
        typer.Option(
            "--history-mode",
            help="History capture mode for each PPO stage: off, highlights, or all-eval.",
        ),
    ] = "highlights",
    envs_per_worker: Annotated[
        int,
        typer.Option(
            "--envs-per-worker",
            min=1,
            help="Active environment streams per rollout worker for batched-gpu inference.",
        ),
    ] = 1,
    policy_server_min_batch: Annotated[
        int,
        typer.Option(
            "--policy-server-min-batch",
            min=1,
            help="Minimum decision requests to batch before GPU policy inference.",
        ),
    ] = 1,
    policy_server_max_wait_ms: Annotated[
        int,
        typer.Option(
            "--policy-server-max-wait-ms",
            min=0,
            help="Maximum milliseconds to wait for a larger GPU policy batch.",
        ),
    ] = 20,
    terminal_progress: Annotated[
        bool,
        typer.Option(
            "--terminal-progress/--no-terminal-progress",
            help="Show live terminal progress bars for every curriculum stage.",
        ),
    ] = True,
) -> None:
    """Train PPO through staged targets, advancing only when comfortable."""

    try:
        backend = _resolve_backend(
            "train-ppo-curriculum",
            ("sts2sim.learning.curriculum", "sts2sim.learning"),
            ("train_masked_ppo_curriculum",),
        )
        terminal_reporter = _TrainingTerminalProgress(terminal_progress)
        with terminal_reporter:
            result = _call_backend(
                backend,
                stages=stages,
                run_name=run_name,
                max_batches=max_batches,
                train_runs_per_batch=train_runs_per_batch,
                eval_runs=eval_runs,
                train_max_steps=train_max_steps,
                eval_max_steps=eval_max_steps,
                seed=seed,
                character_id=character_id,
                ascension=ascension,
                hidden_size=hidden_size,
                hidden_layers=hidden_layers,
                head_hidden_layers=head_hidden_layers,
                activation=activation,
                learning_rate=learning_rate,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_ratio=clip_ratio,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                planning_coef=planning_coef,
                teacher_mix=teacher_mix,
                imitation_coef=imitation_coef,
                ppo_epochs=ppo_epochs,
                minibatch_size=minibatch_size,
                target_reward=target_reward,
                target_eval_successes=target_eval_successes,
                target_consecutive_successes=target_consecutive_successes,
                target_success_rate=target_success_rate,
                resume=resume,
                resume_from_path=resume_from_path,
                checkpoint_dir=checkpoint_dir,
                report_dir=report_dir,
                output_path=output_path,
                report_output_path=report_output_path,
                progress_window=progress_window,
                device=device,
                rollout_workers=rollout_workers,
                rollout_inference=rollout_inference,
                history_mode=history_mode,
                envs_per_worker=envs_per_worker,
                policy_server_min_batch=policy_server_min_batch,
                policy_server_max_wait_ms=policy_server_max_wait_ms,
                progress_reporter=terminal_reporter,
            )
    except BackendUnavailable as exc:
        _backend_error(exc)

    _emit_training_result(result, output_path)


@app.command("learning-progress-report")
def learning_progress_report(
    input_path: Annotated[
        Path,
        typer.Argument(help="Training, rollout, or evaluation JSON file."),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Standalone HTML report path.",
        ),
    ] = Path("reports/learning_progress.html"),
    title: Annotated[
        str,
        typer.Option("--title", help="Report title."),
    ] = "Learning Progress",
    window: Annotated[
        int,
        typer.Option("--window", min=1, help="Rolling window size for trend lines."),
    ] = 10,
) -> None:
    """Create a static HTML dashboard from learning metrics."""

    try:
        backend = _resolve_backend(
            "learning-progress-report",
            ("sts2sim.learning.progress", "sts2sim.learning"),
            ("build_learning_progress_report",),
        )
        result = _call_backend(
            backend,
            input_path=input_path,
            output_path=output_path,
            title=title,
            window=window,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)
    _emit(result)


@app.command("evaluate-agents")
def evaluate_agents(
    runs: Annotated[
        int,
        typer.Option("--runs", "-n", min=1, help="Number of fixed seeds to evaluate."),
    ] = 10,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", min=1, help="Maximum simulator steps per run."),
    ] = 500,
    start_seed: Annotated[
        int,
        typer.Option("--start-seed", help="First simulator seed."),
    ] = 0,
    character_id: Annotated[
        str,
        typer.Option("--character", "-c", help="Character id for simulator runs."),
    ] = "IRONCLAD",
    ascension: Annotated[
        int,
        typer.Option("--ascension", "-a", min=0, help="Ascension level."),
    ] = 0,
    policies: Annotated[
        list[str] | None,
        typer.Option(
            "--policy",
            "-p",
            help="Policy to include: random, q_learning, or strategic. Repeatable.",
        ),
    ] = None,
    model_path: Annotated[
        Path | None,
        typer.Option("--model", help="Optional Q-learning checkpoint path."),
    ] = None,
    output_path: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Baseline comparison JSON path."),
    ] = Path("reports/agent_baselines.json"),
    report_output_path: Annotated[
        Path | None,
        typer.Option("--report-output", help="Standalone HTML comparison report path."),
    ] = Path("reports/agent_baselines.html"),
    progress_window: Annotated[
        int,
        typer.Option("--progress-window", min=1, help="Rolling window size."),
    ] = 10,
) -> None:
    """Evaluate random, Q-learning, and strategic baselines side by side."""

    try:
        backend = _resolve_backend(
            "evaluate-agents",
            ("sts2sim.learning.agent_eval", "sts2sim.learning"),
            ("evaluate_agent_baselines",),
        )
        result = _call_backend(
            backend,
            runs=runs,
            max_steps=max_steps,
            start_seed=start_seed,
            character_id=character_id,
            ascension=ascension,
            policies=tuple(policies) if policies else None,
            model_path=model_path,
            output_path=output_path,
            report_output_path=report_output_path,
            progress_window=progress_window,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)

    payload = _emit(result)
    _write_result_if_missing(output_path, payload)


@app.command("evaluate-learning-agent")
def evaluate_learning_agent(
    policy: Annotated[
        str,
        typer.Option("--policy", help="Policy to evaluate: random or q_learning."),
    ] = "random",
    model_path: Annotated[
        Path | None,
        typer.Option("--model", help="Q-learning checkpoint path."),
    ] = None,
    runs: Annotated[
        int,
        typer.Option("--runs", "-n", min=1, help="Number of evaluation runs."),
    ] = 10,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", min=1, help="Maximum steps per run."),
    ] = 500,
    start_seed: Annotated[
        int,
        typer.Option("--start-seed", help="First evaluation seed."),
    ] = 0,
    character_id: Annotated[
        str,
        typer.Option("--character", "-c", help="Character id for simulator runs."),
    ] = "IRONCLAD",
    ascension: Annotated[
        int,
        typer.Option("--ascension", "-a", min=0, help="Ascension level."),
    ] = 0,
) -> None:
    """Evaluate a self-learning policy over fixed seeds."""

    normalized_policy = policy.strip().lower().replace("-", "_")
    if normalized_policy not in {"random", "q_learning"}:
        raise typer.BadParameter("policy must be 'random' or 'q_learning'")
    try:
        backend = _resolve_backend(
            "evaluate-learning-agent",
            ("sts2sim.learning.evaluate", "sts2sim.learning"),
            ("evaluate_learning_agent",),
        )
        result = _call_backend(
            backend,
            policy=normalized_policy,
            model_path=model_path,
            runs=runs,
            max_steps=max_steps,
            start_seed=start_seed,
            character_id=character_id,
            ascension=ascension,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)
    _emit(result)


@app.command("replay")
def replay(
    replay_path: Annotated[
        Path,
        typer.Argument(help="Replay JSON emitted by play-run."),
    ],
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            "-d",
            help="Directory containing normalized game data.",
        ),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict/--no-strict",
            help="Require the replay to match every recorded decision.",
        ),
    ] = True,
) -> None:
    """Replay a recorded simulator episode."""

    try:
        backend = _resolve_backend(
            "replay",
            ("sts2sim.api", "sts2sim.api.replay", "sts2sim.replay"),
            ("replay", "replay_run", "verify_replay"),
        )
        result = _call_replay_backend(backend, replay_path, data_dir, strict)
    except BackendUnavailable as exc:
        _backend_error(exc)
    _emit(result)


@app.command("compare-trace")
def compare_trace(
    trace_path: Annotated[
        Path,
        typer.Argument(help="Parity trace JSON to compare against the simulator."),
    ],
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Comparison mode: subset compares only captured fields; exact also flags extras.",
        ),
    ] = "subset",
    ignore_paths: Annotated[
        list[str] | None,
        typer.Option(
            "--ignore-path",
            "-i",
            help="Snapshot path to ignore; use path.* to ignore a subtree.",
        ),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict/--no-strict",
            help="Exit non-zero after printing the report if mismatches are found.",
        ),
    ] = False,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional JSON path for the comparison report.",
        ),
    ] = None,
) -> None:
    """Compare a golden parity trace against the simulator."""

    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"subset", "exact"}:
        raise typer.BadParameter("mode must be 'subset' or 'exact'")

    try:
        backend = _resolve_backend(
            "compare-trace",
            ("sts2sim.parity",),
            ("compare_trace_file",),
        )
        result = _call_backend(
            backend,
            trace_path=trace_path,
            mode=normalized_mode,
            ignore_paths=tuple(ignore_paths or ()),
            strict=False,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)

    payload = _emit(result)
    _write_result_if_missing(output_path, payload)
    if strict and isinstance(payload, Mapping) and not payload.get("matched", False):
        raise typer.Exit(1)


@app.command("trace-template")
def trace_template(
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional JSON path for the template trace.",
        ),
    ] = None,
) -> None:
    """Print a starter parity trace JSON payload."""

    try:
        backend = _resolve_backend(
            "trace-template",
            ("sts2sim.parity",),
            ("trace_template",),
        )
        result = _call_backend(backend)
    except BackendUnavailable as exc:
        _backend_error(exc)

    payload = _emit(result)
    _write_result_if_missing(output_path, payload)


@app.command("import-run")
def import_run(
    run_path: Annotated[
        Path,
        typer.Argument(help="Local .run or run-history JSON file."),
    ],
    trace_output_path: Annotated[
        Path | None,
        typer.Option(
            "--trace-output",
            help="Optional path to write a non-replayable parity trace scaffold.",
        ),
    ] = None,
) -> None:
    """Import a finished run-history file into a summary and parity scaffold."""

    try:
        backend = _resolve_backend(
            "import-run",
            ("sts2sim.run_files",),
            ("import_run_file",),
        )
        result = _call_backend(
            backend,
            run_path=run_path,
            trace_output_path=trace_output_path,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)
    _emit(result)


@app.command("find-run-files")
def find_run_files(
    root: Annotated[
        Path,
        typer.Argument(help="Directory to search recursively for .run or run JSON files."),
    ],
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            "-n",
            min=1,
            help="Maximum number of newest files to return.",
        ),
    ] = 20,
) -> None:
    """Find local run-history files under a directory."""

    try:
        backend = _resolve_backend(
            "find-run-files",
            ("sts2sim.run_files",),
            ("find_run_files",),
        )
        result = _call_backend(backend, root=root, limit=limit)
    except BackendUnavailable as exc:
        _backend_error(exc)
    _emit({"root": str(root), "files": [str(path) for path in result]})


@app.command("capture-live")
def capture_live(
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="Base URL for a local STS2 live-state mod API, or 'auto'.",
        ),
    ] = "auto",
    state_path: Annotated[
        str | None,
        typer.Option(
            "--state-path",
            help="Override the live state endpoint path.",
        ),
    ] = None,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional path to write the captured parity trace.",
        ),
    ] = None,
) -> None:
    """Capture the current live game state without taking an action."""

    try:
        backend = _resolve_backend(
            "capture-live",
            ("sts2sim.live_capture",),
            ("capture_live_state",),
        )
        result = _call_backend(
            backend,
            base_url=base_url,
            state_path=state_path,
            output_path=output_path,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)
    _emit(result)


@app.command("live-play")
def live_play(
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="Base URL for a local STS2 live-state/action mod API, or 'auto'.",
        ),
    ] = "auto",
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps",
            min=1,
            help="Maximum live actions to execute.",
        ),
    ] = 5,
    policy: Annotated[
        str,
        typer.Option(
            "--policy",
            help="Action policy: first, random, or prefer_attack.",
        ),
    ] = "first",
    seed: Annotated[
        str,
        typer.Option(
            "--seed",
            help="Seed for the random action policy.",
        ),
    ] = "0",
    state_path: Annotated[
        str | None,
        typer.Option("--state-path", help="Override the live state endpoint path."),
    ] = None,
    actions_path: Annotated[
        str | None,
        typer.Option("--actions-path", help="Override the legal-actions endpoint path."),
    ] = None,
    action_path: Annotated[
        str | None,
        typer.Option("--action-path", help="Override the execute-action endpoint path."),
    ] = None,
    action_envelope: Annotated[
        str,
        typer.Option(
            "--action-envelope",
            help="How to send actions: action, payload, or raw.",
        ),
    ] = "action",
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional path to write the live parity trace.",
        ),
    ] = None,
) -> None:
    """Let a tiny policy take live-game actions and record a parity trace."""

    normalized_policy = policy.strip().lower()
    if normalized_policy not in {"first", "random", "prefer_attack"}:
        raise typer.BadParameter("policy must be 'first', 'random', or 'prefer_attack'")
    normalized_envelope = action_envelope.strip().lower()
    if normalized_envelope not in {"action", "payload", "raw"}:
        raise typer.BadParameter("action-envelope must be 'action', 'payload', or 'raw'")

    try:
        backend = _resolve_backend(
            "live-play",
            ("sts2sim.live_capture",),
            ("live_play",),
        )
        result = _call_backend(
            backend,
            base_url=base_url,
            max_steps=max_steps,
            policy=normalized_policy,
            seed=seed,
            state_path=state_path,
            actions_path=actions_path,
            action_path=action_path,
            action_envelope=normalized_envelope,
            output_path=output_path,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)
    _emit(result)


@app.command("play-live-agent")
def play_live_agent(
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="Base URL for the local STS2MCP live bridge, or 'auto'.",
        ),
    ] = "auto",
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps",
            min=1,
            help="Maximum real-game actions to execute.",
        ),
    ] = 10,
    seed: Annotated[
        str,
        typer.Option(
            "--seed",
            help="Seed for random live-start choices.",
        ),
    ] = "0",
    simulator_seed: Annotated[
        str,
        typer.Option(
            "--simulator-seed",
            help="Simulator seed used for baseline comparison snapshots.",
        ),
    ] = "0",
    start_if_needed: Annotated[
        bool,
        typer.Option(
            "--start-if-needed/--no-start-if-needed",
            help="Start a new standard run if the bridge is still on the menu.",
        ),
    ] = True,
    character: Annotated[
        str,
        typer.Option(
            "--character",
            help="Character to start if needed, or 'random'.",
        ),
    ] = "random",
    ascension: Annotated[
        str,
        typer.Option(
            "--ascension",
            help="Ascension to start if needed: random, max, current, or a number.",
        ),
    ] = "random",
    state_path: Annotated[
        str | None,
        typer.Option("--state-path", help="Override the live state endpoint path."),
    ] = None,
    action_path: Annotated[
        str | None,
        typer.Option("--action-path", help="Override the live action endpoint path."),
    ] = None,
    delay_seconds: Annotated[
        float,
        typer.Option(
            "--delay",
            min=0.0,
            help="Delay after each live action before the next state query.",
        ),
    ] = 0.35,
    settle_timeout_seconds: Annotated[
        float,
        typer.Option(
            "--settle-timeout",
            min=0.0,
            help="Seconds to wait through transient/loading live states.",
        ),
    ] = 3.0,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional path for the live-agent comparison report.",
        ),
    ] = Path("live_traces/live_agent_latest.json"),
) -> None:
    """Let a conservative live agent play and compare snapshots to the simulator."""

    try:
        backend = _resolve_backend(
            "play-live-agent",
            ("sts2sim.live_agent",),
            ("play_live_agent",),
        )
        result = _call_backend(
            backend,
            base_url=base_url,
            max_steps=max_steps,
            seed=seed,
            simulator_seed=simulator_seed,
            start_if_needed=start_if_needed,
            character=character,
            ascension=ascension,
            state_path=state_path,
            action_path=action_path,
            delay_seconds=delay_seconds,
            settle_timeout_seconds=settle_timeout_seconds,
            output_path=output_path,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)
    _emit(result)


@app.command("probe-live")
def probe_live(
    base_urls: Annotated[
        list[str] | None,
        typer.Option(
            "--base-url",
            help="Bridge base URL to probe; may be repeated. Defaults to known ports.",
        ),
    ] = None,
    timeout_seconds: Annotated[
        float,
        typer.Option(
            "--timeout",
            min=0.1,
            help="Per-endpoint HTTP timeout in seconds.",
        ),
    ] = 1.0,
) -> None:
    """Probe known local live-state bridges and report which are reachable."""

    try:
        backend = _resolve_backend(
            "probe-live",
            ("sts2sim.live_capture",),
            ("probe_live_bridges",),
        )
        result = _call_backend(
            backend,
            base_urls=tuple(base_urls) if base_urls else None,
            timeout_seconds=timeout_seconds,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)
    _emit({"bridges": result})


@app.command("start-live-run")
def start_live_run(
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="Base URL for a local STS2 live-state/action mod API, or 'auto'.",
        ),
    ] = "auto",
    character: Annotated[
        str,
        typer.Option(
            "--character",
            help="Character id/name to start, or 'random' for an unlocked random character.",
        ),
    ] = "random",
    ascension: Annotated[
        str,
        typer.Option(
            "--ascension",
            help="'random', 'max', 'current', or an explicit unlocked ascension number.",
        ),
    ] = "random",
    seed: Annotated[
        str,
        typer.Option(
            "--seed",
            help="Seed for random character/ascension selection.",
        ),
    ] = "0",
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps",
            min=1,
            help="Maximum menu actions to try before stopping.",
        ),
    ] = 80,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Optional path to write the start result JSON.",
        ),
    ] = None,
) -> None:
    """Start a standard live singleplayer run with unlocked random choices."""

    try:
        backend = _resolve_backend(
            "start-live-run",
            ("sts2sim.live_start",),
            ("start_live_run",),
        )
        result = _call_backend(
            backend,
            base_url=base_url,
            character=character,
            ascension=ascension,
            seed=seed,
            max_steps=max_steps,
            output_path=output_path,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)
    _emit(result)


@app.command("fuzz-run")
def fuzz_run(
    count: Annotated[
        int,
        typer.Option(
            "--count",
            "-n",
            min=1,
            help="Number of seeds to execute.",
        ),
    ] = 100,
    start_seed: Annotated[
        int,
        typer.Option(
            "--start-seed",
            help="First seed when explicit --seed values are not provided.",
        ),
    ] = 0,
    seeds: Annotated[
        list[int] | None,
        typer.Option(
            "--seed",
            "-s",
            help="Specific seed to execute; may be repeated.",
        ),
    ] = None,
    max_steps: Annotated[
        int | None,
        typer.Option(
            "--max-steps",
            help="Stop each run after this many simulator decisions.",
        ),
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            "-d",
            help="Directory containing normalized game data.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory for per-seed failure artifacts.",
        ),
    ] = None,
) -> None:
    """Run a deterministic fuzz sweep across seeds."""

    try:
        backend = _resolve_backend(
            "fuzz-run",
            ("sts2sim.api", "sts2sim.api.fuzz", "sts2sim.mechanics.fuzz"),
            ("fuzz_run", "run_fuzz", "fuzz"),
        )
        result = _call_backend(
            backend,
            count=count,
            start_seed=start_seed,
            seeds=seeds,
            max_steps=max_steps,
            data_dir=data_dir,
            output_dir=output_dir,
        )
    except BackendUnavailable as exc:
        _backend_error(exc)
    _emit(result)


def main() -> None:
    """Run the Typer application."""

    app()


if __name__ == "__main__":  # pragma: no cover
    main()
