"""Golden trace schema and comparator for simulator parity work.

The comparator intentionally supports sparse snapshots.  A live game capture
may only expose a few fields at first, while simulator snapshots contain the
full engine state.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from sts2sim.api import new_run, serialize, step
from sts2sim.engine.models import Action, RunState

PARITY_TRACE_SCHEMA_VERSION = 1
DEFAULT_IGNORED_PATHS = ("rng", "replay_log")


class ParityModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class ParityCompareConfig(ParityModel):
    mode: Literal["subset", "exact"] = "subset"
    ignored_paths: tuple[str, ...] = DEFAULT_IGNORED_PATHS
    numeric_tolerances: dict[str, float] = Field(default_factory=dict)


class ParityMismatch(ParityModel):
    path: str
    kind: Literal["missing", "extra", "value", "type", "action_invalid", "step_error"]
    expected: Any = None
    actual: Any = None
    message: str = ""


class ParityTraceStep(ParityModel):
    step_index: int = 0
    action: dict[str, Any] = Field(default_factory=dict)
    external_action: Any = None
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParityTrace(ParityModel):
    schema_version: int = PARITY_TRACE_SCHEMA_VERSION
    trace_id: str = "trace"
    source: Literal["manual", "live", "run_file", "codex", "simulator"] = "manual"
    game_version: str | None = None
    seed: int | str = 0
    character_id: str = "IRONCLAD"
    ascension: int = 0
    simulator_replayable: bool = True
    initial_state: dict[str, Any] | None = None
    final_state: dict[str, Any] | None = None
    steps: tuple[ParityTraceStep, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParityStepResult(ParityModel):
    step_index: int
    action: dict[str, Any]
    before_mismatches: tuple[ParityMismatch, ...] = ()
    after_mismatches: tuple[ParityMismatch, ...] = ()
    error: str | None = None

    @property
    def matched(self) -> bool:
        return (
            self.error is None
            and not self.before_mismatches
            and not self.after_mismatches
        )


class ParityReport(ParityModel):
    trace_id: str
    schema_version: int
    source: str
    seed: int | str
    character_id: str
    ascension: int
    mode: Literal["subset", "exact"]
    matched: bool
    mismatch_count: int
    skipped_step_count: int = 0
    initial_mismatches: tuple[ParityMismatch, ...] = ()
    final_mismatches: tuple[ParityMismatch, ...] = ()
    steps: tuple[ParityStepResult, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_trace(trace_path: Path | str) -> ParityTrace:
    path = Path(trace_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Parity trace must be a JSON object: {path}")
    return ParityTrace.model_validate(_payload_with_step_indexes(payload))


def compare_trace_file(
    trace_path: Path | str,
    *,
    mode: Literal["subset", "exact"] = "subset",
    ignore_paths: Sequence[str] | None = None,
    source_data: Any | None = None,
    strict: bool = False,
) -> ParityReport:
    trace = load_trace(trace_path)
    config = ParityCompareConfig(
        mode=mode,
        ignored_paths=DEFAULT_IGNORED_PATHS + tuple(ignore_paths or ()),
    )
    report = compare_trace(trace, config=config, source_data=source_data)
    if strict and not report.matched:
        raise AssertionError("parity trace did not match simulator")
    return report


def compare_trace(
    trace: ParityTrace,
    *,
    config: ParityCompareConfig | None = None,
    source_data: Any | None = None,
) -> ParityReport:
    active_config = config or ParityCompareConfig()
    state = cast(
        RunState,
        new_run(
            seed=trace.seed,
            character_id=trace.character_id,
            ascension=trace.ascension,
            source_data=source_data,
        ),
    )

    initial_mismatches = (
        tuple(compare_snapshots(trace.initial_state, _state_snapshot(state), active_config))
        if trace.initial_state is not None
        else ()
    )

    step_results: list[ParityStepResult] = []
    skipped_step_count = 0
    if trace.simulator_replayable:
        for step_trace in trace.steps:
            state, result = _compare_replayable_step(
                state,
                step_trace,
                active_config,
            )
            step_results.append(result)
            if result.error is not None:
                break
    else:
        skipped_step_count = len(trace.steps)

    final_mismatches = (
        tuple(compare_snapshots(trace.final_state, _state_snapshot(state), active_config))
        if trace.final_state is not None and trace.simulator_replayable
        else ()
    )

    mismatch_count = (
        len(initial_mismatches)
        + len(final_mismatches)
        + sum(
            len(result.before_mismatches) + len(result.after_mismatches)
            for result in step_results
        )
    )
    matched = mismatch_count == 0 and all(result.error is None for result in step_results)
    return ParityReport(
        trace_id=trace.trace_id,
        schema_version=trace.schema_version,
        source=trace.source,
        seed=trace.seed,
        character_id=trace.character_id,
        ascension=trace.ascension,
        mode=active_config.mode,
        matched=matched,
        mismatch_count=mismatch_count,
        skipped_step_count=skipped_step_count,
        initial_mismatches=initial_mismatches,
        final_mismatches=final_mismatches,
        steps=tuple(step_results),
    )


def _compare_replayable_step(
    state: RunState,
    step_trace: ParityTraceStep,
    config: ParityCompareConfig,
) -> tuple[RunState, ParityStepResult]:
    action_payload = step_trace.action
    before_mismatches = (
        tuple(compare_snapshots(step_trace.before, _state_snapshot(state), config))
        if step_trace.before is not None
        else ()
    )

    try:
        action = Action.model_validate(action_payload)
    except Exception as exc:
        return state, ParityStepResult(
            step_index=step_trace.step_index,
            action=action_payload,
            before_mismatches=before_mismatches,
            error=str(exc),
            after_mismatches=(
                ParityMismatch(
                    path=f"steps[{step_trace.step_index}].action",
                    kind="action_invalid",
                    expected=action_payload,
                    message=str(exc),
                ),
            ),
        )

    try:
        next_state = cast(RunState, step(state, action))
    except Exception as exc:
        return state, ParityStepResult(
            step_index=step_trace.step_index,
            action=action_payload,
            before_mismatches=before_mismatches,
            error=str(exc),
            after_mismatches=(
                ParityMismatch(
                    path=f"steps[{step_trace.step_index}]",
                    kind="step_error",
                    expected=step_trace.after,
                    message=str(exc),
                ),
            ),
        )

    after_mismatches = (
        tuple(compare_snapshots(step_trace.after, _state_snapshot(next_state), config))
        if step_trace.after is not None
        else ()
    )
    return next_state, ParityStepResult(
        step_index=step_trace.step_index,
        action=action_payload,
        before_mismatches=before_mismatches,
        after_mismatches=after_mismatches,
    )


def compare_snapshots(
    expected: Mapping[str, Any] | Sequence[Any] | Any,
    actual: Mapping[str, Any] | Sequence[Any] | Any,
    config: ParityCompareConfig | None = None,
) -> tuple[ParityMismatch, ...]:
    active_config = config or ParityCompareConfig()
    mismatches: list[ParityMismatch] = []
    _compare_value(
        expected,
        actual,
        path="",
        config=active_config,
        mismatches=mismatches,
    )
    return tuple(mismatches)


def trace_template() -> dict[str, Any]:
    return {
        "schema_version": PARITY_TRACE_SCHEMA_VERSION,
        "trace_id": "example",
        "source": "manual",
        "game_version": None,
        "seed": 1,
        "character_id": "IRONCLAD",
        "ascension": 0,
        "initial_state": {
            "phase": "ancient",
            "act": 1,
            "floor": 0,
            "player": {"hp": 80, "max_hp": 80},
        },
        "steps": [],
        "metadata": {
            "step_shape_example": {
                "action": {"type": "choose_ancient", "target_id": "option_id"},
                "before": {"phase": "ancient"},
                "after": {"phase": "map"},
            }
        },
    }


def _state_snapshot(state: RunState) -> dict[str, Any]:
    return serialize(state)


def _compare_value(
    expected: Any,
    actual: Any,
    *,
    path: str,
    config: ParityCompareConfig,
    mismatches: list[ParityMismatch],
) -> None:
    if _is_ignored(path, config.ignored_paths):
        return

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            mismatches.append(
                ParityMismatch(
                    path=path or "$",
                    kind="type",
                    expected=_type_name(expected),
                    actual=_type_name(actual),
                    message="Expected object.",
                )
            )
            return
        _compare_mapping(expected, actual, path=path, config=config, mismatches=mismatches)
        return

    if _is_sequence(expected):
        if not _is_sequence(actual):
            mismatches.append(
                ParityMismatch(
                    path=path or "$",
                    kind="type",
                    expected=_type_name(expected),
                    actual=_type_name(actual),
                    message="Expected array.",
                )
            )
            return
        _compare_sequence(
            cast(Sequence[Any], expected),
            cast(Sequence[Any], actual),
            path=path,
            config=config,
            mismatches=mismatches,
        )
        return

    if _values_match(expected, actual, path, config):
        return

    mismatches.append(
        ParityMismatch(
            path=path or "$",
            kind="value",
            expected=expected,
            actual=actual,
            message="Values differ.",
        )
    )


def _compare_mapping(
    expected: Mapping[Any, Any],
    actual: Mapping[Any, Any],
    *,
    path: str,
    config: ParityCompareConfig,
    mismatches: list[ParityMismatch],
) -> None:
    expected_keys = {str(key): key for key in expected}
    actual_keys = {str(key): key for key in actual}
    for key_text, key in expected_keys.items():
        child_path = _join_path(path, key_text)
        if _is_ignored(child_path, config.ignored_paths):
            continue
        actual_key = actual_keys.get(key_text)
        if actual_key is None:
            mismatches.append(
                ParityMismatch(
                    path=child_path,
                    kind="missing",
                    expected=expected[key],
                    actual=None,
                    message="Expected key is missing.",
                )
            )
            continue
        _compare_value(
            expected[key],
            actual[actual_key],
            path=child_path,
            config=config,
            mismatches=mismatches,
        )

    if config.mode != "exact":
        return
    for key_text, key in actual_keys.items():
        child_path = _join_path(path, key_text)
        if key_text in expected_keys or _is_ignored(child_path, config.ignored_paths):
            continue
        mismatches.append(
            ParityMismatch(
                path=child_path,
                kind="extra",
                expected=None,
                actual=actual[key],
                message="Unexpected key is present.",
            )
        )


def _compare_sequence(
    expected: Sequence[Any],
    actual: Sequence[Any],
    *,
    path: str,
    config: ParityCompareConfig,
    mismatches: list[ParityMismatch],
) -> None:
    for index, expected_item in enumerate(expected):
        child_path = f"{path}[{index}]" if path else f"[{index}]"
        if index >= len(actual):
            mismatches.append(
                ParityMismatch(
                    path=child_path,
                    kind="missing",
                    expected=expected_item,
                    actual=None,
                    message="Expected array item is missing.",
                )
            )
            continue
        _compare_value(
            expected_item,
            actual[index],
            path=child_path,
            config=config,
            mismatches=mismatches,
        )

    if config.mode != "exact":
        return
    for index in range(len(expected), len(actual)):
        child_path = f"{path}[{index}]" if path else f"[{index}]"
        mismatches.append(
            ParityMismatch(
                path=child_path,
                kind="extra",
                expected=None,
                actual=actual[index],
                message="Unexpected array item is present.",
            )
        )


def _values_match(
    expected: Any,
    actual: Any,
    path: str,
    config: ParityCompareConfig,
) -> bool:
    tolerance = config.numeric_tolerances.get(path)
    if tolerance is not None and _is_number(expected) and _is_number(actual):
        return abs(float(expected) - float(actual)) <= tolerance
    return bool(expected == actual)


def _payload_with_step_indexes(payload: Mapping[Any, Any]) -> dict[str, Any]:
    normalized = {str(key): value for key, value in payload.items()}
    steps = normalized.get("steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes, bytearray)):
        return normalized
    indexed_steps: list[Any] = []
    for index, step_payload in enumerate(steps):
        if isinstance(step_payload, Mapping):
            step_dict = {str(key): value for key, value in step_payload.items()}
            step_dict.setdefault("step_index", index)
            indexed_steps.append(step_dict)
        else:
            indexed_steps.append(step_payload)
    normalized["steps"] = indexed_steps
    return normalized


def _join_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _is_ignored(path: str, ignored_paths: Sequence[str]) -> bool:
    if not path:
        return False
    for ignored in ignored_paths:
        if path == ignored:
            return True
        if ignored.endswith(".*") and path.startswith(ignored[:-2]):
            return True
    return False


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _type_name(value: object) -> str:
    return type(value).__name__


__all__ = [
    "DEFAULT_IGNORED_PATHS",
    "PARITY_TRACE_SCHEMA_VERSION",
    "ParityCompareConfig",
    "ParityMismatch",
    "ParityReport",
    "ParityStepResult",
    "ParityTrace",
    "ParityTraceStep",
    "compare_snapshots",
    "compare_trace",
    "compare_trace_file",
    "load_trace",
    "trace_template",
]
