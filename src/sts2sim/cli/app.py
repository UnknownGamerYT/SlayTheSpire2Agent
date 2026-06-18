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
    entries_value = normalized_payload.get("entries", [])
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
    else:
        normalized_payload["entries"] = entries
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
            cache_dir=cache_dir,
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
            cache_dir=cache_dir,
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
