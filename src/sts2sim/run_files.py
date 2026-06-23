"""Import local Slay the Spire 2 run-history files.

Finished ``.run`` files are useful parity artifacts, but they usually do not
contain every in-combat decision.  The importer therefore produces a summary and
a non-replayable parity trace scaffold rather than pretending it can replay the
run exactly.
"""

from __future__ import annotations

import gzip
import json
import zlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from sts2sim.parity import PARITY_TRACE_SCHEMA_VERSION, ParityTrace


class RunFileModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class RunFileSummary(RunFileModel):
    path: str
    seed: int | str | None = None
    character_id: str | None = None
    ascension: int | None = None
    victory: bool | None = None
    floor_reached: int | None = None
    score: int | None = None
    playtime_seconds: int | None = None
    hp: int | None = None
    max_hp: int | None = None
    gold: int | None = None
    relics: tuple[str, ...] = ()
    master_deck: tuple[str, ...] = ()
    path_per_floor: tuple[str, ...] = ()
    raw_key_count: int = 0
    raw_keys: tuple[str, ...] = ()
    replayable: bool = False
    notes: tuple[str, ...] = (
        "Finished run files usually lack full action-by-action combat decisions.",
    )

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class RunFileImportResult(RunFileModel):
    summary: RunFileSummary
    trace: ParityTrace | None = None
    trace_output_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def import_run_file(
    run_path: Path | str,
    *,
    trace_output_path: Path | str | None = None,
) -> RunFileImportResult:
    path = Path(run_path)
    payload = load_run_file(path)
    summary = summarize_run_file(path, payload)
    trace = run_file_to_trace(summary, payload)

    output_path: str | None = None
    if trace_output_path is not None:
        target = Path(trace_output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(trace.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_path = str(target)

    return RunFileImportResult(
        summary=summary,
        trace=trace,
        trace_output_path=output_path,
    )


def load_run_file(run_path: Path | str) -> dict[str, Any]:
    path = Path(run_path)
    data = path.read_bytes()
    text = _decode_run_bytes(data)
    payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Run file must contain a JSON object: {path}")
    return {str(key): value for key, value in payload.items()}


def summarize_run_file(path: Path, payload: Mapping[str, Any]) -> RunFileSummary:
    raw_keys = tuple(sorted(str(key) for key in payload))
    return RunFileSummary(
        path=str(path),
        seed=_first_present(payload, "seed", "seed_played", "seed_string", "special_seed"),
        character_id=_optional_str(
            _first_present(payload, "character_id", "character_chosen", "character")
        ),
        ascension=_optional_int(
            _first_present(payload, "ascension", "ascension_level", "ascensionLevel")
        ),
        victory=_optional_bool(_first_present(payload, "victory", "won", "is_victory")),
        floor_reached=_optional_int(
            _first_present(payload, "floor_reached", "floor", "floor_num", "floorReached")
        ),
        score=_optional_int(_first_present(payload, "score", "score_total")),
        playtime_seconds=_optional_int(
            _first_present(payload, "playtime", "playtime_seconds", "duration")
        ),
        hp=_optional_int(_first_present(payload, "current_hp", "hp", "health")),
        max_hp=_optional_int(_first_present(payload, "max_hp", "max_health")),
        gold=_optional_int(_first_present(payload, "gold", "gold_collected")),
        relics=_string_tuple(_first_present(payload, "relics", "relic_ids")),
        master_deck=_string_tuple(
            _first_present(payload, "master_deck", "deck", "cards", "card_ids")
        ),
        path_per_floor=_string_tuple(
            _first_present(payload, "path_per_floor", "path_taken", "room_history")
        ),
        raw_key_count=len(raw_keys),
        raw_keys=raw_keys,
    )


def run_file_to_trace(
    summary: RunFileSummary,
    payload: Mapping[str, Any],
) -> ParityTrace:
    metadata: dict[str, Any] = {
        "run_file_summary": summary.model_dump(mode="json"),
        "raw_keys": list(summary.raw_keys),
        "raw_payload": _jsonable(payload),
        "trace_note": "Run-history import is not simulator-replayable without action history.",
    }
    final_state = _final_snapshot(summary)
    return ParityTrace(
        schema_version=PARITY_TRACE_SCHEMA_VERSION,
        trace_id=Path(summary.path).stem or "run-file",
        source="run_file",
        seed=summary.seed if summary.seed is not None else 0,
        character_id=summary.character_id or "UNKNOWN",
        ascension=summary.ascension or 0,
        simulator_replayable=False,
        final_state=final_state or None,
        steps=(),
        metadata=metadata,
    )


def find_run_files(root: Path | str, *, limit: int | None = None) -> tuple[Path, ...]:
    base = Path(root)
    candidates = sorted(
        (
            path
            for pattern in ("*.run", "*.json")
            for path in base.rglob(pattern)
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return tuple(candidates[:limit] if limit is not None else candidates)


def _decode_run_bytes(data: bytes) -> str:
    errors: list[str] = []
    for decoder in (
        lambda value: value.decode("utf-8-sig"),
        lambda value: gzip.decompress(value).decode("utf-8-sig"),
        lambda value: zlib.decompress(value).decode("utf-8-sig"),
    ):
        try:
            return decoder(data)
        except (OSError, UnicodeDecodeError, zlib.error) as exc:
            errors.append(str(exc))
    raise ValueError(f"Could not decode run file as plain JSON, gzip, or zlib: {errors}")


def _final_snapshot(summary: RunFileSummary) -> dict[str, Any]:
    final: dict[str, Any] = {}
    if summary.floor_reached is not None:
        final["floor"] = summary.floor_reached
    player: dict[str, Any] = {}
    if summary.hp is not None:
        player["hp"] = summary.hp
    if summary.max_hp is not None:
        player["max_hp"] = summary.max_hp
    if summary.gold is not None:
        player["gold"] = summary.gold
    if player:
        final["player"] = player
    if summary.relics:
        final["relics"] = list(summary.relics)
    if summary.master_deck:
        final["master_deck_ids"] = list(summary.master_deck)
    if summary.path_per_floor:
        final["room_history"] = list(summary.path_per_floor)
    return final


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "win", "victory"}:
            return True
        if normalized in {"false", "no", "0", "loss", "defeat"}:
            return False
    if isinstance(value, int):
        return bool(value)
    return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return (str(value),) if isinstance(value, str) else ()
    if not isinstance(value, Iterable):
        return ()
    return tuple(str(item) for item in value if item is not None)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return str(value)


__all__ = [
    "RunFileImportResult",
    "RunFileSummary",
    "find_run_files",
    "import_run_file",
    "load_run_file",
    "run_file_to_trace",
    "summarize_run_file",
]
