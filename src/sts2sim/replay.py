"""Replay helpers for recorded simulator transcripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sts2sim.api import replay as _replay


def replay(
    replay_path: Path | str,
    *,
    strict: bool = True,
    path: Path | str | None = None,
    **kwargs: Any,
) -> Any:
    """Replay and optionally verify a replay JSON file."""
    return _replay(replay_path=replay_path, path=path, strict=strict, **kwargs)


def replay_run(
    replay_path: Path | str,
    *,
    strict: bool = True,
    path: Path | str | None = None,
    **kwargs: Any,
) -> Any:
    """Alias used by CLI/test discovery."""
    return replay(replay_path=replay_path, path=path, strict=strict, **kwargs)


def verify_replay(
    replay_path: Path | str,
    *,
    strict: bool = True,
    path: Path | str | None = None,
    **kwargs: Any,
) -> Any:
    """Alias used by CLI/test discovery."""
    return replay(replay_path=replay_path, path=path, strict=strict, **kwargs)

