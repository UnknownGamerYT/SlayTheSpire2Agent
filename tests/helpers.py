"""Small adapters for planned simulator interfaces used by tests."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return the repository root regardless of how deeply a test file is nested."""

    return Path(__file__).resolve().parents[1]


def import_attr(
    module_names: Sequence[str],
    attr_names: Sequence[str],
) -> Callable[..., Any] | None:
    """Return the first available planned interface attribute."""

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name and (
                exc.name == module_name or module_name.startswith(f"{exc.name}.")
            ):
                continue
            continue
        for attr_name in attr_names:
            attr = getattr(module, attr_name, None)
            if callable(attr):
                return attr
    return None


def call_with_supported_kwargs(func: Callable[..., Any], **kwargs: Any) -> Any:
    """Call ``func`` with the subset of keyword arguments it accepts."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
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


def jsonable(value: Any) -> Any:
    """Normalize result objects for assertions."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return jsonable(value.model_dump())
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_asdict"):
        return jsonable(value._asdict())
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(jsonable(item) for item in value)
    if hasattr(value, "__dict__"):
        return {
            key: jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return repr(value)


def canonical_digest(value: Any) -> str:
    """Hash a normalized value with stable key ordering."""

    payload = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
