"""Start a live Slay the Spire 2 run through a localhost bridge."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from sts2sim.live_capture import AUTO_BASE_URL, LiveApiError, detect_live_bridge

DEFAULT_SINGLEPLAYER_PATH = "/api/v1/singleplayer"


class LiveStartModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LiveStartStep(LiveStartModel):
    screen: str
    option: str
    response: dict[str, Any] = Field(default_factory=dict)


class LiveStartResult(LiveStartModel):
    base_url: str
    started: bool
    stopped_reason: str
    selected_character_id: str | None = None
    selected_character_name: str | None = None
    selected_ascension: int | None = None
    final_state_type: str | None = None
    final_menu_screen: str | None = None
    run: dict[str, Any] | None = None
    steps: tuple[LiveStartStep, ...] = ()
    final_state: dict[str, Any] = Field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def start_live_run(
    *,
    base_url: str = AUTO_BASE_URL,
    character: str = "random",
    ascension: str = "random",
    seed: int | str = 0,
    max_steps: int = 80,
    delay_seconds: float = 0.75,
    singleplayer_path: str = DEFAULT_SINGLEPLAYER_PATH,
    output_path: Path | str | None = None,
) -> LiveStartResult:
    """Navigate the live game menu and start a standard singleplayer run."""

    if _is_auto_base_url(base_url):
        base_url = detect_live_bridge().base_url

    rng = random.Random(str(seed))
    steps: list[LiveStartStep] = []
    selected_character: dict[str, Any] | None = None
    selected_ascension: int | None = None

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=10.0) as client:
        state = _get_state(client, singleplayer_path)
        for _ in range(max(1, max_steps)):
            state_type = _string(state.get("state_type"))
            menu_screen = _normalize(_string(state.get("menu_screen")))
            if state_type != "menu":
                result = _result(
                    base_url=base_url,
                    started=True,
                    stopped_reason=(
                        "run_started" if selected_character is not None else "run_already_active"
                    ),
                    selected_character=selected_character,
                    selected_ascension=selected_ascension,
                    steps=steps,
                    final_state=state,
                )
                _write_result(result, output_path)
                return result

            option: str | None = None
            if menu_screen == "main":
                option = _first_enabled_option(state, "singleplayer")
            elif menu_screen == "singleplayer":
                option = _first_enabled_option(state, "standard")
            elif menu_screen == "character_select":
                if selected_character is None:
                    selected_character = _choose_character(state, character, rng)
                    option = str(selected_character["id"])
                elif selected_ascension is None:
                    target = _choose_ascension(state, ascension, rng)
                    selected_ascension = target
                    current = _optional_int(state.get("ascension"))
                    if current is not None and current != target:
                        option = f"ascension_{target}"
                    else:
                        option = _first_enabled_option(state, "confirm", "embark")
                else:
                    option = _first_enabled_option(state, "confirm", "embark")
            elif menu_screen in {"popup", "tutorial_prompt"}:
                option = _first_enabled_option(
                    state,
                    "ignore",
                    "no",
                    "confirm",
                    "continue",
                    "ok",
                    "yes",
                )
            else:
                option = _fallback_menu_option(state)

            if option is None:
                if _state_has_run(state):
                    time.sleep(max(0.0, delay_seconds))
                    state = _get_state(client, singleplayer_path)
                    continue
                result = _result(
                    base_url=base_url,
                    started=False,
                    stopped_reason=f"no_supported_option_on_{menu_screen or 'unknown_menu'}",
                    selected_character=selected_character,
                    selected_ascension=selected_ascension,
                    steps=steps,
                    final_state=state,
                )
                _write_result(result, output_path)
                return result

            response = _menu_select(client, singleplayer_path, option)
            steps.append(
                LiveStartStep(
                    screen=menu_screen or "menu",
                    option=option,
                    response=response,
                )
            )
            time.sleep(max(0.0, delay_seconds))
            state = _get_state(client, singleplayer_path)

    result = _result(
        base_url=base_url,
        started=False,
        stopped_reason="max_steps_exhausted",
        selected_character=selected_character,
        selected_ascension=selected_ascension,
        steps=steps,
        final_state=state,
    )
    _write_result(result, output_path)
    return result


def _get_state(client: httpx.Client, path: str) -> dict[str, Any]:
    response = client.get(path)
    if response.status_code >= 400:
        raise LiveApiError(f"GET {path} returned HTTP {response.status_code}: {response.text}")
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise LiveApiError(f"GET {path} returned non-object JSON")
    return {str(key): value for key, value in payload.items()}


def _menu_select(client: httpx.Client, path: str, option: str) -> dict[str, Any]:
    response = client.post(path, json={"action": "menu_select", "option": option})
    if response.status_code >= 400:
        raise LiveApiError(
            f"POST {path} menu_select {option!r} returned HTTP "
            f"{response.status_code}: {response.text}"
        )
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise LiveApiError(f"POST {path} menu_select {option!r} returned non-object JSON")
    result = {str(key): value for key, value in payload.items()}
    if _normalize(_string(result.get("status"))) == "error":
        raise LiveApiError(f"menu_select {option!r} failed: {result.get('message')}")
    return result


def _choose_character(
    state: Mapping[str, Any],
    wanted: str,
    rng: random.Random,
) -> dict[str, Any]:
    characters = _unlocked_characters(state)
    if not characters:
        raise LiveApiError("Character select has no unlocked characters")

    normalized = _normalize(wanted)
    if normalized and normalized != "random":
        for character in characters:
            if normalized in {
                _normalize(_string(character.get("id"))),
                _normalize(_string(character.get("name"))),
            }:
                return character
        available = ", ".join(str(character.get("id")) for character in characters)
        raise LiveApiError(
            f"Character {wanted!r} is not unlocked/available. Available: {available}"
        )

    actual_characters = [
        character
        for character in characters
        if _normalize(_string(character.get("id"))) != "random"
    ]
    return rng.choice(actual_characters or characters)


def _choose_ascension(
    state: Mapping[str, Any],
    wanted: str,
    rng: random.Random,
) -> int:
    current = _optional_int(state.get("ascension")) or 0
    max_ascension = _optional_int(state.get("max_ascension"))
    if max_ascension is None:
        return current
    max_ascension = max(0, max_ascension)
    normalized = _normalize(wanted)
    if normalized == "random":
        return rng.randint(0, max_ascension)
    if normalized == "max":
        return max_ascension
    if normalized in {"current", "selected"}:
        return min(current, max_ascension)
    target = _optional_int(wanted)
    if target is None:
        raise LiveApiError("ascension must be 'random', 'max', 'current', or an integer")
    if target < 0 or target > max_ascension:
        raise LiveApiError(
            f"Ascension {target} is not unlocked. Available range: 0-{max_ascension}"
        )
    return target


def _unlocked_characters(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    enabled_option_names = {
        _normalize(option["name"])
        for option in _options(state)
        if option["enabled"]
    }
    characters: list[dict[str, Any]] = []
    raw_characters = state.get("characters")
    if not isinstance(raw_characters, Sequence) or isinstance(raw_characters, (str, bytes)):
        return characters
    for raw in raw_characters:
        if not isinstance(raw, Mapping):
            continue
        character = {str(key): value for key, value in raw.items()}
        character_id = _string(character.get("id"))
        if not character_id or _truthy(character.get("locked")):
            continue
        if enabled_option_names and _normalize(character_id) not in enabled_option_names:
            continue
        characters.append(character)
    return characters


def _first_enabled_option(state: Mapping[str, Any], *wanted_names: str) -> str | None:
    enabled = {
        _normalize(option["name"]): option["name"]
        for option in _options(state)
        if option["enabled"]
    }
    for wanted in wanted_names:
        found = enabled.get(_normalize(wanted))
        if found is not None:
            return str(found)
    return None


def _fallback_menu_option(state: Mapping[str, Any]) -> str | None:
    for wanted in (
        "singleplayer",
        "standard",
        "ignore",
        "no",
        "confirm",
        "embark",
        "continue",
        "ok",
    ):
        found = _first_enabled_option(state, wanted)
        if found is not None:
            return found
    return None


def _options(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_options = state.get("options")
    if not isinstance(raw_options, Sequence) or isinstance(raw_options, (str, bytes)):
        return []
    options: list[dict[str, Any]] = []
    for raw in raw_options:
        if isinstance(raw, str):
            options.append({"name": raw, "enabled": True})
        elif isinstance(raw, Mapping):
            name = _string(raw.get("name") or raw.get("id") or raw.get("option"))
            if name:
                options.append(
                    {
                        "name": name,
                        "enabled": not _is_false(raw.get("enabled")),
                    }
                )
    return options


def _result(
    *,
    base_url: str,
    started: bool,
    stopped_reason: str,
    selected_character: Mapping[str, Any] | None,
    selected_ascension: int | None,
    steps: Sequence[LiveStartStep],
    final_state: Mapping[str, Any],
) -> LiveStartResult:
    run = final_state.get("run")
    return LiveStartResult(
        base_url=base_url,
        started=started,
        stopped_reason=stopped_reason,
        selected_character_id=(
            _string(selected_character.get("id")) if selected_character is not None else None
        ),
        selected_character_name=(
            _string(selected_character.get("name")) if selected_character is not None else None
        ),
        selected_ascension=selected_ascension,
        final_state_type=_string(final_state.get("state_type")),
        final_menu_screen=_string(final_state.get("menu_screen")),
        run={str(key): value for key, value in run.items()} if isinstance(run, Mapping) else None,
        steps=tuple(steps),
        final_state={str(key): value for key, value in final_state.items()},
    )


def _state_has_run(state: Mapping[str, Any]) -> bool:
    return isinstance(state.get("run"), Mapping)


def _write_result(result: LiveStartResult, output_path: Path | str | None) -> None:
    if output_path is None:
        return
    import json

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "enabled", "locked"}
    return bool(value)


def _is_false(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"0", "false", "no", "n", "disabled"}
    return False


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _normalize(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _is_auto_base_url(value: str) -> bool:
    return value.strip().lower() == AUTO_BASE_URL


__all__ = [
    "LiveStartResult",
    "LiveStartStep",
    "start_live_run",
]
