"""Capture live game state from localhost STS2 control mods.

The client is deliberately endpoint-configurable.  STS2MCP and STS2-Agent both
expose local APIs, but their exact action payloads are owned by those projects.
This module records raw live payloads and lightweight normalized snapshots,
then stores them as non-replayable parity traces until an action mapper exists.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sts2sim.parity import PARITY_TRACE_SCHEMA_VERSION, ParityTrace, ParityTraceStep

DEFAULT_BASE_URL = "http://localhost:15526"
AUTO_BASE_URL = "auto"
KNOWN_BRIDGE_BASE_URLS = ("http://localhost:15526", "http://127.0.0.1:8080")
DEFAULT_HEALTH_PATHS = ("/", "/health", "/api/v1/health")
DEFAULT_STATE_PATHS = (
    "/api/v1/singleplayer",
    "/api/v1/multiplayer",
    "/api/v1/state",
    "/api/v1/game_state",
    "/api/v1/run",
    "/state",
    "/game_state",
)
DEFAULT_ACTIONS_PATHS = (
    "/api/v1/actions",
    "/api/v1/legal_actions",
    "/actions",
    "/legal_actions",
)
DEFAULT_ACTION_PATHS = (
    "/api/v1/singleplayer",
    "/api/v1/multiplayer",
    "/api/v1/action",
    "/api/v1/actions",
    "/action",
    "/actions",
)
STS2MCP_ACTION_NAMES = frozenset(
    {
        "advance_dialogue",
        "cancel_bundle_selection",
        "cancel_selection",
        "choose_event_option",
        "choose_map_node",
        "choose_rest_option",
        "claim_reward",
        "claim_treasure_relic",
        "combat_confirm_selection",
        "combat_select_card",
        "confirm_bundle_selection",
        "confirm_selection",
        "crystal_sphere_click_cell",
        "crystal_sphere_proceed",
        "crystal_sphere_set_tool",
        "discard_potion",
        "end_turn",
        "menu_select",
        "play_card",
        "proceed",
        "select_bundle",
        "select_card",
        "select_card_reward",
        "select_relic",
        "shop_purchase",
        "skip_card_reward",
        "skip_relic_selection",
        "undo_end_turn",
        "use_potion",
    }
)


class LiveModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class LiveApiConfig(LiveModel):
    base_url: str = DEFAULT_BASE_URL
    health_path: str | None = None
    state_path: str | None = None
    actions_path: str | None = None
    action_path: str | None = None
    action_method: Literal["post", "put"] = "post"
    action_envelope: Literal["action", "payload", "raw"] = "action"
    timeout_seconds: float = 5.0

    @model_validator(mode="after")
    def normalize_base_url(self) -> LiveApiConfig:
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        object.__setattr__(self, "timeout_seconds", max(0.1, float(self.timeout_seconds)))
        return self


class LiveCaptureStats(LiveModel):
    steps_requested: int
    steps_taken: int
    stopped_reason: str = ""
    action_type_counts: dict[str, int] = Field(default_factory=dict)
    phase_counts: dict[str, int] = Field(default_factory=dict)
    legal_action_counts: tuple[int, ...] = ()
    average_legal_actions: float = 0.0
    selected_probabilities: tuple[float, ...] = ()
    random_policy_likelihood: float = 1.0
    random_policy_log_likelihood: float = 0.0


class LiveBridgeProbe(LiveModel):
    base_url: str
    available: bool
    health_ok: bool = False
    state_ok: bool = False
    actions_ok: bool = False
    action_count: int = 0
    health: dict[str, Any] | None = None
    state_snapshot: dict[str, Any] | None = None
    errors: tuple[str, ...] = ()


class LiveCaptureResult(LiveModel):
    trace: ParityTrace
    stats: LiveCaptureStats
    health: dict[str, Any] | None = None
    output_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class LiveApiClient:
    def __init__(
        self,
        config: LiveApiConfig | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config or LiveApiConfig()
        self._client = client or httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, Any] | None:
        try:
            return self._get_first_json(
                (self.config.health_path,) if self.config.health_path else DEFAULT_HEALTH_PATHS
            )
        except LiveApiError:
            return None

    def state(self) -> dict[str, Any]:
        return self._get_first_json(
            (self.config.state_path,) if self.config.state_path else DEFAULT_STATE_PATHS
        )

    def actions(self, state_payload: Mapping[str, Any] | None = None) -> list[Any]:
        from_state = _extract_actions(state_payload or {})
        if from_state:
            return from_state
        try:
            payload = self._get_first_json(
                (self.config.actions_path,)
                if self.config.actions_path
                else DEFAULT_ACTIONS_PATHS
            )
        except LiveApiError:
            return []
        return _extract_actions(payload)

    def apply_action(self, action: Any) -> dict[str, Any] | None:
        payload = self._action_payload(action)
        paths = (self.config.action_path,) if self.config.action_path else DEFAULT_ACTION_PATHS
        errors: list[str] = []
        for path in paths:
            if path is None:
                continue
            try:
                response = self._request_json(self.config.action_method, path, json=payload)
                return response
            except LiveApiError as exc:
                errors.append(str(exc))
        raise LiveApiError(f"No live action endpoint accepted the action: {errors}")

    def _action_payload(self, action: Any) -> Any:
        if self.config.action_envelope == "raw":
            return action
        if self.config.action_envelope == "action" and _is_raw_sts2mcp_action(action):
            return dict(action)
        return {self.config.action_envelope: action}

    def _get_first_json(self, paths: Sequence[str | None]) -> dict[str, Any]:
        errors: list[str] = []
        for path in paths:
            if path is None:
                continue
            try:
                return self._request_json("get", path)
            except LiveApiError as exc:
                errors.append(str(exc))
        raise LiveApiError(f"No live API endpoint responded with JSON: {errors}")

    def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self._client.request(method.upper(), path, **kwargs)
        if response.status_code >= 400:
            raise LiveApiError(f"{method.upper()} {path} returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise LiveApiError(f"{method.upper()} {path} did not return JSON") from exc
        if not isinstance(payload, Mapping):
            raise LiveApiError(f"{method.upper()} {path} returned non-object JSON")
        return {str(key): value for key, value in payload.items()}


class LiveApiError(RuntimeError):
    """Raised when a live game API cannot be reached or parsed."""


def capture_live_state(
    *,
    base_url: str = DEFAULT_BASE_URL,
    state_path: str | None = None,
    output_path: Path | str | None = None,
    client: LiveApiClient | None = None,
) -> LiveCaptureResult:
    if client is None and _is_auto_base_url(base_url):
        base_url = detect_live_bridge().base_url
    config = LiveApiConfig(base_url=base_url, state_path=state_path)
    active_client = client or LiveApiClient(config)
    try:
        health = active_client.health()
        state = active_client.state()
    finally:
        if client is None:
            active_client.close()

    trace = _trace_from_live_payloads(
        initial_payload=state,
        steps=(),
        stats_metadata={"mode": "capture"},
    )
    stats = LiveCaptureStats(steps_requested=0, steps_taken=0, stopped_reason="captured_state")
    return _capture_result(trace, stats, health=health, output_path=output_path)


def live_play(
    *,
    base_url: str = DEFAULT_BASE_URL,
    max_steps: int = 10,
    policy: Literal["first", "random", "prefer_attack"] = "first",
    seed: int | str = 0,
    state_path: str | None = None,
    actions_path: str | None = None,
    action_path: str | None = None,
    action_envelope: Literal["action", "payload", "raw"] = "action",
    output_path: Path | str | None = None,
    client: LiveApiClient | None = None,
) -> LiveCaptureResult:
    if client is None and _is_auto_base_url(base_url):
        base_url = detect_live_bridge().base_url
    config = LiveApiConfig(
        base_url=base_url,
        state_path=state_path,
        actions_path=actions_path,
        action_path=action_path,
        action_envelope=action_envelope,
    )
    active_client = client or LiveApiClient(config)
    rng = random.Random(str(seed))
    steps: list[ParityTraceStep] = []
    legal_counts: list[int] = []
    selected_probabilities: list[float] = []
    action_counter: Counter[str] = Counter()
    phase_counter: Counter[str] = Counter()
    stopped_reason = "max_steps"

    try:
        health = active_client.health()
        current_state = active_client.state()
        initial_state = current_state
        for index in range(max(0, max_steps)):
            actions = active_client.actions(current_state)
            if not actions:
                current_state, actions = _wait_for_live_actions(active_client, current_state)
            legal_count = len(actions)
            legal_counts.append(legal_count)
            phase = _snapshot_phase(normalize_live_snapshot(current_state))
            if phase:
                phase_counter[phase] += 1
            if not actions:
                stopped_reason = "no_actions"
                break
            action = _choose_action(actions, policy=policy, rng=rng)
            probability = 1.0 / legal_count if legal_count else 0.0
            selected_probabilities.append(probability)
            action_counter[_action_type(action)] += 1
            response = active_client.apply_action(action)
            next_state = _wait_for_live_state_update(active_client, current_state)
            state_changed = _state_fingerprint(current_state) != _state_fingerprint(next_state)
            steps.append(
                ParityTraceStep(
                    step_index=index,
                    external_action=action,
                    before=normalize_live_snapshot(current_state),
                    after=normalize_live_snapshot(next_state),
                    metadata={
                        "legal_action_count": legal_count,
                        "selected_probability_if_uniform": probability,
                        "action_type": _action_type(action),
                        "raw_action": _jsonable(action),
                        "response": _jsonable(response),
                        "state_changed": state_changed,
                    },
                )
            )
            current_state = next_state
            if _response_is_error(response):
                stopped_reason = "action_error"
                break
            if not state_changed:
                stopped_reason = "state_unchanged"
                break
    finally:
        if client is None:
            active_client.close()

    stats = _live_stats(
        steps_requested=max_steps,
        steps_taken=len(steps),
        stopped_reason=stopped_reason,
        legal_counts=legal_counts,
        probabilities=selected_probabilities,
        action_counter=action_counter,
        phase_counter=phase_counter,
    )
    trace = _trace_from_live_payloads(
        initial_payload=initial_state,
        steps=tuple(steps),
        stats_metadata={
            "mode": "live_play",
            "policy": policy,
            "stats": stats.model_dump(mode="json"),
        },
    )
    return _capture_result(trace, stats, health=health, output_path=output_path)


def _wait_for_live_state_update(
    client: LiveApiClient,
    previous_state: Mapping[str, Any],
    *,
    timeout_seconds: float = 3.0,
    poll_seconds: float = 0.15,
    initial_delay_seconds: float = 0.25,
    stable_reads_required: int = 2,
) -> dict[str, Any]:
    previous_fingerprint = _state_fingerprint(previous_state)
    deadline = time.monotonic() + timeout_seconds
    if initial_delay_seconds > 0:
        time.sleep(initial_delay_seconds)
    latest = client.state()
    latest_fingerprint = _state_fingerprint(latest)
    changed = latest_fingerprint != previous_fingerprint
    stable_reads = 1 if changed else 0
    while time.monotonic() < deadline:
        time.sleep(poll_seconds)
        candidate = client.state()
        candidate_fingerprint = _state_fingerprint(candidate)
        if candidate_fingerprint != latest_fingerprint:
            latest = candidate
            latest_fingerprint = candidate_fingerprint
            changed = changed or candidate_fingerprint != previous_fingerprint
            stable_reads = 1 if changed else 0
            continue
        latest = candidate
        if changed:
            stable_reads += 1
        if changed and stable_reads >= stable_reads_required:
            return latest
    return latest


def _wait_for_live_actions(
    client: LiveApiClient,
    current_state: Mapping[str, Any],
    *,
    timeout_seconds: float = 12.0,
    poll_seconds: float = 0.25,
) -> tuple[dict[str, Any], list[Any]]:
    if not _should_wait_for_actions(current_state):
        return dict(current_state), []
    deadline = time.monotonic() + timeout_seconds
    latest = dict(current_state)
    while time.monotonic() < deadline:
        time.sleep(poll_seconds)
        latest = client.state()
        actions = client.actions(latest)
        if actions:
            return latest, actions
        if not _should_wait_for_actions(latest):
            return latest, []
    return latest, []


def _should_wait_for_actions(state: Mapping[str, Any]) -> bool:
    source = _unwrap_payload(state)
    state_type = _first_string(source, "state_type")
    return state_type in {"monster", "elite", "boss"}


def probe_live_bridges(
    *,
    base_urls: Sequence[str] | None = None,
    timeout_seconds: float = 1.0,
) -> tuple[LiveBridgeProbe, ...]:
    urls = tuple(base_urls or KNOWN_BRIDGE_BASE_URLS)
    probes: list[LiveBridgeProbe] = []
    for url in urls:
        config = LiveApiConfig(base_url=url, timeout_seconds=timeout_seconds)
        client = LiveApiClient(config)
        errors: list[str] = []
        health: dict[str, Any] | None = None
        state: dict[str, Any] | None = None
        actions: list[Any] = []
        try:
            health = client.health()
        except Exception as exc:  # pragma: no cover - defensive local bridge probe.
            errors.append(f"health: {exc}")
        try:
            state = client.state()
        except Exception as exc:
            errors.append(f"state: {exc}")
        try:
            actions = client.actions(state)
        except Exception as exc:
            errors.append(f"actions: {exc}")
        finally:
            client.close()
        probes.append(
            LiveBridgeProbe(
                base_url=url,
                available=health is not None or state is not None,
                health_ok=health is not None,
                state_ok=state is not None,
                actions_ok=bool(actions),
                action_count=len(actions),
                health=health,
                state_snapshot=normalize_live_snapshot(state) if state is not None else None,
                errors=tuple(errors),
            )
        )
    return tuple(probes)


def detect_live_bridge(
    *,
    base_urls: Sequence[str] | None = None,
    timeout_seconds: float = 1.0,
) -> LiveBridgeProbe:
    probes = probe_live_bridges(base_urls=base_urls, timeout_seconds=timeout_seconds)
    for probe in probes:
        if probe.state_ok:
            return probe
    for probe in probes:
        if probe.available:
            return probe
    raise LiveApiError("No known live bridge is reachable.")


def normalize_live_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = _unwrap_payload(payload)
    run_source = _first_mapping(source, "run")
    snapshot: dict[str, Any] = {}
    phase = _first_string(
        source,
        "state_type",
        "phase",
        "screen",
        "current_screen",
        "room_phase",
        "state",
    )
    if phase is not None:
        snapshot["phase"] = phase
    menu_screen = _first_string(source, "menu_screen")
    if menu_screen is not None:
        snapshot["menu_screen"] = menu_screen
    options = _first_sequence(source, "options", "available_options")
    if options:
        snapshot["options"] = [
            label for item in options if (label := _option_label(item)) is not None
        ]
    hand_select_source = _first_mapping(source, "hand_select")
    if hand_select_source is not None:
        snapshot["hand_select"] = _hand_select_snapshot(hand_select_source)
    card_select_source = _first_mapping(source, "card_select")
    if card_select_source is not None:
        snapshot["card_select"] = _card_select_snapshot(card_select_source)
    event_source = _first_mapping(source, "event")
    if event_source is not None:
        snapshot["event"] = _event_snapshot(event_source)
    for key in ("act", "floor", "ascension"):
        value = _find_scalar(source, key)
        if value is None and run_source is not None:
            value = _find_scalar(run_source, key)
        if value is not None:
            snapshot[key] = value

    player_source = _first_mapping(source, "player", "player_state", "hero")
    if player_source is None:
        player_source = source
    player = _player_snapshot(player_source)
    if player:
        snapshot["player"] = player

    relics = _first_sequence(source, "relics", "relic_ids")
    if not relics and player_source is not None:
        relics = _first_sequence(player_source, "relics", "relic_ids")
    if relics:
        snapshot["relics"] = [_relic_id(item) for item in relics]
    potions = _first_sequence(source, "potions", "potion_ids")
    if not potions and player_source is not None:
        potions = _first_sequence(player_source, "potions", "potion_ids")
    if potions:
        snapshot["potions"] = [_relic_id(item) for item in potions]

    combat_source = _first_mapping(source, "combat", "combat_state", "battle")
    combat: dict[str, Any] = {}
    if combat_source is not None:
        combat.update(_combat_snapshot(combat_source))
    if player_source is not None:
        combat.update(_combat_snapshot(player_source))
    if combat:
        snapshot["combat"] = combat

    return snapshot


def _capture_result(
    trace: ParityTrace,
    stats: LiveCaptureStats,
    *,
    health: dict[str, Any] | None,
    output_path: Path | str | None,
) -> LiveCaptureResult:
    target_path: str | None = None
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(trace.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        target_path = str(target)
    return LiveCaptureResult(trace=trace, stats=stats, health=health, output_path=target_path)


def _trace_from_live_payloads(
    *,
    initial_payload: Mapping[str, Any],
    steps: tuple[ParityTraceStep, ...],
    stats_metadata: Mapping[str, Any],
) -> ParityTrace:
    source = _unwrap_payload(initial_payload)
    metadata = {
        **dict(stats_metadata),
        "raw_initial_payload": _jsonable(initial_payload),
        "trace_note": "Live traces use external action payloads and are not replayable yet.",
    }
    seed = _first_value(source, "seed", "run_seed", "seed_string") or 0
    character_id = (
        _first_string(source, "character_id", "character", "character_chosen") or "UNKNOWN"
    )
    ascension = _optional_int(_first_value(source, "ascension", "ascension_level")) or 0
    return ParityTrace(
        schema_version=PARITY_TRACE_SCHEMA_VERSION,
        trace_id=f"live-{seed}",
        source="live",
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        simulator_replayable=False,
        initial_state=normalize_live_snapshot(initial_payload),
        final_state=steps[-1].after if steps else normalize_live_snapshot(initial_payload),
        steps=steps,
        metadata=metadata,
    )


def _state_fingerprint(payload: Mapping[str, Any]) -> str:
    return json.dumps(normalize_live_snapshot(payload), sort_keys=True, default=str)


def _live_stats(
    *,
    steps_requested: int,
    steps_taken: int,
    stopped_reason: str,
    legal_counts: Sequence[int],
    probabilities: Sequence[float],
    action_counter: Counter[str],
    phase_counter: Counter[str],
) -> LiveCaptureStats:
    average = sum(legal_counts) / len(legal_counts) if legal_counts else 0.0
    likelihood = 1.0
    log_likelihood = 0.0
    for probability in probabilities:
        likelihood *= probability
        if probability > 0:
            log_likelihood += math.log(probability)
    return LiveCaptureStats(
        steps_requested=steps_requested,
        steps_taken=steps_taken,
        stopped_reason=stopped_reason,
        action_type_counts=dict(action_counter),
        phase_counts=dict(phase_counter),
        legal_action_counts=tuple(legal_counts),
        average_legal_actions=average,
        selected_probabilities=tuple(probabilities),
        random_policy_likelihood=likelihood,
        random_policy_log_likelihood=log_likelihood,
    )


def _choose_action(
    actions: Sequence[Any],
    *,
    policy: str,
    rng: random.Random,
) -> Any:
    if policy == "random":
        return rng.choice(list(actions))
    if policy == "prefer_attack":
        menu_choice = _preferred_menu_action(actions)
        if menu_choice is not None:
            return menu_choice
        for wanted in (
            "play_card",
            "attack",
            "play",
            "confirm",
            "card",
            "reward",
            "map",
            "event",
            "proceed",
            "end",
        ):
            for action in actions:
                if wanted in _action_type(action):
                    return action
    return actions[0]


def _extract_actions(payload: Mapping[str, Any]) -> list[Any]:
    sts2mcp_actions = _extract_sts2mcp_actions(payload)
    if sts2mcp_actions:
        return sts2mcp_actions
    for key in ("actions", "legal_actions", "available_actions", "choices", "options"):
        value = payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return list(value)
    nested = _first_mapping(payload, "state", "game_state", "run", "current_run")
    if nested is not None and nested is not payload:
        return _extract_actions(nested)
    return []


def _preferred_menu_action(actions: Sequence[Any]) -> Any | None:
    menu_actions = [
        action
        for action in actions
        if isinstance(action, Mapping) and action.get("action") == "menu_select"
    ]
    if not menu_actions:
        return None
    for wanted in ("embark", "confirm", "standard", "singleplayer", "IRONCLAD"):
        for action in menu_actions:
            if str(action.get("option") or "").lower() == wanted.lower():
                return action
    for action in menu_actions:
        if str(action.get("option") or "").lower() not in {"back", "quit"}:
            return action
    return None


def _extract_sts2mcp_actions(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    state_type = _first_string(payload, "state_type")
    if state_type is None:
        return []
    actions: list[dict[str, Any]] = []

    if state_type in {"menu", "game_over"}:
        options = _first_sequence(payload, "options")
        game_over = _first_mapping(payload, "game_over")
        if not options and game_over is not None:
            options = _first_sequence(game_over, "options")
        return [
            {"action": "menu_select", "option": option_name}
            for option in options
            if (option_name := _enabled_option_name(option)) is not None
        ]

    if state_type in {"monster", "elite", "boss"}:
        battle = _first_mapping(payload, "battle") or {}
        player = _first_mapping(payload, "player") or {}
        enemies = [
            enemy
            for enemy in _first_sequence(battle, "enemies")
            if isinstance(enemy, Mapping)
        ]
        first_target = _first_enemy_target(enemies)
        if bool(battle.get("is_play_phase")):
            for card in _first_sequence(player, "hand"):
                if not isinstance(card, Mapping) or not bool(card.get("can_play")):
                    continue
                card_index = _optional_int(card.get("index"))
                if card_index is None:
                    continue
                action: dict[str, Any] = {"action": "play_card", "card_index": card_index}
                target_type = str(card.get("target_type") or "")
                if "Enemy" in target_type:
                    if first_target is None:
                        continue
                    action["target"] = first_target
                actions.append(action)
            actions.append({"action": "end_turn"})
        return actions

    if state_type == "map":
        map_state = _first_mapping(payload, "map") or {}
        return _indexed_actions(
            "choose_map_node",
            _first_sequence(map_state, "next_options"),
        )

    if state_type == "event":
        event = _first_mapping(payload, "event") or {}
        if bool(event.get("in_dialogue")):
            actions.append({"action": "advance_dialogue"})
        actions.extend(
            _indexed_actions(
                "choose_event_option",
                _first_sequence(event, "options"),
                enabled_key="is_locked",
                enabled_when=False,
            )
        )
        return actions

    if state_type == "rewards":
        rewards = _first_mapping(payload, "rewards") or {}
        actions.extend(_indexed_actions("claim_reward", _first_sequence(rewards, "items")))
        if bool(rewards.get("can_proceed")):
            actions.append({"action": "proceed"})
        return actions

    if state_type == "card_reward":
        card_reward = _first_mapping(payload, "card_reward") or {}
        actions.extend(
            _indexed_actions(
                "select_card_reward",
                _first_sequence(card_reward, "cards"),
                index_key="card_index",
            )
        )
        if bool(card_reward.get("can_skip")):
            actions.append({"action": "skip_card_reward"})
        return actions

    if state_type == "card_select":
        card_select = _first_mapping(payload, "card_select") or {}
        if bool(card_select.get("can_confirm")):
            actions.append({"action": "confirm_selection"})
        actions.extend(_indexed_actions("select_card", _first_sequence(card_select, "cards")))
        if bool(card_select.get("can_skip")) or bool(card_select.get("can_cancel")):
            actions.append({"action": "cancel_selection"})
        return actions

    if state_type == "hand_select":
        hand_select = _first_mapping(payload, "hand_select") or {}
        actions.extend(
            _indexed_actions(
                "combat_select_card",
                _first_sequence(hand_select, "cards"),
                index_key="card_index",
            )
        )
        if bool(hand_select.get("can_confirm")):
            actions.append({"action": "combat_confirm_selection"})
        return actions

    if state_type == "relic_select":
        relic_select = _first_mapping(payload, "relic_select") or {}
        actions.extend(_indexed_actions("select_relic", _first_sequence(relic_select, "relics")))
        if bool(relic_select.get("can_skip")):
            actions.append({"action": "skip_relic_selection"})
        return actions

    if state_type == "bundle_select":
        bundle_select = _first_mapping(payload, "bundle_select") or {}
        actions.extend(_indexed_actions("select_bundle", _first_sequence(bundle_select, "bundles")))
        if bool(bundle_select.get("can_confirm")):
            actions.append({"action": "confirm_bundle_selection"})
        if bool(bundle_select.get("can_cancel")):
            actions.append({"action": "cancel_bundle_selection"})
        return actions

    if state_type == "rest_site":
        rest_site = _first_mapping(payload, "rest_site") or {}
        actions.extend(
            _indexed_actions(
                "choose_rest_option",
                _first_sequence(rest_site, "options"),
                enabled_key="is_enabled",
            )
        )
        if bool(rest_site.get("can_proceed")):
            actions.append({"action": "proceed"})
        return actions

    if state_type in {"shop", "fake_merchant"}:
        shop = _first_mapping(payload, "shop")
        if shop is None:
            fake_merchant = _first_mapping(payload, "fake_merchant") or {}
            shop = _first_mapping(fake_merchant, "shop") or {}
        actions.extend(
            _indexed_actions(
                "shop_purchase",
                _first_sequence(shop, "items"),
                enabled_key="can_afford",
            )
        )
        if bool(shop.get("can_proceed")):
            actions.append({"action": "proceed"})
        return actions

    if state_type == "treasure":
        treasure = _first_mapping(payload, "treasure") or {}
        actions.extend(
            _indexed_actions("claim_treasure_relic", _first_sequence(treasure, "relics"))
        )
        if bool(treasure.get("can_proceed")):
            actions.append({"action": "proceed"})
        return actions

    return actions


def _indexed_actions(
    action_name: str,
    items: Sequence[Any],
    *,
    enabled_key: str | None = None,
    enabled_when: bool = True,
    index_key: str = "index",
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for fallback_index, item in enumerate(items):
        if not isinstance(item, Mapping):
            continue
        if enabled_key is not None and bool(item.get(enabled_key)) is not enabled_when:
            continue
        index = _optional_int(item.get("index"))
        actions.append(
            {"action": action_name, index_key: fallback_index if index is None else index}
        )
    return actions


def _response_is_error(response: Mapping[str, Any] | None) -> bool:
    if response is None:
        return False
    status = response.get("status")
    return status == "error" or response.get("error") is not None


def _first_enemy_target(enemies: Sequence[Mapping[str, Any]]) -> str | None:
    for enemy in enemies:
        target = _first_value(enemy, "entity_id", "combat_id")
        if target is not None:
            return str(target)
    return None


def _enabled_option_name(option: Any) -> str | None:
    if isinstance(option, Mapping):
        if "enabled" in option and not bool(option.get("enabled")):
            return None
        return _option_label(option)
    return _option_label(option)


def _option_label(option: Any) -> str | None:
    if isinstance(option, Mapping):
        value = _first_value(option, "name", "option", "id", "title")
    else:
        value = option
    if value is None:
        return None
    label = str(value).strip()
    return label or None


def _action_type(action: Any) -> str:
    if isinstance(action, Mapping):
        for key in ("type", "action_type", "id", "name", "command"):
            value = action.get(key)
            if value is not None:
                return _normalize_token(str(value))
        value = action.get("action")
        if value is not None:
            return _normalize_token(str(value))
    return _normalize_token(str(action))


def _unwrap_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    current: Mapping[str, Any] = payload
    for key in ("state", "game_state", "current_run"):
        value = current.get(key)
        if isinstance(value, Mapping):
            current = {str(item_key): item_value for item_key, item_value in value.items()}
    return current


def _is_raw_sts2mcp_action(action: Any) -> bool:
    if not isinstance(action, Mapping):
        return False
    value = action.get("action")
    return isinstance(value, str) and value in STS2MCP_ACTION_NAMES


def _player_snapshot(source: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {
        "hp": ("hp", "current_hp", "health", "currentHealth"),
        "max_hp": ("max_hp", "max_health", "maxHealth"),
        "block": ("block", "armor"),
        "energy": ("energy", "current_energy"),
        "gold": ("gold", "money"),
    }
    result: dict[str, Any] = {}
    for target, keys in aliases.items():
        value = _first_value(source, *keys)
        if value is not None:
            result[target] = value
    return result


def _combat_snapshot(source: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    monsters = _first_sequence(source, "monsters", "enemies")
    if monsters:
        result["monsters"] = [
            _monster_snapshot(item)
            for item in monsters
            if isinstance(item, Mapping)
        ]
    for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        cards = _first_sequence(source, zone, zone.replace("_pile", ""))
        if cards:
            result[zone] = [_card_id(item) for item in cards]
    return result


def _hand_select_snapshot(source: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("mode", "prompt", "can_confirm", "can_cancel"):
        value = source.get(key)
        if value is not None:
            result[key] = value
    cards = _first_sequence(source, "cards")
    if cards:
        result["cards"] = [_card_id(item) for item in cards]
    selected_cards = _first_sequence(source, "selected_cards")
    if selected_cards:
        result["selected_cards"] = [_card_id(item) for item in selected_cards]
    return result


def _card_select_snapshot(source: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("screen_type", "prompt", "can_confirm", "can_cancel", "preview_showing"):
        value = source.get(key)
        if value is not None:
            result[key] = value
    cards = _first_sequence(source, "cards")
    if cards:
        result["cards"] = [_card_id(item) for item in cards]
    return result


def _event_snapshot(source: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("event_id", "event_name", "is_ancient", "in_dialogue"):
        value = source.get(key)
        if value is not None:
            result[key] = value
    options = _first_sequence(source, "options")
    if options:
        result["options"] = [
            option
            for item in options
            if (option := _event_option_snapshot(item)) is not None
        ]
    return result


def _event_option_snapshot(option: Any) -> dict[str, Any] | None:
    if not isinstance(option, Mapping):
        label = _option_label(option)
        return {"title": label} if label is not None else None
    result: dict[str, Any] = {}
    for key in ("index", "title", "description", "is_locked", "is_proceed", "was_chosen"):
        value = option.get(key)
        if value is not None:
            result[key] = value
    return result or None


def _monster_snapshot(source: Mapping[str, Any]) -> dict[str, Any]:
    monster = {str(key): value for key, value in source.items()}
    result: dict[str, Any] = {}
    for target, keys in {
        "monster_id": ("monster_id", "id", "enemy_id"),
        "name": ("name",),
        "hp": ("hp", "current_hp", "health"),
        "max_hp": ("max_hp", "max_health"),
        "block": ("block",),
        "intent": ("intent",),
        "move_id": ("move_id", "move"),
    }.items():
        value = _first_value(monster, *keys)
        if value is not None:
            result[target] = value
    return result


def _card_id(value: Any) -> str:
    if isinstance(value, Mapping):
        found = _first_value(value, "card_id", "id", "name")
        if found is not None:
            return str(found)
    return str(value)


def _relic_id(value: Any) -> str:
    if isinstance(value, Mapping):
        found = _first_value(value, "relic_id", "potion_id", "id", "name")
        if found is not None:
            return str(found)
    return str(value)


def _snapshot_phase(snapshot: Mapping[str, Any]) -> str | None:
    value = snapshot.get("phase")
    return str(value) if value is not None else None


def _first_mapping(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any] | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return {str(item_key): item_value for item_key, item_value in value.items()}
    return None


def _first_sequence(payload: Mapping[str, Any], *keys: str) -> Sequence[Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return value
    return ()


def _first_string(payload: Mapping[str, Any], *keys: str) -> str | None:
    value = _first_value(payload, *keys)
    if value is None:
        return None
    return str(value)


def _first_value(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _find_scalar(payload: Mapping[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value is not None and not isinstance(value, (Mapping, Sequence)):
        return value
    return None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _is_auto_base_url(value: str) -> bool:
    return value.strip().lower() == AUTO_BASE_URL


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return str(value)


__all__ = [
    "AUTO_BASE_URL",
    "DEFAULT_BASE_URL",
    "KNOWN_BRIDGE_BASE_URLS",
    "LiveApiClient",
    "LiveApiConfig",
    "LiveApiError",
    "LiveBridgeProbe",
    "LiveCaptureResult",
    "LiveCaptureStats",
    "capture_live_state",
    "detect_live_bridge",
    "live_play",
    "normalize_live_snapshot",
    "probe_live_bridges",
]
