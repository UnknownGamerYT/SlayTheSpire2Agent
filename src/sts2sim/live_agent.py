"""Play a real STS2 run through a live bridge and compare it to the simulator.

The live bridge is authoritative for the real game.  The simulator comparison
is intentionally a baseline snapshot comparison until every bridge action has a
lossless simulator action mapper.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from sts2sim.api import new_run, serialize
from sts2sim.live_capture import (
    AUTO_BASE_URL,
    DEFAULT_BASE_URL,
    LiveApiClient,
    LiveApiConfig,
    detect_live_bridge,
)
from sts2sim.live_parity import LiveStepParityResult, compare_live_step_to_simulator
from sts2sim.live_start import LiveStartResult, start_live_run
from sts2sim.parity import ParityCompareConfig, ParityMismatch, compare_snapshots

DEFAULT_SINGLEPLAYER_PATH = "/api/v1/singleplayer"
DEFAULT_OUTPUT_PATH = Path("live_traces/live_agent_latest.json")


class _LiveClient(Protocol):
    def health(self) -> dict[str, Any] | None: ...

    def state(self) -> dict[str, Any]: ...

    def apply_action(self, action: Any) -> dict[str, Any] | None: ...

    def close(self) -> None: ...


class LiveAgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class LiveAgentDecision(LiveAgentModel):
    state_type: str
    action: dict[str, Any] | None = None
    reason: str


class LiveAgentComparison(LiveAgentModel):
    supported: bool = True
    character_id: str | None = None
    ascension: int | None = None
    simulator_seed: int | str = 0
    note: str = (
        "Baseline snapshot comparison only; bridge actions are not replayed in "
        "the simulator yet."
    )
    live_summary: dict[str, Any] = Field(default_factory=dict)
    simulator_summary: dict[str, Any] = Field(default_factory=dict)
    mismatch_count: int = 0
    mismatches: tuple[ParityMismatch, ...] = ()


class LiveAgentStep(LiveAgentModel):
    step_index: int
    state_type: str
    decision: LiveAgentDecision
    before: dict[str, Any]
    after: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    comparison_before: LiveAgentComparison
    comparison_after: LiveAgentComparison | None = None
    true_step_comparison: LiveStepParityResult | None = None


class LiveAgentResult(LiveAgentModel):
    base_url: str
    steps_requested: int
    steps_taken: int
    stopped_reason: str
    started_run: bool = False
    start_result: dict[str, Any] | None = None
    health: dict[str, Any] | None = None
    initial: dict[str, Any] = Field(default_factory=dict)
    final: dict[str, Any] = Field(default_factory=dict)
    final_state_type: str | None = None
    steps: tuple[LiveAgentStep, ...] = ()
    output_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def play_live_agent(
    *,
    base_url: str = AUTO_BASE_URL,
    max_steps: int = 10,
    seed: int | str = 0,
    simulator_seed: int | str = 0,
    start_if_needed: bool = True,
    character: str = "random",
    ascension: str = "random",
    state_path: str | None = None,
    action_path: str | None = None,
    delay_seconds: float = 0.35,
    settle_timeout_seconds: float = 3.0,
    output_path: Path | str | None = DEFAULT_OUTPUT_PATH,
    client: _LiveClient | None = None,
) -> LiveAgentResult:
    """Run a conservative live-game agent and emit a comparison report."""

    if client is None and _is_auto_base_url(base_url):
        base_url = detect_live_bridge().base_url
    resolved_base_url = base_url.rstrip("/")
    singleplayer_path = state_path or DEFAULT_SINGLEPLAYER_PATH

    start_result: LiveStartResult | None = None
    if client is None and start_if_needed:
        initial_probe = LiveApiClient(
            LiveApiConfig(
                base_url=resolved_base_url,
                state_path=singleplayer_path,
                action_path=action_path or singleplayer_path,
                action_envelope="raw",
            )
        )
        try:
            initial_state = initial_probe.state()
        finally:
            initial_probe.close()
        if _state_type(initial_state) == "menu":
            start_result = start_live_run(
                base_url=resolved_base_url,
                character=character,
                ascension=ascension,
                seed=seed,
                singleplayer_path=singleplayer_path,
                delay_seconds=delay_seconds,
            )

    config = LiveApiConfig(
        base_url=resolved_base_url,
        state_path=singleplayer_path,
        action_path=action_path or singleplayer_path,
        action_envelope="raw",
    )
    active_client: _LiveClient = client or LiveApiClient(config)
    rng = random.Random(str(seed))
    steps: list[LiveAgentStep] = []
    stopped_reason = "max_steps"

    try:
        health = active_client.health()
        current_state = active_client.state()
        current_state = _wait_for_settled_state(
            active_client,
            current_state,
            timeout_seconds=settle_timeout_seconds,
            interval_seconds=delay_seconds,
        )
        initial_summary = live_state_summary(current_state)

        for step_index in range(max(0, max_steps)):
            current_state = _wait_for_settled_state(
                active_client,
                current_state,
                timeout_seconds=settle_timeout_seconds,
                interval_seconds=delay_seconds,
            )
            before_summary = live_state_summary(current_state)
            comparison_before = compare_live_state_to_simulator(
                current_state,
                simulator_seed=simulator_seed,
            )
            decision = choose_live_action(current_state, rng=rng)
            if decision.action is None:
                stopped_reason = decision.reason
                break

            response = active_client.apply_action(decision.action)
            time.sleep(max(0.0, delay_seconds))
            next_state = active_client.state()
            next_state = _wait_for_settled_state(
                active_client,
                next_state,
                timeout_seconds=settle_timeout_seconds,
                interval_seconds=delay_seconds,
            )
            after_summary = live_state_summary(next_state)
            comparison_after = compare_live_state_to_simulator(
                next_state,
                simulator_seed=simulator_seed,
            )
            true_step_comparison = compare_live_step_to_simulator(
                before=current_state,
                action=decision.action,
                after=next_state,
                seed=simulator_seed,
            )
            steps.append(
                LiveAgentStep(
                    step_index=step_index,
                    state_type=decision.state_type,
                    decision=decision,
                    before=before_summary,
                    after=after_summary,
                    response=_optional_json_object(response),
                    comparison_before=comparison_before,
                    comparison_after=comparison_after,
                    true_step_comparison=true_step_comparison,
                )
            )
            if _response_failed(response):
                stopped_reason = "action_failed"
                current_state = next_state
                break
            current_state = next_state
        else:
            stopped_reason = "max_steps"

        final_summary = live_state_summary(current_state)
    finally:
        if client is None:
            active_client.close()

    result = LiveAgentResult(
        base_url=resolved_base_url,
        steps_requested=max_steps,
        steps_taken=len(steps),
        stopped_reason=stopped_reason,
        started_run=start_result is not None and start_result.started,
        start_result=start_result.as_dict() if start_result is not None else None,
        health=health,
        initial=initial_summary,
        final=final_summary,
        final_state_type=_optional_string(final_summary.get("state_type")),
        steps=tuple(steps),
    )
    return _write_result(result, output_path)


def choose_live_action(
    state: Mapping[str, Any],
    *,
    rng: random.Random | None = None,
) -> LiveAgentDecision:
    """Choose one conservative STS2MCP action for the current live state."""

    del rng
    state_type = _state_type(state)
    if state_type == "menu":
        return LiveAgentDecision(
            state_type=state_type,
            reason="menu_state_requires_start_live_run",
        )
    if state_type == "map":
        return _choose_map_action(state)
    if state_type in {"monster", "elite", "boss"}:
        return _choose_combat_action(state, state_type)
    if state_type == "hand_select":
        return _choose_hand_select_action(state)
    if state_type == "rewards":
        return _choose_rewards_action(state)
    if state_type == "card_reward":
        return _choose_card_reward_action(state)
    if state_type == "event":
        return _choose_event_action(state)
    if state_type == "rest_site":
        return _choose_rest_action(state)
    if state_type in {"shop", "fake_merchant"}:
        return _choose_shop_action(state, state_type)
    if state_type == "treasure":
        return _choose_treasure_action(state)
    if state_type == "card_select":
        return _choose_card_select_action(state)
    if state_type == "bundle_select":
        return _choose_bundle_select_action(state)
    if state_type == "relic_select":
        return _choose_relic_select_action(state)
    if state_type == "crystal_sphere":
        return _choose_crystal_sphere_action(state)
    if state_type == "game_over":
        return LiveAgentDecision(state_type=state_type, reason="game_over")
    return LiveAgentDecision(
        state_type=state_type,
        reason=f"unsupported_state_type:{state_type or 'unknown'}",
    )


def _wait_for_settled_state(
    client: _LiveClient,
    state: dict[str, Any],
    *,
    timeout_seconds: float,
    interval_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    current = state
    while _state_needs_settle(current) and time.monotonic() < deadline:
        time.sleep(max(0.05, interval_seconds))
        current = client.state()
    return current


def _state_needs_settle(state: Mapping[str, Any]) -> bool:
    decision = choose_live_action(state)
    return decision.reason in {
        "map_has_no_next_options",
        "shop_cannot_proceed_yet",
        "treasure_not_ready",
        "waiting_for_player_turn",
        "waiting_for_play_phase",
    }


def live_state_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact live fields that are useful for simulator comparison."""

    run = _mapping(state.get("run"))
    player = _mapping(state.get("player"))
    summary: dict[str, Any] = {
        "state_type": _state_type(state),
        "phase": _live_phase(state),
    }
    for key in ("act", "floor", "ascension"):
        value = _optional_int(run.get(key))
        if value is not None:
            summary[key] = value

    character = _optional_string(player.get("character"))
    character_id = live_character_id(character)
    if character is not None:
        summary["character"] = character
    if character_id is not None:
        summary["character_id"] = character_id

    player_summary = _live_player_summary(player)
    if player_summary:
        summary["player"] = player_summary

    relics = [_item_id(item) for item in _sequence(player.get("relics"))]
    relics = [item for item in relics if item is not None]
    if relics:
        summary["relics"] = relics

    potions = [_item_id(item) for item in _sequence(player.get("potions"))]
    potions = [item for item in potions if item is not None]
    if potions:
        summary["potions"] = potions
    max_potion_slots = _optional_int(player.get("max_potion_slots"))
    if max_potion_slots is not None:
        summary["max_potion_slots"] = max_potion_slots

    battle = _mapping(state.get("battle"))
    if battle:
        summary["combat"] = _live_combat_summary(battle, player)

    map_state = _mapping(state.get("map"))
    if map_state:
        boss = _mapping(map_state.get("boss"))
        next_options = _sequence(map_state.get("next_options"))
        summary["map"] = {
            "next_option_count": len(next_options),
            "boss": {
                key: value
                for key, value in {
                    "id": _optional_string(boss.get("id")),
                    "name": _optional_string(boss.get("name")),
                }.items()
                if value is not None
            },
        }
    return summary


def compare_live_state_to_simulator(
    state: Mapping[str, Any],
    *,
    simulator_seed: int | str = 0,
) -> LiveAgentComparison:
    live_summary = live_state_summary(state)
    character_id = live_character_id(_optional_string(live_summary.get("character")))
    if character_id is None:
        character_id = _optional_string(live_summary.get("character_id"))
    ascension = _optional_int(live_summary.get("ascension")) or 0
    if character_id is None:
        return LiveAgentComparison(
            supported=False,
            character_id=None,
            ascension=ascension,
            simulator_seed=simulator_seed,
            note="Live state does not expose a character id/name that can be mapped.",
            live_summary=live_summary,
        )

    try:
        simulator = serialize(new_run(simulator_seed, character_id, ascension))
    except Exception as exc:
        return LiveAgentComparison(
            supported=False,
            character_id=character_id,
            ascension=ascension,
            simulator_seed=simulator_seed,
            note=f"Could not create simulator baseline: {exc}",
            live_summary=live_summary,
        )

    simulator_summary = _simulator_summary(simulator)
    expected = _comparable_live_summary(live_summary)
    actual = _comparable_simulator_summary(simulator_summary, expected)
    mismatches = compare_snapshots(
        expected,
        actual,
        ParityCompareConfig(mode="subset", ignored_paths=()),
    )
    return LiveAgentComparison(
        supported=True,
        character_id=character_id,
        ascension=ascension,
        simulator_seed=simulator_seed,
        live_summary=expected,
        simulator_summary=actual,
        mismatch_count=len(mismatches),
        mismatches=mismatches,
    )


def live_character_id(character: str | None) -> str | None:
    if character is None:
        return None
    normalized = character.strip().lower().replace("the ", "").replace(" ", "_")
    aliases = {
        "ironclad": "IRONCLAD",
        "the_ironclad": "IRONCLAD",
        "silent": "SILENT",
        "the_silent": "SILENT",
        "defect": "DEFECT",
        "the_defect": "DEFECT",
        "watcher": "WATCHER",
        "the_watcher": "WATCHER",
        "necrobinder": "NECROBINDER",
        "the_necrobinder": "NECROBINDER",
        "regent": "REGENT",
        "the_regent": "REGENT",
    }
    if normalized.upper() in aliases.values():
        return normalized.upper()
    return aliases.get(normalized)


def _choose_map_action(state: Mapping[str, Any]) -> LiveAgentDecision:
    map_state = _mapping(state.get("map"))
    options = _sequence(map_state.get("next_options"))
    if not options:
        return LiveAgentDecision(state_type="map", reason="map_has_no_next_options")
    first = _first_mapping_item(options)
    index = _optional_int(first.get("index")) if first is not None else None
    if index is None:
        return LiveAgentDecision(state_type="map", reason="map_next_option_missing_index")
    return LiveAgentDecision(
        state_type="map",
        action={"action": "choose_map_node", "index": index},
        reason="choose_first_map_node",
    )


def _choose_combat_action(
    state: Mapping[str, Any],
    state_type: str,
) -> LiveAgentDecision:
    battle = _mapping(state.get("battle"))
    if _optional_string(battle.get("turn")) not in {None, "player"}:
        return LiveAgentDecision(state_type=state_type, reason="waiting_for_player_turn")
    if battle.get("is_play_phase") is False:
        return LiveAgentDecision(state_type=state_type, reason="waiting_for_play_phase")

    player = _mapping(state.get("player"))
    hand = [_mapping(item) for item in _sequence(player.get("hand")) if isinstance(item, Mapping)]
    target_id = _first_enemy_target_id(battle)
    playable = [card for card in hand if card.get("can_play") is True]
    attacks = [
        card
        for card in playable
        if (_optional_string(card.get("type")) or "").lower() == "attack"
        and _card_can_be_targeted(card, target_id)
    ]
    for group, reason in ((attacks, "play_first_attack"), (playable, "play_first_playable_card")):
        for card in group:
            card_index = _optional_int(card.get("index"))
            if card_index is None:
                continue
            action: dict[str, Any] = {"action": "play_card", "card_index": card_index}
            if _requires_enemy_target(card) and target_id is not None:
                action["target"] = target_id
            elif _requires_enemy_target(card):
                continue
            return LiveAgentDecision(state_type=state_type, action=action, reason=reason)

    return LiveAgentDecision(
        state_type=state_type,
        action={"action": "end_turn"},
        reason="no_playable_cards",
    )


def _choose_hand_select_action(state: Mapping[str, Any]) -> LiveAgentDecision:
    hand_select = _mapping(state.get("hand_select"))
    if hand_select.get("can_confirm") is True:
        return LiveAgentDecision(
            state_type="hand_select",
            action={"action": "combat_confirm_selection"},
            reason="confirm_current_hand_selection",
        )
    first = _first_mapping_item(_sequence(hand_select.get("cards")))
    index = _optional_int(first.get("index")) if first is not None else None
    if index is not None:
        return LiveAgentDecision(
            state_type="hand_select",
            action={"action": "combat_select_card", "card_index": index},
            reason="select_first_hand_card",
        )
    return LiveAgentDecision(state_type="hand_select", reason="hand_select_has_no_cards")


def _choose_rewards_action(state: Mapping[str, Any]) -> LiveAgentDecision:
    rewards = _mapping(state.get("rewards"))
    items = [
        _mapping(item)
        for item in _sequence(rewards.get("items"))
        if isinstance(item, Mapping)
    ]
    potion_slots_available = _potion_slots_available(_mapping(state.get("player")))
    for item in items:
        reward_type = (_optional_string(item.get("type")) or "").lower()
        if reward_type == "card":
            continue
        if reward_type == "potion" and not potion_slots_available:
            continue
        index = _optional_int(item.get("index"))
        if index is not None:
            return LiveAgentDecision(
                state_type="rewards",
                action={"action": "claim_reward", "index": index},
                reason=f"claim_{reward_type or 'reward'}_reward",
            )
    if rewards.get("can_proceed") is True:
        return LiveAgentDecision(
            state_type="rewards",
            action={"action": "proceed"},
            reason="leave_rewards",
        )
    card_item = next(
        (
            item
            for item in items
            if (_optional_string(item.get("type")) or "").lower() in {"card", "special_card"}
        ),
        None,
    )
    index = _optional_int(card_item.get("index")) if card_item is not None else None
    if index is not None:
        return LiveAgentDecision(
            state_type="rewards",
            action={"action": "claim_reward", "index": index},
            reason="open_forced_card_reward",
        )
    return LiveAgentDecision(state_type="rewards", reason="rewards_have_no_claimable_items")


def _choose_card_reward_action(state: Mapping[str, Any]) -> LiveAgentDecision:
    card_reward = _mapping(state.get("card_reward"))
    if card_reward.get("can_skip") is True:
        return LiveAgentDecision(
            state_type="card_reward",
            action={"action": "skip_card_reward"},
            reason="skip_optional_card_reward",
        )
    first = _first_mapping_item(_sequence(card_reward.get("cards")))
    index = _optional_int(first.get("index")) if first is not None else None
    if index is not None:
        return LiveAgentDecision(
            state_type="card_reward",
            action={"action": "select_card_reward", "card_index": index},
            reason="select_first_forced_card_reward",
        )
    return LiveAgentDecision(state_type="card_reward", reason="card_reward_has_no_cards")


def _choose_event_action(state: Mapping[str, Any]) -> LiveAgentDecision:
    event = _mapping(state.get("event"))
    if event.get("in_dialogue") is True:
        return LiveAgentDecision(
            state_type="event",
            action={"action": "advance_dialogue"},
            reason="advance_event_dialogue",
        )
    options = [
        _mapping(item)
        for item in _sequence(event.get("options"))
        if isinstance(item, Mapping)
    ]
    proceed = _first_event_option(options, proceed=True)
    choice = proceed or _first_event_option(options, proceed=False)
    index = _optional_int(choice.get("index")) if choice is not None else None
    if index is not None:
        return LiveAgentDecision(
            state_type="event",
            action={"action": "choose_event_option", "index": index},
            reason="choose_first_unlocked_event_option",
        )
    return LiveAgentDecision(state_type="event", reason="event_has_no_unlocked_options")


def _choose_rest_action(state: Mapping[str, Any]) -> LiveAgentDecision:
    rest = _mapping(state.get("rest_site"))
    options = [
        _mapping(item)
        for item in _sequence(rest.get("options"))
        if isinstance(item, Mapping)
    ]
    rest_option = _first_rest_option(options, "rest")
    choice = rest_option or _first_enabled_option(options)
    index = _optional_int(choice.get("index")) if choice is not None else None
    if index is not None:
        return LiveAgentDecision(
            state_type="rest_site",
            action={"action": "choose_rest_option", "index": index},
            reason="choose_rest_site_option",
        )
    if rest.get("can_proceed") is True:
        return LiveAgentDecision(
            state_type="rest_site",
            action={"action": "proceed"},
            reason="leave_rest_site",
        )
    return LiveAgentDecision(state_type="rest_site", reason="rest_site_has_no_enabled_options")


def _choose_shop_action(state: Mapping[str, Any], state_type: str) -> LiveAgentDecision:
    shop = _mapping(state.get("shop"))
    if state_type == "fake_merchant":
        fake_merchant = _mapping(state.get("fake_merchant"))
        shop = _mapping(fake_merchant.get("shop")) or shop
    if shop.get("can_proceed") is True:
        return LiveAgentDecision(
            state_type=state_type,
            action={"action": "proceed"},
            reason="leave_shop_without_purchase",
        )
    return LiveAgentDecision(state_type=state_type, reason="shop_cannot_proceed_yet")


def _choose_treasure_action(state: Mapping[str, Any]) -> LiveAgentDecision:
    treasure = _mapping(state.get("treasure"))
    first = _first_mapping_item(_sequence(treasure.get("relics")))
    index = _optional_int(first.get("index")) if first is not None else None
    if index is not None:
        return LiveAgentDecision(
            state_type="treasure",
            action={"action": "claim_treasure_relic", "index": index},
            reason="claim_first_treasure_relic",
        )
    if treasure.get("can_proceed") is True:
        return LiveAgentDecision(
            state_type="treasure",
            action={"action": "proceed"},
            reason="leave_treasure",
        )
    return LiveAgentDecision(state_type="treasure", reason="treasure_not_ready")


def _choose_card_select_action(state: Mapping[str, Any]) -> LiveAgentDecision:
    card_select = _mapping(state.get("card_select"))
    if card_select.get("preview_showing") is True and card_select.get("can_confirm") is True:
        return LiveAgentDecision(
            state_type="card_select",
            action={"action": "confirm_selection"},
            reason="confirm_card_selection_preview",
        )
    if card_select.get("can_cancel") is True:
        return LiveAgentDecision(
            state_type="card_select",
            action={"action": "cancel_selection"},
            reason="cancel_optional_card_selection",
        )
    if card_select.get("can_confirm") is True:
        return LiveAgentDecision(
            state_type="card_select",
            action={"action": "confirm_selection"},
            reason="confirm_card_selection",
        )
    first = _first_mapping_item(_sequence(card_select.get("cards")))
    index = _optional_int(first.get("index")) if first is not None else None
    if index is not None:
        return LiveAgentDecision(
            state_type="card_select",
            action={"action": "select_card", "index": index},
            reason="select_first_required_card",
        )
    return LiveAgentDecision(state_type="card_select", reason="card_select_has_no_cards")


def _choose_bundle_select_action(state: Mapping[str, Any]) -> LiveAgentDecision:
    bundle_select = _mapping(state.get("bundle_select"))
    if bundle_select.get("can_confirm") is True:
        return LiveAgentDecision(
            state_type="bundle_select",
            action={"action": "confirm_bundle_selection"},
            reason="confirm_bundle_selection",
        )
    if bundle_select.get("can_cancel") is True:
        return LiveAgentDecision(
            state_type="bundle_select",
            action={"action": "cancel_bundle_selection"},
            reason="cancel_optional_bundle_selection",
        )
    first = _first_mapping_item(_sequence(bundle_select.get("bundles")))
    index = _optional_int(first.get("index")) if first is not None else None
    if index is not None:
        return LiveAgentDecision(
            state_type="bundle_select",
            action={"action": "select_bundle", "index": index},
            reason="select_first_required_bundle",
        )
    return LiveAgentDecision(state_type="bundle_select", reason="bundle_select_has_no_bundles")


def _choose_relic_select_action(state: Mapping[str, Any]) -> LiveAgentDecision:
    relic_select = _mapping(state.get("relic_select"))
    if relic_select.get("can_skip") is True:
        return LiveAgentDecision(
            state_type="relic_select",
            action={"action": "skip_relic_selection"},
            reason="skip_optional_relic_selection",
        )
    first = _first_mapping_item(_sequence(relic_select.get("relics")))
    index = _optional_int(first.get("index")) if first is not None else None
    if index is not None:
        return LiveAgentDecision(
            state_type="relic_select",
            action={"action": "select_relic", "index": index},
            reason="select_first_required_relic",
        )
    return LiveAgentDecision(state_type="relic_select", reason="relic_select_has_no_relics")


def _choose_crystal_sphere_action(state: Mapping[str, Any]) -> LiveAgentDecision:
    sphere = _mapping(state.get("crystal_sphere"))
    if sphere.get("can_proceed") is True:
        return LiveAgentDecision(
            state_type="crystal_sphere",
            action={"action": "crystal_sphere_proceed"},
            reason="leave_crystal_sphere",
        )
    first_cell = _first_mapping_item(_sequence(sphere.get("clickable_cells")))
    x = _optional_int(first_cell.get("x")) if first_cell is not None else None
    y = _optional_int(first_cell.get("y")) if first_cell is not None else None
    if x is not None and y is not None:
        return LiveAgentDecision(
            state_type="crystal_sphere",
            action={"action": "crystal_sphere_click_cell", "x": x, "y": y},
            reason="click_first_crystal_sphere_cell",
        )
    if sphere.get("can_use_big_tool") is True:
        return LiveAgentDecision(
            state_type="crystal_sphere",
            action={"action": "crystal_sphere_set_tool", "tool": "big"},
            reason="select_big_crystal_sphere_tool",
        )
    return LiveAgentDecision(state_type="crystal_sphere", reason="crystal_sphere_no_action")


def _live_player_summary(player: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("hp", "max_hp", "block", "energy", "max_energy", "gold", "stars"):
        value = _optional_int(player.get(key))
        if value is not None:
            summary[key] = value
    return summary


def _live_combat_summary(
    battle: Mapping[str, Any],
    player: Mapping[str, Any],
) -> dict[str, Any]:
    enemies = [
        _mapping(item)
        for item in _sequence(battle.get("enemies"))
        if isinstance(item, Mapping)
    ]
    return {
        key: value
        for key, value in {
            "round": _optional_int(battle.get("round")),
            "turn": _optional_string(battle.get("turn")),
            "is_play_phase": battle.get("is_play_phase"),
            "hand_count": len(_sequence(player.get("hand"))),
            "draw_pile_count": _optional_int(player.get("draw_pile_count")),
            "discard_pile_count": _optional_int(player.get("discard_pile_count")),
            "exhaust_pile_count": _optional_int(player.get("exhaust_pile_count")),
            "enemies": [
                {
                    key: value
                    for key, value in {
                        "entity_id": _optional_string(enemy.get("entity_id")),
                        "name": _optional_string(enemy.get("name")),
                        "hp": _optional_int(enemy.get("hp")),
                        "max_hp": _optional_int(enemy.get("max_hp")),
                        "block": _optional_int(enemy.get("block")),
                    }.items()
                    if value is not None
                }
                for enemy in enemies
            ],
        }.items()
        if value is not None
    }


def _simulator_summary(simulator: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("character_id", "ascension", "act", "floor", "phase", "relics", "potions"):
        value = simulator.get(key)
        if value is not None:
            summary[key] = value
    player = _mapping(simulator.get("player"))
    player_summary = _live_player_summary(player)
    if player_summary:
        summary["player"] = player_summary
    return summary


def _comparable_live_summary(live_summary: Mapping[str, Any]) -> dict[str, Any]:
    comparable: dict[str, Any] = {}
    for key in (
        "phase",
        "character_id",
        "ascension",
        "act",
        "floor",
        "player",
        "relics",
        "potions",
    ):
        value = live_summary.get(key)
        if value is not None:
            comparable[key] = value
    return comparable


def _comparable_simulator_summary(
    simulator_summary: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    actual: dict[str, Any] = {}
    for key in expected:
        value = simulator_summary.get(key)
        if value is not None:
            actual[key] = value
    return actual


def _first_enemy_target_id(battle: Mapping[str, Any]) -> str | None:
    for item in _sequence(battle.get("enemies")):
        if not isinstance(item, Mapping):
            continue
        hp = _optional_int(item.get("hp"))
        if hp is not None and hp <= 0:
            continue
        target_id = _optional_string(item.get("entity_id"))
        if target_id is not None:
            return target_id
    return None


def _requires_enemy_target(card: Mapping[str, Any]) -> bool:
    target_type = (_optional_string(card.get("target_type")) or "").lower()
    return target_type in {"anyenemy", "enemy", "any_enemy"}


def _card_can_be_targeted(card: Mapping[str, Any], target_id: str | None) -> bool:
    return not _requires_enemy_target(card) or target_id is not None


def _first_event_option(
    options: Sequence[Mapping[str, Any]],
    *,
    proceed: bool,
) -> Mapping[str, Any] | None:
    for option in options:
        if option.get("is_locked") is True:
            continue
        if _optional_int(option.get("index")) is None:
            continue
        if bool(option.get("is_proceed")) == proceed:
            return option
    return None


def _first_rest_option(
    options: Sequence[Mapping[str, Any]],
    option_id: str,
) -> Mapping[str, Any] | None:
    for option in options:
        if option.get("is_enabled") is False:
            continue
        if _optional_string(option.get("id")) == option_id:
            return option
    return None


def _first_enabled_option(options: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for option in options:
        if option.get("is_enabled") is False:
            continue
        if _optional_int(option.get("index")) is not None:
            return option
    return None


def _potion_slots_available(player: Mapping[str, Any]) -> bool:
    max_slots = _optional_int(player.get("max_potion_slots"))
    if max_slots is None:
        return True
    return len(_sequence(player.get("potions"))) < max_slots


def _live_phase(state: Mapping[str, Any]) -> str:
    state_type = _state_type(state)
    if state_type in {"monster", "elite", "boss", "hand_select"}:
        return "combat"
    if state_type == "rest_site":
        return "rest"
    if state_type == "rewards":
        return "reward"
    if state_type == "card_reward":
        return "reward"
    if state_type == "fake_merchant":
        return "shop"
    if state_type in {"card_select", "bundle_select", "relic_select", "crystal_sphere"}:
        return "overlay"
    if state_type:
        return state_type
    return "unknown"


def _state_type(state: Mapping[str, Any]) -> str:
    return (_optional_string(state.get("state_type")) or "").strip().lower()


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _sequence(value: object) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _first_mapping_item(values: Sequence[Any]) -> dict[str, Any] | None:
    for value in values:
        if isinstance(value, Mapping):
            return {str(key): item for key, item in value.items()}
    return None


def _item_id(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("id", "relic_id", "potion_id", "card_id", "name"):
            found = value.get(key)
            if found is not None:
                return str(found)
    if value is not None:
        return str(value)
    return None


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_json_object(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {str(key): _jsonable(item) for key, item in value.items()}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return str(value)


def _response_failed(response: Mapping[str, Any] | None) -> bool:
    if response is None:
        return False
    status = (_optional_string(response.get("status")) or "").lower()
    if status in {"error", "failed", "failure"}:
        return True
    return "error" in response and status != "ok"


def _is_auto_base_url(value: str) -> bool:
    return value.strip().lower() == AUTO_BASE_URL


def _write_result(
    result: LiveAgentResult,
    output_path: Path | str | None,
) -> LiveAgentResult:
    if output_path is None:
        return result
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump(mode="json")
    payload["output_path"] = str(target)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result.model_copy(update={"output_path": str(target)})


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_OUTPUT_PATH",
    "LiveAgentComparison",
    "LiveAgentDecision",
    "LiveAgentResult",
    "LiveAgentStep",
    "choose_live_action",
    "compare_live_state_to_simulator",
    "live_character_id",
    "live_state_summary",
    "play_live_agent",
]
