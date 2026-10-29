from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from sts2sim import serialize
from sts2sim.cli.app import app
from sts2sim.game_installation import STABLE_BUILD
from sts2sim.live_agent import choose_live_action
from sts2sim.live_capture import LiveApiClient, LiveApiConfig
from sts2sim.live_comparison import (
    PLAYER_STREAMS,
    RUN_STREAMS,
    ComparisonField,
    LiveComparisonCase,
    capture_live_comparison,
    compare_live_comparison,
)
from sts2sim.live_parity import live_state_to_simulator_state


def _case() -> LiveComparisonCase:
    card = {"id": "BASH", "name": "Bash", "type": "Attack", "cost": "2",
            "local_cost": 2, "upgrade_level": 0, "is_upgraded": False,
            "has_enchantment": False, "target_type": "AnyEnemy", "can_play": True,
            "index": 0}
    before: dict[str, Any] = {
        "observation_schema_version": 2, "state_type": "monster",
        "run": {"act": 1, "floor": 1, "ascension": 0},
        "battle": {"round": 1, "turn": "player", "is_play_phase": True,
                   "enemies": [{"entity_id": "NIBBIT_0", "name": "Nibbit", "hp": 45,
                                "max_hp": 45, "block": 0, "status": [],
                                "intents": [{"type": "Attack", "label": "12"}]}]},
        "player": {"character": "The Ironclad", "hp": 80, "max_hp": 80, "block": 0,
                   "energy": 3, "max_energy": 3, "gold": 99, "status": [],
                   "relics": [], "potions": [], "hand": [card], "draw_pile": [],
                   "discard_pile": [], "exhaust_pile": [], "draw_pile_count": 0,
                   "discard_pile_count": 0, "exhaust_pile_count": 0},
    }
    sim = live_state_to_simulator_state(before, seed=0)
    assert sim.combat
    after = deepcopy(before)
    after["battle"]["enemies"][0].update({
        "hp": 37, "status": [{"id": "VULNERABLE_POWER", "amount": 2}],
    })
    after["player"].update({"energy": 1, "hand": [], "discard_pile": [card],
                            "discard_pile_count": 1})
    return LiveComparisonCase(
        case_id="source-bash-fixture", simulator_before=serialize(sim),
        simulator_action={"type": "play_card", "card_instance_id": sim.combat.hand[0].instance_id,
                          "target_id": "NIBBIT_0"},
        external_action={"action": "play_card", "card_index": 0, "target": "NIBBIT_0"},
        live_before=before, live_after=after, compare_rng=False,
    )


def test_recorded_case_replays_explicit_checkpoint_and_detects_item_card_mismatch() -> None:
    case = _case()
    report = compare_live_comparison(case)
    assert report["matched"] is True
    assert report["origin"] == "fixture"
    assert report["game_parity_certified"] is False
    after = deepcopy(case.live_after)
    after["player"]["discard_pile"][0]["enchantment"] = {"id": "SHARP", "amount": 2}
    bad = compare_live_comparison(case.model_copy(update={"live_after": after}))
    assert bad["matched"] is False
    assert any("enchantment" in row["path"] for row in bad["mismatches"])


def test_missing_rng_evidence_is_unsupported_even_when_visible_outcome_matches() -> None:
    result = compare_live_comparison(_case().model_copy(update={"compare_rng": True}))
    assert result["supported"] is False
    assert result["matched"] is False
    assert any("schema1" in reason for reason in result["unsupported_reasons"])


def test_positive_regent_stars_use_canonical_resource_and_detect_a_real_difference() -> None:
    case = _case()
    sim = deepcopy(case.simulator_before)
    sim["character_id"] = "REGENT"
    sim["player"]["resources"] = {"star": 4}
    sim["combat"]["player"]["resources"] = {"star": 4}
    before, after = deepcopy(case.live_before), deepcopy(case.live_after)
    for payload in (before, after):
        payload["player"].update({"character": "The Regent", "character_id": "REGENT", "stars": 4})
    regent = case.model_copy(update={
        "simulator_before": sim, "live_before": before, "live_after": after,
    })
    assert compare_live_comparison(regent)["matched"] is True
    after["player"]["stars"] = 5
    mismatched = compare_live_comparison(regent.model_copy(update={"live_after": after}))
    assert mismatched["matched"] is False
    assert any(row["stage"] == "after" and "stars" in row["path"]
               for row in mismatched["mismatches"])


def test_comparison_distinguishes_physical_potion_holes_from_packed_inventory() -> None:
    case = _case()
    sim = deepcopy(case.simulator_before)
    sim["potions"] = ["fire_potion"]
    sim["flags"]["potion_slot_indices"] = [2]
    before, after = deepcopy(case.live_before), deepcopy(case.live_after)
    for payload in (before, after):
        payload["player"]["potions"] = [{"slot": 2, "id": "FIRE_POTION"}]
    physical = case.model_copy(update={
        "simulator_before": sim, "live_before": before, "live_after": after,
    })
    assert compare_live_comparison(physical)["matched"] is True
    sim["flags"]["potion_slot_indices"] = [0]
    wrong_slot = compare_live_comparison(physical.model_copy(update={"simulator_before": sim}))
    assert wrong_slot["matched"] is False
    assert any("potion_inventory" in row["path"] for row in wrong_slot["mismatches"])
    before["player"]["potions"] = [{"id": "FIRE_POTION"}]
    absent_slot = compare_live_comparison(physical.model_copy(update={"live_before": before}))
    assert absent_slot["supported"] is False
    assert any("physical potion slots" in reason for reason in absent_slot["unsupported_reasons"])


def _rng_case() -> LiveComparisonCase:
    case = _case()
    rng = case.simulator_before["rng"]
    streams = rng["streams"]
    validation = {
        "schema_version": 1,
        "run_rng": {"seed": rng["seed"], "counters": {
            name: streams.get(name, {}).get("draws", 0) for name in RUN_STREAMS}},
        "players_rng": [{"slot": 0, "rng": {"seed": 372029326,
            "counters": {name: streams.get(name, {}).get("draws", 0)
                         for name in PLAYER_STREAMS}}}],
        "event_rngs": {},
    }
    # The C# deterministic hash of the displayed string '0'.
    from sts2sim.engine.rng import stable_seed_hash
    validation["players_rng"][0]["rng"]["seed"] = stable_seed_hash("0")
    before, after = deepcopy(case.live_before), deepcopy(case.live_after)
    before["validation"] = deepcopy(validation)
    after["validation"] = deepcopy(validation)
    return case.model_copy(update={"live_before": before, "live_after": after, "compare_rng": True})


def test_native_persistent_rng_positions_are_compared_instead_of_ignored() -> None:
    case = _rng_case()
    assert compare_live_comparison(case)["matched"] is True
    after = deepcopy(case.live_after)
    after["validation"]["players_rng"][0]["rng"]["counters"]["rewards"] += 1
    bad = compare_live_comparison(case.model_copy(update={"live_after": after}))
    assert bad["matched"] is False
    assert any(row.get("scope") == "rng" and row["path"] == "rewards"
               for row in bad["mismatches"])
    del after["validation"]["run_rng"]["counters"]["niche"]
    missing = compare_live_comparison(case.model_copy(update={"live_after": after}))
    assert not missing["supported"]
    assert any("niche" in reason for reason in missing["unsupported_reasons"])


def test_explicit_dynamic_var_paths_compare_actual_fields_and_reject_missing_paths() -> None:
    case = _case().model_copy(update={"fields": (ComparisonField(
        live_path="battle.enemies[0].hp", simulator_path="combat.monsters[0].hp"),)})
    assert compare_live_comparison(case)["matched"]
    bad = case.model_copy(update={"fields": (ComparisonField(
        live_path="battle.enemies[0].missing", simulator_path="combat.monsters[0].hp"),)})
    assert not compare_live_comparison(bad)["supported"]


class _Client:
    def __init__(self, before: dict[str, Any], after: dict[str, Any]):
        self.before, self.after = before, after
        self.calls = 0
        self.actions: list[dict[str, Any]] = []

    def health(self) -> dict[str, Any]:
        return {"observation_schema_version": 2, "observer_actions_are_explicit": True}

    def state(self) -> dict[str, Any]:
        self.calls += 1
        return self.before if not self.actions else self.after

    def apply_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.actions.append(action)
        return {"status": "ok"}


def _plan(case: LiveComparisonCase) -> dict[str, Any]:
    result = case.model_dump(mode="json", exclude={"live_before", "live_after"})
    result["installation"] = {"matches_target_metadata": True, "target": STABLE_BUILD}
    return result


def test_capture_records_one_real_action_only_after_starting_checkpoint_matches() -> None:
    case = _rng_case()
    client = _Client(case.live_before, case.live_after)
    capture = capture_live_comparison(_plan(case), execute=True, client=client)
    assert capture["report"]["matched"]
    assert capture["case"]["origin"] == "installed_game"
    assert len(client.actions) == 1
    assert client.actions[0] == case.external_action
    wrong_before = deepcopy(case.live_before)
    wrong_before["player"]["hp"] -= 1
    blocked = _Client(wrong_before, case.live_after)
    with pytest.raises(ValueError, match="before-state"):
        capture_live_comparison(_plan(case), execute=True, client=blocked)
    assert blocked.actions == []
    untouched = _Client(case.live_before, case.live_after)
    with pytest.raises(ValueError, match="explicit --execute"):
        capture_live_comparison(_plan(case), client=untouched)
    assert untouched.calls == 0


def test_comparison_cli_replays_captured_envelope_and_fails_on_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "case.json"
    case = _case()
    path.write_text(json.dumps({"case": case.model_dump(mode="json"), "report": {}}))
    result = CliRunner().invoke(app, ["compare-live-comparison", str(path)])
    assert result.exit_code == 0, result.output
    after = deepcopy(case.live_after)
    after["player"]["hp"] -= 1
    path.write_text(case.model_copy(update={"live_after": after}).model_dump_json())
    result = CliRunner().invoke(app, ["compare-live-comparison", str(path)])
    assert result.exit_code == 1
    assert json.loads(result.output)["matched"] is False


def test_closed_chest_is_an_explicit_bridge_action_and_get_does_not_send_it() -> None:
    state = {"state_type": "treasure", "treasure": {
        "can_open_chest": True, "can_proceed": True, "relics": []}}
    requests = []
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=state if request.method == "GET" else {"status": "ok"})
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://bridge") as http:
        client = LiveApiClient(LiveApiConfig(base_url="http://bridge"), client=http)
        assert client.state() == client.state()
        assert all(request.method == "GET" for request in requests)
        assert client.actions(state) == [{"action": "open_chest"}, {"action": "proceed"}]
        action = choose_live_action(state).action
        assert action == {"action": "open_chest"}
        client.apply_action(action)
    assert json.loads(requests[-1].content) == {"action": "open_chest"}


def test_bridge_actions_keep_alternatives_distinct_and_ignore_chosen_event_controls() -> None:
    state = {"state_type": "card_reward", "card_reward": {
        "cards": [], "can_skip": False, "alternatives": [
            {"index": 0, "option_id": "SACRIFICE", "is_enabled": True},
            {"index": 1, "option_id": "REROLL", "is_enabled": False},
        ]}}
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={})), base_url="http://bridge") as http:
        client = LiveApiClient(LiveApiConfig(base_url="http://bridge"), client=http)
        assert client.actions(state) == [{"action": "select_card_reward_alternative", "index": 0}]
        with_skip = deepcopy(state)
        with_skip["card_reward"]["can_skip"] = True
        with_skip["card_reward"]["alternatives"].insert(0, {
            "index": 2, "option_id": "Skip", "is_enabled": True})
        assert client.actions(with_skip) == [
            {"action": "select_card_reward_alternative", "index": 0},
            {"action": "skip_card_reward"},
        ]
        event = {"state_type": "event", "event": {"options": [
            {"index": 0, "is_locked": False, "is_enabled": True, "was_chosen": True},
            {"index": 1, "is_locked": False, "is_enabled": False},
            {"index": 2, "is_locked": False, "is_enabled": True},
        ]}}
        assert client.actions(event) == [{"action": "choose_event_option", "index": 2}]


def test_capture_waits_for_source_effect_and_enemy_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    busy = deepcopy(case.live_after)
    busy["run"]["action_running"] = True
    busy["battle"]["is_executing_effect"] = True
    snapshots = iter([busy, case.live_after])
    monkeypatch.setattr("sts2sim.live_comparison._wait_for_live_state_update",
                        lambda *args, **kwargs: next(snapshots))
    client = _Client(case.live_before, case.live_after)
    result = capture_live_comparison(_plan(case), execute=True, client=client)
    assert result["report"]["live_after_settled"] is True
    assert result["report"]["matched"] is True
    assert len(client.actions) == 1
    before = deepcopy(case.live_before)
    before["battle"]["turn"] = "enemy"
    blocked = _Client(before, case.live_after)
    with pytest.raises(ValueError, match="resolving"):
        capture_live_comparison(_plan(case), execute=True, client=blocked)
    assert blocked.actions == []


def test_capture_refuses_an_old_observer_before_any_state_request() -> None:
    case = _case()
    client = _Client(case.live_before, case.live_after)
    client.health = lambda: {"status": "ok"}
    with pytest.raises(ValueError, match="health does not confirm read-only"):
        capture_live_comparison(_plan(case), execute=True, client=client)
    assert client.calls == 0
    assert client.actions == []


def test_numeric_bindings_compare_decimal_values_to_exact_serialized_fractions() -> None:
    case = _case()
    state = deepcopy(case.simulator_before)
    state["flags"]["fraction_probe"] = "3/2"
    before, after = deepcopy(case.live_before), deepcopy(case.live_after)
    before["numeric_probe"] = after["numeric_probe"] = 1.5
    updated = case.model_copy(update={"simulator_before": state,
        "live_before": before, "live_after": after, "fields": (ComparisonField(
            live_path="numeric_probe", simulator_path="flags.fraction_probe", numeric=True),)})
    assert compare_live_comparison(updated)["matched"]
    after["numeric_probe"] = 1.500001
    assert not compare_live_comparison(updated.model_copy(update={"live_after": after}))["matched"]


def test_plain_capture_refuses_identified_legacy_sts2mcp_without_requesting_state() -> None:
    from sts2sim.live_capture import LiveApiError, capture_live_state

    calls = []
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/":
            return httpx.Response(200, json={"message": "Hello from STS2 MCP"})
        raise AssertionError("Legacy game GET must not be requested")
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://bridge") as http:
        client = LiveApiClient(LiveApiConfig(base_url="http://bridge"), client=http)
        with pytest.raises(LiveApiError, match="Read-only observation schema2"):
            capture_live_state(client=client)
    assert calls == ["/"]
