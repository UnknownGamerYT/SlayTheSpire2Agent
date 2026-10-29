from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from sts2sim.agent_api import action_space
from sts2sim.cli.app import app
from sts2sim.engine import Action
from sts2sim.learning.masked_ppo import train_masked_ppo
from sts2sim.learning.policy import CheckpointPolicyError, MaskedPpoPolicy
from sts2sim.live_agent import play_live_agent
from sts2sim.live_capture import LiveApiClient, LiveApiConfig
from sts2sim.live_policy import (
    LivePolicyUnsupported,
    bridge_action_for_policy,
    import_policy_combat,
)


@pytest.fixture(scope="module")
def live_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("torch")
    checkpoint = tmp_path_factory.mktemp("live-policy") / "tiny.pt"
    train_masked_ppo(
        max_batches=1,
        train_runs_per_batch=1,
        train_max_steps=2,
        eval_runs=0,
        eval_max_steps=1,
        champion_eval_runs=1,
        hidden_size=8,
        hidden_layers=1,
        head_hidden_layers=1,
        ppo_epochs=1,
        minibatch_size=2,
        history_mode="off",
        model_output_path=checkpoint,
        output_path=None,
        progress_output_path=None,
        report_output_path=None,
        resume=False,
        device="cpu",
        seed="live-inference-fixture",
    )
    return checkpoint


def _combat() -> dict[str, Any]:
    hand = [
        {
            "index": 0,
            "id": "BASH",
            "name": "Bash",
            "type": "Attack",
            "cost": "2",
            "target_type": "AnyEnemy",
            "is_upgraded": False,
            "can_play": True,
        },
        {
            "index": 1,
            "id": "DEFEND_IRONCLAD",
            "name": "Defend",
            "type": "Skill",
            "cost": "1",
            "target_type": "Self",
            "is_upgraded": False,
            "can_play": True,
        },
    ]
    return {
        "state_type": "monster",
        "run": {"act": 1, "floor": 7, "ascension": 0},
        "battle": {
            "round": 1,
            "turn": "player",
            "is_play_phase": True,
            "enemies": [
                {
                    "entity_id": "NIBBIT_0",
                    "name": "Nibbit",
                    "hp": 45,
                    "max_hp": 45,
                    "block": 0,
                    "status": [],
                    "intents": [{"type": "Attack", "label": "12"}],
                },
            ],
        },
        "player": {
            "character": "The Ironclad",
            "hp": 80,
            "max_hp": 80,
            "block": 0,
            "energy": 3,
            "max_energy": 3,
            "gold": 99,
            "status": [],
            "relics": [{"id": "BURNING_BLOOD"}],
            "potions": [],
            "max_potion_slots": 3,
            "hand": hand,
            "deck": deepcopy(hand),
            "draw_pile": [],
            "draw_pile_count": 0,
            "discard_pile": [],
            "discard_pile_count": 0,
            "exhaust_pile": [],
            "exhaust_pile_count": 0,
        },
    }


class Bridge:
    """Exercise the real HTTP bridge client and live-agent loop with fixture states."""

    def __init__(self, before: dict[str, Any], after: dict[str, Any]) -> None:
        self.current = before
        self.after = after
        self.actions: list[dict[str, Any]] = []
        self.requests: list[str] = []
        self.http = httpx.Client(
            base_url="http://bridge", transport=httpx.MockTransport(self.handle)
        )
        self.client = LiveApiClient(
            LiveApiConfig(
                base_url="http://bridge",
                health_path="/health",
                state_path="/state",
                action_path="/action",
                action_envelope="raw",
            ),
            client=self.http,
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request.method)
        if request.method == "POST":
            self.actions.append(json.loads(request.content))
            self.current = self.after
            return httpx.Response(200, json={"success": True})
        if request.url.path == "/health":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(200, json=self.current)


def _play(bridge: Bridge, checkpoint: Path, **kwargs: Any):
    return play_live_agent(
        base_url="http://bridge",
        checkpoint_path=checkpoint,
        device="cpu",
        client=bridge.client,
        max_steps=kwargs.pop("max_steps", 1),
        start_if_needed=False,
        delay_seconds=0,
        settle_timeout_seconds=0,
        output_path=None,
        **kwargs,
    )


def test_trained_model_runs_through_real_http_client_loop(live_checkpoint: Path) -> None:
    before = _combat()
    policy = MaskedPpoPolicy(live_checkpoint, device="cpu")
    sim = import_policy_combat(before)
    decision = policy.choose_action(sim)
    expected = bridge_action_for_policy(before, sim, Action.model_validate(decision.action))
    assert expected is not None
    bridge = Bridge(before, {"state_type": "game_over"})
    result = _play(bridge, live_checkpoint)
    assert bridge.actions == [expected]
    assert result.steps_taken == 1
    assert result.policy == "masked_ppo"
    assert result.steps[0].decision.policy == "masked_ppo"
    assert result.steps[0].decision.policy_details["action_id"] == decision.action_id
    assert result.steps[0].decision.policy_details["simulator_action"] == decision.action
    assert result.steps[0].decision.policy_details["policy_output"] == decision.policy_output


def test_bridge_can_play_flags_mask_actions_before_model_selection(live_checkpoint: Path) -> None:
    state = _combat()
    for card in state["player"]["hand"]:
        card["can_play"] = False
    bridge = Bridge(state, {"state_type": "game_over"})
    result = _play(bridge, live_checkpoint)
    assert bridge.actions == [{"action": "end_turn"}]
    assert result.steps[0].decision.policy == "masked_ppo"


def test_live_zero_energy_and_status_values_are_preserved() -> None:
    state = _combat()
    state["player"]["energy"] = 0
    state["player"]["status"] = [{"id": "STRENGTH_POWER", "amount": 0}]
    state["battle"]["enemies"][0]["intents"][0]["label"] = "4x2"
    imported = import_policy_combat(state)
    assert imported.player.energy == imported.combat.player.energy == 0
    assert imported.player.statuses["strength"] == 0
    assert imported.combat.monsters[0].intent_damage == 8
    assert imported.combat.monsters[0].hit_count == 2
    assert imported.map is None
    assert len(imported.master_deck) == 2


@pytest.mark.parametrize(
    "damage",
    [
        "missing-deck",
        "missing-pile",
        "unknown-card",
        "bad-index",
        "enchantment",
        "missing-hp",
        "bad-intent",
        "advanced-character",
    ],
)
def test_sparse_or_unmapped_combat_stops_without_sending_actions(
    live_checkpoint: Path,
    damage: str,
) -> None:
    state = _combat()
    if damage == "missing-deck":
        state["player"].pop("deck")
    elif damage == "missing-pile":
        state["player"].pop("draw_pile")
    elif damage == "unknown-card":
        state["player"]["hand"][0]["id"] = "NOT_A_REAL_CARD"
        state["player"]["hand"][0]["name"] = "Not a real card"
    elif damage == "bad-index":
        state["player"]["hand"][0]["index"] = 7
    elif damage == "enchantment":
        state["player"]["hand"][0]["enchantments"] = [{"id": "SWIFT"}]
    elif damage == "missing-hp":
        state["player"].pop("hp")
    elif damage == "bad-intent":
        state["battle"]["enemies"][0]["intents"][0]["label"] = "unknown"
    else:
        state["player"]["character"] = "The Defect"
    bridge = Bridge(state, {"state_type": "game_over"})
    result = _play(bridge, live_checkpoint)
    assert bridge.actions == []
    assert result.stopped_reason == "checkpoint_observation_unsupported"
    assert result.stopped_decision.policy_details["diagnostic"]
    assert result.steps_taken == 0


def test_unsupported_screen_requires_explicit_and_labeled_heuristic_fallback(
    live_checkpoint: Path,
) -> None:
    state = {"state_type": "map", "map": {"next_options": [{"index": 2, "type": "Monster"}]}}
    stopped = Bridge(state, {"state_type": "game_over"})
    result = _play(stopped, live_checkpoint)
    assert stopped.actions == []
    assert result.stopped_reason == "checkpoint_observation_unsupported"
    fallback = Bridge(state, {"state_type": "game_over"})
    result = _play(fallback, live_checkpoint, heuristic_fallback=True)
    assert fallback.actions == [{"action": "choose_map_node", "index": 2}]
    assert result.steps[0].decision.policy == "heuristic_fallback"
    assert "unsupported_model_observation" in result.steps[0].decision.policy_details
    assert "policy_output" not in result.steps[0].decision.policy_details


def test_invalid_checkpoint_is_rejected_before_contacting_bridge(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bad.pt"
    checkpoint.write_bytes(b"not weights")
    bridge = Bridge(_combat(), {"state_type": "game_over"})
    with pytest.raises(CheckpointPolicyError):
        _play(bridge, checkpoint)
    assert bridge.requests == []


def test_bridge_mapper_rejects_wrong_target_and_false_can_play() -> None:
    state = _combat()
    sim = import_policy_combat(state)
    attack = next(
        Action.model_validate(d["action"])
        for d in action_space(sim)
        if d["type"] == "play_card" and d["action"].get("target_id") == "NIBBIT_0"
    )
    state["player"]["hand"][0]["valid_targets"] = ["SOMEONE_ELSE"]
    assert bridge_action_for_policy(state, sim, attack) is None
    state["player"]["hand"][0].pop("valid_targets")
    state["player"]["hand"][0]["can_play"] = False
    assert bridge_action_for_policy(state, sim, attack) is None


def test_live_cli_passes_checkpoint_flags_and_stops_with_diagnostic(
    live_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import sts2sim.live_agent as live_module

    bridge = Bridge({"state_type": "map", "map": {"next_options": [{"index": 1}]}}, {})
    original = live_module.play_live_agent

    def with_bridge(**kwargs: Any):
        return original(**kwargs, client=bridge.client)

    monkeypatch.setattr(live_module, "play_live_agent", with_bridge)
    report = tmp_path / "live.json"
    result = CliRunner().invoke(
        app,
        [
            "play-live-agent",
            "--checkpoint",
            str(live_checkpoint),
            "--device",
            "cpu",
            "--base-url",
            "http://bridge",
            "--max-steps",
            "1",
            "--no-start-if-needed",
            "--delay",
            "0",
            "--settle-timeout",
            "0",
            "--output",
            str(report),
        ],
    )
    assert result.exit_code == 1, result.output
    assert not bridge.actions
    payload = json.loads(report.read_text())
    assert payload["policy"] == "masked_ppo"
    assert payload["stopped_decision"]["policy_details"]["diagnostic"]


def test_unrecognized_state_import_fails_explicitly() -> None:
    with pytest.raises(LivePolicyUnsupported, match="combat only"):
        import_policy_combat({"state_type": "card_select"})


def test_multiple_live_model_steps_keep_observed_memory_and_stop_at_game_over(
    live_checkpoint: Path,
) -> None:
    from sts2sim.live_policy import LiveMaskedPpoPolicy

    state = _combat()
    policy = LiveMaskedPpoPolicy(MaskedPpoPolicy(live_checkpoint, device="cpu"))
    first, _ = policy.choose_action(state)
    after = deepcopy(state)
    after["player"]["hp"] = 75
    policy.record_transition(state, first, after)
    assert policy._memory[0]["hp_delta"] == -5
    assert "reward" not in policy._memory[0]
    second, action = policy.choose_action(after)
    assert second.action_id >= 0 and action["action"] in {"play_card", "end_turn"}
    policy.record_transition(after, second, {"state_type": "game_over", "player": {"hp": None}})
    assert len(policy._memory) == 2
    bridge = Bridge(state, {"state_type": "game_over"})
    result = _play(bridge, live_checkpoint, max_steps=2)
    assert result.steps_taken == 1
    assert result.stopped_reason == "game_over"


def test_unknown_live_id_cannot_fall_back_to_a_known_display_name() -> None:
    state = _combat()
    state["player"]["hand"][0]["id"] = "MODDED_BASH"
    with pytest.raises(LivePolicyUnsupported, match="Unknown live card ID"):
        import_policy_combat(state)


def test_prepared_bridge_schema_preserves_identity_and_blocks_unmapped_modifiers(
    live_checkpoint: Path,
) -> None:
    state = _combat()
    player = state["player"]
    player["character"] = "localized display label"
    player["character_id"] = "IRONCLAD"
    player["master_deck"] = player.pop("deck")
    for zone in ("hand", "master_deck"):
        for card in player[zone]:
            card["has_enchantment"] = False
            card["upgrade_level"] = 0
    player["hand"][0]["is_upgraded"] = True
    player["hand"][0]["upgrade_level"] = 1
    player["draw_pile"] = [deepcopy(player["master_deck"][0])]
    player["draw_pile_count"] = 1
    sim = import_policy_combat(state)
    assert sim.character_id == "IRONCLAD"
    assert sim.combat.hand[0].upgraded is True
    assert sim.combat.draw_pile[0].card_id.lower() == "bash"
    assert sim.combat.draw_pile[0].upgraded is False
    bridge = Bridge(state, {"state_type": "game_over"})
    result = _play(bridge, live_checkpoint)
    assert len(bridge.actions) == 1
    assert result.steps[0].decision.policy == "masked_ppo"
    for field, value in (("has_enchantment", True), ("upgrade_level", 2)):
        bad = deepcopy(state)
        bad["player"]["master_deck"][0][field] = value
        with pytest.raises(LivePolicyUnsupported, match="losslessly"):
            import_policy_combat(bad)


def test_real_recorded_public_bridge_payload_stops_without_inventing_pile_identity(
    live_checkpoint: Path,
) -> None:
    trace = json.loads(Path("traces/live_mcp_smoke_combat_settled.json").read_text())
    raw = trace["metadata"]["raw_initial_payload"]
    assert raw["player"]["draw_pile"]
    assert "is_upgraded" not in raw["player"]["draw_pile"][0]
    bridge = Bridge(raw, {"state_type": "game_over"})
    result = _play(bridge, live_checkpoint)
    assert not bridge.actions
    assert result.stopped_reason == "checkpoint_observation_unsupported"
    assert "upgrade state" in result.stopped_decision.policy_details["diagnostic"]
