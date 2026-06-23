from __future__ import annotations

import gzip
import json
from pathlib import Path

import httpx
from typer.testing import CliRunner

from sts2sim.cli.app import app
from sts2sim.live_capture import LiveApiClient, LiveApiConfig, live_play, normalize_live_snapshot
from sts2sim.run_files import import_run_file, load_run_file


def test_import_run_file_summarizes_json_and_writes_trace(tmp_path: Path) -> None:
    run_path = tmp_path / "sample.run"
    trace_path = tmp_path / "sample.parity.json"
    run_path.write_text(
        json.dumps(
            {
                "seed_played": "ABC123",
                "character_chosen": "IRONCLAD",
                "ascension_level": 3,
                "victory": True,
                "floor_reached": 51,
                "current_hp": 44,
                "max_hp": 80,
                "gold": 123,
                "relics": ["burning_blood", "anchor"],
                "master_deck": ["Strike", "Defend"],
                "path_per_floor": ["M", "?", "R"],
            }
        ),
        encoding="utf-8",
    )

    result = import_run_file(run_path, trace_output_path=trace_path)

    assert result.summary.seed == "ABC123"
    assert result.summary.character_id == "IRONCLAD"
    assert result.summary.ascension == 3
    assert result.summary.victory is True
    assert result.summary.replayable is False
    assert result.trace is not None
    assert result.trace.simulator_replayable is False
    assert result.trace.final_state == {
        "floor": 51,
        "player": {"hp": 44, "max_hp": 80, "gold": 123},
        "relics": ["burning_blood", "anchor"],
        "master_deck_ids": ["Strike", "Defend"],
        "room_history": ["M", "?", "R"],
    }
    assert trace_path.exists()


def test_load_run_file_accepts_gzip_json(tmp_path: Path) -> None:
    run_path = tmp_path / "sample.run"
    run_path.write_bytes(gzip.compress(b'{"seed": 7, "character_id": "SILENT"}'))

    payload = load_run_file(run_path)

    assert payload == {"seed": 7, "character_id": "SILENT"}


def test_find_run_files_cli_lists_newest_files(tmp_path: Path) -> None:
    first = tmp_path / "first.run"
    second = tmp_path / "nested" / "second.run"
    second.parent.mkdir()
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")

    result = CliRunner().invoke(app, ["find-run-files", str(tmp_path), "--limit", "2"])
    payload = _json_output(result)

    assert payload["root"] == str(tmp_path)
    assert set(payload["files"]) == {str(first), str(second)}


def test_import_run_cli_emits_summary_and_trace_path(tmp_path: Path) -> None:
    run_path = tmp_path / "cli.run"
    trace_path = tmp_path / "cli.parity.json"
    run_path.write_text(
        json.dumps({"seed": 9, "character_id": "DEFECT", "floor_reached": 12}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        ["import-run", str(run_path), "--trace-output", str(trace_path)],
    )
    payload = _json_output(result)

    assert payload["summary"]["seed"] == 9
    assert payload["summary"]["character_id"] == "DEFECT"
    assert payload["trace"]["simulator_replayable"] is False
    assert payload["trace_output_path"] == str(trace_path)


def test_normalize_live_snapshot_extracts_common_state_fields() -> None:
    snapshot = normalize_live_snapshot(
        {
            "game_state": {
                "phase": "combat",
                "act": 1,
                "floor": 4,
                "player": {"current_hp": 70, "max_hp": 80, "block": 5, "gold": 99},
                "combat": {
                    "monsters": [
                        {
                            "id": "jaw_worm",
                            "current_hp": 32,
                            "max_hp": 40,
                            "intent": "attack",
                        }
                    ],
                    "hand": [{"id": "strike_ironclad"}],
                },
                "relics": ["burning_blood"],
            }
        }
    )

    assert snapshot["phase"] == "combat"
    assert snapshot["player"] == {"hp": 70, "max_hp": 80, "block": 5, "gold": 99}
    assert snapshot["combat"]["monsters"][0]["monster_id"] == "jaw_worm"
    assert snapshot["combat"]["hand"] == ["strike_ironclad"]


def test_normalize_live_snapshot_extracts_nested_event_and_player_inventory() -> None:
    snapshot = normalize_live_snapshot(
        {
            "state_type": "event",
            "event": {
                "event_id": "NEOW",
                "event_name": "Neow",
                "is_ancient": True,
                "options": [
                    {
                        "index": 0,
                        "title": "Proceed",
                        "is_locked": False,
                        "is_proceed": True,
                    }
                ],
            },
            "player": {
                "relics": [{"id": "BURNING_BLOOD"}, {"id": "NEOWS_TORMENT"}],
                "potions": [{"id": "FIRE_POTION"}],
            },
        }
    )

    assert snapshot["event"] == {
        "event_id": "NEOW",
        "event_name": "Neow",
        "is_ancient": True,
        "options": [
            {
                "index": 0,
                "title": "Proceed",
                "is_locked": False,
                "is_proceed": True,
            }
        ],
    }
    assert snapshot["relics"] == ["BURNING_BLOOD", "NEOWS_TORMENT"]
    assert snapshot["potions"] == ["FIRE_POTION"]


def test_live_play_records_external_actions_and_probabilities() -> None:
    state_index = {"value": 0}
    posted_actions: list[dict[str, object]] = []
    states = [
        {
            "state": {
                "seed": 42,
                "character_id": "IRONCLAD",
                "ascension": 0,
                "phase": "combat",
                "player": {"hp": 80, "max_hp": 80},
                "actions": [{"type": "attack"}, {"type": "end_turn"}],
            }
        },
        {
            "state": {
                "seed": 42,
                "character_id": "IRONCLAD",
                "ascension": 0,
                "phase": "reward",
                "player": {"hp": 78, "max_hp": 80},
                "actions": [],
            }
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/":
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "GET" and request.url.path == "/api/v1/state":
            return httpx.Response(200, json=states[state_index["value"]])
        if request.method == "POST" and request.url.path == "/api/v1/action":
            posted_actions.append(json.loads(request.content.decode("utf-8")))
            state_index["value"] = 1
            return httpx.Response(200, json={"accepted": True})
        return httpx.Response(404, json={"error": "missing"})

    http_client = httpx.Client(
        base_url="http://live.test",
        transport=httpx.MockTransport(handler),
    )
    client = LiveApiClient(LiveApiConfig(base_url="http://live.test"), client=http_client)

    result = live_play(max_steps=2, policy="prefer_attack", client=client)

    assert posted_actions == [{"action": {"type": "attack"}}]
    assert result.trace.simulator_replayable is False
    assert result.trace.source == "live"
    assert len(result.trace.steps) == 1
    assert result.trace.steps[0].external_action == {"type": "attack"}
    assert result.stats.steps_taken == 1
    assert result.stats.stopped_reason == "no_actions"
    assert result.stats.legal_action_counts == (2, 0)
    assert result.stats.selected_probabilities == (0.5,)
    assert result.stats.action_type_counts == {"attack": 1}


def test_live_play_maps_sts2mcp_menu_options_to_raw_action_payloads() -> None:
    posted_actions: list[dict[str, object]] = []
    states = [
        {
            "state_type": "menu",
            "menu_screen": "main",
            "options": ["singleplayer", "settings"],
        },
        {
            "state_type": "menu",
            "menu_screen": "character_select",
            "options": [],
        },
    ]
    state_index = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/":
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "GET" and request.url.path == "/api/v1/singleplayer":
            return httpx.Response(200, json=states[state_index["value"]])
        if request.method == "POST" and request.url.path == "/api/v1/singleplayer":
            posted_actions.append(json.loads(request.content.decode("utf-8")))
            state_index["value"] = 1
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"error": "missing"})

    http_client = httpx.Client(
        base_url="http://live.test",
        transport=httpx.MockTransport(handler),
    )
    client = LiveApiClient(LiveApiConfig(base_url="http://live.test"), client=http_client)

    result = live_play(max_steps=1, client=client)

    assert posted_actions == [{"action": "menu_select", "option": "singleplayer"}]
    assert result.trace.steps[0].external_action == {
        "action": "menu_select",
        "option": "singleplayer",
    }
    assert result.stats.action_type_counts == {"menu_select": 1}


def test_prefer_attack_policy_prefers_forward_sts2mcp_menu_options() -> None:
    posted_actions: list[dict[str, object]] = []
    states = [
        {
            "state_type": "menu",
            "menu_screen": "character_select",
            "options": [
                {"name": "IRONCLAD", "enabled": True},
                {"name": "embark", "enabled": True},
                {"name": "back", "enabled": True},
            ],
        },
        {"state_type": "event", "event": {"options": []}},
    ]
    state_index = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/":
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "GET" and request.url.path == "/api/v1/singleplayer":
            return httpx.Response(200, json=states[state_index["value"]])
        if request.method == "POST" and request.url.path == "/api/v1/singleplayer":
            posted_actions.append(json.loads(request.content.decode("utf-8")))
            state_index["value"] = 1
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"error": "missing"})

    http_client = httpx.Client(
        base_url="http://live.test",
        transport=httpx.MockTransport(handler),
    )
    client = LiveApiClient(LiveApiConfig(base_url="http://live.test"), client=http_client)

    result = live_play(max_steps=1, policy="prefer_attack", client=client)

    assert posted_actions == [{"action": "menu_select", "option": "embark"}]
    assert result.stats.action_type_counts == {"menu_select": 1}


def test_live_play_maps_sts2mcp_card_reward_to_card_index_payload() -> None:
    posted_actions: list[dict[str, object]] = []
    states = [
        {
            "state_type": "card_reward",
            "card_reward": {
                "cards": [{"index": 2, "id": "STRIKE_SILENT"}],
                "can_skip": True,
            },
        },
        {"state_type": "event", "event": {"options": []}},
    ]
    state_index = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/":
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "GET" and request.url.path == "/api/v1/singleplayer":
            return httpx.Response(200, json=states[state_index["value"]])
        if request.method == "POST" and request.url.path == "/api/v1/singleplayer":
            posted_actions.append(json.loads(request.content.decode("utf-8")))
            state_index["value"] = 1
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"error": "missing"})

    http_client = httpx.Client(
        base_url="http://live.test",
        transport=httpx.MockTransport(handler),
    )
    client = LiveApiClient(LiveApiConfig(base_url="http://live.test"), client=http_client)

    result = live_play(max_steps=1, client=client)

    assert posted_actions == [{"action": "select_card_reward", "card_index": 2}]
    assert result.stats.action_type_counts == {"select_card_reward": 1}


def test_live_play_maps_sts2mcp_hand_select_to_card_index_payload() -> None:
    posted_actions: list[dict[str, object]] = []
    states = [
        {
            "state_type": "hand_select",
            "hand_select": {
                "cards": [{"index": 3, "id": "DEFEND_SILENT"}],
                "can_confirm": False,
            },
        },
        {"state_type": "event", "event": {"options": []}},
    ]
    state_index = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/":
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "GET" and request.url.path == "/api/v1/singleplayer":
            return httpx.Response(200, json=states[state_index["value"]])
        if request.method == "POST" and request.url.path == "/api/v1/singleplayer":
            posted_actions.append(json.loads(request.content.decode("utf-8")))
            state_index["value"] = 1
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"error": "missing"})

    http_client = httpx.Client(
        base_url="http://live.test",
        transport=httpx.MockTransport(handler),
    )
    client = LiveApiClient(LiveApiConfig(base_url="http://live.test"), client=http_client)

    result = live_play(max_steps=1, client=client)

    assert posted_actions == [{"action": "combat_select_card", "card_index": 3}]
    assert result.stats.action_type_counts == {"combat_select_card": 1}


def test_live_play_prefers_confirm_on_sts2mcp_card_select_when_available() -> None:
    posted_actions: list[dict[str, object]] = []
    states = [
        {
            "state_type": "card_select",
            "card_select": {
                "screen_type": "NCombatPileCardSelectScreen",
                "prompt": "Choose up to 2 cards.",
                "cards": [{"index": 0, "id": "DEFEND_IRONCLAD"}],
                "can_confirm": True,
                "can_cancel": False,
            },
        },
        {"state_type": "monster", "battle": {"is_play_phase": True}, "player": {"hand": []}},
    ]
    state_index = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/":
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "GET" and request.url.path == "/api/v1/singleplayer":
            return httpx.Response(200, json=states[state_index["value"]])
        if request.method == "POST" and request.url.path == "/api/v1/singleplayer":
            posted_actions.append(json.loads(request.content.decode("utf-8")))
            state_index["value"] = 1
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"error": "missing"})

    http_client = httpx.Client(
        base_url="http://live.test",
        transport=httpx.MockTransport(handler),
    )
    client = LiveApiClient(LiveApiConfig(base_url="http://live.test"), client=http_client)

    result = live_play(max_steps=1, policy="prefer_attack", client=client)

    assert posted_actions == [{"action": "confirm_selection"}]
    assert result.stats.action_type_counts == {"confirm_selection": 1}


def _json_output(result):
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, dict)
    return payload
