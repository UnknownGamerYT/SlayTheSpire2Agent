from __future__ import annotations

import json

from typer.testing import CliRunner

from sts2sim import legal_actions, new_run, step
from sts2sim.agents import play_strategic_run
from sts2sim.cli.app import app
from sts2sim.engine import MapEdgeState, MapNodeState, MapState, RoomKind, RunPhase
from sts2sim.history import (
    append_history_step,
    record_history_step,
    run_history_html,
    run_history_map_text,
    start_run_history,
    write_run_history_html,
    write_run_history_map_text,
)
from sts2sim.learning import collect_random_rollouts


def test_history_records_readable_ancient_and_map_steps() -> None:
    state = new_run(seed=80, character_id="TEST", ascension=0)
    history = start_run_history(state, policy="test")

    ancient_action = next(
        action for action in legal_actions(state) if action.type == "choose_ancient"
    )
    next_state = step(state, ancient_action)
    ancient_step = record_history_step(
        step_index=0,
        before_state=state,
        action=ancient_action,
        after_state=next_state,
    )
    history = append_history_step(history, ancient_step, next_state)

    map_action = next(
        action for action in legal_actions(next_state) if action.type == "choose_node"
    )
    after_map = step(next_state, map_action)
    map_step = record_history_step(
        step_index=1,
        before_state=next_state,
        action=map_action,
        after_state=after_map,
    )
    history = append_history_step(history, map_step, after_map)

    assert "Choose ancient option" in history.steps[0].action_summary
    assert history.steps[0].events
    assert "Choose map node" in history.steps[1].action_summary
    assert history.steps[1].context_before["map"]["reachable"]
    assert history.summary["nodes_chosen"] == 1


def test_history_renders_html_and_text_map_for_chosen_path(tmp_path) -> None:
    state = new_run(seed=80, character_id="TEST", ascension=0)
    history = start_run_history(state, policy="test")
    ancient_action = next(
        action for action in legal_actions(state) if action.type == "choose_ancient"
    )
    after_ancient = step(state, ancient_action)
    history = append_history_step(
        history,
        record_history_step(
            step_index=0,
            before_state=state,
            action=ancient_action,
            after_state=after_ancient,
        ),
        after_ancient,
    )
    map_action = next(
        action for action in legal_actions(after_ancient) if action.type == "choose_node"
    )
    after_map = step(after_ancient, map_action)
    history = append_history_step(
        history,
        record_history_step(
            step_index=1,
            before_state=after_ancient,
            action=map_action,
            after_state=after_map,
        ),
        after_map,
    )

    html = run_history_html(history)
    text_map = run_history_map_text(history)
    html_path = tmp_path / "history.html"
    map_path = tmp_path / "history_map.txt"
    write_run_history_html(history, html_path)
    write_run_history_map_text(history, map_path)

    assert "Map Path" in html
    assert "Timeline" in html
    assert "Choose map node" in html
    assert f":{map_action.target_id}_" in text_map
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert "Legend:" in map_path.read_text(encoding="utf-8")


def test_history_records_combat_card_context_and_events() -> None:
    state = _enter_test_combat()
    play_action = next(action for action in legal_actions(state) if action.type == "play_card")
    next_state = step(state, play_action)

    history_step = record_history_step(
        step_index=0,
        before_state=state,
        action=play_action,
        after_state=next_state,
    )

    assert history_step.action_summary.startswith("Play ")
    assert history_step.context_before["combat"]["hand"]
    assert history_step.context_after["combat"]
    assert history_step.events


def test_strategic_run_returns_and_writes_history(tmp_path) -> None:
    history_path = tmp_path / "history.json"

    result = play_strategic_run(
        seed=81,
        character_id="TEST",
        ascension=0,
        max_steps=2,
        history_path=history_path,
    )

    assert result["history"]["steps"]
    assert history_path.exists()
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    assert payload["policy"] == "strategic_v0"
    assert payload["steps"][0]["action_summary"]


def test_strategic_run_cli_writes_history_output(tmp_path) -> None:
    history_path = tmp_path / "cli_history.json"

    result = CliRunner().invoke(
        app,
        [
            "play-strategic-run",
            "--seed",
            "84",
            "--character",
            "TEST",
            "--max-steps",
            "1",
            "--history-output",
            str(history_path),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(history_path.read_text(encoding="utf-8"))["steps"][0]["action_summary"]


def test_random_rollout_includes_readable_history_by_default() -> None:
    result = collect_random_rollouts(runs=1, max_steps=2, start_seed=82, character_id="TEST")

    history = result.runs[0].history
    assert history is not None
    assert history["summary"]["steps_taken"] == result.runs[0].steps_taken
    assert history["steps"][0]["action_summary"]


def _enter_test_combat():
    state = new_run(
        seed=83,
        character_id="TEST",
        ascension=0,
        source_data={
            "deck": (
                {
                    "id": "TEST_STRIKE",
                    "name": "Test Strike",
                    "type": "Attack",
                    "target": "AnyEnemy",
                    "cost": 1,
                    "damage": 6,
                },
                {
                    "id": "TEST_DEFEND",
                    "name": "Test Defend",
                    "type": "Skill",
                    "target": "Self",
                    "cost": 1,
                    "block": 5,
                },
            ),
            "flags": {"draw_per_turn": 2},
        },
    )
    start = MapNodeState(node_id="start", act=state.act, floor=0, lane=0, kind=RoomKind.START)
    target = MapNodeState(node_id="target", act=state.act, floor=1, lane=0, kind=RoomKind.MONSTER)
    game_map = MapState(
        act=state.act,
        nodes=(start, target),
        edges=(MapEdgeState(from_id=start.node_id, to_id=target.node_id),),
        current_node_id=start.node_id,
        completed_node_ids=(start.node_id,),
    )
    state = state.model_copy(update={"phase": RunPhase.MAP, "map": game_map, "floor": 0})
    return step(
        state,
        next(action for action in legal_actions(state) if action.type == "choose_node"),
    )
