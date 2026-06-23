from __future__ import annotations

import json

from typer.testing import CliRunner

from sts2sim import action_space, legal_actions, new_run, step
from sts2sim.cli.app import app
from sts2sim.engine import MapEdgeState, MapNodeState, MapState, RoomKind, RunPhase
from sts2sim.learning import (
    collect_random_rollouts,
    encode_rich_observation,
    evaluate_learning_agent,
    learning_reward,
    load_q_learning_model,
    train_q_learning,
)
from sts2sim.learning.features import state_action_key


def test_learning_reward_tracks_floor_progress_without_strategy_advice() -> None:
    state = new_run(seed=31, character_id="TEST", ascension=0)
    state = step(
        state,
        next(action for action in legal_actions(state) if action.type == "choose_ancient"),
    )
    next_state = step(
        state,
        next(action for action in legal_actions(state) if action.type == "choose_node"),
    )

    assert learning_reward(state, next_state) > 0


def test_collect_random_rollouts_records_legal_masked_steps(tmp_path) -> None:
    output_path = tmp_path / "random_rollouts.json"

    result = collect_random_rollouts(
        runs=2,
        max_steps=3,
        start_seed=40,
        character_id="TEST",
        output_path=output_path,
    )

    assert result.policy == "masked_random"
    assert result.runs_completed == 2
    assert result.total_steps > 0
    assert output_path.exists()
    assert result.runs[0].steps
    assert result.runs[0].steps[0].action_mask
    assert result.runs[0].steps[0].observation_mode == "rich"
    assert result.runs[0].steps[0].observation is not None
    assert result.runs[0].steps[0].action_id in {
        index for index, value in enumerate(result.runs[0].steps[0].action_mask) if value
    }


def test_rich_observation_includes_deck_map_paths_and_legal_actions() -> None:
    state = new_run(seed=44, character_id="TEST", ascension=0)

    observation = encode_rich_observation(state)

    assert observation["mode"] == "rich_v1"
    assert observation["compact"]["vector"]
    assert observation["aggression"]["target"] >= 0.0
    assert observation["card_zones"]["master_deck"]
    assert observation["map"]["paths"]
    assert observation["ancient"]["options"]
    assert observation["legal_actions"]["actions"]


def test_rich_observation_includes_combat_zones_and_monsters() -> None:
    state = _enter_test_combat()

    observation = encode_rich_observation(state)

    assert observation["combat"]["monsters"]
    assert observation["card_zones"]["hand"]
    assert "draw_pile" in observation["card_zones"]
    assert "discard_pile" in observation["card_zones"]
    assert observation["player"]["relics"] == list(state.relics)


def test_random_rollout_can_store_rich_observations(tmp_path) -> None:
    output_path = tmp_path / "rich_rollouts.json"

    result = collect_random_rollouts(
        runs=1,
        max_steps=2,
        start_seed=45,
        character_id="TEST",
        output_path=output_path,
        observation_mode="rich",
    )

    step_payload = result.runs[0].steps[0]
    assert step_payload.observation_mode == "rich"
    assert step_payload.observation is not None
    assert step_payload.observation["mode"] == "rich_v1"
    assert step_payload.observation["card_zones"]["master_deck"]


def test_random_rollout_can_still_store_compact_steps(tmp_path) -> None:
    output_path = tmp_path / "compact_rollouts.json"

    result = collect_random_rollouts(
        runs=1,
        max_steps=2,
        start_seed=47,
        character_id="TEST",
        output_path=output_path,
        observation_mode="compact",
    )

    step_payload = result.runs[0].steps[0]
    assert step_payload.observation_mode == "compact"
    assert step_payload.observation is None
    assert step_payload.vector
    assert step_payload.action_mask


def test_train_q_learning_writes_checkpoint_and_evaluates(tmp_path) -> None:
    model_path = tmp_path / "q_model.json"

    training = train_q_learning(
        runs=2,
        max_steps=4,
        seed=50,
        character_id="TEST",
        output_path=model_path,
    )
    model = load_q_learning_model(model_path)
    evaluation = evaluate_learning_agent(
        policy="q_learning",
        model_path=model_path,
        runs=1,
        max_steps=3,
        character_id="TEST",
    )

    assert training.algorithm == "state_action_signature_q_learning"
    assert training.total_steps > 0
    assert model.q_values
    assert evaluation.runs_completed == 1


def test_q_learning_action_key_distinguishes_card_context() -> None:
    state = _enter_test_combat()
    observation = {
        "phase": "combat",
        "vector_schema": ("act", "floor", "player_hp", "player_max_hp"),
        "vector": (1, 1, 80, 80),
    }
    descriptors = [
        descriptor for descriptor in action_space(state) if descriptor["type"] == "play_card"
    ]

    assert len({state_action_key(observation, descriptor) for descriptor in descriptors}) == 2


def test_learning_cli_smoke(tmp_path) -> None:
    runner = CliRunner()
    rollout_path = tmp_path / "rollout.json"
    model_path = tmp_path / "model.json"

    rollout = runner.invoke(
        app,
        [
            "rollout-random",
            "--runs",
            "1",
            "--max-steps",
            "2",
            "--character",
            "TEST",
            "--output",
            str(rollout_path),
        ],
    )
    train = runner.invoke(
        app,
        [
            "train-learning-agent",
            "--runs",
            "1",
            "--max-steps",
            "2",
            "--character",
            "TEST",
            "--output",
            str(model_path),
        ],
    )

    assert rollout.exit_code == 0
    assert train.exit_code == 0
    assert json.loads(rollout_path.read_text(encoding="utf-8"))["policy"] == "masked_random"
    assert json.loads(model_path.read_text(encoding="utf-8"))["algorithm"] == (
        "state_action_signature_q_learning"
    )


def _enter_test_combat():
    state = new_run(
        seed=46,
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
