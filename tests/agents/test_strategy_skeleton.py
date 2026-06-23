from __future__ import annotations

from typer.testing import CliRunner

from sts2sim import new_run, step
from sts2sim.agent_api import decode_action
from sts2sim.agents import StrategicAgent, analyze_run, play_strategic_run
from sts2sim.cli.app import app
from sts2sim.engine.models import CombatState, MonsterState, PlayerState, RunPhase


def test_analyze_run_builds_actionable_plan() -> None:
    state = new_run(seed=10, character_id="TEST", ascension=0)

    plan = analyze_run(state)

    assert plan.strategy in {"safe", "balanced", "greedy", "scaling"}
    assert plan.risk_level in {"low", "medium", "high", "critical"}
    assert plan.deck.total_cards > 0
    assert plan.economy.gold >= 0
    assert 0.0 <= plan.aggression.target <= 1.0
    assert 0.0 <= plan.aggression.hp_floor <= 1.0
    assert plan.notes


def test_analyze_run_reduces_aggression_when_hp_is_low() -> None:
    state = new_run(seed=14, character_id="TEST", ascension=0)
    player = state.player.model_copy(update={"hp": 18, "max_hp": 80})

    plan = analyze_run(state.model_copy(update={"player": player}))

    assert plan.risk_level in {"high", "critical"}
    assert plan.aggression.target < 0.35
    assert plan.aggression.hp_spend_budget == 0
    assert plan.aggression.allow_chip_damage is False


def test_analyze_run_rushes_visible_scaling_enemy() -> None:
    combat_player = PlayerState(hp=80, max_hp=80, energy=3)
    state = new_run(seed=15, character_id="TEST", ascension=0).model_copy(
        update={
            "phase": RunPhase.COMBAT,
            "player": combat_player,
            "combat": CombatState(
                player=combat_player,
                monsters=(
                    MonsterState(
                        monster_id="cultist",
                        hp=48,
                        max_hp=48,
                        intent="attack_buff",
                        move_id="grow_strength",
                        statuses={"ritual": 10, "strength": 5},
                        metadata={"move_powers": ({"power_id": "strength"},)},
                    ),
                ),
            ),
        }
    )

    plan = analyze_run(state)

    assert plan.aggression.scaling_pressure >= 0.6
    assert plan.aggression.combat_pace == "rush"
    assert "aggression_rush" in plan.notes


def test_strategic_agent_chooses_legal_action() -> None:
    state = new_run(seed=11, character_id="TEST", ascension=0)
    agent = StrategicAgent()

    decision = agent.choose_action(state)

    assert decision.chosen is not None
    assert decision.chosen.action_type == "choose_ancient"
    assert decision.candidates
    action = decode_action(state, decision.chosen.action_id)
    next_state = step(state, action)
    assert next_state.phase.value == "map"


def test_play_strategic_run_records_explainable_trace() -> None:
    result = play_strategic_run(
        seed=12,
        character_id="TEST",
        ascension=0,
        max_steps=3,
    )

    assert result["policy"] == "strategic_v0"
    assert result["steps_taken"] >= 1
    first_decision = result["decisions"][0]
    assert first_decision["chosen"]["action_type"] == "choose_ancient"
    assert first_decision["plan"]["notes"]
    assert first_decision["top_candidates"]


def test_play_strategic_run_cli_smoke() -> None:
    result = CliRunner().invoke(
        app,
        [
            "play-strategic-run",
            "--seed",
            "13",
            "--character",
            "TEST",
            "--max-steps",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert '"policy": "strategic_v0"' in result.stdout
