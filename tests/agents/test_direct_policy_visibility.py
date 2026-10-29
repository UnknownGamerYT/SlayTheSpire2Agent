"""Direct strategic use has the same visibility boundary as PPO teacher use."""
from __future__ import annotations

from learning.test_policy_visibility import _combat, _different_hidden_state

from sts2sim import serialize
from sts2sim.agents import StrategicAgent
from sts2sim.agents.combat_search import CombatSearchConfig
from sts2sim.policy_visibility import policy_state


def test_direct_strategic_planning_and_choice_ignore_hidden_only_state_changes(monkeypatch):
    first = _combat().model_copy(update={'flags': {
        'player_relic_bag': {'Common': ['hidden_relic_a']},
        'act_room_sets': {'1': {'events': ['hidden_event_a']}},
    }})
    second = _different_hidden_state(first).model_copy(update={'flags': {
        'player_relic_bag': {'Common': ['hidden_relic_b']},
        'act_room_sets': {'1': {'events': ['hidden_event_b']}},
    }})
    originals = (serialize(first), serialize(second))
    agent = StrategicAgent(combat_search_config=CombatSearchConfig(
        max_depth=2, max_turns=1, beam_width=4, branch_width=3))
    planner_inputs = []
    original_score = agent._combat.score_all

    def record_input(context):
        planner_inputs.append(context.state_summary)
        return original_score(context)

    monkeypatch.setattr(agent._combat, 'score_all', record_input)
    direct_first = agent.choose_action(first)
    direct_second = agent.choose_action(second)
    teacher_style = agent.choose_action(policy_state(first))
    assert direct_first.chosen is not None
    assert direct_first == direct_second == teacher_style
    assert planner_inputs[0] == planner_inputs[1] == planner_inputs[2]
    assert planner_inputs[0] == serialize(policy_state(first))
    assert 'player_relic_bag' not in planner_inputs[0]['flags']
    assert (serialize(first), serialize(second)) == originals
