from sts2sim import new_run
from sts2sim.agent_api import action_space, encode_observation
from sts2sim.engine import Action


def test_explicit_actions_preserve_canonical_ids_and_observation_counts():
    state = new_run("explicit-visible-actions", "IRONCLAD", 0)
    actions = [Action(type="proceed", target_id="ui:confirm"), Action(type="rest")]
    descriptors = action_space(state, actions=actions)
    reversed_descriptors = action_space(state, actions=list(reversed(actions)))
    assert descriptors == reversed_descriptors
    assert [entry["id"] for entry in descriptors] == [0, 1]
    assert {entry["action"]["type"] for entry in descriptors} == {"proceed", "rest"}
    observation = encode_observation(state, actions=actions)
    assert observation["counts"]["legal_actions"] == 2
    assert all(entry["preview"] == {"preview_available": 0} for entry in descriptors)
