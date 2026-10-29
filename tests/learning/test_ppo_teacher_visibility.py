from __future__ import annotations

from types import SimpleNamespace

import pytest
from learning.test_policy_visibility import _combat, _different_hidden_state

from sts2sim import action_space, decode_action, serialize
from sts2sim.agents import StrategicAgent
from sts2sim.agents.combat_search import CombatSearchConfig
from sts2sim.engine.rng import new_rng_state
from sts2sim.learning.masked_ppo import _teacher_action_index


class _DrawPeekingTeacher:
    def __init__(self) -> None:
        self.states = []

    def choose_action(self, state):
        self.states.append(state)
        preferred = "hit" if state.combat.draw_pile[0].type == "attack" else "draw"
        descriptor = next(
            descriptor
            for descriptor in action_space(state)
            if descriptor["card_instance_id"] == preferred
        )
        return SimpleNamespace(action_id=descriptor["id"])


def test_ppo_teacher_uses_a_visible_copy_with_stable_physical_action_ids() -> None:
    first = _combat()
    second = _different_hidden_state(first)
    before = [serialize(first), serialize(second)]
    teacher = _DrawPeekingTeacher()
    first_info = {"action_space": action_space(first)}
    second_info = {"action_space": action_space(second)}

    first_index = _teacher_action_index(first, first_info, teacher)
    second_index = _teacher_action_index(second, second_info, teacher)

    assert first_index == second_index
    assert first_index is not None
    for state, info, index in [
        (first, first_info, first_index),
        (second, second_info, second_index),
    ]:
        descriptor = info["action_space"][index]
        assert decode_action(state, descriptor["id"]).card_instance_id == "hit"
    assert all(state.seed == 0 and state.rng == new_rng_state(0) for state in teacher.states)
    assert teacher.states[0] is not first
    assert serialize(teacher.states[0]) == serialize(teacher.states[1])
    assert [serialize(first), serialize(second)] == before


def test_ppo_teacher_cannot_mutate_the_actual_run() -> None:
    state = _combat()
    before = serialize(state)

    class MutatingTeacher:
        def choose_action(self, copy):
            descriptor = action_space(copy)[0]
            copy.flags["teacher_modified"] = True
            copy.combat.hand[0].custom["teacher_modified"] = True
            return SimpleNamespace(action_id=descriptor["id"])

    index = _teacher_action_index(state, {"action_space": action_space(state)}, MutatingTeacher())
    assert index == 0
    assert serialize(state) == before


def test_ppo_teacher_preserves_frozen_eye_known_draw_order() -> None:
    first = _combat(frozen_eye=True)
    second = _different_hidden_state(first)
    teacher = _DrawPeekingTeacher()
    first_index = _teacher_action_index(first, {"action_space": action_space(first)}, teacher)
    second_index = _teacher_action_index(second, {"action_space": action_space(second)}, teacher)

    assert first_index is not None and second_index is not None
    assert first_index != second_index
    assert teacher.states[0].combat.draw_pile == first.combat.draw_pile
    assert teacher.states[1].combat.draw_pile == second.combat.draw_pile
    assert all(state.rng == new_rng_state(0) for state in teacher.states)


def test_real_strategic_teacher_is_invariant_to_hidden_order_and_rng() -> None:
    first = _combat()
    second = _different_hidden_state(first)
    teacher = StrategicAgent(
        combat_search_config=CombatSearchConfig(
            max_depth=2, max_turns=1, beam_width=2, branch_width=2
        )
    )
    first_index = _teacher_action_index(first, {"action_space": action_space(first)}, teacher)
    second_index = _teacher_action_index(second, {"action_space": action_space(second)}, teacher)

    assert first_index is not None
    assert first_index == second_index


@pytest.mark.parametrize("kind", ["missing", "invalid", "error"])
def test_teacher_failures_and_invalid_actions_keep_the_existing_fallback(kind: str) -> None:
    state = _combat()

    class InvalidTeacher:
        def choose_action(self, _state):
            if kind == "error":
                raise ValueError("teacher error")
            return SimpleNamespace(action_id=999) if kind == "invalid" else SimpleNamespace()

    assert (
        _teacher_action_index(state, {"action_space": action_space(state)}, InvalidTeacher())
        is None
    )
