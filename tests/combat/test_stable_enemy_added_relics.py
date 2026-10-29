"""PhilosophersStone's source creature-added hook and missing spawn diagnostics."""

from __future__ import annotations

import pytest
from combat.test_stable_enemy_lifecycle import _end, _fight, _play, _weaken
from combat.test_stable_osty_lifecycle import _start

from sts2sim import load_state, serialize
from sts2sim.engine import EngineError
from sts2sim.engine.transitions import _apply_enemy_added_relics


def _stone(state):
    return state.model_copy(update={"relics": ("philosophers_stone",)})


def test_philosophers_stone_applies_once_to_opening_room_enemies():
    state = _start(relics=["philosophers_stone"])
    assert state.combat.monsters[0].statuses["strength"] == 1


def test_fabricator_additions_receive_one_strength_and_updated_intents():
    state = _end(_stone(_fight(("FABRICATOR",))))
    children = [m for m in state.combat.monsters if m.metadata.get("summoned")]
    assert len(children) == 2
    for child in children:
        assert child.statuses["strength"] == 1
        assert child.metadata["philosophers_stone_added"] is True
        if child.intent_damage:
            assert child.intent_damage == child.metadata["base_intent_damage"] + child.hit_count
    assert (
        sum(
            e.source_id == "philosophers_stone" and e.kind == "status_applied"
            for e in state.combat.last_events
        )
        == 2
    )


@pytest.mark.parametrize("parent,child_count", [("PHROG_PARASITE", 4), ("GREMLIN_MERC", 2)])
def test_death_additions_receive_stone_and_replay_identically(parent, child_count):
    state = _stone(_weaken(_fight((parent,))))
    restored = load_state(serialize(state))
    state = _play(state)
    assert serialize(state) == serialize(_play(restored))
    children = [m for m in state.combat.monsters if m.hp > 0]
    assert len(children) == child_count
    assert all(m.statuses["strength"] == 1 for m in children)


def test_stock_creates_a_new_stone_buffed_creature_each_wave():
    state = _stone(_fight(effects={"damage": 10000}))
    state = _play(state)
    child = next(m for m in state.combat.monsters if m.hp > 0)
    assert child.statuses["strength"] == 1
    state = _end(state)
    child = next(m for m in state.combat.monsters if m.hp > 0)
    assert child.statuses["strength"] == 4  # Added1 + first BootUp3.
    state = _play(state)
    child = next(m for m in state.combat.monsters if m.hp > 0)
    assert child.statuses["strength"] == 1


def test_same_creature_hatch_clears_stone_strength_without_reapplying_it():
    state = _stone(_fight(("TOUGH_EGG",)))
    egg, _ = _apply_enemy_added_relics(state.combat.monsters[0], state)
    assert egg.statuses["strength"] == 1
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (egg,)})}
    )
    state = _end(state)
    assert state.combat.monsters[0].name == "Hatchling"
    assert "strength" not in state.combat.monsters[0].statuses


def test_missing_summon_definition_stops_the_transition_instead_of_omitting_enemies():
    state = _fight(("FABRICATOR",))
    state = state.model_copy(
        update={
            "flags": {
                **state.flags,
                "monsters": [
                    row
                    for row in state.flags["monsters"]
                    if row["id"] not in {"GUARDBOT", "NOISEBOT", "ZAPBOT", "STABBOT"}
                ],
            }
        }
    )
    with pytest.raises(EngineError, match="requires summoned monster definition"):
        _end(state)


def test_missing_death_addition_definition_stops_the_transition():
    state = _weaken(_fight(("PHROG_PARASITE",)))
    state = state.model_copy(
        update={
            "flags": {
                **state.flags,
                "monsters": [row for row in state.flags["monsters"] if row["id"] != "WRIGGLER"],
            }
        }
    )
    with pytest.raises(EngineError, match="WRIGGLER"):
        _play(state)
