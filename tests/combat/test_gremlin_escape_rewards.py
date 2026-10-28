from __future__ import annotations

import pytest
from test_named_monster_scripts import _combat
from test_remaining_monster_scripts import _end, _play, _with_child_definitions

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import RunPhase


def _merc_fight():
    state = _combat("GREMLIN_MERC", statuses=(("SURPRISE", 1),))
    state = _with_child_definitions(state, ("FAT_GREMLIN", "SNEAKY_GREMLIN"))
    monsters = []
    for row in state.flags["monsters"]:
        row = dict(row)
        if row["id"] == "FAT_GREMLIN":
            row["moves"] = ({"id": "SPAWNED", "intent": "Stun"},
                            {"id": "FLEE", "intent": "Escape"})
            row["attack_pattern"] = {
                "initial_move": "SPAWNED",
                "states": (
                    {"id": "SPAWNED", "move_id": "SPAWNED", "next": "FLEE"},
                    {"id": "FLEE", "move_id": "FLEE", "next": "FLEE"},
                ),
            }
        monsters.append(row)
    player = state.combat.player.model_copy(update={"gold": 60})
    return state.model_copy(update={
        "player": player,
        "flags": {**state.flags, "monsters": tuple(monsters), "combat_reward_gold": 30},
        "combat": state.combat.model_copy(update={"player": player}),
    })


@pytest.mark.parametrize("stole_gold,expected_gold", [(False, 15), (True, 0)])
def test_fat_gremlin_escape_changes_only_combat_gold_reward(stole_gold, expected_gold) -> None:
    state = _merc_fight()
    if stole_gold:
        state = _end(state)
        assert state.combat.player.gold == 40
    state = _play(state, effects={"damage": 999})
    state = _end(state)  # Spawned consumes the Fat Gremlin's first own turn.
    state = load_state(serialize(state))
    state = _end(state)
    fat = next(m for m in state.combat.monsters
               if m.metadata.get("source_monster_id") == "FAT_GREMLIN")
    assert fat.metadata["escaped"] is True
    sneaky = next(m for m in state.combat.monsters if m.hp > 0)
    state = _play(state, effects={"damage": 999}, target=sneaky.monster_id)
    assert state.phase is RunPhase.REWARD
    assert state.reward.gold == expected_gold
    assert state.reward.metadata["fat_gremlin_escape_gold_percent"] == (0 if stole_gold else 50)
    assert state.reward.metadata["gold_before_fat_gremlin_escape"] == 30
    assert state.player.gold == (40 if stole_gold else 60)
    if expected_gold:
        state = step(state, next(a for a in legal_actions(state) if a.type == "take_reward_gold"))
        assert state.player.gold == 60 + expected_gold


def test_killing_fat_gremlin_returns_stolen_gold_and_preserves_normal_reward() -> None:
    state = _end(_merc_fight())
    state = _play(state, effects={"damage": 999})
    fat = next(m for m in state.combat.monsters
               if m.metadata.get("source_monster_id") == "FAT_GREMLIN")
    state = _play(state, effects={"damage": 999}, target=fat.monster_id)
    assert state.combat.player.gold == 60
    sneaky = next(m for m in state.combat.monsters if m.hp > 0)
    state = _play(state, effects={"damage": 999}, target=sneaky.monster_id)
    assert state.reward.gold == 30
    assert "fat_gremlin_escape_gold_percent" not in state.reward.metadata


def test_unrelated_escaping_monster_preserves_normal_combat_gold_reward() -> None:
    state = _combat("OTHER_FUGITIVE", moves=({"id": "FLEE", "intent": "Escape"},))
    state = state.model_copy(update={"flags": {**state.flags, "combat_reward_gold": 30}})
    state = _end(state)
    assert state.phase is RunPhase.REWARD
    assert state.reward.gold == 30
    assert "fat_gremlin_escape_gold_percent" not in state.reward.metadata
