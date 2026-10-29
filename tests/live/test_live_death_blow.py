"""The two source DeathBlow intents are attacks followed by owner death."""
from __future__ import annotations

from copy import deepcopy

import pytest
from live.test_live_ppo_policy import _combat

from sts2sim import load_state, serialize, step
from sts2sim.agent_api import action_space, encode_observation
from sts2sim.engine import Action, RunPhase
from sts2sim.live_policy import LivePolicyUnsupported, import_policy_combat


def _state(identity, *, block=0, base=8, strength=0):
    raw = _combat()
    enemy = raw["battle"]["enemies"][0]
    ally = deepcopy(enemy)
    ally.update(entity_id="remaining_enemy", intents=[{"type": "Stun", "label": ""}])
    enemy.update(entity_id="exploding_enemy", monster_id=identity,
        hp=7 if identity == "GAS_BOMB" else 999999999,
        max_hp=7 if identity == "GAS_BOMB" else 999999999,
        status=[{"id": "STRENGTH_POWER", "amount": strength}] if strength else [],
        intents=[{"type": "DeathBlow", "label": str(base + strength),
                  "base_damage": base, "repeats": 1}])
    raw["battle"]["enemies"].append(ally)
    raw["player"]["block"] = block
    return raw, import_policy_combat(raw)


@pytest.mark.parametrize("identity,base,strength", [
    ("GAS_BOMB", 8, 0), ("WATERFALL_GIANT", 13, 2), ("WATERFALL_GIANT", 0, 0),
])
@pytest.mark.parametrize("block", [0, 99])
def test_public_live_death_blow_damages_then_kills_owner_once_even_when_fully_blocked(
    identity, base, strength, block,
):
    raw, state = _state(identity, block=block, base=base, strength=strength)
    monster = state.combat.monsters[0]
    assert monster.intent == "attack_death_blow"
    assert monster.intent_damage == base + strength
    assert monster.metadata["live_visible_intents"][0]["type"] == "DeathBlow"
    assert monster.metadata["live_self_destruct_move_id"] == "EXPLODE"
    # The actor recognizes the attack despite the distinct source display type.
    observation = encode_observation(state)
    assert observation["combat"]["incoming_damage"] == base + strength
    action = Action(type="end_turn")
    before = serialize(state)
    result = step(state, action)
    assert state.player.hp - result.player.hp == max(0, base + strength - block)
    dead = next(m for m in result.combat.monsters if m.monster_id == "exploding_enemy")
    assert dead.hp == 0 and dead.block == 0 and dead.intent is None
    assert dead.metadata["self_destructed"]
    if identity == "WATERFALL_GIANT":
        assert dead.metadata["waterfall_giant_exploded"]
        assert not dead.metadata.get("waterfall_giant_about_to_blow")
    events = result.combat.last_events
    damaged = next(i for i, e in enumerate(events)
                   if e.kind == "player_damaged" and e.source_id == "exploding_enemy")
    died = next(i for i, e in enumerate(events) if e.kind == "monster_self_destructed")
    assert damaged < died
    assert sum(e.kind == "monster_self_destructed" for e in events) == 1
    assert serialize(result) == serialize(step(load_state(before), action))
    assert serialize(state) == before
    assert raw["battle"]["enemies"][0]["intents"][0]["type"] == "DeathBlow"


def test_death_blow_with_unknown_monster_does_not_guess_a_self_destruct_body():
    raw, _ = _state("GAS_BOMB")
    raw["battle"]["enemies"][0]["monster_id"] = "UNREVIEWED_MODDED_MONSTER"
    with pytest.raises(LivePolicyUnsupported, match="known source monster identity"):
        import_policy_combat(raw)


@pytest.mark.parametrize("intent", ["DeathBlow", "Special"])
def test_native_gas_bomb_damage_reaches_potion_survival_strategy_and_actual_self_kill(intent):
    from combat.test_combat_engine_monsters import _enter_source_combat, _source_data

    data = _source_data(({
        "id": "GAS_BOMB", "name": "Gas Bomb", "type": "Normal", "min_hp": 7, "max_hp": 7,
        "moves": ({"id": "EXPLODE", "name": "Explode", "intent": intent,
                   "damage": {"normal": 8}},),
        "attack_pattern": {"initial_move": "EXPLODE_MOVE", "states": (
            {"id": "EXPLODE_MOVE", "type": "move", "move_id": "EXPLODE"},)},
    },), ({"id": "BOMB_ENCOUNTER", "name": "Bomb", "room_type": "Monster", "is_weak": True,
           "monsters": ({"id": "GAS_BOMB"},)},), "BOMB_ENCOUNTER")
    state = _enter_source_combat(data)
    player = state.player.model_copy(update={"hp": 8})
    state = state.model_copy(update={"player": player, "potions": ("block_potion",),
        "combat": state.combat.model_copy(update={"player": player})})
    assert state.combat.monsters[0].intent == intent
    assert encode_observation(state)["combat"]["incoming_damage"] == 8
    potion = next(row for row in action_space(state) if row["type"] == "use_potion")
    assert potion["potion_strategy"]["incoming_damage"] == 8
    assert potion["potion_strategy"]["survival_enabling"] is True
    result = step(state, Action(type="end_turn"))
    assert result.phase == RunPhase.FAILED and result.player.hp == 0
    assert result.combat.monsters[0].hp == 0
    assert result.combat.monsters[0].metadata["self_destructed"]
