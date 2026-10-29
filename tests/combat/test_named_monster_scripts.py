from __future__ import annotations

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import MapEdgeState, MapNodeState, MapState, RoomKind, RunPhase
from sts2sim.engine.transitions import (
    _apply_aeonglass_increasing_intensity,
    _apply_after_card_play_hooks,
    _apply_end_turn_hand_card_effects,
    _card_from_spec,
    _damage_combat_player,
)


def _combat(monster_id: str, *, statuses=(), moves=(), ascension=0):
    moves = moves or ({"id": "ATTACK", "damage": {"normal": 5, "hit_count": 3}},)
    monster = {
        "id": monster_id,
        "name": monster_id,
        "type": "Normal",
        "min_hp": 100,
        "max_hp": 100,
        "innate_powers": tuple({"power_id": key, "amount": value} for key, value in statuses),
        "moves": moves,
        "attack_pattern": {"initial_move": moves[0]["id"]},
    }
    state = new_run(seed=500, character_id="TEST", ascension=ascension, source_data={
        "monsters": (monster,),
        "encounters": ({"id": "TEST", "monsters": ({"id": monster_id},)},),
        "combat_encounter_id": "TEST",
        "deck": tuple({
            "card_id": f"skill_{i}", "type": "skill", "cost": 0,
            "target": "self", "effects": {"block": 1},
        } for i in range(10)),
        "player": {"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3},
    })
    start = MapNodeState(node_id="start", act=1, floor=0, lane=0, kind=RoomKind.START)
    target = MapNodeState(node_id="fight", act=1, floor=1, lane=0, kind=RoomKind.MONSTER)
    state = state.model_copy(update={
        "phase": RunPhase.MAP,
        "map": MapState(
            nodes=(start, target), edges=(MapEdgeState(from_id="start", to_id="fight"),),
            current_node_id="start", completed_node_ids=("start",),
        ),
    })
    state = step(state, next(a for a in legal_actions(state) if a.type == "choose_node"))
    assert state.combat is not None
    return state


def test_vital_spark_skill_stacks_apply_to_each_attack_hit_and_expire() -> None:
    state = _combat("INFESTED_PRISM", statuses=(("VITAL_SPARK", 2),))
    for _ in range(2):
        state = step(state, next(a for a in legal_actions(state) if a.type == "play_card"))
    assert state.combat.player.statuses["tainted"] == 4
    assert state.combat.monsters[0].intent_damage == (5 + 4) * 3
    before = state.combat.player.hp
    state = step(state, next(a for a in legal_actions(state) if a.type == "end_turn"))
    assert state.combat.player.hp == before - (5 + 4) * 3 + 2
    assert "tainted" not in state.combat.player.statuses
    hits = [e for e in state.combat.last_events if e.kind == "player_damaged"]
    assert [e.metadata["tainted"] for e in hits] == [4, 4, 4]


def test_tainted_does_not_increase_nonattack_damage_and_artifact_blocks_application() -> None:
    state = _combat("INFESTED_PRISM", statuses=(("VITAL_SPARK", 3),))
    combat = state.combat.model_copy(update={
        "player": state.combat.player.model_copy(update={"statuses": {"artifact": 1}}),
    })
    combat, events = _apply_after_card_play_hooks(combat, combat.hand[0])
    assert "tainted" not in combat.player.statuses
    assert "artifact" not in combat.player.statuses
    assert any(e.kind == "status_blocked_by_artifact" for e in events)
    combat, _ = _apply_after_card_play_hooks(combat, combat.hand[0])
    before = combat.player.hp
    combat, events = _damage_combat_player(combat, 5, "nonattack")
    assert combat.player.hp == before - 5
    assert events[-1].metadata["tainted"] == 0


def test_presence_counts_across_turns_and_survives_serialization() -> None:
    state = _combat("AEONGLASS")
    for _ in range(5):
        state = step(state, next(a for a in legal_actions(state) if a.type == "play_card"))
    assert not any(c.card_id == "wither" for c in state.combat.hand)
    state = load_state(serialize(state))
    state = step(state, next(a for a in legal_actions(state) if a.type == "end_turn"))
    state = step(state, next(a for a in legal_actions(state) if a.type == "play_card"))
    withers = [c for c in state.combat.hand if c.card_id == "wither"]
    assert len(withers) == 1
    assert withers[0].cost < 0
    before = state.combat.player.hp
    combat, events = _apply_end_turn_hand_card_effects(state.combat, withers)
    assert combat.player.hp == before - 2  # The last Skill granted 1 Block.
    assert events[0].metadata["incoming"] == 3
    assert events[0].metadata["reason"] == "end_turn_hand_card"


@pytest.mark.parametrize("ascension,created,strength", [(0, 1, 3), (9, 2, 4)])
def test_intensity_repeated_upgrades_and_new_presence_cards(ascension, created, strength) -> None:
    # Pinned v0.107.1 Aeonglass source grants 3/4 Strength before adding uses.
    state = _combat("AEONGLASS", ascension=ascension)
    combat = state.combat
    base_wither = _card_from_spec({
        "card_id": "wither", "instance_id": "existing", "type": "status", "cost": -1,
        "description": "At the end of your turn, if this is in your Hand, take 3 damage.",
    }, 1)
    combat = combat.model_copy(update={"hand": (base_wither,), "draw_pile": ()})
    combat, _ = _apply_aeonglass_increasing_intensity(combat, 0, ascension)
    assert len(combat.discard_pile) == created
    assert combat.monsters[0].statuses["strength"] == strength
    assert combat.hand[0].custom["wither_upgrade_count"] == 1
    combat, _ = _apply_aeonglass_increasing_intensity(combat, 0, ascension)
    assert combat.hand[0].custom["wither_upgrade_count"] == 2
    before = combat.player.hp
    combat, _ = _apply_end_turn_hand_card_effects(combat, combat.hand)
    assert combat.player.hp == before - 9
    attack = _card_from_spec({"card_id": "a", "type": "attack", "cost": 0}, 99)
    for _ in range(6):
        combat, _ = _apply_after_card_play_hooks(combat, attack)
    fresh = combat.hand[-1]
    assert fresh.card_id == "wither"
    assert fresh.custom["wither_upgrade_count"] == 2


def test_dead_prism_and_presence_owners_do_not_trigger() -> None:
    state = _combat("AEONGLASS", statuses=(("VITAL_SPARK", 2),))
    combat = state.combat.model_copy(update={
        "monsters": (state.combat.monsters[0].model_copy(update={"hp": 0}),),
    })
    for _ in range(6):
        combat, _ = _apply_after_card_play_hooks(combat, combat.hand[0])
    assert "tainted" not in combat.player.statuses
    assert not any(c.card_id == "wither" for c in combat.hand)
