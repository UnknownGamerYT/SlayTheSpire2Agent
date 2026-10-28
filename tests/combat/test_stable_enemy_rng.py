"""Stream ownership and state-selection contracts from stable engine source."""

from __future__ import annotations

from random import Random

from combat.test_stable_enemy_lifecycle import _end, _fight

from sts2sim import load_state, serialize
from sts2sim.engine.rng import RunRng, capture_random_state, random_stream
from sts2sim.mechanics.monsters import (
    build_monster_definitions,
    initial_monster_move,
    next_monster_move,
    next_move_counts,
    roll_monster_hp,
    spawn_monsters,
    synthetic_encounter,
)


def test_creature_creation_uses_niche_and_avoids_other_current_max_hp():
    # CombatState.CreateCreature -> Creature.SetUniqueMonsterHpValue uses one
    # Niche draw per creature, with other current MaxHp values removed first.
    definitions = build_monster_definitions([{
        "id": "FIXTURE", "type": "Normal", "min_hp": 10, "max_hp": 11,
        "moves": [{"id": "HIT", "damage": {"normal": 1}}],
        "attack_pattern": {"initial_move": "HIT"},
    }])
    rng = RunRng("ENEMY-RNG")
    spawned = spawn_monsters(
        synthetic_encounter(encounter_id="FIXTURE", room_type="monster",
                            monster_ids=("FIXTURE", "FIXTURE")),
        definitions, rng, ascension_level=0,
        hp_rng=random_stream(rng, "niche"), ai_rng=random_stream(rng, "monster_ai"),
    )
    assert {monster.hp for monster in spawned} == {10, 11}
    snapshot = capture_random_state(rng)
    assert snapshot.draws == 0
    assert snapshot.streams["niche"].draws == 2
    assert "monster_ai" not in snapshot.streams or snapshot.streams["monster_ai"].draws == 0


def test_hp_candidate_exhaustion_falls_back_to_full_range():
    definition = build_monster_definitions([{
        "id": "FIXTURE", "min_hp": 10, "max_hp": 11,
    }])["FIXTURE"]
    assert roll_monster_hp(definition, Random(1), ascension_level=0,
                           excluded_max_hp=(10, 11)) in {10, 11}


def test_enemy_ai_uses_its_stream_and_reload_preserves_it():
    state = _fight(("INKLET",))
    state = state.model_copy(update={
        "rng": capture_random_state(RunRng("ENEMY-RNG")),
        "combat": state.combat.model_copy(update={
            "draw_per_turn": 0,
            "monsters": (state.combat.monsters[0].model_copy(update={"move_id": "JAB"}),),
        }),
    })
    restored = load_state(serialize(state))
    after = _end(state)
    assert serialize(after) == serialize(_end(restored))
    assert after.rng.draws == 0
    assert after.rng.streams["monster_ai"].draws == 1
    assert "niche" not in after.rng.streams


def test_can_repeat_twice_limits_consecutive_uses_and_resets_after_other_move():
    # RandomBranchState.GetStateWeight looks at the last two log entries;
    # JAB may return after a different move even when total JAB uses exceed two.
    definition = build_monster_definitions([{
        "id": "FIXTURE", "moves": [{"id": "JAB"}, {"id": "OTHER"}],
        "attack_pattern": {"initial_move": "RAND", "states": [
            {"id": "JAB_MOVE", "move_id": "JAB", "next": "RAND"},
            {"id": "OTHER_MOVE", "move_id": "OTHER", "next": "RAND"},
            {"id": "RAND", "type": "random", "branches": [
                {"move_id": "JAB", "repeat": "CanRepeatXTimes", "max_times": 2, "weight": 1},
                {"move_id": "OTHER", "weight": 1},
            ]},
        ]},
    }])["FIXTURE"]
    counts = next_move_counts(next_move_counts({}, "JAB"), "JAB")
    assert next_monster_move(definition, "JAB", Random(1), move_counts=counts).move_id == "OTHER"
    counts = next_move_counts(counts, "OTHER")
    assert next_monster_move(definition, "OTHER", Random(1), move_counts=counts).move_id == "JAB"
    assert initial_monster_move(definition, Random(1)).move_id == "JAB"
