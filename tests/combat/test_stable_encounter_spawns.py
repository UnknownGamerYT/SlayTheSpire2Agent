"""Opening roster outcomes derived from all 87 stable GenerateMonsters methods."""

from __future__ import annotations

import json
from pathlib import Path
from random import Random

import pytest
from combat.test_stable_enemy_lifecycle import _end, _fight

from sts2sim import load_state, serialize
from sts2sim.engine.rng import RunRng, capture_random_state, model_random, random_stream
from sts2sim.mechanics.encounter_spawns import encounter_spawn_specs
from sts2sim.mechanics.monsters import (
    build_encounter_definitions,
    build_monster_definitions,
    spawn_monsters,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = json.loads((ROOT / "tests/fixtures/stable_encounter_roster_contracts.json").read_text())


@pytest.mark.parametrize("contract", CONTRACTS["encounters"], ids=lambda row: row["encounter_id"])
def test_all_registered_encounters_have_executable_opening_roster_contract(contract):
    definitions = build_monster_definitions(
        json.loads((ROOT / "data/cache/eng/monsters.json").read_text())
    )
    encounters = build_encounter_definitions(
        json.loads((ROOT / "data/cache/eng/encounters.json").read_text())
    )
    encounter = next(e for e in encounters if e.encounter_id == contract["encounter_id"])
    if encounter.encounter_id == "BATTLEWORN_DUMMY_EVENT_ENCOUNTER":
        from dataclasses import replace

        encounter = replace(encounter, monster_ids=("BATTLE_FRIEND_V2",))
    rng = RunRng("SOURCE-ROSTER")
    spawned = spawn_monsters(
        encounter,
        definitions,
        rng,
        ascension_level=0,
        hp_rng=random_stream(rng, "niche"),
        ai_rng=random_stream(rng, "monster_ai"),
        encounter_rng=model_random(rng, encounter.encounter_id, mixin=2),
    )
    if "opening_ids" in contract:
        assert [monster.source_monster_id for monster in spawned] == contract["opening_ids"]
    else:
        assert 1 <= len(spawned) <= 4
    assert all(monster.hp > 0 and monster.move is not None for monster in spawned)
    assert capture_random_state(rng).draws == 0


@pytest.mark.parametrize(
    "encounter_id,expected",
    [
        ("INKLETS_NORMAL", ["JAB", "WHIRLWIND", "JAB"]),
        ("CHOMPERS_NORMAL", ["CLAMP", "SCREECH"]),
        ("TUNNELER_NORMAL", ["SCREECH", None]),
        ("THE_KIN_BOSS", ["POWER_DANCE", "QUICK_SLASH", None]),
        ("PUNCH_OFF_EVENT_ENCOUNTER", ["FAST_PUNCH", "READY"]),
        ("DENSE_VEGETATION_EVENT_ENCOUNTER", ["NASTY_BITE", "WRIGGLE", "NASTY_BITE", "WRIGGLE"]),
    ],
)
def test_encounter_specific_initial_model_properties(encounter_id, expected):
    specs = encounter_spawn_specs(encounter_id, Random(1))
    assert [spec.initial_move_id for spec in specs] == expected


@pytest.mark.parametrize(
    "encounter_id",
    [
        "CORPSE_SLUGS_NORMAL",
        "DECIMILLIPEDE_ELITE",
        "SCROLLS_OF_BITING_NORMAL",
        "TWO_TAILED_RATS_NORMAL",
    ],
)
def test_phased_openings_are_distinct_then_use_source_fourth_scroll_default(encounter_id):
    specs = encounter_spawn_specs(encounter_id, Random(9))
    assert len({spec.initial_move_id for spec in specs[:3]}) == 3
    if encounter_id == "SCROLLS_OF_BITING_NORMAL":
        assert specs[3].initial_move_id == "MORE_TEETH"


@pytest.mark.parametrize("seed", range(20))
def test_random_rosters_choose_source_cardinalities_and_exclusions(seed):
    raiders = encounter_spawn_specs("RUBY_RAIDERS_NORMAL", Random(seed))
    assert len(raiders) == len({spec.monster_id for spec in raiders}) == 3
    workers = encounter_spawn_specs("BOWLBUGS_NORMAL", Random(seed))
    assert workers[0].monster_id == "BOWLBUG_ROCK"
    assert len(workers) == len({spec.monster_id for spec in workers}) == 3
    slimes = encounter_spawn_specs("SLIMES_WEAK", Random(seed))
    assert len(slimes) == 3
    assert {slimes[0].monster_id, slimes[2].monster_id} == {"LEAF_SLIME_S", "TWIG_SLIME_S"}


def test_dummy_requires_selected_setting_and_punch_off_hp_reduction_is_two_through_nine():
    with pytest.raises(ValueError, match="selected setting"):
        encounter_spawn_specs("BATTLEWORN_DUMMY_EVENT_ENCOUNTER", Random(1))
    for seed in range(20):
        assert all(
            2 <= spec.hp_reduction <= 9
            for spec in encounter_spawn_specs(
                "PUNCH_OFF_EVENT_ENCOUNTER",
                Random(seed),
            )
        )


@pytest.mark.parametrize("ascension,minimum,maximum", [(0, 19, 21), (8, 20, 22)])
def test_tough_egg_hatches_same_creature_cleans_powers_and_stays_hatched(
    ascension, minimum, maximum
):
    state = _fight(("TOUGH_EGG",), ascension=ascension)
    egg = state.combat.monsters[0]
    assert egg.statuses["hatch"] == 1
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": (
                        egg.model_copy(
                            update={
                                "hp": 1,
                                "statuses": {
                                    "hatch": 1,
                                    "weak": 1,
                                    "strength": 9,
                                    "minion": 1,
                                },
                            }
                        ),
                    ),
                }
            )
        }
    )
    state = _end(load_state(serialize(state)))
    hatched = state.combat.monsters[0]
    assert hatched.monster_id == egg.monster_id
    assert minimum <= hatched.hp == hatched.max_hp <= maximum
    assert hatched.name == "Hatchling"
    assert hatched.statuses == {"minion": 1}
    assert hatched.move_id == "NIBBLE"
    state = _end(state)
    assert state.combat.monsters[0].move_id == "NIBBLE"
