"""Source monster Move Block and unpowered Block modifier outcomes."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
from combat.test_stable_enemy_lifecycle import CACHE, _end, _fight

from sts2sim import load_state, serialize
from sts2sim.engine.transitions import _gain_monster_block, _modified_monster_block_gain
from sts2sim.mechanics.monsters import (
    build_monster_definitions,
    monster_move_for_ascension,
    move_by_id,
)

CONTRACTS = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures/stable_enemy_block_contracts.json").read_text()
)["blocks"]


@pytest.fixture(scope="module")
def definitions():
    return build_monster_definitions(json.loads(CACHE.read_text()))


@pytest.mark.parametrize("ascension", [0, 7, 8, 9])
@pytest.mark.parametrize(
    "contract", CONTRACTS, ids=lambda row: row["monster_id"] + "." + row["move_id"]
)
def test_source_move_block_contracts(definitions, contract, ascension):
    definition = definitions[contract["monster_id"]]
    move = move_by_id(definition, contract["move_id"])
    assert (
        monster_move_for_ascension(definition, move, ascension_level=ascension).block
        == (contract["block_by_ascension"][str(ascension)])
    )


@pytest.mark.parametrize(
    "statuses,amount,powered,expected",
    [
        ({"dexterity": 3}, 8, True, 11),
        ({"dexterity": -3}, 8, True, 5),
        ({"dexterity": 3, "temporary_dexterity": -2, "fasten": 2}, 8, True, 11),
        ({"frail": 1, "shadowmeld": 1}, 5, True, 7),  # floor after 5*.75*2.
        ({"frail": 1, "shadowmeld": 1}, Fraction(13, 2), True, 9),
        ({"dexterity": -20, "frail": 1}, 8, True, 0),
        ({"no_block": 1, "unmovable": 99}, 8, True, 8),  # No card source; monster target.
        ({"dexterity": 30, "frail": 1, "fasten": 20}, 8, False, 8),
        ({"dexterity": 30, "frail": 1, "shadowmeld": 1}, 8, False, 16),
    ],
)
def test_source_enemy_block_modifiers(statuses, amount, powered, expected):
    assert _modified_monster_block_gain(statuses, amount, powered_move=powered) == expected


def test_enemy_block_caps_at_source_creature_limit():
    monster = _fight().combat.monsters[0].model_copy(update={"block": 999_999_998})
    result, amount = _gain_monster_block(monster, 4)
    assert result.block == 999_999_999
    assert amount == 4  # Source history receives modified gain, before the storage cap.


@pytest.mark.parametrize("ascension,base_damage", [(0, 13), (8, 13), (9, 15)])
def test_forgotten_miasma_gains_block_from_existing_dexterity_then_grows_it(ascension, base_damage):
    state = _end(_fight(("THE_FORGOTTEN",), ascension=ascension))
    forgotten = state.combat.monsters[0]
    assert forgotten.block == 8
    assert forgotten.statuses["dexterity"] == 2
    assert forgotten.intent_damage == base_damage + 2
    state = _end(state)  # Dread, then plan Miasma using the existing two Dexterity.
    assert state.combat.monsters[0].intent_block == 10
    state = _end(load_state(serialize(state)))
    forgotten = state.combat.monsters[0]
    assert forgotten.block == 10
    assert forgotten.statuses["dexterity"] == 4
    assert forgotten.intent_damage == base_damage + 4


def test_unpowered_guard_and_plating_ignore_dexterity_and_frail():
    state = _fight(("GUARDBOT", "FABRICATOR"))
    monsters = list(state.combat.monsters)
    monsters[1] = monsters[1].model_copy(
        update={
            "statuses": {
                **monsters[1].statuses,
                "dexterity": 20,
                "frail": 2,
            }
        }
    )
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": tuple(monsters),
                }
            )
        }
    )
    state = _end(state)
    assert state.combat.monsters[1].block == 15
