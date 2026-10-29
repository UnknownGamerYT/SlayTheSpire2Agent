"""Base damage/hit-count contracts derived from pinned source attack methods."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from combat.test_stable_enemy_lifecycle import CACHE, _end, _fight

from sts2sim.mechanics.monsters import (
    build_monster_definitions,
    monster_move_damage,
    monster_move_for_ascension,
    move_by_id,
)

CONTRACTS = json.loads(
    (
        Path(__file__).resolve().parents[1] / "fixtures/stable_enemy_attack_contracts.json"
    ).read_text()
)["attacks"]


@pytest.fixture(scope="module")
def definitions():
    return build_monster_definitions(json.loads(CACHE.read_text()))


@pytest.mark.parametrize("ascension", [0, 7, 8, 9])
@pytest.mark.parametrize(
    "contract", CONTRACTS, ids=lambda row: row["monster_id"] + "." + row["move_id"]
)
def test_source_attack_numeric_contracts(definitions, contract, ascension):
    definition = definitions[contract["monster_id"]]
    move = move_by_id(definition, contract["move_id"])
    damage = contract["damage_by_ascension"][str(ascension)]
    hits = contract["hits_by_ascension"][str(ascension)]
    if damage is not None:
        assert monster_move_damage(definition, move, ascension_level=ascension) == damage
    if hits is not None:
        assert (
            monster_move_for_ascension(definition, move, ascension_level=ascension).hit_count
            == hits
        )


def test_fake_merchant_spew_coins_executes_eight_hits():
    state = _fight(("FAKE_MERCHANT_MONSTER",))
    monster = state.combat.monsters[0].model_copy(update={"move_id": "SPEW_COINS"})
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )
    state = _end(state)
    assert state.combat.player.hp == 9984
    assert len([e for e in state.combat.last_events if e.kind == "player_damaged"]) == 8
