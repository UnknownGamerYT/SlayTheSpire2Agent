from __future__ import annotations

import json
from pathlib import Path

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _state
from combat.test_glam import _end_turn, _play

from sts2sim import serialize
from sts2sim.engine import CardEnchantment
from sts2sim.engine.transitions import _card_from_spec

# Exact stable Spire Codex source rows fetched 2026-10-28 (v0.107.1).
SOURCE = {
    row["id"]: row
    for row in json.loads(
        (Path(__file__).parents[1] / "fixtures" / "source_damage_fidelity.json").read_text()
    )
}


def _source(card_id, *, upgraded=False, replay=False):
    card = _card_from_spec(
        {**SOURCE[card_id], "upgraded": upgraded, "instance_id": card_id.lower()}, 0
    )
    if replay:
        card = card.model_copy(update={
            "enchantments": (CardEnchantment(keyword="replay", amount=1),),
        })
    return card


@pytest.mark.parametrize("upgraded,base", [(False, 13), (True, 16)])
@pytest.mark.parametrize("other_cards", [0, 2, 8])
def test_precise_cut_counts_remaining_hand_and_floors_damage(upgraded, base, other_cards):
    card = _source("PRECISE_CUT", upgraded=upgraded)
    others = tuple(_simple(f"other:{i}") for i in range(other_cards))
    state = _play(_reload(_state(card, *others)), "precise_cut")
    assert state.combat.monsters[0].hp == 100 - max(0, base - 2 * other_cards)


@pytest.mark.parametrize("upgraded,per_card", [(False, 3), (True, 4)])
@pytest.mark.parametrize("exhausted", [0, 2])
def test_ashen_strike_counts_exhaust_pile_at_play_time(upgraded, per_card, exhausted):
    card = _source("ASHEN_STRIKE", upgraded=upgraded)
    state = _state(card)
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "exhaust_pile": tuple(_simple(f"old:{i}") for i in range(exhausted)),
    })})
    state = _play(_reload(state), "ashen_strike")
    assert state.combat.monsters[0].hp == 100 - 6 - per_card * exhausted


def test_ashen_strike_observes_card_exhausted_by_previous_play_and_replays():
    card = _source("ASHEN_STRIKE", replay=True)
    burn = _simple("burn", exhausts=True)
    state = _play(_state(burn, card), "burn")
    state = _play(_reload(state), "ashen_strike")
    assert state.combat.monsters[0].hp == 82  # Two 9-damage hits.
    assert len(state.combat.exhaust_pile) == 1


@pytest.mark.parametrize("upgraded,damage", [(False, 3), (True, 5)])
@pytest.mark.parametrize("energy_spent", [0, 1, 3])
def test_helix_drill_repeats_once_per_energy_actually_spent(upgraded, damage, energy_spent):
    drill = _source("HELIX_DRILL", upgraded=upgraded)
    spend = _simple("spend", cost=energy_spent)
    refund = _simple("refund", effects={"energy": 3})
    state = _play(_state(spend, refund, drill), "spend")
    state = _play(_reload(state), "refund")
    state = _play(state, "helix_drill")
    assert state.combat.monsters[0].hp == 100 - energy_spent * damage
    hits = [e for e in state.combat.last_events if e.kind == "monster_damaged"]
    assert len(hits) == energy_spent


def test_helix_drill_replay_does_not_spend_again_and_survives_reload():
    drill = _source("HELIX_DRILL", replay=True)
    spend = _simple("spend", cost=2)
    state = _play(_state(spend, drill, statuses={"strength": 2}), "spend")
    result, restored = (_play(candidate, "helix_drill") for candidate in (state, _reload(state)))
    assert serialize(result) == serialize(restored)
    assert result.combat.monsters[0].hp == 80  # 2 repetitions of 2 hits at 3+2.
    assert result.player.energy == 18


def test_helix_drill_counter_resets_next_turn():
    drill = _source("HELIX_DRILL")
    spend = _simple("spend", cost=2)
    state = _play(_state(spend, draw=(drill,), draw_per_turn=1), "spend")
    state = _play(_end_turn(_reload(state)), "helix_drill")
    assert state.combat.monsters[0].hp == 100


def test_helix_drill_stops_at_dead_selected_target_without_hitting_other_enemy():
    drill = _source("HELIX_DRILL")
    state = _state(_simple("spend", cost=3), drill)
    first = state.combat.monsters[0].model_copy(update={"hp": 2})
    other = first.model_copy(update={"monster_id": "other", "hp": 100})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (first, other),
    })})
    state = _play(_play(state, "spend"), "helix_drill")
    assert [monster.hp for monster in state.combat.monsters] == [0, 100]
    assert len([e for e in state.combat.last_events if e.kind == "monster_damaged"]) == 1
