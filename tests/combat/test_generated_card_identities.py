from __future__ import annotations

import json
from pathlib import Path

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _source, _state
from combat.test_glam import _end_turn, _play

from sts2sim import legal_actions, serialize, step
from sts2sim.engine import MonsterState
from sts2sim.engine.transitions import (
    _apply_combat_relic_marker,
    _generated_status_cards,
    _unique_generated_card_id,
)
from sts2sim.mechanics.relic_combat import CombatRelicHook, CombatRelicMarker


def _ids(state):
    return [card.instance_id for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile")
            for card in getattr(state.combat, zone)]


def _power_with_id_gap():
    power = _source("AFTERIMAGE").model_copy(update={"instance_id": "card_001"})
    # After the Power leaves the piles, a length-based allocator would choose
    # card_003 even though this permanent card is still in the combat.
    state = _state(power, _simple("card_002"), _simple("card_003"))
    state = state.model_copy(update={"master_deck": state.combat.hand})
    return _play(state, "card_001")


@pytest.mark.parametrize("enemy,move,status,count", [
    ("NOISEBOT", "NOISE", "dazed", 4), ("MYTE", "TOXIC", "toxic", 2),
])
def test_source_power_install_then_enemy_status_generation_keeps_every_id_unique(
    enemy, move, status, count,
):
    state = _power_with_id_gap()
    assert set(_ids(state)) == {"card_002", "card_003"}
    assert state.player.statuses["afterimage"] == 1
    rows = json.loads((Path("data/cache/eng/monsters.json")).read_text())
    source = next(row for row in rows if row["id"] == enemy)
    monster = MonsterState(monster_id=enemy, hp=100, max_hp=100, move_id=move,
                           metadata={"source_monster_id": enemy})
    state = state.model_copy(update={
        "flags": {**state.flags, "monsters": (source,)},
        "combat": state.combat.model_copy(update={"monsters": (monster,)}),
    })
    direct = _end_turn(state)
    restored = _end_turn(_reload(state))
    assert serialize(direct) == serialize(restored)
    state = direct
    ids = _ids(state)
    assert len(ids) == len(set(ids)) == 2 + count
    created = [card for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile")
               for card in getattr(state.combat, zone) if card.card_id == status]
    assert len(created) == count
    assert all(card.instance_id.startswith(f"status:{status}:") for card in created)
    assert "card_001" in state.combat.metadata["reserved_card_instance_ids"]
    state = _end_turn(_reload(state))
    assert len(_ids(state)) == len(set(_ids(state)))


def test_installed_generated_power_ids_and_creation_hooks_are_not_reused_after_reload() -> None:
    state = _power_with_id_gap()
    spec = _source("AFTERIMAGE").model_dump(mode="json")
    spec["instance_id"] = "card_001"  # Already installed, absent from all piles.
    factories = tuple(_simple(f"make:{i}", effects={"add_card_to_hand": {"card": spec}})
                      for i in range(2))
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "hand": state.combat.hand + factories,
        "player": state.player.model_copy(update={
            "statuses": {**state.player.statuses, "arsenal": 1},
        }),
    })})
    generated_ids = []
    for index in range(2):
        state = _play(_reload(state), f"make:{index}")
        generated = next(card for card in state.combat.hand if card.card_id == "afterimage")
        generated_ids.append(generated.instance_id)
        assert generated.instance_id != "card_001"
        state = _play(state, generated.instance_id)
        assert generated.instance_id not in _ids(state)
    assert len(set(generated_ids)) == 2
    assert state.player.statuses["afterimage"] == 3
    assert state.player.statuses["strength"] == 2  # Arsenal fires for both distinct creations.
    assert set(generated_ids) <= set(state.combat.metadata["created_card_hook_ids"])
    assert set(generated_ids) <= set(state.combat.metadata["reserved_card_instance_ids"])


def test_generated_choices_get_new_ids_after_the_previous_chosen_power_is_installed() -> None:
    discover = _simple("discover", effects={"choose_card": {
        "zone": "generated", "action": "move_to_hand", "count": 1, "choices": 1,
        "card_types": ("power",),
    }})
    retrieve = _simple("retrieve", effects={"choose_card": {
        "zone": "discard_pile", "action": "move_to_hand", "count": 1,
    }})
    source_power = _source("AFTERIMAGE").custom["source_spec"]
    state = _state(discover, retrieve)
    state = state.model_copy(update={"flags": {**state.flags, "card_pool": (source_power,)}})
    generated_ids = []
    for iteration in range(2):
        state = _play(_reload(state), "discover")
        choice = next(action for action in legal_actions(state) if action.type == "choose_card")
        generated_ids.append(choice.card_instance_id)
        state = step(state, choice)
        state = _play(state, choice.card_instance_id)
        if iteration == 0:
            state = _play(state, "retrieve")
            state = step(state, next(action for action in legal_actions(state)
                                     if action.card_instance_id == "discover"))
    assert len(set(generated_ids)) == 2
    assert state.player.statuses["afterimage"] == 2


def test_relic_generated_batches_reserve_ids_against_all_existing_combat_cards() -> None:
    state = _power_with_id_gap()
    marker = CombatRelicMarker(kind="add_card_to_hand", relic_id="ninja_scroll",
                               hook=CombatRelicHook.START_OF_COMBAT, amount=3,
                               metadata={"card_id": "shiv"})
    for _ in range(2):
        combat, _events = _apply_combat_relic_marker(state.combat, marker)
        state = _reload(state.model_copy(update={"combat": combat}))
    assert len(_ids(state)) == len(set(_ids(state))) == 8
    assert len([card for card in state.combat.hand if card.card_id == "shiv"]) == 6


def test_allocator_respects_saved_hook_and_power_ids_and_reserves_a_whole_status_batch() -> None:
    state = _state(_simple("status:dazed:1:1"), _simple("status:dazed:1:1:copy:1"))
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"metadata": {
        "created_card_hook_ids": ("status:dazed:1:2",),
        "reserved_card_instance_ids": ("status:dazed:1:2:copy:1",),
        "combat_triggers": ({"source_instance_id": "status:dazed:1:3"},),
    }})})
    state = _reload(state)
    assert _unique_generated_card_id(state.combat, "status:dazed:1:2") == "status:dazed:1:2:copy:2"
    cards = _generated_status_cards(state.combat, "dazed", 4)
    ids = _ids(state) + [card.instance_id for card in cards]
    assert len(ids) == len(set(ids)) == 6
    assert not {"status:dazed:1:2", "status:dazed:1:2:copy:1", "status:dazed:1:3"} & {
        card.instance_id for card in cards
    }
