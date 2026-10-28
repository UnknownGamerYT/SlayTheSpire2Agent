"""Stable Osty/Phylactery outcomes from pinned OstyCmd and relic source."""

from __future__ import annotations

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import MapEdgeState, MapNodeState, MapState, RoomKind, RunPhase
from sts2sim.engine.transitions import (
    _apply_osty_action_effect,
    _apply_osty_hp_loss_triggers,
    _apply_osty_summon,
    _card_from_spec,
    _damage_combat_player,
    _osty_state,
    _set_osty_state,
)


def _start(character="TEST", relics=None):
    source = {
        "monsters": [
            {
                "id": "FIXTURE",
                "min_hp": 500,
                "max_hp": 500,
                "initial_move": "WAIT",
                "moves": [{"id": "WAIT", "follow_up": "WAIT"}],
            }
        ],
        "encounters": [{"id": "FIXTURE", "monsters": [{"id": "FIXTURE"}]}],
        "combat_encounter_id": "FIXTURE",
        "deck": [
            {
                "id": "defend",
                "type": "skill",
                "target": "self",
                "cost": 0,
                "effects": {"block": 1},
                "copies": 10,
            }
        ],
    }
    if relics is not None:
        source["relics"] = relics
    state = new_run(seed=113, character_id=character, ascension=0, source_data=source)
    start = MapNodeState(node_id="s", act=1, floor=0, lane=0, kind=RoomKind.START)
    fight = MapNodeState(node_id="f", act=1, floor=1, lane=0, kind=RoomKind.MONSTER)
    state = state.model_copy(
        update={
            "phase": RunPhase.MAP,
            "map": MapState(
                nodes=(start, fight),
                edges=(MapEdgeState(from_id="s", to_id="f"),),
                current_node_id="s",
                completed_node_ids=("s",),
            ),
        }
    )
    return step(state, next(a for a in legal_actions(state) if a.type == "choose_node"))


def _end(state):
    return step(state, next(a for a in legal_actions(state) if a.type == "end_turn"))


@pytest.mark.parametrize("character", ["IRONCLAD", "SILENT", "REGENT", "DEFECT"])
def test_other_character_starters_do_not_create_osty(character):
    state = _start(character)
    assert "osty" not in state.combat.metadata
    assert not _osty_state(state.combat)["alive"]


def test_necrobinder_character_without_pet_relic_does_not_invent_a_pet():
    assert not _osty_state(_start("NECROBINDER", []).combat)["alive"]


def test_bound_phylactery_stable_opening_one_and_later_turn_one():
    state = _start("NECROBINDER")
    assert state.combat.metadata["osty"]["hp"] == 1
    assert [e.amount for e in state.combat.last_events if e.kind == "osty_summoned"] == [1]
    state = _end(state)
    assert state.combat.metadata["osty"]["hp"] == 2
    assert state.combat.metadata["osty"]["max_hp"] == 2


def test_unbound_opens_five_then_summons_two_on_first_and_later_turns():
    state = _start("NECROBINDER", ["phylactery_unbound"])
    assert state.combat.metadata["osty"]["hp"] == 7
    assert [e.amount for e in state.combat.last_events if e.kind == "osty_summoned"] == [5, 2]
    assert _end(state).combat.metadata["osty"]["hp"] == 9


@pytest.mark.parametrize(
    "initial,expected,created,revived",
    [
        (None, (6, 6), True, False),
        ({"hp": 18, "max_hp": 20, "alive": True}, (24, 26), False, False),
        ({"hp": 0, "max_hp": 50, "alive": False}, (6, 6), False, True),
    ],
)
def test_summon_creates_grows_or_resets_the_same_revived_pet(initial, expected, created, revived):
    combat = _start().combat
    if initial is not None:
        combat = _set_osty_state(combat, initial)
    combat, events = _apply_osty_summon(combat, 6, "bodyguard", reason="card")
    osty = combat.metadata["osty"]
    assert (osty["hp"], osty["max_hp"]) == expected
    assert osty["die_for_you"] is True
    assert events[0].metadata["created"] is created
    assert events[0].metadata["revived"] is revived


def test_missing_pet_attack_and_heal_do_not_fabricate_twenty_hp():
    state = _start()
    card = _card_from_spec({"id": "fixture", "type": "skill", "cost": 0}, 1)
    combat, _, events = _apply_osty_action_effect(
        state.combat,
        state.rng,
        card,
        {"osty_action": [{"action": "damage", "amount": 6}, {"action": "heal", "amount": 5}]},
    )
    assert not _osty_state(combat)["alive"]
    assert combat.monsters[0].hp == 500
    assert [e.kind for e in events] == ["osty_action_blocked", "osty_action_blocked"]


def test_attack_block_pet_death_overkill_and_revival_survive_reload():
    state = _start("NECROBINDER")
    combat = _set_osty_state(state.combat, {"hp": 4, "max_hp": 10, "alive": True})
    combat = combat.model_copy(update={"player": combat.player.model_copy(update={"block": 3})})
    combat, events = _damage_combat_player(combat, 9, "enemy", is_attack=True)
    assert combat.player.hp == state.combat.player.hp - 2
    assert combat.metadata["osty"]["hp"] == 0
    assert combat.metadata["osty"]["max_hp"] == 10
    assert sum(e.kind == "osty_died" for e in events) == 1
    state = load_state(serialize(state.model_copy(update={"combat": combat})))
    state = _end(state)
    assert state.combat.metadata["osty"]["hp"] == 1
    assert state.combat.metadata["osty"]["max_hp"] == 1
    assert [
        e.metadata["revived"] for e in state.combat.last_events if e.kind == "osty_summoned"
    ] == [True]


def test_necro_mastery_stacks_and_bypasses_enemy_block():
    combat = _start().combat
    combat = _set_osty_state(
        combat, {"hp": 3, "max_hp": 3, "alive": True, "loss_damage_multiplier": 2}
    )
    monster = combat.monsters[0].model_copy(update={"block": 100})
    combat = combat.model_copy(update={"monsters": (monster,)})
    combat, events = _apply_osty_hp_loss_triggers(combat, "enemy", 3, relics=())
    assert combat.monsters[0].hp == 494
    assert combat.monsters[0].block == 100
    assert events[0].amount == 6


@pytest.mark.parametrize("target,stock", [("enemy", 1), ("all_enemies", 0), ("random_enemy", 0)])
def test_pet_multihit_target_identity_and_immediate_stock_replacements(target, stock):
    from combat.test_stable_enemy_lifecycle import _fight

    state = _fight()
    combat = _set_osty_state(
        state.combat,
        {
            "hp": 4,
            "max_hp": 4,
            "alive": True,
            "attacks_this_turn": 1,
        },
    )
    card = _card_from_spec({"id": "fixture", "type": "attack", "cost": 0}, 1)
    old_id = combat.monsters[0].monster_id
    combat, _, _ = _apply_osty_action_effect(
        combat,
        state.rng,
        card,
        {
            "osty_action": {
                "action": "damage",
                "amount": 10000,
                "target": target,
                "extra_hits": "attacks_this_turn",
            }
        },
        state=state,
        target_id=old_id,
        relics=("bone_flute",),
    )
    living = [m for m in combat.monsters if m.hp > 0]
    assert len(living) == 1
    assert living[0].statuses.get("stock", 0) == stock
    # The Rattle command counts once and Bone Flute runs after that command.
    assert combat.metadata["osty_attacks_this_turn"] == 2
    assert combat.player.block == 2


def test_necro_mastery_repeated_installations_and_zero_summon():
    state = _start()
    before = serialize(state)
    combat, events = _apply_osty_summon(state.combat, 0, "fixture", reason="card")
    assert combat is state.combat
    assert events == ()
    assert serialize(state) == before
    card = _card_from_spec({"id": "necro_mastery", "type": "power", "cost": 0}, 1)
    for _ in range(2):
        combat, _, _ = _apply_osty_action_effect(
            combat,
            state.rng,
            card,
            {"osty_action": {"action": "enable_loss_damage"}},
        )
    assert combat.metadata["osty"]["loss_damage_multiplier"] == 2
    assert combat.metadata["osty"]["alive"] is False


def test_thorns_kills_pet_before_damage_but_finishes_current_aoe_batch():
    state = _start()
    combat = _set_osty_state(
        state.combat,
        {
            "hp": 2,
            "max_hp": 2,
            "alive": True,
            "attacks_this_turn": 1,
        },
    )
    first = combat.monsters[0].model_copy(update={"statuses": {"thorns": 3}})
    second = first.model_copy(update={"monster_id": "second"})
    combat = combat.model_copy(
        update={
            "monsters": (first, second),
            "player": combat.player.model_copy(update={"block": 1}),
        }
    )
    card = _card_from_spec({"id": "fixture", "type": "attack", "cost": 0}, 1)
    combat, _, events = _apply_osty_action_effect(
        combat,
        state.rng,
        card,
        {
            "osty_action": {
                "action": "damage",
                "amount": 6,
                "target": "all_enemies",
                "extra_hits": "attacks_this_turn",
            }
        },
    )
    assert [m.hp for m in combat.monsters] == [494, 494]
    assert combat.player.hp == state.combat.player.hp
    assert combat.player.block == 0
    assert combat.metadata["osty"]["alive"] is False
    assert sum(e.kind == "osty_died" for e in events) == 1
    assert next(i for i, e in enumerate(events) if e.kind == "osty_died") < next(
        i for i, e in enumerate(events) if e.kind == "monster_damaged"
    )


def test_single_target_pet_attack_respects_selected_enemy_identity():
    state = _start()
    combat = _set_osty_state(state.combat, {"hp": 3, "max_hp": 3, "alive": True})
    second = combat.monsters[0].model_copy(update={"monster_id": "second"})
    combat = combat.model_copy(update={"monsters": combat.monsters + (second,)})
    card = _card_from_spec({"id": "fixture", "type": "attack", "cost": 0}, 1)
    combat, _, _ = _apply_osty_action_effect(
        combat,
        state.rng,
        card,
        {"osty_action": {"action": "damage", "amount": 6}},
        target_id="second",
    )
    assert [m.hp for m in combat.monsters] == [500, 494]
