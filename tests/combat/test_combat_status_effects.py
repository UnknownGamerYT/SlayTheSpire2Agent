from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import MapEdgeState, MapNodeState, MapState, RoomKind, RunPhase

TRAINING_MONSTERS = (
    {
        "id": "STATUS_DUMMY",
        "name": "Status Dummy",
        "type": "Normal",
        "min_hp": 50,
        "max_hp": 50,
        "moves": (
            {
                "id": "STRIKE",
                "name": "Strike",
                "intent": "Attack",
                "damage": {"normal": 10, "ascension": 10, "hit_count": 1},
                "block": None,
                "heal": None,
                "powers": None,
            },
        ),
        "attack_pattern": {
            "initial_move": "STRIKE",
            "states": (
                {
                    "id": "STRIKE_MOVE",
                    "move_id": "STRIKE",
                    "next": "STRIKE_MOVE",
                    "type": "move",
                },
            ),
            "type": "cycle",
        },
    },
)

TRAINING_ENCOUNTERS = (
    {
        "id": "STATUS_DUMMY_ENCOUNTER",
        "name": "Status Dummy",
        "act": "Act 1 - Overgrowth",
        "room_type": "Monster",
        "is_weak": True,
        "monsters": ({"id": "STATUS_DUMMY"},),
    },
)


def _choose_first_ancient(state):
    action = next(action for action in legal_actions(state) if action.type == "choose_ancient")
    return step(state, action)


def _force_next_room(state, room_kind: RoomKind):
    start = MapNodeState(node_id="start", act=state.act, floor=0, lane=0, kind=RoomKind.START)
    target = MapNodeState(node_id="target", act=state.act, floor=1, lane=0, kind=room_kind)
    game_map = MapState(
        act=state.act,
        nodes=(start, target),
        edges=(MapEdgeState(from_id=start.node_id, to_id=target.node_id),),
        current_node_id=start.node_id,
        completed_node_ids=(start.node_id,),
    )
    return state.model_copy(update={"phase": RunPhase.MAP, "map": game_map, "floor": 0})


def _combat_card(
    card_id: str,
    *,
    card_type: str,
    effects: Mapping[str, Any],
    target: str = "AnyEnemy",
    tags: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "card_id": card_id,
        "name": card_id.replace("_", " ").title(),
        "type": card_type,
        "target": target,
        "cost": 0,
        "effects": dict(effects),
        "tags": tuple(tags),
    }


def _enter_status_combat(
    deck: Sequence[Mapping[str, Any]],
    *,
    player_statuses: Mapping[str, int] | None = None,
    monster_statuses: Mapping[str, int] | None = None,
):
    state = new_run(
        seed=6200,
        character_id="TEST",
        ascension=0,
        source_data={
            "monsters": TRAINING_MONSTERS,
            "encounters": TRAINING_ENCOUNTERS,
            "combat_encounter_id": "STATUS_DUMMY_ENCOUNTER",
            "deck": tuple(dict(card) for card in deck),
            "player": {"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3},
        },
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.MONSTER)
    state = step(
        state,
        next(action for action in legal_actions(state) if action.type == "choose_node"),
    )
    if player_statuses:
        state = _set_player_statuses(state, player_statuses)
    if monster_statuses:
        state = _set_monster_statuses(state, monster_statuses)
    return state


def _set_player_statuses(state, statuses: Mapping[str, int]):
    assert state.combat is not None
    player = state.combat.player.model_copy(update={"statuses": dict(statuses)})
    return state.model_copy(
        update={
            "player": state.player.model_copy(update={"statuses": dict(statuses)}),
            "combat": state.combat.model_copy(update={"player": player}),
        }
    )


def _set_monster_statuses(state, statuses: Mapping[str, int]):
    assert state.combat is not None
    monster = state.combat.monsters[0].model_copy(update={"statuses": dict(statuses)})
    return state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )


def _play_card(state, card_id: str):
    assert state.combat is not None
    card = next(card for card in state.combat.hand if card.card_id == card_id)
    return step(
        state,
        next(
            action
            for action in legal_actions(state)
            if action.type == "play_card" and action.card_instance_id == card.instance_id
        ),
    )


def _end_turn(state):
    return step(state, next(action for action in legal_actions(state) if action.type == "end_turn"))


def test_strength_and_vulnerable_increase_card_damage() -> None:
    state = _enter_status_combat(
        (_combat_card("heavy_strike", card_type="attack", effects={"damage": 10}),),
        player_statuses={"strength": 2},
        monster_statuses={"vulnerable": 1},
    )

    state = _play_card(state, "heavy_strike")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == 32


def test_wrath_stance_doubles_player_attack_damage_and_incoming_damage() -> None:
    state = _enter_status_combat(
        (_combat_card("heavy_strike", card_type="attack", effects={"damage": 10}),),
        player_statuses={"stance_wrath": 1},
    )

    state = _play_card(state, "heavy_strike")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == 30

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.player.hp == 60


def test_mantra_reaching_ten_enters_divinity_and_triples_attack_damage() -> None:
    state = _enter_status_combat(
        (
            _combat_card(
                "meditate",
                card_type="skill",
                effects={"player_resource": {"resource": "mantra", "amount": 10}},
                target="Self",
            ),
            _combat_card("divine_strike", card_type="attack", effects={"damage": 6}),
        )
    )

    state = _play_card(state, "meditate")

    assert state.combat is not None
    assert state.combat.player.resources["mantra"] == 0
    assert state.combat.player.statuses["stance_divinity"] == 1
    assert state.combat.player.energy == 6
    assert any(event.kind == "stance_changed" for event in state.combat.last_events)

    state = _play_card(state, "divine_strike")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == 32


def test_weak_dexterity_and_frail_modify_player_card_values() -> None:
    state = _enter_status_combat(
        (
            _combat_card("quick_strike", card_type="attack", effects={"damage": 10}),
            _combat_card("guard", card_type="skill", effects={"block": 10}, target="Self"),
        ),
        player_statuses={"weak": 1, "dexterity": 2, "frail": 1},
    )

    state = _play_card(state, "quick_strike")
    state = _play_card(state, "guard")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == 43
    assert state.combat.player.block == 9


def test_poison_ticks_before_monster_attack_and_decrements() -> None:
    state = _enter_status_combat(
        (_combat_card("wait", card_type="skill", effects={"block": 0}, target="Self"),),
        monster_statuses={"poison": 3},
    )

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.monsters[0].hp == 47
    assert state.combat.monsters[0].statuses["poison"] == 2
    assert state.combat.player.hp == 70
    assert any(event.kind == "monster_poison_damage" for event in state.combat.last_events)


def test_poison_at_one_is_removed_after_tick() -> None:
    state = _enter_status_combat(
        (_combat_card("wait", card_type="skill", effects={"block": 10}, target="Self"),),
        monster_statuses={"poison": 1},
    )

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.monsters[0].hp == 49
    assert "poison" not in state.combat.monsters[0].statuses
    poison_event = next(
        event for event in state.combat.last_events if event.kind == "monster_poison_damage"
    )
    assert poison_event.metadata["remaining_poison"] == 0


def test_artifact_blocks_one_debuff_application() -> None:
    state = _enter_status_combat(
        (
            _combat_card(
                "hex",
                card_type="skill",
                effects={"apply_status": {"target": "enemy", "weak": 2, "vulnerable": 2}},
            ),
        ),
        monster_statuses={"artifact": 1},
    )

    state = _play_card(state, "hex")

    assert state.combat is not None
    monster_statuses = state.combat.monsters[0].statuses
    assert "artifact" not in monster_statuses
    assert "weak" not in monster_statuses
    assert monster_statuses["vulnerable"] == 2
    assert any(event.kind == "status_blocked_by_artifact" for event in state.combat.last_events)


def test_intangible_caps_unblocked_attack_damage_and_ticks_down() -> None:
    state = _enter_status_combat(
        (_combat_card("wait", card_type="skill", effects={"block": 0}, target="Self"),),
        player_statuses={"intangible": 1},
    )

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.player.hp == 79
    assert "intangible" not in state.combat.player.statuses
    assert any(event.metadata.get("intangible") for event in state.combat.last_events)


def test_metallicize_and_plated_armor_gain_block_before_monster_attack() -> None:
    state = _enter_status_combat(
        (_combat_card("wait", card_type="skill", effects={"block": 0}, target="Self"),),
        player_statuses={"metallicize": 2, "plated_armor": 4},
    )

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.player.hp == 76
    assert state.combat.player.block == 0
    assert state.combat.player.statuses["metallicize"] == 2
    assert state.combat.player.statuses["plated_armor"] == 3


def test_thorns_damages_attacking_monster() -> None:
    state = _enter_status_combat(
        (_combat_card("wait", card_type="skill", effects={"block": 0}, target="Self"),),
        player_statuses={"thorns": 3},
    )

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.player.hp == 70
    assert state.combat.monsters[0].hp == 47
    assert any(
        event.kind == "monster_damaged" and event.metadata.get("status") == "thorns"
        for event in state.combat.last_events
    )


def test_thorns_triggers_per_hit_for_fallback_multihit_monster() -> None:
    state = _enter_status_combat(
        (_combat_card("wait", card_type="skill", effects={"block": 0}, target="Self"),),
        player_statuses={"thorns": 3},
    )
    assert state.combat is not None
    monster = state.combat.monsters[0].model_copy(
        update={
            "monster_id": "fallback_multihit",
            "name": "Fallback Multihit",
            "intent": "Attack",
            "intent_damage": 8,
            "hit_count": 4,
            "move_id": None,
            "statuses": {},
        }
    )
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.player.hp == 72
    assert state.combat.monsters[0].hp == 38
    thorns_events = [
        event
        for event in state.combat.last_events
        if event.kind == "monster_damaged" and event.metadata.get("status") == "thorns"
    ]
    assert len(thorns_events) == 4
    assert all(event.amount == 3 for event in thorns_events)
    damage_hits = [
        event
        for event in state.combat.last_events
        if event.kind == "player_damaged" and event.source_id == "fallback_multihit"
    ]
    assert [event.amount for event in damage_hits] == [2, 2, 2, 2]
    assert [event.metadata["hit_index"] for event in damage_hits] == [0, 1, 2, 3]


def test_retain_hand_keeps_cards_until_next_turn_and_ticks_down() -> None:
    state = _enter_status_combat(
        (
            _combat_card("strike_a", card_type="attack", effects={"damage": 1}),
            _combat_card("strike_b", card_type="attack", effects={"damage": 1}),
        ),
        player_statuses={"retain_hand": 1},
    )
    assert state.combat is not None
    starting_hand_ids = [card.instance_id for card in state.combat.hand]

    state = _end_turn(state)

    assert state.combat is not None
    assert [card.instance_id for card in state.combat.hand] == starting_hand_ids
    assert state.combat.discard_pile == ()
    assert "retain_hand" not in state.combat.player.statuses


def test_next_turn_energy_draw_block_and_star_apply_before_draw() -> None:
    deck = tuple(
        _combat_card(f"card_{index}", card_type="skill", effects={"block": 0}, target="Self")
        for index in range(8)
    )
    state = _enter_status_combat(
        deck,
        player_statuses={
            "next_turn_energy": 2,
            "next_turn_draw": 1,
            "next_turn_block": 4,
            "next_turn_star": 1,
        },
    )

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.player.energy == 5
    assert state.combat.player.block == 4
    assert state.combat.player.resources["star"] == 1
    assert len(state.combat.hand) == 6
    assert not {
        "next_turn_energy",
        "next_turn_draw",
        "next_turn_block",
        "next_turn_star",
    } & set(state.combat.player.statuses)


def test_temporary_strength_and_dexterity_apply_for_one_turn() -> None:
    state = _enter_status_combat(
        (
            _combat_card("temporary_strike", card_type="attack", effects={"damage": 10}),
            _combat_card("temporary_guard", card_type="skill", effects={"block": 5}, target="Self"),
        ),
        player_statuses={"temporary_strength": 3, "temporary_dexterity": 2},
    )

    state = _play_card(state, "temporary_strike")
    state = _play_card(state, "temporary_guard")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == 37
    assert state.combat.player.block == 7

    state = _end_turn(state)

    assert state.combat is not None
    assert "temporary_strength" not in state.combat.player.statuses
    assert "temporary_dexterity" not in state.combat.player.statuses


def test_strength_down_and_dexterity_down_remove_base_buffs_at_turn_end() -> None:
    state = _enter_status_combat(
        (_combat_card("wait", card_type="skill", effects={"block": 0}, target="Self"),),
        player_statuses={
            "strength": 5,
            "strength_down": 5,
            "dexterity": 5,
            "dexterity_down": 5,
        },
    )

    state = _end_turn(state)

    assert state.combat is not None
    assert "strength" not in state.combat.player.statuses
    assert "strength_down" not in state.combat.player.statuses
    assert "dexterity" not in state.combat.player.statuses
    assert "dexterity_down" not in state.combat.player.statuses


def test_accuracy_and_afterimage_statuses_modify_shivs_and_card_play_block() -> None:
    state = _enter_status_combat(
        (
            _combat_card(
                "shiv",
                card_type="attack",
                effects={"damage": 4},
                tags=("shiv",),
            ),
        ),
        player_statuses={"accuracy": 2, "afterimage": 1},
    )

    state = _play_card(state, "shiv")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == 44
    assert state.combat.player.block == 1


def test_focus_is_stored_without_non_orb_combat_side_effects() -> None:
    state = _enter_status_combat(
        (_combat_card("plain_guard", card_type="skill", effects={"block": 5}, target="Self"),),
        player_statuses={"focus": 3},
    )

    state = _play_card(state, "plain_guard")

    assert state.combat is not None
    assert state.combat.player.block == 5
    assert state.combat.player.statuses["focus"] == 3
