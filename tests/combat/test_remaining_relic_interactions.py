from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    MapEdgeState,
    MapNodeState,
    MapState,
    RoomKind,
    RunPhase,
)
from sts2sim.mechanics import start_of_combat

DUMMY_MONSTERS = (
    {
        "id": "RELIC_REGRESSION_DUMMY",
        "name": "Relic Regression Dummy",
        "type": "Normal",
        "min_hp": 50,
        "max_hp": 50,
        "moves": (
            {
                "id": "WAIT",
                "name": "Wait",
                "intent": "Unknown",
            },
        ),
        "attack_pattern": {
            "initial_move": "WAIT",
            "states": ({"id": "WAIT_MOVE", "move_id": "WAIT", "type": "move"},),
        },
    },
)

DUMMY_ENCOUNTERS = (
    {
        "id": "RELIC_REGRESSION_ENCOUNTER",
        "name": "Relic Regression Dummy",
        "act": "Act 1 - Overgrowth",
        "room_type": "Monster",
        "is_weak": True,
        "monsters": ({"id": "RELIC_REGRESSION_DUMMY"},),
    },
)


def _card_spec(
    card_id: str,
    *,
    card_type: str = "Skill",
    target: str = "Self",
    cost: int = 0,
    damage: int | None = None,
    block: int | None = None,
    custom: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "id": card_id,
        "name": card_id.replace("_", " ").title(),
        "type": card_type,
        "target": target,
        "cost": cost,
    }
    if damage is not None:
        spec["damage"] = damage
    if block is not None:
        spec["block"] = block
    if custom is not None:
        spec["custom"] = dict(custom)
    return spec


def _choose_first_ancient(state):
    action = next(action for action in legal_actions(state) if action.type == "choose_ancient")
    return step(state, action)


def _force_next_room(state, room_kind: RoomKind = RoomKind.MONSTER):
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


def _enter_combat(
    deck: Sequence[Mapping[str, Any]],
    *,
    relics: Sequence[str],
    draw_per_turn: int = 5,
    potions: Sequence[str] = (),
    player: Mapping[str, Any] | None = None,
):
    state = new_run(
        seed=9301,
        character_id="TEST",
        ascension=0,
        source_data={
            "deck": tuple(dict(card) for card in deck),
            "player": {
                "hp": 80,
                "max_hp": 80,
                "energy": 3,
                "max_energy": 3,
                **dict(player or {}),
            },
            "monsters": DUMMY_MONSTERS,
            "encounters": DUMMY_ENCOUNTERS,
            "combat_encounter_id": "RELIC_REGRESSION_ENCOUNTER",
            "draw_per_turn": draw_per_turn,
        },
    )
    state = _choose_first_ancient(state)
    updates: dict[str, Any] = {"relics": tuple(relics), "potions": tuple(potions)}
    if player is not None:
        updates["player"] = state.player.model_copy(update=dict(player))
    state = state.model_copy(update=updates)
    state = _force_next_room(state)
    return step(
        state,
        next(action for action in legal_actions(state) if action.type == "choose_node"),
    )


def _end_turn(state):
    return step(
        state,
        next(action for action in legal_actions(state) if action.type == "end_turn"),
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


def _use_potion_action(state, potion_slot: str = "potion:0", target_id: str | None = None):
    return next(
        action
        for action in legal_actions(state)
        if action.type == "use_potion"
        and action.payload.get("potion_slot") == potion_slot
        and (target_id is None or action.target_id == target_id)
    )


def _event_kinds(state) -> list[str]:
    assert state.combat is not None
    return [event.kind for event in state.combat.last_events]


def test_exhaust_relic_moves_top_draw_card_and_emits_exhaust_event() -> None:
    deck = tuple(_card_spec(f"draw_card_{index}") for index in range(4))

    state = _enter_combat(deck, relics=("toasty_mittens",), draw_per_turn=1)

    assert state.combat is not None
    assert len(state.combat.exhaust_pile) == 1
    assert state.combat.player.statuses["strength"] == 1
    assert "draw_pile_cards_exhausted" in _event_kinds(state)
    assert any(
        event.kind == "card_exhausted"
        and event.metadata.get("reason") == "relic_exhaust_top_draw_pile"
        for event in state.combat.last_events
    )


def test_discard_relic_retains_hand_instead_of_end_turn_discard() -> None:
    state = _enter_combat(
        (_card_spec("keep_one"), _card_spec("keep_two")),
        relics=("runic_pyramid",),
        draw_per_turn=2,
    )
    assert state.combat is not None
    starting_ids = [card.instance_id for card in state.combat.hand]

    state = _end_turn(state)

    assert state.combat is not None
    assert [card.instance_id for card in state.combat.hand] == starting_ids
    assert state.combat.discard_pile == ()
    assert any(
        event.kind == "hand_retained"
        and event.metadata.get("relic_id") == "runic_pyramid"
        for event in state.combat.last_events
    )


def test_potion_use_relic_heals_after_consuming_a_potion() -> None:
    state = _enter_combat(
        (),
        relics=("toy_ornithopter",),
        potions=("fire_potion",),
        draw_per_turn=0,
        player={"hp": 70},
    )
    assert state.combat is not None
    monster_id = state.combat.monsters[0].monster_id

    state = step(state, _use_potion_action(state, target_id=monster_id))

    assert state.potions == ()
    assert state.combat is not None
    assert state.combat.player.hp == 75
    assert any(
        event.kind == "trigger_potion_use_heal"
        and event.source_id == "toy_ornithopter"
        for event in state.combat.last_events
    )


def test_shuffle_and_block_relics_apply_in_the_same_short_combat() -> None:
    state = _enter_combat(
        (),
        relics=("blessed_antler", "orichalcum"),
        draw_per_turn=0,
    )

    assert state.combat is not None
    assert [card.card_id for card in state.combat.draw_pile[-3:]] == [
        "dazed",
        "dazed",
        "dazed",
    ]
    assert any(
        event.kind == "status_cards_shuffled_into_draw_pile"
        and event.metadata.get("relic_id") == "blessed_antler"
        for event in state.combat.last_events
    )

    state = _end_turn(state)

    assert state.combat is not None
    assert any(
        event.kind == "player_block"
        and event.metadata.get("relic_id") == "orichalcum"
        and event.metadata.get("condition") == "player_block_is_zero"
        and event.amount == 6
        for event in state.combat.last_events
    )


def test_card_created_relic_adds_luminesce_with_exhaust() -> None:
    state = _enter_combat((), relics=("radiant_pearl",), draw_per_turn=0)

    assert state.combat is not None
    assert [card.card_id for card in state.combat.hand] == ["luminesce"]
    luminesce = state.combat.hand[0]
    assert luminesce.exhausts is True
    assert luminesce.type.value == "skill"
    assert any(
        event.kind == "relic_cards_added"
        and event.metadata.get("relic_id") == "radiant_pearl"
        and event.metadata.get("card_ids") == ["luminesce"]
        for event in state.combat.last_events
    )


def test_resource_relic_grant_can_be_spent_by_star_cost_card() -> None:
    state = _enter_combat(
        (
            _card_spec(
                "star_spender",
                card_type="Attack",
                target="Enemy",
                damage=1,
                custom={"star_cost": 2},
            ),
        ),
        relics=("divine_right",),
        draw_per_turn=1,
    )

    assert state.combat is not None
    assert state.combat.player.resources["star"] == 3

    state = _play_card(state, "star_spender")

    assert state.combat is not None
    assert state.combat.player.resources["star"] == 1
    assert any(
        event.kind == "player_resource_spent"
        and event.metadata.get("resource") == "star"
        and event.amount == 2
        for event in state.combat.last_events
    )


def test_first_power_and_empty_hand_relic_conditions_are_independent() -> None:
    state = _enter_combat(
        (
            _card_spec("opening_power", card_type="Power"),
        ),
        relics=("permafrost", "screaming_flagon"),
        draw_per_turn=1,
    )
    assert state.combat is not None

    state = _play_card(state, "opening_power")
    assert state.combat is not None
    assert state.combat.player.block == 7
    assert any(
        event.kind == "player_block"
        and event.metadata.get("relic_id") == "permafrost"
        and event.metadata.get("condition") == "first_power_this_combat"
        for event in state.combat.last_events
    )

    hp_before = state.combat.monsters[0].hp
    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.monsters[0].hp == hp_before - 20
    assert any(
        event.kind == "monster_damaged"
        and event.metadata.get("relic_id") == "screaming_flagon"
        and event.metadata.get("condition") == "empty_hand"
        for event in state.combat.last_events
    )


def test_start_combat_choice_and_reward_style_relics_emit_actionable_markers() -> None:
    result = start_of_combat(("choices_paradox", "toolbox", "vexing_puzzlebox"))

    assert [
        (marker.relic_id, marker.kind, marker.amount, marker.metadata)
        for marker in result.markers
    ] == [
        (
            "choices_paradox",
            "add_card_to_hand",
            1,
            {"selection": "choose_one_of_random", "choice_count": 5, "retain_once": True},
        ),
        (
            "toolbox",
            "add_card_to_hand",
            1,
            {
                "selection": "choose_one_of_random",
                "choice_count": 3,
                "card_pool": "colorless",
            },
        ),
        (
            "vexing_puzzlebox",
            "add_card_to_hand",
            1,
            {"selection": "random", "free_to_play_this_turn": True},
        ),
    ]
