from __future__ import annotations

from random import Random

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    MapEdgeState,
    MapNodeState,
    MapState,
    RoomKind,
    RunPhase,
)
from sts2sim.mechanics import build_monster_definitions, next_monster_move

MONSTERS = (
    {
        "id": "TRAINING_AUTOMATON",
        "name": "Training Automaton",
        "type": "Normal",
        "min_hp": 30,
        "max_hp": 30,
        "min_hp_ascension": 40,
        "max_hp_ascension": 40,
        "moves": (
            {
                "id": "DOUBLE_STRIKE",
                "name": "Double Strike",
                "intent": "Attack",
                "damage": {"normal": 5, "ascension": 7, "hit_count": 2},
                "block": None,
                "heal": None,
                "powers": None,
            },
            {
                "id": "FORTIFY",
                "name": "Fortify",
                "intent": "Defend + Buff",
                "damage": None,
                "block": 4,
                "heal": None,
                "powers": ({"power_id": "STRENGTH", "amount": 2, "target": "self"},),
            },
        ),
        "attack_pattern": {
            "initial_move": "DOUBLE_STRIKE",
            "states": (
                {
                    "id": "DOUBLE_STRIKE_MOVE",
                    "move_id": "DOUBLE_STRIKE",
                    "next": "FORTIFY_MOVE",
                    "type": "move",
                },
                {
                    "id": "FORTIFY_MOVE",
                    "move_id": "FORTIFY",
                    "next": "DOUBLE_STRIKE_MOVE",
                    "type": "move",
                },
            ),
            "type": "cycle",
        },
    },
)

ENCOUNTERS = (
    {
        "id": "TRAINING_AUTOMATON_ENCOUNTER",
        "name": "Training Automaton",
        "act": "Act 1 - Overgrowth",
        "room_type": "Monster",
        "is_weak": True,
        "monsters": ({"id": "TRAINING_AUTOMATON"},),
    },
)

KNOWLEDGE_DEMON = (
    {
        "id": "KNOWLEDGE_DEMON",
        "name": "Knowledge Demon",
        "type": "Boss",
        "min_hp": 200,
        "max_hp": 200,
        "moves": (
            {
                "id": "CURSE_OF_KNOWLEDGE",
                "name": "Curse of Knowledge",
                "intent": "Debuff",
            },
            {
                "id": "SLAP",
                "name": "Slap",
                "intent": "Attack",
                "damage": {"normal": 17, "ascension": 18},
            },
            {
                "id": "KNOWLEDGE_OVERWHELMING",
                "name": "Knowledge Overwhelming",
                "intent": "Attack",
                "damage": {"normal": 8, "ascension": 9, "hit_count": 3},
            },
            {
                "id": "PONDER",
                "name": "Ponder",
                "intent": "Attack + Buff",
                "damage": {"normal": 11, "ascension": 13},
                "heal": 30,
                "powers": ({"power_id": "STRENGTH", "amount": 2, "target": "self"},),
            },
        ),
        "attack_pattern": {
            "initial_move": "CURSE_OF_KNOWLEDGE",
            "states": (
                {
                    "id": "CURSE_OF_KNOWLEDGE_MOVE",
                    "move_id": "CURSE_OF_KNOWLEDGE",
                    "next": "SLAP_MOVE",
                    "type": "move",
                },
                {
                    "id": "SLAP_MOVE",
                    "move_id": "SLAP",
                    "next": "KNOWLEDGE_OVERWHELMING_MOVE",
                    "type": "move",
                },
                {
                    "id": "KNOWLEDGE_OVERWHELMING_MOVE",
                    "move_id": "KNOWLEDGE_OVERWHELMING",
                    "next": "PONDER_MOVE",
                    "type": "move",
                },
                {
                    "id": "PONDER_MOVE",
                    "move_id": "PONDER",
                    "next": "CURSE_BRANCH",
                    "type": "move",
                },
                {
                    "id": "CURSE_BRANCH",
                    "type": "conditional",
                    "branches": (
                        {
                            "condition": "_curseOfKnowledgeCounter < 3",
                            "move_id": "CURSE_OF_KNOWLEDGE",
                        },
                        {
                            "condition": "_curseOfKnowledgeCounter >= 3",
                            "move_id": "SLAP",
                        },
                    ),
                },
            ),
            "type": "cycle",
        },
    },
)

KNOWLEDGE_DEMON_ENCOUNTER = (
    {
        "id": "KNOWLEDGE_DEMON_ENCOUNTER",
        "name": "Knowledge Demon",
        "room_type": "Monster",
        "monsters": ({"id": "KNOWLEDGE_DEMON"},),
    },
)

TEST_SUBJECT = (
    {
        "id": "TEST_SUBJECT",
        "name": "Test Subject #C14",
        "type": "Boss",
        "min_hp": None,
        "max_hp": None,
        "innate_powers": ({"power_id": "ENRAGE", "amount": 2, "amount_ascension": 3},),
        "moves": (
            {
                "id": "RESPAWN",
                "name": "Respawn",
                "intent": "Heal + Buff",
                "powers": (
                    {"power_id": "PAINFUL_STABS", "amount": 1, "target": "self"},
                    {"power_id": "NEMESIS", "amount": 1, "target": "self"},
                ),
            },
            {"id": "BITE", "name": "Bite", "intent": "Attack", "damage": {"normal": 20}},
            {
                "id": "SKULL_BASH",
                "name": "Skull Bash",
                "intent": "Attack + Debuff",
                "damage": {"normal": 14},
                "powers": ({"power_id": "VULNERABLE", "amount": 1, "target": "player"},),
            },
            {
                "id": "MULTI_CLAW",
                "name": "Multi Claw",
                "intent": "Attack",
                "damage": {"normal": 10},
            },
            {
                "id": "PHASE3_LACERATE",
                "name": "Phase3 Lacerate",
                "intent": "Attack",
                "damage": {"normal": 10, "hit_count": 3},
            },
            {
                "id": "BIG_POUNCE",
                "name": "Big Pounce",
                "intent": "Attack",
                "damage": {"normal": 45},
            },
            {
                "id": "BURNING_GROWL",
                "name": "Burning Growl",
                "intent": "Status + Buff",
                "powers": ({"power_id": "STRENGTH", "amount": 2, "target": "self"},),
            },
        ),
        "attack_pattern": {
            "initial_move": "BITE",
            "states": (
                {"id": "BITE_MOVE", "move_id": "BITE", "next": "SKULL_BASH_MOVE", "type": "move"},
                {
                    "id": "SKULL_BASH_MOVE",
                    "move_id": "SKULL_BASH",
                    "next": "BITE_MOVE",
                    "type": "move",
                },
                {
                    "id": "MULTI_CLAW_MOVE",
                    "move_id": "MULTI_CLAW",
                    "next": "MULTI_CLAW_MOVE",
                    "type": "move",
                },
                {
                    "id": "PHASE3_LACERATE_MOVE",
                    "move_id": "PHASE3_LACERATE",
                    "next": "BIG_POUNCE_MOVE",
                    "type": "move",
                },
                {
                    "id": "BIG_POUNCE_MOVE",
                    "move_id": "BIG_POUNCE",
                    "next": "BURNING_GROWL_MOVE",
                    "type": "move",
                },
                {
                    "id": "BURNING_GROWL_MOVE",
                    "move_id": "BURNING_GROWL",
                    "next": "PHASE3_LACERATE_MOVE",
                    "type": "move",
                },
                {
                    "id": "RESPAWN_MOVE",
                    "move_id": "RESPAWN",
                    "next": "REVIVE_BRANCH",
                    "type": "move",
                    "must_perform_once": True,
                },
                {
                    "id": "REVIVE_BRANCH",
                    "type": "conditional",
                    "branches": (
                        {"condition": "Respawns < 2", "move_id": "MULTI_CLAW"},
                        {"condition": "Respawns >= 2", "move_id": "PHASE3_LACERATE"},
                    ),
                },
            ),
        },
    },
)

TEST_SUBJECT_ENCOUNTER = (
    {
        "id": "TEST_SUBJECT_BOSS",
        "name": "Test Subject",
        "room_type": "Boss",
        "monsters": ({"id": "TEST_SUBJECT"},),
    },
)

WATERFALL_GIANT = (
    {
        "id": "WATERFALL_GIANT",
        "name": "Waterfall Giant",
        "type": "Boss",
        "min_hp": 240,
        "max_hp": None,
        "min_hp_ascension": 250,
        "max_hp_ascension": None,
        "moves": (
            {
                "id": "PRESSURIZE",
                "name": "Pressurize",
                "intent": "Buff",
                "powers": ({"power_id": "STEAM_ERUPTION", "amount": 15, "target": "self"},),
            },
            {
                "id": "STOMP",
                "name": "Stomp",
                "intent": "Attack + Debuff + Buff",
                "damage": {"normal": 15, "ascension": 16},
                "powers": (
                    {"power_id": "WEAK", "amount": 1, "target": "player"},
                    {"power_id": "STEAM_ERUPTION", "amount": 3, "target": "self"},
                ),
            },
            {
                "id": "RAM",
                "name": "Ram",
                "intent": "Attack + Buff",
                "damage": {"normal": 10, "ascension": 11},
                "powers": ({"power_id": "STEAM_ERUPTION", "amount": 3, "target": "self"},),
            },
            {
                "id": "SIPHON",
                "name": "Siphon",
                "intent": "Heal + Buff",
                "powers": ({"power_id": "STEAM_ERUPTION", "amount": 3, "target": "self"},),
            },
            {
                "id": "PRESSURE_GUN",
                "name": "Pressure Gun",
                "intent": "Attack + Buff",
                "powers": ({"power_id": "STEAM_ERUPTION", "amount": 3, "target": "self"},),
            },
            {
                "id": "PRESSURE_UP",
                "name": "Pressure Up",
                "intent": "Attack + Buff",
                "damage": {"normal": 13, "ascension": 14},
                "powers": ({"power_id": "STEAM_ERUPTION", "amount": 3, "target": "self"},),
            },
            {"id": "ABOUT_TO_BLOW", "name": "About To Blow", "intent": "Stun"},
            {"id": "EXPLODE", "name": "Explode", "intent": "Special"},
        ),
        "attack_pattern": {
            "initial_move": "PRESSURIZE",
            "states": (
                {
                    "id": "PRESSURIZE_MOVE",
                    "move_id": "PRESSURIZE",
                    "next": "STOMP_MOVE",
                    "type": "move",
                },
                {"id": "STOMP_MOVE", "move_id": "STOMP", "next": "RAM_MOVE", "type": "move"},
                {"id": "RAM_MOVE", "move_id": "RAM", "next": "SIPHON_MOVE", "type": "move"},
                {
                    "id": "SIPHON_MOVE",
                    "move_id": "SIPHON",
                    "next": "PRESSURE_GUN_MOVE",
                    "type": "move",
                },
                {
                    "id": "PRESSURE_GUN_MOVE",
                    "move_id": "PRESSURE_GUN",
                    "next": "PRESSURE_UP_MOVE",
                    "type": "move",
                },
                {
                    "id": "PRESSURE_UP_MOVE",
                    "move_id": "PRESSURE_UP",
                    "next": "STOMP_MOVE",
                    "type": "move",
                },
                {
                    "id": "ABOUT_TO_BLOW_MOVE",
                    "move_id": "ABOUT_TO_BLOW",
                    "next": "EXPLODE_MOVE",
                    "type": "move",
                    "must_perform_once": True,
                },
                {
                    "id": "EXPLODE_MOVE",
                    "move_id": "EXPLODE",
                    "next": "EXPLODE_MOVE",
                    "type": "move",
                },
            ),
        },
    },
)

WATERFALL_GIANT_ENCOUNTER = (
    {
        "id": "WATERFALL_GIANT_BOSS",
        "name": "Waterfall Giant",
        "room_type": "Boss",
        "monsters": ({"id": "WATERFALL_GIANT"},),
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


def _enter_training_combat(*, ascension: int = 0, deck: tuple[dict, ...] = ()):
    source_data = _source_data(
        MONSTERS,
        ENCOUNTERS,
        "TRAINING_AUTOMATON_ENCOUNTER",
        deck=deck,
    )
    return _enter_source_combat(source_data, ascension=ascension)


def _enter_knowledge_demon_combat(*, ascension: int = 0):
    source_data = _source_data(
        KNOWLEDGE_DEMON,
        KNOWLEDGE_DEMON_ENCOUNTER,
        "KNOWLEDGE_DEMON_ENCOUNTER",
        deck=tuple(
            {
                "card_id": f"defend_{index}",
                "name": "Defend",
                "type": "skill",
                "cost": 1,
                "target": "self",
                "effects": {"block": 5},
            }
            for index in range(12)
        ),
    )
    return _enter_source_combat(source_data, ascension=ascension)


def _enter_test_subject_combat(*, deck: tuple[dict, ...] = (), ascension: int = 0):
    source_data = _source_data(
        TEST_SUBJECT,
        TEST_SUBJECT_ENCOUNTER,
        "TEST_SUBJECT_BOSS",
        deck=deck,
    )
    return _enter_source_combat(source_data, ascension=ascension)


def _enter_waterfall_giant_combat(*, deck: tuple[dict, ...] = (), ascension: int = 0):
    source_data = _source_data(
        WATERFALL_GIANT,
        WATERFALL_GIANT_ENCOUNTER,
        "WATERFALL_GIANT_BOSS",
        deck=deck,
    )
    return _enter_source_combat(source_data, ascension=ascension)


def _source_data(
    monsters: tuple[dict, ...],
    encounters: tuple[dict, ...],
    encounter_id: str,
    *,
    deck: tuple[dict, ...] = (),
) -> dict:
    return {
        "monsters": monsters,
        "encounters": encounters,
        "combat_encounter_id": encounter_id,
        "deck": deck
        or (
            {
                "card_id": "defend",
                "name": "Defend",
                "type": "skill",
                "cost": 1,
                "target": "self",
                "effects": {"block": 5},
            },
        ),
        "player": {"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3},
    }


def _enter_source_combat(source_data: dict, *, ascension: int = 0):
    state = new_run(
        seed=4100 + ascension,
        character_id="TEST",
        ascension=ascension,
        source_data=source_data,
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.MONSTER)
    action = next(action for action in legal_actions(state) if action.type == "choose_node")
    return step(state, action)


def _end_turn_action(state):
    return next(action for action in legal_actions(state) if action.type == "end_turn")


def _play_first_card(state, card_id: str):
    action = next(
        action
        for action in legal_actions(state)
        if action.type == "play_card" and action.card_instance_id
        and any(
            card.instance_id == action.card_instance_id and card.card_id == card_id
            for card in state.combat.hand
        )
    )
    return step(state, action)


def _play_card_targeting(state, card_id: str, target_id: str):
    action = next(
        action
        for action in legal_actions(state)
        if action.type == "play_card"
        and action.target_id == target_id
        and action.card_instance_id
        and any(
            card.instance_id == action.card_instance_id and card.card_id == card_id
            for card in state.combat.hand
        )
    )
    return step(state, action)


def _choose_generated_pending_card(state, card_id: str):
    assert state.combat is not None
    choice = state.combat.pending_choices[0]
    generated = choice.metadata["generated_cards"]
    instance_id = next(card["instance_id"] for card in generated if card["card_id"] == card_id)
    action = next(
        action
        for action in legal_actions(state)
        if action.type == "choose_card" and action.card_instance_id == instance_id
    )
    return step(state, action)


def _execute_card(card_id: str = "execute", damage: int = 150) -> dict:
    return {
        "card_id": card_id,
        "name": "Execute",
        "type": "attack",
        "cost": 0,
        "target": "enemy",
        "effects": {"damage": damage},
    }


def test_combat_spawns_source_monster_with_scaled_initial_intent() -> None:
    state = _enter_training_combat()

    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.monster_id == "TRAINING_AUTOMATON"
    assert monster.hp == 30
    assert monster.move_id == "DOUBLE_STRIKE"
    assert monster.intent_damage == 10
    assert monster.hit_count == 2
    assert monster.metadata["encounter_id"] == "TRAINING_AUTOMATON_ENCOUNTER"

    ascension_state = _enter_training_combat(ascension=7)
    assert ascension_state.combat is not None
    ascension_monster = ascension_state.combat.monsters[0]
    assert ascension_monster.hp == 40
    assert ascension_monster.intent_damage == 14


def test_stranded_won_combat_can_proceed_to_reward() -> None:
    state = _enter_training_combat()
    assert state.combat is not None
    defeated = state.combat.monsters[0].model_copy(update={"hp": 0})
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (defeated,)})}
    )

    proceed = next(action for action in legal_actions(state) if action.type == "proceed")
    state = step(state, proceed)

    assert state.phase is RunPhase.REWARD
    assert state.reward is not None


def test_stranded_lost_combat_can_proceed_to_failed_state() -> None:
    state = _enter_training_combat()
    assert state.combat is not None
    player = state.combat.player.model_copy(update={"hp": 0})
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"player": player})}
    )

    proceed = next(action for action in legal_actions(state) if action.type == "proceed")
    state = step(state, proceed)

    assert state.phase is RunPhase.FAILED


def test_test_subject_starts_at_phase_one_hp_and_respawns_to_phase_two() -> None:
    state = _enter_test_subject_combat(deck=(_execute_card(damage=150),))

    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.hp == 100
    assert monster.max_hp == 100
    assert monster.move_id == "BITE"
    assert monster.statuses["enrage"] == 2

    state = _play_first_card(state, "execute")

    assert state.phase is RunPhase.COMBAT
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.hp == 200
    assert monster.max_hp == 200
    assert monster.move_id == "MULTI_CLAW"
    assert monster.intent_damage == 10
    assert monster.statuses["painful_stabs"] == 1
    assert monster.statuses["nemesis"] == 1
    assert "enrage" not in monster.statuses
    assert monster.metadata["test_subject_respawns"] == 1
    assert any(event.kind == "test_subject_respawned" for event in state.combat.last_events)


def test_test_subject_second_respawn_enters_phase_three_branch() -> None:
    state = _enter_test_subject_combat(
        deck=(
            _execute_card("execute", damage=150),
            _execute_card("execute", damage=250),
        )
    )

    state = _play_first_card(state, "execute")
    state = _play_first_card(state, "execute")

    assert state.phase is RunPhase.COMBAT
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.hp == 300
    assert monster.max_hp == 300
    assert monster.move_id == "PHASE3_LACERATE"
    assert monster.hit_count == 3
    assert monster.intent_damage == 30
    assert monster.statuses["painful_stabs"] == 1
    assert monster.statuses["nemesis"] == 1
    assert monster.statuses["intangible"] == 1
    assert monster.metadata["test_subject_respawns"] == 2
    assert any(
        event.kind == "monster_nemesis_turn" and event.metadata.get("trigger") == "respawn"
        for event in state.combat.last_events
    )


def test_test_subject_painful_stabs_adds_wounds_on_unblocked_attack_damage() -> None:
    state = _enter_test_subject_combat(deck=(_execute_card(damage=150),))
    state = _play_first_card(state, "execute")

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert state.combat.player.hp == 70
    all_piles = (
        state.combat.hand
        + state.combat.draw_pile
        + state.combat.discard_pile
        + state.combat.exhaust_pile
    )
    assert any(card.card_id == "wound" for card in all_piles)
    assert any(
        event.kind == "card_created" and event.metadata.get("reason") == "painful_stabs"
        for event in state.combat.last_events
    )


def test_test_subject_painful_stabs_respects_full_block() -> None:
    state = _enter_test_subject_combat(deck=(_execute_card(damage=150),))
    state = _play_first_card(state, "execute")
    assert state.combat is not None
    player = state.combat.player.model_copy(update={"block": 20})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"player": player})})

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert state.combat.player.hp == 80
    all_piles = (
        state.combat.hand
        + state.combat.draw_pile
        + state.combat.discard_pile
        + state.combat.exhaust_pile
    )
    assert not any(card.card_id == "wound" for card in all_piles)
    assert not any(
        event.kind == "card_created" and event.metadata.get("reason") == "painful_stabs"
        for event in state.combat.last_events
    )


def test_test_subject_respawn_counter_selects_phase_two_and_three_moves() -> None:
    definition = build_monster_definitions(TEST_SUBJECT)["TEST_SUBJECT"]

    assert (
        next_monster_move(
            definition,
            "RESPAWN",
            Random(1),
            move_counts={"RESPAWNS": 1},
        ).move_id
        == "MULTI_CLAW"
    )
    assert (
        next_monster_move(
            definition,
            "RESPAWN",
            Random(1),
            move_counts={"RESPAWNS": 2},
        ).move_id
        == "PHASE3_LACERATE"
    )


def test_waterfall_giant_can_die_before_steam_eruption_starts() -> None:
    state = _enter_waterfall_giant_combat(deck=(_execute_card(damage=300),))

    state = _play_first_card(state, "execute")

    assert state.phase is RunPhase.REWARD
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.hp == 0
    assert not any(
        event.kind == "waterfall_giant_about_to_blow" for event in state.combat.last_events
    )


def test_waterfall_giant_cycle_heals_and_shows_dynamic_pressure_gun_damage() -> None:
    state = _enter_waterfall_giant_combat()

    state = step(state, _end_turn_action(state))
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.statuses["steam_eruption"] == 15
    assert monster.move_id == "STOMP"
    assert monster.intent_damage == 15

    state = step(state, _end_turn_action(state))
    state = step(state, _end_turn_action(state))
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.statuses["steam_eruption"] == 21
    assert monster.move_id == "SIPHON"

    wounded = monster.model_copy(update={"hp": 220})
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (wounded,)})}
    )
    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.hp == 230
    assert monster.statuses["steam_eruption"] == 24
    assert monster.move_id == "PRESSURE_GUN"
    assert monster.intent_damage == 20

    state = step(state, _end_turn_action(state))
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert state.combat.player.hp == 35
    assert monster.metadata["move_counts"]["PRESSURE_GUN"] == 1
    assert monster.move_id == "PRESSURE_UP"


def test_waterfall_giant_siphon_uses_ascension_heal_amount() -> None:
    state = _enter_waterfall_giant_combat(ascension=4)

    state = step(state, _end_turn_action(state))
    state = step(state, _end_turn_action(state))
    state = step(state, _end_turn_action(state))
    assert state.combat is not None
    wounded = state.combat.monsters[0].model_copy(update={"hp": 220})
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (wounded,)})}
    )

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert state.combat.monsters[0].hp == 235


def test_waterfall_giant_death_triggers_countdown_then_explosion() -> None:
    state = _enter_waterfall_giant_combat(
        deck=(
            _execute_card("execute", damage=300),
            _execute_card("execute", damage=300),
        )
    )

    state = step(state, _end_turn_action(state))
    state = _play_first_card(state, "execute")

    assert state.phase is RunPhase.COMBAT
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.hp == 1
    assert monster.move_id == "ABOUT_TO_BLOW"
    assert monster.metadata["waterfall_giant_about_to_blow"] is True
    assert any(event.kind == "waterfall_giant_about_to_blow" for event in state.combat.last_events)

    state = _play_first_card(state, "execute")
    assert state.combat is not None
    assert state.combat.monsters[0].hp == 1

    state = step(state, _end_turn_action(state))
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.hp == 1
    assert monster.move_id == "EXPLODE"
    assert monster.intent_damage == 0

    player = state.combat.player.model_copy(update={"block": 20})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"player": player})})
    state = step(state, _end_turn_action(state))

    assert state.phase is RunPhase.REWARD
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.hp == 0
    assert monster.metadata["waterfall_giant_exploded"] is True
    assert state.combat.player.hp == 80
    assert any(event.kind == "waterfall_giant_exploded" for event in state.combat.last_events)


def test_monster_turn_applies_multi_hit_damage_block_buff_and_next_intent() -> None:
    state = _enter_training_combat()
    assert state.combat is not None

    state = step(state, _end_turn_action(state))
    assert state.combat is not None
    assert state.combat.player.hp == 70
    monster = state.combat.monsters[0]
    assert monster.move_id == "FORTIFY"
    assert monster.intent_block == 4
    assert monster.intent_damage == 0

    state = step(state, _end_turn_action(state))
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.block == 4
    assert monster.statuses["strength"] == 2
    assert monster.move_id == "DOUBLE_STRIKE"
    assert monster.intent_damage == 14


def test_knowledge_demon_forced_mind_rot_choice_resumes_next_turn() -> None:
    state = _enter_knowledge_demon_combat()
    assert state.combat is not None
    assert state.combat.monsters[0].move_id == "CURSE_OF_KNOWLEDGE"

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert state.combat.turn == 1
    assert state.combat.pending_choices
    choice = state.combat.pending_choices[0]
    assert choice.kind == "knowledge_demon_curse"
    assert choice.required is True
    assert state.combat.metadata["resume_after_pending_choice"] == "player_turn_start"
    assert state.combat.metadata["knowledge_demon_curse_counter"] == 1
    assert {
        card["card_id"] for card in choice.metadata["generated_cards"]
    } == {"disintegration", "mind_rot"}

    state = _choose_generated_pending_card(state, "mind_rot")

    assert state.combat is not None
    assert state.combat.turn == 2
    assert not state.combat.pending_choices
    assert state.combat.player.statuses["mind_rot"] == 1
    assert len(state.combat.hand) == 4
    assert state.combat.monsters[0].move_id == "SLAP"
    assert any(event.kind == "draw_reduced" for event in state.combat.last_events)


def test_knowledge_demon_disintegration_deals_end_turn_damage() -> None:
    state = _enter_knowledge_demon_combat()
    assert state.combat is not None

    state = step(state, _end_turn_action(state))
    state = _choose_generated_pending_card(state, "disintegration")

    assert state.combat is not None
    assert state.combat.player.statuses["disintegration"] == 6
    hp_before = state.combat.player.hp

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert state.combat.player.hp == hp_before - 23
    assert any(
        event.kind == "player_damaged"
        and event.source_id == "disintegration"
        and event.amount == 6
        for event in state.combat.last_events
    )


def test_sloth_and_waste_away_turn_start_limits() -> None:
    state = _enter_training_combat()
    assert state.combat is not None
    combat = state.combat.model_copy(
        update={
            "player": state.combat.player.model_copy(
                update={"statuses": {"sloth": 3, "waste_away": 1}}
            )
        }
    )
    state = state.model_copy(update={"combat": combat})

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert state.combat.player.energy == 2
    assert state.combat.metadata["turn_card_play_limit"] == 3
    assert any(
        event.kind == "energy_changed"
        and event.source_id == "waste_away"
        and event.amount == -1
        for event in state.combat.last_events
    )
    assert any(
        event.kind == "turn_card_play_limit_added"
        and event.source_id == "sloth"
        and event.amount == 3
        for event in state.combat.last_events
    )


def test_knowledge_demon_curse_branch_stops_after_three_uses() -> None:
    definition = build_monster_definitions(KNOWLEDGE_DEMON)["KNOWLEDGE_DEMON"]

    curse_again = next_monster_move(
        definition,
        "PONDER",
        Random(1),
        move_counts={"CURSE_OF_KNOWLEDGE": 2},
    )
    skip_curse = next_monster_move(
        definition,
        "PONDER",
        Random(1),
        move_counts={"CURSE_OF_KNOWLEDGE": 3},
    )

    assert curse_again is not None
    assert curse_again.move_id == "CURSE_OF_KNOWLEDGE"
    assert skip_curse is not None
    assert skip_curse.move_id == "SLAP"


def test_monster_attack_hits_block_then_osty_before_player_hp() -> None:
    state = _enter_training_combat()
    assert state.combat is not None
    combat = state.combat.model_copy(
        update={
            "player": state.combat.player.model_copy(update={"block": 3}),
            "metadata": {
                **state.combat.metadata,
                "osty": {"alive": True, "hp": 4, "max_hp": 4},
            },
        }
    )
    state = state.model_copy(update={"combat": combat})

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert state.combat.player.hp == 77
    assert state.combat.metadata["osty"]["alive"] is False
    assert state.combat.metadata["osty"]["hp"] == 0
    osty_absorbs = [
        event for event in state.combat.last_events if event.kind == "osty_damage_absorbed"
    ]
    assert [event.amount for event in osty_absorbs] == [2, 2]
    assert [event.metadata["blocked"] for event in osty_absorbs] == [3, 0]
    assert any(
        event.kind == "player_damaged"
        and event.amount == 3
        and event.metadata["osty_absorbed"] == 2
        for event in state.combat.last_events
    )


def test_non_osty_combat_damage_still_hits_player_after_block() -> None:
    state = _enter_training_combat()
    assert state.combat is not None
    combat = state.combat.model_copy(
        update={"player": state.combat.player.model_copy(update={"block": 3})}
    )
    state = state.model_copy(update={"combat": combat})

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert state.combat.player.hp == 73
    assert "osty" not in state.combat.metadata


def test_end_turn_cycles_discard_back_into_draw_pile() -> None:
    state = _enter_training_combat(
        deck=(
            {
                "card_id": "small_block",
                "name": "Small Block",
                "type": "skill",
                "cost": 1,
                "target": "self",
                "effects": {"block": 3},
            },
        )
    )

    assert state.combat is not None
    assert [card.card_id for card in state.combat.hand] == ["small_block"]
    assert state.combat.draw_pile == ()

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert [card.card_id for card in state.combat.hand] == ["small_block"]
    assert any(event.kind == "discard_shuffled" for event in state.combat.last_events)
    assert any(event.kind == "draw_pile_shuffled" for event in state.combat.last_events)


def test_use_only_once_monster_branch_is_suppressed_after_first_use() -> None:
    definition = build_monster_definitions(
        (
            {
                "id": "ONE_SHOT_BRANCHER",
                "name": "One Shot Brancher",
                "type": "Normal",
                "min_hp": 20,
                "max_hp": 20,
                "moves": (
                    {"id": "ROAR", "name": "Roar", "intent": "Debuff"},
                    {
                        "id": "CLAW",
                        "name": "Claw",
                        "intent": "Attack",
                        "damage": {"normal": 5},
                    },
                ),
                "attack_pattern": {
                    "initial_move": "ROAR_MOVE",
                    "states": (
                        {"id": "ROAR_MOVE", "type": "move", "move_id": "ROAR", "next": "RAND"},
                        {
                            "id": "RAND",
                            "type": "random",
                            "branches": (
                                {"move_id": "ROAR", "repeat": "UseOnlyOnce"},
                                {"move_id": "CLAW"},
                            ),
                        },
                    ),
                },
            },
        )
    )["ONE_SHOT_BRANCHER"]

    move = next_monster_move(
        definition,
        "ROAR",
        Random(1),
        move_counts={"ROAR": 1},
    )

    assert move is not None
    assert move.move_id == "CLAW"


def test_empty_random_monster_selector_falls_back_to_non_previous_move() -> None:
    definition = build_monster_definitions(
        (
            {
                "id": "EMPTY_RANDOM",
                "name": "Empty Random",
                "type": "Normal",
                "min_hp": 20,
                "max_hp": 20,
                "moves": (
                    {"id": "HIT", "name": "Hit", "intent": "Attack", "damage": {"normal": 5}},
                    {"id": "GROWL", "name": "Growl", "intent": "Debuff"},
                ),
                "attack_pattern": {
                    "initial_move": "HIT_MOVE",
                    "states": (
                        {"id": "HIT_MOVE", "type": "move", "move_id": "HIT", "next": "RAND"},
                        {"id": "RAND", "type": "random", "branches": ()},
                    ),
                },
            },
        )
    )["EMPTY_RANDOM"]

    move = next_monster_move(definition, "HIT", Random(1))

    assert move is not None
    assert move.move_id == "GROWL"


def test_decimillipede_segment_cycle_skips_dead_and_reattach_moves() -> None:
    definition = build_monster_definitions(
        (
            {
                "id": "DECIMILLIPEDE_SEGMENT_FRONT",
                "name": "Decimillipede Segment (Front)",
                "type": "Elite",
                "min_hp": 40,
                "max_hp": 40,
                "moves": (
                    {
                        "id": "WRITHE",
                        "name": "Writhe",
                        "intent": "Attack",
                        "damage": {"normal": 5, "hit_count": 2},
                    },
                    {
                        "id": "BULK",
                        "name": "Bulk",
                        "intent": "Attack + Buff",
                        "damage": {"normal": 6},
                    },
                    {
                        "id": "CONSTRICT",
                        "name": "Constrict",
                        "intent": "Attack + Debuff",
                        "damage": {"normal": 8},
                    },
                    {"id": "DEAD", "name": "Dead", "intent": "Unknown"},
                    {"id": "REATTACH", "name": "Reattach", "intent": "Heal"},
                ),
            },
        )
    )["DECIMILLIPEDE_SEGMENT_FRONT"]

    after_writhe = next_monster_move(definition, "WRITHE", Random(1))
    after_constrict = next_monster_move(definition, "CONSTRICT", Random(1))
    after_bulk = next_monster_move(definition, "BULK", Random(1))
    after_reattach = next_monster_move(definition, "REATTACH", Random(1))

    assert after_writhe is not None and after_writhe.move_id == "CONSTRICT"
    assert after_constrict is not None and after_constrict.move_id == "BULK"
    assert after_bulk is not None and after_bulk.move_id == "WRITHE"
    assert after_reattach is not None and after_reattach.move_id == "WRITHE"


def test_decimillipede_defeated_segment_reattaches_after_two_monster_turns() -> None:
    segment_moves = (
        {
            "id": "WRITHE",
            "name": "Writhe",
            "intent": "Attack",
            "damage": {"normal": 5, "hit_count": 2},
        },
        {
            "id": "BULK",
            "name": "Bulk",
            "intent": "Attack + Buff",
            "damage": {"normal": 6},
            "powers": ({"power_id": "STRENGTH", "amount": 2, "target": "self"},),
        },
        {
            "id": "CONSTRICT",
            "name": "Constrict",
            "intent": "Attack + Debuff",
            "damage": {"normal": 8},
            "powers": ({"power_id": "WEAK", "amount": 1, "target": "player"},),
        },
        {"id": "DEAD", "name": "Dead", "intent": "Unknown"},
        {"id": "REATTACH", "name": "Reattach", "intent": "Heal"},
    )
    monsters = tuple(
        {
            "id": monster_id,
            "name": monster_id.title(),
            "type": "Elite",
            "min_hp": 40,
            "max_hp": 40,
            "moves": segment_moves,
        }
        for monster_id in (
            "DECIMILLIPEDE_SEGMENT_BACK",
            "DECIMILLIPEDE_SEGMENT_FRONT",
            "DECIMILLIPEDE_SEGMENT_MIDDLE",
        )
    )
    source_data = _source_data(
        monsters,
        (
            {
                "id": "DECIMILLIPEDE_TEST",
                "name": "The Decimillipede",
                "room_type": "Elite",
                "monsters": (
                    {"id": "DECIMILLIPEDE_SEGMENT_BACK"},
                    {"id": "DECIMILLIPEDE_SEGMENT_FRONT"},
                    {"id": "DECIMILLIPEDE_SEGMENT_MIDDLE"},
                ),
            },
        ),
        "DECIMILLIPEDE_TEST",
        deck=(
            {
                "card_id": "hard_focus",
                "name": "Hard Focus",
                "type": "attack",
                "cost": 0,
                "target": "enemy",
                "effects": {"damage": 50},
            },
        ),
    )
    state = _enter_source_combat(source_data)
    assert state.combat is not None
    back = state.combat.monsters[0].model_copy(update={"statuses": {"strength": 9}})
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={"monsters": (back,) + state.combat.monsters[1:]}
            )
        }
    )

    state = _play_card_targeting(state, "hard_focus", "DECIMILLIPEDE_SEGMENT_BACK")

    assert state.phase is RunPhase.COMBAT
    assert state.combat is not None
    defeated = state.combat.monsters[0]
    assert defeated.hp == 0
    assert defeated.move_id == "DEAD"
    assert defeated.statuses == {}
    assert defeated.metadata["reattach_turns"] == 2
    assert defeated.metadata["reattach_hp"] == 25
    assert any(event.kind == "decimillipede_segment_dead" for event in state.combat.last_events)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    defeated = state.combat.monsters[0]
    assert defeated.hp == 0
    assert defeated.metadata["reattach_turns"] == 1
    assert any(event.kind == "decimillipede_dead_turn" for event in state.combat.last_events)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    revived = state.combat.monsters[0]
    assert revived.hp == 25
    assert revived.statuses == {}
    assert revived.move_id == "WRITHE"
    assert any(event.kind == "decimillipede_reattached" for event in state.combat.last_events)


def test_aeonglass_source_data_starts_with_ebb_and_cycles_to_eye_lasers() -> None:
    source_data = _source_data(
        (
            {
                "id": "AEONGLASS",
                "name": "Aeonglass",
                "type": "Boss",
                "min_hp": 512,
                "max_hp": 512,
                "moves": (
                    {
                        "id": "EBB",
                        "name": "Ebb",
                        "intent": "Attack + Defend",
                        "damage": {"normal": 26, "ascension": 32},
                        "block": 33,
                    },
                    {
                        "id": "EYE_LASERS",
                        "name": "Eye Lasers",
                        "intent": "Attack",
                        "damage": {"normal": 11, "ascension": 12},
                    },
                    {
                        "id": "INCREASING_INTENSITY",
                        "name": "Increasing Intensity",
                        "intent": "Status + Buff",
                    },
                ),
                "attack_pattern": {
                    "initial_move": "EBB",
                    "states": (
                        {
                            "id": "EBB_MOVE",
                            "type": "move",
                            "move_id": "EBB",
                            "next": "EYE_LASERS_MOVE",
                        },
                        {
                            "id": "EYE_LASERS_MOVE",
                            "type": "move",
                            "move_id": "EYE_LASERS",
                            "next": "INCREASING_INTENSITY_MOVE",
                        },
                        {
                            "id": "INCREASING_INTENSITY_MOVE",
                            "type": "move",
                            "move_id": "INCREASING_INTENSITY",
                            "next": "EBB_MOVE",
                        },
                    ),
                },
            },
        ),
        (
            {
                "id": "AEONGLASS_TEST",
                "name": "Aeonglass Test",
                "room_type": "Boss",
                "monsters": ({"id": "AEONGLASS"},),
            },
        ),
        "AEONGLASS_TEST",
    )
    state = _enter_source_combat(source_data)
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.move_id == "EBB"
    assert monster.intent_damage == 26
    assert monster.intent_block == 33

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert state.combat.player.hp == 54
    assert monster.hp == 512
    assert monster.block == 33
    assert monster.move_id == "EYE_LASERS"
    assert monster.intent_damage == 11


def test_queen_amalgam_death_condition_changes_next_move_branch() -> None:
    definition = build_monster_definitions(
        (
            {
                "id": "QUEEN",
                "name": "Queen",
                "type": "Boss",
                "min_hp": 400,
                "max_hp": 400,
                "moves": (
                    {"id": "YOUR_MINE", "name": "Your Mine", "intent": "Debuff"},
                    {
                        "id": "BURN_BRIGHT_FOR_ME",
                        "name": "Burn Bright for Me",
                        "intent": "Buff + Defend",
                        "block": 20,
                    },
                    {
                        "id": "OFF_WITH_YOUR_HEAD",
                        "name": "Off with Your Head",
                        "intent": "Attack",
                        "damage": {"normal": 3, "hit_count": 5},
                    },
                ),
                "attack_pattern": {
                    "initial_move": "YOUR_MINE",
                    "states": (
                        {
                            "id": "YOUR_MINE_MOVE",
                            "type": "move",
                            "move_id": "YOUR_MINE",
                            "next": "YOURE_MINE_NOW_BRANCH",
                        },
                        {
                            "id": "YOURE_MINE_NOW_BRANCH",
                            "type": "conditional",
                            "branches": (
                                {
                                    "condition": "!HasAmalgamDied",
                                    "move_id": "BURN_BRIGHT_FOR_ME",
                                },
                                {
                                    "condition": "HasAmalgamDied",
                                    "move_id": "OFF_WITH_YOUR_HEAD",
                                },
                            ),
                        },
                    ),
                },
            },
        )
    )["QUEEN"]

    torch_alive = next_monster_move(
        definition,
        "YOUR_MINE",
        Random(1),
        move_counts={"YOUR_MINE": 1},
    )
    torch_dead = next_monster_move(
        definition,
        "YOUR_MINE",
        Random(1),
        move_counts={"YOUR_MINE": 1, "HAS_AMALGAM_DIED": 1},
    )

    assert torch_alive is not None and torch_alive.move_id == "BURN_BRIGHT_FOR_ME"
    assert torch_dead is not None and torch_dead.move_id == "OFF_WITH_YOUR_HEAD"


def test_queen_runtime_tracks_torch_head_amalgam_death_for_branching() -> None:
    queen = {
        "id": "QUEEN",
        "name": "Queen",
        "type": "Boss",
        "min_hp": 400,
        "max_hp": 400,
        "moves": (
            {"id": "PUPPET_STRINGS", "name": "Puppet Strings", "intent": "Debuff"},
            {"id": "YOUR_MINE", "name": "Your Mine", "intent": "Debuff"},
            {
                "id": "BURN_BRIGHT_FOR_ME",
                "name": "Burn Bright for Me",
                "intent": "Buff + Defend",
                "block": 20,
            },
            {
                "id": "OFF_WITH_YOUR_HEAD",
                "name": "Off with Your Head",
                "intent": "Attack",
                "damage": {"normal": 3, "hit_count": 5},
            },
        ),
        "attack_pattern": {
            "initial_move": "PUPPET_STRINGS",
            "states": (
                {
                    "id": "PUPPET_STRINGS_MOVE",
                    "type": "move",
                    "move_id": "PUPPET_STRINGS",
                    "next": "YOUR_MINE_MOVE",
                },
                {
                    "id": "YOUR_MINE_MOVE",
                    "type": "move",
                    "move_id": "YOUR_MINE",
                    "next": "YOURE_MINE_NOW_BRANCH",
                },
                {
                    "id": "YOURE_MINE_NOW_BRANCH",
                    "type": "conditional",
                    "branches": (
                        {"condition": "!HasAmalgamDied", "move_id": "BURN_BRIGHT_FOR_ME"},
                        {"condition": "HasAmalgamDied", "move_id": "OFF_WITH_YOUR_HEAD"},
                    ),
                },
            ),
        },
    }
    torch = {
        "id": "TORCH_HEAD_AMALGAM",
        "name": "Torch Head Amalgam",
        "type": "Boss",
        "min_hp": 20,
        "max_hp": 20,
        "moves": ({"id": "TACKLE", "name": "Tackle", "intent": "Attack"},),
        "attack_pattern": {
            "initial_move": "TACKLE",
            "states": ({"id": "TACKLE_MOVE", "type": "move", "move_id": "TACKLE"},),
        },
    }
    source_data = _source_data(
        (queen, torch),
        (
            {
                "id": "QUEEN_TEST",
                "name": "Queen Test",
                "room_type": "Boss",
                "monsters": ({"id": "QUEEN"}, {"id": "TORCH_HEAD_AMALGAM"}),
            },
        ),
        "QUEEN_TEST",
        deck=(
            {
                "card_id": "snuff_torch",
                "name": "Snuff Torch",
                "type": "attack",
                "cost": 0,
                "target": "enemy",
                "effects": {"damage": 25},
            },
        ),
    )
    state = _enter_source_combat(source_data)

    state = _play_card_targeting(state, "snuff_torch", "TORCH_HEAD_AMALGAM")
    state = step(state, _end_turn_action(state))
    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    queen_state = state.combat.monsters[0]
    assert queen_state.move_id == "OFF_WITH_YOUR_HEAD"
    assert queen_state.intent_damage == 15


def test_escape_monster_move_removes_monster_without_defeat_event() -> None:
    source_data = _source_data(
        (
            {
                "id": "RUNNER",
                "name": "Runner",
                "type": "Normal",
                "min_hp": 20,
                "max_hp": 20,
                "moves": ({"id": "FLEE", "name": "Flee", "intent": "Escape"},),
                "attack_pattern": {
                    "initial_move": "FLEE_MOVE",
                    "states": ({"id": "FLEE_MOVE", "type": "move", "move_id": "FLEE"},),
                },
            },
        ),
        (
            {
                "id": "RUNNER_ENCOUNTER",
                "name": "Runner",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "RUNNER"},),
            },
        ),
        "RUNNER_ENCOUNTER",
    )
    state = _enter_source_combat(source_data)

    state = step(state, _end_turn_action(state))

    assert state.phase is RunPhase.REWARD
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.hp == 0
    assert monster.metadata["escaped"] is True
    assert any(event.kind == "monster_escaped" for event in state.combat.last_events)
    assert not any(event.kind == "monster_defeated" for event in state.combat.last_events)


def test_formation_conditions_select_first_matching_branch() -> None:
    definition = build_monster_definitions(
        (
            {
                "id": "FORMATION_BRANCHER",
                "name": "Formation Brancher",
                "type": "Normal",
                "min_hp": 20,
                "max_hp": 20,
                "moves": (
                    {"id": "ALONE", "name": "Alone", "intent": "Attack"},
                    {"id": "FRONT", "name": "Front", "intent": "Attack"},
                    {"id": "BACK", "name": "Back", "intent": "Buff"},
                ),
                "attack_pattern": {
                    "initial_move": "INIT",
                    "states": (
                        {
                            "id": "INIT",
                            "type": "conditional",
                            "branches": (
                                {
                                    "condition": "base.Creature.GetAllyCount() == 0",
                                    "move_id": "ALONE",
                                },
                                {
                                    "condition": "((Nibbit)base.Creature.Monster).IsFront",
                                    "move_id": "FRONT",
                                },
                                {
                                    "condition": "!((Nibbit)base.Creature.Monster).IsFront",
                                    "move_id": "BACK",
                                },
                            ),
                        },
                    ),
                },
            },
        )
    )["FORMATION_BRANCHER"]

    alone = next_monster_move(
        definition,
        None,
        Random(1),
        ally_count=0,
        is_front=True,
    )
    front = next_monster_move(
        definition,
        None,
        Random(1),
        ally_count=1,
        is_front=True,
    )
    back = next_monster_move(
        definition,
        None,
        Random(1),
        ally_count=1,
        is_front=False,
    )

    assert alone is not None
    assert front is not None
    assert back is not None
    assert alone.move_id == "ALONE"
    assert front.move_id == "FRONT"
    assert back.move_id == "BACK"


def test_wriggler_named_slot_conditions_use_slot_index() -> None:
    definition = build_monster_definitions(
        (
            {
                "id": "WRIGGLER",
                "name": "Wriggler",
                "type": "Normal",
                "min_hp": 20,
                "max_hp": 20,
                "moves": (
                    {"id": "BITE", "name": "Bite", "intent": "Attack"},
                    {"id": "WRIGGLE", "name": "Wriggle", "intent": "Buff"},
                ),
                "attack_pattern": {
                    "initial_move": "INIT",
                    "states": (
                        {
                            "id": "INIT",
                            "type": "conditional",
                            "branches": (
                                {
                                    "condition": 'base.Creature.SlotName == "wriggler1"',
                                    "move_id": "BITE",
                                },
                                {
                                    "condition": 'base.Creature.SlotName == "wriggler2"',
                                    "move_id": "WRIGGLE",
                                },
                            ),
                        },
                    ),
                },
            },
        )
    )["WRIGGLER"]

    first = next_monster_move(definition, None, Random(1), slot_index=0)
    second = next_monster_move(definition, None, Random(1), slot_index=1)

    assert first is not None
    assert second is not None
    assert first.move_id == "BITE"
    assert second.move_id == "WRIGGLE"


def test_entomancer_bees_repeats_from_source_specific_selector() -> None:
    definition = build_monster_definitions(
        (
            {
                "id": "ENTOMANCER",
                "name": "Entomancer",
                "type": "Elite",
                "min_hp": 145,
                "max_hp": 145,
                "moves": (
                    {"id": "PHEROMONE_SPIT", "name": "Pheromone Spit", "intent": "Buff"},
                    {
                        "id": "BEES",
                        "name": "Bees",
                        "intent": "Attack",
                        "damage": {"normal": 3, "hit_count": 7},
                    },
                    {"id": "SPEAR", "name": "Spear", "intent": "Attack"},
                ),
                "attack_pattern": {
                    "initial_move": "BEES",
                    "states": (
                        {"id": "PHEROMONE_SPIT_MOVE", "type": "move", "move_id": "PHEROMONE_SPIT"},
                        {"id": "BEES_MOVE", "type": "move", "move_id": "BEES"},
                        {"id": "SPEAR_MOVE", "type": "move", "move_id": "SPEAR"},
                    ),
                },
            },
        )
    )["ENTOMANCER"]

    move = next_monster_move(definition, "BEES", Random(1))

    assert move is not None
    assert move.move_id == "BEES"


def test_spawn_capacity_conditions_select_summon_or_fallback_branch() -> None:
    definition = build_monster_definitions(
        (
            {
                "id": "OVICOPTER",
                "name": "Ovicopter",
                "type": "Normal",
                "min_hp": 120,
                "max_hp": 120,
                "moves": (
                    {"id": "LAY_EGGS", "name": "Lay Eggs", "intent": "Summon"},
                    {"id": "NUTRITIONAL_PASTE", "name": "Nutritional Paste", "intent": "Buff"},
                ),
                "attack_pattern": {
                    "initial_move": "SUMMON_BRANCH",
                    "states": (
                        {
                            "id": "SUMMON_BRANCH",
                            "type": "conditional",
                            "branches": (
                                {"condition": "CanLay", "move_id": "LAY_EGGS"},
                                {"condition": "!CanLay", "move_id": "NUTRITIONAL_PASTE"},
                            ),
                        },
                    ),
                },
            },
        )
    )["OVICOPTER"]

    can_lay = next_monster_move(
        definition,
        None,
        Random(1),
        can_spawn=True,
    )
    cannot_lay = next_monster_move(
        definition,
        None,
        Random(1),
        can_spawn=False,
    )

    assert can_lay is not None
    assert cannot_lay is not None
    assert can_lay.move_id == "LAY_EGGS"
    assert cannot_lay.move_id == "NUTRITIONAL_PASTE"


def test_hp_threshold_condition_uses_live_hp_and_once_move_count() -> None:
    definition = build_monster_definitions(
        (
            {
                "id": "FROG_KNIGHT",
                "name": "Frog Knight",
                "type": "Normal",
                "min_hp": 100,
                "max_hp": 100,
                "moves": (
                    {"id": "TONGUE_LASH", "name": "Tongue Lash", "intent": "Attack"},
                    {"id": "BEETLE_CHARGE", "name": "Beetle Charge", "intent": "Attack"},
                ),
                "attack_pattern": {
                    "initial_move": "HALF_HEALTH",
                    "states": (
                        {
                            "id": "HALF_HEALTH",
                            "type": "conditional",
                            "branches": (
                                {
                                    "condition": "HasBeetleCharged || "
                                    "base.Creature.CurrentHp >= base.Creature.MaxHp / 2",
                                    "move_id": "TONGUE_LASH",
                                },
                                {
                                    "condition": "!HasBeetleCharged && "
                                    "base.Creature.CurrentHp < base.Creature.MaxHp / 2",
                                    "move_id": "BEETLE_CHARGE",
                                },
                            ),
                        },
                    ),
                },
            },
        )
    )["FROG_KNIGHT"]

    above_half = next_monster_move(
        definition,
        None,
        Random(1),
        current_hp=60,
        max_hp=100,
    )
    below_half_before_charge = next_monster_move(
        definition,
        None,
        Random(1),
        current_hp=49,
        max_hp=100,
    )
    below_half_after_charge = next_monster_move(
        definition,
        None,
        Random(1),
        current_hp=49,
        max_hp=100,
        move_counts={"BEETLE_CHARGE": 1},
    )

    assert above_half is not None
    assert below_half_before_charge is not None
    assert below_half_after_charge is not None
    assert above_half.move_id == "TONGUE_LASH"
    assert below_half_before_charge.move_id == "BEETLE_CHARGE"
    assert below_half_after_charge.move_id == "TONGUE_LASH"


def test_summon_move_appends_source_monsters_that_do_not_act_immediately() -> None:
    source_data = _source_data(
        (
            {
                "id": "FABRICATOR",
                "name": "Fabricator",
                "type": "Normal",
                "min_hp": 100,
                "max_hp": 100,
                "moves": (
                    {"id": "FABRICATE", "name": "Fabricate", "intent": "Summon"},
                    {
                        "id": "DISINTEGRATE",
                        "name": "Disintegrate",
                        "intent": "Attack",
                        "damage": {"normal": 11},
                    },
                ),
                "attack_pattern": {
                    "initial_move": "FABRICATE_MOVE",
                    "states": (
                        {
                            "id": "FABRICATE_MOVE",
                            "type": "move",
                            "move_id": "FABRICATE",
                            "next": "DISINTEGRATE_MOVE",
                        },
                        {
                            "id": "DISINTEGRATE_MOVE",
                            "type": "move",
                            "move_id": "DISINTEGRATE",
                            "next": "DISINTEGRATE_MOVE",
                        },
                    ),
                },
            },
            {
                "id": "GUARDBOT",
                "name": "Guardbot",
                "type": "Normal",
                "min_hp": 16,
                "max_hp": 16,
                "moves": (
                    {
                        "id": "GUARD",
                        "name": "Guard",
                        "intent": "Defend",
                        "block": 15,
                    },
                ),
                "attack_pattern": {
                    "initial_move": "GUARD_MOVE",
                    "states": (
                        {
                            "id": "GUARD_MOVE",
                            "type": "move",
                            "move_id": "GUARD",
                            "next": "GUARD_MOVE",
                        },
                    ),
                },
            },
        ),
        (
            {
                "id": "FABRICATOR_ENCOUNTER",
                "name": "Fabricator",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "FABRICATOR"},),
            },
        ),
        "FABRICATOR_ENCOUNTER",
    )
    state = _enter_source_combat(source_data)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert [monster.monster_id for monster in state.combat.monsters] == [
        "FABRICATOR",
        "GUARDBOT",
    ]
    guardbot = state.combat.monsters[1]
    assert guardbot.hp == 16
    assert guardbot.intent_block == 15
    assert guardbot.block == 0
    assert guardbot.metadata["summoned"] is True
    assert any(
        event.kind == "monster_summoned" and event.target_id == "GUARDBOT"
        for event in state.combat.last_events
    )


def test_two_tailed_rat_call_for_backup_summons_another_rat() -> None:
    source_data = _source_data(
        (
            {
                "id": "TWO_TAILED_RAT",
                "name": "Two-Tailed Rat",
                "type": "Normal",
                "min_hp": 17,
                "max_hp": 17,
                "moves": (
                    {
                        "id": "CALL_FOR_BACKUP",
                        "name": "Call for Backup",
                        "intent": "Summon",
                    },
                    {
                        "id": "SCRATCH",
                        "name": "Scratch",
                        "intent": "Attack",
                        "damage": {"normal": 8},
                    },
                ),
                "attack_pattern": {
                    "initial_move": "CALL_FOR_BACKUP_MOVE",
                    "states": (
                        {
                            "id": "CALL_FOR_BACKUP_MOVE",
                            "type": "move",
                            "move_id": "CALL_FOR_BACKUP",
                            "next": "SCRATCH_MOVE",
                        },
                        {"id": "SCRATCH_MOVE", "type": "move", "move_id": "SCRATCH"},
                    ),
                },
            },
        ),
        (
            {
                "id": "RAT_BACKUP_ENCOUNTER",
                "name": "Rat Backup",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "TWO_TAILED_RAT"},),
            },
        ),
        "RAT_BACKUP_ENCOUNTER",
    )
    state = _enter_source_combat(source_data)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert len(state.combat.monsters) == 2
    summoned = state.combat.monsters[1]
    assert summoned.monster_id == "TWO_TAILED_RAT#2"
    assert summoned.metadata["source_monster_id"] == "TWO_TAILED_RAT"
    assert summoned.metadata["summoned"] is True
    assert any(
        event.kind == "monster_summoned" and event.target_id == "TWO_TAILED_RAT#2"
        for event in state.combat.last_events
    )


def test_tough_egg_hatch_advances_to_nibble_without_summoning() -> None:
    source_data = _source_data(
        (
            {
                "id": "TOUGH_EGG",
                "name": "Tough Egg",
                "type": "Normal",
                "min_hp": 14,
                "max_hp": 14,
                "moves": (
                    {"id": "HATCH", "name": "HATCH", "intent": "Summon"},
                    {
                        "id": "NIBBLE",
                        "name": "Nibble",
                        "intent": "Attack",
                        "damage": {"normal": 4},
                    },
                ),
                "attack_pattern": {
                    "initial_move": "HATCH_MOVE",
                    "states": (
                        {"id": "HATCH_MOVE", "type": "move", "move_id": "HATCH"},
                        {"id": "NIBBLE_MOVE", "type": "move", "move_id": "NIBBLE"},
                    ),
                },
            },
        ),
        (
            {
                "id": "TOUGH_EGG_ENCOUNTER",
                "name": "Tough Egg",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "TOUGH_EGG"},),
            },
        ),
        "TOUGH_EGG_ENCOUNTER",
    )
    state = _enter_source_combat(source_data)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert len(state.combat.monsters) == 1
    egg = state.combat.monsters[0]
    assert egg.move_id == "NIBBLE"
    assert egg.intent_damage == 4
    assert any(event.kind == "monster_hatched" for event in state.combat.last_events)
    assert not any(event.kind == "monster_summoned" for event in state.combat.last_events)


def test_gas_bomb_explode_damages_player_and_self_destructs() -> None:
    source_data = _source_data(
        (
            {
                "id": "GAS_BOMB",
                "name": "Gas Bomb",
                "type": "Normal",
                "min_hp": 7,
                "max_hp": 7,
                "moves": (
                    {
                        "id": "EXPLODE",
                        "name": "Explode",
                        "intent": "Special",
                        "damage": {"normal": 8},
                    },
                ),
                "attack_pattern": {
                    "initial_move": "EXPLODE_MOVE",
                    "states": (
                        {"id": "EXPLODE_MOVE", "type": "move", "move_id": "EXPLODE"},
                    ),
                },
            },
        ),
        (
            {
                "id": "GAS_BOMB_ENCOUNTER",
                "name": "Gas Bomb",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "GAS_BOMB"},),
            },
        ),
        "GAS_BOMB_ENCOUNTER",
    )
    state = _enter_source_combat(source_data)

    state = step(state, _end_turn_action(state))

    assert state.phase is RunPhase.REWARD
    assert state.combat is not None
    bomb = state.combat.monsters[0]
    assert bomb.hp == 0
    assert bomb.metadata["self_destructed"] is True
    assert state.combat.player.hp == 72
    assert any(event.kind == "monster_self_destructed" for event in state.combat.last_events)
    assert not any(event.kind == "monster_defeated" for event in state.combat.last_events)


def test_plating_reduces_at_start_and_grants_end_turn_block() -> None:
    source_data = _source_data(
        (
            {
                "id": "PLATED",
                "name": "Plated",
                "type": "Normal",
                "min_hp": 20,
                "max_hp": 20,
                "innate_powers": ({"power_id": "PLATING", "amount": 3},),
                "moves": ({"id": "WAIT", "name": "Wait", "intent": "Unknown"},),
                "attack_pattern": {
                    "initial_move": "WAIT_MOVE",
                    "states": (
                        {"id": "WAIT_MOVE", "type": "move", "move_id": "WAIT"},
                    ),
                },
            },
        ),
        (
            {
                "id": "PLATED_ENCOUNTER",
                "name": "Plated",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "PLATED"},),
            },
        ),
        "PLATED_ENCOUNTER",
    )
    state = _enter_source_combat(source_data)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.statuses["plating"] == 2
    assert monster.block == 2
    assert any(event.kind == "monster_plating_reduced" for event in state.combat.last_events)
    assert any(event.kind == "monster_plating_block" for event in state.combat.last_events)


def test_curl_up_grants_block_once_after_hp_damage() -> None:
    source_data = _source_data(
        (
            {
                "id": "CURLER",
                "name": "Curler",
                "type": "Normal",
                "min_hp": 20,
                "max_hp": 20,
                "innate_powers": ({"power_id": "CURL_UP", "amount": 4},),
                "moves": ({"id": "WAIT", "name": "Wait", "intent": "Unknown"},),
                "attack_pattern": {
                    "initial_move": "WAIT_MOVE",
                    "states": (
                        {"id": "WAIT_MOVE", "type": "move", "move_id": "WAIT"},
                    ),
                },
            },
        ),
        (
            {
                "id": "CURLER_ENCOUNTER",
                "name": "Curler",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "CURLER"},),
            },
        ),
        "CURLER_ENCOUNTER",
        deck=(
            {
                "card_id": "poke",
                "name": "Poke",
                "type": "attack",
                "cost": 0,
                "target": "enemy",
                "effects": {"damage": 2},
            },
            {
                "card_id": "poke",
                "name": "Poke",
                "type": "attack",
                "cost": 0,
                "target": "enemy",
                "effects": {"damage": 2},
            },
        ),
    )
    state = _enter_source_combat(source_data)

    state = _play_first_card(state, "poke")
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.hp == 18
    assert monster.block == 4
    assert monster.metadata["curl_up_used"] is True
    assert any(event.kind == "monster_curl_up" for event in state.combat.last_events)

    state = _play_first_card(state, "poke")
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.hp == 18
    assert monster.block == 2
    assert not any(event.kind == "monster_curl_up" for event in state.combat.last_events)


def test_skittish_grants_block_once_per_turn_on_first_hit() -> None:
    source_data = _source_data(
        (
            {
                "id": "SKITTISH_TARGET",
                "name": "Skittish Target",
                "type": "Normal",
                "min_hp": 20,
                "max_hp": 20,
                "innate_powers": ({"power_id": "SKITTISH", "amount": 3},),
                "moves": ({"id": "WAIT", "name": "Wait", "intent": "Unknown"},),
                "attack_pattern": {
                    "initial_move": "WAIT_MOVE",
                    "states": (
                        {"id": "WAIT_MOVE", "type": "move", "move_id": "WAIT"},
                    ),
                },
            },
        ),
        (
            {
                "id": "SKITTISH_ENCOUNTER",
                "name": "Skittish",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "SKITTISH_TARGET"},),
            },
        ),
        "SKITTISH_ENCOUNTER",
        deck=(
            {
                "card_id": "tap",
                "name": "Tap",
                "type": "attack",
                "cost": 0,
                "target": "enemy",
                "effects": {"damage": 1},
            },
            {
                "card_id": "tap",
                "name": "Tap",
                "type": "attack",
                "cost": 0,
                "target": "enemy",
                "effects": {"damage": 1},
            },
        ),
    )
    state = _enter_source_combat(source_data)

    state = _play_first_card(state, "tap")
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.hp == 19
    assert monster.block == 3
    assert any(event.kind == "monster_skittish_block" for event in state.combat.last_events)

    state = _play_first_card(state, "tap")
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.hp == 19
    assert monster.block == 2
    assert not any(event.kind == "monster_skittish_block" for event in state.combat.last_events)


def test_enrage_gains_strength_when_player_plays_skill() -> None:
    source_data = _source_data(
        (
            {
                "id": "ENRAGED_TARGET",
                "name": "Enraged Target",
                "type": "Normal",
                "min_hp": 20,
                "max_hp": 20,
                "innate_powers": ({"power_id": "ENRAGE", "amount": 2},),
                "moves": ({"id": "WAIT", "name": "Wait", "intent": "Unknown"},),
                "attack_pattern": {
                    "initial_move": "WAIT_MOVE",
                    "states": (
                        {"id": "WAIT_MOVE", "type": "move", "move_id": "WAIT"},
                    ),
                },
            },
        ),
        (
            {
                "id": "ENRAGED_ENCOUNTER",
                "name": "Enraged",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "ENRAGED_TARGET"},),
            },
        ),
        "ENRAGED_ENCOUNTER",
        deck=(
            {
                "card_id": "brace",
                "name": "Brace",
                "type": "skill",
                "cost": 0,
                "target": "self",
                "effects": {"block": 3},
            },
        ),
    )
    state = _enter_source_combat(source_data)

    state = _play_first_card(state, "brace")

    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.statuses["strength"] == 2
    assert any(
        event.kind == "status_applied" and event.metadata.get("trigger") == "enrage"
        for event in state.combat.last_events
    )


def test_personal_hive_adds_dazed_to_draw_pile_when_hit_by_attack() -> None:
    source_data = _source_data(
        (
            {
                "id": "HIVE_TARGET",
                "name": "Hive Target",
                "type": "Normal",
                "min_hp": 20,
                "max_hp": 20,
                "innate_powers": ({"power_id": "PERSONAL_HIVE", "amount": 1},),
                "moves": ({"id": "WAIT", "name": "Wait", "intent": "Unknown"},),
                "attack_pattern": {
                    "initial_move": "WAIT_MOVE",
                    "states": (
                        {"id": "WAIT_MOVE", "type": "move", "move_id": "WAIT"},
                    ),
                },
            },
        ),
        (
            {
                "id": "HIVE_ENCOUNTER",
                "name": "Hive",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "HIVE_TARGET"},),
            },
        ),
        "HIVE_ENCOUNTER",
        deck=(
            {
                "card_id": "sting",
                "name": "Sting",
                "type": "attack",
                "cost": 0,
                "target": "enemy",
                "effects": {"damage": 1},
            },
        ),
    )
    state = _enter_source_combat(source_data)

    state = _play_first_card(state, "sting")

    assert state.combat is not None
    assert state.combat.draw_pile[0].card_id == "dazed"
    assert state.combat.draw_pile[0].custom["unplayable"] is True
    assert any(event.kind == "monster_personal_hive" for event in state.combat.last_events)
    assert any(
        event.kind == "card_created" and event.metadata.get("reason") == "personal_hive"
        for event in state.combat.last_events
    )


def test_slippery_caps_hp_loss_and_decrements_stack() -> None:
    source_data = _source_data(
        (
            {
                "id": "VANTOM",
                "name": "Vantom",
                "type": "Boss",
                "min_hp": 40,
                "max_hp": 40,
                "moves": ({"id": "WAIT", "name": "Wait", "intent": "Unknown"},),
                "attack_pattern": {
                    "initial_move": "WAIT_MOVE",
                    "states": (
                        {"id": "WAIT_MOVE", "type": "move", "move_id": "WAIT"},
                    ),
                },
            },
        ),
        (
            {
                "id": "VANTOM_TEST",
                "name": "Vantom Test",
                "room_type": "Boss",
                "monsters": ({"id": "VANTOM"},),
            },
        ),
        "VANTOM_TEST",
        deck=(
            {
                "card_id": "heavy_hit",
                "name": "Heavy Hit",
                "type": "attack",
                "cost": 0,
                "target": "enemy",
                "effects": {"damage": 20},
            },
        ),
    )
    state = _enter_source_combat(source_data)

    assert state.combat is not None
    assert state.combat.monsters[0].statuses["slippery"] == 9

    state = _play_first_card(state, "heavy_hit")

    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.hp == 39
    assert monster.statuses["slippery"] == 8
    assert any(event.kind == "monster_slippery" for event in state.combat.last_events)
    assert any(
        event.kind == "monster_damaged"
        and event.amount == 1
        and event.metadata.get("slippery") is True
        for event in state.combat.last_events
    )


def test_soul_fysh_beckon_adds_beckons_to_draw_and_discard() -> None:
    source_data = _source_data(
        (
            {
                "id": "SOUL_FYSH",
                "name": "Soul Fysh",
                "type": "Boss",
                "min_hp": 120,
                "max_hp": 120,
                "moves": ({"id": "BECKON", "name": "Beckon", "intent": "Status"},),
                "attack_pattern": {
                    "initial_move": "BECKON_MOVE",
                    "states": (
                        {"id": "BECKON_MOVE", "type": "move", "move_id": "BECKON"},
                    ),
                },
            },
        ),
        (
            {
                "id": "SOUL_FYSH_TEST",
                "name": "Soul Fysh Test",
                "room_type": "Boss",
                "monsters": ({"id": "SOUL_FYSH"},),
            },
        ),
        "SOUL_FYSH_TEST",
    )
    state = _enter_source_combat(source_data)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    beckons = [
        card
        for pile in (state.combat.hand, state.combat.draw_pile, state.combat.discard_pile)
        for card in pile
        if card.card_id == "beckon"
    ]
    assert len(beckons) == 2
    assert all(card.cost == 1 for card in beckons)
    assert all(not card.custom.get("unplayable", False) for card in beckons)
    assert any(
        event.kind == "card_created"
        and event.metadata.get("reason") == "soul_fysh_beckon"
        and event.metadata.get("zone") == "draw_pile"
        for event in state.combat.last_events
    )
    assert any(
        event.kind == "card_created"
        and event.metadata.get("reason") == "soul_fysh_beckon"
        and event.metadata.get("zone") == "discard_pile"
        for event in state.combat.last_events
    )


def test_soul_fysh_gaze_attacks_and_adds_beckon_to_draw_pile() -> None:
    source_data = _source_data(
        (
            {
                "id": "SOUL_FYSH",
                "name": "Soul Fysh",
                "type": "Boss",
                "min_hp": 120,
                "max_hp": 120,
                "moves": (
                    {
                        "id": "GAZE",
                        "name": "Gaze",
                        "intent": "Attack + Status",
                        "damage": {"normal": 7},
                    },
                ),
                "attack_pattern": {
                    "initial_move": "GAZE_MOVE",
                    "states": (
                        {"id": "GAZE_MOVE", "type": "move", "move_id": "GAZE"},
                    ),
                },
            },
        ),
        (
            {
                "id": "SOUL_FYSH_GAZE_TEST",
                "name": "Soul Fysh Gaze Test",
                "room_type": "Boss",
                "monsters": ({"id": "SOUL_FYSH"},),
            },
        ),
        "SOUL_FYSH_GAZE_TEST",
    )
    state = _enter_source_combat(source_data)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert state.combat.player.hp == 73
    beckons = [
        card
        for pile in (state.combat.hand, state.combat.draw_pile, state.combat.discard_pile)
        for card in pile
        if card.card_id == "beckon"
    ]
    assert len(beckons) == 1
    assert any(
        event.kind == "card_created"
        and event.metadata.get("reason") == "soul_fysh_gaze"
        and event.metadata.get("zone") == "draw_pile"
        for event in state.combat.last_events
    )


def test_vantom_dismember_attacks_and_adds_wounds_to_discard() -> None:
    source_data = _source_data(
        (
            {
                "id": "VANTOM",
                "name": "Vantom",
                "type": "Boss",
                "min_hp": 120,
                "max_hp": 120,
                "moves": (
                    {
                        "id": "DISMEMBER",
                        "name": "Dismember",
                        "intent": "Attack + Status",
                        "damage": {"normal": 27},
                    },
                ),
                "attack_pattern": {
                    "initial_move": "DISMEMBER_MOVE",
                    "states": (
                        {
                            "id": "DISMEMBER_MOVE",
                            "type": "move",
                            "move_id": "DISMEMBER",
                        },
                    ),
                },
            },
        ),
        (
            {
                "id": "VANTOM_DISMEMBER_TEST",
                "name": "Vantom Dismember Test",
                "room_type": "Boss",
                "monsters": ({"id": "VANTOM"},),
            },
        ),
        "VANTOM_DISMEMBER_TEST",
    )
    state = _enter_source_combat(source_data)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert state.combat.player.hp == 53
    wounds = [
        card
        for pile in (state.combat.hand, state.combat.draw_pile, state.combat.discard_pile)
        for card in pile
        if card.card_id == "wound"
    ]
    assert len(wounds) == 3
    assert all(card.custom["unplayable"] for card in wounds)
    assert any(
        event.kind == "card_created"
        and event.metadata.get("reason") == "vantom_dismember"
        for event in state.combat.last_events
    )


def test_insatiable_liquify_ground_starts_sandpit_and_shuffles_escape_cards() -> None:
    source_data = _source_data(
        (
            {
                "id": "THE_INSATIABLE",
                "name": "The Insatiable",
                "type": "Boss",
                "min_hp": 200,
                "max_hp": 200,
                "moves": (
                    {
                        "id": "LIQUIFY_GROUND",
                        "name": "Liquify Ground",
                        "intent": "Buff + Status",
                    },
                    {"id": "WAIT", "name": "Wait", "intent": "Unknown"},
                ),
                "attack_pattern": {
                    "initial_move": "LIQUIFY_GROUND_MOVE",
                    "states": (
                        {
                            "id": "LIQUIFY_GROUND_MOVE",
                            "type": "move",
                            "move_id": "LIQUIFY_GROUND",
                            "next": "WAIT_MOVE",
                        },
                        {"id": "WAIT_MOVE", "type": "move", "move_id": "WAIT"},
                    ),
                },
            },
        ),
        (
            {
                "id": "INSATIABLE_TEST",
                "name": "Insatiable Test",
                "room_type": "Boss",
                "monsters": ({"id": "THE_INSATIABLE"},),
            },
        ),
        "INSATIABLE_TEST",
    )
    state = _enter_source_combat(source_data)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert state.combat.player.resources["sandpit"] == 4
    escapes = [
        card
        for pile in (state.combat.hand, state.combat.draw_pile, state.combat.discard_pile)
        for card in pile
        if card.card_id == "frantic_escape"
    ]
    assert len(escapes) == 6
    assert all(card.cost == 1 for card in escapes)
    assert all(not card.custom.get("unplayable", False) for card in escapes)
    assert any(
        event.kind == "card_created"
        and event.metadata.get("reason") == "the_insatiable_liquify_ground"
        and event.metadata.get("to_pile") == "draw_pile_shuffled"
        for event in state.combat.last_events
    )


def test_frantic_escape_extends_sandpit_before_end_turn_tick() -> None:
    source_data = _source_data(
        (
            {
                "id": "THE_INSATIABLE",
                "name": "The Insatiable",
                "type": "Boss",
                "min_hp": 200,
                "max_hp": 200,
                "moves": (
                    {
                        "id": "LIQUIFY_GROUND",
                        "name": "Liquify Ground",
                        "intent": "Buff + Status",
                    },
                    {"id": "WAIT", "name": "Wait", "intent": "Unknown"},
                ),
                "attack_pattern": {
                    "initial_move": "LIQUIFY_GROUND_MOVE",
                    "states": (
                        {
                            "id": "LIQUIFY_GROUND_MOVE",
                            "type": "move",
                            "move_id": "LIQUIFY_GROUND",
                            "next": "WAIT_MOVE",
                        },
                        {"id": "WAIT_MOVE", "type": "move", "move_id": "WAIT"},
                    ),
                },
            },
        ),
        (
            {
                "id": "INSATIABLE_ESCAPE_TEST",
                "name": "Insatiable Escape Test",
                "room_type": "Boss",
                "monsters": ({"id": "THE_INSATIABLE"},),
            },
        ),
        "INSATIABLE_ESCAPE_TEST",
    )
    state = _enter_source_combat(source_data)
    state = step(state, _end_turn_action(state))

    state = _play_first_card(state, "frantic_escape")
    assert state.combat is not None
    assert state.combat.player.resources["sandpit"] == 5
    escaped = next(card for card in state.combat.discard_pile if card.card_id == "frantic_escape")
    assert escaped.cost == 2

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert state.combat.player.resources["sandpit"] == 4
    assert any(event.kind == "sandpit_timer_ticked" for event in state.combat.last_events)


def test_sandpit_expiration_fails_combat_before_monster_turn() -> None:
    source_data = _source_data(
        (
            {
                "id": "THE_INSATIABLE",
                "name": "The Insatiable",
                "type": "Boss",
                "min_hp": 200,
                "max_hp": 200,
                "moves": ({"id": "WAIT", "name": "Wait", "intent": "Unknown"},),
                "attack_pattern": {
                    "initial_move": "WAIT_MOVE",
                    "states": (
                        {"id": "WAIT_MOVE", "type": "move", "move_id": "WAIT"},
                    ),
                },
            },
        ),
        (
            {
                "id": "INSATIABLE_EXPIRE_TEST",
                "name": "Insatiable Expire Test",
                "room_type": "Boss",
                "monsters": ({"id": "THE_INSATIABLE"},),
            },
        ),
        "INSATIABLE_EXPIRE_TEST",
    )
    state = _enter_source_combat(source_data)
    assert state.combat is not None
    combat = state.combat.model_copy(
        update={
            "player": state.combat.player.model_copy(update={"resources": {"sandpit": 1}}),
            "metadata": {**state.combat.metadata, "sandpit_active": True},
        }
    )
    state = state.model_copy(update={"combat": combat})

    state = step(state, _end_turn_action(state))

    assert state.phase is RunPhase.FAILED
    assert state.combat is not None
    assert state.combat.player.hp == 0
    assert any(event.kind == "sandpit_timer_expired" for event in state.combat.last_events)


def test_lagavulin_matriarch_starts_asleep_with_plating_and_wakes_after_three_turns() -> None:
    source_data = _source_data(
        (
            {
                "id": "LAGAVULIN_MATRIARCH",
                "name": "Lagavulin Matriarch",
                "type": "Boss",
                "min_hp": 222,
                "max_hp": 222,
                "moves": (
                    {"id": "SLEEP", "name": "Sleep", "intent": "Sleep"},
                    {
                        "id": "SLASH",
                        "name": "Slash",
                        "intent": "Attack",
                        "damage": {"normal": 19},
                    },
                ),
                "attack_pattern": {
                    "initial_move": "SLEEP_MOVE",
                    "states": (
                        {"id": "SLEEP_MOVE", "type": "move", "move_id": "SLEEP"},
                        {"id": "SLASH_MOVE", "type": "move", "move_id": "SLASH"},
                    ),
                },
            },
        ),
        (
            {
                "id": "LAGAVULIN_SLEEP_TEST",
                "name": "Lagavulin Sleep Test",
                "room_type": "Boss",
                "monsters": ({"id": "LAGAVULIN_MATRIARCH"},),
            },
        ),
        "LAGAVULIN_SLEEP_TEST",
    )
    state = _enter_source_combat(source_data)

    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.statuses["asleep"] == 3
    assert monster.statuses["plating"] == 12

    state = step(state, _end_turn_action(state))
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.move_id == "SLEEP"
    assert monster.statuses["asleep"] == 2
    assert state.combat.player.hp == 80

    state = step(state, _end_turn_action(state))
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.move_id == "SLEEP"
    assert monster.statuses["asleep"] == 1
    assert state.combat.player.hp == 80

    state = step(state, _end_turn_action(state))
    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.move_id == "SLASH"
    assert "asleep" not in monster.statuses
    assert state.combat.player.hp == 80
    assert any(event.kind == "monster_awakened" for event in state.combat.last_events)


def test_lagavulin_matriarch_hp_damage_wakes_and_stuns_next_action() -> None:
    source_data = _source_data(
        (
            {
                "id": "LAGAVULIN_MATRIARCH",
                "name": "Lagavulin Matriarch",
                "type": "Boss",
                "min_hp": 50,
                "max_hp": 50,
                "moves": (
                    {"id": "SLEEP", "name": "Sleep", "intent": "Sleep"},
                    {
                        "id": "SLASH",
                        "name": "Slash",
                        "intent": "Attack",
                        "damage": {"normal": 19},
                    },
                ),
                "attack_pattern": {
                    "initial_move": "SLEEP_MOVE",
                    "states": (
                        {"id": "SLEEP_MOVE", "type": "move", "move_id": "SLEEP"},
                        {"id": "SLASH_MOVE", "type": "move", "move_id": "SLASH"},
                    ),
                },
            },
        ),
        (
            {
                "id": "LAGAVULIN_WAKE_TEST",
                "name": "Lagavulin Wake Test",
                "room_type": "Boss",
                "monsters": ({"id": "LAGAVULIN_MATRIARCH"},),
            },
        ),
        "LAGAVULIN_WAKE_TEST",
        deck=(
            {
                "card_id": "poke",
                "name": "Poke",
                "type": "attack",
                "cost": 0,
                "target": "enemy",
                "effects": {"damage": 1},
            },
        ),
    )
    state = _enter_source_combat(source_data)

    state = _play_first_card(state, "poke")

    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.hp == 49
    assert "asleep" not in monster.statuses
    assert monster.metadata["lagavulin_woke_by_damage"] is True
    assert any(
        event.kind == "monster_awakened" and event.metadata.get("reason") == "hp_damage"
        for event in state.combat.last_events
    )

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert state.combat.player.hp == 80
    assert monster.move_id == "SLASH"
    assert "stunned" not in monster.statuses
    assert any(event.kind == "monster_woke_stunned" for event in state.combat.last_events)
    assert any(event.kind == "monster_stunned" for event in state.combat.last_events)


def test_lagavulin_matriarch_soul_siphon_debuffs_player_and_buffs_self() -> None:
    source_data = _source_data(
        (
            {
                "id": "LAGAVULIN_MATRIARCH",
                "name": "Lagavulin Matriarch",
                "type": "Boss",
                "min_hp": 50,
                "max_hp": 50,
                "moves": (
                    {
                        "id": "SOUL_SIPHON",
                        "name": "Soul Siphon",
                        "intent": "Debuff + Buff",
                        "powers": (
                            {"power_id": "STRENGTH", "amount": 2, "target": "self"},
                        ),
                    },
                ),
                "attack_pattern": {
                    "initial_move": "SOUL_SIPHON_MOVE",
                    "states": (
                        {
                            "id": "SOUL_SIPHON_MOVE",
                            "type": "move",
                            "move_id": "SOUL_SIPHON",
                        },
                    ),
                },
            },
        ),
        (
            {
                "id": "LAGAVULIN_SIPHON_TEST",
                "name": "Lagavulin Siphon Test",
                "room_type": "Boss",
                "monsters": ({"id": "LAGAVULIN_MATRIARCH"},),
            },
        ),
        "LAGAVULIN_SIPHON_TEST",
    )
    state = _enter_source_combat(source_data)
    assert state.combat is not None
    monster = state.combat.monsters[0].model_copy(update={"statuses": {}})
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert state.combat.player.statuses["strength"] == -2
    assert state.combat.player.statuses["dexterity"] == -2
    assert state.combat.monsters[0].statuses["strength"] == 2
    assert any(
        event.kind == "status_applied"
        and event.metadata.get("reason") == "lagavulin_soul_siphon"
        for event in state.combat.last_events
    )


def test_ceremonial_beast_plow_threshold_queues_stun_and_clears_strength() -> None:
    source_data = _source_data(
        (
            {
                "id": "CEREMONIAL_BEAST",
                "name": "Ceremonial Beast",
                "type": "Boss",
                "min_hp": 252,
                "max_hp": 252,
                "moves": (
                    {
                        "id": "PLOW",
                        "name": "Plow",
                        "intent": "Attack + Buff",
                        "damage": {"normal": 18},
                        "powers": ({"power_id": "STRENGTH", "amount": 2, "target": "self"},),
                    },
                    {"id": "STUN", "name": "Stun", "intent": "Stun"},
                    {
                        "id": "BEAST_CRY",
                        "name": "Beast Cry",
                        "intent": "Debuff",
                        "powers": ({"power_id": "RINGING", "amount": 1, "target": "player"},),
                    },
                ),
                "attack_pattern": {
                    "initial_move": "PLOW_MOVE",
                    "states": (
                        {"id": "PLOW_MOVE", "type": "move", "move_id": "PLOW"},
                        {
                            "id": "STUN_MOVE",
                            "type": "move",
                            "move_id": "STUN",
                            "next": "BEAST_CRY_MOVE",
                        },
                        {"id": "BEAST_CRY_MOVE", "type": "move", "move_id": "BEAST_CRY"},
                    ),
                },
            },
        ),
        (
            {
                "id": "CEREMONIAL_PLOW_TEST",
                "name": "Ceremonial Plow Test",
                "room_type": "Boss",
                "monsters": ({"id": "CEREMONIAL_BEAST"},),
            },
        ),
        "CEREMONIAL_PLOW_TEST",
        deck=(
            {
                "card_id": "threshold_hit",
                "name": "Threshold Hit",
                "type": "attack",
                "cost": 0,
                "target": "enemy",
                "effects": {"damage": 2},
            },
        ),
    )
    state = _enter_source_combat(source_data)
    assert state.combat is not None
    monster = state.combat.monsters[0].model_copy(
        update={
            "hp": 151,
            "statuses": {
                "plow": 150,
                "strength": 4,
                "temporary_strength": 3,
                "strength_down": 2,
            },
        }
    )
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )

    state = _play_first_card(state, "threshold_hit")

    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.hp == 149
    assert "plow" not in monster.statuses
    assert "strength" not in monster.statuses
    assert "temporary_strength" not in monster.statuses
    assert monster.metadata["force_stun_move"] is True
    assert monster.metadata["ceremonial_beast_phase"] == 2
    assert any(event.kind == "monster_plow_triggered" for event in state.combat.last_events)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert state.combat.player.hp == 80
    monster = state.combat.monsters[0]
    assert monster.move_id == "BEAST_CRY"
    assert any(event.kind == "monster_stun_forced" for event in state.combat.last_events)


def test_ceremonial_beast_ringing_limits_player_to_one_card_next_turn() -> None:
    source_data = _source_data(
        (
            {
                "id": "CEREMONIAL_BEAST",
                "name": "Ceremonial Beast",
                "type": "Boss",
                "min_hp": 252,
                "max_hp": 252,
                "moves": (
                    {
                        "id": "BEAST_CRY",
                        "name": "Beast Cry",
                        "intent": "Debuff",
                        "powers": ({"power_id": "RINGING", "amount": 1, "target": "player"},),
                    },
                ),
                "attack_pattern": {
                    "initial_move": "BEAST_CRY_MOVE",
                    "states": (
                        {"id": "BEAST_CRY_MOVE", "type": "move", "move_id": "BEAST_CRY"},
                    ),
                },
            },
        ),
        (
            {
                "id": "CEREMONIAL_RINGING_TEST",
                "name": "Ceremonial Ringing Test",
                "room_type": "Boss",
                "monsters": ({"id": "CEREMONIAL_BEAST"},),
            },
        ),
        "CEREMONIAL_RINGING_TEST",
        deck=(
            {
                "card_id": "free_hit_a",
                "name": "Free Hit A",
                "type": "attack",
                "cost": 0,
                "target": "enemy",
                "effects": {"damage": 1},
            },
            {
                "card_id": "free_hit_b",
                "name": "Free Hit B",
                "type": "attack",
                "cost": 0,
                "target": "enemy",
                "effects": {"damage": 1},
            },
        ),
    )
    state = _enter_source_combat(source_data)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert state.combat.metadata["turn_card_play_limit"] == 1
    assert any(
        event.kind == "turn_card_play_limit_added" for event in state.combat.last_events
    )

    state = _play_first_card(state, "free_hit_a")

    assert state.combat is not None
    assert state.combat.cards_played_this_turn
    assert not any(action.type == "play_card" for action in legal_actions(state))


def test_phrog_parasite_infect_adds_infections_to_discard_pile() -> None:
    source_data = _source_data(
        (
            {
                "id": "PHROG_PARASITE",
                "name": "Phrog Parasite",
                "type": "Normal",
                "min_hp": 40,
                "max_hp": 40,
                "moves": (
                    {"id": "INFECT", "name": "Infect", "intent": "Status"},
                    {
                        "id": "LASH",
                        "name": "Lash",
                        "intent": "Attack",
                        "damage": {"normal": 4, "hit_count": 4},
                    },
                ),
                "attack_pattern": {
                    "initial_move": "INFECT_MOVE",
                    "states": (
                        {"id": "INFECT_MOVE", "type": "move", "move_id": "INFECT"},
                    ),
                },
            },
        ),
        (
            {
                "id": "PHROG_ENCOUNTER",
                "name": "Phrog",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "PHROG_PARASITE"},),
            },
        ),
        "PHROG_ENCOUNTER",
    )
    state = _enter_source_combat(source_data)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    infection_cards = [
        card
        for pile in (state.combat.hand, state.combat.draw_pile, state.combat.discard_pile)
        for card in pile
        if card.card_id == "infection"
    ]
    assert len(infection_cards) == 3
    assert all(card.custom["unplayable"] for card in infection_cards)
    assert any(
        event.kind == "card_created"
        and event.metadata.get("reason") == "phrog_parasite_infect"
        for event in state.combat.last_events
    )


def test_mecha_knight_flamethrower_adds_burns_to_hand() -> None:
    source_data = _source_data(
        (
            {
                "id": "MECHA_KNIGHT",
                "name": "Mecha Knight",
                "type": "Normal",
                "min_hp": 80,
                "max_hp": 80,
                "moves": (
                    {"id": "FLAMETHROWER", "name": "Flamethrower", "intent": "Status"},
                    {
                        "id": "CHARGE",
                        "name": "Charge",
                        "intent": "Attack",
                        "damage": {"normal": 25},
                    },
                ),
                "attack_pattern": {
                    "initial_move": "FLAMETHROWER_MOVE",
                    "states": (
                        {
                            "id": "FLAMETHROWER_MOVE",
                            "type": "move",
                            "move_id": "FLAMETHROWER",
                        },
                    ),
                },
            },
        ),
        (
            {
                "id": "MECHA_ENCOUNTER",
                "name": "Mecha",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "MECHA_KNIGHT"},),
            },
        ),
        "MECHA_ENCOUNTER",
    )
    state = _enter_source_combat(source_data)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    burn_cards = [card for card in state.combat.hand if card.card_id == "burn"]
    assert len(burn_cards) == 4
    assert all(card.custom["unplayable"] for card in burn_cards)
    assert any(
        event.kind == "card_created"
        and event.metadata.get("reason") == "mecha_knight_flamethrower"
        for event in state.combat.last_events
    )


def test_magi_knight_dampen_downgrades_upgraded_cards_while_alive() -> None:
    source_data = _source_data(
        (
            {
                "id": "MAGI_KNIGHT",
                "name": "Magi Knight",
                "type": "Normal",
                "min_hp": 40,
                "max_hp": 40,
                "innate_powers": ({"power_id": "DAMPEN", "amount": 1},),
                "moves": ({"id": "WAIT", "name": "Wait", "intent": "Unknown"},),
                "attack_pattern": {
                    "initial_move": "WAIT_MOVE",
                    "states": (
                        {"id": "WAIT_MOVE", "type": "move", "move_id": "WAIT"},
                    ),
                },
            },
        ),
        (
            {
                "id": "MAGI_ENCOUNTER",
                "name": "Magi",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "MAGI_KNIGHT"},),
            },
        ),
        "MAGI_ENCOUNTER",
        deck=(
            {
                "card_id": "upgraded_hit",
                "name": "Upgraded Hit",
                "type": "attack",
                "cost": 1,
                "target": "enemy",
                "effects": {"damage": 6},
                "upgraded": True,
                "upgrade": {"damage": 4},
            },
        ),
    )
    state = _enter_source_combat(source_data)

    state = _play_first_card(state, "upgraded_hit")

    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.hp == 34
    assert state.combat.discard_pile[0].upgraded is True
    assert any(
        event.kind == "card_played" and event.metadata.get("dynamic_downgrade") is True
        for event in state.combat.last_events
    )


def test_magi_knight_dampen_move_applies_dampen_status() -> None:
    source_data = _source_data(
        (
            {
                "id": "MAGI_KNIGHT",
                "name": "Magi Knight",
                "type": "Normal",
                "min_hp": 40,
                "max_hp": 40,
                "moves": (
                    {"id": "DAMPEN", "name": "Dampen", "intent": "Debuff"},
                ),
                "attack_pattern": {
                    "initial_move": "DAMPEN_MOVE",
                    "states": (
                        {"id": "DAMPEN_MOVE", "type": "move", "move_id": "DAMPEN"},
                    ),
                },
            },
        ),
        (
            {
                "id": "MAGI_DAMPEN_ENCOUNTER",
                "name": "Magi Dampen",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "MAGI_KNIGHT"},),
            },
        ),
        "MAGI_DAMPEN_ENCOUNTER",
    )
    state = _enter_source_combat(source_data)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    monster = state.combat.monsters[0]
    assert monster.statuses["dampen"] == 1
    assert any(
        event.kind == "status_applied"
        and event.metadata.get("reason") == "magi_knight_dampen"
        for event in state.combat.last_events
    )


def test_spectral_knight_hex_exhausts_hand_cards_at_end_turn() -> None:
    source_data = _source_data(
        (
            {
                "id": "SPECTRAL_KNIGHT",
                "name": "Spectral Knight",
                "type": "Normal",
                "min_hp": 40,
                "max_hp": 40,
                "innate_powers": ({"power_id": "HEX", "amount": 1},),
                "moves": ({"id": "WAIT", "name": "Wait", "intent": "Unknown"},),
                "attack_pattern": {
                    "initial_move": "WAIT_MOVE",
                    "states": (
                        {"id": "WAIT_MOVE", "type": "move", "move_id": "WAIT"},
                    ),
                },
            },
        ),
        (
            {
                "id": "SPECTRAL_ENCOUNTER",
                "name": "Spectral",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "SPECTRAL_KNIGHT"},),
            },
        ),
        "SPECTRAL_ENCOUNTER",
        deck=(
            {
                "card_id": "plain_strike",
                "name": "Plain Strike",
                "type": "attack",
                "cost": 1,
                "target": "enemy",
                "effects": {"damage": 1},
            },
        ),
    )
    state = _enter_source_combat(source_data)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert [card.card_id for card in state.combat.exhaust_pile] == ["plain_strike"]
    assert not state.combat.discard_pile
    assert any(event.kind == "hex_ethereal_cards_exhausted" for event in state.combat.last_events)


def test_ravenous_eats_defeated_enemy_gains_strength_and_stuns() -> None:
    source_data = _source_data(
        (
            {
                "id": "CORPSE_SLUG",
                "name": "Corpse Slug",
                "type": "Normal",
                "min_hp": 20,
                "max_hp": 20,
                "innate_powers": ({"power_id": "RAVENOUS", "amount": 4},),
                "moves": (
                    {
                        "id": "WHIP_SLAP",
                        "name": "Whip Slap",
                        "intent": "Attack",
                        "damage": {"normal": 3, "hit_count": 2},
                    },
                ),
                "attack_pattern": {
                    "initial_move": "WHIP_SLAP_MOVE",
                    "states": (
                        {"id": "WHIP_SLAP_MOVE", "type": "move", "move_id": "WHIP_SLAP"},
                    ),
                },
            },
            {
                "id": "MEAL",
                "name": "Meal",
                "type": "Normal",
                "min_hp": 1,
                "max_hp": 1,
                "moves": ({"id": "WAIT", "name": "Wait", "intent": "Unknown"},),
                "attack_pattern": {
                    "initial_move": "WAIT_MOVE",
                    "states": (
                        {"id": "WAIT_MOVE", "type": "move", "move_id": "WAIT"},
                    ),
                },
            },
        ),
        (
            {
                "id": "RAVENOUS_ENCOUNTER",
                "name": "Ravenous",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "CORPSE_SLUG"}, {"id": "MEAL"}),
            },
        ),
        "RAVENOUS_ENCOUNTER",
        deck=(
            {
                "card_id": "feed_slug",
                "name": "Feed Slug",
                "type": "attack",
                "cost": 0,
                "target": "enemy",
                "effects": {"damage": 2},
            },
        ),
    )
    state = _enter_source_combat(source_data)

    state = _play_card_targeting(state, "feed_slug", "MEAL")

    assert state.combat is not None
    slug = state.combat.monsters[0]
    assert state.phase is RunPhase.COMBAT
    assert slug.statuses["strength"] == 4
    assert slug.statuses["stunned"] == 1
    assert any(event.kind == "monster_ravenous" for event in state.combat.last_events)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    slug = state.combat.monsters[0]
    assert state.combat.player.hp == 80
    assert "stunned" not in slug.statuses
    assert slug.move_id == "WHIP_SLAP"
    assert any(event.kind == "monster_stunned" for event in state.combat.last_events)


def test_shriek_forces_stun_move_once_when_hp_threshold_is_reached() -> None:
    source_data = _source_data(
        (
            {
                "id": "TERROR_EEL",
                "name": "Terror Eel",
                "type": "Normal",
                "min_hp": 70,
                "max_hp": 70,
                "innate_powers": ({"power_id": "SHRIEK", "amount": 35},),
                "moves": (
                    {
                        "id": "CRASH",
                        "name": "Crash",
                        "intent": "Attack",
                        "damage": {"normal": 16},
                    },
                    {"id": "STUN", "name": "Stun", "intent": "Stun"},
                    {
                        "id": "TERROR",
                        "name": "Terror",
                        "intent": "Debuff",
                        "powers": (
                            {"power_id": "VULNERABLE", "amount": 99, "target": "player"},
                        ),
                    },
                ),
                "attack_pattern": {
                    "initial_move": "CRASH",
                    "states": (
                        {"id": "CRASH_MOVE", "type": "move", "move_id": "CRASH"},
                        {
                            "id": "STUN_MOVE",
                            "type": "move",
                            "move_id": "STUN",
                            "next": "TERROR_MOVE",
                        },
                        {
                            "id": "TERROR_MOVE",
                            "type": "move",
                            "move_id": "TERROR",
                            "next": "CRASH_MOVE",
                        },
                    ),
                },
            },
        ),
        (
            {
                "id": "SHRIEK_ENCOUNTER",
                "name": "Shriek",
                "room_type": "Monster",
                "is_weak": True,
                "monsters": ({"id": "TERROR_EEL"},),
            },
        ),
        "SHRIEK_ENCOUNTER",
        deck=(
            {
                "card_id": "heavy_hit",
                "name": "Heavy Hit",
                "type": "attack",
                "cost": 0,
                "target": "enemy",
                "effects": {"damage": 35},
            },
        ),
    )
    state = _enter_source_combat(source_data)

    state = _play_first_card(state, "heavy_hit")

    assert state.combat is not None
    eel = state.combat.monsters[0]
    assert eel.hp == 35
    assert eel.metadata["shriek_used"] is True
    assert eel.metadata["force_stun_move"] is True
    assert any(event.kind == "monster_shriek" for event in state.combat.last_events)

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    eel = state.combat.monsters[0]
    assert state.combat.player.hp == 80
    assert eel.move_id == "TERROR"
    assert "force_stun_move" not in eel.metadata
    assert any(event.kind == "monster_stun_forced" for event in state.combat.last_events)
