from __future__ import annotations

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    Action,
    ActionType,
    MapEdgeState,
    MapNodeState,
    MapState,
    OrbState,
    RoomKind,
    RunPhase,
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


def _enter_combat(deck, *, player=None, relics=None, flags=None):
    state = new_run(
        seed=5100,
        character_id="TEST",
        ascension=0,
        source_data={
            "deck": deck,
            "player": player or {"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3},
            "flags": dict(flags or {}),
        },
    )
    state = _choose_first_ancient(state)
    if player is not None:
        state = state.model_copy(update={"player": state.player.model_copy(update=player)})
    if relics is not None:
        state = state.model_copy(update={"relics": tuple(relics)})
    state = _force_next_room(state, RoomKind.MONSTER)
    action = next(action for action in legal_actions(state) if action.type == "choose_node")
    return step(state, action)


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
    return step(
        state,
        next(action for action in legal_actions(state) if action.type == "end_turn"),
    )


def _choose_pending_card(state, card_id: str):
    assert state.combat is not None
    card = next(card for card in state.combat.hand if card.card_id == card_id)
    return step(
        state,
        next(
            action
            for action in legal_actions(state)
            if action.type == "choose_card" and action.card_instance_id == card.instance_id
        ),
    )


def _exhaust_pending_card(state, card_id: str):
    assert state.combat is not None
    card = next(card for card in state.combat.hand if card.card_id == card_id)
    return step(
        state,
        next(
            action
            for action in legal_actions(state)
            if action.type == "exhaust_card" and action.card_instance_id == card.instance_id
        ),
    )


def test_initial_draw_emits_per_card_draw_triggers() -> None:
    state = _enter_combat(
        (
            {"id": "DRAW_A", "name": "Draw A", "type": "Skill", "target": "Self", "cost": 0},
            {"id": "DRAW_B", "name": "Draw B", "type": "Skill", "target": "Self", "cost": 0},
        ),
        flags={"draw_per_turn": 2},
    )

    assert state.combat is not None
    draw_events = [event for event in state.combat.last_events if event.kind == "card_drawn"]
    assert len(draw_events) == 2
    assert {event.metadata["to_pile"] for event in draw_events} == {"hand"}
    assert {event.metadata["trigger"] for event in draw_events} == {"card_drawn"}


def test_targetless_card_play_normalizes_away_stale_target() -> None:
    state = _enter_combat(
        (
            {
                "id": "CENTER_SELF",
                "name": "Center Self",
                "type": "Skill",
                "target": "None",
                "cost": 0,
            },
        ),
        flags={"draw_per_turn": 1},
    )
    assert state.combat is not None
    card = state.combat.hand[0]
    stale_target = state.combat.monsters[0].monster_id

    state = step(
        state,
        Action(
            type=ActionType.PLAY_CARD,
            card_instance_id=card.instance_id,
            target_id=stale_target,
        ),
    )

    assert state.combat is not None
    assert state.combat.discard_pile[-1].instance_id == card.instance_id
    assert state.replay_log[-1].action.target_id is None


def test_single_enemy_card_play_normalizes_stale_target_to_alive_enemy() -> None:
    state = _enter_combat(
        (
            {
                "id": "QUICK_STAB",
                "name": "Quick Stab",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 4,
            },
        ),
        flags={"draw_per_turn": 1},
    )
    assert state.combat is not None
    card = state.combat.hand[0]
    alive_target = state.combat.monsters[0].monster_id

    state = step(
        state,
        Action(
            type=ActionType.PLAY_CARD,
            card_instance_id=card.instance_id,
            target_id="removed_monster",
        ),
    )

    assert state.combat is not None
    assert state.combat.discard_pile[-1].instance_id == card.instance_id
    assert state.replay_log[-1].action.target_id == alive_target
    assert state.combat.monsters[0].hp == state.combat.monsters[0].max_hp - 4


def test_source_card_fields_execute_hp_loss_energy_and_star_gain() -> None:
    state = _enter_combat(
        (
            {
                "id": "BLOODLETTING",
                "name": "Bloodletting",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "hp_loss": 3,
                "energy_gain": 2,
            },
            {
                "id": "GLOW",
                "name": "Glow",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Gain [star:1].",
            },
        )
    )

    state = _play_card(state, "bloodletting")
    assert state.combat is not None
    assert state.combat.player.hp == 77
    assert state.combat.player.energy == 5

    state = _play_card(state, "glow")
    assert state.combat is not None
    assert state.combat.player.resources["star"] == 1


def test_star_cost_cards_require_and_spend_stars() -> None:
    state = _enter_combat(
        (
            {
                "id": "COMET",
                "name": "Comet",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "star_cost": 2,
                "damage": 10,
            },
        ),
        player={"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3, "resources": {"star": 1}},
    )

    assert not any(action.type == "play_card" for action in legal_actions(state))

    assert state.combat is not None
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "player": state.combat.player.model_copy(update={"resources": {"star": 3}})
                }
            )
        }
    )
    state = _play_card(state, "comet")

    assert state.combat is not None
    assert state.combat.player.resources["star"] == 1
    assert state.combat.monsters[0].hp == state.combat.monsters[0].max_hp - 10


def test_generated_cards_use_source_destination_and_count() -> None:
    state = _enter_combat(
        (
            {
                "id": "BLADE_DANCE",
                "name": "Blade Dance",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Add 3 Shivs into your Hand.",
                "spawns_cards": ["SHIV"],
            },
        )
    )

    state = _play_card(state, "blade_dance")

    assert state.combat is not None
    assert [card.card_id for card in state.combat.hand] == ["shiv", "shiv", "shiv"]


def test_special_soul_source_text_adds_executable_soul_cards_to_piles() -> None:
    state = _enter_combat(
        (
            {
                "id": "SOUL_CALLER",
                "name": "Soul Caller",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Add 2 Souls into your Draw Pile.",
            },
        )
    )

    state = _play_card(state, "soul_caller")

    assert state.combat is not None
    assert [card.card_id for card in state.combat.draw_pile] == ["soul", "soul"]
    soul = state.combat.draw_pile[0]
    assert soul.exhausts is True
    assert soul.effects["sequence"] == [{"draw": 2}]


def test_soul_play_triggers_grant_summon_and_enemy_hp_loss() -> None:
    state = _enter_combat(
        (
            {
                "id": "DEVOUR_LIFE",
                "name": "Devour Life",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "Whenever you play a Soul, Summon 1.",
            },
            {
                "id": "HAUNT",
                "name": "Haunt",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "Whenever you play a Soul, a random enemy loses 6 HP.",
            },
            {
                "id": "SOUL",
                "name": "Soul",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Draw 0 cards.",
                "keywords_key": ("Exhaust",),
            },
        )
    )
    assert state.combat is not None
    monster_hp = state.combat.monsters[0].hp

    state = _play_card(state, "devour_life")
    state = _play_card(state, "haunt")
    state = _play_card(state, "soul")

    assert state.combat is not None
    assert state.combat.player.resources["summon"] == 1
    assert state.combat.monsters[0].hp == monster_hp - 6
    assert any(event.kind == "monster_hp_loss" for event in state.combat.last_events)


def test_dynamic_summon_and_soul_creation_scale_with_x_cost_energy() -> None:
    state = _enter_combat(
        (
            {
                "id": "DIRGE",
                "name": "Dirge",
                "type": "Skill",
                "target": "Self",
                "cost": -1,
                "is_x_cost": True,
                "description": "Summon 3 X times.\nAdd X Souls into your Draw Pile.",
                "spawns_cards": ("SOUL",),
            },
        )
    )

    state = _play_card(state, "dirge")

    assert state.combat is not None
    assert state.combat.player.resources["summon"] == 9
    assert [card.card_id for card in state.combat.draw_pile] == ["soul", "soul", "soul"]


def test_dynamic_forge_uses_previous_hits_on_target_this_turn() -> None:
    state = _enter_combat(
        (
            {
                "id": "SETUP_HIT",
                "name": "Setup Hit",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 1,
            },
            {
                "id": "BEAT_INTO_SHAPE",
                "name": "Beat Into Shape",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 1,
                "damage": 5,
                "description": (
                    "Forge X.\n"
                    "Forges an additional 5 for every other time you've hit the enemy this turn."
                ),
            },
        )
    )

    state = _play_card(state, "setup_hit")
    state = _play_card(state, "beat_into_shape")

    assert state.combat is not None
    assert state.combat.player.resources["forge"] == 6


def test_temporary_focus_scales_orb_passive_triggers_until_turn_end() -> None:
    state = _enter_combat(
        (
            {
                "id": "SYNCHRONIZE",
                "name": "Synchronize",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Gain 2 Focus this turn for each unique Orb you have.",
            },
            {
                "id": "FROST_TAP",
                "name": "Frost Tap",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Trigger the passive ability of all Frost Orbs.",
            },
        )
    )
    assert state.combat is not None
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={"orbs": (OrbState(orb_id="frost"), OrbState(orb_id="lightning"))}
            )
        }
    )

    state = _play_card(state, "synchronize")
    state = _play_card(state, "frost_tap")

    assert state.combat is not None
    assert state.combat.player.block == 6
    assert state.combat.player.statuses["temporary_focus"] == 4

    state = _end_turn(state)
    assert state.combat is not None
    assert "temporary_focus" not in state.combat.player.statuses


def test_sovereign_blade_modifiers_update_and_double_the_blade() -> None:
    state = _enter_combat(
        (
            {
                "id": "CONQUEROR",
                "name": "Conqueror",
                "type": "Skill",
                "target": "AnyEnemy",
                "cost": 0,
                "description": "Sovereign Blade deals double damage to the enemy this turn.",
            },
            {
                "id": "SEEKING_EDGE",
                "name": "Seeking Edge",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "Sovereign Blade now deals damage to ALL enemies.",
            },
            {
                "id": "SWORD_SAGE",
                "name": "Sword Sage",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "Sovereign Blade now hits an additional time.",
            },
            {
                "id": "SOVEREIGN_BLADE",
                "name": "Sovereign Blade",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 10,
            },
        )
    )
    assert state.combat is not None
    monster = state.combat.monsters[0].model_copy(update={"hp": 100, "max_hp": 100})
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )
    assert state.combat is not None
    monster_hp = state.combat.monsters[0].hp

    state = _play_card(state, "conqueror")
    state = _play_card(state, "seeking_edge")
    state = _play_card(state, "sword_sage")

    assert state.combat is not None
    blade = next(card for card in state.combat.hand if card.card_id == "sovereign_blade")
    assert blade.target.value == "all_enemies"
    assert blade.effects["sequence"] == [{"all_damage": 10}, {"all_damage": 10}]

    state = _play_card(state, "sovereign_blade")
    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 40


def test_special_stance_source_text_changes_attack_damage() -> None:
    state = _enter_combat(
        (
            {
                "id": "ENTER_WRATH",
                "name": "Enter Wrath",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Enter Wrath.",
            },
            {
                "id": "FOLLOW_UP",
                "name": "Follow Up",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 5,
            },
        )
    )

    state = _play_card(state, "enter_wrath")
    state = _play_card(state, "follow_up")

    assert state.combat is not None
    assert state.combat.player.statuses["stance_wrath"] == 1
    assert state.combat.monsters[0].hp == state.combat.monsters[0].max_hp - 10


def test_chosen_discard_waits_for_selected_hand_card() -> None:
    state = _enter_combat(
        (
            {
                "id": "ACROBATICS",
                "name": "Acrobatics",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Discard 1 card.",
            },
            {"id": "JUNK_A", "name": "Junk A", "type": "Skill", "target": "Self", "cost": 0},
            {"id": "JUNK_B", "name": "Junk B", "type": "Skill", "target": "Self", "cost": 0},
        )
    )

    state = _play_card(state, "acrobatics")

    assert state.combat is not None
    assert len(state.combat.discard_pile) == 1
    assert state.combat.discard_pile[0].card_id == "acrobatics"
    assert any(event.kind == "card_discard_choice_pending" for event in state.combat.last_events)
    assert len(state.combat.pending_choices) == 1
    assert state.combat.pending_choices[0].kind == "discard"
    assert state.combat.pending_choices[0].remaining == 1
    assert set(state.combat.pending_choices[0].candidate_ids) == {
        card.instance_id for card in state.combat.hand
    }

    discard_actions = [action for action in legal_actions(state) if action.type == "discard_card"]
    assert len(discard_actions) == 2
    assert {action.payload["choice_id"] for action in discard_actions} == {
        state.combat.pending_choices[0].choice_id
    }
    assert not any(action.type == "end_turn" for action in legal_actions(state))

    chosen_card_id = discard_actions[0].card_instance_id
    assert chosen_card_id is not None
    chosen = next(card for card in state.combat.hand if card.instance_id == chosen_card_id)
    state = step(state, discard_actions[0])

    assert state.combat is not None
    assert len(state.combat.discard_pile) == 2
    assert state.combat.discard_pile[-1].instance_id == chosen.instance_id
    assert all(card.instance_id != chosen.instance_id for card in state.combat.hand)
    assert "pending_card_choice" not in state.combat.metadata
    assert state.combat.pending_choices == ()
    assert state.combat.last_events[0].kind == "card_discarded_by_choice"
    assert any(
        event.kind == "card_discarded"
        and event.target_id == chosen.instance_id
        and event.metadata["reason"] == "chosen_discard"
        for event in state.combat.last_events
    )


def test_random_discard_resolves_immediately_without_choice() -> None:
    state = _enter_combat(
        (
            {
                "id": "RANDOM_DISCARD",
                "name": "Random Discard",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Discard 1 random card.",
            },
            {"id": "JUNK_A", "name": "Junk A", "type": "Skill", "target": "Self", "cost": 0},
            {"id": "JUNK_B", "name": "Junk B", "type": "Skill", "target": "Self", "cost": 0},
        )
    )

    state = _play_card(state, "random_discard")

    assert state.combat is not None
    assert len(state.combat.discard_pile) == 2
    assert "pending_card_choice" not in state.combat.metadata
    assert not any(action.type == "discard_card" for action in legal_actions(state))
    assert any(event.kind == "cards_discarded" for event in state.combat.last_events)
    assert any(
        event.kind == "card_discarded" and event.metadata["reason"] == "random_discard"
        for event in state.combat.last_events
    )


def test_chosen_exhaust_waits_for_selected_hand_card() -> None:
    state = _enter_combat(
        (
            {
                "id": "TRUE_GRIT_TEST",
                "name": "True Grit Test",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Exhaust 1 card.",
            },
            {"id": "JUNK_A", "name": "Junk A", "type": "Skill", "target": "Self", "cost": 0},
            {"id": "JUNK_B", "name": "Junk B", "type": "Skill", "target": "Self", "cost": 0},
        )
    )

    state = _play_card(state, "true_grit_test")

    assert state.combat is not None
    assert [card.card_id for card in state.combat.discard_pile] == ["true_grit_test"]
    assert any(event.kind == "card_exhaust_choice_pending" for event in state.combat.last_events)
    assert len(state.combat.pending_choices) == 1
    assert state.combat.pending_choices[0].kind == "exhaust"
    assert state.combat.pending_choices[0].remaining == 1

    exhaust_actions = [action for action in legal_actions(state) if action.type == "exhaust_card"]
    assert len(exhaust_actions) == 2
    assert {action.payload["choice_id"] for action in exhaust_actions} == {
        state.combat.pending_choices[0].choice_id
    }
    assert not any(action.type == "end_turn" for action in legal_actions(state))

    chosen_card_id = exhaust_actions[0].card_instance_id
    assert chosen_card_id is not None
    chosen = next(card for card in state.combat.hand if card.instance_id == chosen_card_id)
    state = step(state, exhaust_actions[0])

    assert state.combat is not None
    assert state.combat.exhaust_pile[-1].instance_id == chosen.instance_id
    assert all(card.instance_id != chosen.instance_id for card in state.combat.hand)
    assert "pending_card_choice" not in state.combat.metadata
    assert state.combat.pending_choices == ()
    assert state.combat.last_events[0].kind == "card_exhausted_by_choice"
    assert any(
        event.kind == "card_exhausted"
        and event.target_id == chosen.instance_id
        and event.metadata["reason"] == "chosen_exhaust"
        for event in state.combat.last_events
    )


def test_random_exhaust_resolves_immediately_without_choice() -> None:
    state = _enter_combat(
        (
            {
                "id": "RANDOM_EXHAUST",
                "name": "Random Exhaust",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Exhaust 1 random card.",
            },
            {"id": "JUNK_A", "name": "Junk A", "type": "Skill", "target": "Self", "cost": 0},
            {"id": "JUNK_B", "name": "Junk B", "type": "Skill", "target": "Self", "cost": 0},
        )
    )

    state = _play_card(state, "random_exhaust")

    assert state.combat is not None
    assert len(state.combat.exhaust_pile) == 1
    assert "pending_card_choice" not in state.combat.metadata
    assert not any(action.type == "exhaust_card" for action in legal_actions(state))
    assert any(event.kind == "cards_exhausted" for event in state.combat.last_events)
    assert any(
        event.kind == "card_exhausted" and event.metadata["reason"] == "random_exhaust"
        for event in state.combat.last_events
    )


def test_choose_card_add_retain_marks_selected_hand_card() -> None:
    state = _enter_combat(
        (
            {
                "id": "SNAP_TEST",
                "name": "Snap Test",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Add Retain to a card in your Hand.",
            },
            {"id": "JUNK_A", "name": "Junk A", "type": "Skill", "target": "Self", "cost": 0},
            {"id": "JUNK_B", "name": "Junk B", "type": "Skill", "target": "Self", "cost": 0},
        ),
        flags={"draw_per_turn": 3},
    )

    state = _play_card(state, "snap_test")

    assert state.combat is not None
    assert len(state.combat.pending_choices) == 1
    assert state.combat.pending_choices[0].kind == "add_retain"
    assert not any(action.type == "end_turn" for action in legal_actions(state))

    chosen = next(card for card in state.combat.hand if card.card_id == "junk_a")
    choose_action = next(
        action
        for action in legal_actions(state)
        if action.type == "choose_card" and action.card_instance_id == chosen.instance_id
    )
    state = step(state, choose_action)

    assert state.combat is not None
    chosen = next(card for card in state.combat.hand if card.card_id == "junk_a")
    assert chosen.custom["retain"] is True
    assert state.combat.pending_choices == ()
    assert any(event.kind == "card_retain_added" for event in state.combat.last_events)


def test_choose_card_exhausts_selected_card_from_draw_pile() -> None:
    state = _enter_combat(
        (
            {
                "id": "CLEANSE_TEST",
                "name": "Cleanse Test",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Exhaust 1 card from your Draw Pile.",
                "keywords_key": ("Innate",),
            },
            {"id": "JUNK_A", "name": "Junk A", "type": "Skill", "target": "Self", "cost": 0},
            {"id": "JUNK_B", "name": "Junk B", "type": "Skill", "target": "Self", "cost": 0},
        ),
        flags={"draw_per_turn": 1},
    )

    state = _play_card(state, "cleanse_test")

    assert state.combat is not None
    assert len(state.combat.pending_choices) == 1
    assert state.combat.pending_choices[0].kind == "exhaust"
    assert state.combat.pending_choices[0].zone == "draw_pile"

    chosen = state.combat.draw_pile[0]
    choose_action = next(
        action
        for action in legal_actions(state)
        if action.type == "choose_card" and action.card_instance_id == chosen.instance_id
    )
    state = step(state, choose_action)

    assert state.combat is not None
    assert state.combat.exhaust_pile[-1].instance_id == chosen.instance_id
    assert all(card.instance_id != chosen.instance_id for card in state.combat.draw_pile)
    assert any(
        event.kind == "card_exhausted"
        and event.target_id == chosen.instance_id
        and event.metadata["from_pile"] == "draw_pile"
        for event in state.combat.last_events
    )


def test_frantic_escape_gains_sandpit_and_increases_own_cost() -> None:
    state = _enter_combat(
        (
            {
                "id": "FRANTIC_ESCAPE",
                "name": "Frantic Escape",
                "type": "Status",
                "target": "Self",
                "cost": 1,
                "description": (
                    "Get farther away. Increase Sandpit by 1. "
                    "Increase the cost of this card by 1."
                ),
            },
        )
    )

    state = _play_card(state, "frantic_escape")

    assert state.combat is not None
    assert state.combat.player.resources["sandpit"] == 1
    escaped = next(card for card in state.combat.discard_pile if card.card_id == "frantic_escape")
    assert escaped.cost == 2


def test_hidden_gem_adds_replay_to_random_draw_pile_card() -> None:
    state = _enter_combat(
        (
            {
                "id": "HIDDEN_GEM",
                "name": "Hidden Gem",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "keywords_key": ("Innate",),
                "description": "A random card without Replay in your Draw Pile gains Replay 2.",
            },
            {"id": "JUNK_A", "name": "Junk A", "type": "Skill", "target": "Self", "cost": 0},
            {"id": "JUNK_B", "name": "Junk B", "type": "Skill", "target": "Self", "cost": 0},
        ),
        flags={"draw_per_turn": 1},
    )

    state = _play_card(state, "hidden_gem")

    assert state.combat is not None
    replay_cards = [card for card in state.combat.draw_pile if card.custom.get("replay")]
    assert len(replay_cards) == 1
    assert replay_cards[0].custom["replay"] == 2


def test_aggression_moves_upgraded_attack_from_discard_on_turn_start() -> None:
    state = _enter_combat(
        (
            {
                "id": "AGGRESSION",
                "name": "Aggression",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "keywords_key": ("Innate",),
                "description": (
                    "At the start of your turn, put a random Attack from your "
                    "Discard Pile into your Hand and Upgrade it."
                ),
            },
            {
                "id": "BASH_TEST",
                "name": "Bash Test",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 4,
            },
        ),
        flags={"draw_per_turn": 1},
    )
    state = _play_card(state, "aggression")
    assert state.combat is not None
    attack = state.combat.draw_pile[0]
    combat = state.combat.model_copy(
        update={"draw_pile": (), "discard_pile": (attack,), "hand": ()}
    )
    state = state.model_copy(update={"combat": combat, "player": combat.player})

    state = _end_turn(state)

    assert state.combat is not None
    moved = next(card for card in state.combat.hand if card.card_id == "bash_test")
    assert moved.upgraded is True
    assert any(event.kind == "random_card_moved" for event in state.combat.last_events)


def test_hellraiser_plays_drawn_strike_against_random_enemy() -> None:
    state = _enter_combat(
        (
            {
                "id": "HELLRAISER",
                "name": "Hellraiser",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "keywords_key": ("Innate",),
                "description": (
                    "Whenever you draw a card containing Strike, it is played "
                    "against a random enemy."
                ),
            },
            {
                "id": "STRIKE_TEST",
                "name": "Strike Test",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 5,
            },
        ),
        flags={"draw_per_turn": 1},
    )
    state = _play_card(state, "hellraiser")
    assert state.combat is not None
    strike = state.combat.draw_pile[0]
    monster_hp = state.combat.monsters[0].hp
    combat = state.combat.model_copy(update={"draw_pile": (strike,), "hand": ()})
    state = state.model_copy(update={"combat": combat, "player": combat.player})

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 5
    assert any(event.kind == "card_played_by_trigger" for event in state.combat.last_events)


def test_juggling_copies_every_third_attack_to_hand() -> None:
    state = _enter_combat(
        (
            {
                "id": "JUGGLING",
                "name": "Juggling",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "Add a copy of the third Attack you play each turn into your Hand.",
            },
            {
                "id": "HIT_A",
                "name": "Hit A",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 1,
            },
            {
                "id": "HIT_B",
                "name": "Hit B",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 1,
            },
            {
                "id": "HIT_C",
                "name": "Hit C",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 1,
            },
        ),
        flags={"draw_per_turn": 4},
    )

    state = _play_card(state, "juggling")
    state = _play_card(state, "hit_a")
    state = _play_card(state, "hit_b")
    state = _play_card(state, "hit_c")

    assert state.combat is not None
    assert any(event.kind == "card_copied_to_hand" for event in state.combat.last_events)
    assert any(card.custom.get("copied_from_card_id") == "hit_c" for card in state.combat.hand)


def test_master_planner_adds_sly_to_played_skill() -> None:
    state = _enter_combat(
        (
            {
                "id": "MASTER_PLANNER",
                "name": "Master Planner",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "When you play a Skill, it gains Sly.",
            },
            {"id": "TACTIC", "name": "Tactic", "type": "Skill", "target": "Self", "cost": 0},
        ),
        flags={"draw_per_turn": 2},
    )

    state = _play_card(state, "master_planner")
    state = _play_card(state, "tactic")

    assert state.combat is not None
    tactic = next(card for card in state.combat.discard_pile if card.card_id == "tactic")
    assert tactic.custom["sly"] is True


def test_well_laid_plans_pends_retain_choice_before_end_turn_discard() -> None:
    state = _enter_combat(
        (
            {
                "id": "WELL_LAID_PLANS",
                "name": "Well-Laid Plans",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "At the end of your turn, Retain up to 1 card.",
            },
            {"id": "JUNK_A", "name": "Junk A", "type": "Skill", "target": "Self", "cost": 0},
            {"id": "JUNK_B", "name": "Junk B", "type": "Skill", "target": "Self", "cost": 0},
        ),
        flags={"draw_per_turn": 3},
    )

    state = _play_card(state, "well_laid_plans")
    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.pending_choices
    assert state.combat.pending_choices[0].kind == "add_retain"
    assert {card.card_id for card in state.combat.hand} == {"junk_a", "junk_b"}

    state = _choose_pending_card(state, "junk_a")
    assert state.combat is not None
    assert next(card for card in state.combat.hand if card.card_id == "junk_a").custom["retain"]
    state = _end_turn(state)

    assert state.combat is not None
    assert any(
        event.kind == "cards_retained" and "junk_a" in event.metadata["card_ids"]
        for event in state.combat.last_events
    )
    assert any(event.kind == "hand_discarded" for event in state.combat.last_events)


def test_stampede_plays_random_hand_attack_before_end_turn_discard() -> None:
    state = _enter_combat(
        (
            {
                "id": "STAMPEDE",
                "name": "Stampede",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": (
                    "At the end of your turn, 1 random Attack in your Hand is "
                    "played against a random enemy."
                ),
            },
            {
                "id": "HIT_A",
                "name": "Hit A",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 4,
            },
        ),
        flags={"draw_per_turn": 2},
    )
    state = _play_card(state, "stampede")
    assert state.combat is not None
    monster_hp = state.combat.monsters[0].hp

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 4
    assert any(event.kind == "card_played_by_trigger" for event in state.combat.last_events)


def test_largesse_adds_random_colorless_card_to_hand() -> None:
    state = _enter_combat(
        (
            {
                "id": "LARGESSE",
                "name": "Largesse",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Another player adds 1 random Colorless card to their Hand.",
            },
        ),
        flags={
            "draw_per_turn": 1,
            "card_pool": (
                {
                    "id": "COLORLESS_TEST",
                    "name": "Colorless Test",
                    "type": "Skill",
                    "target": "Self",
                    "cost": 0,
                    "color": "Colorless",
                    "rarity": "Uncommon",
                },
            ),
        },
    )

    state = _play_card(state, "largesse")

    assert state.combat is not None
    assert any(card.card_id == "colorless_test" for card in state.combat.hand)


def test_child_of_the_stars_grants_block_when_star_is_spent() -> None:
    state = _enter_combat(
        (
            {
                "id": "CHILD_OF_THE_STARS",
                "name": "Child of the Stars",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "Whenever you spend [star:1], gain 2 Block for each [star:1] spent.",
            },
            {
                "id": "STAR_SKILL",
                "name": "Star Skill",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "star_cost": 1,
            },
        ),
        player={"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3, "resources": {"star": 1}},
        flags={"draw_per_turn": 2},
    )

    state = _play_card(state, "child_of_the_stars")
    state = _play_card(state, "star_skill")

    assert state.combat is not None
    assert state.combat.player.block == 2


def test_beacon_of_hope_emits_ally_block_when_player_gains_block() -> None:
    state = _enter_combat(
        (
            {
                "id": "BEACON_OF_HOPE",
                "name": "Beacon of Hope",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": (
                    "Whenever you gain Block on your turn, other players gain "
                    "half that much Block."
                ),
            },
            {
                "id": "DEFEND_TEST",
                "name": "Defend Test",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "block": 6,
            },
        ),
        flags={"draw_per_turn": 2},
    )

    state = _play_card(state, "beacon_of_hope")
    state = _play_card(state, "defend_test")

    assert state.combat is not None
    assert any(
        event.kind == "ally_block_gained" and event.amount == 3
        for event in state.combat.last_events
    )


def test_flanking_applies_enemy_status_marker() -> None:
    state = _enter_combat(
        (
            {
                "id": "FLANKING",
                "name": "Flanking",
                "type": "Skill",
                "target": "Enemy",
                "cost": 0,
                "description": "The enemy takes double attack damage from other players this turn.",
            },
        )
    )

    state = _play_card(state, "flanking")

    assert state.combat is not None
    assert state.combat.monsters[0].statuses["flanking"] == 1


def test_reaper_form_applies_doom_equal_to_attack_damage() -> None:
    state = _enter_combat(
        (
            {
                "id": "REAPER_FORM",
                "name": "Reaper Form",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "Whenever Attacks deal damage, they also apply that much Doom.",
            },
            {
                "id": "CUT",
                "name": "Cut",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 6,
            },
        ),
        flags={"draw_per_turn": 2},
    )

    state = _play_card(state, "reaper_form")
    state = _play_card(state, "cut")

    assert state.combat is not None
    assert state.combat.monsters[0].statuses["doom"] == 6


def test_monarchs_gaze_applies_temporary_strength_loss_on_attack_damage() -> None:
    state = _enter_combat(
        (
            {
                "id": "MONARCHS_GAZE",
                "name": "Monarch's Gaze",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "Whenever you attack an enemy, it loses 1 Strength this turn.",
            },
            {
                "id": "CUT",
                "name": "Cut",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 3,
            },
        ),
        flags={"draw_per_turn": 2},
    )

    state = _play_card(state, "monarchs_gaze")
    state = _play_card(state, "cut")

    assert state.combat is not None
    assert state.combat.monsters[0].statuses["temporary_strength"] == -1


def test_tracking_doubles_attack_damage_against_weak_enemy() -> None:
    state = _enter_combat(
        (
            {
                "id": "TRACKING",
                "name": "Tracking",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "Weak enemies take double damage from Attacks.",
            },
            {
                "id": "CUT",
                "name": "Cut",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 4,
            },
        ),
        flags={"draw_per_turn": 2},
    )
    assert state.combat is not None
    weak_monster = state.combat.monsters[0].model_copy(update={"statuses": {"weak": 1}})
    combat = state.combat.model_copy(update={"monsters": (weak_monster,)})
    state = state.model_copy(update={"combat": combat, "player": combat.player})
    monster_hp = weak_monster.hp

    state = _play_card(state, "tracking")
    state = _play_card(state, "cut")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 8


def test_tank_doubles_incoming_player_damage() -> None:
    state = _enter_combat(
        (
            {
                "id": "TANK",
                "name": "Tank",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": (
                    "Take double damage from enemies. Allies take half damage from enemies."
                ),
            },
        ),
        flags={"draw_per_turn": 1},
    )
    state = _play_card(state, "tank")
    assert state.combat is not None
    hp_before = state.combat.player.hp
    incoming = state.combat.monsters[0].intent_damage

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.player.hp == hp_before - incoming * 2


def test_choose_card_moves_selected_discard_card_to_draw_top() -> None:
    state = _enter_combat(
        (
            {
                "id": "HEADBUTT_TEST",
                "name": "Headbutt Test",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": (
                    "Put a card from your Discard Pile on top of your Draw Pile."
                ),
            },
            {"id": "JUNK_A", "name": "Junk A", "type": "Skill", "target": "Self", "cost": 0},
        ),
        flags={"draw_per_turn": 1},
    )
    assert state.combat is not None
    discarded = state.combat.draw_pile[0]
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={"draw_pile": (), "discard_pile": (discarded,)}
            )
        }
    )

    state = _play_card(state, "headbutt_test")

    assert state.combat is not None
    assert state.combat.pending_choices[0].kind == "move_to_draw_top"
    choose_action = next(
        action
        for action in legal_actions(state)
        if action.type == "choose_card" and action.card_instance_id == discarded.instance_id
    )
    state = step(state, choose_action)

    assert state.combat is not None
    assert state.combat.draw_pile[0].instance_id == discarded.instance_id
    assert all(card.instance_id != discarded.instance_id for card in state.combat.discard_pile)
    assert any(
        event.kind == "card_moved_by_choice"
        and event.metadata["to_pile"] == "draw_pile_top"
        for event in state.combat.last_events
    )


def test_choose_card_transforms_selected_draw_cards_into_target_cards() -> None:
    state = _enter_combat(
        (
            {
                "id": "CHARGE_TEST",
                "name": "Charge Test",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": (
                    "Choose 2 cards in your Draw Pile to Transform into "
                    "Minion Dive Bombs."
                ),
                "keywords_key": ("Innate",),
            },
            {"id": "JUNK_A", "name": "Junk A", "type": "Skill", "target": "Self", "cost": 0},
            {"id": "JUNK_B", "name": "Junk B", "type": "Skill", "target": "Self", "cost": 0},
        ),
        flags={
            "draw_per_turn": 1,
            "card_library": (
                {
                    "id": "MINION_DIVE_BOMB",
                    "name": "Minion Dive Bomb",
                    "type": "Attack",
                    "target": "Enemy",
                    "cost": 1,
                    "damage": 12,
                },
            ),
        },
    )

    state = _play_card(state, "charge_test")

    assert state.combat is not None
    assert state.combat.pending_choices[0].kind == "transform"
    assert state.combat.pending_choices[0].remaining == 2
    first_id = state.combat.draw_pile[0].instance_id
    first_action = next(
        action
        for action in legal_actions(state)
        if action.type == "choose_card" and action.card_instance_id == first_id
    )
    state = step(state, first_action)

    assert state.combat is not None
    first = next(card for card in state.combat.draw_pile if card.instance_id == first_id)
    assert first.card_id == "minion_dive_bomb"
    assert "transformed_from_card_id" not in first.custom
    assert state.combat.pending_choices[0].remaining == 1

    second_action = next(action for action in legal_actions(state) if action.type == "choose_card")
    second_id = second_action.card_instance_id
    assert second_id is not None
    assert second_id != first_id
    state = step(state, second_action)

    assert state.combat is not None
    assert state.combat.pending_choices == ()
    transformed = [card for card in state.combat.draw_pile if card.card_id == "minion_dive_bomb"]
    assert len(transformed) == 2
    assert any(event.kind == "card_transformed_by_choice" for event in state.combat.last_events)


def test_generated_discovery_choice_adds_selected_card_to_hand() -> None:
    state = _enter_combat(
        (
            {
                "id": "DISCOVERY_TEST",
                "name": "Discovery Test",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": (
                    "Choose 1 of 3 random Colorless cards to add into your Hand. "
                    "It's free to play this turn."
                ),
            },
        ),
        flags={
            "draw_per_turn": 1,
            "card_library": (
                {
                    "id": "COLORLESS_A",
                    "name": "Colorless A",
                    "color": "Colorless",
                    "type": "Skill",
                    "target": "Self",
                    "cost": 1,
                    "block": 3,
                },
                {
                    "id": "COLORLESS_B",
                    "name": "Colorless B",
                    "color": "Colorless",
                    "type": "Attack",
                    "target": "Enemy",
                    "cost": 1,
                    "damage": 4,
                },
                {
                    "id": "COLORLESS_C",
                    "name": "Colorless C",
                    "color": "Colorless",
                    "type": "Skill",
                    "target": "Self",
                    "cost": 2,
                    "block": 5,
                },
                {
                    "id": "STRIKE_A",
                    "name": "Strike A",
                    "color": "Test",
                    "type": "Attack",
                    "target": "Enemy",
                    "cost": 1,
                    "damage": 6,
                },
            ),
        },
    )

    state = _play_card(state, "discovery_test")

    assert state.combat is not None
    assert state.combat.pending_choices[0].kind == "move_to_hand"
    assert state.combat.pending_choices[0].zone == "generated"
    choice_actions = [action for action in legal_actions(state) if action.type == "choose_card"]
    assert len(choice_actions) == 3

    state = step(state, choice_actions[0])

    assert state.combat is not None
    assert state.combat.pending_choices == ()
    assert len(state.combat.hand) == 1
    assert state.combat.hand[0].card_id.startswith("colorless_")
    assert state.combat.hand[0].cost == 0
    assert state.combat.hand[0].custom["free_to_play_this_turn"] is True


def test_choose_card_copies_selected_attack_or_power_to_hand() -> None:
    state = _enter_combat(
        (
            {
                "id": "DUAL_WIELD_TEST",
                "name": "Dual Wield Test",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": (
                    "Choose an Attack or Power card. "
                    "Add a copy of that card into your Hand."
                ),
            },
            {
                "id": "JAB",
                "name": "Jab",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 3,
            },
            {
                "id": "PLAIN_SKILL",
                "name": "Plain Skill",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "block": 2,
            },
        ),
        flags={"draw_per_turn": 3},
    )

    state = _play_card(state, "dual_wield_test")

    assert state.combat is not None
    choice_actions = [action for action in legal_actions(state) if action.type == "choose_card"]
    assert len(choice_actions) == 1
    original = next(card for card in state.combat.hand if card.card_id == "jab")
    assert choice_actions[0].card_instance_id == original.instance_id

    state = step(state, choice_actions[0])

    assert state.combat is not None
    copies = [card for card in state.combat.hand if card.card_id == "jab"]
    assert len(copies) == 2
    assert len({card.instance_id for card in copies}) == 2
    assert any(event.kind == "card_copied_by_choice" for event in state.combat.last_events)


def test_choose_card_plays_selected_skill_multiple_times() -> None:
    state = _enter_combat(
        (
            {
                "id": "DECISIONS_TEST",
                "name": "Decisions Test",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Choose a Skill in your Hand and play it 3 times.",
            },
            {
                "id": "FOCUS_BLOCK",
                "name": "Focus Block",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "block": 2,
            },
        ),
        flags={"draw_per_turn": 2},
    )

    state = _play_card(state, "decisions_test")

    assert state.combat is not None
    choice_action = next(action for action in legal_actions(state) if action.type == "choose_card")
    state = step(state, choice_action)

    assert state.combat is not None
    assert state.combat.player.block == 6
    assert state.combat.pending_choices == ()
    assert sum(event.kind == "card_played_by_choice" for event in state.combat.last_events) == 3


def test_sly_card_triggers_effect_when_discarded_from_hand() -> None:
    state = _enter_combat(
        (
            {
                "id": "DISCARD_TOOL",
                "name": "Discard Tool",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Discard 1 card.",
            },
            {
                "id": "SLY_DAGGER",
                "name": "Sly Dagger",
                "type": "Attack",
                "target": "Enemy",
                "cost": 1,
                "damage": 5,
                "keywords_key": ("Sly",),
            },
        ),
        flags={"draw_per_turn": 2},
    )

    state = _play_card(state, "discard_tool")

    assert state.combat is not None
    sly_card = next(card for card in state.combat.hand if card.card_id == "sly_dagger")
    discard_action = next(
        action
        for action in legal_actions(state)
        if action.type == "discard_card" and action.card_instance_id == sly_card.instance_id
    )
    monster_hp = state.combat.monsters[0].hp
    state = step(state, discard_action)

    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 5
    assert any(event.kind == "sly_card_triggered" for event in state.combat.last_events)


def test_sly_card_does_not_trigger_from_end_turn_hand_cleanup() -> None:
    state = _enter_combat(
        (
            {
                "id": "SLY_DAGGER",
                "name": "Sly Dagger",
                "type": "Attack",
                "target": "Enemy",
                "cost": 1,
                "damage": 5,
                "keywords_key": ("Sly",),
            },
        ),
        flags={"draw_per_turn": 1},
    )

    assert state.combat is not None
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"draw_per_turn": 0})}
    )
    assert state.combat is not None
    monster_hp = state.combat.monsters[0].hp
    end_turn = next(action for action in legal_actions(state) if action.type == "end_turn")
    state = step(state, end_turn)

    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp
    assert [card.card_id for card in state.combat.discard_pile] == ["sly_dagger"]
    assert not any(event.kind == "sly_card_triggered" for event in state.combat.last_events)


def test_unplayable_card_cannot_be_played_but_end_turn_hand_effect_resolves() -> None:
    state = _enter_combat(
        (
            {
                "id": "BURN_TEST",
                "name": "Burn Test",
                "type": "Status",
                "target": "Self",
                "cost": -1,
                "description": "At the end of your turn, if this is in your Hand, take 2 damage.",
                "keywords_key": ("Unplayable",),
            },
        ),
        flags={"draw_per_turn": 1},
    )

    assert state.combat is not None
    assert not any(action.type == "play_card" for action in legal_actions(state))

    end_turn = next(action for action in legal_actions(state) if action.type == "end_turn")
    state = step(state, end_turn)

    assert state.combat is not None
    assert any(
        event.kind == "player_damaged"
        and event.amount == 2
        and event.metadata.get("reason") == "end_turn_hand_card"
        for event in state.combat.last_events
    )


def test_innate_cards_are_drawn_first_and_can_exceed_normal_opening_draw() -> None:
    deck = tuple(
        {
            "id": f"INNATE_{index}",
            "name": f"Innate {index}",
            "type": "Skill",
            "target": "Self",
            "cost": 0,
            "keywords_key": ("Innate",),
        }
        for index in range(6)
    ) + tuple(
        {
            "id": f"NORMAL_{index}",
            "name": f"Normal {index}",
            "type": "Skill",
            "target": "Self",
            "cost": 0,
        }
        for index in range(4)
    )
    state = _enter_combat(deck, flags={"draw_per_turn": 5})

    assert state.combat is not None
    assert len(state.combat.hand) == 6
    assert {card.card_id for card in state.combat.hand} == {
        f"innate_{index}" for index in range(6)
    }
    assert any(event.kind == "innate_cards_prioritized" for event in state.combat.last_events)
    assert len([event for event in state.combat.last_events if event.kind == "card_drawn"]) == 6


def test_more_than_ten_innate_cards_leave_extras_on_top_of_draw_pile() -> None:
    deck = tuple(
        {
            "id": f"INNATE_{index}",
            "name": f"Innate {index}",
            "type": "Skill",
            "target": "Self",
            "cost": 0,
            "keywords_key": ("Innate",),
        }
        for index in range(12)
    )
    state = _enter_combat(deck, flags={"draw_per_turn": 5})

    assert state.combat is not None
    assert len(state.combat.hand) == 10
    assert [card.card_id for card in state.combat.draw_pile[:2]] == ["innate_10", "innate_11"]


def test_osty_action_card_uses_companion_state_to_damage_enemy() -> None:
    state = _enter_combat(
        (
            {
                "id": "OSTY_STRIKE",
                "name": "Osty Strike",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Osty deals 6 damage to a random enemy.",
            },
        ),
        flags={"draw_per_turn": 1},
    )
    assert state.combat is not None
    monster_hp = state.combat.monsters[0].hp

    state = _play_card(state, "osty_strike")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 6
    assert state.combat.metadata["osty"]["alive"] is True
    assert any(event.kind == "osty_damaged_enemies" for event in state.combat.last_events)


def test_osty_attack_source_damage_is_not_applied_twice() -> None:
    state = _enter_combat(
        (
            {
                "id": "POKE",
                "name": "Poke",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 6,
                "description": "Osty deals 6 damage.",
                "tags": ("OstyAttack",),
            },
        ),
        flags={"draw_per_turn": 1},
    )
    assert state.combat is not None
    monster_hp = state.combat.monsters[0].hp

    state = _play_card(state, "poke")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 6


def test_calcify_adds_persistent_osty_attack_damage() -> None:
    state = _enter_combat(
        (
            {
                "id": "CALCIFY",
                "name": "Calcify",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "Osty's attacks deal 4 additional damage.",
                "powers_applied": [{"power": "Calcify", "amount": 4}],
            },
            {
                "id": "POKE",
                "name": "Poke",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 6,
                "description": "Osty deals 6 damage.",
                "tags": ("OstyAttack",),
            },
        ),
        flags={"draw_per_turn": 2},
    )
    assert state.combat is not None
    state = _play_card(state, "calcify")
    monster_hp = state.combat.monsters[0].hp

    state = _play_card(state, "poke")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 10
    assert state.combat.metadata["osty"]["damage_bonus"] == 4


def test_bone_shards_damage_block_and_death_are_gated_by_osty_state() -> None:
    state = _enter_combat(
        (
            {
                "id": "BONE_SHARDS",
                "name": "Bone Shards",
                "type": "Attack",
                "target": "AllEnemies",
                "cost": 0,
                "damage": 9,
                "block": 9,
                "description": (
                    "If Osty is alive, he deals 9 damage to ALL enemies "
                    "and you gain 9 Block. Osty dies."
                ),
                "tags": ("OstyAttack",),
            },
        ),
        flags={"draw_per_turn": 1},
    )
    assert state.combat is not None
    monster_hp = state.combat.monsters[0].hp

    state = _play_card(state, "bone_shards")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 9
    assert state.combat.player.block == 9
    assert state.combat.metadata["osty"]["alive"] is False
    assert state.combat.metadata["osty"]["hp"] == 0


def test_sacrifice_blocks_for_double_osty_max_hp_and_kills_osty() -> None:
    state = _enter_combat(
        (
            {
                "id": "SACRIFICE",
                "name": "Sacrifice",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": (
                    "If Osty is alive, he dies and you gain Block equal to double his Max HP."
                ),
            },
        ),
        flags={"draw_per_turn": 1},
    )
    assert state.combat is not None
    metadata = {**state.combat.metadata, "osty": {"alive": True, "hp": 12, "max_hp": 12}}
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"metadata": metadata})}
    )

    state = _play_card(state, "sacrifice")

    assert state.combat is not None
    assert state.combat.player.block == 24
    assert state.combat.metadata["osty"]["alive"] is False


def test_flatten_costs_zero_after_osty_has_attacked_this_turn() -> None:
    state = _enter_combat(
        (
            {
                "id": "POKE",
                "name": "Poke",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 6,
                "description": "Osty deals 6 damage.",
                "tags": ("OstyAttack",),
            },
            {
                "id": "FLATTEN",
                "name": "Flatten",
                "type": "Attack",
                "target": "Enemy",
                "cost": 2,
                "damage": 12,
                "description": (
                    "Osty deals 12 damage. "
                    "This card costs 0 [energy:1] if Osty has attacked this turn."
                ),
                "tags": ("OstyAttack",),
            },
        ),
        player={"hp": 80, "max_hp": 80, "energy": 1, "max_energy": 1},
        flags={"draw_per_turn": 2},
    )

    assert state.combat is not None
    flatten = next(card for card in state.combat.hand if card.card_id == "flatten")
    assert not any(
        action.type == "play_card" and action.card_instance_id == flatten.instance_id
        for action in legal_actions(state)
    )

    state = _play_card(state, "poke")

    assert state.combat is not None
    flatten = next(card for card in state.combat.hand if card.card_id == "flatten")
    assert any(
        action.type == "play_card" and action.card_instance_id == flatten.instance_id
        for action in legal_actions(state)
    )
    state = _play_card(state, "flatten")
    assert state.combat is not None
    assert state.combat.last_events[0].kind == "card_played"
    assert state.combat.last_events[0].amount == 0


def test_rattle_hits_once_plus_previous_osty_attacks_this_turn() -> None:
    state = _enter_combat(
        (
            {
                "id": "POKE",
                "name": "Poke",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 6,
                "description": "Osty deals 6 damage.",
                "tags": ("OstyAttack",),
            },
            {
                "id": "RATTLE",
                "name": "Rattle",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 7,
                "description": (
                    "Osty deals 7 damage. "
                    "Hits an additional time for each other time he has attacked this turn."
                ),
                "tags": ("OstyAttack",),
            },
        ),
        flags={"draw_per_turn": 2},
    )
    assert state.combat is not None
    state = _play_card(state, "poke")
    monster_hp = state.combat.monsters[0].hp

    state = _play_card(state, "rattle")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 14
    assert state.combat.metadata["osty_attacks_this_turn"] == 3


def test_sic_em_grants_summon_when_osty_hits_marked_enemy() -> None:
    state = _enter_combat(
        (
            {
                "id": "SIC_EM",
                "name": "Sic 'Em",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 5,
                "description": (
                    "Osty deals 5 damage. "
                    "Whenever Osty hits this enemy this turn, Summon 2."
                ),
                "powers_applied": [{"power": "SicEm", "amount": 2}],
                "tags": ("OstyAttack",),
            },
        ),
        flags={"draw_per_turn": 1},
    )

    state = _play_card(state, "sic_em")

    assert state.combat is not None
    assert state.combat.player.resources["summon"] == 2
    assert state.combat.metadata["osty"]["hp"] == 22
    assert state.combat.metadata["osty"]["max_hp"] == 22
    assert any(
        event.kind == "player_resource_changed"
        and event.metadata.get("trigger") == "sicem"
        for event in state.combat.last_events
    )


def test_necro_mastery_mirrors_osty_hp_loss_to_enemies() -> None:
    state = _enter_combat(
        (
            {
                "id": "NECRO_MASTERY",
                "name": "Necro Mastery",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": (
                    "Summon 5. Whenever Osty loses HP, ALL enemies lose that much HP as well."
                ),
            },
            {
                "id": "SACRIFICE",
                "name": "Sacrifice",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": (
                    "If Osty is alive, he dies and you gain Block equal to double his Max HP."
                ),
            },
        ),
        flags={"draw_per_turn": 2},
    )
    assert state.combat is not None
    monster_hp = state.combat.monsters[0].hp

    state = _play_card(state, "necro_mastery")
    state = _play_card(state, "sacrifice")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 25
    assert any(event.kind == "osty_hp_loss_damaged_enemies" for event in state.combat.last_events)


def test_summon_restores_osty_and_raises_max_hp_when_overfilled() -> None:
    state = _enter_combat(
        (
            {
                "id": "BODYGUARD",
                "name": "Bodyguard",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Summon 6.",
            },
        ),
        flags={"draw_per_turn": 1},
    )
    assert state.combat is not None
    metadata = {
        **state.combat.metadata,
        "osty": {"alive": True, "hp": 18, "max_hp": 20},
    }
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"metadata": metadata})}
    )

    state = _play_card(state, "bodyguard")

    assert state.combat is not None
    assert state.combat.player.resources["summon"] == 6
    assert state.combat.metadata["osty"]["alive"] is True
    assert state.combat.metadata["osty"]["hp"] == 24
    assert state.combat.metadata["osty"]["max_hp"] == 24
    assert any(
        event.kind == "osty_summoned"
        and event.amount == 6
        and event.metadata["max_hp_gained"] == 4
        for event in state.combat.last_events
    )


def test_bone_flute_blocks_when_osty_attacks() -> None:
    state = _enter_combat(
        (
            {
                "id": "POKE",
                "name": "Poke",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 6,
                "description": "Osty deals 6 damage.",
                "tags": ("OstyAttack",),
            },
        ),
        relics=("BONE_FLUTE",),
        flags={"draw_per_turn": 1},
    )

    state = _play_card(state, "poke")

    assert state.combat is not None
    assert state.combat.player.block == 2


def test_played_card_destination_emits_discard_or_exhaust_trigger() -> None:
    state = _enter_combat(
        (
            {
                "id": "BURN_NOW",
                "name": "Burn Now",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "exhaust": True,
            },
        )
    )

    state = _play_card(state, "burn_now")

    assert state.combat is not None
    assert [card.card_id for card in state.combat.exhaust_pile] == ["burn_now"]
    assert any(
        event.kind == "card_exhausted"
        and event.metadata["reason"] == "played_card_destination"
        for event in state.combat.last_events
    )


def test_upgrade_source_cards_mutate_values() -> None:

    state = new_run(
        seed=5101,
        character_id="TEST",
        ascension=0,
        source_data={
            "deck": (
                {
                    "id": "STRIKE_TEST",
                    "name": "Strike Test",
                    "type": "Attack",
                    "target": "AnyEnemy",
                    "cost": 1,
                    "damage": 6,
                    "upgrade": {"damage": "+5"},
                },
            )
        },
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.REST)
    state = step(
        state,
        next(action for action in legal_actions(state) if action.type == "choose_node"),
    )
    strike = state.master_deck[0]
    state = step(
        state,
        next(
            action
            for action in legal_actions(state)
            if action.type == "smith" and action.target_id == strike.instance_id
        ),
    )

    assert state.master_deck[0].upgraded is True
    assert state.master_deck[0].effects["sequence"][0]["damage"] == 11


def test_start_of_combat_relics_apply_block_heal_and_statuses() -> None:
    state = _enter_combat(
        (),
        player={"hp": 70, "max_hp": 80, "energy": 3, "max_energy": 3},
        relics=("anchor", "blood_vial", "vajra"),
    )

    assert state.combat is not None
    assert state.combat.player.block == 10
    assert state.combat.player.hp == 72
    assert state.combat.player.statuses["strength"] == 1
    assert {event.kind for event in state.combat.last_events} >= {
        "player_block",
        "player_healed",
        "status_applied",
    }


def test_bag_of_preparation_draws_extra_opening_cards() -> None:
    state = _enter_combat(
        tuple(
            {
                "id": f"SETUP_{index}",
                "name": f"Setup {index}",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
            }
            for index in range(1, 8)
        ),
        relics=("bag_of_preparation",),
    )

    assert state.combat is not None
    assert len(state.combat.hand) == 7
    assert state.combat.draw_pile == ()
    assert any(event.kind == "relic_bonus_draw_applied" for event in state.combat.last_events)


def test_akabeko_vigor_adds_damage_to_next_attack_and_is_consumed() -> None:
    state = _enter_combat(
        (
            {
                "id": "TAP",
                "name": "Tap",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 1,
            },
        ),
        relics=("akabeko",),
        flags={"draw_per_turn": 1},
    )

    assert state.combat is not None
    assert state.combat.player.statuses["vigor"] == 8
    starting_hp = state.combat.monsters[0].hp

    state = _play_card(state, "tap")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == starting_hp - 9
    assert "vigor" not in state.combat.player.statuses
    assert any(event.kind == "status_consumed" for event in state.combat.last_events)


def test_data_disk_and_brimstone_apply_combat_statuses() -> None:
    state = _enter_combat(
        (),
        relics=("data_disk", "brimstone"),
    )

    assert state.combat is not None
    assert state.combat.player.statuses["focus"] == 1
    assert state.combat.player.statuses["strength"] == 2
    assert state.combat.monsters[0].statuses["strength"] == 1


def test_attack_counter_relics_trigger_after_third_attack() -> None:
    state = _enter_combat(
        tuple(
            {
                "id": f"JAB_{index}",
                "name": f"Jab {index}",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 1,
            }
            for index in range(1, 4)
        ),
        relics=("shuriken", "kunai", "ornamental_fan"),
    )

    for card_id in ("jab_1", "jab_2", "jab_3"):
        state = _play_card(state, card_id)

    assert state.combat is not None
    assert state.combat.player.statuses["strength"] == 1
    assert state.combat.player.statuses["dexterity"] == 1
    assert state.combat.player.block == 4
    assert state.combat.metadata["attacks_played_this_turn"] == 3


def test_turn_relics_apply_energy_limits_and_end_turn_block() -> None:
    state = _enter_combat(
        tuple(
            {
                "id": f"ZERO_{index}",
                "name": f"Zero {index}",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
            }
            for index in range(1, 8)
        ),
        relics=("velvet_choker", "orichalcum", "happy_flower"),
        flags={"draw_per_turn": 7},
    )

    assert state.combat is not None
    assert state.combat.player.energy == 4
    for card_id in tuple(f"zero_{index}" for index in range(1, 7)):
        state = _play_card(state, card_id)

    assert not any(action.type == "play_card" for action in legal_actions(state))
    state = step(
        state,
        next(action for action in legal_actions(state) if action.type == "end_turn"),
    )

    assert state.combat is not None
    assert state.combat.player.hp == 80
    assert any(
        event.kind == "player_block"
        and event.source_id == "orichalcum"
        and event.metadata["hook"] == "turn_end"
        for event in state.combat.last_events
    )
    assert state.combat.metadata["relic_counters"]["happy_flower"] == 2


def test_combat_relics_modify_vulnerable_damage_math_and_end_heal() -> None:
    state = _enter_combat(
        (
            {
                "id": "BONK",
                "name": "Bonk",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 10,
            },
        ),
        player={"hp": 70, "max_hp": 80, "energy": 3, "max_energy": 3},
        relics=("paper_phrog", "burning_blood"),
    )
    assert state.combat is not None
    monster = state.combat.monsters[0]
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": (
                        monster.model_copy(
                            update={"hp": 17, "max_hp": 17, "statuses": {"vulnerable": 1}}
                        ),
                    )
                }
            )
        }
    )

    state = _play_card(state, "bonk")

    assert state.phase == RunPhase.REWARD
    assert state.player.hp == 76


def test_end_turn_discards_only_cards_without_retain() -> None:
    state = _enter_combat(
        (
            {
                "id": "SAFETY",
                "name": "Safety",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Retain.",
            },
            {
                "id": "PLAIN_BLOCK",
                "name": "Plain Block",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "block": 5,
            },
        ),
        flags={"draw_per_turn": 2},
    )
    assert state.combat is not None
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"draw_per_turn": 0})}
    )

    state = step(
        state,
        next(action for action in legal_actions(state) if action.type == "end_turn"),
    )

    assert state.combat is not None
    assert [card.card_id for card in state.combat.hand] == ["safety"]
    assert [card.card_id for card in state.combat.discard_pile] == ["plain_block"]
    assert any(event.kind == "cards_retained" for event in state.combat.last_events)
    assert any(
        event.kind == "card_discarded"
        and event.metadata["reason"] == "end_turn_discard"
        and event.metadata["card_id"] == "plain_block"
        for event in state.combat.last_events
    )


def test_temporary_retain_is_consumed_after_one_end_turn() -> None:
    state = _enter_combat(
        (
            {
                "id": "SET_ASIDE",
                "name": "Set Aside",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "custom": {"retain_once": True},
            },
        ),
        flags={"draw_per_turn": 1},
    )
    assert state.combat is not None
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"draw_per_turn": 0})}
    )

    state = step(
        state,
        next(action for action in legal_actions(state) if action.type == "end_turn"),
    )

    assert state.combat is not None
    assert [card.card_id for card in state.combat.hand] == ["set_aside"]
    assert "retain_once" not in state.combat.hand[0].custom

    state = step(
        state,
        next(action for action in legal_actions(state) if action.type == "end_turn"),
    )

    assert state.combat is not None
    assert state.combat.hand == ()
    assert [card.card_id for card in state.combat.discard_pile] == ["set_aside"]


def test_nightmare_copies_chosen_card_into_next_turn_hand() -> None:
    state = _enter_combat(
        (
            {
                "id": "NIGHTMARE",
                "name": "Nightmare",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": (
                    "Choose a card. Next turn, add 3 copies of that card into your Hand."
                ),
            },
            {
                "id": "TWIN",
                "name": "Twin",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
            },
            {
                "id": "FILLER",
                "name": "Filler",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
            },
        ),
        flags={"draw_per_turn": 3},
    )
    assert state.combat is not None
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"draw_per_turn": 0})}
    )

    state = _play_card(state, "nightmare")
    assert state.combat is not None
    assert state.combat.pending_choices[0].kind == "copy_to_hand_next_turn"

    state = _choose_pending_card(state, "twin")
    assert state.combat is not None
    assert state.combat.pending_choices == ()
    assert len(state.combat.metadata["timed_card_triggers"]) == 1

    state = _end_turn(state)

    assert state.combat is not None
    assert [card.card_id for card in state.combat.hand] == ["twin", "twin", "twin"]
    assert not state.combat.metadata.get("timed_card_triggers")
    assert any(
        event.kind == "timed_card_copies_added_to_hand" for event in state.combat.last_events
    )


def test_entropy_prompts_random_transform_at_start_of_turn() -> None:
    state = _enter_combat(
        (
            {
                "id": "ENTROPY",
                "name": "Entropy",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "At the start of your turn, Transform 1 card in your Hand.",
            },
            {
                "id": "KEEP",
                "name": "Keep",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Retain.",
            },
        ),
        flags={
            "draw_per_turn": 2,
            "card_pool": (
                {
                    "id": "NEW_CARD",
                    "name": "New Card",
                    "type": "Skill",
                    "target": "Self",
                    "cost": 0,
                    "color": "test",
                },
            ),
        },
    )
    assert state.combat is not None
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"draw_per_turn": 0})}
    )

    state = _play_card(state, "entropy")
    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.pending_choices[0].kind == "transform"
    assert state.combat.hand[0].card_id == "keep"

    state = _choose_pending_card(state, "keep")

    assert state.combat is not None
    assert state.combat.hand[0].card_id == "new_card"
    assert state.combat.pending_choices == ()
    assert any(event.kind == "card_transformed_by_choice" for event in state.combat.last_events)


def test_tyranny_draws_then_requires_exhaust_choice_at_start_of_turn() -> None:
    state = _enter_combat(
        (
            {
                "id": "TYRANNY",
                "name": "Tyranny",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": (
                    "At the start of your turn, draw 1 card and Exhaust 1 card from your Hand."
                ),
            },
            {
                "id": "KEEP",
                "name": "Keep",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Retain.",
            },
            {
                "id": "DRAWN",
                "name": "Drawn",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
            },
        ),
        flags={"draw_per_turn": 3},
    )
    assert state.combat is not None
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"draw_per_turn": 0})}
    )

    state = _play_card(state, "tyranny")
    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.pending_choices[0].kind == "exhaust"
    assert sorted(card.card_id for card in state.combat.hand) == ["drawn", "keep"]

    state = _exhaust_pending_card(state, "keep")

    assert state.combat is not None
    assert [card.card_id for card in state.combat.exhaust_pile] == ["keep"]
    assert [card.card_id for card in state.combat.hand] == ["drawn"]


def test_stratagem_prompts_for_draw_pile_card_after_shuffle() -> None:
    state = _enter_combat(
        (
            {
                "id": "STRATAGEM",
                "name": "Stratagem",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": (
                    "Whenever you shuffle your Draw Pile, choose a card from it to put into "
                    "your Hand."
                ),
            },
            {
                "id": "DRAW_TOOL",
                "name": "Draw Tool",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "draw": 1,
            },
            {
                "id": "CHOICE_A",
                "name": "Choice A",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
            },
            {
                "id": "CHOICE_B",
                "name": "Choice B",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
            },
        ),
        flags={"draw_per_turn": 4},
    )

    state = _play_card(state, "stratagem")
    assert state.combat is not None
    cards_by_id = {card.card_id: card for card in state.combat.hand}
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (cards_by_id["draw_tool"],),
                    "draw_pile": (),
                    "discard_pile": (cards_by_id["choice_a"], cards_by_id["choice_b"]),
                }
            )
        }
    )

    state = _play_card(state, "draw_tool")

    assert state.combat is not None
    assert state.combat.pending_choices[0].kind == "move_to_hand"
    assert any(event.kind == "timed_card_trigger_resolved" for event in state.combat.last_events)
    choice_id = state.combat.pending_choices[0].candidate_ids[0]
    chosen = next(card for card in state.combat.draw_pile if card.instance_id == choice_id)
    state = step(
        state,
        next(
            action
            for action in legal_actions(state)
            if action.type == "choose_card" and action.card_instance_id == choice_id
        ),
    )

    assert state.combat is not None
    assert chosen.card_id in [card.card_id for card in state.combat.hand]
    assert chosen.instance_id not in {card.instance_id for card in state.combat.draw_pile}


def test_turn_limited_attack_trigger_grants_block() -> None:
    state = _enter_combat(
        (
            {
                "id": "RAGE",
                "name": "Rage",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Whenever you play an Attack this turn, gain 3 Block.",
            },
            {
                "id": "JAB",
                "name": "Jab",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 1,
            },
        ),
        flags={"draw_per_turn": 2},
    )

    state = _play_card(state, "rage")
    state = _play_card(state, "jab")

    assert state.combat is not None
    assert state.combat.player.block == 3
    assert any(event.kind == "combat_trigger_resolved" for event in state.combat.last_events)


def test_every_five_cards_trigger_damages_all_enemies() -> None:
    state = _enter_combat(
        (
            {
                "id": "PANACHE",
                "name": "Panache",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": (
                    "Every time you play 5 cards in a single turn, deal 10 damage to "
                    "ALL enemies."
                ),
            },
            *(
                {
                    "id": f"ZERO_{index}",
                    "name": f"Zero {index}",
                    "type": "Skill",
                    "target": "Self",
                    "cost": 0,
                }
                for index in range(1, 6)
            ),
        ),
        flags={"draw_per_turn": 6},
    )
    assert state.combat is not None
    starting_hp = state.combat.monsters[0].hp

    state = _play_card(state, "panache")
    for index in range(1, 6):
        state = _play_card(state, f"zero_{index}")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == starting_hp - 10


def test_card_exhausted_trigger_grants_block() -> None:
    state = _enter_combat(
        (
            {
                "id": "FEEL_NO_PAIN",
                "name": "Feel No Pain",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "Whenever a card is Exhausted, gain 3 Block.",
            },
            {
                "id": "BURN_UP",
                "name": "Burn Up",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "keywords_key": ("Exhaust",),
            },
        ),
        flags={"draw_per_turn": 2},
    )

    state = _play_card(state, "feel_no_pain")
    state = _play_card(state, "burn_up")

    assert state.combat is not None
    assert state.combat.player.block == 3
    assert [card.card_id for card in state.combat.exhaust_pile] == ["burn_up"]


def test_turn_start_trigger_applies_poison_to_all_enemies() -> None:
    state = _enter_combat(
        (
            {
                "id": "NOXIOUS_FUMES",
                "name": "Noxious Fumes",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "At the start of your turn, apply 2 Poison to ALL enemies.",
            },
        ),
        flags={"draw_per_turn": 1},
    )
    assert state.combat is not None
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"draw_per_turn": 0})}
    )

    state = _play_card(state, "noxious_fumes")
    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.monsters[0].statuses["poison"] == 2


def test_turn_start_trigger_adds_random_card_from_pool() -> None:
    state = _enter_combat(
        (
            {
                "id": "CREATIVE_AI",
                "name": "Creative AI",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "At the start of your turn, add a random Power into your Hand.",
            },
        ),
        flags={
            "draw_per_turn": 1,
            "card_pool": (
                {
                    "id": "ONLY_POWER",
                    "name": "Only Power",
                    "type": "Power",
                    "target": "Self",
                    "cost": 0,
                    "color": "test",
                },
            ),
        },
    )
    assert state.combat is not None
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"draw_per_turn": 0})}
    )

    state = _play_card(state, "creative_ai")
    state = _end_turn(state)

    assert state.combat is not None
    assert [card.card_id for card in state.combat.hand] == ["only_power"]


def test_combat_end_trigger_grants_gold() -> None:
    state = _enter_combat(
        (
            {
                "id": "ROYALTIES",
                "name": "Royalties",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "At the end of combat, gain 30 Gold.",
            },
            {
                "id": "FINISH",
                "name": "Finish",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 999,
            },
        ),
        flags={"draw_per_turn": 2},
    )
    starting_gold = state.player.gold

    state = _play_card(state, "royalties")
    state = _play_card(state, "finish")

    assert state.phase == RunPhase.REWARD
    assert state.player.gold == starting_gold + 30


def test_forbidden_grimoire_offers_optional_deck_removal_after_combat() -> None:
    state = _enter_combat(
        (
            {
                "id": "FORBIDDEN_GRIMOIRE",
                "name": "Forbidden Grimoire",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "At the end of combat, you may remove a card from your Deck.",
                "keywords": ("Eternal",),
            },
            {
                "id": "STRIKE",
                "name": "Strike",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 6,
            },
            {
                "id": "FINISH",
                "name": "Finish",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 999,
            },
        ),
        flags={"draw_per_turn": 3},
    )

    state = _play_card(state, "forbidden_grimoire")
    state = _play_card(state, "finish")

    assert state.phase == RunPhase.REWARD
    assert state.reward is not None
    assert state.reward.metadata["optional_remove_card_count"] == 1
    assert "forbidden_grimoire" not in state.reward.metadata["optional_remove_card_ids"]
    assert any(action.type == "proceed" for action in legal_actions(state))

    remove_strike = next(
        action
        for action in legal_actions(state)
        if action.type == "take_reward_card"
        and action.target_id is not None
        and action.target_id.startswith("reward:remove_card:")
        and any(
            card.instance_id
            == state.reward.metadata["optional_remove_card_instance_ids"][
                int(action.target_id.rsplit(":", 1)[1])
            ]
            and card.card_id == "strike"
            for card in state.master_deck
        )
    )
    state = step(state, remove_strike)

    assert all(card.card_id != "strike" for card in state.master_deck)
    assert any(card.card_id == "forbidden_grimoire" for card in state.master_deck)
    assert state.reward is not None
    assert state.reward.metadata["optional_removed_card_instance_ids"]


def test_guilty_removes_itself_after_five_won_combats() -> None:
    state = _enter_combat(
        (
            {
                "id": "GUILTY",
                "name": "Guilty",
                "type": "Curse",
                "target": "None",
                "cost": -1,
                "description": "Removed from your Deck after 5 combats.",
                "keywords": ("Unplayable",),
            },
            {
                "id": "FINISH",
                "name": "Finish",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 999,
            },
        ),
        flags={"draw_per_turn": 2},
    )

    for combat_number in range(1, 6):
        state = _play_card(state, "finish")
        assert state.phase == RunPhase.REWARD
        if combat_number < 5:
            guilty = next(card for card in state.master_deck if card.card_id == "guilty")
            assert guilty.custom["guilty_combats_completed"] == combat_number
            assert guilty.custom["guilty_combats_remaining"] == 5 - combat_number
            state = step(
                state,
                next(action for action in legal_actions(state) if action.type == "proceed"),
            )
            state = _force_next_room(state, RoomKind.MONSTER)
            state = step(
                state,
                next(action for action in legal_actions(state) if action.type == "choose_node"),
            )

    assert all(card.card_id != "guilty" for card in state.master_deck)
    assert state.combat is not None
    assert any(
        event.kind == "guilty_removed_after_combats" for event in state.combat.last_events
    )


def test_ignition_channels_plasma_to_ally_state() -> None:
    state = _enter_combat(
        (
            {
                "id": "IGNITION",
                "name": "Ignition",
                "type": "Skill",
                "target": "AnyAlly",
                "cost": 0,
                "description": "Another player Channels Plasma.",
                "keywords": ("Exhaust",),
            },
            {
                "id": "FINISH",
                "name": "Finish",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 999,
            },
        ),
        flags={
            "allies": (
                {
                    "id": "ally_a",
                    "block": 7,
                    "orb_slots": 2,
                    "orbs": ("frost", "lightning"),
                },
            ),
            "draw_per_turn": 2,
        },
    )

    state = _play_card(state, "ignition")

    assert state.combat is not None
    ally = state.combat.metadata["allies"][0]
    assert ally["orbs"] == ("lightning", "plasma")
    assert any(event.kind == "ally_orb_evoked" for event in state.combat.last_events)
    assert any(
        event.kind == "ally_orb_channeled" and event.metadata["orb"] == "plasma"
        for event in state.combat.last_events
    )


def test_mimic_gains_block_from_ally_state() -> None:
    state = _enter_combat(
        (
            {
                "id": "MIMIC",
                "name": "Mimic",
                "type": "Skill",
                "target": "AnyAlly",
                "cost": 0,
                "description": "Gain Block equal to the Block on another player.",
                "keywords": ("Exhaust",),
            },
            {
                "id": "FINISH",
                "name": "Finish",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 999,
            },
        ),
        flags={"allies": ({"id": "ally_a", "block": 11},), "draw_per_turn": 2},
    )

    state = _play_card(state, "mimic")

    assert state.combat is not None
    assert state.combat.player.block == 11
    assert any(
        event.kind == "player_block"
        and event.amount == 11
        and event.metadata.get("formula") == "ally_block"
        for event in state.combat.last_events
    )


def test_barricade_status_keeps_block_between_turns() -> None:
    state = _enter_combat(
        (
            {
                "id": "BARRICADE",
                "name": "Barricade",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "Block is not removed at the start of your turn.",
            },
            {
                "id": "BIG_BLOCK",
                "name": "Big Block",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "block": 99,
            },
        ),
        flags={"draw_per_turn": 2},
    )
    assert state.combat is not None
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"draw_per_turn": 0})}
    )

    state = _play_card(state, "barricade")
    state = _play_card(state, "big_block")
    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.player.block > 0
    assert state.combat.player.statuses["retain_block"] == 1


def test_dynamic_damage_and_block_formulas_use_current_combat_state() -> None:
    state = _enter_combat(
        (
            {
                "id": "BODY_SLAM",
                "name": "Body Slam",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "description": "Deal damage equal to your Block.",
            },
            {
                "id": "STACK",
                "name": "Stack",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": (
                    "Gain Block equal to the number of cards in your Discard Pile."
                ),
            },
            {"id": "FILLER_A", "name": "Filler A", "type": "Skill", "target": "Self", "cost": 0},
            {"id": "FILLER_B", "name": "Filler B", "type": "Skill", "target": "Self", "cost": 0},
        ),
        flags={"draw_per_turn": 2},
    )
    assert state.combat is not None
    cards = {card.card_id: card for card in state.combat.hand + state.combat.draw_pile}
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "player": state.combat.player.model_copy(update={"block": 7}),
                    "hand": (cards["body_slam"], cards["stack"]),
                    "discard_pile": (cards["filler_a"], cards["filler_b"]),
                    "draw_pile": (),
                }
            )
        }
    )
    assert state.combat is not None
    monster_hp = state.combat.monsters[0].hp

    state = _play_card(state, "body_slam")
    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 7

    state = _play_card(state, "stack")
    assert state.combat is not None
    assert state.combat.player.block == 10


def test_dynamic_energy_draw_and_enemy_defense_removal_cards_execute() -> None:
    draw_cards = tuple(
        {
            "id": f"DRAW_{index}",
            "name": f"Draw {index}",
            "type": "Skill",
            "target": "Self",
            "cost": 0,
        }
        for index in range(8)
    )
    state = _enter_combat(
        (
            {
                "id": "DOUBLE_ENERGY",
                "name": "Double Energy",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Double your Energy.",
            },
            {
                "id": "SCRAWL",
                "name": "Scrawl",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Draw cards until your [gold]Hand[/gold] is full.",
            },
            *draw_cards,
        ),
        flags={"draw_per_turn": 2},
    )
    assert state.combat is not None
    cards = {
        card.card_id: card
        for card in state.combat.hand + state.combat.draw_pile + state.combat.discard_pile
    }
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (cards["double_energy"], cards["scrawl"]),
                    "draw_pile": tuple(cards[f"draw_{index}"] for index in range(8)),
                    "discard_pile": (),
                    "player": state.combat.player.model_copy(update={"energy": 2}),
                }
            )
        }
    )

    state = _play_card(state, "double_energy")
    assert state.combat is not None
    assert state.combat.player.energy == 4

    state = _play_card(state, "scrawl")
    assert state.combat is not None
    assert len(state.combat.hand) == 9
    assert "double_energy" in {card.card_id for card in state.combat.hand}
    assert state.combat.draw_pile == ()

    state = _enter_combat(
        (
            {
                "id": "EXPOSE",
                "name": "Expose",
                "type": "Skill",
                "target": "AnyEnemy",
                "cost": 0,
                "description": (
                    "Remove all [gold]Artifact[/gold] and [gold]Block[/gold] from the enemy.\n"
                    "Apply 2 [gold]Vulnerable[/gold]."
                ),
            },
        ),
        flags={"draw_per_turn": 1},
    )
    assert state.combat is not None
    monster = state.combat.monsters[0].model_copy(
        update={"block": 12, "statuses": {"artifact": 2}}
    )
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )

    state = _play_card(state, "expose")

    assert state.combat is not None
    assert state.combat.monsters[0].block == 0
    assert "artifact" not in state.combat.monsters[0].statuses
    assert state.combat.monsters[0].statuses["vulnerable"] == 2

    state = _enter_combat(
        (
            {
                "id": "MALAISE",
                "name": "Malaise",
                "type": "Skill",
                "target": "AnyEnemy",
                "cost": -1,
                "is_x_cost": True,
                "description": (
                    "Enemy loses X [gold]Strength[/gold]. Apply X [gold]Weak[/gold]."
                ),
            },
            {
                "id": "TIMES_UP",
                "name": "Time's Up",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "description": "Deal damage equal to the enemy's [gold]Doom[/gold].",
            },
        ),
        flags={"draw_per_turn": 2},
    )
    assert state.combat is not None
    monster_hp = state.combat.monsters[0].hp
    monster = state.combat.monsters[0].model_copy(update={"statuses": {"doom": 7}})
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": (monster,),
                    "player": state.combat.player.model_copy(update={"energy": 3}),
                }
            )
        }
    )

    state = _play_card(state, "malaise")

    assert state.combat is not None
    assert state.combat.player.energy == 0
    assert state.combat.monsters[0].statuses["strength"] == -3
    assert state.combat.monsters[0].statuses["weak"] == 3

    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={"player": state.combat.player.model_copy(update={"energy": 3})}
            )
        }
    )
    state = _play_card(state, "times_up")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 7


def test_apotheosis_upgrades_combat_zones_and_enlightenment_costs_reset() -> None:
    state = _enter_combat(
        (
            {
                "id": "APOTHEOSIS",
                "name": "Apotheosis",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "[gold]Upgrade[/gold] ALL your cards.",
            },
            {
                "id": "HEAVY_SKILL",
                "name": "Heavy Skill",
                "type": "Skill",
                "target": "Self",
                "cost": 3,
                "block": 5,
                "upgrade": {"block": 3},
            },
            {
                "id": "DRAW_ATTACK",
                "name": "Draw Attack",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 1,
                "damage": 4,
                "upgrade": {"damage": 2},
            },
        ),
        flags={"draw_per_turn": 2},
    )
    assert state.combat is not None
    cards = {card.card_id: card for card in state.combat.hand + state.combat.draw_pile}
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (cards["apotheosis"], cards["heavy_skill"]),
                    "draw_pile": (cards["draw_attack"],),
                }
            )
        }
    )

    state = _play_card(state, "apotheosis")

    assert state.combat is not None
    heavy = next(card for card in state.combat.hand if card.card_id == "heavy_skill")
    draw_attack = next(card for card in state.combat.draw_pile if card.card_id == "draw_attack")
    assert heavy.upgraded is True
    assert heavy.effects == {"sequence": [{"block": 8}]}
    assert draw_attack.upgraded is True
    assert draw_attack.effects == {"sequence": [{"damage": 6}]}

    state = _enter_combat(
        (
            {
                "id": "ENLIGHTENMENT",
                "name": "Enlightenment",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": (
                    "Reduce the cost of ALL cards in your [gold]Hand[/gold] to 1 this turn."
                ),
            },
            {
                "id": "EXPENSIVE_CARD",
                "name": "Expensive Card",
                "type": "Skill",
                "target": "Self",
                "cost": 3,
                "block": 1,
            },
        ),
        flags={"draw_per_turn": 2},
    )

    state = _play_card(state, "enlightenment")
    assert state.combat is not None
    expensive = next(card for card in state.combat.hand if card.card_id == "expensive_card")
    assert expensive.cost == 1

    state = _end_turn(state)

    assert state.combat is not None
    all_cards = state.combat.hand + state.combat.draw_pile + state.combat.discard_pile
    expensive = next(card for card in all_cards if card.card_id == "expensive_card")
    assert expensive.cost == 3


def test_alchemize_generates_random_potion_into_open_slot() -> None:
    state = _enter_combat(
        (
            {
                "id": "ALCHEMIZE",
                "name": "Alchemize",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Procure a random potion.",
            },
        ),
        flags={"draw_per_turn": 1, "potion_pool": ("fire_potion",)},
    )
    state = state.model_copy(update={"potions": ()})

    state = _play_card(state, "alchemize")

    assert state.potions == ("fire_potion",)
    assert state.combat is not None
    assert any(event.kind == "potion_generated" for event in state.combat.last_events)


def test_enthralled_must_be_played_before_other_cards() -> None:
    state = _enter_combat(
        (
            {
                "id": "ENTHRALLED",
                "name": "Enthralled",
                "type": "Curse",
                "target": "Self",
                "cost": 0,
                "description": (
                    "If this is in your [gold]Hand[/gold], it must be played before other cards."
                ),
            },
            {
                "id": "STRIKE",
                "name": "Strike",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 3,
            },
        ),
        flags={"draw_per_turn": 2},
    )
    assert state.combat is not None
    ids_by_instance = {card.instance_id: card.card_id for card in state.combat.hand}

    playable_ids = {
        ids_by_instance[action.card_instance_id]
        for action in legal_actions(state)
        if action.type == "play_card"
    }

    assert playable_ids == {"enthralled"}


def test_bullet_time_blocks_extra_draws_and_makes_current_hand_free() -> None:
    state = _enter_combat(
        (
            {
                "id": "BULLET_TIME",
                "name": "Bullet Time",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": (
                    "You cannot draw additional cards this turn. "
                    "All cards in your [gold]Hand[/gold] are free to play this turn."
                ),
            },
            {
                "id": "DRAW_TOOL",
                "name": "Draw Tool",
                "type": "Skill",
                "target": "Self",
                "cost": 2,
                "draw": 1,
            },
            {
                "id": "WAITING_CARD",
                "name": "Waiting Card",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
            },
        ),
        player={"hp": 80, "max_hp": 80, "energy": 0, "max_energy": 0},
        flags={"draw_per_turn": 2},
    )

    assert state.combat is not None
    cards = {card.card_id: card for card in state.combat.hand + state.combat.draw_pile}
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (cards["bullet_time"], cards["draw_tool"]),
                    "draw_pile": (cards["waiting_card"],),
                    "discard_pile": (),
                }
            )
        }
    )

    state = _play_card(state, "bullet_time")
    assert state.combat is not None
    draw_tool = next(card for card in state.combat.hand if card.card_id == "draw_tool")
    assert draw_tool.custom["free_to_play_this_turn"] is True

    state = _play_card(state, "draw_tool")
    assert state.combat is not None
    assert [card.card_id for card in state.combat.draw_pile] == ["waiting_card"]
    assert any(event.kind == "draw_blocked" for event in state.combat.last_events)


def test_corruption_makes_skills_free_and_exhaust_on_play() -> None:
    state = _enter_combat(
        (
            {
                "id": "CORRUPTION",
                "name": "Corruption",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": (
                    "Skills cost 0 [energy:1]. Whenever you play a Skill, "
                    "[gold]Exhaust[/gold] it."
                ),
            },
            {
                "id": "EXPENSIVE_SKILL",
                "name": "Expensive Skill",
                "type": "Skill",
                "target": "Self",
                "cost": 2,
                "block": 5,
            },
        ),
        player={"hp": 80, "max_hp": 80, "energy": 0, "max_energy": 0},
        flags={"draw_per_turn": 2},
    )

    state = _play_card(state, "corruption")
    state = _play_card(state, "expensive_skill")

    assert state.combat is not None
    assert state.combat.player.block == 5
    assert [card.card_id for card in state.combat.exhaust_pile] == ["expensive_skill"]


def test_next_matching_card_extra_play_repeats_card_effect_once() -> None:
    state = _enter_combat(
        (
            {
                "id": "BURST_TEST",
                "name": "Burst Test",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "This turn, your next Skill is played an extra time.",
            },
            {
                "id": "FOCUSED_BLOCK",
                "name": "Focused Block",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "block": 4,
            },
        ),
        flags={"draw_per_turn": 2},
    )

    state = _play_card(state, "burst_test")
    state = _play_card(state, "focused_block")

    assert state.combat is not None
    assert state.combat.player.block == 8
    assert any(event.kind == "card_extra_played" for event in state.combat.last_events)


def test_next_card_extra_play_persists_across_turns_without_this_turn_text() -> None:
    state = _enter_combat(
        (
            {
                "id": "SETUP_NEXT",
                "name": "Setup Next",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "Your next Skill is played an extra time.",
            },
            {
                "id": "RETAINED_BLOCK",
                "name": "Retained Block",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "block": 4,
                "custom": {"retain": True},
            },
        ),
        flags={"draw_per_turn": 2},
    )
    assert state.combat is not None
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"draw_per_turn": 0})}
    )

    state = _play_card(state, "setup_next")
    state = _end_turn(state)
    state = _play_card(state, "retained_block")

    assert state.combat is not None
    assert state.combat.player.block == 8
    assert any(event.kind == "card_extra_played" for event in state.combat.last_events)


def test_next_card_extra_play_expires_at_turn_end_when_text_says_this_turn() -> None:
    state = _enter_combat(
        (
            {
                "id": "SETUP_NEXT",
                "name": "Setup Next",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "This turn, your next Skill is played an extra time.",
            },
            {
                "id": "RETAINED_BLOCK",
                "name": "Retained Block",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "block": 4,
                "custom": {"retain": True},
            },
        ),
        flags={"draw_per_turn": 2},
    )
    assert state.combat is not None
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"draw_per_turn": 0})}
    )

    state = _play_card(state, "setup_next")
    state = _end_turn(state)
    state = _play_card(state, "retained_block")

    assert state.combat is not None
    assert state.combat.player.block == 4
    assert not any(event.kind == "card_extra_played" for event in state.combat.last_events)


def test_next_card_extra_play_matches_generic_card_text() -> None:
    state = _enter_combat(
        (
            {
                "id": "DUPLICATE_NEXT",
                "name": "Duplicate Next",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "description": "Your next card is played an extra time.",
            },
            {
                "id": "JAB",
                "name": "Jab",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 3,
            },
        ),
        flags={"draw_per_turn": 2},
    )
    assert state.combat is not None
    monster_hp = state.combat.monsters[0].hp

    state = _play_card(state, "duplicate_next")
    state = _play_card(state, "jab")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 6
    assert any(event.kind == "card_extra_played" for event in state.combat.last_events)


def test_tools_of_the_trade_draws_then_requires_discard_choice() -> None:
    state = _enter_combat(
        (
            {
                "id": "TOOLS_OF_THE_TRADE",
                "name": "Tools of the Trade",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "At the start of your turn, draw 1 card and discard 1 card.",
            },
            {
                "id": "DRAWN_CARD",
                "name": "Drawn Card",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
            },
        ),
        flags={"draw_per_turn": 1},
    )
    assert state.combat is not None
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"draw_per_turn": 0})}
    )

    state = _play_card(state, "tools_of_the_trade")
    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.pending_choices[0].kind == "discard"
    assert [card.card_id for card in state.combat.hand] == ["drawn_card"]
    discard_action = next(
        action for action in legal_actions(state) if action.type == "discard_card"
    )
    state = step(state, discard_action)

    assert state.combat is not None
    assert state.combat.pending_choices == ()
    assert [card.card_id for card in state.combat.discard_pile][-1] == "drawn_card"


def test_mayhem_style_trigger_plays_top_draw_card_at_turn_start() -> None:
    state = _enter_combat(
        (
            {
                "id": "MAYHEM",
                "name": "Mayhem",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": "At the start of your turn, play the top card of your Draw Pile.",
            },
            {
                "id": "TOP_STRIKE",
                "name": "Top Strike",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 1,
                "damage": 6,
            },
        ),
        flags={"draw_per_turn": 1},
    )
    assert state.combat is not None
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"draw_per_turn": 0})}
    )
    monster_hp = state.combat.monsters[0].hp

    state = _play_card(state, "mayhem")
    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 6
    assert [card.card_id for card in state.combat.discard_pile][-1] == "top_strike"
    assert any(event.kind == "card_played_by_trigger" for event in state.combat.last_events)


def test_first_attack_or_skill_can_return_to_top_of_draw_pile_once_per_turn() -> None:
    state = _enter_combat(
        (
            {
                "id": "NOSTALGIA",
                "name": "Nostalgia",
                "type": "Power",
                "target": "Self",
                "cost": 0,
                "description": (
                    "The first Attack or Skill you play each turn is placed on top "
                    "of your Draw Pile."
                ),
            },
            {
                "id": "POKE",
                "name": "Poke",
                "type": "Attack",
                "target": "AnyEnemy",
                "cost": 0,
                "damage": 1,
            },
            {
                "id": "GUARD",
                "name": "Guard",
                "type": "Skill",
                "target": "Self",
                "cost": 0,
                "block": 3,
            },
        ),
        flags={"draw_per_turn": 3},
    )

    state = _play_card(state, "nostalgia")
    state = _play_card(state, "poke")

    assert state.combat is not None
    assert state.combat.draw_pile[0].card_id == "poke"
    assert "poke" not in [card.card_id for card in state.combat.discard_pile]

    state = _play_card(state, "guard")

    assert state.combat is not None
    assert [card.card_id for card in state.combat.discard_pile][-1] == "guard"
