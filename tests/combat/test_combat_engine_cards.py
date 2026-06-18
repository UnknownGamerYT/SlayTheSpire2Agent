from __future__ import annotations

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import MapEdgeState, MapNodeState, MapState, RoomKind, RunPhase


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
    assert state.combat.monsters[0].hp == monster_hp - 20
    assert any(event.kind == "osty_hp_loss_damaged_enemies" for event in state.combat.last_events)


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
