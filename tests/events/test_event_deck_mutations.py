from __future__ import annotations

from random import Random

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import MapEdgeState, MapNodeState, MapState, RoomKind, RunPhase
from sts2sim.engine.transitions import _apply_event_flow_markers
from sts2sim.mechanics.event_flows import EventFlowMarker, EventFlowMarkerKind

EVENT_TEST_CARDS = (
    {
        "id": "STRIKE",
        "name": "Strike",
        "rarity": "Common",
        "color": "test",
        "type": "Attack",
        "target": "AnyEnemy",
        "damage": 6,
    },
    {
        "id": "DEFEND",
        "name": "Defend",
        "rarity": "Common",
        "color": "test",
        "type": "Skill",
        "target": "Self",
        "block": 5,
    },
    {
        "id": "POMMEL_STRIKE",
        "name": "Pommel Strike",
        "rarity": "Common",
        "color": "test",
        "type": "Attack",
        "target": "AnyEnemy",
        "damage": 9,
        "draw": 1,
    },
    {
        "id": "TRUE_GRIT",
        "name": "True Grit",
        "rarity": "Uncommon",
        "color": "test",
        "type": "Skill",
        "target": "Self",
        "block": 7,
    },
    {
        "id": "RARE_POWER",
        "name": "Rare Power",
        "rarity": "Rare",
        "color": "test",
        "type": "Power",
        "target": "Self",
        "effects": {"apply_status": {"target": "self", "strength": 1}},
    },
    {
        "id": "APOTHEOSIS",
        "name": "Apotheosis",
        "rarity": "Rare",
        "color": "colorless",
        "type": "Skill",
        "target": "Self",
        "effects": {"upgrade_all": True},
    },
    {
        "id": "BLIND",
        "name": "Blind",
        "rarity": "Uncommon",
        "color": "colorless",
        "type": "Skill",
        "target": "AllEnemies",
        "effects": {"apply_status": {"target": "enemy", "weak": 2}},
    },
)


def _choose_first_ancient(state):
    return state.model_copy(update={"phase": RunPhase.MAP, "ancient": None})


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


def _choose_first_node(state):
    action = next(action for action in legal_actions(state) if action.type == "choose_node")
    return step(state, action)


def _choose_event_action(state, option_id: str, card_instance_id: str | None = None):
    return next(
        action
        for action in legal_actions(state)
        if action.type == "choose_event"
        and action.target_id == option_id
        and (card_instance_id is None or action.card_instance_id == card_instance_id)
    )


def _proceed(state):
    action = next(action for action in legal_actions(state) if action.type == "proceed")
    return step(state, action)


def test_self_help_book_enchants_selected_attack_and_persists_into_combat() -> None:
    state = new_run(
        seed=1200,
        character_id="TEST",
        ascension=0,
        source_data={"event_id": "SELF_HELP_BOOK"},
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)

    strike = next(card for card in state.master_deck if card.card_id == "strike")
    state = step(state, _choose_event_action(state, "READ_THE_BACK", strike.instance_id))

    enchanted = next(card for card in state.master_deck if card.instance_id == strike.instance_id)
    assert state.event is not None
    assert state.event.resolved_option_id == "READ_THE_BACK"
    assert enchanted.effects["damage"] == 8
    assert enchanted.enchantments[0].keyword == "Sharp"
    assert enchanted.enchantments[0].amount == 2
    assert "enchant:sharp" in enchanted.tags

    state = _proceed(state)
    state = _force_next_room(state, RoomKind.MONSTER)
    state = _choose_first_node(state)

    assert state.combat is not None
    combat_cards = state.combat.hand + state.combat.draw_pile + state.combat.discard_pile
    carried_card = next(card for card in combat_cards if card.instance_id == strike.instance_id)
    assert carried_card.effects["damage"] == 8
    assert carried_card.enchantments[0].keyword == "Sharp"


def test_generic_event_enchant_parsed_from_codex_text_mutates_target_card() -> None:
    state = new_run(
        seed=1201,
        character_id="TEST",
        ascension=0,
        source_data={"event_id": "STONE_OF_ALL_TIME", "player": {"hp": 50, "max_hp": 80}},
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)

    strike = next(card for card in state.master_deck if card.card_id == "strike")
    previous_hp = state.player.hp
    state = step(state, _choose_event_action(state, "push", strike.instance_id))

    enchanted = next(card for card in state.master_deck if card.instance_id == strike.instance_id)
    assert state.player.hp == previous_hp - 6
    assert enchanted.effects["damage"] == 14
    assert enchanted.enchantments[0].keyword == "Vigorous"
    assert enchanted.enchantments[0].amount == 8


def test_tinker_time_creates_real_custom_card_in_master_deck() -> None:
    state = new_run(
        seed=1202,
        character_id="TEST",
        ascension=0,
        source_data={"event_id": "TINKER_TIME"},
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)

    state = step(state, _choose_event_action(state, "CHOOSE_CARD_TYPE"))
    state = step(state, _choose_event_action(state, "SKILL"))
    rider_action_ids = {
        action.target_id for action in legal_actions(state) if action.type == "choose_event"
    }
    assert rider_action_ids == {
        "ENERGIZED",
        "WISDOM",
        "CHAOS",
    }

    state = step(state, _choose_event_action(state, "WISDOM"))

    created = state.master_deck[-1]
    assert state.event is not None
    assert state.event.resolved_option_id == "WISDOM"
    assert created.card_id == "mad_science_skill_wisdom"
    assert created.name == "Mad Science (Wisdom)"
    assert created.type.value == "skill"
    assert created.effects == {"sequence": [{"block": 8}, {"draw": 3}]}
    assert created.custom == {
        "source_event_id": "TINKER_TIME",
        "base_card_id": "MAD_SCIENCE",
        "card_type": "skill",
        "rider_id": "wisdom",
        "rider_effect": "draw_cards",
    }


def test_colossal_flower_fixed_relic_marker_adds_relic_to_run() -> None:
    state = new_run(
        seed=1203,
        character_id="TEST",
        ascension=0,
        source_data={"event_id": "COLOSSAL_FLOWER", "player": {"hp": 40, "max_hp": 80}},
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)

    state = step(state, _choose_event_action(state, "REACH_DEEPER_1"))
    state = step(state, _choose_event_action(state, "REACH_DEEPER_2"))
    state = step(state, _choose_event_action(state, "POLLINOUS_CORE"))

    assert "pollinous_core" in state.relics
    assert state.event is not None
    assert state.event.resolved_option_id == "POLLINOUS_CORE"
    assert any(
        event.kind == "event_relic_obtained" and event.target_id == "pollinous_core"
        for event in state.replay_log[-1].events
    )


def test_tablet_of_truth_upgrade_all_marker_upgrades_master_deck() -> None:
    state = new_run(
        seed=1204,
        character_id="TEST",
        ascension=0,
        source_data={"event_id": "TABLET_OF_TRUTH", "player": {"hp": 50, "max_hp": 80}},
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)

    assert any(not card.upgraded for card in state.master_deck)

    state = step(state, _choose_event_action(state, "DECIPHER_1"))
    for _ in range(4):
        state = step(state, _choose_event_action(state, "DECIPHER"))

    assert state.event is not None
    assert state.event.resolved_option_id == "DECIPHER"
    assert all(card.upgraded for card in state.master_deck)
    assert any(event.kind == "card_upgraded" for event in state.replay_log[-1].events)


def test_wongos_random_relic_marker_adds_seeded_relic() -> None:
    state = new_run(
        seed=1205,
        character_id="TEST",
        ascension=0,
        source_data={
            "event_id": "WELCOME_TO_WONGOS",
            "player": {"hp": 50, "max_hp": 80, "gold": 150},
        },
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)
    old_relics = set(state.relics)

    state = step(state, _choose_event_action(state, "BARGAIN_BIN"))

    assert state.player.gold == 50
    assert len(set(state.relics) - old_relics) == 1
    assert any(event.kind == "event_relic_obtained" for event in state.replay_log[-1].events)


def test_endless_conveyor_transform_marker_replaces_selected_card() -> None:
    state = new_run(
        seed=1206,
        character_id="TEST",
        ascension=0,
        source_data={
            "event_id": "ENDLESS_CONVEYOR",
            "event_flow_page_id": "ALL",
            "player": {"hp": 50, "max_hp": 80, "gold": 80},
            "cards": (
                {
                    "id": "STRIKE",
                    "name": "Strike",
                    "rarity": "Common",
                    "color": "test",
                    "type": "Attack",
                    "target": "AnyEnemy",
                    "damage": 6,
                },
                {
                    "id": "DEFEND",
                    "name": "Defend",
                    "rarity": "Common",
                    "color": "test",
                    "type": "Skill",
                    "target": "Self",
                    "block": 5,
                },
                {
                    "id": "POMMEL_STRIKE",
                    "name": "Pommel Strike",
                    "rarity": "Common",
                    "color": "test",
                    "type": "Attack",
                    "target": "AnyEnemy",
                    "damage": 9,
                    "draw": 1,
                },
                {
                    "id": "TRUE_GRIT",
                    "name": "True Grit",
                    "rarity": "Uncommon",
                    "color": "test",
                    "type": "Skill",
                    "target": "Self",
                    "block": 7,
                },
            ),
        },
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)
    strike = next(card for card in state.master_deck if card.card_id == "strike")
    transform_actions = [
        action
        for action in legal_actions(state)
        if action.type == "choose_event" and action.target_id == "JELLY_LIVER"
    ]

    assert transform_actions
    assert not any(
        action.card_instance_id is None and not action.payload for action in transform_actions
    )

    state = step(state, _choose_event_action(state, "JELLY_LIVER", strike.instance_id))

    transformed = next(card for card in state.master_deck if card.instance_id == strike.instance_id)
    assert state.player.gold == 40
    assert transformed.card_id in {"defend", "pommel_strike", "true_grit"}
    assert transformed.card_id != "strike"
    assert transformed.custom["transformed_from_card_id"] == "strike"
    assert any(event.kind == "event_card_transformed" for event in state.replay_log[-1].events)


def test_random_transform_marker_resolves_without_selected_card() -> None:
    state = new_run(
        seed=1207,
        character_id="TEST",
        ascension=0,
        source_data={
            "cards": (
                {
                    "id": "STRIKE",
                    "name": "Strike",
                    "rarity": "Common",
                    "color": "test",
                    "type": "Attack",
                    "target": "AnyEnemy",
                    "damage": 6,
                },
                {
                    "id": "POMMEL_STRIKE",
                    "name": "Pommel Strike",
                    "rarity": "Common",
                    "color": "test",
                    "type": "Attack",
                    "target": "AnyEnemy",
                    "damage": 9,
                },
            ),
            "deck": (
                {
                    "id": "STRIKE",
                    "name": "Strike",
                    "type": "Attack",
                    "target": "AnyEnemy",
                    "damage": 6,
                },
            ),
        },
    )
    marker = EventFlowMarker(
        kind=EventFlowMarkerKind.CARD_TRANSFORM,
        description="Transform a random card.",
    )

    next_state, events = _apply_event_flow_markers(state, (marker,), Random(7))

    assert next_state.master_deck[0].card_id == "pommel_strike"
    assert events[0].kind == "event_card_transformed"


def test_endless_conveyor_fried_eel_adds_random_colorless_card() -> None:
    state = new_run(
        seed=1208,
        character_id="TEST",
        ascension=0,
        source_data={
            "event_id": "ENDLESS_CONVEYOR",
            "event_flow_page_id": "ALL",
            "player": {"hp": 50, "max_hp": 80, "gold": 80},
            "cards": EVENT_TEST_CARDS,
        },
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)
    deck_count = len(state.master_deck)

    state = step(state, _choose_event_action(state, "FRIED_EEL"))

    assert state.player.gold == 40
    assert len(state.master_deck) == deck_count + 1
    assert state.master_deck[-1].card_id.lower() in {"apotheosis", "blind"}
    assert any(
        event.kind == "event_random_card_added" for event in state.replay_log[-1].events
    )


def test_trial_nondescript_guilty_creates_two_pickable_card_reward_groups() -> None:
    state = new_run(
        seed=1209,
        character_id="TEST",
        ascension=0,
        source_data={
            "event_id": "TRIAL",
            "event_flow_data": {"trial_case": "NONDESCRIPT"},
            "player": {"hp": 50, "max_hp": 80},
            "cards": EVENT_TEST_CARDS,
        },
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)

    state = step(state, _choose_event_action(state, "ACCEPT"))
    state = step(state, _choose_event_action(state, "GUILTY"))

    assert state.reward is not None
    assert len(state.reward.card_options) == 3
    assert len(state.reward.card_option_groups) == 1
    assert len(state.reward.card_option_groups[0]) == 3
    assert any(card.card_id == "doubt" for card in state.master_deck)
    assert {
        action.target_id for action in legal_actions(state) if action.type == "take_reward_card"
    } >= {"reward:card:0", "reward:card_group:0:0"}

    first_group_card = state.reward.card_options[0]
    second_group_card = state.reward.card_option_groups[0][0]
    state = step(
        state,
        next(
            action
            for action in legal_actions(state)
            if action.type == "take_reward_card" and action.target_id == "reward:card:0"
        ),
    )
    state = step(
        state,
        next(
            action
            for action in legal_actions(state)
            if action.type == "take_reward_card"
            and action.target_id == "reward:card_group:0:0"
        ),
    )

    assert state.reward is not None
    assert state.reward.card_claimed is True
    assert state.reward.claimed_card_option_group_indices == (0,)
    assert state.master_deck[-2].card_id.lower() == first_group_card
    assert state.master_deck[-1].card_id.lower() == second_group_card
