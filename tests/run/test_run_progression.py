from __future__ import annotations

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    AncientOptionState,
    AncientState,
    MapEdgeState,
    MapNodeState,
    MapState,
    RewardState,
    RoomKind,
    RunPhase,
)
from sts2sim.mechanics.mapgen import map_layout_from_state, validate_act_map_parity


def _choose_first_ancient(state):
    target_id = None
    if state.ancient is not None:
        available = [
            option
            for option in state.ancient.options
            if option.option_id not in state.ancient.chosen_option_ids
        ]
        preferred = next(
            (
                option
                for option in available
                if not option.metadata.get("choice", {}).get("card_reward_count")
            ),
            available[0] if available else None,
        )
        target_id = preferred.option_id if preferred is not None else None
    action = next(
        action
        for action in legal_actions(state)
        if action.type == "choose_ancient"
        and (target_id is None or action.target_id == target_id)
    )
    return _skip_ancient_reward(step(state, action))


def _skip_ancient_reward(state):
    if (
        state.phase == RunPhase.REWARD
        and state.reward is not None
        and state.reward.source == "ancient"
    ):
        while (
            state.reward is not None
            and any(action.type == "take_reward_card" for action in legal_actions(state))
        ):
            action = next(
                action for action in legal_actions(state) if action.type == "take_reward_card"
            )
            state = step(state, action)
        return _proceed(state)
    return state


def _choose_ancient_option(state, option_index: int):
    assert state.ancient is not None
    option = state.ancient.options[option_index]
    action = next(
        action
        for action in legal_actions(state)
        if action.type == "choose_ancient" and action.target_id == option.option_id
    )
    return step(state, action)


def _choose_first_node(state):
    action = next(action for action in legal_actions(state) if action.type == "choose_node")
    return step(state, action)


def _play_first_card(state):
    action = next(action for action in legal_actions(state) if action.type == "play_card")
    return step(state, action)


def _play_card_by_id(state, card_id: str):
    assert state.combat is not None
    card_ids = {card.instance_id: card.card_id for card in state.combat.hand}
    action = next(
        action
        for action in legal_actions(state)
        if action.type == "play_card" and card_ids.get(action.card_instance_id) == card_id
    )
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
    return state.model_copy(
        update={"phase": RunPhase.MAP, "map": game_map, "floor": 0, "reward": None}
    )


def _proceed(state):
    action = next(action for action in legal_actions(state) if action.type == "proceed")
    return step(state, action)


def _action(state, action_type: str, target_id: str | None = None):
    return next(
        action
        for action in legal_actions(state)
        if action.type == action_type
        and (target_id is None or action.target_id == target_id)
    )


def _path_can_reach_boss_without_target(map_state: MapState, target_id: str) -> bool:
    assert map_state.current_node_id is not None
    assert map_state.boss_node_id is not None
    outgoing = map_state.outgoing_by_id
    seen: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id == target_id:
            return False
        if node_id == map_state.boss_node_id:
            return True
        if node_id in seen:
            return False
        seen.add(node_id)
        return any(visit(next_id) for next_id in outgoing.get(node_id, ()))

    return visit(map_state.current_node_id)


def _move_to_predecessor_of(state, target_id: str):
    assert state.map is not None
    edge = next(edge for edge in state.map.edges if edge.to_id == target_id)
    predecessor = state.map.node_by_id[edge.from_id]
    game_map = state.map.model_copy(
        update={
            "current_node_id": predecessor.node_id,
            "completed_node_ids": tuple(
                dict.fromkeys(state.map.completed_node_ids + (predecessor.node_id,))
            ),
        }
    )
    return state.model_copy(
        update={"phase": RunPhase.MAP, "map": game_map, "floor": predecessor.floor}
    )


def test_run_progresses_through_map_boss_and_next_act() -> None:
    state = new_run(
        seed=7,
        character_id="TEST",
        ascension=0,
        source_data={
            "max_acts": 2,
            "map_floors": 4,
            "map_width": 1,
            "deck": [
                {
                    "card_id": "debug_kill",
                    "name": "Debug Kill",
                    "type": "attack",
                    "cost": 0,
                    "target": "enemy",
                    "effects": {"damage": 999},
                }
            ],
            "player": {"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3},
        },
    )

    assert state.phase.value == "ancient"
    assert len([action for action in legal_actions(state) if action.type == "choose_ancient"]) == 3
    state = _choose_first_ancient(state)
    assert state.phase.value == "map"
    assert state.act == 1

    state = _choose_first_node(state)
    assert state.phase.value == "combat"
    state = _play_first_card(state)
    assert state.phase.value == "reward"
    state = _proceed(state)
    assert state.phase.value == "map"

    state = _choose_first_node(state)
    assert state.phase.value == "rest"
    state = _proceed(state)
    assert state.phase.value == "map"

    state = _choose_first_node(state)
    assert state.phase.value == "combat"
    state = _play_first_card(state)
    assert state.phase.value == "reward"
    state = _proceed(state)
    assert state.phase.value == "ancient"
    assert state.act == 2
    state = _choose_first_ancient(state)
    assert state.phase.value == "map"
    assert state.act == 2

    state = _choose_first_node(state)
    state = _play_first_card(state)
    state = _proceed(state)
    state = _choose_first_node(state)
    state = _proceed(state)
    state = _choose_first_node(state)
    state = _play_first_card(state)
    state = _proceed(state)

    assert state.phase.value == "complete"
    assert state.act == 2
    assert len(state.room_history) == 6


def test_ancient_offerings_heal_and_grant_selected_relics() -> None:
    state = new_run(
        seed=42,
        character_id="TEST",
        ascension=0,
        source_data={"player": {"hp": 20, "max_hp": 80}},
    )

    assert state.phase.value == "ancient"
    assert state.player.hp == 80
    assert state.ancient is not None
    assert len(state.ancient.options) == 3
    assert [option.kind for option in state.ancient.options].count("positive_relic") == 2
    assert [option.kind for option in state.ancient.options].count("curse_relic") == 1

    chosen = state.ancient.options[0]
    state = _choose_ancient_option(state, 0)
    assert state.phase.value == "map"
    assert chosen.relic_id in state.relics
    assert state.ancient is not None
    assert state.ancient.chosen_option_ids == (chosen.option_id,)


def test_ancient_choice_payload_applies_direct_effects() -> None:
    state = new_run(seed=43, character_id="TEST", ascension=0)
    option = AncientOptionState(
        option_id="test_blessing",
        name="Test Blessing",
        kind="positive_relic",
        relic_id="golden_pearl",
        description="Gain Golden Pearl and more.",
        metadata={
            "choice": {
                "option_id": "test_blessing",
                "name": "Test Blessing",
                "kind": "positive_relic",
                "relic_id": "golden_pearl",
                "gold_delta": 20,
                "max_hp_delta": 5,
                "fixed_card_ids": ("test_card",),
                "fixed_potion_ids": ("fire_potion",),
            }
        },
    )
    state = state.model_copy(
        update={
            "ancient": AncientState(act=1, ancient_id="neow", options=(option,)),
            "player": state.player.model_copy(update={"hp": 40, "max_hp": 50, "gold": 5}),
        }
    )

    state = step(state, _action(state, "choose_ancient", "test_blessing"))

    assert state.phase.value == "map"
    assert state.player.gold == 25
    assert state.player.max_hp == 55
    assert state.player.hp == 40
    assert "golden_pearl" in state.relics
    assert state.master_deck[-1].card_id == "test_card"
    assert state.potions == ("fire_potion",)
    assert any(event.kind == "ancient_card_added" for event in state.replay_log[-1].events)


def test_ancient_card_reward_opens_reward_screen_and_uses_reward_triggers() -> None:
    state = new_run(
        seed=44,
        character_id="TEST",
        ascension=0,
        source_data={
            "cards": [
                {"id": "COMMON_A", "name": "Common A", "rarity": "Common", "type": "Attack"},
                {"id": "COMMON_B", "name": "Common B", "rarity": "Common", "type": "Skill"},
                {"id": "COMMON_C", "name": "Common C", "rarity": "Common", "type": "Skill"},
                {"id": "COMMON_D", "name": "Common D", "rarity": "Common", "type": "Power"},
                {"id": "COMMON_E", "name": "Common E", "rarity": "Common", "type": "Attack"},
            ]
        },
    )
    option = AncientOptionState(
        option_id="choose_cards",
        name="Choose Cards",
        kind="positive_relic",
        relic_id="anchor",
        metadata={
            "choice": {
                "option_id": "choose_cards",
                "name": "Choose Cards",
                "kind": "positive_relic",
                "relic_id": "anchor",
                "card_reward_count": 1,
                "card_reward_size": 3,
            }
        },
    )
    state = state.model_copy(
        update={
            "ancient": AncientState(act=1, ancient_id="neow", options=(option,)),
            "relics": ("question_card",),
        }
    )

    state = step(state, _action(state, "choose_ancient", "choose_cards"))

    assert state.phase == RunPhase.REWARD
    assert state.reward is not None
    assert state.reward.source == "ancient"
    assert state.reward.forced is True
    assert len(state.reward.card_options) == 4
    assert state.reward.metadata["reward_trigger_effects"][0]["content_id"] == "question_card"
    assert not any(action.type == "proceed" for action in legal_actions(state))

    chosen = state.reward.card_options[0]
    state = step(state, _action(state, "take_reward_card", "reward:card:0"))
    assert state.master_deck[-1].card_id.lower() == chosen
    assert any(action.type == "proceed" for action in legal_actions(state))

    state = step(state, _action(state, "proceed"))
    assert state.phase == RunPhase.MAP
    assert state.reward is None


def test_combat_reward_potion_is_visible_but_requires_open_slot() -> None:
    state = new_run(
        seed=90,
        character_id="TEST",
        ascension=0,
        source_data={
            "max_acts": 1,
            "map_floors": 4,
            "map_width": 1,
            "combat_reward_potion_id": "essence_of_steel",
            "deck": [
                {
                    "card_id": "debug_kill",
                    "name": "Debug Kill",
                    "type": "attack",
                    "cost": 0,
                    "target": "enemy",
                    "effects": {"damage": 999},
                }
            ],
            "player": {"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3},
        },
    )
    state = state.model_copy(
        update={"potions": ("fire_potion", "skill_potion", "foul_potion")}
    )

    state = _choose_first_ancient(state)
    state = _choose_first_node(state)
    state = _play_first_card(state)

    assert state.phase == RunPhase.REWARD
    assert state.reward is not None
    assert state.reward.potion_id == "essence_of_steel"
    assert not any(action.type == "take_reward_potion" for action in legal_actions(state))

    state = step(state, _action(state, "discard_potion", "potion:1"))

    assert state.potions == ("fire_potion", "foul_potion")
    assert any(action.type == "take_reward_potion" for action in legal_actions(state))

    state = step(state, _action(state, "take_reward_potion", "reward:potion"))

    assert state.potions == ("fire_potion", "foul_potion", "essence_of_steel")
    assert state.reward is not None
    assert state.reward.potion_claimed is True


def test_event_reward_potion_uses_same_visible_offer_rules() -> None:
    state = new_run(seed=91, character_id="TEST", ascension=0)
    state = state.model_copy(
        update={
            "phase": RunPhase.EVENT,
            "reward": RewardState(
                reward_id="event:test",
                source="event",
                potion_id="foul_potion",
            ),
            "potions": ("fire_potion", "skill_potion", "essence_of_steel"),
        }
    )

    assert state.reward is not None
    assert state.reward.potion_id == "foul_potion"
    assert not any(action.type == "take_reward_potion" for action in legal_actions(state))

    state = step(state, _action(state, "discard_potion", "potion:0"))
    state = step(state, _action(state, "take_reward_potion", "reward:potion"))

    assert state.potions == ("skill_potion", "essence_of_steel", "foul_potion")


def test_reward_can_offer_multiple_potions_one_slot_at_a_time() -> None:
    state = new_run(seed=910, character_id="TEST", ascension=0)
    state = state.model_copy(
        update={
            "phase": RunPhase.REWARD,
            "reward": RewardState(
                reward_id="reward:multi_potion",
                source="other",
                potion_id="fire_potion",
                potion_ids=("skill_potion",),
            ),
            "potions": ("foul_potion", "essence_of_steel"),
        }
    )

    assert _action(state, "take_reward_potion", "reward:potion")
    assert _action(state, "take_reward_potion", "reward:potion:0")

    state = step(state, _action(state, "take_reward_potion", "reward:potion:0"))

    assert state.potions == ("foul_potion", "essence_of_steel", "skill_potion")
    assert state.reward is not None
    assert state.reward.claimed_potion_indices == (0,)
    assert not any(action.type == "take_reward_potion" for action in legal_actions(state))

    state = step(state, _action(state, "discard_potion", "potion:0"))
    state = step(state, _action(state, "take_reward_potion", "reward:potion"))

    assert state.potions == ("essence_of_steel", "skill_potion", "fire_potion")
    assert state.reward is not None
    assert state.reward.potion_claimed is True


def test_reward_can_offer_fixed_card_alongside_card_choices() -> None:
    state = new_run(
        seed=911,
        character_id="TEST",
        ascension=0,
        source_data={
            "cards": {
                "anger": {
                    "card_id": "anger",
                    "name": "Anger",
                    "type": "attack",
                    "cost": 0,
                    "target": "enemy",
                    "effects": {"damage": 6},
                },
                "LANTERN_KEY": {
                    "card_id": "LANTERN_KEY",
                    "name": "Lantern Key",
                    "type": "status",
                    "cost": -1,
                    "target": "self",
                },
            },
        },
    )
    state = state.model_copy(
        update={
            "phase": RunPhase.REWARD,
            "reward": RewardState(
                reward_id="reward:fixed_card",
                source="other",
                card_options=("anger",),
                card_ids=("lantern_key",),
            ),
        }
    )

    state = step(state, _action(state, "take_reward_card", "reward:fixed_card:0"))

    assert state.master_deck[-1].card_id == "LANTERN_KEY"
    assert state.master_deck[-1].name == "Lantern Key"
    assert state.reward is not None
    assert state.reward.claimed_card_indices == (0,)
    assert _action(state, "take_reward_card", "reward:card:0")


def test_optional_reward_potion_can_be_skipped_without_opening_slot() -> None:
    state = new_run(
        seed=92,
        character_id="TEST",
        ascension=0,
        source_data={
            "max_acts": 1,
            "map_floors": 4,
            "map_width": 1,
            "combat_reward_potion_id": "essence_of_steel",
            "deck": [
                {
                    "card_id": "debug_kill",
                    "name": "Debug Kill",
                    "type": "attack",
                    "cost": 0,
                    "target": "enemy",
                    "effects": {"damage": 999},
                }
            ],
            "player": {"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3},
        },
    )
    state = state.model_copy(
        update={"potions": ("fire_potion", "skill_potion", "foul_potion")}
    )

    state = _choose_first_ancient(state)
    state = _choose_first_node(state)
    state = _play_first_card(state)

    assert state.reward is not None
    assert state.reward.potion_id == "essence_of_steel"
    assert any(action.type == "proceed" for action in legal_actions(state))
    assert not any(action.type == "take_reward_potion" for action in legal_actions(state))

    state = _proceed(state)

    assert state.phase == RunPhase.MAP
    assert state.reward is None
    assert state.potions == ("fire_potion", "skill_potion", "foul_potion")


def test_reward_gold_relic_and_card_choices_are_explicit_actions() -> None:
    state = new_run(
        seed=93,
        character_id="TEST",
        ascension=0,
        source_data={
            "cards": {
                "shrug_it_off": {
                    "card_id": "shrug_it_off",
                    "name": "Shrug It Off",
                    "type": "skill",
                    "cost": 1,
                    "target": "self",
                    "effects": {"block": 8, "draw": 1},
                },
                "anger": {
                    "card_id": "anger",
                    "name": "Anger",
                    "type": "attack",
                    "cost": 0,
                    "target": "enemy",
                    "effects": {"damage": 6},
                },
            }
        },
    )
    state = state.model_copy(
        update={
            "phase": RunPhase.REWARD,
            "reward": RewardState(
                reward_id="reward:test",
                source="other",
                gold=42,
                relic_id="anchor",
                card_options=("shrug_it_off", "anger"),
            ),
        }
    )

    assert any(action.type == "proceed" for action in legal_actions(state))
    assert _action(state, "take_reward_gold", "reward:gold")
    assert _action(state, "take_reward_relic", "reward:relic")
    assert _action(state, "take_reward_card", "reward:card:0")
    assert _action(state, "take_reward_card", "reward:card:1")

    state = step(state, _action(state, "take_reward_gold", "reward:gold"))
    state = step(state, _action(state, "take_reward_relic", "reward:relic"))
    state = step(state, _action(state, "take_reward_card", "reward:card:0"))

    assert state.player.gold == 42
    assert "anchor" in state.relics
    assert state.master_deck[-1].card_id == "shrug_it_off"
    assert state.master_deck[-1].name == "Shrug It Off"
    assert state.reward is not None
    assert state.reward.gold_claimed is True
    assert state.reward.relic_claimed is True
    assert state.reward.card_claimed is True


def test_forced_reward_blocks_proceed_until_taken() -> None:
    state = new_run(seed=94, character_id="TEST", ascension=0)
    state = state.model_copy(
        update={
            "phase": RunPhase.EVENT,
            "reward": RewardState(
                reward_id="event:forced",
                source="event",
                forced=True,
                relic_id="odd_mushroom",
            ),
        }
    )

    assert not any(action.type == "proceed" for action in legal_actions(state))
    assert any(action.type == "take_reward_relic" for action in legal_actions(state))

    state = step(state, _action(state, "take_reward_relic", "reward:relic"))

    assert "odd_mushroom" in state.relics
    assert any(action.type == "proceed" for action in legal_actions(state))


def test_war_historian_cage_reward_consumes_lantern_key_for_history_course() -> None:
    state = new_run(
        seed=930,
        character_id="TEST",
        ascension=0,
        source_data={
            "deck": [
                {
                    "card_id": "lantern_key",
                    "name": "Lantern Key",
                    "type": "status",
                    "cost": -1,
                    "target": "self",
                }
            ],
            "event_reward_remove_card_ids": ("lantern_key",),
            "event_reward_relic_id": "history_course",
        },
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)

    assert state.phase == RunPhase.EVENT
    assert all(card.card_id != "lantern_key" for card in state.master_deck)
    assert state.reward is not None
    assert state.reward.relic_id == "history_course"

    state = step(state, _action(state, "take_reward_relic", "reward:relic"))

    assert "history_course" in state.relics


def test_war_historian_chest_reward_consumes_lantern_key_for_relics_and_potions() -> None:
    state = new_run(
        seed=931,
        character_id="TEST",
        ascension=0,
        source_data={
            "deck": [
                {
                    "card_id": "lantern_key",
                    "name": "Lantern Key",
                    "type": "status",
                    "cost": -1,
                    "target": "self",
                }
            ],
            "relic_pool": (
                {"id": "COMMON_RELIC", "rarity_key": "Common", "pool": "shared"},
                {"id": "UNCOMMON_RELIC", "rarity_key": "Uncommon", "pool": "shared"},
                {"id": "RARE_RELIC", "rarity_key": "Rare", "pool": "shared"},
            ),
            "potion_pool": ("fire_potion", "skill_potion"),
            "event_reward_remove_card_ids": ("lantern_key",),
            "event_reward_relic_count": 2,
            "event_reward_potion_count": 2,
        },
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)

    assert state.phase == RunPhase.EVENT
    assert all(card.card_id != "lantern_key" for card in state.master_deck)
    assert state.reward is not None
    assert len(state.reward.relic_ids) == 2
    assert len(state.reward.potion_ids) == 2

    state = step(state, _action(state, "take_reward_potion", "reward:potion:0"))
    state = step(state, _action(state, "take_reward_potion", "reward:potion:1"))

    assert len(state.potions) == 2
    assert state.reward is not None
    assert state.reward.claimed_potion_indices == (0, 1)


def test_forced_event_room_exposes_cached_event_options() -> None:
    state = new_run(
        seed=940,
        character_id="TEST",
        ascension=0,
        source_data={"event_id": "WAR_HISTORIAN_REPY"},
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)

    assert state.phase == RunPhase.EVENT
    assert state.event is not None
    assert state.event.event_id == "WAR_HISTORIAN_REPY"
    assert {option.option_id for option in state.event.options} == {
        "unlock_cage",
        "unlock_chest",
    }
    assert _action(state, "choose_event", "unlock_chest")


def test_war_historian_cached_chest_choice_consumes_key_and_rewards_loot() -> None:
    state = new_run(
        seed=941,
        character_id="TEST",
        ascension=0,
        source_data={
            "event_id": "WAR_HISTORIAN_REPY",
            "deck": [
                {
                    "card_id": "lantern_key",
                    "name": "Lantern Key",
                    "type": "status",
                    "cost": -1,
                    "target": "self",
                }
            ],
            "relic_pool": (
                {"id": "COMMON_RELIC", "rarity_key": "Common", "pool": "shared"},
                {"id": "UNCOMMON_RELIC", "rarity_key": "Uncommon", "pool": "shared"},
                {"id": "RARE_RELIC", "rarity_key": "Rare", "pool": "shared"},
            ),
            "potion_pool": ("fire_potion", "skill_potion"),
        },
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)
    state = step(state, _action(state, "choose_event", "unlock_chest"))

    assert state.event is not None
    assert state.event.resolved_option_id == "unlock_chest"
    assert all(card.card_id != "lantern_key" for card in state.master_deck)
    assert state.reward is not None
    assert len(state.reward.relic_ids) == 2
    assert len(state.reward.potion_ids) == 2


def test_lantern_key_cached_fight_choice_starts_combat_with_quest_card_reward() -> None:
    state = new_run(
        seed=942,
        character_id="TEST",
        ascension=0,
        source_data={
            "event_id": "THE_LANTERN_KEY",
            "deck": [
                {
                    "card_id": "debug_kill",
                    "name": "Debug Kill",
                    "type": "attack",
                    "cost": 0,
                    "target": "enemy",
                    "effects": {"damage": 999},
                }
            ],
            "cards": (
                {
                    "id": "COMMON_ATTACK",
                    "name": "Common Attack",
                    "rarity": "Common",
                    "color": "test",
                },
                {"id": "COMMON_SKILL", "name": "Common Skill", "rarity": "Common", "color": "test"},
                {
                    "id": "UNCOMMON_SKILL",
                    "name": "Uncommon Skill",
                    "rarity": "Uncommon",
                    "color": "test",
                },
                {"id": "RARE_POWER", "name": "Rare Power", "rarity": "Rare", "color": "test"},
                {
                    "id": "LANTERN_KEY",
                    "name": "Lantern Key",
                    "rarity": "Quest",
                    "color": "quest",
                    "type": "Quest",
                },
            ),
        },
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)
    state = step(state, _action(state, "choose_event", "keep_the_key"))

    assert state.phase == RunPhase.COMBAT
    assert state.flags["combat_reward_card_ids"] == ("lantern_key",)

    state = _play_first_card(state)

    assert state.phase == RunPhase.REWARD
    assert state.reward is not None
    assert state.reward.card_ids == ("lantern_key",)


def test_cached_event_choice_handles_ranged_gold_and_random_relic() -> None:
    state = new_run(
        seed=943,
        character_id="TEST",
        ascension=0,
        source_data={
            "event_id": "THIS_OR_THAT",
            "relic_pool": (
                {"id": "COMMON_RELIC", "rarity_key": "Common", "pool": "shared"},
                {"id": "UNCOMMON_RELIC", "rarity_key": "Uncommon", "pool": "shared"},
            ),
            "cards": (
                {
                    "id": "CLUMSY",
                    "name": "Clumsy",
                    "rarity": "Special",
                    "color": "curse",
                    "type": "Curse",
                },
            ),
        },
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)
    state = step(state, _action(state, "choose_event", "plain"))

    assert state.player.hp == 74
    assert state.reward is not None
    assert 41 <= state.reward.gold <= 68

    state = step(state, _action(state, "take_reward_gold", "reward:gold"))
    state = _force_next_room(_proceed(state), RoomKind.EVENT)
    state = state.model_copy(update={"flags": {**state.flags, "event_id": "THIS_OR_THAT"}})
    state = _choose_first_node(state)
    state = step(state, _action(state, "choose_event", "ornate"))

    assert state.reward is not None
    assert state.reward.card_ids == ("clumsy",)
    assert len(state.reward.relic_ids) == 1


def test_cached_event_choice_handles_fixed_potions_and_max_hp_loss() -> None:
    state = new_run(
        seed=944,
        character_id="TEST",
        ascension=0,
        source_data={
            "event_id": "POTION_COURIER",
            "potions": (
                {"id": "FOUL_POTION", "name": "Foul Potion", "pool": "shared"},
                {"id": "GLOWWATER_POTION", "name": "Glowwater Potion", "pool": "shared"},
            ),
        },
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)
    state = step(state, _action(state, "choose_event", "grab_potions"))

    assert state.reward is not None
    assert state.reward.potion_ids == ("foul_potion", "foul_potion", "foul_potion")

    state = new_run(
        seed=945,
        character_id="TEST",
        ascension=0,
        source_data={"event_id": "DROWNING_BEACON"},
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)
    state = step(state, _action(state, "choose_event", "climb"))

    assert state.player.max_hp == 67
    assert state.reward is not None
    assert state.reward.relic_id == "fresnel_lens"


def test_ancient_curse_option_is_tracked_separately() -> None:
    state = new_run(seed=43, character_id="TEST", ascension=0)

    assert state.ancient is not None
    curse_option = state.ancient.options[-1]
    assert curse_option.kind == "curse_relic"

    state = _choose_ancient_option(state, -1)
    assert curse_option.relic_id in state.relics
    assert curse_option.relic_id in state.curses


def test_ancient_heal_is_reduced_on_ascension_two_plus() -> None:
    state = new_run(
        seed=44,
        character_id="TEST",
        ascension=2,
        source_data={"player": {"hp": 20, "max_hp": 80}},
    )

    assert state.phase.value == "ancient"
    assert state.player.hp == 68


def test_spoils_map_marks_next_act_treasure_and_redeems_if_card_is_kept() -> None:
    state = new_run(
        seed=145,
        character_id="TEST",
        ascension=0,
        source_data={
            "max_acts": 2,
            "map_floors": 8,
            "map_width": 5,
            "boss_reward_card_count": 0,
            "boss_reward_relic_count": 0,
            "deck": [
                {
                    "card_id": "debug_kill",
                    "name": "Debug Kill",
                    "type": "attack",
                    "cost": 0,
                    "target": "enemy",
                    "effects": {"damage": 999},
                }
            ],
            "cards": [
                {
                    "id": "SPOILS_MAP",
                    "name": "Spoils Map",
                    "type": "Quest",
                    "color": "quest",
                    "rarity": "Quest",
                    "cost": -1,
                }
            ],
            "player": {"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3},
        },
    )
    state = state.model_copy(
        update={
            "phase": RunPhase.REWARD,
            "reward": RewardState(
                reward_id="event:the_legends_were_true:nab_the_map",
                source="event",
                forced=True,
                card_ids=("spoils_map",),
            ),
        }
    )

    state = step(state, _action(state, "take_reward_card", "reward:fixed_card:0"))

    assert state.flags["spoils_map_pending_act"] == 2
    assert any(card.card_id == "SPOILS_MAP" for card in state.master_deck)

    state = _force_next_room(state, RoomKind.BOSS)
    state = step(state, _action(state, "choose_node", "target"))
    state = _play_card_by_id(state, "debug_kill")
    state = _proceed(state)

    assert state.phase == RunPhase.ANCIENT
    assert state.act == 2
    assert state.map is not None
    target_id = state.flags["spoils_map_target_node_id"]
    target = state.map.node_by_id[target_id]
    assert target.kind == RoomKind.TREASURE
    assert not _path_can_reach_boss_without_target(state.map, target_id)

    marked_state = state
    missing_state = marked_state.model_copy(
        update={
            "master_deck": tuple(
                card for card in marked_state.master_deck if card.card_id != "SPOILS_MAP"
            )
        }
    )
    missing_state = _move_to_predecessor_of(missing_state, target_id)
    missing_gold = missing_state.player.gold
    missing_state = step(missing_state, _action(missing_state, "choose_node", target_id))

    assert missing_state.player.gold == missing_gold
    assert "spoils_map_target_node_id" not in missing_state.flags

    redeeming_state = _move_to_predecessor_of(marked_state, target_id)
    starting_gold = redeeming_state.player.gold
    redeeming_state = step(
        redeeming_state,
        _action(redeeming_state, "choose_node", target_id),
    )

    assert redeeming_state.player.gold == starting_gold + 600
    assert all(card.card_id != "SPOILS_MAP" for card in redeeming_state.master_deck)
    assert redeeming_state.phase == RunPhase.TREASURE
    assert redeeming_state.flags["spoils_map_completed"] is True


def test_the_legends_were_true_map_choice_starts_spoils_map_quest() -> None:
    state = new_run(
        seed=148,
        character_id="TEST",
        ascension=0,
        source_data={
            "max_acts": 2,
            "event_id": "THE_LEGENDS_WERE_TRUE",
            "cards": [
                {
                    "id": "SPOILS_MAP",
                    "name": "Spoils Map",
                    "type": "Quest",
                    "color": "quest",
                    "rarity": "Quest",
                    "cost": -1,
                }
            ],
        },
    )
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.EVENT)
    state = _choose_first_node(state)
    state = step(state, _action(state, "choose_event", "nab_the_map"))

    assert state.reward is not None
    assert state.reward.card_ids == ("spoils_map",)

    state = step(state, _action(state, "take_reward_card", "reward:fixed_card:0"))

    assert state.flags["spoils_map_pending_act"] == 2


def test_golden_compass_act_two_map_is_single_special_path() -> None:
    state = new_run(
        seed=146,
        character_id="TEST",
        ascension=0,
        source_data={"act": 2, "golden_compass_act2_map": True},
    )

    assert state.map is not None
    nodes_by_floor: dict[int, list[MapNodeState]] = {}
    for node in state.map.nodes:
        nodes_by_floor.setdefault(node.floor, []).append(node)

    expected_kinds = (
        RoomKind.START,
        RoomKind.MONSTER,
        RoomKind.EVENT,
        RoomKind.MONSTER,
        RoomKind.REST,
        RoomKind.MONSTER,
        RoomKind.REST,
        RoomKind.EVENT,
        RoomKind.TREASURE,
        RoomKind.EVENT,
        RoomKind.TREASURE,
        RoomKind.EVENT,
        RoomKind.SHOP,
        RoomKind.ELITE,
        RoomKind.REST,
        RoomKind.ELITE,
        RoomKind.REST,
        RoomKind.BOSS,
    )

    assert tuple(nodes_by_floor) == tuple(range(len(expected_kinds)))
    assert all(len(nodes) == 1 for nodes in nodes_by_floor.values())
    assert tuple(nodes[0].kind for nodes in nodes_by_floor.values()) == expected_kinds
    assert len(state.map.edges) == len(expected_kinds) - 1
    assert all(
        state.map.node_by_id[edge.to_id].floor == state.map.node_by_id[edge.from_id].floor + 1
        for edge in state.map.edges
    )


def test_golden_compass_remarks_existing_spoils_map_target_on_act_two_pickup() -> None:
    state = new_run(seed=147, character_id="TEST", ascension=0, source_data={"act": 2})
    assert state.map is not None
    old_target = next(node for node in state.map.nodes if node.kind == RoomKind.TREASURE)
    state = state.model_copy(
        update={
            "ancient": AncientState(
                act=2,
                ancient_id="tezcatara",
                options=(
                    AncientOptionState(
                        option_id="golden_compass",
                        name="Golden Compass",
                        kind="positive_relic",
                        relic_id="golden_compass",
                    ),
                ),
            ),
            "flags": {
                **state.flags,
                "spoils_map_target_act": 2,
                "spoils_map_target_node_id": old_target.node_id,
                "spoils_map_reward_gold": 600,
            },
        }
    )

    state = step(state, _action(state, "choose_ancient", "golden_compass"))

    assert state.map is not None
    assert state.flags["golden_compass_act2_map"] is True
    assert state.flags["spoils_map_target_node_id"] in state.map.node_by_id
    assert state.flags["spoils_map_target_node_id"] != old_target.node_id
    assert len({node.lane for node in state.map.nodes}) == 1


def test_generated_map_edges_are_sparse_and_adjacent() -> None:
    state = new_run(
        seed=123,
        character_id="TEST",
        ascension=0,
        source_data={"map_width": 5, "map_floors": 8, "map_paths": 6},
    )

    assert state.map is not None
    nodes = state.map.node_by_id
    outgoing: dict[str, list[str]] = {}
    for edge in state.map.edges:
        start = nodes[edge.from_id]
        end = nodes[edge.to_id]
        outgoing.setdefault(edge.from_id, []).append(edge.to_id)

        assert end.floor == start.floor + 1
        assert abs(end.lane - start.lane) <= 1

    exit_counts = [
        len(targets)
        for node_id, targets in outgoing.items()
        if nodes[node_id].kind.value not in {"start", "boss"}
    ]
    assert any(count == 1 for count in exit_counts)
    assert any(count > 1 for count in exit_counts)


def test_default_act_maps_have_realistic_rows_and_fire_boss_row() -> None:
    expected_floors = {1: 17, 2: 16, 3: 15}

    for act, floor_count in expected_floors.items():
        state = new_run(
            seed=1,
            character_id="TEST",
            ascension=0,
            source_data={"act": act},
        )

        assert state.map is not None
        report = validate_act_map_parity(map_layout_from_state(state.map), act=act)
        assert report.ok, [issue.code for issue in report.issues]

        nodes = state.map.node_by_id
        max_floor = max(node.floor for node in nodes.values())
        rest_floor = max_floor - 1
        treasure_floor = max(2, max_floor - 7)

        assert max_floor + 1 == floor_count
        assert 35 <= len(nodes) <= 70

        for floor in range(floor_count):
            floor_nodes = [node for node in nodes.values() if node.floor == floor]
            assert 1 <= len(floor_nodes) <= 7

        assert {node.kind.value for node in nodes.values() if node.floor == 1} == {"monster"}
        assert {node.kind.value for node in nodes.values() if node.floor == rest_floor} == {
            "rest"
        }
        assert {
            node.kind.value for node in nodes.values() if node.floor == treasure_floor
        } == {"treasure"}
        assert not any(
            node.kind == RoomKind.REST for node in nodes.values() if node.floor == rest_floor - 1
        )

        outgoing: dict[str, list[str]] = {}
        for edge in state.map.edges:
            start = nodes[edge.from_id]
            end = nodes[edge.to_id]
            outgoing.setdefault(edge.from_id, []).append(edge.to_id)
            assert end.floor == start.floor + 1
            assert abs(end.lane - start.lane) <= 1
            assert not (
                start.kind == end.kind
                and start.kind in {RoomKind.ELITE, RoomKind.SHOP, RoomKind.REST, RoomKind.TREASURE}
            )

        for node_id, targets in outgoing.items():
            if nodes[node_id].kind == RoomKind.START:
                continue
            if len(targets) < 2:
                continue
            target_kinds = [nodes[target_id].kind for target_id in targets]
            assert len(target_kinds) == len(set(target_kinds))

        for fire in [node for node in nodes.values() if node.floor == rest_floor]:
            assert outgoing[fire.node_id] == [state.map.boss_node_id]
