from __future__ import annotations

from random import Random

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    MapEdgeState,
    MapNodeState,
    MapState,
    RewardState,
    RoomKind,
    RunPhase,
)
from sts2sim.mechanics import (
    TreasureContext,
    build_treasure_relic_pool,
    draw_treasure_reward,
)

TREASURE_POOL = (
    {"id": "ANCHOR", "name": "Anchor", "rarity_key": "Common", "pool": "shared"},
    {"id": "AKABEKO", "name": "Akabeko", "rarity_key": "Uncommon", "pool": "shared"},
    {"id": "OLD_COIN", "name": "Old Coin", "rarity_key": "Rare", "pool": "shared"},
)

PICKUP_TEST_CARDS = (
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
        "id": "STRIKE",
        "name": "Strike",
        "rarity": "Common",
        "color": "test",
        "type": "Attack",
        "target": "Enemy",
        "damage": 6,
        "upgrade": {"damage": "+3"},
    },
    {
        "id": "DEFEND",
        "name": "Defend",
        "rarity": "Common",
        "color": "test",
        "type": "Skill",
        "target": "Self",
        "block": 5,
        "upgrade": {"block": "+3"},
    },
    {
        "id": "TRUE_GRIT",
        "name": "True Grit",
        "rarity": "Uncommon",
        "color": "test",
        "type": "Skill",
        "target": "Self",
        "block": 7,
        "upgrade": {"block": "+3"},
    },
    {
        "id": "BASH",
        "name": "Bash",
        "rarity": "Uncommon",
        "color": "test",
        "type": "Attack",
        "target": "Enemy",
        "damage": 8,
    },
)


def _treasure_state(*, seed: int = 1, ascension: int = 0, relics=()):
    state = new_run(
        seed=seed,
        character_id="TEST",
        ascension=ascension,
        source_data={
            "treasure_relic_pool": TREASURE_POOL,
            "map_floors": 4,
            "map_width": 1,
        },
    )
    start = MapNodeState(
        node_id="start",
        act=1,
        floor=0,
        lane=0,
        kind=RoomKind.START,
    )
    treasure = MapNodeState(
        node_id="treasure",
        act=1,
        floor=1,
        lane=0,
        kind=RoomKind.TREASURE,
    )
    map_state = MapState(
        act=1,
        nodes=(start, treasure),
        edges=(MapEdgeState(from_id=start.node_id, to_id=treasure.node_id),),
        current_node_id=start.node_id,
        completed_node_ids=(start.node_id,),
        boss_node_id=None,
    )
    return state.model_copy(
        update={
            "phase": RunPhase.MAP,
            "map": map_state,
            "relics": tuple(relics),
            "ancient": None,
            "room_history": (),
            "replay_log": (),
        }
    )


def _action(state, action_type: str, target_id: str | None = None):
    return next(
        action
        for action in legal_actions(state)
        if action.type == action_type
        and (target_id is None or action.target_id == target_id)
    )


def _enter_treasure(state):
    return step(state, _action(state, "choose_node", "treasure"))


def test_treasure_node_offers_gold_and_source_backed_relic() -> None:
    state = _enter_treasure(_treasure_state(seed=10))

    assert state.phase == RunPhase.TREASURE
    assert state.reward is not None
    assert 42 <= state.reward.gold <= 52
    assert state.reward.relic_id in {"anchor", "akabeko", "old_coin"}
    assert state.reward.metadata["relic_rarity"] in {"common", "uncommon", "rare"}
    assert state.flags["treasure_chests_opened"] == 1
    assert any(action.type == "take_reward_gold" for action in legal_actions(state))
    assert any(action.type == "take_reward_relic" for action in legal_actions(state))
    assert any(action.type == "proceed" for action in legal_actions(state))


def test_treasure_reward_can_be_taken_or_skipped() -> None:
    state = _enter_treasure(_treasure_state(seed=11))
    assert state.reward is not None
    gold = state.reward.gold
    relic_id = state.reward.relic_id

    state = step(state, _action(state, "take_reward_gold", "reward:gold"))
    state = step(state, _action(state, "take_reward_relic", "reward:relic"))

    assert state.player.gold == gold
    assert relic_id in state.relics
    assert state.reward is not None
    assert state.reward.gold_claimed is True
    assert state.reward.relic_claimed is True

    state = step(state, _action(state, "proceed"))

    assert state.phase == RunPhase.MAP
    assert state.reward is None


def test_treasure_poverty_ascension_reduces_chest_gold() -> None:
    state = _enter_treasure(_treasure_state(seed=12, ascension=3))

    assert state.reward is not None
    assert state.reward.gold == int(state.reward.metadata["base_gold"] * 0.75)


def test_silver_crucible_makes_first_treasure_chest_empty() -> None:
    state = _enter_treasure(_treasure_state(seed=13, relics=("silver_crucible",)))

    assert state.reward is not None
    assert state.reward.gold == 0
    assert state.reward.relic_id is None
    assert state.reward.metadata["empty"] is True
    assert state.reward.metadata["empty_reason"] == "first_chest_empty_relic"
    assert state.flags["treasure_chests_opened"] == 1
    assert not any(action.type == "take_reward_gold" for action in legal_actions(state))
    assert not any(action.type == "take_reward_relic" for action in legal_actions(state))
    assert any(action.type == "proceed" for action in legal_actions(state))


def test_treasure_relic_rarity_falls_forward_when_pool_is_exhausted() -> None:
    pool = build_treasure_relic_pool(
        (
            {"id": "ANCHOR", "rarity_key": "Common", "pool": "shared"},
            {"id": "AKABEKO", "rarity_key": "Uncommon", "pool": "shared"},
        ),
        character_id="TEST",
    )

    reward = draw_treasure_reward(
        Random(1),
        pool,
        TreasureContext(character_id="TEST", owned_relics=("anchor",)),
    )

    assert reward.relic_id == "akabeko"
    assert reward.relic_rarity is not None
    assert reward.relic_rarity.value == "uncommon"


def test_old_coin_treasure_pickup_grants_gold() -> None:
    state = new_run(seed=14, character_id="TEST", ascension=0)
    state = state.model_copy(
        update={
            "phase": RunPhase.TREASURE,
            "reward": RewardState(
                reward_id="treasure:test",
                source="treasure",
                relic_id="old_coin",
            ),
        }
    )
    before_gold = state.player.gold

    state = step(state, _action(state, "take_reward_relic", "reward:relic"))

    assert "old_coin" in state.relics
    assert state.player.gold == before_gold + 300
    assert [event.kind for event in state.replay_log[-1].events] == [
        "reward_relic_taken",
        "relic_gold_gained",
    ]


def test_max_hp_relic_pickup_updates_player_and_waffle_heals_to_full() -> None:
    state = new_run(seed=15, character_id="TEST", ascension=0)
    state = state.model_copy(
        update={
            "phase": RunPhase.TREASURE,
            "player": state.player.model_copy(update={"hp": 30, "max_hp": 80}),
            "reward": RewardState(
                reward_id="treasure:test",
                source="treasure",
                relic_id="mango",
            ),
        }
    )

    state = step(state, _action(state, "take_reward_relic", "reward:relic"))

    assert state.player.max_hp == 94
    assert state.player.hp == 44
    assert any(event.kind == "relic_max_hp_changed" for event in state.replay_log[-1].events)

    state = state.model_copy(
        update={
            "phase": RunPhase.TREASURE,
            "reward": RewardState(
                reward_id="treasure:test:waffle",
                source="treasure",
                relic_id="lees_waffle",
            ),
        }
    )

    state = step(state, _action(state, "take_reward_relic", "reward:relic"))

    assert state.player.max_hp == 101
    assert state.player.hp == 101


def test_fixed_card_pickup_relic_adds_card_to_master_deck() -> None:
    state = new_run(
        seed=16,
        character_id="TEST",
        ascension=0,
        source_data={"deck": (), "cards": PICKUP_TEST_CARDS},
    )
    state = state.model_copy(
        update={
            "phase": RunPhase.TREASURE,
            "reward": RewardState(
                reward_id="treasure:jewelry_box",
                source="treasure",
                relic_id="jewelry_box",
            ),
        }
    )

    state = step(state, _action(state, "take_reward_relic", "reward:relic"))

    assert "jewelry_box" in state.relics
    assert state.master_deck[-1].card_id.lower() == "apotheosis"
    assert any(event.kind == "relic_deck_card_added" for event in state.replay_log[-1].events)


def test_ghost_seed_pickup_makes_strikes_and_defends_ethereal() -> None:
    state = new_run(
        seed=17,
        character_id="TEST",
        ascension=0,
        source_data={
            "deck": ("STRIKE", "DEFEND", "TRUE_GRIT"),
            "cards": PICKUP_TEST_CARDS,
        },
    )
    state = state.model_copy(
        update={
            "phase": RunPhase.TREASURE,
            "reward": RewardState(
                reward_id="treasure:ghost_seed",
                source="treasure",
                relic_id="ghost_seed",
            ),
        }
    )

    state = step(state, _action(state, "take_reward_relic", "reward:relic"))

    by_id = {card.card_id.lower(): card for card in state.master_deck}
    assert by_id["strike"].custom["ethereal"] is True
    assert by_id["defend"].custom["ethereal"] is True
    assert "ethereal" not in by_id["true_grit"].custom


def test_war_paint_pickup_upgrades_two_random_skills_only() -> None:
    state = new_run(
        seed=18,
        character_id="TEST",
        ascension=0,
        source_data={
            "deck": ("STRIKE", "DEFEND", "TRUE_GRIT"),
            "cards": PICKUP_TEST_CARDS,
        },
    )
    state = state.model_copy(
        update={
            "phase": RunPhase.TREASURE,
            "reward": RewardState(
                reward_id="treasure:war_paint",
                source="treasure",
                relic_id="war_paint",
            ),
        }
    )

    state = step(state, _action(state, "take_reward_relic", "reward:relic"))

    by_id = {card.card_id.lower(): card for card in state.master_deck}
    assert by_id["strike"].upgraded is False
    assert by_id["defend"].upgraded is True
    assert by_id["true_grit"].upgraded is True
    upgraded_events = [
        event for event in state.replay_log[-1].events if event.kind == "relic_deck_card_upgraded"
    ]
    assert len(upgraded_events) == 2


def test_bone_tea_pickup_stores_next_combat_charge() -> None:
    state = new_run(seed=19, character_id="TEST", ascension=0)
    state = state.model_copy(
        update={
            "phase": RunPhase.TREASURE,
            "reward": RewardState(
                reward_id="treasure:bone_tea",
                source="treasure",
                relic_id="bone_tea",
            ),
        }
    )

    state = step(state, _action(state, "take_reward_relic", "reward:relic"))

    assert "bone_tea" in state.relics
    assert state.flags["relic_counters"]["bone_tea"] == 1
    assert any(
        event.kind == "relic_counter_changed"
        and event.source_id == "bone_tea"
        and event.amount == 1
        for event in state.replay_log[-1].events
    )


def test_calling_bell_pickup_adds_curse_and_random_relics() -> None:
    state = new_run(
        seed=20,
        character_id="TEST",
        ascension=0,
        source_data={
            "deck": (),
            "cards": PICKUP_TEST_CARDS
            + (
                {
                    "id": "CURSE_OF_THE_BELL",
                    "name": "Curse of the Bell",
                    "type": "Curse",
                    "color": "curse",
                },
            ),
            "treasure_relic_pool": TREASURE_POOL,
        },
    )
    state = state.model_copy(
        update={
            "phase": RunPhase.TREASURE,
            "reward": RewardState(
                reward_id="treasure:calling_bell",
                source="treasure",
                relic_id="calling_bell",
            ),
        }
    )

    state = step(state, _action(state, "take_reward_relic", "reward:relic"))

    assert "calling_bell" in state.relics
    assert state.master_deck[-1].card_id.lower() == "curse_of_the_bell"
    assert len(set(state.relics) & {"anchor", "akabeko", "old_coin"}) == 3
    assert any(event.kind == "event_relic_obtained" for event in state.replay_log[-1].events)


def test_cauldron_pickup_adds_random_potions_until_slots_are_full() -> None:
    state = new_run(
        seed=21,
        character_id="TEST",
        ascension=0,
        source_data={"event_reward_potion_pool": ("fire_potion", "skill_potion")},
    )
    state = state.model_copy(
        update={
            "phase": RunPhase.TREASURE,
            "reward": RewardState(
                reward_id="treasure:cauldron",
                source="treasure",
                relic_id="cauldron",
            ),
        }
    )

    state = step(state, _action(state, "take_reward_relic", "reward:relic"))

    assert len(state.potions) == 3
    assert set(state.potions) <= {"fire_potion", "skill_potion"}
    potion_events = [
        event for event in state.replay_log[-1].events if event.kind == "relic_potion_obtained"
    ]
    skipped_events = [
        event
        for event in state.replay_log[-1].events
        if event.kind == "event_potion_skipped_no_slot"
    ]
    assert len(potion_events) == 3
    assert len(skipped_events) == 2


def test_pandoras_box_pickup_transforms_strikes_and_defends() -> None:
    state = new_run(
        seed=22,
        character_id="TEST",
        ascension=0,
        source_data={
            "deck": ("STRIKE", "DEFEND", "TRUE_GRIT"),
            "cards": PICKUP_TEST_CARDS,
        },
    )
    state = state.model_copy(
        update={
            "phase": RunPhase.TREASURE,
            "reward": RewardState(
                reward_id="treasure:pandoras_box",
                source="treasure",
                relic_id="pandoras_box",
            ),
        }
    )

    state = step(state, _action(state, "take_reward_relic", "reward:relic"))

    card_ids = {card.card_id.lower() for card in state.master_deck}
    assert "strike" not in card_ids
    assert "defend" not in card_ids
    assert len(state.master_deck) == 3
    assert any(
        event.kind == "relic_deck_card_transformed" for event in state.replay_log[-1].events
    )


def test_orrery_pickup_opens_card_reward_groups() -> None:
    state = new_run(
        seed=23,
        character_id="TEST",
        ascension=0,
        source_data={"deck": (), "cards": PICKUP_TEST_CARDS},
    )
    state = state.model_copy(
        update={
            "phase": RunPhase.TREASURE,
            "reward": RewardState(
                reward_id="treasure:orrery",
                source="treasure",
                relic_id="orrery",
            ),
        }
    )

    state = step(state, _action(state, "take_reward_relic", "reward:relic"))

    assert state.reward is not None
    assert state.reward.relic_claimed is True
    assert state.reward.card_options == ()
    assert len(state.reward.card_option_groups) == 5
    assert all(len(group) == 3 for group in state.reward.card_option_groups)
    assert any(
        event.kind == "reward_card_group_generated" and event.source_id == "orrery"
        for event in state.replay_log[-1].events
    )


def test_chosen_pickup_deck_mutations_report_choice_required() -> None:
    state = new_run(
        seed=24,
        character_id="TEST",
        ascension=0,
        source_data={
            "deck": ("STRIKE", "DEFEND", "TRUE_GRIT"),
            "cards": PICKUP_TEST_CARDS,
        },
    )
    state = state.model_copy(
        update={
            "phase": RunPhase.TREASURE,
            "reward": RewardState(
                reward_id="treasure:empty_cage",
                source="treasure",
                relic_id="empty_cage",
            ),
        }
    )

    state = step(state, _action(state, "take_reward_relic", "reward:relic"))

    assert len(state.master_deck) == 3
    event = next(
        event for event in state.replay_log[-1].events if event.kind == "relic_deck_choice_required"
    )
    assert event.source_id == "empty_cage"
    assert event.amount == 2
    assert len(event.metadata["candidate_card_instance_ids"]) == 3
