from __future__ import annotations

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    MapEdgeState,
    MapNodeState,
    MapState,
    RoomKind,
    RunPhase,
)

REWARD_CARDS = (
    {
        "id": "COMMON_ATTACK",
        "name": "Common Attack",
        "rarity": "Common",
        "color": "test",
        "type": "Attack",
        "target": "AnyEnemy",
        "damage": 6,
        "upgrade": {"damage": "+3"},
    },
    {
        "id": "COMMON_SKILL",
        "name": "Common Skill",
        "rarity": "Common",
        "color": "test",
        "type": "Skill",
        "target": "Self",
        "block": 5,
        "upgrade": {"block": "+3"},
    },
    {
        "id": "UNCOMMON_SKILL",
        "name": "Uncommon Skill",
        "rarity": "Uncommon",
        "color": "test",
        "type": "Skill",
        "target": "Self",
        "block": 7,
        "upgrade": {"block": "+3"},
    },
    {
        "id": "RARE_POWER_A",
        "name": "Rare Power A",
        "rarity": "Rare",
        "color": "test",
        "type": "Power",
        "target": "Self",
        "effects": {"apply_status": {"target": "self", "strength": 1}},
    },
    {
        "id": "RARE_POWER_B",
        "name": "Rare Power B",
        "rarity": "Rare",
        "color": "test",
        "type": "Power",
        "target": "Self",
        "effects": {"apply_status": {"target": "self", "dexterity": 1}},
    },
    {
        "id": "RARE_POWER_C",
        "name": "Rare Power C",
        "rarity": "Rare",
        "color": "test",
        "type": "Power",
        "target": "Self",
        "effects": {"apply_status": {"target": "self", "focus": 1}},
    },
)
REWARD_RELICS = (
    {"id": "COMMON_RELIC", "name": "Common Relic", "rarity_key": "Common", "pool": "shared"},
    {"id": "UNCOMMON_RELIC", "name": "Uncommon Relic", "rarity_key": "Uncommon", "pool": "shared"},
    {"id": "RARE_RELIC", "name": "Rare Relic", "rarity_key": "Rare", "pool": "shared"},
    {"id": "BOSS_RELIC", "name": "Boss Relic", "rarity_key": "Ancient", "pool": "shared"},
)


def _choose_first_ancient(state):
    action = next(action for action in legal_actions(state) if action.type == "choose_ancient")
    return step(state, action)


def _action(state, action_type: str, target_id: str | None = None):
    return next(
        action
        for action in legal_actions(state)
        if action.type == action_type
        and (target_id is None or action.target_id == target_id)
    )


def _force_next_room(state, room_kind: RoomKind):
    start = MapNodeState(node_id="start", act=1, floor=0, lane=0, kind=RoomKind.START)
    target = MapNodeState(node_id="target", act=1, floor=1, lane=0, kind=room_kind)
    game_map = MapState(
        act=1,
        nodes=(start, target),
        edges=(MapEdgeState(from_id=start.node_id, to_id=target.node_id),),
        current_node_id=start.node_id,
        completed_node_ids=(start.node_id,),
        boss_node_id=target.node_id if room_kind is RoomKind.BOSS else None,
    )
    return state.model_copy(update={"phase": RunPhase.MAP, "map": game_map, "floor": 0})


def _reward_after_kill(
    room_kind: RoomKind,
    *,
    seed: int,
    extra_source: dict | None = None,
    relics: tuple[str, ...] = (),
):
    source_data = {
        "max_acts": 1,
        "map_floors": 4,
        "map_width": 1,
        "cards": REWARD_CARDS,
        "relic_pool": REWARD_RELICS,
        "boss_relic_pool": REWARD_RELICS,
        "potion_pool": ("fire_potion",),
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
    }
    if extra_source:
        source_data.update(extra_source)

    state = new_run(seed=seed, character_id="TEST", ascension=0, source_data=source_data)
    state = _choose_first_ancient(state)
    if relics:
        state = state.model_copy(update={"relics": relics})
    state = _force_next_room(state, room_kind)
    state = step(state, _action(state, "choose_node", "target"))
    return step(state, _action(state, "play_card"))


def test_normal_combat_reward_uses_default_gold_and_card_choices() -> None:
    state = _reward_after_kill(RoomKind.MONSTER, seed=201)

    assert state.phase == RunPhase.REWARD
    assert state.reward is not None
    assert state.reward.metadata["encounter"] == "normal"
    assert 10 <= state.reward.gold <= 20
    assert len(state.reward.card_options) == 3
    assert state.reward.relic_ids == ()
    assert "potion_chance_bonus" in state.flags


def test_elite_combat_reward_adds_random_relic_reward() -> None:
    state = _reward_after_kill(RoomKind.ELITE, seed=202)

    assert state.reward is not None
    assert state.reward.metadata["encounter"] == "elite"
    assert 35 <= state.reward.gold <= 45
    assert len(state.reward.relic_ids) == 1
    assert state.reward.metadata["relic_rarities"][0] in {"common", "uncommon", "rare"}

    relic_id = state.reward.relic_ids[0]
    state = step(state, _action(state, "take_reward_relic", "reward:relic:0"))

    assert relic_id in state.relics
    assert state.reward is not None
    assert state.reward.claimed_relic_ids == (relic_id,)


def test_black_star_adds_extra_default_elite_relic_reward() -> None:
    state = _reward_after_kill(RoomKind.ELITE, seed=212, relics=("black_star",))

    assert state.reward is not None
    assert state.reward.metadata["encounter"] == "elite"
    assert len(state.reward.relic_ids) == 2
    assert set(state.reward.metadata["relic_rarities"]) <= {"common", "uncommon", "rare"}


def test_egg_relics_upgrade_reward_cards_before_adding_to_deck() -> None:
    state = _reward_after_kill(
        RoomKind.MONSTER,
        seed=213,
        relics=("molten_egg",),
        extra_source={"combat_reward_card_options": ("COMMON_ATTACK",)},
    )

    assert state.reward is not None
    state = step(state, _action(state, "take_reward_card", "reward:card:0"))

    gained = state.master_deck[-1]
    assert gained.card_id.lower() == "common_attack"
    assert gained.upgraded is True
    assert gained.effects["sequence"][0]["damage"] == 9
    assert state.reward is not None
    assert any(event.kind == "relic_card_upgraded" for event in state.replay_log[-1].events)


def test_boss_combat_reward_adds_gold_rare_cards_and_boss_relic() -> None:
    state = _reward_after_kill(RoomKind.BOSS, seed=203)

    assert state.reward is not None
    assert state.reward.metadata["encounter"] == "boss"
    assert state.reward.gold == 100
    assert len(state.reward.card_options) == 3
    assert state.reward.metadata["card_rarities"] == ("rare", "rare", "rare")
    assert state.reward.relic_ids == ("boss_relic",)
    assert state.reward.metadata["relic_rarities"] == ("ancient",)


def test_fake_merchant_event_combat_reward_gives_rug_and_unsold_relics() -> None:
    state = _reward_after_kill(
        RoomKind.MONSTER,
        seed=204,
        extra_source={
            "combat_reward_encounter": "event",
            "combat_reward_event_id": "fake_merchant",
            "fake_merchant_unsold_relic_ids": ("fake_anchor", "fake_mango"),
        },
    )

    assert state.reward is not None
    assert state.reward.gold == 0
    assert state.reward.card_options == ()
    assert state.reward.relic_ids == (
        "fake_merchants_rug",
        "fake_anchor",
        "fake_mango",
    )

    for index, relic_id in enumerate(state.reward.relic_ids):
        state = step(state, _action(state, "take_reward_relic", f"reward:relic:{index}"))
        assert relic_id in state.relics

    assert state.reward is not None
    assert state.reward.claimed_relic_ids == (
        "fake_merchants_rug",
        "fake_anchor",
        "fake_mango",
    )
