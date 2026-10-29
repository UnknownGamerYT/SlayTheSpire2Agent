"""RewardsSet.WithRewardsFromRoom and AmethystAubergine stable source boundaries."""

from __future__ import annotations

import pytest

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    Action,
    ActionType,
    MapEdgeState,
    MapNodeState,
    MapState,
    RoomKind,
    RunPhase,
    load_state,
    serialize_state,
)
from sts2sim.engine.rng import capture_random_state, random_from_state
from sts2sim.engine.transitions import _combat_reward_state
from sts2sim.mechanics.reward_triggers import (
    RewardGenerationContext,
    resolve_reward_generation_triggers,
)


@pytest.mark.parametrize("act,character,empty", [(3, "IRONCLAD", True), (3, "TEST", False),
                                               (1, "IRONCLAD", False)])
def test_final_act_boss_base_rewards_and_amethyst_do_not_draw(act, character, empty) -> None:
    state = new_run("FINAL-REWARDS", character, 10, source_data={"relics": [
        "amethyst_aubergine", "white_beast_statue", "question_card",
    ]})
    state = state.model_copy(update={"act": act, "flags": {
        **state.flags, "combat_reward_encounter": "boss", "max_acts": 1,
    }})
    rng = random_from_state(state.rng)
    before = capture_random_state(rng)
    reward, _, flags = _combat_reward_state(state, rng)
    if empty:
        assert reward.gold == 0
        assert not (
            reward.card_options or reward.relic_ids or reward.potion_id or reward.potion_ids
        )
        assert capture_random_state(rng) == before
        assert flags == state.flags
    else:
        assert reward.card_options
        assert reward.gold > 0


def test_amethyst_modifier_explicitly_rejects_final_boss_context() -> None:
    result = resolve_reward_generation_triggers(RewardGenerationContext(
        source="combat", encounter_type="boss", card_choice_count=0,
        owned_relics=("amethyst_aubergine",), metadata={"final_act_boss": True},
    ))
    assert result.gold_delta == 0


@pytest.mark.parametrize("ascension,boss_index", [(0, 0), (10, 0), (10, 1)])
def test_public_final_boss_empty_rewards_proceed_to_next_boss_or_completion(
    ascension, boss_index,
) -> None:
    state = new_run("FINAL-BOSS", "IRONCLAD", ascension, source_data={
        "deck": [{"card_id": "debug_kill", "cost": 0, "type": "attack",
                  "target": "all_enemies", "effects": {"all_damage": 999999}}],
        "draw_per_turn": 1, "relics": ["burning_blood", "amethyst_aubergine"],
        "monsters": [{"id": "FIXTURE", "hp": 1, "max_hp": 1,
                      "moves": [{"id": "WAIT", "intent": "defend"}]}],
        "combat_monster_ids": ["FIXTURE"],
    })
    start = MapNodeState(node_id="start", act=3, floor=0, lane=3, kind=RoomKind.START)
    boss = MapNodeState(node_id="boss", act=3, floor=14, lane=3, kind=RoomKind.BOSS)
    second = MapNodeState(node_id="second", act=3, floor=16, lane=3, kind=RoomKind.BOSS)
    nodes = (start, boss, second) if ascension == 10 else (start, boss)
    edges = (MapEdgeState(from_id="start", to_id="boss"),)
    if ascension == 10:
        edges += (MapEdgeState(from_id="boss", to_id="second"),)
    game_map = MapState(act=3, nodes=nodes, edges=edges, current_node_id="start",
                       completed_node_ids=("start",), boss_node_id=nodes[-1].node_id)
    state = state.model_copy(update={"act": 3, "phase": RunPhase.MAP, "ancient": None,
                                     "map": game_map})
    for index in range(boss_index + 1):
        target = "boss" if index == 0 else "second"
        state = step(state, Action(type=ActionType.CHOOSE_NODE, target_id=target))
        kill = next(a for a in legal_actions(state) if a.type == ActionType.PLAY_CARD)
        state = step(state, kill)
        assert state.phase == RunPhase.REWARD and state.reward is not None
        assert state.reward.gold == 0 and state.reward.card_options == ()
        proceed = next(a for a in legal_actions(state) if a.type == ActionType.PROCEED)
        restored = step(load_state(serialize_state(state)), proceed)
        state = step(state, proceed)
        assert serialize_state(state) == serialize_state(restored)
    assert state.phase == (RunPhase.MAP if ascension == 10 and boss_index == 0
                           else RunPhase.COMPLETE)
