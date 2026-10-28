from __future__ import annotations

import json

import pytest

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import (
    CardInstance,
    MapEdgeState,
    MapNodeState,
    MapState,
    PlayerState,
    RoomKind,
    RunPhase,
    RunState,
)
from sts2sim.engine.rng import new_rng_state


def _run(*, relics: tuple[str, ...] = ("FISHING_ROD",)) -> RunState:
    return RunState(
        seed=173,
        character_id="TEST",
        rng=new_rng_state(173),
        player=PlayerState(hp=80, max_hp=80),
        relics=relics,
        master_deck=(
            CardInstance(
                instance_id="finisher:1",
                card_id="finisher",
                type="attack",
                cost=0,
                target="enemy",
                effects={"damage": 999},
            ),
            *(
                CardInstance(
                    instance_id=f"guard:{index}",
                    card_id="guard",
                    type="skill",
                    cost=0,
                    target="self",
                    effects={"block": 5},
                )
                for index in range(4)
            ),
        ),
        flags={
            "combat_reward_card_count": 0,
            "combat_reward_relic_count": 0,
            "combat_reward_gold": 10,
            "combat_reward_potion_chance_percent": 0,
        },
    )


def _enter_combat(state: RunState, kind: RoomKind = RoomKind.MONSTER) -> RunState:
    start = MapNodeState(
        node_id="start", act=state.act, floor=state.floor, lane=0, kind=RoomKind.START
    )
    fight = MapNodeState(
        node_id="fight",
        act=state.act,
        floor=state.floor + 1,
        lane=0,
        kind=RoomKind.MONSTER if kind == RoomKind.EVENT else kind,
    )
    game_map = MapState(
        act=state.act,
        nodes=(start, fight),
        edges=(MapEdgeState(from_id="start", to_id="fight"),),
        current_node_id="start",
        completed_node_ids=("start",),
    )
    state = state.model_copy(
        update={"phase": RunPhase.MAP, "map": game_map, "combat": None, "reward": None}
    )
    state = step(state, next(a for a in legal_actions(state) if a.type == "choose_node"))
    if kind == RoomKind.EVENT:
        # Event fights retain their event node rather than becoming normal rooms.
        state = state.model_copy(
            update={
                "map": game_map.model_copy(
                    update={
                        "current_node_id": "fight",
                        "nodes": (start, fight.model_copy(update={"kind": kind})),
                    }
                )
            }
        )
    return state


def _win(state: RunState) -> RunState:
    state = step(
        state,
        next(
            action
            for action in legal_actions(state)
            if action.type == "play_card" and action.card_instance_id == "finisher:1"
        ),
    )
    assert state.phase == RunPhase.REWARD
    return state


def test_fishing_rod_upgrades_once_every_three_normal_victories() -> None:
    state = _run()
    for wins in range(1, 7):
        state = _win(_enter_combat(state))

        assert state.flags["relic_counters"]["fishing_rod"] == wins % 3
        assert sum(card.upgraded for card in state.master_deck) == wins // 3
        upgrade_events = [
            event
            for event in state.replay_log[-1].events
            if event.kind == "relic_deck_card_upgraded" and event.source_id == "fishing_rod"
        ]
        assert len(upgrade_events) == (1 if wins % 3 == 0 else 0)


@pytest.mark.parametrize("kind", [RoomKind.ELITE, RoomKind.BOSS, RoomKind.EVENT])
def test_fishing_rod_does_not_count_other_encounter_victories(kind: RoomKind) -> None:
    state = _win(_enter_combat(_run()))
    state = _win(_enter_combat(state, kind))

    assert state.flags["relic_counters"]["fishing_rod"] == 1
    assert not any(card.upgraded for card in state.master_deck)


def test_fishing_rod_does_not_count_turns_or_defeats() -> None:
    state = _enter_combat(_run())
    state = step(state, next(a for a in legal_actions(state) if a.type == "end_turn"))
    assert state.phase == RunPhase.COMBAT
    assert "fishing_rod" not in state.combat.metadata["relic_counters"]

    # A lethal enemy turn must not be treated as a completed normal combat.
    player = state.player.model_copy(update={"hp": 1})
    combat = state.combat.model_copy(update={"player": player})
    state = state.model_copy(update={"player": player, "combat": combat})
    state = step(state, next(a for a in legal_actions(state) if a.type == "end_turn"))
    assert state.phase == RunPhase.FAILED
    assert "fishing_rod" not in state.flags.get("relic_counters", {})
    assert not any(card.upgraded for card in state.master_deck)


def test_fishing_rod_upgrade_and_counter_survive_save_reload() -> None:
    state = _run()
    for _ in range(2):
        state = _win(_enter_combat(state))
    restored = load_state(json.loads(json.dumps(serialize(state))))

    continuous = _win(_enter_combat(state))
    resumed = _win(_enter_combat(restored))

    assert serialize(resumed) == serialize(continuous)
    assert sum(card.upgraded for card in resumed.master_deck) == 1
    assert resumed.flags["relic_counters"]["fishing_rod"] == 0

    # Claiming a reward does not trigger another combat completion.
    resumed = step(resumed, next(a for a in legal_actions(resumed) if a.type == "take_reward_gold"))
    assert resumed.flags["relic_counters"]["fishing_rod"] == 0
    assert sum(card.upgraded for card in resumed.master_deck) == 1


def test_fishing_rod_handles_no_eligible_upgrade_without_repeating_trigger() -> None:
    state = _run()
    state = state.model_copy(
        update={
            "master_deck": tuple(
                card.model_copy(update={"upgraded": True}) for card in state.master_deck
            ),
            "flags": {**state.flags, "relic_counters": {"fishing_rod": 2}},
        }
    )
    state = _win(_enter_combat(state))
    assert state.flags["relic_counters"]["fishing_rod"] == 0
    assert not any(
        event.kind == "relic_deck_card_upgraded" for event in state.replay_log[-1].events
    )


def test_normal_combat_does_not_add_fishing_rod_counter_without_relic() -> None:
    state = _win(_enter_combat(_run(relics=())))
    assert "fishing_rod" not in state.flags.get("relic_counters", {})
    assert not any(card.upgraded for card in state.master_deck)
