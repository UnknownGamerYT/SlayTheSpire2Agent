from __future__ import annotations

import json

import pytest

from sts2sim import action_space, agent_previews, encode_observation, legal_actions, serialize
from sts2sim.engine import (
    CardInstance,
    CombatState,
    MapEdgeState,
    MapNodeState,
    MapState,
    MonsterState,
    PlayerState,
    RoomKind,
    RunPhase,
    RunState,
)
from sts2sim.engine.rng import new_rng_state
from sts2sim.learning.masked_ppo import _policy_input
from sts2sim.learning.observation import encode_rich_observation
from sts2sim.mechanics.belief import belief_summary


def _combat(*, frozen_eye: bool = False) -> RunState:
    attack = CardInstance(
        instance_id="hit",
        card_id="hit",
        type="attack",
        cost=0,
        target="enemy",
        effects={"damage": 6},
    )
    draw = CardInstance(
        instance_id="draw",
        card_id="draw",
        type="skill",
        cost=0,
        target="self",
        effects={"draw": 1},
    )
    potion = CardInstance(
        instance_id="alchemy",
        card_id="alchemy",
        type="skill",
        cost=0,
        target="self",
        effects={"add_random_potion": 1},
    )
    pile = tuple(
        CardInstance(
            instance_id=f"card:{i}",
            card_id=f"card_{i}",
            type="attack" if i < 3 else "skill",
            cost=0,
            target="enemy" if i < 3 else "self",
            effects={"damage": i + 1} if i < 3 else {"block": i + 1},
        )
        for i in range(7)
    )
    player = PlayerState(hp=70, max_hp=80, energy=3)
    return RunState(
        seed=123,
        character_id="TEST",
        rng=new_rng_state(123),
        phase=RunPhase.COMBAT,
        player=player,
        master_deck=(attack, draw, potion, *pile),
        relics=("FROZEN_EYE",) if frozen_eye else (),
        combat=CombatState(
            player=player,
            hand=(attack, draw, potion),
            draw_pile=pile,
            draw_per_turn=2,
            monsters=(MonsterState(monster_id="enemy", hp=40, max_hp=40),),
        ),
    )


def _different_hidden_state(state: RunState) -> RunState:
    return state.model_copy(
        update={
            "rng": new_rng_state(987),
            "combat": state.combat.model_copy(update={"draw_pile": state.combat.draw_pile[::-1]}),
        }
    )


def test_policy_inputs_are_invariant_to_hidden_draw_order_and_rng() -> None:
    first = _combat()
    second = _different_hidden_state(first)
    snapshots = [serialize(first), serialize(second)]

    first_observation = encode_observation(first, include_state=True)
    second_observation = encode_observation(second, include_state=True)
    first_actions = action_space(first)
    second_actions = action_space(second)

    assert first_observation == second_observation
    assert first_actions == second_actions
    assert encode_rich_observation(first, include_serialized_state=True) == (
        encode_rich_observation(second, include_serialized_state=True)
    )
    assert _policy_input(first_observation, {"action_space": first_actions}) == (
        _policy_input(second_observation, {"action_space": second_actions})
    )
    assert [serialize(first), serialize(second)] == snapshots


def test_hidden_draw_composition_has_no_top_positions_in_compact_or_rich_inputs() -> None:
    state = _combat()
    observation = encode_observation(state, include_state=True)
    rich = encode_rich_observation(state)

    assert observation["visibility_mode"] == rich["visibility_mode"] == "visible"
    assert observation["policy_observation_schema_version"] == 2
    assert observation["visibility"]["draw_order_visible"] is False
    assert all(card["position_from_top"] is None for card in rich["card_zones"]["draw_pile"])
    assert all(card["position"] is None for card in rich["card_zones"]["draw_pile"])
    assert all(
        card["position"] == 0 for card in observation["visibility"]["card_slots"]["draw_pile"]
    )
    assert "rng" not in observation["state"]
    assert "seed" not in observation["state"]
    assert "replay_log" not in observation["state"]
    json.dumps(rich)


def test_belief_draw_probabilities_use_all_unseen_cards_not_the_actual_top_window() -> None:
    state = _combat()
    first = belief_summary(serialize(state))
    second = belief_summary(serialize(_different_hidden_state(state)))
    assert first == second
    assert first["draw_attack_chance"] == round(3 / 7, 4)
    assert first["draw_block_chance"] == round(4 / 7, 4)


@pytest.mark.parametrize("oracle", [False, True])
def test_frozen_eye_and_explicit_oracle_can_reveal_draw_order(oracle: bool) -> None:
    first = _combat(frozen_eye=not oracle)
    second = _different_hidden_state(first)
    first_observation = encode_observation(first, oracle=oracle)
    second_observation = encode_observation(second, oracle=oracle)

    assert first_observation["visibility"]["draw_order_visible"] is True
    assert first_observation["vector"] != second_observation["vector"]
    assert first_observation["positions"]["cards"]["draw_pile"][0]["position_from_top"] == 0
    assert (
        encode_rich_observation(first, oracle=oracle)["card_zones"]["draw_pile"][0]["card_id"]
        == first.combat.draw_pile[0].card_id
    )


def test_safe_previews_keep_visible_damage_and_suppress_random_draw_and_next_turn() -> None:
    descriptors = action_space(_combat())
    by_card = {d["card_instance_id"]: d for d in descriptors if d["type"] == "play_card"}
    assert by_card["hit"]["preview"]["target_hp_delta"] == -6
    assert by_card["draw"]["preview"] == {"preview_available": 0}
    assert by_card["alchemy"]["preview"] == {"preview_available": 0}
    assert next(d for d in descriptors if d["type"] == "end_turn")["preview"] == {
        "preview_available": 0
    }
    assert all(not any(key.startswith("second_turn") for key in d["preview"]) for d in descriptors)


def test_default_previews_never_use_the_saved_game_rng(monkeypatch) -> None:
    real_step = agent_previews._step
    seen_rngs = []

    def inspect_step(state, action):
        seen_rngs.append(state.rng)
        return real_step(state, action)

    monkeypatch.setattr(agent_previews, "_step", inspect_step)
    state = _combat()
    agent_previews.preview_actions(state, legal_actions(state))
    assert seen_rngs
    assert all(rng == new_rng_state(0) for rng in seen_rngs)
    assert all(rng != state.rng for rng in seen_rngs)


def test_map_previews_do_not_simulate_unknown_rooms_or_encounters(monkeypatch) -> None:
    state = _combat()
    start = MapNodeState(node_id="start", act=1, floor=0, lane=0, kind=RoomKind.START)
    nodes = [
        MapNodeState(node_id=kind.value, act=1, floor=1, lane=i, kind=kind)
        for i, kind in enumerate([RoomKind.MONSTER, RoomKind.EVENT, RoomKind.SHOP])
    ]
    state = state.model_copy(
        update={
            "phase": RunPhase.MAP,
            "combat": None,
            "map": MapState(
                nodes=(start, *nodes),
                current_node_id="start",
                completed_node_ids=("start",),
                edges=tuple(MapEdgeState(from_id="start", to_id=n.node_id) for n in nodes),
            ),
        }
    )

    def unexpected_step(*_args, **_kwargs):
        pytest.fail("Default policy preview tried to simulate an unseen room")

    monkeypatch.setattr(agent_previews, "_step", unexpected_step)
    first_actions = action_space(state)
    second_actions = action_space(state.model_copy(update={"rng": new_rng_state(999)}))
    assert first_actions == second_actions
    assert all(d["preview"] == {"preview_available": 0} for d in first_actions)


def test_oracle_is_explicit_and_retains_lossless_debug_state_and_lookahead() -> None:
    state = _combat()
    observation = encode_observation(state, oracle=True, include_state=True)
    descriptors = action_space(state, oracle=True)

    assert observation["visibility_mode"] == "oracle"
    assert observation["state"] == serialize(state)
    assert any("second_turn_best_damage" in descriptor["preview"] for descriptor in descriptors)
