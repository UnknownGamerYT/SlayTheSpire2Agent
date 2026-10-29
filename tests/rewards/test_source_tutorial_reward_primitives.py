from dataclasses import replace

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import MapNodeState, MapState, RoomKind, RunPhase
from sts2sim.engine.rng import capture_random_state, random_from_state, random_stream
from sts2sim.engine.source_tutorial_rewards import (
    HISTORY,
    draw_tutorial_bundle,
    is_tutorial_profile,
    record_combat_entry,
    tutorial_reward_metadata,
    tutorial_reward_spec,
)
from sts2sim.mechanics.rewards import EncounterType, RewardPityState


def _first_run(ascension=0):
    return new_run("tutorial-rewards", "IRONCLAD", ascension, source_data={
        "number_of_runs": 0, "revealed_epochs": [],
    })


def _enter_counted_room(state, kind, index):
    node = MapNodeState(node_id=f"tutorial:{kind.value}:{index}", act=state.act,
                        floor=index + 1, lane=0, kind=kind)
    state = state.model_copy(update={"map": MapState(
        nodes=(node,), current_node_id=node.node_id, act=state.act,
    )})
    return record_combat_entry(state, node)


def test_first_run_public_initial_combat_records_its_map_point_once():
    state = _first_run()
    assert state.phase is RunPhase.MAP and is_tutorial_profile(state)
    combat = step(load_state(serialize(state)), legal_actions(state)[0])
    assert combat.phase is RunPhase.COMBAT
    spec = tutorial_reward_spec(combat, EncounterType.NORMAL)
    assert spec.index == 0 and spec.cards == ("setup_strike", "tremble", "blood_wall")
    assert len(combat.flags[HISTORY]) == 1
    assert tutorial_reward_spec(load_state(serialize(combat)), EncounterType.NORMAL) == spec


def test_profile_requires_exact_first_run_ironclad_and_no_revealed_epochs():
    state = _first_run()
    assert is_tutorial_profile(state)
    assert not is_tutorial_profile(state.model_copy(update={"character_id": "SILENT"}))
    assert not is_tutorial_profile(state.model_copy(update={
        "flags": {**state.flags, "number_of_runs": 1},
    }))
    assert not is_tutorial_profile(state.model_copy(update={
        "flags": {**state.flags, "revealed_epochs": ["IRONCLAD_1"]},
    }))
    assert not is_tutorial_profile(new_run("ordinary", "IRONCLAD", 0))
    assert not is_tutorial_profile(new_run("partially-specified", "IRONCLAD", 0,
                                         source_data={"number_of_runs": 0}))


def test_all_seven_fixed_monster_sets_preserve_pity_and_only_draw_gold():
    expected = (
        (("setup_strike", "tremble", "blood_wall"), None),
        (("breakthrough", "inflame", "anger"), None),
        (("iron_wave", "dismantle", "cinder"), "fire_potion"),
        (("stomp", "shrug_it_off", "armaments"), None),
        (("thunderclap", "setup_strike", "rage"), "strength_potion"),
        (("battle_trance", "true_grit", "uppercut"), None),
        (("bloodletting", "whirlwind", "tremble"), "energy_potion"),
    )
    state = _first_run(3)
    pity = RewardPityState(card_non_rare_count=8, card_rarity_offset=0.125,
                           potion_current_value=0.7, potion_chance_bonus=30)
    for index, (cards, potion) in enumerate(expected):
        state = _enter_counted_room(state, RoomKind.MONSTER, index)
        spec = tutorial_reward_spec(state, EncounterType.NORMAL)
        assert spec.index == index
        expected_rng = random_from_state(state.rng)
        expected_gold = random_stream(expected_rng, "rewards").randint(10, 20)
        rng = random_stream(random_from_state(state.rng), "rewards")
        bundle = draw_tutorial_bundle(state, rng, spec, pity)
        assert bundle.card_ids == cards and bundle.potion_id == potion
        assert bundle.gold == expected_gold and 10 <= bundle.gold <= 20  # No Poverty scaling.
        assert bundle.card_upgraded == (False, False, False)
        assert bundle.potion_roll is None and bundle.pity_state == pity
        assert capture_random_state(rng) == capture_random_state(expected_rng)
        assert not bundle.relic_ids
        state = load_state(serialize(state))
    state = _enter_counted_room(state, RoomKind.MONSTER, 7)
    assert tutorial_reward_spec(state, EncounterType.NORMAL) is None


@pytest.mark.parametrize("kind,expected", [
    (RoomKind.ELITE, (
        (("bludgeon", "pyre", "evil_eye"), "block_potion", "vajra"),
        (("pillage", "rampage", "flame_barrier"), None, "ornamental_fan"),
    )),
    (RoomKind.BOSS, ((("primal_force", "demon_form", "thrash"), None, None),)),
])
def test_elite_and_boss_manual_rewards_do_not_roll_potions_cards_or_relics(kind, expected):
    state = _first_run(3)
    encounter = EncounterType.ELITE if kind is RoomKind.ELITE else EncounterType.BOSS
    for index, (cards, potion, relic) in enumerate(expected):
        state = _enter_counted_room(state, kind, index)
        spec = tutorial_reward_spec(state, encounter)
        rng = random_stream(random_from_state(state.rng), "rewards")
        bundle = draw_tutorial_bundle(state, rng, spec, RewardPityState())
        assert bundle.card_ids == cards and bundle.potion_id == potion
        assert bundle.relic_ids == ((relic,) if relic else ())
        assert bundle.card_upgraded == (False, False, False) and bundle.potion_roll is None
        before = state.rng.streams.get("rewards")
        expected_draws = (before.draws if before else 0) + 1
        assert capture_random_state(rng).streams["rewards"].draws == expected_draws
        assert (26 <= bundle.gold <= 33) if kind is RoomKind.ELITE else bundle.gold == 75
    state = _enter_counted_room(state, kind, len(expected))
    assert tutorial_reward_spec(state, encounter) is None


def test_nested_event_combats_count_once_and_do_not_become_elite_map_points():
    state = _first_run()
    point = MapNodeState(node_id="event:two-fights", act=1, floor=2, lane=0, kind=RoomKind.EVENT)
    state = state.model_copy(update={"map": MapState(nodes=(point,),
                                                     current_node_id=point.node_id)})
    room = point.model_copy(update={"kind": RoomKind.MONSTER})
    state = record_combat_entry(record_combat_entry(state, room), room)
    assert len(state.flags[HISTORY]) == 1
    assert tutorial_reward_spec(state, EncounterType.NORMAL).index == 0
    assert tutorial_reward_spec(state, EncounterType.ELITE) is None
    state = state.model_copy(update={"act": 2})
    state = _enter_counted_room(state, RoomKind.MONSTER, 0)
    assert tutorial_reward_spec(load_state(serialize(state)), EncounterType.NORMAL).index == 1


def test_manual_initial_flags_and_for_room_reroll_flags_are_distinct():
    state = _enter_counted_room(_first_run(), RoomKind.MONSTER, 0)
    spec = tutorial_reward_spec(state, EncounterType.NORMAL)
    context = tutorial_reward_metadata(spec)["card_reroll_contexts"]["primary"]
    assert context["initial_is_card_reward"] is True and context["is_card_reward"] is False
    assert context["updates_on_relic_obtain"] is False and context["count"] == 3
    assert tutorial_reward_metadata(replace(spec, encounter=EncounterType.BOSS))[
        "card_reroll_contexts"]["primary"]["encounter"] == "boss"
