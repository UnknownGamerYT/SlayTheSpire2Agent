"""v0.107.1 encounter generation compared with the independent public C port."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from random import Random

import pytest

from sts2sim.mechanics.monsters import (
    EncounterDefinition,
    build_encounter_definitions,
    choose_encounter,
)
from sts2sim.mechanics.run_encounters import (
    ACT_ENCOUNTER_POOLS,
    ENCOUNTER_TAGS,
    EncounterBagState,
    draw_encounter_sequence,
    draw_from_encounter_bag,
    encounters_for_act,
    generate_act_encounters,
    shares_encounter_tags,
    weak_count_for_act,
)

FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "stable_encounter_source_oracle.json").read_text()
)


class TapeRng:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.draws = 0

    def random(self) -> float:
        value = self.values[self.draws]
        self.draws += 1
        return value


def _encounter(encounter_id: str, *, weak: bool = False, act: int = 1) -> EncounterDefinition:
    return EncounterDefinition(encounter_id, encounter_id, "monster", ("X",), act, weak)


def _source_encounters() -> tuple[EncounterDefinition, ...]:
    return tuple(
        EncounterDefinition(
            encounter_id,
            encounter_id,
            room_type,
            ("X",),
            {"OVERGROWTH": 1, "UNDERDOCKS": 1, "HIVE": 2, "GLORY": 3}[act_id],
            room_type == "monster" and pool_name == "weak",
        )
        for act_id, pools in FIXTURE["pools"].items()
        for pool_name, room_type in (
            ("weak", "monster"),
            ("regular", "monster"),
            ("elite", "elite"),
            ("boss", "boss"),
        )
        for encounter_id in pools[pool_name]
    )


@pytest.mark.parametrize("act_id", ("OVERGROWTH", "UNDERDOCKS", "HIVE", "GLORY"))
def test_pools_match_decompiled_order_and_not_api_sorting(act_id: str) -> None:
    pools = ACT_ENCOUNTER_POOLS[act_id]
    expected = FIXTURE["pools"][act_id]
    for pool_name in ("weak", "regular", "elite", "boss"):
        assert list(getattr(pools, pool_name)) == expected[pool_name]


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["act_id"])
def test_complete_act_plan_matches_independent_c_source_tape(case: dict) -> None:
    rng = TapeRng(case["samples"])
    result = generate_act_encounters(_source_encounters(), rng, act_id=case["act_id"])
    assert result.as_metadata() == {
        key: case[key] for key in ("normal_encounters", "elite_encounters", "boss_encounter")
    }
    assert rng.draws == len(case["samples"])


def test_rejection_uses_full_bag_and_consumes_every_rejected_draw() -> None:
    pool = (_encounter("FLYCONID_NORMAL"), _encounter("CUBEX_CONSTRUCT_NORMAL"))
    rng = TapeRng([0.1, 0.9])
    selected, state = draw_from_encounter_bag(
        pool, rng, EncounterBagState(previous_id="SLIMES_WEAK")
    )
    assert selected == pool[1]
    assert state.remaining_ids == ("FLYCONID_NORMAL",)
    assert rng.draws == 2


def test_no_compatible_candidate_uses_one_unrestricted_draw() -> None:
    pool = (_encounter("SLIMES_WEAK"), _encounter("SLIMES_NORMAL"))
    rng = TapeRng([0.0])
    selected, state = draw_from_encounter_bag(
        pool, rng, EncounterBagState(previous_id="SLIMES_WEAK")
    )
    assert selected == pool[0]
    assert state.remaining_ids == ("SLIMES_NORMAL",)
    assert rng.draws == 1


def test_single_remaining_entry_still_consumes_a_draw() -> None:
    rng = TapeRng([0.8])
    pool = (_encounter("ONLY"),)
    assert draw_encounter_sequence(pool, rng, 1) == pool
    assert rng.draws == 1


def test_bag_state_json_reload_preserves_remaining_pool_and_previous() -> None:
    pool = tuple(_encounter(f"E{index}") for index in range(3))
    rng = Random(732)
    _, state = draw_from_encounter_bag(pool, rng)
    payload = json.loads(json.dumps(asdict(state)))
    restored = EncounterBagState(tuple(payload["remaining_ids"]), payload["previous_id"])
    resumed_rng = Random()
    resumed_rng.setstate(rng.getstate())
    assert draw_from_encounter_bag(pool, resumed_rng, restored) == draw_from_encounter_bag(
        pool, rng, state
    )


def test_refill_blocks_last_encounter_when_an_alternative_remains() -> None:
    pool = tuple(_encounter(f"E{index}") for index in range(3))
    sequence = draw_encounter_sequence(pool, Random(4), 18)
    assert len(sequence) == 18
    for index in range(0, 18, 3):
        assert set(sequence[index : index + 3]) == set(pool)
    assert all(left != right for left, right in zip(sequence, sequence[1:], strict=False))


def test_stable_tags_do_not_invent_newer_beta_jaxfruit_or_nibbit_tags() -> None:
    assert shares_encounter_tags("SLIMES_WEAK", "FLYCONID_NORMAL")
    assert shares_encounter_tags("SNAPPING_JAXFRUIT_NORMAL", "FLYCONID_NORMAL")
    assert "SLITHERING_STRANGLER_NORMAL" not in ENCOUNTER_TAGS
    assert "NIBBITS_NORMAL" not in ENCOUNTER_TAGS
    assert not shares_encounter_tags("NIBBITS_NORMAL", "NIBBITS_WEAK")


def test_act_binding_never_falls_back_to_other_act_or_other_act_one_variant() -> None:
    encounters = _source_encounters()
    underdocks = encounters_for_act(encounters, "UNDERDOCKS")
    assert underdocks
    assert {encounter.encounter_id for encounter in underdocks} == (
        ACT_ENCOUNTER_POOLS["UNDERDOCKS"].all_ids
    )
    assert not encounters_for_act(encounters, "INVALID_ACT")
    assert (
        choose_encounter(
            tuple(
                encounter
                for encounter in encounters
                if encounter.encounter_id in ACT_ENCOUNTER_POOLS["OVERGROWTH"].all_ids
            ),
            Random(3),
            act=1,
            room_type="monster",
            act_id="UNDERDOCKS",
        )
        is None
    )
    assert choose_encounter(encounters, Random(3), act=4, room_type="monster") is None


def test_named_act_source_label_is_retained_for_custom_encounters() -> None:
    encounters = build_encounter_definitions(
        (
            {
                "id": "CUSTOM",
                "act": "Act 1 - Underdocks",
                "room_type": "Monster",
                "monsters": [{"id": "X"}],
            },
        )
    )
    assert encounters[0].act_id == "UNDERDOCKS"
    assert encounters_for_act(encounters, "UNDERDOCKS") == encounters
    assert not encounters_for_act(encounters, "OVERGROWTH")


def test_source_weak_counts_and_multiplayer_room_count_override() -> None:
    assert [weak_count_for_act(act) for act in ACT_ENCOUNTER_POOLS] == [3, 3, 2, 2]
    for act_id, pools in ACT_ENCOUNTER_POOLS.items():
        plan = generate_act_encounters(
            _source_encounters(), Random(8), act_id=act_id, room_count=pools.room_count - 1
        )
        assert len(plan.normal_ids) == pools.room_count - 1
        assert set(plan.normal_ids[: pools.weak_count]) <= set(pools.weak)
        assert set(plan.normal_ids[pools.weak_count :]) <= set(pools.regular)
        assert len(plan.elite_ids) == 15


def test_profile_discovery_and_first_run_overrides_apply_after_rolls() -> None:
    ordinary_rng = Random(53)
    tutorial_rng = Random(53)
    ordinary = generate_act_encounters(_source_encounters(), ordinary_rng, act_id="OVERGROWTH")
    tutorial = generate_act_encounters(
        _source_encounters(),
        tutorial_rng,
        act_id="OVERGROWTH",
        number_of_runs=0,
        seen_encounters=[],
    )
    assert tutorial.normal_ids[:7] == (
        "NIBBITS_WEAK",
        "SLIMES_WEAK",
        "SHRINKER_BEETLE_WEAK",
        "INKLETS_NORMAL",
        "MAWLER_NORMAL",
        "RUBY_RAIDERS_NORMAL",
        "NIBBITS_NORMAL",
    )
    assert tutorial.elite_ids[:2] == ("BYRDONIS_ELITE", "PHROG_PARASITE_ELITE")
    assert tutorial.boss_id == "VANTOM_BOSS"
    assert len(tutorial.normal_ids) == len(ordinary.normal_ids)
    assert ordinary_rng.getstate() == tutorial_rng.getstate()
    next_boss = generate_act_encounters(
        _source_encounters(),
        Random(53),
        act_id="OVERGROWTH",
        seen_encounters=["VANTOM_BOSS"],
    )
    assert next_boss.boss_id == "CEREMONIAL_BEAST_BOSS"


def test_missing_source_pool_does_not_admit_unrelated_fights() -> None:
    only_other_act = (_encounter("SCROLLS_OF_BITING_WEAK", weak=True, act=3),)
    assert generate_act_encounters(only_other_act, Random(3), act_id="OVERGROWTH").normal_ids == ()
    assert generate_act_encounters(only_other_act, Random(3), act_id="INVALID").boss_id is None
