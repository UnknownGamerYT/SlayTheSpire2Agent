from __future__ import annotations

import pytest

from sts2sim.agent_api import _potion_belt_summary
from sts2sim.engine import PlayerState, RunState
from sts2sim.engine.rng import new_rng_state


@pytest.mark.parametrize("owned", [("potion_belt",), ()])
def test_permanent_potion_slot_pickup_is_not_counted_again_when_relic_is_owned_or_melts(
    owned: tuple[str, ...],
) -> None:
    state = RunState(
        seed=4,
        character_id="TEST",
        rng=new_rng_state(4),
        player=PlayerState(hp=80, max_hp=80),
        relics=owned,
        potions=("fire_potion",),
        flags={"bonus_potion_slots": 2, "potion_slot_relic_pickups": ("potion_belt",)},
    )

    summary = _potion_belt_summary(state)

    assert summary["capacity"] == 5
    assert summary["open_slots"] == 4


@pytest.mark.parametrize("base_slots", [0, 4])
def test_explicit_potion_slots_is_a_base_before_relic_and_permanent_bonuses(
    base_slots: int,
) -> None:
    state = RunState(
        seed=4,
        character_id="TEST",
        rng=new_rng_state(4),
        player=PlayerState(hp=80, max_hp=80),
        relics=("potion_belt",),
        flags={"potion_slots": base_slots, "bonus_potion_slots": 1},
    )

    summary = _potion_belt_summary(state)

    assert summary["base_slots"] == base_slots
    assert summary["capacity"] == base_slots + 3
