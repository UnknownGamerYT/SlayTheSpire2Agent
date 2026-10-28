from __future__ import annotations

from random import Random

import pytest

from sts2sim.mechanics.unknown_rooms import BASE_UNKNOWN_ROOM_ODDS, resolve_unknown_room


class FixedRoll(Random):
    def __init__(self, roll: float) -> None:
        super().__init__(0)
        self.roll = roll
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return self.roll


@pytest.mark.parametrize(
    "roll,outcome", [(0.05, "monster"), (0.11, "treasure"), (0.14, "shop"), (0.8, "event")]
)
def test_source_roll_order_and_single_rng_draw(roll: float, outcome: str) -> None:
    rng = FixedRoll(roll)
    result = resolve_unknown_room(rng)
    assert result.outcome == outcome
    assert rng.calls == 1
    assert result.next_odds["elite"] == -1.0
    for key in ("monster", "treasure", "shop"):
        expected = BASE_UNKNOWN_ROOM_ODDS[key] * (1 if key == outcome else 2)
        assert result.next_odds[key] == pytest.approx(expected)


def test_misses_accumulate_and_a_hit_resets_only_its_own_odds() -> None:
    missed = resolve_unknown_room(FixedRoll(0.99))
    second = resolve_unknown_room(FixedRoll(0.99), previous_odds=missed.next_odds)
    hit = resolve_unknown_room(FixedRoll(0.25), previous_odds=second.next_odds)
    assert hit.outcome == "monster"
    assert hit.next_odds["monster"] == 0.1
    assert hit.next_odds["shop"] == pytest.approx(0.12)
    # Calling without prior odds starts a new act at the base probabilities.
    reset = resolve_unknown_room(FixedRoll(0.25))
    assert reset.outcome == "event"


def test_blacklist_blocks_roll_and_pity_increase() -> None:
    previous = {**BASE_UNKNOWN_ROOM_ODDS, "monster": 0.9}
    result = resolve_unknown_room(FixedRoll(0.05), previous_odds=previous, blacklisted=["monster"])
    assert result.outcome == "event"
    assert result.next_odds["monster"] == 0.9


def test_modifier_can_enable_elites_and_non_event_odds_need_no_normalization() -> None:
    result = resolve_unknown_room(FixedRoll(0.2), base_odds={"elite": 0.25})
    assert result.outcome == "elite"
    assert result.next_odds["elite"] == 0.25
    result = resolve_unknown_room(FixedRoll(0.99), previous_odds={"monster": 1.2})
    assert result.outcome == "monster"


def test_invalid_snapshot_does_not_consume_randomness() -> None:
    rng = FixedRoll(0.5)
    with pytest.raises(ValueError, match="finite"):
        resolve_unknown_room(rng, previous_odds={"shop": float("nan")})
    assert rng.calls == 0
