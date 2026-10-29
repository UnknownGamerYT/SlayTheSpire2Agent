from __future__ import annotations

import pytest

from sts2sim.live_relic_state import relic_runtime_projection


def _project(relic_id: str, counter: object = None, **runtime: object) -> dict:
    player = {"relics": [{"id": relic_id, "counter": counter, "runtime_state": runtime}]}
    flags, metadata = relic_runtime_projection(player, {"round": 4, "player_turn": 5})
    assert flags["relic_counters"] == metadata["relic_counters"]
    return flags["relic_counters"]


@pytest.mark.parametrize(
    ("relic_id", "counter", "runtime", "expected"),
    [
        ("NUNCHAKU", 9, {"attacks_played": 29}, 9),
        ("PEN_NIB", 10, {"attacks_played": 0}, 0),
        ("NUNCHAKU", 10, {"attacks_played": 21}, 1),
        ("IRON_CLUB", 3, {"cards_played": 43}, 43),
        ("IRON_CLUB", 4, {"cards_played": 43}, 43),
        ("HAPPY_FLOWER", 2, {"turns_seen": 2}, 2),
        ("FAKE_HAPPY_FLOWER", 5, {"turns_seen": 0}, 0),
        ("JOSS_PAPER", 4, {"cards_exhausted": 4}, 4),
        ("GALACTIC_DUST", 7, {"stars_spent": 7}, 7),
        ("METRONOME", 7, {"orbs_channeled": 19}, 19),
        ("PUMPKIN_CANDLE", 8, {"kindle_count": 8}, 8),
        ("PAELS_LEGION", None, {"cooldown": 0}, 0),
        ("PAELS_LEGION", None, {"cooldown": -4}, 0),
        ("PAELS_LEGION", 2, {"cooldown": 2}, 2),
        ("PENDULUM", 3, {"turns_seen": 1}, 1),
        ("POLLINOUS_CORE", 3, {}, 3),
        ("FISHING_ROD", 2, {}, 2),
        ("BOOK_OF_FIVE_RINGS", 4, {}, 4),
        ("WONGOS_MYSTERY_TICKET", None, {"gave_relic": True}, 0),
    ],
)
def test_exact_relic_progress_and_saturation_contracts(
    relic_id: str, counter: object, runtime: dict, expected: int
) -> None:
    assert _project(relic_id, counter, **runtime) == {relic_id.lower(): expected}


@pytest.mark.parametrize(
    ("relic_id", "field"),
    [
        ("burning_sticks", "used_this_combat"),
        ("centennial_puzzle", "used_this_combat"),
        ("permafrost", "used_this_combat"),
        ("ruined_helmet", "used_this_combat"),
        ("throwing_axe", "used_this_combat"),
        ("paels_eye", "used_this_combat"),
        ("demon_tongue", "triggered_this_turn"),
        ("mini_regent", "used_this_turn"),
        ("red_skull", "strength_applied"),
        ("belt_buckle", "dexterity_applied"),
        ("vambrace", "block_gained_this_combat"),
    ],
)
def test_one_shot_history_must_be_explicit_not_inferred_from_generic_status(
    relic_id: str, field: str
) -> None:
    assert _project(relic_id, **{field: True}) == {relic_id: 1}
    assert _project(relic_id, **{field: False}) == {relic_id: 0}
    with pytest.raises(ValueError, match="boolean"):
        _project(relic_id)
    with pytest.raises(ValueError, match="boolean"):
        _project(relic_id, **{field: 1})


def test_lamp_remembers_attempt_even_without_a_successful_power_application() -> None:
    assert _project(
        "unsettling_lamp", triggering_card_set=True, finished_triggering=False
    ) == {"unsettling_lamp": 1}
    assert _project(
        "unsettling_lamp", triggering_card_set=False, finished_triggering=True
    ) == {"unsettling_lamp": 1}


@pytest.mark.parametrize("bad", [True, -1, 2.5, "2", None])
def test_invalid_counter_is_not_silently_coerced(bad: object) -> None:
    with pytest.raises(ValueError, match="integer"):
        _project("nunchaku", 2, attacks_played=bad)


def test_conflicting_visible_counter_and_runtime_fails_closed() -> None:
    with pytest.raises(ValueError, match="disagrees"):
        _project("pen_nib", 8, attacks_played=9)
    with pytest.raises(ValueError, match="source duration"):
        _project("paels_legion", 3, cooldown=3)
    with pytest.raises(ValueError, match="integer"):
        _project("iron_club", 3)
    with pytest.raises(ValueError, match="activation display"):
        _project("pendulum", 3)


def test_spent_ticket_import_cannot_reinitialize_and_pay_again() -> None:
    flags, metadata = relic_runtime_projection({"relics": [{
        "id": "WONGOS_MYSTERY_TICKET", "counter": None,
        "runtime_state": {"gave_relic": True},
    }]}, {"round": 5})
    assert flags["wongos_mystery_ticket_used_up"] is True
    assert flags["initialized_relic_lifecycles"] == ("wongos_mystery_ticket",)
    assert flags["relic_counters"] == metadata["relic_counters"] == {"wongos_mystery_ticket": 0}


def test_noncombat_does_not_require_expired_combat_history_or_private_oracle() -> None:
    player = {"relics": [{"id": "centennial_puzzle"}, {"id": "anchor"}]}
    assert relic_runtime_projection(player, {}) == (
        {"relic_counters": {}}, {"relic_counters": {}}
    )
    # Multiple held quest relics are valid; no generic uniqueness assumption.
    player = {"relics": [{"id": "spoils_map"}, {"id": "spoils_map"}]}
    assert relic_runtime_projection(player, {})[0] == {"relic_counters": {}}
