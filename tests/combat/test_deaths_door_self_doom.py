from __future__ import annotations

from combat.test_combat_engine_monsters import _play_first_card
from combat.test_monster_innate_hooks import _combat, _monster
from combat.test_source_card_conditions import _no_stubs, _source


def test_deaths_door_counts_successful_doom_applied_to_yourself() -> None:
    doom = _source(
        "self_doom",
        "Apply 2 Doom to yourself.",
        target="self",
        powers_applied=({"power_key": "Doom", "amount": 2, "target": "self"},),
    )
    door = _source(
        "deaths_door",
        "Gain 6 Block. If you applied Doom this turn, gain Block 2 additional times.",
        block=6,
    )
    state = _combat(_monster("VICTIM", {}), deck=(doom, door))
    state = _play_first_card(state, "self_doom")
    assert state.combat.player.statuses["doom"] == 2
    state = _play_first_card(state, "deaths_door")
    assert state.combat.player.block == 18
    _no_stubs(state)
