from __future__ import annotations

from sts2sim import new_run
from sts2sim.engine import CardType
from sts2sim.engine.transitions import (
    _card_from_spec,
    _card_library,
    _card_spec_from_library,
    _reward_card_spec,
)


def test_card_library_resolves_source_id_aliases() -> None:
    library = _card_library(
        {
            "cards": (
                {
                    "id": "TWIN_STRIKE",
                    "name": "Twin Strike",
                    "type": "Attack",
                },
            )
        }
    )

    assert _card_spec_from_library(library, "TWIN_STRIKE")["type"] == "Attack"
    assert _card_spec_from_library(library, "twin_strike")["type"] == "Attack"
    assert _card_spec_from_library(library, "Twin Strike")["type"] == "Attack"


def test_reward_cards_fall_back_to_cached_source_specs_when_state_library_is_partial() -> None:
    state = new_run(
        seed=303,
        character_id="TEST",
        ascension=0,
        source_data={
            "cards": (
                {
                    "id": "DEBUG_ONLY",
                    "name": "Debug Only",
                    "type": "Skill",
                },
            )
        },
    )

    expected_types = {
        "havoc": CardType.SKILL,
        "twin_strike": CardType.ATTACK,
        "spore_mind": CardType.CURSE,
    }
    for index, (card_id, expected_type) in enumerate(expected_types.items(), start=1):
        card = _card_from_spec(_reward_card_spec(state, card_id), index)

        assert card.type is expected_type
