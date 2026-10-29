from __future__ import annotations

from collections import Counter

import pytest

from sts2sim import new_run
from sts2sim.engine import transitions

EXPECTED = (
    ("IRONCLAD", 80, "burning_blood", 5, 4, ("bash",)),
    ("SILENT", 70, "ring_of_the_snake", 5, 5, ("neutralize", "survivor")),
    ("REGENT", 75, "divine_right", 4, 4, ("falling_star", "venerate")),
    ("NECROBINDER", 66, "bound_phylactery", 4, 4, ("bodyguard", "unleash")),
    ("DEFECT", 75, "cracked_core", 4, 4, ("zap", "dualcast")),
)


@pytest.mark.parametrize("character,hp,relic,strikes,defends,specials", EXPECTED)
@pytest.mark.parametrize("cache_available", [True, False])
def test_known_character_starts_match_stable_sources(
    monkeypatch: pytest.MonkeyPatch,
    character: str,
    hp: int,
    relic: str,
    strikes: int,
    defends: int,
    specials: tuple[str, ...],
    cache_available: bool,
) -> None:
    if not cache_available:
        monkeypatch.setattr(transitions, "_initial_cached_source_rows", lambda *_: ())
    state = new_run(seed=87, character_id=character, ascension=0)
    assert state.player.hp == state.player.max_hp == hp
    assert state.player.gold == 99
    assert state.player.energy == state.player.max_energy == 3
    assert state.relics == (relic,)
    expected = Counter(
        {
            f"strike_{character.lower()}": strikes,
            f"defend_{character.lower()}": defends,
            **dict.fromkeys(specials, 1),
        }
    )
    assert Counter(card.card_id for card in state.master_deck) == expected
    assert all(card.type.value != "unknown" and card.effects for card in state.master_deck)
    assert len({card.instance_id for card in state.master_deck}) == len(state.master_deck)


@pytest.mark.parametrize("relic_override", [None, (), []])
def test_explicit_empty_relic_override_is_respected(relic_override: object) -> None:
    state = new_run(
        seed=5, character_id="IRONCLAD", ascension=0, source_data={"relics": relic_override}
    )
    assert state.relics == ()


def test_explicit_character_and_player_overrides_remain_authoritative() -> None:
    state = new_run(
        seed=5,
        character_id="DEFECT",
        ascension=0,
        source_data={
            "player": {"hp": 13, "max_hp": 42, "gold": 8, "energy": 2, "max_energy": 2},
            "relics": ("Anchor",),
            "deck": ({"card_id": "fixture", "type": "skill", "effects": {"block": 1}},),
            "orb_slots": 1,
        },
    )
    assert state.player.max_hp == 42
    assert state.player.gold == 8
    assert state.player.max_energy == 2
    assert state.relics == ("anchor",)
    assert [card.card_id for card in state.master_deck] == ["fixture"]
    assert state.flags["orb_slots"] == 1


def test_unknown_fixture_character_keeps_original_defaults() -> None:
    state = new_run(seed=5, character_id="TEST", ascension=0)
    assert state.relics == ()
    assert state.player.hp == 80
    assert state.player.gold == 0
    assert Counter(card.card_id for card in state.master_deck) == {
        "strike": 5,
        "defend": 4,
        "bash": 1,
    }


def test_explicit_character_rows_accept_codex_field_names_and_camel_case_ids() -> None:
    state = new_run(
        seed=5,
        character_id="CUSTOM",
        ascension=0,
        source_data={
            "characters": [
                {
                    "id": "CUSTOM",
                    "starting_hp": 33,
                    "starting_gold": 17,
                    "max_energy": 4,
                    "starting_relics": ["CrackedCore"],
                    "starting_deck": ["Zap"],
                }
            ],
            "cards": [
                {"id": "ZAP", "type": "Skill", "cost": 1, "description": "Channel 1 Lightning."}
            ],
        },
    )
    assert (state.player.max_hp, state.player.gold, state.player.max_energy) == (33, 17, 4)
    assert state.relics == ("cracked_core",)
    assert state.master_deck[0].card_id == "zap"
