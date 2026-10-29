"""Exact stable rest amount projection, including low maximum HP and Ascension."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2sim.mechanics.ascension import (
    AscensionFlag,
    flags_for_ascension,
    rest_site_heal_fraction_at_ascension,
)
from sts2sim.mechanics.campfire import rest_heal_amount
from sts2sim.mechanics.source_rest import source_rest_heal_amount

SOURCE = json.loads((Path(__file__).parents[1] / "fixtures/stable_rest_source.json").read_text())


@pytest.mark.parametrize("max_hp,expected", SOURCE["floor_vectors"])
def test_source_fraction_truncation_and_no_minimum_heal(max_hp, expected) -> None:
    assert rest_heal_amount(max_hp) == expected
    assert source_rest_heal_amount(max_hp) == expected
    assert source_rest_heal_amount(max_hp, relics=("REGAL_PILLOW",)) == expected + 15


@pytest.mark.parametrize("ascension", range(11))
def test_rest_fraction_is_unchanged_at_every_stable_ascension(ascension) -> None:
    assert rest_heal_amount(83, ascension_level=ascension) == 24
    assert rest_site_heal_fraction_at_ascension(ascension) == 0.3
    assert AscensionFlag.REDUCED_REST_HEAL not in flags_for_ascension(ascension)


def test_night_terrors_overrides_regal_pillow_after_relic_hook() -> None:
    assert source_rest_heal_amount(83, relics=("regal_pillow",),
                                   modifiers=("NIGHT_TERRORS",)) == 83
