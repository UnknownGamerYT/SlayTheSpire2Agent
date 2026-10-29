"""Constructor, costs and keywords from stable source; no effect-parity claim."""

import json
from pathlib import Path

import pytest

from sts2sim.engine.transitions import _card_from_spec, _card_has_keyword

FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures/source_class_metadata.json").read_text()
)
ROWS = FIXTURE["rows"]


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["id"])
@pytest.mark.parametrize("upgraded", [False, True])
def test_class_constructor_costs_and_keywords_match_source(row, upgraded):
    card = _card_from_spec({**row["input"], "upgraded": upgraded}, 1)
    expected = row["expected"]
    assert card.card_id == row["id"]
    assert card.cost == expected["upgraded_cost" if upgraded else "cost"]
    assert card.type.value == expected["type"]
    assert bool(card.custom.get("x_cost")) == expected["x_cost"]
    assert (card.custom.get("star_cost") == "X") == expected["x_star_cost"]
    if expected["star_cost"] is not None and not expected["x_star_cost"]:
        assert card.custom.get("star_cost") == expected["star_cost"]
    expected_keywords = expected["upgraded_keywords" if upgraded else "keywords"]
    for keyword in ("innate", "ethereal", "exhaust", "retain", "sly", "unplayable", "eternal"):
        assert _card_has_keyword(card, keyword) == (keyword in expected_keywords), keyword


def test_class_metadata_fixture_covers_the_full_catalog_scope():
    assert FIXTURE["game_version"] == "0.107.1"
    assert len(ROWS) == 338
