"""All 239 core card constructors and upgrade keyword/cost declarations.

Pinned to Razhuo/sts2-rl-agent@5d874e855ae1d0550d5c30416fadd7dbb0d133b7,
C# MegaCrit.Sts2.Core.Models.Cards, stable 0.107.1 build23811903.
This metadata crosswalk does not certify card effects or timing parity.
"""

import json
from pathlib import Path

import pytest

from sts2sim.engine.transitions import _card_from_spec, _card_has_keyword

ROWS = json.loads((Path(__file__).parents[1] / "fixtures/source_core_metadata.json").read_text())


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
@pytest.mark.parametrize("upgraded", [False, True])
def test_core_normalized_constructor_and_upgrade_metadata(row, upgraded):
    card = _card_from_spec({**row["input"], "upgraded": upgraded}, 0)
    expected = row["expected"]
    assert card.card_id == row["id"]
    want_cost = (
        -1 if row["input"].get("is_x_cost") else expected["upgraded_cost" if upgraded else "cost"]
    )
    assert card.cost == want_cost
    assert bool(card.custom.get("x_cost")) == bool(row["input"].get("is_x_cost"))
    assert card.type.value == expected["type"]
    for keyword in ("innate", "ethereal", "exhaust", "retain", "sly", "unplayable", "eternal"):
        want = keyword in expected["upgraded_keywords" if upgraded else "keywords"]
        assert _card_has_keyword(card, keyword) == want
