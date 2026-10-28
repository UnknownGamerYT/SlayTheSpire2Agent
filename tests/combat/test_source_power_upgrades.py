from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2sim.engine.transitions import _card_from_spec


@pytest.mark.parametrize(
    "card_id,status,amount",
    [
        ("PHANTOM_BLADES", "phantomblades", 12),
        ("SERPENT_FORM", "serpentform", 6),
        ("LETHALITY", "lethality", 75),
        ("ITERATION", "iteration", 3),
        ("JUGGERNAUT", "juggernaut", 8),
    ],
)
def test_source_power_upgrade_updates_applied_power_amount(card_id, status, amount) -> None:
    path = Path("data/cache/eng/cards.json")
    if not path.exists():
        pytest.skip("Stable source cache not present")
    row = next(row for row in json.loads(path.read_text()) if row["id"] == card_id)
    card = _card_from_spec({**row, "upgraded": True}, 0)
    statuses = [step["apply_status"] for step in card.effects["sequence"] if "apply_status" in step]
    assert any(payload.get(status) == amount for payload in statuses)


def test_source_neurosurge_energy_upgrade_updates_energy_gain() -> None:
    path = Path("data/cache/eng/cards.json")
    if not path.exists():
        pytest.skip("Stable source cache not present")
    row = next(row for row in json.loads(path.read_text()) if row["id"] == "NEUROSURGE")
    card = _card_from_spec({**row, "upgraded": True}, 0)
    assert any(step.get("energy") == 4 for step in card.effects["sequence"])


@pytest.mark.parametrize("card_id,cost", [("DUALCAST", 0), ("FERAL", 1), ("BANSHEES_CRY", 7)])
def test_numeric_source_upgrade_cost_is_absolute(card_id, cost):
    rows = json.loads(Path("data/cache/eng/cards.json").read_text())
    row = next(row for row in rows if row["id"] == card_id)
    card = _card_from_spec({**row, "upgraded": True}, 0)
    assert card.cost == cost


@pytest.mark.parametrize("card_id,keyword,present", [
    ("AFTERIMAGE", "innate", True), ("DREDGE", "retain", True),
    ("ECHO_FORM", "ethereal", False), ("APPARITION", "ethereal", False),
    ("CHILL", "exhaust", False),
])
def test_source_upgrade_keywords_are_applied_and_downgrade_restores_base(card_id, keyword, present):
    from sts2sim.engine.transitions import _card_has_keyword, _downgrade_card_instance
    rows = json.loads(Path("data/cache/eng/cards.json").read_text())
    row = next(row for row in rows if row["id"] == card_id)
    card = _card_from_spec({**row, "upgraded": True}, 0)
    assert _card_has_keyword(card, keyword) is present
    if keyword == "exhaust":
        assert card.exhausts is False
    base = _downgrade_card_instance(card)
    assert _card_has_keyword(base, keyword) is not present


def test_damage_upgrade_can_be_downgraded_without_retaining_bonus():
    from sts2sim.engine.transitions import _downgrade_card_instance
    row = {"id": "TEST_STRIKE", "type": "Attack", "cost": 1, "damage": 6,
           "description": "Deal 6 damage.", "upgrade": {"damage": "+3"}}
    card = _card_from_spec({**row, "upgraded": True}, 0)
    base = _downgrade_card_instance(card)
    assert card.effects["sequence"][0]["damage"] == 9
    assert base.effects["sequence"][0]["damage"] == 6


def test_conditional_upgrade_field_does_not_inflate_immediate_base_damage():
    from sts2sim.engine.transitions import _apply_card_upgrade_to_spec
    rows = json.loads(Path("data/cache/eng/cards.json").read_text())
    row = next(row for row in rows if row["id"] == "ASHEN_STRIKE")
    upgraded = _apply_card_upgrade_to_spec(row)
    assert upgraded["damage"] == 6  # ExtraDamage changes per Exhausted card, not base damage.


def test_glitterstream_future_block_upgrade_does_not_apply_to_immediate_block():
    from sts2sim.engine.transitions import _apply_card_upgrade_to_spec
    rows = json.loads(Path("data/cache/eng/cards.json").read_text())
    row = next(row for row in rows if row["id"] == "GLITTERSTREAM")
    assert _apply_card_upgrade_to_spec(row)["block"] == 13
