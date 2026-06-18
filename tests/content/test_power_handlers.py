from __future__ import annotations

from sts2sim.mechanics.powers import (
    apply_power_modifiers_to_effect,
    apply_status_delta,
    end_of_combat_status_events,
    modified_card_cost,
    modifier_snapshot_from_statuses,
    normalize_statuses,
)


def test_status_normalization_builds_modifier_snapshot() -> None:
    statuses = [
        {"power": "Strength", "amount": 2},
        {"name": "Temporary Dexterity", "type": "Buff"},
        {"Weak": 1, "Vulnerable": 2},
    ]

    assert normalize_statuses(statuses) == {
        "strength": 2,
        "temporary_dexterity": 1,
        "weak": 1,
        "vulnerable": 2,
    }
    snapshot = modifier_snapshot_from_statuses(statuses)
    assert snapshot.strength == 2
    assert snapshot.dexterity == 1
    assert snapshot.weak is True
    assert snapshot.vulnerable is True


def test_power_modifiers_adjust_damage_and_block_steps() -> None:
    result = apply_power_modifiers_to_effect(
        {"damage": 10, "block": 5},
        actor_statuses={"strength": 2, "weak": 1, "dexterity": 1},
        defender_statuses={"vulnerable": 1},
        card_type="attack",
        source_id="test_card",
    )

    assert result.effect == {"damage": 13, "block": 6}
    assert [event["metadata"]["field"] for event in result.events] == ["damage", "block"]


def test_modified_card_cost_applies_bounded_type_reductions() -> None:
    assert modified_card_cost({"cost": 3, "type": "Power"}, {"power_cost_reduction": 1}) == 2
    assert (
        modified_card_cost(
            {"cost": 2, "type": "Attack"},
            {"card_cost_reduction": 1, "attack_cost_reduction": 2},
        )
        == 0
    )
    assert modified_card_cost({"cost": "X", "type": "Skill"}, available_energy=3) == 3


def test_status_delta_and_end_of_combat_markers_emit_events() -> None:
    result = apply_status_delta(
        {"Weak": 1},
        {"Weak": 2, "Strength": -1},
        source_id="bash",
        target_id="m1",
    )

    assert result.statuses == {"strength": -1, "weak": 3}
    assert result.events == (
        {
            "kind": "status_applied",
            "source_id": "bash",
            "target_id": "m1",
            "amount": -1,
            "metadata": {"status": "strength"},
        },
        {
            "kind": "status_applied",
            "source_id": "bash",
            "target_id": "m1",
            "amount": 2,
            "metadata": {"status": "weak"},
        },
    )

    assert end_of_combat_status_events({"upgrade_random_end_of_combat": 2}) == (
        {
            "kind": "card_upgrade_random_pending",
            "source_id": None,
            "target_id": "player",
            "amount": 2,
            "metadata": {"status": "end_of_combat_upgrade_random"},
        },
    )
