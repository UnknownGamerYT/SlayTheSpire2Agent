from __future__ import annotations

from sts2sim.mechanics.mechanic_atoms import (
    CARD_SLOT_KEYS,
    card_atom_summary,
    card_slot_summary,
    card_slot_vector,
    card_slots_from_payload,
    status_atom_summary,
    status_atom_vector,
)


def test_chosen_discard_is_distinct_from_random_discard() -> None:
    chosen = card_atom_summary({"effects": {"discard": 1}})
    random = card_atom_summary({"effects": {"random_discard": 1}})

    assert chosen["chosen_discard"] == 1.0
    assert chosen["random_discard"] == 0.0
    assert random["chosen_discard"] == 0.0
    assert random["random_discard"] == 1.0


def test_chosen_transform_is_distinct_from_random_transform() -> None:
    chosen = card_atom_summary({"effects": {"choose_transform": 1}})
    random = card_atom_summary({"effects": {"random_transform": 1}})

    assert chosen["chosen_transform"] == 1.0
    assert chosen["random_transform"] == 0.0
    assert random["chosen_transform"] == 0.0
    assert random["random_transform"] == 1.0


def test_card_keyword_flags_are_exposed_as_atoms() -> None:
    summary = card_atom_summary(
        {
            "unplayable": True,
            "custom": {"innate": True, "eternal": True, "retain": True},
            "tags": ("keyword_ethereal",),
        }
    )

    assert summary["unplayable"] == 1.0
    assert summary["innate"] == 1.0
    assert summary["eternal"] == 1.0
    assert summary["retain"] == 1.0
    assert summary["ethereal"] == 1.0


def test_status_vector_has_explicit_status_values() -> None:
    summary = status_atom_summary(
        {
            "poison": 7,
            "artifact": 2,
            "intangible": 1,
            "strength": -3,
        }
    )
    vector = status_atom_vector(summary)

    assert summary["poison"] == 7.0
    assert summary["artifact"] == 2.0
    assert summary["intangible"] == 1.0
    assert summary["strength"] == -3.0
    assert vector[0] == -3.0
    assert vector[3] == 7.0
    assert vector[7] == 1.0
    assert vector[11] == 2.0


def test_card_slot_vector_length_equals_schema() -> None:
    slot = card_slot_summary(
        {
            "card_id": "bash",
            "type": "attack",
            "cost": 2,
            "upgraded": True,
            "effects": {"damage": 8, "apply_status": {"vulnerable": 2}},
        },
        zone="hand",
        position=3,
    )
    vector = card_slot_vector(slot)

    assert len(vector) == len(CARD_SLOT_KEYS)
    assert slot["card_type_id"] == 1.0
    assert slot["zone_id"] == 2.0
    assert slot["position"] == 3.0
    assert slot["damage"] == 8.0
    assert slot["apply_vulnerable"] == 2.0


def test_card_slots_from_payload_reads_master_deck_and_combat_zones() -> None:
    payload = {
        "master_deck": (
            {"card_id": "strike", "type": "attack", "effects": {"damage": 6}},
            {"card_id": "defend", "type": "skill", "effects": {"block": 5}},
        ),
        "combat": {
            "hand": ({"card_id": "bash", "type": "attack", "effects": {"damage": 8}},),
            "draw_pile": ({"card_id": "shrug", "type": "skill", "effects": {"draw": 1}},),
        },
    }

    deck_slots = card_slots_from_payload(payload, zone="master_deck", limit=1)
    combat_slots = card_slots_from_payload(payload, zone="combat", limit=2)

    assert [slot["position"] for slot in deck_slots] == [0.0]
    assert [slot["zone_id"] for slot in combat_slots] == [2.0, 3.0]
    assert combat_slots[1]["draw_now"] == 1.0
