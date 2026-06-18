from __future__ import annotations

from sts2sim.mechanics import (
    has_open_potion_slot,
    normalize_potion_use,
    potion_capacity,
    potion_discard_decision,
    potion_pickup_decision,
)


def test_potion_capacity_uses_ascension_and_slot_relics() -> None:
    plain = potion_capacity(
        base_slots=3,
        ascension_level=11,
        current_potions=("FIRE_POTION", "SKILL_POTION"),
    )
    belted = potion_capacity(
        base_slots=3,
        ascension_level=11,
        relics=("POTION_BELT",),
        current_potions=("FIRE_POTION", "SKILL_POTION"),
    )
    coffer = potion_capacity(base_slots=3, relics=({"id": "ALCHEMICAL_COFFER"},))
    holster = potion_capacity(base_slots=3, relics=({"id": "PHIAL_HOLSTER"},))
    buckled = potion_capacity(base_slots=3, relics=({"id": "BELT_BUCKLE"},))

    assert plain.capacity == 2
    assert plain.open_slots == 0
    assert belted.capacity == 4
    assert belted.open_slots == 2
    assert coffer.capacity == 7
    assert holster.capacity == 4
    assert buckled.capacity == 3


def test_pickup_and_discard_decisions_are_bounded_and_pure() -> None:
    potions = ("FIRE_POTION", "SKILL_POTION", "FOUL_POTION")
    blocked = potion_pickup_decision(potions, "BLOCK_POTION")

    assert blocked.can_pick_up is False
    assert blocked.requires_discard is True
    assert blocked.next_potions == ("fire_potion", "skill_potion", "foul_potion")

    discarded = potion_discard_decision(blocked.next_potions, 1)
    assert discarded.can_discard is True
    assert discarded.discarded_potion_id == "skill_potion"
    assert discarded.next_potions == ("fire_potion", "foul_potion")

    picked = potion_pickup_decision(discarded.next_potions, {"id": "BLOCK_POTION"})
    assert picked.can_pick_up is True
    assert picked.next_potions == ("fire_potion", "foul_potion", "block_potion")


def test_has_open_potion_slot_respects_capacity_relics() -> None:
    potions = ("fire_potion", "skill_potion", "foul_potion")

    assert has_open_potion_slot(potions) is False
    assert has_open_potion_slot(potions, relics=("potion_belt",)) is True


def test_foul_potion_merchant_throw_normalizes_to_gold_marker() -> None:
    use = normalize_potion_use("FOUL_POTION", target_id="merchant")

    assert use.mode == "merchant_throw"
    assert use.consumes_potion is True
    assert use.effects[0].kind == "merchant_gold"
    assert use.effects[0].target == "merchant"
    assert use.effects[0].amount == 100


def test_basic_potion_use_normalization_for_common_categories() -> None:
    fire = normalize_potion_use("FIRE_POTION", target_id="jaw_worm")
    block = normalize_potion_use({"id": "BLOCK_POTION", "name": "Block Potion"})
    strength = normalize_potion_use("STRENGTH_POTION")
    dexterity = normalize_potion_use("DEXTERITY_POTION")
    weak = normalize_potion_use("WEAK_POTION", target_id="cultist")
    energy = normalize_potion_use("ENERGY_POTION")
    swift = normalize_potion_use("SWIFT_POTION")
    cure_all = normalize_potion_use("CURE_ALL")

    assert fire.effects[0].kind == "damage"
    assert fire.effects[0].amount == 20
    assert fire.effects[0].target_id == "jaw_worm"
    assert block.effects[0].kind == "block"
    assert block.effects[0].amount == 12
    assert strength.effects[0].status == "strength"
    assert dexterity.effects[0].status == "dexterity"
    assert weak.effects[0].status == "weak"
    assert weak.effects[0].target_id == "cultist"
    assert energy.effects[0].kind == "energy"
    assert energy.effects[0].amount == 2
    assert swift.effects[0].kind == "draw"
    assert swift.effects[0].amount == 3
    assert [effect.kind for effect in cure_all.effects] == ["energy", "draw"]
    assert [effect.amount for effect in cure_all.effects] == [1, 2]


def test_orb_potion_use_normalization() -> None:
    capacity = normalize_potion_use("POTION_OF_CAPACITY")
    darkness = normalize_potion_use("ESSENCE_OF_DARKNESS")

    assert capacity.effects[0].kind == "orb_slot_delta"
    assert capacity.effects[0].amount == 2
    assert darkness.effects[0].kind == "channel_orb"
    assert darkness.effects[0].amount is None
    assert darkness.effects[0].metadata == {"orb": "dark", "amount": "orb_slots"}


def test_codex_potion_spec_can_be_normalized_without_explicit_id_mapping() -> None:
    use = normalize_potion_use(
        {
            "id": "CUSTOM_TEST_POTION",
            "description": "Gain [energy:1]. Draw [blue]2[/blue] cards.",
        }
    )

    assert use.unsupported is False
    assert [(effect.kind, effect.amount) for effect in use.effects] == [
        ("energy", 1),
        ("draw", 2),
    ]


def test_codex_orb_potion_spec_can_be_normalized_without_explicit_mapping() -> None:
    capacity = normalize_potion_use(
        {
            "id": "CUSTOM_CAPACITY_POTION",
            "description": "Gain [blue]2[/blue] [gold]Orb Slots[/gold].",
        }
    )
    darkness = normalize_potion_use(
        {
            "id": "CUSTOM_DARKNESS_POTION",
            "description": (
                "[gold]Channel[/gold] a [gold]Dark[/gold] for each of your "
                "[gold]Orb Slots[/gold]."
            ),
        }
    )

    assert capacity.unsupported is False
    assert [(effect.kind, effect.amount) for effect in capacity.effects] == [
        ("orb_slot_delta", 2)
    ]
    assert darkness.unsupported is False
    assert darkness.effects[0].kind == "channel_orb"
    assert darkness.effects[0].metadata == {"orb": "dark", "amount": "orb_slots"}
