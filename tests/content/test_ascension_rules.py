from __future__ import annotations

import pytest

from sts2sim.mechanics.ascension import (
    ascension_economy_modifiers,
    card_removal_price_at_ascension,
    potion_slots_at_ascension,
    rarity_scarcity_at_ascension,
    rest_site_heal_fraction_at_ascension,
    reward_gold_multiplier_at_ascension,
    validate_ascension_level,
)


def test_a0_economy_modifiers_are_baseline() -> None:
    modifiers = ascension_economy_modifiers(0)

    assert modifiers.potion_slots == 3
    assert modifiers.card_removal_base_price == 75
    assert modifiers.card_removal_increment == 25
    assert modifiers.rest_heal_fraction == 0.30
    assert modifiers.reward_gold_multiplier == 1.0
    assert modifiers.rarity_scarcity_enabled is False
    assert modifiers.ascender_curse_enabled is False


def test_a0_to_a10_non_combat_rule_boundaries() -> None:
    assert reward_gold_multiplier_at_ascension(2) == 1.0
    assert reward_gold_multiplier_at_ascension(3) == 0.75
    assert rest_site_heal_fraction_at_ascension(4) == 0.30
    assert rest_site_heal_fraction_at_ascension(5) == 0.20
    assert card_removal_price_at_ascension(5, card_removals_bought=2) == 125
    assert card_removal_price_at_ascension(6, card_removals_bought=2) == 200
    assert rarity_scarcity_at_ascension(6) is False
    assert rarity_scarcity_at_ascension(7) is True
    assert ascension_economy_modifiers(9).ascender_curse_enabled is False
    assert ascension_economy_modifiers(10).ascender_curse_enabled is True


def test_potion_slot_reduction_starts_after_a10() -> None:
    assert potion_slots_at_ascension(10) == 3
    assert potion_slots_at_ascension(11) == 2
    assert potion_slots_at_ascension(11, base_slots=1) == 0


def test_ascension_validation_is_shared_by_rule_helpers() -> None:
    with pytest.raises(ValueError, match="between 0 and"):
        validate_ascension_level(21)

    with pytest.raises(ValueError, match="between 0 and"):
        potion_slots_at_ascension(-1)
