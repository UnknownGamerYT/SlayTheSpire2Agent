"""Stable powered attack modifiers for Osty's dealer and card owner's hooks."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from . import source_enchantment_hooks
from .models import CardInstance, CombatState


def damage_amount(
    combat: CombatState, card: CardInstance, osty: Mapping[str, Any], base: int,
    *, relics: Sequence[str] = (), target_statuses: Mapping[str, int] | None = None,
) -> Fraction:
    from . import transitions as t

    dealer = {str(key): int(value) for key, value in t._mapping_from(
        osty.get("statuses")
    ).items()}
    value = source_enchantment_hooks.damage_base(combat, card, base)
    value += t._signed_status_amount(dealer, "strength")
    value += t._signed_status_amount(dealer, "temporary_strength")
    value += t._status_amount(dealer, "vigor")
    value += t._status_amount(target_statuses or {}, "tainted")
    if card.custom.get("source_core_inky"):
        value += 1
    if t._has_relic_id(relics, "strike_dummy") and t._has_source_card_tag(card, "strike"):
        value += 3
    if t._has_relic_id(relics, "fake_strike_dummy") and t._has_source_card_tag(card, "strike"):
        value += 1
    if t._has_relic_id(relics, "miniature_cannon") and card.upgraded:
        value += 3
    if t._has_relic_id(relics, "mystic_lighter") and card.enchantments:
        value += 9
    if t._has_source_card_tag(card, "shiv"):
        value += t._status_amount(dealer, "accuracy")
    lethality = t._coerce_int(card.custom.get("lethality_damage_percent"), default=0)
    value *= Fraction(100 + lethality, 100)
    if t._status_amount(dealer, "weak"):
        value *= Fraction(1, 2) if t._status_amount(dealer, "debilitate") else Fraction(3, 4)
    if t._signed_status_amount(dealer, "shrink"):
        value *= Fraction(7, 10)
    if t._status_amount(combat.player.statuses, "source_core_double_damage"):
        value *= 2
    if t._status_amount(dealer, "source_core_double_damage"):
        value *= 2
    value *= t._queued_damage_multiplier(combat, card)
    if t._has_relic_id(relics, "vitruvian_minion") and t._has_source_card_tag(card, "minion"):
        value *= 2
    return value


def receiver_attacker_statuses(combat: CombatState, osty: Mapping[str, Any]) -> dict[str, int]:
    """Tracking permits owner pets; Vulnerable's Cruelty modifier belongs to the dealer."""
    from . import transitions as t

    dealer = {str(key): int(value) for key, value in t._mapping_from(
        osty.get("statuses")
    ).items()}
    tracking = t._status_amount(combat.player.statuses, "tracking")
    pet_tracking = t._status_amount(dealer, "tracking")
    if tracking or pet_tracking:
        dealer["tracking"] = max(1, tracking) * max(1, pet_tracking)
    return dealer
