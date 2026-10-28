"""Ordered local energy modifiers from stable0.107.1 CardEnergyCost.cs.

Independent implementation against Razhuo/sts2-rl-agent
5d874e855ae1d0550d5c30416fadd7dbb0d133b7. LocalCostModifier.Modify
preserves insertion order; expiry removes matching entries without restoring
an obsolete snapshot over a later modifier. Star costs have their own scope.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .models import CardInstance

_KEY = "source_local_energy_modifiers"
_BASE = "source_local_energy_base"
_PLAYED = 1
_TURN = 2
_COMBAT = 4
_EXPIRY = {"played": _PLAYED, "turn": _TURN,
           "turn_or_played": _TURN | _PLAYED, "combat": _COMBAT}


def _mods(card: CardInstance) -> list[dict[str, Any]]:
    values = card.custom.get(_KEY, ())
    return [dict(value) for value in values if isinstance(value, Mapping)] if isinstance(
        values, Sequence
    ) else []


def base_cost(card: CardInstance) -> int | None:
    value = card.custom.get(_BASE, card.cost)
    return int(value) if value is not None else None


def _entry(amount: int, absolute: bool, expiration: str, reduce_only: bool) -> dict[str, Any]:
    return {"amount": amount, "absolute": absolute, "expiration": _EXPIRY[expiration],
            "reduce_only": reduce_only}


def adopt_legacy(card: CardInstance) -> CardInstance:
    """Read previous schema markers; all new mutations then append ordered entries."""
    if _KEY in card.custom:
        return card
    custom = dict(card.custom)
    base = custom.get("cost_before_set_hand_cost", custom.get("cost_before_randomize", card.cost))
    mods: list[dict[str, Any]] = []
    if custom.get("free_to_play_this_combat"):
        mods.append(_entry(0, True, "combat", False))
    if custom.get("free_to_play_this_turn") and not custom.get("free_energy_consumed"):
        mods.append(_entry(0, True, "turn_or_played", False))
    if "source_enchantment_slither_cost" in custom:
        mods.append(_entry(int(custom["source_enchantment_slither_cost"]), True, "combat", False))
    if "confused_cost_this_combat" in custom:
        mods.append(_entry(int(custom["confused_cost_this_combat"]), True, "combat", False))
    if "randomized_cost_this_turn" in custom:
        mods.append(_entry(int(custom["randomized_cost_this_turn"]), True, "turn_or_played", False))
    if "cost_before_set_hand_cost" in custom:
        duration = "turn_or_played" if custom.get("set_hand_cost_duration") == "turn" else "combat"
        mods.append(_entry(card.cost or 0, True, duration, False))
    if custom.get("source_enchantment_slumber_discount"):
        mods.append(_entry(-int(custom["source_enchantment_slumber_discount"]), False,
                           "played", False))
    custom.update({_BASE: base, _KEY: mods})
    return _display(card.model_copy(update={"custom": custom}))


def local_cost(card: CardInstance) -> int:
    card = adopt_legacy(card)
    value = base_cost(card) or 0
    if value < 0:
        return value
    for modifier in _mods(card):
        amount = int(modifier.get("amount", 0))
        updated = amount if modifier.get("absolute") else value + amount
        value = min(value, updated) if modifier.get("reduce_only") else updated
    return value


def _display(card: CardInstance) -> CardInstance:
    base = base_cost(card)
    if base is None or base < 0 or card.custom.get("x_cost"):
        return card.model_copy(update={"cost": base})
    return card.model_copy(update={"cost": max(0, local_cost(card))})


def _append(
    card: CardInstance, amount: int, expiration: str, *, absolute: bool, reduce_only: bool,
) -> CardInstance:
    card = adopt_legacy(card)
    base = base_cost(card)
    # Source Set*(0) skips a genuinely unplayable negative canonical cost.
    # X cards still collect modifiers, while GetAmountToSpend ignores them.
    if absolute and amount == 0 and base is not None and base < 0 and not card.custom.get("x_cost"):
        return card
    modifiers = _mods(card)
    modifiers.append(_entry(amount, absolute, expiration, reduce_only))
    return _display(card.model_copy(update={"custom": {**card.custom, _KEY: modifiers}}))


def set_local_energy_cost(
    card: CardInstance, amount: int, expiration: str, *, reduce_only: bool = False,
) -> CardInstance:
    return _append(card, amount, expiration, absolute=True, reduce_only=reduce_only)


def add_local_energy_cost(
    card: CardInstance, amount: int, expiration: str, *, reduce_only: bool = False,
) -> CardInstance:
    if not amount:
        return card
    return _append(card, amount, expiration, absolute=False, reduce_only=reduce_only)


def free_to_play(card: CardInstance, *, combat: bool = False) -> CardInstance:
    card = set_local_energy_cost(card, 0, "combat" if combat else "turn_or_played")
    flag = "free_to_play_this_combat" if combat else "free_to_play_this_turn"
    custom = {**card.custom, flag: True}
    if not combat:
        custom.pop("free_energy_consumed", None)
    return card.model_copy(update={"custom": custom})


def expire(
    card: CardInstance, *, when_played: bool = False, turn_end: bool = False,
) -> CardInstance:
    card = adopt_legacy(card)
    mask = (_PLAYED if when_played else 0) | (_TURN if turn_end else 0)
    custom = dict(card.custom)
    custom[_KEY] = [m for m in _mods(card) if not int(m.get("expiration", 0)) & mask]
    if when_played and custom.get("free_to_play_this_turn"):
        custom["free_energy_consumed"] = True
    if turn_end:
        custom.pop("free_to_play_this_turn", None)
        custom.pop("free_energy_consumed", None)
    for key in ("cost_before_randomize", "randomized_cost_this_turn",
                "cost_before_set_hand_cost", "set_hand_cost_duration"):
        custom.pop(key, None)
    if when_played:
        custom.pop("source_enchantment_slumber_discount", None)
    return _display(card.model_copy(update={"custom": custom}))


def preserve_changes(card: CardInstance, updated: CardInstance) -> CardInstance:
    if _KEY not in card.custom:
        return updated
    modifiers = _mods(card)
    old_base = base_cost(card)
    new_base = updated.cost
    if (not card.upgraded and updated.upgraded and old_base is not None
            and new_base is not None and new_base < old_base):
        for modifier in modifiers:
            if modifier.get("absolute") and int(modifier.get("amount", 0)) > new_base:
                modifier["amount"] = new_base
    return _display(updated.model_copy(update={"custom": {
        **updated.custom, _BASE: new_base, _KEY: modifiers,
        **{key: value for key, value in card.custom.items() if key in {
            "free_to_play_this_turn", "free_to_play_this_combat", "free_energy_consumed",
        }},
    }}))
