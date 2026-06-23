"""Per-option reward and shop visibility for learning agents.

The functions in this module expose option mechanics as facts.  They do not
encode preferences such as "take this card"; they split visible choices into
fixed-schema slots so the policy does not have to infer mechanics from a single
bundle count.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sts2sim.mechanics.planning_context import reward_plan_summary
from sts2sim.mechanics.semantics import action_mechanic_profile
from sts2sim.mechanics.synergy import action_synergy_profile

OPTION_SLOT_KEYS: tuple[str, ...] = (
    "present",
    "option_kind_id",
    "source_kind_id",
    "rarity_id",
    "price",
    "affordable",
    "position",
    "selection_set_size",
    "group_size",
    "leaves_other_choices_open",
    "skip_action",
    "forced",
    "card_gain",
    "relic_gain",
    "potion_gain",
    "gold_gain",
    "card_remove",
    "damage",
    "block",
    "draw",
    "energy",
    "scaling",
    "poison",
    "exhaust",
    "retain",
    "discard_payoff",
    "summon",
    "orb",
    "forge",
    "max_hp_payoff",
    "potion_slot_pressure",
    "gold_pressure",
    "improves_current_need",
    "duplicates_existing_engine",
    "adds_bloat",
    "enables_combo",
    "conflicts_with_plan",
)

OPTION_KIND_IDS: dict[str, int] = {
    "": 0,
    "card": 1,
    "fixed_card": 2,
    "card_group": 3,
    "relic": 4,
    "potion": 5,
    "gold": 6,
    "card_removal": 7,
    "skip": 8,
    "proceed": 9,
}

SOURCE_KIND_IDS: dict[str, int] = {
    "": 0,
    "reward": 1,
    "shop": 2,
    "combat": 3,
    "event": 4,
    "treasure": 5,
    "ancient": 6,
    "other": 7,
}

RARITY_IDS: dict[str, int] = {
    "": 0,
    "starter": 1,
    "basic": 2,
    "common": 3,
    "uncommon": 4,
    "rare": 5,
    "shop": 6,
    "event": 7,
    "boss": 8,
    "ancient": 9,
}


def reward_option_slots(
    payload: Mapping[str, Any],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Return separate visible reward choices plus skip/proceed controls."""

    reward = _mapping(payload.get("reward"))
    if not reward:
        return []
    plan = reward_plan_summary(payload)
    choices = [
        _reward_slot(payload, reward, plan, _mapping(choice))
        for choice in _sequence(plan.get("available_choices"))
    ]
    if bool(plan.get("can_skip")):
        choices.extend(_reward_skip_slots(payload, reward, plan))
        choices.append(_reward_proceed_slot(payload, reward, plan))
    return choices[: max(0, limit)]


def shop_option_slots(
    payload: Mapping[str, Any],
    *,
    limit: int = 16,
) -> list[dict[str, Any]]:
    """Return separate visible shop item slots."""

    shop = _mapping(payload.get("shop"))
    player_gold = _player_gold(payload)
    slots: list[dict[str, Any]] = []
    for position, raw_item in enumerate(_sequence(shop.get("items"))):
        item = _mapping(raw_item)
        if not item or bool(item.get("purchased", False)):
            continue
        kind = str(item.get("kind", ""))
        content_id = str(item.get("item_id", ""))
        descriptor = _shop_descriptor(item)
        slot = _slot_from_descriptor(
            payload,
            descriptor,
            kind=kind,
            content_id=content_id,
            source="shop",
            position=_int(item.get("slot_index", position)),
            price=_float(item.get("price")),
            rarity=str(item.get("rarity") or ""),
            forced=False,
            skip_action=False,
            leaves_other_choices_open=True,
            selection_set_size=1,
            group_size=1,
        )
        slot["affordable"] = player_gold >= _float(slot["price"])
        slots.append(slot)
    return slots[: max(0, limit)]


def option_slot_vector(slot: Mapping[str, Any]) -> tuple[float, ...]:
    """Return a stable vector ordered by OPTION_SLOT_KEYS."""

    values = _mapping(slot.get("values"))
    flattened = {
        "present": slot.get("present"),
        "option_kind_id": OPTION_KIND_IDS.get(str(slot.get("kind", "")), 0),
        "source_kind_id": SOURCE_KIND_IDS.get(str(slot.get("source", "")), 0),
        "rarity_id": RARITY_IDS.get(str(slot.get("rarity", "")), 0),
        "price": slot.get("price"),
        "affordable": slot.get("affordable"),
        "position": slot.get("position"),
        "selection_set_size": slot.get("selection_set_size"),
        "group_size": slot.get("group_size"),
        "leaves_other_choices_open": slot.get("leaves_other_choices_open"),
        "skip_action": slot.get("skip_action"),
        "forced": slot.get("forced"),
        **values,
    }
    return tuple(_number(flattened.get(key)) for key in OPTION_SLOT_KEYS)


def option_slots_vector(
    payload: Mapping[str, Any],
    *,
    reward_limit: int = 12,
    shop_limit: int = 16,
) -> tuple[float, ...]:
    """Return flattened fixed-size reward and shop option slots."""

    slots = [
        *reward_option_slots(payload, limit=reward_limit),
        *_pad_slots(reward_option_slots(payload, limit=reward_limit), reward_limit),
        *shop_option_slots(payload, limit=shop_limit),
        *_pad_slots(shop_option_slots(payload, limit=shop_limit), shop_limit),
    ]
    return tuple(value for slot in slots for value in option_slot_vector(slot))


def _reward_slot(
    payload: Mapping[str, Any],
    reward: Mapping[str, Any],
    plan: Mapping[str, Any],
    choice: Mapping[str, Any],
) -> dict[str, Any]:
    kind = str(choice.get("kind", ""))
    content_id = str(choice.get("content_id", ""))
    descriptor = _reward_descriptor(choice)
    available_sets = _available_selection_set_count(plan)
    leaves_open = available_sets > 1 or not bool(choice.get("exclusive_within_set"))
    return _slot_from_descriptor(
        payload,
        descriptor,
        kind=kind,
        content_id=content_id,
        source=str(reward.get("source", "reward")),
        position=_int(choice.get("position")),
        price=0.0,
        rarity=str(choice.get("rarity") or ""),
        forced=bool(reward.get("forced", False)),
        skip_action=False,
        leaves_other_choices_open=leaves_open,
        selection_set_size=_int(choice.get("selection_set_size"), 1),
        group_size=_int(choice.get("selection_set_size"), 1),
    )


def _reward_skip_slots(
    payload: Mapping[str, Any],
    reward: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    selection_sets = _sequence(plan.get("selection_sets"))
    for position, raw_set in enumerate(selection_sets):
        selection_set = _mapping(raw_set)
        if _int(selection_set.get("available_count")) <= 0:
            continue
        if str(selection_set.get("selection_set_id", "")) == "card_removal":
            # Optional reward removals are declined through the shared proceed
            # action; there is no engine skip action for this selection set.
            continue
        descriptor = {
            "type": "skip_reward",
            "reward_choice": {
                "kind": "skip",
                "skip_kind": str(selection_set.get("selection_set_id", "")),
                "content_id": "",
            },
        }
        slots.append(
            _slot_from_descriptor(
                payload,
                descriptor,
                kind="skip",
                content_id=str(selection_set.get("selection_set_id", "")),
                source=str(reward.get("source", "reward")),
                position=position,
                price=0.0,
                rarity="",
                forced=False,
                skip_action=True,
                leaves_other_choices_open=len(selection_sets) > 1,
                selection_set_size=_int(selection_set.get("selection_set_size"), 1),
                group_size=_int(selection_set.get("available_count"), 1),
            )
        )
    return slots


def _reward_proceed_slot(
    payload: Mapping[str, Any],
    reward: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    descriptor = {
        "type": "proceed",
        "reward_choice": {
            "kind": "proceed",
            "content_id": "",
            "available_remaining_count": _int(
                _mapping(plan.get("available_counts")).get("total")
            ),
        },
    }
    return _slot_from_descriptor(
        payload,
        descriptor,
        kind="proceed",
        content_id="proceed",
        source=str(reward.get("source", "reward")),
        position=len(_sequence(plan.get("selection_sets"))),
        price=0.0,
        rarity="",
        forced=False,
        skip_action=True,
        leaves_other_choices_open=False,
        selection_set_size=1,
        group_size=1,
    )


def _slot_from_descriptor(
    payload: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    *,
    kind: str,
    content_id: str,
    source: str,
    position: int,
    price: float,
    rarity: str,
    forced: bool,
    skip_action: bool,
    leaves_other_choices_open: bool,
    selection_set_size: int,
    group_size: int,
) -> dict[str, Any]:
    mechanics = action_mechanic_profile(descriptor)
    synergy = action_synergy_profile(payload, {**descriptor, "mechanics": mechanics})
    values = _option_values(mechanics, synergy, kind=kind, descriptor=descriptor)
    return {
        "present": True,
        "kind": kind,
        "source": source,
        "content_id": content_id,
        "rarity": rarity,
        "price": price,
        "affordable": True,
        "position": position,
        "selection_set_size": selection_set_size,
        "group_size": group_size,
        "leaves_other_choices_open": leaves_other_choices_open,
        "skip_action": skip_action,
        "forced": forced,
        "descriptor": dict(descriptor),
        "values": values,
        "mechanics": mechanics,
        "synergy": synergy,
    }


def _option_values(
    mechanics: Mapping[str, Any],
    synergy: Mapping[str, Any],
    *,
    kind: str,
    descriptor: Mapping[str, Any],
) -> dict[str, float]:
    mechanic_values = _mapping(mechanics.get("values"))
    synergy_values = _mapping(synergy.get("values"))
    values = {
        "card_gain": _float(mechanic_values.get("card_gain")),
        "relic_gain": _float(mechanic_values.get("relic_gain")),
        "potion_gain": _float(mechanic_values.get("potion_gain")),
        "gold_gain": max(0.0, _float(mechanic_values.get("gold_delta"))),
        "card_remove": _float(mechanic_values.get("card_remove")),
        "damage": _float(mechanic_values.get("damage"))
        + _float(mechanic_values.get("aoe_damage")),
        "block": _float(mechanic_values.get("block")),
        "draw": _float(mechanic_values.get("draw")),
        "energy": _float(mechanic_values.get("energy")),
        "scaling": _scaling_value(mechanic_values),
        "poison": _float(mechanic_values.get("poison")),
        "exhaust": _float(mechanic_values.get("exhaust")),
        "retain": _float(mechanic_values.get("retain")),
        "discard_payoff": _float(synergy_values.get("discard_payoff")),
        "summon": _float(mechanic_values.get("summon")),
        "orb": _float(mechanic_values.get("orb_channel"))
        + _float(mechanic_values.get("orb_evoke")),
        "forge": _float(mechanic_values.get("forge")),
        "max_hp_payoff": max(
            _float(synergy_values.get("max_hp_payoff")),
            _float(mechanic_values.get("max_hp_delta")),
        ),
        "potion_slot_pressure": _float(synergy_values.get("potion_slot_pressure")),
        "gold_pressure": _float(synergy_values.get("gold_pressure")),
        "improves_current_need": _float(synergy_values.get("improves_current_need")),
        "duplicates_existing_engine": _float(
            synergy_values.get("duplicates_existing_engine")
        ),
        "adds_bloat": _float(synergy_values.get("adds_bloat")),
        "enables_combo": _float(synergy_values.get("enables_combo")),
        "conflicts_with_plan": _float(synergy_values.get("conflicts_with_plan")),
    }
    if kind == "card_removal":
        values["card_remove"] = max(1.0, values["card_remove"])
    if kind == "gold":
        reward_choice = _mapping(descriptor.get("reward_choice"))
        values["gold_gain"] = max(values["gold_gain"], _float(reward_choice.get("amount")))
    return values


def _reward_descriptor(choice: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(choice.get("kind", ""))
    content_id = str(choice.get("content_id", ""))
    reward_choice = {
        "kind": "card" if kind in {"card", "fixed_card", "card_group"} else kind,
        "content_id": content_id,
        "source": "reward",
        "position": _int(choice.get("position")),
        "selection_set_id": str(choice.get("selection_set_id", "")),
        "selection_set_size": _int(choice.get("selection_set_size"), 1),
        "amount": _int(choice.get("amount")),
    }
    descriptor: dict[str, Any] = {
        "type": f"take_reward_{reward_choice['kind']}",
        "reward_choice": reward_choice,
    }
    if reward_choice["kind"] == "card":
        descriptor["card"] = {"card_id": content_id, "zone": "reward"}
    elif reward_choice["kind"] == "relic":
        descriptor["relic"] = {"relic_id": content_id}
    elif reward_choice["kind"] == "potion":
        descriptor["potion"] = {"potion_id": content_id}
    return descriptor


def _shop_descriptor(item: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(item.get("kind", ""))
    item_id = str(item.get("item_id", ""))
    descriptor: dict[str, Any] = {
        "type": "shop_buy",
        "item": dict(item),
    }
    if kind in {"card", "colorless_card"}:
        descriptor["card"] = {"card_id": item_id, "zone": "shop"}
    elif kind == "relic":
        descriptor["relic"] = {"relic_id": item_id}
    elif kind == "potion":
        descriptor["potion"] = {"potion_id": item_id}
    elif kind == "card_removal":
        descriptor["type"] = "shop_remove_card"
    return descriptor


def _pad_slots(slots: Sequence[Mapping[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [_empty_slot() for _index in range(max(0, limit - len(slots)))]


def _empty_slot() -> dict[str, Any]:
    return {
        "present": False,
        "kind": "",
        "source": "",
        "content_id": "",
        "rarity": "",
        "price": 0.0,
        "affordable": False,
        "position": 0,
        "selection_set_size": 0,
        "group_size": 0,
        "leaves_other_choices_open": False,
        "skip_action": False,
        "forced": False,
        "values": {key: 0.0 for key in OPTION_SLOT_KEYS},
    }


def _available_selection_set_count(plan: Mapping[str, Any]) -> int:
    return sum(
        1
        for raw_set in _sequence(plan.get("selection_sets"))
        if _int(_mapping(raw_set).get("available_count")) > 0
    )


def _scaling_value(values: Mapping[str, Any]) -> float:
    return (
        _float(values.get("strength"))
        + _float(values.get("dexterity"))
        + _float(values.get("focus"))
        + _float(values.get("orb_slot_delta"))
        + _float(values.get("repeating_effect"))
        + _float(values.get("periodic_effect"))
    )


def _player_gold(payload: Mapping[str, Any]) -> float:
    return _float(_mapping(payload.get("player")).get("gold"))


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _sequence(value: object) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(value)
    return ()


def _number(value: object) -> float:
    if isinstance(value, bool):
        return float(int(value))
    return _float(value)


def _float(value: object) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default
