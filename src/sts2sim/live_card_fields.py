"""Source-defined card fields shared by visible live import and parity capture.

No game objects are reconstructed here: values must be supplied by the bridge.
The source has one enchantment and one affliction slot per physical card.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from sts2sim.content.source_card_pools import NON_UPGRADABLE_CARD_IDS
from sts2sim.engine import source_cost_hooks, source_enchantment_hooks, source_modifier_hooks
from sts2sim.engine.models import CardInstance

_KEYWORDS = frozenset(
    {"none", "exhaust", "ethereal", "innate", "unplayable", "retain", "sly", "eternal"}
)
_AFFLICTIONS = frozenset(
    {"bound", "entangled", "galvanized", "hexed", "ringing", "smog", "tainted"}
)


def integer(value: Any, field: str, *, minimum: int | None = 0) -> int:
    try:
        if isinstance(value, bool) or value is None or float(value) != int(value):
            raise ValueError
        result = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Missing or invalid live {field} integer") from exc
    if minimum is not None and result < minimum:
        raise ValueError(f"Live {field} is below {minimum}")
    return result


def rows(value: Any, field: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list | tuple) or not all(isinstance(v, Mapping) for v in value):
        raise ValueError(f"Live {field} must be a list of objects")
    return tuple(value)


def _slot(row: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    singular = row.get(name)
    plural = row.get(name + "s")
    if singular is not None and not isinstance(singular, Mapping):
        raise ValueError(f"Live {name} is not an identity/amount object")
    if plural is not None:
        values = rows(plural, name + "s")
        if len(values) > 1:
            raise ValueError(f"Source cards have only one {name} slot")
        if values:
            if singular is not None and dict(singular) != dict(values[0]):
                raise ValueError(f"Conflicting live {name} fields")
            singular = values[0]
    present = row.get("has_" + name)
    if present is not None and not isinstance(present, bool):
        raise ValueError(f"Invalid live has_{name} flag")
    if present is True and not singular:
        raise ValueError(f"Live {name} identity/amount cannot be losslessly imported")
    if present is False and singular:
        raise ValueError(f"Live has_{name} contradicts its payload")
    return singular


def _expiration(value: Any) -> int:
    # Source LocalCostModifierExpiration is 0=Combat,2=Turn,4=Played;
    # the simulator's saved schema is independently 4=Combat,2=Turn,1=Played.
    raw = integer(value, "local cost expiration")
    if raw & ~6:
        raise ValueError("Unknown source local-cost expiration flags")
    return 4 if raw == 0 else (2 if raw & 2 else 0) | (1 if raw & 4 else 0)


def enrich_card(card: CardInstance, row: Mapping[str, Any]) -> CardInstance:
    from sts2sim.engine.transitions import _enchant_card, _wither_with_upgrade_count

    level = (
        integer(row["upgrade_level"], "upgrade_level")
        if "upgrade_level" in row
        else int(card.upgraded)
    )
    maximum = 0 if card.card_id.lower() in NON_UPGRADABLE_CARD_IDS else 1
    if level > maximum:
        raise ValueError("Invalid source upgrade level; cannot be losslessly imported")
    if "is_upgraded" in row and row["is_upgraded"] is not (level > 0):
        raise ValueError("Live upgrade level contradicts is_upgraded")
    custom = dict(card.custom)
    if row.get("instance_id"):
        custom["live_instance_id"] = str(row["instance_id"])
        instance_id = str(row.get("deck_instance_id") or row["instance_id"])
        card = card.model_copy(update={"instance_id": instance_id})
    if row.get("deck_instance_id"):
        custom["live_deck_instance_id"] = str(row["deck_instance_id"])
    card = card.model_copy(update={"custom": custom})
    enchantment = _slot(row, "enchantment")
    if enchantment:
        identity = source_enchantment_hooks.enchantment_id(str(enchantment.get("id", "")))
        if identity not in source_enchantment_hooks.SOURCE_ENCHANTMENTS - {
            "deprecated_enchantment"
        }:
            raise ValueError(f"Unknown live enchantment {identity!r}")
        card = _enchant_card(
            card,
            identity,
            integer(enchantment.get("amount"), "enchantment amount"),
            source_id="live",
        )
    affliction = _slot(row, "affliction")
    if affliction:
        identity = str(affliction.get("id", "")).lower().removesuffix("_affliction")
        if identity not in _AFFLICTIONS:
            raise ValueError(f"Unknown live affliction {identity!r}")
        card = source_modifier_hooks.afflict(
            card, identity, integer(affliction.get("amount"), "affliction amount")
        )
    custom = dict(card.custom)
    if row.get("fake_upgrade_level") is not None:
        count = integer(row["fake_upgrade_level"], "fake_upgrade_level")
        if count and card.card_id != "wither":
            raise ValueError("Only source Wither supports fake upgrade levels")
        if card.card_id == "wither":
            card = _wither_with_upgrade_count(card, count).model_copy(
                update={"upgraded": level > 0}
            )
            custom = dict(card.custom)
    if "base_replay_count" in row:
        custom["replay"] = integer(row["base_replay_count"], "base_replay_count")
    if "local_cost_modifiers" in row:
        base = integer(row.get("base_cost"), "base_cost", minimum=None)
        modifiers = []
        for raw in rows(row["local_cost_modifiers"], "local_cost_modifiers"):
            kind = str(raw.get("type", "")).lower()
            if kind == "none":
                continue
            if kind not in {"absolute", "relative"} or not isinstance(raw.get("reduce_only"), bool):
                raise ValueError("Invalid source local-cost modifier")
            modifiers.append(
                {
                    "amount": integer(raw.get("amount"), "local modifier amount", minimum=None),
                    "absolute": kind == "absolute",
                    "expiration": _expiration(raw.get("expiration")),
                    "reduce_only": raw["reduce_only"],
                }
            )
        custom.update(source_local_energy_base=base, source_local_energy_modifiers=modifiers)
    elif "local_cost" in row:
        local = integer(row["local_cost"], "local_cost", minimum=None)
        base = integer(row.get("base_cost", local), "base_cost", minimum=None)
        if local != (base if base < 0 else max(0, base)):
            raise ValueError("Changed local cost lacks ordered modifier telemetry")
        custom["source_local_energy_base"] = base
        custom["source_local_energy_modifiers"] = []
    if "temporary_star_costs" in row:
        base = row.get("base_star_cost", custom.get("star_cost", 0))
        custom["source_local_star_base"] = (
            "X" if custom.get("star_cost") == "X" else integer(base, "base_star_cost", minimum=None)
        )
        modifiers = []
        for raw in rows(row["temporary_star_costs"], "temporary_star_costs"):
            if not all(
                isinstance(raw.get(k), bool)
                for k in ("clears_when_turn_ends", "clears_when_card_is_played")
            ):
                raise ValueError("Missing temporary-star expiry flags")
            expiration = (2 if raw["clears_when_turn_ends"] else 0) | (
                1 if raw["clears_when_card_is_played"] else 0
            )
            modifiers.append(
                {
                    "amount": integer(raw.get("cost"), "temporary star cost"),
                    "expiration": expiration or 4,
                }
            )
        custom["source_local_star_modifiers"] = modifiers
    if "keyword_ids" in row:
        values = row["keyword_ids"]
        if not isinstance(values, list | tuple):
            raise ValueError("Live keyword_ids must be a list")
        keywords = {str(k).lower() for k in values}
        if keywords - _KEYWORDS:
            raise ValueError("Unknown live card keyword")
        custom["live_keyword_ids"] = sorted(keywords)
        source = dict(custom.get("source_spec", {}))
        source["keywords"] = sorted(keywords)
        source["keywords_key"] = sorted(keywords)
        source["tags"] = [
            tag for tag in (source.get("tags") or ()) if str(tag).lower() not in _KEYWORDS
        ]
        custom["source_spec"] = source
        for keyword in _KEYWORDS - {"none"}:
            custom.pop(keyword, None)
        custom.update({keyword: True for keyword in keywords - {"none"}})
        card = card.model_copy(
            update={
                "exhausts": "exhaust" in keywords,
                "tags": tuple(tag for tag in card.tags if tag.lower() not in _KEYWORDS),
            }
        )
    for field in ("exhaust_on_next_play", "retain_this_turn", "is_sly_this_turn"):
        if field in row and not isinstance(row[field], bool):
            raise ValueError(f"Invalid live {field}")
    if row.get("exhaust_on_next_play"):
        custom["source_exhaust_on_next_play"] = True
    if row.get("retain_this_turn"):
        custom["retain_once"] = True
    if row.get("is_sly_this_turn") and not custom.get("sly"):
        custom.update(sly=True, source_core_temporary_sly=True)
    dynamic = row.get("dynamic_vars")
    if dynamic is not None:
        if not isinstance(dynamic, Mapping):
            raise ValueError("Live dynamic_vars must be a numeric object")
        if any(
            isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v)
            for v in dynamic.values()
        ):
            raise ValueError("Invalid live dynamic variable value")
        custom["live_dynamic_vars"] = dict(dynamic)
        damage = dynamic.get("Damage")
        if damage is not None:
            canonical = card.custom.get("source_spec", {}).get("damage")
            if card.card_id == "sovereign_blade":
                custom["sovereign_blade_base_damage"] = integer(damage, "blade damage")
                if "Repeat" in dynamic:
                    custom["sovereign_blade_hits"] = integer(
                        dynamic["Repeat"], "blade repeats", minimum=1
                    )
            elif isinstance(canonical, int | float) and card.card_id != "wither":
                delta = Fraction(str(damage)) - Fraction(str(canonical))
                custom["source_core_damage_bonus"] = str(delta)
    card = card.model_copy(update={"custom": custom})
    if "local_cost_modifiers" in row or "local_cost" in row:
        energy_base = source_cost_hooks.base_cost(card)
        local = source_cost_hooks.local_cost(card)
        card = card.model_copy(
            update={
                "cost": -1
                if custom.get("x_cost")
                else (energy_base if energy_base is None or energy_base < 0 else max(0, local))
            }
        )
        if "local_cost" in row and (
            local if energy_base is not None and energy_base < 0 else max(0, local)
        ) != integer(row["local_cost"], "local_cost", minimum=None):
            raise ValueError("Live local-cost list does not reproduce observed local_cost")
    return card
