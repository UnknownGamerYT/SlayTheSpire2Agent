"""Stable 0.107.1 enchantment properties and combat hooks.

Independent rules from Razhuo/sts2-rl-agent commit
5d874e855ae1d0550d5c30416fadd7dbb0d133b7, EnchantmentModel.cs and
MegaCrit.Sts2.Core.Models.Enchantments/*.cs. There are 22 registered current
models plus DeprecatedEnchantment, not 32 distinct enchantment models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sts2sim.content.source_card_tags import SOURCE_CARD_TAGS

from .models import CardInstance

SOURCE_ENCHANTMENTS = frozenset(
    {
        "adroit",
        "clone",
        "corrupted",
        "deprecated_enchantment",
        "glam",
        "goopy",
        "imbued",
        "inky",
        "instinct",
        "momentum",
        "nimble",
        "perfect_fit",
        "royally_approved",
        "sharp",
        "slither",
        "slumbering_essence",
        "souls_power",
        "sown",
        "spiral",
        "steady",
        "swift",
        "tezcataras_ember",
        "vigorous",
    }
)
_NAMES = {name.replace("_", ""): name for name in SOURCE_ENCHANTMENTS}

_SOURCE_BLOCK_CARDS = frozenset(
    {
        "armaments",
        "backflip",
        "blood_wall",
        "blur",
        "bone_shards",
        "boost_away",
        "boot_sequence",
        "bulwark",
        "charge_battery",
        "cloak_and_dagger",
        "cloak_of_stars",
        "colossus",
        "compact",
        "cosmic_indifference",
        "dash",
        "deaths_door",
        "defend_defect",
        "defend_ironclad",
        "defend_necrobinder",
        "defend_regent",
        "defend_silent",
        "deflect",
        "defy",
        "delay",
        "demonic_shield",
        "dodge_and_roll",
        "entrench",
        "equilibrium",
        "escape_plan",
        "evil_eye",
        "fight_through",
        "finesse",
        "fisticuffs",
        "flame_barrier",
        "gather_light",
        "genetic_algorithm",
        "glacier",
        "glasswork",
        "glitterstream",
        "grave_warden",
        "hand_trick",
        "hologram",
        "i_am_invincible",
        "impervious",
        "intercept",
        "iron_wave",
        "leap",
        "leg_sweep",
        "lift",
        "lightning_rod",
        "manifest_authority",
        "melancholy",
        "mimic",
        "minion_sacrifice",
        "mirage",
        "negative_pulse",
        "panic_button",
        "particle_wall",
        "patter",
        "pull_aggro",
        "rally",
        "reflect",
        "relax",
        "sacrifice",
        "second_wind",
        "shadow_shield",
        "shrug_it_off",
        "stack",
        "survivor",
        "taunt",
        "the_gambit",
        "toric_toughness",
        "true_grit",
        "ultimate_defend",
        "undeath",
        "untouchable",
    }
)
_SOURCE_BASIC_CARDS = frozenset(
    {
        "bash",
        "bodyguard",
        "defend_defect",
        "defend_ironclad",
        "defend_necrobinder",
        "defend_regent",
        "defend_silent",
        "dualcast",
        "falling_star",
        "neutralize",
        "strike_defect",
        "strike_ironclad",
        "strike_necrobinder",
        "strike_regent",
        "strike_silent",
        "survivor",
        "unleash",
        "venerate",
        "zap",
    }
)


def enchantment_id(value: str) -> str:
    key = "".join(c for c in value.lower() if c.isalnum())
    return _NAMES.get(key, value.lower().replace(" ", "_"))


def _keywords(card: CardInstance) -> set[str]:
    from . import transitions as t

    result = {
        key
        for key in ("exhaust", "unplayable", "innate", "retain", "eternal")
        if t._card_has_keyword(card, key)
    }
    if card.exhausts:
        result.add("exhaust")
    return result


def _has_block(effects: Mapping[str, Any]) -> bool:
    if any(key in effects for key in ("block", "block_formula", "block_from_ally")):
        return True
    return any(
        _has_block(item) for item in effects.get("sequence", ()) if isinstance(item, Mapping)
    )


def can_enchant(card: CardInstance, keyword: str, *, in_deck: bool = True) -> bool:
    """EnchantmentModel.CanEnchant plus the source model's extra predicate."""
    name = enchantment_id(keyword)
    if card.type.value in {"status", "curse", "quest"} or card.enchantments:
        return False
    keywords = _keywords(card)
    if in_deck and "unplayable" in keywords:
        return False
    if name in {"corrupted", "instinct", "momentum", "sharp", "vigorous"}:
        return card.type.value == "attack"
    if name == "imbued":
        return card.type.value == "skill"
    if name == "royally_approved":
        return card.type.value in {"attack", "skill"}
    tags = {str(tag).lower() for tag in card.tags} | set(SOURCE_CARD_TAGS.get(card.card_id, ()))
    if name == "goopy":
        return "defend" in tags
    if name == "nimble":
        return card.card_id in _SOURCE_BLOCK_CARDS or _has_block(card.effects)
    if name == "slither":
        source = card.custom.get("source_spec", {})
        return (
            "unplayable" not in keywords
            and not card.custom.get("x_cost")
            and not (isinstance(source, Mapping) and source.get("is_x_cost"))
        )
    if name == "souls_power":
        return "exhaust" in keywords
    if name == "spiral":
        source = card.custom.get("source_spec", {})
        basic = card.card_id in _SOURCE_BASIC_CARDS or (
            isinstance(source, Mapping) and str(source.get("rarity", "")).lower() == "basic"
        )
        return basic and bool(tags & {"strike", "defend"})
    return True


def refresh_properties(card: CardInstance) -> CardInstance:
    """Source OnEnchant effects also run when deserializing or upgrading."""
    custom = dict(card.custom)
    cost = card.cost
    exhausts = card.exhausts
    tags = card.tags
    for enchantment in card.enchantments:
        name = enchantment_id(enchantment.keyword)
        if name in {"royally_approved", "steady"}:
            custom["retain"] = True
        if name == "royally_approved":
            custom["innate"] = True
        if name == "goopy":
            exhausts = True
        if name == "souls_power":
            exhausts = False
            tags = tuple(
                tag for tag in tags if str(tag).lower() not in {"exhaust", "keyword_exhaust"}
            )
            source = custom.get("source_spec")
            if isinstance(source, Mapping):
                source = dict(source)
                for key in ("keywords", "keywords_key", "tags"):
                    values = source.get(key, ())
                    if isinstance(values, Sequence) and not isinstance(values, str):
                        source[key] = tuple(v for v in values if str(v).lower() != "exhaust")
                custom["source_spec"] = source
            custom["exhaust_on_play"] = False
        if name == "tezcataras_ember":
            custom["eternal"] = True
            if not custom.get("x_cost"):
                cost = 0
    return card.model_copy(
        update={"custom": custom, "cost": cost, "exhausts": exhausts, "tags": tags}
    )
