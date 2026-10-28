"""Stable 0.107.1 enchantment properties and combat hooks.

Independent rules from Razhuo/sts2-rl-agent commit
5d874e855ae1d0550d5c30416fadd7dbb0d133b7, EnchantmentModel.cs and
MegaCrit.Sts2.Core.Models.Enchantments/*.cs. There are 22 registered current
models plus DeprecatedEnchantment, not 32 distinct enchantment models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from random import Random
from typing import Any

from sts2sim.content.source_card_tags import SOURCE_CARD_TAGS

from .models import CardInstance, CombatState, EffectEvent, RngState, RunState
from .rng import random_stream

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
    effects = dict(card.effects)
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
            effects.pop("exhaust_on_play", None)
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
        update={"custom": custom, "cost": cost, "exhausts": exhausts, "tags": tags,
                "effects": effects}
    )


def has(card: CardInstance, name: str) -> bool:
    return any(enchantment_id(e.keyword) == name for e in card.enchantments)


def _values(combat: CombatState, key: str) -> dict[str, int]:
    raw = combat.metadata.get(key, {})
    return {str(k): int(v) for k, v in raw.items()} if isinstance(raw, Mapping) else {}


def damage_base(combat: CombatState, card: CardInstance, amount: int | Fraction) -> Fraction:
    """Card-local enchantment arithmetic precedes Strength and other hooks."""
    value = Fraction(amount)
    disabled = _values(combat, "source_enchantment_disabled")
    momentum = _values(combat, "source_enchantment_momentum")
    for enchantment in card.enchantments:
        name = enchantment_id(enchantment.keyword)
        if name == "sharp" or (name == "vigorous" and not disabled.get(card.instance_id)):
            value += enchantment.amount
        elif name == "inky":
            value += 1
        elif name == "tezcataras_ember":
            value += 3
        elif name == "momentum":
            value += momentum.get(card.instance_id, 0)
        if name == "corrupted":
            value *= Fraction(3, 2)
        elif name == "instinct":
            value *= 2
    return value


def block_base(combat: CombatState, card: CardInstance, amount: int | Fraction) -> Fraction:
    value = Fraction(amount)
    growth = _values(combat, "source_enchantment_goopy")
    for enchantment in card.enchantments:
        name = enchantment_id(enchantment.keyword)
        if name == "nimble":
            value += enchantment.amount
        elif name == "goopy":
            value += growth.get(card.instance_id, enchantment.amount) - 1
    return value


def after_play(
    state: RunState | None,
    combat: CombatState,
    rng_state: RngState,
    card: CardInstance,
    target_id: str | None,
    relics: Sequence[str],
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    """OnPlay runs after the card's main effects, then AfterCardPlayed mutates it."""
    from . import transitions as t

    events: list[EffectEvent] = []
    for enchantment in card.enchantments:
        if combat.player.hp <= 0:
            break
        name = enchantment_id(enchantment.keyword)
        disabled = _values(combat, "source_enchantment_disabled")
        effect: dict[str, Any] = {}
        if name == "adroit":
            effect = {"block": enchantment.amount}
        elif name == "corrupted":
            # CreatureCmd.Damage is unpowered and unblockable; it still passes
            # through incoming damage caps, Buffer, and HP-loss hooks.
            block = combat.player.block
            combat = combat.model_copy(update={"player": combat.player.model_copy(
                update={"block": 0},
            )})
            combat, emitted = t._damage_combat_player(
                combat, 2, card.instance_id, relics=relics, is_attack=False,
            )
            combat = combat.model_copy(update={"player": combat.player.model_copy(
                update={"block": combat.player.block + block},
            )})
            events.extend(event.model_copy(update={"metadata": {
                **event.metadata, "unblockable": True, "unpowered": True,
                "enchantment": "corrupted",
            }}) for event in emitted)
        elif name == "inky":
            effect = {"apply_status": {
                "weak": 1,
                "target": "all_enemies" if card.target.value == "all_enemies" else "enemy",
            }}
        elif name in {"swift", "sown"} and not disabled.get(card.instance_id):
            disabled[card.instance_id] = 1
            combat = combat.model_copy(update={"metadata": {
                **combat.metadata, "source_enchantment_disabled": disabled,
            }})
            effect = {"draw" if name == "swift" else "energy": enchantment.amount}
        elif name == "momentum":
            values = _values(combat, "source_enchantment_momentum")
            values[card.instance_id] = values.get(card.instance_id, 0) + enchantment.amount
            combat = combat.model_copy(update={"metadata": {
                **combat.metadata, "source_enchantment_momentum": values,
            }})
        if effect:
            combat, rng_state, emitted = t._apply_effect_step(
                state=state, combat=combat, rng_state=rng_state, card=card,
                target_id=target_id, energy_spent=0, effect=effect, relics=relics,
            )
            events.extend(emitted)
        if name == "vigorous":
            disabled[card.instance_id] = 1
            combat = combat.model_copy(update={"metadata": {
                **combat.metadata, "source_enchantment_disabled": disabled,
            }})
        elif name == "goopy":
            values = _values(combat, "source_enchantment_goopy")
            values[card.instance_id] = values.get(card.instance_id, enchantment.amount) + 1
            combat = combat.model_copy(update={"metadata": {
                **combat.metadata, "source_enchantment_goopy": values,
            }})
    return combat, rng_state, tuple(events)


def played_card(combat: CombatState, card: CardInstance) -> CardInstance:
    custom = dict(card.custom)
    custom.pop("source_enchantment_slumber_discount", None)
    growth = _values(combat, "source_enchantment_goopy")
    enchantments = tuple(
        e.model_copy(update={"amount": growth[card.instance_id]})
        if enchantment_id(e.keyword) == "goopy" and card.instance_id in growth else e
        for e in card.enchantments
    )
    return card.model_copy(update={"custom": custom, "enchantments": enchantments})


def preserve_changes(card: CardInstance, updated: CardInstance) -> CardInstance:
    """Upgrade/downgrade modifies base fields without discarding local modifiers."""
    custom = {**updated.custom, **{
        key: value for key, value in card.custom.items() if key.startswith("source_enchantment_")
    }}
    if (not card.upgraded and updated.upgraded and card.cost is not None
            and updated.cost is not None and updated.cost < card.cost
            and "source_enchantment_slither_cost" in custom):
        custom["source_enchantment_slither_cost"] = min(
            int(custom["source_enchantment_slither_cost"]), updated.cost,
        )
    return refresh_properties(updated.model_copy(update={"custom": custom}))


def sync_master_deck(state: RunState, combat: CombatState) -> RunState:
    growth = _values(combat, "source_enchantment_goopy")
    if not growth:
        return state
    return state.model_copy(update={"master_deck": tuple(
        card.model_copy(update={"enchantments": tuple(
            e.model_copy(update={"amount": growth[card.instance_id]})
            if enchantment_id(e.keyword) == "goopy" else e
            for e in card.enchantments
        )}) if card.instance_id in growth else card
        for card in state.master_deck
    )})


def after_draw(card: CardInstance, rng: Random) -> CardInstance:
    if not has(card, "slither"):
        return card
    cost = random_stream(rng, "combat_energy_costs").randrange(4)
    return card.model_copy(update={"custom": {
        **card.custom, "source_enchantment_slither_cost": cost,
    }})


def energy_cost(card: CardInstance, cost: int) -> int:
    cost = int(card.custom.get("source_enchantment_slither_cost", cost))
    return cost - int(card.custom.get("source_enchantment_slumber_discount", 0))


def before_flush(combat: CombatState) -> CombatState:
    if combat.metadata.get("source_enchantment_flush_turn") == combat.turn:
        return combat
    hand = tuple(
        card.model_copy(update={"custom": {
            **card.custom, "source_enchantment_slumber_discount": (
                int(card.custom.get("source_enchantment_slumber_discount", 0)) + 1
            ),
        }}) if has(card, "slumbering_essence") else card
        for card in combat.hand
    )
    return combat.model_copy(update={"hand": hand, "metadata": {
        **combat.metadata, "source_enchantment_flush_turn": combat.turn,
    }})


def shuffle_order(cards: Sequence[CardInstance], *, initial: bool) -> tuple[CardInstance, ...]:
    if initial:
        return tuple(c for c in cards if not has(c, "imbued")) + tuple(
            c for c in cards if has(c, "imbued")
        )
    ordered = list(cards)
    for card in cards:
        if has(card, "perfect_fit"):
            ordered.remove(card)
            ordered.insert(0, card)
    return tuple(ordered)


def opening_autoplay(
    state: RunState, combat: CombatState, rng_state: RngState,
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    requests = tuple({
        "card": card.model_dump(mode="json"), "autoplay_from_zone": "draw_pile",
        "target_mode": "random_enemy", "event_kind": "imbued_card_triggered",
        "metadata": {"enchantment": "imbued"},
    } for card in combat.draw_pile if has(card, "imbued"))
    if not requests:
        return combat, rng_state, ()
    return t._resolve_queued_autoplays(state, combat, rng_state, requests)
