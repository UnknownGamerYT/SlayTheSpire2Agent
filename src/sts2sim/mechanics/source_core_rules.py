"""Independent rules from stable 0.107.1 C# card models.

Reference: Razhuo/sts2-rl-agent, commit
5d874e855ae1d0550d5c30416fadd7dbb0d133b7, decompiled/
MegaCrit.Sts2.Core.Models.Cards/<PascalCase card id>.cs. Values and ordering
come from CanonicalVars, OnUpgrade and OnPlay, including associated powers.
This module deliberately contains rules, never copies of the decompiled code.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# These cards need state, physical-card mutations, or ordered child operations.
RUNTIME_CARDS = frozenset(
    {
        "anger",
        "anointed",
        "armaments",
        "beat_down",
        "bouncing_flask",
        "calculated_gamble",
        "catastrophe",
        "echoing_slash",
        "expertise",
        "fiend_fire",
        "finisher",
        "fisticuffs",
        "flechettes",
        "havoc",
        "jack_of_all_trades",
        "jackpot",
        "knife_trap",
        "memento_mori",
        "molten_fist",
        "murder",
        "omnislice",
        "perfected_strike",
        "pillage",
        "primal_force",
        "rampage",
        "second_wind",
        "stoke",
        "storm_of_steel",
        "thrash",
        "brand",
        "burning_pact",
        "hidden_daggers",
        "gold_axe",
        "dodge_and_roll",
        "hidden_gem",
        "cinder",
        "true_grit",
        "tracking",
    }
)


def source_core_steps(
    source: Mapping[str, Any],
    steps: tuple[Mapping[str, Any], ...],
    *,
    card_library: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Replace known lossy source-field/text interpretations with source rules."""
    if source.get("effects") or source.get("effect") or not source.get("description"):
        return steps
    card_id = str(source.get("id", source.get("card_id", ""))).lower()
    upgraded = bool(source.get("upgraded"))
    if card_id in RUNTIME_CARDS:
        return ({"source_core": card_id},)

    def status(key: str, amount: int, target: str = "self") -> dict[str, Any]:
        return {"apply_status": {"target": target, key: amount}}

    def shivs(count: int, *, inky: bool = False, upgraded: bool = False) -> dict[str, Any]:
        return {
            "add_card_to_hand": {"count": count, "card": shiv_spec(upgraded=upgraded, inky=inky)}
        }

    if card_id == "expose":
        return (
            {"remove_block": {"target": "enemy"}},
            {"remove_status": {"target": "enemy", "statuses": ("artifact",)}},
            status("vulnerable", 3 if upgraded else 2, "enemy"),
        )
    if card_id == "entrench":
        return ({"block_formula": {"formula": "player_block"}, "unpowered_block": True},)
    if card_id == "accuracy":
        return (status("accuracy", 6 if upgraded else 4),)
    if card_id == "well_laid_plans":
        return (status("well_laid_plans", 2 if upgraded else 1),)
    if card_id == "stampede":
        return (status("stampede", 1),)
    if card_id in {"blade_dance", "blade_of_ink", "cloak_and_dagger", "up_my_sleeve"}:
        count = {"blade_dance": 3, "blade_of_ink": 2, "cloak_and_dagger": 1, "up_my_sleeve": 3}[
            card_id
        ] + upgraded
        result = [{"block": 6}] if card_id == "cloak_and_dagger" else []
        result.append(shivs(count, inky=card_id == "blade_of_ink"))
        if card_id == "up_my_sleeve":
            result.append({"self_cost_delta": -1})
        return tuple(result)
    if card_id == "automation":
        return (
            {
                "combat_trigger": {
                    "trigger": "card_drawn",
                    "duration": "combat",
                    "every": 10,
                    "effects": ({"energy": 1},),
                }
            },
        )
    if card_id == "anticipate":
        return (status("temporary_dexterity", 3 if upgraded else 2),)
    if card_id == "setup_strike":
        return (
            {"damage": 9 if upgraded else 7},
            status("temporary_strength", 3 if upgraded else 2),
        )
    if card_id == "fight_me":
        return (
            {"damage": 6 if upgraded else 5},
            {"damage": 6 if upgraded else 5},
            status("strength", 4 if upgraded else 3),
            status("strength", 1, "enemy"),
        )
    if card_id == "piercing_wail":
        return (status("temporary_strength", -8 if upgraded else -6, "all_enemies"),)
    if card_id in {"equilibrium", "salvo"}:
        return steps + ({"retain_hand": True},)
    if card_id == "shadowmeld":
        return (status("shadowmeld", 1),)
    if card_id == "shadow_step":
        return (
            {"discard_hand": {"mode": "all"}},
            {
                "combat_trigger": {
                    "trigger": "turn_start",
                    "duration": "once",
                    "effects": (status("source_core_double_damage", 1),),
                }
            },
        )
    if card_id == "dominate":
        return (status("vulnerable", 2 if upgraded else 1, "enemy"), {"source_core": "dominate"})
    if card_id in {"pounce", "unrelenting"}:
        return (
            {"damage": 20 if upgraded else 14},
            status(
                "source_core_free_skill" if card_id == "pounce" else "source_core_free_attack", 1
            ),
        )
    if card_id in {"ricochet", "sword_boomerang"}:
        return ({"source_core": card_id},)
    if card_id == "dagger_throw":
        return ({"damage": 12 if upgraded else 9}, {"draw": 1}, {"discard_choice": 1})
    if card_id in {"blood_wall", "bloodletting", "breakthrough", "hemokinesis", "offering"}:
        # HP-loss reactions (Rupture/Inferno) happen before the following effects.
        return tuple(step for step in steps if "hp_loss" in step) + tuple(
            step for step in steps if "hp_loss" not in step
        )
    if card_id in {"dash", "iron_wave"}:
        return tuple(step for step in steps if "block" in step) + tuple(
            step for step in steps if "block" not in step
        )
    if card_id == "cinder":
        return ({"damage": 24 if upgraded else 18}, {"exhaust_random": 1})
    if card_id == "true_grit":
        return (
            {"block": 9 if upgraded else 7},
            {"exhaust_choice" if upgraded else "exhaust_random": 1},
        )
    if card_id == "purity":
        return (
            {
                "choose_card": {
                    "action": "exhaust",
                    "zone": "hand",
                    "count": 5 if upgraded else 3,
                    "up_to": True,
                }
            },
        )
    if card_id == "colossus":
        return ({"block": 8 if upgraded else 5}, status("source_core_colossus", 1))
    if card_id == "fan_of_knives":
        return (status("source_core_fan_of_knives", 1), shivs(5 if upgraded else 4))
    if card_id == "leading_strike":
        return ({"damage": 6 if upgraded else 3}, shivs(2))
    if card_id == "infinite_blades":
        return (
            {
                "combat_trigger": {
                    "trigger": "turn_start",
                    "duration": "combat",
                    "effects": (shivs(1),),
                }
            },
        )
    if card_id == "hand_trick":
        return (
            {"block": 10 if upgraded else 7},
            {
                "choose_card": {
                    "action": "add_sly",
                    "zone": "hand",
                    "card_types": ("skill",),
                    "count": 1,
                }
            },
        )
    if card_id == "juggling":
        return (status("source_core_juggling", 1),)
    if card_id == "restlessness":
        return (
            {
                "conditional": {
                    "condition": {"hand_empty": True},
                    "effects": ({"draw": 3 if upgraded else 2}, {"energy": 3 if upgraded else 2}),
                }
            },
        )
    if card_id == "cascade":
        return ({"play_top_card": {"amount": {"amount": 1 if upgraded else 0, "per_energy": 1}}},)
    if card_id == "splash":
        return (
            {
                "choose_card": {
                    "action": "move_to_hand",
                    "zone": "generated",
                    "count": 1,
                    "choices": 3,
                    "pool": "other_character",
                    "card_types": ("attack",),
                    "upgraded": upgraded,
                    "up_to": True,
                    "free_to_play_this_turn": True,
                }
            },
        )
    if card_id == "seeker_strike":
        return tuple(step for step in steps if "draw" not in step)
    return steps


def shiv_spec(*, upgraded: bool = False, inky: bool = False) -> dict[str, Any]:
    """Canonical Shiv fields retain an independent upgradeable source spec."""
    return {
        "card_id": "shiv",
        "name": "Shiv",
        "type": "attack",
        "target": "enemy",
        "cost": 0,
        "damage": 4,
        "upgrade": {"damage": 2},
        "upgraded": upgraded,
        "enchantments": ({"keyword": "inky", "amount": 1},) if inky else (),
        "exhausts": True,
        "tags": ("shiv",),
    }
