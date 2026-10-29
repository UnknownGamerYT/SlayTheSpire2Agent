"""Source-defined card rules requiring play eligibility or physical-card state."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


def source_execution_steps(
    source: Mapping[str, Any],
    steps: tuple[Mapping[str, Any], ...],
) -> tuple[Mapping[str, Any], ...]:
    text = re.sub(r"\[/?gold\]", "", str(source.get("description", ""))).lower()
    card_id = str(source.get("id", source.get("card_id", ""))).upper()
    if not text:
        return steps
    damage = next((step["damage"] for step in steps if "damage" in step), 0)
    if card_id == "BULLY":
        extra = re.search(r"deals (\d+) additional damage for each vulnerable", text)
        if extra:
            return ({"damage_formula": {
                "formula": "target_vulnerable", "base": damage, "multiplier": int(extra[1]),
            }},)
    if card_id == "REND":
        extra = re.search(r"deals (\d+) additional damage for each unique debuff", text)
        if extra:
            # Stable Rend+ is 18 + 8 per debuff (Untapped corroborates the
            # CalculationBase upgrade); Codex's upgraded first sentence is stale.
            upgrade = source.get("upgrade")
            if source.get("upgraded") and isinstance(upgrade, Mapping):
                damage += int(upgrade.get("calculationbase", 0))
            return ({"damage_formula": {
                "formula": "target_unique_debuffs", "base": damage,
                "multiplier": int(extra[1]),
            }},)
    if card_id == "ASHEN_STRIKE":
        extra = re.search(r"deals (\d+) additional damage for each card", text)
        if extra:
            return ({"damage_formula": {
                "formula": "exhaust_pile_count", "base": damage,
                "multiplier": int(extra[1]),
            }},)
    if card_id == "PRECISE_CUT":
        penalty = re.search(r"deals (\d+) less damage for each other card", text)
        if penalty:
            # This source upgrade changes CalculationBase, not a generic Damage
            # field. Its stable row has no upgraded description to replace it.
            upgrade = source.get("upgrade")
            if source.get("upgraded") and not source.get("upgrade_description") and isinstance(
                upgrade, Mapping
            ):
                damage += int(upgrade.get("calculationbase", 0))
            return ({"damage_formula": {
                "formula": "hand_count", "base": damage,
                "multiplier": -int(penalty[1]),
            }},)
    if card_id == "HELIX_DRILL":
        return ({"repeat_damage": {
            "amount": damage, "basis": "energy_spent_this_turn",
        }},)
    if card_id == "RADIATE":
        amount = next((step["all_damage"] for step in steps if "all_damage" in step), damage)
        return ({"repeat_damage": {
            "amount": amount, "basis": "stars_gained_this_turn", "target": "all enemies",
        }},)
    x_damage = re.fullmatch(
        r"deal (\d+) damage(?: to (a random enemy|all enemies))? x(?:\+(\d+))? times\.?", text,
    )
    if (source.get("is_x_cost") or source.get("is_x_star_cost")) and x_damage:
        return ({"repeat_damage": {
            "amount": int(x_damage[1]), "basis": "energy_spent",
            "extra_hits": int(x_damage[3] or 0), "target": x_damage[2] or "enemy",
        }},)
    if source.get("is_x_cost") and "evoke your rightmost orb x" in text:
        extra = re.search(r"orb x\+(\d+)", text)
        return ({"evoke_orb": {"selector": "rightmost", "amount": {
            "amount": int(extra[1]) if extra else 0, "per_energy": 1,
        }}},)
    if card_id == "HEAVENLY_DRILL":
        return ({"repeat_damage": {"amount": damage, "basis": "energy_spent", "double_at": 4}},)
    if card_id == "SPITE":
        match = re.search(r"hits\s+(\d+)\s+times", text)
        return (
            {
                "repeat_damage": {
                    "amount": damage,
                    "basis": "lost_hp_this_turn",
                    "hits": int(match[1]) if match else 2,
                }
            },
        )
    if card_id == "TEAR_ASUNDER":
        return ({"repeat_damage": {"amount": damage, "basis": "hp_loss_count_this_combat"}},)
    if card_id == "EIDOLON":
        return ({"exhaust_hand": {"mode": "all"}, "intangible_if_exhausted": 9},)
    if card_id == "THE_GAMBIT":
        return steps + ({"apply_status": {"target": "self", "the_gambit": 1}},)
    if card_id == "BLUR":
        return steps + ({"apply_status": {"target": "self", "blur": 1}},)
    if card_id == "FLAME_BARRIER":
        match = re.search(r"deal\s+(\d+)\s+damage back", text)
        if match:
            return steps + ({"apply_status": {"target": "self", "flame_barrier":
                                             int(match[1])}},)
    if card_id in {"BOLAS", "THRUMMING_HATCHET"}:
        return steps + ({"return_next_turn": True},)
    return steps


def source_play_restrictions(source: Mapping[str, Any]) -> dict[str, bool]:
    text = str(source.get("description", "")).lower()
    if "can only be played if every card" in text:
        return {"requires_attack_only_hand": True}
    if "can only be played if there are no cards in your" in text:
        return {"requires_empty_draw_pile": True}
    return {}
