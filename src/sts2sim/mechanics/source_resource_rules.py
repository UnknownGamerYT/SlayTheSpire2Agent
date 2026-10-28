"""Source card resource timing and gain restrictions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


def source_resource_steps(
    source: Mapping[str, Any],
    steps: tuple[Mapping[str, Any], ...],
) -> tuple[Mapping[str, Any], ...]:
    if source.get("effects"):
        return steps
    text = " ".join(re.sub(r"\[/?gold\]", "", str(source.get("description") or "")).lower().split())
    card_id = str(source.get("id", source.get("card_id", ""))).upper()
    if card_id == "CONVERGENCE":
        match = re.search(r"next turn,\s*gain \[energy:(\d+)\] and \[star:(\d+)\]", text)
        if match:
            return (
                {"next_turn": {"energy": int(match[1]), "star": int(match[2])}},
                {"retain_hand": True},
            )
    if card_id == "EXPECT_A_FIGHT" and "for each attack in your hand" in text:
        match = re.search(r"gain \[energy:(\d+)\]", text)
        if match:
            return (
                {"energy_formula": {"formula": "hand_attack_count", "multiplier": int(match[1])}},
                {"apply_status": {"target": "self", "no_energy_gain_this_turn": 1}},
            )
    if card_id == "PANIC_BUTTON":
        match = re.search(r"cannot gain block from cards for (\d+) turns", text)
        if match:
            return steps + (
                {"apply_status": {"target": "self", "no_card_block_gain": int(match[1])}},
            )
    return steps


def permitted_energy_delta(statuses: Mapping[str, int], amount: int) -> int:
    """Block positive gains while preserving energy spending and reductions."""
    if amount > 0 and statuses.get("no_energy_gain_this_turn", 0) > 0:
        return 0
    return amount
