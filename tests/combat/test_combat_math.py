from __future__ import annotations

import importlib
import inspect
from typing import Any

from fakes import apply_block_damage, calculate_attack_damage
from helpers import call_with_supported_kwargs, import_attr, jsonable


def _damage_calculator() -> Any:
    return import_attr(
        ("sts2sim.mechanics.combat", "sts2sim.mechanics.combat_math", "sts2sim.api.combat"),
        ("calculate_attack_damage", "modified_attack_damage", "attack_damage"),
    ) or calculate_attack_damage


def _block_applier() -> Any:
    return import_attr(
        ("sts2sim.mechanics.combat", "sts2sim.mechanics.combat_math", "sts2sim.api.combat"),
        ("apply_block_damage", "resolve_block_damage", "resolve_hp_loss", "apply_damage"),
    ) or apply_block_damage


def _attack_damage(base: int, strength: int, weak: bool, vulnerable: bool) -> int:
    calculator = _damage_calculator()
    signature = inspect.signature(calculator)
    if "attacker" in signature.parameters or "defender" in signature.parameters:
        module = importlib.import_module(calculator.__module__)
        modifier = module.ModifierSnapshot
        return int(
            call_with_supported_kwargs(
                calculator,
                base_damage=base,
                base=base,
                attacker=modifier(strength=strength, weak=weak),
                defender=modifier(vulnerable=vulnerable),
                is_attack=True,
            )
        )
    return int(
        call_with_supported_kwargs(
            calculator,
            base=base,
            base_damage=base,
            strength=strength,
            weak=weak,
            vulnerable=vulnerable,
        )
    )


def _hp_loss(result: Any) -> int:
    payload = jsonable(result)
    if isinstance(payload, dict):
        for key in ("hp_loss", "health_loss", "unblocked", "damage_to_hp"):
            if key in payload:
                return int(payload[key])
    if isinstance(payload, (list, tuple)):
        return int(payload[0])
    return int(payload)


def _remaining_block(result: Any) -> int:
    payload = jsonable(result)
    if isinstance(payload, dict):
        for key in ("remaining_block", "block", "block_remaining"):
            if key in payload:
                return int(payload[key])
    if isinstance(payload, (list, tuple)) and len(payload) > 2:
        return int(payload[2])
    if isinstance(payload, (list, tuple)) and len(payload) > 1:
        return int(payload[1])
    raise AssertionError(f"cannot read remaining block from {payload!r}")


def test_attack_modifiers_use_sts_rounding_order() -> None:
    damage = _attack_damage(base=10, strength=2, weak=True, vulnerable=True)

    assert damage == 13


def test_block_absorbs_damage_before_hp_loss() -> None:
    result = call_with_supported_kwargs(
        _block_applier(),
        damage=13,
        block=5,
        incoming_damage=13,
        current_block=5,
    )

    assert _hp_loss(result) == 8
    assert _remaining_block(result) == 0


def test_negative_strength_cannot_create_negative_attack_damage() -> None:
    damage = _attack_damage(base=3, strength=-10, weak=False, vulnerable=False)

    assert damage == 0
