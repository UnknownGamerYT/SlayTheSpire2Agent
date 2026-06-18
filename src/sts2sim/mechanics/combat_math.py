"""Pure combat math helpers.

These helpers resolve the deterministic pieces of damage and block math.  They
do not know about cards, relic hooks, enemy intents, or engine state objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor

from sts2sim.content.sources import STS1_COMPAT_SOURCE, SourceRef


@dataclass(frozen=True, slots=True)
class CombatMathRules:
    weak_attack_multiplier: float = 0.75
    vulnerable_damage_multiplier: float = 1.5
    frail_block_multiplier: float = 0.75
    intangible_hp_loss_cap: int = 1
    min_attack_damage: int = 0
    source: SourceRef = STS1_COMPAT_SOURCE


@dataclass(frozen=True, slots=True)
class ModifierSnapshot:
    strength: int = 0
    dexterity: int = 0
    weak: bool = False
    vulnerable: bool = False
    frail: bool = False
    intangible: bool = False
    damage_multiplier: float = 1.0
    block_multiplier: float = 1.0


@dataclass(frozen=True, slots=True)
class DamageRequest:
    base_damage: int
    hits: int = 1
    block: int = 0
    is_attack: bool = True
    attacker: ModifierSnapshot = field(default_factory=ModifierSnapshot)
    defender: ModifierSnapshot = field(default_factory=ModifierSnapshot)


@dataclass(frozen=True, slots=True)
class DamageResult:
    per_hit_damage: tuple[int, ...]
    hp_loss: int
    block_lost: int
    remaining_block: int
    source: SourceRef


@dataclass(frozen=True, slots=True)
class BlockResult:
    block: int
    source: SourceRef


DEFAULT_COMBAT_RULES = CombatMathRules()


def _scaled(value: int, multiplier: float) -> int:
    return floor(value * multiplier)


def modified_attack_damage(
    base_damage: int,
    *,
    attacker: ModifierSnapshot | None = None,
    defender: ModifierSnapshot | None = None,
    is_attack: bool = True,
    rules: CombatMathRules = DEFAULT_COMBAT_RULES,
) -> int:
    """Return single-hit damage after strength, weak, vulnerable, and multipliers."""

    attacker = attacker or ModifierSnapshot()
    defender = defender or ModifierSnapshot()
    damage = base_damage
    if is_attack:
        damage += attacker.strength
        if attacker.weak:
            damage = _scaled(damage, rules.weak_attack_multiplier)
    if defender.vulnerable:
        damage = _scaled(damage, rules.vulnerable_damage_multiplier)
    damage = _scaled(damage, attacker.damage_multiplier)
    return max(rules.min_attack_damage, damage)


def block_gain(
    base_block: int,
    *,
    actor: ModifierSnapshot | None = None,
    rules: CombatMathRules = DEFAULT_COMBAT_RULES,
) -> BlockResult:
    """Return block gained after dexterity, frail, and custom block multiplier."""

    actor = actor or ModifierSnapshot()
    block = base_block + actor.dexterity
    if actor.frail:
        block = _scaled(block, rules.frail_block_multiplier)
    block = _scaled(block, actor.block_multiplier)
    return BlockResult(block=max(0, block), source=rules.source)


def resolve_hp_loss(
    incoming_damage: int,
    *,
    block: int,
    defender: ModifierSnapshot | None = None,
    rules: CombatMathRules = DEFAULT_COMBAT_RULES,
) -> tuple[int, int, int]:
    """Resolve one hit against block.

    Returns ``(hp_loss, block_lost, remaining_block)``.
    """

    defender = defender or ModifierSnapshot()
    incoming_damage = max(0, incoming_damage)
    block = max(0, block)
    block_lost = min(block, incoming_damage)
    unblocked = incoming_damage - block_lost
    if defender.intangible and unblocked > rules.intangible_hp_loss_cap:
        unblocked = rules.intangible_hp_loss_cap
    return unblocked, block_lost, block - block_lost


def resolve_attack(
    request: DamageRequest,
    *,
    rules: CombatMathRules = DEFAULT_COMBAT_RULES,
) -> DamageResult:
    """Resolve a multi-hit attack request against block and defensive modifiers."""

    if request.hits < 1:
        raise ValueError(f"DamageRequest.hits must be >= 1: {request.hits}")

    per_hit = modified_attack_damage(
        request.base_damage,
        attacker=request.attacker,
        defender=request.defender,
        is_attack=request.is_attack,
        rules=rules,
    )
    remaining_block = max(0, request.block)
    total_hp_loss = 0
    total_block_lost = 0
    hits: list[int] = []
    for _ in range(request.hits):
        hp_loss, block_lost, remaining_block = resolve_hp_loss(
            per_hit,
            block=remaining_block,
            defender=request.defender,
            rules=rules,
        )
        total_hp_loss += hp_loss
        total_block_lost += block_lost
        hits.append(per_hit)
    return DamageResult(
        per_hit_damage=tuple(hits),
        hp_loss=total_hp_loss,
        block_lost=total_block_lost,
        remaining_block=remaining_block,
        source=rules.source,
    )


def apply_percent_change(value: int, percent: int) -> int:
    """Apply an integer percentage change with floor rounding."""

    return floor(value * (100 + percent) / 100)
