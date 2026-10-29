"""Import the current visible attack from its source base and displayed label.

AttackIntent labels already include damage hooks. Keeping the independently
captured DamageCalc value lets previews apply later visible power changes once,
without attempting to invert rounded displayed damage or reconstruct future AI.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from sts2sim.engine.models import CombatState, MonsterState
from sts2sim.live_parity import _status_id

_ATTACK_MODIFIERS = frozenset({
    "strength", "temporary_strength", "vigor", "attack_damage_percent",
    "temporary_attack_damage_percent", "shrink", "weak", "debilitate",
})
_DEFENSE_MODIFIERS = frozenset({
    "tainted", "tank", "diamond_diadem", "stance_wrath", "vulnerable",
    "intangible", "debilitate",
})


def _rows(value: object, name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list | tuple) or not all(isinstance(v, Mapping) for v in value):
        raise ValueError(f"Live {name} must be a list of objects")
    return tuple(value)


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"Invalid live {name} integer")
    return value


def import_monster_intents(
    raw: Mapping[str, Any], monster: MonsterState, combat: CombatState,
    relics: Sequence[str],
) -> MonsterState:
    """Validate current uniform attacks and retain source raw damage separately.

    Old unmodified fixture labels remain readable. An attack affected by powers
    needs explicit base telemetry, including when rounding hides the modifier.
    """
    from sts2sim.engine import transitions as t

    statuses = {}
    for status in _rows(raw.get("status"), "enemy.status"):
        identity = _status_id(status)
        if not identity:
            raise ValueError("Live status is missing its identity")
        statuses[identity] = _integer(status.get("amount"), "status amount")
    kinds = []
    base_damage: Fraction | None = None
    shown_damage: int | None = None
    hit_count = 0
    block = 0
    intents = _rows(raw.get("intents"), "enemy.intents")
    if not intents:
        raise ValueError("Live monster requires explicit visible intents")
    for intent in intents:
        kind = str(intent.get("type", "")).lower()
        kinds.append(kind)
        if "attack" in kind:
            label = str(intent.get("label", ""))
            match = re.fullmatch(r"\s*(\d+)(?:\s*[x×]\s*(\d+))?\s*", label)
            if match is None:
                raise ValueError(f"Unmapped live attack intent amount: {label!r}")
            damage, repeats = int(match[1]), int(match[2] or 1)
            if repeats < 1:
                raise ValueError("Live attack hit count must be positive")
            if "repeats" in intent and _integer(intent["repeats"], "attack repeats",
                                                minimum=1) != repeats:
                raise ValueError("Live attack repeats contradict its displayed label")
            if "base_damage" in intent:
                value = intent["base_damage"]
                if (isinstance(value, bool) or not isinstance(value, int | float)
                        or not math.isfinite(value)):
                    raise ValueError("Live attack base_damage must be finite numeric telemetry")
                base = Fraction(str(value))
            else:
                if (any(statuses.get(key) for key in _ATTACK_MODIFIERS)
                        or any(combat.player.statuses.get(key) for key in _DEFENSE_MODIFIERS)):
                    raise ValueError("Modified live attack requires explicit base_damage telemetry")
                base = Fraction(damage)
            if (base_damage is not None and (base_damage != base or shown_damage != damage)):
                raise ValueError("Distinct simultaneous attack amounts need a hit sequence")
            base_damage, shown_damage = base, damage
            hit_count += repeats
        elif "defend" in kind:
            label = str(intent.get("label", "")).strip()
            if label.isdecimal():
                block += int(label)
    metadata = {**monster.metadata, "live_visible_intents": [
        {key: intent[key] for key in ("type", "label", "base_damage", "repeats") if key in intent}
        for intent in intents
    ]}
    if raw.get("monster_id"):
        metadata["source_monster_id"] = str(raw["monster_id"]).lower()
    if base_damage is not None:
        metadata["live_attack_base_per_hit"] = str(base_damage)
    result = monster.model_copy(update={
        "intent": "_".join(kinds), "intent_damage": (shown_damage or 0) * hit_count,
        "hit_count": hit_count or 1, "intent_block": block,
        "metadata": metadata, "statuses": statuses,
    })
    if base_damage is not None:
        expected = t._refresh_monster_intent_from_metadata(result, combat.player, relics)
        if expected.intent_damage != result.intent_damage:
            raise ValueError("Live attack base and damage hooks contradict its displayed label")
    return result
