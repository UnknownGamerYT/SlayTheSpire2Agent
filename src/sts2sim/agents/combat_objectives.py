"""Combat projection scoring for search-backed agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sts2sim.mechanics.aggression import aggression_summary


@dataclass(frozen=True)
class CombatProjectionScore:
    """Score and explanation for one projected combat line."""

    score: float
    reasons: tuple[str, ...]
    won: bool
    lost: bool
    reached_phase: str
    hp_delta: int
    max_hp_delta: int
    resource_delta: int
    monster_hp_delta: int
    alive_monster_delta: int
    projected_unblocked_damage: int


def evaluate_combat_projection(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
    *,
    path_length: int,
    event_kinds: Sequence[str] = (),
) -> CombatProjectionScore:
    """Score a projected combat state as a future run state, not a single turn."""

    initial_player = _active_player(initial)
    final_player = _active_player(final)
    initial_phase = _phase(initial)
    final_phase = _phase(final)
    initial_combat = _mapping(initial.get("combat"))
    final_combat = _mapping(final.get("combat"))

    initial_hp = _int(initial_player.get("hp"))
    final_hp = _int(final_player.get("hp"))
    initial_max_hp = _int(initial_player.get("max_hp"))
    final_max_hp = _int(final_player.get("max_hp"))
    hp_delta = final_hp - initial_hp
    max_hp_delta = final_max_hp - initial_max_hp
    resource_delta = _resource_total(final_player) - _resource_total(initial_player)
    potion_delta = len(_sequence(final.get("potions"))) - len(_sequence(initial.get("potions")))
    gold_delta = _int(_mapping(final.get("player")).get("gold")) - _int(
        _mapping(initial.get("player")).get("gold")
    )

    initial_monster_hp = _monster_hp_total(initial_combat)
    final_monster_hp = _monster_hp_total(final_combat)
    monster_hp_delta = initial_monster_hp - final_monster_hp
    initial_alive = _alive_monsters(initial_combat)
    final_alive = _alive_monsters(final_combat)
    alive_monster_delta = initial_alive - final_alive
    projected_unblocked = _projected_unblocked_damage(final_combat)
    turns_elapsed = max(0, _int(final_combat.get("turn"), 1) - _int(initial_combat.get("turn"), 1))
    aggression = aggression_summary(initial)
    aggression_target = _float(aggression.get("target"), 0.5)
    hp_spend_budget = _int(aggression.get("hp_spend_budget"))
    block_priority = _float(aggression.get("block_priority"), 0.5)
    scaling_pressure = _float(aggression.get("scaling_pressure"))

    lost = final_phase == "failed" or final_hp <= 0
    won = initial_phase == "combat" and final_phase != "combat" and not lost

    score = 0.0
    reasons: list[str] = []
    if lost:
        score -= 100_000.0
        reasons.append("projection_loses_combat")
    if won:
        score += 10_000.0
        reasons.append("projection_wins_combat")

    score += monster_hp_delta * 4.0
    if monster_hp_delta:
        reasons.append("projection_reduces_enemy_hp")
    score += alive_monster_delta * 150.0
    if alive_monster_delta:
        reasons.append("projection_kills_enemy")

    hp_loss_weight = max(12.0, 22.0 + block_priority * 8.0 - aggression_target * 8.0)
    hp_gain_weight = 18.0 + block_priority * 4.0
    score += hp_delta * (hp_loss_weight if hp_delta < 0 else hp_gain_weight)
    if hp_delta < 0:
        reasons.append("projection_loses_player_hp")
    elif hp_delta > 0:
        reasons.append("projection_heals_player_hp")

    score += max_hp_delta * 180.0
    if max_hp_delta > 0:
        reasons.append("projection_gains_permanent_max_hp")
    elif max_hp_delta < 0:
        reasons.append("projection_loses_permanent_max_hp")

    score += resource_delta * 18.0
    if resource_delta > 0:
        reasons.append("projection_gains_combat_resource")
    elif resource_delta < 0:
        reasons.append("projection_spends_combat_resource")

    score += gold_delta * 1.5
    if gold_delta > 0:
        reasons.append("projection_gains_gold")

    if potion_delta < 0:
        score += potion_delta * 45.0
        reasons.append("projection_spends_potion")
    elif potion_delta > 0:
        score += potion_delta * 18.0
        reasons.append("projection_gains_potion")

    if projected_unblocked:
        over_budget = max(0, projected_unblocked - hp_spend_budget)
        score -= projected_unblocked * (6.0 + block_priority * 8.0)
        score -= over_budget * 8.0
        reasons.append("projection_accounts_for_next_incoming_damage")
        if over_budget:
            reasons.append("projection_exceeds_hp_spend_budget")

    score -= turns_elapsed * (2.0 + scaling_pressure * 6.0 + aggression_target * 1.5)
    score -= path_length * 0.2

    if any(kind == "player_resource_changed" for kind in event_kinds):
        score += 60.0
        reasons.append("projection_triggered_card_specific_resource_payoff")
    if any("max_hp" in kind for kind in event_kinds):
        score += 90.0
        reasons.append("projection_triggered_max_hp_payoff")

    if not reasons:
        reasons.append("projection_preserves_state")

    return CombatProjectionScore(
        score=score,
        reasons=tuple(dict.fromkeys(reasons)),
        won=won,
        lost=lost,
        reached_phase=final_phase,
        hp_delta=hp_delta,
        max_hp_delta=max_hp_delta,
        resource_delta=resource_delta,
        monster_hp_delta=monster_hp_delta,
        alive_monster_delta=alive_monster_delta,
        projected_unblocked_damage=projected_unblocked,
    )


def _active_player(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    combat = _mapping(payload.get("combat"))
    combat_player = _mapping(combat.get("player"))
    if combat_player:
        return combat_player
    return _mapping(payload.get("player"))


def _monster_hp_total(combat: Mapping[str, Any]) -> int:
    return sum(_int(_mapping(monster).get("hp")) for monster in _sequence(combat.get("monsters")))


def _alive_monsters(combat: Mapping[str, Any]) -> int:
    return sum(
        1
        for monster in _sequence(combat.get("monsters"))
        if _int(_mapping(monster).get("hp")) > 0
    )


def _projected_unblocked_damage(combat: Mapping[str, Any]) -> int:
    player = _mapping(combat.get("player"))
    block = _int(player.get("block"))
    incoming = sum(
        _int(_mapping(monster).get("intent_damage"))
        for monster in _sequence(combat.get("monsters"))
        if _int(_mapping(monster).get("hp")) > 0
    )
    return max(0, incoming - block)


def _resource_total(player: Mapping[str, Any]) -> int:
    resources = _mapping(player.get("resources"))
    return sum(_int(value) for value in resources.values())


def _phase(payload: Mapping[str, Any]) -> str:
    return str(payload.get("phase", "unknown"))


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _int(value: object, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _float(value: object, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default
