"""Build high-level strategic plans from simulator state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from sts2sim.agent_api import encode_observation
from sts2sim.api import serialize
from sts2sim.mechanics.aggression import aggression_summary

from .models import (
    AggressionProfile,
    CombatPace,
    DeckProfile,
    EconomyProfile,
    RiskLevel,
    RunPlan,
    StrategyMode,
    ThreatProfile,
)


def analyze_run(state: Any) -> RunPlan:
    """Return a strategic plan for the current simulator state."""

    state_summary = serialize(state)
    observation = encode_observation(state, include_state=False)
    return analyze_serialized_state(state_summary, observation=observation)


def analyze_serialized_state(
    state_summary: Mapping[str, Any],
    *,
    observation: Mapping[str, Any] | None = None,
) -> RunPlan:
    """Build a ``RunPlan`` from a serialized simulator state."""

    player = _mapping(state_summary.get("player"))
    deck = _deck_profile(_sequence(state_summary.get("master_deck")))
    threat = _threat_profile(state_summary, observation)
    economy = _economy_profile(state_summary, observation, deck)
    hp = _int(player.get("hp"))
    max_hp = max(1, _int(player.get("max_hp"), 1))
    hp_ratio = max(0.0, min(1.0, hp / max_hp))
    risk_level = _risk_level(hp_ratio, threat)
    strategy = _strategy(hp_ratio, risk_level, deck, economy, threat)
    aggression = _aggression_profile(state_summary)

    must_find = _must_find(deck, economy, threat, hp_ratio)
    avoid = _avoid(deck, economy, risk_level)
    remove_targets = _remove_targets(_sequence(state_summary.get("master_deck")))
    upgrade_targets = _upgrade_targets(_sequence(state_summary.get("master_deck")))
    elite_budget = _elite_budget(strategy, risk_level, hp_ratio, economy, aggression)

    return RunPlan(
        strategy=strategy,
        risk_level=risk_level,
        elite_budget=elite_budget,
        hp_ratio=round(hp_ratio, 4),
        deck=deck,
        threat=threat,
        economy=economy,
        aggression=aggression,
        must_find=must_find,
        avoid=avoid,
        upgrade_targets=upgrade_targets,
        remove_targets=remove_targets,
        potion_policy=_potion_policy(risk_level, economy),
        notes=_notes(strategy, risk_level, must_find, aggression),
    )


def _deck_profile(cards: Sequence[object]) -> DeckProfile:
    attacks = 0
    skills = 0
    powers = 0
    statuses = 0
    curses = 0
    upgraded = 0
    strike_like = 0
    defend_like = 0
    draw_cards = 0
    block_cards = 0
    damage_cards = 0

    for raw_card in cards:
        card = _mapping(raw_card)
        card_type = _normalized(card.get("type"))
        card_id = _normalized(card.get("card_id"))
        name = _normalized(card.get("name"))
        effects = _mapping(card.get("effects"))
        tags = {_normalized(tag) for tag in _sequence(card.get("tags"))}
        text_blob = " ".join(
            str(value).lower()
            for value in (
                card.get("card_id"),
                card.get("name"),
                effects,
                tuple(tags),
            )
        )

        attacks += int(card_type == "attack")
        skills += int(card_type == "skill")
        powers += int(card_type == "power")
        statuses += int(card_type == "status")
        curses += int(card_type == "curse")
        upgraded += int(bool(card.get("upgraded")))
        strike_like += int("strike" in card_id or "strike" in name)
        defend_like += int("defend" in card_id or "defend" in name)
        draw_cards += int("draw" in effects or "draw" in text_blob)
        block_cards += int("block" in effects or "block" in text_blob)
        damage_cards += int("damage" in effects or card_type == "attack")

    return DeckProfile(
        total_cards=len(cards),
        attacks=attacks,
        skills=skills,
        powers=powers,
        statuses=statuses,
        curses=curses,
        upgraded=upgraded,
        strike_like=strike_like,
        defend_like=defend_like,
        draw_cards=draw_cards,
        block_cards=block_cards,
        damage_cards=damage_cards,
    )


def _threat_profile(
    state_summary: Mapping[str, Any],
    observation: Mapping[str, Any] | None,
) -> ThreatProfile:
    combat = _mapping(state_summary.get("combat"))
    map_state = _mapping(state_summary.get("map"))
    boss_node_id = _optional_str(map_state.get("boss_node_id"))

    combat_observation = _mapping(_mapping(observation).get("combat")) if observation else {}
    aggression = _aggression_profile(state_summary)
    return ThreatProfile(
        phase=str(state_summary.get("phase", "unknown")),
        act=_int(state_summary.get("act"), 1),
        floor=_int(state_summary.get("floor")),
        incoming_damage=_int(
            combat_observation.get("incoming_damage", _incoming_damage(combat))
        ),
        alive_monsters=_int(
            combat_observation.get("alive_monster_count", _alive_monsters(combat))
        ),
        monster_hp_total=_int(
            combat_observation.get("monster_hp_total", _monster_hp_total(combat))
        ),
        scaling_pressure=aggression.scaling_pressure,
        enemy_attack_pressure=aggression.enemy_attack_pressure,
        next_boss_id=boss_node_id,
        known_elite_id=aggression.known_elite_id,
        possible_elite_ids=aggression.possible_elite_ids,
        unknown_elite_count=aggression.unknown_elite_count,
    )


def _aggression_profile(state_summary: Mapping[str, Any]) -> AggressionProfile:
    summary = aggression_summary(state_summary)
    pace = str(summary.get("combat_pace", "balanced"))
    if pace not in {"stall", "balanced", "rush"}:
        pace = "balanced"
    return AggressionProfile(
        target=round(_float(summary.get("target")), 4),
        hp_floor=round(_float(summary.get("hp_floor")), 4),
        hp_spend_budget=max(0, _int(summary.get("hp_spend_budget"))),
        block_priority=round(_float(summary.get("block_priority")), 4),
        combat_pace=cast(CombatPace, pace),
        combat_pace_pressure=round(_float(summary.get("combat_pace_pressure")), 4),
        allow_chip_damage=bool(summary.get("allow_chip_damage")),
        scaling_pressure=round(_float(summary.get("scaling_pressure")), 4),
        enemy_attack_pressure=round(_float(summary.get("enemy_attack_pressure")), 4),
        elite_pressure=round(_float(summary.get("elite_pressure")), 4),
        future_elite_count=max(0, _int(summary.get("future_elite_count"))),
        future_rest_count=max(0, _int(summary.get("future_rest_count"))),
        nearest_elite_distance=max(0, _int(summary.get("nearest_elite_distance"))),
        nearest_rest_distance=max(0, _int(summary.get("nearest_rest_distance"))),
        boss_distance=max(0, _int(summary.get("boss_distance"))),
        known_elite_id=_optional_str(summary.get("known_elite_id")),
        possible_elite_ids=tuple(
            str(item) for item in _sequence(summary.get("possible_elite_ids"))
        ),
        unknown_elite_count=max(0, _int(summary.get("unknown_elite_count"))),
    )


def _economy_profile(
    state_summary: Mapping[str, Any],
    observation: Mapping[str, Any] | None,
    deck: DeckProfile,
) -> EconomyProfile:
    del observation
    player = _mapping(state_summary.get("player"))
    flags = _mapping(state_summary.get("flags"))
    potions = _sequence(state_summary.get("potions"))
    potion_capacity = _optional_int(flags.get("potion_slots"))
    return EconomyProfile(
        gold=_int(player.get("gold")),
        potion_count=len(potions),
        potion_capacity=potion_capacity,
        relic_count=len(_sequence(state_summary.get("relics"))),
        removable_cards=deck.curses + deck.strike_like + deck.defend_like,
    )


def _risk_level(hp_ratio: float, threat: ThreatProfile) -> RiskLevel:
    if hp_ratio <= 0.25:
        return "critical"
    if threat.incoming_damage > 0 and threat.incoming_damage >= max(1, int(hp_ratio * 40)):
        return "high"
    if hp_ratio <= 0.45:
        return "high"
    if hp_ratio <= 0.7:
        return "medium"
    return "low"


def _strategy(
    hp_ratio: float,
    risk_level: RiskLevel,
    deck: DeckProfile,
    economy: EconomyProfile,
    threat: ThreatProfile,
) -> StrategyMode:
    if risk_level in {"critical", "high"}:
        return "safe"
    if threat.act >= 2 and deck.powers == 0:
        return "scaling"
    if hp_ratio >= 0.75 and economy.potion_count > 0:
        return "greedy"
    return "balanced"


def _must_find(
    deck: DeckProfile,
    economy: EconomyProfile,
    threat: ThreatProfile,
    hp_ratio: float,
) -> tuple[str, ...]:
    needs: list[str] = []
    if deck.damage_cards < 5:
        needs.append("frontload_damage")
    if deck.block_cards < 4 or hp_ratio < 0.5:
        needs.append("block")
    if deck.draw_cards < 2 and deck.total_cards >= 14:
        needs.append("card_draw")
    if deck.powers == 0 and threat.act >= 2:
        needs.append("scaling")
    if economy.potion_count == 0:
        needs.append("potion")
    if deck.curses:
        needs.append("card_remove")
    return tuple(dict.fromkeys(needs))


def _avoid(
    deck: DeckProfile,
    economy: EconomyProfile,
    risk_level: RiskLevel,
) -> tuple[str, ...]:
    avoid: list[str] = []
    if deck.total_cards >= 26:
        avoid.append("deck_bloat")
    if risk_level in {"high", "critical"}:
        avoid.append("hp_loss")
    if economy.gold < 75:
        avoid.append("low_impact_shop_spend")
    return tuple(avoid)


def _remove_targets(cards: Sequence[object]) -> tuple[str, ...]:
    targets: list[str] = []
    for raw_card in cards:
        card = _mapping(raw_card)
        card_id = _normalized(card.get("card_id"))
        name = _normalized(card.get("name"))
        tags = {_normalized(tag) for tag in _sequence(card.get("tags"))}
        if "eternal" in tags:
            continue
        if (
            _normalized(card.get("type")) == "curse"
            or "curse" in tags
            or "strike" in card_id
            or "strike" in name
        ):
            targets.append(str(card.get("card_id", card.get("name", ""))))
    return tuple(item for item in targets if item)[:5]


def _upgrade_targets(cards: Sequence[object]) -> tuple[str, ...]:
    targets: list[str] = []
    priority_terms = ("bash", "eruption", "zap", "neutralize", "inflame", "defragment")
    for raw_card in cards:
        card = _mapping(raw_card)
        if bool(card.get("upgraded")):
            continue
        card_id = _normalized(card.get("card_id"))
        name = _normalized(card.get("name"))
        if any(term in card_id or term in name for term in priority_terms):
            targets.append(str(card.get("card_id", card.get("name", ""))))
    if targets:
        return tuple(dict.fromkeys(targets))[:5]
    for raw_card in cards:
        card = _mapping(raw_card)
        if not bool(card.get("upgraded")) and _normalized(card.get("type")) in {
            "attack",
            "skill",
            "power",
        }:
            value = str(card.get("card_id", card.get("name", "")))
            if value:
                targets.append(value)
    return tuple(dict.fromkeys(targets))[:5]


def _elite_budget(
    strategy: StrategyMode,
    risk_level: RiskLevel,
    hp_ratio: float,
    economy: EconomyProfile,
    aggression: AggressionProfile,
) -> int:
    if risk_level in {"critical", "high"}:
        return 0
    if aggression.target >= 0.8 and aggression.hp_spend_budget >= 18:
        return 3 if economy.potion_count else 2
    if strategy == "greedy" and hp_ratio >= 0.8:
        return 3 if economy.potion_count else 2
    if hp_ratio >= 0.65:
        return 2
    return 1


def _potion_policy(risk_level: RiskLevel, economy: EconomyProfile) -> str:
    if risk_level in {"critical", "high"}:
        return "use_to_prevent_large_hp_loss"
    if economy.potion_count >= 2:
        return "spend_for_elite_or_boss_advantage"
    return "save_for_elite_or_boss"


def _notes(
    strategy: StrategyMode,
    risk_level: RiskLevel,
    must_find: Sequence[str],
    aggression: AggressionProfile,
) -> tuple[str, ...]:
    notes = [f"{strategy}_plan", f"{risk_level}_risk"]
    notes.append(f"aggression_{aggression.combat_pace}")
    if aggression.allow_chip_damage:
        notes.append("hp_budget_available")
    if aggression.known_elite_id:
        notes.append(f"known_elite_{aggression.known_elite_id}")
    notes.extend(f"needs_{need}" for need in must_find)
    return tuple(notes)


def _incoming_damage(combat: Mapping[str, Any]) -> int:
    return sum(
        _int(_mapping(monster).get("intent_damage"))
        for monster in _sequence(combat.get("monsters"))
        if _int(_mapping(monster).get("hp")) > 0
    )


def _alive_monsters(combat: Mapping[str, Any]) -> int:
    return sum(
        1
        for monster in _sequence(combat.get("monsters"))
        if _int(_mapping(monster).get("hp")) > 0
    )


def _monster_hp_total(combat: Mapping[str, Any]) -> int:
    return sum(_int(_mapping(monster).get("hp")) for monster in _sequence(combat.get("monsters")))


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


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalized(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
