"""Deterministic belief and probability inputs for learning observations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

BELIEF_OBSERVATION_KEYS: tuple[str, ...] = (
    "draw_attack_chance",
    "draw_block_chance",
    "draw_damage_chance",
    "draw_setup_chance",
    "deck_cycle_distance",
    "reshuffle_risk",
    "visible_lethal_now",
    "likely_damage_taken_after_end_turn",
    "turns_to_kill_estimate",
    "survival_margin",
    "route_expected_fights_before_boss",
    "route_expected_elites_before_boss",
    "route_expected_rests_before_boss",
    "route_expected_shops_before_boss",
    "route_expected_rewards_before_boss",
    "reward_visible_card_count",
    "reward_visible_relic_count",
    "reward_visible_potion_count",
    "reward_card_attack_ev",
    "reward_card_block_ev",
    "reward_card_setup_ev",
    "reward_relic_ev",
    "reward_potion_ev",
)

_ATTACK_TOKENS = ("attack", "strike", "slash", "blow", "hit")
_BLOCK_TOKENS = ("block", "defend", "defense", "guard", "shield", "armor")
_DAMAGE_TOKENS = ("damage", "attack", "strike", "slash", "blow", "poison")
_SETUP_TOKENS = (
    "draw",
    "energy",
    "power",
    "skill",
    "setup",
    "discard",
    "exhaust",
    "retain",
    "status",
)


def belief_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic belief/probability features from serialized state.

    The inputs are intentionally lightweight and visible-state based. They do
    not sample hidden RNG or require full engine objects, which keeps them safe
    for offline datasets and live captures with partial fields.
    """

    combat = _mapping(payload.get("combat"))
    player = _active_player(payload)
    draw_per_turn = max(1, _int(combat.get("draw_per_turn"), 5))
    draw_pile = _cards(combat.get("draw_pile"))
    discard_pile = _cards(combat.get("discard_pile"))
    hand = _cards(combat.get("hand"))
    deck = _cards(payload.get("master_deck"))
    draw_window = draw_pile[: min(draw_per_turn, len(draw_pile))]
    if len(draw_window) < draw_per_turn and discard_pile:
        draw_window = (*draw_window, *discard_pile[: draw_per_turn - len(draw_window)])

    draw_denominator = max(1, min(draw_per_turn, len(draw_window)))
    draw_attack_chance = _chance(draw_window, _is_attack_card, draw_denominator)
    draw_block_chance = _chance(draw_window, _is_block_card, draw_denominator)
    draw_damage_chance = _chance(draw_window, _is_damage_card, draw_denominator)
    draw_setup_chance = _chance(draw_window, _is_setup_card, draw_denominator)
    deck_cycle_distance = _deck_cycle_distance(len(draw_pile), draw_per_turn)
    reshuffle_risk = _clamp((draw_per_turn - len(draw_pile)) / draw_per_turn)

    monsters = [
        monster
        for monster in (_mapping(raw) for raw in _sequence(combat.get("monsters")))
        if _int(monster.get("hp")) > 0
    ]
    visible_damage = sum(_card_damage(card) for card in hand)
    monster_effective_hp = sum(
        _int(monster.get("hp")) + _int(monster.get("block")) for monster in monsters
    )
    visible_lethal_now = 1.0 if monsters and visible_damage >= monster_effective_hp else 0.0
    incoming_damage = sum(_monster_incoming_damage(monster) for monster in monsters)
    likely_damage_taken = max(0, incoming_damage - _int(player.get("block")))
    hp = _int(player.get("hp"))
    survival_margin = hp - likely_damage_taken
    turns_to_kill = _turns_to_kill_estimate(
        monster_effective_hp=monster_effective_hp,
        visible_damage=visible_damage,
        deck=deck or (*hand, *draw_pile, *discard_pile),
        draw_per_turn=draw_per_turn,
    )

    route = _route_expectations(payload)
    reward = _reward_counts(payload)
    deck_needs = _deck_needs(deck or (*hand, *draw_pile, *discard_pile))
    reward_evs = _reward_expected_values(reward, deck_needs)

    return {
        "draw_attack_chance": round(draw_attack_chance, 4),
        "draw_block_chance": round(draw_block_chance, 4),
        "draw_damage_chance": round(draw_damage_chance, 4),
        "draw_setup_chance": round(draw_setup_chance, 4),
        "deck_cycle_distance": round(deck_cycle_distance, 4),
        "reshuffle_risk": round(reshuffle_risk, 4),
        "visible_lethal_now": visible_lethal_now,
        "likely_damage_taken_after_end_turn": likely_damage_taken,
        "turns_to_kill_estimate": round(turns_to_kill, 4),
        "survival_margin": survival_margin,
        "route_expected_fights_before_boss": round(route["fights"], 4),
        "route_expected_elites_before_boss": round(route["elites"], 4),
        "route_expected_rests_before_boss": round(route["rests"], 4),
        "route_expected_shops_before_boss": round(route["shops"], 4),
        "route_expected_rewards_before_boss": round(route["rewards"], 4),
        "reward_visible_card_count": reward["cards"],
        "reward_visible_relic_count": reward["relics"],
        "reward_visible_potion_count": reward["potions"],
        "reward_card_attack_ev": round(reward_evs["card_attack"], 4),
        "reward_card_block_ev": round(reward_evs["card_block"], 4),
        "reward_card_setup_ev": round(reward_evs["card_setup"], 4),
        "reward_relic_ev": round(reward_evs["relic"], 4),
        "reward_potion_ev": round(reward_evs["potion"], 4),
    }


def belief_vector(summary: Mapping[str, Any]) -> list[float]:
    """Return fixed-order numeric belief features."""

    return [_float(summary.get(key)) for key in BELIEF_OBSERVATION_KEYS]


def _active_player(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    combat_player = _mapping(_mapping(payload.get("combat")).get("player"))
    if combat_player:
        return combat_player
    return _mapping(payload.get("player"))


def _cards(value: object) -> tuple[Mapping[str, Any], ...]:
    cards: list[Mapping[str, Any]] = []
    for index, raw in enumerate(_sequence(value)):
        if isinstance(raw, Mapping):
            cards.append(_mapping(raw))
        elif raw is not None:
            text = str(raw)
            cards.append({"card_id": text, "name": text, "position": index})
    return tuple(cards)


def _chance(
    cards: Sequence[Mapping[str, Any]],
    predicate: Callable[[Mapping[str, Any]], bool],
    denominator: int,
) -> float:
    if denominator <= 0:
        return 0.0
    return _clamp(sum(1 for card in cards if predicate(card)) / denominator)


def _is_attack_card(card: Mapping[str, Any]) -> bool:
    return _card_type(card) == "attack" or _has_any_token(card, _ATTACK_TOKENS)


def _is_block_card(card: Mapping[str, Any]) -> bool:
    effects = _mapping(card.get("effects"))
    return _positive_effect(effects, "block") or _has_any_token(card, _BLOCK_TOKENS)


def _is_damage_card(card: Mapping[str, Any]) -> bool:
    return _card_damage(card) > 0 or _has_any_token(card, _DAMAGE_TOKENS)


def _is_setup_card(card: Mapping[str, Any]) -> bool:
    effects = _mapping(card.get("effects"))
    if any(_positive_effect(effects, key) for key in ("draw", "energy", "discard", "exhaust")):
        return True
    return _card_type(card) in {"power", "skill"} or _has_any_token(card, _SETUP_TOKENS)


def _card_type(card: Mapping[str, Any]) -> str:
    return _normalized(card.get("type"))


def _card_damage(card: Mapping[str, Any]) -> int:
    effects = _mapping(card.get("effects"))
    damage = _int(effects.get("damage")) + _int(effects.get("aoe_damage"))
    if damage:
        return damage
    if _is_attack_card_by_identity(card):
        return 6
    return 0


def _is_attack_card_by_identity(card: Mapping[str, Any]) -> bool:
    return _card_type(card) == "attack" or _has_any_token(card, _ATTACK_TOKENS)


def _positive_effect(effects: Mapping[str, Any], key: str) -> bool:
    value = effects.get(key)
    if isinstance(value, Mapping):
        return bool(value)
    return _int(value) > 0


def _has_any_token(card: Mapping[str, Any], tokens: Sequence[str]) -> bool:
    haystack = " ".join(
        _normalized(value)
        for value in (
            card.get("card_id"),
            card.get("name"),
            card.get("type"),
            *(_sequence(card.get("tags"))),
        )
    )
    effects = _mapping(card.get("effects"))
    haystack = f"{haystack} {' '.join(_normalized(key) for key in effects)}"
    return any(token in haystack for token in tokens)


def _monster_incoming_damage(monster: Mapping[str, Any]) -> int:
    damage = _int(monster.get("intent_damage"))
    hit_count = max(1, _int(monster.get("hit_count"), 1))
    metadata = _mapping(monster.get("metadata"))
    if metadata.get("intent_damage_total") is not None:
        return _int(metadata.get("intent_damage_total"))
    return damage * hit_count


def _turns_to_kill_estimate(
    *,
    monster_effective_hp: int,
    visible_damage: int,
    deck: Sequence[Mapping[str, Any]],
    draw_per_turn: int,
) -> float:
    if monster_effective_hp <= 0:
        return 0.0
    if visible_damage >= monster_effective_hp:
        return 0.0
    average_card_damage = (
        sum(_card_damage(card) for card in deck) / max(1, len(deck))
        if deck
        else 0.0
    )
    expected_turn_damage = max(float(visible_damage), average_card_damage * max(1, draw_per_turn))
    if expected_turn_damage <= 0:
        return 99.0
    return min(99.0, monster_effective_hp / expected_turn_damage)


def _deck_cycle_distance(draw_pile_count: int, draw_per_turn: int) -> float:
    if draw_per_turn <= 0:
        return 0.0
    return draw_pile_count / draw_per_turn


def _route_expectations(payload: Mapping[str, Any]) -> dict[str, float]:
    game_map = _mapping(payload.get("map"))
    nodes = {
        str(node.get("node_id")): node
        for node in (_mapping(raw) for raw in _sequence(game_map.get("nodes")))
        if node.get("node_id") is not None
    }
    current_id = str(game_map.get("current_node_id") or "")
    if not current_id or current_id not in nodes:
        return {"fights": 0.0, "elites": 0.0, "rests": 0.0, "shops": 0.0, "rewards": 0.0}

    completed = {str(item) for item in _sequence(game_map.get("completed_node_ids"))}
    outgoing: dict[str, list[str]] = {}
    for raw_edge in _sequence(game_map.get("edges")):
        edge = _mapping(raw_edge)
        from_id = edge.get("from_id")
        to_id = edge.get("to_id")
        if from_id is not None and to_id is not None:
            outgoing.setdefault(str(from_id), []).append(str(to_id))

    paths: list[list[str]] = []
    stack: list[tuple[str, list[str]]] = [(current_id, [current_id])]
    while stack and len(paths) < 256:
        node_id, path = stack.pop()
        next_ids = [
            next_id
            for next_id in outgoing.get(node_id, ())
            if next_id not in completed and next_id not in path
        ]
        if not next_ids:
            paths.append(path)
            continue
        for next_id in reversed(next_ids):
            stack.append((next_id, [*path, next_id]))
    if not paths:
        return {"fights": 0.0, "elites": 0.0, "rests": 0.0, "shops": 0.0, "rewards": 0.0}

    totals = {"fights": 0.0, "elites": 0.0, "rests": 0.0, "shops": 0.0, "rewards": 0.0}
    for path in paths:
        for node_id in path[1:]:
            kind = _normalized(nodes.get(node_id, {}).get("kind"))
            if kind in {"monster", "elite"}:
                totals["fights"] += 1.0
                totals["rewards"] += 1.0
            if kind == "elite":
                totals["elites"] += 1.0
                totals["rewards"] += 1.0
            elif kind == "rest":
                totals["rests"] += 1.0
            elif kind == "shop":
                totals["shops"] += 1.0
            elif kind == "treasure":
                totals["rewards"] += 1.0
    return {key: value / len(paths) for key, value in totals.items()}


def _reward_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    reward = _mapping(payload.get("reward"))
    card_count = (
        len(_sequence(reward.get("card_ids")))
        + len(_sequence(reward.get("card_options")))
        + sum(len(_sequence(group)) for group in _sequence(reward.get("card_option_groups")))
    )
    relic_count = int(bool(reward.get("relic_id"))) + len(_sequence(reward.get("relic_ids")))
    potion_count = int(bool(reward.get("potion_id"))) + len(_sequence(reward.get("potion_ids")))
    return {"cards": card_count, "relics": relic_count, "potions": potion_count}


def _deck_needs(deck: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    total = max(1, len(deck))
    attack_ratio = sum(1 for card in deck if _is_attack_card(card)) / total
    block_ratio = sum(1 for card in deck if _is_block_card(card)) / total
    setup_ratio = sum(1 for card in deck if _is_setup_card(card)) / total
    return {
        "attack": _clamp(0.42 - attack_ratio, maximum=0.42) / 0.42,
        "block": _clamp(0.32 - block_ratio, maximum=0.32) / 0.32,
        "setup": _clamp(0.24 - setup_ratio, maximum=0.24) / 0.24,
    }


def _reward_expected_values(
    reward: Mapping[str, int],
    deck_needs: Mapping[str, float],
) -> dict[str, float]:
    cards = _scaled(reward["cards"], 3)
    relics = _scaled(reward["relics"], 2)
    potions = _scaled(reward["potions"], 2)
    return {
        "card_attack": cards * (0.35 + 0.65 * deck_needs["attack"]),
        "card_block": cards * (0.35 + 0.65 * deck_needs["block"]),
        "card_setup": cards * (0.30 + 0.70 * deck_needs["setup"]),
        "relic": relics * 0.85,
        "potion": potions * 0.65,
    }


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _sequence(value: object) -> tuple[Any, ...]:
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


def _scaled(value: int, maximum: int) -> float:
    if maximum <= 0:
        return 0.0
    return _clamp(value / maximum)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _normalized(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
