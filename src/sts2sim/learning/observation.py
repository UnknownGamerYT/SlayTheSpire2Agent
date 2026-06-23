"""Rich observation encoder for learning agents.

The compact observation vector is useful for smoke tests and simple baselines.
This module exposes the larger structured view needed by stronger policies:
cards by zone, visible map paths, monsters, rewards, shops, events, relics,
potions, statuses, and legal action descriptors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sts2sim.agent_api import action_space, encode_observation
from sts2sim.api import serialize


def encode_rich_observation(
    state: Any,
    *,
    max_map_paths: int = 64,
    max_path_depth: int = 16,
    include_serialized_state: bool = False,
) -> dict[str, Any]:
    """Encode a state with detailed symbolic context for learning agents."""

    payload = serialize(state)
    compact = encode_observation(state, include_state=False)
    rich: dict[str, Any] = {
        "schema_version": 1,
        "mode": "rich_v1",
        "compact": compact,
        "aggression": compact.get("aggression", {}),
        "belief": compact.get("belief", {}),
        "reward_plan": compact.get("reward_plan", {}),
        "route_plan": compact.get("route_plan", {}),
        "visibility": compact.get("visibility", {}),
        "positions": compact.get("positions", {}),
        "targets": compact.get("targets", {}),
        "run": _run_summary(payload),
        "player": _player_summary(payload),
        "card_zones": _card_zones(payload),
        "combat": _combat_summary(payload),
        "map": _map_summary(
            payload,
            max_paths=max_map_paths,
            max_depth=max_path_depth,
        ),
        "reward": _reward_summary(payload),
        "shop": _shop_summary(payload),
        "event": _event_summary(payload),
        "ancient": _ancient_summary(payload),
        "legal_actions": {
            "count": len(action_space(state)),
            "actions": action_space(state),
        },
    }
    if include_serialized_state:
        rich["state"] = payload
    return rich


def _run_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": payload.get("seed"),
        "character_id": payload.get("character_id"),
        "ascension": _int(payload.get("ascension")),
        "phase": str(payload.get("phase", "unknown")),
        "act": _int(payload.get("act")),
        "floor": _int(payload.get("floor")),
        "room_history": list(_sequence(payload.get("room_history"))),
        "flags": dict(_mapping(payload.get("flags"))),
    }


def _player_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    player = _mapping(payload.get("player"))
    return {
        "hp": _int(player.get("hp")),
        "max_hp": _int(player.get("max_hp")),
        "block": _int(player.get("block")),
        "energy": _int(player.get("energy")),
        "max_energy": _int(player.get("max_energy")),
        "gold": _int(player.get("gold")),
        "statuses": dict(_mapping(player.get("statuses"))),
        "resources": dict(_mapping(player.get("resources"))),
        "relics": list(_sequence(payload.get("relics"))),
        "relic_positions": [
            {"relic_id": str(relic_id), "position": index}
            for index, relic_id in enumerate(_sequence(payload.get("relics")))
        ],
        "potions": list(_sequence(payload.get("potions"))),
        "potion_slots": [
            {"potion_id": str(potion_id), "slot_index": index, "position": index}
            for index, potion_id in enumerate(_sequence(payload.get("potions")))
        ],
        "curses": list(_sequence(payload.get("curses"))),
    }


def _card_zones(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    combat = _mapping(payload.get("combat"))
    return {
        "master_deck": [
            _card_summary(card, zone="master_deck", position=index)
            for index, card in enumerate(_sequence(payload.get("master_deck")))
        ],
        "hand": [
            _card_summary(card, zone="hand", position=index)
            for index, card in enumerate(_sequence(combat.get("hand")))
        ],
        "draw_pile": [
            _card_summary(card, zone="draw_pile", position=index)
            for index, card in enumerate(_sequence(combat.get("draw_pile")))
        ],
        "discard_pile": [
            _card_summary(card, zone="discard_pile", position=index)
            for index, card in enumerate(_sequence(combat.get("discard_pile")))
        ],
        "exhaust_pile": [
            _card_summary(card, zone="exhaust_pile", position=index)
            for index, card in enumerate(_sequence(combat.get("exhaust_pile")))
        ],
    }


def _card_summary(raw_card: object, *, zone: str, position: int) -> dict[str, Any]:
    card = _mapping(raw_card)
    effects = _mapping(card.get("effects"))
    return {
        "instance_id": _optional_str(card.get("instance_id")),
        "card_id": _optional_str(card.get("card_id")),
        "zone": zone,
        "position": position,
        "position_from_top": position if zone == "draw_pile" else None,
        "name": _optional_str(card.get("name")),
        "type": _optional_str(card.get("type")),
        "cost": card.get("cost"),
        "target": _optional_str(card.get("target")),
        "upgraded": bool(card.get("upgraded")),
        "exhausts": bool(card.get("exhausts")),
        "tags": list(_sequence(card.get("tags"))),
        "enchantments": [_compact_mapping(item) for item in _sequence(card.get("enchantments"))],
        "custom": dict(_mapping(card.get("custom"))),
        "effect_keys": sorted(effects.keys()),
        "effects": dict(effects),
    }


def _combat_summary(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    combat = _mapping(payload.get("combat"))
    if not combat:
        return None
    player = _mapping(combat.get("player"))
    return {
        "turn": _int(combat.get("turn")),
        "draw_per_turn": _int(combat.get("draw_per_turn")),
        "player": {
            "hp": _int(player.get("hp")),
            "max_hp": _int(player.get("max_hp")),
            "block": _int(player.get("block")),
            "energy": _int(player.get("energy")),
            "max_energy": _int(player.get("max_energy")),
            "statuses": dict(_mapping(player.get("statuses"))),
            "resources": dict(_mapping(player.get("resources"))),
        },
        "monsters": [_monster_summary(monster) for monster in _sequence(combat.get("monsters"))],
        "orbs": [_compact_mapping(orb) for orb in _sequence(combat.get("orbs"))],
        "orb_slots": _int(combat.get("orb_slots")),
        "cards_played_this_turn": list(_sequence(combat.get("cards_played_this_turn"))),
        "pending_choices": [
            _pending_choice_summary(choice) for choice in _sequence(combat.get("pending_choices"))
        ],
        "metadata": dict(_mapping(combat.get("metadata"))),
    }


def _monster_summary(raw_monster: object) -> dict[str, Any]:
    monster = _mapping(raw_monster)
    return {
        "monster_id": _optional_str(monster.get("monster_id")),
        "name": _optional_str(monster.get("name")),
        "hp": _int(monster.get("hp")),
        "max_hp": _int(monster.get("max_hp")),
        "block": _int(monster.get("block")),
        "intent": _optional_str(monster.get("intent")),
        "intent_damage": _int(monster.get("intent_damage")),
        "intent_block": _int(monster.get("intent_block")),
        "move_id": _optional_str(monster.get("move_id")),
        "next_move_id": _optional_str(monster.get("next_move_id")),
        "hit_count": _int(monster.get("hit_count"), 1),
        "statuses": dict(_mapping(monster.get("statuses"))),
        "metadata": dict(_mapping(monster.get("metadata"))),
    }


def _pending_choice_summary(raw_choice: object) -> dict[str, Any]:
    choice = _mapping(raw_choice)
    return {
        "choice_id": _optional_str(choice.get("choice_id")),
        "kind": _optional_str(choice.get("kind")),
        "source_id": _optional_str(choice.get("source_id")),
        "prompt": _optional_str(choice.get("prompt")),
        "zone": _optional_str(choice.get("zone")),
        "candidate_ids": list(_sequence(choice.get("candidate_ids"))),
        "selected_ids": list(_sequence(choice.get("selected_ids"))),
        "min_choices": _int(choice.get("min_choices")),
        "max_choices": _int(choice.get("max_choices")),
        "remaining": _int(choice.get("remaining")),
        "required": bool(choice.get("required")),
        "metadata": dict(_mapping(choice.get("metadata"))),
    }


def _map_summary(
    payload: Mapping[str, Any],
    *,
    max_paths: int,
    max_depth: int,
) -> dict[str, Any] | None:
    game_map = _mapping(payload.get("map"))
    if not game_map:
        return None
    nodes = [_map_node_summary(node) for node in _sequence(game_map.get("nodes"))]
    edges = [_compact_mapping(edge) for edge in _sequence(game_map.get("edges"))]
    node_by_id = {str(node["node_id"]): node for node in nodes if node.get("node_id") is not None}
    outgoing = _outgoing_by_id(edges)
    current_node_id = _optional_str(game_map.get("current_node_id"))
    completed = tuple(str(item) for item in _sequence(game_map.get("completed_node_ids")))
    reachable = tuple(
        node_id
        for node_id in outgoing.get(str(current_node_id), ())
        if node_id not in set(completed)
    )
    paths = _enumerate_map_paths(
        current_node_id=str(current_node_id) if current_node_id is not None else "",
        node_by_id=node_by_id,
        outgoing=outgoing,
        completed=set(completed),
        max_paths=max_paths,
        max_depth=max_depth,
    )
    return {
        "act": _int(game_map.get("act")),
        "current_node_id": current_node_id,
        "completed_node_ids": list(completed),
        "boss_node_id": _optional_str(game_map.get("boss_node_id")),
        "nodes": nodes,
        "edges": edges,
        "reachable_next_node_ids": list(reachable),
        "paths": paths,
    }


def _map_node_summary(raw_node: object) -> dict[str, Any]:
    node = _mapping(raw_node)
    return {
        "node_id": _optional_str(node.get("node_id")),
        "act": _int(node.get("act")),
        "floor": _int(node.get("floor")),
        "lane": _int(node.get("lane")),
        "kind": _optional_str(node.get("kind")),
    }


def _enumerate_map_paths(
    *,
    current_node_id: str,
    node_by_id: Mapping[str, Mapping[str, Any]],
    outgoing: Mapping[str, tuple[str, ...]],
    completed: set[str],
    max_paths: int,
    max_depth: int,
) -> list[dict[str, Any]]:
    if not current_node_id:
        return []
    paths: list[list[str]] = []
    stack: list[tuple[str, list[str]]] = [(current_node_id, [current_node_id])]
    while stack and len(paths) < max_paths:
        node_id, path = stack.pop()
        next_ids = [
            next_id
            for next_id in outgoing.get(node_id, ())
            if next_id not in completed and next_id not in path
        ]
        if not next_ids or len(path) >= max_depth:
            paths.append(path)
            continue
        for next_id in reversed(next_ids):
            stack.append((next_id, [*path, next_id]))
    return [_path_summary(path, node_by_id) for path in paths]


def _path_summary(
    path: Sequence[str],
    node_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    nodes = [node_by_id.get(node_id, {"node_id": node_id}) for node_id in path]
    kinds = [str(node.get("kind", "unknown")) for node in nodes]
    return {
        "node_ids": list(path),
        "kinds": kinds,
        "floors": [_int(node.get("floor")) for node in nodes],
        "lanes": [_int(node.get("lane")) for node in nodes],
        "length": len(path),
        "elite_count": kinds.count("elite"),
        "rest_count": kinds.count("rest"),
        "shop_count": kinds.count("shop"),
        "treasure_count": kinds.count("treasure"),
        "event_count": kinds.count("event"),
        "monster_count": kinds.count("monster"),
        "ends_at_boss": bool(kinds and kinds[-1] == "boss"),
    }


def _reward_summary(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    reward = _mapping(payload.get("reward"))
    if not reward:
        return None
    return {
        "reward_id": _optional_str(reward.get("reward_id")),
        "source": _optional_str(reward.get("source")),
        "forced": bool(reward.get("forced")),
        "gold": _int(reward.get("gold")),
        "gold_claimed": bool(reward.get("gold_claimed")),
        "relic_id": _optional_str(reward.get("relic_id")),
        "relic_ids": list(_sequence(reward.get("relic_ids"))),
        "claimed_relic_ids": list(_sequence(reward.get("claimed_relic_ids"))),
        "card_ids": list(_sequence(reward.get("card_ids"))),
        "card_options": list(_sequence(reward.get("card_options"))),
        "card_option_groups": [
            list(_sequence(group)) for group in _sequence(reward.get("card_option_groups"))
        ],
        "potion_id": _optional_str(reward.get("potion_id")),
        "potion_ids": list(_sequence(reward.get("potion_ids"))),
        "choice_positions": _reward_choice_positions(reward),
        "metadata": dict(_mapping(reward.get("metadata"))),
    }


def _reward_choice_positions(reward: Mapping[str, Any]) -> list[dict[str, Any]]:
    choices: list[dict[str, Any]] = []
    if _int(reward.get("gold")) > 0:
        choices.append({"kind": "gold", "content_id": "gold", "position": 0})
    if reward.get("relic_id"):
        choices.append({"kind": "relic", "content_id": str(reward.get("relic_id")), "position": 0})
    for index, relic_id in enumerate(_sequence(reward.get("relic_ids"))):
        choices.append({"kind": "relic", "content_id": str(relic_id), "position": index})
    for index, card_id in enumerate(_sequence(reward.get("card_options"))):
        choices.append({"kind": "card", "content_id": str(card_id), "position": index})
    for index, card_id in enumerate(_sequence(reward.get("card_ids"))):
        choices.append({"kind": "fixed_card", "content_id": str(card_id), "position": index})
    group_position = 0
    for group_index, group in enumerate(_sequence(reward.get("card_option_groups"))):
        for card_index, card_id in enumerate(_sequence(group)):
            choices.append(
                {
                    "kind": "card_group",
                    "content_id": str(card_id),
                    "group_index": group_index,
                    "card_index": card_index,
                    "position": group_position,
                }
            )
            group_position += 1
    if reward.get("potion_id"):
        choices.append(
            {
                "kind": "potion",
                "content_id": str(reward.get("potion_id")),
                "position": 0,
            }
        )
    for index, potion_id in enumerate(_sequence(reward.get("potion_ids"))):
        choices.append({"kind": "potion", "content_id": str(potion_id), "position": index})
    return choices


def _shop_summary(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    shop = _mapping(payload.get("shop"))
    if not shop:
        return None
    return {
        "node_id": _optional_str(shop.get("node_id")),
        "card_removals_bought": _int(shop.get("card_removals_bought")),
        "items": [
            {**_compact_mapping(item), "position": index, "slot_index": index}
            for index, item in enumerate(_sequence(shop.get("items")))
        ],
    }


def _event_summary(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    event = _mapping(payload.get("event"))
    if not event:
        return None
    return {
        "event_id": _optional_str(event.get("event_id")),
        "name": _optional_str(event.get("name")),
        "page_id": _optional_str(event.get("page_id")),
        "resolved_option_id": _optional_str(event.get("resolved_option_id")),
        "options": [
            {**_compact_mapping(option), "position": index}
            for index, option in enumerate(_sequence(event.get("options")))
        ],
        "metadata": dict(_mapping(event.get("metadata"))),
    }


def _ancient_summary(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    ancient = _mapping(payload.get("ancient"))
    if not ancient:
        return None
    return {
        "act": _int(ancient.get("act")),
        "ancient_id": _optional_str(ancient.get("ancient_id")),
        "chosen_option_ids": list(_sequence(ancient.get("chosen_option_ids"))),
        "options": [
            {**_compact_mapping(option), "position": index}
            for index, option in enumerate(_sequence(ancient.get("options")))
        ],
    }


def _outgoing_by_id(edges: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    outgoing: dict[str, list[str]] = {}
    for edge in edges:
        from_id = edge.get("from_id")
        to_id = edge.get("to_id")
        if from_id is None or to_id is None:
            continue
        outgoing.setdefault(str(from_id), []).append(str(to_id))
    return {key: tuple(values) for key, values in outgoing.items()}


def _compact_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _sequence(value: object) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


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
