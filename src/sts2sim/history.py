"""Human-readable run history for simulator-driven agents."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sts2sim.api import serialize
from sts2sim.engine.serialization import state_digest


class HistoryModel(BaseModel):
    """Base model for history payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class RunHistoryStep(HistoryModel):
    """One readable simulator transition."""

    step_index: int
    phase_before: str
    phase_after: str
    action: dict[str, Any]
    action_summary: str
    state_hash_before: str
    state_hash_after: str
    events: tuple[dict[str, Any], ...] = ()
    context_before: dict[str, Any] = Field(default_factory=dict)
    context_after: dict[str, Any] = Field(default_factory=dict)
    reward: float | None = None
    decision: dict[str, Any] | None = None


class RunHistory(HistoryModel):
    """A complete readable timeline for one simulator run."""

    seed: int | str
    character_id: str
    ascension: int
    policy: str
    initial: dict[str, Any]
    final: dict[str, Any]
    steps: tuple[RunHistoryStep, ...] = ()
    summary: dict[str, Any] = Field(default_factory=dict)


def start_run_history(state: Any, *, policy: str) -> RunHistory:
    """Create an empty history from the initial simulator state."""

    payload = serialize(state)
    initial = summarize_payload(payload)
    return RunHistory(
        seed=_seed(payload),
        character_id=str(payload.get("character_id", "")),
        ascension=_int(payload.get("ascension")),
        policy=policy,
        initial=initial,
        final=initial,
        summary=_history_summary((), initial),
    )


def record_history_step(
    *,
    step_index: int,
    before_state: Any,
    action: Any,
    after_state: Any,
    reward: float | None = None,
    decision: Mapping[str, Any] | None = None,
) -> RunHistoryStep:
    """Build a readable history entry for one already-applied transition."""

    before_payload = serialize(before_state)
    after_payload = serialize(after_state)
    action_payload = action_to_payload(action)
    events = _latest_replay_events(after_state)
    return RunHistoryStep(
        step_index=step_index,
        phase_before=str(before_payload.get("phase", "")),
        phase_after=str(after_payload.get("phase", "")),
        action=action_payload,
        action_summary=summarize_action(before_payload, action_payload),
        state_hash_before=state_digest(before_state),
        state_hash_after=state_digest(after_state),
        events=events,
        context_before=summarize_payload(before_payload),
        context_after=summarize_payload(after_payload),
        reward=None if reward is None else round(float(reward), 6),
        decision=dict(decision) if decision is not None else None,
    )


def append_history_step(history: RunHistory, step: RunHistoryStep, final_state: Any) -> RunHistory:
    """Return a history with ``step`` appended and final/summary refreshed."""

    steps = history.steps + (step,)
    final = summarize_state(final_state)
    return history.model_copy(
        update={
            "steps": steps,
            "final": final,
            "summary": _history_summary(steps, final),
        }
    )


def summarize_state(state: Any) -> dict[str, Any]:
    """Return the compact readable context used by history entries."""

    return summarize_payload(serialize(state))


def summarize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact, JSON-friendly context from a serialized run state."""

    player = _mapping(payload.get("player"))
    return {
        "phase": str(payload.get("phase", "")),
        "act": _int(payload.get("act")),
        "floor": _int(payload.get("floor")),
        "player": {
            "hp": _int(player.get("hp")),
            "max_hp": _int(player.get("max_hp")),
            "block": _int(player.get("block")),
            "energy": _int(player.get("energy")),
            "max_energy": _int(player.get("max_energy")),
            "gold": _int(player.get("gold")),
            "statuses": dict(_mapping(player.get("statuses"))),
            "resources": dict(_mapping(player.get("resources"))),
            "relics": list(_sequence(payload.get("relics"))),
            "potions": list(_sequence(payload.get("potions"))),
            "deck_count": len(_sequence(payload.get("master_deck"))),
        },
        "map": _map_summary(_mapping(payload.get("map"))),
        "ancient": _ancient_summary(_mapping(payload.get("ancient"))),
        "event": _event_summary(_mapping(payload.get("event"))),
        "shop": _shop_summary(_mapping(payload.get("shop"))),
        "reward": _reward_summary(_mapping(payload.get("reward"))),
        "combat": _combat_summary(_mapping(payload.get("combat"))),
        "room_history": list(_sequence(payload.get("room_history"))),
        "flags": _public_flags(_mapping(payload.get("flags"))),
    }


def summarize_action(state_payload: Mapping[str, Any], action: Mapping[str, Any]) -> str:
    """Return a short human-readable action description."""

    action_type = str(action.get("type", "unknown"))
    target_id = _optional_str(action.get("target_id"))
    card_instance_id = _optional_str(action.get("card_instance_id"))

    if action_type == "choose_ancient":
        option = _find_ancient_option(_mapping(state_payload.get("ancient")), target_id)
        if option:
            return (
                f"Choose ancient option {option.get('name', target_id)} "
                f"for relic {option.get('relic_id', 'unknown')}"
            )
        return f"Choose ancient option {target_id}"

    if action_type == "choose_node":
        node = _find_map_node(_mapping(state_payload.get("map")), target_id)
        if node:
            kind = str(node.get("kind", "node"))
            return (
                f"Choose map node {target_id} "
                f"({kind}, floor {_int(node.get('floor'))}, lane {_int(node.get('lane'))})"
            )
        return f"Choose map node {target_id}"

    if action_type == "choose_event":
        option = _find_event_option(_mapping(state_payload.get("event")), target_id)
        if option:
            return f"Choose event option {option.get('title', target_id)}"
        return f"Choose event option {target_id}"

    if action_type == "play_card":
        card = _find_card_by_instance_id(state_payload, card_instance_id)
        target = _target_name(state_payload, target_id)
        card_name = (
            str(card.get("name", card.get("card_id", card_instance_id)))
            if card
            else card_instance_id
        )
        return f"Play {card_name} -> {target}" if target else f"Play {card_name}"

    if action_type == "end_turn":
        return "End turn"

    if action_type == "choose_card":
        card = _find_card_by_instance_id(state_payload, card_instance_id)
        name = (
            str(card.get("name", card.get("card_id", card_instance_id)))
            if card
            else card_instance_id
        )
        return f"Choose card {name}"

    if action_type == "discard_card":
        card = _find_card_by_instance_id(state_payload, card_instance_id)
        name = (
            str(card.get("name", card.get("card_id", card_instance_id)))
            if card
            else card_instance_id
        )
        return f"Discard card {name}"

    if action_type == "exhaust_card":
        card = _find_card_by_instance_id(state_payload, card_instance_id)
        name = (
            str(card.get("name", card.get("card_id", card_instance_id)))
            if card
            else card_instance_id
        )
        return f"Exhaust card {name}"

    if action_type == "use_potion":
        payload = _mapping(action.get("payload"))
        potion_id = str(payload.get("potion_id", "potion"))
        target = _target_name(state_payload, target_id)
        return f"Use potion {potion_id} -> {target}" if target else f"Use potion {potion_id}"

    if action_type == "discard_potion":
        potion_id = _potion_name_for_slot(state_payload, target_id)
        return f"Discard potion {potion_id}"

    if action_type == "shop_buy":
        item = _find_shop_item(_mapping(state_payload.get("shop")), target_id)
        if item:
            return (
                f"Buy {item.get('kind', 'shop item')} {item.get('item_id', target_id)} "
                f"for {_int(item.get('price'))} gold"
            )
        return f"Buy shop item {target_id}"

    if action_type == "shop_leave":
        return "Leave shop"

    if action_type == "throw_potion_at_merchant":
        return "Throw Foul Potion at the merchant"

    if action_type in {"rest", "recall", "dig", "lift"}:
        return action_type.replace("_", " ").title()

    if action_type == "smith":
        card = _find_card_by_instance_id(state_payload, target_id)
        name = str(card.get("name", card.get("card_id", target_id))) if card else target_id
        return f"Smith card {name}"

    if action_type == "toke":
        card = _find_card_by_instance_id(state_payload, target_id)
        name = str(card.get("name", card.get("card_id", target_id))) if card else target_id
        return f"Remove card {name}"

    if action_type.startswith("take_reward"):
        return _reward_action_summary(_mapping(state_payload.get("reward")), action_type, target_id)

    if action_type == "proceed":
        return f"Proceed from {state_payload.get('phase', 'current phase')}"

    return action_type.replace("_", " ").title()


def action_to_payload(action: Any) -> dict[str, Any]:
    """Return a JSON-friendly engine action payload."""

    model_dump = getattr(action, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="json", exclude_none=True))
    if isinstance(action, Mapping):
        return dict(action)
    return {"type": str(action)}


def write_run_history(history: RunHistory, path: Path | str) -> None:
    """Write history as formatted JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(history.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _history_summary(
    steps: Sequence[RunHistoryStep],
    final: Mapping[str, Any],
) -> dict[str, Any]:
    action_types = [step.action.get("type") for step in steps]
    events = [event.get("kind") for step in steps for event in step.events]
    return {
        "steps_taken": len(steps),
        "final_phase": final.get("phase"),
        "final_act": final.get("act"),
        "final_floor": final.get("floor"),
        "cards_played": sum(1 for action_type in action_types if action_type == "play_card"),
        "nodes_chosen": sum(1 for action_type in action_types if action_type == "choose_node"),
        "turns_ended": sum(1 for action_type in action_types if action_type == "end_turn"),
        "rewards_taken": sum(
            1 for action_type in action_types if str(action_type).startswith("take_reward")
        ),
        "event_count": len(events),
        "event_kinds": sorted({str(kind) for kind in events if kind is not None}),
    }


def _latest_replay_events(after_state: Any) -> tuple[dict[str, Any], ...]:
    replay_log = getattr(after_state, "replay_log", ())
    if not replay_log:
        return ()
    latest = replay_log[-1]
    raw_events = getattr(latest, "events", ())
    return tuple(_model_payload(event) for event in raw_events)


def _model_payload(value: Any) -> dict[str, Any]:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="json", exclude_none=True))
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": value}


def _map_summary(game_map: Mapping[str, Any]) -> dict[str, Any]:
    if not game_map:
        return {}
    nodes = [_mapping(node) for node in _sequence(game_map.get("nodes"))]
    node_by_id = {str(node.get("node_id")): node for node in nodes}
    current_id = _optional_str(game_map.get("current_node_id"))
    edges = [_mapping(edge) for edge in _sequence(game_map.get("edges"))]
    reachable_ids = [
        str(edge.get("to_id"))
        for edge in edges
        if current_id is not None and str(edge.get("from_id")) == current_id
    ]
    return {
        "act": _int(game_map.get("act")),
        "current_node_id": current_id,
        "completed_node_ids": list(_sequence(game_map.get("completed_node_ids"))),
        "boss_node_id": _optional_str(game_map.get("boss_node_id")),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "reachable": [
            _node_summary(node_by_id[node_id])
            for node_id in reachable_ids
            if node_id in node_by_id
        ],
    }


def _node_summary(node: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "node_id": str(node.get("node_id", "")),
        "kind": str(node.get("kind", "")),
        "floor": _int(node.get("floor")),
        "lane": _int(node.get("lane")),
    }


def _ancient_summary(ancient: Mapping[str, Any]) -> dict[str, Any]:
    if not ancient:
        return {}
    return {
        "act": _int(ancient.get("act")),
        "ancient_id": str(ancient.get("ancient_id", "")),
        "chosen_option_ids": list(_sequence(ancient.get("chosen_option_ids"))),
        "options": [
            {
                "option_id": str(option.get("option_id", "")),
                "name": str(option.get("name", "")),
                "kind": str(option.get("kind", "")),
                "relic_id": str(option.get("relic_id", "")),
            }
            for option in (_mapping(item) for item in _sequence(ancient.get("options")))
        ],
    }


def _event_summary(event: Mapping[str, Any]) -> dict[str, Any]:
    if not event:
        return {}
    return {
        "event_id": str(event.get("event_id", "")),
        "name": str(event.get("name", "")),
        "page_id": str(event.get("page_id", "")),
        "resolved_option_id": _optional_str(event.get("resolved_option_id")),
        "options": [
            {
                "option_id": str(option.get("option_id", "")),
                "title": str(option.get("title", "")),
                "disabled": bool(option.get("disabled", False)),
            }
            for option in (_mapping(item) for item in _sequence(event.get("options")))
        ],
    }


def _shop_summary(shop: Mapping[str, Any]) -> dict[str, Any]:
    if not shop:
        return {}
    return {
        "node_id": str(shop.get("node_id", "")),
        "card_removals_bought": _int(shop.get("card_removals_bought")),
        "items": [
            {
                "slot_id": str(item.get("slot_id", "")),
                "item_id": str(item.get("item_id", "")),
                "kind": str(item.get("kind", "")),
                "rarity": _optional_str(item.get("rarity")),
                "price": _int(item.get("price")),
                "purchased": bool(item.get("purchased", False)),
            }
            for item in (_mapping(raw) for raw in _sequence(shop.get("items")))
        ],
    }


def _reward_summary(reward: Mapping[str, Any]) -> dict[str, Any]:
    if not reward:
        return {}
    return {
        "reward_id": str(reward.get("reward_id", "")),
        "source": str(reward.get("source", "")),
        "forced": bool(reward.get("forced", False)),
        "gold": _int(reward.get("gold")),
        "gold_claimed": bool(reward.get("gold_claimed", False)),
        "relic_id": _optional_str(reward.get("relic_id")),
        "relic_ids": list(_sequence(reward.get("relic_ids"))),
        "claimed_relic_ids": list(_sequence(reward.get("claimed_relic_ids"))),
        "card_options": list(_sequence(reward.get("card_options"))),
        "card_option_groups": [
            list(_sequence(group)) for group in _sequence(reward.get("card_option_groups"))
        ],
        "card_ids": list(_sequence(reward.get("card_ids"))),
        "potion_id": _optional_str(reward.get("potion_id")),
        "potion_ids": list(_sequence(reward.get("potion_ids"))),
        "claimed_potion_indices": list(_sequence(reward.get("claimed_potion_indices"))),
    }


def _combat_summary(combat: Mapping[str, Any]) -> dict[str, Any]:
    if not combat:
        return {}
    return {
        "turn": _int(combat.get("turn")),
        "player": _combat_player_summary(_mapping(combat.get("player"))),
        "monsters": [
            _monster_summary(_mapping(monster)) for monster in _sequence(combat.get("monsters"))
        ],
        "hand": [_card_summary(_mapping(card)) for card in _sequence(combat.get("hand"))],
        "draw_pile": [_card_summary(_mapping(card)) for card in _sequence(combat.get("draw_pile"))],
        "discard_pile": [
            _card_summary(_mapping(card)) for card in _sequence(combat.get("discard_pile"))
        ],
        "exhaust_pile": [
            _card_summary(_mapping(card)) for card in _sequence(combat.get("exhaust_pile"))
        ],
        "orbs": [
            {"orb_id": str(orb.get("orb_id", "")), "value": _int(orb.get("value"))}
            for orb in (_mapping(raw) for raw in _sequence(combat.get("orbs")))
        ],
        "orb_slots": _int(combat.get("orb_slots")),
        "cards_played_this_turn": list(_sequence(combat.get("cards_played_this_turn"))),
        "pending_choices": [
            {
                "choice_id": str(choice.get("choice_id", "")),
                "kind": str(choice.get("kind", "")),
                "prompt": str(choice.get("prompt", "")),
                "candidate_ids": list(_sequence(choice.get("candidate_ids"))),
                "remaining": _int(choice.get("remaining")),
                "required": bool(choice.get("required", False)),
            }
            for choice in (_mapping(raw) for raw in _sequence(combat.get("pending_choices")))
        ],
    }


def _combat_player_summary(player: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "hp": _int(player.get("hp")),
        "max_hp": _int(player.get("max_hp")),
        "block": _int(player.get("block")),
        "energy": _int(player.get("energy")),
        "statuses": dict(_mapping(player.get("statuses"))),
        "resources": dict(_mapping(player.get("resources"))),
    }


def _monster_summary(monster: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "monster_id": str(monster.get("monster_id", "")),
        "name": str(monster.get("name", "")),
        "hp": _int(monster.get("hp")),
        "max_hp": _int(monster.get("max_hp")),
        "block": _int(monster.get("block")),
        "intent": _optional_str(monster.get("intent")),
        "intent_damage": _int(monster.get("intent_damage")),
        "hit_count": _int(monster.get("hit_count")),
        "statuses": dict(_mapping(monster.get("statuses"))),
    }


def _card_summary(card: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "instance_id": str(card.get("instance_id", "")),
        "card_id": str(card.get("card_id", "")),
        "name": str(card.get("name", "")),
        "type": str(card.get("type", "")),
        "cost": card.get("cost"),
        "upgraded": bool(card.get("upgraded", False)),
        "tags": list(_sequence(card.get("tags"))),
    }


def _public_flags(flags: Mapping[str, Any]) -> dict[str, Any]:
    skipped = {"debug", "source_data", "rng"}
    return {str(key): value for key, value in flags.items() if str(key) not in skipped}


def _find_ancient_option(
    ancient: Mapping[str, Any],
    option_id: str | None,
) -> Mapping[str, Any] | None:
    for option in (_mapping(item) for item in _sequence(ancient.get("options"))):
        if str(option.get("option_id")) == option_id:
            return option
    return None


def _find_map_node(
    game_map: Mapping[str, Any],
    node_id: str | None,
) -> Mapping[str, Any] | None:
    for node in (_mapping(item) for item in _sequence(game_map.get("nodes"))):
        if str(node.get("node_id")) == node_id:
            return node
    return None


def _find_event_option(
    event: Mapping[str, Any],
    option_id: str | None,
) -> Mapping[str, Any] | None:
    for option in (_mapping(item) for item in _sequence(event.get("options"))):
        if str(option.get("option_id")) == option_id:
            return option
    return None


def _find_shop_item(
    shop: Mapping[str, Any],
    slot_id: str | None,
) -> Mapping[str, Any] | None:
    for item in (_mapping(raw) for raw in _sequence(shop.get("items"))):
        if str(item.get("slot_id")) == slot_id:
            return item
    return None


def _find_card_by_instance_id(
    state_payload: Mapping[str, Any],
    instance_id: str | None,
) -> Mapping[str, Any] | None:
    if instance_id is None:
        return None
    for card in _all_cards(state_payload):
        if str(card.get("instance_id")) == instance_id:
            return card
    return None


def _all_cards(state_payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    cards: list[Mapping[str, Any]] = []
    cards.extend(_mapping(card) for card in _sequence(state_payload.get("master_deck")))
    combat = _mapping(state_payload.get("combat"))
    for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        cards.extend(_mapping(card) for card in _sequence(combat.get(zone)))
    return tuple(cards)


def _target_name(state_payload: Mapping[str, Any], target_id: str | None) -> str | None:
    if target_id in {None, "", "none"}:
        return None
    if target_id == "player":
        return "player"
    combat = _mapping(state_payload.get("combat"))
    for monster in (_mapping(item) for item in _sequence(combat.get("monsters"))):
        if str(monster.get("monster_id")) == target_id:
            return str(monster.get("name", target_id))
    return target_id


def _potion_name_for_slot(state_payload: Mapping[str, Any], slot_id: str | None) -> str:
    if slot_id is None:
        return "unknown"
    parts = slot_id.split(":")
    if len(parts) == 2 and parts[0] == "potion":
        index = _int(parts[1])
        potions = _sequence(state_payload.get("potions"))
        if 0 <= index < len(potions):
            return str(potions[index])
    return slot_id


def _reward_action_summary(
    reward: Mapping[str, Any],
    action_type: str,
    target_id: str | None,
) -> str:
    if action_type == "take_reward_gold":
        return f"Take reward gold ({_int(reward.get('gold'))})"
    if action_type == "take_reward_relic":
        relic_id = _reward_relic_for_target(reward, target_id)
        return f"Take reward relic {relic_id}"
    if action_type == "take_reward_potion":
        potion_id = _reward_potion_for_target(reward, target_id)
        return f"Take reward potion {potion_id}"
    if action_type == "take_reward_card":
        card_id = _reward_card_for_target(reward, target_id)
        return f"Take reward card {card_id}"
    return f"Take reward {target_id}"


def _reward_relic_for_target(reward: Mapping[str, Any], target_id: str | None) -> str:
    if target_id == "reward:relic":
        return str(reward.get("relic_id", target_id))
    parts = (target_id or "").split(":")
    if len(parts) == 3 and parts[:2] == ["reward", "relic"]:
        relics = _sequence(reward.get("relic_ids"))
        index = _int(parts[2])
        if 0 <= index < len(relics):
            return str(relics[index])
    return str(target_id)


def _reward_potion_for_target(reward: Mapping[str, Any], target_id: str | None) -> str:
    if target_id == "reward:potion":
        return str(reward.get("potion_id", target_id))
    parts = (target_id or "").split(":")
    if len(parts) == 3 and parts[:2] == ["reward", "potion"]:
        potions = _sequence(reward.get("potion_ids"))
        index = _int(parts[2])
        if 0 <= index < len(potions):
            return str(potions[index])
    return str(target_id)


def _reward_card_for_target(reward: Mapping[str, Any], target_id: str | None) -> str:
    parts = (target_id or "").split(":")
    if len(parts) == 3 and parts[:2] == ["reward", "card"]:
        cards = _sequence(reward.get("card_options"))
        index = _int(parts[2])
        if 0 <= index < len(cards):
            return str(cards[index])
    if len(parts) == 3 and parts[:2] == ["reward", "fixed_card"]:
        cards = _sequence(reward.get("card_ids"))
        index = _int(parts[2])
        if 0 <= index < len(cards):
            return str(cards[index])
    if len(parts) == 4 and parts[:2] == ["reward", "card_group"]:
        groups = _sequence(reward.get("card_option_groups"))
        group_index = _int(parts[2])
        card_index = _int(parts[3])
        if 0 <= group_index < len(groups):
            group = _sequence(groups[group_index])
            if 0 <= card_index < len(group):
                return str(group[card_index])
    if len(parts) == 3 and parts[:2] == ["reward", "remove_card"]:
        return str(target_id)
    return str(target_id)


def _seed(payload: Mapping[str, Any]) -> int | str:
    seed = payload.get("seed", 0)
    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
    return str(seed)


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, str | bytes | bytearray) or value is None or isinstance(value, Mapping):
        return ()
    try:
        return tuple(value)
    except TypeError:
        return ()


def _int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
