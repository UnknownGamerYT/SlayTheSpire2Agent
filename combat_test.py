from __future__ import annotations

import argparse
import json
import random
import re
import threading
import webbrowser
from collections.abc import Mapping, Sequence
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sts2sim import legal_actions, new_run, step
from sts2sim.engine.models import (
    Action,
    ActionType,
    CardInstance,
    EffectEvent,
    MapEdgeState,
    MapNodeState,
    MapState,
    MonsterState,
    PlayerState,
    RoomKind,
    RunPhase,
)
from sts2sim.engine.transitions import _apply_orb_effects, _card_from_spec, _draw_cards

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8791
DEFAULT_CHARACTER_ID = "IRONCLAD"
DEFAULT_CARD_CACHE = Path("data/cache/eng/cards.json")
DEFAULT_CHARACTER_CACHE = Path("data/cache/eng/characters.json")
DEFAULT_MONSTER_CACHE = Path("data/cache/eng/monsters.json")
DEFAULT_POTION_CACHE = Path("data/cache/eng/potions.json")
DEFAULT_POWER_CACHE = Path("data/cache/eng/powers.json")
DEFAULT_RELIC_CACHE = Path("data/cache/eng/relics.json")
DEFAULT_SOURCE_FLAGS: dict[str, Any] = {
    "max_acts": 1,
    "draw_per_turn": 5,
    "combat_reward_potion_chance_percent": 0,
    "combat_reward_card_count": 0,
    "combat_reward_relic_count": 0,
}
FALLBACK_SOURCE_DATA: dict[str, Any] = {
    "flags": {
        **DEFAULT_SOURCE_FLAGS,
    },
}
COMMON_STATUS_OPTIONS: tuple[tuple[str, str, str], ...] = (
    ("strength", "Strength", "Core"),
    ("dexterity", "Dexterity", "Core"),
    ("vulnerable", "Vulnerable", "Core"),
    ("weak", "Weak", "Core"),
    ("frail", "Frail", "Core"),
    ("poison", "Poison", "Core"),
    ("artifact", "Artifact", "Core"),
    ("intangible", "Intangible", "Core"),
    ("plated_armor", "Plated Armor", "Core"),
    ("metallicize", "Metallicize", "Core"),
    ("thorns", "Thorns", "Core"),
    ("focus", "Focus", "Core"),
    ("retain_hand", "Retain Hand", "Engine"),
    ("temporary_strength", "Temporary Strength", "Engine"),
    ("temporary_dexterity", "Temporary Dexterity", "Engine"),
    ("next_turn_energy", "Next Turn Energy", "Engine"),
    ("next_turn_draw", "Next Turn Draw", "Engine"),
    ("next_turn_block", "Next Turn Block", "Engine"),
)
ORB_OPTIONS: tuple[tuple[str, str], ...] = (
    ("lightning", "Lightning"),
    ("frost", "Frost"),
    ("dark", "Dark"),
    ("plasma", "Plasma"),
    ("glass", "Glass"),
    ("random_orb", "Random"),
)


def create_combat_state(
    *,
    seed: int,
    ascension: int,
    character_id: str = DEFAULT_CHARACTER_ID,
    relics: Sequence[str] = (),
    potions: Sequence[str] = (),
    source_data: Mapping[str, Any] | None = None,
) -> Any:
    character_id = _normalized_character_id(character_id)
    data = deepcopy(dict(source_data or _runtime_source_data(character_id)))
    state = new_run(
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        source_data=data,
    )
    start_node = MapNodeState(
        node_id="debug_start",
        act=1,
        floor=0,
        lane=0,
        kind=RoomKind.START,
    )
    combat_node = MapNodeState(
        node_id="debug_combat",
        act=1,
        floor=1,
        lane=0,
        kind=RoomKind.MONSTER,
    )
    map_state = MapState(
        act=1,
        nodes=(start_node, combat_node),
        edges=(MapEdgeState(from_id=start_node.node_id, to_id=combat_node.node_id),),
        current_node_id=start_node.node_id,
        completed_node_ids=(start_node.node_id,),
        boss_node_id=None,
    )
    starter_relics = _starting_relics_from_source(data, character_id)
    state = state.model_copy(
        update={
            "phase": RunPhase.MAP,
            "act": 1,
            "floor": 0,
            "relics": tuple(
                dict.fromkeys(
                    (
                        *state.relics,
                        *starter_relics,
                        *(_source_relic_id_to_runtime_id(relic) for relic in relics if relic),
                    )
                )
            ),
            "potions": tuple(_normalized_id(potion) for potion in potions if potion),
            "ancient": None,
            "combat": None,
            "map": map_state,
            "room_history": (),
            "replay_log": (),
        }
    )
    action = next(
        action for action in legal_actions(state)
        if action.type == ActionType.CHOOSE_NODE
        and action.target_id == combat_node.node_id
    )
    return step(state, action)


def _state_payload(state: Any, *, seed: int, message: str) -> dict[str, Any]:
    actions = [_action_payload(action) for action in legal_actions(state)]
    combat = state.combat
    return {
        "seed": seed,
        "character_id": state.character_id,
        "ascension": state.ascension,
        "character_options": _character_options_payload(),
        "phase": _value(state.phase),
        "message": message,
        "flags": dict(state.flags),
        "player": _player_payload(state.player),
        "potions": list(state.potions),
        "relics": list(state.relics),
        "master_deck": [_card_payload(card) for card in state.master_deck],
        "combat": _combat_payload(combat),
        "actions": actions,
        "event_log": _event_log(state),
        "card_library": _card_library_payload(),
        "status_options": _status_options_payload(),
        "relic_options": _relic_options_payload(),
        "potion_options": _potion_options_payload(),
        "monster_options": _monster_options_payload(),
        "orb_options": _orb_options_payload(),
    }


def _combat_payload(combat: Any | None) -> dict[str, Any] | None:
    if combat is None:
        return None
    return {
        "turn": combat.turn,
        "player": _player_payload(combat.player),
        "monsters": [_monster_payload(monster) for monster in combat.monsters],
        "hand": [_card_payload(card) for card in combat.hand],
        "draw_pile": [_card_payload(card) for card in combat.draw_pile],
        "discard_pile": [_card_payload(card) for card in combat.discard_pile],
        "exhaust_pile": [_card_payload(card) for card in combat.exhaust_pile],
        "orbs": [_orb_payload(orb) for orb in combat.orbs],
        "orb_slots": combat.orb_slots,
        "orb_slots_open": max(0, combat.orb_slots - len(combat.orbs)),
        "draw_count": len(combat.draw_pile),
        "discard_count": len(combat.discard_pile),
        "exhaust_count": len(combat.exhaust_pile),
        "cards_played_this_turn": list(combat.cards_played_this_turn),
        "draw_per_turn": combat.draw_per_turn,
        "metadata": dict(combat.metadata),
        "last_events": [_event_payload(event) for event in combat.last_events],
    }


def _orb_payload(orb: Any) -> dict[str, Any]:
    return {
        "orb_id": orb.orb_id,
        "name": _display_name(orb.orb_id),
        "value": orb.value,
    }


def _player_payload(player: PlayerState) -> dict[str, Any]:
    return {
        "hp": player.hp,
        "max_hp": player.max_hp,
        "block": player.block,
        "energy": player.energy,
        "max_energy": player.max_energy,
        "gold": player.gold,
        "statuses": dict(player.statuses),
        "resources": dict(player.resources),
    }


def _monster_payload(monster: MonsterState) -> dict[str, Any]:
    return {
        "monster_id": monster.monster_id,
        "name": monster.name,
        "hp": monster.hp,
        "max_hp": monster.max_hp,
        "block": monster.block,
        "intent": monster.intent,
        "intent_damage": monster.intent_damage,
        "intent_block": monster.intent_block,
        "move_id": monster.move_id,
        "hit_count": monster.hit_count,
        "statuses": dict(monster.statuses),
        "metadata": dict(monster.metadata),
        "alive": monster.hp > 0,
    }


def _card_payload(card: CardInstance) -> dict[str, Any]:
    return {
        "instance_id": card.instance_id,
        "card_id": card.card_id,
        "name": card.name or _display_name(card.card_id),
        "type": _value(card.type),
        "cost": card.cost,
        "target": _value(card.target),
        "effects": dict(card.effects),
        "tags": list(card.tags),
        "exhausts": card.exhausts,
        "upgraded": card.upgraded,
        "custom": dict(card.custom),
    }


def _action_payload(action: Action) -> dict[str, Any]:
    return {
        "type": _value(action.type),
        "card_instance_id": action.card_instance_id,
        "target_id": action.target_id,
        "payload": dict(action.payload),
    }


def _event_log(state: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for entry in state.replay_log[-8:]:
        for event in entry.events:
            events.append(_event_payload(event))
    if state.combat is not None:
        events.extend(_event_payload(event) for event in state.combat.last_events[-12:])
    return events[-24:]


def _event_payload(event: EffectEvent) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "source_id": event.source_id,
        "target_id": event.target_id,
        "amount": event.amount,
        "message": event.message,
        "metadata": dict(event.metadata),
    }


def apply_engine_action(state: Any, action_payload: Mapping[str, Any]) -> tuple[Any, str]:
    try:
        requested = Action.model_validate(dict(action_payload))
    except Exception as exc:
        return state, f"Invalid action payload: {exc}"
    for action in legal_actions(state):
        if _same_action(action, requested):
            next_state = step(state, action)
            next_state = _apply_debug_flags(next_state)
            return next_state, _event_message(next_state)
    return state, f"Action is not legal right now: {requested.type.value}"


def apply_debug_action(state: Any, payload: Mapping[str, Any]) -> tuple[Any, str]:
    action = _normalized_id(payload.get("action", ""))
    if action == "toggle_infinite_energy":
        enabled = bool(payload.get("enabled", False))
        flags = dict(state.flags)
        flags["debug_infinite_energy"] = enabled
        state = state.model_copy(update={"flags": flags})
        state = _apply_debug_flags(state)
        return state, "Infinite energy enabled." if enabled else "Infinite energy disabled."
    if action == "heal_full":
        return _set_player_fields(state, {"hp": state.player.max_hp}), "Player healed to full."
    if action == "energy_full":
        return _set_player_fields(state, {"energy": state.player.max_energy}), "Energy restored."
    if action == "clear_player_statuses":
        return _set_player_statuses(state, {}), "Player statuses cleared."
    if action == "set_player":
        field = str(payload.get("field", ""))
        value = _int(payload.get("value"), 0)
        return _set_player_fields(state, {field: value}), f"Player {field} set to {value}."
    if action == "add_player_status":
        status_id = _normalized_id(payload.get("status_id", "strength"))
        amount = _int(payload.get("amount"), 1)
        return _add_player_status(state, status_id, amount), f"Added {amount} {status_id}."
    if action == "add_enemy_status":
        monster_id = str(payload.get("monster_id", ""))
        status_id = _normalized_id(payload.get("status_id", "vulnerable"))
        amount = _int(payload.get("amount"), 1)
        state = _update_monster(state, monster_id, status_delta={status_id: amount})
        return state, f"Added {amount} {status_id} to {monster_id}."
    if action == "damage_enemy":
        monster_id = str(payload.get("monster_id", ""))
        amount = _int(payload.get("amount"), 10)
        state = _update_monster(state, monster_id, hp_delta=-abs(amount))
        return state, f"Dealt {amount} debug damage to {monster_id}."
    if action == "kill_enemy":
        monster_id = str(payload.get("monster_id", ""))
        return _update_monster(state, monster_id, hp=0), f"Killed {monster_id}."
    if action == "kill_all":
        return _kill_all_monsters(state), "All monsters set to 0 HP."
    if action == "set_enemy":
        monster_id = str(payload.get("monster_id", ""))
        field = str(payload.get("field", "hp"))
        value = _int(payload.get("value"), 1)
        return _update_monster_field(state, monster_id, field, value), f"Enemy {field} set."
    if action == "spawn_monster":
        return _spawn_monster(state, payload), "Spawned debug monster."
    if action == "add_card":
        return _add_card_to_zone(state, payload), "Added debug card."
    if action == "draw_cards":
        amount = _int(payload.get("amount"), 1)
        return _debug_draw_cards(state, amount), f"Drew {amount} card(s)."
    if action == "set_orb_slots":
        if state.combat is None:
            return state, "No active combat."
        slots = min(10, max(0, _int(payload.get("slots"), state.combat.orb_slots)))
        delta = slots - state.combat.orb_slots
        state = _apply_debug_orb_effect(state, {"orb_slot_delta": delta})
        return state, f"Orb slots set to {slots}."
    if action == "channel_orb":
        orb_id = _normalized_id(payload.get("orb_id", "lightning"))
        amount = max(1, _int(payload.get("amount"), 1))
        target_id = str(payload.get("target_id", "")) or None
        state = _apply_debug_orb_effect(
            state,
            {"channel_orb": {"orb": orb_id, "amount": amount}},
            target_id=target_id,
        )
        return state, f"Channeled {amount} {orb_id} orb(s)."
    if action == "evoke_orb":
        selector = _normalized_id(payload.get("selector", "leftmost"))
        amount: int | str = _int(payload.get("amount"), 1)
        if selector == "all":
            amount = "all"
        target_id = str(payload.get("target_id", "")) or None
        state = _apply_debug_orb_effect(
            state,
            {"evoke_orb": {"selector": selector, "amount": amount}},
            target_id=target_id,
        )
        return state, f"Evoked {selector} orb(s)."
    if action == "clear_orbs":
        return _clear_debug_orbs(state), "Cleared combat orbs."
    if action == "add_relic":
        relic_id = _normalized_id(payload.get("relic_id", "anchor"))
        relics = tuple(dict.fromkeys((*state.relics, relic_id)))
        return state.model_copy(update={"relics": relics}), f"Added relic {relic_id}."
    if action == "add_potion":
        potion_id = _normalized_id(payload.get("potion_id", "fire_potion"))
        return state.model_copy(update={"potions": (*state.potions, potion_id)}), "Potion added."
    if action == "discard_potion_debug":
        index = _int(payload.get("index"), 0)
        potions = tuple(potion for slot, potion in enumerate(state.potions) if slot != index)
        return state.model_copy(update={"potions": potions}), "Potion removed."
    return state, f"Unknown debug action: {action or '(missing)'}"


def _same_action(left: Action, right: Action) -> bool:
    return (
        left.type == right.type
        and left.card_instance_id == right.card_instance_id
        and left.target_id == right.target_id
        and dict(left.payload) == dict(right.payload)
    )


def _apply_debug_flags(state: Any) -> Any:
    if not bool(state.flags.get("debug_infinite_energy", False)):
        return state
    return _set_player_fields(state, {"energy": 99})


def _set_player_fields(state: Any, fields: Mapping[str, int]) -> Any:
    allowed = {"hp", "max_hp", "block", "energy", "max_energy", "gold"}
    update = {key: value for key, value in fields.items() if key in allowed}
    if not update:
        return state
    player = state.player.model_copy(update=update)
    combat = state.combat
    if combat is not None:
        combat = combat.model_copy(update={"player": combat.player.model_copy(update=update)})
    return state.model_copy(update={"player": player, "combat": combat})


def _set_player_statuses(state: Any, statuses: Mapping[str, int]) -> Any:
    player = state.player.model_copy(update={"statuses": dict(statuses)})
    combat = state.combat
    if combat is not None:
        combat = combat.model_copy(
            update={"player": combat.player.model_copy(update={"statuses": dict(statuses)})}
        )
    return state.model_copy(update={"player": player, "combat": combat})


def _add_player_status(state: Any, status_id: str, amount: int) -> Any:
    statuses = dict(state.player.statuses)
    statuses[status_id] = statuses.get(status_id, 0) + amount
    return _set_player_statuses(state, statuses)


def _update_monster(
    state: Any,
    monster_id: str,
    *,
    hp: int | None = None,
    hp_delta: int = 0,
    status_delta: Mapping[str, int] | None = None,
) -> Any:
    if state.combat is None:
        return state
    monsters: list[MonsterState] = []
    for monster in state.combat.monsters:
        if monster.monster_id != monster_id:
            monsters.append(monster)
            continue
        update: dict[str, Any] = {}
        if hp is not None:
            update["hp"] = max(0, hp)
        elif hp_delta:
            update["hp"] = max(0, monster.hp + hp_delta)
        if status_delta:
            statuses = dict(monster.statuses)
            for status, amount in status_delta.items():
                statuses[status] = statuses.get(status, 0) + amount
                if statuses[status] <= 0:
                    statuses.pop(status, None)
            update["statuses"] = statuses
        monsters.append(monster.model_copy(update=update))
    combat = state.combat.model_copy(update={"monsters": tuple(monsters)})
    return state.model_copy(update={"combat": combat})


def _update_monster_field(state: Any, monster_id: str, field: str, value: int) -> Any:
    if field not in {"hp", "max_hp", "block", "intent_damage", "intent_block", "hit_count"}:
        return state
    if state.combat is None:
        return state
    monsters = tuple(
        monster.model_copy(update={field: value}) if monster.monster_id == monster_id else monster
        for monster in state.combat.monsters
    )
    combat = state.combat.model_copy(update={"monsters": monsters})
    return state.model_copy(update={"combat": combat})


def _kill_all_monsters(state: Any) -> Any:
    if state.combat is None:
        return state
    monsters = tuple(monster.model_copy(update={"hp": 0}) for monster in state.combat.monsters)
    combat = state.combat.model_copy(update={"monsters": monsters})
    return state.model_copy(update={"combat": combat})


def _spawn_monster(state: Any, payload: Mapping[str, Any]) -> Any:
    if state.combat is None:
        return state
    index = len(state.combat.monsters) + 1
    monster_id = _normalized_id(payload.get("monster_id", f"debug_monster_{index}"))
    hp = max(1, _int(payload.get("hp"), 30))
    damage = max(0, _int(payload.get("damage"), 6))
    monster = MonsterState(
        monster_id=f"{monster_id}_{index}",
        name=_display_name(monster_id),
        hp=hp,
        max_hp=hp,
        intent="attack" if damage else None,
        intent_damage=damage,
    )
    combat = state.combat.model_copy(update={"monsters": (*state.combat.monsters, monster)})
    return state.model_copy(update={"combat": combat})


def _add_card_to_zone(state: Any, payload: Mapping[str, Any]) -> Any:
    zone = _normalized_id(payload.get("zone", "hand"))
    card = _debug_card_from_payload(payload, len(state.master_deck) + _combat_card_count(state) + 1)
    if zone == "deck":
        return state.model_copy(update={"master_deck": (*state.master_deck, card)})
    if state.combat is None:
        return state
    if zone == "draw":
        combat = state.combat.model_copy(update={"draw_pile": (card, *state.combat.draw_pile)})
    elif zone == "discard":
        combat = state.combat.model_copy(
            update={"discard_pile": (*state.combat.discard_pile, card)}
        )
    elif zone == "exhaust":
        combat = state.combat.model_copy(
            update={"exhaust_pile": (*state.combat.exhaust_pile, card)}
        )
    else:
        combat = state.combat.model_copy(update={"hand": (*state.combat.hand, card)})
    return state.model_copy(update={"combat": combat})


def _debug_card_from_payload(payload: Mapping[str, Any], instance_counter: int) -> CardInstance:
    library = _cached_card_library()
    card_id = str(payload.get("card_id", "strike_debug")).strip() or "strike_debug"
    source = dict(library.get(_normalized_id(card_id), {"id": card_id}))
    if payload.get("name"):
        source["name"] = str(payload["name"])
    if payload.get("type"):
        source["type"] = str(payload["type"])
    if payload.get("target"):
        source["target"] = str(payload["target"])
    if "cost" in payload and str(payload.get("cost", "")) != "":
        source["cost"] = _int(payload.get("cost"), 0)
    effects: dict[str, Any] = {}
    damage = _optional_int(payload.get("damage"))
    block = _optional_int(payload.get("block"))
    draw = _optional_int(payload.get("draw"))
    discard_choice = _optional_int(payload.get("discard_choice"))
    discard_random = _optional_int(payload.get("discard_random"))
    orb_slot_delta = _optional_int(payload.get("orb_slot_delta"))
    channel_orb = _normalized_id(payload.get("channel_orb", ""))
    evoke_orb = _normalized_id(payload.get("evoke_orb", ""))
    if damage is not None:
        effects["damage"] = damage
        source.setdefault("type", "Attack")
        source.setdefault("target", "AnyEnemy")
    if block is not None:
        effects["block"] = block
        source.setdefault("type", "Skill")
        source.setdefault("target", "Self")
    if draw is not None:
        effects["draw"] = draw
    if discard_choice is not None:
        effects["discard_choice"] = max(0, discard_choice)
        source.setdefault("type", "Skill")
        source.setdefault("target", "Self")
    if discard_random is not None:
        effects["discard_random"] = max(0, discard_random)
        source.setdefault("type", "Skill")
        source.setdefault("target", "Self")
    if orb_slot_delta is not None:
        effects["orb_slot_delta"] = orb_slot_delta
        source.setdefault("type", "Skill")
        source.setdefault("target", "Self")
    if channel_orb:
        effects["channel_orb"] = {"orb": channel_orb, "amount": 1}
        source.setdefault("type", "Skill")
        source.setdefault("target", "Self")
    if evoke_orb:
        effects["evoke_orb"] = {"selector": evoke_orb, "amount": 1}
        source.setdefault("type", "Skill")
        source.setdefault("target", "Self")
    if effects:
        source["effects"] = effects
    source["instance_id"] = f"debug_{_normalized_id(card_id)}_{instance_counter}"
    source["upgraded"] = bool(payload.get("upgraded", False))
    custom = dict(source.get("custom", {})) if isinstance(source.get("custom"), Mapping) else {}
    if bool(payload.get("retain", False)):
        custom["retain"] = True
    if bool(payload.get("ethereal", False)):
        custom["ethereal"] = True
    if custom:
        source["custom"] = custom
    return _card_from_spec(source, instance_counter, card_library=library)


def _debug_draw_cards(state: Any, amount: int) -> Any:
    if state.combat is None:
        return state
    combat, rng_state, events = _draw_cards(state.combat, state.rng, max(0, amount))
    combat = combat.model_copy(update={"last_events": events})
    return state.model_copy(update={"combat": combat, "rng": rng_state})


def _apply_debug_orb_effect(
    state: Any,
    effect: Mapping[str, Any],
    *,
    target_id: str | None = None,
) -> Any:
    if state.combat is None:
        return state
    combat, rng_state, events = _apply_orb_effects(
        state.combat,
        state.rng,
        "debug_orb",
        effect,
        target_id=target_id,
        relics=state.relics,
    )
    if events:
        combat = combat.model_copy(update={"last_events": events})
    return state.model_copy(update={"combat": combat, "rng": rng_state})


def _clear_debug_orbs(state: Any) -> Any:
    if state.combat is None:
        return state
    event = EffectEvent(
        kind="orbs_cleared",
        source_id="debug_orb",
        target_id="player",
        amount=len(state.combat.orbs),
    )
    combat = state.combat.model_copy(update={"orbs": (), "last_events": (event,)})
    return state.model_copy(update={"combat": combat})


def _combat_card_count(state: Any) -> int:
    combat = state.combat
    if combat is None:
        return 0
    return len(combat.hand) + len(combat.draw_pile) + len(combat.discard_pile)


def _event_message(state: Any) -> str:
    if state.combat is not None and state.combat.last_events:
        return "; ".join(_event_text(event) for event in state.combat.last_events[-4:])
    if state.replay_log:
        events = state.replay_log[-1].events
        if events:
            return "; ".join(_event_text(event) for event in events[-4:])
    return "Combat ready."


def _event_text(event: EffectEvent) -> str:
    text = event.kind.replace("_", " ")
    if event.target_id:
        text += f": {event.target_id}"
    if event.amount is not None:
        text += f" ({event.amount:+d})"
    return text


class CombatTestContext:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.seed = int(args.seed)
        self.character_id = _normalized_character_id(args.character)
        self.lock = threading.Lock()
        self.message = "Combat ready."
        self.state = self._new_state()

    def _new_state(self) -> Any:
        return create_combat_state(
            seed=self.seed,
            ascension=self.args.ascension,
            character_id=self.character_id,
            relics=tuple(self.args.relic or ()),
            potions=tuple(self.args.potion or ()),
        )

    def reset(self, payload: Mapping[str, Any]) -> None:
        if payload.get("random_seed"):
            self.seed = random.randrange(1_000_000_000)
        elif payload.get("seed") not in (None, ""):
            self.seed = _int(payload.get("seed"), self.seed)
        if payload.get("character_id") not in (None, ""):
            self.character_id = _normalized_character_id(payload.get("character_id"))
        if payload.get("ascension") not in (None, ""):
            self.args.ascension = _int(payload.get("ascension"), self.args.ascension)
        self.state = self._new_state()
        self.message = "Combat reset."

    def payload(self) -> dict[str, Any]:
        return _state_payload(self.state, seed=self.seed, message=self.message)


def run_web(args: argparse.Namespace) -> None:
    context = CombatTestContext(args)
    handler = _make_handler(context)
    server = _bind_server(args.host, args.port, handler)
    host, port = server.server_address
    url = f"http://{host}:{port}"
    print(f"Combat test web UI running at {url}")
    print("Press Ctrl+C to stop it.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping combat test server.")
    finally:
        server.server_close()


def _bind_server(
    host: str,
    port: int,
    handler: type[BaseHTTPRequestHandler],
) -> ThreadingHTTPServer:
    for candidate_port in range(port, port + 20):
        try:
            return ThreadingHTTPServer((host, candidate_port), handler)
        except OSError:
            continue
    raise OSError(f"Could not bind a combat test server on {host}:{port}-{port + 19}")


def _make_handler(context: CombatTestContext) -> type[BaseHTTPRequestHandler]:
    class CombatTestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_html(_html_page())
                return
            if path == "/api/state":
                with context.lock:
                    self._send_json(context.payload())
                return
            self.send_error(404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            body = self._read_json_body()
            with context.lock:
                if path == "/api/reset":
                    context.reset(body)
                elif path == "/api/action":
                    context.state, context.message = apply_engine_action(context.state, body)
                elif path == "/api/debug":
                    context.state, context.message = apply_debug_action(context.state, body)
                    context.state = _apply_debug_flags(context.state)
                else:
                    self.send_error(404)
                    return
                self._send_json(context.payload())

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                return {}
            return payload if isinstance(payload, dict) else {}

        def _send_json(self, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_html(self, content: str) -> None:
            encoded = content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return CombatTestHandler


def _card_library_payload() -> list[dict[str, Any]]:
    rows = _cached_card_rows()
    items: list[dict[str, Any]] = []
    for row in rows:
        card_id = str(row.get("id", row.get("card_id", "")))
        if not card_id:
            continue
        statuses = _card_status_ids(row)
        items.append(
            {
                "id": _normalized_id(card_id),
                "name": str(row.get("name", _display_name(card_id))),
                "type": str(row.get("type", row.get("card_type", ""))),
                "rarity": str(row.get("rarity", "")),
                "statuses": list(statuses),
                "search_text": " ".join(
                    (
                        _normalized_id(card_id),
                        _normalized_id(row.get("name", "")),
                        _normalized_id(row.get("type", row.get("card_type", ""))),
                        _normalized_id(row.get("rarity", "")),
                        " ".join(statuses),
                    )
                ),
            }
        )
    return items


def _card_status_ids(row: Mapping[str, Any]) -> tuple[str, ...]:
    statuses: list[str] = []

    def add_status(value: object) -> None:
        status_id = _normalized_id(value)
        if status_id and status_id not in statuses:
            statuses.append(status_id)

    powers_applied = row.get("powers_applied")
    if isinstance(powers_applied, Sequence) and not isinstance(
        powers_applied,
        (str, bytes, bytearray),
    ):
        for power in powers_applied:
            if not isinstance(power, Mapping):
                continue
            add_status(power.get("power_key", power.get("power", power.get("id"))))

    effects = row.get("effects", row.get("effect"))
    if isinstance(effects, Mapping):
        _collect_statuses_from_effects(effects, add_status)

    description = str(row.get("description", row.get("description_raw", "")) or "")
    status_pattern = r"\b(?:apply|gain)\s+\d*\s*\[gold\]([^\[]+?)\[/gold\]"
    for match in re.finditer(status_pattern, description, re.I):
        add_status(match.group(1))
    return tuple(statuses)


def _collect_statuses_from_effects(
    effect: Mapping[str, Any],
    add_status: Any,
) -> None:
    sequence = effect.get("sequence", effect.get("effects"))
    if isinstance(sequence, Sequence) and not isinstance(sequence, (str, bytes, bytearray)):
        for item in sequence:
            if isinstance(item, Mapping):
                _collect_statuses_from_effects(item, add_status)
    apply_status = effect.get("apply_status", effect.get("status"))
    if isinstance(apply_status, Mapping):
        for key in ("status_id", "status", "power", "power_key", "id"):
            if key in apply_status:
                add_status(apply_status[key])
    elif apply_status is not None:
        add_status(apply_status)


def _character_options_payload() -> list[dict[str, str]]:
    rows = _cached_character_rows()
    items: list[dict[str, str]] = []
    for row in rows:
        character_id = str(row.get("id", "")).strip().upper()
        if not character_id:
            continue
        items.append(
            {
                "id": character_id,
                "name": str(row.get("name", _display_name(character_id))),
            }
        )
    if items:
        return items
    return [{"id": DEFAULT_CHARACTER_ID, "name": _display_name(DEFAULT_CHARACTER_ID)}]


def _status_options_payload() -> list[dict[str, str]]:
    return _source_options_payload(
        _cached_power_rows(),
        fallback=COMMON_STATUS_OPTIONS,
    )


def _relic_options_payload() -> list[dict[str, str]]:
    return _source_options_payload(_cached_relic_rows())


def _potion_options_payload() -> list[dict[str, str]]:
    return _source_options_payload(_cached_potion_rows())


def _monster_options_payload() -> list[dict[str, str]]:
    return _source_options_payload(
        _cached_monster_rows(),
        fallback=(("debug_enemy", "Debug Enemy", "Debug"),),
    )


def _orb_options_payload() -> list[dict[str, str]]:
    return [
        {"id": orb_id, "name": name, "group": "Orb"}
        for orb_id, name in ORB_OPTIONS
    ]


def _source_options_payload(
    rows: Sequence[Mapping[str, Any]],
    *,
    fallback: Sequence[tuple[str, str, str]] = (),
) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    seen: set[str] = set()

    def add_option(option_id: str, name: str, group: str = "") -> None:
        normalized = _normalized_id(option_id)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        options.append(
            {
                "id": normalized,
                "name": name or _display_name(normalized),
                "group": group,
            }
        )

    for option_id, name, group in fallback:
        add_option(option_id, name, group)
    for row in rows:
        source_id = str(row.get("id", row.get("card_id", ""))).strip()
        if not source_id:
            continue
        group = str(row.get("type", row.get("rarity", row.get("pool", ""))) or "")
        add_option(source_id, str(row.get("name", _display_name(source_id))), group)
    return sorted(options, key=lambda item: (item["name"].lower(), item["id"]))


def _cached_card_library() -> dict[str, Mapping[str, Any]]:
    library: dict[str, Mapping[str, Any]] = {}
    for row in _cached_card_rows():
        card_id = str(row.get("id", row.get("card_id", "")))
        if not card_id:
            continue
        library[_normalized_id(card_id)] = row
        library[card_id] = row
        library[card_id.upper()] = row
    return library


def _cached_card_rows() -> tuple[Mapping[str, Any], ...]:
    return _cached_json_rows(DEFAULT_CARD_CACHE)


def _cached_character_rows() -> tuple[Mapping[str, Any], ...]:
    return _cached_json_rows(DEFAULT_CHARACTER_CACHE)


def _cached_monster_rows() -> tuple[Mapping[str, Any], ...]:
    return _cached_json_rows(DEFAULT_MONSTER_CACHE)


def _cached_potion_rows() -> tuple[Mapping[str, Any], ...]:
    return _cached_json_rows(DEFAULT_POTION_CACHE)


def _cached_power_rows() -> tuple[Mapping[str, Any], ...]:
    return _cached_json_rows(DEFAULT_POWER_CACHE)


def _cached_relic_rows() -> tuple[Mapping[str, Any], ...]:
    return _cached_json_rows(DEFAULT_RELIC_CACHE)


def _cached_json_rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        return ()
    return tuple(row for row in payload if isinstance(row, Mapping))


def _runtime_source_data(character_id: str) -> dict[str, Any]:
    cards_by_id = _source_cards_by_id()
    character_rows = _cached_character_rows()
    characters: dict[str, dict[str, Any]] = {}
    for row in character_rows:
        runtime_id = _normalized_character_id(row.get("id", ""))
        if not runtime_id:
            continue
        starter_deck = tuple(
            _runtime_card_spec(raw_card_id, cards_by_id)
            for raw_card_id in _string_sequence(row.get("starting_deck"))
        )
        player = _runtime_player_spec(row)
        characters[runtime_id] = {
            "starter_deck": starter_deck,
            "player": player,
            "starting_relics": tuple(
                _source_relic_id_to_runtime_id(relic_id)
                for relic_id in _string_sequence(row.get("starting_relics"))
            ),
        }
    if not characters:
        return deepcopy(FALLBACK_SOURCE_DATA)
    if character_id not in characters:
        character_id = (
            DEFAULT_CHARACTER_ID
            if DEFAULT_CHARACTER_ID in characters
            else next(iter(characters))
        )
    return {
        "characters": {character_id: characters[character_id]},
        "flags": {**DEFAULT_SOURCE_FLAGS},
    }


def _source_cards_by_id() -> dict[str, Mapping[str, Any]]:
    cards: dict[str, Mapping[str, Any]] = {}
    for row in _cached_card_rows():
        card_id = str(row.get("id", row.get("card_id", ""))).strip()
        if not card_id:
            continue
        cards[card_id.upper()] = row
    return cards


def _runtime_card_spec(
    source_card_id: str,
    cards_by_id: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    runtime_id = _source_card_id_to_runtime_id(source_card_id, cards_by_id.keys())
    row = cards_by_id.get(runtime_id.upper())
    if row is None:
        return {"card_id": runtime_id}
    spec = dict(row)
    spec["card_id"] = str(row.get("id", runtime_id))
    return spec


def _runtime_player_spec(character_row: Mapping[str, Any]) -> dict[str, int]:
    max_hp = _int(character_row.get("starting_hp"), 80)
    max_energy = _int(character_row.get("max_energy"), 3)
    return {
        "hp": max_hp,
        "max_hp": max_hp,
        "energy": max_energy,
        "max_energy": max_energy,
        "gold": _int(character_row.get("starting_gold"), 99),
    }


def _starting_relics_from_source(
    source_data: Mapping[str, Any],
    character_id: str,
) -> tuple[str, ...]:
    characters = source_data.get("characters")
    if not isinstance(characters, Mapping):
        return ()
    character_source = characters.get(_normalized_character_id(character_id))
    if not isinstance(character_source, Mapping):
        return ()
    return tuple(
        _source_relic_id_to_runtime_id(relic_id)
        for relic_id in _string_sequence(character_source.get("starting_relics"))
    )


def _source_card_id_to_runtime_id(source_card_id: str, known_card_ids: Sequence[str]) -> str:
    raw = str(source_card_id).strip()
    known = {str(card_id).upper() for card_id in known_card_ids}
    candidates = (
        raw,
        raw.upper(),
        _camel_to_screaming_snake(raw),
        _normalized_id(raw).upper(),
    )
    for candidate in candidates:
        if candidate.upper() in known:
            return candidate.upper()
    return candidates[2] or raw


def _source_relic_id_to_runtime_id(source_relic_id: object) -> str:
    return _camel_to_snake(str(source_relic_id)).lower()


def _normalized_character_id(value: object) -> str:
    character_id = str(value or DEFAULT_CHARACTER_ID).strip().upper()
    return character_id or DEFAULT_CHARACTER_ID


def _camel_to_screaming_snake(value: str) -> str:
    return _camel_to_snake(value).upper()


def _camel_to_snake(value: str) -> str:
    stripped = str(value or "").strip().replace("-", "_").replace(" ", "_")
    first_pass = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", stripped)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass)


def _string_sequence(value: object) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value if str(item).strip())
    if value in (None, ""):
        return ()
    return (str(value),)


def _display_name(item_id: str) -> str:
    return str(item_id).replace("_", " ").title()


def _value(value: Any) -> Any:
    enum_value = getattr(value, "value", None)
    return enum_value if enum_value is not None else value


def _int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return _int(value, 0)


def _int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    parsed: dict[str, int] = {}
    for key, raw_value in value.items():
        parsed[_normalized_id(key)] = _int(raw_value, 0)
    return parsed


def _normalized_id(value: object) -> str:
    return str(value or "").strip().lower().replace("'", "").replace(" ", "_").replace("-", "_")


def _html_page() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STS2 Combat Lab</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fa;
      --surface: #ffffff;
      --surface-2: #eef3f6;
      --line: #d8e0e6;
      --text: #16212a;
      --muted: #61707d;
      --teal: #0f8a8a;
      --teal-2: #dff5f2;
      --red: #c73a4a;
      --red-2: #fde8eb;
      --amber: #a66a00;
      --shadow: 0 18px 42px rgba(21, 34, 45, .08);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); }
    button, input, select {
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
      color: var(--text);
      min-height: 32px;
    }
    button {
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 700;
    }
    button:hover { border-color: var(--teal); color: var(--teal); }
    button.primary { background: var(--teal); border-color: var(--teal); color: #fff; }
    button.danger { background: var(--red-2); border-color: #efb8bf; color: var(--red); }
    input, select { width: 100%; padding: 6px 8px; font-size: 12px; }
    label { display: grid; gap: 4px; font-size: 11px; color: var(--muted); font-weight: 700; }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 260px minmax(520px, 1fr) 360px;
      gap: 14px;
      padding: 14px;
    }
    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .rail, .inspect { padding: 14px; display: flex; flex-direction: column; gap: 12px; }
    .main {
      display: grid;
      grid-template-rows: auto auto minmax(260px, 1fr);
      overflow: hidden;
    }
    .topbar {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    h1, h2, h3 { margin: 0; line-height: 1.1; }
    h1 { font-size: 17px; }
    h2 { font-size: 13px; margin-bottom: 8px; }
    h3 { font-size: 12px; color: var(--muted); text-transform: uppercase; }
    .muted { color: var(--muted); font-size: 12px; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    .stack { display: flex; flex-direction: column; gap: 8px; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .stat-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(70px, 1fr));
      gap: 8px;
    }
    .stat {
      background: var(--surface-2);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      min-height: 50px;
    }
    .stat b { display: block; font-size: 11px; color: var(--muted); }
    .stat span { font-size: 16px; font-weight: 800; }
    .arena { padding: 14px; overflow: auto; display: grid; gap: 14px; }
    .monster-table { display: grid; gap: 8px; }
    .monster {
      display: grid;
      grid-template-columns: minmax(120px, 1fr) 110px 110px 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
    }
    .bar { height: 8px; border-radius: 999px; background: #e8edf1; overflow: hidden; }
    .bar > i { display: block; height: 100%; background: var(--red); }
    .badges { display: flex; flex-wrap: wrap; gap: 5px; }
    .badge {
      padding: 3px 6px;
      border-radius: 999px;
      background: var(--teal-2);
      color: #075f61;
      font-size: 11px;
      font-weight: 800;
    }
    .badge.red { background: var(--red-2); color: var(--red); }
    .hand {
      padding: 12px 14px;
      border-top: 1px solid var(--line);
      background: #fbfcfd;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      align-content: start;
      overflow: auto;
    }
    .card {
      min-height: 132px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .card .name { font-weight: 900; font-size: 13px; }
    .card .meta { font-size: 11px; color: var(--muted); }
    .card pre.meta {
      max-height: 54px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .card .actions { margin-top: auto; display: flex; flex-wrap: wrap; gap: 6px; }
    .section {
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }
    .section:first-child { border-top: 0; padding-top: 0; }
    .log {
      max-height: 180px;
      overflow: auto;
      background: #0f1720;
      color: #dce8ef;
      border-radius: 8px;
      padding: 10px;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 11px;
      line-height: 1.45;
    }
    .raw {
      max-height: 210px;
      overflow: auto;
      background: #101820;
      color: #d8e6ef;
      border-radius: 8px;
      padding: 10px;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 10px;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 11px;
      font-weight: 800;
      background: var(--surface-2);
    }
    .debug-tools { display: none; }
    body.debug-open .debug-tools { display: block; }
    body.debug-open .debug-tools.stack { display: flex; }
    body.debug-open button.debug-tools { display: inline-block; }
    .pile-tabs {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 6px;
    }
    .pile-tabs button.active {
      border-color: var(--teal);
      background: var(--teal-2);
      color: #075f61;
    }
    .pile-list {
      max-height: 260px;
      overflow: auto;
      display: grid;
      gap: 6px;
    }
    .pile-card {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px;
      background: #fbfcfd;
      display: grid;
      gap: 3px;
    }
    .pile-card b { font-size: 12px; }
    .pile-card span { color: var(--muted); font-size: 11px; }
    .orb-belt {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(86px, 1fr));
      gap: 8px;
    }
    .orb-slot {
      min-height: 64px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 8px;
      display: grid;
      gap: 4px;
      align-content: start;
    }
    .orb-slot.filled { border-color: #b8d9da; background: var(--teal-2); }
    .orb-slot b { font-size: 12px; }
    .orb-slot span { font-size: 11px; color: var(--muted); }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 1fr; }
      .monster { grid-template-columns: 1fr 1fr; }
      .stat-grid { grid-template-columns: repeat(3, 1fr); }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="panel rail">
      <section class="section">
        <h1>Combat Lab</h1>
        <div id="message" class="muted">Loading...</div>
      </section>
      <section class="section stack">
        <h2>Run</h2>
        <label>Character<select id="character"></select></label>
        <div class="grid2">
          <label>Seed<input id="seed" type="number"></label>
          <label>Ascension<input id="ascension" type="number" value="0"></label>
        </div>
        <button class="primary" onclick="reset(false)">Reset Combat</button>
        <button onclick="reset(true)">Random Seed</button>
        <button id="debug-toggle" onclick="toggleDebugTools()">Show Debug Tools</button>
      </section>
      <section class="section stack debug-tools">
        <h2>Debug Toggles</h2>
        <button id="energy-toggle" onclick="toggleEnergy()">Infinite Energy Off</button>
        <button onclick="debugAction({action:'heal_full'})">Heal Full</button>
        <button onclick="debugAction({action:'energy_full'})">Energy Full</button>
        <button onclick="debugAction({action:'kill_all'})" class="danger">Kill All</button>
        <button onclick="debugAction({action:'clear_player_statuses'})">Clear Statuses</button>
      </section>
      <section class="section stack debug-tools">
        <h2>Quick Effects</h2>
        <div class="grid2">
          <button onclick="quickStatus('strength',1)">+ Strength</button>
          <button onclick="quickStatus('dexterity',1)">+ Dexterity</button>
          <button onclick="quickEnemyStatus('vulnerable',1)">Vuln Enemy</button>
          <button onclick="quickEnemyStatus('weak',1)">Weak Enemy</button>
        </div>
      </section>
      <section class="section">
        <h2>Piles</h2>
        <div id="piles" class="row"></div>
      </section>
    </aside>

    <main class="panel main">
      <div class="topbar">
        <div>
          <h1 id="phase">combat</h1>
          <div id="turn" class="muted"></div>
        </div>
        <div class="row">
          <button onclick="engineAction(endTurnAction())">End Turn</button>
          <button class="debug-tools" onclick="debugAction({action:'draw_cards', amount: 5})">
            Draw 5
          </button>
        </div>
      </div>
      <div class="arena">
        <section>
          <div class="stat-grid" id="stats"></div>
        </section>
        <section>
          <h2>Orbs</h2>
          <div id="orbs" class="orb-belt"></div>
        </section>
        <section>
          <h2>Monsters</h2>
          <div id="monsters" class="monster-table"></div>
        </section>
      </div>
      <section class="hand" id="hand"></section>
    </main>

    <aside class="panel inspect">
      <section class="section stack">
        <h2>Pile Inspector</h2>
        <div class="pile-tabs" id="pile-tabs"></div>
        <div id="pile-list" class="pile-list"></div>
      </section>

      <section class="section stack debug-tools">
        <h2>Add Card</h2>
        <label>Card Search<input id="card-id" list="card-options" value="strike_debug"></label>
        <datalist id="card-options"></datalist>
        <datalist id="status-options"></datalist>
        <datalist id="monster-options"></datalist>
        <datalist id="relic-options"></datalist>
        <datalist id="potion-options"></datalist>
        <datalist id="orb-options"></datalist>
        <div class="grid3">
          <label>Zone<select id="card-zone">
            <option>hand</option><option>draw</option><option>discard</option>
            <option>deck</option><option>exhaust</option>
          </select></label>
          <label>Cost<input id="card-cost" type="number" placeholder=""></label>
          <label>Damage<input id="card-damage" type="number" placeholder=""></label>
        </div>
        <div class="grid2">
          <label>Block<input id="card-block" type="number" placeholder=""></label>
          <label>Draw<input id="card-draw" type="number" placeholder=""></label>
        </div>
        <div class="grid2">
          <label>Choose Discard<input id="card-discard-choice" type="number" placeholder=""></label>
          <label>Random Discard<input id="card-discard-random" type="number" placeholder=""></label>
        </div>
        <div class="grid3">
          <label>Channel Orb<select id="card-channel-orb"></select></label>
          <label>Evoke<select id="card-evoke-orb">
            <option value="">None</option>
            <option value="leftmost">Leftmost</option>
            <option value="rightmost">Rightmost</option>
            <option value="all">All</option>
          </select></label>
          <label>Slot Delta<input id="card-orb-slot-delta" type="number" placeholder=""></label>
        </div>
        <div class="row">
          <label class="row"><input id="card-retain" type="checkbox"> Retain</label>
          <label class="row"><input id="card-ethereal" type="checkbox"> Ethereal</label>
          <button class="primary" onclick="addCard()">Add Card</button>
        </div>
      </section>

      <section class="section stack debug-tools">
        <h2>Player / Enemy</h2>
        <div class="grid3">
          <label>Field<select id="player-field">
            <option>hp</option><option>max_hp</option><option>block</option>
            <option>energy</option><option>max_energy</option><option>gold</option>
          </select></label>
          <label>Value<input id="player-value" type="number" value="99"></label>
          <button onclick="setPlayer()">Set Player</button>
        </div>
        <div class="grid3">
          <label>Status<input id="status-id" list="status-options" value="strength"></label>
          <label>Amount<input id="status-amount" type="number" value="1"></label>
          <button onclick="addPlayerStatus()">Add Player Status</button>
        </div>
        <div class="grid3">
          <label>Enemy<select id="enemy-id"></select></label>
          <label>Amount<input id="enemy-amount" type="number" value="10"></label>
          <button onclick="damageEnemy()">Damage Enemy</button>
        </div>
        <div class="grid3">
          <label>Enemy Status
            <input id="enemy-status-id" list="status-options" value="vulnerable">
          </label>
          <label>Amount<input id="enemy-status-amount" type="number" value="1"></label>
          <button onclick="addEnemyStatus()">Add Enemy Status</button>
        </div>
      </section>

      <section class="section stack debug-tools">
        <h2>Orbs</h2>
        <div class="grid3">
          <label>Slots<input id="orb-slots" type="number" min="0" max="10" value="3"></label>
          <button onclick="setOrbSlots()">Set Slots</button>
          <button onclick="debugAction({action:'clear_orbs'})">Clear Orbs</button>
        </div>
        <div class="grid3">
          <label>Orb<select id="orb-id"></select></label>
          <label>Amount<input id="orb-amount" type="number" min="1" value="1"></label>
          <button onclick="channelOrb()">Channel</button>
        </div>
        <div class="grid3">
          <label>Evoke<select id="evoke-selector">
            <option value="leftmost">Leftmost</option>
            <option value="rightmost">Rightmost</option>
            <option value="all">All</option>
          </select></label>
          <label>Amount<input id="evoke-amount" type="number" min="1" value="1"></label>
          <button onclick="evokeOrb()">Evoke</button>
        </div>
      </section>

      <section class="section stack debug-tools">
        <h2>Spawn / Items</h2>
        <div class="grid3">
          <label>ID<input id="spawn-id" list="monster-options" value="debug_enemy"></label>
          <label>HP<input id="spawn-hp" type="number" value="30"></label>
          <label>Damage<input id="spawn-damage" type="number" value="6"></label>
        </div>
        <button onclick="spawnMonster()">Spawn Monster</button>
        <div class="grid2">
          <label>Relic<input id="relic-id" list="relic-options" value="anchor"></label>
          <button onclick="debugAction({action:'add_relic', relic_id: val('relic-id')})">
            Add Relic
          </button>
        </div>
        <div class="grid2">
          <label>Potion<input id="potion-id" list="potion-options" value="fire_potion"></label>
          <button onclick="debugAction({action:'add_potion', potion_id: val('potion-id')})">
            Add Potion
          </button>
        </div>
      </section>

      <section class="section">
        <h2>Event Log</h2>
        <div id="log" class="log"></div>
      </section>
      <section class="section debug-tools">
        <h2>Raw Combat</h2>
        <pre id="raw" class="raw"></pre>
      </section>
    </aside>
  </div>

  <script>
    let current = null;
    let selectedEnemy = null;
    let selectedPile = "draw_pile";
    let debugOpen = false;

    const el = (id) => document.getElementById(id);
    const val = (id) => el(id).value;
    const numberVal = (id) => {
      const value = val(id);
      return value === "" ? null : Number(value);
    };

    async function api(path, body=null) {
      const options = body === null ? {} : {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      };
      const response = await fetch(path, options);
      current = await response.json();
      render();
    }

    function reset(randomSeed) {
      api("/api/reset", {
        seed: val("seed"),
        ascension: Number(val("ascension") || 0),
        character_id: val("character"),
        random_seed: randomSeed,
      });
    }

    function debugAction(body) { api("/api/debug", body); }
    function engineAction(action) { if (action) api("/api/action", action); }

    function render() {
      if (!current) return;
      el("seed").value = current.seed;
      el("ascension").value = current.ascension;
      el("phase").textContent = `Phase: ${current.phase}`;
      el("message").textContent = current.message || "";
      const combat = current.combat;
      el("turn").textContent = combat ? `Turn ${combat.turn}` : "No active combat";
      renderDebugVisibility();
      renderCharacterOptions();
      renderCardOptions();
      renderSourceOptionLists();
      renderOrbSelects();
      renderStats();
      renderOrbs();
      renderPiles();
      renderPileInspector();
      renderMonsters();
      renderHand();
      renderLog();
      el("raw").textContent = JSON.stringify(combat || current, null, 2);
      const enabled = Boolean(current.flags?.debug_infinite_energy);
      el("energy-toggle").textContent = enabled ? "Infinite Energy On" : "Infinite Energy Off";
    }

    function renderDebugVisibility() {
      document.body.classList.toggle("debug-open", debugOpen);
      el("debug-toggle").textContent = debugOpen ? "Hide Debug Tools" : "Show Debug Tools";
    }

    function renderCharacterOptions() {
      const options = current.character_options || [];
      el("character").innerHTML = options.map((character) =>
        `<option value="${character.id}">${character.name}</option>`
      ).join("");
      el("character").value = current.character_id || "IRONCLAD";
    }

    function renderCardOptions() {
      el("card-options").innerHTML = (current.card_library || [])
        .map((card) => {
          const applies = card.statuses?.length ? ` | applies ${card.statuses.join(", ")}` : "";
          const cardType = card.type || "unknown";
          const rarity = card.rarity || "unknown";
          const label = `${card.name} | ${cardType} | ${rarity}${applies}`;
          return `<option value="${card.id}" label="${label}">${label}</option>`;
        })
        .join("");
    }

    function renderSourceOptionLists() {
      renderDataList("status-options", current.status_options || []);
      renderDataList("monster-options", current.monster_options || []);
      renderDataList("relic-options", current.relic_options || []);
      renderDataList("potion-options", current.potion_options || []);
      renderDataList("orb-options", current.orb_options || []);
    }

    function renderOrbSelects() {
      const options = current.orb_options || [];
      const previousOrb = el("orb-id").value || "lightning";
      const previousCardOrb = el("card-channel-orb").value || "";
      el("orb-id").innerHTML = options.map((option) =>
        `<option value="${option.id}">${option.name}</option>`
      ).join("");
      el("card-channel-orb").innerHTML = [
        `<option value="">None</option>`,
        ...options.map((option) => `<option value="${option.id}">${option.name}</option>`),
      ].join("");
      el("orb-id").value = previousOrb;
      el("card-channel-orb").value = previousCardOrb;
    }

    function renderDataList(id, options) {
      el(id).innerHTML = options.map((option) => {
        const label = option.group ? `${option.name} - ${option.group}` : option.name;
        return `<option value="${option.id}">${label}</option>`;
      }).join("");
    }

    function renderStats() {
      const player = current.combat?.player || current.player;
      const stats = [
        ["HP", `${player.hp}/${player.max_hp}`],
        ["Block", player.block],
        ["Energy", `${player.energy}/${player.max_energy}`],
        ["Gold", player.gold],
        ["Statuses", statusText(player.statuses)],
        ["Potions", current.potions.length],
        ["Orbs", orbSummary(current.combat)],
      ];
      el("stats").innerHTML = stats.map(([name, value]) =>
        `<div class="stat"><b>${name}</b><span>${value || "-"}</span></div>`
      ).join("");
    }

    function renderOrbs() {
      const combat = current.combat;
      if (!combat) {
        el("orbs").innerHTML = `<div class="muted">No active combat.</div>`;
        return;
      }
      el("orb-slots").value = combat.orb_slots;
      const slots = [];
      for (let index = 0; index < combat.orb_slots; index += 1) {
        const orb = combat.orbs[index];
        slots.push(orb ? `
          <div class="orb-slot filled">
            <b>${index + 1}. ${orb.name}</b>
            <span>${orb.orb_id}</span>
            <span>Value ${orb.value}</span>
          </div>
        ` : `
          <div class="orb-slot">
            <b>${index + 1}. Empty</b>
            <span>Open slot</span>
          </div>
        `);
      }
      el("orbs").innerHTML = slots.join("") || `<div class="muted">No orb slots.</div>`;
    }

    function renderPiles() {
      const combat = current.combat;
      if (!combat) {
        el("piles").innerHTML = "";
        return;
      }
      el("piles").innerHTML = [
        ["Draw", combat.draw_count],
        ["Discard", combat.discard_count],
        ["Exhaust", combat.exhaust_count],
        ["Hand", combat.hand.length],
      ].map(([label, count]) => `<span class="pill">${label}: ${count}</span>`).join("");
    }

    function renderPileInspector() {
      const piles = [
        ["master_deck", "Deck", current.master_deck || []],
        ["draw_pile", "Draw", current.combat?.draw_pile || []],
        ["discard_pile", "Discard", current.combat?.discard_pile || []],
        ["exhaust_pile", "Exhaust", current.combat?.exhaust_pile || []],
        ["hand", "Hand", current.combat?.hand || []],
      ];
      if (!piles.some(([key]) => key === selectedPile)) selectedPile = "draw_pile";
      el("pile-tabs").innerHTML = piles.map(([key, label, cards]) =>
        `<button class="${key === selectedPile ? "active" : ""}"
          onclick="setPileView('${key}')">${label} (${cards.length})</button>`
      ).join("");
      const selected = piles.find(([key]) => key === selectedPile);
      const cards = selected ? selected[2] : [];
      el("pile-list").innerHTML = cards.map((card, index) => `
        <div class="pile-card">
          <b>${index + 1}. ${card.name}</b>
          <span>${card.card_id}</span>
          <span>${card.type} | cost ${card.cost} | ${card.target}</span>
        </div>
      `).join("") || `<div class="muted">This pile is empty.</div>`;
    }

    function renderMonsters() {
      const monsters = current.combat?.monsters || [];
      const enemySelect = el("enemy-id");
      enemySelect.innerHTML = monsters.map((monster) =>
        `<option value="${monster.monster_id}">${monster.name}</option>`
      ).join("");
      if (!selectedEnemy && monsters.length) selectedEnemy = monsters[0].monster_id;
      if (selectedEnemy) enemySelect.value = selectedEnemy;
      el("monsters").innerHTML = monsters.map((monster) => {
        const hpPct = monster.max_hp ? Math.max(0, monster.hp / monster.max_hp * 100) : 0;
        return `<div class="monster">
          <div>
            <b>${monster.name}</b>
            <div class="muted">${monster.monster_id}</div>
          </div>
          <div>
            <div class="muted">HP ${monster.hp}/${monster.max_hp}</div>
            <div class="bar"><i style="width:${hpPct}%"></i></div>
          </div>
          <div><b>Block</b> ${monster.block}</div>
          <div>
            <b>Intent</b> ${monster.intent || "-"} ${monster.intent_damage || ""}
            <div class="badges">${badges(monster.statuses, "red")}</div>
          </div>
          <div class="row">
            <button class="debug-tools" onclick="selectEnemy('${monster.monster_id}')">
              Target
            </button>
            <button onclick="debugAction({action:'kill_enemy', monster_id:'${monster.monster_id}'})"
              class="danger debug-tools">Kill</button>
          </div>
        </div>`;
      }).join("") || `<div class="muted">No monsters.</div>`;
    }

    function renderHand() {
      const hand = current.combat?.hand || [];
      el("hand").innerHTML = hand.map((card) => {
        const actions = current.actions.filter((action) => {
          const combatCardAction = action.type === "play_card" || action.type === "discard_card";
          return combatCardAction && action.card_instance_id === card.instance_id;
        });
        const buttons = actions.map((action) =>
          `<button class="primary" onclick='engineAction(${JSON.stringify(action)})'>
            ${action.type === "discard_card" ? "Discard" : `Play ${action.target_id || ""}`}
          </button>`
        ).join("");
        return `<article class="card">
          <div class="name">${card.name}</div>
          <div class="meta">${card.type} | cost ${card.cost} | ${card.target}</div>
          <div class="badges">${cardBadges(card)}</div>
          <pre class="meta">${JSON.stringify(card.effects)}</pre>
          <div class="actions">${buttons || '<span class="muted">No legal play</span>'}</div>
        </article>`;
      }).join("") || `<div class="muted">No cards in hand.</div>`;
    }

    function renderLog() {
      const events = current.event_log || [];
      el("log").innerHTML = events.map((event) => {
        const amount =
          event.amount === null || event.amount === undefined ? "" : ` ${event.amount}`;
        const route = `${event.source_id || ""} -> ${event.target_id || ""}`;
        return `<div>${event.kind}${amount} ${route}</div>`;
      }).join("") || "No events yet.";
    }

    function endTurnAction() {
      return current.actions.find((action) => action.type === "end_turn");
    }

    function toggleDebugTools() {
      debugOpen = !debugOpen;
      renderDebugVisibility();
    }

    function setPileView(pileKey) {
      selectedPile = pileKey;
      renderPileInspector();
    }

    function toggleEnergy() {
      debugAction({
        action: "toggle_infinite_energy",
        enabled: !Boolean(current.flags?.debug_infinite_energy),
      });
    }

    function selectEnemy(monsterId) {
      selectedEnemy = monsterId;
      el("enemy-id").value = monsterId;
      render();
    }

    function selectedEnemyId() {
      return val("enemy-id") || selectedEnemy || current.combat?.monsters?.[0]?.monster_id || "";
    }

    function quickStatus(statusId, amount) {
      debugAction({action: "add_player_status", status_id: statusId, amount});
    }

    function quickEnemyStatus(statusId, amount) {
      debugAction({
        action: "add_enemy_status",
        monster_id: selectedEnemyId(),
        status_id: statusId,
        amount,
      });
    }

    function setPlayer() {
      debugAction({
        action: "set_player",
        field: val("player-field"),
        value: Number(val("player-value") || 0),
      });
    }

    function addPlayerStatus() {
      debugAction({
        action: "add_player_status",
        status_id: val("status-id"),
        amount: Number(val("status-amount") || 0),
      });
    }

    function addEnemyStatus() {
      debugAction({
        action: "add_enemy_status",
        monster_id: selectedEnemyId(),
        status_id: val("enemy-status-id"),
        amount: Number(val("enemy-status-amount") || 0),
      });
    }

    function setOrbSlots() {
      debugAction({
        action: "set_orb_slots",
        slots: Number(val("orb-slots") || 0),
      });
    }

    function channelOrb() {
      debugAction({
        action: "channel_orb",
        orb_id: val("orb-id"),
        amount: Number(val("orb-amount") || 1),
        target_id: selectedEnemyId(),
      });
    }

    function evokeOrb() {
      debugAction({
        action: "evoke_orb",
        selector: val("evoke-selector"),
        amount: Number(val("evoke-amount") || 1),
        target_id: selectedEnemyId(),
      });
    }

    function damageEnemy() {
      debugAction({
        action: "damage_enemy",
        monster_id: selectedEnemyId(),
        amount: Number(val("enemy-amount") || 0),
      });
    }

    function spawnMonster() {
      debugAction({
        action: "spawn_monster",
        monster_id: val("spawn-id"),
        hp: Number(val("spawn-hp") || 1),
        damage: Number(val("spawn-damage") || 0),
      });
    }

    function addCard() {
      debugAction({
        action: "add_card",
        card_id: val("card-id"),
        zone: val("card-zone"),
        cost: numberVal("card-cost"),
        damage: numberVal("card-damage"),
        block: numberVal("card-block"),
        draw: numberVal("card-draw"),
        discard_choice: numberVal("card-discard-choice"),
        discard_random: numberVal("card-discard-random"),
        channel_orb: val("card-channel-orb"),
        evoke_orb: val("card-evoke-orb"),
        orb_slot_delta: numberVal("card-orb-slot-delta"),
        retain: el("card-retain").checked,
        ethereal: el("card-ethereal").checked,
      });
    }

    function statusText(statuses) {
      const entries = Object.entries(statuses || {});
      return entries.length ? entries.map(([k, v]) => `${k} ${v}`).join(", ") : "-";
    }

    function orbSummary(combat) {
      if (!combat) return "-";
      return `${combat.orbs.length}/${combat.orb_slots}`;
    }

    function badges(statuses, color="") {
      return Object.entries(statuses || {}).map(([key, value]) =>
        `<span class="badge ${color}">${key} ${value}</span>`
      ).join("");
    }

    function cardBadges(card) {
      const tags = [];
      if (card.exhausts) tags.push("exhaust");
      if (card.custom?.retain) tags.push("retain");
      if (card.custom?.ethereal) tags.push("ethereal");
      if (card.upgraded) tags.push("upgraded");
      return tags.map((tag) => `<span class="badge">${tag}</span>`).join("");
    }

    api("/api/state");
  </script>
</body>
</html>
"""


def run_terminal(args: argparse.Namespace) -> None:
    state = create_combat_state(
        seed=args.seed,
        ascension=args.ascension,
        character_id=args.character,
        relics=tuple(args.relic or ()),
        potions=tuple(args.potion or ()),
    )
    payload = _state_payload(state, seed=args.seed, message="Combat ready.")
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive combat tester for the headless sts2sim engine."
    )
    parser.add_argument("--seed", type=int, default=12345, help="Deterministic combat seed.")
    parser.add_argument(
        "--character",
        default=DEFAULT_CHARACTER_ID,
        help="Character id, for example IRONCLAD, SILENT, DEFECT, REGENT, or NECROBINDER.",
    )
    parser.add_argument("--ascension", type=int, default=0, help="Ascension level.")
    parser.add_argument("--relic", action="append", help="Starting relic id. Repeatable.")
    parser.add_argument("--potion", action="append", help="Starting potion id. Repeatable.")
    parser.add_argument("--web", action="store_true", help="Open a small local web UI.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Web server host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Web server port.")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open a browser.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.web:
        run_web(args)
    else:
        run_terminal(args)


if __name__ == "__main__":
    main()
