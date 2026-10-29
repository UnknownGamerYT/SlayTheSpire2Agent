"""Visibility boundaries shared by policy observations and action previews."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

POLICY_OBSERVATION_SCHEMA_VERSION = 2


def draw_order_is_visible(payload: Mapping[str, Any], *, oracle: bool = False) -> bool:
    """Frozen Eye reveals draw order; debug callers can explicitly opt into it."""

    return oracle or any(
        str(relic).strip().lower().replace(" ", "_") == "frozen_eye"
        for relic in _sequence(payload.get("relics"))
    )


def policy_payload(payload: Mapping[str, Any], *, oracle: bool = False) -> dict[str, Any]:
    """Copy facts while hiding RNG, seed, history hashes and unknown draw order.

    Unordered piles retain public composition in a canonical order. This is an
    observation snapshot, not an engine save; use ``oracle=True`` for an exact
    debugging snapshot. The source payload and engine state are never mutated.
    """

    visible = deepcopy(dict(payload))
    if oracle:
        return visible
    visible.pop("rng", None)
    visible.pop("seed", None)
    visible.pop("replay_log", None)
    flags = dict(_mapping(visible.get("flags")))
    for key in (
        "act_room_sets", "player_relic_bag", "shared_relic_bag", "shared_relic_bag_originals",
        "player_relic_bag_multiplayer_fallback",
        "source_pending_treasure",
    ):
        flags.pop(key, None)
    visible["flags"] = flags
    # Event snapshots also occur inside pickup/resume frames. Project these
    # known internal payloads wherever they appear, including saved outer events.
    visible = dict(_mapping(_project_event_payloads(visible)))
    raw_combat = visible.get("combat")
    if isinstance(raw_combat, Mapping):
        combat = dict(raw_combat)
        combat.pop("last_events", None)
        if not draw_order_is_visible(visible):
            cards = []
            for raw_card in _sequence(combat.get("draw_pile")):
                card = dict(raw_card) if isinstance(raw_card, Mapping) else raw_card
                if isinstance(card, dict):
                    card.pop("position", None)
                    card.pop("position_from_top", None)
                cards.append(card)
            combat["draw_pile"] = sorted(cards, key=_canonical_card_key)
        monsters = []
        for raw_monster in _sequence(combat.get("monsters")):
            if not isinstance(raw_monster, Mapping):
                monsters.append(raw_monster)
                continue
            monster = dict(raw_monster)
            monster["next_move_id"] = None
            metadata = dict(_mapping(monster.get("metadata")))
            metadata.pop("next_move_id", None)
            monster["metadata"] = metadata
            monsters.append(monster)
        combat["monsters"] = monsters
        visible["combat"] = combat
    return visible


def _project_event_payloads(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: (_visible_sphere_board(item) if key == "crystal_sphere_board"
                  and isinstance(item, Mapping) else _project_event_payloads(item))
            for key, item in value.items() if key != "source_startup_flags"
        }
    if isinstance(value, list | tuple):
        projected = [_project_event_payloads(item) for item in value]
        return tuple(projected) if isinstance(value, tuple) else projected
    return value


def _visible_sphere_board(board: Mapping[str, Any]) -> dict[str, Any]:
    """Reveal cell contents only through cleared cells, and completed items.

    Item bounds, array positions, retry subscriptions and placement success are
    engine facts until their cells have been revealed. This projection is also
    idempotent when action descriptors project an already sanitized state.
    """
    if "items" not in board:
        return {key: deepcopy(board[key]) for key in (
            "remaining", "cleared", "visible_cells", "revealed_items") if key in board}
    cleared = sorted({int(cell) for cell in _sequence(board.get("cleared"))})
    cell_items: dict[int, Mapping[str, Any]] = {}
    items = _sequence(board.get("items"))
    for raw in items:
        item = _mapping(raw)
        if not item.get("placed"):
            continue
        for dx in range(int(item.get("width", 1))):
            for dy in range(int(item.get("height", 1))):
                cell = (int(item.get("x", 0)) + dx) * 11 + int(item.get("y", 0)) + dy
                cell_items.setdefault(cell, item)
    revealed = []
    for index in sorted({int(index) for index in _sequence(board.get("revealed"))}):
        if not 0 <= index < len(items):
            continue
        item = _mapping(items[index])
        revealed.append({key: item[key] for key in (
            "kind", "rarity", "x", "y", "width", "height") if key in item})
    return {
        "remaining": int(board.get("remaining", 0)),
        "cleared": cleared,
        "visible_cells": [{
            "x": cell // 11, "y": cell % 11,
            "kind": str(cell_items.get(cell, {}).get("kind", "empty")),
            "rarity": str(cell_items.get(cell, {}).get("rarity", "")),
        } for cell in cleared],
        "revealed_items": sorted(revealed, key=lambda item: json.dumps(item, sort_keys=True)),
    }


def policy_state(state: Any, *, oracle: bool = False) -> Any:
    """Build an independent state for descriptors and deterministic previews.

    The preview RNG uses a public constant, never the run seed, hidden state
    digest or saved RNG. The preview caller suppresses random transitions.
    """

    if oracle:
        return state
    from sts2sim.api import load_state, serialize
    from sts2sim.engine.rng import (
        PYTHON_RANDOM_ALGORITHM,
        STS2_STABLE_DERIVATION,
        capture_random_state,
        run_random_from_seed,
    )

    original = serialize(state)
    source_mode = _mapping(original.get("rng")).get("derivation") == STS2_STABLE_DERIVATION
    payload = policy_payload(original)
    payload["seed"] = 0
    # Only the public source/legacy mode survives. No saved stream states,
    # counters, seeds or arbitrary derivations are reused by planning.
    rng = (run_random_from_seed(0) if source_mode else
           run_random_from_seed(0, algorithm=PYTHON_RANDOM_ALGORITHM))
    payload["rng"] = capture_random_state(rng).model_dump(mode="json")
    return load_state(payload)


def public_run_flags(flags: Mapping[str, Any]) -> dict[str, Any]:
    """Expose visible progress and relic counters, without source/debug config."""

    public_keys = {
        "relic_counters",
        "bonus_potion_slots",
        "potion_slot_indices",
        "campfire_used_actions",
        "fur_coat_marked_nodes",
        "golden_compass_act2_map",
        "spoils_map_pending_act",
        "spoils_map_target_act",
        "spoils_map_target_node_id",
        "spoils_map_reward_gold",
        "silken_tress_reward_used",
        "treasure_chests_opened",
        "winged_boots_charges",
    }
    return {key: deepcopy(value) for key, value in flags.items() if key in public_keys}


def _canonical_card_key(card: Any) -> str:
    if isinstance(card, Mapping):
        return str(card.get("card_id", "")) + ":" + json.dumps(card, sort_keys=True)
    return str(card)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()
