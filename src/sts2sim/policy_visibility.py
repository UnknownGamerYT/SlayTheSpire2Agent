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
    ):
        flags.pop(key, None)
    visible["flags"] = flags
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


def policy_state(state: Any, *, oracle: bool = False) -> Any:
    """Build an independent state for descriptors and deterministic previews.

    The preview RNG uses a public constant, never the run seed, hidden state
    digest or saved RNG. The preview caller suppresses random transitions.
    """

    if oracle:
        return state
    from sts2sim.api import load_state, serialize
    from sts2sim.engine.rng import new_rng_state

    payload = policy_payload(serialize(state))
    payload["seed"] = 0
    payload["rng"] = new_rng_state(0).model_dump(mode="json")
    return load_state(payload)


def public_run_flags(flags: Mapping[str, Any]) -> dict[str, Any]:
    """Expose visible progress and relic counters, without source/debug config."""

    public_keys = {
        "relic_counters",
        "bonus_potion_slots",
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
