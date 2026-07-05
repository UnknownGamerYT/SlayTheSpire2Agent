"""Agent-facing helpers for deterministic action IDs and observations."""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping, Sequence
from typing import Any

from sts2sim.agent_previews import preview_action_key, preview_actions
from sts2sim.api import legal_actions as _legal_actions
from sts2sim.api import serialize as _serialize
from sts2sim.engine.models import Action, ActionType, RunPhase
from sts2sim.mechanics.aggression import (
    AGGRESSION_OBSERVATION_KEYS,
    aggression_summary,
    aggression_vector,
)
from sts2sim.mechanics.belief import BELIEF_OBSERVATION_KEYS, belief_summary, belief_vector
from sts2sim.mechanics.enemy_traits import (
    ENEMY_TRAIT_AGGREGATE_KEYS,
    enemy_trait_aggregate_vector,
)
from sts2sim.mechanics.mechanic_atoms import (
    CARD_SLOT_KEYS,
    STATUS_ATOM_KEYS,
    card_slot_summary,
    card_slot_vector,
    card_slots_from_payload,
    status_atom_summary,
    status_atom_vector,
)
from sts2sim.mechanics.option_slots import (
    OPTION_SLOT_KEYS,
    option_slots_vector,
    reward_option_slots,
    shop_option_slots,
)
from sts2sim.mechanics.planning_context import (
    REWARD_PLAN_KEYS,
    ROUTE_PLAN_KEYS,
    reward_plan_summary,
    reward_plan_vector,
    route_plan_summary,
    route_plan_vector,
)
from sts2sim.mechanics.potions import normalize_potion_use, potion_capacity
from sts2sim.mechanics.semantics import (
    MECHANIC_TAG_BUCKETS,
    MECHANIC_VALUE_KEYS,
    action_mechanic_profile,
    profile_value_vector,
    state_mechanic_profile,
)
from sts2sim.mechanics.synergy import action_synergy_profile
from sts2sim.mechanics.trigger_visibility import (
    TRIGGER_VISIBILITY_KEYS,
    potion_slot_summary,
    trigger_visibility_summary,
    trigger_visibility_vector,
)

DEFAULT_MAX_ACTIONS = 512
RELIC_HASH_BUCKETS = 32
POTION_HASH_BUCKETS = 16
PLAYER_STATUS_HASH_BUCKETS = 16
DECK_CARD_HASH_BUCKETS = 32
DECK_EFFECT_HASH_BUCKETS = 16
MONSTER_HASH_BUCKETS = 32
MONSTER_STATUS_HASH_BUCKETS = 32
MONSTER_INTENT_HASH_BUCKETS = 16
CARD_POSITION_HASH_BUCKETS = 64
RELIC_POSITION_HASH_BUCKETS = 32
POTION_POSITION_HASH_BUCKETS = 16
SHOP_POSITION_HASH_BUCKETS = 32
REWARD_POSITION_HASH_BUCKETS = 32
EVENT_POSITION_HASH_BUCKETS = 16
ANCIENT_POSITION_HASH_BUCKETS = 16
HAND_CARD_SLOT_LIMIT = 10
DRAW_CARD_SLOT_LIMIT = 10
DISCARD_CARD_SLOT_LIMIT = 10
EXHAUST_CARD_SLOT_LIMIT = 10
MASTER_DECK_CARD_SLOT_LIMIT = 30
REWARD_OPTION_SLOT_LIMIT = 12
SHOP_OPTION_SLOT_LIMIT = 16
AGENT_MEMORY_STEPS = 4
AGENT_MEMORY_FEATURE_KEYS: tuple[str, ...] = (
    "present",
    "age",
    "action_type_id",
    "action_id",
    "action_index",
    "confidence",
    "log_prob",
    "value",
    "reward",
    "hp_delta",
    "block_delta",
    "energy_delta",
    "gold_delta",
    "floor_delta",
    "phase_changed",
    "target_hp_delta",
    "monster_hp_total_delta",
    "kills",
    "incoming_damage_delta",
    "preview_error",
    "done",
    "plan_aggression_target",
    "plan_hp_floor",
    "plan_hp_spend_budget",
    "plan_combat_pace",
    "plan_route_preference",
    "plan_potion_policy",
    "plan_reward_pickiness",
    "plan_expected_hp_loss",
    "plan_expected_turns_to_kill",
    "plan_boss_readiness",
)

ACTION_TYPE_IDS: dict[str, int] = {
    action_type.value: index for index, action_type in enumerate(ActionType)
}
PHASE_IDS: dict[str, int] = {phase.value: index for index, phase in enumerate(RunPhase)}

OBSERVATION_VECTOR_SCHEMA: tuple[str, ...] = (
    "phase_id",
    "act",
    "floor",
    "ascension",
    "player_hp",
    "player_max_hp",
    "player_block",
    "player_energy",
    "player_max_energy",
    "player_stars",
    "player_resource_total",
    "player_gold",
    "master_deck_count",
    "relic_count",
    "curse_count",
    "potion_count",
    "legal_action_count",
    "combat_turn",
    "combat_hand_count",
    "combat_draw_pile_count",
    "combat_discard_pile_count",
    "combat_exhaust_pile_count",
    "combat_orb_count",
    "combat_orb_slots",
    "combat_orb_value_total",
    "alive_monster_count",
    "monster_hp_total",
    "monster_block_total",
    "incoming_damage",
    "map_node_count",
    "map_edge_count",
    "map_completed_count",
    "reward_gold",
    "reward_card_count",
    "reward_relic_count",
    "reward_potion_count",
    *(f"aggression_{key}" for key in AGGRESSION_OBSERVATION_KEYS),
    *(f"belief_{key}" for key in BELIEF_OBSERVATION_KEYS),
    *(f"reward_plan_{key}" for key in REWARD_PLAN_KEYS),
    *(f"route_plan_{key}" for key in ROUTE_PLAN_KEYS),
    *(f"player_status_atom_{key}" for key in STATUS_ATOM_KEYS),
    *(f"enemy_trait_{key}" for key in ENEMY_TRAIT_AGGREGATE_KEYS),
    *(f"trigger_visibility_{key}" for key in TRIGGER_VISIBILITY_KEYS),
    *(
        f"hand_slot_{slot_index}_{key}"
        for slot_index in range(HAND_CARD_SLOT_LIMIT)
        for key in CARD_SLOT_KEYS
    ),
    *(
        f"draw_slot_{slot_index}_{key}"
        for slot_index in range(DRAW_CARD_SLOT_LIMIT)
        for key in CARD_SLOT_KEYS
    ),
    *(
        f"discard_slot_{slot_index}_{key}"
        for slot_index in range(DISCARD_CARD_SLOT_LIMIT)
        for key in CARD_SLOT_KEYS
    ),
    *(
        f"exhaust_slot_{slot_index}_{key}"
        for slot_index in range(EXHAUST_CARD_SLOT_LIMIT)
        for key in CARD_SLOT_KEYS
    ),
    *(
        f"deck_slot_{slot_index}_{key}"
        for slot_index in range(MASTER_DECK_CARD_SLOT_LIMIT)
        for key in CARD_SLOT_KEYS
    ),
    *(
        f"reward_option_slot_{slot_index}_{key}"
        for slot_index in range(REWARD_OPTION_SLOT_LIMIT)
        for key in OPTION_SLOT_KEYS
    ),
    *(
        f"shop_option_slot_{slot_index}_{key}"
        for slot_index in range(SHOP_OPTION_SLOT_LIMIT)
        for key in OPTION_SLOT_KEYS
    ),
    *(f"owned_relic_{index}" for index in range(RELIC_HASH_BUCKETS)),
    *(f"owned_potion_{index}" for index in range(POTION_HASH_BUCKETS)),
    *(f"player_status_{index}" for index in range(PLAYER_STATUS_HASH_BUCKETS)),
    *(f"deck_card_{index}" for index in range(DECK_CARD_HASH_BUCKETS)),
    *(f"deck_effect_{index}" for index in range(DECK_EFFECT_HASH_BUCKETS)),
    *(f"monster_id_{index}" for index in range(MONSTER_HASH_BUCKETS)),
    *(f"monster_status_{index}" for index in range(MONSTER_STATUS_HASH_BUCKETS)),
    *(f"monster_intent_{index}" for index in range(MONSTER_INTENT_HASH_BUCKETS)),
    *(f"state_mechanic_{key}" for key in MECHANIC_VALUE_KEYS),
    *(f"state_mechanic_tag_{index}" for index in range(MECHANIC_TAG_BUCKETS)),
    *(f"card_position_{index}" for index in range(CARD_POSITION_HASH_BUCKETS)),
    *(f"relic_position_{index}" for index in range(RELIC_POSITION_HASH_BUCKETS)),
    *(f"potion_slot_position_{index}" for index in range(POTION_POSITION_HASH_BUCKETS)),
    *(f"shop_position_{index}" for index in range(SHOP_POSITION_HASH_BUCKETS)),
    *(f"reward_position_{index}" for index in range(REWARD_POSITION_HASH_BUCKETS)),
    *(f"event_option_position_{index}" for index in range(EVENT_POSITION_HASH_BUCKETS)),
    *(f"ancient_option_position_{index}" for index in range(ANCIENT_POSITION_HASH_BUCKETS)),
    *(
        f"agent_memory_{step_index}_{key}"
        for step_index in range(AGENT_MEMORY_STEPS)
        for key in AGENT_MEMORY_FEATURE_KEYS
    ),
)


def action_space(state: Any) -> list[dict[str, Any]]:
    """Return state-local action descriptors with deterministic integer IDs.

    IDs are assigned after sorting legal engine actions by their canonical JSON
    payload. They are stable for equivalent states, but intentionally local to
    the state because card instance IDs, map nodes, and reward targets are
    generated by the simulator.
    """

    actions = _ordered_legal_actions(state)
    previews = preview_actions(state, actions)
    state_payload = _serialize(state)
    return [
        _action_descriptor(
            action_id,
            action,
            state,
            preview=previews.get(preview_action_key(action)),
            state_payload=state_payload,
        )
        for action_id, action in enumerate(actions)
    ]


def action_mask(state: Any, *, max_actions: int | None = None) -> tuple[int, ...]:
    """Return a binary mask for legal action IDs.

    With ``max_actions=None`` the mask length matches the state-local action
    space, so every entry is legal. Pass ``max_actions`` to get a fixed-width
    mask padded with zeros for Gymnasium-style discrete policies.
    """

    action_count = len(_ordered_legal_actions(state))
    if max_actions is None:
        return tuple(1 for _ in range(action_count))
    if max_actions < 0:
        raise ValueError("max_actions must be non-negative")
    if action_count > max_actions:
        raise ValueError(
            f"state has {action_count} legal actions, which exceeds max_actions={max_actions}"
        )
    return tuple(1 if index < action_count else 0 for index in range(max_actions))


def decode_action(state: Any, action_id: int) -> Action:
    """Map a deterministic state-local action ID back to an engine ``Action``."""

    if isinstance(action_id, bool):
        raise TypeError("action_id must be an integer, not bool")
    action_index = operator.index(action_id)
    if action_index < 0:
        raise IndexError(f"action_id {action_id} is not legal for this state")
    actions = _ordered_legal_actions(state)
    try:
        return actions[action_index]
    except IndexError as exc:
        raise IndexError(f"action_id {action_id} is not legal for this state") from exc


def encode_observation(
    state: Any,
    *,
    include_state: bool = True,
    agent_memory: object | None = None,
) -> dict[str, Any]:
    """Encode a simulator state as a JSON-friendly observation payload.

    The ``vector`` field is fixed-length and numeric. Structured sections keep
    useful symbolic context available for agents that are not purely tensor
    based. The full serialized state is included by default for lossless
    debugging and can be omitted with ``include_state=False``.
    """

    payload = _serialize(state)
    phase = str(payload.get("phase", ""))
    legal_action_count = len(_ordered_legal_actions(state))

    player = _mapping(payload.get("player"))
    combat = _mapping(payload.get("combat"))
    game_map = _mapping(payload.get("map"))
    reward = _mapping(payload.get("reward"))

    monsters = _sequence(combat.get("monsters"))
    alive_monsters = [
        monster for monster in monsters if _number(_mapping(monster).get("hp")) > 0
    ]

    resources = _mapping(player.get("resources"))
    player_summary = {
        "hp": _int(player.get("hp")),
        "max_hp": _int(player.get("max_hp")),
        "block": _int(player.get("block")),
        "energy": _int(player.get("energy")),
        "max_energy": _int(player.get("max_energy")),
        "stars": _int(resources.get("star", resources.get("stars", 0))),
        "resource_total": sum(_int(value) for value in resources.values()),
        "gold": _int(player.get("gold")),
        "resources": dict(resources),
    }
    counts = {
        "master_deck": len(_sequence(payload.get("master_deck"))),
        "relics": len(_sequence(payload.get("relics"))),
        "curses": len(_sequence(payload.get("curses"))),
        "potions": len(_sequence(payload.get("potions"))),
        "legal_actions": legal_action_count,
    }
    combat_summary = {
        "turn": _int(combat.get("turn")),
        "hand_count": len(_sequence(combat.get("hand"))),
        "draw_pile_count": len(_sequence(combat.get("draw_pile"))),
        "discard_pile_count": len(_sequence(combat.get("discard_pile"))),
        "exhaust_pile_count": len(_sequence(combat.get("exhaust_pile"))),
        "orb_count": len(_sequence(combat.get("orbs"))),
        "orb_slots": _int(combat.get("orb_slots")),
        "orb_value_total": sum(
            _int(_mapping(orb).get("value")) for orb in _sequence(combat.get("orbs"))
        ),
        "monster_count": len(monsters),
        "alive_monster_count": len(alive_monsters),
        "monster_hp_total": sum(_int(_mapping(monster).get("hp")) for monster in monsters),
        "monster_block_total": sum(
            _int(_mapping(monster).get("block")) for monster in monsters
        ),
        "incoming_damage": sum(
            _int(_mapping(monster).get("intent_damage")) for monster in alive_monsters
        ),
    }
    map_summary = {
        "node_count": len(_sequence(game_map.get("nodes"))),
        "edge_count": len(_sequence(game_map.get("edges"))),
        "completed_count": len(_sequence(game_map.get("completed_node_ids"))),
    }
    reward_summary = {
        "gold": _int(reward.get("gold")),
        "card_count": len(_sequence(reward.get("card_options")))
        + len(_sequence(reward.get("card_ids")))
        + sum(len(_sequence(group)) for group in _sequence(reward.get("card_option_groups"))),
        "relic_count": int(bool(reward.get("relic_id")))
        + len(_sequence(reward.get("relic_ids"))),
        "potion_count": int(bool(reward.get("potion_id")))
        + len(_sequence(reward.get("potion_ids"))),
    }
    aggression = aggression_summary(payload)
    belief = belief_summary(payload)
    reward_plan = reward_plan_summary(payload)
    route_plan = route_plan_summary(payload)
    player_status_atoms = status_atom_summary(_mapping(player.get("statuses")))
    enemy_traits_vector = enemy_trait_aggregate_vector(payload)
    trigger_visibility = trigger_visibility_summary(payload)
    potion_slots = potion_slot_summary(payload)
    card_slots = _card_visibility_slots(payload)
    reward_slots = reward_option_slots(payload, limit=REWARD_OPTION_SLOT_LIMIT)
    shop_slots = shop_option_slots(payload, limit=SHOP_OPTION_SLOT_LIMIT)

    vector = [
        _float(PHASE_IDS.get(phase, -1)),
        _float(payload.get("act")),
        _float(payload.get("floor")),
        _float(payload.get("ascension")),
        _float(player_summary["hp"]),
        _float(player_summary["max_hp"]),
        _float(player_summary["block"]),
        _float(player_summary["energy"]),
        _float(player_summary["max_energy"]),
        _float(player_summary["stars"]),
        _float(player_summary["resource_total"]),
        _float(player_summary["gold"]),
        _float(counts["master_deck"]),
        _float(counts["relics"]),
        _float(counts["curses"]),
        _float(counts["potions"]),
        _float(counts["legal_actions"]),
        _float(combat_summary["turn"]),
        _float(combat_summary["hand_count"]),
        _float(combat_summary["draw_pile_count"]),
        _float(combat_summary["discard_pile_count"]),
        _float(combat_summary["exhaust_pile_count"]),
        _float(combat_summary["orb_count"]),
        _float(combat_summary["orb_slots"]),
        _float(combat_summary["orb_value_total"]),
        _float(combat_summary["alive_monster_count"]),
        _float(combat_summary["monster_hp_total"]),
        _float(combat_summary["monster_block_total"]),
        _float(combat_summary["incoming_damage"]),
        _float(map_summary["node_count"]),
        _float(map_summary["edge_count"]),
        _float(map_summary["completed_count"]),
        _float(reward_summary["gold"]),
        _float(reward_summary["card_count"]),
        _float(reward_summary["relic_count"]),
        _float(reward_summary["potion_count"]),
    ]
    vector.extend(aggression_vector(aggression))
    vector.extend(belief_vector(belief))
    vector.extend(reward_plan_vector(payload))
    vector.extend(route_plan_vector(payload))
    vector.extend(status_atom_vector(player_status_atoms))
    vector.extend(enemy_traits_vector)
    vector.extend(trigger_visibility_vector(trigger_visibility))
    vector.extend(_card_slot_vectors(card_slots))
    vector.extend(
        option_slots_vector(
            payload,
            reward_limit=REWARD_OPTION_SLOT_LIMIT,
            shop_limit=SHOP_OPTION_SLOT_LIMIT,
        )
    )
    vector.extend(
        _hash_presence_features(payload.get("relics"), bucket_count=RELIC_HASH_BUCKETS)
    )
    vector.extend(
        _hash_presence_features(payload.get("potions"), bucket_count=POTION_HASH_BUCKETS)
    )
    vector.extend(
        _hash_presence_features(
            _mapping(player.get("statuses")).keys(),
            bucket_count=PLAYER_STATUS_HASH_BUCKETS,
        )
    )
    vector.extend(
        _hash_presence_features(
            (_mapping(card).get("card_id") for card in _sequence(payload.get("master_deck"))),
            bucket_count=DECK_CARD_HASH_BUCKETS,
        )
    )
    vector.extend(
        _hash_presence_features(
            (
                key
                for card in _sequence(payload.get("master_deck"))
                for key in _mapping(_mapping(card).get("effects"))
            ),
            bucket_count=DECK_EFFECT_HASH_BUCKETS,
        )
    )
    targets = _target_summary(payload)
    vector.extend(
        _hash_presence_features(
            _target_tokens(targets, "monster"),
            bucket_count=MONSTER_HASH_BUCKETS,
        )
    )
    vector.extend(
        _hash_presence_features(
            _target_tokens(targets, "status"),
            bucket_count=MONSTER_STATUS_HASH_BUCKETS,
        )
    )
    vector.extend(
        _hash_presence_features(
            _target_tokens(targets, "intent"),
            bucket_count=MONSTER_INTENT_HASH_BUCKETS,
        )
    )
    mechanics = state_mechanic_profile(payload)
    vector.extend(profile_value_vector(mechanics))
    vector.extend(
        _hash_presence_features(
            _mapping(mechanics).get("tags"),
            bucket_count=MECHANIC_TAG_BUCKETS,
        )
    )
    positions = _position_summary(payload)
    vector.extend(
        _hash_presence_features(
            _position_tokens(positions, "cards"),
            bucket_count=CARD_POSITION_HASH_BUCKETS,
        )
    )
    vector.extend(
        _hash_presence_features(
            _position_tokens(positions, "relics"),
            bucket_count=RELIC_POSITION_HASH_BUCKETS,
        )
    )
    vector.extend(
        _hash_presence_features(
            _position_tokens(positions, "potions"),
            bucket_count=POTION_POSITION_HASH_BUCKETS,
        )
    )
    vector.extend(
        _hash_presence_features(
            _position_tokens(positions, "shop_items"),
            bucket_count=SHOP_POSITION_HASH_BUCKETS,
        )
    )
    vector.extend(
        _hash_presence_features(
            _position_tokens(positions, "reward_choices"),
            bucket_count=REWARD_POSITION_HASH_BUCKETS,
        )
    )
    vector.extend(
        _hash_presence_features(
            _position_tokens(positions, "event_options"),
            bucket_count=EVENT_POSITION_HASH_BUCKETS,
        )
    )
    vector.extend(
        _hash_presence_features(
            _position_tokens(positions, "ancient_options"),
            bucket_count=ANCIENT_POSITION_HASH_BUCKETS,
        )
    )
    memory_summary = _agent_memory_summary(agent_memory)
    vector.extend(_agent_memory_vector(memory_summary))

    observation: dict[str, Any] = {
        "schema_version": _int(payload.get("schema_version")),
        "phase": phase,
        "phase_id": PHASE_IDS.get(phase, -1),
        "vector_schema": list(OBSERVATION_VECTOR_SCHEMA),
        "vector": vector,
        "player": player_summary,
        "counts": counts,
        "combat": combat_summary,
        "map": map_summary,
        "reward": reward_summary,
        "aggression": aggression,
        "belief": belief,
        "reward_plan": reward_plan,
        "route_plan": route_plan,
        "visibility": {
            "player_status_atoms": player_status_atoms,
            "trigger_visibility": trigger_visibility,
            "potion_slots": potion_slots,
            "card_slots": card_slots,
            "reward_option_slots": reward_slots,
            "shop_option_slots": shop_slots,
        },
        "targets": targets,
        "mechanics": mechanics,
        "positions": positions,
        "agent_memory": memory_summary,
        "legal_actions": {
            "count": legal_action_count,
            "ids": list(range(legal_action_count)),
        },
    }
    if include_state:
        observation["state"] = payload
    return observation


def _agent_memory_summary(agent_memory: object | None) -> dict[str, Any]:
    if agent_memory is None:
        raw_entries: tuple[Any, ...] = ()
    elif isinstance(agent_memory, Mapping):
        raw_entries = _sequence(agent_memory.get("entries"))
    else:
        raw_entries = _sequence(agent_memory)
    entries: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(raw_entries[:AGENT_MEMORY_STEPS]):
        entry = _mapping(raw_entry)
        entries.append(
            {
                "age": _int(entry.get("age", index)),
                "action_type": str(entry.get("action_type", "")),
                "action_type_id": _int(entry.get("action_type_id")),
                "action_id": _int(entry.get("action_id")),
                "action_index": _int(entry.get("action_index")),
                "confidence": _float(entry.get("confidence")),
                "log_prob": _float(entry.get("log_prob")),
                "value": _float(entry.get("value")),
                "reward": _float(entry.get("reward")),
                "hp_delta": _int(entry.get("hp_delta")),
                "block_delta": _int(entry.get("block_delta")),
                "energy_delta": _int(entry.get("energy_delta")),
                "gold_delta": _int(entry.get("gold_delta")),
                "floor_delta": _int(entry.get("floor_delta")),
                "phase_changed": _int(entry.get("phase_changed")),
                "target_hp_delta": _int(entry.get("target_hp_delta")),
                "monster_hp_total_delta": _int(entry.get("monster_hp_total_delta")),
                "kills": _int(entry.get("kills")),
                "incoming_damage_delta": _int(entry.get("incoming_damage_delta")),
                "preview_error": _int(entry.get("preview_error")),
                "done": _int(entry.get("done")),
                "plan_aggression_target": _float(entry.get("plan_aggression_target")),
                "plan_hp_floor": _float(entry.get("plan_hp_floor")),
                "plan_hp_spend_budget": _float(entry.get("plan_hp_spend_budget")),
                "plan_combat_pace": _float(entry.get("plan_combat_pace")),
                "plan_route_preference": _float(entry.get("plan_route_preference")),
                "plan_potion_policy": _float(entry.get("plan_potion_policy")),
                "plan_reward_pickiness": _float(entry.get("plan_reward_pickiness")),
                "plan_expected_hp_loss": _float(entry.get("plan_expected_hp_loss")),
                "plan_expected_turns_to_kill": _float(
                    entry.get("plan_expected_turns_to_kill")
                ),
                "plan_boss_readiness": _float(entry.get("plan_boss_readiness")),
            }
        )
    return {
        "max_steps": AGENT_MEMORY_STEPS,
        "feature_keys": list(AGENT_MEMORY_FEATURE_KEYS),
        "entries": entries,
    }


def _agent_memory_vector(memory_summary: Mapping[str, Any]) -> list[float]:
    entries = _sequence(memory_summary.get("entries"))
    vector: list[float] = []
    for index in range(AGENT_MEMORY_STEPS):
        entry = _mapping(entries[index]) if index < len(entries) else {}
        vector.extend(
            (
                1.0 if entry else 0.0,
                _scaled(_int(entry.get("age", index)), AGENT_MEMORY_STEPS),
                _scaled(_int(entry.get("action_type_id")), len(ACTION_TYPE_IDS)),
                _scaled(_int(entry.get("action_id")), DEFAULT_MAX_ACTIONS),
                _scaled(_int(entry.get("action_index")), DEFAULT_MAX_ACTIONS),
                _scaled_fraction(entry.get("confidence")),
                _signed_scaled(_float(entry.get("log_prob")), 10.0),
                _signed_scaled(_float(entry.get("value")), 100.0),
                _signed_scaled(_float(entry.get("reward")), 100.0),
                _signed_scaled(_float(entry.get("hp_delta")), 100.0),
                _signed_scaled(_float(entry.get("block_delta")), 100.0),
                _signed_scaled(_float(entry.get("energy_delta")), 10.0),
                _signed_scaled(_float(entry.get("gold_delta")), 500.0),
                _signed_scaled(_float(entry.get("floor_delta")), 20.0),
                _scaled(_int(entry.get("phase_changed")), 1),
                _signed_scaled(_float(entry.get("target_hp_delta")), 300.0),
                _signed_scaled(_float(entry.get("monster_hp_total_delta")), 600.0),
                _scaled(_int(entry.get("kills")), 5),
                _signed_scaled(_float(entry.get("incoming_damage_delta")), 120.0),
                _scaled(_int(entry.get("preview_error")), 1),
                _scaled(_int(entry.get("done")), 1),
                _scaled_fraction(entry.get("plan_aggression_target")),
                _scaled_fraction(entry.get("plan_hp_floor")),
                _scaled_fraction(entry.get("plan_hp_spend_budget")),
                _scaled_fraction(entry.get("plan_combat_pace")),
                _scaled_fraction(entry.get("plan_route_preference")),
                _scaled_fraction(entry.get("plan_potion_policy")),
                _scaled_fraction(entry.get("plan_reward_pickiness")),
                _scaled_fraction(entry.get("plan_expected_hp_loss")),
                _scaled_fraction(entry.get("plan_expected_turns_to_kill")),
                _scaled_fraction(entry.get("plan_boss_readiness")),
            )
        )
    return vector


def _card_visibility_slots(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        "hand": card_slots_from_payload(payload, zone="hand", limit=HAND_CARD_SLOT_LIMIT),
        "draw_pile": card_slots_from_payload(
            payload,
            zone="draw_pile",
            limit=DRAW_CARD_SLOT_LIMIT,
        ),
        "discard_pile": card_slots_from_payload(
            payload,
            zone="discard_pile",
            limit=DISCARD_CARD_SLOT_LIMIT,
        ),
        "exhaust_pile": card_slots_from_payload(
            payload,
            zone="exhaust_pile",
            limit=EXHAUST_CARD_SLOT_LIMIT,
        ),
        "master_deck": card_slots_from_payload(
            payload,
            zone="master_deck",
            limit=MASTER_DECK_CARD_SLOT_LIMIT,
        ),
    }


def _card_slot_vectors(slots_by_zone: Mapping[str, Any]) -> list[float]:
    vector: list[float] = []
    zone_limits = (
        ("hand", HAND_CARD_SLOT_LIMIT),
        ("draw_pile", DRAW_CARD_SLOT_LIMIT),
        ("discard_pile", DISCARD_CARD_SLOT_LIMIT),
        ("exhaust_pile", EXHAUST_CARD_SLOT_LIMIT),
        ("master_deck", MASTER_DECK_CARD_SLOT_LIMIT),
    )
    for zone, limit in zone_limits:
        slots = [_mapping(slot) for slot in _sequence(slots_by_zone.get(zone))]
        for index in range(limit):
            slot = slots[index] if index < len(slots) else {}
            vector.extend(card_slot_vector(slot))
    return vector


def _target_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    player = _mapping(payload.get("player"))
    combat = _mapping(payload.get("combat"))
    monsters = [
        _serialized_monster_descriptor(monster, position=index)
        for index, monster in enumerate(_sequence(combat.get("monsters")))
    ]
    return {
        "player": {
            "kind": "player",
            "hp": _int(player.get("hp")),
            "max_hp": _int(player.get("max_hp")),
            "block": _int(player.get("block")),
            "statuses": dict(_mapping(player.get("statuses"))),
            "resources": dict(_mapping(player.get("resources"))),
        },
        "monsters": monsters,
        "alive_monsters": [monster for monster in monsters if monster.get("alive")],
    }


def _serialized_monster_descriptor(
    raw_monster: object,
    *,
    position: int,
) -> dict[str, Any]:
    monster = _mapping(raw_monster)
    statuses = dict(_mapping(monster.get("statuses")))
    metadata = _mapping(monster.get("metadata"))
    hp = _int(monster.get("hp"))
    max_hp = max(1, _int(monster.get("max_hp")))
    return {
        "kind": "monster",
        "target_id": str(monster.get("monster_id", "")),
        "monster_id": str(monster.get("monster_id", "")),
        "source_monster_id": str(metadata.get("source_monster_id", monster.get("monster_id", ""))),
        "name": str(monster.get("name", "")),
        "position": position,
        "slot_index": _int(metadata.get("slot_index", position)),
        "hp": hp,
        "max_hp": max_hp,
        "hp_fraction": hp / max_hp,
        "block": _int(monster.get("block")),
        "intent": str(monster.get("intent", "")),
        "intent_damage": _int(monster.get("intent_damage")),
        "intent_block": _int(monster.get("intent_block")),
        "move_id": str(monster.get("move_id", "")),
        "next_move_id": str(monster.get("next_move_id", "")),
        "hit_count": _int(monster.get("hit_count")),
        "statuses": statuses,
        "status_keys": sorted(str(key) for key in statuses),
        "status_total": sum(_int(value) for value in statuses.values()),
        "alive": hp > 0,
        "metadata": dict(metadata),
    }


def _target_tokens(targets: Mapping[str, Any], kind: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for monster in _sequence(targets.get("monsters")):
        monster_map = _mapping(monster)
        position = _int(monster_map.get("position"))
        if kind == "monster":
            monster_id = str(
                monster_map.get("source_monster_id") or monster_map.get("monster_id", "")
            )
            if monster_id:
                tokens.append(f"monster:{position}:{monster_id}")
        elif kind == "status":
            statuses = _mapping(monster_map.get("statuses"))
            for status_id, amount in statuses.items():
                tokens.append(f"monster_status:{position}:{status_id}:{_int(amount)}")
        elif kind == "intent":
            intent = str(monster_map.get("intent", ""))
            move_id = str(monster_map.get("move_id", ""))
            if intent:
                tokens.append(f"monster_intent:{position}:{intent}")
            if move_id:
                tokens.append(f"monster_move:{position}:{move_id}")
    return tuple(tokens)


def _position_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    combat = _mapping(payload.get("combat"))
    cards = {
        "master_deck": _card_position_list(payload.get("master_deck"), zone="master_deck"),
        "hand": _card_position_list(combat.get("hand"), zone="hand"),
        "draw_pile": _card_position_list(combat.get("draw_pile"), zone="draw_pile"),
        "discard_pile": _card_position_list(combat.get("discard_pile"), zone="discard_pile"),
        "exhaust_pile": _card_position_list(combat.get("exhaust_pile"), zone="exhaust_pile"),
    }
    return {
        "cards": cards,
        "relics": [
            {"relic_id": str(relic_id), "position": index}
            for index, relic_id in enumerate(_sequence(payload.get("relics")))
        ],
        "potions": [
            {"potion_id": str(potion_id), "slot_index": index, "position": index}
            for index, potion_id in enumerate(_sequence(payload.get("potions")))
        ],
        "shop_items": _shop_position_list(payload.get("shop")),
        "reward_choices": _reward_position_list(payload.get("reward")),
        "event_options": _event_position_list(payload.get("event")),
        "ancient_options": _ancient_position_list(payload.get("ancient")),
    }


def _card_position_list(raw_cards: object, *, zone: str) -> list[dict[str, Any]]:
    return [
        {
            "instance_id": str(_mapping(card).get("instance_id", "")),
            "card_id": str(_mapping(card).get("card_id", "")),
            "zone": zone,
            "position": index,
            "position_from_top": index if zone == "draw_pile" else None,
        }
        for index, card in enumerate(_sequence(raw_cards))
    ]


def _shop_position_list(raw_shop: object) -> list[dict[str, Any]]:
    shop = _mapping(raw_shop)
    return [
        {
            "slot_id": str(_mapping(item).get("slot_id", "")),
            "slot_index": index,
            "position": index,
            "item_id": str(_mapping(item).get("item_id", "")),
            "kind": str(_mapping(item).get("kind", "")),
            "purchased": bool(_mapping(item).get("purchased", False)),
        }
        for index, item in enumerate(_sequence(shop.get("items")))
    ]


def _reward_position_list(raw_reward: object) -> list[dict[str, Any]]:
    reward = _mapping(raw_reward)
    choices: list[dict[str, Any]] = []
    if not reward:
        return choices
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


def _event_position_list(raw_event: object) -> list[dict[str, Any]]:
    event = _mapping(raw_event)
    return [
        {
            "event_id": str(event.get("event_id", "")),
            "page_id": str(event.get("page_id", "")),
            "option_id": str(_mapping(option).get("option_id", "")),
            "position": index,
            "disabled": bool(_mapping(option).get("disabled", False)),
        }
        for index, option in enumerate(_sequence(event.get("options")))
    ]


def _ancient_position_list(raw_ancient: object) -> list[dict[str, Any]]:
    ancient = _mapping(raw_ancient)
    return [
        {
            "ancient_id": str(ancient.get("ancient_id", "")),
            "option_id": str(_mapping(option).get("option_id", "")),
            "position": index,
            "kind": str(_mapping(option).get("kind", "")),
            "relic_id": str(_mapping(option).get("relic_id", "")),
        }
        for index, option in enumerate(_sequence(ancient.get("options")))
    ]


def _position_tokens(positions: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw = positions.get(key)
    tokens: list[str] = []
    if key == "cards" and isinstance(raw, Mapping):
        for zone, cards in raw.items():
            for card in _sequence(cards):
                card_map = _mapping(card)
                card_id = str(card_map.get("card_id", ""))
                instance_id = str(card_map.get("instance_id", ""))
                position = _int(card_map.get("position"))
                if card_id:
                    tokens.append(f"{zone}:{position}:card:{card_id}")
                if instance_id:
                    tokens.append(f"{zone}:{position}:instance:{instance_id}")
        return tuple(tokens)
    for item in _sequence(raw):
        item_map = _mapping(item)
        position = _int(item_map.get("position", item_map.get("slot_index")))
        identity = _position_identity(item_map)
        kind = str(item_map.get("kind", key))
        if identity:
            tokens.append(f"{key}:{position}:{kind}:{identity}")
    return tuple(tokens)


def _position_identity(item: Mapping[str, Any]) -> str:
    for key in ("card_id", "relic_id", "potion_id", "item_id", "content_id", "option_id"):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _ordered_legal_actions(state: Any) -> tuple[Action, ...]:
    keyed_actions = [
        (_action_key(action), index, action)
        for index, action in enumerate(_legal_actions(state))
    ]
    return tuple(action for _key, _index, action in sorted(keyed_actions))


def _action_descriptor(
    action_id: int,
    action: Action,
    state: Any,
    *,
    preview: Mapping[str, Any] | None = None,
    state_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _action_payload(action)
    descriptor = {
        "id": action_id,
        "key": _canonical_json(payload),
        "type": str(payload["type"]),
        "action_type_id": ACTION_TYPE_IDS.get(str(payload["type"]), -1),
        "card_instance_id": payload.get("card_instance_id"),
        "target_id": payload.get("target_id"),
        "payload": dict(_mapping(payload.get("payload"))),
        "action": payload,
    }
    if preview is not None:
        descriptor["preview"] = _json_safe(preview)
    card = _action_card_descriptor(state, action)
    if card:
        descriptor["card"] = card
    selected_cards = _action_selected_card_descriptors(state, action)
    if selected_cards:
        descriptor["selected_cards"] = selected_cards
        descriptor["selected_card_count"] = len(selected_cards)
        if not card:
            descriptor["card"] = (
                selected_cards[0]
                if len(selected_cards) == 1
                else _aggregate_selected_card_descriptor(selected_cards)
            )
    node = _action_node_descriptor(state, action)
    if node:
        descriptor["node"] = node
    target = _action_target_descriptor(state, action)
    if target:
        descriptor["target"] = target
    item = _action_item_descriptor(state, action)
    if item:
        descriptor["item"] = item
    potion = _action_potion_descriptor(state, action)
    if potion:
        descriptor["potion"] = potion
        potion_strategy = _action_potion_strategy_descriptor(
            state,
            action,
            potion,
            preview=preview,
        )
        if potion_strategy:
            descriptor["potion_strategy"] = potion_strategy
    relic = _action_relic_descriptor(state, action)
    if relic:
        descriptor["relic"] = relic
    reward_choice = _action_reward_descriptor(state, action)
    if reward_choice:
        descriptor["reward_choice"] = reward_choice
    reward_bundle = _reward_bundle_descriptor(state)
    if reward_bundle:
        descriptor["reward_bundle"] = reward_bundle
    event_option = _action_event_option_descriptor(state, action)
    if event_option:
        descriptor["event_option"] = event_option
    ancient_option = _action_ancient_option_descriptor(state, action)
    if ancient_option:
        descriptor["ancient_option"] = ancient_option
    option_slot = _action_option_slot_descriptor(state_payload, descriptor)
    if option_slot:
        descriptor["option_slot"] = option_slot
    descriptor["mechanics"] = action_mechanic_profile(descriptor)
    descriptor["synergy"] = action_synergy_profile(
        state_payload if state_payload is not None else _serialize(state),
        descriptor,
    )
    return descriptor


def _action_option_slot_descriptor(
    state_payload: Mapping[str, Any] | None,
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    if state_payload is None:
        return {}
    descriptor_type = str(descriptor.get("type", ""))
    target_id = str(descriptor.get("target_id", ""))
    content_id = _action_option_content_id(descriptor)
    slots: list[dict[str, Any]] = []
    if descriptor_type.startswith("take_reward") or descriptor_type in {
        "skip_reward",
        "proceed",
    }:
        slots = reward_option_slots(state_payload, limit=REWARD_OPTION_SLOT_LIMIT)
    elif descriptor_type in {"shop_buy", "shop_remove_card"}:
        slots = shop_option_slots(state_payload, limit=SHOP_OPTION_SLOT_LIMIT)
    for slot in slots:
        slot_map = _mapping(slot)
        slot_target = str(_mapping(slot_map.get("descriptor")).get("target_id", ""))
        if descriptor_type == "proceed" and str(slot_map.get("kind", "")) == "proceed":
            return dict(slot_map)
        if descriptor_type == "skip_reward":
            reward_choice = _mapping(descriptor.get("reward_choice"))
            selection_set_id = str(reward_choice.get("selection_set_id", ""))
            skip_kind = str(reward_choice.get("skip_kind", ""))
            slot_content_id = str(slot_map.get("content_id", ""))
            if (
                str(slot_map.get("kind", "")) == "skip"
                and slot_content_id in {selection_set_id, skip_kind}
            ):
                return dict(slot_map)
        if target_id and slot_target and slot_target == target_id:
            return dict(slot_map)
        if content_id and str(slot_map.get("content_id", "")) == content_id:
            return dict(slot_map)
    return {}


def _action_option_content_id(descriptor: Mapping[str, Any]) -> str:
    for key, identity_key in (
        ("reward_choice", "content_id"),
        ("item", "item_id"),
        ("card", "card_id"),
        ("relic", "relic_id"),
        ("potion", "potion_id"),
    ):
        value = _mapping(descriptor.get(key)).get(identity_key)
        if value not in (None, ""):
            return str(value)
    return ""


def _action_card_descriptor(state: Any, action: Action) -> dict[str, Any]:
    card_instance_id = _action_card_instance_id(action)
    if card_instance_id is not None:
        reference = _find_card_reference(state, card_instance_id)
        if reference is not None:
            card, zone, position = reference
            return _card_descriptor(card, zone=zone, position=position)

    if (
        action.type == ActionType.TAKE_REWARD_CARD
        and action.target_id is not None
        and action.target_id.startswith("reward:remove_card:")
    ):
        card = _reward_optional_remove_card_for_target(state, action.target_id)
        if card is not None:
            position = _master_deck_position(state, getattr(card, "instance_id", ""))
            descriptor = _card_descriptor(
                card,
                zone="master_deck",
                position=position,
            )
            descriptor["reward_remove"] = True
            return descriptor

    reward_card_id = _reward_card_id_for_action(state, action)
    if reward_card_id is not None:
        return _card_id_descriptor(
            reward_card_id,
            zone="reward",
            position=_reward_position_for_action(state, action),
        )

    item = _shop_item_for_action(state, action)
    if item and str(getattr(item, "kind", "")) in {"card", "colorless_card"}:
        return _card_id_descriptor(
            getattr(item, "item_id", ""),
            zone="shop",
            position=_shop_item_index(action.target_id or ""),
        )
    return {}


def _action_card_instance_id(action: Action) -> str | None:
    if action.card_instance_id is not None:
        return action.card_instance_id
    if action.type in {ActionType.SMITH, ActionType.TOKE}:
        return action.target_id
    if action.type == ActionType.SHOP_BUY and action.target_id is not None:
        return _shop_remove_card_instance_id(action.target_id)
    return None


def _action_selected_card_descriptors(state: Any, action: Action) -> list[dict[str, Any]]:
    card_ids = _action_payload_card_instance_ids(action)
    if not card_ids and action.type == ActionType.CHOOSE_EVENT and action.card_instance_id:
        card_ids = (action.card_instance_id,)
    if not card_ids:
        return []

    selected: list[dict[str, Any]] = []
    for card_instance_id in card_ids:
        reference = _find_card_reference(state, card_instance_id)
        if reference is None:
            continue
        card, zone, position = reference
        selected.append(_card_descriptor(card, zone=zone, position=position))
    return selected


def _action_payload_card_instance_ids(action: Action) -> tuple[str, ...]:
    raw_ids = action.payload.get("card_instance_ids")
    if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes, bytearray)):
        return tuple(str(card_id) for card_id in raw_ids)
    return ()


def _aggregate_selected_card_descriptor(cards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    effect_keys: set[str] = set()
    effect_amounts: dict[str, float] = {}
    card_ids: list[str] = []
    positions: list[int] = []
    upgraded_count = 0
    exhaust_count = 0
    for card in cards:
        card_id = str(card.get("card_id", ""))
        if card_id:
            card_ids.append(card_id)
        effect_keys.update(str(key) for key in _sequence(card.get("effect_keys")))
        for key, value in _mapping(card.get("effect_amounts")).items():
            effect_amounts[str(key)] = effect_amounts.get(str(key), 0.0) + _float(value)
        if card.get("upgraded"):
            upgraded_count += 1
        if card.get("exhausts"):
            exhaust_count += 1
        position = _parse_int(card.get("position"))
        if position is not None:
            positions.append(position)

    descriptor: dict[str, Any] = {
        "card_id": "selected:" + "+".join(card_ids[:8]),
        "type": "selection",
        "cost": sum(_int(card.get("cost")) for card in cards),
        "target": "",
        "upgraded": upgraded_count == len(cards) if cards else False,
        "upgraded_count": upgraded_count,
        "exhausts": bool(exhaust_count),
        "exhaust_count": exhaust_count,
        "effect_keys": sorted(effect_keys),
        "effect_amounts": effect_amounts,
        "effects": {"selected_card_ids": tuple(card_ids)},
        "zone": "selection",
        "position": min(positions, default=0),
        "selected_count": len(cards),
    }
    descriptor.update(
        card_slot_summary(
            descriptor,
            zone="selection",
            position=_int(descriptor.get("position")),
        )
    )
    return descriptor


def _find_card_reference(state: Any, instance_id: str) -> tuple[Any, str, int] | None:
    combat = getattr(state, "combat", None)
    if combat is not None:
        for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
            for index, card in enumerate(getattr(combat, zone, ())):
                if getattr(card, "instance_id", None) == instance_id:
                    return card, zone, index
    for index, card in enumerate(getattr(state, "master_deck", ())):
        if getattr(card, "instance_id", None) == instance_id:
            return card, "master_deck", index
    return None


def _master_deck_position(state: Any, instance_id: str) -> int:
    for index, card in enumerate(getattr(state, "master_deck", ())):
        if getattr(card, "instance_id", None) == instance_id:
            return index
    return 0


def _card_descriptor(
    card: Any,
    *,
    zone: str | None = None,
    position: int | None = None,
) -> dict[str, Any]:
    effects = _mapping(getattr(card, "effects", {}))
    descriptor: dict[str, Any] = {
        "instance_id": str(getattr(card, "instance_id", "")),
        "card_id": str(getattr(card, "card_id", "")),
        "type": _enum_or_str(getattr(card, "type", "")),
        "cost": _int(getattr(card, "cost", 0)),
        "target": _enum_or_str(getattr(card, "target", "")),
        "upgraded": bool(getattr(card, "upgraded", False)),
        "exhausts": bool(getattr(card, "exhausts", False)),
        "effect_keys": sorted(str(key) for key in _effect_keys(effects)),
        "effect_amounts": _effect_amounts(effects),
        "effects": _json_safe(effects),
    }
    if zone is not None:
        descriptor["zone"] = zone
    if position is not None:
        descriptor["position"] = position
        if zone == "draw_pile":
            descriptor["position_from_top"] = position
    descriptor.update(
        card_slot_summary(
            descriptor,
            zone=zone or str(descriptor.get("zone", "")),
            position=position if position is not None else _int(descriptor.get("position")),
        )
    )
    return descriptor


def _card_id_descriptor(
    card_id: object,
    *,
    zone: str | None = None,
    position: int | None = None,
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "card_id": str(card_id),
        "type": "",
        "cost": 0,
        "target": "",
        "upgraded": False,
        "exhausts": False,
        "effect_keys": (),
        "effect_amounts": {},
        "effects": {},
    }
    if zone is not None:
        descriptor["zone"] = zone
    if position is not None:
        descriptor["position"] = position
    descriptor.update(
        card_slot_summary(
            descriptor,
            zone=zone or str(descriptor.get("zone", "")),
            position=position if position is not None else _int(descriptor.get("position")),
        )
    )
    return descriptor


def _action_node_descriptor(state: Any, action: Action) -> dict[str, Any]:
    if action.type != ActionType.CHOOSE_NODE or action.target_id is None:
        return {}
    game_map = getattr(state, "map", None)
    if game_map is None:
        return {}
    node_by_id = getattr(game_map, "node_by_id", {})
    node = node_by_id.get(action.target_id) if isinstance(node_by_id, Mapping) else None
    if node is None:
        for candidate in getattr(game_map, "nodes", ()):
            if getattr(candidate, "node_id", None) == action.target_id:
                node = candidate
                break
    if node is None:
        return {}
    descriptor = {
        "kind": _enum_or_str(getattr(node, "kind", "")),
        "act": _int(getattr(node, "act", 0)),
        "floor": _int(getattr(node, "floor", 0)),
        "lane": _int(getattr(node, "lane", 0)),
    }
    path = _map_path_summary_for_node(game_map, action.target_id, state=state)
    if path:
        descriptor["path"] = path
    return descriptor


def _action_target_descriptor(state: Any, action: Action) -> dict[str, Any]:
    target_id = action.target_id
    if target_id is None:
        return {}
    combat = getattr(state, "combat", None)
    if combat is None:
        return {}
    if target_id == "player":
        player = getattr(combat, "player", getattr(state, "player", None))
        if player is None:
            return {}
        statuses = dict(getattr(player, "statuses", {}))
        return {
            "kind": "player",
            "target_id": "player",
            "hp": _int(getattr(player, "hp", 0)),
            "max_hp": _int(getattr(player, "max_hp", 0)),
            "block": _int(getattr(player, "block", 0)),
            "statuses": statuses,
            "status_keys": sorted(str(key) for key in statuses),
            "status_total": sum(_int(value) for value in statuses.values()),
        }
    for index, monster in enumerate(getattr(combat, "monsters", ())):
        if getattr(monster, "monster_id", None) == target_id:
            return _monster_descriptor(monster, position=index)
    return {"kind": "unknown", "target_id": str(target_id)}


def _monster_descriptor(monster: Any, *, position: int) -> dict[str, Any]:
    statuses = dict(getattr(monster, "statuses", {}))
    metadata = _mapping(getattr(monster, "metadata", {}))
    hp = _int(getattr(monster, "hp", 0))
    max_hp = max(1, _int(getattr(monster, "max_hp", 1)))
    monster_id = str(getattr(monster, "monster_id", ""))
    return {
        "kind": "monster",
        "target_id": monster_id,
        "monster_id": monster_id,
        "source_monster_id": str(metadata.get("source_monster_id", monster_id)),
        "name": str(getattr(monster, "name", "")),
        "position": position,
        "slot_index": _int(metadata.get("slot_index", position)),
        "hp": hp,
        "max_hp": max_hp,
        "hp_fraction": hp / max_hp,
        "block": _int(getattr(monster, "block", 0)),
        "intent": _optional_text(getattr(monster, "intent", "")) or "",
        "intent_damage": _int(getattr(monster, "intent_damage", 0)),
        "intent_block": _int(getattr(monster, "intent_block", 0)),
        "move_id": _optional_text(getattr(monster, "move_id", "")) or "",
        "next_move_id": _optional_text(getattr(monster, "next_move_id", "")) or "",
        "hit_count": _int(getattr(monster, "hit_count", 1)),
        "statuses": statuses,
        "status_keys": sorted(str(key) for key in statuses),
        "status_total": sum(_int(value) for value in statuses.values()),
        "alive": hp > 0,
        "metadata": dict(metadata),
    }


def _map_path_summary_for_node(
    game_map: Any,
    start_node_id: str,
    *,
    state: Any | None = None,
    max_depth: int = 16,
) -> dict[str, Any]:
    node_by_id = _map_nodes_by_id(game_map)
    if start_node_id not in node_by_id:
        return {}
    outgoing = _map_outgoing_by_id(game_map)
    completed = {str(node_id) for node_id in getattr(game_map, "completed_node_ids", ())}
    paths: list[list[str]] = []
    stack: list[tuple[str, list[str]]] = [(start_node_id, [start_node_id])]
    while stack and len(paths) < 64:
        node_id, path = stack.pop()
        if len(path) >= max_depth or _node_kind(node_by_id.get(node_id)) == "boss":
            paths.append(path)
            continue
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
        return {}
    path_summaries = [_path_counts(path, node_by_id) for path in paths]
    depths = [summary["depth"] for summary in path_summaries]
    first_rest_depths = [
        summary["first_rest_depth"]
        for summary in path_summaries
        if summary["first_rest_depth"] > 0
    ]
    current_hp_fraction = _player_hp_fraction(state)
    upgradeable_cards = _upgradeable_card_count(state)
    avg_aggression = _average(summary["aggression_score"] for summary in path_summaries)
    avg_rests = _average(summary["rest_count"] for summary in path_summaries)
    avg_card_rewards = _average(
        summary["future_card_reward_groups"] for summary in path_summaries
    )
    avg_relic_rewards = _average(
        summary["future_relic_rewards"] for summary in path_summaries
    )
    avg_fights = _average(summary["fight_count"] for summary in path_summaries)
    return {
        "path_count": len(path_summaries),
        "min_depth": min(depths),
        "max_depth": max(depths),
        "avg_depth": sum(depths) / max(1, len(depths)),
        "max_elites": max(summary["elite_count"] for summary in path_summaries),
        "min_elites": min(summary["elite_count"] for summary in path_summaries),
        "avg_elites": _average(summary["elite_count"] for summary in path_summaries),
        "max_rests": max(summary["rest_count"] for summary in path_summaries),
        "min_rests": min(summary["rest_count"] for summary in path_summaries),
        "avg_rests": avg_rests,
        "max_shops": max(summary["shop_count"] for summary in path_summaries),
        "avg_shops": _average(summary["shop_count"] for summary in path_summaries),
        "max_events": max(summary["event_count"] for summary in path_summaries),
        "avg_events": _average(summary["event_count"] for summary in path_summaries),
        "max_monsters": max(summary["monster_count"] for summary in path_summaries),
        "avg_monsters": _average(summary["monster_count"] for summary in path_summaries),
        "max_treasures": max(summary["treasure_count"] for summary in path_summaries),
        "avg_treasures": _average(summary["treasure_count"] for summary in path_summaries),
        "min_fights": min(summary["fight_count"] for summary in path_summaries),
        "max_fights": max(summary["fight_count"] for summary in path_summaries),
        "avg_fights": avg_fights,
        "has_boss_path": any(summary["ends_at_boss"] for summary in path_summaries),
        "boss_path_count": sum(1 for summary in path_summaries if summary["ends_at_boss"]),
        "boss_path_fraction": sum(1 for summary in path_summaries if summary["ends_at_boss"])
        / max(1, len(path_summaries)),
        "min_aggression_score": min(
            summary["aggression_score"] for summary in path_summaries
        ),
        "max_aggression_score": max(
            summary["aggression_score"] for summary in path_summaries
        ),
        "avg_aggression_score": avg_aggression,
        "min_safety_score": min(summary["safety_score"] for summary in path_summaries),
        "max_safety_score": max(summary["safety_score"] for summary in path_summaries),
        "avg_safety_score": _average(summary["safety_score"] for summary in path_summaries),
        "future_card_reward_groups_min": min(
            summary["future_card_reward_groups"] for summary in path_summaries
        ),
        "future_card_reward_groups_max": max(
            summary["future_card_reward_groups"] for summary in path_summaries
        ),
        "future_card_reward_groups_avg": avg_card_rewards,
        "future_relic_rewards_min": min(
            summary["future_relic_rewards"] for summary in path_summaries
        ),
        "future_relic_rewards_max": max(
            summary["future_relic_rewards"] for summary in path_summaries
        ),
        "future_relic_rewards_avg": avg_relic_rewards,
        "paths_with_rest_fraction": len(first_rest_depths) / max(1, len(path_summaries)),
        "first_rest_depth_min": min(first_rest_depths, default=0),
        "first_rest_depth_avg": _average(first_rest_depths),
        "fights_before_first_rest_min": min(
            summary["fights_before_first_rest"] for summary in path_summaries
        ),
        "fights_before_first_rest_avg": _average(
            summary["fights_before_first_rest"] for summary in path_summaries
        ),
        "elites_before_first_rest_max": max(
            summary["elites_before_first_rest"] for summary in path_summaries
        ),
        "upgrade_opportunity_avg": avg_rests,
        "heal_opportunity_avg": avg_rests,
        "current_hp_fraction": current_hp_fraction,
        "upgradeable_card_count": upgradeable_cards,
        "rest_upgrade_flexibility": min(avg_rests, float(upgradeable_cards)),
        "low_hp_aggression_risk_avg": avg_aggression * (1.0 - current_hp_fraction),
        "boss_prep_score_avg": (
            avg_relic_rewards * 2.0
            + avg_card_rewards
            + avg_rests * 0.75
            - avg_fights * 0.35
        ),
    }


def _path_counts(path: list[str], node_by_id: Mapping[str, Any]) -> dict[str, Any]:
    kinds = [_node_kind(node_by_id.get(node_id)) for node_id in path]
    boss_count = kinds.count("boss")
    fight_count = kinds.count("monster") + kinds.count("elite") + boss_count
    first_rest_index = next(
        (index for index, kind in enumerate(kinds) if kind == "rest"),
        None,
    )
    if first_rest_index is None:
        first_rest_depth = 0
        prefix = kinds
    else:
        first_rest_depth = first_rest_index + 1
        prefix = kinds[:first_rest_index]
    elites = kinds.count("elite")
    monsters = kinds.count("monster")
    rests = kinds.count("rest")
    shops = kinds.count("shop")
    events = kinds.count("event")
    treasures = kinds.count("treasure")
    return {
        "depth": len(path),
        "elite_count": elites,
        "rest_count": rests,
        "shop_count": shops,
        "event_count": events,
        "monster_count": monsters,
        "treasure_count": treasures,
        "fight_count": fight_count,
        "future_card_reward_groups": monsters + elites + boss_count,
        "future_relic_rewards": elites + treasures + boss_count,
        "first_rest_depth": first_rest_depth,
        "fights_before_first_rest": (
            prefix.count("monster") + prefix.count("elite") + prefix.count("boss")
        ),
        "elites_before_first_rest": prefix.count("elite"),
        "aggression_score": (
            elites * 3.0
            + monsters
            + boss_count * 2.0
            - rests * 1.2
            - shops * 0.4
            - treasures * 0.2
        ),
        "safety_score": (
            rests * 1.8
            + shops * 0.5
            + treasures * 0.7
            - elites * 2.5
            - monsters * 0.5
            - boss_count
        ),
        "ends_at_boss": bool(kinds and kinds[-1] == "boss"),
    }


def _map_nodes_by_id(game_map: Any) -> dict[str, Any]:
    node_by_id = getattr(game_map, "node_by_id", {})
    if isinstance(node_by_id, Mapping):
        return {str(node_id): node for node_id, node in node_by_id.items()}
    return {str(getattr(node, "node_id", "")): node for node in getattr(game_map, "nodes", ())}


def _map_outgoing_by_id(game_map: Any) -> dict[str, tuple[str, ...]]:
    outgoing_by_id = getattr(game_map, "outgoing_by_id", {})
    if isinstance(outgoing_by_id, Mapping):
        return {
            str(node_id): tuple(str(next_id) for next_id in _sequence(next_ids))
            for node_id, next_ids in outgoing_by_id.items()
        }
    outgoing: dict[str, list[str]] = {}
    for edge in getattr(game_map, "edges", ()):
        outgoing.setdefault(str(getattr(edge, "from_id", "")), []).append(
            str(getattr(edge, "to_id", ""))
        )
    return {node_id: tuple(next_ids) for node_id, next_ids in outgoing.items()}


def _node_kind(node: Any) -> str:
    if node is None:
        return ""
    return _enum_or_str(getattr(node, "kind", ""))


def _player_hp_fraction(state: Any | None) -> float:
    player = getattr(state, "player", None)
    max_hp = max(1, _int(getattr(player, "max_hp", 1)))
    return max(0.0, min(1.0, _float(getattr(player, "hp", 0)) / max_hp))


def _upgradeable_card_count(state: Any | None) -> int:
    return sum(
        1
        for card in getattr(state, "master_deck", ())
        if not bool(getattr(card, "upgraded", False))
    )


def _average(values: Any) -> float:
    items = [_float(value) for value in values]
    return sum(items) / max(1, len(items))


def _action_item_descriptor(state: Any, action: Action) -> dict[str, Any]:
    item = _shop_item_for_action(state, action)
    if item is None:
        return {}
    rarity = getattr(item, "rarity", None)
    position = _shop_item_index(action.target_id or "")
    return {
        "slot_id": str(getattr(item, "slot_id", "")),
        "slot_index": position,
        "position": position,
        "item_id": str(getattr(item, "item_id", "")),
        "kind": str(getattr(item, "kind", "")),
        "rarity": "" if rarity is None else str(rarity),
        "price": _int(getattr(item, "price", 0)),
        "base_price": _int(getattr(item, "base_price", 0)),
        "purchased": bool(getattr(item, "purchased", False)),
    }


def _action_potion_descriptor(state: Any, action: Action) -> dict[str, Any]:
    potion_id: object | None = None
    slot_id: str | None = None
    if action.type == ActionType.USE_POTION:
        potion_id = _mapping(action.payload).get("potion_id")
        slot_id = _optional_text(_mapping(action.payload).get("potion_slot"))
    elif action.type == ActionType.THROW_POTION_AT_MERCHANT:
        potion_id = "foul_potion"
        slot_id = _optional_text(_mapping(action.payload).get("potion_slot"))
        if slot_id is None:
            slot_id = _first_potion_slot_id(state, "foul_potion")
    elif action.type == ActionType.DISCARD_POTION:
        slot_id = action.target_id
        potion_id = _potion_id_for_slot(state, action.target_id)
    elif action.type == ActionType.TAKE_REWARD_POTION:
        potion_id = _reward_potion_id_for_action(state, action)
    elif action.type == ActionType.SHOP_BUY:
        item = _shop_item_for_action(state, action)
        if item and str(getattr(item, "kind", "")) == "potion":
            potion_id = getattr(item, "item_id", "")
            slot_id = str(getattr(item, "slot_id", "")) or action.target_id
    if potion_id is None:
        return {}
    slot_index = _potion_slot_index(slot_id)
    if slot_index is None and action.type == ActionType.SHOP_BUY:
        slot_index = _shop_item_index(action.target_id or "")
    return {
        "potion_id": str(potion_id),
        "slot_id": slot_id,
        "slot_index": slot_index,
    }


def _action_potion_strategy_descriptor(
    state: Any,
    action: Action,
    potion: Mapping[str, Any],
    *,
    preview: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    potion_id = _normalized_content_id(potion.get("potion_id"))
    if not potion_id:
        return {}
    preview_map = _mapping(preview)
    effect_summary = _potion_effect_summary(potion_id, action)
    belt = _potion_belt_summary(state)
    combat = getattr(state, "combat", None)
    combat_summary = _potion_combat_timing_summary(
        combat,
        action,
        effect_summary,
        preview_map,
    )
    takes_slot = action.type in {ActionType.TAKE_REWARD_POTION, ActionType.SHOP_BUY}
    frees_slot = action.type in {
        ActionType.USE_POTION,
        ActionType.DISCARD_POTION,
        ActionType.THROW_POTION_AT_MERCHANT,
    }
    requires_discard = bool(takes_slot and _int(belt.get("open_slots")) <= 0)
    save_priority = _potion_save_priority(potion_id, effect_summary, combat_summary)
    roles = sorted(
        {
            *_potion_effect_roles(effect_summary),
            *_potion_timing_roles(action, effect_summary, combat_summary, belt),
        }
    )
    return {
        "potion_id": potion_id,
        "slot_index": potion.get("slot_index"),
        "action_kind": action.type.value,
        "roles": roles,
        "combat_present": combat is not None,
        "capacity": _int(belt.get("capacity")),
        "current_potions": _int(belt.get("current_potions")),
        "open_slots": _int(belt.get("open_slots")),
        "slot_pressure": _float(belt.get("slot_pressure")),
        "belt_full": bool(belt.get("belt_full")),
        "frees_slot": frees_slot,
        "takes_slot": takes_slot,
        "requires_discard": requires_discard,
        "save_priority": save_priority,
        **effect_summary,
        **combat_summary,
    }


def _potion_effect_summary(potion_id: str, action: Action) -> dict[str, Any]:
    normalized = _normalized_content_id(potion_id)
    summary: dict[str, Any] = {
        "damage": 0,
        "aoe_damage": 0,
        "self_damage": 0,
        "block": 0,
        "draw": 0,
        "energy": 0,
        "heal": 0,
        "max_hp_delta": 0,
        "poison": 0,
        "weak": 0,
        "vulnerable": 0,
        "strength": 0,
        "dexterity": 0,
        "focus": 0,
        "regen": 0,
        "intangible": 0,
        "buffer": 0,
        "status_enemy": 0,
        "status_self": 0,
        "card_generation": 0,
        "card_recovery": 0,
        "random_card_play": 0,
        "free_card_play": 0,
        "hand_control": 0,
        "potion_generation": 0,
        "persistent_setup": 0,
        "temporary_setup": 0,
        "passive_revive": 0,
    }
    normalization_target_id = action.target_id
    if action.type == ActionType.THROW_POTION_AT_MERCHANT:
        if action.target_id == "fake_merchant" or _mapping(action.payload).get(
            "merchant"
        ) == "fake_merchant":
            return summary
        normalization_target_id = "merchant"
    normalization = normalize_potion_use(normalized, target_id=normalization_target_id)
    for effect in normalization.effects:
        kind = str(getattr(effect, "kind", ""))
        amount = _int(getattr(effect, "amount", 0))
        target = str(getattr(effect, "target", "") or "")
        status = _normalized_content_id(getattr(effect, "status", ""))
        duration = _int(getattr(effect, "duration", 0))
        if kind == "damage":
            if target in {"all_enemies", "all_combatants"}:
                summary["aoe_damage"] += amount
                if target == "all_combatants":
                    summary["self_damage"] += amount
            else:
                summary["damage"] += amount
        elif kind == "block":
            summary["block"] += amount
        elif kind == "draw":
            summary["draw"] += amount
        elif kind == "energy":
            summary["energy"] += amount
        elif kind == "heal":
            summary["heal"] += amount
        elif kind == "max_hp":
            summary["max_hp_delta"] += amount
        elif kind == "status":
            _add_status_summary(summary, status, amount, target)
            if duration > 1:
                summary["persistent_setup"] += 1
        elif kind == "temporary_status":
            _add_status_summary(summary, status, amount, target)
            summary["temporary_setup"] += 1
        elif kind in {"start_turn_energy", "next_turn_block"}:
            summary["persistent_setup"] += max(1, duration)
            if kind == "start_turn_energy":
                summary["energy"] += amount
            elif kind == "next_turn_block":
                summary["block"] += amount
        elif kind == "upgrade_hand":
            summary["hand_control"] += 1
            summary["persistent_setup"] += 1
        elif kind in {"channel_orb", "orb_slot_delta", "player_resource"}:
            summary["persistent_setup"] += 1
        elif kind in {"exhaust_hand", "block_multiplier"}:
            summary["hand_control"] += 1

    _add_runtime_potion_summary(summary, normalized)
    return summary


def _add_status_summary(
    summary: dict[str, Any],
    status: str,
    amount: int,
    target: str,
) -> None:
    if target in {"enemy", "all_enemies"}:
        summary["status_enemy"] += max(1, amount)
    else:
        summary["status_self"] += max(1, amount)
    if status in {
        "poison",
        "weak",
        "vulnerable",
        "strength",
        "dexterity",
        "focus",
        "regen",
        "intangible",
        "buffer",
    }:
        summary[status] += amount


def _add_runtime_potion_summary(summary: dict[str, Any], potion_id: str) -> None:
    if potion_id in {"attack_potion", "skill_potion", "power_potion", "colorless_potion"}:
        summary["card_generation"] += 3
        summary["free_card_play"] += 1
        summary["hand_control"] += 1
        summary["temporary_setup"] += 1
    elif potion_id == "liquid_memories":
        summary["card_recovery"] += 1
        summary["free_card_play"] += 1
        summary["hand_control"] += 1
        summary["persistent_setup"] += 1
    elif potion_id == "distilled_chaos":
        summary["random_card_play"] += 3
        summary["temporary_setup"] += 1
    elif potion_id == "duplicator":
        summary["free_card_play"] += 1
        summary["temporary_setup"] += 1
    elif potion_id in {"gamblers_brew", "snecko_oil", "bottled_potential"}:
        summary["hand_control"] += 1
        summary["draw"] += {"gamblers_brew": 5, "snecko_oil": 7, "bottled_potential": 5}[
            potion_id
        ]
        summary["temporary_setup"] += 1
    elif potion_id in {"droplet_of_precognition", "touch_of_insanity", "ashwater"}:
        summary["hand_control"] += 1
        summary["persistent_setup"] += 1
    elif potion_id == "entropic_brew":
        summary["potion_generation"] += 1
        summary["persistent_setup"] += 1
    elif potion_id == "cunning_potion":
        summary["card_generation"] += 3
        summary["damage"] += 18
        summary["free_card_play"] += 1
    elif potion_id in {"cosmic_concoction", "orobic_acid", "pot_of_ghouls"}:
        summary["card_generation"] += 3 if potion_id != "pot_of_ghouls" else 2
        summary["temporary_setup"] += 1
    elif potion_id in {"clarity", "radiant_tincture", "powdered_demise"}:
        summary["persistent_setup"] += 3
    elif potion_id == "fairy_in_a_bottle":
        summary["passive_revive"] = 1


def _potion_belt_summary(state: Any) -> dict[str, Any]:
    flags = _mapping(getattr(state, "flags", {}))
    explicit_slots = _parse_int(flags.get("potion_slots"))
    potions = tuple(str(potion_id) for potion_id in _sequence(getattr(state, "potions", ())))
    if explicit_slots is not None:
        capacity = max(0, explicit_slots)
        open_slots = max(0, capacity - len(potions))
        base_slots = capacity
    else:
        base_slots = _parse_int(flags.get("base_potion_slots")) or 3
        result = potion_capacity(
            base_slots=base_slots,
            ascension_level=_int(getattr(state, "ascension", 0)),
            relics=tuple(str(relic_id) for relic_id in _sequence(getattr(state, "relics", ()))),
            bonus_slots=_parse_int(flags.get("bonus_potion_slots")) or 0,
            current_potions=potions,
        )
        capacity = result.capacity
        open_slots = result.open_slots
    current = len(potions)
    return {
        "capacity": capacity,
        "base_slots": base_slots,
        "current_potions": current,
        "open_slots": open_slots,
        "slot_pressure": current / max(1, capacity),
        "belt_full": current >= capacity,
    }


def _potion_combat_timing_summary(
    combat: Any,
    action: Action,
    effect_summary: Mapping[str, Any],
    preview: Mapping[str, Any],
) -> dict[str, Any]:
    if combat is None:
        return {
            "incoming_damage": 0,
            "baseline_damage_if_end_turn": 0,
            "projected_damage_taken_after_use": 0,
            "damage_prevented_this_turn": 0,
            "survival_enabling": False,
            "lethal_now": False,
            "target_lethal_now": False,
            "kills_now": 0,
            "target_hp": 0,
            "overkill_damage": 0,
        }
    player = getattr(combat, "player", None)
    player_hp = _int(getattr(player, "hp", 0))
    player_block = _int(getattr(player, "block", 0))
    incoming = _incoming_combat_damage(combat)
    baseline_damage = max(0, incoming - player_block)
    preview_projected = preview.get("projected_damage_taken_after_end")
    projected_damage = (
        _int(preview_projected)
        if preview_projected is not None
        else max(0, incoming - player_block - _int(effect_summary.get("block")))
    )
    target_hp = _target_hp(combat, action.target_id)
    direct_damage = _int(effect_summary.get("damage"))
    aoe_damage = _int(effect_summary.get("aoe_damage"))
    target_hp_delta = _int(preview.get("target_hp_delta"))
    target_lethal = bool(
        target_hp > 0
        and (
            target_hp_delta <= -target_hp
            or (direct_damage > 0 and direct_damage >= target_hp)
        )
    )
    aoe_kills = _aoe_kill_count(combat, aoe_damage)
    kills_now = max(_int(preview.get("kills")), int(target_lethal), aoe_kills)
    lethal_now = bool(preview.get("combat_ended")) or kills_now >= _alive_monster_count(combat) > 0
    overkill = max(0, direct_damage - target_hp) if target_hp > 0 and direct_damage else 0
    return {
        "incoming_damage": incoming,
        "baseline_damage_if_end_turn": baseline_damage,
        "projected_damage_taken_after_use": projected_damage,
        "damage_prevented_this_turn": max(0, baseline_damage - projected_damage),
        "survival_enabling": baseline_damage >= player_hp > projected_damage,
        "lethal_now": lethal_now,
        "target_lethal_now": target_lethal,
        "kills_now": kills_now,
        "target_hp": target_hp,
        "overkill_damage": overkill,
    }


def _incoming_combat_damage(combat: Any) -> int:
    total = 0
    for monster in getattr(combat, "monsters", ()):
        if _int(getattr(monster, "hp", 0)) <= 0:
            continue
        intent = str(getattr(monster, "intent", "") or "")
        if "attack" not in intent:
            continue
        total += _int(getattr(monster, "intent_damage", 0)) * max(
            1,
            _int(getattr(monster, "hit_count", 1)),
        )
    return total


def _target_hp(combat: Any, target_id: str | None) -> int:
    if target_id is None or target_id == "player":
        return 0
    for monster in getattr(combat, "monsters", ()):
        if str(getattr(monster, "monster_id", "")) == str(target_id):
            return _int(getattr(monster, "hp", 0))
    return 0


def _alive_monster_count(combat: Any) -> int:
    return sum(
        1
        for monster in getattr(combat, "monsters", ())
        if _int(getattr(monster, "hp", 0)) > 0
    )


def _aoe_kill_count(combat: Any, damage: int) -> int:
    if damage <= 0:
        return 0
    return sum(
        1
        for monster in getattr(combat, "monsters", ())
        if 0 < _int(getattr(monster, "hp", 0)) <= damage
    )


def _potion_save_priority(
    potion_id: str,
    effect_summary: Mapping[str, Any],
    combat_summary: Mapping[str, Any],
) -> float:
    score = 0.0
    if effect_summary.get("passive_revive"):
        score += 1.0
    if effect_summary.get("intangible") or effect_summary.get("buffer"):
        score += 0.8
    if effect_summary.get("potion_generation"):
        score += 0.5
    if effect_summary.get("card_generation") or effect_summary.get("card_recovery"):
        score += 0.35
    if combat_summary.get("lethal_now") or combat_summary.get("survival_enabling"):
        score -= 0.7
    if potion_id in {"fairy_in_a_bottle", "ghost_in_a_jar", "lucky_tonic"}:
        score += 0.25
    return max(0.0, min(1.0, score))


def _potion_effect_roles(effect_summary: Mapping[str, Any]) -> tuple[str, ...]:
    roles: list[str] = []
    if _int(effect_summary.get("damage")) or _int(effect_summary.get("aoe_damage")):
        roles.append("damage")
    if (
        _int(effect_summary.get("block"))
        or effect_summary.get("intangible")
        or effect_summary.get("buffer")
    ):
        roles.append("defense")
    if _int(effect_summary.get("draw")) or _int(effect_summary.get("energy")):
        roles.append("tempo")
    if _int(effect_summary.get("card_generation")):
        roles.append("card_generation")
    if _int(effect_summary.get("card_recovery")):
        roles.append("card_recovery")
    if _int(effect_summary.get("hand_control")):
        roles.append("hand_control")
    if _int(effect_summary.get("potion_generation")):
        roles.append("potion_generation")
    if _int(effect_summary.get("persistent_setup")):
        roles.append("persistent_setup")
    if _int(effect_summary.get("temporary_setup")):
        roles.append("temporary_setup")
    if _int(effect_summary.get("passive_revive")):
        roles.append("passive_revive")
    return tuple(roles)


def _potion_timing_roles(
    action: Action,
    effect_summary: Mapping[str, Any],
    combat_summary: Mapping[str, Any],
    belt: Mapping[str, Any],
) -> tuple[str, ...]:
    roles: list[str] = []
    if combat_summary.get("lethal_now") or combat_summary.get("target_lethal_now"):
        roles.append("lethal_now")
    if combat_summary.get("survival_enabling"):
        roles.append("prevents_death")
    if _int(combat_summary.get("damage_prevented_this_turn")) > 0:
        roles.append("prevents_damage")
    if _int(effect_summary.get("card_generation")) or _int(effect_summary.get("draw")):
        roles.append("preemptive_fight_setup")
    if action.type in {
        ActionType.USE_POTION,
        ActionType.DISCARD_POTION,
        ActionType.THROW_POTION_AT_MERCHANT,
    } and bool(belt.get("belt_full")):
        roles.append("frees_belt_slot")
    if action.type in {ActionType.TAKE_REWARD_POTION, ActionType.SHOP_BUY}:
        roles.append("pickup")
        if bool(belt.get("belt_full")):
            roles.append("blocked_by_full_belt")
    return tuple(roles)


def _normalized_content_id(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _action_relic_descriptor(state: Any, action: Action) -> dict[str, Any]:
    relic_id: object | None = None
    if action.type == ActionType.CHOOSE_ANCIENT:
        option = _ancient_option_for_action(state, action)
        relic_id = getattr(option, "relic_id", None) if option is not None else None
    elif action.type == ActionType.TAKE_REWARD_RELIC:
        relic_id = _reward_relic_id_for_action(state, action)
    elif action.type == ActionType.SHOP_BUY:
        item = _shop_item_for_action(state, action)
        if item and str(getattr(item, "kind", "")) == "relic":
            relic_id = getattr(item, "item_id", "")
    if relic_id is None:
        return {}
    return {"relic_id": str(relic_id)}


def _action_reward_descriptor(state: Any, action: Action) -> dict[str, Any]:
    if action.type not in {
        ActionType.TAKE_REWARD_GOLD,
        ActionType.TAKE_REWARD_RELIC,
        ActionType.TAKE_REWARD_CARD,
        ActionType.TAKE_REWARD_POTION,
        ActionType.SKIP_REWARD,
        ActionType.PROCEED,
    }:
        return {}
    reward = getattr(state, "reward", None)
    if reward is None:
        return {}
    if action.type == ActionType.PROCEED:
        bundle = _reward_bundle_descriptor(state)
        counts = _mapping(bundle.get("available_counts"))
        return {
            "source": str(getattr(reward, "source", "")),
            "forced": bool(getattr(reward, "forced", False)),
            "kind": "proceed",
            "content_id": "skip_all_remaining_rewards",
            "gold": 0,
            "position": None,
            "skips_remaining": True,
            "skips_selection": False,
            "skip_scope": "all_remaining",
            "available_remaining_count": _int(counts.get("total")),
            "selection_set_id": "all_remaining",
            "selection_set_size": _int(counts.get("total")),
            "closes_selection_set": True,
        }
    if action.type == ActionType.SKIP_REWARD:
        return _reward_skip_descriptor_for_action(state, action)
    if (
        action.type == ActionType.TAKE_REWARD_CARD
        and action.target_id is not None
        and action.target_id.startswith("reward:remove_card:")
    ):
        card = _reward_optional_remove_card_for_target(state, action.target_id)
        selection = _reward_selection_context_for_action(state, action)
        return {
            "source": str(getattr(reward, "source", "")),
            "forced": bool(getattr(reward, "forced", False)),
            "kind": "card_removal",
            "content_id": str(getattr(card, "card_id", "")) if card is not None else "",
            "card_instance_id": str(getattr(card, "instance_id", ""))
            if card is not None
            else "",
            "gold": 0,
            "position": _reward_position_for_action(state, action),
            "skips_remaining": False,
            "skips_selection": False,
            "skip_scope": "",
            **selection,
        }
    content_id = (
        _reward_card_id_for_action(state, action)
        or _reward_relic_id_for_action(state, action)
        or _reward_potion_id_for_action(state, action)
        or "gold"
    )
    gold = _int(getattr(reward, "gold", 0)) if action.type == ActionType.TAKE_REWARD_GOLD else 0
    selection = _reward_selection_context_for_action(state, action)
    return {
        "source": str(getattr(reward, "source", "")),
        "forced": bool(getattr(reward, "forced", False)),
        "kind": action.type.value.removeprefix("take_reward_"),
        "content_id": str(content_id),
        "gold": gold,
        "position": _reward_position_for_action(state, action),
        "skips_remaining": False,
        "skips_selection": False,
        "skip_scope": "",
        **selection,
    }


def _reward_bundle_descriptor(state: Any) -> dict[str, Any]:
    reward = getattr(state, "reward", None)
    if reward is None:
        return {}
    choices = _available_reward_choice_descriptors(state)
    counts = _reward_choice_counts(choices)
    claimed_counts = {
        "gold": int(bool(getattr(reward, "gold_claimed", False))),
        "primary_card_group": int(bool(getattr(reward, "card_claimed", False))),
        "card_groups": len(_sequence(getattr(reward, "claimed_card_option_group_indices", ()))),
        "fixed_cards": len(_sequence(getattr(reward, "claimed_card_indices", ()))),
        "relics": int(bool(getattr(reward, "relic_claimed", False)))
        + len(_sequence(getattr(reward, "claimed_relic_ids", ()))),
        "potions": int(bool(getattr(reward, "potion_claimed", False)))
        + len(_sequence(getattr(reward, "claimed_potion_indices", ()))),
    }
    can_skip = not bool(getattr(reward, "forced", False))
    metadata = _mapping(getattr(reward, "metadata", {}))
    return {
        "reward_id": str(getattr(reward, "reward_id", "")),
        "source": str(getattr(reward, "source", "")),
        "forced": bool(getattr(reward, "forced", False)),
        "can_skip": can_skip,
        "available_counts": counts,
        "claimed_counts": claimed_counts,
        "skipped_counts": _reward_skipped_counts(reward),
        "available_choices": choices,
        "available_skip_targets": _available_reward_skip_descriptors(state),
        "available_content_ids": [
            str(choice.get("content_id", "")) for choice in choices if choice.get("content_id")
        ],
        "metadata_keys": sorted(str(key) for key in metadata),
    }


def _available_reward_choice_descriptors(state: Any) -> list[dict[str, Any]]:
    reward = getattr(state, "reward", None)
    if reward is None:
        return []
    choices: list[dict[str, Any]] = []
    optional_remove_cards = _reward_optional_remove_cards(state)
    for index, card in enumerate(optional_remove_cards):
        choices.append(
            {
                "kind": "card_removal",
                "content_id": str(getattr(card, "card_id", "")),
                "card_instance_id": str(getattr(card, "instance_id", "")),
                "target_id": f"reward:remove_card:{index}",
                "selection_set_id": "card_removal",
                "selection_set_size": len(optional_remove_cards),
                "group_index": None,
                "card_index": index,
                "position": index,
            }
        )
    if (
        _int(getattr(reward, "gold", 0)) > 0
        and not bool(getattr(reward, "gold_claimed", False))
        and not bool(getattr(reward, "gold_skipped", False))
    ):
        choices.append(
            {
                "kind": "gold",
                "content_id": "gold",
                "target_id": "reward:gold",
                "amount": _int(getattr(reward, "gold", 0)),
                "selection_set_id": "gold",
                "selection_set_size": 1,
                "position": 0,
            }
        )
    if (
        getattr(reward, "relic_id", None)
        and not bool(getattr(reward, "relic_claimed", False))
        and not bool(getattr(reward, "relic_skipped", False))
    ):
        choices.append(
            {
                "kind": "relic",
                "content_id": str(getattr(reward, "relic_id", "")),
                "target_id": "reward:relic",
                "selection_set_id": "relic:single",
                "selection_set_size": 1,
                "position": 0,
            }
        )
    claimed_relics = {
        str(item)
        for item in (
            *_sequence(getattr(reward, "claimed_relic_ids", ())),
            *_sequence(getattr(reward, "skipped_relic_ids", ())),
        )
    }
    for index, relic_id in enumerate(_sequence(getattr(reward, "relic_ids", ()))):
        if str(relic_id) in claimed_relics:
            continue
        choices.append(
            {
                "kind": "relic",
                "content_id": str(relic_id),
                "target_id": f"reward:relic:{index}",
                "selection_set_id": f"relic:{index}",
                "selection_set_size": 1,
                "position": index,
            }
        )
    if _sequence(getattr(reward, "card_options", ())) and not bool(
        getattr(reward, "card_claimed", False)
    ) and not bool(getattr(reward, "card_skipped", False)):
        group = _sequence(getattr(reward, "card_options", ()))
        for index, card_id in enumerate(group):
            choices.append(
                {
                    "kind": "card",
                    "content_id": str(card_id),
                    "target_id": f"reward:card:{index}",
                    "selection_set_id": "card_options",
                    "selection_set_size": len(group),
                    "group_index": None,
                    "card_index": index,
                    "position": index,
                    "exclusive_within_set": True,
                }
            )
    claimed_fixed_cards = set(
        _sequence(getattr(reward, "claimed_card_indices", ()))
    ) | set(_sequence(getattr(reward, "skipped_card_indices", ())))
    for index, card_id in enumerate(_sequence(getattr(reward, "card_ids", ()))):
        if index in claimed_fixed_cards:
            continue
        choices.append(
            {
                "kind": "fixed_card",
                "content_id": str(card_id),
                "target_id": f"reward:fixed_card:{index}",
                "selection_set_id": f"fixed_card:{index}",
                "selection_set_size": 1,
                "position": index,
            }
        )
    claimed_card_groups = set(
        _sequence(getattr(reward, "claimed_card_option_group_indices", ()))
    ) | set(_sequence(getattr(reward, "skipped_card_option_group_indices", ())))
    for group_index, group in enumerate(_sequence(getattr(reward, "card_option_groups", ()))):
        if group_index in claimed_card_groups:
            continue
        group_items = _sequence(group)
        for card_index, card_id in enumerate(group_items):
            choices.append(
                {
                    "kind": "card_group",
                    "content_id": str(card_id),
                    "target_id": f"reward:card_group:{group_index}:{card_index}",
                    "selection_set_id": f"card_group:{group_index}",
                    "selection_set_size": len(group_items),
                    "group_index": group_index,
                    "card_index": card_index,
                    "position": len(
                        [
                            choice
                            for choice in choices
                            if str(choice.get("kind", "")).startswith("card")
                        ]
                    ),
                    "exclusive_within_set": True,
                }
            )
    if (
        getattr(reward, "potion_id", None)
        and not bool(getattr(reward, "potion_claimed", False))
        and not bool(getattr(reward, "potion_skipped", False))
    ):
        choices.append(
            {
                "kind": "potion",
                "content_id": str(getattr(reward, "potion_id", "")),
                "target_id": "reward:potion",
                "selection_set_id": "potion:single",
                "selection_set_size": 1,
                "position": 0,
            }
        )
    claimed_potions = set(_sequence(getattr(reward, "claimed_potion_indices", ()))) | set(
        _sequence(getattr(reward, "skipped_potion_indices", ()))
    )
    for index, potion_id in enumerate(_sequence(getattr(reward, "potion_ids", ()))):
        if index in claimed_potions:
            continue
        choices.append(
            {
                "kind": "potion",
                "content_id": str(potion_id),
                "target_id": f"reward:potion:{index}",
                "selection_set_id": f"potion:{index}",
                "selection_set_size": 1,
                "position": index,
            }
        )
    return choices


def _reward_choice_counts(choices: list[dict[str, Any]]) -> dict[str, int]:
    selection_sets = {str(choice.get("selection_set_id", "")) for choice in choices}
    card_sets = {
        str(choice.get("selection_set_id", ""))
        for choice in choices
        if str(choice.get("kind", "")).startswith("card")
    }
    return {
        "total": len(choices),
        "selection_sets": len(selection_sets),
        "cards": sum(1 for choice in choices if str(choice.get("kind")) == "card"),
        "card_groups": sum(1 for item in card_sets if item.startswith("card_group")),
        "fixed_cards": sum(1 for choice in choices if str(choice.get("kind")) == "fixed_card"),
        "card_removals": sum(
            1 for choice in choices if str(choice.get("kind")) == "card_removal"
        ),
        "relics": sum(1 for choice in choices if str(choice.get("kind")) == "relic"),
        "potions": sum(1 for choice in choices if str(choice.get("kind")) == "potion"),
        "gold": sum(1 for choice in choices if str(choice.get("kind")) == "gold"),
    }


def _reward_skipped_counts(reward: Any) -> dict[str, int]:
    return {
        "gold": int(bool(getattr(reward, "gold_skipped", False))),
        "primary_card_group": int(bool(getattr(reward, "card_skipped", False))),
        "card_groups": len(_sequence(getattr(reward, "skipped_card_option_group_indices", ()))),
        "fixed_cards": len(_sequence(getattr(reward, "skipped_card_indices", ()))),
        "relics": int(bool(getattr(reward, "relic_skipped", False)))
        + len(_sequence(getattr(reward, "skipped_relic_ids", ()))),
        "potions": int(bool(getattr(reward, "potion_skipped", False)))
        + len(_sequence(getattr(reward, "skipped_potion_indices", ()))),
    }


def _available_reward_skip_descriptors(state: Any) -> list[dict[str, Any]]:
    reward = getattr(state, "reward", None)
    if reward is None or bool(getattr(reward, "forced", False)):
        return []
    choices = _available_reward_choice_descriptors(state)
    by_set: dict[str, list[dict[str, Any]]] = {}
    for choice in choices:
        by_set.setdefault(str(choice.get("selection_set_id", "")), []).append(choice)
    descriptors: list[dict[str, Any]] = []
    for selection_set_id, set_choices in by_set.items():
        target_id = _skip_target_id_for_selection_set(selection_set_id)
        if target_id is None:
            continue
        descriptors.append(
            {
                "kind": "skip",
                "skip_kind": _skip_kind_for_selection_set(selection_set_id),
                "target_id": target_id,
                "content_id": f"skip:{selection_set_id}",
                "selection_set_id": selection_set_id,
                "selection_set_size": len(set_choices),
                "available_remaining_count": len(choices),
                "sibling_content_ids": [
                    str(choice.get("content_id", ""))
                    for choice in set_choices
                    if choice.get("content_id")
                ],
                "closes_selection_set": True,
            }
        )
    return descriptors


def _reward_skip_descriptor_for_action(state: Any, action: Action) -> dict[str, Any]:
    reward = getattr(state, "reward", None)
    if reward is None or action.target_id is None:
        return {}
    selected = next(
        (
            descriptor
            for descriptor in _available_reward_skip_descriptors(state)
            if str(descriptor.get("target_id", "")) == str(action.target_id)
        ),
        None,
    )
    if selected is None:
        return {}
    return {
        "source": str(getattr(reward, "source", "")),
        "forced": bool(getattr(reward, "forced", False)),
        "kind": "skip",
        "content_id": str(selected.get("content_id", "")),
        "gold": 0,
        "position": None,
        "skips_remaining": False,
        "skips_selection": True,
        "skip_scope": "selection_set",
        "skip_kind": str(selected.get("skip_kind", "")),
        "available_remaining_count": _int(selected.get("available_remaining_count")),
        "selection_set_id": str(selected.get("selection_set_id", "")),
        "selection_set_size": _int(selected.get("selection_set_size")),
        "closes_selection_set": True,
        "sibling_content_ids": list(_sequence(selected.get("sibling_content_ids"))),
    }


def _skip_target_id_for_selection_set(selection_set_id: str) -> str | None:
    if selection_set_id in {"gold", "relic:single", "potion:single"}:
        return {
            "gold": "reward:gold",
            "relic:single": "reward:relic",
            "potion:single": "reward:potion",
        }[selection_set_id]
    if selection_set_id == "card_options":
        return "reward:card_options"
    for prefix, target_prefix in (
        ("relic:", "reward:relic:"),
        ("fixed_card:", "reward:fixed_card:"),
        ("card_group:", "reward:card_group:"),
        ("potion:", "reward:potion:"),
    ):
        if selection_set_id.startswith(prefix):
            suffix = selection_set_id.removeprefix(prefix)
            return f"{target_prefix}{suffix}"
    return None


def _skip_kind_for_selection_set(selection_set_id: str) -> str:
    if selection_set_id == "gold":
        return "gold"
    if selection_set_id == "card_options":
        return "card_options"
    return selection_set_id.split(":", 1)[0]


def _reward_selection_context_for_action(state: Any, action: Action) -> dict[str, Any]:
    if action.target_id is None:
        return {}
    target_id = str(action.target_id)
    choices = _available_reward_choice_descriptors(state)
    selected = next(
        (choice for choice in choices if str(choice.get("target_id", "")) == target_id),
        None,
    )
    if selected is None:
        return {}
    selection_set_id = str(selected.get("selection_set_id", ""))
    set_choices = [
        choice
        for choice in choices
        if str(choice.get("selection_set_id", "")) == selection_set_id
    ]
    return {
        "selection_set_id": selection_set_id,
        "selection_set_size": len(set_choices),
        "available_remaining_count": len(choices),
        "closes_selection_set": bool(
            selected.get("exclusive_within_set", False) or len(set_choices) <= 1
        ),
        "group_index": selected.get("group_index"),
        "card_index": selected.get("card_index"),
        "group_size": _int(selected.get("selection_set_size")),
        "sibling_content_ids": [
            str(choice.get("content_id", ""))
            for choice in set_choices
            if choice.get("content_id") != selected.get("content_id")
        ],
    }


def _action_event_option_descriptor(state: Any, action: Action) -> dict[str, Any]:
    event = getattr(state, "event", None)
    if action.type == ActionType.PROCEED and getattr(state, "phase", None) == RunPhase.EVENT:
        if event is None or getattr(event, "resolved_option_id", None) is not None:
            return {}
        available_options = [
            str(getattr(option, "option_id", ""))
            for option in _sequence(getattr(event, "options", ()))
            if not bool(getattr(option, "disabled", False))
        ]
        skip_metadata = {
            "skip_event": True,
            "available_option_count": len(available_options),
            "available_option_ids": tuple(available_options),
        }
        return {
            "event_id": str(getattr(event, "event_id", "")),
            "page_id": str(getattr(event, "page_id", "")),
            "option_id": "__skip_event__",
            "position": len(available_options),
            "title": "Proceed",
            "description": "Leave the event without choosing a visible option.",
            "disabled": False,
            "skip_action": True,
            "available_option_ids": available_options,
            "metadata": _json_safe(skip_metadata),
            "metadata_keys": sorted(str(key) for key in skip_metadata),
        }
    if action.type != ActionType.CHOOSE_EVENT:
        return {}
    option = _event_option_for_action(state, action)
    if event is None or option is None:
        return {}
    metadata = _mapping(getattr(option, "metadata", {}))
    return {
        "event_id": str(getattr(event, "event_id", "")),
        "page_id": str(getattr(event, "page_id", "")),
        "option_id": str(getattr(option, "option_id", "")),
        "position": _event_option_position_for_action(state, action),
        "title": str(getattr(option, "title", "")),
        "description": str(getattr(option, "description", "")),
        "disabled": bool(getattr(option, "disabled", False)),
        "skip_action": False,
        "metadata": _json_safe(metadata),
        "metadata_keys": sorted(str(key) for key in metadata),
    }


def _action_ancient_option_descriptor(state: Any, action: Action) -> dict[str, Any]:
    if action.type != ActionType.CHOOSE_ANCIENT:
        return {}
    ancient = getattr(state, "ancient", None)
    option = _ancient_option_for_action(state, action)
    if ancient is None or option is None:
        return {}
    metadata = _mapping(getattr(option, "metadata", {}))
    return {
        "ancient_id": str(getattr(ancient, "ancient_id", "")),
        "option_id": str(getattr(option, "option_id", "")),
        "position": _ancient_option_position_for_action(state, action),
        "kind": str(getattr(option, "kind", "")),
        "relic_id": str(getattr(option, "relic_id", "")),
        "gold_delta": _int(getattr(option, "gold_delta", 0)),
        "hp_delta": _int(getattr(option, "hp_delta", 0)),
        "heal_amount": _int(getattr(option, "heal_amount", 0)),
        "max_hp_delta": _int(getattr(option, "max_hp_delta", 0)),
        "potion_slot_delta": _int(getattr(option, "potion_slot_delta", 0)),
        "card_reward_count": _int(getattr(option, "card_reward_count", 0)),
        "random_relic_count": _int(getattr(option, "random_relic_count", 0)),
        "random_potion_count": _int(getattr(option, "random_potion_count", 0)),
        "upgrade_random_count": _int(getattr(option, "upgrade_random_count", 0)),
        "transform_random_count": _int(getattr(option, "transform_random_count", 0)),
        "remove_random_count": _int(getattr(option, "remove_random_count", 0)),
        "metadata": _json_safe(metadata),
        "metadata_keys": sorted(str(key) for key in metadata),
    }


def _shop_item_for_action(state: Any, action: Action) -> Any | None:
    if action.type != ActionType.SHOP_BUY or action.target_id is None:
        return None
    index = _shop_item_index(action.target_id)
    if index is None:
        return None
    shop = getattr(state, "shop", None)
    items = getattr(shop, "items", ()) if shop is not None else ()
    return items[index] if 0 <= index < len(items) else None


def _shop_item_index(target_id: str) -> int | None:
    parts = target_id.split(":")
    if len(parts) < 2 or parts[0] != "shop":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _potion_slot_index(slot_id: str | None) -> int | None:
    if slot_id is None:
        return None
    parts = slot_id.split(":")
    if len(parts) != 2 or parts[0] != "potion":
        return None
    return _parse_int(parts[1])


def _shop_remove_card_instance_id(target_id: str) -> str | None:
    parts = target_id.split(":")
    if len(parts) == 4 and parts[0] == "shop" and parts[2] == "remove":
        return parts[3]
    return None


def _reward_position_for_action(state: Any, action: Action) -> int | None:
    if action.target_id is None:
        return None
    if action.type == ActionType.TAKE_REWARD_GOLD:
        return 0
    target_id = action.target_id
    if target_id in {"reward:relic", "reward:potion"}:
        return 0
    if target_id.startswith("reward:card_group:"):
        parts = target_id.split(":")
        if len(parts) != 4:
            return None
        group_index = _parse_int(parts[2])
        card_index = _parse_int(parts[3])
        if group_index is None or card_index is None:
            return None
        reward = getattr(state, "reward", None)
        groups = getattr(reward, "card_option_groups", ()) if reward is not None else ()
        prior_count = sum(len(group) for group in groups[:group_index])
        return prior_count + card_index
    return _target_index(target_id, 2)


def _event_option_position_for_action(state: Any, action: Action) -> int | None:
    event = getattr(state, "event", None)
    if event is None or action.target_id is None:
        return None
    for index, option in enumerate(getattr(event, "options", ())):
        if getattr(option, "option_id", None) == action.target_id:
            return index
    return None


def _ancient_option_position_for_action(state: Any, action: Action) -> int | None:
    ancient = getattr(state, "ancient", None)
    if ancient is None or action.target_id is None:
        return None
    for index, option in enumerate(getattr(ancient, "options", ())):
        if getattr(option, "option_id", None) == action.target_id:
            return index
    return None


def _reward_card_id_for_action(state: Any, action: Action) -> str | None:
    if action.type != ActionType.TAKE_REWARD_CARD or action.target_id is None:
        return None
    reward = getattr(state, "reward", None)
    if reward is None:
        return None
    target_id = action.target_id
    if target_id.startswith("reward:card:"):
        index = _target_index(target_id, 2)
        options = getattr(reward, "card_options", ())
        return options[index] if index is not None and index < len(options) else None
    if target_id.startswith("reward:fixed_card:"):
        index = _target_index(target_id, 2)
        cards = getattr(reward, "card_ids", ())
        return cards[index] if index is not None and index < len(cards) else None
    if target_id.startswith("reward:card_group:"):
        parts = target_id.split(":")
        if len(parts) != 4:
            return None
        group_index = _parse_int(parts[2])
        card_index = _parse_int(parts[3])
        groups = getattr(reward, "card_option_groups", ())
        if group_index is None or card_index is None or group_index >= len(groups):
            return None
        group = groups[group_index]
        return group[card_index] if card_index < len(group) else None
    if target_id.startswith("reward:remove_card:"):
        card = _reward_optional_remove_card_for_target(state, target_id)
        return str(getattr(card, "card_id", "")) if card is not None else None
    return None


def _reward_relic_id_for_action(state: Any, action: Action) -> str | None:
    if action.type != ActionType.TAKE_REWARD_RELIC or action.target_id is None:
        return None
    reward = getattr(state, "reward", None)
    if reward is None:
        return None
    if action.target_id == "reward:relic":
        relic_id = getattr(reward, "relic_id", None)
        return str(relic_id) if relic_id is not None else None
    if action.target_id.startswith("reward:relic:"):
        index = _target_index(action.target_id, 2)
        relics = getattr(reward, "relic_ids", ())
        return relics[index] if index is not None and index < len(relics) else None
    return None


def _reward_potion_id_for_action(state: Any, action: Action) -> str | None:
    if action.type != ActionType.TAKE_REWARD_POTION or action.target_id is None:
        return None
    reward = getattr(state, "reward", None)
    if reward is None:
        return None
    if action.target_id == "reward:potion":
        potion_id = getattr(reward, "potion_id", None)
        return str(potion_id) if potion_id is not None else None
    if action.target_id.startswith("reward:potion:"):
        index = _target_index(action.target_id, 2)
        potions = getattr(reward, "potion_ids", ())
        return potions[index] if index is not None and index < len(potions) else None
    return None


def _reward_optional_remove_card_for_target(state: Any, target_id: str) -> Any | None:
    index = _target_index(target_id, 2)
    candidates = _reward_optional_remove_cards(state)
    return candidates[index] if index is not None and index < len(candidates) else None


def _reward_optional_remove_cards(state: Any) -> list[Any]:
    reward = getattr(state, "reward", None)
    if reward is None:
        return []
    metadata = _mapping(getattr(reward, "metadata", {}))
    remove_count = _int(metadata.get("optional_remove_card_count"))
    removed_ids = set(_sequence(metadata.get("optional_removed_card_instance_ids")))
    if remove_count <= len(removed_ids):
        return []
    candidate_ids = set(
        str(item)
        for item in _sequence(
            metadata.get("optional_remove_card_instance_ids")
        )
    )
    return [
        card
        for card in getattr(state, "master_deck", ())
        if getattr(card, "instance_id", None) in candidate_ids
        and getattr(card, "instance_id", None) not in removed_ids
    ]


def _potion_id_for_slot(state: Any, slot_id: str | None) -> str | None:
    if slot_id is None:
        return None
    parts = slot_id.split(":")
    if len(parts) != 2 or parts[0] != "potion":
        return None
    index = _parse_int(parts[1])
    potions = getattr(state, "potions", ())
    if index is None or index >= len(potions):
        return None
    return str(potions[index])


def _first_potion_slot_id(state: Any, potion_id: str) -> str | None:
    for index, owned_potion_id in enumerate(getattr(state, "potions", ())):
        if str(owned_potion_id) == potion_id:
            return f"potion:{index}"
    return None


def _event_option_for_action(state: Any, action: Action) -> Any | None:
    event = getattr(state, "event", None)
    if event is None or action.target_id is None:
        return None
    for option in getattr(event, "options", ()):
        if getattr(option, "option_id", None) == action.target_id:
            return option
    return None


def _ancient_option_for_action(state: Any, action: Action) -> Any | None:
    ancient = getattr(state, "ancient", None)
    if ancient is None or action.target_id is None:
        return None
    for option in getattr(ancient, "options", ()):
        if getattr(option, "option_id", None) == action.target_id:
            return option
    return None


def _target_index(target_id: str, part_index: int) -> int | None:
    parts = target_id.split(":")
    if part_index >= len(parts):
        return None
    return _parse_int(parts[part_index])


def _parse_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _action_key(action: Action) -> str:
    return _canonical_json(_action_payload(action))


def _action_payload(action: Action | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(action, Action):
        return dict(action.model_dump(mode="json", exclude_none=True))
    return dict(Action.model_validate(dict(action)).model_dump(mode="json", exclude_none=True))


def _enum_or_str(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _effect_keys(value: object) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_effect_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.extend(_effect_keys(item))
    return tuple(dict.fromkeys(keys))


def _effect_amounts(effects: Mapping[str, Any]) -> dict[str, int]:
    totals = {"damage": 0, "block": 0, "draw": 0, "energy": 0, "heal": 0, "status": 0}

    def walk(value: object, parent_key: str = "") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized = str(key).lower()
                if normalized in totals and isinstance(item, int | float):
                    totals[normalized] += int(item)
                    continue
                elif normalized in {"apply_status", "status", "statuses"}:
                    totals["status"] += 1
                walk(item, normalized)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item, parent_key)
        elif parent_key in totals and isinstance(value, int | float):
            totals[parent_key] += int(value)

    walk(effects)
    return totals


def _hash_presence_features(values: object, *, bucket_count: int) -> list[float]:
    features = [0.0 for _index in range(max(0, bucket_count))]
    for value in _sequence(values):
        if value is None:
            continue
        features[_hash_bucket(str(value), bucket_count)] = 1.0
    return features


def _hash_bucket(value: str, bucket_count: int) -> int:
    if bucket_count <= 1:
        return 0
    total = 0
    for character in value:
        total = ((total * 33) + ord(character)) % bucket_count
    return total


def _scaled(value: int | float, maximum: int | float) -> float:
    return max(0.0, min(1.0, float(value) / max(1.0, float(maximum))))


def _scaled_fraction(value: object) -> float:
    return max(0.0, min(1.0, _float(value)))


def _signed_scaled(value: float, maximum: float) -> float:
    limit = max(1.0, abs(float(maximum)))
    return max(-1.0, min(1.0, float(value) / limit))


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_safe(item) for item in value), key=str)
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or value is None:
        return ()
    if isinstance(value, Mapping):
        return ()
    try:
        return tuple(value)
    except TypeError:
        return ()


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _int(value: Any) -> int:
    return int(_number(value))


def _float(value: Any) -> float:
    return float(_number(value))
