"""Stable single-player chest preparation, explicit opening and quest payouts.

TreasureRoom.EnterInternal prepares the shared relic offer. Opening then grants
normal chest gold immediately, completes Spoils Map and reveals the relic. The
source contracts and hashes are retained in source_treasure_edges.json.
"""
from __future__ import annotations

from typing import Any

from sts2sim.mechanics.relic_bag_runtime import (
    draw_source_relic,
    has_source_relic_bag,
    source_relic_rarity,
)
from sts2sim.mechanics.relic_bags import remove_bag_relics

from .errors import EngineError
from .models import Action, ActionType, EffectEvent, MapNodeState, RewardState, RunPhase, RunState
from .rng import capture_random_state, random_from_state, random_stream

PENDING = "source_pending_treasure"


def is_unopened(state: RunState) -> bool:
    return (state.phase == RunPhase.TREASURE and state.reward is None
            and not state.flags.get("source_treasure_opened"))


def enter(state: RunState, node: MapNodeState) -> tuple[RunState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    flags: dict[str, Any] = dict(state.flags)
    visits = t._flag_int(state, "source_treasure_rooms_entered",
                         t._flag_int(state, "treasure_chests_opened", 0)) + 1
    flags["source_treasure_rooms_entered"] = visits
    flags["source_treasure_opened"] = False
    empty = False
    if t._has_relic(state, "silver_crucible"):
        count = t._flag_int(state, "silver_crucible_treasure_rooms_entered", 0) + 1
        flags["silver_crucible_treasure_rooms_entered"] = count
        empty = count == 1
    rng = random_from_state(state.rng)
    relic_id: str | None = None
    rarity: str | None = None
    if not empty:
        rarity = source_relic_rarity(random_stream(rng, "treasure_room_relics"))
        if flags.get("number_of_runs") == 0 and visits == 1:
            # The rarity draw still happens for the first tutorial chest.
            relic_id = "gorget"
            bag = t._mapping_from(flags.get("shared_relic_bag"))
            flags["shared_relic_bag"] = remove_bag_relics(bag, ("GORGET",))
            rarity = "common"
        elif has_source_relic_bag(state, shared=True):
            original = t._mapping_from(flags.get("shared_relic_bag"))
            relic_id = draw_source_relic(state, flags, rarity, shared=True)
            rarity = next((key for key, values in original.items()
                           if any(str(value).lower() == relic_id for value in values)), "none")
        else:
            raise EngineError("Source treasure preparation requires its shared relic bag")
    flags[PENDING] = {"node_id": node.node_id, "relic_id": relic_id,
                      "relic_rarity": rarity, "empty": empty}
    result = state.model_copy(update={"phase": RunPhase.TREASURE, "combat": None,
        "event": None, "reward": None, "flags": flags, "rng": capture_random_state(rng)})
    return result, (EffectEvent(kind="treasure_ready", target_id=node.node_id,
                               metadata={"opened": False}),)


def legal_open_actions(state: RunState) -> tuple[Action, ...]:
    # NRun.SetCurrentRoom updates ActiveScreenContext, whose treasure listener
    # disables Proceed until _hasChestBeenOpened. Map travel is also disabled
    # on entering the room. Skipping is available only after revealing it.
    return (Action(type=ActionType.OPEN_CHEST),)


def open_chest(state: RunState) -> tuple[RunState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    pending = t._mapping_from(state.flags.get(PENDING))
    node = t._current_map_node(state)
    if not pending or node is None or pending.get("node_id") != node.node_id:
        raise EngineError("Source unopened treasure is missing its prepared room offer")
    flags = dict(state.flags)
    flags.pop(PENDING, None)
    flags["source_treasure_opened"] = True
    flags["treasure_chests_opened"] = t._flag_int(state, "treasure_chests_opened", 0) + 1
    state = state.model_copy(update={"flags": flags})
    events: list[EffectEvent] = [EffectEvent(kind="treasure_chest_opened", target_id=node.node_id)]
    amount = 0
    if not pending.get("empty"):
        rng = random_from_state(state.rng)
        # OneOffSynchronizer: roll first, then truncate Poverty's double product.
        amount = random_stream(rng, "rewards").randrange(42, 53)
        if state.ascension >= 3:
            amount = int(amount * .75)
        player, emitted = t._gain_player_gold(state.player, amount, relics=state.relics,
            source_id=None,
            kind="treasure_gold_gained", target_id=node.node_id,
            metadata={"requested_chest_gold": amount})
        state = state.model_copy(update={"player": player, "rng": capture_random_state(rng)})
        events.extend(emitted)
        state, emitted = _complete_spoils(state, node)
        events.extend(emitted)
    else:
        events.append(EffectEvent(kind="treasure_chest_empty", source_id="silver_crucible",
                                  target_id=node.node_id))
    reward = RewardState(reward_id=f"treasure:{node.node_id}", source="treasure",
        relic_id=pending.get("relic_id"), metadata={
            "empty": bool(pending.get("empty")), "relic_rarity": pending.get("relic_rarity"),
            "chest_opened": True, "chest_gold_already_granted": amount,
        })
    state = state.model_copy(update={"reward": reward})
    events.extend(t._reward_generated_events(reward))
    return state, tuple(events)


def _complete_spoils(
    state: RunState, node: MapNodeState,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    from . import transitions as t

    if (state.flags.get("spoils_map_target_act") != node.act
        or state.flags.get("spoils_map_target_node_id") != node.node_id):
        return state, ()
    cards = tuple(card for card in state.master_deck
                  if t._normalized_id(card.card_id) == "spoils_map")
    flags = dict(state.flags)
    for key in ("spoils_map_target_act", "spoils_map_target_node_id", "spoils_map_reward_gold"):
        flags.pop(key, None)
    flags["spoils_map_completed"] = bool(cards)
    events: list[EffectEvent] = []
    player = state.player
    for card in cards:
        player, emitted = t._gain_player_gold(player, 600, relics=state.relics, source_id=None,
            kind="spoils_map_redeemed", target_id=node.node_id,
            metadata={"removed_card_instance_id": card.instance_id, "act": node.act,
                      "floor": node.floor})
        events.extend(emitted)
        events.append(EffectEvent(kind="event_card_removed", target_id=card.instance_id,
            metadata={"card_id": card.card_id, "quest_card": "spoils_map"}))
    ids = {card.instance_id for card in cards}
    return state.model_copy(update={"player": player, "flags": flags,
        "master_deck": tuple(card for card in state.master_deck if card.instance_id not in ids)}), (
            tuple(events))
