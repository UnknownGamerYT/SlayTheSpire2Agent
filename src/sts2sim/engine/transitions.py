from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from itertools import combinations
from pathlib import Path
from random import Random
from typing import Any, Literal

from sts2sim.content.sources import STS1_COMPAT_SOURCE
from sts2sim.data import load_cached_json
from sts2sim.mechanics.campfire import (
    CampfireAction,
    CampfireChoice,
    CampfireState,
    available_campfire_actions,
    resolve_campfire_action,
    rest_heal_amount,
)
from sts2sim.mechanics.card_effects import normalize_card_spec
from sts2sim.mechanics.combat_rewards import (
    CombatRewardContext,
    build_boss_relic_pool,
    build_combat_card_pool,
    build_combat_potion_pool,
    draw_card_reward_options,
    draw_combat_reward,
    fake_merchant_reward_relic_ids,
)
from sts2sim.mechanics.event_flows import (
    EventFlowMarker,
    EventFlowMarkerKind,
    EventFlowOption,
    EventFlowPage,
    EventFlowState,
    current_event_flow_page,
    event_flow_state,
    resolve_event_flow_option,
)
from sts2sim.mechanics.event_specials import self_help_book_options
from sts2sim.mechanics.monsters import (
    EncounterDefinition,
    MonsterDefinition,
    MonsterMove,
    MonsterPower,
    SpawnedMonster,
    build_encounter_definitions,
    build_monster_definitions,
    choose_encounter,
    monster_move_damage,
    monster_power_amount,
    move_by_id,
    next_monster_move,
    next_move_counts,
    spawn_monsters,
    synthetic_encounter,
)
from sts2sim.mechanics.relic_combat import (
    CombatRelicMarker,
    CombatRelicResolution,
    resolve_damage_dealt_relics,
    resolve_damage_taken_relics,
)
from sts2sim.mechanics.reward_pools import (
    build_character_card_pool,
    build_colorless_card_pool,
)
from sts2sim.mechanics.rewards import (
    CardRarity,
    EncounterType,
    PotionRarity,
    RelicRarity,
    RewardPityState,
    potion_slots_for_ascension,
)
from sts2sim.mechanics.shop_rooms import (
    ShopRoomAction,
    ShopRoomChoice,
    ShopRoomState,
    available_shop_actions,
    build_basic_shop_inventory,
    resolve_shop_action,
)
from sts2sim.mechanics.shops import (
    PricedShopItem,
    ShopInventory,
    ShopInventoryPlan,
    ShopItem,
    ShopItemKind,
)
from sts2sim.mechanics.treasure import (
    TreasureContext,
    TreasureRelic,
    build_treasure_relic_pool,
    draw_treasure_reward,
)
from sts2sim.mechanics.triggers import GameTrigger, TriggerContext, resolve_game_trigger

from .errors import IllegalActionError
from .models import (
    PLAYER_TARGET_ID,
    Action,
    ActionType,
    AncientOptionState,
    AncientState,
    CardEnchantment,
    CardInstance,
    CardType,
    CombatState,
    EffectEvent,
    EventOptionState,
    EventState,
    MapEdgeState,
    MapNodeState,
    MapState,
    MonsterState,
    OrbState,
    PlayerState,
    ReplayEntry,
    RewardState,
    RngState,
    RoomKind,
    RunPhase,
    RunState,
    ShopItemState,
    ShopState,
    TargetType,
)
from .rng import capture_random_state, random_from_seed, random_from_state
from .serialization import SerializedState, load_state, state_digest

ActionInput = Action | Mapping[str, Any]
SourceData = Mapping[str, Any] | None
SPOILS_MAP_CARD_ID = "spoils_map"
SPOILS_MAP_GOLD = 600
GOLDEN_COMPASS_RELIC_ID = "golden_compass"
_DELAYED_EVENT_REWARDS_FLAG = "delayed_event_rewards"
_CAMPFIRE_ACTION_TYPES = frozenset(
    {
        ActionType.REST,
        ActionType.SMITH,
        ActionType.RECALL,
        ActionType.DIG,
        ActionType.LIFT,
        ActionType.TOKE,
    }
)
_FLOW_EVENT_IDS = frozenset(
    {
        "abyssal_baths",
        "colossal_flower",
        "endless_conveyor",
        "slippery_bridge",
        "tablet_of_truth",
        "tinker_time",
        "trial",
        "the_trial",
        "welcome_to_wongos",
    }
)
_ENCHANT_METADATA_KEYS = frozenset(
    {
        "enchant",
        "enchant_keyword",
        "enchant_amount",
        "enchant_count",
        "enchant_card_type",
        "enchant_card_ids",
        "enchant_requires_exhaust",
        "enchant_basic_only",
    }
)


def new_run_state(
    seed: int | str,
    character_id: str,
    ascension: int,
    source_data: SourceData = None,
) -> RunState:
    source = dict(source_data or {})
    rng = random_from_seed(seed)
    deck = _starter_deck(character_id, source)
    player = _initial_player(character_id, ascension, source)
    map_state = _generate_act_map(act=_source_int(source, "act", 1), rng=rng, source=source)
    player = _apply_ancient_heal(player, ascension)
    ancient_state = _generate_ancient_state(act=map_state.act, rng=rng)
    rng_state = capture_random_state(rng)

    return RunState(
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        rng=rng_state,
        phase=RunPhase.ANCIENT,
        act=map_state.act,
        floor=_source_int(source, "floor", 0),
        player=player,
        master_deck=deck,
        ancient=ancient_state,
        map=map_state,
        combat=None,
        flags=_initial_flags(source),
    )


def legal_actions(state: RunState) -> tuple[Action, ...]:
    if state.phase == RunPhase.ANCIENT:
        if state.ancient is None:
            return _with_potion_discard_actions(state, ())
        return _with_potion_discard_actions(
            state,
            tuple(
                Action(type=ActionType.CHOOSE_ANCIENT, target_id=option.option_id)
                for option in state.ancient.options
                if option.option_id not in state.ancient.chosen_option_ids
            ),
        )

    if state.phase == RunPhase.MAP:
        return _with_potion_discard_actions(
            state,
            tuple(
                Action(type=ActionType.CHOOSE_NODE, target_id=node_id)
                for node_id in _reachable_node_ids(state)
            ),
        )

    if state.phase == RunPhase.REST:
        return _with_potion_discard_actions(state, _legal_campfire_actions(state))

    if state.phase == RunPhase.SHOP:
        return _with_potion_discard_actions(state, _legal_shop_actions(state))

    if state.phase in {
        RunPhase.EVENT,
        RunPhase.TREASURE,
        RunPhase.REWARD,
    }:
        if (
            state.phase == RunPhase.EVENT
            and state.event is not None
            and state.event.resolved_option_id is None
            and state.reward is None
        ):
            event_actions = list(_legal_event_actions(state))
            event_actions.append(Action(type=ActionType.PROCEED))
            return _with_potion_discard_actions(state, tuple(event_actions))
        reward_actions = list(_legal_reward_actions(state))
        if not _reward_has_forced_pending_items(state):
            reward_actions.append(Action(type=ActionType.PROCEED))
        return _with_potion_discard_actions(state, tuple(reward_actions))

    if state.phase != RunPhase.COMBAT or state.combat is None:
        return _with_potion_discard_actions(state, ())

    combat = state.combat
    if combat.player.hp <= 0 or not _alive_monsters(combat):
        return _with_potion_discard_actions(state, ())

    discard_choice = _pending_discard_choice(combat)
    if discard_choice is not None:
        return tuple(
            Action(
                type=ActionType.DISCARD_CARD,
                card_instance_id=card.instance_id,
                payload={
                    "source_card_instance_id": discard_choice.get("source_card_instance_id"),
                    "remaining": discard_choice["remaining"],
                },
            )
            for card in combat.hand
        )

    actions: list[Action] = []
    if not _turn_card_play_limit_reached(combat):
        for card in combat.hand:
            cost = _energy_cost(card, combat.player.energy)
            if cost > combat.player.energy:
                continue
            if not _can_pay_resource_costs(combat.player, card):
                continue
            for target_id in _legal_target_ids(combat, card):
                actions.append(
                    Action(
                        type=ActionType.PLAY_CARD,
                        card_instance_id=card.instance_id,
                        target_id=target_id,
                    )
                )

    actions.extend(_legal_potion_use_actions(state, combat))
    actions.append(Action(type=ActionType.END_TURN))
    return _with_potion_discard_actions(state, tuple(actions))


def step_state(state: RunState | SerializedState, action: ActionInput) -> RunState:
    state = load_state(state)
    action = _coerce_action(action)
    legal = legal_actions(state)
    action = _normalize_action(state, action)
    if not _action_is_legal(action, legal):
        raise IllegalActionError(action, legal)

    before_hash = state_digest(state)
    if action.type == ActionType.CHOOSE_ANCIENT:
        next_state, events = _choose_ancient(state, action)
    elif action.type == ActionType.CHOOSE_NODE:
        next_state, events = _choose_node(state, action)
    elif action.type == ActionType.CHOOSE_EVENT:
        next_state, events = _choose_event(state, action)
    elif action.type == ActionType.TAKE_REWARD_GOLD:
        next_state, events = _take_reward_gold(state, action)
    elif action.type == ActionType.TAKE_REWARD_RELIC:
        next_state, events = _take_reward_relic(state, action)
    elif action.type == ActionType.TAKE_REWARD_CARD:
        next_state, events = _take_reward_card(state, action)
    elif action.type == ActionType.TAKE_REWARD_POTION:
        next_state, events = _take_reward_potion(state, action)
    elif action.type in _CAMPFIRE_ACTION_TYPES:
        next_state, events = _resolve_campfire_action(state, action)
    elif action.type in {
        ActionType.SHOP_BUY,
        ActionType.SHOP_LEAVE,
        ActionType.THROW_POTION_AT_MERCHANT,
    }:
        next_state, events = _resolve_shop_action(state, action)
    elif action.type == ActionType.USE_POTION:
        next_state, events = _use_potion(state, action)
    elif action.type == ActionType.DISCARD_POTION:
        next_state, events = _discard_potion(state, action)
    elif action.type == ActionType.DISCARD_CARD:
        next_state, events = _discard_card_from_hand(state, action)
    elif action.type == ActionType.PROCEED:
        next_state, events = _proceed(state)
    elif action.type == ActionType.END_TURN:
        next_state, events = _end_turn(state)
    else:
        next_state, events = _play_card(state, action)

    after_hash = state_digest(next_state)
    entry = ReplayEntry(
        step_index=len(state.replay_log),
        action=action,
        state_hash_before=before_hash,
        state_hash_after=after_hash,
        events=events,
    )
    return next_state.model_copy(update={"replay_log": state.replay_log + (entry,)})


def replay_actions(
    initial_state: RunState | SerializedState,
    actions: Iterable[ActionInput],
) -> RunState:
    state = load_state(initial_state)
    for action in actions:
        state = step_state(state, action)
    return state


def _with_potion_discard_actions(
    state: RunState,
    actions: Sequence[Action],
) -> tuple[Action, ...]:
    return tuple(actions) + tuple(
        Action(type=ActionType.DISCARD_POTION, target_id=_potion_slot_id(index))
        for index, _potion_id in enumerate(state.potions)
    )


def _potion_slot_id(index: int) -> str:
    return f"potion:{index}"


def _parse_potion_slot_id(target_id: str) -> int:
    parts = target_id.split(":")
    if len(parts) != 2 or parts[0] != "potion":
        raise ValueError(f"Invalid potion slot id: {target_id}")
    return int(parts[1])


def _legal_potion_use_actions(
    state: RunState,
    combat: CombatState,
) -> tuple[Action, ...]:
    actions: list[Action] = []
    for index, potion_id in enumerate(state.potions):
        if not _potion_is_manually_usable(potion_id):
            continue
        slot_id = _potion_slot_id(index)
        for target_id in _legal_potion_target_ids(combat, potion_id):
            actions.append(
                Action(
                    type=ActionType.USE_POTION,
                    target_id=target_id,
                    payload={"potion_slot": slot_id, "potion_id": potion_id},
                )
            )
    return tuple(actions)


def _legal_potion_target_ids(
    combat: CombatState,
    potion_id: str,
) -> tuple[str | None, ...]:
    target = _potion_target_type(potion_id)
    if target == TargetType.SELF:
        return (PLAYER_TARGET_ID,)
    if target == TargetType.ENEMY:
        return tuple(monster.monster_id for monster in _alive_monsters(combat))
    if target == TargetType.ALL_ENEMIES:
        return (None,)
    return (PLAYER_TARGET_ID,)


def _potion_is_manually_usable(potion_id: str) -> bool:
    return _normalized_id(potion_id) not in {"fairy_in_a_bottle"}


def _choose_ancient(
    state: RunState, action: Action
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if state.ancient is None or action.target_id is None:
        return state, ()

    option = next(
        (
            candidate
            for candidate in state.ancient.options
            if candidate.option_id == action.target_id
        ),
        None,
    )
    if option is None:
        return state, ()

    relics = state.relics + (option.relic_id,)
    curses = state.curses
    if option.kind == "curse_relic":
        curses = curses + (option.relic_id,)

    ancient = state.ancient.model_copy(
        update={
            "chosen_option_ids": state.ancient.chosen_option_ids
            + (option.option_id,),
        }
    )
    next_state = state.model_copy(
        update={
            "phase": RunPhase.MAP,
            "ancient": ancient,
            "relics": relics,
            "curses": curses,
        }
    )
    next_state, pickup_events = _apply_relic_pickup_effects(next_state, option.relic_id)
    return next_state, (
        (
            EffectEvent(
                kind="ancient_option_chosen",
                source_id=state.ancient.ancient_id,
                target_id=option.option_id,
                metadata={
                    "act": state.ancient.act,
                    "option_kind": option.kind,
                    "relic_id": option.relic_id,
                },
            ),
        )
        + pickup_events
    )


def _choose_node(state: RunState, action: Action) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if state.map is None or action.target_id is None:
        return state, ()
    node = state.map.node_by_id[action.target_id]
    map_state = state.map.model_copy(update={"current_node_id": node.node_id})
    state = state.model_copy(update={"map": map_state, "floor": node.floor})
    entered = EffectEvent(
        kind="room_entered",
        target_id=node.node_id,
        metadata={"act": node.act, "floor": node.floor, "room_kind": node.kind.value},
    )
    state, quest_events = _apply_spoils_map_room_entry(state, node)
    next_state, events = _enter_room(state, node)
    return next_state, (entered,) + quest_events + events


def _enter_room(
    state: RunState, node: MapNodeState
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if node.kind in {RoomKind.MONSTER, RoomKind.ELITE, RoomKind.BOSS}:
        combat, rng_state, events = _start_combat_for_node(state, node)
        return (
            state.model_copy(
                update={
                    "phase": RunPhase.COMBAT,
                    "combat": combat,
                    "event": None,
                    "rng": rng_state,
                    "player": combat.player,
                }
            ),
            events,
    )
    if node.kind == RoomKind.EVENT:
        return _enter_event_room(state, node)
    if node.kind == RoomKind.SHOP:
        return _enter_shop_room(state.model_copy(update={"event": None}), node)
    if node.kind == RoomKind.REST:
        return state.model_copy(update={"phase": RunPhase.REST, "combat": None, "event": None}), (
            EffectEvent(kind="rest_ready", target_id=node.node_id),
        )
    if node.kind == RoomKind.TREASURE:
        rng = random_from_state(state.rng)
        reward, reward_events = _treasure_reward_state(state, node, rng)
        flags = dict(state.flags)
        flags["treasure_chests_opened"] = _flag_int(state, "treasure_chests_opened", 0) + 1
        return state.model_copy(
            update={
                "phase": RunPhase.TREASURE,
                "combat": None,
                "event": None,
                "reward": reward,
                "flags": flags,
                "rng": capture_random_state(rng),
            }
        ), (EffectEvent(kind="treasure_ready", target_id=node.node_id),) + reward_events
    return _complete_current_room(state, (EffectEvent(kind="start_room_skipped"),))


def _apply_spoils_map_room_entry(
    state: RunState,
    node: MapNodeState,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    target_act = _optional_int(state.flags.get("spoils_map_target_act"))
    target_node_id = state.flags.get("spoils_map_target_node_id")
    if target_act != node.act or target_node_id != node.node_id:
        return state, ()

    flags = dict(state.flags)
    for key in (
        "spoils_map_target_act",
        "spoils_map_target_node_id",
        "spoils_map_reward_gold",
    ):
        flags.pop(key, None)

    deck = list(state.master_deck)
    deck, removed = _remove_first_card_by_id(deck, SPOILS_MAP_CARD_ID)
    if removed is None:
        return state.model_copy(update={"flags": flags}), (
            EffectEvent(
                kind="spoils_map_missing",
                target_id=node.node_id,
                metadata={"act": node.act, "floor": node.floor},
            ),
        )

    gold = _flag_int(state, "spoils_map_reward_gold", SPOILS_MAP_GOLD)
    flags["spoils_map_completed"] = True
    player = state.player.model_copy(update={"gold": state.player.gold + gold})
    return state.model_copy(
        update={"player": player, "master_deck": tuple(deck), "flags": flags}
    ), (
        EffectEvent(
            kind="spoils_map_redeemed",
            target_id=node.node_id,
            amount=gold,
            metadata={
                "act": node.act,
                "floor": node.floor,
                "removed_card_instance_id": removed.instance_id,
            },
        ),
        EffectEvent(
            kind="event_card_removed",
            target_id=removed.instance_id,
            metadata={"card_id": removed.card_id, "quest_card": SPOILS_MAP_CARD_ID},
        ),
    )


def _proceed(state: RunState) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if state.phase == RunPhase.REST:
        return _rest_at_campfire(state)
    if state.phase == RunPhase.EVENT:
        return _complete_current_room(state, (EffectEvent(kind="event_resolved"),))
    if state.phase == RunPhase.SHOP:
        return _leave_shop(state)
    if state.phase == RunPhase.TREASURE:
        return _complete_current_room(state, (EffectEvent(kind="treasure_opened"),))
    if state.phase == RunPhase.REWARD:
        return _complete_current_room(state, (EffectEvent(kind="reward_skipped"),))
    return state, ()


def _legal_campfire_actions(state: RunState) -> tuple[Action, ...]:
    campfire = _campfire_state(state)
    available = available_campfire_actions(campfire, ascension_level=state.ascension)
    actions: list[Action] = []

    if CampfireAction.REST in available:
        actions.append(Action(type=ActionType.REST))
    if CampfireAction.SMITH in available:
        actions.extend(
            Action(type=ActionType.SMITH, target_id=card.instance_id)
            for card in state.master_deck
            if card.instance_id in campfire.upgradeable_card_ids
        )
    if CampfireAction.RECALL in available:
        actions.append(Action(type=ActionType.RECALL))
    if CampfireAction.DIG in available:
        actions.append(Action(type=ActionType.DIG))
    if CampfireAction.LIFT in available:
        actions.append(Action(type=ActionType.LIFT))
    if CampfireAction.TOKE in available:
        actions.extend(
            Action(type=ActionType.TOKE, target_id=card.instance_id)
            for card in state.master_deck
            if card.instance_id in campfire.removable_card_ids
        )

    # Legacy policy runners can still select proceed at a rest site; it resolves as Rest.
    if CampfireAction.REST in available:
        actions.append(Action(type=ActionType.PROCEED))

    return tuple(actions)


def _resolve_campfire_action(
    state: RunState, action: Action
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if action.type == ActionType.REST:
        return _rest_at_campfire(state)
    if action.type == ActionType.SMITH:
        return _smith_at_campfire(state, action)
    if action.type == ActionType.RECALL:
        return _recall_at_campfire(state)
    if action.type == ActionType.DIG:
        return _dig_at_campfire(state)
    if action.type == ActionType.LIFT:
        return _lift_at_campfire(state)
    if action.type == ActionType.TOKE:
        return _toke_at_campfire(state, action)
    return state, ()


def _rest_at_campfire(state: RunState) -> tuple[RunState, tuple[EffectEvent, ...]]:
    heal_amount = rest_heal_amount(state.player.max_hp, ascension_level=state.ascension)
    healed = min(max(0, state.player.max_hp - state.player.hp), heal_amount)
    player = state.player.model_copy(update={"hp": state.player.hp + healed})
    return _complete_current_room(
        state.model_copy(update={"player": player}),
        (
            EffectEvent(
                kind="rest_healed",
                amount=healed,
                metadata={"heal_amount": heal_amount, "ascension": state.ascension},
            ),
        ),
    )


def _smith_at_campfire(
    state: RunState, action: Action
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    campfire = _campfire_state(state)
    result = resolve_campfire_action(
        CampfireChoice(CampfireAction.SMITH, target_id=action.target_id),
        campfire,
        ascension_level=state.ascension,
    )
    upgraded_card_id = result.upgraded_card_id
    card = _find_card(state.master_deck, upgraded_card_id)
    if upgraded_card_id is None or card is None:
        return state, ()

    master_deck = tuple(
        _upgrade_card_instance(deck_card)
        if deck_card.instance_id == upgraded_card_id
        else deck_card
        for deck_card in state.master_deck
    )
    return _complete_current_room(
        state.model_copy(update={"master_deck": master_deck}),
        (
            EffectEvent(
                kind="card_upgraded",
                target_id=upgraded_card_id,
                metadata={"card_id": card.card_id, "campfire_action": CampfireAction.SMITH.value},
            ),
        ),
    )


def _recall_at_campfire(state: RunState) -> tuple[RunState, tuple[EffectEvent, ...]]:
    campfire = _campfire_state(state)
    resolve_campfire_action(
        CampfireChoice(CampfireAction.RECALL),
        campfire,
        ascension_level=state.ascension,
    )
    flags = dict(state.flags)
    flags["has_ruby_key"] = True
    return _complete_current_room(
        state.model_copy(update={"flags": flags}),
        (
            EffectEvent(
                kind="ruby_key_recalled",
                metadata={"campfire_action": CampfireAction.RECALL.value},
            ),
        ),
    )


def _dig_at_campfire(state: RunState) -> tuple[RunState, tuple[EffectEvent, ...]]:
    campfire = _campfire_state(state)
    result = resolve_campfire_action(
        CampfireChoice(CampfireAction.DIG),
        campfire,
        ascension_level=state.ascension,
    )
    if not result.grants_relic:
        return state, ()

    rng = random_from_state(state.rng)
    relic_id = _dig_relic_id(state, rng)
    flags = dict(state.flags)
    dig_count = _flag_int(state, "campfire_dig_count", 0) + 1
    flags["campfire_dig_count"] = dig_count
    return _complete_current_room(
        state.model_copy(
            update={
                "flags": flags,
                "relics": state.relics + (relic_id,),
                "rng": capture_random_state(rng),
            }
        ),
        (
            EffectEvent(
                kind="campfire_dug_relic",
                target_id=relic_id,
                metadata={
                    "campfire_action": CampfireAction.DIG.value,
                    "dig_count": dig_count,
                },
            ),
        ),
    )


def _lift_at_campfire(state: RunState) -> tuple[RunState, tuple[EffectEvent, ...]]:
    campfire = _campfire_state(state)
    resolve_campfire_action(
        CampfireChoice(CampfireAction.LIFT),
        campfire,
        ascension_level=state.ascension,
    )
    lift_count = campfire.lift_count + 1
    flags = dict(state.flags)
    flags["girya_lift_count"] = lift_count
    flags["girya_strength_bonus"] = lift_count
    return _complete_current_room(
        state.model_copy(update={"flags": flags}),
        (
            EffectEvent(
                kind="girya_lifted",
                amount=1,
                metadata={
                    "campfire_action": CampfireAction.LIFT.value,
                    "lift_count": lift_count,
                    "strength_bonus": lift_count,
                },
            ),
        ),
    )


def _toke_at_campfire(
    state: RunState, action: Action
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    campfire = _campfire_state(state)
    result = resolve_campfire_action(
        CampfireChoice(CampfireAction.TOKE, target_id=action.target_id),
        campfire,
        ascension_level=state.ascension,
    )
    removed_card_id = result.removed_card_id
    card = _find_card(state.master_deck, removed_card_id)
    if removed_card_id is None or card is None:
        return state, ()

    removed_cards = list(_flag_str_sequence(state, "peace_pipe_removed_card_ids"))
    removed_cards.append(removed_card_id)
    flags = dict(state.flags)
    flags["peace_pipe_removed_card_ids"] = removed_cards
    master_deck = tuple(
        deck_card for deck_card in state.master_deck if deck_card.instance_id != removed_card_id
    )
    return _complete_current_room(
        state.model_copy(update={"flags": flags, "master_deck": master_deck}),
        (
            EffectEvent(
                kind="card_removed",
                target_id=removed_card_id,
                metadata={
                    "card_id": card.card_id,
                    "campfire_action": CampfireAction.TOKE.value,
                },
            ),
        ),
    )


def _campfire_state(state: RunState) -> CampfireState:
    return CampfireState(
        current_hp=state.player.hp,
        max_hp=state.player.max_hp,
        upgradeable_card_ids=frozenset(
            card.instance_id for card in state.master_deck if not card.upgraded
        ),
        removable_card_ids=frozenset(card.instance_id for card in state.master_deck),
        has_ruby_key=bool(state.flags.get("has_ruby_key", state.flags.get("ruby_key", False))),
        can_recall=bool(state.flags.get("can_recall", True)),
        has_shovel="shovel" in state.relics,
        has_girya="girya" in state.relics,
        lift_count=_flag_int(state, "girya_lift_count", 0),
        has_peace_pipe="peace_pipe" in state.relics,
    )


def _enter_event_room(
    state: RunState,
    node: MapNodeState,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    rng = random_from_state(state.rng)
    if _has_immediate_event_reward_flags(state):
        state, cost_events = _apply_event_reward_card_costs(state)
        reward, events = _event_reward_state(state, node, rng)
        return (
            state.model_copy(
                update={
                    "phase": RunPhase.EVENT,
                    "combat": None,
                    "event": None,
                    "reward": reward,
                    "rng": capture_random_state(rng),
                }
            ),
            (EffectEvent(kind="event_ready", target_id=node.node_id),)
            + cost_events
            + events,
        )

    event = _event_state_for_node(state, node, rng)
    return (
        state.model_copy(
            update={
                "phase": RunPhase.EVENT,
                "combat": None,
                "event": event,
                "reward": None,
                "rng": capture_random_state(rng),
            }
        ),
        (
            EffectEvent(
                kind="event_ready",
                target_id=node.node_id,
                source_id=event.event_id,
                metadata={"event_name": event.name},
            ),
        ),
    )


def _has_immediate_event_reward_flags(state: RunState) -> bool:
    if state.flags.get("event_id") is not None or state.flags.get("event_room_id") is not None:
        return False
    return any(
        key in state.flags
        for key in (
            "event_reward_potion_id",
            "event_reward_potion_ids",
            "event_reward_potion_count",
            "event_reward_potion_chance_percent",
            "event_reward_gold",
            "event_reward_relic_id",
            "event_reward_relic_ids",
            "event_reward_relic_count",
            "event_reward_card_ids",
            "event_reward_card_options",
        )
    )


def _event_state_for_node(
    state: RunState,
    node: MapNodeState,
    rng: Random,
) -> EventState:
    event = _event_source_for_room(state, rng)
    event_id = str(event.get("id", f"EVENT_{node.act}_{node.floor}_{node.lane}"))
    if _event_uses_flow(event_id):
        flow_state = event_flow_state(
            event_id,
            hp=state.player.hp,
            max_hp=state.player.max_hp,
            gold=state.player.gold,
            page_id=str(state.flags.get("event_flow_page_id", "INITIAL")),
            counters=_mapping_from(state.flags.get("event_flow_counters", {})),
            data=_mapping_from(state.flags.get("event_flow_data", {})),
        )
        return _event_state_from_flow(
            state,
            event_id=event_id,
            name=str(event.get("name", event_id)),
            node=node,
            flow_state=flow_state,
        )
    if _is_self_help_book_event(event_id):
        options = _self_help_book_event_option_states(state)
        return EventState(
            event_id=event_id,
            name=str(event.get("name", event_id)),
            page_id="INITIAL",
            options=options,
            metadata={"node_id": node.node_id, "act": node.act, "floor": node.floor},
        )
    options = _event_option_states(event)
    if not options:
        options = (
            EventOptionState(
                option_id="leave",
                title="Leave",
                description="Leave.",
            ),
        )
    return EventState(
        event_id=event_id,
        name=str(event.get("name", event_id)),
        page_id="INITIAL",
        options=options,
        metadata={"node_id": node.node_id, "act": node.act, "floor": node.floor},
    )


def _event_source_for_room(state: RunState, rng: Random) -> Mapping[str, Any]:
    event_id = state.flags.get("event_id", state.flags.get("event_room_id"))
    events = _event_source_rows(state)
    if event_id is not None:
        normalized = _normalized_id(str(event_id))
        for event in events:
            if _normalized_id(str(event.get("id", ""))) == normalized:
                return event

    configured_pool = _flag_str_sequence(state, "event_pool")
    candidates = [
        event
        for event in events
        if (not configured_pool or _normalized_id(str(event.get("id", ""))) in configured_pool)
        and _event_is_available_in_act(event, state.act)
    ]
    if not candidates:
        candidates = list(events)
    if candidates:
        return rng.choice(candidates)
    return {"id": "UNKNOWN_EVENT", "name": "Unknown Event", "options": []}


def _event_source_rows(state: RunState) -> tuple[Mapping[str, Any], ...]:
    raw_events = _source_items(state.flags.get("events")) or _cached_source_rows(state, "events")
    return tuple(event for event in (_mapping_from(item) for item in raw_events) if event)


def _event_is_available_in_act(event: Mapping[str, Any], act: int) -> bool:
    raw_act = str(event.get("act") or "").lower()
    return not raw_act or f"act {act}" in raw_act


def _event_uses_flow(event_id: str) -> bool:
    return _normalized_id(event_id) in _FLOW_EVENT_IDS


def _is_self_help_book_event(event_id: str) -> bool:
    return _normalized_id(event_id) == "self_help_book"


def _event_option_states(event: Mapping[str, Any]) -> tuple[EventOptionState, ...]:
    raw_options = list(event.get("options") or [])
    if not raw_options:
        for page in event.get("pages") or []:
            page_map = _mapping_from(page)
            if str(page_map.get("id", "")).upper() == "INITIAL":
                raw_options.extend(page_map.get("options") or [])
                break
    options: list[EventOptionState] = []
    seen: set[str] = set()
    for raw_option in raw_options:
        option = _mapping_from(raw_option)
        if not option:
            continue
        option_id = _normalized_id(str(option.get("id", option.get("title", "option"))))
        if option_id in seen:
            continue
        seen.add(option_id)
        description = _clean_event_text(str(option.get("description", "")))
        metadata = _event_option_metadata(description)
        options.append(
            EventOptionState(
                option_id=option_id,
                title=str(option.get("title", option_id)),
                description=description,
                disabled=_event_option_is_disabled(option),
                metadata=metadata,
            )
        )
    return tuple(options)


def _event_option_is_disabled(option: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(option.get(key, "")) for key in ("id", "title", "description")
    )
    return bool(
        re.search(r"\blocked\b", text, re.IGNORECASE)
        or re.search(
            r"\b(?:no|none\s+of\s+your|don't\s+have\s+any)\b.*\bEnchanted\b",
            text,
            re.IGNORECASE,
        )
    )


def _event_option_metadata(description: str) -> dict[str, Any]:
    return _event_option_enchant_metadata(description)


def _event_option_enchant_metadata(description: str) -> dict[str, Any]:
    if not re.search(r"\bEnchant(?:ed)?\b", description, re.IGNORECASE):
        return {}
    if re.search(
        r"\b(?:no|none\s+of\s+your|don't\s+have\s+any)\b.*\bEnchanted\b",
        description,
        re.IGNORECASE,
    ):
        return {"enchant": True, "locked": True}

    keyword_match = re.search(
        r"\bwith\s+([A-Z][A-Za-z' -]*?)(?:\s+(\d+))?(?:\.|$)",
        description,
    )
    metadata: dict[str, Any] = {"enchant": True, "enchant_count": 1}
    if keyword_match is not None:
        metadata["enchant_keyword"] = keyword_match.group(1).strip()
        if keyword_match.group(2):
            metadata["enchant_amount"] = int(keyword_match.group(2))

    count_match = re.search(
        r"\bEnchant\s+(?:(\d+)\s+)?(?:a|an)?\s*cards?\b",
        description,
        re.IGNORECASE,
    )
    if count_match is not None and count_match.group(1):
        metadata["enchant_count"] = int(count_match.group(1))

    type_match = re.search(r"\b(?:an?|Choose\s+an?)\s+(Attack|Skill|Power)\b", description)
    if type_match is not None:
        metadata["enchant_card_type"] = _normalized_id(type_match.group(1))
    if re.search(r"\bBasic\s+Strike\s+or\s+Defend\b", description, re.IGNORECASE):
        metadata["enchant_basic_only"] = True
        metadata["enchant_card_ids"] = ("strike", "defend")
    if re.search(r"\bcard\s+that\s+Exhausts\b", description, re.IGNORECASE):
        metadata["enchant_requires_exhaust"] = True
    return metadata


def _self_help_book_event_option_states(state: RunState) -> tuple[EventOptionState, ...]:
    deck_types = {
        card.type.value
        for card in state.master_deck
        if card.type in {CardType.ATTACK, CardType.SKILL, CardType.POWER}
    }
    any_enchantable = bool(deck_types)
    options: list[EventOptionState] = []
    for option in self_help_book_options(None):
        metadata = dict(option.metadata)
        if "enchant_keyword" in metadata:
            metadata["enchant"] = True
            metadata.setdefault("enchant_count", 1)
        card_type = str(metadata.get("enchant_card_type", metadata.get("card_type", "")))
        locked_variant = bool(metadata.get("locked"))
        if card_type:
            if card_type in deck_types and locked_variant:
                continue
            if card_type not in deck_types and not locked_variant:
                continue
        if option.option_id == "NO_OPTIONS":
            disabled = any_enchantable
            metadata = {
                **metadata,
                **(
                    {"locked": True, "disabled_reason": "requires_no_enchantable_cards"}
                    if any_enchantable
                    else {}
                ),
            }
        else:
            disabled = bool(metadata.get("locked"))
        options.append(
            EventOptionState(
                option_id=option.option_id,
                title=option.label,
                description=option.description,
                disabled=disabled,
                metadata=metadata,
            )
        )
    return tuple(options)


def _event_state_from_flow(
    state: RunState,
    *,
    event_id: str,
    name: str,
    node: MapNodeState | None,
    flow_state: EventFlowState,
    resolved_option_id: str | None = None,
) -> EventState:
    page = current_event_flow_page(flow_state)
    metadata: dict[str, Any] = {
        "flow": True,
        "flow_page_id": flow_state.page_id,
        "flow_counters": dict(flow_state.counters),
        "flow_data": dict(flow_state.data),
        "flow_selected_option_ids": flow_state.selected_option_ids,
        "flow_terminal": flow_state.terminal,
    }
    if node is not None:
        metadata.update({"node_id": node.node_id, "act": node.act, "floor": node.floor})
    return EventState(
        event_id=event_id,
        name=name,
        page_id=page.page_id,
        options=_event_flow_option_states(page, state.player.gold),
        resolved_option_id=resolved_option_id,
        metadata=metadata,
    )


def _event_flow_option_states(
    page: EventFlowPage,
    gold: int,
) -> tuple[EventOptionState, ...]:
    return tuple(_event_flow_option_state(option, gold) for option in page.options)


def _event_flow_option_state(option: EventFlowOption, gold: int) -> EventOptionState:
    metadata = dict(option.metadata)
    if option.required_gold:
        metadata["required_gold"] = option.required_gold
    if option.markers:
        metadata["markers"] = tuple(marker.kind.value for marker in option.markers)
        transform_markers = tuple(
            marker
            for marker in option.markers
            if marker.kind is EventFlowMarkerKind.CARD_TRANSFORM
        )
        transform_count = sum(marker.count for marker in transform_markers)
        if transform_count:
            metadata["transform_card_count"] = transform_count
            transform_random_count = sum(
                marker.count
                for marker in transform_markers
                if _event_transform_marker_is_random(marker)
            )
            if transform_random_count:
                metadata["transform_random_card_count"] = transform_random_count
    return EventOptionState(
        option_id=option.option_id,
        title=option.label or option.option_id,
        description=option.description,
        disabled=option.locked or gold < option.required_gold,
        metadata=metadata,
    )


def _legal_event_actions(state: RunState) -> tuple[Action, ...]:
    if state.event is None:
        return ()
    actions: list[Action] = []
    for option in state.event.options:
        if option.disabled:
            continue
        if _event_option_is_enchant_choice(option):
            actions.extend(_legal_event_enchant_actions(state, option))
            continue
        if _event_option_is_transform_choice(option):
            actions.extend(_legal_event_transform_actions(state, option))
            continue
        actions.append(Action(type=ActionType.CHOOSE_EVENT, target_id=option.option_id))
    return tuple(actions)


def _event_option_is_enchant_choice(option: EventOptionState) -> bool:
    return bool(option.metadata.get("enchant")) and not bool(option.metadata.get("locked"))


def _event_option_is_transform_choice(option: EventOptionState) -> bool:
    return _event_transform_count(option) > 0 and not bool(option.metadata.get("locked"))


def _event_transform_count(option: EventOptionState) -> int:
    return _nonnegative_int(option.metadata.get("transform_card_count"), 0)


def _event_transform_random_count(option: EventOptionState) -> int:
    return _nonnegative_int(option.metadata.get("transform_random_card_count"), 0)


def _event_transform_choice_count(option: EventOptionState) -> int:
    return max(0, _event_transform_count(option) - _event_transform_random_count(option))


def _legal_event_transform_actions(
    state: RunState,
    option: EventOptionState,
) -> tuple[Action, ...]:
    total_count = _event_transform_count(option)
    choice_count = _event_transform_choice_count(option)
    if total_count <= 0:
        return ()
    if choice_count <= 0:
        return (Action(type=ActionType.CHOOSE_EVENT, target_id=option.option_id),)
    eligible_cards = tuple(state.master_deck)
    if choice_count == 1:
        return tuple(
            Action(
                type=ActionType.CHOOSE_EVENT,
                target_id=option.option_id,
                card_instance_id=card.instance_id,
            )
            for card in eligible_cards
        )
    return tuple(
        Action(
            type=ActionType.CHOOSE_EVENT,
            target_id=option.option_id,
            payload={"card_instance_ids": tuple(card.instance_id for card in card_group)},
        )
        for card_group in combinations(eligible_cards, choice_count)
    )


def _legal_event_enchant_actions(
    state: RunState,
    option: EventOptionState,
) -> tuple[Action, ...]:
    eligible_cards = _eligible_enchant_cards(state, option)
    count = _event_enchant_count(option)
    if count <= 0 or len(eligible_cards) < count:
        return ()
    if count == 1:
        return tuple(
            Action(
                type=ActionType.CHOOSE_EVENT,
                target_id=option.option_id,
                card_instance_id=card.instance_id,
            )
            for card in eligible_cards
        )
    return tuple(
        Action(
            type=ActionType.CHOOSE_EVENT,
            target_id=option.option_id,
            payload={"card_instance_ids": tuple(card.instance_id for card in card_group)},
        )
        for card_group in combinations(eligible_cards, count)
    )


def _choose_event(
    state: RunState,
    action: Action,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if state.phase != RunPhase.EVENT or state.event is None or action.target_id is None:
        return state, ()
    if state.event.resolved_option_id is not None:
        return state, ()

    option = _event_option_for_id(state.event, action.target_id)
    if option is None or option.disabled:
        return state, ()

    if _event_uses_flow(state.event.event_id):
        return _choose_event_flow_option(state, action, option)

    if _event_option_is_enchant_choice(option):
        return _choose_event_enchant_option(state, action, option)

    rng = random_from_state(state.rng)
    event = state.event.model_copy(update={"resolved_option_id": option.option_id})
    chosen_event = EffectEvent(
        kind="event_option_chosen",
        source_id=state.event.event_id,
        target_id=option.option_id,
        metadata={"title": option.title},
    )

    next_state, effect_events = _apply_event_option_immediate_effects(state, option, rng)
    reward, reward_events = _event_reward_from_option(next_state, state.event, option, rng)
    next_state = next_state.model_copy(
        update={
            "event": event,
            "reward": reward,
            "rng": capture_random_state(rng),
        }
    )

    if _event_option_starts_combat(state.event, option):
        return _start_event_option_combat(
            next_state,
            state.event,
            option,
            (chosen_event,) + effect_events,
        )

    return next_state, (chosen_event,) + effect_events + reward_events


def _event_option_for_id(
    event: EventState,
    option_id: str,
) -> EventOptionState | None:
    normalized = _normalized_id(option_id)
    for option in event.options:
        if _normalized_id(option.option_id) == normalized:
            return option
    return None


def _choose_event_flow_option(
    state: RunState,
    action: Action,
    option: EventOptionState,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if state.event is None or action.target_id is None:
        return state, ()

    flow_state = _flow_state_from_event_state(state, state.event)
    rng = random_from_state(state.rng)
    resolution = resolve_event_flow_option(flow_state, action.target_id)
    player = state.player.model_copy(
        update={
            "hp": resolution.state.hp,
            "max_hp": resolution.state.max_hp,
            "gold": resolution.state.gold,
        }
    )
    next_state = state.model_copy(update={"player": player})

    chosen_event = EffectEvent(
        kind="event_option_chosen",
        source_id=state.event.event_id,
        target_id=option.option_id,
        metadata={"title": option.title, "flow_page_id": flow_state.page_id},
    )
    effect_events = _event_flow_resolution_events(state, resolution)
    next_state, marker_events = _apply_event_flow_markers(
        next_state,
        resolution.markers,
        rng,
        selected_card_instance_ids=_event_action_card_instance_ids(action),
    )
    event = _event_state_from_flow(
        next_state,
        event_id=state.event.event_id,
        name=state.event.name,
        node=_current_map_node(state),
        flow_state=resolution.state,
        resolved_option_id=option.option_id if resolution.terminal else None,
    )
    next_phase = RunPhase.FAILED if next_state.player.hp <= 0 else RunPhase.EVENT
    next_state = next_state.model_copy(
        update={
            "phase": next_phase,
            "event": event,
            "reward": next_state.reward,
            "rng": capture_random_state(rng),
        }
    )
    return next_state, (chosen_event,) + effect_events + marker_events


def _flow_state_from_event_state(state: RunState, event: EventState) -> EventFlowState:
    metadata = event.metadata
    return event_flow_state(
        event.event_id,
        hp=state.player.hp,
        max_hp=state.player.max_hp,
        gold=state.player.gold,
        page_id=str(metadata.get("flow_page_id", event.page_id)),
        counters=_mapping_from(metadata.get("flow_counters", {})),
        data=_mapping_from(metadata.get("flow_data", {})),
    )


def _event_flow_resolution_events(
    state: RunState,
    resolution: Any,
) -> tuple[EffectEvent, ...]:
    events: list[EffectEvent] = []
    if resolution.gold_delta < 0:
        events.append(EffectEvent(kind="event_gold_lost", amount=abs(resolution.gold_delta)))
    elif resolution.gold_delta > 0:
        events.append(EffectEvent(kind="event_gold_gained", amount=resolution.gold_delta))
    if resolution.hp_delta < 0:
        events.append(EffectEvent(kind="event_hp_lost", amount=abs(resolution.hp_delta)))
    elif resolution.heal_amount > 0 or resolution.hp_delta > 0:
        events.append(
            EffectEvent(
                kind="event_healed",
                amount=resolution.heal_amount or resolution.hp_delta,
            )
        )
    if resolution.max_hp_delta < 0:
        events.append(EffectEvent(kind="event_max_hp_lost", amount=abs(resolution.max_hp_delta)))
    elif resolution.max_hp_delta > 0:
        events.append(EffectEvent(kind="event_max_hp_gained", amount=resolution.max_hp_delta))
    if state.player.hp > 0 and resolution.state.hp <= 0:
        events.append(EffectEvent(kind="event_player_died", source_id=resolution.option.option_id))
    return tuple(events)


def _apply_event_flow_markers(
    state: RunState,
    markers: Sequence[EventFlowMarker],
    rng: Random,
    *,
    selected_card_instance_ids: Sequence[str] = (),
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    next_state = state
    events: list[EffectEvent] = []
    selected_card_ids = list(selected_card_instance_ids)
    for marker in markers:
        if marker.kind is EventFlowMarkerKind.CUSTOM_CARD:
            card = _custom_card_from_event_marker(marker, len(next_state.master_deck) + 1)
            next_state = next_state.model_copy(
                update={"master_deck": next_state.master_deck + (card,)}
            )
            events.append(
                EffectEvent(
                    kind="event_custom_card_created",
                    target_id=card.instance_id,
                    metadata={
                        "card_id": card.card_id,
                        "card_type": card.type.value,
                        "source_event_id": marker.metadata.get("source_event_id"),
                        "rider_id": card.custom.get("rider_id"),
                    },
                )
            )
        elif marker.kind is EventFlowMarkerKind.CARD_ADD and marker.item_id:
            card = _card_from_spec(
                _reward_card_spec(next_state, marker.item_id),
                len(next_state.master_deck) + 1,
            )
            card, card_upgrade_event = _upgrade_card_for_add_relics(next_state, card)
            next_state = next_state.model_copy(
                update={"master_deck": next_state.master_deck + (card,)}
            )
            events.append(
                EffectEvent(
                    kind="event_card_added",
                    target_id=card.instance_id,
                    metadata={"card_id": card.card_id},
                )
            )
            if card_upgrade_event is not None:
                events.append(card_upgrade_event)
        elif marker.kind is EventFlowMarkerKind.RANDOM_CARD:
            next_state, card_events = _add_event_flow_random_cards(next_state, rng, marker)
            events.extend(card_events)
        elif marker.kind is EventFlowMarkerKind.CARD_REWARD:
            next_state, reward_events = _add_event_flow_card_rewards(next_state, rng, marker)
            events.extend(reward_events)
        elif marker.kind is EventFlowMarkerKind.CARD_UPGRADE_RANDOM:
            next_state, upgrade_events = _upgrade_random_deck_cards(
                next_state,
                rng,
                marker.count,
                source_id="event_flow",
            )
            events.extend(upgrade_events)
        elif marker.kind is EventFlowMarkerKind.CARD_UPGRADE_ALL:
            next_state, upgrade_events = _upgrade_all_deck_cards(
                next_state,
                source_id="event_flow",
            )
            events.extend(upgrade_events)
        elif marker.kind is EventFlowMarkerKind.CARD_REMOVE_RANDOM:
            next_state, remove_events = _remove_random_deck_cards(
                next_state,
                rng,
                marker.count,
                source_id="event_flow",
            )
            events.extend(remove_events)
        elif marker.kind is EventFlowMarkerKind.CARD_DOWNGRADE_RANDOM:
            next_state, downgrade_events = _downgrade_random_deck_cards(
                next_state,
                rng,
                marker.count,
                source_id="event_flow",
            )
            events.extend(downgrade_events)
        elif marker.kind is EventFlowMarkerKind.CARD_TRANSFORM:
            if _event_transform_marker_is_random(marker):
                selected_for_marker: tuple[str, ...] = ()
            else:
                selected_for_marker = tuple(selected_card_ids[: marker.count])
                del selected_card_ids[: marker.count]
            next_state, transform_events, transformed_count = _transform_deck_cards(
                next_state,
                rng,
                marker,
                selected_card_instance_ids=selected_for_marker,
                source_id="event_flow",
            )
            events.extend(transform_events)
            if not _event_transform_marker_is_random(marker) and transformed_count < marker.count:
                events.append(
                    EffectEvent(
                        kind="event_card_transform_choice_required",
                        source_id="event_flow",
                        amount=marker.count - transformed_count,
                        metadata={
                            "marker_kind": marker.kind.value,
                            "count": marker.count,
                            "selected_count": len(selected_for_marker),
                            "description": marker.description,
                        },
                    )
                )
        elif marker.kind is EventFlowMarkerKind.FIXED_RELIC and marker.item_id:
            next_state, relic_events = _add_event_relic(next_state, marker.item_id)
            events.extend(relic_events)
        elif marker.kind is EventFlowMarkerKind.RANDOM_RELIC:
            relic_ids = _draw_event_flow_relic_ids(next_state, rng, marker)
            for relic_id in relic_ids:
                next_state, relic_events = _add_event_relic(next_state, relic_id)
                events.extend(relic_events)
        elif marker.kind is EventFlowMarkerKind.RANDOM_POTION:
            potion_ids = _draw_event_flow_potion_ids(next_state, rng, marker)
            next_state, potion_events = _add_event_potions(next_state, potion_ids)
            events.extend(potion_events)
        elif marker.kind is EventFlowMarkerKind.FIXED_POTION and marker.item_id:
            next_state, potion_events = _add_event_potions(next_state, (marker.item_id,))
            events.extend(potion_events)
        elif marker.kind is EventFlowMarkerKind.DELAYED_REWARD:
            next_state, delayed_events = _schedule_delayed_event_reward(next_state, marker)
            events.extend(delayed_events)
        elif marker.kind is EventFlowMarkerKind.RUN_DEATH:
            player = next_state.player.model_copy(update={"hp": 0})
            next_state = next_state.model_copy(update={"player": player})
            events.append(
                EffectEvent(
                    kind="event_player_died",
                    source_id="event_flow",
                    metadata={"marker_kind": marker.kind.value},
                )
            )
        else:
            events.append(
                EffectEvent(
                    kind="event_flow_marker_recorded",
                    target_id=marker.item_id,
                    metadata={
                        "marker_kind": marker.kind.value,
                        "count": marker.count,
                        "qualifier": marker.qualifier,
                        "description": marker.description,
                    },
                )
            )
    return next_state, tuple(events)


def _schedule_delayed_event_reward(
    state: RunState,
    marker: EventFlowMarker,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    request = _delayed_event_reward_from_marker(marker)
    if request is None:
        return state, (
            EffectEvent(
                kind="event_delayed_reward_blocked",
                metadata={
                    "marker_kind": marker.kind.value,
                    "count": marker.count,
                    "qualifier": marker.qualifier,
                    "description": marker.description,
                    "reason": "missing_reward_kind",
                },
            ),
        )

    flags = dict(state.flags)
    queue = list(_delayed_event_rewards_from_flags(flags))
    queue.append(request)
    flags = _write_delayed_event_rewards(flags, queue)
    return (
        state.model_copy(update={"flags": flags}),
        (
            EffectEvent(
                kind="event_delayed_reward_scheduled",
                amount=request["count"],
                metadata={
                    "reward_kind": request["reward_kind"],
                    "remaining_combats": request["remaining_combats"],
                    "qualifier": request.get("qualifier"),
                    "item_id": request.get("item_id"),
                    "source_event_id": request.get("source_event_id"),
                    "description": request.get("description", ""),
                },
            ),
        ),
    )


def _delayed_event_reward_from_marker(marker: EventFlowMarker) -> dict[str, Any] | None:
    reward_kinds = {
        "fixed_potion",
        "fixed_relic",
        "random_card",
        "random_potion",
        "random_relic",
        "card_reward",
    }
    raw_qualifier = _normalized_id(str(marker.qualifier or ""))
    raw_metadata = dict(marker.metadata)
    raw_kind = _normalized_id(str(raw_metadata.get("reward_kind", "")))
    if raw_qualifier in reward_kinds:
        reward_kind = raw_qualifier
        qualifier = raw_metadata.get("reward_qualifier")
    elif raw_kind:
        reward_kind = raw_kind
        qualifier = marker.qualifier
    elif marker.kind is not EventFlowMarkerKind.DELAYED_REWARD:
        reward_kind = marker.kind.value
        qualifier = marker.qualifier
    else:
        return None

    source_event_id = raw_metadata.get("source_event_id")
    item_id = _normalized_id(marker.item_id) if marker.item_id else None
    return {
        "reward_kind": reward_kind,
        "count": max(0, marker.count),
        "remaining_combats": max(0, marker.delay_combat_count),
        "qualifier": _normalized_id(str(qualifier)) if qualifier else None,
        "item_id": item_id,
        "source_event_id": str(source_event_id) if source_event_id is not None else None,
        "description": marker.description,
        "metadata": {
            "source_marker_kind": marker.kind.value,
            **raw_metadata,
        },
    }


def _delayed_event_rewards_from_flags(flags: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_queue = flags.get(_DELAYED_EVENT_REWARDS_FLAG, ())
    if isinstance(raw_queue, Mapping):
        raw_items: Sequence[Any] = (raw_queue,)
    elif isinstance(raw_queue, Sequence) and not isinstance(raw_queue, (bytes, bytearray, str)):
        raw_items = raw_queue
    else:
        return ()

    queue: list[dict[str, Any]] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            continue
        reward_kind = _normalized_id(
            str(raw_item.get("reward_kind") or raw_item.get("kind") or "")
        )
        if not reward_kind:
            continue
        metadata = dict(_mapping_from(raw_item.get("metadata")))
        source_event_id = raw_item.get("source_event_id", metadata.get("source_event_id"))
        qualifier = raw_item.get("qualifier")
        item_id = raw_item.get("item_id")
        queue.append(
            {
                "reward_kind": reward_kind,
                "count": _nonnegative_int(raw_item.get("count"), 1),
                "remaining_combats": _nonnegative_int(
                    raw_item.get(
                        "remaining_combats",
                        raw_item.get("delay_combat_count", raw_item.get("remaining", 0)),
                    ),
                    0,
                ),
                "qualifier": _normalized_id(str(qualifier)) if qualifier else None,
                "item_id": _normalized_id(str(item_id)) if item_id else None,
                "source_event_id": str(source_event_id) if source_event_id is not None else None,
                "description": str(raw_item.get("description", "")),
                "metadata": metadata,
            }
        )
    return tuple(queue)


def _write_delayed_event_rewards(
    flags: Mapping[str, Any],
    queue: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    next_flags = dict(flags)
    serialized = [dict(item) for item in queue if _nonnegative_int(item.get("count"), 0) > 0]
    if serialized:
        next_flags[_DELAYED_EVENT_REWARDS_FLAG] = serialized
    else:
        next_flags.pop(_DELAYED_EVENT_REWARDS_FLAG, None)
    return next_flags


def _nonnegative_int(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, default)


def _upgrade_all_deck_cards(
    state: RunState,
    *,
    source_id: str,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    deck = list(state.master_deck)
    events: list[EffectEvent] = []
    for index, card in enumerate(deck):
        if card.upgraded:
            continue
        deck[index] = _upgrade_card_instance(card)
        events.append(
            EffectEvent(
                kind="card_upgraded",
                source_id=source_id,
                target_id=card.instance_id,
                metadata={"card_id": card.card_id},
            )
        )
    return state.model_copy(update={"master_deck": tuple(deck)}), tuple(events)


def _remove_random_deck_cards(
    state: RunState,
    rng: Random,
    count: int,
    *,
    source_id: str,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    deck = list(state.master_deck)
    indices = list(range(len(deck)))
    rng.shuffle(indices)
    remove_indices = set(indices[: max(0, count)])
    removed = [card for index, card in enumerate(deck) if index in remove_indices]
    deck = [card for index, card in enumerate(deck) if index not in remove_indices]
    events = tuple(
        EffectEvent(
            kind="event_card_removed_random",
            source_id=source_id,
            target_id=card.instance_id,
            metadata={"card_id": card.card_id},
        )
        for card in removed
    )
    return state.model_copy(update={"master_deck": tuple(deck)}), events


def _downgrade_random_deck_cards(
    state: RunState,
    rng: Random,
    count: int,
    *,
    source_id: str,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    deck = list(state.master_deck)
    candidates = [index for index, card in enumerate(deck) if card.upgraded]
    rng.shuffle(candidates)
    events: list[EffectEvent] = []
    for index in candidates[: max(0, count)]:
        card = deck[index]
        deck[index] = _downgrade_card_instance(card)
        events.append(
            EffectEvent(
                kind="event_card_downgraded_random",
                source_id=source_id,
                target_id=card.instance_id,
                metadata={"card_id": card.card_id},
            )
        )
    return state.model_copy(update={"master_deck": tuple(deck)}), tuple(events)


def _transform_deck_cards(
    state: RunState,
    rng: Random,
    marker: EventFlowMarker,
    *,
    selected_card_instance_ids: Sequence[str],
    source_id: str,
) -> tuple[RunState, tuple[EffectEvent, ...], int]:
    if marker.count <= 0:
        return state, (), 0

    deck = list(state.master_deck)
    if selected_card_instance_ids:
        candidate_indices = [
            index
            for card_instance_id in selected_card_instance_ids
            for index, card in enumerate(deck)
            if card.instance_id == card_instance_id
        ]
    elif _event_transform_marker_is_random(marker):
        candidate_indices = list(range(len(deck)))
        rng.shuffle(candidate_indices)
        candidate_indices = candidate_indices[: marker.count]
    else:
        candidate_indices = []

    events: list[EffectEvent] = []
    transformed_count = 0
    used_indices: set[int] = set()
    for index in candidate_indices:
        if index in used_indices or not 0 <= index < len(deck):
            continue
        used_indices.add(index)
        old_card = deck[index]
        new_card_id = _draw_transformed_card_id(
            state,
            rng,
            marker,
            excluded_card_ids=(old_card.card_id,),
        )
        if new_card_id is None:
            events.append(
                EffectEvent(
                    kind="event_card_transform_blocked",
                    source_id=source_id,
                    target_id=old_card.instance_id,
                    metadata={
                        "old_card_id": old_card.card_id,
                        "reason": "empty_transform_pool",
                        "qualifier": marker.qualifier,
                    },
                )
            )
            continue

        new_spec = _reward_card_spec(state, new_card_id)
        new_spec["card_id"] = new_card_id
        new_spec["instance_id"] = old_card.instance_id
        if old_card.upgraded:
            new_spec["upgraded"] = True
        new_card = _card_from_spec(new_spec, index + 1)
        custom = dict(new_card.custom)
        custom.update(
            {
                "transformed_from_card_id": old_card.card_id,
                "transformed_from_instance_id": old_card.instance_id,
            }
        )
        deck[index] = new_card.model_copy(update={"custom": custom})
        transformed_count += 1
        events.append(
            EffectEvent(
                kind="event_card_transformed",
                source_id=source_id,
                target_id=old_card.instance_id,
                metadata={
                    "old_card_id": old_card.card_id,
                    "new_card_id": deck[index].card_id,
                    "upgraded": deck[index].upgraded,
                    "qualifier": marker.qualifier,
                },
            )
        )
    return state.model_copy(update={"master_deck": tuple(deck)}), tuple(events), transformed_count


def _event_transform_marker_is_random(marker: EventFlowMarker) -> bool:
    if marker.kind is not EventFlowMarkerKind.CARD_TRANSFORM:
        return False
    metadata = _mapping_from(marker.metadata)
    if _truthy(metadata.get("random")) or _truthy(metadata.get("random_transform")):
        return True
    qualifier = _normalized_id(str(marker.qualifier or ""))
    if qualifier in {"random", "random_card"}:
        return True
    description = marker.description.lower()
    return bool(re.search(r"\brandom(?:ly)?\b", description))


def _add_event_flow_random_cards(
    state: RunState,
    rng: Random,
    marker: EventFlowMarker,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    card_ids = _draw_event_flow_card_ids(state, rng, marker, count=marker.count)
    deck = list(state.master_deck)
    events: list[EffectEvent] = []
    next_state = state
    for card_id in card_ids:
        card = _card_from_spec(_reward_card_spec(next_state, card_id), len(deck) + 1)
        card, card_upgrade_event = _upgrade_card_for_add_relics(next_state, card)
        deck.append(card)
        next_state = next_state.model_copy(update={"master_deck": tuple(deck)})
        events.append(
            EffectEvent(
                kind="event_random_card_added",
                source_id="event_flow",
                target_id=card.instance_id,
                metadata={
                    "card_id": card.card_id,
                    "qualifier": marker.qualifier,
                    "description": marker.description,
                },
            )
        )
        if card_upgrade_event is not None:
            events.append(card_upgrade_event)
    if not card_ids:
        events.append(
            EffectEvent(
                kind="event_random_card_blocked",
                source_id="event_flow",
                metadata={
                    "reason": "empty_card_pool",
                    "qualifier": marker.qualifier,
                    "description": marker.description,
                },
            )
        )
    return next_state, tuple(events)


def _add_event_flow_card_rewards(
    state: RunState,
    rng: Random,
    marker: EventFlowMarker,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    groups, rarities, next_pity = _draw_event_flow_card_reward_groups(state, rng, marker)
    if not groups:
        return state, (
            EffectEvent(
                kind="event_card_reward_blocked",
                source_id="event_flow",
                metadata={
                    "reason": "empty_card_pool",
                    "qualifier": marker.qualifier,
                    "description": marker.description,
                },
            ),
        )

    reward = _reward_with_card_option_groups(
        state.reward,
        reward_id=f"event_flow:card_reward:{len(state.room_history)}",
        groups=groups,
        rarities=rarities,
        metadata={
            "marker_kind": marker.kind.value,
            "qualifier": marker.qualifier,
            "description": marker.description,
            **dict(marker.metadata),
        },
    )
    flags = dict(state.flags)
    flags["card_non_rare_count"] = next_pity.card_non_rare_count
    next_state = state.model_copy(update={"reward": reward, "flags": flags})
    return next_state, _reward_generated_events(reward)


def _reward_with_card_option_groups(
    reward: RewardState | None,
    *,
    reward_id: str,
    groups: Sequence[Sequence[str]],
    rarities: Sequence[Sequence[str]],
    metadata: Mapping[str, Any],
) -> RewardState:
    normalized_groups = tuple(
        tuple(_normalized_id(card_id) for card_id in group) for group in groups
    )
    normalized_rarities = tuple(
        tuple(str(rarity) for rarity in group) for group in rarities
    )
    if reward is None:
        card_options = normalized_groups[0] if normalized_groups else ()
        extra_groups = normalized_groups[1:]
        reward_metadata = {
            **dict(metadata),
            "card_group_rarities": normalized_rarities,
            "card_reward_group_count": len(normalized_groups),
        }
        return RewardState(
            reward_id=reward_id,
            source="event",
            forced=False,
            card_options=card_options,
            card_option_groups=extra_groups,
            metadata=reward_metadata,
        )

    reward_metadata = dict(reward.metadata)
    existing_rarities = reward_metadata.get("card_group_rarities", ())
    if isinstance(existing_rarities, Sequence) and not isinstance(
        existing_rarities,
        (str, bytes, bytearray),
    ):
        rarity_groups = tuple(tuple(str(item) for item in group) for group in existing_rarities)
    else:
        rarity_groups = ()
    reward_metadata.update(
        {
            **dict(metadata),
            "card_group_rarities": rarity_groups + normalized_rarities,
            "card_reward_group_count": len(reward.card_option_groups) + len(normalized_groups),
        }
    )
    return reward.model_copy(
        update={
            "card_option_groups": reward.card_option_groups + normalized_groups,
            "metadata": reward_metadata,
        }
    )


def _draw_event_flow_card_reward_groups(
    state: RunState,
    rng: Random,
    marker: EventFlowMarker,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...], RewardPityState]:
    group_count = max(1, marker.count)
    card_count = _card_reward_choice_count(state, marker=marker)
    current_pity = RewardPityState(
        card_non_rare_count=_flag_int(state, "card_non_rare_count", 0),
        potion_chance_bonus=_flag_int(state, "potion_chance_bonus", 0),
    )
    groups: list[tuple[str, ...]] = []
    rarities: list[tuple[str, ...]] = []
    for _ in range(group_count):
        group, group_rarities, current_pity = _draw_event_flow_card_reward_group(
            state,
            rng,
            marker,
            card_count=card_count,
            pity_state=current_pity,
        )
        if group:
            groups.append(group)
            rarities.append(group_rarities)
    return tuple(groups), tuple(rarities), current_pity


def _draw_event_flow_card_reward_group(
    state: RunState,
    rng: Random,
    marker: EventFlowMarker,
    *,
    card_count: int,
    pity_state: RewardPityState,
) -> tuple[tuple[str, ...], tuple[str, ...], RewardPityState]:
    rarity = _event_flow_card_rarity_qualifier(marker)
    if rarity is not None:
        cards = _draw_event_flow_card_ids(state, rng, marker, count=card_count)
        return cards, tuple(rarity.value for _ in cards), pity_state

    cards, rarities, next_pity = draw_card_reward_options(
        rng,
        card_pool=_event_flow_card_pool(state, marker),
        context=CombatRewardContext(
            character_id=state.character_id,
            encounter=EncounterType.NORMAL,
            act=state.act,
            floor=state.floor,
            ascension_level=state.ascension,
            owned_relics=state.relics,
        ),
        pity_state=pity_state,
        card_count=card_count,
    )
    return cards, rarities, next_pity


def _draw_event_flow_card_ids(
    state: RunState,
    rng: Random,
    marker: EventFlowMarker,
    *,
    count: int,
) -> tuple[str, ...]:
    pool = _event_flow_card_pool(state, marker)
    rarity = _event_flow_card_rarity_qualifier(marker)
    if rarity is not None:
        pool = tuple(card for card in pool if card.rarity is rarity)
    candidates = list(pool)
    rng.shuffle(candidates)
    return tuple(card.card_id for card in candidates[: max(0, count)])


def _event_flow_card_pool(
    state: RunState,
    marker: EventFlowMarker,
) -> tuple[Any, ...]:
    qualifier = _normalized_id(str(marker.qualifier or ""))
    raw_cards = _raw_card_pool_items(state)
    if qualifier == "colorless":
        return build_colorless_card_pool(raw_cards)
    if qualifier in {"character", "class"}:
        return build_character_card_pool(raw_cards, character_id=state.character_id)
    return _combat_card_pool(state)


def _event_flow_card_rarity_qualifier(marker: EventFlowMarker) -> CardRarity | None:
    qualifier = _normalized_id(str(marker.qualifier or ""))
    for rarity in CardRarity:
        if qualifier == rarity.value:
            return rarity
    return None


def _card_reward_choice_count(
    state: RunState,
    *,
    marker: EventFlowMarker | None = None,
    default: int = 3,
) -> int:
    metadata = _mapping_from(marker.metadata) if marker is not None else {}
    explicit = metadata.get("card_reward_choice_count", state.flags.get("card_reward_choice_count"))
    with suppress(TypeError, ValueError):
        if explicit is not None:
            default = int(explicit)

    count = default
    if _has_relic(state, "question_card"):
        count += 1
    if _has_relic(state, "busted_crown"):
        count -= 2
    count += _flag_int(state, "card_reward_choice_bonus", 0)
    count += _flag_int(state, "card_reward_choice_delta", 0)
    return max(0, count)


def _draw_transformed_card_id(
    state: RunState,
    rng: Random,
    marker: EventFlowMarker,
    *,
    excluded_card_ids: Sequence[str] = (),
) -> str | None:
    excluded = {_normalized_id(card_id) for card_id in excluded_card_ids}
    candidates = [
        str(card.card_id)
        for card in _transform_card_pool(state, marker)
        if _normalized_id(card.card_id) not in excluded
    ]
    if not candidates:
        return None
    return rng.choice(tuple(candidates))


def _transform_card_pool(
    state: RunState,
    marker: EventFlowMarker,
) -> tuple[Any, ...]:
    qualifier = _normalized_id(str(marker.qualifier or ""))
    raw_cards = _raw_card_pool_items(state)
    if qualifier == "colorless":
        return build_colorless_card_pool(raw_cards)
    if qualifier in {"character", "class"}:
        return build_character_card_pool(raw_cards, character_id=state.character_id)

    pool = _combat_card_pool(state)
    if qualifier in {rarity.value for rarity in CardRarity}:
        pool = tuple(card for card in pool if card.rarity.value == qualifier)
    return tuple(pool)


def _add_event_relic(
    state: RunState,
    relic_id: str,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    normalized = _normalized_id(relic_id)
    if _has_relic(state, normalized):
        return state, (
            EffectEvent(
                kind="event_relic_duplicate_skipped",
                target_id=normalized,
                metadata={"reason": "already_owned"},
            ),
        )
    next_state = state.model_copy(update={"relics": state.relics + (normalized,)})
    next_state, pickup_events = _apply_relic_pickup_effects(next_state, normalized)
    return next_state, (
        EffectEvent(kind="event_relic_obtained", target_id=normalized),
    ) + pickup_events


def _draw_event_flow_relic_ids(
    state: RunState,
    rng: Random,
    marker: EventFlowMarker,
) -> tuple[str, ...]:
    pool = _treasure_relic_pool(state)
    qualifier = _normalized_id(str(marker.qualifier or ""))
    if qualifier in {rarity.value for rarity in RelicRarity}:
        pool = tuple(relic for relic in pool if relic.rarity.value == qualifier)
    return _draw_event_relic_ids_from_pool(state, rng, max(0, marker.count), pool)


def _draw_event_relic_ids_from_pool(
    state: RunState,
    rng: Random,
    count: int,
    pool: Sequence[TreasureRelic],
) -> tuple[str, ...]:
    relic_ids: list[str] = []
    for _ in range(max(0, count)):
        reward = draw_treasure_reward(
            rng,
            pool,
            TreasureContext(
                character_id=state.character_id,
                act=state.act,
                floor=state.floor,
                ascension_level=state.ascension,
                owned_relics=state.relics + tuple(relic_ids),
                opened_chests=1,
            ),
        )
        if reward.relic_id is None:
            break
        relic_ids.append(reward.relic_id)
    return tuple(relic_ids)


def _draw_event_flow_potion_ids(
    state: RunState,
    rng: Random,
    marker: EventFlowMarker,
) -> tuple[str, ...]:
    pool = _reward_potion_pool(state, "event_reward_potion_pool") or (
        "fire_potion",
        "skill_potion",
        "essence_of_steel",
    )
    return tuple(rng.choice(tuple(pool)) for _ in range(max(0, marker.count)))


def _add_event_potions(
    state: RunState,
    potion_ids: Sequence[str],
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    potions = list(state.potions)
    capacity = _potion_capacity(state)
    events: list[EffectEvent] = []
    for potion_id in potion_ids:
        normalized = _normalized_id(potion_id)
        if len(potions) >= capacity:
            events.append(
                EffectEvent(
                    kind="event_potion_skipped_no_slot",
                    target_id=normalized,
                    metadata={"potion_slots": capacity},
                )
            )
            continue
        potions.append(normalized)
        events.append(
            EffectEvent(
                kind="event_potion_obtained",
                target_id=normalized,
                metadata={"potion_slots": capacity},
            )
        )
    return state.model_copy(update={"potions": tuple(potions)}), tuple(events)


def _choose_event_enchant_option(
    state: RunState,
    action: Action,
    option: EventOptionState,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if state.event is None:
        return state, ()

    selected_ids = _event_enchant_selected_card_ids(action)
    if len(selected_ids) != _event_enchant_count(option):
        return state, ()

    eligible_ids = {card.instance_id for card in _eligible_enchant_cards(state, option)}
    if any(card_id not in eligible_ids for card_id in selected_ids):
        return state, ()

    rng = random_from_state(state.rng)
    chosen_event = EffectEvent(
        kind="event_option_chosen",
        source_id=state.event.event_id,
        target_id=option.option_id,
        metadata={"title": option.title, "card_instance_ids": selected_ids},
    )
    next_state, effect_events = _apply_event_option_immediate_effects(state, option, rng)
    next_state, enchant_events = _apply_event_enchants(next_state, option, selected_ids)
    reward, reward_events = _event_reward_from_option(next_state, state.event, option, rng)
    reward = _drop_enchant_only_reward(reward)
    event = state.event.model_copy(update={"resolved_option_id": option.option_id})
    next_state = next_state.model_copy(
        update={
            "event": event,
            "reward": reward,
            "rng": capture_random_state(rng),
        }
    )
    return next_state, (chosen_event,) + effect_events + enchant_events + reward_events


def _event_enchant_selected_card_ids(action: Action) -> tuple[str, ...]:
    payload_ids = action.payload.get("card_instance_ids")
    if isinstance(payload_ids, Sequence) and not isinstance(
        payload_ids,
        (str, bytes, bytearray),
    ):
        return tuple(str(card_id) for card_id in payload_ids)
    if action.card_instance_id is not None:
        return (action.card_instance_id,)
    return ()


def _apply_event_enchants(
    state: RunState,
    option: EventOptionState,
    card_instance_ids: Sequence[str],
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    keyword = str(option.metadata.get("enchant_keyword", "Unknown"))
    amount = _event_enchant_amount(option)
    selected = set(card_instance_ids)
    events: list[EffectEvent] = []
    deck: list[CardInstance] = []
    for card in state.master_deck:
        if card.instance_id not in selected:
            deck.append(card)
            continue
        source_id = state.event.event_id if state.event else None
        enchanted = _enchant_card(card, keyword, amount, source_id=source_id)
        deck.append(enchanted)
        events.append(
            EffectEvent(
                kind="card_enchanted",
                source_id=state.event.event_id if state.event else None,
                target_id=card.instance_id,
                amount=amount,
                metadata={
                    "card_id": card.card_id,
                    "keyword": keyword,
                    "previous_cost": card.cost,
                    "new_cost": enchanted.cost,
                },
            )
        )
    return state.model_copy(update={"master_deck": tuple(deck)}), tuple(events)


def _drop_enchant_only_reward(reward: RewardState | None) -> RewardState | None:
    if reward is None:
        return None
    if (
        reward.gold <= 0
        and reward.relic_id is None
        and not reward.relic_ids
        and not reward.card_ids
        and not reward.card_options
        and not reward.card_option_groups
        and reward.potion_id is None
        and not reward.potion_ids
        and set(reward.metadata).issubset(
            {"event_id", "option_id", "potion_slots", "current_potions", *_ENCHANT_METADATA_KEYS}
        )
    ):
        return None
    return reward


def _eligible_enchant_cards(
    state: RunState,
    option: EventOptionState,
) -> tuple[CardInstance, ...]:
    card_type = option.metadata.get("enchant_card_type")
    allowed_card_ids = {
        _normalized_id(card_id)
        for card_id in _metadata_str_sequence(option.metadata.get("enchant_card_ids"))
    }
    requires_exhaust = bool(option.metadata.get("enchant_requires_exhaust"))
    return tuple(
        card
        for card in state.master_deck
        if (card_type is None or card.type.value == str(card_type))
        and (not allowed_card_ids or _normalized_id(card.card_id) in allowed_card_ids)
        and (not requires_exhaust or card.exhausts or bool(card.effects.get("exhaust_on_play")))
    )


def _event_enchant_count(option: EventOptionState) -> int:
    value = option.metadata.get("enchant_count", 1)
    with suppress(TypeError, ValueError):
        return max(1, int(value))
    return 1


def _event_enchant_amount(option: EventOptionState) -> int:
    value = option.metadata.get("enchant_amount", 0)
    with suppress(TypeError, ValueError):
        return max(0, int(value))
    return 0


def _metadata_str_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value)
    return ()


def _enchant_card(
    card: CardInstance,
    keyword: str,
    amount: int,
    *,
    source_id: str | None,
) -> CardInstance:
    normalized_keyword = _normalized_id(keyword)
    cost = card.cost
    if normalized_keyword == "swift" and cost is not None and cost >= 0:
        cost = max(0, cost - amount)

    return card.model_copy(
        update={
            "cost": cost,
            "effects": _enchant_card_effects(card.effects, normalized_keyword, amount),
            "tags": tuple(
                dict.fromkeys(card.tags + (f"enchant:{normalized_keyword}",))
            ),
            "enchantments": card.enchantments
            + (
                CardEnchantment(
                    keyword=keyword,
                    amount=amount,
                    source_id=source_id,
                    metadata={"normalized_keyword": normalized_keyword},
                ),
            ),
        }
    )


def _enchant_card_effects(
    effects: Mapping[str, Any],
    normalized_keyword: str,
    amount: int,
) -> dict[str, Any]:
    if amount <= 0:
        return dict(effects)
    if "sequence" in effects and isinstance(effects["sequence"], Sequence):
        return {
            **dict(effects),
            "sequence": [
                _enchant_card_effects(step, normalized_keyword, amount)
                if isinstance(step, Mapping)
                else step
                for step in effects["sequence"]
            ],
        }
    if "effects" in effects and isinstance(effects["effects"], Sequence):
        return {
            **dict(effects),
            "effects": [
                _enchant_card_effects(step, normalized_keyword, amount)
                if isinstance(step, Mapping)
                else step
                for step in effects["effects"]
            ],
        }

    updated = dict(effects)
    if normalized_keyword in {"sharp", "vigorous"}:
        if "damage" in updated:
            updated["damage"] = _increment_effect_value(updated["damage"], amount)
        if "all_damage" in updated:
            updated["all_damage"] = _increment_effect_value(updated["all_damage"], amount)
    elif normalized_keyword == "nimble" and "block" in updated:
        updated["block"] = _increment_effect_value(updated["block"], amount)
    return updated


def _increment_effect_value(value: Any, amount: int) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value + amount
    if isinstance(value, float):
        return value + amount
    if isinstance(value, Mapping):
        current = value.get("amount", 0)
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            return {**dict(value), "amount": current + amount}
    return value


def _custom_card_from_event_marker(
    marker: EventFlowMarker,
    instance_counter: int,
) -> CardInstance:
    if _normalized_id(str(marker.metadata.get("source_event_id", ""))) == "tinker_time":
        return _tinker_custom_card(marker, instance_counter)
    return _card_from_spec(
        {
            "card_id": marker.item_id or f"custom_card_{instance_counter:03d}",
            "name": marker.item_id or "Custom Card",
            "type": marker.qualifier or "unknown",
            "custom": dict(marker.metadata),
        },
        instance_counter,
    )


def _tinker_custom_card(marker: EventFlowMarker, instance_counter: int) -> CardInstance:
    metadata = dict(marker.metadata)
    card_type = _normalized_id(str(metadata.get("card_type", marker.qualifier or "unknown")))
    rider_id = _normalized_id(str(metadata.get("rider_id", "unknown")))
    rider = _mapping_from(metadata.get("rider", {}))
    effect = str(rider.get("effect", metadata.get("rider_effect", "")))
    sequence: list[dict[str, Any]] = []
    target = TargetType.SELF.value

    if card_type == "attack":
        target = TargetType.ENEMY.value
        hit_count = 1 + int(rider.get("additional_hits", 0) or 0)
        sequence.extend({"damage": 12} for _ in range(max(1, hit_count)))
    elif card_type == "skill":
        target = TargetType.SELF.value
        sequence.append({"block": 8})
    elif card_type == "power":
        target = TargetType.SELF.value

    if effect == "apply_debuffs":
        sequence.append(
            {
                "apply_status": {
                    "target": "enemy",
                    "weak": int(rider.get("weak", 2)),
                    "vulnerable": int(rider.get("vulnerable", 2)),
                }
            }
        )
    elif effect == "choking":
        sequence.append(
            {
                "apply_status": {
                    "target": "enemy",
                    "choking": int(rider.get("hp_loss_per_card", 6)),
                }
            }
        )
    elif effect == "gain_energy":
        sequence.append({"energy": int(rider.get("energy", 2))})
    elif effect == "draw_cards":
        sequence.append({"draw": int(rider.get("draw", 3))})
    elif effect == "add_random_free_card_to_hand":
        sequence.append({"add_random_card_to_hand": int(rider.get("random_card_count", 1))})
    elif effect == "gain_strength_and_dexterity":
        sequence.append(
            {
                "apply_status": {
                    "target": "self",
                    "strength": int(rider.get("strength", 2)),
                    "dexterity": int(rider.get("dexterity", 2)),
                }
            }
        )
    elif effect == "power_cost_reduction":
        sequence.append(
            {
                "apply_status": {
                    "target": "self",
                    "power_cost_reduction": int(rider.get("power_cost_reduction", 1)),
                }
            }
        )
    elif effect == "end_of_combat_upgrade_random":
        sequence.append(
            {
                "apply_status": {
                    "target": "self",
                    "end_of_combat_upgrade_random": int(rider.get("upgrade_random_count", 1)),
                }
            }
        )

    effects: dict[str, Any] = sequence[0] if len(sequence) == 1 else {"sequence": sequence}

    title = rider_id.replace("_", " ").title()
    return _card_from_spec(
        {
            "instance_id": f"tinker_{instance_counter:03d}",
            "card_id": f"mad_science_{card_type}_{rider_id}",
            "name": f"Mad Science ({title})",
            "type": card_type,
            "cost": 1,
            "target": target,
            "effects": effects,
            "tags": ("custom", "event:tinker_time", f"rider:{rider_id}"),
            "custom": {
                "source_event_id": "TINKER_TIME",
                "base_card_id": "MAD_SCIENCE",
                "card_type": card_type,
                "rider_id": rider_id,
                "rider_effect": effect,
            },
        },
        instance_counter,
    )


def _upgrade_random_deck_cards(
    state: RunState,
    rng: Random,
    count: int,
    *,
    source_id: str,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    deck = list(state.master_deck)
    candidates = [index for index, card in enumerate(deck) if not card.upgraded]
    rng.shuffle(candidates)
    events: list[EffectEvent] = []
    for index in candidates[: max(0, count)]:
        card = deck[index]
        deck[index] = _upgrade_card_instance(card)
        events.append(
            EffectEvent(
                kind="card_upgraded",
                source_id=source_id,
                target_id=card.instance_id,
                metadata={"card_id": card.card_id},
            )
        )
    return state.model_copy(update={"master_deck": tuple(deck)}), tuple(events)


def _apply_event_option_immediate_effects(
    state: RunState,
    option: EventOptionState,
    rng: Random,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    text = option.description
    player = state.player
    deck = list(state.master_deck)
    events: list[EffectEvent] = []
    gold = state.player.gold

    if re.search(r"\bLose\s+ALL\b.*\bGold\b", text, re.IGNORECASE):
        events.append(EffectEvent(kind="event_gold_lost", amount=gold))
        gold = 0

    for amount in _event_gold_cost_amounts(text, rng):
        spent = min(gold, amount)
        gold -= spent
        events.append(EffectEvent(kind="event_gold_lost", amount=spent))

    for amount in _event_number_matches(text, r"\bLose\s+(\d+)\s+HP\b"):
        next_hp = max(0, player.hp - amount)
        events.append(EffectEvent(kind="event_hp_lost", amount=player.hp - next_hp))
        player = player.model_copy(update={"hp": next_hp})

    for amount in _event_number_matches(text, r"\bGain\s+(\d+)\s+Max\s+HP\b"):
        player = player.model_copy(update={"max_hp": player.max_hp + amount})
        events.append(EffectEvent(kind="event_max_hp_gained", amount=amount))

    for amount in _event_number_matches(text, r"\bLose\s+(\d+)\s+Max\s+HP\b"):
        next_max_hp = max(1, player.max_hp - amount)
        player = player.model_copy(
            update={"max_hp": next_max_hp, "hp": min(player.hp, next_max_hp)}
        )
        events.append(EffectEvent(kind="event_max_hp_lost", amount=amount))

    for amount in _event_number_matches(text, r"\bSet\s+Max\s+HP\s+to\s+(\d+)\b"):
        next_max_hp = max(1, amount)
        player = player.model_copy(
            update={"max_hp": next_max_hp, "hp": min(player.hp, next_max_hp)}
        )
        events.append(EffectEvent(kind="event_max_hp_set", amount=amount))

    if re.search(r"\bHeal\s+to\s+full\s+HP\b", text, re.IGNORECASE):
        healed = player.max_hp - player.hp
        player = player.model_copy(update={"hp": player.max_hp})
        events.append(EffectEvent(kind="event_healed", amount=healed))

    for percent in _event_number_matches(text, r"\bHeal\s+(\d+)%\s+Max\s+HP\b"):
        amount = max(0, int(player.max_hp * percent / 100))
        next_hp = min(player.max_hp, player.hp + amount)
        events.append(EffectEvent(kind="event_healed", amount=next_hp - player.hp))
        player = player.model_copy(update={"hp": next_hp})

    for card_name in re.findall(r"\bLose\s+([A-Z][A-Za-z' -]+?)(?:\.|$)", text):
        card_id = _lookup_content_id_by_name(state, "cards", card_name)
        if card_id is None:
            continue
        deck, removed = _remove_first_card_by_id(deck, card_id)
        if removed is not None:
            events.append(
                EffectEvent(
                    kind="event_card_removed",
                    target_id=removed.instance_id,
                    metadata={"card_id": removed.card_id},
                )
            )

    return (
        state.model_copy(
            update={
                "player": player.model_copy(update={"gold": gold}),
                "master_deck": tuple(deck),
            }
        ),
        tuple(events),
    )


def _event_reward_from_option(
    state: RunState,
    event: EventState,
    option: EventOptionState,
    rng: Random,
) -> tuple[RewardState | None, tuple[EffectEvent, ...]]:
    if _event_option_starts_combat(event, option):
        return None, ()

    text = option.description
    gold = _event_gold_reward_amount(text, rng)
    card_ids = _event_reward_card_ids(state, text)
    relic_ids = _event_reward_relic_ids(state, text, rng)
    relic_id = _event_reward_fixed_relic_id(state, text)
    potion_ids = _event_reward_potion_ids(state, text, rng)
    fixed_potion_ids = _event_reward_fixed_potion_ids(state, text)
    potion_ids = fixed_potion_ids + potion_ids
    metadata = _event_reward_metadata(text)

    if (
        gold <= 0
        and relic_id is None
        and not relic_ids
        and not card_ids
        and not potion_ids
        and not metadata
    ):
        return None, ()

    reward = RewardState(
        reward_id=f"event:{event.event_id}:{option.option_id}",
        source="event",
        forced=True,
        gold=gold,
        relic_id=relic_id,
        relic_ids=relic_ids,
        card_ids=card_ids,
        potion_ids=potion_ids,
        metadata={
            "event_id": event.event_id,
            "option_id": option.option_id,
            **metadata,
            "potion_slots": _potion_capacity(state),
            "current_potions": len(state.potions),
        },
    )
    return reward, _reward_generated_events(reward)


def _event_option_starts_combat(event: EventState, option: EventOptionState) -> bool:
    event_id = _normalized_id(event.event_id)
    if event_id == "round_tea_party":
        return False
    text = f"{option.option_id} {option.title} {option.description}"
    return bool(re.search(r"\bfight\b", text, re.IGNORECASE))


def _start_event_option_combat(
    state: RunState,
    event: EventState,
    option: EventOptionState,
    pre_events: tuple[EffectEvent, ...],
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    node = _current_map_node(state) or MapNodeState(
        node_id=f"event:{event.event_id}",
        act=state.act,
        floor=state.floor,
        lane=0,
        kind=RoomKind.MONSTER,
    )
    flags = dict(state.flags)
    flags.update(_event_combat_reward_flags(event, option))
    combat, rng_state, combat_events = _start_combat_for_node(
        state.model_copy(update={"flags": flags}),
        node.model_copy(update={"kind": RoomKind.MONSTER}),
    )
    return (
        state.model_copy(
            update={
                "phase": RunPhase.COMBAT,
                "combat": combat,
                "reward": None,
                "player": combat.player,
                "flags": flags,
                "rng": rng_state,
            }
        ),
        pre_events + combat_events,
    )


def _event_combat_reward_flags(
    event: EventState,
    option: EventOptionState,
) -> dict[str, Any]:
    event_id = _normalized_id(event.event_id)
    option_id = _normalized_id(option.option_id)
    flags: dict[str, Any] = {
        "event_fight_id": event_id,
        "combat_reward_event_id": event_id,
        "combat_reward_encounter": "normal",
    }
    if event_id == "punch_off":
        flags.update(
            {
                "combat_reward_relic_count": 1,
                "combat_reward_extra_potion_count": 1,
            }
        )
    elif event_id == "dense_vegetation":
        flags["combat_reward_encounter"] = "normal"
    elif event_id == "the_lantern_key":
        flags.update(
            {
                "combat_reward_encounter": "normal",
                "combat_reward_card_ids": ("lantern_key",),
            }
        )
    elif event_id == "battleworn_dummy":
        flags.update(
            {
                "combat_reward_encounter": "event",
                "combat_reward_card_count": 0,
                "combat_reward_relic_count": 0,
                "combat_reward_potion_chance_percent": 0,
            }
        )
        if option_id == "setting_1":
            flags["combat_reward_extra_potion_count"] = 1
        elif option_id == "setting_2":
            flags["combat_reward_upgrade_random_cards"] = 2
        elif option_id == "setting_3":
            flags["combat_reward_relic_count"] = 1
    return flags


def _event_reward_card_ids(state: RunState, text: str) -> tuple[str, ...]:
    card_ids: list[str] = []
    for card_name in re.findall(
        r"\bAdd\s+(?:\d+\s+)?([A-Z][A-Za-z' -]+?)\s+to\s+your\s+Deck\b",
        text,
    ):
        card_id = _lookup_content_id_by_name(state, "cards", card_name)
        if card_id is not None:
            card_ids.append(card_id)
    for name in _event_fixed_item_names(text):
        card_id = _lookup_content_id_by_name(state, "cards", name)
        if card_id is not None:
            card_ids.append(card_id)
    return tuple(dict.fromkeys(card_ids))


def _event_reward_fixed_relic_id(state: RunState, text: str) -> str | None:
    for name in _event_fixed_item_names(text):
        relic_id = _lookup_content_id_by_name(state, "relics", name)
        if relic_id is not None:
            return relic_id
    return None


def _event_reward_relic_ids(
    state: RunState,
    text: str,
    rng: Random,
) -> tuple[str, ...]:
    count = _event_random_count(text, r"\bObtain\s+(?:(\d+)\s+)?(?:a\s+)?random\s+Relics?\b")
    if count <= 0:
        return ()
    return _draw_event_relic_ids(state, rng, count)


def _event_reward_potion_ids(
    state: RunState,
    text: str,
    rng: Random,
) -> tuple[str, ...]:
    count = _event_random_count(
        text,
        r"\b(?:Procure|Obtain|Receive)\s+(?:(\d+)\s+)?(?:a\s+)?random\s+Potions?\b",
    )
    if count <= 0:
        return ()
    pool = _reward_potion_pool(state, "event_reward_potion_pool") or (
        "fire_potion",
        "skill_potion",
        "essence_of_steel",
    )
    return tuple(rng.choice(tuple(pool)) for _ in range(count))


def _event_reward_fixed_potion_ids(state: RunState, text: str) -> tuple[str, ...]:
    potion_ids: list[str] = []
    for raw_count, name in re.findall(
        r"\b(?:Procure|Obtain|Receive|Gain)\s+(?:(\d+)\s+)?"
        r"((?!random\b)[A-Z][A-Za-z' -]+?Potions?)\b",
        text,
        flags=re.IGNORECASE,
    ):
        potion_id = _lookup_content_id_by_name(state, "potions", _singular_item_name(name))
        if potion_id is None:
            continue
        count = int(raw_count or "1")
        potion_ids.extend(potion_id for _ in range(max(1, count)))
    return tuple(potion_ids)


def _event_gold_reward_amount(text: str, rng: Random) -> int:
    total = 0
    for value in re.findall(
        r"\b(?:Gain|Obtain)\s+(\d+(?:-\d+)?)\s+Gold\b",
        text,
        flags=re.IGNORECASE,
    ):
        total += _event_roll_amount(value, rng)
    return total


def _event_gold_cost_amounts(text: str, rng: Random) -> tuple[int, ...]:
    return tuple(
        _event_roll_amount(value, rng)
        for value in re.findall(
            r"\b(?:Lose|Pay)\s+(\d+(?:-\d+)?)\s+(?:of\s+your\s+)?Gold\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _event_reward_metadata(text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    upgrade_count = _event_random_count(text, r"\bUpgrade\s+(\d+)\s+random\s+cards?\b")
    if upgrade_count:
        metadata["upgrade_random_cards"] = upgrade_count
    if re.search(r"\bTransform\b", text, re.IGNORECASE):
        metadata["transform_card"] = True
    if re.search(r"\bRemove\b", text, re.IGNORECASE):
        metadata["remove_card"] = True
    metadata.update(_event_option_enchant_metadata(text))
    return metadata


def _event_fixed_item_names(text: str) -> tuple[str, ...]:
    names: list[str] = []
    for name in re.findall(
        r"\b(?:Obtain|Receive)\s+(?:the\s+)?([A-Z][A-Za-z' -]+?)(?:\.|$)",
        text,
    ):
        if re.search(r"\brandom\b|\bRelics?\b|\bPotions?\b", name, re.IGNORECASE):
            continue
        names.append(name.strip())
    return tuple(names)


def _event_random_count(text: str, pattern: str) -> int:
    total = 0
    for raw in re.findall(pattern, text, flags=re.IGNORECASE):
        value = raw[0] if isinstance(raw, tuple) else raw
        total += int(value or "1")
    return total


def _event_number_matches(text: str, pattern: str) -> tuple[int, ...]:
    return tuple(int(match) for match in re.findall(pattern, text, flags=re.IGNORECASE))


def _event_roll_amount(value: str, rng: Random) -> int:
    if "-" not in value:
        return int(value)
    low, high = (int(part) for part in value.split("-", 1))
    return rng.randint(min(low, high), max(low, high))


def _singular_item_name(name: str) -> str:
    name = name.strip()
    if name.endswith(" Potions"):
        return f"{name[:-8]} Potion"
    return name


def _draw_event_relic_ids(state: RunState, rng: Random, count: int) -> tuple[str, ...]:
    relic_ids: list[str] = []
    for _ in range(max(0, count)):
        reward = draw_treasure_reward(
            rng,
            _treasure_relic_pool(state),
            TreasureContext(
                character_id=state.character_id,
                act=state.act,
                floor=state.floor,
                ascension_level=state.ascension,
                owned_relics=state.relics + tuple(relic_ids),
                opened_chests=1,
            ),
        )
        if reward.relic_id is None:
            break
        relic_ids.append(reward.relic_id)
    return tuple(relic_ids)


def _remove_first_card_by_id(
    deck: list[CardInstance],
    card_id: str,
) -> tuple[list[CardInstance], CardInstance | None]:
    normalized = _normalized_id(card_id)
    for index, card in enumerate(deck):
        if _normalized_id(card.card_id) != normalized:
            continue
        removed = deck.pop(index)
        return deck, removed
    return deck, None


def _lookup_content_id_by_name(
    state: RunState,
    dataset: str,
    name: str,
) -> str | None:
    normalized_name = _normalized_text_key(name)
    rows = _source_items(state.flags.get(dataset)) or _cached_source_rows(state, dataset)
    for raw_item in rows:
        item = _mapping_from(raw_item)
        if not item:
            continue
        item_name = str(item.get("name", ""))
        item_id = str(item.get("id", item.get("card_id", item.get("relic_id", ""))))
        if _normalized_text_key(item_name) == normalized_name:
            return _normalized_id(item_id)
    return None


def _clean_event_text(text: str) -> str:
    text = re.sub(r"\[(?:/?[a-zA-Z_]+|/?[a-zA-Z_]+=[^\]]+)\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalized_text_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _apply_event_reward_card_costs(
    state: RunState,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    card_ids = tuple(
        _normalized_id(card_id)
        for card_id in _flag_str_sequence(state, "event_reward_remove_card_ids")
    )
    if not card_ids:
        return state, ()

    deck = list(state.master_deck)
    events: list[EffectEvent] = []
    for card_id in card_ids:
        for index, card in enumerate(deck):
            if _normalized_id(card.card_id) != card_id:
                continue
            removed = deck.pop(index)
            events.append(
                EffectEvent(
                    kind="event_card_removed",
                    target_id=removed.instance_id,
                    metadata={"card_id": removed.card_id},
                )
            )
            break
    return state.model_copy(update={"master_deck": tuple(deck)}), tuple(events)


def _enter_shop_room(
    state: RunState, node: MapNodeState
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    rng = random_from_state(state.rng)
    inventory = build_basic_shop_inventory(
        rng,
        card_pool=_shop_item_pool(state, "shop_card_pool", _default_shop_card_pool()),
        colorless_card_pool=_shop_item_pool(
            state,
            "shop_colorless_card_pool",
            _default_shop_colorless_pool(),
            default_kind=ShopItemKind.COLORLESS_CARD,
        ),
        relic_pool=_shop_item_pool(
            state,
            "shop_relic_pool",
            _default_shop_relic_pool(),
            default_kind=ShopItemKind.RELIC,
        ),
        potion_pool=_shop_item_pool(
            state,
            "shop_potion_pool",
            _default_shop_potion_pool(),
            default_kind=ShopItemKind.POTION,
        ),
        plan=_shop_inventory_plan(state),
        ascension_level=state.ascension,
        rare_offset_percent=_flag_float(state, "card_rare_offset_percent", 0.0),
        card_removals_bought=_flag_int(state, "shop_card_removals_bought", 0),
    )
    shop = _shop_state_from_inventory(
        node.node_id,
        inventory,
        card_removals_bought=_flag_int(state, "shop_card_removals_bought", 0),
    )
    shop = _apply_shop_price_modifiers(state, shop)
    player, entry_events = _apply_shop_entry_relics(state.player, state)
    state = state.model_copy(update={"player": player})
    parasol_state, parasol_events = _apply_lords_parasol(state, shop)
    if parasol_state is not None:
        state = parasol_state
        shop = parasol_state.shop if parasol_state.shop is not None else shop
    return (
        state.model_copy(
            update={
                "phase": RunPhase.SHOP,
                "combat": None,
                "shop": shop,
                "rng": capture_random_state(rng),
            }
        ),
        (
            EffectEvent(
                kind="shop_ready",
                target_id=node.node_id,
                amount=len(shop.items),
            ),
        )
        + entry_events
        + parasol_events,
    )


def _legal_shop_actions(state: RunState) -> tuple[Action, ...]:
    if state.shop is None:
        return ()
    room_state = _shop_room_state(state)
    actions: list[Action] = []
    for choice in available_shop_actions(room_state):
        if choice.action is ShopRoomAction.LEAVE:
            actions.append(Action(type=ActionType.SHOP_LEAVE))
        elif choice.item_index is not None:
            item = room_state.inventory.items[choice.item_index].item
            if item.kind is ShopItemKind.POTION and not _has_open_potion_slot(state):
                continue
            actions.append(
                Action(
                    type=ActionType.SHOP_BUY,
                    target_id=_shop_choice_target_id(choice),
                )
            )
    if "foul_potion" in state.potions:
        actions.append(Action(type=ActionType.THROW_POTION_AT_MERCHANT))
    actions.append(Action(type=ActionType.PROCEED))
    return tuple(actions)


def _resolve_shop_action(
    state: RunState, action: Action
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if action.type == ActionType.SHOP_LEAVE:
        return _leave_shop(state)
    if action.type == ActionType.THROW_POTION_AT_MERCHANT:
        return _throw_potion_at_merchant(state)
    if action.type != ActionType.SHOP_BUY or action.target_id is None:
        return state, ()

    item_index, remove_card_id = _parse_shop_target_id(action.target_id)
    room_state = _shop_room_state(state)
    result = resolve_shop_action(
        ShopRoomChoice(
            ShopRoomAction.BUY_ITEM,
            item_index=item_index,
            target_card_id=remove_card_id,
        ),
        room_state,
    )
    item = result.purchased_item
    if item is None:
        return state, ()

    player = state.player.model_copy(update={"gold": result.state.gold})
    shop = _shop_after_purchase(state, item_index, item.item.kind)
    flags = dict(state.flags)
    flags["shop_card_removals_bought"] = result.state.card_removals_bought
    rng_state = state.rng
    restocked_item_id: str | None = None
    if (
        _has_relic(state, "the_courier", "courier")
        and item.item.kind is not ShopItemKind.CARD_REMOVAL
    ):
        rng = random_from_state(state.rng)
        shop, restocked_item_id = _restock_shop_slot(state, shop, item_index, item.item.kind, rng)
        rng_state = capture_random_state(rng)

    update: dict[str, Any] = {
        "player": player,
        "shop": shop,
        "flags": flags,
        "rng": rng_state,
    }
    event_kind = "shop_item_bought"
    metadata: dict[str, Any] = {
        "item_id": item.item.item_id,
        "item_kind": item.item.kind.value,
        "price": item.price,
        "base_price": item.base_price,
        "slot_id": _shop_slot_id(item_index),
    }
    if restocked_item_id is not None:
        metadata["restocked_item_id"] = restocked_item_id

    card_upgrade_event: EffectEvent | None = None
    if item.item.kind in {ShopItemKind.CARD, ShopItemKind.COLORLESS_CARD}:
        card_spec = _reward_card_spec(state, item.item.item_id)
        card_spec["instance_id"] = f"shop_{len(state.replay_log)}_{item_index}"
        card = _card_from_spec(card_spec, len(state.master_deck) + 1)
        card, card_upgrade_event = _upgrade_card_for_add_relics(state, card)
        update["master_deck"] = state.master_deck + (card,)
        metadata["card_instance_id"] = card.instance_id
        if card_upgrade_event is not None:
            metadata["upgraded_by_relic"] = card_upgrade_event.source_id
    elif item.item.kind is ShopItemKind.RELIC:
        update["relics"] = state.relics + (item.item.item_id,)
    elif item.item.kind is ShopItemKind.POTION:
        if not _has_open_potion_slot(state):
            return state, ()
        update["potions"] = state.potions + (item.item.item_id,)
    elif item.item.kind is ShopItemKind.CARD_REMOVAL and result.removed_card_id is not None:
        removed_card = _find_card(state.master_deck, result.removed_card_id)
        update["master_deck"] = tuple(
            deck_card
            for deck_card in state.master_deck
            if deck_card.instance_id != result.removed_card_id
        )
        event_kind = "shop_card_removed"
        metadata["removed_card_instance_id"] = result.removed_card_id
        if removed_card is not None:
            metadata["removed_card_id"] = removed_card.card_id

    next_state = state.model_copy(update=update)
    if next_state.shop is not None:
        next_state = next_state.model_copy(
            update={"shop": _apply_shop_price_modifiers(next_state, next_state.shop)}
        )

    events = [
        EffectEvent(
            kind=event_kind,
            target_id=item.item.item_id,
            amount=result.gold_delta,
            metadata=metadata,
        ),
    ]
    if (
        item.item.kind in {ShopItemKind.CARD, ShopItemKind.COLORLESS_CARD}
        and card_upgrade_event is not None
    ):
        events.append(card_upgrade_event)

    return next_state, tuple(events)


def _use_potion(
    state: RunState,
    action: Action,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if state.phase != RunPhase.COMBAT or state.combat is None:
        return state, ()

    slot_id = action.payload.get("potion_slot")
    if not isinstance(slot_id, str):
        return state, ()
    try:
        slot_index = _parse_potion_slot_id(slot_id)
    except ValueError:
        return state, ()
    if slot_index < 0 or slot_index >= len(state.potions):
        return state, ()

    potion_id = state.potions[slot_index]
    if not _potion_is_manually_usable(potion_id):
        return state, ()

    potions = tuple(
        potion
        for index, potion in enumerate(state.potions)
        if index != slot_index
    )
    potion_source_id = f"{slot_id}:{potion_id}"
    combat = state.combat
    rng_state = state.rng
    events: list[EffectEvent] = [
        EffectEvent(
            kind="potion_used",
            source_id=potion_id,
            target_id=action.target_id,
            metadata={"potion_slot": slot_id},
        )
    ]

    combat, direct_events = _apply_direct_potion_effects(
        combat,
        potion_id,
        potion_source_id,
        relics=state.relics,
    )
    events.extend(direct_events)

    effects = _potion_effects(potion_id, combat)
    if effects:
        potion_card = CardInstance(
            instance_id=potion_source_id,
            card_id=potion_id,
            name=_potion_name(state, potion_id),
            type=CardType.SKILL,
            cost=0,
            target=_potion_target_type(potion_id),
            effects=effects,
            exhausts=True,
        )
        combat, rng_state, effect_events = _apply_card_effects(
            combat=combat,
            rng_state=rng_state,
            card=potion_card,
            target_id=action.target_id,
            energy_spent=0,
            relics=state.relics,
        )
        events.extend(effect_events)
    elif not direct_events:
        events.append(
            EffectEvent(
                kind="potion_effect_stubbed",
                source_id=potion_id,
                target_id=action.target_id,
                message="Potion use is legal, but this potion effect is not implemented yet.",
            )
        )

    event_tuple = tuple(events)
    combat = combat.model_copy(update={"last_events": event_tuple})
    return _state_after_combat(
        state.model_copy(update={"potions": potions}),
        combat,
        rng_state,
    ), event_tuple


def _apply_direct_potion_effects(
    combat: CombatState,
    potion_id: str,
    source_id: str,
    *,
    relics: Sequence[str] = (),
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    normalized = _normalized_id(potion_id)
    player = combat.player
    hand = combat.hand
    events: list[EffectEvent] = []

    if normalized == "fruit_juice":
        player = player.model_copy(update={"max_hp": player.max_hp + 5, "hp": player.hp + 5})
        events.append(
            EffectEvent(
                kind="player_max_hp_gained",
                source_id=source_id,
                target_id=PLAYER_TARGET_ID,
                amount=5,
            )
        )
    elif normalized == "fortifier":
        gained = player.block * 2
        player = player.model_copy(update={"block": player.block * 3})
        events.append(
            EffectEvent(
                kind="player_block",
                source_id=source_id,
                target_id=PLAYER_TARGET_ID,
                amount=gained,
                metadata={"mode": "triple_block"},
            )
        )
    elif normalized == "blessing_of_the_forge":
        hand = tuple(_upgrade_card_instance(card) for card in hand)
        events.append(
            EffectEvent(
                kind="hand_upgraded",
                source_id=source_id,
                amount=len(hand),
                metadata={"card_instance_ids": [card.instance_id for card in hand]},
            )
        )
    elif normalized == "foul_potion":
        player, event = _damage_player(player, 12, source_id, relics=relics)
        events.append(event)

    return combat.model_copy(update={"player": player, "hand": hand}), tuple(events)


def _potion_effects(potion_id: str, combat: CombatState) -> dict[str, Any]:
    normalized = _normalized_id(potion_id)
    if normalized == "fire_potion":
        return {"damage": 20}
    if normalized == "explosive_ampoule":
        return {"all_damage": 10}
    if normalized == "foul_potion":
        return {"all_damage": 12}
    if normalized == "block_potion":
        return {"block": 12}
    if normalized == "blood_potion":
        return {"heal": max(1, int(combat.player.max_hp * 0.2))}
    if normalized == "energy_potion":
        return {"energy": 2}
    if normalized == "focus_potion":
        return {"apply_status": {"target": "self", "focus": 2}}
    if normalized == "swift_potion":
        return {"draw": 3}
    if normalized == "strength_potion":
        return {"apply_status": {"target": "self", "strength": 2}}
    if normalized == "dexterity_potion":
        return {"apply_status": {"target": "self", "dexterity": 2}}
    if normalized == "liquid_bronze":
        return {"apply_status": {"target": "self", "thorns": 3}}
    if normalized == "flex_potion":
        return {"apply_status": {"target": "self", "strength": 5, "strength_down": 5}}
    if normalized == "speed_potion":
        return {"apply_status": {"target": "self", "dexterity": 5, "dexterity_down": 5}}
    if normalized == "fysh_oil":
        return {"apply_status": {"target": "self", "strength": 1, "dexterity": 1}}
    if normalized == "weak_potion":
        return {"apply_status": {"target": "enemy", "weak": 3}}
    if normalized == "vulnerable_potion":
        return {"apply_status": {"target": "enemy", "vulnerable": 3}}
    if normalized == "poison_potion":
        return {"apply_status": {"target": "enemy", "poison": 6}}
    if normalized == "potion_of_doom":
        return {"apply_status": {"target": "enemy", "doom": 33}}
    if normalized == "potion_of_binding":
        return {"apply_status": {"target": "all_enemies", "weak": 1, "vulnerable": 1}}
    if normalized == "potion_shaped_rock":
        return {"damage": 15}
    if normalized == "essence_of_steel":
        return {"apply_status": {"target": "self", "plated_armor": 4}}
    if normalized == "heart_of_iron":
        return {"apply_status": {"target": "self", "plated_armor": 7}}
    if normalized == "regen_potion":
        return {"apply_status": {"target": "self", "regen": 5}}
    if normalized == "ship_in_a_bottle":
        return {"block": 10, "next_turn": {"block": 10}}
    if normalized == "potion_of_capacity":
        return {"orb_slot_delta": 2}
    if normalized == "essence_of_darkness":
        return {"channel_orb": {"orb": "dark", "amount": max(0, combat.orb_slots)}}
    return {}


def _potion_target_type(potion_id: str) -> TargetType:
    normalized = _normalized_id(potion_id)
    if normalized in {
        "fire_potion",
        "poison_potion",
        "potion_of_doom",
        "potion_shaped_rock",
        "weak_potion",
        "vulnerable_potion",
    }:
        return TargetType.ENEMY
    if normalized in {"explosive_ampoule", "foul_potion", "potion_of_binding"}:
        return TargetType.ALL_ENEMIES
    return TargetType.SELF


def _potion_name(state: RunState, potion_id: str) -> str:
    normalized = _normalized_id(potion_id)
    rows = _source_items(state.flags.get("potions")) or _cached_source_rows(state, "potions")
    for raw_item in rows:
        item = _mapping_from(raw_item)
        item_id = str(item.get("id", item.get("potion_id", "")))
        if _normalized_id(item_id) == normalized:
            return str(item.get("name", potion_id))
    return potion_id


def _discard_potion(
    state: RunState,
    action: Action,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if action.target_id is None:
        return state, ()
    slot_index = _parse_potion_slot_id(action.target_id)
    if slot_index < 0 or slot_index >= len(state.potions):
        return state, ()

    potion_id = state.potions[slot_index]
    potions = tuple(
        potion
        for index, potion in enumerate(state.potions)
        if index != slot_index
    )
    return (
        state.model_copy(update={"potions": potions}),
        (
            EffectEvent(
                kind="potion_discarded",
                target_id=potion_id,
                metadata={
                    "slot_id": action.target_id,
                    "potion_slots": _potion_capacity(state),
                },
            ),
        ),
    )


def _legal_reward_actions(state: RunState) -> tuple[Action, ...]:
    reward = state.reward
    if reward is None:
        return ()

    actions: list[Action] = []
    if reward.gold > 0 and not reward.gold_claimed:
        actions.append(Action(type=ActionType.TAKE_REWARD_GOLD, target_id="reward:gold"))
    actions.extend(
        Action(type=ActionType.TAKE_REWARD_RELIC, target_id=target_id)
        for target_id, _relic_id in _unclaimed_reward_relic_targets(reward)
    )
    if reward.card_options and not reward.card_claimed:
        actions.extend(
            Action(type=ActionType.TAKE_REWARD_CARD, target_id=f"reward:card:{index}")
            for index, _card_id in enumerate(reward.card_options)
        )
    actions.extend(
        Action(type=ActionType.TAKE_REWARD_CARD, target_id=target_id)
        for target_id, _card_id in _unclaimed_reward_card_group_targets(reward)
    )
    actions.extend(
        Action(type=ActionType.TAKE_REWARD_CARD, target_id=target_id)
        for target_id, _card_id in _unclaimed_reward_fixed_card_targets(reward)
    )
    if _has_open_potion_slot(state):
        actions.extend(
            Action(type=ActionType.TAKE_REWARD_POTION, target_id=target_id)
            for target_id, _potion_id in _unclaimed_reward_potion_targets(reward)
        )
    return tuple(actions)


def _reward_has_forced_pending_items(state: RunState) -> bool:
    reward = state.reward
    if reward is None or not reward.forced:
        return False
    return (
        (reward.gold > 0 and not reward.gold_claimed)
        or bool(_unclaimed_reward_relic_targets(reward))
        or bool(_unclaimed_reward_fixed_card_targets(reward))
        or (bool(reward.card_options) and not reward.card_claimed)
        or bool(_unclaimed_reward_card_group_targets(reward))
        or bool(_unclaimed_reward_potion_targets(reward))
    )


def _unclaimed_reward_relic_targets(reward: RewardState) -> tuple[tuple[str, str], ...]:
    targets: list[tuple[str, str]] = []
    if reward.relic_id is not None and not reward.relic_claimed:
        targets.append(("reward:relic", reward.relic_id))
    claimed = set(reward.claimed_relic_ids)
    targets.extend(
        (f"reward:relic:{index}", relic_id)
        for index, relic_id in enumerate(reward.relic_ids)
        if relic_id not in claimed
    )
    return tuple(targets)


def _unclaimed_reward_fixed_card_targets(reward: RewardState) -> tuple[tuple[str, str], ...]:
    claimed = set(reward.claimed_card_indices)
    return tuple(
        (f"reward:fixed_card:{index}", card_id)
        for index, card_id in enumerate(reward.card_ids)
        if index not in claimed
    )


def _unclaimed_reward_card_group_targets(reward: RewardState) -> tuple[tuple[str, str], ...]:
    claimed_groups = set(reward.claimed_card_option_group_indices)
    targets: list[tuple[str, str]] = []
    for group_index, group in enumerate(reward.card_option_groups):
        if group_index in claimed_groups:
            continue
        targets.extend(
            (f"reward:card_group:{group_index}:{card_index}", card_id)
            for card_index, card_id in enumerate(group)
        )
    return tuple(targets)


def _unclaimed_reward_potion_targets(reward: RewardState) -> tuple[tuple[str, str], ...]:
    targets: list[tuple[str, str]] = []
    if reward.potion_id is not None and not reward.potion_claimed:
        targets.append(("reward:potion", reward.potion_id))
    claimed = set(reward.claimed_potion_indices)
    targets.extend(
        (f"reward:potion:{index}", potion_id)
        for index, potion_id in enumerate(reward.potion_ids)
        if index not in claimed
    )
    return tuple(targets)


def _take_reward_gold(
    state: RunState,
    action: Action,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if action.target_id != "reward:gold" or state.reward is None:
        return state, ()
    if state.reward.gold <= 0 or state.reward.gold_claimed:
        return state, ()

    reward = state.reward.model_copy(update={"gold_claimed": True})
    player = state.player.model_copy(update={"gold": state.player.gold + state.reward.gold})
    return (
        state.model_copy(update={"player": player, "reward": reward}),
        (
            EffectEvent(
                kind="reward_gold_taken",
                amount=state.reward.gold,
                metadata={
                    "reward_id": state.reward.reward_id,
                    "reward_source": state.reward.source,
                },
            ),
        ),
    )


def _take_reward_relic(
    state: RunState,
    action: Action,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if action.target_id is None or state.reward is None:
        return state, ()
    relic_target = _reward_relic_for_target_id(state.reward, action.target_id)
    if relic_target is None:
        return state, ()

    relic_id, target_index = relic_target
    if target_index is None:
        reward = state.reward.model_copy(update={"relic_claimed": True})
    else:
        reward = state.reward.model_copy(
            update={
                "claimed_relic_ids": tuple(
                    dict.fromkeys(state.reward.claimed_relic_ids + (relic_id,))
                ),
            }
        )
    next_state = state.model_copy(
        update={
            "relics": state.relics + (relic_id,),
            "reward": reward,
        }
    )
    next_state, pickup_events = _apply_relic_pickup_effects(next_state, relic_id)
    return (
        next_state,
        (
            EffectEvent(
                kind="reward_relic_taken",
                target_id=relic_id,
                metadata={
                    "reward_id": state.reward.reward_id,
                    "reward_source": state.reward.source,
                },
            ),
        )
        + pickup_events,
    )


def _reward_relic_for_target_id(
    reward: RewardState,
    target_id: str,
) -> tuple[str, int | None] | None:
    if target_id == "reward:relic":
        if reward.relic_id is None or reward.relic_claimed:
            return None
        return reward.relic_id, None

    parts = target_id.split(":")
    if len(parts) != 3 or parts[0] != "reward" or parts[1] != "relic":
        return None
    with suppress(ValueError, IndexError):
        index = int(parts[2])
        relic_id = reward.relic_ids[index]
        if relic_id in set(reward.claimed_relic_ids):
            return None
        return relic_id, index
    return None


def _apply_relic_pickup_effects(
    state: RunState,
    relic_id: str,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    normalized = _normalized_id(relic_id)
    if normalized == "old_coin":
        player = state.player.model_copy(update={"gold": state.player.gold + 300})
        return state.model_copy(update={"player": player}), (
            EffectEvent(kind="relic_gold_gained", source_id=relic_id, amount=300),
        )
    if normalized == GOLDEN_COMPASS_RELIC_ID:
        flags = dict(state.flags)
        flags["golden_compass_act2_map"] = True
        next_state = state.model_copy(update={"flags": flags})
        events: list[EffectEvent] = [
            EffectEvent(kind="golden_compass_activated", source_id=relic_id)
        ]
        if state.act == 2:
            compass_map = _generate_golden_compass_map(act=2)
            compass_map, flags, map_events = _mark_spoils_map_site(
                compass_map,
                flags,
                force=True,
            )
            next_state = next_state.model_copy(
                update={
                    "phase": RunPhase.MAP,
                    "floor": 0,
                    "map": compass_map,
                    "flags": flags,
                }
            )
            events.append(EffectEvent(kind="golden_compass_map_replaced", source_id=relic_id))
            events.extend(map_events)
        return next_state, tuple(events)
    return state, ()


def _take_reward_card(
    state: RunState,
    action: Action,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if state.reward is None or action.target_id is None:
        return state, ()

    card_target = _reward_card_for_target_id(state.reward, action.target_id)
    if card_target is None:
        return state, ()

    card_id, fixed_index, group_index = card_target
    card_spec = _reward_card_spec(state, card_id)
    card = _card_from_spec(card_spec, len(state.master_deck) + 1)
    card, card_upgrade_event = _upgrade_card_for_add_relics(state, card)
    if group_index is not None:
        reward = state.reward.model_copy(
            update={
                "claimed_card_option_group_indices": tuple(
                    sorted(
                        set(state.reward.claimed_card_option_group_indices + (group_index,))
                    )
                ),
            }
        )
        fixed = False
    elif fixed_index is None:
        reward = state.reward.model_copy(update={"card_claimed": True})
        fixed = False
    else:
        reward = state.reward.model_copy(
            update={
                "claimed_card_indices": tuple(
                    sorted(set(state.reward.claimed_card_indices + (fixed_index,)))
                ),
            }
        )
        fixed = True
    next_state = state.model_copy(
        update={"master_deck": state.master_deck + (card,), "reward": reward}
    )
    events: list[EffectEvent] = [
        EffectEvent(
            kind="reward_card_taken",
            target_id=card.instance_id,
            metadata={
                "reward_id": state.reward.reward_id,
                "reward_source": state.reward.source,
                "card_id": card.card_id,
                "fixed": fixed,
                "card_group_index": group_index,
            },
        )
    ]
    if card_upgrade_event is not None:
        events.append(card_upgrade_event)
    if _normalized_id(card.card_id) == SPOILS_MAP_CARD_ID:
        next_act = state.act + 1
        if next_act <= _max_acts(state):
            flags = dict(next_state.flags)
            flags["spoils_map_pending_act"] = next_act
            flags["spoils_map_source_act"] = state.act
            next_state = next_state.model_copy(update={"flags": flags})
            events.append(
                EffectEvent(
                    kind="spoils_map_quest_started",
                    target_id=card.instance_id,
                    metadata={"source_act": state.act, "target_act": next_act},
                )
            )
        else:
            events.append(
                EffectEvent(
                    kind="spoils_map_no_next_act",
                    target_id=card.instance_id,
                    metadata={"source_act": state.act},
                )
            )
    return next_state, tuple(events)


def _reward_card_for_target_id(
    reward: RewardState,
    target_id: str,
) -> tuple[str, int | None, int | None] | None:
    parts = target_id.split(":")
    if len(parts) not in {3, 4} or parts[0] != "reward":
        return None
    with suppress(ValueError, IndexError):
        if len(parts) == 3 and parts[1] == "card":
            index = int(parts[2])
            if reward.card_claimed:
                return None
            return reward.card_options[index], None, None
        if len(parts) == 3 and parts[1] == "fixed_card":
            index = int(parts[2])
            if index in set(reward.claimed_card_indices):
                return None
            return reward.card_ids[index], index, None
        if len(parts) == 4 and parts[1] == "card_group":
            group_index = int(parts[2])
            card_index = int(parts[3])
            if group_index in set(reward.claimed_card_option_group_indices):
                return None
            return reward.card_option_groups[group_index][card_index], None, group_index
    return None


def _reward_card_spec(state: RunState, card_id: str) -> dict[str, Any]:
    library = _card_library(state.flags)
    for key in (card_id, card_id.upper(), _normalized_id(card_id)):
        if key in library:
            card_spec = dict(library[key])
            break
    else:
        card_spec = {"card_id": card_id}
    card_spec["card_id"] = str(card_spec.get("card_id", card_spec.get("id", card_id)))
    card_spec.pop("instance_id", None)
    return card_spec


def _take_reward_potion(
    state: RunState,
    action: Action,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if action.target_id is None or state.reward is None:
        return state, ()
    potion_target = _reward_potion_for_target_id(state.reward, action.target_id)
    if potion_target is None:
        return state, ()
    if not _has_open_potion_slot(state):
        return state, ()

    potion_id, target_index = potion_target
    if target_index is None:
        reward = state.reward.model_copy(update={"potion_claimed": True})
    else:
        reward = state.reward.model_copy(
            update={
                "claimed_potion_indices": tuple(
                    sorted(set(state.reward.claimed_potion_indices + (target_index,)))
                ),
            }
        )
    potions = state.potions + (potion_id,)
    return (
        state.model_copy(update={"potions": potions, "reward": reward}),
        (
            EffectEvent(
                kind="reward_potion_taken",
                target_id=potion_id,
                metadata={
                    "reward_id": state.reward.reward_id,
                    "reward_source": state.reward.source,
                    "potion_slots": _potion_capacity(state),
                },
            ),
        ),
    )


def _reward_potion_for_target_id(
    reward: RewardState,
    target_id: str,
) -> tuple[str, int | None] | None:
    if target_id == "reward:potion":
        if reward.potion_id is None or reward.potion_claimed:
            return None
        return reward.potion_id, None

    parts = target_id.split(":")
    if len(parts) != 3 or parts[0] != "reward" or parts[1] != "potion":
        return None
    with suppress(ValueError, IndexError):
        index = int(parts[2])
        if index in set(reward.claimed_potion_indices):
            return None
        return reward.potion_ids[index], index
    return None


def _leave_shop(state: RunState) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if state.shop is not None:
        resolve_shop_action(
            ShopRoomChoice(ShopRoomAction.LEAVE),
            _shop_room_state(state),
        )
    return _complete_current_room(
        state.model_copy(update={"shop": None}),
        (EffectEvent(kind="shop_left"),),
    )


def _throw_potion_at_merchant(state: RunState) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if "foul_potion" not in state.potions:
        return state, ()
    potions = list(state.potions)
    potions.remove("foul_potion")
    player = state.player.model_copy(update={"gold": state.player.gold + 100})
    return (
        state.model_copy(update={"player": player, "potions": tuple(potions)}),
        (
            EffectEvent(
                kind="foul_potion_thrown_at_merchant",
                source_id="foul_potion",
                target_id="merchant",
                amount=100,
            ),
        ),
    )


def _apply_shop_entry_relics(
    player: PlayerState,
    state: RunState,
) -> tuple[PlayerState, tuple[EffectEvent, ...]]:
    events: list[EffectEvent] = []
    if _has_relic(state, "meal_ticket"):
        old_hp = player.hp
        player = player.model_copy(update={"hp": min(player.max_hp, player.hp + 15)})
        events.append(EffectEvent(kind="meal_ticket_healed", amount=player.hp - old_hp))
    return player, tuple(events)


def _apply_lords_parasol(
    state: RunState,
    shop: ShopState,
) -> tuple[RunState | None, tuple[EffectEvent, ...]]:
    if not _has_relic(state, "lords_parasol", "lord's_parasol", "lord_parasol"):
        return None, ()

    update, gained_ids = _shop_gain_all_non_service_items(state, shop)
    update["shop"] = shop.model_copy(
        update={
            "items": tuple(
                item.model_copy(update={"purchased": item.kind != ShopItemKind.CARD_REMOVAL.value})
                for item in shop.items
            )
        }
    )
    return (
        state.model_copy(update=update),
        (
            EffectEvent(
                kind="lords_parasol_claimed_shop",
                amount=len(gained_ids),
                metadata={"item_ids": gained_ids},
            ),
        ),
    )


def _shop_gain_all_non_service_items(
    state: RunState,
    shop: ShopState,
) -> tuple[dict[str, Any], list[str]]:
    deck = list(state.master_deck)
    relics = list(state.relics)
    potions = list(state.potions)
    gained_ids: list[str] = []
    for index, item in enumerate(shop.items):
        if item.purchased or item.kind == ShopItemKind.CARD_REMOVAL.value:
            continue
        kind = ShopItemKind(item.kind)
        if kind in {ShopItemKind.CARD, ShopItemKind.COLORLESS_CARD}:
            gained_ids.append(item.item_id)
            deck.append(
                _card_from_spec(
                    {
                        "instance_id": f"shop_free_{len(state.replay_log)}_{index}",
                        "card_id": item.item_id,
                        "name": item.item_id,
                    },
                    len(deck) + 1,
                )
            )
        elif kind is ShopItemKind.RELIC:
            gained_ids.append(item.item_id)
            relics.append(item.item_id)
        elif kind is ShopItemKind.POTION:
            if len(potions) < _potion_capacity(state, relics=relics):
                gained_ids.append(item.item_id)
                potions.append(item.item_id)
    return {
        "master_deck": tuple(deck),
        "relics": tuple(relics),
        "potions": tuple(potions),
    }, gained_ids


def _shop_state_from_inventory(
    node_id: str,
    inventory: ShopInventory,
    *,
    card_removals_bought: int = 0,
) -> ShopState:
    return ShopState(
        node_id=node_id,
        card_removals_bought=card_removals_bought,
        items=tuple(
            _shop_item_state_from_priced(index, item)
            for index, item in enumerate(inventory.items)
        ),
    )


def _shop_item_state_from_priced(item_index: int, item: PricedShopItem) -> ShopItemState:
    return ShopItemState(
        slot_id=_shop_slot_id(item_index),
        item_id=item.item.item_id,
        kind=item.item.kind.value,
        rarity=item.item.rarity.value if item.item.rarity is not None else None,
        price=item.price,
        base_price=item.base_price,
    )


def _apply_shop_price_modifiers(state: RunState, shop: ShopState) -> ShopState:
    return shop.model_copy(
        update={
            "items": tuple(
                item.model_copy(update={"price": _modified_shop_price(state, item)})
                for item in shop.items
            )
        }
    )


def _modified_shop_price(state: RunState, item: ShopItemState) -> int:
    if item.kind == ShopItemKind.CARD_REMOVAL.value and _has_relic(state, "smiling_mask"):
        return 50

    multiplier = 1.0
    if _has_relic(state, "membership_card"):
        multiplier *= 0.5
    if _has_relic(state, "the_courier", "courier"):
        multiplier *= 0.8
    return max(0, int(item.base_price * multiplier))


def _shop_room_state(state: RunState) -> ShopRoomState:
    if state.shop is None:
        return ShopRoomState(
            gold=state.player.gold,
            inventory=ShopInventory((), False, source=STS1_COMPAT_SOURCE),
        )
    return ShopRoomState(
        gold=state.player.gold,
        inventory=_shop_inventory_from_state(state.shop),
        removable_card_ids=frozenset(card.instance_id for card in state.master_deck),
        purchased_item_indices=frozenset(
            index for index, item in enumerate(state.shop.items) if item.purchased
        ),
        card_removals_bought=state.shop.card_removals_bought,
    )


def _shop_inventory_from_state(shop: ShopState) -> ShopInventory:
    return ShopInventory(
        items=tuple(
            PricedShopItem(
                item=ShopItem(
                    item_id=item.item_id,
                    kind=ShopItemKind(item.kind),
                    rarity=_shop_rarity(ShopItemKind(item.kind), item.rarity),
                ),
                price=item.price,
                base_price=item.base_price,
                source=STS1_COMPAT_SOURCE,
            )
            for item in shop.items
        ),
        includes_card_removal=any(
            item.kind == ShopItemKind.CARD_REMOVAL.value for item in shop.items
        ),
        source=STS1_COMPAT_SOURCE,
    )


def _shop_after_purchase(
    state: RunState,
    item_index: int,
    kind: ShopItemKind,
) -> ShopState | None:
    if _has_relic(state, "the_courier", "courier") and kind is not ShopItemKind.CARD_REMOVAL:
        return state.shop
    return _shop_mark_purchased(state.shop, item_index)


def _shop_mark_purchased(shop: ShopState | None, item_index: int) -> ShopState | None:
    if shop is None:
        return None
    return shop.model_copy(
        update={
            "items": tuple(
                item.model_copy(update={"purchased": True})
                if index == item_index
                else item
                for index, item in enumerate(shop.items)
            ),
            "card_removals_bought": shop.card_removals_bought + (
                1
                if shop.items[item_index].kind == ShopItemKind.CARD_REMOVAL.value
                else 0
            ),
        }
    )


def _restock_shop_slot(
    state: RunState,
    shop: ShopState | None,
    item_index: int,
    kind: ShopItemKind,
    rng: Random,
) -> tuple[ShopState | None, str | None]:
    if shop is None:
        return None, None
    existing_item_ids = {
        item.item_id
        for index, item in enumerate(shop.items)
        if index != item_index and not item.purchased
    }
    inventory = build_basic_shop_inventory(
        rng,
        card_pool=_shop_restock_pool(
            _shop_item_pool(state, "shop_card_pool", _default_shop_card_pool()),
            existing_item_ids,
        ),
        colorless_card_pool=_shop_restock_pool(
            _shop_item_pool(
                state,
                "shop_colorless_card_pool",
                _default_shop_colorless_pool(),
                default_kind=ShopItemKind.COLORLESS_CARD,
            ),
            existing_item_ids,
        ),
        relic_pool=_shop_restock_pool(
            _shop_item_pool(
                state,
                "shop_relic_pool",
                _default_shop_relic_pool(),
                default_kind=ShopItemKind.RELIC,
            ),
            existing_item_ids,
        ),
        potion_pool=_shop_restock_pool(
            _shop_item_pool(
                state,
                "shop_potion_pool",
                _default_shop_potion_pool(),
                default_kind=ShopItemKind.POTION,
            ),
            existing_item_ids,
        ),
        plan=ShopInventoryPlan(
            colored_cards=1 if kind is ShopItemKind.CARD else 0,
            colorless_cards=1 if kind is ShopItemKind.COLORLESS_CARD else 0,
            relics=1 if kind is ShopItemKind.RELIC else 0,
            potions=1 if kind is ShopItemKind.POTION else 0,
            include_card_removal=False,
        ),
        ascension_level=state.ascension,
        rare_offset_percent=_flag_float(state, "card_rare_offset_percent", 0.0),
        card_removals_bought=_flag_int(state, "shop_card_removals_bought", 0),
    )
    if not inventory.items:
        return _shop_mark_purchased(shop, item_index), None
    restocked = _shop_item_state_from_priced(item_index, inventory.items[0])
    restocked = restocked.model_copy(update={"price": _modified_shop_price(state, restocked)})
    return (
        shop.model_copy(
            update={
                "items": tuple(
                    restocked if index == item_index else item
                    for index, item in enumerate(shop.items)
                )
            }
        ),
        restocked.item_id,
    )


def _shop_restock_pool(
    pool: Sequence[ShopItem],
    existing_item_ids: set[str],
) -> tuple[ShopItem, ...]:
    return tuple(item for item in pool if item.item_id not in existing_item_ids)


def _shop_slot_id(item_index: int) -> str:
    return f"shop:{item_index}"


def _shop_choice_target_id(choice: ShopRoomChoice) -> str:
    if choice.item_index is None:
        raise ValueError("Shop buy choices require item_index.")
    slot_id = _shop_slot_id(choice.item_index)
    if choice.target_card_id is None:
        return slot_id
    return f"{slot_id}:remove:{choice.target_card_id}"


def _parse_shop_target_id(target_id: str) -> tuple[int, str | None]:
    parts = target_id.split(":")
    if len(parts) < 2 or parts[0] != "shop":
        raise ValueError(f"Invalid shop target id: {target_id}")
    item_index = int(parts[1])
    if len(parts) == 2:
        return item_index, None
    if len(parts) == 4 and parts[2] == "remove":
        return item_index, parts[3]
    raise ValueError(f"Invalid shop target id: {target_id}")


def _shop_inventory_plan(state: RunState) -> ShopInventoryPlan:
    raw_plan = state.flags.get("shop_plan")
    plan = _mapping_from(raw_plan)
    return ShopInventoryPlan(
        colored_cards=_int_from_mapping(plan, "colored_cards", 5),
        colorless_cards=_int_from_mapping(plan, "colorless_cards", 2),
        relics=_int_from_mapping(plan, "relics", 3),
        potions=_int_from_mapping(plan, "potions", 3),
        include_card_removal=bool(plan.get("include_card_removal", True)),
    )


def _shop_item_pool(
    state: RunState,
    flag_key: str,
    default_pool: Sequence[ShopItem],
    *,
    default_kind: ShopItemKind = ShopItemKind.CARD,
) -> tuple[ShopItem, ...]:
    raw_pool = state.flags.get(flag_key)
    if raw_pool is None:
        return tuple(default_pool)
    if not isinstance(raw_pool, Sequence) or isinstance(raw_pool, (str, bytes, bytearray)):
        return tuple(default_pool)

    items: list[ShopItem] = []
    for raw_item in raw_pool:
        item = _shop_item_from_raw(raw_item, default_kind=default_kind)
        if item is not None:
            items.append(item)
    return tuple(items) or tuple(default_pool)


def _shop_item_from_raw(raw_item: Any, *, default_kind: ShopItemKind) -> ShopItem | None:
    if isinstance(raw_item, str):
        return ShopItem(raw_item, default_kind, _default_shop_rarity(default_kind))
    item = _mapping_from(raw_item)
    if not item:
        return None

    kind = ShopItemKind(str(item.get("kind", default_kind.value)))
    item_id = str(
        item.get(
            "item_id",
            item.get("card_id", item.get("relic_id", item.get("potion_id", item.get("id", "")))),
        )
    )
    if not item_id:
        return None
    price = item.get("price")
    return ShopItem(
        item_id,
        kind,
        _shop_rarity(kind, item.get("rarity")),
        int(price) if price is not None else None,
        card_type=str(item.get("card_type", item.get("type"))).lower()
        if kind is ShopItemKind.CARD and item.get("card_type", item.get("type")) is not None
        else None,
    )


def _shop_rarity(kind: ShopItemKind, rarity: Any) -> Any:
    if rarity is None:
        return _default_shop_rarity(kind)
    if kind in {ShopItemKind.CARD, ShopItemKind.COLORLESS_CARD}:
        return CardRarity(str(rarity))
    if kind is ShopItemKind.POTION:
        return PotionRarity(str(rarity))
    if kind is ShopItemKind.RELIC:
        return RelicRarity(str(rarity))
    return None


def _default_shop_rarity(kind: ShopItemKind) -> Any:
    if kind in {ShopItemKind.CARD, ShopItemKind.COLORLESS_CARD}:
        return CardRarity.COMMON
    if kind is ShopItemKind.POTION:
        return PotionRarity.COMMON
    if kind is ShopItemKind.RELIC:
        return RelicRarity.COMMON
    return None


def _default_shop_card_pool() -> tuple[ShopItem, ...]:
    return (
        ShopItem("pommel_strike", ShopItemKind.CARD, CardRarity.COMMON, card_type="attack"),
        ShopItem("cleave", ShopItemKind.CARD, CardRarity.COMMON, card_type="attack"),
        ShopItem("uppercut", ShopItemKind.CARD, CardRarity.UNCOMMON, card_type="attack"),
        ShopItem("feed", ShopItemKind.CARD, CardRarity.RARE, card_type="attack"),
        ShopItem("shrug_it_off", ShopItemKind.CARD, CardRarity.COMMON, card_type="skill"),
        ShopItem("true_grit", ShopItemKind.CARD, CardRarity.COMMON, card_type="skill"),
        ShopItem("flame_barrier", ShopItemKind.CARD, CardRarity.UNCOMMON, card_type="skill"),
        ShopItem("impervious", ShopItemKind.CARD, CardRarity.RARE, card_type="skill"),
        ShopItem("inflame", ShopItemKind.CARD, CardRarity.UNCOMMON, card_type="power"),
        ShopItem("demon_form", ShopItemKind.CARD, CardRarity.RARE, card_type="power"),
    )


def _default_shop_colorless_pool() -> tuple[ShopItem, ...]:
    return (
        ShopItem("flash_of_steel", ShopItemKind.COLORLESS_CARD, CardRarity.UNCOMMON),
        ShopItem("trip", ShopItemKind.COLORLESS_CARD, CardRarity.UNCOMMON),
        ShopItem("apotheosis", ShopItemKind.COLORLESS_CARD, CardRarity.RARE),
        ShopItem("master_of_strategy", ShopItemKind.COLORLESS_CARD, CardRarity.RARE),
    )


def _default_shop_relic_pool() -> tuple[ShopItem, ...]:
    return (
        ShopItem("anchor", ShopItemKind.RELIC, RelicRarity.COMMON),
        ShopItem("kunai", ShopItemKind.RELIC, RelicRarity.UNCOMMON),
        ShopItem("shovel", ShopItemKind.RELIC, RelicRarity.RARE),
        ShopItem("cauldron", ShopItemKind.RELIC, RelicRarity.SHOP),
        ShopItem("chemical_x", ShopItemKind.RELIC, RelicRarity.SHOP),
        ShopItem("frozen_eye", ShopItemKind.RELIC, RelicRarity.SHOP),
        ShopItem("medical_kit", ShopItemKind.RELIC, RelicRarity.SHOP),
        ShopItem("membership_card", ShopItemKind.RELIC, RelicRarity.SHOP),
        ShopItem("orange_pellets", ShopItemKind.RELIC, RelicRarity.SHOP),
        ShopItem("prismatic_shard", ShopItemKind.RELIC, RelicRarity.SHOP),
    )


def _default_shop_potion_pool() -> tuple[ShopItem, ...]:
    return (
        ShopItem("fire_potion", ShopItemKind.POTION, PotionRarity.COMMON),
        ShopItem("skill_potion", ShopItemKind.POTION, PotionRarity.UNCOMMON),
        ShopItem("essence_of_steel", ShopItemKind.POTION, PotionRarity.RARE),
    )


def _complete_current_room(
    state: RunState, events: tuple[EffectEvent, ...]
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if state.map is None or state.map.current_node_id is None:
        return state, events

    node = state.map.node_by_id[state.map.current_node_id]
    completed = tuple(dict.fromkeys(state.map.completed_node_ids + (node.node_id,)))
    room_history = state.room_history + (node.node_id,)
    map_state = state.map.model_copy(update={"completed_node_ids": completed})
    completion_event = EffectEvent(
        kind="room_completed",
        target_id=node.node_id,
        metadata={"act": node.act, "floor": node.floor, "room_kind": node.kind.value},
    )

    if node.kind == RoomKind.BOSS:
        if state.act >= _max_acts(state):
            return (
                state.model_copy(
                    update={
                        "phase": RunPhase.COMPLETE,
                        "map": map_state,
                        "combat": None,
                        "reward": None,
                        "shop": None,
                        "room_history": room_history,
                    }
                ),
                events + (completion_event, EffectEvent(kind="run_completed")),
            )
        next_act = state.act + 1
        rng = random_from_state(state.rng)
        next_map = _generate_act_map(act=next_act, rng=rng, source=state.flags)
        flags = dict(state.flags)
        next_map, flags, map_events = _mark_spoils_map_site(next_map, flags)
        player = _apply_ancient_heal(state.player, state.ascension)
        ancient_state = _generate_ancient_state(act=next_act, rng=rng)
        return (
        state.model_copy(
            update={
                "phase": RunPhase.ANCIENT,
                "act": next_act,
                "floor": 0,
                "player": player,
                "map": next_map,
                "ancient": ancient_state,
                "combat": None,
                "event": None,
                "reward": None,
                "shop": None,
                "rng": capture_random_state(rng),
                "room_history": room_history,
                "flags": flags,
                }
            ),
            events
            + (
                completion_event,
                EffectEvent(kind="act_advanced", amount=next_act),
                *map_events,
                EffectEvent(kind="ancient_ready", source_id=ancient_state.ancient_id),
            ),
        )

    return (
        state.model_copy(
            update={
                "phase": RunPhase.MAP,
                "map": map_state,
                "combat": None,
                "event": None,
                "reward": None,
                "shop": None,
                "room_history": room_history,
            }
        ),
        events + (completion_event,),
    )


def _start_combat_for_node(
    state: RunState, node: MapNodeState
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    rng = random_from_state(state.rng)
    draw_pile = list(state.master_deck)
    rng.shuffle(draw_pile)
    combat = CombatState(
        player=state.player.model_copy(update={"block": 0, "energy": state.player.max_energy}),
        monsters=_monsters_for_node(state, node, rng),
        orb_slots=_base_orb_slots(state),
        draw_pile=tuple(draw_pile),
        draw_per_turn=_flag_int(state, "draw_per_turn", 5),
        metadata={"relic_counters": _relic_counters_from_flags(state)},
    )
    start_relics = _resolve_combat_relic_trigger(
        GameTrigger.COMBAT_START,
        state.relics,
        player_hp=combat.player.hp,
        player_max_hp=combat.player.max_hp,
        encounter_type=node.kind.value,
        relic_counters=_combat_relic_counters(combat),
    )
    combat, start_relic_events = _apply_combat_relic_resolution(combat, start_relics)
    turn_relics = _resolve_combat_relic_trigger(
        GameTrigger.TURN_START,
        state.relics,
        turn_number=1,
        player_hp=combat.player.hp,
        player_max_hp=combat.player.max_hp,
        player_block=combat.player.block,
        encounter_type=node.kind.value,
        player_statuses=combat.player.statuses,
        relic_counters=_combat_relic_counters_for(state.relics, combat),
    )
    combat, turn_relic_events = _apply_combat_relic_resolution(combat, turn_relics)
    combat, bonus_draw, bonus_draw_events = _pop_pending_relic_draw(combat)
    rng_state = capture_random_state(rng)
    combat, rng_state, draw_events = _draw_cards(
        combat,
        rng_state,
        combat.draw_per_turn + bonus_draw,
    )
    combat = combat.model_copy(
        update={
            "last_events": start_relic_events
            + turn_relic_events
            + bonus_draw_events
            + draw_events
        }
    )
    return combat, rng_state, (
        EffectEvent(
            kind="combat_started",
            target_id=node.node_id,
            metadata={"room_kind": node.kind.value, "act": node.act, "floor": node.floor},
        ),
    ) + start_relic_events + turn_relic_events + bonus_draw_events + draw_events


def _resolve_combat_relic_trigger(
    trigger: GameTrigger,
    relics: Sequence[str],
    **context_values: Any,
) -> CombatRelicResolution:
    resolution = resolve_game_trigger(
        trigger,
        relics=relics,
        context=TriggerContext(trigger, **context_values),
    )
    combat_resolution = resolution.combat_relic_resolution
    if combat_resolution is None:
        raise ValueError(f"Trigger {trigger.value!r} does not have a combat relic adapter.")
    return combat_resolution


def _pop_pending_relic_draw(
    combat: CombatState,
) -> tuple[CombatState, int, tuple[EffectEvent, ...]]:
    amount = max(0, _metadata_int(combat.metadata, "pending_relic_draw", 0))
    if amount <= 0:
        return combat, 0, ()

    metadata = dict(combat.metadata)
    metadata.pop("pending_relic_draw", None)
    return (
        combat.model_copy(update={"metadata": metadata}),
        amount,
        (
            EffectEvent(
                kind="relic_bonus_draw_applied",
                amount=amount,
                metadata={"source": "pending_relic_draw"},
            ),
        ),
    )


def _base_orb_slots(state: RunState) -> int:
    if "orb_slots" in state.flags:
        return min(10, max(0, _flag_int(state, "orb_slots", 0)))
    if _normalized_id(state.character_id) == "defect" or _has_relic(state, "cracked_core"):
        return 3
    return 0


def _coerce_action(action: ActionInput) -> Action:
    if isinstance(action, Action):
        return action
    if isinstance(action, Mapping):
        return Action.model_validate(dict(action))
    raise TypeError(f"Unsupported action payload type: {type(action)!r}")


def _normalize_action(state: RunState, action: Action) -> Action:
    if action.type == ActionType.USE_POTION:
        return _normalize_potion_action(state, action)
    if action.type in {
        ActionType.CHOOSE_ANCIENT,
        ActionType.CHOOSE_NODE,
        ActionType.CHOOSE_EVENT,
        ActionType.TAKE_REWARD_GOLD,
        ActionType.TAKE_REWARD_RELIC,
        ActionType.TAKE_REWARD_CARD,
        ActionType.TAKE_REWARD_POTION,
        *_CAMPFIRE_ACTION_TYPES,
        ActionType.SHOP_BUY,
        ActionType.SHOP_LEAVE,
        ActionType.DISCARD_POTION,
        ActionType.DISCARD_CARD,
        ActionType.THROW_POTION_AT_MERCHANT,
        ActionType.PROCEED,
    }:
        return action
    if action.type != ActionType.PLAY_CARD or state.combat is None:
        return action

    card = _find_card(state.combat.hand, action.card_instance_id)
    if card is None:
        return action

    target_id = action.target_id
    if card.target == TargetType.SELF and target_id is None:
        target_id = PLAYER_TARGET_ID
    if card.target == TargetType.ENEMY and target_id is None:
        alive = _alive_monsters(state.combat)
        if len(alive) == 1:
            target_id = alive[0].monster_id
    return action.model_copy(update={"target_id": target_id})


def _normalize_potion_action(state: RunState, action: Action) -> Action:
    if state.combat is None:
        return action
    potion_id = _action_potion_id(state, action)
    if potion_id is None:
        return action

    target_id = action.target_id
    target = _potion_target_type(potion_id)
    if target == TargetType.SELF and target_id is None:
        target_id = PLAYER_TARGET_ID
    elif target == TargetType.ENEMY and target_id is None:
        alive = _alive_monsters(state.combat)
        if len(alive) == 1:
            target_id = alive[0].monster_id
    return action.model_copy(update={"target_id": target_id})


def _action_potion_id(state: RunState, action: Action) -> str | None:
    slot_id = action.payload.get("potion_slot")
    if not isinstance(slot_id, str):
        return None
    with suppress(ValueError, IndexError):
        return state.potions[_parse_potion_slot_id(slot_id)]
    return None


def _action_is_legal(action: Action, legal_actions_: Sequence[Action]) -> bool:
    for legal in legal_actions_:
        if legal.type != action.type:
            continue
        if legal.type in {
            ActionType.END_TURN,
            ActionType.PROCEED,
            ActionType.SHOP_LEAVE,
            ActionType.THROW_POTION_AT_MERCHANT,
        }:
            return True
        if (
            legal.type == ActionType.REST
            and action.card_instance_id is None
            and action.target_id is None
        ):
            return True
        if legal.type in {
            ActionType.CHOOSE_ANCIENT,
            ActionType.CHOOSE_NODE,
            ActionType.TAKE_REWARD_GOLD,
            ActionType.TAKE_REWARD_RELIC,
            ActionType.TAKE_REWARD_CARD,
            ActionType.TAKE_REWARD_POTION,
            ActionType.SHOP_BUY,
            ActionType.DISCARD_POTION,
        } and legal.target_id == action.target_id:
            return True
        if (
            legal.type == ActionType.DISCARD_CARD
            and legal.card_instance_id == action.card_instance_id
        ):
            return True
        if legal.type == ActionType.CHOOSE_EVENT and legal.target_id == action.target_id:
            legal_payload_ids = _payload_card_instance_ids(legal)
            action_payload_ids = _payload_card_instance_ids(action)
            if legal_payload_ids or action_payload_ids:
                if legal_payload_ids == action_payload_ids:
                    return True
                continue
            if legal.card_instance_id == action.card_instance_id:
                return True
            continue
        if legal.type == ActionType.USE_POTION and legal.target_id == action.target_id:
            return legal.payload.get("potion_slot") == action.payload.get("potion_slot")
        if (
            legal.type not in {ActionType.CHOOSE_EVENT, ActionType.USE_POTION}
            and legal.card_instance_id == action.card_instance_id
            and legal.target_id == action.target_id
        ):
            return True
    return False


def _payload_card_instance_ids(action: Action) -> tuple[str, ...]:
    payload_ids = action.payload.get("card_instance_ids")
    if isinstance(payload_ids, Sequence) and not isinstance(
        payload_ids,
        (str, bytes, bytearray),
    ):
        return tuple(str(card_id) for card_id in payload_ids)
    return ()


def _event_action_card_instance_ids(action: Action) -> tuple[str, ...]:
    payload_ids = _payload_card_instance_ids(action)
    if payload_ids:
        return payload_ids
    if action.card_instance_id is not None:
        return (action.card_instance_id,)
    return ()


def _discard_card_from_hand(
    state: RunState,
    action: Action,
) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if state.combat is None or action.card_instance_id is None:
        return state, ()

    combat = state.combat
    discard_choice = _pending_discard_choice(combat)
    if discard_choice is None:
        return state, ()

    discarded = _find_card(combat.hand, action.card_instance_id)
    if discarded is None:
        return state, ()

    remaining = max(0, discard_choice["remaining"] - 1)
    hand = tuple(card for card in combat.hand if card.instance_id != discarded.instance_id)
    metadata = dict(combat.metadata)
    if remaining > 0 and hand:
        metadata["pending_card_choice"] = {
            **discard_choice,
            "remaining": remaining,
        }
    else:
        metadata.pop("pending_card_choice", None)

    event = EffectEvent(
        kind="card_discarded_by_choice",
        source_id=discard_choice.get("source_card_instance_id"),
        target_id=discarded.instance_id,
        amount=1,
        metadata={
            "card_id": discarded.card_id,
            "remaining": remaining if hand else 0,
        },
    )
    movement_events = _card_movement_events(
        GameTrigger.CARD_DISCARDED,
        (discarded,),
        from_pile="hand",
        to_pile="discard_pile",
        source_id=discard_choice.get("source_card_instance_id"),
        reason="chosen_discard",
        metadata={"remaining": remaining if hand else 0},
    )
    combat = combat.model_copy(
        update={
            "hand": hand,
            "discard_pile": combat.discard_pile + (discarded,),
            "metadata": metadata,
            "last_events": (event,) + movement_events,
        }
    )
    return (
        state.model_copy(update={"combat": combat, "player": combat.player}),
        (event,) + movement_events,
    )


def _play_card(state: RunState, action: Action) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if state.combat is None or action.card_instance_id is None:
        return state, ()

    combat = state.combat
    card = _find_card(combat.hand, action.card_instance_id)
    if card is None:
        return state, ()

    hand = tuple(item for item in combat.hand if item.instance_id != card.instance_id)
    energy_spent = _energy_cost(card, combat.player.energy)
    player, resource_events = _pay_card_resource_costs(
        combat.player.model_copy(update={"energy": combat.player.energy - energy_spent}),
        card,
    )
    combat = combat.model_copy(update={"hand": hand, "player": player})

    combat, rng_state, effect_events = _apply_card_effects(
        combat=combat,
        rng_state=state.rng,
        card=card,
        target_id=action.target_id,
        energy_spent=energy_spent,
        relics=state.relics,
    )
    combat, vigor_events = _consume_vigor_after_attack(combat, card)

    destination = _card_destination(card)
    combat, destination_events = _move_played_card_to_destination(combat, card, destination)

    events = (
        EffectEvent(
            kind="card_played",
            source_id=card.instance_id,
            target_id=action.target_id,
            amount=energy_spent,
            metadata={"card_id": card.card_id, "destination": destination},
        ),
    ) + destination_events + resource_events + effect_events + vigor_events

    combat, hook_events = _apply_after_card_play_hooks(combat, card, state.relics)
    events += hook_events

    combat = combat.model_copy(
        update={
            "cards_played_this_turn": combat.cards_played_this_turn + (card.instance_id,),
            "last_events": events,
        }
    )

    return _state_after_combat(state, combat, rng_state), events


def _end_turn(state: RunState) -> tuple[RunState, tuple[EffectEvent, ...]]:
    if state.combat is None:
        return state, ()

    combat = state.combat
    rng = random_from_state(state.rng)
    events: list[EffectEvent] = []
    player = combat.player
    retained_hand: tuple[CardInstance, ...]
    discarded_hand: tuple[CardInstance, ...]
    if _status_amount(player.statuses, "retain_hand") > 0:
        retained_hand = combat.hand
        discarded_hand = ()
        events.append(
            EffectEvent(
                kind="hand_retained",
                amount=len(retained_hand),
                metadata={"card_instance_ids": [card.instance_id for card in retained_hand]},
            )
        )
    else:
        retained_hand, discarded_hand = _split_retained_end_turn_cards(combat.hand)
        if retained_hand:
            events.append(
                EffectEvent(
                    kind="cards_retained",
                    amount=len(retained_hand),
                    metadata={
                        "card_instance_ids": [card.instance_id for card in retained_hand],
                        "card_ids": [card.card_id for card in retained_hand],
                    },
                )
            )
    exhaust_hand = tuple(card for card in discarded_hand if bool(card.custom.get("ethereal")))
    exhausted_ids = {card.instance_id for card in exhaust_hand}
    discard_hand = tuple(card for card in discarded_hand if card.instance_id not in exhausted_ids)

    if discard_hand:
        events.append(
            EffectEvent(
                kind="hand_discarded",
                amount=len(discard_hand),
                metadata={"card_instance_ids": [card.instance_id for card in discard_hand]},
            )
        )
        events.extend(
            _card_movement_events(
                GameTrigger.CARD_DISCARDED,
                discard_hand,
                from_pile="hand",
                to_pile="discard_pile",
                reason="end_turn_discard",
            )
        )
    if exhaust_hand:
        events.append(
            EffectEvent(
                kind="ethereal_cards_exhausted",
                amount=len(exhaust_hand),
                metadata={"card_instance_ids": [card.instance_id for card in exhaust_hand]},
            )
        )
        events.extend(
            _card_movement_events(
                GameTrigger.CARD_EXHAUSTED,
                exhaust_hand,
                from_pile="hand",
                to_pile="exhaust_pile",
                reason="ethereal_end_turn",
            )
        )
    combat = combat.model_copy(
        update={
            "discard_pile": combat.discard_pile + discard_hand,
            "exhaust_pile": combat.exhaust_pile + exhaust_hand,
            "hand": retained_hand,
        }
    )

    turn_end_relics = _resolve_combat_relic_trigger(
        GameTrigger.TURN_END,
        state.relics,
        player_hp=combat.player.hp,
        player_max_hp=combat.player.max_hp,
        player_block=combat.player.block,
        encounter_type=_current_encounter_type(state),
        player_statuses=combat.player.statuses,
        relic_counters=_combat_relic_counters_for(state.relics, combat),
    )
    combat, turn_end_relic_events = _apply_combat_relic_resolution(combat, turn_end_relics)
    events.extend(turn_end_relic_events)
    player = combat.player

    player, status_block_events = _apply_player_end_turn_status_block(player)
    events.extend(status_block_events)
    combat = combat.model_copy(update={"player": player})
    combat, passive_rng_state, orb_passive_events = _apply_orb_passives(
        combat,
        capture_random_state(rng),
        relics=state.relics,
    )
    rng = random_from_state(passive_rng_state)
    events.extend(orb_passive_events)
    player = combat.player

    monster_definitions = _monster_definitions(state)
    monsters: list[MonsterState] = []
    for monster in combat.monsters:
        if monster.hp <= 0:
            monsters.append(monster)
            continue
        monster, poison_events = _apply_monster_poison(monster)
        events.extend(poison_events)
        if monster.hp <= 0:
            monsters.append(monster)
            continue
        monster, player, monster_events = _execute_monster_turn(
            state=state,
            monster=monster,
            player=player,
            monster_definitions=monster_definitions,
            rng=rng,
        )
        events.extend(monster_events)
        monsters.append(monster)

    player = player.model_copy(
        update={
            "block": 0,
            "energy": player.max_energy,
        }
    )
    combat = combat.model_copy(
        update={
            "turn": combat.turn + 1,
            "player": player,
            "monsters": tuple(monsters),
            "cards_played_this_turn": (),
            "metadata": _reset_turn_combat_metadata(combat.metadata),
        }
    )
    turn_start_relics = _resolve_combat_relic_trigger(
        GameTrigger.TURN_START,
        state.relics,
        turn_number=combat.turn,
        player_hp=combat.player.hp,
        player_max_hp=combat.player.max_hp,
        player_block=combat.player.block,
        encounter_type=_current_encounter_type(state),
        player_statuses=combat.player.statuses,
        relic_counters=_combat_relic_counters_for(state.relics, combat),
    )
    combat, turn_start_relic_events = _apply_combat_relic_resolution(combat, turn_start_relics)
    events.extend(turn_start_relic_events)
    combat, turn_start_events, extra_draw = _apply_player_turn_start_effects(combat)
    events.extend(turn_start_events)
    combat, tick_events = _tick_end_turn_statuses(combat)
    events.extend(tick_events)

    phase = _phase_after_combat(combat)
    rng_state = capture_random_state(rng)
    if phase == RunPhase.COMBAT:
        combat, rng_state, draw_events = _draw_cards(
            combat,
            rng_state,
            combat.draw_per_turn + extra_draw,
        )
        events.extend(draw_events)
        phase = _phase_after_combat(combat)

    event_tuple = tuple(events)
    combat = combat.model_copy(update={"last_events": event_tuple})
    return _state_after_combat(state, combat, rng_state), event_tuple


def _split_retained_end_turn_cards(
    hand: Sequence[CardInstance],
) -> tuple[tuple[CardInstance, ...], tuple[CardInstance, ...]]:
    retained: list[CardInstance] = []
    discarded: list[CardInstance] = []
    for card in hand:
        if _card_retains_at_end_of_turn(card):
            retained.append(_card_after_end_turn_retain(card))
        else:
            discarded.append(card)
    return tuple(retained), tuple(discarded)


def _card_retains_at_end_of_turn(card: CardInstance) -> bool:
    custom = card.custom
    if any(
        _truthy(custom.get(key))
        for key in ("retain", "retained", "retain_once", "temporary_retain")
    ):
        return True
    if _custom_int(custom, "retain_turns", 0) > 0:
        return True
    if any(_normalized_tag(tag) in {"retain", "retained", "keyword_retain"} for tag in card.tags):
        return True
    return _source_spec_has_retain(_mapping_from(custom.get("source_spec")))


def _card_after_end_turn_retain(card: CardInstance) -> CardInstance:
    custom = dict(card.custom)
    changed = False
    for key in ("retain_once", "temporary_retain"):
        if key in custom:
            custom.pop(key, None)
            changed = True
    if "retain_turns" in custom:
        turns = _custom_int(custom, "retain_turns", 0)
        if turns <= 1:
            custom.pop("retain_turns", None)
        else:
            custom["retain_turns"] = turns - 1
        changed = True
    return card.model_copy(update={"custom": custom}) if changed else card


def _execute_monster_turn(
    *,
    state: RunState,
    monster: MonsterState,
    player: PlayerState,
    monster_definitions: Mapping[str, MonsterDefinition],
    rng: Random,
) -> tuple[MonsterState, PlayerState, tuple[EffectEvent, ...]]:
    events: list[EffectEvent] = []
    if monster.block:
        events.append(
            EffectEvent(
                kind="monster_block_expired",
                source_id=monster.monster_id,
                target_id=monster.monster_id,
                amount=monster.block,
            )
        )
        monster = monster.model_copy(update={"block": 0})

    definition = _definition_for_monster(monster, monster_definitions)
    move = move_by_id(definition, monster.move_id) if definition is not None else None
    if definition is None or move is None:
        monster, player, fallback_events = _execute_fallback_monster_turn(
            monster,
            player,
            relics=state.relics,
        )
        return monster, player, tuple(events) + fallback_events

    per_hit_damage = _monster_attack_hit_damage(
        definition,
        move,
        statuses=monster.statuses,
        ascension_level=state.ascension,
    )
    if per_hit_damage:
        for hit_index in range(move.hit_count):
            player, damage_event = _damage_player(
                player,
                per_hit_damage,
                monster.monster_id,
                relics=state.relics,
            )
            events.append(
                damage_event.model_copy(
                    update={
                        "metadata": {
                            **damage_event.metadata,
                            "hit_index": hit_index,
                            "hit_count": move.hit_count,
                        }
                    }
                )
            )
            monster, thorns_events = _apply_player_thorns_to_attacker(
                monster,
                player,
                source_id=monster.monster_id,
            )
            events.extend(thorns_events)
            if monster.hp <= 0:
                break

    if monster.hp <= 0:
        return monster, player, tuple(events)

    if move.block:
        monster = monster.model_copy(update={"block": monster.block + move.block})
        events.append(
            EffectEvent(
                kind="monster_block",
                source_id=monster.monster_id,
                target_id=monster.monster_id,
                amount=move.block,
            )
        )

    if move.heal:
        old_hp = monster.hp
        monster = monster.model_copy(update={"hp": min(monster.max_hp, monster.hp + move.heal)})
        events.append(
            EffectEvent(
                kind="monster_healed",
                source_id=monster.monster_id,
                target_id=monster.monster_id,
                amount=monster.hp - old_hp,
            )
        )

    for power in move.powers:
        monster, player, power_event = _apply_monster_power(
            definition,
            monster,
            player,
            power,
            ascension_level=state.ascension,
        )
        if power_event is not None:
            events.append(power_event)

    counts = next_move_counts(_monster_move_counts(monster), move.move_id)
    next_move = next_monster_move(
        definition,
        move.move_id,
        rng,
        slot_index=_monster_slot_index(monster),
        move_counts=counts,
    )
    monster = _set_monster_move(
        monster,
        definition,
        next_move,
        player=player,
        ascension_level=state.ascension,
        move_counts=counts,
    )

    return monster, player, tuple(events)


def _apply_after_card_play_hooks(
    combat: CombatState,
    card: CardInstance,
    relics: Sequence[str] = (),
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    player = combat.player
    events: list[EffectEvent] = []
    afterimage = _status_amount(player.statuses, "afterimage")
    if afterimage > 0:
        player = player.model_copy(update={"block": player.block + afterimage})
        events.append(
            EffectEvent(
                kind="player_block",
                source_id=card.instance_id,
                target_id=PLAYER_TARGET_ID,
                amount=afterimage,
                metadata={"status": "afterimage"},
            )
        )
    combat = combat.model_copy(update={"player": player})
    attack_count = _metadata_int(combat.metadata, "attacks_played_this_turn", 0)
    relic_result = _resolve_combat_relic_trigger(
        GameTrigger.CARD_PLAYED,
        relics,
        card_type=card.type.value,
        card_id=card.card_id,
        player_hp=player.hp,
        player_max_hp=player.max_hp,
        player_block=player.block,
        player_statuses=player.statuses,
        relic_counters=_combat_relic_counters(combat),
        metadata={"attacks_played_this_turn": attack_count},
    )
    combat, relic_events = _apply_combat_relic_resolution(combat, relic_result)
    metadata = dict(combat.metadata)
    if card.type == CardType.ATTACK:
        metadata["attacks_played_this_turn"] = attack_count + 1
    combat = combat.model_copy(update={"metadata": metadata})
    return combat, tuple(events) + relic_events


def _execute_fallback_monster_turn(
    monster: MonsterState,
    player: PlayerState,
    *,
    relics: Sequence[str] = (),
) -> tuple[MonsterState, PlayerState, tuple[EffectEvent, ...]]:
    events: list[EffectEvent] = []
    if monster.intent_damage:
        player, event = _damage_player(
            player,
            monster.intent_damage,
            monster.monster_id,
            relics=relics,
        )
        events.append(event)
        monster, thorns_events = _apply_player_thorns_to_attacker(
            monster,
            player,
            source_id=monster.monster_id,
        )
        events.extend(thorns_events)
    if monster.hp <= 0:
        return monster, player, tuple(events)
    if monster.intent_block:
        monster = monster.model_copy(update={"block": monster.block + monster.intent_block})
        events.append(
            EffectEvent(
                kind="monster_block",
                source_id=monster.monster_id,
                target_id=monster.monster_id,
                amount=monster.intent_block,
            )
        )
    return monster, player, tuple(events)


def _definition_for_monster(
    monster: MonsterState,
    monster_definitions: Mapping[str, MonsterDefinition],
) -> MonsterDefinition | None:
    source_monster_id = monster.metadata.get("source_monster_id")
    if isinstance(source_monster_id, str):
        definition = monster_definitions.get(source_monster_id)
        if definition is not None:
            return definition
    return monster_definitions.get(monster.monster_id)


def _apply_monster_power(
    definition: MonsterDefinition,
    monster: MonsterState,
    player: PlayerState,
    power: MonsterPower,
    *,
    ascension_level: int,
) -> tuple[MonsterState, PlayerState, EffectEvent | None]:
    if not power.power_id:
        return monster, player, None

    amount = monster_power_amount(definition, power, ascension_level=ascension_level)
    if amount <= 0:
        return monster, player, None

    status = _normalized_id(power.power_id)
    target = _normalized_id(power.target or "self")
    if target in {"player", "adventurer"}:
        statuses, event = _apply_status_value(
            player.statuses,
            status,
            amount,
            source_id=monster.monster_id,
            target_id=PLAYER_TARGET_ID,
        )
        player = player.model_copy(update={"statuses": statuses})
        return (
            monster,
            player,
            event,
        )

    statuses, event = _apply_status_value(
        monster.statuses,
        status,
        amount,
        source_id=monster.monster_id,
        target_id=monster.monster_id,
    )
    monster = monster.model_copy(update={"statuses": statuses})
    return (
        monster,
        player,
        event,
    )


def _set_monster_move(
    monster: MonsterState,
    definition: MonsterDefinition,
    move: MonsterMove | None,
    *,
    player: PlayerState,
    ascension_level: int,
    move_counts: Mapping[str, int],
) -> MonsterState:
    metadata = dict(monster.metadata)
    metadata["move_counts"] = {str(key): int(value) for key, value in move_counts.items()}
    if move is None:
        return monster.model_copy(
            update={
                "intent": None,
                "intent_damage": 0,
                "intent_block": 0,
                "move_id": None,
                "next_move_id": None,
                "hit_count": 1,
                "metadata": metadata,
            }
        )

    metadata.update(_monster_move_metadata(definition, move, ascension_level))
    return monster.model_copy(
        update={
            "intent": move.intent,
            "intent_damage": _monster_intent_damage(
                definition,
                move,
                statuses=monster.statuses,
                player=player,
                ascension_level=ascension_level,
            ),
            "intent_block": move.block,
            "move_id": move.move_id,
            "next_move_id": None,
            "hit_count": move.hit_count,
            "metadata": metadata,
        }
    )


def _monster_move_counts(monster: MonsterState) -> Mapping[str, int]:
    counts = monster.metadata.get("move_counts", {})
    if isinstance(counts, Mapping):
        parsed: dict[str, int] = {}
        for key, value in counts.items():
            with suppress(TypeError, ValueError):
                parsed[str(key)] = int(value)
        return parsed
    return {}


def _monster_slot_index(monster: MonsterState) -> int:
    value = monster.metadata.get("slot_index", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _apply_card_effects(
    *,
    combat: CombatState,
    rng_state: RngState,
    card: CardInstance,
    target_id: str | None,
    energy_spent: int,
    relics: Sequence[str] = (),
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    events: list[EffectEvent] = []
    for effect in _effect_steps(card.effects):
        combat, rng_state, effect_events = _apply_effect_step(
            combat=combat,
            rng_state=rng_state,
            card=card,
            target_id=target_id,
            energy_spent=energy_spent,
            effect=effect,
            relics=relics,
        )
        events.extend(effect_events)
    return combat, rng_state, tuple(events)


def _apply_effect_step(
    *,
    combat: CombatState,
    rng_state: RngState,
    card: CardInstance,
    target_id: str | None,
    energy_spent: int,
    effect: Mapping[str, Any],
    relics: Sequence[str] = (),
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    events: list[EffectEvent] = []
    player = combat.player
    monsters = list(combat.monsters)

    damage = _effect_amount(effect, "damage", energy_spent)
    if damage:
        damage = _modified_card_damage(combat, card, damage)
        targets = _effect_enemy_targets(combat, card, target_id, all_enemies=False)
        monsters, damage_events = _damage_monsters(
            monsters,
            targets,
            damage,
            card.instance_id,
            relics=relics,
        )
        events.extend(damage_events)

    all_damage = _effect_amount(effect, "all_damage", energy_spent)
    if all_damage:
        all_damage = _modified_card_damage(combat, card, all_damage)
        targets = tuple(monster.monster_id for monster in monsters if monster.hp > 0)
        monsters, damage_events = _damage_monsters(
            monsters,
            targets,
            all_damage,
            card.instance_id,
            relics=relics,
        )
        events.extend(damage_events)

    block = _effect_amount(effect, "block", energy_spent)
    if block:
        block = _modified_card_block(combat, card, block)
        player = player.model_copy(update={"block": player.block + block})
        events.append(
            EffectEvent(
                kind="player_block",
                source_id=card.instance_id,
                target_id=PLAYER_TARGET_ID,
                amount=block,
            )
        )

    energy = _effect_amount(effect, "energy", energy_spent)
    if energy:
        player = player.model_copy(update={"energy": max(0, player.energy + energy)})
        events.append(
            EffectEvent(
                kind="energy_changed",
                source_id=card.instance_id,
                target_id=PLAYER_TARGET_ID,
                amount=energy,
            )
        )

    heal = _effect_amount(effect, "heal", energy_spent)
    if heal:
        old_hp = player.hp
        player = player.model_copy(update={"hp": min(player.max_hp, player.hp + heal)})
        events.append(
            EffectEvent(
                kind="player_healed",
                source_id=card.instance_id,
                target_id=PLAYER_TARGET_ID,
                amount=player.hp - old_hp,
            )
        )

    hp_loss = _effect_amount(effect, "hp_loss", energy_spent)
    if hp_loss:
        player, hp_loss_event = _lose_player_hp(player, hp_loss, card.instance_id)
        events.append(hp_loss_event)

    max_hp = _effect_amount(effect, "max_hp", energy_spent)
    if max_hp:
        new_max_hp = max(1, player.max_hp + max_hp)
        player = player.model_copy(update={"max_hp": new_max_hp, "hp": min(player.hp, new_max_hp)})
        events.append(
            EffectEvent(
                kind="player_max_hp_changed",
                source_id=card.instance_id,
                target_id=PLAYER_TARGET_ID,
                amount=max_hp,
            )
        )

    player, resource_events = _apply_player_resource_effect(player, card, effect, energy_spent)
    events.extend(resource_events)

    combat = combat.model_copy(update={"player": player, "monsters": tuple(monsters)})
    combat, rng_state, orb_events = _apply_orb_effects(
        combat,
        rng_state,
        card.instance_id,
        effect,
        target_id=target_id,
        relics=relics,
    )
    events.extend(orb_events)

    draw = _effect_amount(effect, "draw", energy_spent)
    if draw:
        combat, rng_state, draw_events = _draw_cards(combat, rng_state, draw)
        events.extend(draw_events)

    add_random_cards = _effect_amount(effect, "add_random_card_to_hand", energy_spent)
    if add_random_cards:
        combat, rng_state, generated_events = _add_random_free_cards_to_hand(
            combat,
            rng_state,
            card,
            add_random_cards,
        )
        events.extend(generated_events)

    combat, rng_state, pile_events = _apply_generated_card_effects(combat, rng_state, card, effect)
    events.extend(pile_events)

    combat, rng_state, hand_events = _apply_hand_manipulation_effects(
        combat,
        rng_state,
        card,
        effect,
    )
    events.extend(hand_events)

    combat, next_turn_events = _apply_next_turn_effects(combat, card, effect)
    events.extend(next_turn_events)

    combat, status_events = _apply_status_effects(combat, card, target_id, effect)
    events.extend(status_events)

    stubbed = sorted(set(effect) - _SUPPORTED_EFFECT_KEYS)
    if stubbed:
        events.append(
            EffectEvent(
                kind="effect_stubbed",
                source_id=card.instance_id,
                target_id=target_id,
                message="Unsupported effect keys were preserved but not executed.",
                metadata={"keys": stubbed},
            )
        )

    return combat, rng_state, tuple(events)


def _apply_orb_effects(
    combat: CombatState,
    rng_state: RngState,
    source_id: str,
    effect: Mapping[str, Any],
    *,
    target_id: str | None,
    relics: Sequence[str] = (),
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    events: list[EffectEvent] = []

    slot_delta = _orb_slot_delta(effect.get("orb_slot_delta"))
    if slot_delta:
        combat, slot_events = _change_orb_slots(combat, slot_delta, source_id)
        events.extend(slot_events)

    rng = random_from_state(rng_state)
    for orb_id, amount in _channel_orb_requests(effect.get("channel_orb"), rng):
        for _ in range(amount):
            combat, channel_events = _channel_orb(
                combat,
                rng,
                source_id,
                orb_id,
                target_id=target_id,
                relics=relics,
            )
            events.extend(channel_events)

    evoke_payload = effect.get("evoke_orb")
    if evoke_payload is not None:
        combat, evoke_events = _evoke_orbs(
            combat,
            rng,
            source_id,
            evoke_payload,
            target_id=target_id,
            relics=relics,
        )
        events.extend(evoke_events)

    return combat, capture_random_state(rng), tuple(events)


def _change_orb_slots(
    combat: CombatState,
    delta: int,
    source_id: str,
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    old_slots = combat.orb_slots
    next_slots = min(10, max(0, old_slots + delta))
    next_orbs = combat.orbs[:next_slots]
    trimmed = len(combat.orbs) - len(next_orbs)
    events = [
        EffectEvent(
            kind="orb_slots_changed",
            source_id=source_id,
            amount=next_slots - old_slots,
            metadata={
                "old_slots": old_slots,
                "new_slots": next_slots,
                "trimmed_orbs": trimmed,
            },
        )
    ]
    return combat.model_copy(update={"orb_slots": next_slots, "orbs": next_orbs}), tuple(events)


def _channel_orb(
    combat: CombatState,
    rng: Random,
    source_id: str,
    orb_id: str,
    *,
    target_id: str | None,
    relics: Sequence[str],
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    orb = _new_orb(orb_id)
    events: list[EffectEvent] = []
    if combat.orb_slots <= 0:
        events.append(
            EffectEvent(
                kind="orb_channel_blocked",
                source_id=source_id,
                target_id=PLAYER_TARGET_ID,
                metadata={"orb": orb.orb_id, "reason": "no_orb_slots"},
            )
        )
        return combat, tuple(events)

    orbs = list(combat.orbs)
    if len(orbs) >= combat.orb_slots:
        evoked = orbs.pop(0)
        combat = combat.model_copy(update={"orbs": tuple(orbs)})
        combat, evoke_events = _apply_orb_evoke(
            combat,
            rng,
            evoked,
            source_id,
            target_id=target_id,
            relics=relics,
            metadata={"reason": "channel_overflow"},
        )
        events.extend(evoke_events)
        orbs = list(combat.orbs)

    orbs.append(orb)
    combat = combat.model_copy(update={"orbs": tuple(orbs)})
    events.append(
        EffectEvent(
            kind="orb_channeled",
            source_id=source_id,
            target_id=PLAYER_TARGET_ID,
            amount=1,
            metadata={"orb": orb.orb_id, "orb_slots": combat.orb_slots},
        )
    )
    return combat, tuple(events)


def _evoke_orbs(
    combat: CombatState,
    rng: Random,
    source_id: str,
    payload: Any,
    *,
    target_id: str | None,
    relics: Sequence[str],
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    selector = "leftmost"
    amount: object = 1
    if isinstance(payload, Mapping):
        selector = _normalized_id(str(payload.get("selector", payload.get("target", selector))))
        amount = payload.get("amount", payload.get("times", 1))
    elif isinstance(payload, str):
        selector = _normalized_id(payload)
    elif isinstance(payload, int):
        amount = payload

    if selector == "all" or amount == "all":
        events: list[EffectEvent] = []
        while combat.orbs:
            orb = combat.orbs[0]
            combat = combat.model_copy(update={"orbs": combat.orbs[1:]})
            combat, evoke_events = _apply_orb_evoke(
                combat,
                rng,
                orb,
                source_id,
                target_id=target_id,
                relics=relics,
                metadata={"selector": "all"},
            )
            events.extend(evoke_events)
        return combat, tuple(events)

    repeats = max(1, _coerce_positive_int(amount, default=1))
    index = -1 if selector == "rightmost" else 0
    if not combat.orbs:
        return (
            combat,
            (
                EffectEvent(
                    kind="orb_evoke_blocked",
                    source_id=source_id,
                    target_id=PLAYER_TARGET_ID,
                    metadata={"reason": "no_orbs", "selector": selector},
                ),
            ),
        )

    orbs = list(combat.orbs)
    orb = orbs.pop(index)
    combat = combat.model_copy(update={"orbs": tuple(orbs)})
    events = []
    for _ in range(repeats):
        combat, evoke_events = _apply_orb_evoke(
            combat,
            rng,
            orb,
            source_id,
            target_id=target_id,
            relics=relics,
            metadata={"selector": selector, "repeat_count": repeats},
        )
        events.extend(evoke_events)
    return combat, tuple(events)


def _apply_orb_passives(
    combat: CombatState,
    rng_state: RngState,
    *,
    relics: Sequence[str] = (),
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    if not combat.orbs:
        return combat, rng_state, ()

    rng = random_from_state(rng_state)
    events: list[EffectEvent] = []
    updated_orbs = list(combat.orbs)
    for index in range(len(updated_orbs) - 1, -1, -1):
        orb = updated_orbs[index]
        combat = combat.model_copy(update={"orbs": tuple(updated_orbs)})
        combat, passive_events, next_orb = _apply_orb_passive(
            combat,
            rng,
            orb,
            relics=relics,
        )
        events.extend(passive_events)
        updated_orbs = list(combat.orbs)
        updated_orbs[index] = next_orb
    combat = combat.model_copy(update={"orbs": tuple(updated_orbs)})
    return combat, capture_random_state(rng), tuple(events)


def _apply_orb_passive(
    combat: CombatState,
    rng: Random,
    orb: OrbState,
    *,
    relics: Sequence[str],
) -> tuple[CombatState, tuple[EffectEvent, ...], OrbState]:
    focus = _status_amount(combat.player.statuses, "focus")
    if orb.orb_id == "lightning":
        amount = _orb_amount(3, focus)
        combat, events = _damage_random_monster(
            combat,
            rng,
            amount,
            "orb:lightning",
            relics=relics,
            metadata={"orb": "lightning", "trigger": "passive"},
        )
        return combat, events, orb
    if orb.orb_id == "frost":
        amount = _orb_amount(2, focus)
        player = combat.player.model_copy(update={"block": combat.player.block + amount})
        return (
            combat.model_copy(update={"player": player}),
            (
                EffectEvent(
                    kind="player_block",
                    source_id="orb:frost",
                    target_id=PLAYER_TARGET_ID,
                    amount=amount,
                    metadata={"orb": "frost", "trigger": "passive"},
                ),
            ),
            orb,
        )
    if orb.orb_id == "dark":
        amount = _orb_amount(6, focus)
        next_orb = orb.model_copy(update={"value": orb.value + amount})
        return (
            combat,
            (
                EffectEvent(
                    kind="orb_dark_charged",
                    source_id="orb:dark",
                    target_id=PLAYER_TARGET_ID,
                    amount=amount,
                    metadata={"value": next_orb.value, "trigger": "passive"},
                ),
            ),
            next_orb,
        )
    if orb.orb_id == "plasma":
        statuses = dict(combat.player.statuses)
        statuses["next_turn_energy"] = statuses.get("next_turn_energy", 0) + 1
        player = combat.player.model_copy(update={"statuses": statuses})
        return (
            combat.model_copy(update={"player": player}),
            (
                EffectEvent(
                    kind="next_turn_effect_added",
                    source_id="orb:plasma",
                    target_id=PLAYER_TARGET_ID,
                    amount=1,
                    metadata={
                        "orb": "plasma",
                        "trigger": "passive",
                        "status": "next_turn_energy",
                    },
                ),
            ),
            orb,
        )
    if orb.orb_id == "glass":
        amount = _orb_amount(orb.value, focus)
        combat, events = _damage_all_monsters(
            combat,
            amount,
            "orb:glass",
            relics=relics,
            metadata={"orb": "glass", "trigger": "passive"},
        )
        return combat, events, orb.model_copy(update={"value": max(0, orb.value - 1)})
    return combat, (), orb


def _apply_orb_evoke(
    combat: CombatState,
    rng: Random,
    orb: OrbState,
    source_id: str,
    *,
    target_id: str | None,
    relics: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    focus = _status_amount(combat.player.statuses, "focus")
    base_metadata = {"orb": orb.orb_id, "trigger": "evoke", **dict(metadata or {})}
    if orb.orb_id == "lightning":
        return _damage_random_monster(
            combat,
            rng,
            _orb_amount(8, focus),
            source_id,
            relics=relics,
            metadata=base_metadata,
        )
    if orb.orb_id == "frost":
        amount = _orb_amount(5, focus)
        player = combat.player.model_copy(update={"block": combat.player.block + amount})
        return (
            combat.model_copy(update={"player": player}),
            (
                EffectEvent(
                    kind="player_block",
                    source_id=source_id,
                    target_id=PLAYER_TARGET_ID,
                    amount=amount,
                    metadata=base_metadata,
                ),
            ),
        )
    if orb.orb_id == "dark":
        target = _dark_orb_target(combat, target_id)
        if target is None:
            return combat, ()
        monsters, damage_events = _damage_monsters(
            combat.monsters,
            (target,),
            orb.value,
            source_id,
            relics=relics,
        )
        return (
            combat.model_copy(update={"monsters": tuple(monsters)}),
            tuple(
                event.model_copy(update={"metadata": {**event.metadata, **base_metadata}})
                for event in damage_events
            ),
        )
    if orb.orb_id == "plasma":
        player = combat.player.model_copy(update={"energy": combat.player.energy + 2})
        return (
            combat.model_copy(update={"player": player}),
            (
                EffectEvent(
                    kind="energy_changed",
                    source_id=source_id,
                    target_id=PLAYER_TARGET_ID,
                    amount=2,
                    metadata=base_metadata,
                ),
            ),
        )
    if orb.orb_id == "glass":
        return _damage_all_monsters(
            combat,
            _orb_amount(orb.value * 2, focus),
            source_id,
            relics=relics,
            metadata=base_metadata,
        )
    return combat, ()


def _damage_random_monster(
    combat: CombatState,
    rng: Random,
    amount: int,
    source_id: str,
    *,
    relics: Sequence[str],
    metadata: Mapping[str, Any],
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    target = _random_alive_monster_id(combat, rng)
    if target is None or amount <= 0:
        return combat, ()
    monsters, damage_events = _damage_monsters(
        combat.monsters,
        (target,),
        amount,
        source_id,
        relics=relics,
    )
    return (
        combat.model_copy(update={"monsters": tuple(monsters)}),
        tuple(
            event.model_copy(update={"metadata": {**event.metadata, **dict(metadata)}})
            for event in damage_events
        ),
    )


def _damage_all_monsters(
    combat: CombatState,
    amount: int,
    source_id: str,
    *,
    relics: Sequence[str],
    metadata: Mapping[str, Any],
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    targets = tuple(monster.monster_id for monster in combat.monsters if monster.hp > 0)
    if not targets or amount <= 0:
        return combat, ()
    monsters, damage_events = _damage_monsters(
        combat.monsters,
        targets,
        amount,
        source_id,
        relics=relics,
    )
    return (
        combat.model_copy(update={"monsters": tuple(monsters)}),
        tuple(
            event.model_copy(update={"metadata": {**event.metadata, **dict(metadata)}})
            for event in damage_events
        ),
    )


def _channel_orb_requests(payload: Any, rng: Random) -> tuple[tuple[str, int], ...]:
    if payload is None:
        return ()
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray, Mapping)):
        requests: list[tuple[str, int]] = []
        for item in payload:
            requests.extend(_channel_orb_requests(item, rng))
        return tuple(requests)
    if isinstance(payload, Mapping):
        orb_id = _normalized_orb_id(str(payload.get("orb", payload.get("orb_id", "lightning"))))
        amount = _coerce_positive_int(payload.get("amount", payload.get("count", 1)), default=1)
    else:
        orb_id = _normalized_orb_id(str(payload))
        amount = 1
    if orb_id == "random_orb":
        return tuple((rng.choice(_ORB_TYPES), 1) for _ in range(max(0, amount)))
    return ((orb_id, max(0, amount)),)


def _orb_slot_delta(payload: Any) -> int:
    if payload in (None, ""):
        return 0
    if isinstance(payload, Mapping):
        payload = payload.get("amount", payload.get("delta", 0))
    return _coerce_int(payload, default=0)


def _new_orb(orb_id: str) -> OrbState:
    normalized = _normalized_orb_id(orb_id)
    if normalized == "dark":
        return OrbState(orb_id=normalized, value=6)
    if normalized == "glass":
        return OrbState(orb_id=normalized, value=4)
    return OrbState(orb_id=normalized)


def _random_alive_monster_id(combat: CombatState, rng: Random) -> str | None:
    alive = [monster.monster_id for monster in combat.monsters if monster.hp > 0]
    return rng.choice(alive) if alive else None


def _dark_orb_target(combat: CombatState, target_id: str | None) -> str | None:
    if target_id and any(
        monster.monster_id == target_id and monster.hp > 0 for monster in combat.monsters
    ):
        return target_id
    alive = [monster for monster in combat.monsters if monster.hp > 0]
    if not alive:
        return None
    return min(alive, key=lambda monster: (monster.hp, monster.monster_id)).monster_id


def _orb_amount(base: int, focus: int) -> int:
    return max(0, base + focus)


def _normalized_orb_id(value: str) -> str:
    normalized = _normalized_id(value)
    if normalized == "random":
        return "random_orb"
    if normalized not in _ORB_TYPES and normalized != "random_orb":
        return "lightning"
    return normalized


def _card_movement_event(
    trigger: GameTrigger,
    card: CardInstance,
    *,
    from_pile: str,
    to_pile: str,
    source_id: str | None = None,
    reason: str,
    metadata: Mapping[str, Any] | None = None,
) -> EffectEvent:
    return EffectEvent(
        kind=trigger.value,
        source_id=source_id,
        target_id=card.instance_id,
        amount=1,
        metadata={
            "trigger": trigger.value,
            "card_id": card.card_id,
            "from_pile": from_pile,
            "to_pile": to_pile,
            "reason": reason,
            **dict(metadata or {}),
        },
    )


def _card_movement_events(
    trigger: GameTrigger,
    cards: Sequence[CardInstance],
    *,
    from_pile: str,
    to_pile: str,
    source_id: str | None = None,
    reason: str,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[EffectEvent, ...]:
    return tuple(
        _card_movement_event(
            trigger,
            card,
            from_pile=from_pile,
            to_pile=to_pile,
            source_id=source_id,
            reason=reason,
            metadata=metadata,
        )
        for card in cards
    )


def _pile_trigger_event(
    trigger: GameTrigger,
    *,
    source_pile: str,
    target_pile: str,
    amount: int,
    reason: str,
    metadata: Mapping[str, Any] | None = None,
) -> EffectEvent:
    return EffectEvent(
        kind=trigger.value,
        amount=amount,
        metadata={
            "trigger": trigger.value,
            "source_pile": source_pile,
            "target_pile": target_pile,
            "reason": reason,
            **dict(metadata or {}),
        },
    )


def _append_cards_to_pile(
    combat: CombatState,
    pile_name: Literal["discard_pile", "exhaust_pile"],
    cards: Sequence[CardInstance],
) -> CombatState:
    if not cards:
        return combat
    if pile_name == "discard_pile":
        return combat.model_copy(update={"discard_pile": combat.discard_pile + tuple(cards)})
    return combat.model_copy(update={"exhaust_pile": combat.exhaust_pile + tuple(cards)})


def _move_played_card_to_destination(
    combat: CombatState,
    card: CardInstance,
    destination: str,
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    if destination == "discard":
        return (
            _append_cards_to_pile(combat, "discard_pile", (card,)),
            _card_movement_events(
                GameTrigger.CARD_DISCARDED,
                (card,),
                from_pile="play_area",
                to_pile="discard_pile",
                source_id=card.instance_id,
                reason="played_card_destination",
            ),
        )
    if destination == "exhaust":
        return (
            _append_cards_to_pile(combat, "exhaust_pile", (card,)),
            _card_movement_events(
                GameTrigger.CARD_EXHAUSTED,
                (card,),
                from_pile="play_area",
                to_pile="exhaust_pile",
                source_id=card.instance_id,
                reason="played_card_destination",
            ),
        )
    return combat, ()


def _coerce_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_positive_int(value: Any, *, default: int = 1) -> int:
    if isinstance(value, str) and value.strip().lower() == "twice":
        return 2
    return max(0, _coerce_int(value, default=default))


def _draw_cards(
    combat: CombatState,
    rng_state: RngState,
    amount: int,
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    if amount <= 0:
        return combat, rng_state, ()

    rng = random_from_state(rng_state)
    draw_pile = list(combat.draw_pile)
    discard_pile = list(combat.discard_pile)
    hand = list(combat.hand)
    events: list[EffectEvent] = []
    drawn_cards: list[CardInstance] = []

    for _ in range(amount):
        if not draw_pile:
            if not discard_pile:
                break
            rng.shuffle(discard_pile)
            draw_pile = discard_pile
            discard_pile = []
            events.append(EffectEvent(kind="discard_shuffled", amount=len(draw_pile)))
            events.append(
                _pile_trigger_event(
                    GameTrigger.DRAW_PILE_SHUFFLED,
                    source_pile="discard_pile",
                    target_pile="draw_pile",
                    amount=len(draw_pile),
                    reason="draw_from_empty_draw_pile",
                    metadata={"card_instance_ids": [card.instance_id for card in draw_pile]},
                )
            )
        if not draw_pile:
            break
        drawn_card = draw_pile.pop(0)
        hand.append(drawn_card)
        drawn_cards.append(drawn_card)

    drawn_count = len(hand) - len(combat.hand)
    if drawn_count:
        events.append(
            EffectEvent(
                kind="cards_drawn",
                amount=drawn_count,
                metadata={
                    "card_instance_ids": [card.instance_id for card in hand[-drawn_count:]]
                },
            )
        )
        events.extend(
            _card_movement_events(
                GameTrigger.CARD_DRAWN,
                drawn_cards,
                from_pile="draw_pile",
                to_pile="hand",
                reason="draw_cards",
            )
        )

    return (
        combat.model_copy(
            update={
                "draw_pile": tuple(draw_pile),
                "discard_pile": tuple(discard_pile),
                "hand": tuple(hand),
            }
        ),
        capture_random_state(rng),
        tuple(events),
    )


def _add_random_free_cards_to_hand(
    combat: CombatState,
    rng_state: RngState,
    source_card: CardInstance,
    amount: int,
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    rng = random_from_state(rng_state)
    hand = list(combat.hand)
    events: list[EffectEvent] = []
    templates: tuple[dict[str, Any], ...] = (
        {
            "card_id": "generated_free_attack",
            "name": "Generated Free Attack",
            "type": "attack",
            "target": "enemy",
            "effects": {"damage": 6},
        },
        {
            "card_id": "generated_free_skill",
            "name": "Generated Free Skill",
            "type": "skill",
            "target": "self",
            "effects": {"block": 5},
        },
    )
    for index in range(max(0, amount)):
        template = dict(rng.choice(templates))
        template.update(
            {
                "instance_id": f"{source_card.instance_id}:generated:{combat.turn}:{index}",
                "cost": 0,
                "exhaust": True,
                "tags": ("generated", "temporary"),
            }
        )
        generated = _card_from_spec(template, len(hand) + 1)
        hand.append(generated)
        events.append(
            EffectEvent(
                kind="card_added_to_hand",
                source_id=source_card.instance_id,
                target_id=generated.instance_id,
                metadata={"card_id": generated.card_id, "temporary": True},
            )
        )
    return (
        combat.model_copy(update={"hand": tuple(hand)}),
        capture_random_state(rng),
        tuple(events),
    )


def _apply_generated_card_effects(
    combat: CombatState,
    rng_state: RngState,
    source_card: CardInstance,
    effect: Mapping[str, Any],
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    events: list[EffectEvent] = []
    hand = list(combat.hand)
    draw_pile = list(combat.draw_pile)
    discard_pile = list(combat.discard_pile)
    exhaust_pile = list(combat.exhaust_pile)
    generated_index = 0

    for key, destination in (
        ("add_card_to_hand", "hand"),
        ("add_card_to_draw", "draw"),
        ("add_card_to_discard", "discard"),
        ("add_card_to_exhaust", "exhaust"),
    ):
        payload = effect.get(key)
        if payload is None:
            continue
        for card_payload in _generated_card_payloads(payload):
            generated_index += 1
            card_spec = dict(_mapping_from(card_payload.get("card", card_payload)))
            card_spec.setdefault(
                "instance_id",
                f"{source_card.instance_id}:generated:{combat.turn}:{generated_index}",
            )
            card = _card_from_spec(card_spec, len(hand) + len(draw_pile) + len(discard_pile) + 1)
            if destination == "hand":
                hand.append(card)
            elif destination == "draw":
                draw_pile.insert(0, card)
            elif destination == "discard":
                discard_pile.append(card)
            else:
                exhaust_pile.append(card)
            events.append(
                EffectEvent(
                    kind=f"card_added_to_{destination}",
                    source_id=source_card.instance_id,
                    target_id=card.instance_id,
                    metadata={"card_id": card.card_id, "temporary": True},
                )
            )

    return (
        combat.model_copy(
            update={
                "hand": tuple(hand),
                "draw_pile": tuple(draw_pile),
                "discard_pile": tuple(discard_pile),
                "exhaust_pile": tuple(exhaust_pile),
            }
        ),
        rng_state,
        tuple(events),
    )


def _generated_card_payloads(payload: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(payload, Mapping):
        return (payload,)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return tuple(item for item in payload if isinstance(item, Mapping))
    return ()


def _apply_hand_manipulation_effects(
    combat: CombatState,
    rng_state: RngState,
    source_card: CardInstance,
    effect: Mapping[str, Any],
) -> tuple[CombatState, RngState, tuple[EffectEvent, ...]]:
    rng = random_from_state(rng_state)
    hand = list(combat.hand)
    discard_pile = list(combat.discard_pile)
    exhaust_pile = list(combat.exhaust_pile)
    events: list[EffectEvent] = []

    discard_count = _effect_amount(effect, "discard_random", 0)
    discard_hand = effect.get("discard_hand")
    if isinstance(discard_hand, Mapping) and discard_hand.get("mode") == "all":
        discard_count = len(hand)
    if discard_count:
        indexes = list(range(len(hand)))
        rng.shuffle(indexes)
        selected = sorted(indexes[: max(0, min(discard_count, len(hand)))], reverse=True)
        discarded: list[CardInstance] = []
        for index in selected:
            discarded.append(hand.pop(index))
        discard_pile.extend(reversed(discarded))
        events.append(
            EffectEvent(
                kind="cards_discarded",
                source_id=source_card.instance_id,
                amount=len(discarded),
                metadata={"card_instance_ids": [card.instance_id for card in discarded]},
            )
        )
        events.extend(
            _card_movement_events(
                GameTrigger.CARD_DISCARDED,
                tuple(reversed(discarded)),
                from_pile="hand",
                to_pile="discard_pile",
                source_id=source_card.instance_id,
                reason="random_discard",
            )
        )

    discard_choice_count = _effect_amount(effect, "discard_choice", 0)
    if discard_choice_count and hand:
        metadata = dict(combat.metadata)
        pending = _pending_discard_choice(combat)
        pending_remaining = pending["remaining"] if pending is not None else 0
        metadata["pending_card_choice"] = {
            "kind": "discard",
            "source_card_instance_id": source_card.instance_id,
            "remaining": min(len(hand), pending_remaining + discard_choice_count),
        }
        combat = combat.model_copy(update={"metadata": metadata})
        events.append(
            EffectEvent(
                kind="card_discard_choice_pending",
                source_id=source_card.instance_id,
                amount=metadata["pending_card_choice"]["remaining"],
                metadata={
                    "card_instance_ids": [card.instance_id for card in hand],
                    "card_ids": [card.card_id for card in hand],
                },
            )
        )

    exhaust_count = _effect_amount(effect, "exhaust_random", 0)
    exhaust_hand = effect.get("exhaust_hand")
    if isinstance(exhaust_hand, Mapping) and exhaust_hand.get("mode") == "all":
        exhaust_count = len(hand)
    if exhaust_count:
        indexes = list(range(len(hand)))
        rng.shuffle(indexes)
        selected = sorted(indexes[: max(0, min(exhaust_count, len(hand)))], reverse=True)
        exhausted: list[CardInstance] = []
        for index in selected:
            exhausted.append(hand.pop(index))
        exhaust_pile.extend(reversed(exhausted))
        events.append(
            EffectEvent(
                kind="cards_exhausted",
                source_id=source_card.instance_id,
                amount=len(exhausted),
                metadata={"card_instance_ids": [card.instance_id for card in exhausted]},
            )
        )
        events.extend(
            _card_movement_events(
                GameTrigger.CARD_EXHAUSTED,
                tuple(reversed(exhausted)),
                from_pile="hand",
                to_pile="exhaust_pile",
                source_id=source_card.instance_id,
                reason="random_exhaust",
            )
        )

    return (
        combat.model_copy(
            update={
                "hand": tuple(hand),
                "discard_pile": tuple(discard_pile),
                "exhaust_pile": tuple(exhaust_pile),
            }
        ),
        capture_random_state(rng),
        tuple(events),
    )


def _pending_discard_choice(combat: CombatState) -> dict[str, Any] | None:
    pending = combat.metadata.get("pending_card_choice")
    if not isinstance(pending, Mapping):
        return None
    if pending.get("kind") != "discard":
        return None
    remaining = _int_from_mapping(pending, "remaining", 0)
    if remaining <= 0:
        return None
    return {
        "kind": "discard",
        "source_card_instance_id": pending.get("source_card_instance_id"),
        "remaining": remaining,
    }


def _apply_next_turn_effects(
    combat: CombatState,
    source_card: CardInstance,
    effect: Mapping[str, Any],
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    player = combat.player
    statuses = dict(player.statuses)
    events: list[EffectEvent] = []

    next_turn = effect.get("next_turn")
    if isinstance(next_turn, Mapping):
        for key, status_id in (
            ("energy", "next_turn_energy"),
            ("star", "next_turn_star"),
            ("draw", "next_turn_draw"),
            ("block", "next_turn_block"),
        ):
            amount = _int_from_mapping(next_turn, key, 0)
            if amount:
                statuses[status_id] = statuses.get(status_id, 0) + amount
                events.append(
                    EffectEvent(
                        kind="next_turn_effect_added",
                        source_id=source_card.instance_id,
                        target_id=PLAYER_TARGET_ID,
                        amount=amount,
                        metadata={"status": status_id},
                    )
                )

    if bool(effect.get("retain_hand")):
        statuses["retain_hand"] = max(1, statuses.get("retain_hand", 0))
        events.append(
            EffectEvent(
                kind="retain_hand_added",
                source_id=source_card.instance_id,
                target_id=PLAYER_TARGET_ID,
                amount=1,
            )
        )

    return (
        combat.model_copy(update={"player": player.model_copy(update={"statuses": statuses})}),
        tuple(events),
    )


def _apply_player_resource_effect(
    player: PlayerState,
    source_card: CardInstance,
    effect: Mapping[str, Any],
    energy_spent: int,
) -> tuple[PlayerState, tuple[EffectEvent, ...]]:
    payload = effect.get("player_resource", effect.get("resource"))
    if not isinstance(payload, Mapping):
        return player, ()
    resource = _normalized_resource_id(str(payload.get("resource", payload.get("name", ""))))
    if not resource:
        return player, ()
    amount = _effect_amount(payload, "amount", energy_spent)
    if amount == 0:
        return player, ()
    player = _set_player_resource(player, resource, _player_resource(player, resource) + amount)
    events: list[EffectEvent] = [
        EffectEvent(
            kind="player_resource_changed",
            source_id=source_card.instance_id,
            target_id=PLAYER_TARGET_ID,
            amount=amount,
            metadata={"resource": resource, "value": _player_resource(player, resource)},
        )
    ]
    if resource == "mantra" and _player_resource(player, "mantra") >= 10:
        player = _set_player_resource(player, "mantra", _player_resource(player, "mantra") - 10)
        events.append(
            EffectEvent(
                kind="player_resource_changed",
                source_id=source_card.instance_id,
                target_id=PLAYER_TARGET_ID,
                amount=-10,
                metadata={"resource": "mantra", "value": _player_resource(player, "mantra")},
            )
        )
        player, stance_events = _apply_player_stance_status(
            player,
            "stance_divinity",
            source_id=source_card.instance_id,
        )
        events.extend(stance_events)
    return player, tuple(events)


def _apply_player_stance_status(
    player: PlayerState,
    stance_status: str,
    *,
    source_id: str | None,
) -> tuple[PlayerState, tuple[EffectEvent, ...]]:
    normalized = _normalized_status_id(stance_status)
    target_stance = "" if normalized in _STANCE_EXIT_STATUS_IDS else normalized
    if target_stance and target_stance not in _STANCE_STATUS_IDS:
        return player, ()

    previous_stance = _current_player_stance(player)
    statuses = {
        status: amount
        for status, amount in player.statuses.items()
        if _normalized_status_id(status) not in _STANCE_STATUS_IDS | _STANCE_EXIT_STATUS_IDS
    }
    if target_stance:
        statuses[target_stance] = 1

    events: list[EffectEvent] = []
    energy_delta = 0
    if previous_stance == "stance_calm" and target_stance != "stance_calm":
        energy_delta += 2
    if target_stance == "stance_divinity":
        energy_delta += 3

    if energy_delta:
        player = player.model_copy(update={"energy": player.energy + energy_delta})
        events.append(
            EffectEvent(
                kind="energy_changed",
                source_id=source_id,
                target_id=PLAYER_TARGET_ID,
                amount=energy_delta,
                metadata={"status": target_stance or "stance_none"},
            )
        )

    player = player.model_copy(update={"statuses": statuses})
    events.append(
        EffectEvent(
            kind="stance_changed",
            source_id=source_id,
            target_id=PLAYER_TARGET_ID,
            amount=1 if target_stance else 0,
            metadata={"previous_stance": previous_stance, "stance": target_stance or "none"},
        )
    )
    return player, tuple(events)


def _current_player_stance(player: PlayerState) -> str | None:
    for stance in _STANCE_STATUS_IDS:
        if _status_amount(player.statuses, stance) > 0:
            return stance
    return None


def _is_stance_status(status: str) -> bool:
    normalized = _normalized_status_id(status)
    return normalized in _STANCE_STATUS_IDS or normalized in _STANCE_EXIT_STATUS_IDS


def _lose_player_hp(
    player: PlayerState,
    amount: int,
    source_id: str,
) -> tuple[PlayerState, EffectEvent]:
    hp_loss = min(player.hp, max(0, amount))
    return (
        player.model_copy(update={"hp": player.hp - hp_loss}),
        EffectEvent(
            kind="player_hp_lost",
            source_id=source_id,
            target_id=PLAYER_TARGET_ID,
            amount=hp_loss,
        ),
    )


def _apply_player_turn_start_effects(
    combat: CombatState,
) -> tuple[CombatState, tuple[EffectEvent, ...], int]:
    player = combat.player
    statuses = dict(player.statuses)
    events: list[EffectEvent] = []
    extra_draw = max(0, int(statuses.pop("next_turn_draw", 0)))

    energy = max(0, int(statuses.pop("next_turn_energy", 0)))
    if energy:
        player = player.model_copy(update={"energy": player.energy + energy})
        events.append(EffectEvent(kind="energy_changed", target_id=PLAYER_TARGET_ID, amount=energy))

    block = max(0, int(statuses.pop("next_turn_block", 0)))
    if block:
        player = player.model_copy(update={"block": player.block + block})
        events.append(EffectEvent(kind="player_block", target_id=PLAYER_TARGET_ID, amount=block))

    star = max(0, int(statuses.pop("next_turn_star", 0)))
    if star:
        player = _set_player_resource(player, "star", _player_resource(player, "star") + star)
        events.append(
            EffectEvent(
                kind="player_resource_changed",
                target_id=PLAYER_TARGET_ID,
                amount=star,
                metadata={"resource": "star", "value": _player_resource(player, "star")},
            )
        )

    player = player.model_copy(update={"statuses": statuses})
    return combat.model_copy(update={"player": player}), tuple(events), extra_draw


def _apply_player_end_turn_status_block(
    player: PlayerState,
) -> tuple[PlayerState, tuple[EffectEvent, ...]]:
    events: list[EffectEvent] = []
    block_sources = (
        ("metallicize", _status_amount(player.statuses, "metallicize")),
        ("plated_armor", _status_amount(player.statuses, "plated_armor")),
    )
    total_block = sum(amount for _status, amount in block_sources if amount > 0)
    if total_block <= 0:
        return player, ()

    player = player.model_copy(update={"block": player.block + total_block})
    for status, amount in block_sources:
        if amount <= 0:
            continue
        events.append(
            EffectEvent(
                kind="player_block",
                target_id=PLAYER_TARGET_ID,
                amount=amount,
                metadata={"status": status},
            )
        )
    return player, tuple(events)


def _tick_end_turn_statuses(combat: CombatState) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    events: list[EffectEvent] = []
    player, player_events = _tick_player_statuses(combat.player)
    events.extend(player_events)
    monsters: list[MonsterState] = []
    for monster in combat.monsters:
        ticked, monster_events = _tick_monster_statuses(monster)
        monsters.append(ticked)
        events.extend(monster_events)
    return combat.model_copy(update={"player": player, "monsters": tuple(monsters)}), tuple(events)


def _tick_player_statuses(player: PlayerState) -> tuple[PlayerState, tuple[EffectEvent, ...]]:
    ticked, events = _tick_status_map(player, PLAYER_TARGET_ID)
    if isinstance(ticked, PlayerState):
        return ticked, events
    return player, events


def _tick_monster_statuses(monster: MonsterState) -> tuple[MonsterState, tuple[EffectEvent, ...]]:
    ticked, events = _tick_status_map(monster, monster.monster_id)
    if isinstance(ticked, MonsterState):
        return ticked, events
    return monster, events


def _tick_status_map(
    combatant: PlayerState | MonsterState,
    target_id: str,
) -> tuple[PlayerState | MonsterState, tuple[EffectEvent, ...]]:
    statuses = dict(combatant.statuses)
    events: list[EffectEvent] = []
    for status in ("weak", "vulnerable", "frail", "intangible", "retain_hand"):
        amount = statuses.get(status, 0)
        if amount <= 0:
            continue
        next_amount = amount - 1
        if next_amount > 0:
            statuses[status] = next_amount
        else:
            statuses.pop(status, None)
        events.append(
            EffectEvent(
                kind="status_ticked",
                target_id=target_id,
                amount=-1,
                metadata={"status": status, "remaining": max(0, next_amount)},
            )
        )
    for temporary_status in ("temporary_strength", "temporary_dexterity"):
        amount = statuses.pop(temporary_status, 0)
        if amount:
            events.append(
                EffectEvent(
                    kind="temporary_status_expired",
                    target_id=target_id,
                    amount=-amount,
                    metadata={"status": temporary_status},
                )
            )
    for down_status, base_status in (
        ("strength_down", "strength"),
        ("dexterity_down", "dexterity"),
    ):
        amount = statuses.pop(down_status, 0)
        if amount:
            statuses[base_status] = statuses.get(base_status, 0) - amount
            if statuses[base_status] == 0:
                statuses.pop(base_status, None)
            events.append(
                EffectEvent(
                    kind="temporary_status_expired",
                    target_id=target_id,
                    amount=-amount,
                    metadata={"status": down_status, "base_status": base_status},
                )
            )
    return combatant.model_copy(update={"statuses": statuses}), tuple(events)


def _apply_monster_poison(monster: MonsterState) -> tuple[MonsterState, tuple[EffectEvent, ...]]:
    poison = _status_amount(monster.statuses, "poison")
    if poison <= 0:
        return monster, ()
    statuses = dict(monster.statuses)
    if poison > 1:
        statuses["poison"] = poison - 1
    else:
        statuses.pop("poison", None)
    hp_loss = min(monster.hp, poison)
    monster = monster.model_copy(update={"hp": monster.hp - hp_loss, "statuses": statuses})
    events = [
        EffectEvent(
            kind="monster_poison_damage",
            target_id=monster.monster_id,
            amount=hp_loss,
            metadata={"remaining_poison": max(0, poison - 1)},
        )
    ]
    if monster.hp <= 0:
        events.append(EffectEvent(kind="monster_defeated", target_id=monster.monster_id))
    return monster, tuple(events)


def _damage_player(
    player: PlayerState,
    amount: int,
    source_id: str,
    *,
    relics: Sequence[str] = (),
) -> tuple[PlayerState, EffectEvent]:
    incoming = _incoming_player_damage(player, amount, relics=relics)
    blocked = min(player.block, incoming)
    hp_loss = max(0, incoming - blocked)
    intangible_applied = False
    if hp_loss > 1 and _status_amount(player.statuses, "intangible") > 0:
        hp_loss = 1
        intangible_applied = True
    statuses = dict(player.statuses)
    plated_armor_lost = 0
    if hp_loss > 0 and _status_amount(statuses, "plated_armor") > 0:
        plated_armor = _status_amount(statuses, "plated_armor")
        plated_armor_lost = 1
        if plated_armor > 1:
            statuses["plated_armor"] = plated_armor - 1
        else:
            statuses.pop("plated_armor", None)
    return (
        player.model_copy(
            update={
                "block": player.block - blocked,
                "hp": max(0, player.hp - hp_loss),
                "statuses": statuses,
            }
        ),
        EffectEvent(
            kind="player_damaged",
            source_id=source_id,
            target_id=PLAYER_TARGET_ID,
            amount=hp_loss,
            metadata={
                "incoming": incoming,
                "base": amount,
                "blocked": blocked,
                "intangible": intangible_applied,
                "plated_armor_lost": plated_armor_lost,
            },
        ),
    )


def _incoming_player_damage(
    player: PlayerState,
    amount: int,
    *,
    relics: Sequence[str] = (),
) -> int:
    if amount <= 0:
        return amount
    if _status_amount(player.statuses, "stance_wrath") > 0:
        amount *= 2
    if _status_amount(player.statuses, "vulnerable") > 0:
        multiplier = _damage_taken_vulnerable_multiplier(relics, player)
        return int(amount * multiplier)
    return amount


def _damage_monsters(
    monsters: Sequence[MonsterState],
    target_ids: Sequence[str],
    amount: int,
    source_id: str,
    *,
    relics: Sequence[str] = (),
) -> tuple[list[MonsterState], tuple[EffectEvent, ...]]:
    targets = set(target_ids)
    updated: list[MonsterState] = []
    events: list[EffectEvent] = []
    for monster in monsters:
        if monster.monster_id not in targets or monster.hp <= 0:
            updated.append(monster)
            continue
        incoming = _incoming_monster_damage(monster, amount, relics=relics)
        blocked = min(monster.block, incoming)
        hp_loss = max(0, incoming - blocked)
        intangible_applied = False
        if hp_loss > 1 and _status_amount(monster.statuses, "intangible") > 0:
            hp_loss = 1
            intangible_applied = True
        updated_monster = monster.model_copy(
            update={"block": monster.block - blocked, "hp": max(0, monster.hp - hp_loss)}
        )
        updated.append(updated_monster)
        events.append(
            EffectEvent(
                kind="monster_damaged",
                source_id=source_id,
                target_id=monster.monster_id,
                amount=hp_loss,
                metadata={
                    "incoming": incoming,
                    "base": amount,
                    "blocked": blocked,
                    "intangible": intangible_applied,
                },
            )
        )
        if updated_monster.hp <= 0:
            events.append(
                EffectEvent(
                    kind="monster_defeated",
                    source_id=source_id,
                    target_id=monster.monster_id,
                )
            )
    return updated, tuple(events)


def _apply_player_thorns_to_attacker(
    monster: MonsterState,
    player: PlayerState,
    *,
    source_id: str,
) -> tuple[MonsterState, tuple[EffectEvent, ...]]:
    thorns = _status_amount(player.statuses, "thorns")
    if thorns <= 0 or monster.hp <= 0:
        return monster, ()

    hp_loss = min(monster.hp, thorns)
    monster = monster.model_copy(update={"hp": monster.hp - hp_loss})
    events = [
        EffectEvent(
            kind="monster_damaged",
            source_id=PLAYER_TARGET_ID,
            target_id=monster.monster_id,
            amount=hp_loss,
            metadata={"status": "thorns", "base": thorns, "source_attack_id": source_id},
        )
    ]
    if monster.hp <= 0:
        events.append(
            EffectEvent(
                kind="monster_defeated",
                source_id=PLAYER_TARGET_ID,
                target_id=monster.monster_id,
                metadata={"status": "thorns"},
            )
        )
    return monster, tuple(events)


def _incoming_monster_damage(
    monster: MonsterState,
    amount: int,
    *,
    relics: Sequence[str] = (),
) -> int:
    if amount <= 0:
        return amount
    if _status_amount(monster.statuses, "vulnerable") > 0:
        multiplier = _damage_dealt_vulnerable_multiplier(relics, monster)
        return int(amount * multiplier)
    return amount


def _apply_status_effects(
    combat: CombatState,
    card: CardInstance,
    target_id: str | None,
    effect: Mapping[str, Any],
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    status_payload = effect.get("apply_status", effect.get("status"))
    if not isinstance(status_payload, Mapping):
        return combat, ()

    events: list[EffectEvent] = []
    player = combat.player
    monsters = list(combat.monsters)
    targets = _status_targets(combat, card, target_id, status_payload)
    status_values = _status_values(status_payload)

    if PLAYER_TARGET_ID in targets:
        statuses = dict(player.statuses)
        for status, value in status_values.items():
            if _is_stance_status(status):
                player = player.model_copy(update={"statuses": statuses})
                player, stance_events = _apply_player_stance_status(
                    player,
                    status,
                    source_id=card.instance_id,
                )
                statuses = dict(player.statuses)
                events.extend(stance_events)
                continue
            statuses, event = _apply_status_value(
                statuses,
                status,
                value,
                source_id=card.instance_id,
                target_id=PLAYER_TARGET_ID,
            )
            if event is not None:
                events.append(event)
        player = player.model_copy(update={"statuses": statuses})

    updated_monsters: list[MonsterState] = []
    for monster in monsters:
        if monster.monster_id not in targets:
            updated_monsters.append(monster)
            continue
        statuses = dict(monster.statuses)
        for status, value in status_values.items():
            statuses, event = _apply_status_value(
                statuses,
                status,
                value,
                source_id=card.instance_id,
                target_id=monster.monster_id,
            )
            if event is not None:
                events.append(event)
        updated_monsters.append(monster.model_copy(update={"statuses": statuses}))

    return (
        combat.model_copy(update={"player": player, "monsters": tuple(updated_monsters)}),
        tuple(events),
    )


def _apply_status_value(
    statuses: Mapping[str, int],
    status: str,
    value: int,
    *,
    source_id: str | None,
    target_id: str,
) -> tuple[dict[str, int], EffectEvent | None]:
    updated = dict(statuses)
    if value == 0:
        return updated, None

    if _status_blocked_by_artifact(updated, status, value):
        artifact = _status_amount(updated, "artifact")
        if artifact > 1:
            updated["artifact"] = artifact - 1
        else:
            updated.pop("artifact", None)
        return (
            updated,
            EffectEvent(
                kind="status_blocked_by_artifact",
                source_id=source_id,
                target_id=target_id,
                amount=value,
                metadata={"status": status},
            ),
        )

    next_amount = updated.get(status, 0) + value
    if next_amount == 0:
        updated.pop(status, None)
    else:
        updated[status] = next_amount
    return (
        updated,
        EffectEvent(
            kind="status_applied",
            source_id=source_id,
            target_id=target_id,
            amount=value,
            metadata={"status": status},
        ),
    )


def _status_blocked_by_artifact(
    statuses: Mapping[str, int],
    status: str,
    value: int,
) -> bool:
    if _status_amount(statuses, "artifact") <= 0:
        return False
    if status in _ARTIFACT_BLOCKED_STATUSES:
        return True
    return value < 0 and status not in _ARTIFACT_IGNORED_STATUSES


def _apply_combat_relic_resolution(
    combat: CombatState,
    resolution: CombatRelicResolution,
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    combat = combat.model_copy(update={"metadata": dict(combat.metadata)})
    events: list[EffectEvent] = []

    for marker in resolution.markers:
        combat, marker_events = _apply_combat_relic_marker(combat, marker)
        events.extend(marker_events)

    for blocker in resolution.blockers:
        events.append(
            EffectEvent(
                kind="combat_relic_blocked",
                source_id=blocker.source_id,
                message=blocker.reason,
                metadata={
                    "hook": blocker.hook.value,
                    "relic_id": blocker.relic_id,
                    "name": blocker.name,
                },
            )
        )

    return combat, tuple(events)


def _apply_combat_relic_marker(
    combat: CombatState,
    marker: CombatRelicMarker,
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    events: list[EffectEvent] = []
    player = combat.player
    monsters = list(combat.monsters)
    metadata = dict(combat.metadata)
    metadata = _metadata_with_relic_counter(metadata, marker)
    combat = combat.model_copy(update={"metadata": metadata})

    if marker.kind == "heal_player":
        heal = max(0, marker.amount or 0)
        old_hp = player.hp
        player = player.model_copy(update={"hp": min(player.max_hp, player.hp + heal)})
        events.append(_combat_relic_event(marker, "player_healed", player.hp - old_hp))
    elif marker.kind == "gain_block" and _marker_targets_player(marker):
        amount = max(0, marker.amount or 0)
        player = player.model_copy(update={"block": player.block + amount})
        events.append(_combat_relic_event(marker, "player_block", amount))
    elif marker.kind == "gain_energy" and _marker_targets_player(marker):
        amount = marker.amount or 0
        player = player.model_copy(update={"energy": max(0, player.energy + amount)})
        events.append(_combat_relic_event(marker, "energy_changed", amount))
    elif marker.kind == "draw_cards" and _marker_targets_player(marker):
        amount = max(0, marker.amount or 0)
        if amount:
            metadata["pending_relic_draw"] = (
                _metadata_int(metadata, "pending_relic_draw", 0) + amount
            )
        events.append(_combat_relic_event(marker, "relic_draw_scheduled", amount))
    elif marker.kind == "channel_orb" and _marker_targets_player(marker):
        orb_id = _normalized_orb_id(str(marker.metadata.get("orb", "lightning")))
        combat, channel_events = _channel_orb(
            combat.model_copy(
                update={
                    "player": player,
                    "monsters": tuple(monsters),
                    "metadata": metadata,
                }
            ),
            Random(0),
            marker.source_id or marker.relic_id,
            orb_id,
            target_id=None,
            relics=(),
        )
        player = combat.player
        monsters = list(combat.monsters)
        metadata = dict(combat.metadata)
        events.extend(channel_events)
    elif marker.kind in {"gain_status", "apply_status"}:
        player, monsters, status_events = _apply_combat_relic_status(player, monsters, marker)
        events.extend(status_events)
    elif marker.kind == "elite_monster_hp_multiplier":
        monsters, hp_events = _apply_combat_relic_enemy_hp_multiplier(monsters, marker)
        events.extend(hp_events)
    elif marker.kind == "turn_card_play_limit":
        limit = max(0, marker.amount or _metadata_int(marker.metadata, "limit", 0))
        metadata["turn_card_play_limit"] = limit
        events.append(_combat_relic_event(marker, "turn_card_play_limit_added", limit))
    elif marker.kind == "relic_counter_changed":
        events.append(
            _combat_relic_event(
                marker,
                "relic_counter_changed",
                _metadata_int(marker.metadata, "counter", marker.amount or 0),
            )
        )
    elif marker.kind == "periodic_energy_check":
        events.append(_combat_relic_event(marker, "combat_relic_pending", marker.amount))
    elif marker.kind.startswith("modify_"):
        events.append(_combat_relic_event(marker, "combat_relic_modifier_available", marker.amount))
    else:
        events.append(
            EffectEvent(
                kind="combat_relic_marker_stubbed",
                source_id=marker.source_id,
                target_id=marker.target_id,
                amount=marker.amount,
                metadata={
                    "kind": marker.kind,
                    "hook": marker.hook.value,
                    "relic_id": marker.relic_id,
                    **dict(marker.metadata),
                },
            )
        )

    return (
        combat.model_copy(
            update={
                "player": player,
                "monsters": tuple(monsters),
                "metadata": metadata,
            }
        ),
        tuple(events),
    )


def _apply_combat_relic_status(
    player: PlayerState,
    monsters: list[MonsterState],
    marker: CombatRelicMarker,
) -> tuple[PlayerState, list[MonsterState], tuple[EffectEvent, ...]]:
    status = _normalized_id(str(marker.metadata.get("status", "")))
    amount = marker.amount or 0
    if not status or amount == 0:
        return player, monsters, ()

    events: list[EffectEvent] = []
    if _marker_targets_player(marker):
        statuses, event = _apply_status_value(
            player.statuses,
            status,
            amount,
            source_id=marker.source_id,
            target_id=PLAYER_TARGET_ID,
        )
        player = player.model_copy(update={"statuses": statuses})
        if event is not None:
            events.append(
                event.model_copy(
                    update={
                        "metadata": {
                            **event.metadata,
                            "hook": marker.hook.value,
                            "relic_id": marker.relic_id,
                            "marker_kind": marker.kind,
                        }
                    }
                )
            )
        return player, monsters, tuple(events)

    target_ids = _combat_relic_marker_monster_targets(monsters, marker)
    updated: list[MonsterState] = []
    for monster in monsters:
        if monster.monster_id not in target_ids or monster.hp <= 0:
            updated.append(monster)
            continue
        statuses, event = _apply_status_value(
            monster.statuses,
            status,
            amount,
            source_id=marker.source_id,
            target_id=monster.monster_id,
        )
        updated.append(monster.model_copy(update={"statuses": statuses}))
        if event is not None:
            events.append(
                event.model_copy(
                    update={
                        "metadata": {
                            **event.metadata,
                            "hook": marker.hook.value,
                            "relic_id": marker.relic_id,
                            "marker_kind": marker.kind,
                        }
                    }
                )
            )
    return player, updated, tuple(events)


def _apply_combat_relic_enemy_hp_multiplier(
    monsters: list[MonsterState],
    marker: CombatRelicMarker,
) -> tuple[list[MonsterState], tuple[EffectEvent, ...]]:
    percent = max(0, marker.amount or 100)
    target_ids = _combat_relic_marker_monster_targets(monsters, marker)
    updated: list[MonsterState] = []
    events: list[EffectEvent] = []
    for monster in monsters:
        if monster.monster_id not in target_ids or monster.hp <= 0:
            updated.append(monster)
            continue
        max_hp = max(1, int(monster.max_hp * percent / 100))
        hp = max(1, min(max_hp, int(monster.hp * percent / 100)))
        updated.append(monster.model_copy(update={"max_hp": max_hp, "hp": hp}))
        events.append(
            _combat_relic_event(
                marker,
                "monster_max_hp_changed",
                max_hp - monster.max_hp,
                target_id=monster.monster_id,
            )
        )
    return updated, tuple(events)


def _combat_relic_marker_monster_targets(
    monsters: Sequence[MonsterState],
    marker: CombatRelicMarker,
) -> set[str]:
    if marker.target_id in {None, "all_enemies", "enemy"}:
        return {monster.monster_id for monster in monsters if monster.hp > 0}
    return {str(marker.target_id)}


def _marker_targets_player(marker: CombatRelicMarker) -> bool:
    return marker.target_id in {None, PLAYER_TARGET_ID, "player"}


def _combat_relic_event(
    marker: CombatRelicMarker,
    kind: str,
    amount: int | None,
    *,
    target_id: str | None = None,
    status: str | None = None,
) -> EffectEvent:
    metadata = {
        "hook": marker.hook.value,
        "relic_id": marker.relic_id,
        "marker_kind": marker.kind,
        **dict(marker.metadata),
    }
    if status is not None:
        metadata["status"] = status
    return EffectEvent(
        kind=kind,
        source_id=marker.source_id,
        target_id=target_id if target_id is not None else marker.target_id,
        amount=amount,
        metadata=metadata,
    )


def _metadata_with_relic_counter(
    metadata: Mapping[str, Any],
    marker: CombatRelicMarker,
) -> dict[str, Any]:
    updated = dict(metadata)
    counter: int | None = None
    if marker.kind == "relic_counter_changed":
        counter = _metadata_int(marker.metadata, "counter", marker.amount or 0)
    elif "next_counter" in marker.metadata:
        counter = _metadata_int(marker.metadata, "next_counter", 0)
    if counter is None:
        return updated
    counters = dict(_coerced_int_mapping(updated.get("relic_counters", {})))
    counters[marker.relic_id] = max(0, counter)
    updated["relic_counters"] = counters
    return updated


def _combat_relic_counters(combat: CombatState) -> dict[str, int]:
    return _coerced_int_mapping(combat.metadata.get("relic_counters", {}))


def _combat_relic_counters_for(
    relics: Sequence[str],
    combat: CombatState,
) -> dict[str, int]:
    counters = _combat_relic_counters(combat)
    for relic_id in relics:
        normalized = _normalized_id(str(relic_id))
        if normalized in {"happy_flower", "fake_happy_flower"}:
            counters.setdefault(normalized, 0)
    return counters


def _relic_counters_from_flags(state: RunState) -> dict[str, int]:
    return _coerced_int_mapping(state.flags.get("relic_counters", {}))


def _coerced_int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    parsed: dict[str, int] = {}
    for key, raw_value in value.items():
        with suppress(TypeError, ValueError):
            parsed[_normalized_id(str(key))] = max(0, int(raw_value))
    return parsed


def _reset_turn_combat_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(metadata)
    updated.pop("turn_card_play_limit", None)
    updated["attacks_played_this_turn"] = 0
    return updated


def _turn_card_play_limit_reached(combat: CombatState) -> bool:
    limit = _metadata_int(combat.metadata, "turn_card_play_limit", 0)
    return limit > 0 and len(combat.cards_played_this_turn) >= limit


def _metadata_int(metadata: Mapping[str, Any], key: str, default: int = 0) -> int:
    with suppress(TypeError, ValueError):
        return int(metadata.get(key, default))
    return default


def _damage_taken_vulnerable_multiplier(relics: Sequence[str], player: PlayerState) -> float:
    result = resolve_damage_taken_relics(
        relics,
        player_statuses=player.statuses,
        target_id=PLAYER_TARGET_ID,
    )
    for marker in result.markers:
        if marker.kind == "modify_vulnerable_damage_taken" and marker.amount:
            return marker.amount / 100
    return 1.5


def _damage_dealt_vulnerable_multiplier(relics: Sequence[str], monster: MonsterState) -> float:
    result = resolve_damage_dealt_relics(
        relics,
        target_statuses=monster.statuses,
        target_id=monster.monster_id,
    )
    for marker in result.markers:
        if marker.kind == "modify_vulnerable_damage_dealt" and marker.amount:
            return marker.amount / 100
    return 1.5


def _current_encounter_type(state: RunState) -> str | None:
    if state.map is None or state.map.current_node_id is None:
        return None
    node = state.map.node_by_id.get(state.map.current_node_id)
    return node.kind.value if node is not None else None


def _phase_after_combat(combat: CombatState) -> RunPhase:
    if combat.player.hp <= 0:
        return RunPhase.FAILED
    if not _alive_monsters(combat):
        return RunPhase.REWARD
    return RunPhase.COMBAT


def _state_after_combat(state: RunState, combat: CombatState, rng_state: RngState) -> RunState:
    phase = _phase_after_combat(combat)
    reward = state.reward
    events_rng_state = rng_state
    flags = dict(state.flags)
    if phase == RunPhase.REWARD and state.phase == RunPhase.COMBAT:
        combat_end_relics = _resolve_combat_relic_trigger(
            GameTrigger.COMBAT_END,
            state.relics,
            player_hp=combat.player.hp,
            player_max_hp=combat.player.max_hp,
            player_block=combat.player.block,
            player_statuses=combat.player.statuses,
            relic_counters=_combat_relic_counters(combat),
        )
        combat, combat_end_relic_events = _apply_combat_relic_resolution(
            combat,
            combat_end_relics,
        )
        if combat_end_relic_events:
            combat = combat.model_copy(
                update={"last_events": combat.last_events + combat_end_relic_events}
            )
        flags["relic_counters"] = _combat_relic_counters(combat)
        rng = random_from_state(rng_state)
        reward, _events, flags = _combat_reward_state(state, rng)
        reward, flags, delayed_reward_events = _apply_due_delayed_event_rewards(
            state,
            rng,
            reward,
            flags,
        )
        if delayed_reward_events:
            combat = combat.model_copy(
                update={"last_events": combat.last_events + delayed_reward_events}
            )
        flags["relic_counters"] = _combat_relic_counters(combat)
        events_rng_state = capture_random_state(rng)
    elif phase != RunPhase.REWARD:
        reward = None
    return state.model_copy(
        update={
            "combat": combat,
            "rng": events_rng_state,
            "phase": phase,
            "player": combat.player,
            "reward": reward,
            "flags": flags,
        }
    )


def _combat_reward_state(
    state: RunState,
    rng: Random,
) -> tuple[RewardState, tuple[EffectEvent, ...], dict[str, Any]]:
    encounter = _combat_reward_encounter(state)
    bundle = draw_combat_reward(
        rng,
        card_pool=_combat_card_pool(state),
        potion_pool=_combat_potion_pool(state),
        relic_pool=_treasure_relic_pool(state),
        boss_relic_pool=_boss_relic_pool(state),
        context=CombatRewardContext(
            character_id=state.character_id,
            encounter=encounter,
            act=state.act,
            floor=state.floor,
            ascension_level=state.ascension,
            owned_relics=state.relics,
        ),
        pity_state=RewardPityState(
            card_non_rare_count=_flag_int(state, "card_non_rare_count", 0),
            potion_chance_bonus=_flag_int(state, "potion_chance_bonus", 0),
        ),
        card_count=_combat_reward_card_count(state, encounter),
        relic_count=_combat_reward_relic_count(state, encounter),
    )
    flags = dict(state.flags)
    flags["card_non_rare_count"] = bundle.pity_state.card_non_rare_count
    flags["potion_chance_bonus"] = bundle.pity_state.potion_chance_bonus

    gold = bundle.gold
    if "combat_reward_gold" in state.flags:
        gold = _flag_int(state, "combat_reward_gold", bundle.gold)

    relic_id = _explicit_reward_id(state, "combat_reward_relic_id")
    relic_ids = _combat_reward_relic_ids(state, default_relic_ids=bundle.relic_ids)
    card_ids = _fixed_reward_card_ids(state, "combat_reward_card_ids")
    card_options = _flag_str_sequence(state, "combat_reward_card_options") or bundle.card_ids
    card_option_groups, card_group_rarities, card_pity_state = _combat_extra_card_option_groups(
        state,
        rng,
        encounter,
        bundle.pity_state,
    )
    flags["card_non_rare_count"] = card_pity_state.card_non_rare_count
    potion_id = _combat_reward_potion_id(state, rng, default_potion_id=bundle.potion_id)
    potion_ids = _reward_potion_ids(
        state,
        rng,
        explicit_key="combat_reward_potion_ids",
        count_key="combat_reward_extra_potion_count",
        pool_key="combat_reward_potion_pool",
    )

    reward = RewardState(
        reward_id=f"combat:{len(state.room_history)}",
        source="combat",
        forced=_flag_bool(state, "combat_reward_forced", False),
        gold=gold,
        relic_id=relic_id,
        relic_ids=relic_ids,
        card_ids=card_ids,
        card_options=card_options,
        card_option_groups=card_option_groups,
        potion_id=potion_id,
        potion_ids=potion_ids,
        metadata={
            "encounter": encounter.value,
            "base_gold": bundle.base_gold,
            "card_rarities": bundle.card_rarities,
            "card_group_rarities": card_group_rarities,
            "relic_rarities": bundle.relic_rarities,
            "potion_drop": _combat_potion_roll_metadata(
                state,
                bundle,
                potion_id=potion_id,
            ),
            "extra_potion_ids": potion_ids,
            "potion_slots": _potion_capacity(state),
            "current_potions": len(state.potions),
        },
    )
    return reward, _reward_generated_events(reward), flags


def _combat_extra_card_option_groups(
    state: RunState,
    rng: Random,
    encounter: EncounterType,
    pity_state: RewardPityState,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...], RewardPityState]:
    if encounter is not EncounterType.NORMAL or not _has_relic(state, "prayer_wheel"):
        return (), (), pity_state
    cards, rarities, next_pity = draw_card_reward_options(
        rng,
        card_pool=_combat_card_pool(state),
        context=CombatRewardContext(
            character_id=state.character_id,
            encounter=encounter,
            act=state.act,
            floor=state.floor,
            ascension_level=state.ascension,
            owned_relics=state.relics,
        ),
        pity_state=pity_state,
        card_count=_card_reward_choice_count(state),
    )
    if not cards:
        return (), (), next_pity
    return (cards,), (rarities,), next_pity


def _apply_due_delayed_event_rewards(
    state: RunState,
    rng: Random,
    reward: RewardState,
    flags: Mapping[str, Any],
) -> tuple[RewardState, dict[str, Any], tuple[EffectEvent, ...]]:
    queue = _delayed_event_rewards_from_flags(flags)
    if not queue:
        return reward, dict(flags), ()

    pending: list[dict[str, Any]] = []
    events: list[EffectEvent] = []
    next_reward = reward
    for item in queue:
        remaining = max(0, _nonnegative_int(item.get("remaining_combats"), 0) - 1)
        ticked = {**item, "remaining_combats": remaining}
        tick_metadata = _delayed_reward_event_metadata(ticked)
        if remaining > 0:
            pending.append(ticked)
            events.append(
                EffectEvent(
                    kind="delayed_event_reward_pending",
                    amount=remaining,
                    metadata=tick_metadata,
                )
            )
            continue

        next_reward, resolved, ready_events = _add_delayed_reward_to_reward_screen(
            state,
            rng,
            next_reward,
            ticked,
        )
        events.extend(ready_events)
        if not resolved:
            pending.append(ticked)

    next_flags = _write_delayed_event_rewards(flags, pending)
    return next_reward, next_flags, tuple(events)


def _add_delayed_reward_to_reward_screen(
    state: RunState,
    rng: Random,
    reward: RewardState,
    item: Mapping[str, Any],
) -> tuple[RewardState, bool, tuple[EffectEvent, ...]]:
    reward_kind = _normalized_id(str(item.get("reward_kind", "")))
    count = _nonnegative_int(item.get("count"), 1)
    metadata = _delayed_reward_event_metadata(item)
    if count <= 0:
        return reward, True, (
            EffectEvent(
                kind="delayed_event_reward_skipped",
                metadata={**metadata, "reason": "empty_reward"},
            ),
        )

    if reward_kind == "random_relic":
        relic_ids = _draw_delayed_reward_relic_ids(
            state,
            rng,
            count,
            qualifier=item.get("qualifier"),
            existing_reward_relics=_reward_screen_relic_ids(reward),
        )
        next_reward = _reward_with_delayed_metadata(
            reward.model_copy(update={"relic_ids": reward.relic_ids + relic_ids}),
            {**metadata, "relic_ids": relic_ids},
        )
        return next_reward, True, (
            EffectEvent(
                kind="delayed_event_reward_ready",
                amount=len(relic_ids),
                metadata={**metadata, "relic_ids": relic_ids},
            ),
        )

    if reward_kind == "fixed_relic":
        item_id = item.get("item_id")
        if not item_id:
            return reward, False, (
                EffectEvent(
                    kind="delayed_event_reward_blocked",
                    metadata={**metadata, "reason": "missing_item_id"},
                ),
            )
        relic_ids = (_normalized_id(str(item_id)),)
        next_reward = _reward_with_delayed_metadata(
            reward.model_copy(update={"relic_ids": reward.relic_ids + relic_ids}),
            {**metadata, "relic_ids": relic_ids},
        )
        return next_reward, True, (
            EffectEvent(
                kind="delayed_event_reward_ready",
                amount=len(relic_ids),
                metadata={**metadata, "relic_ids": relic_ids},
            ),
        )

    if reward_kind == "random_potion":
        marker = EventFlowMarker(kind=EventFlowMarkerKind.RANDOM_POTION, count=count)
        potion_ids = _draw_event_flow_potion_ids(state, rng, marker)
        next_reward = _reward_with_delayed_metadata(
            reward.model_copy(update={"potion_ids": reward.potion_ids + potion_ids}),
            {**metadata, "potion_ids": potion_ids},
        )
        return next_reward, True, (
            EffectEvent(
                kind="delayed_event_reward_ready",
                amount=len(potion_ids),
                metadata={**metadata, "potion_ids": potion_ids},
            ),
        )

    if reward_kind == "fixed_potion":
        item_id = item.get("item_id")
        if not item_id:
            return reward, False, (
                EffectEvent(
                    kind="delayed_event_reward_blocked",
                    metadata={**metadata, "reason": "missing_item_id"},
                ),
            )
        potion_ids = (_normalized_id(str(item_id)),)
        next_reward = _reward_with_delayed_metadata(
            reward.model_copy(update={"potion_ids": reward.potion_ids + potion_ids}),
            {**metadata, "potion_ids": potion_ids},
        )
        return next_reward, True, (
            EffectEvent(
                kind="delayed_event_reward_ready",
                amount=len(potion_ids),
                metadata={**metadata, "potion_ids": potion_ids},
            ),
        )

    return reward, False, (
        EffectEvent(
            kind="delayed_event_reward_blocked",
            metadata={
                **metadata,
                "reason": "unsupported_reward_kind",
                "supported_reward_kinds": (
                    "fixed_potion",
                    "fixed_relic",
                    "random_potion",
                    "random_relic",
                ),
            },
        ),
    )


def _draw_delayed_reward_relic_ids(
    state: RunState,
    rng: Random,
    count: int,
    *,
    qualifier: Any,
    existing_reward_relics: Sequence[str],
) -> tuple[str, ...]:
    pool = _treasure_relic_pool(state)
    normalized_qualifier = _normalized_id(str(qualifier or ""))
    if normalized_qualifier in {rarity.value for rarity in RelicRarity}:
        pool = tuple(relic for relic in pool if relic.rarity.value == normalized_qualifier)
    draw_state = state.model_copy(update={"relics": state.relics + tuple(existing_reward_relics)})
    return _draw_event_relic_ids_from_pool(draw_state, rng, count, pool)


def _reward_screen_relic_ids(reward: RewardState) -> tuple[str, ...]:
    relic_ids = list(reward.relic_ids)
    if reward.relic_id is not None:
        relic_ids.append(reward.relic_id)
    return tuple(relic_ids)


def _reward_with_delayed_metadata(
    reward: RewardState,
    entry: Mapping[str, Any],
) -> RewardState:
    metadata = dict(reward.metadata)
    raw_entries = metadata.get("delayed_rewards", ())
    if isinstance(raw_entries, Sequence) and not isinstance(raw_entries, (bytes, bytearray, str)):
        entries = [dict(item) for item in raw_entries if isinstance(item, Mapping)]
    elif isinstance(raw_entries, Mapping):
        entries = [dict(raw_entries)]
    else:
        entries = []
    entries.append(dict(entry))
    metadata["delayed_rewards"] = entries
    return reward.model_copy(update={"metadata": metadata})


def _delayed_reward_event_metadata(item: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        "reward_kind": _normalized_id(str(item.get("reward_kind", ""))),
        "count": _nonnegative_int(item.get("count"), 1),
        "remaining_combats": _nonnegative_int(item.get("remaining_combats"), 0),
        "qualifier": item.get("qualifier"),
        "item_id": item.get("item_id"),
        "source_event_id": item.get("source_event_id"),
        "description": str(item.get("description", "")),
    }
    marker_metadata = _mapping_from(item.get("metadata"))
    if marker_metadata:
        metadata["marker_metadata"] = dict(marker_metadata)
    return metadata


def _combat_reward_encounter(state: RunState) -> EncounterType:
    raw_encounter = state.flags.get("combat_reward_encounter", state.flags.get("combat_encounter"))
    if raw_encounter is not None:
        with suppress(ValueError):
            return EncounterType(_normalized_id(str(raw_encounter)))

    node = _current_map_node(state)
    if node is None:
        return EncounterType.NORMAL
    if node.kind == RoomKind.ELITE:
        return EncounterType.ELITE
    if node.kind == RoomKind.BOSS:
        return EncounterType.BOSS
    if node.kind == RoomKind.EVENT:
        return EncounterType.EVENT
    return EncounterType.NORMAL


def _combat_reward_card_count(
    state: RunState,
    encounter: EncounterType,
) -> int | None:
    if _flag_str_sequence(state, "combat_reward_card_options"):
        return 0
    if "combat_reward_card_count" in state.flags:
        return max(0, _flag_int(state, "combat_reward_card_count", 3))
    if encounter is EncounterType.BOSS and "boss_reward_card_count" in state.flags:
        return max(0, _flag_int(state, "boss_reward_card_count", 3))
    if encounter is EncounterType.ELITE and "elite_reward_card_count" in state.flags:
        return max(0, _flag_int(state, "elite_reward_card_count", 3))
    if encounter is EncounterType.EVENT:
        return 0
    return _card_reward_choice_count(state)


def _combat_reward_relic_count(
    state: RunState,
    encounter: EncounterType,
) -> int | None:
    if (
        _explicit_reward_id(state, "combat_reward_relic_id") is not None
        or _flag_str_sequence(state, "combat_reward_relic_ids")
        or _flag_str_sequence(state, "event_combat_reward_relic_ids")
        or _is_fake_merchant_combat(state)
    ):
        return 0
    if "combat_reward_relic_count" in state.flags:
        return max(0, _flag_int(state, "combat_reward_relic_count", 0))
    if encounter is EncounterType.BOSS and "boss_reward_relic_count" in state.flags:
        return max(0, _flag_int(state, "boss_reward_relic_count", 1))
    if encounter is EncounterType.ELITE and "elite_reward_relic_count" in state.flags:
        return max(0, _flag_int(state, "elite_reward_relic_count", 1))
    if encounter is EncounterType.EVENT:
        return 0
    if encounter is EncounterType.ELITE and _has_relic(state, "black_star"):
        return 2
    return None


def _combat_reward_relic_ids(
    state: RunState,
    *,
    default_relic_ids: Sequence[str],
) -> tuple[str, ...]:
    if _is_fake_merchant_combat(state):
        return fake_merchant_reward_relic_ids(
            _cached_relic_source_rows(state),
            unsold_relic_ids=_flag_str_sequence(state, "fake_merchant_unsold_relic_ids"),
        )

    explicit = _flag_str_sequence(state, "combat_reward_relic_ids")
    if explicit:
        return tuple(_normalized_id(relic_id) for relic_id in explicit)

    event_explicit = _flag_str_sequence(state, "event_combat_reward_relic_ids")
    if event_explicit:
        return tuple(_normalized_id(relic_id) for relic_id in event_explicit)

    if _explicit_reward_id(state, "combat_reward_relic_id") is not None:
        return ()
    return tuple(default_relic_ids)


def _combat_reward_potion_id(
    state: RunState,
    rng: Random,
    *,
    default_potion_id: str | None,
) -> str | None:
    if state.flags.get("combat_reward_potion_id") is not None:
        return str(state.flags["combat_reward_potion_id"])
    if "combat_reward_potion_chance_percent" in state.flags:
        return _reward_potion_id(
            state,
            rng,
            explicit_key="combat_reward_potion_id",
            pool_key="combat_reward_potion_pool",
            chance_key="combat_reward_potion_chance_percent",
            default_chance=40,
        )
    return default_potion_id


def _combat_potion_roll_metadata(
    state: RunState,
    bundle: Any,
    *,
    potion_id: str | None,
) -> Mapping[str, Any] | None:
    if "combat_reward_potion_chance_percent" in state.flags:
        return {
            "dropped": potion_id is not None,
            "override": True,
            "chance_percent": _flag_int(state, "combat_reward_potion_chance_percent", 0),
        }
    potion_roll = getattr(bundle, "potion_roll", None)
    if potion_roll is None:
        return None
    return {
        "dropped": potion_roll.dropped,
        "base_chance_percent": potion_roll.base_chance_percent,
        "effective_chance_tenths": potion_roll.effective_chance_tenths,
        "elite_bonus_tenths": potion_roll.elite_bonus_tenths,
        "roll_tenths": potion_roll.roll_tenths,
        "next_potion_chance_bonus": potion_roll.state.potion_chance_bonus,
    }


def _combat_card_pool(state: RunState) -> tuple[Any, ...]:
    return build_combat_card_pool(_raw_card_pool_items(state), character_id=state.character_id)


def _raw_card_pool_items(state: RunState) -> tuple[Any, ...]:
    return (
        _source_items(state.flags.get("combat_reward_card_pool"))
        or _source_items(state.flags.get("card_pool"))
        or _source_items(state.flags.get("card_library"))
        or _source_items(state.flags.get("cards"))
        or _cached_source_rows(state, "cards")
    )


def _combat_potion_pool(state: RunState) -> tuple[str, ...]:
    raw_pool = (
        _source_items(state.flags.get("combat_reward_potion_pool"))
        or _source_items(state.flags.get("potion_pool"))
        or _cached_source_rows(state, "potions")
    )
    pool = build_combat_potion_pool(raw_pool, character_id=state.character_id)
    return pool or ("fire_potion", "skill_potion", "essence_of_steel")


def _boss_relic_pool(state: RunState) -> tuple[Any, ...]:
    raw_pool = (
        _source_items(state.flags.get("combat_boss_relic_pool"))
        or _source_items(state.flags.get("boss_relic_pool"))
        or _source_items(state.flags.get("relic_pool"))
        or _cached_relic_source_rows(state)
    )
    return build_boss_relic_pool(raw_pool, character_id=state.character_id)


def _current_map_node(state: RunState) -> MapNodeState | None:
    if state.map is None or state.map.current_node_id is None:
        return None
    return state.map.node_by_id.get(state.map.current_node_id)


def _is_fake_merchant_combat(state: RunState) -> bool:
    event_id = _normalized_id(
        str(
            state.flags.get(
                "combat_reward_event_id",
                state.flags.get("event_fight_id", ""),
            )
        )
    )
    return (
        event_id in {"fake_merchant", "the_merchant", "the_merchant???"}
        or _flag_bool(state, "fake_merchant_combat", False)
        or "fake_merchant_unsold_relic_ids" in state.flags
    )


def _treasure_reward_state(
    state: RunState,
    node: MapNodeState,
    rng: Random,
) -> tuple[RewardState, tuple[EffectEvent, ...]]:
    treasure_reward = draw_treasure_reward(
        rng,
        _treasure_relic_pool(state),
        TreasureContext(
            character_id=state.character_id,
            act=node.act,
            floor=node.floor,
            ascension_level=state.ascension,
            owned_relics=state.relics,
            opened_chests=_flag_int(state, "treasure_chests_opened", 0),
        ),
    )
    relic_id = treasure_reward.relic_id
    explicit_relic_id = _explicit_reward_id(state, "treasure_reward_relic_id")
    if explicit_relic_id is not None and not treasure_reward.empty:
        relic_id = _normalized_id(explicit_relic_id)

    gold = treasure_reward.gold
    if "treasure_reward_gold" in state.flags and not treasure_reward.empty:
        gold = _flag_int(state, "treasure_reward_gold", treasure_reward.gold)

    reward = RewardState(
        reward_id=f"treasure:{node.node_id}",
        source="treasure",
        forced=_flag_bool(state, "treasure_reward_forced", False),
        gold=gold,
        relic_id=relic_id,
        card_options=_flag_str_sequence(state, "treasure_reward_card_options"),
        metadata={
            "node_id": node.node_id,
            "base_gold": treasure_reward.base_gold,
            "relic_rarity": treasure_reward.relic_rarity.value
            if treasure_reward.relic_rarity is not None
            else None,
            "source_relic_id": treasure_reward.source_relic_id,
            "empty": treasure_reward.empty,
            "empty_reason": treasure_reward.empty_reason,
            "treasure_chests_opened": _flag_int(state, "treasure_chests_opened", 0),
            "next_treasure_chests_opened": _flag_int(
                state,
                "treasure_chests_opened",
                0,
            )
            + 1,
        },
    )
    events = _reward_generated_events(reward)
    if treasure_reward.empty:
        events += (
            EffectEvent(
                kind="treasure_chest_empty",
                source_id=treasure_reward.source_relic_id,
                target_id=node.node_id,
                metadata={"reason": treasure_reward.empty_reason},
            ),
        )
    return reward, events


def _event_reward_state(
    state: RunState,
    node: MapNodeState,
    rng: Random,
) -> tuple[RewardState | None, tuple[EffectEvent, ...]]:
    gold = _flag_int(state, "event_reward_gold", 0)
    relic_id = _explicit_reward_id(state, "event_reward_relic_id")
    relic_ids = tuple(
        _normalized_id(relic_id)
        for relic_id in _flag_str_sequence(state, "event_reward_relic_ids")
    )
    relic_ids += _event_random_relic_ids(state, rng)
    card_ids = _fixed_reward_card_ids(state, "event_reward_card_ids")
    card_options = _flag_str_sequence(state, "event_reward_card_options")
    potion_id = _reward_potion_id(
        state,
        rng,
        explicit_key="event_reward_potion_id",
        pool_key="event_reward_potion_pool",
        chance_key="event_reward_potion_chance_percent",
        default_chance=0,
    )
    potion_ids = _reward_potion_ids(
        state,
        rng,
        explicit_key="event_reward_potion_ids",
        count_key="event_reward_potion_count",
        pool_key="event_reward_potion_pool",
    )
    if (
        gold <= 0
        and relic_id is None
        and not relic_ids
        and not card_ids
        and not card_options
        and potion_id is None
        and not potion_ids
    ):
        return None, ()

    reward = RewardState(
        reward_id=f"event:{node.node_id}",
        source="event",
        forced=_flag_bool(state, "event_reward_forced", False),
        gold=gold,
        relic_id=relic_id,
        relic_ids=relic_ids,
        card_ids=card_ids,
        card_options=card_options,
        potion_id=potion_id,
        potion_ids=potion_ids,
        metadata={
            "node_id": node.node_id,
            "potion_slots": _potion_capacity(state),
            "current_potions": len(state.potions),
        },
    )
    return reward, _reward_generated_events(reward, node_id=node.node_id)


def _treasure_relic_pool(state: RunState) -> tuple[TreasureRelic, ...]:
    raw_pool = (
        _source_sequence(state.flags.get("treasure_relic_pool"))
        or _source_sequence(state.flags.get("relic_pool"))
        or _cached_relic_source_rows(state)
    )
    return build_treasure_relic_pool(raw_pool, character_id=state.character_id)


def _source_sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _source_items(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        return tuple(value.values())
    return _source_sequence(value)


def _cached_relic_source_rows(state: RunState) -> tuple[Any, ...]:
    return _cached_source_rows(state, "relics")


def _cached_source_rows(state: RunState, dataset: str) -> tuple[Any, ...]:
    cache_dirs: list[Path] = []
    configured_cache_dir = state.flags.get("data_cache_dir")
    if configured_cache_dir is not None:
        cache_dirs.append(Path(str(configured_cache_dir)))
    cache_dirs.append(Path("data") / "cache")
    cache_dirs.append(Path(__file__).resolve().parents[3] / "data" / "cache")

    seen: set[Path] = set()
    for cache_dir in cache_dirs:
        cache_dir = cache_dir.resolve()
        if cache_dir in seen:
            continue
        seen.add(cache_dir)
        with suppress(OSError, KeyError, FileNotFoundError):
            payload = load_cached_json(cache_dir, dataset, "eng")
            return _source_sequence(payload)
    return ()


def _reward_generated_events(
    reward: RewardState,
    *,
    node_id: str | None = None,
) -> tuple[EffectEvent, ...]:
    events: list[EffectEvent] = []
    metadata = {"reward_id": reward.reward_id, "reward_source": reward.source}
    if node_id is not None:
        metadata["node_id"] = node_id
    if reward.gold > 0:
        events.append(
            EffectEvent(
                kind="reward_gold_generated",
                amount=reward.gold,
                metadata=metadata,
            )
        )
    if reward.relic_id is not None:
        events.append(
            EffectEvent(
                kind="reward_relic_generated",
                target_id=reward.relic_id,
                metadata=metadata,
            )
        )
    for index, relic_id in enumerate(reward.relic_ids):
        events.append(
            EffectEvent(
                kind="reward_relic_generated",
                target_id=relic_id,
                metadata={**metadata, "reward_relic_index": index},
            )
        )
    if reward.card_options:
        events.append(
            EffectEvent(
                kind="reward_cards_generated",
                metadata={**metadata, "card_options": reward.card_options},
            )
        )
    for group_index, group in enumerate(reward.card_option_groups):
        events.append(
            EffectEvent(
                kind="reward_card_group_generated",
                metadata={
                    **metadata,
                    "card_group_index": group_index,
                    "card_options": group,
                },
            )
        )
    for index, card_id in enumerate(reward.card_ids):
        events.append(
            EffectEvent(
                kind="reward_card_generated",
                target_id=card_id,
                metadata={**metadata, "reward_card_index": index, "fixed": True},
            )
        )
    if reward.potion_id is not None:
        events.append(
            EffectEvent(
                kind="reward_potion_generated",
                target_id=reward.potion_id,
                metadata=metadata,
            )
        )
    for index, potion_id in enumerate(reward.potion_ids):
        events.append(
            EffectEvent(
                kind="reward_potion_generated",
                target_id=potion_id,
                metadata={**metadata, "reward_potion_index": index},
            )
        )
    return tuple(events)


def _explicit_reward_id(state: RunState, key: str) -> str | None:
    value = state.flags.get(key)
    return str(value) if value is not None else None


def _fixed_reward_card_ids(state: RunState, key: str) -> tuple[str, ...]:
    return tuple(_normalized_id(card_id) for card_id in _flag_str_sequence(state, key))


def _event_random_relic_ids(state: RunState, rng: Random) -> tuple[str, ...]:
    count = _flag_int(state, "event_reward_relic_count", 0)
    if count <= 0:
        return ()

    relic_ids: list[str] = []
    for _ in range(count):
        reward = draw_treasure_reward(
            rng,
            _treasure_relic_pool(state),
            TreasureContext(
                character_id=state.character_id,
                act=state.act,
                floor=state.floor,
                ascension_level=state.ascension,
                owned_relics=state.relics + tuple(relic_ids),
                opened_chests=1,
            ),
        )
        if reward.relic_id is None:
            break
        relic_ids.append(reward.relic_id)
    return tuple(relic_ids)


def _treasure_relic_id(state: RunState, node: MapNodeState, rng: Random) -> str:
    pool = _flag_str_sequence(state, "treasure_relic_pool") or _flag_str_sequence(
        state,
        "relic_pool",
    )
    if pool:
        candidates = [relic_id for relic_id in pool if relic_id not in state.relics]
        return rng.choice(candidates or list(pool))
    return f"treasure_relic_{node.act}_{node.floor}_{node.lane}"


def _reward_potion_id(
    state: RunState,
    rng: Random,
    *,
    explicit_key: str,
    pool_key: str,
    chance_key: str,
    default_chance: int,
) -> str | None:
    explicit = state.flags.get(explicit_key)
    if explicit is not None:
        return str(explicit)

    chance = _flag_int(state, chance_key, default_chance)
    if chance <= 0 or rng.randrange(100) >= min(100, chance):
        return None

    pool = _reward_potion_pool(state, pool_key)
    if not pool:
        pool = ("fire_potion", "skill_potion", "essence_of_steel")
    return rng.choice(tuple(pool))


def _reward_potion_ids(
    state: RunState,
    rng: Random,
    *,
    explicit_key: str,
    count_key: str,
    pool_key: str,
) -> tuple[str, ...]:
    explicit = _flag_str_sequence(state, explicit_key)
    if explicit:
        return tuple(_normalized_id(potion_id) for potion_id in explicit)

    count = _flag_int(state, count_key, 0)
    if count <= 0:
        return ()

    pool = _reward_potion_pool(state, pool_key)
    if not pool:
        pool = ("fire_potion", "skill_potion", "essence_of_steel")
    return tuple(rng.choice(tuple(pool)) for _ in range(count))


def _reward_potion_pool(state: RunState, pool_key: str) -> tuple[str, ...]:
    raw_pool = (
        _source_items(state.flags.get(pool_key))
        or _source_items(state.flags.get("potion_pool"))
        or _cached_source_rows(state, "potions")
    )
    return build_combat_potion_pool(raw_pool, character_id=state.character_id)


def _alive_monsters(combat: CombatState) -> tuple[MonsterState, ...]:
    return tuple(monster for monster in combat.monsters if monster.hp > 0)


def _legal_target_ids(combat: CombatState, card: CardInstance) -> tuple[str | None, ...]:
    if card.target == TargetType.NONE:
        return (None,)
    if card.target == TargetType.SELF:
        return (PLAYER_TARGET_ID,)
    if card.target == TargetType.ALL_ENEMIES:
        return (None,)
    if card.target == TargetType.ANY:
        return (PLAYER_TARGET_ID,) + tuple(
            monster.monster_id for monster in _alive_monsters(combat)
        )
    return tuple(monster.monster_id for monster in _alive_monsters(combat))


def _effect_enemy_targets(
    combat: CombatState,
    card: CardInstance,
    target_id: str | None,
    *,
    all_enemies: bool,
) -> tuple[str, ...]:
    if all_enemies or card.target == TargetType.ALL_ENEMIES:
        return tuple(monster.monster_id for monster in _alive_monsters(combat))
    if target_id and target_id != PLAYER_TARGET_ID:
        return (target_id,)
    alive = _alive_monsters(combat)
    if len(alive) == 1 and card.target == TargetType.ENEMY:
        return (alive[0].monster_id,)
    return ()


def _energy_cost(card: CardInstance, available_energy: int) -> int:
    if card.cost is None:
        return 0
    if card.cost < 0:
        return max(0, available_energy)
    return max(0, card.cost)


def _can_pay_resource_costs(
    player: PlayerState,
    card: CardInstance,
) -> bool:
    for resource, cost in _card_resource_costs(
        card,
        available_star=_player_resource(player, "star"),
    ).items():
        if _player_resource(player, resource) < cost:
            return False
    return True


def _pay_card_resource_costs(
    player: PlayerState,
    card: CardInstance,
) -> tuple[PlayerState, tuple[EffectEvent, ...]]:
    events: list[EffectEvent] = []
    star_spent = 0
    for resource, cost in _card_resource_costs(
        card,
        available_star=_player_resource(player, "star"),
    ).items():
        if cost <= 0:
            continue
        player = _set_player_resource(player, resource, _player_resource(player, resource) - cost)
        if resource == "star":
            star_spent += cost
        events.append(
            EffectEvent(
                kind="player_resource_spent",
                source_id=card.instance_id,
                target_id=PLAYER_TARGET_ID,
                amount=cost,
                metadata={"resource": resource, "value": _player_resource(player, resource)},
            )
        )
    if star_spent and _status_amount(player.statuses, "child_of_the_stars") > 0:
        block = star_spent * _status_amount(player.statuses, "child_of_the_stars")
        player = player.model_copy(update={"block": player.block + block})
        events.append(
            EffectEvent(
                kind="player_block",
                source_id=card.instance_id,
                target_id=PLAYER_TARGET_ID,
                amount=block,
                metadata={"status": "child_of_the_stars"},
            )
        )
    return player, tuple(events)


def _card_resource_costs(card: CardInstance, *, available_star: int) -> dict[str, int]:
    raw_star_cost = card.custom.get("star_cost", 0)
    if raw_star_cost in (None, "", 0):
        return {}
    if isinstance(raw_star_cost, str) and raw_star_cost.strip().lower() == "x":
        return {"star": max(0, available_star)}
    try:
        return {"star": max(0, int(raw_star_cost))}
    except (TypeError, ValueError):
        return {}


def _player_resource(player: PlayerState, resource: str) -> int:
    normalized = _normalized_resource_id(resource)
    if normalized == "star":
        return max(
            0,
            int(player.resources.get("star", player.resources.get("stars", 0))),
        )
    return max(0, int(player.resources.get(normalized, 0)))


def _set_player_resource(player: PlayerState, resource: str, amount: int) -> PlayerState:
    normalized = _normalized_resource_id(resource)
    resources = dict(player.resources)
    resources[normalized] = max(0, amount)
    if normalized == "star":
        resources.pop("stars", None)
    return player.model_copy(update={"resources": resources})


def _normalized_resource_id(value: str) -> str:
    normalized = _normalized_id(value)
    if normalized in {"stars", "starlight"}:
        return "star"
    return normalized


def _card_destination(card: CardInstance) -> str:
    destination = card.effects.get("destination")
    if isinstance(destination, str) and destination in {"discard", "exhaust", "none"}:
        return destination
    if card.exhausts or bool(card.effects.get("exhaust_on_play")):
        return "exhaust"
    return "discard"


def _consume_vigor_after_attack(
    combat: CombatState,
    card: CardInstance,
) -> tuple[CombatState, tuple[EffectEvent, ...]]:
    if card.type != CardType.ATTACK:
        return combat, ()

    vigor = _status_amount(combat.player.statuses, "vigor")
    if vigor <= 0:
        return combat, ()

    statuses = dict(combat.player.statuses)
    for key in tuple(statuses):
        if _normalized_id(key).replace("_", "") == "vigor":
            statuses.pop(key, None)
    player = combat.player.model_copy(update={"statuses": statuses})
    return (
        combat.model_copy(update={"player": player}),
        (
            EffectEvent(
                kind="status_consumed",
                source_id=card.instance_id,
                target_id=PLAYER_TARGET_ID,
                amount=vigor,
                metadata={"status": "vigor"},
            ),
        ),
    )


def _find_card(cards: Sequence[CardInstance], instance_id: str | None) -> CardInstance | None:
    if instance_id is None:
        return None
    for card in cards:
        if card.instance_id == instance_id:
            return card
    return None


def _effect_steps(effects: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    sequence = effects.get("sequence", effects.get("effects"))
    if isinstance(sequence, Sequence) and not isinstance(sequence, (str, bytes, bytearray)):
        return tuple(item for item in sequence if isinstance(item, Mapping))
    return (effects,)


def _effect_amount(effect: Mapping[str, Any], key: str, energy_spent: int) -> int:
    value = effect.get(key, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, Mapping):
        base = int(value.get("amount", 0))
        per_energy = int(value.get("per_energy", 0))
        return base + per_energy * energy_spent
    return 0


def _modified_card_damage(
    combat: CombatState,
    card: CardInstance,
    amount: int,
) -> int:
    if amount <= 0:
        return amount
    modified = amount
    if card.type == CardType.ATTACK:
        modified += _status_amount(combat.player.statuses, "strength")
        modified += _status_amount(combat.player.statuses, "temporary_strength")
        modified += _status_amount(combat.player.statuses, "vigor")
        if _is_shiv_card(card):
            modified += _status_amount(combat.player.statuses, "accuracy")
        if _status_amount(combat.player.statuses, "weak") > 0:
            modified = int(modified * 0.75)
        if _status_amount(combat.player.statuses, "stance_wrath") > 0:
            modified *= 2
        elif _status_amount(combat.player.statuses, "stance_divinity") > 0:
            modified *= 3
    return max(0, modified)


def _modified_card_block(
    combat: CombatState,
    card: CardInstance,
    amount: int,
) -> int:
    if amount <= 0:
        return amount
    modified = amount
    if card.type == CardType.SKILL:
        modified += _status_amount(combat.player.statuses, "dexterity")
        modified += _status_amount(combat.player.statuses, "temporary_dexterity")
        if _status_amount(combat.player.statuses, "frail") > 0:
            modified = int(modified * 0.75)
    return max(0, modified)


def _is_shiv_card(card: CardInstance) -> bool:
    return _normalized_id(card.card_id) == "shiv" or any(
        _normalized_id(tag) == "shiv" for tag in card.tags
    )


def _status_amount(statuses: Mapping[str, int], status: str) -> int:
    candidates = {
        status,
        _normalized_id(status),
        _normalized_id(status).replace("_", ""),
    }
    aliases = _STATUS_LOOKUP_ALIASES.get(_normalized_id(status), ())
    candidates.update(aliases)
    return max(0, max(int(statuses.get(candidate, 0)) for candidate in candidates))


def _status_targets(
    combat: CombatState,
    card: CardInstance,
    target_id: str | None,
    status_payload: Mapping[str, Any],
) -> tuple[str, ...]:
    target = status_payload.get("target")
    if target == "self":
        return (PLAYER_TARGET_ID,)
    if target == "all_enemies":
        return tuple(monster.monster_id for monster in _alive_monsters(combat))
    if target == "enemy" and target_id and target_id != PLAYER_TARGET_ID:
        return (target_id,)
    if card.target == TargetType.SELF:
        return (PLAYER_TARGET_ID,)
    if target_id == PLAYER_TARGET_ID:
        return (PLAYER_TARGET_ID,)
    if target_id:
        return (target_id,)
    return ()


def _status_values(status_payload: Mapping[str, Any]) -> dict[str, int]:
    if "name" in status_payload or "status" in status_payload:
        name = _normalized_status_id(str(status_payload.get("name", status_payload.get("status"))))
        amount = int(status_payload.get("amount", status_payload.get("value", 1)))
        return {name: amount}

    return {
        _normalized_status_id(str(key)): int(value)
        for key, value in status_payload.items()
        if key not in {"target"} and isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _normalized_status_id(value: str) -> str:
    normalized = _normalized_id(value)
    return _STATUS_APPLICATION_ALIASES.get(normalized, normalized)


ANCIENT_SOURCE_URL = "https://spire-codex.com/api/mechanics/sections/neow"
ANCIENT_POSITIVE_RELICS = (
    "arcane_scroll",
    "booming_conch",
    "golden_pearl",
    "lead_paperweight",
    "lost_coffer",
    "massive_scroll",
    "neows_torment",
    "new_leaf",
    "phial_holster",
    "precise_scissors",
    "winged_boots",
)
ANCIENT_POSITIVE_RELIC_PAIRS = (
    ("lava_rock", "small_capsule"),
    ("nutritious_oyster", "stone_humidifier"),
    ("neows_talisman", "pomander"),
)
ANCIENT_CURSE_RELICS = (
    "cursed_pearl",
    "hefty_tablet",
    "large_capsule",
    "leafy_poultice",
    "neows_bones",
    "precarious_shears",
    "scroll_boxes",
    "silver_crucible",
)
ANCIENT_CURSE_BLOCKS: dict[str, frozenset[str]] = {
    "cursed_pearl": frozenset({"golden_pearl"}),
    "hefty_tablet": frozenset({"arcane_scroll"}),
    "large_capsule": frozenset({"lava_rock", "small_capsule"}),
    "leafy_poultice": frozenset({"new_leaf"}),
    "precarious_shears": frozenset({"precise_scissors"}),
}
ANCIENT_RELIC_NAMES = {
    "arcane_scroll": "Arcane Scroll",
    "booming_conch": "Booming Conch",
    "cursed_pearl": "Cursed Pearl",
    "golden_compass": "Golden Compass",
    "golden_pearl": "Golden Pearl",
    "hefty_tablet": "Hefty Tablet",
    "large_capsule": "Large Capsule",
    "lava_rock": "Lava Rock",
    "lead_paperweight": "Lead Paperweight",
    "leafy_poultice": "Leafy Poultice",
    "lost_coffer": "Lost Coffer",
    "massive_scroll": "Massive Scroll",
    "neows_bones": "Neow's Bones",
    "neows_talisman": "Neow's Talisman",
    "neows_torment": "Neow's Torment",
    "new_leaf": "New Leaf",
    "nutritious_oyster": "Nutritious Oyster",
    "phial_holster": "Phial Holster",
    "pomander": "Pomander",
    "precarious_shears": "Precarious Shears",
    "precise_scissors": "Precise Scissors",
    "scroll_boxes": "Scroll Boxes",
    "silver_crucible": "Silver Crucible",
    "small_capsule": "Small Capsule",
    "stone_humidifier": "Stone Humidifier",
    "winged_boots": "Winged Boots",
}


def _apply_ancient_heal(player: PlayerState, ascension: int) -> PlayerState:
    missing_hp = max(0, player.max_hp - player.hp)
    if missing_hp <= 0:
        return player
    heal = int(missing_hp * 0.8) if ascension >= 2 else missing_hp
    return player.model_copy(update={"hp": min(player.max_hp, player.hp + heal)})


def _generate_ancient_state(act: int, rng: Random) -> AncientState:
    curse_relic = rng.choice(ANCIENT_CURSE_RELICS)
    blocked = ANCIENT_CURSE_BLOCKS.get(curse_relic, frozenset())
    positive_pool = [relic for relic in ANCIENT_POSITIVE_RELICS if relic not in blocked]
    if act == 2:
        positive_pool.append(GOLDEN_COMPASS_RELIC_ID)

    for relic_pair in ANCIENT_POSITIVE_RELIC_PAIRS:
        pair_candidates = [relic for relic in relic_pair if relic not in blocked]
        if pair_candidates:
            positive_pool.append(rng.choice(pair_candidates))

    rng.shuffle(positive_pool)
    positive_relics = positive_pool[:2]
    options = [
        _ancient_option(act, index + 1, relic_id, "positive_relic")
        for index, relic_id in enumerate(positive_relics)
    ]
    options.append(_ancient_option(act, 3, curse_relic, "curse_relic"))
    return AncientState(act=act, options=tuple(options))


def _ancient_option(
    act: int,
    index: int,
    relic_id: str,
    kind: Literal["positive_relic", "curse_relic"],
) -> AncientOptionState:
    name = ANCIENT_RELIC_NAMES.get(relic_id, relic_id.replace("_", " ").title())
    return AncientOptionState(
        option_id=f"a{act}:ancient:{index}",
        name=name,
        kind=kind,
        relic_id=relic_id,
        description=f"Gain {name}.",
        metadata={
            "act": act,
            "pool": "curse" if kind == "curse_relic" else "positive",
            "source_url": ANCIENT_SOURCE_URL,
        },
    )


def _initial_player(character_id: str, ascension: int, source: Mapping[str, Any]) -> PlayerState:
    character_source = _character_source(character_id, source)
    player_source = _mapping_from(character_source.get("player")) or _mapping_from(
        source.get("player")
    )
    max_hp = int(player_source.get("max_hp", player_source.get("hp", 80)))
    hp = int(player_source.get("hp", max_hp))
    max_energy = int(player_source.get("max_energy", player_source.get("energy", 3)))
    return PlayerState(
        hp=hp,
        max_hp=max_hp,
        block=int(player_source.get("block", 0)),
        energy=int(player_source.get("energy", max_energy)),
        max_energy=max_energy,
        gold=int(player_source.get("gold", 0)),
        statuses=dict(player_source.get("statuses", {}))
        if isinstance(player_source.get("statuses", {}), Mapping)
        else {},
        resources=_player_resources_from_source(player_source),
    )


def _player_resources_from_source(player_source: Mapping[str, Any]) -> dict[str, int]:
    resources: dict[str, int] = {}
    raw_resources = player_source.get("resources")
    if isinstance(raw_resources, Mapping):
        for key, value in raw_resources.items():
            with suppress(TypeError, ValueError):
                resources[_normalized_resource_id(str(key))] = max(0, int(value))
    for key in ("star", "stars", "forge", "summon"):
        if key in player_source:
            with suppress(TypeError, ValueError):
                resources[_normalized_resource_id(key)] = max(0, int(player_source[key]))
    return resources


def _initial_monsters(source: Mapping[str, Any]) -> tuple[MonsterState, ...]:
    encounter = (
        _mapping_from(source.get("initial_encounter"))
        or _mapping_from(source.get("encounter"))
        or _first_encounter(source.get("encounters"))
    )
    monster_specs = encounter.get("monsters") if encounter else source.get("monsters")
    if not isinstance(monster_specs, Sequence) or isinstance(
        monster_specs, (str, bytes, bytearray)
    ):
        monster_specs = (
            {
                "monster_id": "training_dummy",
                "name": "Training Dummy",
                "hp": 40,
                "intent": "attack",
                "intent_damage": 6,
            },
        )

    monsters: list[MonsterState] = []
    for index, raw_spec in enumerate(monster_specs):
        spec = _mapping_from(raw_spec)
        if not spec:
            continue
        monster_id = str(spec.get("monster_id", spec.get("id", f"monster_{index + 1:03d}")))
        max_hp = int(spec.get("max_hp", spec.get("hp", 20)))
        monsters.append(
            MonsterState(
                monster_id=monster_id,
                name=str(spec.get("name", monster_id)),
                hp=int(spec.get("hp", max_hp)),
                max_hp=max_hp,
                block=int(spec.get("block", 0)),
                intent=str(spec["intent"]) if spec.get("intent") is not None else None,
                intent_damage=int(spec.get("intent_damage", spec.get("damage", 0))),
                intent_block=int(spec.get("intent_block", 0)),
                statuses=dict(spec.get("statuses", {}))
                if isinstance(spec.get("statuses", {}), Mapping)
                else {},
            )
        )
    if not monsters:
        return (
            MonsterState(
                monster_id="training_dummy",
                name="Training Dummy",
                hp=40,
                max_hp=40,
                intent="attack",
                intent_damage=6,
            ),
        )
    return tuple(monsters)


def _starter_deck(character_id: str, source: Mapping[str, Any]) -> tuple[CardInstance, ...]:
    character_source = _character_source(character_id, source)
    raw_deck = (
        character_source.get("starter_deck")
        or character_source.get("deck")
        or source.get("starter_deck")
        or source.get("deck")
        or _default_starter_deck()
    )
    library = _card_library(source)
    return _instantiate_deck(raw_deck, library)


def _instantiate_deck(
    raw_deck: Any, library: Mapping[str, Mapping[str, Any]]
) -> tuple[CardInstance, ...]:
    if not isinstance(raw_deck, Sequence) or isinstance(raw_deck, (str, bytes, bytearray)):
        raw_deck = _default_starter_deck()

    cards: list[CardInstance] = []
    instance_counter = 1
    for raw_item in raw_deck:
        item = _deck_item_spec(raw_item, library)
        if not item:
            continue
        copies = int(item.pop("copies", item.pop("count", item.pop("qty", 1))))
        for copy_index in range(max(1, copies)):
            card_spec = dict(item)
            if copies > 1 and "instance_id" in card_spec:
                card_spec["instance_id"] = f"{card_spec['instance_id']}_{copy_index + 1}"
            cards.append(_card_from_spec(card_spec, instance_counter, card_library=library))
            instance_counter += 1
    return tuple(cards)


def _deck_item_spec(raw_item: Any, library: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if isinstance(raw_item, str):
        return dict(library.get(raw_item, {"card_id": raw_item}))
    spec = _mapping_from(raw_item)
    if not spec:
        return {}
    card_id = str(spec.get("card_id", spec.get("id", "")))
    merged = dict(library.get(card_id, {}))
    merged.update(spec)
    return merged


def _card_from_spec(
    spec: Mapping[str, Any],
    instance_counter: int,
    *,
    card_library: Mapping[str, Mapping[str, Any]] | None = None,
) -> CardInstance:
    source_spec = dict(spec)
    if bool(source_spec.get("upgraded", False)):
        source_spec = _apply_card_upgrade_to_spec(source_spec)
    normalized = normalize_card_spec(source_spec, card_library=card_library)
    merged = dict(normalized)
    merged.update({key: value for key, value in source_spec.items() if key in _CARD_RUNTIME_KEYS})
    custom = dict(_mapping_from(merged.get("custom", {})))
    if source_spec.get("star_cost") not in (None, ""):
        custom["star_cost"] = int(source_spec.get("star_cost", 0))
    if bool(source_spec.get("is_x_star_cost")):
        custom["star_cost"] = "X"
    keywords = tuple(str(keyword) for keyword in source_spec.get("keywords_key") or ())
    if any(_normalized_id(keyword) == "exhaust" for keyword in keywords):
        merged["exhausts"] = True
    if any(_normalized_id(keyword) == "ethereal" for keyword in keywords):
        custom["ethereal"] = True
    if _source_spec_has_retain(source_spec):
        custom["retain"] = True

    spec = merged
    card_id = str(spec.get("card_id", spec.get("id", f"card_{instance_counter:03d}")))
    effects = spec.get("effects", spec.get("effect", {}))
    if isinstance(effects, Sequence) and not isinstance(effects, (str, bytes, bytearray, Mapping)):
        effects = {"sequence": list(effects)}
    if not isinstance(effects, Mapping):
        effects = {}

    card_type = _enum_value(
        CardType,
        spec.get("type", spec.get("card_type", CardType.UNKNOWN.value)),
    )
    target = spec.get("target")
    if target is None:
        target = _infer_target(card_type, effects)

    raw_tags = spec.get("tags", ())
    tags = tuple(str(tag) for tag in raw_tags) if raw_tags is not None else ()
    return CardInstance(
        instance_id=str(spec.get("instance_id", f"card_{instance_counter:03d}")),
        card_id=card_id,
        name=str(spec.get("name", card_id)),
        type=card_type,
        cost=_card_cost_from_spec(spec),
        target=_enum_value(TargetType, target),
        effects=dict(effects),
        tags=tags,
        exhausts=bool(spec.get("exhausts", spec.get("exhaust", False))),
        upgraded=bool(spec.get("upgraded", False)),
        enchantments=tuple(
            CardEnchantment.model_validate(enchantment)
            if isinstance(enchantment, Mapping)
            else enchantment
            for enchantment in spec.get("enchantments", ())
        ),
        custom=_card_custom_metadata(custom, source_spec),
    )


def _card_custom_metadata(
    custom: Mapping[str, Any],
    source_spec: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = dict(custom)
    if "source_event_id" not in metadata:
        metadata.setdefault("source_spec", _jsonish_copy(source_spec))
    return metadata


def _source_spec_has_retain(source_spec: Mapping[str, Any]) -> bool:
    if any(_truthy(source_spec.get(key)) for key in ("retain", "retains", "retained")):
        return True
    keywords = source_spec.get("keywords_key", source_spec.get("keywords", ()))
    if any(
        _normalized_tag(keyword) in {"retain", "retained"}
        for keyword in _string_items(keywords)
    ):
        return True
    if any(
        _normalized_tag(tag) in {"retain", "retained", "keyword_retain"}
        for tag in _string_items(source_spec.get("tags", ()))
    ):
        return True
    description = str(
        source_spec.get("description", source_spec.get("description_raw", "")) or ""
    )
    return _description_has_standalone_retain(description)


def _description_has_standalone_retain(description: str) -> bool:
    if not description:
        return False
    plain = re.sub(r"\[/?[^\]]+\]", "", description)
    sentences = re.split(r"(?:\n|(?<=[.!?])\s+)", plain)
    return any(sentence.strip().strip(".!?").lower() == "retain" for sentence in sentences)


def _normalized_tag(value: object) -> str:
    return _normalized_id(str(value)).replace(":", "_")


def _string_items(value: object) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "retain"}
    return bool(value)


def _custom_int(custom: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(custom.get(key, default))
    except (TypeError, ValueError):
        return default


def _upgrade_card_for_add_relics(
    state: RunState,
    card: CardInstance,
) -> tuple[CardInstance, EffectEvent | None]:
    relic_id = _card_add_upgrade_relic_id(state, card)
    if relic_id is None:
        return card, None

    upgraded = _upgrade_card_instance(card)
    if upgraded.upgraded == card.upgraded:
        return card, None

    return (
        upgraded,
        EffectEvent(
            kind="relic_card_upgraded",
            source_id=relic_id,
            target_id=upgraded.instance_id,
            metadata={
                "card_id": upgraded.card_id,
                "card_type": upgraded.type.value,
            },
        ),
    )


def _card_add_upgrade_relic_id(state: RunState, card: CardInstance) -> str | None:
    if card.upgraded:
        return None
    if card.type == CardType.ATTACK and _has_relic(state, "molten_egg"):
        return "molten_egg"
    if card.type == CardType.SKILL and _has_relic(state, "toxic_egg"):
        return "toxic_egg"
    if card.type == CardType.POWER and _has_relic(state, "frozen_egg"):
        return "frozen_egg"
    return None


def _upgrade_card_instance(card: CardInstance) -> CardInstance:
    if card.upgraded:
        return card
    source_spec = _mapping_from(card.custom.get("source_spec"))
    if source_spec:
        upgraded_spec = dict(source_spec)
        upgraded_spec["upgraded"] = True
        upgraded_spec["instance_id"] = card.instance_id
        upgraded = _card_from_spec(upgraded_spec, 1)
        return upgraded.model_copy(
            update={
                "enchantments": card.enchantments,
                "tags": tuple(sorted(set(upgraded.tags + card.tags))),
            }
        )
    return card.model_copy(update={"upgraded": True})


def _downgrade_card_instance(card: CardInstance) -> CardInstance:
    if not card.upgraded:
        return card
    source_spec = _mapping_from(card.custom.get("source_spec"))
    if source_spec:
        downgraded_spec = dict(source_spec)
        downgraded_spec["upgraded"] = False
        downgraded_spec["instance_id"] = card.instance_id
        downgraded = _card_from_spec(downgraded_spec, 1)
        return downgraded.model_copy(
            update={
                "enchantments": card.enchantments,
                "tags": tuple(sorted(set(downgraded.tags + card.tags))),
            }
        )
    return card.model_copy(update={"upgraded": False})


def _apply_card_upgrade_to_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    upgraded = dict(spec)
    upgrade = _mapping_from(spec.get("upgrade"))
    if not upgrade:
        return upgraded

    normalized_key_to_source_key = {
        _normalized_id(key): key
        for key in (
            "block",
            "cards_draw",
            "cost",
            "damage",
            "energy_gain",
            "heal",
            "hit_count",
            "hp_loss",
            "star_cost",
        )
    }
    for raw_key, raw_delta in upgrade.items():
        normalized_key = _normalized_id(str(raw_key))
        source_key = normalized_key_to_source_key.get(normalized_key)
        if source_key is None:
            for candidate in normalized_key_to_source_key:
                if normalized_key.startswith(candidate) or normalized_key.endswith(candidate):
                    source_key = normalized_key_to_source_key[candidate]
                    break
        if source_key is None:
            continue
        current = upgraded.get(source_key)
        next_value = _upgraded_value(current, raw_delta)
        if next_value is not None:
            upgraded[source_key] = next_value
    return upgraded


def _upgraded_value(current: Any, delta: Any) -> int | None:
    if current is None:
        current_int = 0
    else:
        with suppress(TypeError, ValueError):
            current_int = int(current)
            return current_int + _upgrade_delta(delta)
        return None
    return current_int + _upgrade_delta(delta)


def _upgrade_delta(delta: Any) -> int:
    if isinstance(delta, str):
        text = delta.strip()
        if text.startswith("+"):
            text = text[1:]
        with suppress(ValueError):
            return int(text)
        return 0
    with suppress(TypeError, ValueError):
        return int(delta)
    return 0


def _jsonish_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonish_copy(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_jsonish_copy(item) for item in value)
    return value


def _default_starter_deck() -> tuple[dict[str, Any], ...]:
    return (
        {
            "card_id": "strike",
            "name": "Strike",
            "type": "attack",
            "cost": 1,
            "target": "enemy",
            "effects": {"damage": 6},
            "copies": 5,
        },
        {
            "card_id": "defend",
            "name": "Defend",
            "type": "skill",
            "cost": 1,
            "target": "self",
            "effects": {"block": 5},
            "copies": 4,
        },
        {
            "card_id": "bash",
            "name": "Bash",
            "type": "attack",
            "cost": 2,
            "target": "enemy",
            "effects": {"damage": 8, "apply_status": {"target": "enemy", "vulnerable": 2}},
        },
    )


def _card_library(source: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = source.get("card_library", source.get("cards", {}))
    if isinstance(raw, Mapping):
        return {
            str(key): _mapping_from(value) or {"card_id": str(key)}
            for key, value in raw.items()
        }
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        library: dict[str, Mapping[str, Any]] = {}
        for value in raw:
            spec = _mapping_from(value)
            if spec:
                card_id = str(spec.get("card_id", spec.get("id", "")))
                if card_id:
                    library[card_id] = spec
        return library
    return {}


def _character_source(character_id: str, source: Mapping[str, Any]) -> Mapping[str, Any]:
    characters = source.get("characters")
    if isinstance(characters, Mapping):
        character = characters.get(character_id)
        if isinstance(character, Mapping):
            return character
    return {}


def _first_encounter(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            encounter = _mapping_from(item)
            if encounter:
                return encounter
    return {}


def _mapping_from(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _initial_flags(source: Mapping[str, Any]) -> dict[str, Any]:
    flags = dict(source.get("flags", {})) if isinstance(source.get("flags", {}), Mapping) else {}
    for key in (
        "max_acts",
        "map_floors",
        "map_width",
        "draw_per_turn",
        "orb_slots",
        "base_potion_slots",
        "bonus_potion_slots",
        "potion_slots",
        "potion_pool",
        "cards",
        "card_library",
        "card_pool",
        "relic_pool",
        "monsters",
        "monster_pool",
        "encounters",
        "encounter_pool",
        "events",
        "event_id",
        "event_room_id",
        "event_flow_page_id",
        "event_flow_counters",
        "event_flow_data",
        "event_pool",
        "data_cache_dir",
        "card_non_rare_count",
        "card_reward_choice_bonus",
        "card_reward_choice_count",
        "card_reward_choice_delta",
        "potion_chance_bonus",
        "treasure_chests_opened",
        "combat_encounter",
        "combat_encounter_id",
        "monster_encounter_id",
        "encounter_id",
        "combat_monster_ids",
        "combat_monster_id",
        "monster_ids",
        "monster_id",
        "combat_reward_encounter",
        "combat_reward_event_id",
        "event_fight_id",
        "fake_merchant_combat",
        "fake_merchant_unsold_relic_ids",
        "combat_reward_potion_id",
        "combat_reward_potion_ids",
        "combat_reward_extra_potion_count",
        "combat_reward_potion_pool",
        "combat_reward_potion_chance_percent",
        "combat_reward_gold",
        "combat_reward_relic_id",
        "combat_reward_relic_ids",
        "event_combat_reward_relic_ids",
        "combat_reward_card_ids",
        "combat_reward_card_options",
        "combat_reward_card_pool",
        "combat_reward_card_count",
        "combat_reward_relic_count",
        "combat_reward_forced",
        "boss_reward_card_count",
        "boss_reward_relic_count",
        "boss_relic_pool",
        "combat_boss_relic_pool",
        "elite_reward_card_count",
        "elite_reward_relic_count",
        "event_reward_potion_id",
        "event_reward_potion_ids",
        "event_reward_potion_count",
        "event_reward_potion_pool",
        "event_reward_potion_chance_percent",
        "event_reward_gold",
        "event_reward_relic_id",
        "event_reward_relic_ids",
        "event_reward_relic_count",
        "event_reward_card_ids",
        "event_reward_card_options",
        "event_reward_remove_card_ids",
        "event_reward_forced",
        "treasure_reward_gold",
        "treasure_reward_relic_id",
        "treasure_reward_card_options",
        "treasure_reward_forced",
        "treasure_relic_pool",
        "golden_compass_act2_map",
        "spoils_map_pending_act",
        "spoils_map_source_act",
        "spoils_map_target_act",
        "spoils_map_target_node_id",
        "spoils_map_reward_gold",
        "shop_plan",
        "shop_card_pool",
        "shop_colorless_card_pool",
        "shop_relic_pool",
        "shop_potion_pool",
        "shop_card_removals_bought",
        "card_rare_offset_percent",
    ):
        if key in source:
            flags[key] = source[key]
    return flags


def _source_int(source: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(source.get(key, default))
    except (TypeError, ValueError):
        return default


def _source_bool(source: Mapping[str, Any], key: str, default: bool) -> bool:
    value = source.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _int_from_mapping(source: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(source.get(key, default))
    except (TypeError, ValueError):
        return default


def _card_cost_from_spec(spec: Mapping[str, Any]) -> int | None:
    if "cost" not in spec:
        return 1
    value = spec.get("cost")
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "x":
        return -1
    return int(value)


def _enum_value(enum_type: type[CardType] | type[TargetType], value: Any) -> Any:
    if isinstance(value, enum_type):
        return value
    normalized = _normalized_id(str(value))
    aliases: Mapping[str, str] = {}
    if enum_type is CardType:
        aliases = {
            "attack": CardType.ATTACK.value,
            "curse": CardType.CURSE.value,
            "power": CardType.POWER.value,
            "quest": CardType.STATUS.value,
            "skill": CardType.SKILL.value,
            "status": CardType.STATUS.value,
        }
    else:
        aliases = {
            "all_enemies": TargetType.ALL_ENEMIES.value,
            "allenemies": TargetType.ALL_ENEMIES.value,
            "any": TargetType.ANY.value,
            "any_enemy": TargetType.ENEMY.value,
            "anyenemy": TargetType.ENEMY.value,
            "enemy": TargetType.ENEMY.value,
            "none": TargetType.NONE.value,
            "random_enemy": TargetType.ENEMY.value,
            "randomenemy": TargetType.ENEMY.value,
            "self": TargetType.SELF.value,
        }
    if normalized in aliases:
        return enum_type(aliases[normalized])
    try:
        return enum_type(str(value))
    except ValueError:
        if enum_type is CardType:
            return CardType.UNKNOWN
        return TargetType.ENEMY


def _infer_target(card_type: CardType, effects: Mapping[str, Any]) -> str:
    if "all_damage" in effects:
        return TargetType.ALL_ENEMIES.value
    if "damage" in effects:
        return TargetType.ENEMY.value
    if "block" in effects or "heal" in effects:
        return TargetType.SELF.value
    if card_type == CardType.SKILL:
        return TargetType.SELF.value
    return TargetType.NONE.value


def _generate_act_map(act: int, rng: Random, source: Mapping[str, Any]) -> MapState:
    if act == 2 and _source_bool(source, "golden_compass_act2_map", False):
        return _generate_golden_compass_map(act=act)

    floors = _source_int(source, "map_floors", _default_act_floors(act))
    width = _source_int(source, "map_width", 7)
    width = max(1, min(7, width))
    floors = max(4, floors)
    boss_floor = floors - 1
    rest_floor = boss_floor - 1
    treasure_floor = max(2, boss_floor - 7)

    start_lane = width // 2
    start_node_id = f"a{act}:0:{start_lane}"
    nodes: list[MapNodeState] = [
        MapNodeState(
            node_id=start_node_id,
            act=act,
            floor=0,
            lane=start_lane,
            kind=RoomKind.START,
        )
    ]
    room_cycle = (
        RoomKind.MONSTER,
        RoomKind.EVENT,
        RoomKind.SHOP,
        RoomKind.ELITE,
        RoomKind.MONSTER,
        RoomKind.REST,
        RoomKind.TREASURE,
    )
    pre_rest_cycle = (
        RoomKind.MONSTER,
        RoomKind.EVENT,
        RoomKind.SHOP,
        RoomKind.ELITE,
        RoomKind.MONSTER,
        RoomKind.TREASURE,
    )

    for floor in range(1, boss_floor):
        if floor == rest_floor:
            kinds = [RoomKind.REST for _ in range(width)]
        elif floor == treasure_floor:
            kinds = [RoomKind.TREASURE for _ in range(width)]
        elif floor == 1:
            kinds = [RoomKind.MONSTER for _ in range(width)]
        elif floor <= 5:
            early_cycle = (RoomKind.MONSTER, RoomKind.EVENT, RoomKind.SHOP)
            offset = rng.randrange(len(early_cycle))
            kinds = [
                early_cycle[(floor + lane + offset + act) % len(early_cycle)]
                for lane in range(width)
            ]
        elif floor == rest_floor - 1:
            offset = rng.randrange(len(pre_rest_cycle))
            kinds = [
                pre_rest_cycle[(floor + lane + offset + act) % len(pre_rest_cycle)]
                for lane in range(width)
            ]
        else:
            offset = rng.randrange(len(room_cycle))
            kinds = [
                room_cycle[(floor + lane + offset + act) % len(room_cycle)]
                for lane in range(width)
            ]

        for lane, kind in enumerate(kinds):
            nodes.append(
                MapNodeState(
                    node_id=f"a{act}:{floor}:{lane}",
                    act=act,
                    floor=floor,
                    lane=lane,
                    kind=kind,
                )
            )

    boss_node_id = f"a{act}:{boss_floor}:{start_lane}"
    nodes.append(
        MapNodeState(
            node_id=boss_node_id,
            act=act,
            floor=boss_floor,
            lane=start_lane,
            kind=RoomKind.BOSS,
        )
    )

    nodes_by_floor: dict[int, list[MapNodeState]] = {}
    for node in nodes:
        nodes_by_floor.setdefault(node.floor, []).append(node)
    edges = _generate_path_edges(
        act=act,
        boss_floor=boss_floor,
        width=width,
        rng=rng,
        path_count=_source_int(source, "map_paths", _default_path_count(width)),
        start_lane=start_lane,
    )
    edges = _dedupe_fixed_destination_floor_edges(
        edges,
        fixed_destination_floors={treasure_floor, rest_floor},
    )
    nodes, edges = _prune_nodes_to_edges(nodes, edges)
    nodes = _repair_generated_map_node_kinds(
        nodes,
        edges,
        act=act,
        boss_floor=boss_floor,
        treasure_floor=treasure_floor,
        rest_floor=rest_floor,
    )

    return MapState(
        act=act,
        nodes=tuple(nodes),
        edges=tuple(edges),
        current_node_id=start_node_id,
        completed_node_ids=(start_node_id,),
        boss_node_id=boss_node_id,
    )


_GOLDEN_COMPASS_PATH_KINDS = (
    RoomKind.MONSTER,
    RoomKind.EVENT,
    RoomKind.MONSTER,
    RoomKind.REST,
    RoomKind.MONSTER,
    RoomKind.REST,
    RoomKind.EVENT,
    RoomKind.TREASURE,
    RoomKind.EVENT,
    RoomKind.TREASURE,
    RoomKind.EVENT,
    RoomKind.SHOP,
    RoomKind.ELITE,
    RoomKind.REST,
    RoomKind.ELITE,
    RoomKind.REST,
    RoomKind.BOSS,
)


def _generate_golden_compass_map(act: int) -> MapState:
    start_node_id = f"a{act}:0:0"
    nodes = [
        MapNodeState(
            node_id=start_node_id,
            act=act,
            floor=0,
            lane=0,
            kind=RoomKind.START,
        )
    ]
    nodes.extend(
        MapNodeState(
            node_id=f"a{act}:{floor}:0",
            act=act,
            floor=floor,
            lane=0,
            kind=kind,
        )
        for floor, kind in enumerate(_GOLDEN_COMPASS_PATH_KINDS, start=1)
    )
    edges = tuple(
        MapEdgeState(
            from_id=nodes[index].node_id,
            to_id=nodes[index + 1].node_id,
        )
        for index in range(len(nodes) - 1)
    )
    boss_node = nodes[-1]
    return MapState(
        act=act,
        nodes=tuple(nodes),
        edges=edges,
        current_node_id=start_node_id,
        completed_node_ids=(start_node_id,),
        boss_node_id=boss_node.node_id,
    )


def _mark_spoils_map_site(
    map_state: MapState,
    flags: Mapping[str, Any],
    *,
    force: bool = False,
) -> tuple[MapState, dict[str, Any], tuple[EffectEvent, ...]]:
    next_flags = dict(flags)
    pending_act = _optional_int(next_flags.get("spoils_map_pending_act"))
    target_act = _optional_int(next_flags.get("spoils_map_target_act"))
    should_mark = pending_act == map_state.act or (force and target_act == map_state.act)
    if not should_mark:
        return map_state, next_flags, ()

    map_state, target = _force_map_through_spoils_treasure(map_state)
    for key in (
        "spoils_map_pending_act",
        "spoils_map_target_act",
        "spoils_map_target_node_id",
        "spoils_map_reward_gold",
    ):
        next_flags.pop(key, None)
    next_flags["spoils_map_target_act"] = map_state.act
    next_flags["spoils_map_target_node_id"] = target.node_id
    next_flags["spoils_map_reward_gold"] = SPOILS_MAP_GOLD
    event_kind = (
        "spoils_map_site_remarked"
        if force and target_act == map_state.act
        else "spoils_map_site_marked"
    )
    return map_state, next_flags, (
        EffectEvent(
            kind=event_kind,
            target_id=target.node_id,
            amount=SPOILS_MAP_GOLD,
            metadata={
                "act": target.act,
                "floor": target.floor,
                "room_kind": target.kind.value,
            },
        ),
    )


def _force_map_through_spoils_treasure(map_state: MapState) -> tuple[MapState, MapNodeState]:
    nodes_by_key = {(node.floor, node.lane): node for node in map_state.nodes}
    floors = [node.floor for node in map_state.nodes]
    boss_floor = max(floors)
    width = max((node.lane for node in map_state.nodes), default=0) + 1
    start = _map_start_node(map_state)
    boss = _map_boss_node(map_state)
    target_floor = _spoils_map_target_floor(map_state, boss_floor)
    target_lane = min(max(start.lane, 0), max(0, width - 1))

    floor_lanes = {
        floor: _spoils_map_lanes_for_floor(
            floor,
            width=width,
            start_lane=start.lane,
            target_floor=target_floor,
            target_lane=target_lane,
            boss_floor=boss_floor,
            boss_lane=boss.lane,
        )
        for floor in range(boss_floor + 1)
    }
    nodes: list[MapNodeState] = []
    for floor in range(boss_floor + 1):
        for lane in floor_lanes[floor]:
            kind = _spoils_map_node_kind(
                floor,
                lane,
                existing=nodes_by_key.get((floor, lane)),
                act=map_state.act,
                target_floor=target_floor,
                boss_floor=boss_floor,
            )
            nodes.append(
                MapNodeState(
                    node_id=f"a{map_state.act}:{floor}:{lane}",
                    act=map_state.act,
                    floor=floor,
                    lane=lane,
                    kind=kind,
                )
            )

    nodes_by_floor: dict[int, list[MapNodeState]] = {}
    for node in nodes:
        nodes_by_floor.setdefault(node.floor, []).append(node)

    edges: list[MapEdgeState] = []
    for floor in range(boss_floor):
        next_nodes = nodes_by_floor[floor + 1]
        for node in nodes_by_floor[floor]:
            candidates = [
                target
                for target in next_nodes
                if abs(target.lane - node.lane) <= 1
            ] or [min(next_nodes, key=lambda target: abs(target.lane - node.lane))]
            for target in candidates:
                edges.append(MapEdgeState(from_id=node.node_id, to_id=target.node_id))

    nodes, edges = _prune_nodes_to_edges(nodes, edges)
    nodes = _repair_generated_map_node_kinds(
        nodes,
        edges,
        act=map_state.act,
        boss_floor=boss_floor,
        treasure_floor=target_floor,
        rest_floor=boss_floor - 1,
    )
    nodes = sorted(nodes, key=lambda node: (node.floor, node.lane))
    edges = sorted(edges, key=_edge_state_sort_key)
    start_node_id = f"a{map_state.act}:0:{start.lane}"
    boss_node_id = f"a{map_state.act}:{boss_floor}:{boss.lane}"
    target_node_id = f"a{map_state.act}:{target_floor}:{target_lane}"
    next_map = MapState(
        act=map_state.act,
        nodes=tuple(nodes),
        edges=tuple(edges),
        current_node_id=start_node_id,
        completed_node_ids=(start_node_id,),
        boss_node_id=boss_node_id,
    )
    return next_map, next_map.node_by_id[target_node_id]


def _map_start_node(map_state: MapState) -> MapNodeState:
    for node in map_state.nodes:
        if node.kind == RoomKind.START:
            return node
    return min(map_state.nodes, key=lambda node: (node.floor, node.lane))


def _map_boss_node(map_state: MapState) -> MapNodeState:
    if map_state.boss_node_id is not None:
        with suppress(KeyError):
            return map_state.node_by_id[map_state.boss_node_id]
    bosses = [node for node in map_state.nodes if node.kind == RoomKind.BOSS]
    if bosses:
        return max(bosses, key=lambda node: (node.floor, node.lane))
    return max(map_state.nodes, key=lambda node: (node.floor, node.lane))


def _spoils_map_target_floor(map_state: MapState, boss_floor: int) -> int:
    treasure_floors = sorted(
        {node.floor for node in map_state.nodes if node.kind == RoomKind.TREASURE}
    )
    if treasure_floors:
        return treasure_floors[0]
    return min(max(2, boss_floor - 7), max(1, boss_floor - 2))


def _spoils_map_lanes_for_floor(
    floor: int,
    *,
    width: int,
    start_lane: int,
    target_floor: int,
    target_lane: int,
    boss_floor: int,
    boss_lane: int,
) -> tuple[int, ...]:
    all_lanes = range(width)
    if floor == 0:
        return (start_lane,)
    if floor == target_floor:
        return (target_lane,)
    if floor == boss_floor:
        return (boss_lane,)
    if floor < target_floor:
        lanes = [
            lane
            for lane in all_lanes
            if abs(lane - start_lane) <= floor
            and abs(lane - target_lane) <= target_floor - floor
        ]
    else:
        lanes = [
            lane
            for lane in all_lanes
            if abs(lane - target_lane) <= floor - target_floor
            and abs(lane - boss_lane) <= boss_floor - floor
        ]
    if lanes:
        return tuple(lanes)
    fallback = round(
        target_lane
        + ((boss_lane - target_lane) * max(0, floor - target_floor))
        / max(1, boss_floor - target_floor)
    )
    return (min(max(0, fallback), max(0, width - 1)),)


def _spoils_map_node_kind(
    floor: int,
    lane: int,
    *,
    existing: MapNodeState | None,
    act: int,
    target_floor: int,
    boss_floor: int,
) -> RoomKind:
    if floor == 0:
        return RoomKind.START
    if floor == target_floor:
        return RoomKind.TREASURE
    if floor == boss_floor:
        return RoomKind.BOSS
    if floor == boss_floor - 1:
        return RoomKind.REST
    if existing is not None:
        return existing.kind
    cycle = (
        RoomKind.MONSTER,
        RoomKind.EVENT,
        RoomKind.SHOP,
        RoomKind.MONSTER,
        RoomKind.ELITE,
        RoomKind.EVENT,
    )
    return cycle[(floor + lane + act) % len(cycle)]


def _default_act_floors(act: int) -> int:
    # Spire Codex: rooms + Ancient/start + boss.
    if act <= 1:
        return 17
    if act == 2:
        return 16
    return 15


def _default_path_count(width: int) -> int:
    if width <= 1:
        return 1
    if width <= 3:
        return 4
    return 5


def _generate_path_edges(
    *,
    act: int,
    boss_floor: int,
    width: int,
    rng: Random,
    path_count: int,
    start_lane: int,
) -> list[MapEdgeState]:
    edge_keys: set[tuple[str, str]] = set()
    path_count = max(1, min(12, path_count))
    first_lanes = _adjacent_lanes(start_lane, width)
    paths: list[list[int]] = []
    for index in range(path_count):
        first_lane = first_lanes[index] if index < len(first_lanes) else None
        paths.append(
            _generate_lane_path(
                boss_floor,
                width,
                rng,
                start_lane,
                first_lane=first_lane,
            )
        )

    for path in paths:
        for floor in range(boss_floor):
            from_id = f"a{act}:{floor}:{path[floor]}"
            to_id = f"a{act}:{floor + 1}:{path[floor + 1]}"
            edge_keys.add((from_id, to_id))

    return [
        MapEdgeState(from_id=from_id, to_id=to_id)
        for from_id, to_id in sorted(edge_keys, key=_edge_sort_key)
    ]


_MAP_NON_CONSECUTIVE_KINDS = frozenset(
    {RoomKind.ELITE, RoomKind.SHOP, RoomKind.REST, RoomKind.TREASURE}
)


def _dedupe_fixed_destination_floor_edges(
    edges: Sequence[MapEdgeState],
    *,
    fixed_destination_floors: set[int],
) -> list[MapEdgeState]:
    """Avoid branching from one node into multiple same-kind fixed floors."""

    kept: list[MapEdgeState] = []
    seen_fixed_starts: set[tuple[str, int]] = set()
    for edge in sorted(edges, key=_edge_state_sort_key):
        _from_floor, _from_lane, to_floor, _to_lane = _edge_parts(edge.from_id, edge.to_id)
        key = (edge.from_id, to_floor)
        if to_floor in fixed_destination_floors:
            if key in seen_fixed_starts:
                continue
            seen_fixed_starts.add(key)
        kept.append(edge)
    return kept


def _repair_generated_map_node_kinds(
    nodes: Sequence[MapNodeState],
    edges: Sequence[MapEdgeState],
    *,
    act: int,
    boss_floor: int,
    treasure_floor: int,
    rest_floor: int,
) -> list[MapNodeState]:
    incoming: dict[str, list[str]] = {}
    outgoing: dict[str, list[str]] = {}
    for edge in edges:
        outgoing.setdefault(edge.from_id, []).append(edge.to_id)
        incoming.setdefault(edge.to_id, []).append(edge.from_id)

    sorted_nodes = sorted(nodes, key=lambda node: (node.floor, node.lane, node.node_id))
    assigned = {node.node_id: node.kind for node in sorted_nodes}
    for node in sorted_nodes:
        forced = _forced_map_kind_for_floor(
            node.floor,
            boss_floor=boss_floor,
            treasure_floor=treasure_floor,
            rest_floor=rest_floor,
        )
        if forced is not None:
            assigned[node.node_id] = forced

    for node in sorted_nodes:
        if _forced_map_kind_for_floor(
            node.floor,
            boss_floor=boss_floor,
            treasure_floor=treasure_floor,
            rest_floor=rest_floor,
        ) is not None:
            continue
        for candidate in _candidate_map_kinds_for_floor(
            node.floor,
            node.lane,
            node.kind,
            act=act,
            boss_floor=boss_floor,
            rest_floor=rest_floor,
        ):
            if _candidate_map_kind_is_valid(
                candidate,
                node,
                assigned=assigned,
                incoming=incoming,
                outgoing=outgoing,
            ):
                assigned[node.node_id] = candidate
                break

    return [node.model_copy(update={"kind": assigned[node.node_id]}) for node in nodes]


def _forced_map_kind_for_floor(
    floor: int,
    *,
    boss_floor: int,
    treasure_floor: int,
    rest_floor: int,
) -> RoomKind | None:
    if floor == 0:
        return RoomKind.START
    if floor == boss_floor:
        return RoomKind.BOSS
    if floor == rest_floor:
        return RoomKind.REST
    if floor == treasure_floor:
        return RoomKind.TREASURE
    if floor == 1:
        return RoomKind.MONSTER
    return None


def _candidate_map_kinds_for_floor(
    floor: int,
    lane: int,
    initial_kind: RoomKind,
    *,
    act: int,
    boss_floor: int,
    rest_floor: int,
) -> tuple[RoomKind, ...]:
    base: tuple[RoomKind, ...]
    if floor <= 5:
        base = (RoomKind.MONSTER, RoomKind.EVENT, RoomKind.SHOP)
    elif floor == rest_floor - 1:
        base = (
            RoomKind.MONSTER,
            RoomKind.EVENT,
            RoomKind.SHOP,
            RoomKind.ELITE,
            RoomKind.TREASURE,
        )
    else:
        base = (
            RoomKind.MONSTER,
            RoomKind.EVENT,
            RoomKind.SHOP,
            RoomKind.ELITE,
            RoomKind.REST,
            RoomKind.TREASURE,
        )

    offset = (floor + lane + act) % len(base)
    ordered = (initial_kind, *base[offset:], *base[:offset])
    unique: list[RoomKind] = []
    for kind in ordered:
        if kind not in unique and kind in base:
            unique.append(kind)
    return tuple(unique)


def _candidate_map_kind_is_valid(
    candidate: RoomKind,
    node: MapNodeState,
    *,
    assigned: Mapping[str, RoomKind],
    incoming: Mapping[str, Sequence[str]],
    outgoing: Mapping[str, Sequence[str]],
) -> bool:
    if candidate in _MAP_NON_CONSECUTIVE_KINDS:
        for predecessor_id in incoming.get(node.node_id, ()):
            if assigned.get(predecessor_id) is candidate:
                return False
        for successor_id in outgoing.get(node.node_id, ()):
            if assigned.get(successor_id) is candidate:
                return False

    for predecessor_id in incoming.get(node.node_id, ()):
        for sibling_id in outgoing.get(predecessor_id, ()):
            if sibling_id != node.node_id and assigned.get(sibling_id) is candidate:
                return False
    return True


def _generate_lane_path(
    boss_floor: int,
    width: int,
    rng: Random,
    start_lane: int,
    *,
    first_lane: int | None = None,
) -> list[int]:
    path = [start_lane]
    lane = start_lane
    for floor in range(1, boss_floor):
        if first_lane is not None and floor == 1:
            lane = first_lane
        else:
            choices = _adjacent_lanes(lane, width)
            remaining_steps_to_rest = (boss_floor - 1) - floor
            if remaining_steps_to_rest <= 2:
                max_distance = remaining_steps_to_rest + 1
                funnel_choices = [
                    candidate
                    for candidate in choices
                    if abs(candidate - start_lane) <= max_distance
                ]
                choices = funnel_choices or choices
            # Bias toward moving sometimes, while still allowing straight paths.
            if len(choices) > 1 and rng.randrange(100) < 65:
                moving_choices = [candidate for candidate in choices if candidate != lane]
                lane = rng.choice(moving_choices)
            else:
                lane = rng.choice(choices)
        path.append(lane)
    path.append(start_lane)
    return path


def _adjacent_lanes(lane: int, width: int) -> list[int]:
    return [candidate for candidate in range(width) if abs(candidate - lane) <= 1]


def _prune_nodes_to_edges(
    nodes: Sequence[MapNodeState], edges: Sequence[MapEdgeState]
) -> tuple[list[MapNodeState], list[MapEdgeState]]:
    referenced = {edge.from_id for edge in edges} | {edge.to_id for edge in edges}
    kept_nodes = [node for node in nodes if node.node_id in referenced]
    kept_node_ids = {node.node_id for node in kept_nodes}
    kept_edges = [
        edge for edge in edges if edge.from_id in kept_node_ids and edge.to_id in kept_node_ids
    ]
    return kept_nodes, kept_edges


def _edge_sort_key(edge_key: tuple[str, str]) -> tuple[int, int, int, int]:
    from_id, to_id = edge_key
    return _edge_parts(from_id, to_id)


def _edge_state_sort_key(edge: MapEdgeState) -> tuple[int, int, int, int]:
    return _edge_parts(edge.from_id, edge.to_id)


def _edge_parts(from_id: str, to_id: str) -> tuple[int, int, int, int]:
    _, from_floor, from_lane = from_id.split(":")
    _, to_floor, to_lane = to_id.split(":")
    return (int(from_floor), int(from_lane), int(to_floor), int(to_lane))


def _reachable_node_ids(state: RunState) -> tuple[str, ...]:
    if state.map is None or state.map.current_node_id is None:
        return ()
    completed = set(state.map.completed_node_ids)
    outgoing = state.map.outgoing_by_id.get(state.map.current_node_id, ())
    return tuple(node_id for node_id in outgoing if node_id not in completed)


def _monsters_for_node(
    state: RunState,
    node: MapNodeState,
    rng: Random,
) -> tuple[MonsterState, ...]:
    if _uses_placeholder_monsters(state):
        return _fallback_monsters_for_node(node)

    monster_definitions = _monster_definitions(state)
    encounter = _encounter_for_node(state, node, rng)
    if monster_definitions and encounter is not None:
        spawned = spawn_monsters(
            encounter,
            monster_definitions,
            rng,
            ascension_level=state.ascension,
        )
        monsters = tuple(
            _monster_state_from_spawn(
                spawned_monster,
                monster_definitions[spawned_monster.source_monster_id],
                state,
            )
            for spawned_monster in spawned
            if spawned_monster.source_monster_id in monster_definitions
        )
        if monsters:
            return monsters

    return _fallback_monsters_for_node(node)


def _uses_placeholder_monsters(state: RunState) -> bool:
    if state.character_id.upper() != "TEST":
        return False
    return not any(
        key in state.flags
        for key in (
            "monsters",
            "monster_pool",
            "encounters",
            "encounter_pool",
            "combat_encounter_id",
            "monster_encounter_id",
            "encounter_id",
            "combat_monster_ids",
            "combat_monster_id",
            "monster_ids",
            "monster_id",
        )
    )


def _monster_definitions(state: RunState) -> Mapping[str, MonsterDefinition]:
    raw_monsters = (
        _source_items(state.flags.get("monsters"))
        or _source_items(state.flags.get("monster_pool"))
        or _cached_source_rows(state, "monsters")
    )
    return build_monster_definitions(raw_monsters)


def _encounter_for_node(
    state: RunState,
    node: MapNodeState,
    rng: Random,
) -> EncounterDefinition | None:
    explicit_monster_ids = (
        _flag_str_sequence(state, "combat_monster_ids")
        or _flag_str_sequence(state, "monster_ids")
        or _flag_str_sequence(state, "combat_monster_id")
        or _flag_str_sequence(state, "monster_id")
    )
    if explicit_monster_ids:
        return synthetic_encounter(
            encounter_id=str(
                state.flags.get("combat_encounter_id", f"{node.node_id}:synthetic_encounter")
            ),
            room_type=node.kind.value,
            monster_ids=explicit_monster_ids,
            act_number=node.act,
        )

    raw_encounters = (
        _source_items(state.flags.get("encounters"))
        or _source_items(state.flags.get("encounter_pool"))
        or _cached_source_rows(state, "encounters")
    )
    encounters = build_encounter_definitions(raw_encounters)
    if not encounters:
        return None

    return choose_encounter(
        encounters,
        rng,
        act=node.act,
        room_type=node.kind.value,
        preferred_id=_preferred_encounter_id(state),
        prefer_weak=node.kind == RoomKind.MONSTER and node.floor <= 3,
    )


def _preferred_encounter_id(state: RunState) -> str | None:
    for key in (
        "combat_encounter_id",
        "monster_encounter_id",
        "encounter_id",
        "event_fight_id",
        "combat_encounter",
    ):
        value = state.flags.get(key)
        if isinstance(value, str) and value.strip().lower() not in {
            "normal",
            "monster",
            "elite",
            "boss",
            "event",
        }:
            return value
    return None


def _monster_state_from_spawn(
    spawned_monster: SpawnedMonster,
    definition: MonsterDefinition,
    state: RunState,
) -> MonsterState:
    statuses = _monster_innate_statuses(definition, state.ascension)
    move = spawned_monster.move
    intent_damage = 0
    intent_block = 0
    hit_count = 1
    move_metadata: dict[str, Any] = {}
    if move is not None:
        intent_damage = _monster_intent_damage(
            definition,
            move,
            statuses=statuses,
            player=state.player,
            ascension_level=state.ascension,
        )
        intent_block = move.block
        hit_count = move.hit_count
        move_metadata = _monster_move_metadata(definition, move, state.ascension)

    metadata = {
        "source_monster_id": spawned_monster.source_monster_id,
        "encounter_id": spawned_monster.encounter_id,
        "slot_index": spawned_monster.slot_index,
        "move_counts": {},
        "innate_powers": [
            power.as_metadata() for power in spawned_monster.innate_powers if power.power_id
        ],
        **move_metadata,
    }
    return MonsterState(
        monster_id=spawned_monster.instance_id,
        name=spawned_monster.name,
        hp=spawned_monster.hp,
        max_hp=spawned_monster.max_hp,
        intent=move.intent if move is not None else None,
        intent_damage=intent_damage,
        intent_block=intent_block,
        move_id=move.move_id if move is not None else None,
        next_move_id=spawned_monster.next_move_id,
        hit_count=hit_count,
        statuses=statuses,
        metadata=metadata,
    )


def _monster_innate_statuses(
    definition: MonsterDefinition,
    ascension_level: int,
) -> dict[str, int]:
    statuses: dict[str, int] = {}
    for power in definition.innate_powers:
        if not power.power_id:
            continue
        status = _normalized_id(power.power_id)
        amount = monster_power_amount(definition, power, ascension_level=ascension_level)
        if amount:
            statuses[status] = statuses.get(status, 0) + amount
    return statuses


def _monster_move_metadata(
    definition: MonsterDefinition,
    move: MonsterMove,
    ascension_level: int,
) -> dict[str, Any]:
    damage_per_hit = monster_move_damage(definition, move, ascension_level=ascension_level)
    return {
        "move_id": move.move_id,
        "move_name": move.name,
        "damage_per_hit": damage_per_hit,
        "hit_count": move.hit_count,
        "base_intent_damage": damage_per_hit * move.hit_count,
        "move_heal": move.heal,
        "move_powers": [power.as_metadata() for power in move.powers if power.power_id],
    }


def _monster_intent_damage(
    definition: MonsterDefinition,
    move: MonsterMove,
    *,
    statuses: Mapping[str, int],
    player: PlayerState,
    ascension_level: int,
) -> int:
    per_hit = _monster_attack_hit_damage(
        definition,
        move,
        statuses=statuses,
        ascension_level=ascension_level,
    )
    if _status_amount(player.statuses, "vulnerable") > 0:
        per_hit = int(per_hit * 1.5)
    return max(0, per_hit) * move.hit_count


def _monster_attack_hit_damage(
    definition: MonsterDefinition,
    move: MonsterMove,
    *,
    statuses: Mapping[str, int],
    ascension_level: int,
) -> int:
    damage = monster_move_damage(definition, move, ascension_level=ascension_level)
    if damage <= 0:
        return 0
    damage += _status_amount(statuses, "strength")
    if _status_amount(statuses, "weak") > 0:
        damage = int(damage * 0.75)
    return max(0, damage)


def _fallback_monsters_for_node(node: MapNodeState) -> tuple[MonsterState, ...]:
    hp_bonus = (node.act - 1) * 10 + node.floor
    damage_bonus = node.act - 1
    if node.kind == RoomKind.BOSS:
        return (
            MonsterState(
                monster_id=f"act_{node.act}_boss",
                name=f"Act {node.act} Boss",
                hp=70 + hp_bonus * 2,
                max_hp=70 + hp_bonus * 2,
                intent="attack",
                intent_damage=10 + node.act * 2,
            ),
        )
    if node.kind == RoomKind.ELITE:
        return (
            MonsterState(
                monster_id=f"act_{node.act}_elite",
                name=f"Act {node.act} Elite",
                hp=45 + hp_bonus,
                max_hp=45 + hp_bonus,
                intent="attack",
                intent_damage=8 + damage_bonus,
            ),
        )
    return (
        MonsterState(
            monster_id=f"act_{node.act}_monster_{node.floor}_{node.lane}",
            name=f"Act {node.act} Monster",
            hp=25 + hp_bonus,
            max_hp=25 + hp_bonus,
            intent="attack",
            intent_damage=5 + damage_bonus,
        ),
    )


def _max_acts(state: RunState) -> int:
    value = state.flags.get("max_acts", 3)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 3


def _potion_capacity(state: RunState, relics: Sequence[str] | None = None) -> int:
    base_slots = _flag_int(state, "base_potion_slots", 3)
    explicit_slots = state.flags.get("potion_slots")
    if explicit_slots is not None:
        with suppress(TypeError, ValueError):
            base_slots = int(explicit_slots)

    capacity = potion_slots_for_ascension(base_slots, state.ascension)
    capacity += _flag_int(state, "bonus_potion_slots", 0)

    owned = {
        _normalized_id(relic_id)
        for relic_id in (state.relics if relics is None else relics)
    }
    if "potion_belt" in owned:
        capacity += 2
    if "alchemical_coffer" in owned:
        capacity += 4
    if "phial_holster" in owned:
        capacity += 1
    return max(0, capacity)


def _has_open_potion_slot(state: RunState) -> bool:
    return len(state.potions) < _potion_capacity(state)


def _flag_int(state: RunState, key: str, default: int) -> int:
    try:
        return int(state.flags.get(key, default))
    except (TypeError, ValueError):
        return default


def _flag_float(state: RunState, key: str, default: float) -> float:
    try:
        return float(state.flags.get(key, default))
    except (TypeError, ValueError):
        return default


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _flag_bool(state: RunState, key: str, default: bool) -> bool:
    value = state.flags.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _flag_str_sequence(state: RunState, key: str) -> tuple[str, ...]:
    value = state.flags.get(key, ())
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value)
    return ()


def _dig_relic_id(state: RunState, rng: Random) -> str:
    pool = (
        _flag_str_sequence(state, "campfire_dig_relic_pool")
        or _flag_str_sequence(state, "relic_pool")
    )
    if pool:
        candidates = [relic_id for relic_id in pool if relic_id not in state.relics]
        return rng.choice(candidates or list(pool))
    return f"campfire_dig_relic_{_flag_int(state, 'campfire_dig_count', 0) + 1}"


def _has_relic(state: RunState, *relic_ids: str) -> bool:
    owned = {_normalized_id(relic_id) for relic_id in state.relics}
    return any(_normalized_id(relic_id) in owned for relic_id in relic_ids)


def _normalized_id(value: str) -> str:
    return value.lower().replace("'", "").replace(" ", "_").replace("-", "_")


_SUPPORTED_EFFECT_KEYS = {
    "add_card_to_discard",
    "add_card_to_draw",
    "add_card_to_exhaust",
    "add_card_to_hand",
    "add_random_card_to_hand",
    "all_damage",
    "apply_status",
    "block",
    "channel_orb",
    "damage",
    "destination",
    "discard_choice",
    "discard_hand",
    "discard_random",
    "draw",
    "effects",
    "energy",
    "evoke_orb",
    "exhaust_hand",
    "exhaust_random",
    "exhaust_on_play",
    "heal",
    "hp_loss",
    "max_hp",
    "next_turn",
    "orb_slot_delta",
    "player_resource",
    "resource",
    "retain_hand",
    "sequence",
    "status",
}
_CARD_RUNTIME_KEYS = frozenset(
    {
        "card_id",
        "cost",
        "custom",
        "effects",
        "enchantments",
        "exhaust",
        "exhausts",
        "instance_id",
        "name",
        "tags",
        "target",
        "type",
        "upgraded",
    }
)
_STATUS_LOOKUP_ALIASES = {
    "child_of_the_stars": ("childofthestars",),
    "black_hole": ("blackhole",),
}
_STATUS_APPLICATION_ALIASES = {
    "strengthdown": "strength_down",
    "dexteritydown": "dexterity_down",
    "calm": "stance_calm",
    "divinity": "stance_divinity",
    "wrath": "stance_wrath",
}
_STANCE_STATUS_IDS = frozenset({"stance_calm", "stance_divinity", "stance_wrath"})
_STANCE_EXIT_STATUS_IDS = frozenset({"exit_stance", "no_stance", "stance_none"})
_ARTIFACT_BLOCKED_STATUSES = frozenset(
    {
        "choking",
        "frail",
        "poison",
        "slow",
        "vulnerable",
        "weak",
    }
)
_ARTIFACT_IGNORED_STATUSES = frozenset(
    {
        "temporary_dexterity",
        "temporary_strength",
    }
)
_ORB_TYPES = ("lightning", "frost", "dark", "plasma", "glass")
