"""Bounded action previews for agent-facing action descriptors."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from sts2sim.api import legal_actions as _legal_actions
from sts2sim.api import serialize as _serialize
from sts2sim.api import step as _step
from sts2sim.engine.models import Action, ActionType

SECOND_TURN_ACTION_LIMIT = 32


def preview_action_key(action: Action | Mapping[str, Any]) -> str:
    """Return the same canonical action key used by agent action IDs."""

    return _canonical_json(_action_payload(action))


def preview_actions(
    state: Any,
    actions: Sequence[Action],
    *,
    second_turn_action_limit: int = SECOND_TURN_ACTION_LIMIT,
) -> dict[str, dict[str, Any]]:
    """Preview legal actions without mutating the original state.

    The preview is intentionally bounded: it simulates the selected action, an
    optional forced end turn, and one aggregate pass over the next visible legal
    actions. This keeps the PPO input useful without turning action enumeration
    into a full tree search.
    """

    before_payload = _serialize(state)
    previews: dict[str, dict[str, Any]] = {}
    for action in actions:
        key = preview_action_key(action)
        try:
            next_state = _step(state, action)
            next_payload = _serialize(next_state)
            preview = _transition_preview(before_payload, next_payload, action)
            preview.update(
                _combat_lookahead_preview(
                    before_state=state,
                    before_payload=before_payload,
                    action=action,
                    after_state=next_state,
                    after_payload=next_payload,
                    second_turn_action_limit=second_turn_action_limit,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive runtime capture
            preview = {
                "preview_error": 1,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:200],
            }
        previews[key] = preview
    return previews


def _transition_preview(
    before_payload: Mapping[str, Any],
    after_payload: Mapping[str, Any],
    action: Action,
) -> dict[str, Any]:
    before_player = _effective_player(before_payload)
    after_player = _effective_player(after_payload)
    before_root_player = _mapping(before_payload.get("player"))
    after_root_player = _mapping(after_payload.get("player"))
    before_combat = _mapping(before_payload.get("combat"))
    after_combat = _mapping(after_payload.get("combat"))
    before_monsters = _monster_by_id(before_payload)
    after_monsters = _monster_by_id(after_payload)
    target_id = str(action.target_id or "")
    before_target = _mapping(before_monsters.get(target_id))
    after_target = _mapping(after_monsters.get(target_id))
    before_alive = _alive_monster_count(before_payload)
    after_alive = _alive_monster_count(after_payload)
    before_reward = _reward_counts(before_payload)
    after_reward = _reward_counts(after_payload)
    before_shop = _shop_summary(before_payload)
    after_shop = _shop_summary(after_payload)

    return {
        "preview_error": 0,
        "phase_changed": int(str(before_payload.get("phase")) != str(after_payload.get("phase"))),
        "terminal": int(str(after_payload.get("phase", "")) in {"complete", "failed"}),
        "act_delta": _int(after_payload.get("act")) - _int(before_payload.get("act")),
        "floor_delta": _int(after_payload.get("floor")) - _int(before_payload.get("floor")),
        "player_hp_delta": _int(after_player.get("hp")) - _int(before_player.get("hp")),
        "player_block_delta": _int(after_player.get("block")) - _int(before_player.get("block")),
        "player_energy_delta": _int(after_player.get("energy")) - _int(before_player.get("energy")),
        "player_gold_delta": _int(after_root_player.get("gold"))
        - _int(before_root_player.get("gold")),
        "player_max_hp_delta": _int(after_player.get("max_hp"))
        - _int(before_player.get("max_hp")),
        "deck_count_delta": len(_sequence(after_payload.get("master_deck")))
        - len(_sequence(before_payload.get("master_deck"))),
        "relic_count_delta": len(_sequence(after_payload.get("relics")))
        - len(_sequence(before_payload.get("relics"))),
        "potion_count_delta": len(_sequence(after_payload.get("potions")))
        - len(_sequence(before_payload.get("potions"))),
        "target_is_monster": int(bool(before_target or after_target)),
        "target_hp_delta": _int(after_target.get("hp")) - _int(before_target.get("hp")),
        "target_block_delta": _int(after_target.get("block"))
        - _int(before_target.get("block")),
        "monster_hp_total_delta": _monster_hp_total(after_payload)
        - _monster_hp_total(before_payload),
        "monster_block_total_delta": _monster_block_total(after_payload)
        - _monster_block_total(before_payload),
        "alive_monster_delta": after_alive - before_alive,
        "kills": max(0, before_alive - after_alive),
        "incoming_damage_delta": _incoming_damage(after_payload)
        - _incoming_damage(before_payload),
        "hand_delta": _zone_count(after_combat, "hand") - _zone_count(before_combat, "hand"),
        "draw_pile_delta": _zone_count(after_combat, "draw_pile")
        - _zone_count(before_combat, "draw_pile"),
        "discard_pile_delta": _zone_count(after_combat, "discard_pile")
        - _zone_count(before_combat, "discard_pile"),
        "exhaust_pile_delta": _zone_count(after_combat, "exhaust_pile")
        - _zone_count(before_combat, "exhaust_pile"),
        "reward_opened": int(
            not _mapping(before_payload.get("reward")) and bool(after_reward["total"])
        ),
        "reward_card_count_delta": after_reward["cards"] - before_reward["cards"],
        "reward_relic_count_delta": after_reward["relics"] - before_reward["relics"],
        "reward_potion_count_delta": after_reward["potions"] - before_reward["potions"],
        "reward_gold_delta": after_reward["gold"] - before_reward["gold"],
        "shop_available_item_delta": after_shop["available_items"] - before_shop["available_items"],
        "shop_price_total_delta": after_shop["price_total"] - before_shop["price_total"],
        "ended_turn": int(
            action.type == ActionType.END_TURN
            or _int(after_combat.get("turn")) > _int(before_combat.get("turn"))
        ),
        "combat_ended": int(bool(before_combat) and not bool(after_combat)),
    }


def _combat_lookahead_preview(
    *,
    before_state: Any,
    before_payload: Mapping[str, Any],
    action: Action,
    after_state: Any,
    after_payload: Mapping[str, Any],
    second_turn_action_limit: int,
) -> dict[str, Any]:
    del before_state
    if not _mapping(before_payload.get("combat")):
        return {}
    if not _mapping(after_payload.get("combat")):
        return {"lookahead_combat": 1, "lookahead_combat_ended": 1}

    end_turn_available = 0
    end_state = after_state
    end_payload = after_payload
    end_turn_error = 0
    if action.type != ActionType.END_TURN:
        end_turn = _first_action(after_state, ActionType.END_TURN)
        if end_turn is not None:
            end_turn_available = 1
            try:
                end_state = _step(after_state, end_turn)
                end_payload = _serialize(end_state)
            except Exception:  # pragma: no cover - defensive runtime capture
                end_turn_error = 1
                end_state = after_state
                end_payload = after_payload
    else:
        end_turn_available = 1

    before_player = _effective_player(before_payload)
    end_player = _effective_player(end_payload)
    turn_start_payload = before_payload if action.type == ActionType.END_TURN else after_payload
    turn_sequence = _enemy_turn_preview(
        start_payload=turn_start_payload,
        end_payload=end_payload,
        action=action,
    )
    aggregate = _second_turn_aggregate(
        end_state,
        end_payload,
        second_turn_action_limit=second_turn_action_limit,
    )
    preview = {
        "lookahead_combat": 1,
        "lookahead_combat_ended": int(not bool(_mapping(end_payload.get("combat")))),
        "end_turn_available": end_turn_available,
        "end_turn_preview_error": end_turn_error,
        "projected_player_hp_delta_after_end": _int(end_player.get("hp"))
        - _int(before_player.get("hp")),
        "projected_damage_taken_after_end": turn_sequence["enemy_turn_damage_taken"],
        "next_turn_number": _int(_mapping(end_payload.get("combat")).get("turn")),
        "next_turn_player_hp": _int(end_player.get("hp")),
        "next_turn_player_block": _int(end_player.get("block")),
        "next_turn_player_energy": _int(end_player.get("energy")),
        "next_turn_hand_count": _zone_count(_mapping(end_payload.get("combat")), "hand"),
        "next_turn_draw_pile_count": _zone_count(
            _mapping(end_payload.get("combat")), "draw_pile"
        ),
        "next_turn_discard_pile_count": _zone_count(
            _mapping(end_payload.get("combat")), "discard_pile"
        ),
        "next_turn_exhaust_pile_count": _zone_count(
            _mapping(end_payload.get("combat")), "exhaust_pile"
        ),
        "next_turn_incoming_damage": _incoming_damage(end_payload),
    }
    preview.update(turn_sequence)
    preview.update(aggregate)
    return preview


def _second_turn_aggregate(
    state: Any,
    payload: Mapping[str, Any],
    *,
    second_turn_action_limit: int,
) -> dict[str, Any]:
    if not _mapping(payload.get("combat")):
        return {
            "second_turn_legal_action_count": 0,
            "second_turn_previewed_action_count": 0,
        }
    baseline_hp_total = _monster_hp_total(payload)
    baseline_alive = _alive_monster_count(payload)
    baseline_block = _int(_effective_player(payload).get("block"))
    baseline_hp = _int(_effective_player(payload).get("hp"))
    actions = [
        action
        for action in _safe_legal_actions(state)
        if getattr(action, "type", None) != ActionType.END_TURN
    ]
    previewed = 0
    errors = 0
    best_damage = 0
    best_block = 0
    best_hp_delta = -10_000
    kill_available = 0
    lethal_available = 0
    for action in actions[: max(0, second_turn_action_limit)]:
        try:
            next_state = _step(state, action)
            next_payload = _serialize(next_state)
        except Exception:  # pragma: no cover - defensive runtime capture
            errors += 1
            continue
        previewed += 1
        next_hp_total = _monster_hp_total(next_payload)
        next_alive = _alive_monster_count(next_payload)
        next_player = _effective_player(next_payload)
        best_damage = max(best_damage, max(0, baseline_hp_total - next_hp_total))
        best_block = max(best_block, max(0, _int(next_player.get("block")) - baseline_block))
        best_hp_delta = max(best_hp_delta, _int(next_player.get("hp")) - baseline_hp)
        if next_alive < baseline_alive:
            kill_available = 1
        if baseline_alive > 0 and next_alive == 0:
            lethal_available = 1
    return {
        "second_turn_legal_action_count": len(actions),
        "second_turn_previewed_action_count": previewed,
        "second_turn_preview_error_count": errors,
        "second_turn_best_damage": best_damage,
        "second_turn_best_block": best_block,
        "second_turn_best_hp_delta": 0 if best_hp_delta == -10_000 else best_hp_delta,
        "second_turn_kill_available": kill_available,
        "second_turn_lethal_available": lethal_available,
    }


def _enemy_turn_preview(
    *,
    start_payload: Mapping[str, Any],
    end_payload: Mapping[str, Any],
    action: Action,
) -> dict[str, int]:
    start_player = _effective_player(start_payload)
    end_player = _effective_player(end_payload)
    start_alive = _alive_monster_count(start_payload)
    end_alive = _alive_monster_count(end_payload)
    start_player_hp = _int(start_player.get("hp"))
    end_player_hp = _int(end_player.get("hp"))
    start_combat = _mapping(start_payload.get("combat"))
    end_combat = _mapping(end_payload.get("combat"))
    events = _last_events(end_payload)
    event_summary = _enemy_turn_event_summary(events)
    return {
        "enemy_turn_available": int(
            bool(start_combat) and (action.type == ActionType.END_TURN or bool(end_combat))
        ),
        "enemy_turn_player_hp_delta": end_player_hp - start_player_hp,
        "enemy_turn_damage_taken": max(0, start_player_hp - end_player_hp),
        "enemy_turn_player_block_delta": _int(end_player.get("block"))
        - _int(start_player.get("block")),
        "enemy_turn_player_status_delta": _status_total(end_player)
        - _status_total(start_player),
        "enemy_turn_monster_hp_delta": _monster_hp_total(end_payload)
        - _monster_hp_total(start_payload),
        "enemy_turn_monster_block_delta": _monster_block_total(end_payload)
        - _monster_block_total(start_payload),
        "enemy_turn_monster_status_delta": _monster_status_total(end_payload)
        - _monster_status_total(start_payload),
        "enemy_turn_monsters_killed": max(0, start_alive - end_alive),
        "enemy_turn_retaliation_damage": event_summary["retaliation_damage"],
        "enemy_turn_retaliation_kills": event_summary["retaliation_kills"],
        "enemy_turn_poison_damage": event_summary["poison_damage"],
        "enemy_turn_self_damage": event_summary["self_damage"],
        "enemy_turn_player_damage_events": event_summary["player_damage_events"],
        "enemy_turn_monster_attack_events": event_summary["monster_attack_events"],
        "enemy_turn_block_events": event_summary["block_events"],
        "enemy_turn_buff_events": event_summary["buff_events"],
        "enemy_turn_debuff_events": event_summary["debuff_events"],
        "enemy_turn_next_incoming_damage": _incoming_damage(end_payload),
        "enemy_turn_survives": int(end_player_hp > 0),
        "enemy_turn_death_pending": int(start_player_hp > 0 and end_player_hp <= 0),
    }


def _enemy_turn_event_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    summary = {
        "retaliation_damage": 0,
        "retaliation_kills": 0,
        "poison_damage": 0,
        "self_damage": 0,
        "player_damage_events": 0,
        "monster_attack_events": 0,
        "block_events": 0,
        "buff_events": 0,
        "debuff_events": 0,
    }
    for event in events:
        kind = str(event.get("kind", ""))
        metadata = _mapping(event.get("metadata"))
        amount = _int(event.get("amount"))
        status = str(metadata.get("status", ""))
        if kind == "player_damaged":
            summary["player_damage_events"] += 1
            if event.get("source_id"):
                summary["monster_attack_events"] += 1
        if kind == "monster_damaged" and status == "thorns":
            summary["retaliation_damage"] += amount
        elif kind == "monster_damaged" and status == "poison":
            summary["poison_damage"] += amount
        elif kind in {"monster_self_damaged", "monster_self_destructed"}:
            summary["self_damage"] += amount
        if kind == "monster_defeated" and status == "thorns":
            summary["retaliation_kills"] += 1
        if "block" in kind:
            summary["block_events"] += 1
        if _event_is_buff(kind, metadata):
            summary["buff_events"] += 1
        if _event_is_debuff(kind, metadata):
            summary["debuff_events"] += 1
    return summary


def _event_is_buff(kind: str, metadata: Mapping[str, Any]) -> bool:
    status = str(metadata.get("status", "")).lower()
    if "buff" in kind or kind in {"monster_strength", "monster_healed", "monster_block"}:
        return True
    return status in {
        "strength",
        "ritual",
        "metallicize",
        "plated_armor",
        "thorns",
        "regen",
        "buffer",
        "artifact",
    }


def _event_is_debuff(kind: str, metadata: Mapping[str, Any]) -> bool:
    status = str(metadata.get("status", "")).lower()
    if "debuff" in kind:
        return True
    return status in {
        "weak",
        "vulnerable",
        "frail",
        "poison",
        "doom",
        "temporary_strength",
        "hex",
        "stunned",
    }


def _last_events(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        _mapping(event)
        for event in _sequence(_mapping(payload.get("combat")).get("last_events"))
    )


def _first_action(state: Any, action_type: ActionType) -> Action | None:
    for action in _safe_legal_actions(state):
        if getattr(action, "type", None) == action_type:
            return action
    return None


def _safe_legal_actions(state: Any) -> tuple[Action, ...]:
    try:
        return tuple(_legal_actions(state))
    except Exception:  # pragma: no cover - defensive runtime capture
        return ()


def _effective_player(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    combat_player = _mapping(_mapping(payload.get("combat")).get("player"))
    return combat_player or _mapping(payload.get("player"))


def _monster_by_id(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(_mapping(monster).get("monster_id", "")): _mapping(monster)
        for monster in _sequence(_mapping(payload.get("combat")).get("monsters"))
    }


def _monster_hp_total(payload: Mapping[str, Any]) -> int:
    return sum(
        _int(_mapping(monster).get("hp"))
        for monster in _sequence(_mapping(payload.get("combat")).get("monsters"))
    )


def _monster_block_total(payload: Mapping[str, Any]) -> int:
    return sum(
        _int(_mapping(monster).get("block"))
        for monster in _sequence(_mapping(payload.get("combat")).get("monsters"))
    )


def _monster_status_total(payload: Mapping[str, Any]) -> int:
    return sum(
        _status_total(_mapping(monster))
        for monster in _sequence(_mapping(payload.get("combat")).get("monsters"))
    )


def _status_total(entity: Mapping[str, Any]) -> int:
    return sum(_int(value) for value in _mapping(entity.get("statuses")).values())


def _alive_monster_count(payload: Mapping[str, Any]) -> int:
    return sum(
        1
        for monster in _sequence(_mapping(payload.get("combat")).get("monsters"))
        if _int(_mapping(monster).get("hp")) > 0
    )


def _incoming_damage(payload: Mapping[str, Any]) -> int:
    total = 0
    for monster in _sequence(_mapping(payload.get("combat")).get("monsters")):
        monster_map = _mapping(monster)
        if _int(monster_map.get("hp")) > 0:
            total += _int(monster_map.get("intent_damage"))
    return total


def _zone_count(combat: Mapping[str, Any], zone: str) -> int:
    return len(_sequence(combat.get(zone)))


def _reward_counts(payload: Mapping[str, Any]) -> dict[str, int]:
    reward = _mapping(payload.get("reward"))
    card_count = (
        len(_sequence(reward.get("card_options")))
        + len(_sequence(reward.get("card_ids")))
        + sum(len(_sequence(group)) for group in _sequence(reward.get("card_option_groups")))
    )
    relic_count = int(bool(reward.get("relic_id"))) + len(_sequence(reward.get("relic_ids")))
    potion_count = int(bool(reward.get("potion_id"))) + len(_sequence(reward.get("potion_ids")))
    gold = _int(reward.get("gold"))
    return {
        "cards": card_count,
        "relics": relic_count,
        "potions": potion_count,
        "gold": gold,
        "total": card_count + relic_count + potion_count + int(gold > 0),
    }


def _shop_summary(payload: Mapping[str, Any]) -> dict[str, int]:
    items = tuple(_sequence(_mapping(payload.get("shop")).get("items")))
    available = [
        _mapping(item) for item in items if not bool(_mapping(item).get("purchased", False))
    ]
    return {
        "available_items": len(available),
        "price_total": sum(_int(item.get("price")) for item in available),
    }


def _action_payload(action: Action | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(action, Action):
        return dict(action.model_dump(mode="json", exclude_none=True))
    return dict(Action.model_validate(dict(action)).model_dump(mode="json", exclude_none=True))


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


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


def _int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0
