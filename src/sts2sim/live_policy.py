"""Conservative bridge adapter for a validated PPO policy.

Only complete combat observations and round-trip verified card/end-turn
commands are supported. Other screens require an explicit heuristic fallback
in the caller; this adapter never substitutes an action from another policy.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from sts2sim.agent_api import action_space
from sts2sim.engine.models import Action, ActionType, RunState, TargetType
from sts2sim.learning.policy import MaskedPpoDecision, MaskedPpoPolicy
from sts2sim.live_parity import (
    DEFAULT_CARD_CACHE,
    DEFAULT_CHARACTER_CACHE,
    LiveParityUnsupported,
    _card_library,
    _cards_from_live_zone,
    _live_card_id,
    _source_card_id_to_runtime_id,
    _status_id,
    live_state_to_simulator_state,
    map_live_action_to_simulator_action,
)
from sts2sim.mechanics.relics import relic_potion_slot_bonus
from sts2sim.mechanics.rewards import potion_slots_for_ascension


class LivePolicyUnsupported(ValueError):
    """A bridge observation cannot safely be used by the model adapter."""


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: object, field: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise LivePolicyUnsupported(f"Missing or invalid live {field} list.")
    if not all(isinstance(item, Mapping) for item in value):
        raise LivePolicyUnsupported(f"Live {field} contains non-object entries.")
    return tuple(value)


def _integer(payload: Mapping[str, Any], key: str, *, minimum: int | None = 0) -> int:
    raw = payload.get(key)
    try:
        if isinstance(raw, bool) or raw is None or float(raw) != int(raw):
            raise ValueError
        number = int(raw)
    except (ValueError, TypeError, OverflowError) as exc:
        raise LivePolicyUnsupported(f"Missing or invalid live {key} integer.") from exc
    if minimum is not None and number < minimum:
        raise LivePolicyUnsupported(f"Live {key} is below {minimum}.")
    return number


def _statuses(rows: tuple[Mapping[str, Any], ...]) -> dict[str, int]:
    result = {}
    for status in rows:
        status_id = _status_id(status)
        if not status_id:
            raise LivePolicyUnsupported("Live status is missing its identity.")
        result[status_id] = _integer(status, "amount", minimum=None)
    return result


def import_policy_combat(
    state: Mapping[str, Any],
    *,
    seed: int | str = 0,
    card_cache: Path | str = DEFAULT_CARD_CACHE,
    character_cache: Path | str = DEFAULT_CHARACTER_CACHE,
) -> RunState:
    """Import visible combat fields without silently fabricating missing piles."""
    state_type = str(state.get("state_type", "")).lower()
    if state_type not in {"monster", "elite", "boss"}:
        raise LivePolicyUnsupported(
            f"PPO mapping supports combat only; screen={state_type or 'unknown'}."
        )
    player = _mapping(state.get("player"))
    if player.get("character_id"):
        # The prepared bridge exposes the stable model ID independently of display language.
        player = {**player, "character": player["character_id"]}
        state = {**state, "player": player}
    battle = _mapping(state.get("battle"))
    run = _mapping(state.get("run"))
    if battle.get("turn") != "player" or battle.get("is_play_phase") is not True:
        raise LivePolicyUnsupported("Live combat is not in the player's card-play phase.")
    _integer(battle, "round", minimum=1)
    act = _integer(run, "act", minimum=1)
    floor = _integer(run, "floor", minimum=1)
    _integer(run, "ascension")
    for field in ("hp", "max_hp", "block", "energy", "max_energy", "gold", "max_potion_slots"):
        _integer(player, field, minimum=1 if field in {"hp", "max_hp"} else 0)
    for field in ("status", "relics", "potions"):
        _rows(player.get(field), f"player.{field}")
    zones = {
        zone: _rows(player.get(zone), f"player.{zone}")
        for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile")
    }
    for zone, cards in zones.items():
        count_key = f"{zone}_count"
        if count_key in player and _integer(player, count_key) != len(cards):
            raise LivePolicyUnsupported(f"Live {zone} count differs from its visible card list.")
    for index, card in enumerate(zones["hand"]):
        if _integer(card, "index") != index or not isinstance(card.get("can_play"), bool):
            raise LivePolicyUnsupported("Live hand requires contiguous indices and can_play flags.")
    enemies = _rows(battle.get("enemies"), "battle.enemies")
    if not enemies:
        raise LivePolicyUnsupported("Live combat has no visible enemies.")
    enemy_ids = [str(enemy.get("entity_id") or "") for enemy in enemies]
    if not all(enemy_ids) or len(set(enemy_ids)) != len(enemy_ids):
        raise LivePolicyUnsupported("Live enemies require distinct entity IDs.")
    for enemy in enemies:
        for field in ("hp", "max_hp", "block"):
            _integer(enemy, field, minimum=1 if field == "max_hp" else 0)
        _rows(enemy.get("status"), "enemy.status")
        intents = _rows(enemy.get("intents"), "enemy.intents")
        if len(intents) != 1:
            raise LivePolicyUnsupported("PPO mapping requires one explicit intent per enemy.")
    try:
        sim = live_state_to_simulator_state(
            state,
            seed=seed,
            card_cache=card_cache,
            character_cache=character_cache,
        )
        library = _card_library(Path(card_cache))
    except (OSError, ValueError, LiveParityUnsupported) as exc:
        raise LivePolicyUnsupported(f"Live combat import failed: {exc}") from exc
    for cards in zones.values():
        for card_row in cards:
            raw_id = card_row.get("id")
            if raw_id and _source_card_id_to_runtime_id(str(raw_id), tuple(library)) not in library:
                raise LivePolicyUnsupported(f"Unknown live card ID: {raw_id}.")
            if _live_card_id(card_row, sim.character_id, library) not in library:
                raise LivePolicyUnsupported(
                    f"Unknown live card: {card_row.get('id', card_row.get('name'))}."
                )
            if (
                card_row.get("enchantments")
                or card_row.get("enchantment")
                or card_row.get("has_enchantment")
            ):
                raise LivePolicyUnsupported("Live card enchantments are not yet losslessly mapped.")
            if "upgrade_level" in card_row and _integer(card_row, "upgrade_level") > 1:
                raise LivePolicyUnsupported("Repeated card upgrades are not yet losslessly mapped.")
            if not isinstance(card_row.get("is_upgraded"), bool):
                raise LivePolicyUnsupported("Live cards must expose their upgrade state.")
    assert sim.combat is not None
    combat = sim.combat
    imported_player = combat.player.model_copy(
        update={
            field: _integer(player, field)
            for field in ("hp", "max_hp", "block", "energy", "max_energy", "gold")
        }
    )
    imported_player = imported_player.model_copy(
        update={
            "statuses": _statuses(_rows(player["status"], "player.status")),
        }
    )
    resources = player.get("resources")
    if sim.character_id not in {"IRONCLAD", "SILENT"}:
        raise LivePolicyUnsupported(
            f"PPO live mapping for {sim.character_id} requires "
            "additional orb/Osty/resource support."
        )
    if resources:
        raise LivePolicyUnsupported("Live nonstandard player resources are not yet mapped.")
    hand = []
    for hand_card, raw in zip(combat.hand, zones["hand"], strict=True):
        target_names = {
            "self": "self",
            "anyenemy": "enemy",
            "enemy": "enemy",
            "none": "none",
            "allenemies": "all_enemies",
            "any": "any",
        }
        raw_target = str(raw.get("target_type", "")).replace("_", "").replace(" ", "").lower()
        if target_names.get(raw_target) != hand_card.target.value:
            raise LivePolicyUnsupported("Live card target type does not match the source card.")
        cost = raw.get("cost")
        if str(cost).upper() == "X":
            hand.append(hand_card)
        else:
            hand.append(hand_card.model_copy(update={"cost": _integer(raw, "cost")}))
    monsters = []
    for monster, raw in zip(combat.monsters, enemies, strict=True):
        intent = _mapping(raw["intents"][0])
        kind = str(intent.get("type", "")).lower()
        damage, hits = 0, 1
        if "attack" in kind:
            label = str(intent.get("label", ""))
            match = re.fullmatch(r"\s*(\d+)(?:\s*[x×]\s*(\d+))?\s*", label)
            if match is None:
                raise LivePolicyUnsupported(f"Unmapped live attack intent amount: {label!r}.")
            damage = int(match[1])
            hits = int(match[2] or 1)
        if hits < 1:
            raise LivePolicyUnsupported("Live attack hit count must be positive.")
        monsters.append(
            monster.model_copy(
                update={
                    "intent": kind,
                    "intent_damage": damage,
                    "hit_count": hits,
                    "statuses": _statuses(_rows(raw["status"], "enemy.status")),
                }
            )
        )
    combat = combat.model_copy(
        update={"player": imported_player, "hand": tuple(hand), "monsters": tuple(monsters)}
    )
    raw_deck = player.get("master_deck", player.get("deck"))
    if raw_deck is not None:
        deck_rows = _rows(raw_deck, "player.deck")
        if any(
            _live_card_id(card, sim.character_id, library) not in library
            or (
                card.get("id")
                and _source_card_id_to_runtime_id(str(card["id"]), tuple(library)) not in library
            )
            for card in deck_rows
        ):
            raise LivePolicyUnsupported("Live master deck contains unknown cards.")
        if any(
            not isinstance(card.get("is_upgraded"), bool)
            or card.get("enchantments")
            or card.get("enchantment")
            or card.get("has_enchantment")
            or ("upgrade_level" in card and _integer(card, "upgrade_level") > 1)
            for card in deck_rows
        ):
            raise LivePolicyUnsupported(
                "Live master deck upgrades/enchantments are not losslessly mapped."
            )
        master_deck = _cards_from_live_zone(
            deck_rows,
            zone="master",
            character_id=sim.character_id,
            card_library=library,
            start_index=1001,
        )
    else:
        visible = (*combat.hand, *combat.draw_pile, *combat.discard_pile, *combat.exhaust_pile)

        def signature(cards: Sequence[Any]) -> Counter[tuple[str, bool]]:
            return Counter((card.card_id.lower(), card.upgraded) for card in cards)

        if act != 1 or floor != 1 or signature(visible) != signature(sim.master_deck):
            raise LivePolicyUnsupported(
                "Live snapshot lacks a master deck; visible piles cannot verify the starter deck."
            )
        master_deck = sim.master_deck
    # The bridge exposes final capacity; simulator flags represent capacity
    # before ascension/relic modifiers, so reverse those known modifiers.
    reduction = 100 - potion_slots_for_ascension(100, sim.ascension)
    base_slots = (
        _integer(player, "max_potion_slots") + reduction - relic_potion_slot_bonus(sim.relics)
    )
    if base_slots < 0:
        raise LivePolicyUnsupported("Live potion capacity conflicts with owned relic bonuses.")
    flags = {**sim.flags, "potion_slots": base_slots, "bonus_potion_slots": 0}
    # Map generation and a starter deck from new_run are not live observations.
    # Only the validated deck survives; unknown map context is represented empty.
    return sim.model_copy(
        update={
            "player": imported_player,
            "combat": combat,
            "master_deck": master_deck,
            "map": None,
            "flags": flags,
        }
    )


def bridge_action_for_policy(
    live: Mapping[str, Any],
    sim: RunState,
    action: Action,
) -> dict[str, Any] | None:
    """Return a supported bridge command only when it maps back to this action."""
    if action.type is ActionType.END_TURN:
        if _mapping(live.get("battle")).get("can_end_turn") is False:
            return None
        candidate: dict[str, Any] = {"action": "end_turn"}
    elif action.type is ActionType.PLAY_CARD and sim.combat is not None:
        index = next(
            (
                index
                for index, card in enumerate(sim.combat.hand)
                if card.instance_id == action.card_instance_id
            ),
            None,
        )
        if index is None:
            return None
        raw_card = _rows(_mapping(live.get("player")).get("hand"), "hand")[index]
        if raw_card.get("can_play") is not True:
            return None
        candidate = {"action": "play_card", "card_index": index}
        card = sim.combat.hand[index]
        if card.target in {TargetType.ENEMY, TargetType.ANY}:
            if action.target_id is None or action.target_id == "player":
                return None
            targets = raw_card.get("valid_targets")
            if targets is not None and action.target_id not in targets:
                return None
            candidate["target"] = action.target_id
    else:
        return None
    try:
        mapped = map_live_action_to_simulator_action(live, candidate, sim)
    except LiveParityUnsupported:
        return None
    if action.type is ActionType.PLAY_CARD and sim.combat is not None:
        card = next(card for card in sim.combat.hand if card.instance_id == action.card_instance_id)
        if card.target is TargetType.SELF and mapped.target_id is None:
            mapped = mapped.model_copy(update={"target_id": "player"})
    return candidate if mapped == action else None


class LiveMaskedPpoPolicy:
    def __init__(self, policy: MaskedPpoPolicy, *, simulator_seed: int | str = 0) -> None:
        self.policy = policy
        self.simulator_seed = simulator_seed
        self._memory: list[dict[str, Any]] = []

    def choose_action(self, live: Mapping[str, Any]) -> tuple[MaskedPpoDecision, dict[str, Any]]:
        sim = import_policy_combat(live, seed=self.simulator_seed)
        supported: dict[int, dict[str, Any]] = {}
        for descriptor in action_space(sim):
            action = Action.model_validate(descriptor["action"])
            bridge_action = bridge_action_for_policy(live, sim, action)
            if bridge_action is not None:
                supported[descriptor["id"]] = bridge_action
        if not supported:
            raise LivePolicyUnsupported("No simulator action has a verified live bridge mapping.")
        decision = self.policy.choose_action(
            sim,
            allowed_action_ids=tuple(supported),
            agent_memory={"entries": self._memory},
        )
        return decision, supported[decision.action_id]

    def record_transition(
        self,
        before: Mapping[str, Any],
        decision: MaskedPpoDecision,
        after: Mapping[str, Any],
    ) -> None:
        """Remember observed action/outcome fields; live training rewards are unavailable."""
        previous = _mapping(before.get("player"))
        current = _mapping(after.get("player"))
        entry: dict[str, Any] = {
            "age": 0,
            "action_type": decision.descriptor.get("type", ""),
            "action_type_id": decision.descriptor.get("action_type_id", 0),
            "action_id": decision.action_id,
            **decision.policy_output,
            "phase_changed": int(before.get("state_type") != after.get("state_type")),
            "done": int(after.get("state_type") == "game_over"),
        }
        for key in ("hp", "block", "energy", "gold"):
            if key in previous and key in current:
                with suppress(LivePolicyUnsupported):
                    entry[f"{key}_delta"] = _integer(current, key) - _integer(previous, key)
        for key, value in decision.policy_output.items():
            if key in {
                "aggression_target",
                "hp_floor",
                "hp_spend_budget",
                "combat_pace",
                "route_preference",
                "potion_policy",
                "reward_pickiness",
                "expected_hp_loss",
                "expected_turns_to_kill",
                "boss_readiness",
            }:
                entry[f"plan_{key}"] = value
        self._memory = [
            entry,
            *({**item, "age": index + 1} for index, item in enumerate(self._memory)),
        ][:4]
