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
from sts2sim.engine.models import Action, ActionType, OrbState, RunState, TargetType
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


def _import_resources(player: Mapping[str, Any], character: str) -> dict[str, int]:
    raw = player.get("resources", {})
    if not isinstance(raw, Mapping):
        raise LivePolicyUnsupported("Live resources must be an object.")
    resources: dict[str, int] = {}
    for key, value in raw.items():
        normalized = "star" if str(key).lower() in {"star", "stars"} else str(key).lower()
        if normalized not in {"star", "forge"}:
            raise LivePolicyUnsupported(f"Unknown live player resource: {key}.")
        amount = _integer({"amount": value}, "amount")
        if normalized in resources and resources[normalized] != amount:
            raise LivePolicyUnsupported("Conflicting aliases for a live resource.")
        resources[normalized] = amount
    if "stars" in player:
        amount = _integer(player, "stars")
        if "star" in resources and resources["star"] != amount:
            raise LivePolicyUnsupported("Live stars contradict player.resources.")
        resources["star"] = amount
    if character == "REGENT" and "star" not in resources:
        raise LivePolicyUnsupported("Regent combat requires an explicit live stars value.")
    return resources


def _import_orbs(player: Mapping[str, Any], character: str) -> tuple[tuple[OrbState, ...], int]:
    if character == "DEFECT" and ("orbs" not in player or "orb_slots" not in player):
        raise LivePolicyUnsupported("Defect combat requires explicit live orbs and orb_slots.")
    raw = _rows(player.get("orbs", []), "player.orbs")
    slots = _integer(player, "orb_slots") if "orb_slots" in player else 0
    if slots > 10 or len(raw) > slots:
        raise LivePolicyUnsupported("Live orb occupancy exceeds source capacity.")
    orbs = []
    for orb in raw:
        kind = str(orb.get("id", "")).strip().lower().removesuffix("_orb")
        if kind not in {"lightning", "frost", "dark", "plasma", "glass"}:
            raise LivePolicyUnsupported(f"Unknown live orb ID: {kind}.")
        if kind == "dark":
            value = _integer(orb, "evoke_val")
        elif kind == "glass":
            value = _integer(orb, "base_value")
        else:
            value = 0  # Source these orb models have no mutable accumulator.
        orbs.append(OrbState(orb_id=kind, value=value))
    return tuple(orbs), slots


def _import_osty(player: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    if "pets" not in player:
        raise LivePolicyUnsupported("Necrobinder combat requires an explicit live pets list.")
    pets = _rows(player["pets"], "player.pets")
    if len(pets) > 1 or any(str(p.get("id", "")).lower() != "osty" for p in pets):
        raise LivePolicyUnsupported("Unknown live combat pet.")
    if not pets:
        return {
            "created": False,
            "alive": False,
            "hp": 0,
            "max_hp": 0,
            "block": 0,
            "statuses": {},
            "die_for_you": False,
        }
    pet = pets[0]
    hp, maximum, block = (_integer(pet, field) for field in ("hp", "max_hp", "block"))
    alive = pet.get("alive")
    if not isinstance(alive, bool) or alive != (hp > 0) or hp > maximum:
        raise LivePolicyUnsupported("Contradictory live Osty health/alive state.")
    statuses = _statuses(_rows(pet.get("status"), "pet.status"))
    return {
        "created": True,
        "alive": alive,
        "hp": hp,
        "max_hp": maximum,
        "block": block,
        "statuses": statuses,
        "die_for_you": "die_for_you" in statuses,
        "attacks_this_turn": int(metadata.get("osty_attacks_this_turn", 0)),
    }


_OBSERVED_COMBAT_COUNTERS = frozenset(
    {
        "attacks_played_this_turn",
        "skills_played_this_turn",
        "powers_played_this_turn",
        "cards_played_this_combat",
        "source_class_ethereal_plays",
        "osty_attacks_this_turn",
    }
)


def _import_combat_metadata(
    battle: Mapping[str, Any],
    player: Any,
    combat: Any,
) -> dict[str, Any]:
    raw = battle.get("combat_metadata", {})
    if not isinstance(raw, Mapping):
        raise LivePolicyUnsupported("Live combat_metadata must be an object.")
    result = {key: _integer(raw, key) for key in _OBSERVED_COMBAT_COUNTERS if key in raw}
    if (
        any(
            card.card_id == "banshees_cry"
            for zone in (
                combat.hand,
                combat.draw_pile,
                combat.discard_pile,
                combat.exhaust_pile,
            )
            for card in zone
        )
        and "source_class_ethereal_plays" not in result
    ):
        raise LivePolicyUnsupported("Banshee's Cry requires observed Ethereal play history.")
    return result


def _played_card_ids(battle: Mapping[str, Any], combat: Any) -> tuple[str, ...]:
    raw = battle.get("cards_played_this_turn", [])
    if not isinstance(raw, list | tuple) or not all(isinstance(v, str) for v in raw):
        raise LivePolicyUnsupported("Live cards_played_this_turn must contain physical card IDs.")
    aliases = {
        str(card.custom.get("live_instance_id", card.instance_id)): str(card.instance_id)
        for zone in (combat.hand, combat.draw_pile, combat.discard_pile, combat.exhaust_pile)
        for card in zone
    }
    return tuple(aliases.get(str(value), str(value)) for value in raw)


def _mark_observed_local_costs(combat: Any) -> Any:
    zones = {}
    for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        cards = []
        for card in getattr(combat, zone):
            custom = dict(card.custom)
            if "source_local_energy_modifiers" in custom:
                if card.card_id in {"stomp", "pinpoint"}:
                    custom["source_local_count_turn"] = combat.turn
                if card.card_id == "banshees_cry":
                    custom["source_local_banshee_seen"] = int(
                        combat.metadata.get("source_class_ethereal_plays", 0)
                    )
            cards.append(card.model_copy(update={"custom": custom}))
        zones[zone] = tuple(cards)
    return combat.model_copy(update=zones)


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
    total_floor = (
        _integer(run, "total_floor", minimum=1)
        if "total_floor" in run
        else _integer(run, "floor", minimum=1)
    )
    if "act_floor" not in run and act > 1:
        raise LivePolicyUnsupported("Later acts require an explicit within-act floor.")
    floor = _integer(run, "act_floor", minimum=1) if "act_floor" in run else total_floor
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
        if not intents:
            raise LivePolicyUnsupported("PPO mapping requires explicit visible enemy intents.")
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
    resources = _import_resources(player, sim.character_id)
    imported_player = imported_player.model_copy(update={"resources": resources})
    metadata = _import_combat_metadata(battle, state.get("player", {}), combat)
    metadata["item_relic_ids"] = sim.relics
    metadata["source_round_number"] = _integer(battle, "round", minimum=1)
    if "player_turn" not in battle and "paels_eye" in sim.relics:
        raise LivePolicyUnsupported("Pael's Eye requires an explicit observed player_turn.")
    player_turn = (
        _integer(battle, "player_turn", minimum=1)
        if "player_turn" in battle
        else _integer(battle, "round", minimum=1)
    )
    orbs, orb_slots = _import_orbs(player, sim.character_id)
    if sim.character_id == "NECROBINDER" or "pets" in player:
        metadata["osty"] = _import_osty(player, metadata)
    combat = combat.model_copy(
        update={
            "player": imported_player,
            "turn": player_turn,
            "metadata": metadata,
            "orbs": orbs,
            "orb_slots": orb_slots,
            "cards_played_this_turn": _played_card_ids(battle, combat),
        }
    )
    from sts2sim.engine import source_class_hooks

    combat = _mark_observed_local_costs(source_class_hooks.refresh_blades(combat))
    ids = [
        card.instance_id
        for zone in (combat.hand, combat.draw_pile, combat.discard_pile, combat.exhaust_pile)
        for card in zone
    ]
    if len(set(ids)) != len(ids):
        raise LivePolicyUnsupported("A physical live card appears in multiple combat positions.")
    raw_plays = battle.get("card_plays_this_turn")
    if raw_plays is not None:
        plays = _rows(raw_plays, "battle.card_plays_this_turn")
        if any(not isinstance(play.get("is_auto_play"), bool) for play in plays):
            raise LivePolicyUnsupported("Live finished-card history lacks auto-play identity.")
        manual = sum(
            not play["is_auto_play"] and _integer(play, "play_index") == 0 for play in plays
        )
        metadata = {
            **combat.metadata,
            "card_status_turn_counters": {
                "turn": player_turn,
                "manual_cards": manual,
                "paels_eye_manual": manual,
                "brilliant_scarf_manual": manual,
            },
        }
        combat = combat.model_copy(update={"metadata": metadata})
    elif imported_player.statuses.get("voidform") or "brilliant_scarf" in sim.relics:
        raise LivePolicyUnsupported("Current free-card gates require observed manual card history.")
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
        if str(cost).upper() != "X":
            observed_cost = _integer(raw, "cost")
            if "local_cost" in raw:
                from sts2sim.engine.transitions import _energy_cost

                computed_cost = _energy_cost(hand_card, imported_player.energy, combat=combat)
                if computed_cost != observed_cost and (hand_card.cost or 0) >= 0:
                    raise LivePolicyUnsupported(
                        "Observed effective cost differs from imported modifiers."
                    )
        hand.append(hand_card)
    monsters = []
    for monster, raw in zip(combat.monsters, enemies, strict=True):
        kinds: list[str] = []
        attack: tuple[int, int] | None = None
        block = 0
        for intent in _rows(raw["intents"], "enemy.intents"):
            kind = str(intent.get("type", "")).lower()
            kinds.append(kind)
            if "attack" in kind:
                label = str(intent.get("label", ""))
                match = re.fullmatch(r"\s*(\d+)(?:\s*[x×]\s*(\d+))?\s*", label)
                if match is None:
                    raise LivePolicyUnsupported(f"Unmapped live attack intent amount: {label!r}.")
                damage, hits = int(match[1]), int(match[2] or 1)
                if hits < 1:
                    raise LivePolicyUnsupported("Live attack hit count must be positive.")
                if attack is not None:
                    if damage != attack[0]:
                        raise LivePolicyUnsupported(
                            "Distinct simultaneous attack amounts need a hit sequence."
                        )
                    hits += attack[1]
                attack = damage, hits
            elif "defend" in kind:
                label = str(intent.get("label", "")).strip()
                if label.isdecimal():
                    block += int(label)
        damage, hits = attack or (0, 1)
        source_id = str(raw.get("monster_id") or "")
        monsters.append(
            monster.model_copy(
                update={
                    "intent": "_".join(kinds),
                    "intent_damage": damage * hits,
                    "hit_count": hits,
                    "intent_block": block,
                    "metadata": {
                        **monster.metadata,
                        "live_visible_intents": list(raw["intents"]),
                        **({"source_monster_id": source_id.lower()} if source_id else {}),
                    },
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
        if any(not isinstance(card.get("is_upgraded"), bool) for card in deck_rows):
            raise LivePolicyUnsupported("Live master deck must expose its upgrade state.")
        try:
            master_deck = _cards_from_live_zone(
                deck_rows,
                zone="master",
                character_id=sim.character_id,
                card_library=library,
                start_index=1001,
            )
        except LiveParityUnsupported as exc:
            raise LivePolicyUnsupported(str(exc)) from exc

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
    flags = {
        **sim.flags,
        "potion_slots": base_slots,
        "bonus_potion_slots": 0,
        "total_floor": total_floor,
    }
    if "can_remove_potions" in player:
        if not isinstance(player["can_remove_potions"], bool):
            raise LivePolicyUnsupported("Invalid live can_remove_potions flag.")
        flags["can_remove_potions"] = player["can_remove_potions"]
    live_potions = tuple(
        p
        for p in _rows(player.get("potions"), "player.potions")
        if p.get("id") or p.get("potion_id")
    )
    if any("slot" in potion for potion in live_potions):
        if not all("slot" in potion for potion in live_potions):
            raise LivePolicyUnsupported("Live potion inventory mixes indexed and packed slots.")
        indices = tuple(_integer(potion, "slot") for potion in live_potions)
        if len(set(indices)) != len(indices) or any(
            i >= _integer(player, "max_potion_slots") for i in indices
        ):
            raise LivePolicyUnsupported("Invalid live physical potion slots.")
        flags["potion_slot_indices"] = indices
    # Map generation and a starter deck from new_run are not live observations.
    # Only the validated deck survives; unknown map context is represented empty.
    return sim.model_copy(
        update={
            "player": imported_player,
            "combat": combat,
            "master_deck": master_deck,
            "floor": floor,
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
    elif action.type in {ActionType.USE_POTION, ActionType.DISCARD_POTION}:
        player = _mapping(live.get("player"))
        if player.get("can_remove_potions") is False:
            return None
        slot_id = (
            action.target_id
            if action.type is ActionType.DISCARD_POTION
            else _mapping(action.payload).get("potion_slot")
        )
        if not isinstance(slot_id, str) or not re.fullmatch(r"potion:\d+", slot_id):
            return None
        slot = int(slot_id.split(":")[1])
        raw_potion = next(
            (
                potion
                for index, potion in enumerate(_rows(player.get("potions"), "player.potions"))
                if potion.get("slot", index) == slot and potion.get("id")
            ),
            None,
        )
        if raw_potion is None:
            return None
        if action.type is ActionType.USE_POTION:
            if raw_potion.get("can_use_in_combat") is not True:
                return None
            candidate = {"action": "use_potion", "slot": slot}
            if action.target_id not in {None, "player"}:
                candidate["target"] = action.target_id
        else:
            candidate = {"action": "discard_potion", "slot": slot}
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
