"""True step parity between a live STS2MCP state and the simulator.

This module reconstructs a simulator state from a live bridge payload, maps the
same live action into a simulator action, steps the simulator, and compares the
result to the live post-action payload.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sts2sim.api import new_run, step
from sts2sim.engine.models import (
    Action,
    ActionType,
    CardInstance,
    CombatState,
    MonsterState,
    PlayerState,
    RunPhase,
    RunState,
)
from sts2sim.engine.transitions import _card_from_spec
from sts2sim.parity import ParityCompareConfig, ParityMismatch, compare_snapshots

DEFAULT_CARD_CACHE = Path("data/cache/eng/cards.json")
DEFAULT_CHARACTER_CACHE = Path("data/cache/eng/characters.json")


class LiveParityModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class LiveStepParityResult(LiveParityModel):
    supported: bool
    reason: str = ""
    sim_action: dict[str, Any] | None = None
    live_before: dict[str, Any] = Field(default_factory=dict)
    live_after: dict[str, Any] = Field(default_factory=dict)
    simulator_after: dict[str, Any] = Field(default_factory=dict)
    mismatch_count: int = 0
    mismatches: tuple[ParityMismatch, ...] = ()

    @property
    def matched(self) -> bool:
        return self.supported and not self.mismatches


def compare_live_step_to_simulator(
    *,
    before: Mapping[str, Any],
    action: Mapping[str, Any],
    after: Mapping[str, Any],
    seed: int | str = 0,
    card_cache: Path | str = DEFAULT_CARD_CACHE,
    character_cache: Path | str = DEFAULT_CHARACTER_CACHE,
) -> LiveStepParityResult:
    """Reconstruct, step, and compare one live action against the simulator."""

    try:
        sim_before = live_state_to_simulator_state(
            before,
            seed=seed,
            card_cache=card_cache,
            character_cache=character_cache,
        )
        sim_action = map_live_action_to_simulator_action(before, action, sim_before)
    except LiveParityUnsupported as exc:
        return LiveStepParityResult(
            supported=False,
            reason=str(exc),
            live_before=live_snapshot_for_compare(before),
            live_after=live_snapshot_for_compare(after),
        )

    try:
        sim_after = step(sim_before, sim_action)
    except Exception as exc:
        return LiveStepParityResult(
            supported=False,
            reason=f"simulator_step_failed:{exc}",
            sim_action=sim_action.model_dump(mode="json"),
            live_before=live_snapshot_for_compare(before),
            live_after=live_snapshot_for_compare(after),
        )

    live_after_snapshot = live_snapshot_for_compare(after)
    sim_after_snapshot = simulator_snapshot_for_compare(sim_after)
    mismatches = compare_snapshots(
        live_after_snapshot,
        sim_after_snapshot,
        ParityCompareConfig(mode="subset", ignored_paths=("combat.last_events",)),
    )
    return LiveStepParityResult(
        supported=True,
        sim_action=sim_action.model_dump(mode="json"),
        live_before=live_snapshot_for_compare(before),
        live_after=live_after_snapshot,
        simulator_after=sim_after_snapshot,
        mismatch_count=len(mismatches),
        mismatches=mismatches,
    )


def live_state_to_simulator_state(
    state: Mapping[str, Any],
    *,
    seed: int | str = 0,
    card_cache: Path | str = DEFAULT_CARD_CACHE,
    character_cache: Path | str = DEFAULT_CHARACTER_CACHE,
) -> RunState:
    """Build a simulator combat state from a live STS2MCP payload."""

    if _state_type(state) not in {"monster", "elite", "boss"}:
        raise LiveParityUnsupported("true parity currently supports combat states only")

    player_source = _mapping(state.get("player"))
    run_source = _mapping(state.get("run"))
    character_id = _live_character_id(_optional_string(player_source.get("character")))
    if character_id is None:
        raise LiveParityUnsupported("live state does not expose a supported character")

    ascension = _optional_int(run_source.get("ascension")) or 0
    source_data = _runtime_source_data(
        character_id,
        card_cache=Path(card_cache),
        character_cache=Path(character_cache),
    )
    base = new_run(seed, character_id, ascension, source_data=source_data)
    if not isinstance(base, RunState):
        raise LiveParityUnsupported("simulator did not return a RunState")

    card_library = _card_library(Path(card_cache))
    live_player = _player_state_from_live(player_source)
    relics = tuple(
        _source_id_to_runtime_id(item_id)
        for item_id in _item_ids(_sequence(player_source.get("relics")))
    )
    potions = tuple(
        _source_id_to_runtime_id(item_id)
        for item_id in _item_ids(_sequence(player_source.get("potions")))
    )
    combat = CombatState(
        turn=_optional_int(_mapping(state.get("battle")).get("round")) or 1,
        player=live_player,
        monsters=_monsters_from_live(_mapping(state.get("battle"))),
        hand=_cards_from_live_zone(
            _sequence(player_source.get("hand")),
            zone="hand",
            character_id=character_id,
            card_library=card_library,
            start_index=1,
        ),
        draw_pile=_cards_from_live_zone(
            _sequence(player_source.get("draw_pile")),
            zone="draw",
            character_id=character_id,
            card_library=card_library,
            start_index=101,
        ),
        discard_pile=_cards_from_live_zone(
            _sequence(player_source.get("discard_pile")),
            zone="discard",
            character_id=character_id,
            card_library=card_library,
            start_index=201,
        ),
        exhaust_pile=_cards_from_live_zone(
            _sequence(player_source.get("exhaust_pile")),
            zone="exhaust",
            character_id=character_id,
            card_library=card_library,
            start_index=301,
        ),
        draw_per_turn=5,
    )
    return base.model_copy(
        update={
            "phase": RunPhase.COMBAT,
            "act": _optional_int(run_source.get("act")) or 1,
            "floor": _optional_int(run_source.get("floor")) or 0,
            "player": live_player,
            "relics": relics,
            "potions": potions,
            "ancient": None,
            "event": None,
            "reward": None,
            "shop": None,
            "combat": combat,
            "room_history": (),
            "replay_log": (),
            "flags": {**base.flags, **source_data},
        }
    )


def map_live_action_to_simulator_action(
    state: Mapping[str, Any],
    action: Mapping[str, Any],
    sim_state: RunState,
) -> Action:
    """Map an STS2MCP action payload to a simulator Action."""

    action_name = _normalized_id(action.get("action"))
    if sim_state.combat is None:
        raise LiveParityUnsupported("simulator state has no combat")

    if action_name == "play_card":
        card_index = _optional_int(action.get("card_index"))
        if card_index is None:
            raise LiveParityUnsupported("play_card action is missing card_index")
        if card_index < 0 or card_index >= len(sim_state.combat.hand):
            raise LiveParityUnsupported("play_card card_index is outside simulator hand")
        card = sim_state.combat.hand[card_index]
        return Action(
            type=ActionType.PLAY_CARD,
            card_instance_id=card.instance_id,
            target_id=_optional_string(action.get("target")),
        )

    if action_name == "end_turn":
        return Action(type=ActionType.END_TURN)

    if action_name == "discard_potion":
        slot = _optional_int(action.get("slot"))
        if slot is None:
            raise LiveParityUnsupported("discard_potion action is missing slot")
        return Action(type=ActionType.DISCARD_POTION, target_id=f"potion:{slot}")

    if action_name == "use_potion":
        slot = _optional_int(action.get("slot"))
        if slot is None:
            raise LiveParityUnsupported("use_potion action is missing slot")
        potion_id = sim_state.potions[slot] if 0 <= slot < len(sim_state.potions) else ""
        return Action(
            type=ActionType.USE_POTION,
            target_id=_optional_string(action.get("target")) or "player",
            payload={"potion_slot": f"potion:{slot}", "potion_id": potion_id},
        )

    raise LiveParityUnsupported(f"unsupported live action for true parity:{action_name}")


def live_snapshot_for_compare(state: Mapping[str, Any]) -> dict[str, Any]:
    player = _mapping(state.get("player"))
    run = _mapping(state.get("run"))
    phase = "combat" if _state_type(state) in {"monster", "elite", "boss"} else _state_type(state)
    snapshot: dict[str, Any] = {
        "phase": phase,
        "act": _optional_int(run.get("act")),
        "floor": _optional_int(run.get("floor")),
        "ascension": _optional_int(run.get("ascension")),
        "player": _live_player_snapshot(player),
        "relics": [
            _source_id_to_runtime_id(item_id)
            for item_id in _item_ids(_sequence(player.get("relics")))
        ],
        "potions": [
            _source_id_to_runtime_id(item_id)
            for item_id in _item_ids(_sequence(player.get("potions")))
        ],
    }
    if _state_type(state) in {"monster", "elite", "boss"}:
        snapshot["combat"] = {
            "turn": _optional_int(_mapping(state.get("battle")).get("round")) or 1,
            "player": _live_player_snapshot(player),
            "monsters": [
                _live_monster_snapshot(enemy)
                for enemy in _sequence(_mapping(state.get("battle")).get("enemies"))
                if isinstance(enemy, Mapping)
            ],
            "hand_count": len(_sequence(player.get("hand"))),
            "draw_pile_count": _optional_int(player.get("draw_pile_count")),
            "discard_pile_count": _optional_int(player.get("discard_pile_count")),
            "exhaust_pile_count": _optional_int(player.get("exhaust_pile_count")),
        }
    return _without_none(snapshot)


def simulator_snapshot_for_compare(state: Any) -> dict[str, Any]:
    sim_state = state if isinstance(state, RunState) else RunState.model_validate(state)
    snapshot: dict[str, Any] = {
        "phase": sim_state.phase.value,
        "act": sim_state.act,
        "floor": sim_state.floor,
        "ascension": sim_state.ascension,
        "player": _sim_player_snapshot(sim_state.player),
        "relics": list(sim_state.relics),
        "potions": list(sim_state.potions),
    }
    if sim_state.combat is not None:
        combat = sim_state.combat
        snapshot["combat"] = {
            "turn": combat.turn,
            "player": _sim_player_snapshot(combat.player),
            "monsters": [
                {
                    "monster_id": monster.monster_id,
                    "name": monster.name,
                    "hp": monster.hp,
                    "max_hp": monster.max_hp,
                    "block": monster.block,
                    "statuses": dict(monster.statuses),
                }
                for monster in combat.monsters
                if monster.hp > 0
            ],
            "hand_count": len(combat.hand),
            "hand": [_sim_card_snapshot(card) for card in combat.hand],
            "draw_pile_count": len(combat.draw_pile),
            "discard_pile_count": len(combat.discard_pile),
            "exhaust_pile_count": len(combat.exhaust_pile),
        }
    return snapshot


class LiveParityUnsupported(RuntimeError):
    """Raised when a live state/action is not reconstructable yet."""


def _runtime_source_data(
    character_id: str,
    *,
    card_cache: Path,
    character_cache: Path,
) -> dict[str, Any]:
    card_rows = _json_rows(card_cache)
    cards_by_id = {str(row.get("id", "")).upper(): row for row in card_rows if row.get("id")}
    characters: dict[str, dict[str, Any]] = {}
    for row in _json_rows(character_cache):
        runtime_id = _normalized_character_id(row.get("id", ""))
        if not runtime_id:
            continue
        starter_deck = tuple(
            _runtime_card_spec(raw_card_id, cards_by_id)
            for raw_card_id in _string_sequence(row.get("starting_deck"))
        )
        characters[runtime_id] = {
            "starter_deck": starter_deck,
            "player": {
                "hp": _int(row.get("starting_hp"), 80),
                "max_hp": _int(row.get("starting_hp"), 80),
                "energy": _int(row.get("max_energy"), 3),
                "max_energy": _int(row.get("max_energy"), 3),
                "gold": _int(row.get("starting_gold"), 99),
            },
            "starting_relics": tuple(
                _source_id_to_runtime_id(relic_id)
                for relic_id in _string_sequence(row.get("starting_relics"))
            ),
        }
    return {
        "characters": {character_id: characters.get(character_id, {})},
        "cards": tuple(card_rows),
        "flags": {
            "max_acts": 1,
            "draw_per_turn": 5,
            "combat_reward_potion_chance_percent": 0,
            "combat_reward_card_count": 0,
            "combat_reward_relic_count": 0,
        },
    }


def _runtime_card_spec(
    source_card_id: str,
    cards_by_id: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    runtime_id = _source_card_id_to_runtime_id(source_card_id, tuple(cards_by_id))
    row = cards_by_id.get(runtime_id.upper())
    if row is None:
        return {"card_id": runtime_id}
    spec = dict(row)
    spec["card_id"] = str(row.get("id", runtime_id))
    return spec


def _cards_from_live_zone(
    cards: Sequence[Any],
    *,
    zone: str,
    character_id: str,
    card_library: Mapping[str, Mapping[str, Any]],
    start_index: int,
) -> tuple[CardInstance, ...]:
    result: list[CardInstance] = []
    for offset, item in enumerate(cards):
        if not isinstance(item, Mapping):
            continue
        result.append(
            _live_card_to_sim_card(
                item,
                zone=zone,
                character_id=character_id,
                instance_counter=start_index + offset,
                card_library=card_library,
            )
        )
    return tuple(result)


def _live_card_to_sim_card(
    card: Mapping[str, Any],
    *,
    zone: str,
    character_id: str,
    instance_counter: int,
    card_library: Mapping[str, Mapping[str, Any]],
) -> CardInstance:
    card_id = _live_card_id(card, character_id, card_library)
    source = dict(card_library.get(card_id, {"id": card_id, "card_id": card_id}))
    source["card_id"] = str(source.get("id", card_id))
    source["instance_id"] = f"live_{zone}_{instance_counter:03d}"
    if card.get("is_upgraded") is not None:
        source["upgraded"] = bool(card.get("is_upgraded"))
    return _card_from_spec(source, instance_counter, card_library=card_library)


def _live_card_id(
    card: Mapping[str, Any],
    character_id: str,
    card_library: Mapping[str, Mapping[str, Any]],
) -> str:
    raw_id = _optional_string(card.get("id"))
    if raw_id:
        runtime_id = _source_card_id_to_runtime_id(raw_id, tuple(card_library))
        if runtime_id in card_library:
            return runtime_id
    name = (_optional_string(card.get("name")) or "").strip().lower()
    if name == "strike":
        candidate = f"STRIKE_{character_id}"
        if candidate in card_library:
            return candidate
    if name == "defend":
        candidate = f"DEFEND_{character_id}"
        if candidate in card_library:
            return candidate
    for key, row in card_library.items():
        if str(row.get("name", "")).strip().lower() == name:
            return key
    return _source_card_id_to_runtime_id(raw_id or name or "unknown_card", tuple(card_library))


def _monsters_from_live(battle: Mapping[str, Any]) -> tuple[MonsterState, ...]:
    monsters: list[MonsterState] = []
    for enemy in _sequence(battle.get("enemies")):
        if not isinstance(enemy, Mapping):
            continue
        enemy_map = _mapping(enemy)
        monster_id = _optional_string(enemy_map.get("entity_id")) or f"enemy_{len(monsters)}"
        intent = _first_intent(enemy_map)
        max_hp = _optional_int(enemy_map.get("max_hp")) or _optional_int(enemy_map.get("hp")) or 1
        monsters.append(
            MonsterState(
                monster_id=monster_id,
                name=_optional_string(enemy_map.get("name")) or monster_id,
                hp=_optional_int(enemy_map.get("hp")) or 0,
                max_hp=max_hp,
                block=_optional_int(enemy_map.get("block")) or 0,
                intent=intent.get("type"),
                intent_damage=_optional_int(intent.get("label")) or 0,
                statuses=_status_map(_sequence(enemy_map.get("status"))),
            )
        )
    return tuple(monsters)


def _player_state_from_live(player: Mapping[str, Any]) -> PlayerState:
    max_hp = _optional_int(player.get("max_hp")) or 80
    max_energy = _optional_int(player.get("max_energy")) or 3
    return PlayerState(
        hp=_optional_int(player.get("hp")) or max_hp,
        max_hp=max_hp,
        block=_optional_int(player.get("block")) or 0,
        energy=_optional_int(player.get("energy")) or max_energy,
        max_energy=max_energy,
        gold=_optional_int(player.get("gold")) or 0,
        statuses=_status_map(_sequence(player.get("status"))),
    )


def _live_player_snapshot(player: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "hp": _optional_int(player.get("hp")),
        "max_hp": _optional_int(player.get("max_hp")),
        "block": _optional_int(player.get("block")),
        "energy": _optional_int(player.get("energy")),
        "max_energy": _optional_int(player.get("max_energy")),
        "gold": _optional_int(player.get("gold")),
        "statuses": _status_map(_sequence(player.get("status"))),
    }
    return _without_none(result)


def _sim_player_snapshot(player: PlayerState) -> dict[str, Any]:
    return {
        "hp": player.hp,
        "max_hp": player.max_hp,
        "block": player.block,
        "energy": player.energy,
        "max_energy": player.max_energy,
        "gold": player.gold,
        "statuses": dict(player.statuses),
    }


def _live_monster_snapshot(enemy: Mapping[str, Any]) -> dict[str, Any]:
    enemy_map = _mapping(enemy)
    return _without_none(
        {
            "monster_id": _optional_string(enemy_map.get("entity_id")),
            "name": _optional_string(enemy_map.get("name")),
            "hp": _optional_int(enemy_map.get("hp")),
            "max_hp": _optional_int(enemy_map.get("max_hp")),
            "block": _optional_int(enemy_map.get("block")),
            "statuses": _status_map(_sequence(enemy_map.get("status"))),
        }
    )


def _live_card_zone_snapshot(cards: Sequence[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in cards:
        if not isinstance(item, Mapping):
            continue
        item_map = _mapping(item)
        result.append(
            _without_none(
                {
                    "card_id": _optional_string(item_map.get("id")),
                    "name": _optional_string(item_map.get("name")),
                    "cost": _optional_int(item_map.get("cost")),
                    "type": _normalized_id(item_map.get("type")),
                }
            )
        )
    return result


def _sim_card_snapshot(card: CardInstance) -> dict[str, Any]:
    return {
        "card_id": card.card_id,
        "name": card.name,
        "cost": card.cost,
        "type": card.type.value,
    }


def _first_intent(enemy: Mapping[str, Any]) -> dict[str, Any]:
    first = _first_mapping_item(_sequence(enemy.get("intents")))
    return first or {}


def _status_map(statuses: Sequence[Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in statuses:
        if not isinstance(item, Mapping):
            continue
        status_id = _status_id(item)
        if not status_id:
            continue
        result[status_id] = _optional_int(item.get("amount")) or 1
    return result


def _status_id(item: Mapping[str, Any]) -> str:
    raw = _optional_string(item.get("id")) or _optional_string(item.get("name")) or ""
    normalized = _normalized_id(raw)
    if normalized.endswith("_power"):
        normalized = normalized.removesuffix("_power")
    return normalized


def _card_library(card_cache: Path) -> dict[str, Mapping[str, Any]]:
    library: dict[str, Mapping[str, Any]] = {}
    for row in _json_rows(card_cache):
        card_id = str(row.get("id", row.get("card_id", ""))).strip()
        if not card_id:
            continue
        normalized = _source_card_id_to_runtime_id(card_id, ())
        library[card_id.upper()] = row
        library[normalized] = row
        library[_normalized_id(card_id)] = row
    return library


def _json_rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        return ()
    return tuple(row for row in payload if isinstance(row, Mapping))


def _source_card_id_to_runtime_id(source_card_id: str, known_card_ids: Sequence[str]) -> str:
    raw = str(source_card_id).strip()
    known = {str(card_id).upper() for card_id in known_card_ids}
    candidates = (
        raw,
        raw.upper(),
        _camel_to_snake(raw).upper(),
        _normalized_id(raw).upper(),
    )
    for candidate in candidates:
        if not known or candidate.upper() in known:
            return candidate.upper()
    return candidates[2] or raw


def _item_ids(items: Sequence[Any]) -> tuple[str, ...]:
    result: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            value = item.get("id") or item.get("relic_id") or item.get("potion_id")
            if value:
                result.append(str(value))
        elif item:
            result.append(str(item))
    return tuple(result)


def _source_id_to_runtime_id(value: object) -> str:
    return _camel_to_snake(str(value)).lower()


def _live_character_id(character: str | None) -> str | None:
    if character is None:
        return None
    normalized = character.strip().lower().replace("the ", "").replace(" ", "_")
    aliases = {
        "ironclad": "IRONCLAD",
        "silent": "SILENT",
        "defect": "DEFECT",
        "watcher": "WATCHER",
        "necrobinder": "NECROBINDER",
        "regent": "REGENT",
    }
    if normalized.upper() in aliases.values():
        return normalized.upper()
    return aliases.get(normalized)


def _normalized_character_id(value: object) -> str:
    return str(value or "").strip().upper()


def _camel_to_snake(value: str) -> str:
    stripped = str(value or "").strip().replace("-", "_").replace(" ", "_")
    first_pass = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", stripped)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass)


def _normalized_id(value: object) -> str:
    return str(value or "").strip().lower().replace("'", "").replace(" ", "_").replace("-", "_")


def _state_type(state: Mapping[str, Any]) -> str:
    return _normalized_id(state.get("state_type"))


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _sequence(value: object) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _first_mapping_item(values: Sequence[Any]) -> dict[str, Any] | None:
    for value in values:
        if isinstance(value, Mapping):
            return {str(key): item for key, item in value.items()}
    return None


def _optional_int(value: object) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        if isinstance(value, int):
            return value
        if isinstance(value, float | str | bytes | bytearray):
            return int(value)
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _int(value: object, default: int = 0) -> int:
    parsed = _optional_int(value)
    return default if parsed is None else parsed


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _string_sequence(value: object) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value if str(item).strip())
    if value in (None, ""):
        return ()
    return (str(value),)


def _without_none(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


__all__ = [
    "LiveParityUnsupported",
    "LiveStepParityResult",
    "compare_live_step_to_simulator",
    "live_snapshot_for_compare",
    "live_state_to_simulator_state",
    "map_live_action_to_simulator_action",
    "simulator_snapshot_for_compare",
]
