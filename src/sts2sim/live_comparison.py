"""Explicit installed-game comparison cases with visible and oracle evidence.

A case supplies a full simulator checkpoint and records a separately observed
bridge before/action/after. No sparse bridge payload is treated as a lossless
checkpoint, and a matching case never certifies the entire game.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sts2sim.api import load_state, serialize, step
from sts2sim.engine import source_cost_hooks, source_modifier_hooks
from sts2sim.engine.models import Action, CardInstance, RunState
from sts2sim.engine.rng import STS2_STABLE_DERIVATION, stable_seed_hash
from sts2sim.game_installation import STABLE_BUILD
from sts2sim.live_capture import (
    DEFAULT_BASE_URL,
    LiveApiClient,
    LiveApiConfig,
    _response_is_error,
    _wait_for_live_state_update,
)
from sts2sim.live_parity import live_snapshot_for_compare, simulator_snapshot_for_compare
from sts2sim.mechanics.potion_inventory import occupied_potion_slots
from sts2sim.mechanics.run_progress import source_total_floor
from sts2sim.parity import ParityCompareConfig, compare_snapshots

RUN_STREAMS = (
    "up_front", "shuffle", "unknown_map_point", "combat_card_generation",
    "combat_potion_generation", "combat_card_selection", "combat_energy_costs",
    "combat_targets", "monster_ai", "niche", "combat_orbs", "treasure_room_relics",
)
PLAYER_STREAMS = ("rewards", "shops", "transformations")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComparisonField(_Model):
    """Explicit additional source-field binding; paths are parsed, never evaluated."""
    live_path: str
    simulator_path: str
    numeric: bool = False


class LiveComparisonCase(_Model):
    schema_version: Literal[1] = 1
    case_id: str = "comparison"
    origin: Literal["installed_game", "fixture"] = "fixture"
    game_version: str = "0.107.1"
    installation: dict[str, Any] = Field(default_factory=dict)
    simulator_before: dict[str, Any]
    simulator_action: dict[str, Any]
    external_action: dict[str, Any]
    live_before: dict[str, Any]
    live_after: dict[str, Any]
    compare_rng: bool = True
    fields: tuple[ComparisonField, ...] = ()
    phase_map: dict[str, str] = Field(default_factory=dict)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _name(value: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value).lower()


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be an explicit nonnegative integer")
    return value


def _native_rng(payload: Mapping[str, Any]) -> dict[str, Any]:
    validation = _mapping(payload.get("validation"))
    if validation.get("schema_version") != 1:
        raise ValueError("Bridge validation schema1 is required for RNG comparisons")
    run = _mapping(validation.get("run_rng"))
    if not isinstance(run.get("seed"), str):
        raise ValueError("Native run RNG must expose its displayed seed")
    result: dict[str, Any] = {"displayed_seed": run["seed"]}
    run_counters = {_name(str(k)): v for k, v in _mapping(run.get("counters")).items()}
    for name in RUN_STREAMS:
        result[name] = _integer(run_counters.get(name), f"Native run RNG {name}")
    players = validation.get("players_rng")
    if not isinstance(players, list) or len(players) != 1 or _mapping(players[0]).get("slot") != 0:
        raise ValueError("Comparison supports one explicitly identified player in slot0")
    player = _mapping(_mapping(players[0]).get("rng"))
    result["player_seed"] = _integer(player.get("seed"), "Native player RNG seed")
    counters = {_name(str(k)): v for k, v in _mapping(player.get("counters")).items()}
    for name in PLAYER_STREAMS:
        result[name] = _integer(counters.get(name), f"Native player RNG {name}")
    events = _mapping(validation.get("event_rngs"))
    if not events and isinstance(validation.get("event_rng"), Mapping):
        event = _mapping(validation["event_rng"])
        events = {str(event.get("id", "")): event}
    result["events"] = {}
    for identifier, row in events.items():
        if not identifier:
            raise ValueError("Event RNG is missing a model identity")
        event = _mapping(row)
        result["events"][identifier] = {
            "seed": _integer(event.get("seed"), f"Native {identifier} RNG seed"),
            "counter": _integer(event.get("counter"), f"Native {identifier} RNG counter"),
        }
    if validation.get("event_rng_error"):
        raise ValueError("Bridge reported unavailable event RNG evidence")
    return result


def _simulator_rng(state: RunState, native: Mapping[str, Any]) -> dict[str, Any]:
    if state.rng.derivation != STS2_STABLE_DERIVATION or state.rng.seed is None:
        raise ValueError("RNG comparison requires a stable-source simulator checkpoint")
    displayed = str(state.rng.seed)
    result: dict[str, Any] = {
        "displayed_seed": displayed, "player_seed": stable_seed_hash(displayed),
    }
    for name in (*RUN_STREAMS, *PLAYER_STREAMS):
        stream = state.rng.streams.get(name)
        result[name] = stream.draws if stream is not None else 0
    events: dict[str, Any] = {}
    for identifier in _mapping(native.get("events")):
        stream = state.rng.streams.get(identifier)
        events[identifier] = {
            "seed": (stable_seed_hash(displayed) + stable_seed_hash(identifier)) & 0xFFFFFFFF,
            "counter": stream.draws if stream is not None else 0,
        }
    result["events"] = events
    return result


def _path(payload: Any, path: str) -> Any:
    parts = re.findall(r"([^\.\[\]]+)|\[(\d+)\]", path)
    rebuilt = "".join(("." if i and key else "") + (key or f"[{index}]")
                      for i, (key, index) in enumerate(parts))
    if not parts or rebuilt != path:
        raise ValueError(f"Invalid comparison path: {path!r}")
    value = payload
    for key, index in parts:
        if key:
            if not isinstance(value, Mapping) or key not in value:
                raise ValueError(f"Missing comparison field: {path}")
            value = value[key]
        else:
            if not isinstance(value, list | tuple) or int(index) >= len(value):
                raise ValueError(f"Missing comparison field: {path}")
            value = value[int(index)]
    return value


def _bound_values(
    live: Mapping[str, Any], simulator: Mapping[str, Any], binding: ComparisonField,
) -> tuple[Any, Any]:
    expected = _path(live, binding.live_path)
    actual = _path(simulator, binding.simulator_path)
    if binding.numeric:
        try:
            if isinstance(expected, bool) or isinstance(actual, bool):
                raise ValueError
            expected, actual = str(Fraction(str(expected))), str(Fraction(str(actual)))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("Numeric field binding requires finite numeric values") from exc
    return expected, actual


def _modifier(value: object) -> dict[str, Any] | None:
    raw = _mapping(value)
    if not raw:
        return None
    return {"id": _name(str(raw.get("id", ""))), "amount": raw.get("amount")}


def _live_card(card: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "id": _name(str(card.get("id", ""))),
        "upgrade_level": card.get("upgrade_level"),
        "enchantment": _modifier(card.get("enchantment")),
        "affliction": _modifier(card.get("affliction")),
    }
    if "local_cost" in card:
        result["local_cost"] = card["local_cost"]
    if card.get("fake_upgrade_level") is not None:
        result["fake_upgrade_level"] = card["fake_upgrade_level"]
    return result


def _sim_card(card: CardInstance) -> dict[str, Any]:
    identity = _name(card.card_id)
    enchantments = [{"id": _name(item.keyword), "amount": item.amount}
                    for item in card.enchantments]
    affliction = source_modifier_hooks.affliction_id(card)
    raw = _mapping(card.custom.get("source_core_affliction", card.custom.get("affliction")))
    result = {
        "id": identity,
        "upgrade_level": 0 if identity == "wither" else int(card.upgraded),
        "enchantment": enchantments[0] if len(enchantments) == 1 else
                       enchantments if enchantments else None,
        "affliction": ({"id": affliction, "amount": raw.get("amount", 1)}
                       if affliction else None),
        "local_cost": source_cost_hooks.local_cost(card),
    }
    if identity == "wither":
        result["fake_upgrade_level"] = card.custom.get("wither_upgrade_count", 0)
    return result


def _live_projection(
    raw: Mapping[str, Any], phase_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    result = live_snapshot_for_compare(raw)
    screen = str(raw.get("state_type", ""))
    aliases = {"rest_site": "rest", "card_reward": "reward", "rewards": "reward",
               "relic_select": "reward", "crystal_sphere": "event", "fake_merchant": "event",
               "hand_select": "combat", "abandon_run_confirmation": "event"}
    if screen in {"card_select", "bundle_select"}:
        aliases[screen] = "combat" if _mapping(raw.get("run")).get("in_combat") else "event"
    if screen == "event" and _mapping(raw.get("event")).get("is_ancient"):
        aliases[screen] = "ancient"
    if screen == "game_over" and isinstance(_mapping(raw.get("game_over")).get("won"), bool):
        aliases[screen] = "complete" if _mapping(raw["game_over"])["won"] else "failed"
    result["phase"] = (phase_map or {}).get(screen, aliases.get(screen, result.get("phase")))
    if screen == "abandon_run_confirmation":
        popup = _mapping(raw.get(screen))
        result["event_confirmation"] = {
            "kind": "abandon_run", "event_id": _name(str(popup.get("event_id", ""))),
            "page_id": "ABANDON_CONFIRMATION",
            "can_confirm": popup.get("can_confirm"), "can_cancel": popup.get("can_cancel"),
            "enabled_options": [f"{choice.upper()}_ABANDON" for choice in ("confirm", "cancel")
                                if popup.get(f"can_{choice}") is True],
        }
    battle = _mapping(raw.get("battle"))
    if isinstance(result.get("combat"), dict):
        if "player_turn" in battle:
            result["combat"]["turn"] = battle["player_turn"]
        if "round" in battle:
            result["combat"]["round"] = battle["round"]
    run = _mapping(raw.get("run"))
    if "act_floor" in run:
        result["floor"] = run["act_floor"]
    if "total_floor" in run:
        result["total_floor"] = run["total_floor"]
    player = _mapping(raw.get("player"))
    potion_slots = _live_potion_slots(player)
    if potion_slots is not None:
        result["potion_inventory"] = potion_slots
    cards: dict[str, Any] = {}
    for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile", "master_deck"):
        if zone in player:
            rows = [_live_card(_mapping(card)) for card in player[zone]]
            # The public draw-pile view is sorted; native order has a separate oracle field.
            cards[zone] = (sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))
                           if zone == "draw_pile" else rows)
    if cards:
        result["card_inventory"] = cards
    validation = _mapping(raw.get("validation"))
    if "draw_order" in validation:
        result["native_draw_order"] = [_live_card(_mapping(card))
                                       for card in validation["draw_order"]]
    if "stars" in player:
        result["stars"] = player["stars"]
    if "orbs" in player:
        orbs = []
        for raw_orb in player["orbs"]:
            orb = _mapping(raw_orb)
            identifier = _name(str(orb.get("id", ""))).removesuffix("_orb")
            value = (orb.get("evoke_val") if identifier == "dark" else
                     orb.get("base_value") if identifier == "glass" else 0)
            orbs.append({"id": identifier, "value": value})
        result["orbs"] = orbs
        result["orb_slots"] = player.get("orb_slots")
    pets = player.get("pets")
    if isinstance(pets, list):
        osty = next((_mapping(pet) for pet in pets
                     if _name(str(_mapping(pet).get("id", ""))) == "osty"), {})
        result["osty"] = {key: osty.get(key, 0) for key in ("hp", "max_hp", "block")}
    return result


def _sim_projection(state: RunState) -> dict[str, Any]:
    result = simulator_snapshot_for_compare(state)
    result["potion_inventory"] = [
        {"slot": slot, "id": _name(identifier)}
        for slot, identifier in occupied_potion_slots(state.potions, state.flags)
    ]
    result["total_floor"] = source_total_floor(state)
    if state.event is not None and state.event.page_id == "ABANDON_CONFIRMATION":
        result["event_confirmation"] = {
            "kind": "abandon_run", "event_id": _name(state.event.event_id),
            "page_id": state.event.page_id,
            "can_confirm": any(option.option_id == "CONFIRM_ABANDON" and not option.disabled
                               for option in state.event.options),
            "can_cancel": any(option.option_id == "CANCEL_ABANDON" and not option.disabled
                              for option in state.event.options),
            "enabled_options": [option.option_id for option in state.event.options
                                if not option.disabled],
        }
    inventories: dict[str, Any] = {"master_deck": list(state.master_deck)}
    if state.combat is not None:
        inventories.update({zone: list(getattr(state.combat, zone)) for zone in (
            "hand", "draw_pile", "discard_pile", "exhaust_pile")})
        result["combat"]["round"] = state.combat.metadata.get(
            "source_round_number", state.combat.turn)
        result["native_draw_order"] = [_sim_card(card) for card in state.combat.draw_pile]
        result["orbs"] = [{"id": orb.orb_id, "value": orb.value} for orb in state.combat.orbs]
        result["orb_slots"] = state.combat.orb_slots
        result["stars"] = state.combat.player.resources.get(
            "star", state.combat.player.resources.get("stars", 0))
        osty = _mapping(state.combat.metadata.get("osty"))
        result["osty"] = {key: osty.get(key, 0) for key in ("hp", "max_hp", "block")}
    result["card_inventory"] = {}
    for zone, cards in inventories.items():
        rows = [_sim_card(card) for card in cards]
        result["card_inventory"][zone] = (
            sorted(rows, key=lambda row: json.dumps(row, sort_keys=True))
            if zone == "draw_pile" else rows
        )
    return result


def live_validation_snapshot(
    raw: Mapping[str, Any], *, phase_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Normalize the declared schema2 observed fields, including oracle draw order."""
    return _live_projection(raw, phase_map)


def simulator_validation_snapshot(state: RunState) -> dict[str, Any]:
    """Project a real simulator checkpoint into the same comparison field set."""
    return _sim_projection(state)


def _settled(raw: Mapping[str, Any]) -> bool:
    screen = str(raw.get("state_type", ""))
    if screen == "abandon_run_confirmation":
        popup = _mapping(raw.get(screen))
        return any(popup.get(f"can_{choice}") is True for choice in ("confirm", "cancel"))
    if screen in {"card_select", "hand_select", "bundle_select", "relic_select",
                  "card_reward", "rewards", "crystal_sphere"}:
        return True  # A source action may legitimately await a visible mandatory choice.
    run = _mapping(raw.get("run"))
    if run.get("action_running") is True:
        return False
    if screen in {"monster", "elite", "boss"}:
        battle = _mapping(raw.get("battle"))
        return (battle.get("turn") == "player" and battle.get("is_play_phase") is True
                and battle.get("is_executing_effect") is not True
                and bool(battle.get("enemies")) and _mapping(raw.get("player")).get("hp", 0) > 0)
    return True


def _settled_after(
    client: LiveApiClient, before: Mapping[str, Any], timeout_seconds: float,
) -> tuple[dict[str, Any], bool]:
    deadline = time.monotonic() + max(0.1, timeout_seconds)
    latest = dict(before)
    while time.monotonic() < deadline:
        latest = _wait_for_live_state_update(
            client, before, timeout_seconds=min(3, max(.1, deadline - time.monotonic())))
        if _settled(latest):
            return latest, True
    return latest, False


def _evidence_errors(case: LiveComparisonCase) -> list[str]:
    errors = []
    if case.game_version != STABLE_BUILD["version"]:
        errors.append("Case game version differs from selected stable target")
    if case.origin == "installed_game":
        installation = case.installation
        if installation.get("matches_target_metadata") is not True:
            errors.append("Installed-game case requires a successful installation fingerprint")
        if installation.get("target") != STABLE_BUILD:
            errors.append("Installation fingerprint target differs from selected stable build")
    for label, raw in (("before", case.live_before), ("after", case.live_after)):
        if raw.get("observation_schema_version") != 2:
            errors.append(f"{label}: enriched bridge observation schema2 is required")
        if not raw.get("state_type") or not isinstance(raw.get("run"), Mapping):
            errors.append(f"{label}: explicit screen and run observations are required")
        if raw.get("state_type") == "abandon_run_confirmation":
            popup = _mapping(raw.get("abandon_run_confirmation"))
            if any(not isinstance(popup.get(f"can_{choice}"), bool)
                   for choice in ("confirm", "cancel")):
                errors.append(f"{label}: explicit abandon confirmation controls are required")
        player = _mapping(raw.get("player"))
        for key in ("hp", "max_hp", "block", "gold", "status", "relics", "potions"):
            if key not in player:
                errors.append(f"{label}: missing player.{key}")
        if _live_potion_slots(player) is None:
            errors.append(f"{label}: explicit distinct physical potion slots are required")
    return errors


def _live_potion_slots(player: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    raw = player.get("potions")
    if not isinstance(raw, list) or not all(isinstance(row, Mapping) for row in raw):
        return None
    occupied = [row for row in raw if row.get("id")]
    slots = [row.get("slot") for row in occupied]
    if (any(type(slot) is not int or slot < 0 for slot in slots)
            or len(set(slots)) != len(slots)):
        return None
    return sorted(
        [{"slot": row["slot"], "id": _name(str(row["id"]))} for row in occupied],
        key=lambda row: row["slot"],
    )


def compare_live_comparison(case: LiveComparisonCase) -> dict[str, Any]:
    """Compare known projections before and after one explicit simulator action."""
    errors = _evidence_errors(case)
    mismatches: list[dict[str, Any]] = []
    state = load_state(case.simulator_before)
    config = ParityCompareConfig(mode="subset", ignored_paths=())
    try:
        after = step(state, Action.model_validate(case.simulator_action))
    except (ValueError, RuntimeError) as exc:
        errors.append(f"Simulator action failed: {exc}")
        after = None
    for label, raw, sim in (("before", case.live_before, state),
                            ("after", case.live_after, after)):
        if sim is None:
            continue
        live = _live_projection(raw, case.phase_map)
        observed = _sim_projection(sim)
        for mismatch in compare_snapshots(live, observed, config):
            mismatches.append({"stage": label, **mismatch.model_dump(mode="json")})
        if case.compare_rng:
            try:
                native = _native_rng(raw)
                for mismatch in compare_snapshots(native, _simulator_rng(sim, native), config):
                    mismatches.append({"stage": label, "scope": "rng", **mismatch.model_dump(
                        mode="json")})
            except ValueError as exc:
                errors.append(f"{label}: {exc}")
        for binding in case.fields:
            try:
                expected, actual = _bound_values(raw, serialize(sim), binding)
                for mismatch in compare_snapshots({"value": expected}, {"value": actual}, config):
                    mismatches.append({"stage": label, "live_path": binding.live_path,
                                       "simulator_path": binding.simulator_path,
                                       **mismatch.model_dump(mode="json")})
            except ValueError as exc:
                errors.append(f"{label}: {exc}")
    return {
        "case_id": case.case_id, "origin": case.origin,
        "supported": not errors, "matched": not errors and not mismatches,
        "mismatch_count": len(mismatches), "mismatches": mismatches,
        "unsupported_reasons": errors,
        "rng_compared": case.compare_rng,
        "card_state_scope": "IDs, true/fake upgrades, enchantments, afflictions and local costs",
        "additional_card_dynamic_vars": "Raw captures retained; use explicit field bindings",
        "rng_scope": "Persistent run/player streams and retained observed event RNG counters",
        "explicit_field_bindings": len(case.fields),
        "explicit_phase_map": case.phase_map,
        "external_action": case.external_action,
        "simulator_action": case.simulator_action,
        "game_parity_certified": False,
        "scope": "One recorded before/action/after projection; not a lossless game checkpoint",
    }


def compare_live_comparison_file(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and "case" in payload:
        payload = payload["case"]
    return compare_live_comparison(LiveComparisonCase.model_validate(payload))


def capture_live_comparison(
    plan: Mapping[str, Any], *, base_url: str = DEFAULT_BASE_URL,
    execute: bool = False, client: LiveApiClient | None = None,
    settle_timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    """Record one deliberately selected game action; require explicit execution intent.

    The supplied full simulator state is never reconstructed from sparse bridge
    fields. A before-state mismatch prevents sending an action to the game.
    """
    if not execute:
        raise ValueError("Capture requires explicit --execute to submit the planned game action")
    active = client or LiveApiClient(LiveApiConfig(
        base_url=base_url, state_path="/api/v1/singleplayer",
        action_path="/api/v1/singleplayer", action_envelope="raw"))
    try:
        health = active.health()
        if (not isinstance(health, Mapping) or health.get("observation_schema_version") != 2
                or health.get("observer_actions_are_explicit") is not True):
            raise ValueError("Bridge health does not confirm read-only observation schema2; "
                             "prepare and load the reviewed source patch first")
        before = active.state()
        if not _settled(before):
            raise ValueError("Live before-state is still resolving an action or enemy turn")
        case = LiveComparisonCase.model_validate({
            **plan, "origin": "installed_game", "live_before": before, "live_after": before,
        })
        state = load_state(case.simulator_before)
        # Validate the planned simulator action before any live mutation.
        step(state, Action.model_validate(case.simulator_action))
        errors = _evidence_errors(case)
        before_mismatches = compare_snapshots(
            _live_projection(before, case.phase_map), _sim_projection(state),
            ParityCompareConfig(mode="subset", ignored_paths=()))
        if case.compare_rng:
            native = _native_rng(before)
            before_mismatches += compare_snapshots(
                native, _simulator_rng(state, native),
                ParityCompareConfig(mode="subset", ignored_paths=()))
        for binding in case.fields:
            expected, actual = _bound_values(before, serialize(state), binding)
            before_mismatches += compare_snapshots(
                {"value": expected}, {"value": actual},
                ParityCompareConfig(mode="subset", ignored_paths=()))
        if errors or before_mismatches:
            details: Any = errors if errors else [
                x.model_dump(mode="json") for x in before_mismatches
            ]
            raise ValueError("Live before-state does not match the explicit comparison plan: "
                             + str(details))
        response = active.apply_action(case.external_action)
        after, settled = _settled_after(active, before, settle_timeout_seconds)
        case = case.model_copy(update={"live_after": after})
        report = compare_live_comparison(case)
        report["live_after_settled"] = settled
        if not settled:
            report["matched"] = False
            report["supported"] = False
            report["unsupported_reasons"].append("Live action did not settle before the deadline")
        if _response_is_error(response):
            report["matched"] = False
            report["supported"] = False
            report["unsupported_reasons"].append("Bridge reported an action error")
        return {"case": case.model_dump(mode="json"), "report": report,
                "action_response": response}
    finally:
        if client is None:
            active.close()
