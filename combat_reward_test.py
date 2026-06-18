from __future__ import annotations

import argparse
import html
import json
import random
import re
import threading
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sts2sim import legal_actions, new_run, step
from sts2sim.engine.models import (
    ActionType,
    MapEdgeState,
    MapNodeState,
    MapState,
    RoomKind,
    RunPhase,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8798
DEFAULT_CHARACTER = "ironclad"
DATA_DIR = Path(__file__).resolve().parent / "data" / "cache" / "eng"


@dataclass(frozen=True)
class EventCombatPreset:
    preset_id: str
    event_id: str
    event_name: str
    option_id: str
    option_title: str
    reward_summary: str
    flags: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


EVENT_COMBAT_PRESETS: tuple[EventCombatPreset, ...] = (
    EventCombatPreset(
        preset_id="event_default",
        event_id="EVENT_FIGHT",
        event_name="Generic Event Fight",
        option_id="FIGHT",
        option_title="Default event combat",
        reward_summary="No fixed event reward. Shows only an optional combat potion roll.",
        flags={"combat_reward_potion_chance_percent": 40},
        notes=("Use this when an event starts combat but the post-fight reward is not known yet.",),
    ),
    EventCombatPreset(
        preset_id="fake_merchant",
        event_id="FAKE_MERCHANT",
        event_name="The Merchant???",
        option_id="THROW_FOUL_POTION",
        option_title="Throw Foul Potion",
        reward_summary="Fake Merchant's Rug plus all unsold fake relics.",
        flags={
            "combat_reward_event_id": "fake_merchant",
            "combat_reward_potion_chance_percent": 0,
        },
        notes=("Spire Codex says winning rewards the rug plus all unsold fake relics.",),
    ),
    EventCombatPreset(
        preset_id="battleworn_dummy_potion",
        event_id="BATTLEWORN_DUMMY",
        event_name="Battleworn Dummy",
        option_id="SETTING_1",
        option_title="Setting 1",
        reward_summary="Procure 1 random potion.",
        flags={
            "combat_reward_potion_chance_percent": 100,
            "combat_reward_card_count": 0,
            "combat_reward_relic_count": 0,
        },
    ),
    EventCombatPreset(
        preset_id="battleworn_dummy_upgrade",
        event_id="BATTLEWORN_DUMMY",
        event_name="Battleworn Dummy",
        option_id="SETTING_2",
        option_title="Setting 2",
        reward_summary="Upgrade 2 random cards.",
        flags={
            "combat_reward_potion_chance_percent": 0,
            "combat_reward_card_count": 0,
            "combat_reward_relic_count": 0,
        },
        notes=(
            "Card-upgrade reward is listed as an event effect; card upgrade selection "
            "is not wired yet.",
        ),
    ),
    EventCombatPreset(
        preset_id="battleworn_dummy_relic",
        event_id="BATTLEWORN_DUMMY",
        event_name="Battleworn Dummy",
        option_id="SETTING_3",
        option_title="Setting 3",
        reward_summary="Obtain a random relic.",
        flags={
            "combat_reward_potion_chance_percent": 0,
            "combat_reward_card_count": 0,
            "combat_reward_relic_count": 1,
        },
    ),
    EventCombatPreset(
        preset_id="punch_off_greater_rewards",
        event_id="PUNCH_OFF",
        event_name="Punch Off",
        option_id="I_CAN_TAKE_THEM",
        option_title="I Can Take Them",
        reward_summary=(
            "Normal combat rewards plus an additional potion and random relic."
        ),
        flags={
            "combat_reward_encounter": "normal",
            "combat_reward_card_count": 3,
            "combat_reward_relic_count": 1,
            "combat_reward_extra_potion_count": 1,
        },
        notes=(
            "The extra potion is always generated, and the normal combat potion "
            "roll can still add a second visible potion.",
        ),
    ),
    EventCombatPreset(
        preset_id="dense_vegetation",
        event_id="DENSE_VEGETATION",
        event_name="Dense Vegetation",
        option_id="REST",
        option_title="Rest",
        reward_summary="Heal 30% max HP, then fight for standard combat rewards.",
        flags={"combat_reward_encounter": "normal"},
        notes=(
            "The heal is an event entry effect and is shown as a note, not applied "
            "by the tester.",
        ),
    ),
    EventCombatPreset(
        preset_id="round_tea_party_relic",
        event_id="ROUND_TEA_PARTY",
        event_name="The Round Tea Party",
        option_id="PICK_FIGHT",
        option_title="Pick a Fight",
        reward_summary="Lose 11 HP. Obtain a random relic.",
        flags={
            "combat_reward_potion_chance_percent": 0,
            "combat_reward_card_count": 0,
            "combat_reward_relic_count": 1,
        },
        notes=(
            "HP loss is an event entry cost and is shown as a note, not applied by the "
            "tester.",
        ),
    ),
    EventCombatPreset(
        preset_id="lantern_key",
        event_id="THE_LANTERN_KEY",
        event_name="The Lantern Key",
        option_id="KEEP_THE_KEY",
        option_title="Keep the Key",
        reward_summary="Standard combat rewards plus the Lantern Key quest card.",
        flags={
            "combat_reward_encounter": "normal",
            "combat_reward_card_ids": ("lantern_key",),
        },
        notes=(
            "The quest card can later be redeemed at War Historian, Repy.",
        ),
    ),
)


def create_reward_state(
    *,
    seed: int,
    encounter: str,
    event_preset_id: str,
    ascension: int,
    character_id: str,
    fake_unsold_relics: tuple[str, ...],
    custom_gold: int | None = None,
    custom_card_count: int | None = None,
    custom_relic_count: int | None = None,
    custom_potion_chance: int | None = None,
) -> Any:
    source_data = _base_source_data()
    preset = _event_preset_by_id(event_preset_id)
    flags: dict[str, Any] = {}
    room_kind = _room_kind_for_encounter(encounter)
    if encounter == "event":
        room_kind = RoomKind.MONSTER
        flags.update({"combat_reward_encounter": "event"})
        flags.update(preset.flags)
        if preset.preset_id == "fake_merchant":
            flags["fake_merchant_unsold_relic_ids"] = fake_unsold_relics or _fake_relic_ids()

    if custom_gold is not None:
        flags["combat_reward_gold"] = custom_gold
    if custom_card_count is not None:
        flags["combat_reward_card_count"] = custom_card_count
    if custom_relic_count is not None:
        flags["combat_reward_relic_count"] = custom_relic_count
    if custom_potion_chance is not None:
        flags["combat_reward_potion_chance_percent"] = custom_potion_chance

    source_data.update(flags)
    state = new_run(
        seed=seed,
        character_id=character_id,
        ascension=ascension,
        source_data=source_data,
    )
    state = _force_combat_node(state, room_kind)
    state = step(state, _find_action(state, ActionType.CHOOSE_NODE, "target"))
    state = step(state, _find_action(state, ActionType.PLAY_CARD))
    return state


def _base_source_data() -> dict[str, Any]:
    return {
        "max_acts": 1,
        "map_floors": 4,
        "map_width": 1,
        "cards": _load_cache("cards"),
        "relic_pool": _load_cache("relics"),
        "boss_relic_pool": _load_cache("relics"),
        "potion_pool": _load_cache("potions"),
        "deck": [
            {
                "card_id": "debug_kill",
                "name": "Debug Kill",
                "type": "attack",
                "cost": 0,
                "target": "enemy",
                "effects": {"damage": 999},
            }
        ],
        "player": {"hp": 80, "max_hp": 80, "gold": 0, "energy": 3, "max_energy": 3},
    }


def _force_combat_node(state: Any, room_kind: RoomKind) -> Any:
    start = MapNodeState(node_id="start", act=1, floor=0, lane=0, kind=RoomKind.START)
    target = MapNodeState(node_id="target", act=1, floor=1, lane=0, kind=room_kind)
    game_map = MapState(
        act=1,
        nodes=(start, target),
        edges=(MapEdgeState(from_id=start.node_id, to_id=target.node_id),),
        current_node_id=start.node_id,
        completed_node_ids=(start.node_id,),
        boss_node_id=target.node_id if room_kind is RoomKind.BOSS else None,
    )
    return state.model_copy(
        update={
            "phase": RunPhase.MAP,
            "map": game_map,
            "floor": 0,
            "ancient": None,
            "combat": None,
            "reward": None,
            "room_history": (),
            "replay_log": (),
        }
    )


def _find_action(state: Any, action_type: ActionType, target_id: str | None = None) -> Any:
    for action in legal_actions(state):
        if action.type != action_type:
            continue
        if target_id is not None and action.target_id != target_id:
            continue
        return action
    raise RuntimeError(f"No legal action found: {action_type.value} {target_id or ''}".strip())


def _apply_action(
    state: Any,
    action_type: ActionType,
    target_id: str | None = None,
) -> tuple[Any, str]:
    for action in legal_actions(state):
        if action.type == action_type and (target_id is None or action.target_id == target_id):
            return step(state, action), f"Applied {action_type.value}."
    return state, f"Action is not legal: {action_type.value} {target_id or ''}".strip()


def _state_payload(
    state: Any,
    *,
    seed: int,
    encounter: str,
    event_preset_id: str,
    message: str,
) -> dict[str, Any]:
    reward = state.reward
    preset = _event_preset_by_id(event_preset_id)
    return {
        "seed": seed,
        "encounter": encounter,
        "event_preset": _preset_payload(preset),
        "message": message,
        "phase": state.phase.value,
        "player": {
            "gold": state.player.gold,
            "hp": state.player.hp,
            "max_hp": state.player.max_hp,
        },
        "reward": _reward_payload(reward),
        "legal_actions": [
            {"type": action.type.value, "target_id": action.target_id}
            for action in legal_actions(state)
        ],
        "relics": list(state.relics),
        "potions": list(state.potions),
        "deck": [
            {
                "instance_id": card.instance_id,
                "card_id": card.card_id,
                "name": card.name,
                "upgraded": card.upgraded,
            }
            for card in state.master_deck
        ],
        "events": _event_catalog_payload(),
        "fake_relic_ids": list(_fake_relic_ids()),
    }


def _reward_payload(reward: Any | None) -> dict[str, Any] | None:
    if reward is None:
        return None
    return {
        "reward_id": reward.reward_id,
        "source": reward.source,
        "forced": reward.forced,
        "gold": reward.gold,
        "gold_claimed": reward.gold_claimed,
        "relic_id": reward.relic_id,
        "relic_claimed": reward.relic_claimed,
        "relic_ids": list(reward.relic_ids),
        "claimed_relic_ids": list(reward.claimed_relic_ids),
        "card_ids": list(reward.card_ids),
        "claimed_card_indices": list(reward.claimed_card_indices),
        "card_options": list(reward.card_options),
        "card_claimed": reward.card_claimed,
        "potion_id": reward.potion_id,
        "potion_claimed": reward.potion_claimed,
        "potion_ids": list(reward.potion_ids),
        "claimed_potion_indices": list(reward.claimed_potion_indices),
        "metadata": reward.metadata,
    }


def _event_catalog_payload() -> list[dict[str, Any]]:
    events = _load_cache("events")
    catalog: list[dict[str, Any]] = []
    for event in events:
        event_id = str(event.get("id", ""))
        options = _event_options(event)
        if event_id == "FAKE_MERCHANT" and not options:
            options = [
                {
                    "option_id": "THROW_FOUL_POTION",
                    "title": "Throw Foul Potion",
                    "description": _clean_text(str(event.get("description", ""))),
                    "combat": "true",
                    "reward": (
                        "Winning rewards Fake Merchant's Rug plus all unsold relics."
                    ),
                }
            ]
        combat_options = [option for option in options if _looks_like_combat_option(option)]
        if not combat_options:
            continue
        catalog.append(
            {
                "event_id": event_id,
                "name": event.get("name", event_id),
                "act": event.get("act"),
                "options": combat_options,
            }
        )
    return catalog


def _event_options(event: dict[str, Any]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    results: list[dict[str, str]] = []
    raw_options: list[Any] = list(event.get("options") or [])
    for page in event.get("pages") or []:
        raw_options.extend(page.get("options") or [])

    for option in raw_options:
        if not isinstance(option, dict):
            continue
        option_id = str(option.get("id", ""))
        title = str(option.get("title", option_id))
        description = _clean_text(str(option.get("description", "")))
        key = (option_id, description)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "option_id": option_id,
                "title": title,
                "description": description,
                "combat": str(_looks_like_combat_option(option)).lower(),
                "reward": _reward_hint(description),
            }
        )
    return results


def _looks_like_combat_option(option: dict[str, Any]) -> bool:
    text = " ".join(str(option.get(key, "")) for key in ("id", "title", "description"))
    return bool(re.search(r"\bfight\b", text, re.IGNORECASE))


def _reward_hint(text: str) -> str:
    if not text:
        return ""
    matches = re.findall(
        r"((?:Gain|Obtain|Procure|Add|Upgrade|Transform|Heal|Lose|Receive)[^.]*\.)",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(matches[:3]) or text


def _clean_text(text: str) -> str:
    text = re.sub(r"\[(?:/?[a-zA-Z_]+|/?[a-zA-Z_]+=[^\]]+)\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return html.unescape(text).strip()


def _preset_payload(preset: EventCombatPreset) -> dict[str, Any]:
    return {
        "preset_id": preset.preset_id,
        "event_id": preset.event_id,
        "event_name": preset.event_name,
        "option_id": preset.option_id,
        "option_title": preset.option_title,
        "reward_summary": preset.reward_summary,
        "notes": list(preset.notes),
    }


def _event_preset_by_id(preset_id: str) -> EventCombatPreset:
    for preset in EVENT_COMBAT_PRESETS:
        if preset.preset_id == preset_id:
            return preset
    return EVENT_COMBAT_PRESETS[0]


def _room_kind_for_encounter(encounter: str) -> RoomKind:
    if encounter == "elite":
        return RoomKind.ELITE
    if encounter == "boss":
        return RoomKind.BOSS
    return RoomKind.MONSTER


def _load_cache(name: str) -> list[dict[str, Any]]:
    path = DATA_DIR / f"{name}.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else []


def _fake_relic_ids() -> tuple[str, ...]:
    fake_ids = []
    for relic in _load_cache("relics"):
        relic_id = str(relic.get("id", "")).lower()
        if relic_id.startswith("fake_") and relic_id != "fake_merchants_rug":
            fake_ids.append(relic_id)
    return tuple(fake_ids)


class CombatRewardContext:
    def __init__(self, args: argparse.Namespace) -> None:
        self.lock = threading.Lock()
        self.seed = args.seed if args.seed is not None else random.randrange(1_000_000_000)
        self.encounter = args.encounter
        self.event_preset_id = args.event_preset
        self.ascension = args.ascension
        self.character_id = args.character
        self.fake_unsold_relics = _fake_relic_ids()
        self.state = create_reward_state(
            seed=self.seed,
            encounter=self.encounter,
            event_preset_id=self.event_preset_id,
            ascension=self.ascension,
            character_id=self.character_id,
            fake_unsold_relics=self.fake_unsold_relics,
        )
        self.message = "Reward generated."

    def reset(self, body: dict[str, Any]) -> None:
        if body.get("random_seed"):
            self.seed = random.randrange(1_000_000_000)
        elif "seed" in body:
            self.seed = int(body.get("seed") or self.seed)
        self.encounter = str(body.get("encounter", self.encounter))
        self.event_preset_id = str(body.get("event_preset", self.event_preset_id))
        self.ascension = int(body.get("ascension", self.ascension))
        self.character_id = str(body.get("character", self.character_id))
        self.fake_unsold_relics = tuple(
            str(item) for item in body.get("fake_unsold_relics", self.fake_unsold_relics)
        )
        self.state = create_reward_state(
            seed=self.seed,
            encounter=self.encounter,
            event_preset_id=self.event_preset_id,
            ascension=self.ascension,
            character_id=self.character_id,
            fake_unsold_relics=self.fake_unsold_relics,
            custom_gold=_optional_int(body.get("custom_gold")),
            custom_card_count=_optional_int(body.get("custom_card_count")),
            custom_relic_count=_optional_int(body.get("custom_relic_count")),
            custom_potion_chance=_optional_int(body.get("custom_potion_chance")),
        )
        self.message = "Reward generated."

    def payload(self) -> dict[str, Any]:
        return _state_payload(
            self.state,
            seed=self.seed,
            encounter=self.encounter,
            event_preset_id=self.event_preset_id,
            message=self.message,
        )


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def run_web(args: argparse.Namespace) -> None:
    context = CombatRewardContext(args)
    handler = _make_handler(context)
    server = _bind_server(args.host, args.port, handler)
    host, port = server.server_address
    url = f"http://{host}:{port}"
    print(f"Combat reward test web UI running at {url}")
    print("Press Ctrl+C to stop it.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping combat reward test server.")
    finally:
        server.server_close()


def _bind_server(
    host: str,
    port: int,
    handler: type[BaseHTTPRequestHandler],
) -> ThreadingHTTPServer:
    for candidate_port in range(port, port + 20):
        try:
            return ThreadingHTTPServer((host, candidate_port), handler)
        except OSError:
            continue
    raise OSError(f"Could not bind a combat reward server on {host}:{port}-{port + 19}")


def _make_handler(context: CombatRewardContext) -> type[BaseHTTPRequestHandler]:
    class CombatRewardHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_html(_html_page())
                return
            if path == "/api/state":
                with context.lock:
                    self._send_json(context.payload())
                return
            self.send_error(404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            body = self._read_json_body()
            with context.lock:
                if path == "/api/reset":
                    context.reset(body)
                elif path == "/api/take":
                    action_type = ActionType(str(body.get("action_type", "")))
                    target_id = str(body.get("target_id", "")) or None
                    context.state, context.message = _apply_action(
                        context.state,
                        action_type,
                        target_id,
                    )
                elif path == "/api/proceed":
                    context.state, context.message = _apply_action(
                        context.state,
                        ActionType.PROCEED,
                    )
                else:
                    self.send_error(404)
                    return
                self._send_json(context.payload())

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                return {}
            return data if isinstance(data, dict) else {}

        def _send_json(self, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_html(self, content: str) -> None:
            encoded = content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return CombatRewardHandler


def _html_page() -> str:
    presets_json = json.dumps([_preset_payload(preset) for preset in EVENT_COMBAT_PRESETS])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>sts2sim Combat Reward Test</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f1ea;
      color: #171717;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: #f4f1ea;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      padding: 22px 28px 16px;
      border-bottom: 1px solid #cec7b6;
      background: #262421;
      color: #faf7ef;
    }}
    h1, h2, h3 {{
      margin: 0;
      letter-spacing: 0;
    }}
    h1 {{ font-size: 24px; line-height: 1.1; }}
    h2 {{ font-size: 15px; margin-bottom: 10px; }}
    h3 {{ font-size: 14px; margin-bottom: 8px; }}
    main {{
      display: grid;
      grid-template-columns: minmax(260px, 340px) minmax(0, 1fr);
      gap: 20px;
      padding: 20px 28px 28px;
    }}
    section {{
      background: #fffdf7;
      border: 1px solid #d8d0bf;
      border-radius: 8px;
      padding: 16px;
    }}
    .side {{
      display: grid;
      gap: 16px;
      align-content: start;
    }}
    .stack {{ display: grid; gap: 12px; }}
    .row {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
    }}
    label {{
      display: grid;
      gap: 5px;
      color: #5f5b52;
      font-size: 12px;
      font-weight: 700;
    }}
    input, select, button {{
      min-height: 34px;
      border-radius: 6px;
      border: 1px solid #827a6b;
      background: #ffffff;
      color: #161616;
      padding: 6px 10px;
      font: inherit;
    }}
    input[type="number"] {{ width: 110px; }}
    button {{
      cursor: pointer;
      font-weight: 700;
    }}
    button.primary {{
      background: #1f6f54;
      border-color: #1f6f54;
      color: white;
    }}
    button.warning {{
      background: #8f3d2e;
      border-color: #8f3d2e;
      color: white;
    }}
    button:disabled {{
      cursor: not-allowed;
      opacity: 0.5;
    }}
    .message {{
      padding: 10px 12px;
      border-radius: 6px;
      background: #e5f1ea;
      color: #164330;
      border: 1px solid #b8d1c3;
      font-weight: 650;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .stat {{
      border: 1px solid #ded6c8;
      border-radius: 6px;
      padding: 9px 10px;
      background: #f8f4eb;
    }}
    .stat b {{
      display: block;
      font-size: 12px;
      color: #5f5b52;
      font-weight: 600;
    }}
    .stat span {{
      font-size: 20px;
      font-weight: 700;
    }}
    .reward-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .reward-box {{
      border: 1px solid #ded6c8;
      border-radius: 7px;
      background: #f8f4eb;
      padding: 12px;
      min-height: 96px;
    }}
    .reward-list {{
      display: grid;
      gap: 8px;
    }}
    .reward-row {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      border: 1px solid #d3caba;
      border-radius: 6px;
      background: #fffaf0;
      padding: 8px;
    }}
    .reward-name {{
      min-width: 0;
      font-weight: 750;
      font-size: 13px;
    }}
    .reward-name small {{
      display: block;
      margin-top: 2px;
      color: #716d65;
      font-size: 11px;
      font-weight: 600;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .chip {{
      border: 1px solid #b9b09f;
      background: #eee6d8;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
    }}
    .fake-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
      max-height: 170px;
      overflow: auto;
      padding-right: 4px;
    }}
    .check {{
      display: flex;
      align-items: center;
      gap: 6px;
      color: #34312c;
      font-size: 12px;
      font-weight: 650;
    }}
    .check input {{ min-height: 0; }}
    .catalog {{
      display: grid;
      gap: 10px;
      max-height: 420px;
      overflow: auto;
      padding-right: 4px;
    }}
    .event-card {{
      border: 1px solid #ded6c8;
      border-radius: 7px;
      background: #f8f4eb;
      padding: 10px;
    }}
    .event-card strong {{
      display: block;
      margin-bottom: 4px;
    }}
    .event-option {{
      border-top: 1px solid #e4ddcf;
      margin-top: 7px;
      padding-top: 7px;
      font-size: 12px;
      line-height: 1.35;
    }}
    .muted {{ color: #716d65; }}
    .claimed {{ color: #8f3d2e; font-weight: 700; }}
    .hint {{
      margin: 8px 0 0;
      color: #625d54;
      font-size: 12px;
      line-height: 1.35;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 280px;
      overflow: auto;
      margin: 0;
      background: #262421;
      color: #faf7ef;
      border-radius: 7px;
      padding: 12px;
      font-size: 12px;
      line-height: 1.35;
    }}
    @media (max-width: 900px) {{
      header {{ display: block; }}
      main {{
        grid-template-columns: 1fr;
        padding: 16px;
      }}
      .reward-grid {{ grid-template-columns: 1fr; }}
      .fake-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>sts2sim Combat Reward Test</h1>
      <div id="seed-label" class="muted"></div>
    </div>
    <div class="row">
      <button id="reset">Generate</button>
      <button id="random-reset">New Seed</button>
    </div>
  </header>
  <main>
    <div class="side">
      <section class="stack">
        <h2>Reward Source</h2>
        <label>Encounter
          <select id="encounter">
            <option value="normal">Monster</option>
            <option value="elite">Elite</option>
            <option value="boss">Boss</option>
            <option value="event">Event fight</option>
          </select>
        </label>
        <label id="event-wrap">Event preset
          <select id="event-preset"></select>
        </label>
        <div id="preset-summary" class="hint"></div>
        <div id="fake-wrap">
          <h3>Fake Merchant unsold relics</h3>
          <div id="fake-relics" class="fake-grid"></div>
        </div>
      </section>
      <section class="stack">
        <h2>Seed and Overrides</h2>
        <div class="row">
          <label>Seed <input id="seed-input" type="number"></label>
          <label>Ascension <input id="ascension" type="number" min="0" max="20" value="0"></label>
        </div>
        <div class="row">
          <label>Gold <input id="custom-gold" type="number" placeholder="auto"></label>
          <label>Cards <input id="custom-cards" type="number" placeholder="auto"></label>
        </div>
        <div class="row">
          <label>Relics <input id="custom-relics" type="number" placeholder="auto"></label>
          <label>Potion % <input id="custom-potion" type="number" placeholder="auto"></label>
        </div>
      </section>
      <section>
        <h2>State</h2>
        <div class="stats">
          <div class="stat"><b>Phase</b><span id="phase">reward</span></div>
          <div class="stat"><b>Gold</b><span id="player-gold">0</span></div>
          <div class="stat"><b>HP</b><span id="hp">0/0</span></div>
          <div class="stat"><b>Relics</b><span id="relic-count">0</span></div>
        </div>
      </section>
    </div>
    <div class="stack">
      <section class="stack">
        <div class="message" id="message">Loading...</div>
        <div class="reward-grid">
          <div class="reward-box">
            <h2>Gold</h2>
            <div id="gold-reward" class="reward-list"></div>
          </div>
          <div class="reward-box">
            <h2>Potion</h2>
            <div id="potion-reward" class="reward-list"></div>
          </div>
          <div class="reward-box">
            <h2>Cards</h2>
            <div id="card-reward" class="reward-list"></div>
          </div>
          <div class="reward-box">
            <h2>Relics</h2>
            <div id="relic-reward" class="reward-list"></div>
          </div>
        </div>
        <div class="row">
          <button id="proceed" class="warning">Skip / Proceed</button>
        </div>
      </section>
      <section class="stack">
        <h2>Known Event Fight Data</h2>
        <div id="event-catalog" class="catalog"></div>
      </section>
      <section>
        <h2>Raw Reward State</h2>
        <pre id="raw"></pre>
      </section>
    </div>
  </main>
  <script>
    const PRESETS = {presets_json};
    let current = null;

    const el = (id) => document.getElementById(id);
    const title = (id) => String(id || "").replaceAll("_", " ")
      .replace(/\\b\\w/g, (c) => c.toUpperCase());

    function presetById(id) {{
      return PRESETS.find((preset) => preset.preset_id === id) || PRESETS[0];
    }}

    function action(type, targetId) {{
      return (current?.legal_actions || []).find((item) =>
        item.type === type && (!targetId || item.target_id === targetId)
      );
    }}

    async function getState() {{
      const response = await fetch("/api/state");
      current = await response.json();
      render();
    }}

    function payload(randomSeed=false) {{
      return {{
        random_seed: randomSeed,
        seed: el("seed-input").value,
        encounter: el("encounter").value,
        event_preset: el("event-preset").value,
        ascension: el("ascension").value,
        fake_unsold_relics: [...document.querySelectorAll(".fake-check:checked")]
          .map((item) => item.value),
        custom_gold: el("custom-gold").value,
        custom_card_count: el("custom-cards").value,
        custom_relic_count: el("custom-relics").value,
        custom_potion_chance: el("custom-potion").value,
      }};
    }}

    async function reset(randomSeed=false) {{
      const response = await fetch("/api/reset", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify(payload(randomSeed)),
      }});
      current = await response.json();
      render();
    }}

    async function post(path, body={{}}) {{
      const response = await fetch(path, {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify(body),
      }});
      current = await response.json();
      render();
    }}

    function render() {{
      if (!current) return;
      el("seed-label").textContent = `Seed ${{current.seed}}`;
      el("seed-input").value = current.seed;
      el("message").textContent = current.message;
      el("phase").textContent = current.phase;
      el("player-gold").textContent = current.player.gold;
      el("hp").textContent = `${{current.player.hp}}/${{current.player.max_hp}}`;
      el("relic-count").textContent = current.relics.length;
      renderPresetControls();
      renderReward();
      renderCatalog();
      el("raw").textContent = JSON.stringify(current.reward, null, 2);
    }}

    function renderPresetControls() {{
      if (!el("event-preset").children.length) {{
        el("event-preset").innerHTML = PRESETS.map((preset) =>
          `<option value="${{preset.preset_id}}">` +
          `${{preset.event_name}} - ${{preset.option_title}}</option>`
        ).join("");
      }}
      el("encounter").value = current.encounter;
      el("event-preset").value = current.event_preset.preset_id;
      el("event-wrap").style.display = current.encounter === "event" ? "grid" : "none";
      const preset = presetById(el("event-preset").value);
      el("preset-summary").textContent = current.encounter === "event"
        ? `${{preset.reward_summary}} ${{preset.notes.join(" ")}}`
        : "Standard combat reward generation from the selected room type.";
      el("fake-wrap").style.display =
        current.encounter === "event" && el("event-preset").value === "fake_merchant"
          ? "block"
          : "none";
      if (!el("fake-relics").children.length) {{
        el("fake-relics").innerHTML = current.fake_relic_ids.map((id) =>
          `<label class="check"><input class="fake-check" ` +
          `type="checkbox" value="${{id}}" checked>${{title(id)}}</label>`
        ).join("");
      }}
    }}

    function renderReward() {{
      const reward = current.reward;
      if (!reward) {{
        for (const id of ["gold-reward", "potion-reward", "card-reward", "relic-reward"]) {{
          el(id).innerHTML = `<div class="muted">No active reward.</div>`;
        }}
        el("proceed").disabled = !action("proceed");
        return;
      }}
      renderGold(reward);
      renderPotion(reward);
      renderCards(reward);
      renderRelics(reward);
      el("proceed").disabled = !action("proceed");
    }}

    function renderGold(reward) {{
      const available = action("take_reward_gold", "reward:gold");
      el("gold-reward").innerHTML = reward.gold > 0
        ? rewardRow(
            `${{reward.gold}} gold`,
            reward.gold_claimed ? "claimed" : "available",
            "take_reward_gold",
            "reward:gold",
            available,
          )
        : `<div class="muted">No gold reward.</div>`;
    }}

    function renderPotion(reward) {{
      const rows = [];
      if (reward.potion_id) {{
        rows.push(potionRow(
          reward.potion_id,
          "reward:potion",
          reward.potion_claimed,
        ));
      }}
      reward.potion_ids.forEach((potionId, index) => {{
        rows.push(potionRow(
          potionId,
          `reward:potion:${{index}}`,
          reward.claimed_potion_indices.includes(index),
        ));
      }});
      el("potion-reward").innerHTML = rows.length
        ? rows.join("")
        : `<div class="muted">No potion reward.</div>`;
    }}

    function potionRow(potionId, target, claimed) {{
      const available = action("take_reward_potion", target);
      return rewardRow(
        title(potionId),
        claimed ? "claimed" : "visible if slot is open",
        "take_reward_potion",
        target,
        available,
      );
    }}

    function renderCards(reward) {{
      const rows = reward.card_options.map((cardId, index) => {{
            const target = `reward:card:${{index}}`;
            const available = action("take_reward_card", target);
            const rarity = reward.metadata?.card_rarities?.[index] || "card";
            return rewardRow(
              title(cardId),
              `${{rarity}} ${{reward.card_claimed ? "claimed" : "choice"}}`,
              "take_reward_card",
              target,
              available,
            );
          }});
      reward.card_ids.forEach((cardId, index) => {{
        const target = `reward:fixed_card:${{index}}`;
        const available = action("take_reward_card", target);
        rows.push(rewardRow(
          title(cardId),
          reward.claimed_card_indices.includes(index) ? "claimed" : "guaranteed card",
          "take_reward_card",
          target,
          available,
        ));
      }});
      el("card-reward").innerHTML = rows.length
        ? rows.join("")
        : `<div class="muted">No card choices.</div>`;
    }}

    function renderRelics(reward) {{
      const rows = [];
      if (reward.relic_id) {{
        rows.push(relicRow(
          reward.relic_id,
          "reward:relic",
          reward.relic_claimed,
          reward.metadata?.relic_rarity,
        ));
      }}
      reward.relic_ids.forEach((relicId, index) => {{
        rows.push(relicRow(
          relicId,
          `reward:relic:${{index}}`,
          reward.claimed_relic_ids.includes(relicId),
          reward.metadata?.relic_rarities?.[index],
        ));
      }});
      el("relic-reward").innerHTML = rows.length
        ? rows.join("")
        : `<div class="muted">No relic reward.</div>`;
    }}

    function relicRow(relicId, target, claimed, rarity) {{
      const available = action("take_reward_relic", target);
      return rewardRow(
        title(relicId),
        `${{rarity || "relic"}} ${{claimed ? "claimed" : "available"}}`,
        "take_reward_relic",
        target,
        available,
      );
    }}

    function rewardRow(name, detail, actionType, target, available) {{
      return `<div class="reward-row"><div class="reward-name">${{name}}` +
        `<small>${{detail}}</small></div>` +
        `<button data-action="${{actionType}}" data-target="${{target}}" ` +
        `${{available ? "" : "disabled"}}>Take</button></div>`;
    }}

    function renderCatalog() {{
      el("event-catalog").innerHTML = current.events.map((event) => {{
        const options = event.options.map((option) =>
          `<div class="event-option"><strong>${{option.title}}</strong>` +
          `<div>${{option.description || "No description."}}</div>` +
          `<div class="muted">${{option.reward || ""}}</div></div>`
        ).join("");
        return `<div class="event-card"><strong>${{event.name}}</strong>` +
          `<div class="muted">${{event.event_id}}` +
          `${{event.act ? " / " + event.act : ""}}</div>${{options}}</div>`;
      }}).join("");
    }}

    document.addEventListener("click", (event) => {{
      const button = event.target.closest("button");
      if (!button) return;
      if (button.id === "reset") reset(false);
      if (button.id === "random-reset") reset(true);
      if (button.id === "proceed") post("/api/proceed");
      if (button.dataset.action) {{
        post("/api/take", {{
          action_type: button.dataset.action,
          target_id: button.dataset.target,
        }});
      }}
    }});
    el("encounter").addEventListener("change", () => reset(false));
    el("event-preset").addEventListener("change", () => reset(false));
    getState();
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive combat reward tester for the headless sts2sim engine."
    )
    parser.add_argument("--seed", type=int, help="Deterministic reward seed.")
    parser.add_argument(
        "--encounter",
        choices=("normal", "elite", "boss", "event"),
        default="normal",
        help="Reward source to generate.",
    )
    parser.add_argument(
        "--event-preset",
        choices=tuple(preset.preset_id for preset in EVENT_COMBAT_PRESETS),
        default="event_default",
        help="Event reward preset used when --encounter event.",
    )
    parser.add_argument("--ascension", type=int, default=0, help="Ascension level.")
    parser.add_argument("--character", default=DEFAULT_CHARACTER, help="Character id.")
    parser.add_argument("--web", action="store_true", help="Open a small local web UI.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Web server host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Web server port.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser.")
    args = parser.parse_args()
    if args.web:
        run_web(args)
        return

    seed = args.seed if args.seed is not None else random.randrange(1_000_000_000)
    state = create_reward_state(
        seed=seed,
        encounter=args.encounter,
        event_preset_id=args.event_preset,
        ascension=args.ascension,
        character_id=args.character,
        fake_unsold_relics=_fake_relic_ids(),
    )
    print(json.dumps(_state_payload(
        state,
        seed=seed,
        encounter=args.encounter,
        event_preset_id=args.event_preset,
        message="Reward generated.",
    )["reward"], indent=2))


if __name__ == "__main__":
    main()
