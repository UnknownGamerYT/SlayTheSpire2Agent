from __future__ import annotations

import argparse
import html
import json
import random
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from sts2sim import legal_actions, new_run, step
from sts2sim.engine.models import (
    ActionType,
    MapNodeState,
    MapState,
    PlayerState,
    RoomKind,
    RunPhase,
)
from sts2sim.engine.transitions import _enter_shop_room, _potion_capacity

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_POTIONS = ("foul_potion",)

SHOP_TEST_SOURCE_DATA: dict[str, Any] = {
    "max_acts": 1,
    "shop_plan": {
        "colored_cards": 5,
        "colorless_cards": 2,
        "relics": 3,
        "potions": 3,
        "include_card_removal": True,
    },
    "shop_card_pool": [
        {"id": "pommel_strike", "kind": "card", "rarity": "common", "type": "attack"},
        {"id": "cleave", "kind": "card", "rarity": "common", "type": "attack"},
        {"id": "uppercut", "kind": "card", "rarity": "uncommon", "type": "attack"},
        {"id": "feed", "kind": "card", "rarity": "rare", "type": "attack"},
        {"id": "shrug_it_off", "kind": "card", "rarity": "common", "type": "skill"},
        {"id": "true_grit", "kind": "card", "rarity": "common", "type": "skill"},
        {"id": "flame_barrier", "kind": "card", "rarity": "uncommon", "type": "skill"},
        {"id": "impervious", "kind": "card", "rarity": "rare", "type": "skill"},
        {"id": "inflame", "kind": "card", "rarity": "uncommon", "type": "power"},
        {"id": "demon_form", "kind": "card", "rarity": "rare", "type": "power"},
    ],
    "shop_colorless_card_pool": [
        {"id": "trip", "kind": "colorless_card", "rarity": "uncommon"},
        {"id": "flash_of_steel", "kind": "colorless_card", "rarity": "uncommon"},
        {"id": "apotheosis", "kind": "colorless_card", "rarity": "rare"},
        {"id": "master_of_strategy", "kind": "colorless_card", "rarity": "rare"},
    ],
    "shop_relic_pool": [
        {"id": "anchor", "kind": "relic", "rarity": "common"},
        {"id": "smiling_mask", "kind": "relic", "rarity": "uncommon"},
        {"id": "shovel", "kind": "relic", "rarity": "rare"},
        {"id": "cauldron", "kind": "relic", "rarity": "shop"},
        {"id": "chemical_x", "kind": "relic", "rarity": "shop"},
        {"id": "frozen_eye", "kind": "relic", "rarity": "shop"},
        {"id": "medical_kit", "kind": "relic", "rarity": "shop"},
        {"id": "membership_card", "kind": "relic", "rarity": "shop"},
        {"id": "orange_pellets", "kind": "relic", "rarity": "shop"},
        {"id": "prismatic_shard", "kind": "relic", "rarity": "shop"},
    ],
    "shop_potion_pool": [
        {"id": "fire_potion", "kind": "potion", "rarity": "common"},
        {"id": "skill_potion", "kind": "potion", "rarity": "uncommon"},
        {"id": "essence_of_steel", "kind": "potion", "rarity": "rare"},
        {"id": "foul_potion", "kind": "potion", "rarity": "rare"},
    ],
    "deck": [
        {
            "instance_id": "strike_1",
            "card_id": "strike",
            "name": "Strike",
            "type": "attack",
            "cost": 1,
            "target": "enemy",
            "effects": {"damage": 6},
        },
        {
            "instance_id": "strike_2",
            "card_id": "strike",
            "name": "Strike",
            "type": "attack",
            "cost": 1,
            "target": "enemy",
            "effects": {"damage": 6},
        },
        {
            "instance_id": "defend_1",
            "card_id": "defend",
            "name": "Defend",
            "type": "skill",
            "cost": 1,
            "target": "self",
            "effects": {"block": 5},
        },
        {
            "instance_id": "defend_2",
            "card_id": "defend",
            "name": "Defend",
            "type": "skill",
            "cost": 1,
            "target": "self",
            "effects": {"block": 5},
        },
        {
            "instance_id": "bash_1",
            "card_id": "bash",
            "name": "Bash",
            "type": "attack",
            "cost": 2,
            "target": "enemy",
            "effects": {"damage": 8},
        },
    ],
    "player": {"hp": 70, "max_hp": 80, "gold": 350, "energy": 3, "max_energy": 3},
}

KIND_LABELS = {
    "card": "Character card",
    "colorless_card": "Colorless card",
    "potion": "Potion",
    "relic": "Relic",
    "card_removal": "Card removal",
}


def create_shop_state(
    *,
    seed: int,
    gold: int,
    ascension: int,
    relics: tuple[str, ...],
    potions: tuple[str, ...],
) -> Any:
    source_data = json.loads(json.dumps(SHOP_TEST_SOURCE_DATA))
    source_data["player"]["gold"] = gold
    state = new_run(
        seed=seed,
        character_id="TEST",
        ascension=ascension,
        source_data=source_data,
    )
    start_node = MapNodeState(
        node_id="debug_start",
        act=1,
        floor=0,
        lane=0,
        kind=RoomKind.START,
    )
    shop_node = MapNodeState(
        node_id="debug_shop",
        act=1,
        floor=1,
        lane=0,
        kind=RoomKind.SHOP,
    )
    map_state = MapState(
        act=1,
        nodes=(start_node, shop_node),
        edges=(),
        current_node_id=shop_node.node_id,
        completed_node_ids=(start_node.node_id,),
        boss_node_id=None,
    )
    player = PlayerState(
        hp=70,
        max_hp=80,
        gold=gold,
        energy=3,
        max_energy=3,
    )
    state = state.model_copy(
        update={
            "phase": RunPhase.MAP,
            "act": 1,
            "floor": 1,
            "player": player,
            "relics": relics,
            "potions": potions,
            "ancient": None,
            "combat": None,
            "map": map_state,
            "room_history": (),
            "replay_log": (),
        }
    )
    state, _events = _enter_shop_room(state, shop_node)
    return state


def _display_name(item_id: str) -> str:
    if item_id == "card_removal":
        return "Card Removal"
    return item_id.replace("_", " ").title()


def _find_action(state: Any, action_type: ActionType, target_id: str | None = None) -> Any | None:
    for action in legal_actions(state):
        if action.type != action_type:
            continue
        if target_id is not None and action.target_id != target_id:
            continue
        return action
    return None


def _apply_action(
    state: Any,
    action_type: ActionType,
    *,
    target_id: str | None = None,
) -> tuple[Any, str]:
    action = _find_action(state, action_type, target_id)
    if action is None:
        label = action_type.value
        if target_id is not None:
            label = f"{label} {target_id}"
        return state, f"That action is not legal right now: {label}"
    next_state = step(state, action)
    return next_state, _event_message(next_state)


def _event_message(state: Any) -> str:
    if not state.replay_log:
        return "Shop is ready."
    events = state.replay_log[-1].events
    if not events:
        return "Action resolved."

    parts: list[str] = []
    for event in events:
        text = event.kind.replace("_", " ")
        if event.target_id:
            text += f": {_display_name(event.target_id)}"
        if event.amount is not None:
            text += f" ({event.amount:+d})"
        parts.append(text)
    return "; ".join(parts)


def _state_payload(state: Any, *, seed: int, message: str) -> dict[str, Any]:
    actions = legal_actions(state)
    buy_targets = {
        str(action.target_id)
        for action in actions
        if action.type == ActionType.SHOP_BUY and action.target_id is not None
    }
    discard_targets = {
        str(action.target_id)
        for action in actions
        if action.type == ActionType.DISCARD_POTION and action.target_id is not None
    }
    action_types = {action.type for action in actions}
    items: list[dict[str, Any]] = []
    potion_slots = _potion_capacity(state)

    if state.shop is not None:
        for index, item in enumerate(state.shop.items):
            buy_target = f"shop:{index}"
            remove_prefix = f"{buy_target}:remove:"
            remove_targets = sorted(
                target.removeprefix(remove_prefix)
                for target in buy_targets
                if target.startswith(remove_prefix)
            )
            can_buy = buy_target in buy_targets or bool(remove_targets)
            items.append(
                {
                    "slot": index,
                    "target_id": buy_target,
                    "item_id": item.item_id,
                    "name": _display_name(item.item_id),
                    "kind": item.kind,
                    "kind_label": KIND_LABELS.get(item.kind, item.kind),
                    "rarity": item.rarity or "",
                    "price": item.price,
                    "base_price": item.base_price,
                    "purchased": item.purchased,
                    "can_buy": can_buy,
                    "blocked_reason": _blocked_shop_reason(
                        item_kind=str(item.kind),
                        item_price=item.price,
                        can_buy=can_buy,
                        purchased=item.purchased,
                        gold=state.player.gold,
                        potion_count=len(state.potions),
                        potion_slots=potion_slots,
                        deck_count=len(state.master_deck),
                    ),
                    "remove_targets": remove_targets,
                }
            )

    deck = [
        {
            "instance_id": card.instance_id,
            "card_id": card.card_id,
            "name": card.name or _display_name(card.card_id),
            "upgraded": card.upgraded,
        }
        for card in state.master_deck
    ]
    return {
        "seed": seed,
        "phase": getattr(state.phase, "value", str(state.phase)),
        "message": message,
        "gold": state.player.gold,
        "hp": state.player.hp,
        "max_hp": state.player.max_hp,
        "potion_slots": potion_slots,
        "potion_slots_open": max(0, potion_slots - len(state.potions)),
        "potions": [
            {
                "slot": index,
                "target_id": f"potion:{index}",
                "id": potion,
                "name": _display_name(potion),
                "can_discard": f"potion:{index}" in discard_targets,
            }
            for index, potion in enumerate(state.potions)
        ],
        "relics": [{"id": relic, "name": _display_name(relic)} for relic in state.relics],
        "deck": deck,
        "items": items,
        "can_throw_foul_potion": ActionType.THROW_POTION_AT_MERCHANT in action_types,
        "can_leave": ActionType.SHOP_LEAVE in action_types,
    }


def _blocked_shop_reason(
    *,
    item_kind: str,
    item_price: int,
    can_buy: bool,
    purchased: bool,
    gold: int,
    potion_count: int,
    potion_slots: int,
    deck_count: int,
) -> str:
    if purchased or can_buy:
        return ""
    if item_kind == "potion" and potion_count >= potion_slots:
        return "Potion belt full"
    if item_kind == "card_removal" and deck_count <= 0:
        return "No cards to remove"
    if item_price > gold:
        return "Not enough gold"
    return "Unavailable"


def _print_state(state: Any, *, seed: int, message: str) -> None:
    payload = _state_payload(state, seed=seed, message=message)
    print("\n" + "=" * 88)
    print(f"Shop test seed {payload['seed']} | phase {payload['phase']}")
    print(f"Gold: {payload['gold']} | HP: {payload['hp']}/{payload['max_hp']}")
    print(
        f"Potions ({len(payload['potions'])}/{payload['potion_slots']}):",
        _comma_names(payload["potions"]),
    )
    print("Relics: ", _comma_names(payload["relics"]))
    print(f"Message: {payload['message']}")
    print("-" * 88)

    if not payload["items"]:
        print("No active shop. Use r to reset or q to quit.")
        return

    print(f"{'#':>2}  {'Kind':<16} {'Name':<24} {'Rarity':<9} {'Cost':>5} {'Base':>5} Status")
    print("-" * 88)
    for item in payload["items"]:
        status = "sold" if item["purchased"] else "ready"
        if not item["purchased"] and not item["can_buy"]:
            status = "no legal buy"
        print(
            f"{item['slot']:>2}  {item['kind_label']:<16} {item['name']:<24.24} "
            f"{item['rarity']:<9} {item['price']:>5} {item['base_price']:>5} {status}"
        )

    print("-" * 88)
    print(
        "Commands: number/b <slot> buy, d <potion-slot> discard, "
        "t throw Foul Potion, l leave, r reset, q quit"
    )


def _comma_names(items: list[dict[str, Any]]) -> str:
    if not items:
        return "(none)"
    return ", ".join(str(item["name"]) for item in items)


def run_terminal(args: argparse.Namespace) -> None:
    relics = tuple(args.relic or ())
    potions = tuple(args.potion if args.potion is not None else DEFAULT_POTIONS)
    state = create_shop_state(
        seed=args.seed,
        gold=args.gold,
        ascension=args.ascension,
        relics=relics,
        potions=potions,
    )
    message = "Shop generated. Use --relic the_courier to test Courier restocks."

    while True:
        _print_state(state, seed=args.seed, message=message)
        raw_choice = input("shop> ").strip()
        choice = raw_choice.lower()
        if choice in {"q", "quit", "exit"}:
            return
        if choice in {"r", "reset"}:
            state = create_shop_state(
                seed=args.seed,
                gold=args.gold,
                ascension=args.ascension,
                relics=relics,
                potions=potions,
            )
            message = "Shop reset."
            continue
        if choice in {"l", "leave"}:
            state, message = _apply_action(state, ActionType.SHOP_LEAVE)
            continue
        if choice in {"t", "throw"}:
            state, message = _apply_action(state, ActionType.THROW_POTION_AT_MERCHANT)
            continue
        if choice.startswith("d "):
            slot_text = choice.removeprefix("d ").strip()
            if slot_text.isdigit():
                state, message = _apply_action(
                    state,
                    ActionType.DISCARD_POTION,
                    target_id=f"potion:{slot_text}",
                )
                continue

        slot_text = choice.removeprefix("b ").strip()
        if slot_text.isdigit():
            state, message = _buy_from_terminal(state, int(slot_text))
            continue

        message = f"Unknown command: {html.escape(raw_choice)}"


def _buy_from_terminal(state: Any, slot: int) -> tuple[Any, str]:
    if state.shop is None:
        return state, "There is no active shop."
    if slot < 0 or slot >= len(state.shop.items):
        return state, f"No shop slot exists at index {slot}."

    item = state.shop.items[slot]
    if item.kind == "card_removal":
        target_card_id = _prompt_card_removal_target(state)
        if target_card_id is None:
            return state, "Card removal cancelled."
        return _apply_action(
            state,
            ActionType.SHOP_BUY,
            target_id=f"shop:{slot}:remove:{target_card_id}",
        )
    return _apply_action(state, ActionType.SHOP_BUY, target_id=f"shop:{slot}")


def _prompt_card_removal_target(state: Any) -> str | None:
    if not state.master_deck:
        return None
    print("\nChoose a card to remove:")
    for index, card in enumerate(state.master_deck, start=1):
        upgraded = "+" if card.upgraded else ""
        print(f"{index:>2}. {card.name}{upgraded} [{card.instance_id}]")

    raw_choice = input("remove> ").strip()
    if not raw_choice:
        return None
    if raw_choice.isdigit():
        index = int(raw_choice) - 1
        if 0 <= index < len(state.master_deck):
            return str(state.master_deck[index].instance_id)
        return None
    return raw_choice


class ShopTestContext:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.seed = int(args.seed)
        self.lock = threading.Lock()
        self.message = "Shop generated. Use --relic the_courier to test Courier restocks."
        self.state = self._new_state()

    def _new_state(self) -> Any:
        relics = tuple(self.args.relic or ())
        potions = tuple(self.args.potion if self.args.potion is not None else DEFAULT_POTIONS)
        return create_shop_state(
            seed=self.seed,
            gold=self.args.gold,
            ascension=self.args.ascension,
            relics=relics,
            potions=potions,
        )

    def reset(self, *, random_seed: bool = False) -> None:
        if random_seed:
            self.seed = random.randrange(1_000_000_000)
        self.state = self._new_state()
        self.message = "Shop reset."


def run_web(args: argparse.Namespace) -> None:
    context = ShopTestContext(args)
    handler = _make_handler(context)
    server = _bind_server(args.host, args.port, handler)
    host, port = server.server_address
    url = f"http://{host}:{port}"
    print(f"Shop test web UI running at {url}")
    print("Press Ctrl+C to stop it.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping shop test server.")
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
    raise OSError(f"Could not bind a shop test server on {host}:{port}-{port + 19}")


def _make_handler(context: ShopTestContext) -> type[BaseHTTPRequestHandler]:
    class ShopTestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_html(_html_page())
                return
            if path == "/api/state":
                with context.lock:
                    self._send_json(
                        _state_payload(
                            context.state,
                            seed=context.seed,
                            message=context.message,
                        )
                    )
                return
            self.send_error(404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            body = self._read_json_body()
            with context.lock:
                if path == "/api/reset":
                    context.reset(random_seed=bool(body.get("random_seed", False)))
                elif path == "/api/buy":
                    target_id = str(body.get("target_id", ""))
                    context.state, context.message = _apply_action(
                        context.state,
                        ActionType.SHOP_BUY,
                        target_id=target_id,
                    )
                elif path == "/api/throw-foul-potion":
                    context.state, context.message = _apply_action(
                        context.state,
                        ActionType.THROW_POTION_AT_MERCHANT,
                    )
                elif path == "/api/discard-potion":
                    target_id = str(body.get("target_id", ""))
                    context.state, context.message = _apply_action(
                        context.state,
                        ActionType.DISCARD_POTION,
                        target_id=target_id,
                    )
                elif path == "/api/leave":
                    context.state, context.message = _apply_action(
                        context.state,
                        ActionType.SHOP_LEAVE,
                    )
                else:
                    self.send_error(404)
                    return
                self._send_json(
                    _state_payload(
                        context.state,
                        seed=context.seed,
                        message=context.message,
                    )
                )

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                return {}
            if isinstance(data, dict):
                return data
            return {}

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

    return ShopTestHandler


def _html_page() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>sts2sim Shop Test</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f1ea;
      color: #171717;
    }
    body {
      margin: 0;
      min-height: 100vh;
      background: #f4f1ea;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      padding: 22px 28px 16px;
      border-bottom: 1px solid #cec7b6;
      background: #262421;
      color: #faf7ef;
    }
    h1, h2 {
      margin: 0;
      letter-spacing: 0;
    }
    h1 {
      font-size: 24px;
      line-height: 1.1;
    }
    h2 {
      font-size: 15px;
      margin-bottom: 10px;
    }
    main {
      display: grid;
      grid-template-columns: minmax(240px, 320px) minmax(0, 1fr);
      gap: 20px;
      padding: 20px 28px 28px;
    }
    section {
      background: #fffdf7;
      border: 1px solid #d8d0bf;
      border-radius: 8px;
      padding: 16px;
    }
    .side {
      display: grid;
      gap: 16px;
      align-content: start;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .stat {
      border: 1px solid #ded6c8;
      border-radius: 6px;
      padding: 9px 10px;
      background: #f8f4eb;
    }
    .stat b {
      display: block;
      font-size: 12px;
      color: #5f5b52;
      font-weight: 600;
    }
    .stat span {
      font-size: 20px;
      font-weight: 700;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .chip {
      border: 1px solid #b9b09f;
      background: #eee6d8;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
    }
    .potion-belt {
      display: grid;
      gap: 8px;
    }
    .potion-slot {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      border: 1px solid #cfc5b4;
      border-radius: 6px;
      background: #f8f4eb;
      padding: 8px;
    }
    .potion-slot.empty {
      grid-template-columns: 1fr;
      color: #716d65;
      border-style: dashed;
      background: #fffaf0;
    }
    .potion-name {
      min-width: 0;
      font-size: 13px;
      font-weight: 700;
    }
    .potion-name small {
      display: block;
      margin-top: 2px;
      color: #716d65;
      font-size: 11px;
      font-weight: 600;
    }
    .hint {
      margin-top: 8px;
      color: #625d54;
      font-size: 12px;
      line-height: 1.35;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      border-bottom: 1px solid #e4ddcf;
      padding: 9px 8px;
      text-align: left;
      vertical-align: middle;
      font-size: 14px;
    }
    th {
      color: #5d5a52;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .slot { width: 44px; }
    .kind { width: 138px; }
    .rarity { width: 90px; }
    .price { width: 88px; text-align: right; }
    .action { width: 220px; }
    td.price { font-variant-numeric: tabular-nums; }
    .muted { color: #716d65; }
    .sold { color: #8f3d2e; font-weight: 700; }
    .blocked {
      color: #8f3d2e;
      font-size: 12px;
      font-weight: 700;
      line-height: 1.25;
    }
    button, select {
      min-height: 34px;
      border-radius: 6px;
      border: 1px solid #827a6b;
      background: #ffffff;
      color: #161616;
      padding: 6px 10px;
      font: inherit;
    }
    button {
      cursor: pointer;
      font-weight: 700;
    }
    button.primary {
      background: #1f6f54;
      border-color: #1f6f54;
      color: white;
    }
    button.warning {
      background: #8f3d2e;
      border-color: #8f3d2e;
      color: white;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.5;
    }
    .row-actions {
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
    }
    .message {
      padding: 10px 12px;
      border-radius: 6px;
      background: #e5f1ea;
      color: #164330;
      border: 1px solid #b8d1c3;
      font-weight: 650;
    }
    .deck-list {
      display: grid;
      gap: 6px;
      max-height: 240px;
      overflow: auto;
      padding-right: 4px;
    }
    @media (max-width: 860px) {
      header {
        display: block;
      }
      main {
        grid-template-columns: 1fr;
        padding: 16px;
      }
      table {
        min-width: 760px;
      }
      .table-scroll {
        overflow-x: auto;
      }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>sts2sim Shop Test</h1>
      <div id="seed" class="muted"></div>
    </div>
    <div class="row-actions">
      <button id="reset">Reset</button>
      <button id="random-reset">New Seed</button>
    </div>
  </header>
  <main>
    <div class="side">
      <section>
        <h2>State</h2>
        <div class="stats">
          <div class="stat"><b>Gold</b><span id="gold">0</span></div>
          <div class="stat"><b>HP</b><span id="hp">0/0</span></div>
          <div class="stat"><b>Phase</b><span id="phase">shop</span></div>
          <div class="stat"><b>Deck</b><span id="deck-count">0</span></div>
          <div class="stat"><b>Potions</b><span id="potion-count">0/0</span></div>
        </div>
      </section>
      <section>
        <h2>Potions</h2>
        <div id="potions" class="potion-belt"></div>
        <div id="potion-hint" class="hint"></div>
        <p><button id="throw" class="warning">Throw Foul Potion</button></p>
      </section>
      <section>
        <h2>Relics</h2>
        <div id="relics" class="chips"></div>
      </section>
      <section>
        <h2>Deck</h2>
        <div id="deck" class="deck-list"></div>
      </section>
    </div>
    <section>
      <h2>Merchant Inventory</h2>
      <div id="message" class="message"></div>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th class="slot">#</th>
              <th class="kind">Kind</th>
              <th>Name</th>
              <th class="rarity">Rarity</th>
              <th class="price">Cost</th>
              <th class="action">Action</th>
            </tr>
          </thead>
          <tbody id="items"></tbody>
        </table>
      </div>
      <p><button id="leave">Leave Shop</button></p>
    </section>
  </main>
  <script>
    let currentState = null;

    async function api(path, body) {
      const response = await fetch(path, {
        method: body ? "POST" : "GET",
        headers: body ? {"Content-Type": "application/json"} : {},
        body: body ? JSON.stringify(body) : undefined
      });
      currentState = await response.json();
      render(currentState);
    }

    function chip(name) {
      const node = document.createElement("span");
      node.className = "chip";
      node.textContent = name;
      return node;
    }

    function render(data) {
      document.getElementById("seed").textContent = `Seed ${data.seed}`;
      document.getElementById("gold").textContent = data.gold;
      document.getElementById("hp").textContent = `${data.hp}/${data.max_hp}`;
      document.getElementById("phase").textContent = data.phase;
      document.getElementById("deck-count").textContent = data.deck.length;
      document.getElementById("potion-count").textContent =
        `${data.potions.length}/${data.potion_slots}`;
      document.getElementById("message").textContent = data.message;

      renderPotions(data.potions, data.potion_slots, data.potion_slots_open);
      renderChips("relics", data.relics);
      renderDeck(data.deck);
      renderItems(data.items);

      document.getElementById("throw").disabled = !data.can_throw_foul_potion;
      document.getElementById("leave").disabled = !data.can_leave;
    }

    function renderChips(id, items) {
      const root = document.getElementById(id);
      root.replaceChildren();
      if (!items.length) {
        root.append(chip("(none)"));
        return;
      }
      for (const item of items) root.append(chip(item.name));
    }

    function renderPotions(potions, potionSlots, potionSlotsOpen) {
      const root = document.getElementById("potions");
      const hint = document.getElementById("potion-hint");
      root.replaceChildren();
      hint.textContent = potionSlotsOpen > 0
        ? `${potionSlotsOpen} open potion slot${potionSlotsOpen === 1 ? "" : "s"}.`
        : "Potion belt full. Discard a potion here before buying another potion.";
      if (!potionSlots) {
        root.append(emptyPotionSlot("No potion slots"));
        hint.textContent = "This state has no potion slots.";
        return;
      }
      const bySlot = new Map(potions.map((potion) => [potion.slot, potion]));
      for (let slot = 0; slot < potionSlots; slot += 1) {
        const potion = bySlot.get(slot);
        if (!potion) {
          root.append(emptyPotionSlot(`Slot ${slot}: empty`));
          continue;
        }
        root.append(filledPotionSlot(potion));
      }
    }

    function emptyPotionSlot(text) {
      const wrap = document.createElement("div");
      wrap.className = "potion-slot empty";
      wrap.textContent = text;
      return wrap;
    }

    function filledPotionSlot(potion) {
      const wrap = document.createElement("div");
      wrap.className = "potion-slot";
      const label = document.createElement("div");
      label.className = "potion-name";
      label.textContent = potion.name;
      const meta = document.createElement("small");
      meta.textContent = `Slot ${potion.slot}`;
      label.append(meta);

      const button = document.createElement("button");
      button.className = "warning";
      button.textContent = "Discard";
      button.disabled = !potion.can_discard;
      button.onclick = () => api("/api/discard-potion", {target_id: potion.target_id});
      wrap.append(label, button);
      return wrap;
    }

    function renderDeck(cards) {
      const root = document.getElementById("deck");
      root.replaceChildren();
      for (const card of cards) {
        const line = document.createElement("div");
        line.className = "chip";
        line.textContent = `${card.name}${card.upgraded ? "+" : ""} [${card.instance_id}]`;
        root.append(line);
      }
    }

    function renderItems(items) {
      const tbody = document.getElementById("items");
      tbody.replaceChildren();
      for (const item of items) {
        const row = document.createElement("tr");
        row.append(cell(item.slot));
        row.append(cell(item.kind_label));
        row.append(cell(item.name));
        row.append(cell(item.rarity || "-"));
        const price = cell(
          item.base_price === item.price ? item.price : `${item.price} (${item.base_price})`
        );
        price.className = "price";
        row.append(price);
        row.append(actionCell(item));
        tbody.append(row);
      }
    }

    function cell(text) {
      const td = document.createElement("td");
      td.textContent = text;
      return td;
    }

    function actionCell(item) {
      const td = document.createElement("td");
      const wrap = document.createElement("div");
      wrap.className = "row-actions";
      if (item.purchased) {
        const sold = document.createElement("span");
        sold.className = "sold";
        sold.textContent = "Sold";
        wrap.append(sold);
      } else if (item.kind === "card_removal") {
        const select = document.createElement("select");
        for (const card of currentState.deck) {
          const option = document.createElement("option");
          option.value = card.instance_id;
          option.textContent = `${card.name} [${card.instance_id}]`;
          select.append(option);
        }
        const button = document.createElement("button");
        button.className = "primary";
        button.textContent = "Remove";
        button.disabled = !item.can_buy || !currentState.deck.length;
        button.onclick = () => api(
          "/api/buy",
          {target_id: `${item.target_id}:remove:${select.value}`}
        );
        wrap.append(select, button);
      } else {
        const button = document.createElement("button");
        button.className = "primary";
        button.textContent = item.blocked_reason || "Buy";
        button.disabled = !item.can_buy;
        button.onclick = () => api("/api/buy", {target_id: item.target_id});
        wrap.append(button);
        if (!item.can_buy && item.blocked_reason) {
          const reason = document.createElement("span");
          reason.className = "blocked";
          reason.textContent = item.kind === "potion" && item.blocked_reason === "Potion belt full"
            ? "Discard a potion first"
            : item.blocked_reason;
          wrap.append(reason);
        }
      }
      td.append(wrap);
      return td;
    }

    document.getElementById("reset").onclick = () => api("/api/reset", {});
    document.getElementById("random-reset").onclick = () => api("/api/reset", {random_seed: true});
    document.getElementById("throw").onclick = () => api("/api/throw-foul-potion", {});
    document.getElementById("leave").onclick = () => api("/api/leave", {});
    api("/api/state");
  </script>
</body>
</html>"""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive shop tester for the headless sts2sim engine."
    )
    parser.add_argument("positional_seed", nargs="?", type=int, help="Deterministic shop seed.")
    parser.add_argument("--seed", dest="seed_option", type=int, help="Deterministic shop seed.")
    parser.add_argument("--gold", type=int, default=350, help="Starting gold.")
    parser.add_argument("--ascension", type=int, default=0, help="Ascension level.")
    parser.add_argument("--relic", action="append", help="Starting relic id. May be repeated.")
    parser.add_argument("--potion", action="append", help="Starting potion id. May be repeated.")
    parser.add_argument("--web", action="store_true", help="Open a small local web UI.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Web server host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Web server port.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window.")
    args = parser.parse_args(argv)
    if args.seed_option is not None:
        args.seed = args.seed_option
    elif args.positional_seed is not None:
        args.seed = args.positional_seed
    else:
        args.seed = random.randrange(1_000_000_000)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.web:
        run_web(args)
    else:
        run_terminal(args)


if __name__ == "__main__":
    main()
