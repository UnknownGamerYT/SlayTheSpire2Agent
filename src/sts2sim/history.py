"""Human-readable run history for simulator-driven agents."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from html import escape as html_escape
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sts2sim.api import serialize
from sts2sim.engine.serialization import state_digest


class HistoryModel(BaseModel):
    """Base model for history payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class RunHistoryStep(HistoryModel):
    """One readable simulator transition."""

    step_index: int
    phase_before: str
    phase_after: str
    action: dict[str, Any]
    action_summary: str
    state_hash_before: str
    state_hash_after: str
    events: tuple[dict[str, Any], ...] = ()
    context_before: dict[str, Any] = Field(default_factory=dict)
    context_after: dict[str, Any] = Field(default_factory=dict)
    reward: float | None = None
    decision: dict[str, Any] | None = None


class RunHistory(HistoryModel):
    """A complete readable timeline for one simulator run."""

    seed: int | str
    character_id: str
    ascension: int
    policy: str
    initial: dict[str, Any]
    final: dict[str, Any]
    steps: tuple[RunHistoryStep, ...] = ()
    summary: dict[str, Any] = Field(default_factory=dict)


def start_run_history(state: Any, *, policy: str) -> RunHistory:
    """Create an empty history from the initial simulator state."""

    payload = serialize(state)
    initial = summarize_payload(payload)
    return RunHistory(
        seed=_seed(payload),
        character_id=str(payload.get("character_id", "")),
        ascension=_int(payload.get("ascension")),
        policy=policy,
        initial=initial,
        final=initial,
        summary=_history_summary((), initial),
    )


def record_history_step(
    *,
    step_index: int,
    before_state: Any,
    action: Any,
    after_state: Any,
    reward: float | None = None,
    decision: Mapping[str, Any] | None = None,
) -> RunHistoryStep:
    """Build a readable history entry for one already-applied transition."""

    before_payload = serialize(before_state)
    after_payload = serialize(after_state)
    action_payload = action_to_payload(action)
    events = _latest_replay_events(after_state)
    return RunHistoryStep(
        step_index=step_index,
        phase_before=str(before_payload.get("phase", "")),
        phase_after=str(after_payload.get("phase", "")),
        action=action_payload,
        action_summary=summarize_action(before_payload, action_payload),
        state_hash_before=state_digest(before_state),
        state_hash_after=state_digest(after_state),
        events=events,
        context_before=summarize_payload(before_payload),
        context_after=summarize_payload(after_payload),
        reward=None if reward is None else round(float(reward), 6),
        decision=dict(decision) if decision is not None else None,
    )


def append_history_step(history: RunHistory, step: RunHistoryStep, final_state: Any) -> RunHistory:
    """Return a history with ``step`` appended and final/summary refreshed."""

    steps = history.steps + (step,)
    final = summarize_state(final_state)
    return history.model_copy(
        update={
            "steps": steps,
            "final": final,
            "summary": _history_summary(steps, final),
        }
    )


def summarize_state(state: Any) -> dict[str, Any]:
    """Return the compact readable context used by history entries."""

    return summarize_payload(serialize(state))


def summarize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact, JSON-friendly context from a serialized run state."""

    player = _mapping(payload.get("player"))
    return {
        "phase": str(payload.get("phase", "")),
        "act": _int(payload.get("act")),
        "floor": _int(payload.get("floor")),
        "player": {
            "hp": _int(player.get("hp")),
            "max_hp": _int(player.get("max_hp")),
            "block": _int(player.get("block")),
            "energy": _int(player.get("energy")),
            "max_energy": _int(player.get("max_energy")),
            "gold": _int(player.get("gold")),
            "statuses": dict(_mapping(player.get("statuses"))),
            "resources": dict(_mapping(player.get("resources"))),
            "relics": list(_sequence(payload.get("relics"))),
            "potions": list(_sequence(payload.get("potions"))),
            "deck_count": len(_sequence(payload.get("master_deck"))),
        },
        "map": _map_summary(_mapping(payload.get("map"))),
        "ancient": _ancient_summary(_mapping(payload.get("ancient"))),
        "event": _event_summary(_mapping(payload.get("event"))),
        "shop": _shop_summary(_mapping(payload.get("shop"))),
        "reward": _reward_summary(_mapping(payload.get("reward"))),
        "combat": _combat_summary(_mapping(payload.get("combat"))),
        "room_history": list(_sequence(payload.get("room_history"))),
        "flags": _public_flags(_mapping(payload.get("flags"))),
    }


def summarize_action(state_payload: Mapping[str, Any], action: Mapping[str, Any]) -> str:
    """Return a short human-readable action description."""

    action_type = str(action.get("type", "unknown"))
    target_id = _optional_str(action.get("target_id"))
    card_instance_id = _optional_str(action.get("card_instance_id"))

    if action_type == "choose_ancient":
        option = _find_ancient_option(_mapping(state_payload.get("ancient")), target_id)
        if option:
            return (
                f"Choose ancient option {option.get('name', target_id)} "
                f"for relic {option.get('relic_id', 'unknown')}"
            )
        return f"Choose ancient option {target_id}"

    if action_type == "choose_node":
        node = _find_map_node(_mapping(state_payload.get("map")), target_id)
        if node:
            kind = str(node.get("kind", "node"))
            return (
                f"Choose map node {target_id} "
                f"({kind}, floor {_int(node.get('floor'))}, lane {_int(node.get('lane'))})"
            )
        return f"Choose map node {target_id}"

    if action_type == "choose_event":
        option = _find_event_option(_mapping(state_payload.get("event")), target_id)
        if option:
            return f"Choose event option {option.get('title', target_id)}"
        return f"Choose event option {target_id}"

    if action_type == "play_card":
        card = _find_card_by_instance_id(state_payload, card_instance_id)
        target = _target_name(state_payload, target_id)
        card_name = (
            str(card.get("name", card.get("card_id", card_instance_id)))
            if card
            else card_instance_id
        )
        return f"Play {card_name} -> {target}" if target else f"Play {card_name}"

    if action_type == "end_turn":
        return "End turn"

    if action_type == "choose_card":
        card = _find_card_by_instance_id(state_payload, card_instance_id)
        name = (
            str(card.get("name", card.get("card_id", card_instance_id)))
            if card
            else card_instance_id
        )
        return f"Choose card {name}"

    if action_type == "discard_card":
        card = _find_card_by_instance_id(state_payload, card_instance_id)
        name = (
            str(card.get("name", card.get("card_id", card_instance_id)))
            if card
            else card_instance_id
        )
        return f"Discard card {name}"

    if action_type == "exhaust_card":
        card = _find_card_by_instance_id(state_payload, card_instance_id)
        name = (
            str(card.get("name", card.get("card_id", card_instance_id)))
            if card
            else card_instance_id
        )
        return f"Exhaust card {name}"

    if action_type == "use_potion":
        payload = _mapping(action.get("payload"))
        potion_id = str(payload.get("potion_id", "potion"))
        target = _target_name(state_payload, target_id)
        return f"Use potion {potion_id} -> {target}" if target else f"Use potion {potion_id}"

    if action_type == "discard_potion":
        potion_id = _potion_name_for_slot(state_payload, target_id)
        return f"Discard potion {potion_id}"

    if action_type == "shop_buy":
        item = _find_shop_item(_mapping(state_payload.get("shop")), target_id)
        if item:
            return (
                f"Buy {item.get('kind', 'shop item')} {item.get('item_id', target_id)} "
                f"for {_int(item.get('price'))} gold"
            )
        return f"Buy shop item {target_id}"

    if action_type == "shop_leave":
        return "Leave shop"

    if action_type == "throw_potion_at_merchant":
        return "Throw Foul Potion at the merchant"

    if action_type in {"rest", "recall", "dig", "lift"}:
        return action_type.replace("_", " ").title()

    if action_type == "smith":
        card = _find_card_by_instance_id(state_payload, target_id)
        name = str(card.get("name", card.get("card_id", target_id))) if card else target_id
        return f"Smith card {name}"

    if action_type == "toke":
        card = _find_card_by_instance_id(state_payload, target_id)
        name = str(card.get("name", card.get("card_id", target_id))) if card else target_id
        return f"Remove card {name}"

    if action_type.startswith("take_reward"):
        return _reward_action_summary(_mapping(state_payload.get("reward")), action_type, target_id)

    if action_type == "proceed":
        return f"Proceed from {state_payload.get('phase', 'current phase')}"

    return action_type.replace("_", " ").title()


def action_to_payload(action: Any) -> dict[str, Any]:
    """Return a JSON-friendly engine action payload."""

    model_dump = getattr(action, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="json", exclude_none=True))
    if isinstance(action, Mapping):
        return dict(action)
    return {"type": str(action)}


def write_run_history(history: RunHistory | Mapping[str, Any], path: Path | str) -> None:
    """Write history as formatted JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_history_payload(history), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_run_history_html(history: RunHistory | Mapping[str, Any], path: Path | str) -> None:
    """Write a standalone readable HTML run timeline."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(run_history_html(history), encoding="utf-8")


def write_run_history_map_text(history: RunHistory | Mapping[str, Any], path: Path | str) -> None:
    """Write a compact text map with the chosen path marked."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(run_history_map_text(history) + "\n", encoding="utf-8")


def run_history_map_text(history: RunHistory | Mapping[str, Any]) -> str:
    """Return a compact floor-by-floor map with visited nodes underlined by marker."""

    payload = _history_payload(history)
    map_payload = _largest_map_payload(payload)
    if not map_payload:
        return "No map was generated for this history."

    nodes = tuple(_mapping(node) for node in _sequence(map_payload.get("nodes")))
    edges = tuple(_mapping(edge) for edge in _sequence(map_payload.get("edges")))
    chosen_ids = set(_chosen_node_ids(payload))
    completed_ids = {
        str(node_id)
        for node_id in _sequence(map_payload.get("completed_node_ids"))
        if str(node_id)
    }
    current_id = _optional_str(map_payload.get("current_node_id"))
    by_floor: dict[int, list[Mapping[str, Any]]] = {}
    for node in nodes:
        by_floor.setdefault(_int(node.get("floor")), []).append(node)

    lines = [
        f"Act {_int(map_payload.get('act'))} map",
        "Legend: * visited/chosen, _node_ chosen path marker, -> reachable edges",
    ]
    for floor in sorted(by_floor):
        lane_chunks: list[str] = []
        for node in sorted(by_floor[floor], key=lambda item: _int(item.get("lane"))):
            node_id = str(node.get("node_id", ""))
            kind = _room_letter(str(node.get("kind", "")))
            marker = "*" if node_id in chosen_ids or node_id in completed_ids else " "
            current = "!" if current_id == node_id else " "
            label = f"{kind}{marker}{current}:{node_id}"
            if node_id in chosen_ids:
                label = f"_{label}_"
            lane_chunks.append(f"L{_int(node.get('lane'))}:{label}")
        lines.append(f"F{floor:02d}  " + "  ".join(lane_chunks))
        floor_edges = [
            f"{edge.get('from_id')}->{edge.get('to_id')}"
            for edge in edges
            if _node_floor(nodes, str(edge.get("from_id"))) == floor
        ]
        if floor_edges:
            lines.append("      " + ", ".join(floor_edges[:12]))
    return "\n".join(lines)


def run_history_html(
    history: RunHistory | Mapping[str, Any],
    *,
    title: str = "Run History",
) -> str:
    """Render a readable standalone HTML report for a simulator run history."""

    payload = _history_payload(history)
    summary = _mapping(payload.get("summary"))
    initial = _mapping(payload.get("initial"))
    final = _mapping(payload.get("final"))
    steps = tuple(_mapping(step) for step in _sequence(payload.get("steps")))
    safe_title = html_escape(title)
    overview = {
        "seed": payload.get("seed", ""),
        "character": payload.get("character_id", ""),
        "ascension": payload.get("ascension", 0),
        "policy": payload.get("policy", ""),
        "steps": summary.get("steps_taken", len(steps)),
        "final": (
            f"act {final.get('act', 0)} floor {final.get('floor', 0)} "
            f"{final.get('phase', '')}"
        ),
        "cards played": summary.get("cards_played", 0),
        "nodes chosen": summary.get("nodes_chosen", 0),
        "rewards taken": summary.get("rewards_taken", 0),
    }
    overview_items = "\n".join(
        "<div>"
        f"<strong>{html_escape(str(key).title())}</strong>"
        f"<span>{html_escape(str(value))}</span>"
        "</div>"
        for key, value in overview.items()
    )
    timeline = "\n".join(_step_html(step) for step in steps)
    if not timeline:
        timeline = '<p class="muted">No steps were recorded.</p>'
    map_text = html_escape(run_history_map_text(payload))
    reward_overview = _context_panel_html("Initial", initial) + _context_panel_html("Final", final)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #62707a;
      --line: #d9e0e5;
      --panel: #fff;
      --paper: #f5f7f8;
      --accent: #2459a6;
      --good: #177245;
      --warn: #a26114;
      --bad: #b3332f;
    }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font: 14px/1.45 system-ui, sans-serif;
    }}
    main {{
      width: min(1280px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }}
    h1 {{ margin: 0 0 4px; font-size: 28px; }}
    h2 {{ margin: 28px 0 12px; font-size: 19px; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; }}
    .muted {{ color: var(--muted); }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .summary div, article, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .summary strong {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .summary span {{ display: block; margin-top: 4px; font-weight: 700; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 12px;
      align-items: start;
    }}
    article {{ margin-bottom: 12px; }}
    .step-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 10px;
    }}
    .badge {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      background: #f8fafb;
    }}
    .narrative {{ margin: 0 0 10px; padding-left: 18px; }}
    .narrative li {{ margin: 3px 0; }}
    pre {{
      overflow: auto;
      background: #111820;
      color: #edf4f7;
      border-radius: 8px;
      padding: 12px;
      line-height: 1.35;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
    }}
    th, td {{
      padding: 6px 7px;
      border-bottom: 1px solid #e7ecef;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }}
    .cards, .events {{ display: flex; flex-wrap: wrap; gap: 5px; }}
    .pill {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      background: #f9fbfc;
      font-size: 12px;
    }}
    .diff-good {{ color: var(--good); font-weight: 700; }}
    .diff-bad {{ color: var(--bad); font-weight: 700; }}
  </style>
</head>
<body>
<main>
  <h1>{safe_title}</h1>
  <p class="muted">
    Step-by-step simulator trace with map path, combat state, rewards, shops,
    events, and engine replay events.
  </p>
  <section class="summary">{overview_items}</section>
  <h2>Map Path</h2>
  <pre>{map_text}</pre>
  <h2>Run Start And Finish</h2>
  <div class="grid">{reward_overview}</div>
  <h2>Timeline</h2>
  {timeline}
</main>
</body>
</html>
"""


def _history_payload(history: RunHistory | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(history, RunHistory):
        return history.model_dump(mode="json")
    return history


def _step_html(step: Mapping[str, Any]) -> str:
    context_before = _mapping(step.get("context_before"))
    context_after = _mapping(step.get("context_after"))
    events = tuple(_mapping(event) for event in _sequence(step.get("events")))
    decision = _mapping(step.get("decision"))
    meta = [
        f"phase {step.get('phase_before', '')} -> {step.get('phase_after', '')}",
        f"reward {step.get('reward', 0)}",
        f"before {str(step.get('state_hash_before', ''))[:10]}",
        f"after {str(step.get('state_hash_after', ''))[:10]}",
    ]
    meta_html = "".join(f'<span class="badge">{html_escape(item)}</span>' for item in meta)
    narrative = "".join(
        f"<li>{html_escape(item)}</li>"
        for item in _step_narrative(step, context_before, context_after)
    )
    if not narrative:
        narrative = "<li>No detailed state context was available.</li>"
    before_panel = _context_panel_html("Before", context_before)
    after_panel = _context_panel_html("After", context_after)
    events_html = _events_html(events)
    decision_html = _decision_html(decision)
    return (
        "<article>"
        f"<h3>Step {_int(step.get('step_index'))}: "
        f"{html_escape(str(step.get('action_summary', 'Action')))}</h3>"
        f'<div class="step-meta">{meta_html}</div>'
        f'<ul class="narrative">{narrative}</ul>'
        f'<div class="grid">{before_panel}{after_panel}</div>'
        f"{events_html}{decision_html}"
        "</article>"
    )


def _step_narrative(
    step: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[str, ...]:
    lines = [
        f"Action selected: {step.get('action_summary', 'unknown')}.",
    ]
    before_combat = _mapping(before.get("combat"))
    after_combat = _mapping(after.get("combat"))
    if before_combat:
        before_player = _mapping(before_combat.get("player"))
        lines.append(
            "Turn "
            f"{_int(before_combat.get('turn'))} start: "
            f"HP {_int(before_player.get('hp'))}/{_int(before_player.get('max_hp'))}, "
            f"block {_int(before_player.get('block'))}, "
            f"energy {_int(before_player.get('energy'))}, "
            f"hand {_card_names(before_combat.get('hand'))}."
        )
        intents = _monster_intents(before_combat)
        if intents:
            lines.append(f"Enemy intentions: {intents}.")
    if after_combat:
        after_player = _mapping(after_combat.get("player"))
        lines.append(
            "After action: "
            f"HP {_int(after_player.get('hp'))}/{_int(after_player.get('max_hp'))}, "
            f"block {_int(after_player.get('block'))}, "
            f"energy {_int(after_player.get('energy'))}, "
            f"hand {_card_names(after_combat.get('hand'))}."
        )
        lines.append(
            "Piles after action: "
            f"draw {_zone_count(after_combat, 'draw_pile')}, "
            f"discard {_zone_count(after_combat, 'discard_pile')}, "
            f"exhaust {_zone_count(after_combat, 'exhaust_pile')}."
        )
        after_intents = _monster_intents(after_combat)
        if after_intents:
            lines.append(f"Enemy state after action: {after_intents}.")
    if not before_combat and after_combat:
        lines.append(
            "Combat started; opening hand and enemy intents are visible in the after panel."
        )
    if before_combat and not after_combat:
        lines.append("Combat ended and rewards or the next room state opened.")

    before_reward = _mapping(before.get("reward"))
    after_reward = _mapping(after.get("reward"))
    if before_reward:
        lines.append(f"Reward screen before action: {_reward_line(before_reward)}.")
    if after_reward and after_reward != before_reward:
        lines.append(f"Reward screen after action: {_reward_line(after_reward)}.")

    before_shop = _mapping(before.get("shop"))
    after_shop = _mapping(after.get("shop"))
    if before_shop:
        lines.append(f"Shop before action: {_shop_line(before_shop)}.")
    if after_shop and after_shop != before_shop:
        lines.append(f"Shop after action: {_shop_line(after_shop)}.")

    before_event = _mapping(before.get("event"))
    after_event = _mapping(after.get("event"))
    if before_event:
        lines.append(f"Event before action: {_event_line(before_event)}.")
    if after_event and after_event != before_event:
        lines.append(f"Event after action: {_event_line(after_event)}.")

    before_map = _mapping(before.get("map"))
    after_map = _mapping(after.get("map"))
    if before_map and str(_mapping(step.get("action")).get("type")) == "choose_node":
        lines.append(f"Map choice opened: {_map_line(before_map)}.")
    if after_map and after_map != before_map:
        lines.append(f"Map after action: {_map_line(after_map)}.")
    return tuple(lines)


def _context_panel_html(title: str, context: Mapping[str, Any]) -> str:
    if not context:
        return (
            f'<section class="panel"><h3>{html_escape(title)}</h3>'
            '<p class="muted">Empty.</p></section>'
        )
    player = _mapping(context.get("player"))
    combat = _mapping(context.get("combat"))
    reward = _mapping(context.get("reward"))
    shop = _mapping(context.get("shop"))
    event = _mapping(context.get("event"))
    game_map = _mapping(context.get("map"))
    lines = [
        f"<p><strong>Phase:</strong> {html_escape(str(context.get('phase', '')))} "
        f"<strong>Act/Floor:</strong> {_int(context.get('act'))}/{_int(context.get('floor'))}</p>",
    ]
    if player:
        potions = ", ".join(str(item) for item in _sequence(player.get("potions"))) or "-"
        relics = ", ".join(str(item) for item in _sequence(player.get("relics"))) or "-"
        lines.append(
            "<p><strong>Player:</strong> "
            f"HP {_int(player.get('hp'))}/{_int(player.get('max_hp'))}, "
            f"block {_int(player.get('block'))}, energy {_int(player.get('energy'))}, "
            f"gold {_int(player.get('gold'))}, "
            f"potions {html_escape(potions)}, "
            f"relics {html_escape(relics)}"
            "</p>"
        )
    if combat:
        lines.append(_combat_panel_html(combat))
    if reward:
        lines.append(f"<p><strong>Reward:</strong> {html_escape(_reward_line(reward))}</p>")
    if shop:
        lines.append(f"<p><strong>Shop:</strong> {html_escape(_shop_line(shop))}</p>")
    if event:
        lines.append(f"<p><strong>Event:</strong> {html_escape(_event_line(event))}</p>")
    if game_map:
        lines.append(f"<p><strong>Map:</strong> {html_escape(_map_line(game_map))}</p>")
    flags = _mapping(context.get("flags"))
    if flags:
        lines.append(f"<p><strong>Flags:</strong> {html_escape(_compact_json(flags))}</p>")
    return f'<section class="panel"><h3>{html_escape(title)}</h3>{"".join(lines)}</section>'


def _combat_panel_html(combat: Mapping[str, Any]) -> str:
    monsters = tuple(_mapping(monster) for monster in _sequence(combat.get("monsters")))
    monster_rows = "".join(
        "<tr>"
        f"<td>{html_escape(str(monster.get('name') or monster.get('monster_id')))}</td>"
        f"<td>{_int(monster.get('hp'))}/{_int(monster.get('max_hp'))}</td>"
        f"<td>{_int(monster.get('block'))}</td>"
        f"<td>{html_escape(str(monster.get('intent') or '-'))} "
        f"{_int(monster.get('intent_damage'))}x{max(1, _int(monster.get('hit_count')))}</td>"
        f"<td>{html_escape(_compact_json(_mapping(monster.get('statuses'))))}</td>"
        "</tr>"
        for monster in monsters
    )
    if not monster_rows:
        monster_rows = '<tr><td colspan="5" class="muted">No monsters.</td></tr>'
    zones = "".join(
        "<p>"
        f"<strong>{html_escape(label)}:</strong> "
        f'<span class="cards">{_card_pills(combat.get(zone))}</span>'
        "</p>"
        for label, zone in (
            ("Hand", "hand"),
            ("Draw", "draw_pile"),
            ("Discard", "discard_pile"),
            ("Exhaust", "exhaust_pile"),
        )
    )
    return (
        f"<p><strong>Combat Turn:</strong> {_int(combat.get('turn'))}</p>"
        "<table><thead><tr><th>Enemy</th><th>HP</th><th>Block</th>"
        "<th>Intent</th><th>Status</th></tr></thead>"
        f"<tbody>{monster_rows}</tbody></table>"
        f"{zones}"
    )


def _events_html(events: Sequence[Mapping[str, Any]]) -> str:
    if not events:
        return ""
    event_items = "".join(
        '<span class="pill">'
        f"{html_escape(str(event.get('kind', event.get('value', 'event'))))}: "
        f"{html_escape(_compact_json(_mapping(event.get('metadata'))))}"
        "</span>"
        for event in events
    )
    return f'<h3>Engine Events</h3><div class="events">{event_items}</div>'


def _decision_html(decision: Mapping[str, Any]) -> str:
    if not decision:
        return ""
    cells = "".join(
        "<tr>"
        f"<th>{html_escape(str(key).replace('_', ' ').title())}</th>"
        f"<td>{html_escape(str(value))}</td>"
        "</tr>"
        for key, value in sorted(decision.items())
    )
    return f"<h3>Policy Output</h3><table><tbody>{cells}</tbody></table>"


def _largest_map_payload(history_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    maps: list[Mapping[str, Any]] = []
    for context_key in ("initial", "final"):
        game_map = _mapping(_mapping(history_payload.get(context_key)).get("map"))
        if game_map:
            maps.append(game_map)
    for step in (_mapping(raw) for raw in _sequence(history_payload.get("steps"))):
        for context_key in ("context_before", "context_after"):
            game_map = _mapping(_mapping(step.get(context_key)).get("map"))
            if game_map:
                maps.append(game_map)
    if not maps:
        return {}
    return max(maps, key=lambda game_map: len(_sequence(game_map.get("nodes"))))


def _chosen_node_ids(history_payload: Mapping[str, Any]) -> tuple[str, ...]:
    chosen: list[str] = []
    for step in (_mapping(raw) for raw in _sequence(history_payload.get("steps"))):
        action = _mapping(step.get("action"))
        if str(action.get("type")) == "choose_node":
            target_id = _optional_str(action.get("target_id"))
            if target_id:
                chosen.append(target_id)
    final_map = _mapping(_mapping(history_payload.get("final")).get("map"))
    chosen.extend(str(item) for item in _sequence(final_map.get("completed_node_ids")))
    return tuple(dict.fromkeys(chosen))


def _node_floor(nodes: Sequence[Mapping[str, Any]], node_id: str) -> int:
    for node in nodes:
        if str(node.get("node_id")) == node_id:
            return _int(node.get("floor"))
    return -1


def _room_letter(kind: str) -> str:
    normalized = kind.lower()
    return {
        "start": "S",
        "monster": "M",
        "elite": "L",
        "boss": "B",
        "rest": "F",
        "shop": "$",
        "event": "?",
        "treasure": "T",
    }.get(normalized, (kind[:1] or "?").upper())


def _card_names(cards: object) -> str:
    names = [
        str(_mapping(card).get("name") or _mapping(card).get("card_id") or "")
        for card in _sequence(cards)
    ]
    return ", ".join(name for name in names if name) or "-"


def _card_pills(cards: object) -> str:
    labels = [
        str(_mapping(card).get("name") or _mapping(card).get("card_id") or "")
        for card in _sequence(cards)
    ]
    if not labels:
        return '<span class="pill">empty</span>'
    return "".join(f'<span class="pill">{html_escape(label)}</span>' for label in labels if label)


def _monster_intents(combat: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for monster in (_mapping(raw) for raw in _sequence(combat.get("monsters"))):
        name = str(monster.get("name") or monster.get("monster_id") or "enemy")
        statuses = _mapping(monster.get("statuses"))
        status_text = f", statuses {_compact_json(statuses)}" if statuses else ""
        parts.append(
            f"{name} HP {_int(monster.get('hp'))}/{_int(monster.get('max_hp'))}, "
            f"block {_int(monster.get('block'))}, "
            f"intent {monster.get('intent') or '-'} "
            f"{_int(monster.get('intent_damage'))}x{max(1, _int(monster.get('hit_count')))}"
            f"{status_text}"
        )
    return "; ".join(parts)


def _reward_line(reward: Mapping[str, Any]) -> str:
    parts = [
        f"source {reward.get('source', '')}",
        f"gold {_int(reward.get('gold'))}{' claimed' if reward.get('gold_claimed') else ''}",
    ]
    relics = list(_sequence(reward.get("relic_ids")))
    relic_id = _optional_str(reward.get("relic_id"))
    if relic_id:
        relics.insert(0, relic_id)
    if relics:
        parts.append("relics " + ", ".join(str(item) for item in relics))
    card_groups = _sequence(reward.get("card_option_groups"))
    card_options = _sequence(reward.get("card_options"))
    fixed_cards = _sequence(reward.get("card_ids"))
    if card_groups:
        parts.append(
            "card groups "
            + " | ".join(
                "[" + ", ".join(str(item) for item in _sequence(group)) + "]"
                for group in card_groups
            )
        )
    if card_options:
        parts.append("cards " + ", ".join(str(item) for item in card_options))
    if fixed_cards:
        parts.append("fixed cards " + ", ".join(str(item) for item in fixed_cards))
    potions = list(_sequence(reward.get("potion_ids")))
    potion_id = _optional_str(reward.get("potion_id"))
    if potion_id:
        potions.insert(0, potion_id)
    if potions:
        parts.append("potions " + ", ".join(str(item) for item in potions))
    claimed: list[str] = []
    claimed.extend(str(item) for item in _sequence(reward.get("claimed_relic_ids")))
    claimed.extend(f"potion#{item}" for item in _sequence(reward.get("claimed_potion_indices")))
    if claimed:
        parts.append("claimed " + ", ".join(claimed))
    return "; ".join(parts)


def _shop_line(shop: Mapping[str, Any]) -> str:
    items = [
        f"{item.get('slot_id')}={item.get('kind')}:{item.get('item_id')} "
        f"{_int(item.get('price'))}g{' sold' if item.get('purchased') else ''}"
        for item in (_mapping(raw) for raw in _sequence(shop.get("items")))
    ]
    return f"{len(items)} items; " + "; ".join(items[:8])


def _event_line(event: Mapping[str, Any]) -> str:
    options = [
        (
            f"{option.get('option_id')}:{option.get('title')}"
            f"{' disabled' if option.get('disabled') else ''}"
        )
        for option in (_mapping(raw) for raw in _sequence(event.get("options")))
    ]
    return (
        f"{event.get('name') or event.get('event_id')} page {event.get('page_id', '')}; "
        + "; ".join(options)
    )


def _map_line(game_map: Mapping[str, Any]) -> str:
    reachable = [
        f"{node.get('node_id')}({node.get('kind')},F{node.get('floor')},L{node.get('lane')})"
        for node in (_mapping(raw) for raw in _sequence(game_map.get("reachable")))
    ]
    return (
        f"current {game_map.get('current_node_id')}, "
        f"completed {len(_sequence(game_map.get('completed_node_ids')))}, "
        f"nodes {len(_sequence(game_map.get('nodes')))}, "
        f"reachable {', '.join(reachable) or '-'}"
    )


def _zone_count(combat: Mapping[str, Any], zone: str) -> int:
    return len(_sequence(combat.get(zone)))


def _compact_json(value: Mapping[str, Any]) -> str:
    if not value:
        return "{}"
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def _history_summary(
    steps: Sequence[RunHistoryStep],
    final: Mapping[str, Any],
) -> dict[str, Any]:
    action_types = [step.action.get("type") for step in steps]
    events = [event.get("kind") for step in steps for event in step.events]
    return {
        "steps_taken": len(steps),
        "final_phase": final.get("phase"),
        "final_act": final.get("act"),
        "final_floor": final.get("floor"),
        "cards_played": sum(1 for action_type in action_types if action_type == "play_card"),
        "nodes_chosen": sum(1 for action_type in action_types if action_type == "choose_node"),
        "turns_ended": sum(1 for action_type in action_types if action_type == "end_turn"),
        "rewards_taken": sum(
            1 for action_type in action_types if str(action_type).startswith("take_reward")
        ),
        "event_count": len(events),
        "event_kinds": sorted({str(kind) for kind in events if kind is not None}),
    }


def _latest_replay_events(after_state: Any) -> tuple[dict[str, Any], ...]:
    replay_log = getattr(after_state, "replay_log", ())
    if not replay_log:
        return ()
    latest = replay_log[-1]
    raw_events = getattr(latest, "events", ())
    return tuple(_model_payload(event) for event in raw_events)


def _model_payload(value: Any) -> dict[str, Any]:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return dict(model_dump(mode="json", exclude_none=True))
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": value}


def _map_summary(game_map: Mapping[str, Any]) -> dict[str, Any]:
    if not game_map:
        return {}
    nodes = [_mapping(node) for node in _sequence(game_map.get("nodes"))]
    node_by_id = {str(node.get("node_id")): node for node in nodes}
    current_id = _optional_str(game_map.get("current_node_id"))
    edges = [_mapping(edge) for edge in _sequence(game_map.get("edges"))]
    reachable_ids = [
        str(edge.get("to_id"))
        for edge in edges
        if current_id is not None and str(edge.get("from_id")) == current_id
    ]
    return {
        "act": _int(game_map.get("act")),
        "current_node_id": current_id,
        "completed_node_ids": list(_sequence(game_map.get("completed_node_ids"))),
        "boss_node_id": _optional_str(game_map.get("boss_node_id")),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": [_node_summary(node) for node in nodes],
        "edges": [
            {
                "from_id": str(edge.get("from_id", "")),
                "to_id": str(edge.get("to_id", "")),
            }
            for edge in edges
        ],
        "reachable": [
            _node_summary(node_by_id[node_id])
            for node_id in reachable_ids
            if node_id in node_by_id
        ],
    }


def _node_summary(node: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "node_id": str(node.get("node_id", "")),
        "kind": str(node.get("kind", "")),
        "floor": _int(node.get("floor")),
        "lane": _int(node.get("lane")),
    }


def _ancient_summary(ancient: Mapping[str, Any]) -> dict[str, Any]:
    if not ancient:
        return {}
    return {
        "act": _int(ancient.get("act")),
        "ancient_id": str(ancient.get("ancient_id", "")),
        "chosen_option_ids": list(_sequence(ancient.get("chosen_option_ids"))),
        "options": [
            {
                "option_id": str(option.get("option_id", "")),
                "name": str(option.get("name", "")),
                "kind": str(option.get("kind", "")),
                "relic_id": str(option.get("relic_id", "")),
            }
            for option in (_mapping(item) for item in _sequence(ancient.get("options")))
        ],
    }


def _event_summary(event: Mapping[str, Any]) -> dict[str, Any]:
    if not event:
        return {}
    return {
        "event_id": str(event.get("event_id", "")),
        "name": str(event.get("name", "")),
        "page_id": str(event.get("page_id", "")),
        "resolved_option_id": _optional_str(event.get("resolved_option_id")),
        "options": [
            {
                "option_id": str(option.get("option_id", "")),
                "title": str(option.get("title", "")),
                "disabled": bool(option.get("disabled", False)),
            }
            for option in (_mapping(item) for item in _sequence(event.get("options")))
        ],
    }


def _shop_summary(shop: Mapping[str, Any]) -> dict[str, Any]:
    if not shop:
        return {}
    return {
        "node_id": str(shop.get("node_id", "")),
        "card_removals_bought": _int(shop.get("card_removals_bought")),
        "items": [
            {
                "slot_id": str(item.get("slot_id", "")),
                "item_id": str(item.get("item_id", "")),
                "kind": str(item.get("kind", "")),
                "rarity": _optional_str(item.get("rarity")),
                "price": _int(item.get("price")),
                "purchased": bool(item.get("purchased", False)),
            }
            for item in (_mapping(raw) for raw in _sequence(shop.get("items")))
        ],
    }


def _reward_summary(reward: Mapping[str, Any]) -> dict[str, Any]:
    if not reward:
        return {}
    return {
        "reward_id": str(reward.get("reward_id", "")),
        "source": str(reward.get("source", "")),
        "forced": bool(reward.get("forced", False)),
        "gold": _int(reward.get("gold")),
        "gold_claimed": bool(reward.get("gold_claimed", False)),
        "relic_id": _optional_str(reward.get("relic_id")),
        "relic_ids": list(_sequence(reward.get("relic_ids"))),
        "claimed_relic_ids": list(_sequence(reward.get("claimed_relic_ids"))),
        "card_options": list(_sequence(reward.get("card_options"))),
        "card_option_groups": [
            list(_sequence(group)) for group in _sequence(reward.get("card_option_groups"))
        ],
        "card_ids": list(_sequence(reward.get("card_ids"))),
        "potion_id": _optional_str(reward.get("potion_id")),
        "potion_ids": list(_sequence(reward.get("potion_ids"))),
        "claimed_potion_indices": list(_sequence(reward.get("claimed_potion_indices"))),
    }


def _combat_summary(combat: Mapping[str, Any]) -> dict[str, Any]:
    if not combat:
        return {}
    return {
        "turn": _int(combat.get("turn")),
        "player": _combat_player_summary(_mapping(combat.get("player"))),
        "monsters": [
            _monster_summary(_mapping(monster)) for monster in _sequence(combat.get("monsters"))
        ],
        "hand": [_card_summary(_mapping(card)) for card in _sequence(combat.get("hand"))],
        "draw_pile": [_card_summary(_mapping(card)) for card in _sequence(combat.get("draw_pile"))],
        "discard_pile": [
            _card_summary(_mapping(card)) for card in _sequence(combat.get("discard_pile"))
        ],
        "exhaust_pile": [
            _card_summary(_mapping(card)) for card in _sequence(combat.get("exhaust_pile"))
        ],
        "orbs": [
            {"orb_id": str(orb.get("orb_id", "")), "value": _int(orb.get("value"))}
            for orb in (_mapping(raw) for raw in _sequence(combat.get("orbs")))
        ],
        "orb_slots": _int(combat.get("orb_slots")),
        "cards_played_this_turn": list(_sequence(combat.get("cards_played_this_turn"))),
        "pending_choices": [
            {
                "choice_id": str(choice.get("choice_id", "")),
                "kind": str(choice.get("kind", "")),
                "prompt": str(choice.get("prompt", "")),
                "candidate_ids": list(_sequence(choice.get("candidate_ids"))),
                "remaining": _int(choice.get("remaining")),
                "required": bool(choice.get("required", False)),
            }
            for choice in (_mapping(raw) for raw in _sequence(combat.get("pending_choices")))
        ],
    }


def _combat_player_summary(player: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "hp": _int(player.get("hp")),
        "max_hp": _int(player.get("max_hp")),
        "block": _int(player.get("block")),
        "energy": _int(player.get("energy")),
        "statuses": dict(_mapping(player.get("statuses"))),
        "resources": dict(_mapping(player.get("resources"))),
    }


def _monster_summary(monster: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "monster_id": str(monster.get("monster_id", "")),
        "name": str(monster.get("name", "")),
        "hp": _int(monster.get("hp")),
        "max_hp": _int(monster.get("max_hp")),
        "block": _int(monster.get("block")),
        "intent": _optional_str(monster.get("intent")),
        "intent_damage": _int(monster.get("intent_damage")),
        "hit_count": _int(monster.get("hit_count")),
        "statuses": dict(_mapping(monster.get("statuses"))),
    }


def _card_summary(card: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "instance_id": str(card.get("instance_id", "")),
        "card_id": str(card.get("card_id", "")),
        "name": str(card.get("name", "")),
        "type": str(card.get("type", "")),
        "cost": card.get("cost"),
        "upgraded": bool(card.get("upgraded", False)),
        "tags": list(_sequence(card.get("tags"))),
    }


def _public_flags(flags: Mapping[str, Any]) -> dict[str, Any]:
    skipped = {"debug", "source_data", "rng"}
    return {str(key): value for key, value in flags.items() if str(key) not in skipped}


def _find_ancient_option(
    ancient: Mapping[str, Any],
    option_id: str | None,
) -> Mapping[str, Any] | None:
    for option in (_mapping(item) for item in _sequence(ancient.get("options"))):
        if str(option.get("option_id")) == option_id:
            return option
    return None


def _find_map_node(
    game_map: Mapping[str, Any],
    node_id: str | None,
) -> Mapping[str, Any] | None:
    for node in (_mapping(item) for item in _sequence(game_map.get("nodes"))):
        if str(node.get("node_id")) == node_id:
            return node
    return None


def _find_event_option(
    event: Mapping[str, Any],
    option_id: str | None,
) -> Mapping[str, Any] | None:
    for option in (_mapping(item) for item in _sequence(event.get("options"))):
        if str(option.get("option_id")) == option_id:
            return option
    return None


def _find_shop_item(
    shop: Mapping[str, Any],
    slot_id: str | None,
) -> Mapping[str, Any] | None:
    for item in (_mapping(raw) for raw in _sequence(shop.get("items"))):
        if str(item.get("slot_id")) == slot_id:
            return item
    return None


def _find_card_by_instance_id(
    state_payload: Mapping[str, Any],
    instance_id: str | None,
) -> Mapping[str, Any] | None:
    if instance_id is None:
        return None
    for card in _all_cards(state_payload):
        if str(card.get("instance_id")) == instance_id:
            return card
    return None


def _all_cards(state_payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    cards: list[Mapping[str, Any]] = []
    cards.extend(_mapping(card) for card in _sequence(state_payload.get("master_deck")))
    combat = _mapping(state_payload.get("combat"))
    for zone in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        cards.extend(_mapping(card) for card in _sequence(combat.get(zone)))
    return tuple(cards)


def _target_name(state_payload: Mapping[str, Any], target_id: str | None) -> str | None:
    if target_id in {None, "", "none"}:
        return None
    if target_id == "player":
        return "player"
    combat = _mapping(state_payload.get("combat"))
    for monster in (_mapping(item) for item in _sequence(combat.get("monsters"))):
        if str(monster.get("monster_id")) == target_id:
            return str(monster.get("name", target_id))
    return target_id


def _potion_name_for_slot(state_payload: Mapping[str, Any], slot_id: str | None) -> str:
    if slot_id is None:
        return "unknown"
    parts = slot_id.split(":")
    if len(parts) == 2 and parts[0] == "potion":
        index = _int(parts[1])
        potions = _sequence(state_payload.get("potions"))
        if 0 <= index < len(potions):
            return str(potions[index])
    return slot_id


def _reward_action_summary(
    reward: Mapping[str, Any],
    action_type: str,
    target_id: str | None,
) -> str:
    if action_type == "take_reward_gold":
        return f"Take reward gold ({_int(reward.get('gold'))})"
    if action_type == "take_reward_relic":
        relic_id = _reward_relic_for_target(reward, target_id)
        return f"Take reward relic {relic_id}"
    if action_type == "take_reward_potion":
        potion_id = _reward_potion_for_target(reward, target_id)
        return f"Take reward potion {potion_id}"
    if action_type == "take_reward_card":
        card_id = _reward_card_for_target(reward, target_id)
        return f"Take reward card {card_id}"
    return f"Take reward {target_id}"


def _reward_relic_for_target(reward: Mapping[str, Any], target_id: str | None) -> str:
    if target_id == "reward:relic":
        return str(reward.get("relic_id", target_id))
    parts = (target_id or "").split(":")
    if len(parts) == 3 and parts[:2] == ["reward", "relic"]:
        relics = _sequence(reward.get("relic_ids"))
        index = _int(parts[2])
        if 0 <= index < len(relics):
            return str(relics[index])
    return str(target_id)


def _reward_potion_for_target(reward: Mapping[str, Any], target_id: str | None) -> str:
    if target_id == "reward:potion":
        return str(reward.get("potion_id", target_id))
    parts = (target_id or "").split(":")
    if len(parts) == 3 and parts[:2] == ["reward", "potion"]:
        potions = _sequence(reward.get("potion_ids"))
        index = _int(parts[2])
        if 0 <= index < len(potions):
            return str(potions[index])
    return str(target_id)


def _reward_card_for_target(reward: Mapping[str, Any], target_id: str | None) -> str:
    parts = (target_id or "").split(":")
    if len(parts) == 3 and parts[:2] == ["reward", "card"]:
        cards = _sequence(reward.get("card_options"))
        index = _int(parts[2])
        if 0 <= index < len(cards):
            return str(cards[index])
    if len(parts) == 3 and parts[:2] == ["reward", "fixed_card"]:
        cards = _sequence(reward.get("card_ids"))
        index = _int(parts[2])
        if 0 <= index < len(cards):
            return str(cards[index])
    if len(parts) == 4 and parts[:2] == ["reward", "card_group"]:
        groups = _sequence(reward.get("card_option_groups"))
        group_index = _int(parts[2])
        card_index = _int(parts[3])
        if 0 <= group_index < len(groups):
            group = _sequence(groups[group_index])
            if 0 <= card_index < len(group):
                return str(group[card_index])
    if len(parts) == 3 and parts[:2] == ["reward", "remove_card"]:
        return str(target_id)
    return str(target_id)


def _seed(payload: Mapping[str, Any]) -> int | str:
    seed = payload.get("seed", 0)
    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
    return str(seed)


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, str | bytes | bytearray) or value is None or isinstance(value, Mapping):
        return ()
    try:
        return tuple(value)
    except TypeError:
        return ()


def _int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
