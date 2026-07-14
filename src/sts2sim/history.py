"""Human-readable run history for simulator-driven agents."""

from __future__ import annotations

import json
from collections import Counter
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
    deck = [_card_summary(_mapping(card)) for card in _sequence(payload.get("master_deck"))]
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
            "deck_count": len(deck),
            "deck": deck,
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
        payload = _mapping(action.get("payload"))
        if target_id == "fake_merchant" or payload.get("merchant") == "fake_merchant":
            return "Throw Foul Potion at the fake merchant"
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

    if action_type == "skip_reward":
        return _skip_reward_action_summary(_mapping(state_payload.get("reward")), target_id)

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


def write_run_history_summary(
    history: RunHistory | Mapping[str, Any],
    path: Path | str,
    *,
    links: Mapping[str, str] | None = None,
) -> None:
    """Write the short node-by-node run summary as formatted JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(run_history_summary(history, links=links), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def write_run_history_summary_text(
    history: RunHistory | Mapping[str, Any],
    path: Path | str,
    *,
    links: Mapping[str, str] | None = None,
) -> None:
    """Write a quick text journal of the chosen route and node outcomes."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(run_history_summary_text(history, links=links) + "\n", encoding="utf-8")


def write_run_history_summary_html(
    history: RunHistory | Mapping[str, Any],
    path: Path | str,
    *,
    links: Mapping[str, str] | None = None,
) -> None:
    """Write a standalone HTML short route summary."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(run_history_summary_html(history, links=links), encoding="utf-8")


def run_history_summary(
    history: RunHistory | Mapping[str, Any],
    *,
    links: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a compact route journal grouped by ancient/map node."""

    payload = _history_payload(history)
    steps = tuple(_mapping(step) for step in _sequence(payload.get("steps")))
    initial = _mapping(payload.get("initial"))
    final = _mapping(payload.get("final"))
    nodes = _summary_nodes(steps, links=dict(links or {}))
    totals = _summary_totals(nodes, initial, final)
    return {
        "schema_version": 1,
        "generated_at": payload.get("generated_at", ""),
        "highlight_role": payload.get("highlight_role", ""),
        "seed": payload.get("seed", ""),
        "character_id": payload.get("character_id", ""),
        "ascension": _int(payload.get("ascension")),
        "policy": payload.get("policy", ""),
        "links": dict(links or {}),
        "initial": _summary_player_snapshot(initial),
        "final": _summary_player_snapshot(final) | {
            "act": _int(final.get("act")),
            "floor": _int(final.get("floor")),
            "phase": str(final.get("phase", "")),
        },
        "totals": totals,
        "nodes": nodes,
    }


def run_history_summary_text(
    history: RunHistory | Mapping[str, Any],
    *,
    links: Mapping[str, str] | None = None,
) -> str:
    """Render a compact plain-text route journal."""

    summary = run_history_summary(history, links=links)
    totals = _mapping(summary.get("totals"))
    final = _mapping(summary.get("final"))
    lines = [
        "Short run summary",
        (
            f"Generated at: {summary.get('generated_at') or '-'} | "
            f"Role: {summary.get('highlight_role') or '-'} | "
            f"Seed: {summary.get('seed')} | "
            f"{summary.get('character_id')} A{summary.get('ascension')}"
        ),
        (
            f"Final: act {_int(final.get('act'))} floor {_int(final.get('floor'))} "
            f"{final.get('phase', '')}, HP {_int(final.get('hp'))}/"
            f"{_int(final.get('max_hp'))}, gold {_int(final.get('gold'))}"
        ),
        (
            f"Totals: HP lost {_int(totals.get('hp_lost'))}, healed "
            f"{_int(totals.get('healed'))}, gold +{_int(totals.get('gold_gained'))}/"
            f"-{_int(totals.get('gold_spent'))}, cards +{_int(totals.get('cards_gained'))}/"
            f"-{_int(totals.get('cards_lost'))}, relics +{_int(totals.get('relics_gained'))}, "
            f"potions +{_int(totals.get('potions_gained'))}/"
            f"-{_int(totals.get('potions_lost'))}"
        ),
    ]
    link_lines = _summary_link_lines(_mapping(summary.get("links")))
    if link_lines:
        lines.append("Links: " + " | ".join(link_lines))
    for index, raw_node in enumerate(_sequence(summary.get("nodes")), start=1):
        node = _mapping(raw_node)
        lines.append("")
        lines.append(f"{index:02d}. {_summary_node_title(node)}")
        lines.append(
            f"    Steps {_int(node.get('step_start'))}-{_int(node.get('step_end'))}; "
            f"HP {_int(node.get('hp_before'))}->{_int(node.get('hp_after'))} "
            f"(lost {_int(node.get('hp_lost'))}, healed {_int(node.get('healed'))}); "
            f"gold {_int(node.get('gold_before'))}->{_int(node.get('gold_after'))}"
        )
        gained = _summary_gain_line(node)
        if gained:
            lines.append(f"    Took: {gained}")
        lost = _summary_loss_line(node)
        if lost:
            lines.append(f"    Lost/used: {lost}")
        skipped = _summary_filtered_lines(node, prefixes=("Reward skip:", "Reward screen left"))
        if skipped:
            lines.append("    Skipped: " + "; ".join(skipped[:3]))
        choices = _sequence(node.get("choices"))
        if choices:
            lines.append("    Choices: " + "; ".join(str(item) for item in choices[:4]))
        combat_actions = _sequence(node.get("combat_actions"))
        if combat_actions:
            actions = "; ".join(str(item) for item in combat_actions[:12])
            suffix = " ..." if len(combat_actions) > 12 else ""
            lines.append(f"    Combat: {actions}{suffix}")
        changes = [
            str(line)
            for line in _sequence(node.get("changes"))
            if str(line) not in skipped
        ]
        if changes:
            lines.append("    Changes: " + "; ".join(changes[:5]))
        node_links = _mapping(node.get("links"))
        replay = node_links.get("history")
        if replay:
            lines.append(f"    Replay: {replay}")
    return "\n".join(lines)


def run_history_summary_html(
    history: RunHistory | Mapping[str, Any],
    *,
    links: Mapping[str, str] | None = None,
    title: str = "Short Run Summary",
) -> str:
    """Render a short node-by-node HTML route journal."""

    summary = run_history_summary(history, links=links)
    totals = _mapping(summary.get("totals"))
    final = _mapping(summary.get("final"))
    safe_title = html_escape(title)
    link_html = _summary_links_html(_mapping(summary.get("links")))
    total_items = {
        "Generated": summary.get("generated_at") or "-",
        "Role": summary.get("highlight_role") or "-",
        "Seed": summary.get("seed"),
        "Character": f"{summary.get('character_id')} A{summary.get('ascension')}",
        "Final": (
            f"act {_int(final.get('act'))} floor {_int(final.get('floor'))} "
            f"{final.get('phase', '')}"
        ),
        "HP": f"{_int(final.get('hp'))}/{_int(final.get('max_hp'))}",
        "Gold": _int(final.get("gold")),
        "HP Lost": _int(totals.get("hp_lost")),
        "Gold Gained": _int(totals.get("gold_gained")),
    }
    overview = "".join(
        "<div>"
        f"<strong>{html_escape(str(key))}</strong>"
        f"<span>{html_escape(str(value))}</span>"
        "</div>"
        for key, value in total_items.items()
    )
    node_html = "\n".join(
        _summary_node_html(index, _mapping(node))
        for index, node in enumerate(_sequence(summary.get("nodes")), start=1)
    )
    if not node_html:
        node_html = '<p class="muted">No route steps were recorded.</p>'
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
    }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font: 14px/1.45 system-ui, sans-serif;
    }}
    main {{ width: min(1100px, calc(100vw - 32px)); margin: 0 auto; padding: 28px 0 48px; }}
    h1 {{ margin: 0 0 4px; font-size: 28px; }}
    h2 {{ margin: 28px 0 12px; font-size: 19px; }}
    h3 {{ margin: 0 0 8px; font-size: 17px; }}
    a {{ color: var(--accent); }}
    .muted {{ color: var(--muted); }}
    .overview {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .overview div, article {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .overview strong {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .overview span {{ display: block; margin-top: 4px; font-weight: 700; }}
    article {{ margin-bottom: 12px; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; color: var(--muted); }}
    .badge {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      background: #f8fafb;
      font-size: 12px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
    }}
    .box {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: #fbfcfd;
    }}
    .box strong {{ display: block; margin-bottom: 4px; }}
    ul, ol {{ margin: 6px 0 0; padding-left: 20px; }}
    li {{ margin: 2px 0; }}
  </style>
</head>
<body>
<main>
  <h1>{safe_title}</h1>
  <p class="muted">
    Compact route journal grouped by start choice and map node. Use the links
    to open the full replay, JSON, or visual map when a node needs inspection.
  </p>
  {link_html}
  <section class="overview">{overview}</section>
  <h2>Route</h2>
  {node_html}
</main>
</body>
</html>
"""


def _summary_nodes(
    steps: Sequence[Mapping[str, Any]],
    *,
    links: Mapping[str, str],
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        segment_steps = tuple(_mapping(step) for step in _sequence(current.get("steps")))
        if segment_steps:
            segments.append(_summary_segment(current, segment_steps, links=links))
        current = None

    for raw_step in steps:
        step = _mapping(raw_step)
        action_type = str(_mapping(step.get("action")).get("type", ""))
        if action_type in {"choose_ancient", "choose_node"}:
            flush()
            current = _summary_segment_header(step)
        elif current is None:
            current = _summary_start_header(step)
        current.setdefault("steps", []).append(step)
    flush()
    return segments


def _summary_start_header(step: Mapping[str, Any]) -> dict[str, Any]:
    before = _mapping(step.get("context_before"))
    return {
        "node_id": "start",
        "kind": "start",
        "act": _int(before.get("act")),
        "floor": _int(before.get("floor")),
        "lane": None,
        "label": "Run start",
        "steps": [],
    }


def _summary_segment_header(step: Mapping[str, Any]) -> dict[str, Any]:
    action = _mapping(step.get("action"))
    action_type = str(action.get("type", ""))
    target_id = _optional_str(action.get("target_id"))
    before = _mapping(step.get("context_before"))
    after = _mapping(step.get("context_after"))
    if action_type == "choose_node":
        before_map = _mapping(before.get("map"))
        after_map = _mapping(after.get("map"))
        node = (
            _find_map_node(before_map, target_id)
            or _find_map_node(after_map, target_id)
            or {}
        )
        kind = str(node.get("kind") or "node")
        floor = _int(node.get("floor"))
        lane = _int(node.get("lane"))
        act = _int(before_map.get("act")) or _int(after_map.get("act")) or _int(before.get("act"))
        return {
            "node_id": target_id or "",
            "kind": kind,
            "act": act,
            "floor": floor,
            "lane": lane,
            "label": f"Act {act} floor {floor} {kind}",
            "steps": [],
        }
    ancient = _mapping(before.get("ancient")) or _mapping(after.get("ancient"))
    act = _int(ancient.get("act")) or _int(before.get("act")) or _int(after.get("act"))
    option = _find_ancient_option(ancient, target_id) or {}
    label = str(option.get("name") or target_id or "Ancient option")
    return {
        "node_id": f"ancient:a{act}",
        "kind": "ancient",
        "act": act,
        "floor": _int(before.get("floor")),
        "lane": None,
        "label": f"Act {act} ancient: {label}",
        "steps": [],
    }


def _summary_segment(
    header: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    *,
    links: Mapping[str, str],
) -> dict[str, Any]:
    first = steps[0]
    last = steps[-1]
    before = _mapping(first.get("context_before"))
    after = _mapping(last.get("context_after"))
    before_player = _summary_context_player(before)
    after_player = _summary_context_player(after)
    hp_lost, healed = _summary_metric_flow(steps, "hp")
    gold_spent, gold_gained = _summary_metric_flow(steps, "gold")
    choices: list[str] = []
    combat_actions: list[str] = []
    reward_actions: list[str] = []
    shop_actions: list[str] = []
    other_actions: list[str] = []
    all_actions: list[str] = []
    changes: list[str] = []
    for step in steps:
        action_type = str(_mapping(step.get("action")).get("type", ""))
        action_summary = str(step.get("action_summary", "Action"))
        all_actions.append(action_summary)
        if action_type in {
            "choose_ancient",
            "choose_node",
            "choose_event",
            "smith",
            "rest",
            "toke",
        }:
            choices.append(action_summary)
        elif action_type in {
            "play_card",
            "use_potion",
            "discard_potion",
            "discard_card",
            "exhaust_card",
            "choose_card",
            "end_turn",
        }:
            combat_actions.append(action_summary)
        elif action_type.startswith("take_reward") or action_type in {"skip_reward", "proceed"}:
            reward_actions.append(action_summary)
        elif action_type.startswith("shop_") or action_type == "throw_potion_at_merchant":
            shop_actions.append(action_summary)
        else:
            other_actions.append(action_summary)
        changes.extend(_summary_step_lines(step))

    before_deck = _summary_card_inventory(before_player)
    after_deck = _summary_card_inventory(after_player)
    before_relics = tuple(str(item) for item in _sequence(before_player.get("relics")))
    after_relics = tuple(str(item) for item in _sequence(after_player.get("relics")))
    before_potions = tuple(str(item) for item in _sequence(before_player.get("potions")))
    after_potions = tuple(str(item) for item in _sequence(after_player.get("potions")))
    node_links = _summary_node_links(links, _int(first.get("step_index")))
    return {
        "node_id": header.get("node_id"),
        "kind": header.get("kind"),
        "act": _int(header.get("act")),
        "floor": _int(header.get("floor")),
        "lane": header.get("lane"),
        "label": header.get("label"),
        "step_start": _int(first.get("step_index")),
        "step_end": _int(last.get("step_index")),
        "phase_before": first.get("phase_before", ""),
        "phase_after": last.get("phase_after", ""),
        "hp_before": _int(before_player.get("hp")),
        "hp_after": _int(after_player.get("hp")),
        "max_hp_before": _int(before_player.get("max_hp")),
        "max_hp_after": _int(after_player.get("max_hp")),
        "hp_lost": hp_lost,
        "healed": healed,
        "gold_before": _int(before_player.get("gold")),
        "gold_after": _int(after_player.get("gold")),
        "gold_gained": gold_gained,
        "gold_spent": gold_spent,
        "cards_gained": list(_ordered_counter_delta(after_deck, before_deck)),
        "cards_lost": list(_ordered_counter_delta(before_deck, after_deck)),
        "relics_gained": list(_ordered_counter_delta(after_relics, before_relics)),
        "relics_lost": list(_ordered_counter_delta(before_relics, after_relics)),
        "potions_gained": list(_ordered_counter_delta(after_potions, before_potions)),
        "potions_lost": list(_ordered_counter_delta(before_potions, after_potions)),
        "choices": list(dict.fromkeys(choices)),
        "combat_actions": combat_actions,
        "reward_actions": reward_actions,
        "shop_actions": shop_actions,
        "other_actions": other_actions,
        "actions": all_actions,
        "changes": list(_dedupe_lines(changes)),
        "links": node_links,
    }


def _summary_step_lines(step: Mapping[str, Any]) -> tuple[str, ...]:
    before = _mapping(step.get("context_before"))
    after = _mapping(step.get("context_after"))
    action_type = str(_mapping(step.get("action")).get("type", ""))
    lines: list[str] = list(_compact_step_change_lines(step))
    lines.extend(_shop_outcome_lines(action_type, before, after))
    lines.extend(_event_outcome_lines(action_type, before, after, _sequence(step.get("events"))))
    return tuple(lines)


def _summary_context_player(context: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(context.get("player"))


def _summary_card_inventory(player: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(_card_change_label(_mapping(card)) for card in _sequence(player.get("deck")))


def _summary_metric_flow(steps: Sequence[Mapping[str, Any]], key: str) -> tuple[int, int]:
    gained = 0
    spent = 0
    for step in steps:
        before = _summary_context_player(_mapping(step.get("context_before")))
        after = _summary_context_player(_mapping(step.get("context_after")))
        if not before or not after:
            continue
        delta = _int(after.get(key)) - _int(before.get(key))
        if delta > 0:
            gained += delta
        elif delta < 0:
            spent += -delta
    return spent, gained


def _summary_player_snapshot(context: Mapping[str, Any]) -> dict[str, Any]:
    player = _summary_context_player(context)
    return {
        "hp": _int(player.get("hp")),
        "max_hp": _int(player.get("max_hp")),
        "gold": _int(player.get("gold")),
        "deck_count": _int(player.get("deck_count")),
        "relics": list(_sequence(player.get("relics"))),
        "potions": list(_sequence(player.get("potions"))),
    }


def _summary_totals(
    nodes: Sequence[Mapping[str, Any]],
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
) -> dict[str, Any]:
    initial_player = _summary_context_player(initial)
    final_player = _summary_context_player(final)
    return {
        "nodes": len(nodes),
        "hp_lost": sum(_int(node.get("hp_lost")) for node in nodes),
        "healed": sum(_int(node.get("healed")) for node in nodes),
        "gold_gained": sum(_int(node.get("gold_gained")) for node in nodes),
        "gold_spent": sum(_int(node.get("gold_spent")) for node in nodes),
        "cards_gained": sum(len(_sequence(node.get("cards_gained"))) for node in nodes),
        "cards_lost": sum(len(_sequence(node.get("cards_lost"))) for node in nodes),
        "relics_gained": sum(len(_sequence(node.get("relics_gained"))) for node in nodes),
        "relics_lost": sum(len(_sequence(node.get("relics_lost"))) for node in nodes),
        "potions_gained": sum(len(_sequence(node.get("potions_gained"))) for node in nodes),
        "potions_lost": sum(len(_sequence(node.get("potions_lost"))) for node in nodes),
        "initial_hp": _int(initial_player.get("hp")),
        "final_hp": _int(final_player.get("hp")),
        "initial_gold": _int(initial_player.get("gold")),
        "final_gold": _int(final_player.get("gold")),
        "final_deck_count": _int(final_player.get("deck_count")),
    }


def _summary_node_links(links: Mapping[str, str], step_index: int) -> dict[str, str]:
    result = {key: value for key, value in links.items() if value}
    history_link = result.get("history")
    if history_link:
        result["history"] = f"{history_link}#step-{step_index}"
    return result


def _summary_link_lines(links: Mapping[str, Any]) -> tuple[str, ...]:
    labels = {
        "history": "full replay",
        "history_json": "full json",
        "map": "map",
        "summary_html": "summary html",
        "summary_json": "summary json",
        "summary_txt": "summary text",
    }
    return tuple(
        f"{labels.get(key, key)} {value}"
        for key, value in links.items()
        if value
    )


def _summary_links_html(links: Mapping[str, Any]) -> str:
    if not links:
        return ""
    label_by_key = {
        "history": "Full Replay",
        "history_json": "Full JSON",
        "map": "Map",
        "summary_json": "Summary JSON",
        "summary_txt": "Summary Text",
    }
    anchors = []
    for key, value in links.items():
        if not value or key == "summary_html":
            continue
        label = label_by_key.get(key, key.replace("_", " ").title())
        anchors.append(
            f'<a class="badge" href="{html_escape(str(value))}">{html_escape(label)}</a>'
        )
    if not anchors:
        return ""
    return '<nav class="meta">' + "".join(anchors) + "</nav>"


def _summary_node_title(node: Mapping[str, Any]) -> str:
    kind = str(node.get("kind") or "node")
    label = str(node.get("label") or "")
    node_id = str(node.get("node_id") or "")
    if kind == "ancient":
        return label
    if kind == "start":
        return "Run start"
    return (
        f"Act {_int(node.get('act'))} floor {_int(node.get('floor'))} "
        f"{kind} {node_id}"
    ).strip()


def _summary_gain_line(node: Mapping[str, Any]) -> str:
    chunks: list[str] = []
    if _int(node.get("gold_gained")):
        chunks.append(f"gold {_int(node.get('gold_gained'))}")
    for key, label in (
        ("cards_gained", "cards"),
        ("relics_gained", "relics"),
        ("potions_gained", "potions"),
    ):
        values = [str(item) for item in _sequence(node.get(key))]
        if values:
            chunks.append(f"{label} " + ", ".join(values))
    return "; ".join(chunks)


def _summary_loss_line(node: Mapping[str, Any]) -> str:
    chunks: list[str] = []
    if _int(node.get("gold_spent")):
        chunks.append(f"gold {_int(node.get('gold_spent'))}")
    for key, label in (
        ("cards_lost", "cards"),
        ("relics_lost", "relics"),
        ("potions_lost", "potions"),
    ):
        values = [str(item) for item in _sequence(node.get(key))]
        if values:
            chunks.append(f"{label} " + ", ".join(values))
    return "; ".join(chunks)


def _summary_filtered_lines(
    node: Mapping[str, Any],
    *,
    prefixes: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        str(line)
        for line in _sequence(node.get("changes"))
        if any(str(line).startswith(prefix) for prefix in prefixes)
    )


def _summary_node_html(index: int, node: Mapping[str, Any]) -> str:
    title = _summary_node_title(node)
    node_links = _mapping(node.get("links"))
    replay = node_links.get("history")
    replay_link = (
        f' <a href="{html_escape(str(replay))}">open detailed replay</a>'
        if replay
        else ""
    )
    meta = [
        f"steps {_int(node.get('step_start'))}-{_int(node.get('step_end'))}",
        f"HP {_int(node.get('hp_before'))}->{_int(node.get('hp_after'))}",
        f"lost {_int(node.get('hp_lost'))}",
        f"healed {_int(node.get('healed'))}",
        f"gold {_int(node.get('gold_before'))}->{_int(node.get('gold_after'))}",
    ]
    meta_html = "".join(f'<span class="badge">{html_escape(item)}</span>' for item in meta)
    boxes = [
        _summary_box_html("Took", _summary_gain_line(node)),
        _summary_box_html("Lost / Used", _summary_loss_line(node)),
        _summary_list_box_html("Choices", _sequence(node.get("choices")), limit=6),
        _summary_list_box_html("Combat", _sequence(node.get("combat_actions")), limit=16),
        _summary_list_box_html("Rewards", _sequence(node.get("reward_actions")), limit=8),
        _summary_list_box_html("Changes", _sequence(node.get("changes")), limit=8),
    ]
    return (
        "<article>"
        f"<h3>{index:02d}. {html_escape(title)}{replay_link}</h3>"
        f'<div class="meta">{meta_html}</div>'
        f'<div class="grid">{"".join(boxes)}</div>'
        "</article>"
    )


def _summary_box_html(title: str, value: str) -> str:
    body = html_escape(value) if value else '<span class="muted">None.</span>'
    return f'<div class="box"><strong>{html_escape(title)}</strong>{body}</div>'


def _summary_list_box_html(title: str, values: Sequence[Any], *, limit: int) -> str:
    items = [str(item) for item in values if str(item)]
    if not items:
        return _summary_box_html(title, "")
    visible = items[:limit]
    suffix = f"<li>... {len(items) - limit} more</li>" if len(items) > limit else ""
    body = "<ol>" + "".join(f"<li>{html_escape(item)}</li>" for item in visible) + suffix + "</ol>"
    return f'<div class="box"><strong>{html_escape(title)}</strong>{body}</div>'


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
        f"Generated at: {payload.get('generated_at', '') or '-'}",
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
        "generated at": payload.get("generated_at", ""),
        "highlight role": payload.get("highlight_role", ""),
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
    timeline = _timeline_html(steps)
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
    details {{
      margin-top: 10px;
      border-top: 1px solid var(--line);
      padding-top: 8px;
    }}
    summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 700;
    }}
    .combat-turn {{
      border-left: 4px solid var(--accent);
    }}
    .turn-actions {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin: 10px 0;
    }}
    .turn-actions div {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: #fbfcfd;
    }}
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


def _timeline_html(steps: Sequence[Mapping[str, Any]]) -> str:
    chunks: list[str] = []
    combat_group: list[Mapping[str, Any]] = []
    current_turn: int | None = None

    def flush_combat_group() -> None:
        nonlocal combat_group, current_turn
        if combat_group:
            chunks.append(_combat_turn_group_html(tuple(combat_group)))
            combat_group = []
            current_turn = None

    for step in steps:
        turn = _combat_step_group_turn(step)
        if turn is None:
            flush_combat_group()
            chunks.append(_step_html(step))
            continue
        if combat_group and current_turn != turn:
            flush_combat_group()
        combat_group.append(step)
        current_turn = turn
        if _combat_step_ends_group(step):
            flush_combat_group()
    flush_combat_group()
    return "\n".join(chunks)


def _combat_step_group_turn(step: Mapping[str, Any]) -> int | None:
    before = _mapping(step.get("context_before"))
    after = _mapping(step.get("context_after"))
    before_combat = _mapping(before.get("combat"))
    after_combat = _mapping(after.get("combat"))
    if before_combat:
        return max(1, _int(before_combat.get("turn")) or 1)
    if after_combat and str(step.get("phase_before", "")) != "combat":
        return max(1, _int(after_combat.get("turn")) or 1)
    return None


def _combat_step_ends_group(step: Mapping[str, Any]) -> bool:
    action_type = str(_mapping(step.get("action")).get("type", ""))
    after = _mapping(step.get("context_after"))
    before_combat = _mapping(_mapping(step.get("context_before")).get("combat"))
    after_combat = _mapping(after.get("combat"))
    if before_combat and not after_combat:
        return True
    if action_type == "end_turn":
        return True
    if before_combat and after_combat:
        return _int(after_combat.get("turn")) != _int(before_combat.get("turn"))
    return False


def _combat_turn_group_html(steps: Sequence[Mapping[str, Any]]) -> str:
    first = steps[0]
    last = steps[-1]
    first_step_index = _int(first.get("step_index"))
    first_before = _mapping(first.get("context_before"))
    first_after = _mapping(first.get("context_after"))
    start_combat = _mapping(first_before.get("combat")) or _mapping(first_after.get("combat"))
    last_after = _mapping(last.get("context_after"))
    end_combat = _mapping(last_after.get("combat"))
    turn = _combat_step_group_turn(first) or _int(start_combat.get("turn")) or 1
    reward_total = sum(_float(step.get("reward")) for step in steps)
    meta = [
        f"steps {_int(first.get('step_index'))}-{_int(last.get('step_index'))}",
        f"phase {first.get('phase_before', '')} -> {last.get('phase_after', '')}",
        f"reward {reward_total:.3f}",
    ]
    meta_html = "".join(f'<span class="badge">{html_escape(item)}</span>' for item in meta)

    card_actions = [
        str(step.get("action_summary", "Play card"))
        for step in steps
        if str(_mapping(step.get("action")).get("type", "")) == "play_card"
    ]
    potion_actions = [
        str(step.get("action_summary", "Use potion"))
        for step in steps
        if str(_mapping(step.get("action")).get("type", "")) in {"use_potion", "discard_potion"}
    ]
    other_actions = [
        str(step.get("action_summary", "Action"))
        for step in steps
        if str(_mapping(step.get("action")).get("type", ""))
        not in {"play_card", "use_potion", "discard_potion"}
    ]
    start_lines = _combat_start_lines(start_combat)
    change_lines = _dedupe_lines(
        line
        for step in steps
        for line in _compact_step_change_lines(step)
    )
    end_lines = _combat_end_lines(start_combat, end_combat, last)
    action_cards = _action_list_html("Cards Played This Turn", card_actions)
    potion_cards = _action_list_html("Potions / Discards", potion_actions)
    other_cards = _action_list_html("Other Turn Actions", other_actions)
    main_lines = "".join(
        f"<li>{html_escape(line)}</li>"
        for line in (*start_lines, *change_lines, *end_lines)
    )
    if not main_lines:
        main_lines = "<li>No visible combat state change.</li>"
    details = _combat_step_details_html(steps)
    return (
        f'<article class="combat-turn" id="step-{first_step_index}">'
        f"<h3>Combat Turn {turn}</h3>"
        f'<div class="step-meta">{meta_html}</div>'
        f'<div class="turn-actions">{action_cards}{potion_cards}{other_cards}</div>'
        f'<ul class="narrative">{main_lines}</ul>'
        f"{details}"
        "</article>"
    )


def _combat_start_lines(combat: Mapping[str, Any]) -> tuple[str, ...]:
    if not combat:
        return ()
    player = _mapping(combat.get("player"))
    lines = [
        "Turn start: "
        f"HP {_int(player.get('hp'))}/{_int(player.get('max_hp'))}, "
        f"block {_int(player.get('block'))}, "
        f"energy {_int(player.get('energy'))}, "
        f"hand {_card_names(combat.get('hand'))}."
    ]
    intents = _monster_intents(combat)
    if intents:
        lines.append(f"Enemy intentions: {intents}.")
    return tuple(lines)


def _combat_end_lines(
    start_combat: Mapping[str, Any],
    end_combat: Mapping[str, Any],
    last_step: Mapping[str, Any],
) -> tuple[str, ...]:
    if not start_combat and not end_combat:
        return ()
    if not end_combat:
        return ("Combat ended; reward or next room state opened.",)
    before_turn = _int(start_combat.get("turn"))
    after_turn = _int(end_combat.get("turn"))
    player = _mapping(end_combat.get("player"))
    prefix = "Turn end state"
    if after_turn > before_turn or str(_mapping(last_step.get("action")).get("type")) == "end_turn":
        prefix = "After enemy turn"
    lines = [
        f"{prefix}: HP {_int(player.get('hp'))}/{_int(player.get('max_hp'))}, "
        f"block {_int(player.get('block'))}, energy {_int(player.get('energy'))}, "
        f"hand {_card_names(end_combat.get('hand'))}."
    ]
    if after_turn > before_turn:
        intents = _monster_intents(end_combat)
        if intents:
            lines.append(f"Next turn enemy intentions: {intents}.")
    return tuple(lines)


def _compact_step_change_lines(step: Mapping[str, Any]) -> tuple[str, ...]:
    before = _mapping(step.get("context_before"))
    after = _mapping(step.get("context_after"))
    action_type = str(_mapping(step.get("action")).get("type", ""))
    lines: list[str] = []
    lines.extend(_state_change_lines(before, after))
    lines.extend(
        _combat_change_lines(
            _mapping(before.get("combat")),
            _mapping(after.get("combat")),
        )
    )
    lines.extend(_event_effect_lines(_sequence(step.get("events"))))
    lines.extend(_room_outcome_lines(action_type, before, after))
    lines.extend(_reward_outcome_lines(action_type, before, after))
    return tuple(lines)


def _action_list_html(title: str, actions: Sequence[str]) -> str:
    if not actions:
        body = '<p class="muted">None.</p>'
    else:
        body = "<ol>" + "".join(f"<li>{html_escape(action)}</li>" for action in actions) + "</ol>"
    return f"<div><strong>{html_escape(title)}</strong>{body}</div>"


def _combat_step_details_html(steps: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for step in steps:
        step_index = _int(step.get("step_index"))
        change_text = "; ".join(_compact_step_change_lines(step)) or "no visible change"
        rows.append(
            f'<tr id="detail-step-{step_index}">'
            f"<td>{step_index}</td>"
            f"<td>{html_escape(str(step.get('action_summary', 'Action')))}</td>"
            f"<td>{_float(step.get('reward')):.3f}</td>"
            f"<td>{html_escape(change_text)}</td>"
            "</tr>"
        )
    return (
        "<details><summary>Step Details</summary>"
        "<table><thead><tr><th>Step</th><th>Action</th><th>Reward</th>"
        "<th>Visible Changes</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "</details>"
    )


def _dedupe_lines(lines: Sequence[str] | Any) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        if not line or line in seen:
            continue
        seen.add(line)
        result.append(str(line))
    return tuple(result)


def _step_html(step: Mapping[str, Any]) -> str:
    context_before = _mapping(step.get("context_before"))
    context_after = _mapping(step.get("context_after"))
    events = tuple(_mapping(event) for event in _sequence(step.get("events")))
    decision = _mapping(step.get("decision"))
    meta = [
        f"phase {step.get('phase_before', '')} -> {step.get('phase_after', '')}",
        f"reward {step.get('reward', 0)}",
    ]
    meta_html = "".join(f'<span class="badge">{html_escape(item)}</span>' for item in meta)
    narrative = "".join(
        f"<li>{html_escape(item)}</li>"
        for item in _step_narrative(step, context_before, context_after)
    )
    if not narrative:
        narrative = "<li>No visible state change.</li>"
    events_html = _events_html(events)
    decision_html = _decision_html(decision)
    step_index = _int(step.get("step_index"))
    return (
        f'<article id="step-{step_index}">'
        f"<h3>Step {step_index}: "
        f"{html_escape(str(step.get('action_summary', 'Action')))}</h3>"
        f'<div class="step-meta">{meta_html}</div>'
        f'<ul class="narrative">{narrative}</ul>'
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
    action = _mapping(step.get("action"))
    action_type = str(action.get("type", ""))
    lines.extend(_state_change_lines(before, after))
    lines.extend(_event_effect_lines(_sequence(step.get("events"))))
    lines.extend(_room_outcome_lines(action_type, before, after))
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
    if after_reward and after_reward != before_reward:
        lines.append(f"Reward screen now: {_reward_line(after_reward)}.")
    lines.extend(_reward_outcome_lines(action_type, before, after))

    before_shop = _mapping(before.get("shop"))
    after_shop = _mapping(after.get("shop"))
    if after_shop and after_shop != before_shop:
        lines.append(f"Shop now: {_shop_line(after_shop)}.")
    lines.extend(_shop_outcome_lines(action_type, before, after))

    before_event = _mapping(before.get("event"))
    after_event = _mapping(after.get("event"))
    if after_event and after_event != before_event:
        lines.append(f"Event now: {_event_line(after_event)}.")
    lines.extend(_event_outcome_lines(action_type, before, after, _sequence(step.get("events"))))

    before_map = _mapping(before.get("map"))
    after_map = _mapping(after.get("map"))
    if after_map and after_map != before_map:
        lines.append(f"Map now: {_map_line(after_map)}.")
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
        f"<td>{html_escape(_monster_intent_text(monster))}</td>"
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
    return (
        "<details><summary>Raw Engine Events</summary>"
        f'<div class="events">{event_items}</div>'
        "</details>"
    )


def _decision_html(decision: Mapping[str, Any]) -> str:
    if not decision:
        return ""
    visible_keys = (
        "action_id",
        "action_index",
        "confidence",
        "value",
        "reward_total",
        "reward_aggression_pressure",
    )
    compact = {key: decision[key] for key in visible_keys if key in decision}
    compact_cells = "".join(
        "<tr>"
        f"<th>{html_escape(str(key).replace('_', ' ').title())}</th>"
        f"<td>{html_escape(str(value))}</td>"
        "</tr>"
        for key, value in compact.items()
    )
    full_cells = "".join(
        "<tr>"
        f"<th>{html_escape(str(key).replace('_', ' ').title())}</th>"
        f"<td>{html_escape(str(value))}</td>"
        "</tr>"
        for key, value in sorted(decision.items())
    )
    compact_html = (
        f"<table><tbody>{compact_cells}</tbody></table>" if compact_cells else ""
    )
    return (
        f"{compact_html}"
        "<details><summary>Full Policy Output</summary>"
        f"<table><tbody>{full_cells}</tbody></table>"
        "</details>"
    )


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
            f"intent {_monster_intent_text(monster)}"
            f"{status_text}"
        )
    return "; ".join(parts)


def _monster_intent_text(monster: Mapping[str, Any]) -> str:
    intent = str(monster.get("intent") or "-")
    damage_text = _monster_intent_damage_text(monster)
    if not damage_text:
        return intent
    return f"{intent} {damage_text}"


def _monster_intent_damage_text(monster: Mapping[str, Any]) -> str:
    total = _monster_intent_total_damage(monster)
    hits = _monster_intent_hit_count(monster)
    if total <= 0 and hits <= 1:
        return ""
    if hits <= 1:
        return str(total)
    per_hit = _monster_intent_per_hit_damage(monster)
    return f"{per_hit}x{hits} (total incoming {total})"


def _monster_intent_total_damage(monster: Mapping[str, Any]) -> int:
    return max(
        0,
        _int(
            monster.get(
                "intent_damage_total",
                monster.get("intent_damage"),
            )
        ),
    )


def _monster_intent_hit_count(monster: Mapping[str, Any]) -> int:
    return max(1, _int(monster.get("hit_count")))


def _monster_intent_per_hit_damage(monster: Mapping[str, Any]) -> int:
    explicit = monster.get("intent_damage_per_hit")
    if explicit is not None:
        return max(0, _int(explicit))
    hits = _monster_intent_hit_count(monster)
    total = _monster_intent_total_damage(monster)
    return max(0, total // hits)


def _reward_line(reward: Mapping[str, Any]) -> str:
    parts = [
        f"source {reward.get('source', '')}",
        (
            f"gold {_int(reward.get('gold'))}"
            f"{' claimed' if reward.get('gold_claimed') else ''}"
            f"{' skipped' if reward.get('gold_skipped') else ''}"
        ),
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
    if reward.get("relic_claimed") and relic_id:
        claimed.append(f"relic:{relic_id}")
    claimed.extend(str(item) for item in _sequence(reward.get("claimed_relic_ids")))
    if reward.get("card_claimed") and card_options:
        claimed.append("card-options")
    claimed.extend(
        _indexed_reward_labels("fixed-card", fixed_cards, reward.get("claimed_card_indices"))
    )
    claimed.extend(
        _indexed_reward_labels(
            "card-group",
            card_groups,
            reward.get("claimed_card_option_group_indices"),
            bracket_values=True,
        )
    )
    if reward.get("potion_claimed") and potion_id:
        claimed.append(f"potion:{potion_id}")
    claimed.extend(f"potion#{item}" for item in _sequence(reward.get("claimed_potion_indices")))
    if claimed:
        parts.append("claimed " + ", ".join(claimed))
    skipped: list[str] = []
    if reward.get("gold_skipped"):
        skipped.append("gold")
    if reward.get("relic_skipped") and relic_id:
        skipped.append(f"relic:{relic_id}")
    skipped.extend(str(item) for item in _sequence(reward.get("skipped_relic_ids")))
    if reward.get("card_skipped") and card_options:
        skipped.append("card-options")
    skipped.extend(
        _indexed_reward_labels("fixed-card", fixed_cards, reward.get("skipped_card_indices"))
    )
    skipped.extend(
        _indexed_reward_labels(
            "card-group",
            card_groups,
            reward.get("skipped_card_option_group_indices"),
            bracket_values=True,
        )
    )
    if reward.get("potion_skipped") and potion_id:
        skipped.append(f"potion:{potion_id}")
    skipped.extend(f"potion#{item}" for item in _sequence(reward.get("skipped_potion_indices")))
    if skipped:
        parts.append("skipped " + ", ".join(skipped))
    return "; ".join(parts)


def _reward_has_available_loot(reward: Mapping[str, Any]) -> bool:
    return bool(_reward_unclaimed_labels(reward))


def _reward_available_summary(reward: Mapping[str, Any]) -> str:
    labels = _reward_unclaimed_labels(reward)
    return ", ".join(labels) if labels else "no claimable loot"


def _reward_claimed_labels(reward: Mapping[str, Any]) -> tuple[str, ...]:
    labels: list[str] = []
    if _int(reward.get("gold")) > 0 and bool(reward.get("gold_claimed", False)):
        labels.append(f"gold {_int(reward.get('gold'))}")

    relic_id = _optional_str(reward.get("relic_id"))
    if relic_id and bool(reward.get("relic_claimed", False)):
        labels.append(f"relic {relic_id}")
    relics = tuple(_sequence(reward.get("relic_ids")))
    claimed_relics = {str(item) for item in _sequence(reward.get("claimed_relic_ids"))}
    labels.extend(f"relic {item}" for item in relics if str(item) in claimed_relics)

    card_options = tuple(_sequence(reward.get("card_options")))
    if card_options and bool(reward.get("card_claimed", False)):
        labels.append("card choice from [" + ", ".join(str(item) for item in card_options) + "]")

    fixed_cards = tuple(_sequence(reward.get("card_ids")))
    labels.extend(
        f"card {label}"
        for label in _indexed_reward_values(fixed_cards, reward.get("claimed_card_indices"))
    )

    card_groups = tuple(_sequence(reward.get("card_option_groups")))
    labels.extend(
        f"card group {label}"
        for label in _indexed_reward_group_values(
            card_groups,
            reward.get("claimed_card_option_group_indices"),
        )
    )

    potion_id = _optional_str(reward.get("potion_id"))
    if potion_id and bool(reward.get("potion_claimed", False)):
        labels.append(f"potion {potion_id}")
    potions = tuple(_sequence(reward.get("potion_ids")))
    labels.extend(
        f"potion {label}"
        for label in _indexed_reward_values(potions, reward.get("claimed_potion_indices"))
    )
    return tuple(labels)


def _reward_unclaimed_labels(reward: Mapping[str, Any]) -> tuple[str, ...]:
    labels: list[str] = []
    if (
        _int(reward.get("gold")) > 0
        and not bool(reward.get("gold_claimed", False))
        and not bool(reward.get("gold_skipped", False))
    ):
        labels.append(f"gold {_int(reward.get('gold'))}")

    relic_id = _optional_str(reward.get("relic_id"))
    if (
        relic_id
        and not bool(reward.get("relic_claimed", False))
        and not bool(reward.get("relic_skipped", False))
    ):
        labels.append(f"relic {relic_id}")
    claimed_relics = {str(item) for item in _sequence(reward.get("claimed_relic_ids"))}
    skipped_relics = {str(item) for item in _sequence(reward.get("skipped_relic_ids"))}
    for relic in _sequence(reward.get("relic_ids")):
        relic_text = str(relic)
        if relic_text not in claimed_relics and relic_text not in skipped_relics:
            labels.append(f"relic {relic_text}")

    card_options = tuple(_sequence(reward.get("card_options")))
    if (
        card_options
        and not bool(reward.get("card_claimed", False))
        and not bool(reward.get("card_skipped", False))
    ):
        labels.append("card choice [" + ", ".join(str(item) for item in card_options) + "]")

    fixed_cards = tuple(_sequence(reward.get("card_ids")))
    claimed_card_indices = {_int(item) for item in _sequence(reward.get("claimed_card_indices"))}
    skipped_card_indices = {_int(item) for item in _sequence(reward.get("skipped_card_indices"))}
    for index, card in enumerate(fixed_cards):
        if index not in claimed_card_indices and index not in skipped_card_indices:
            labels.append(f"card {card}")

    card_groups = tuple(_sequence(reward.get("card_option_groups")))
    claimed_group_indices = {
        _int(item) for item in _sequence(reward.get("claimed_card_option_group_indices"))
    }
    skipped_group_indices = {
        _int(item) for item in _sequence(reward.get("skipped_card_option_group_indices"))
    }
    for index, group in enumerate(card_groups):
        if index not in claimed_group_indices and index not in skipped_group_indices:
            labels.append(
                "card group [" + ", ".join(str(item) for item in _sequence(group)) + "]"
            )

    potion_id = _optional_str(reward.get("potion_id"))
    if (
        potion_id
        and not bool(reward.get("potion_claimed", False))
        and not bool(reward.get("potion_skipped", False))
    ):
        labels.append(f"potion {potion_id}")
    claimed_potion_indices = {
        _int(item) for item in _sequence(reward.get("claimed_potion_indices"))
    }
    skipped_potion_indices = {
        _int(item) for item in _sequence(reward.get("skipped_potion_indices"))
    }
    for index, potion in enumerate(_sequence(reward.get("potion_ids"))):
        if index not in claimed_potion_indices and index not in skipped_potion_indices:
            labels.append(f"potion {potion}")
    return tuple(labels)


def _reward_new_claimed_labels(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[str, ...]:
    old = set(_reward_claimed_labels(before))
    return tuple(label for label in _reward_claimed_labels(after) if label not in old)


def _reward_new_skipped_labels(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[str, ...]:
    old = set(_reward_skipped_labels(before))
    return tuple(label for label in _reward_skipped_labels(after) if label not in old)


def _reward_skipped_labels(reward: Mapping[str, Any]) -> tuple[str, ...]:
    labels: list[str] = []
    if bool(reward.get("gold_skipped", False)):
        labels.append(f"gold {_int(reward.get('gold'))}")
    relic_id = _optional_str(reward.get("relic_id"))
    if relic_id and bool(reward.get("relic_skipped", False)):
        labels.append(f"relic {relic_id}")
    labels.extend(f"relic {item}" for item in _sequence(reward.get("skipped_relic_ids")))
    card_options = tuple(_sequence(reward.get("card_options")))
    if card_options and bool(reward.get("card_skipped", False)):
        labels.append("card choice [" + ", ".join(str(item) for item in card_options) + "]")
    labels.extend(
        f"card {label}"
        for label in _indexed_reward_values(
            _sequence(reward.get("card_ids")),
            reward.get("skipped_card_indices"),
        )
    )
    labels.extend(
        f"card group {label}"
        for label in _indexed_reward_group_values(
            _sequence(reward.get("card_option_groups")),
            reward.get("skipped_card_option_group_indices"),
        )
    )
    potion_id = _optional_str(reward.get("potion_id"))
    if potion_id and bool(reward.get("potion_skipped", False)):
        labels.append(f"potion {potion_id}")
    labels.extend(
        f"potion {label}"
        for label in _indexed_reward_values(
            _sequence(reward.get("potion_ids")),
            reward.get("skipped_potion_indices"),
        )
    )
    return tuple(labels)


def _indexed_reward_values(values: Sequence[Any], raw_indices: Any) -> tuple[str, ...]:
    labels: list[str] = []
    items = tuple(values)
    for raw_index in _sequence(raw_indices):
        index = _int(raw_index)
        labels.append(str(items[index]) if 0 <= index < len(items) else f"#{index}")
    return tuple(labels)


def _indexed_reward_group_values(values: Sequence[Any], raw_indices: Any) -> tuple[str, ...]:
    labels: list[str] = []
    items = tuple(values)
    for raw_index in _sequence(raw_indices):
        index = _int(raw_index)
        if 0 <= index < len(items):
            labels.append("[" + ", ".join(str(item) for item in _sequence(items[index])) + "]")
        else:
            labels.append(f"#{index}")
    return tuple(labels)


def _indexed_reward_labels(
    prefix: str,
    values: Sequence[Any],
    raw_indices: Any,
    *,
    bracket_values: bool = False,
) -> list[str]:
    labels: list[str] = []
    items = tuple(values)
    for raw_index in _sequence(raw_indices):
        index = _int(raw_index)
        if not 0 <= index < len(items):
            labels.append(f"{prefix}#{index}")
            continue
        value = items[index]
        if bracket_values:
            label = "[" + ", ".join(str(item) for item in _sequence(value)) + "]"
        else:
            label = str(value)
        labels.append(f"{prefix}#{index}:{label}")
    return labels


def _shop_line(shop: Mapping[str, Any]) -> str:
    items = [
        f"{item.get('slot_id')}={item.get('kind')}:{item.get('item_id')} "
        f"{_int(item.get('price'))}g{' sold' if item.get('purchased') else ''}"
        for item in (_mapping(raw) for raw in _sequence(shop.get("items")))
    ]
    return f"{len(items)} items; " + "; ".join(items[:8])


def _shop_available_summary(shop: Mapping[str, Any]) -> str:
    items = [
        _shop_item_label(item)
        for item in (_mapping(raw) for raw in _sequence(shop.get("items")))
        if not bool(item.get("purchased", False))
    ]
    return ", ".join(items[:8]) if items else "no purchasable items"


def _shop_item_label(item: Mapping[str, Any]) -> str:
    kind = str(item.get("kind") or "item").replace("_", " ")
    item_id = str(item.get("item_id") or item.get("slot_id") or "unknown")
    rarity = _optional_str(item.get("rarity"))
    rarity_text = f" {rarity}" if rarity else ""
    price = _int(item.get("price"))
    return f"{kind}{rarity_text} {item_id} for {price} gold"


def _shop_purchased_items(shop: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        item
        for item in (_mapping(raw) for raw in _sequence(shop.get("items")))
        if bool(item.get("purchased", False))
    )


def _shop_newly_purchased_items(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    before_slots = {
        str(item.get("slot_id"))
        for item in (_mapping(raw) for raw in _sequence(before.get("items")))
        if bool(item.get("purchased", False))
    }
    return tuple(
        item
        for item in _shop_purchased_items(after)
        if str(item.get("slot_id")) not in before_slots
    )


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


def _room_outcome_lines(
    action_type: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[str, ...]:
    if action_type != "choose_node":
        return ()

    phase_after = str(after.get("phase", ""))
    lines: list[str] = []
    if phase_after == "shop":
        shop = _mapping(after.get("shop"))
        if shop:
            lines.append(f"Shop opened with {_shop_available_summary(shop)}.")
            lines.append("Shop outcome so far: no purchases yet.")
    elif phase_after == "treasure":
        reward = _mapping(after.get("reward"))
        if reward:
            if _reward_has_available_loot(reward):
                lines.append(f"Treasure opened with {_reward_available_summary(reward)}.")
            else:
                reason = _mapping(reward.get("metadata")).get("empty_reason")
                suffix = f" ({reason})" if reason else ""
                lines.append(f"Treasure opened with no claimable loot{suffix}.")
    elif phase_after == "event":
        event = _mapping(after.get("event"))
        reward = _mapping(after.get("reward"))
        if event:
            lines.append(f"Event opened: {_event_line(event)}.")
        elif reward:
            lines.append(f"Event opened immediate reward: {_reward_available_summary(reward)}.")
    elif phase_after == "combat":
        lines.append("Room opened combat.")
    elif phase_after == "rest":
        lines.append("Rest site opened.")
    return tuple(lines)


def _reward_outcome_lines(
    action_type: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[str, ...]:
    before_reward = _mapping(before.get("reward"))
    after_reward = _mapping(after.get("reward"))
    if not before_reward:
        return ()

    lines: list[str] = []
    if action_type.startswith("take_reward"):
        gained = _reward_new_claimed_labels(before_reward, after_reward)
        if gained:
            lines.append(f"Reward pickup: took {', '.join(gained)}.")
    elif action_type == "skip_reward":
        skipped = _reward_new_skipped_labels(before_reward, after_reward)
        if skipped:
            lines.append(f"Reward skip: left {', '.join(skipped)}.")
    elif action_type == "proceed":
        phase_before = str(before.get("phase", ""))
        claimed = _reward_claimed_labels(before_reward)
        left = _reward_unclaimed_labels(before_reward)
        label = "Treasure" if phase_before == "treasure" else "Reward screen"
        if claimed:
            lines.append(f"{label} outcome: took {', '.join(claimed)}.")
        elif _reward_has_available_loot(before_reward):
            lines.append(f"{label} outcome: left without taking available loot.")
        else:
            lines.append(f"{label} outcome: no claimable loot was available.")
        if left:
            lines.append(f"{label} left behind: {', '.join(left)}.")
    return tuple(lines)


def _shop_outcome_lines(
    action_type: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[str, ...]:
    before_shop = _mapping(before.get("shop"))
    after_shop = _mapping(after.get("shop"))
    if not before_shop:
        return ()

    lines: list[str] = []
    if action_type == "shop_buy":
        newly_purchased = _shop_newly_purchased_items(before_shop, after_shop)
        if newly_purchased:
            lines.append(
                "Shop pickup: bought "
                + ", ".join(_shop_item_label(item) for item in newly_purchased)
                + "."
            )
        else:
            lines.append("Shop action did not buy an item.")
    elif action_type in {"shop_leave", "proceed"} and str(before.get("phase")) == "shop":
        bought = _shop_purchased_items(before_shop)
        if bought:
            lines.append(
                "Left shop after buying "
                + ", ".join(_shop_item_label(item) for item in bought)
                + "."
            )
        else:
            lines.append("Left shop without buying anything.")
    return tuple(lines)


def _event_outcome_lines(
    action_type: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    raw_events: Sequence[Any],
) -> tuple[str, ...]:
    if action_type != "choose_event":
        return ()

    lines: list[str] = []
    chosen = _chosen_event_from_events(raw_events)
    if chosen:
        lines.append(chosen)

    after_reward = _mapping(after.get("reward"))
    after_combat = _mapping(after.get("combat"))
    after_event = _mapping(after.get("event"))
    phase_after = str(after.get("phase", ""))
    if after_reward:
        lines.append(f"Event reward offered: {_reward_available_summary(after_reward)}.")
    elif after_combat:
        lines.append("Event outcome: started combat.")
    elif phase_after == "map":
        lines.append("Event outcome: completed with no pending reward.")
    elif after_event:
        resolved = after_event.get("resolved_option_id")
        if resolved:
            lines.append(f"Event outcome: option {resolved} resolved.")
        else:
            lines.append(f"Event outcome: moved to {_event_line(after_event)}.")
    return tuple(lines)


def _chosen_event_from_events(raw_events: Sequence[Any]) -> str | None:
    for event in (_mapping(raw) for raw in raw_events):
        if str(event.get("kind")) != "event_option_chosen":
            continue
        metadata = _mapping(event.get("metadata"))
        title = str(metadata.get("title") or event.get("target_id") or "unknown")
        source = str(event.get("source_id") or "event")
        target = str(event.get("target_id") or "")
        if target and target != title:
            return f"Event side chosen: {source} -> {title} ({target})."
        return f"Event side chosen: {source} -> {title}."
    return None


def _zone_count(combat: Mapping[str, Any], zone: str) -> int:
    return len(_sequence(combat.get(zone)))


def _compact_json(value: Mapping[str, Any]) -> str:
    if not value:
        return "{}"
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def _string_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in _sequence(value))


def _state_change_lines(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[str, ...]:
    before_player = _mapping(before.get("player"))
    after_player = _mapping(after.get("player"))
    if not before_player or not after_player:
        return ()

    lines: list[str] = []
    for label, key in (
        ("HP", "hp"),
        ("Max HP", "max_hp"),
        ("Block", "block"),
        ("Energy", "energy"),
        ("Gold", "gold"),
    ):
        old_value = _int(before_player.get(key))
        new_value = _int(after_player.get(key))
        if old_value != new_value:
            delta = new_value - old_value
            lines.append(f"{label} changed {delta:+d} ({old_value} -> {new_value}).")

    lines.extend(
        _inventory_delta_lines(
            "Relics",
            _sequence(before_player.get("relics")),
            _sequence(after_player.get("relics")),
        )
    )
    lines.extend(
        _inventory_delta_lines(
            "Potions",
            _sequence(before_player.get("potions")),
            _sequence(after_player.get("potions")),
        )
    )
    lines.extend(
        _mapping_value_delta_lines(
            "Player status",
            _mapping(before_player.get("statuses")),
            _mapping(after_player.get("statuses")),
        )
    )
    lines.extend(
        _mapping_value_delta_lines(
            "Player resource",
            _mapping(before_player.get("resources")),
            _mapping(after_player.get("resources")),
        )
    )
    lines.extend(
        _deck_delta_lines(
            _sequence(before_player.get("deck")),
            _sequence(after_player.get("deck")),
        )
    )
    return tuple(lines)


def _combat_change_lines(
    before_combat: Mapping[str, Any],
    after_combat: Mapping[str, Any],
) -> tuple[str, ...]:
    if not before_combat and not after_combat:
        return ()
    lines: list[str] = []
    before_player = _mapping(before_combat.get("player"))
    after_player = _mapping(after_combat.get("player"))
    if before_player and after_player:
        for label, key in (("Combat HP", "hp"), ("Combat block", "block"), ("Energy", "energy")):
            old_value = _int(before_player.get(key))
            new_value = _int(after_player.get(key))
            if old_value != new_value:
                lines.append(
                    f"{label} changed {new_value - old_value:+d} "
                    f"({old_value} -> {new_value})."
                )
        lines.extend(
            _mapping_value_delta_lines(
                "Combat player status",
                _mapping(before_player.get("statuses")),
                _mapping(after_player.get("statuses")),
            )
        )
        lines.extend(
            _mapping_value_delta_lines(
                "Combat player resource",
                _mapping(before_player.get("resources")),
                _mapping(after_player.get("resources")),
            )
        )
    lines.extend(_monster_change_lines(before_combat, after_combat))
    pile_changes: list[str] = []
    for label, zone in (
        ("hand", "hand"),
        ("draw", "draw_pile"),
        ("discard", "discard_pile"),
        ("exhaust", "exhaust_pile"),
    ):
        before_count = _zone_count(before_combat, zone)
        after_count = _zone_count(after_combat, zone)
        if before_count != after_count:
            pile_changes.append(f"{label} {before_count}->{after_count}")
    if pile_changes:
        lines.append(f"Combat piles changed: {', '.join(pile_changes)}.")
    return tuple(lines)


def _monster_change_lines(
    before_combat: Mapping[str, Any],
    after_combat: Mapping[str, Any],
) -> tuple[str, ...]:
    before_monsters = {
        _monster_change_key(index, monster): monster
        for index, monster in enumerate(
            _mapping(raw) for raw in _sequence(before_combat.get("monsters"))
        )
    }
    after_monsters = {
        _monster_change_key(index, monster): monster
        for index, monster in enumerate(
            _mapping(raw) for raw in _sequence(after_combat.get("monsters"))
        )
    }
    lines: list[str] = []
    for key in sorted(set(before_monsters) | set(after_monsters)):
        before = before_monsters.get(key, {})
        after = after_monsters.get(key, {})
        name = str(after.get("name") or before.get("name") or after.get("monster_id") or key)
        if before and not after:
            lines.append(f"Enemy left combat: {name}.")
            continue
        if after and not before:
            lines.append(f"Enemy entered combat: {name}.")
            continue
        changes: list[str] = []
        for label, field in (("HP", "hp"), ("block", "block")):
            old_value = _int(before.get(field))
            new_value = _int(after.get(field))
            if old_value != new_value:
                changes.append(f"{label} {old_value}->{new_value}")
        before_intent = _monster_intent_text(before)
        after_intent = _monster_intent_text(after)
        if before_intent != after_intent:
            changes.append(f"intent {before_intent}->{after_intent}")
        status_change = _mapping_value_delta_text(
            _mapping(before.get("statuses")),
            _mapping(after.get("statuses")),
        )
        if status_change:
            changes.append(f"status {status_change}")
        if changes:
            lines.append(f"Enemy {name} changed: {', '.join(changes)}.")
    return tuple(lines)


def _monster_change_key(index: int, monster: Mapping[str, Any]) -> str:
    return str(monster.get("instance_id") or monster.get("monster_id") or index)


def _mapping_value_delta_text(before: Mapping[str, Any], after: Mapping[str, Any]) -> str:
    changes: list[str] = []
    for key in sorted(set(before) | set(after)):
        old_value = before.get(key, 0)
        new_value = after.get(key, 0)
        if old_value != new_value:
            changes.append(f"{key} {old_value}->{new_value}")
    return ", ".join(changes)


def _event_effect_lines(raw_events: Sequence[Any]) -> tuple[str, ...]:
    lines: list[str] = []
    for event in (_mapping(raw) for raw in raw_events):
        kind = str(event.get("kind", ""))
        metadata = _mapping(event.get("metadata"))
        target_id = _optional_str(event.get("target_id"))
        amount = _int(event.get("amount"))
        if kind == "reward_card_taken":
            card_id = str(metadata.get("card_id") or target_id or "unknown")
            lines.append(f"Reward card taken: {card_id}.")
        elif kind == "reward_relic_taken":
            lines.append(f"Reward relic taken: {target_id or 'unknown'}.")
        elif kind == "reward_potion_taken":
            lines.append(f"Reward potion taken: {target_id or 'unknown'}.")
        elif kind == "reward_gold_taken":
            lines.append(f"Reward gold taken: {amount}.")
        elif kind == "reward_item_skipped":
            skipped_kind = str(metadata.get("kind") or "item")
            skipped_id = str(event.get("target_id") or metadata.get("target_id") or skipped_kind)
            lines.append(f"Reward skipped: {skipped_kind} {skipped_id}.")
        elif kind == "reward_card_removed":
            card_id = str(metadata.get("card_id") or target_id or "unknown")
            lines.append(f"Reward removed card: {card_id}.")
        elif kind == "reward_gold_generated":
            lines.append(f"Reward generated gold: {amount}.")
        elif kind == "reward_relic_generated":
            relic_id = str(target_id or metadata.get("relic_id") or "unknown")
            rarity = metadata.get("relic_rarity") or metadata.get("rarity")
            suffix = f" ({rarity})" if rarity else ""
            lines.append(f"Reward generated relic: {relic_id}{suffix}.")
        elif kind == "reward_cards_generated":
            cards = _string_list(metadata.get("card_ids") or metadata.get("cards"))
            lines.append(
                "Reward generated card choices: "
                + (", ".join(cards) if cards else _compact_json(metadata))
                + "."
            )
        elif kind == "reward_card_group_generated":
            cards = _string_list(metadata.get("card_ids") or metadata.get("cards"))
            lines.append(
                "Reward generated extra card group: "
                + (", ".join(cards) if cards else _compact_json(metadata))
                + "."
            )
        elif kind == "reward_card_generated":
            card_id = str(target_id or metadata.get("card_id") or "unknown")
            lines.append(f"Reward generated fixed card: {card_id}.")
        elif kind == "reward_potion_generated":
            potion_id = str(target_id or metadata.get("potion_id") or "unknown")
            lines.append(f"Reward generated potion: {potion_id}.")
        elif kind == "shop_item_bought":
            item_id = str(metadata.get("item_id") or target_id or "unknown")
            item_kind = str(metadata.get("item_kind") or "item").replace("_", " ")
            price = _int(metadata.get("price"))
            lines.append(f"Shop bought {item_kind} {item_id} for {price} gold.")
            if metadata.get("restocked_item_id"):
                lines.append(f"Shop restocked slot with {metadata['restocked_item_id']}.")
        elif kind == "shop_card_removed":
            card_id = str(metadata.get("removed_card_id") or target_id or "unknown")
            price = _int(metadata.get("price"))
            lines.append(f"Shop removed card {card_id} for {price} gold.")
        elif kind == "shop_card_remove_blocked":
            card_id = str(metadata.get("card_id") or target_id or "unknown")
            reason = str(metadata.get("reason") or "blocked")
            lines.append(f"Shop card removal blocked for {card_id}: {reason}.")
        elif kind == "shop_left":
            lines.append("Shop closed.")
        elif kind == "shop_ready":
            lines.append("Shop inventory generated.")
        elif kind == "foul_potion_thrown_at_merchant":
            lines.append(f"Foul Potion thrown at merchant: gained {amount} gold.")
        elif kind == "foul_potion_thrown_at_fake_merchant":
            lines.append("Foul Potion thrown at fake merchant: combat started.")
        elif kind == "treasure_ready":
            lines.append("Treasure chest opened.")
        elif kind == "treasure_opened":
            lines.append("Treasure room finished.")
        elif kind == "treasure_chest_empty":
            reason = str(metadata.get("reason") or metadata.get("empty_reason") or "empty")
            lines.append(f"Treasure chest was empty: {reason}.")
        elif kind == "event_gold_lost":
            lines.append(f"Event cost paid: lost {amount} gold.")
        elif kind == "event_gold_gained":
            lines.append(f"Event granted {amount} gold.")
        elif kind == "event_hp_lost":
            lines.append(f"Event cost paid: lost {amount} HP.")
        elif kind == "event_healed":
            lines.append(f"Event healed {amount} HP.")
        elif kind == "event_max_hp_lost":
            lines.append(f"Event reduced Max HP by {amount}.")
        elif kind == "event_max_hp_gained":
            lines.append(f"Event increased Max HP by {amount}.")
        elif kind == "event_max_hp_set":
            lines.append(f"Event set Max HP to {amount}.")
        elif kind in {"event_card_added", "event_random_card_added"}:
            card_id = str(metadata.get("card_id") or target_id or "unknown")
            lines.append(f"Event added card: {card_id}.")
        elif kind in {"event_card_removed", "event_card_removed_random"}:
            card_id = str(metadata.get("card_id") or target_id or "unknown")
            lines.append(f"Event removed card: {card_id}.")
        elif kind == "event_card_transformed":
            old_card = str(metadata.get("old_card_id") or metadata.get("card_id") or target_id)
            new_card = str(metadata.get("new_card_id") or metadata.get("transformed_card_id"))
            if new_card and new_card != "None":
                lines.append(f"Event transformed card: {old_card} -> {new_card}.")
            else:
                lines.append(f"Event transformed card: {old_card}.")
        elif kind == "event_card_transform_choice_required":
            lines.append(f"Event requires choosing {amount} card(s) to transform.")
        elif kind == "event_relic_obtained":
            lines.append(f"Event relic obtained: {target_id or 'unknown'}.")
        elif kind == "event_relic_duplicate_skipped":
            lines.append(f"Event skipped duplicate relic: {target_id or 'unknown'}.")
        elif kind == "event_potion_obtained":
            lines.append(f"Event potion obtained: {target_id or 'unknown'}.")
        elif kind == "event_potion_skipped_no_slot":
            lines.append(
                "Event potion skipped because no slot was open: "
                f"{target_id or 'unknown'}."
            )
        elif kind == "event_card_reward_blocked":
            lines.append("Event card reward was blocked.")
        elif kind == "event_delayed_reward_scheduled":
            lines.append(f"Event scheduled delayed reward: {_compact_json(metadata)}.")
        elif kind == "event_delayed_reward_blocked":
            lines.append(f"Event delayed reward blocked: {_compact_json(metadata)}.")
        elif kind == "event_player_died":
            lines.append("Event outcome: player died.")
        elif kind == "relic_gold_gained":
            source = event.get("source_id") or target_id or ""
            lines.append(f"Relic {source} granted {amount} gold.")
        elif kind == "relic_gold_set":
            source = event.get("source_id") or ""
            lines.append(f"Relic {source} set gold to {metadata.get('gold')}.")
        elif kind == "relic_max_hp_changed":
            lines.append(f"Relic {event.get('source_id') or ''} changed Max HP by {amount}.")
        elif kind == "relic_healed":
            lines.append(f"Relic {event.get('source_id') or ''} healed {amount} HP.")
        elif kind == "relic_potion_slots_changed":
            lines.append(f"Relic {event.get('source_id') or ''} changed potion slots by {amount}.")
        elif kind in {"relic_deck_card_added", "relic_potion_obtained"}:
            content_id = str(metadata.get("card_id") or target_id or "unknown")
            lines.append(f"Relic {event.get('source_id') or ''} added {content_id}.")
        elif kind == "relic_deck_card_upgraded":
            card_id = str(metadata.get("card_id") or target_id or "unknown")
            lines.append(f"Relic {event.get('source_id') or ''} upgraded {card_id}.")
        elif kind == "relic_deck_card_removed":
            card_id = str(metadata.get("card_id") or target_id or "unknown")
            lines.append(f"Relic {event.get('source_id') or ''} removed {card_id}.")
        elif kind == "relic_deck_card_transformed":
            old_card = str(metadata.get("old_card_id") or metadata.get("card_id") or target_id)
            new_card = str(metadata.get("new_card_id") or metadata.get("transformed_card_id"))
            source = event.get("source_id") or ""
            lines.append(f"Relic {source} transformed {old_card} -> {new_card}.")
        elif kind == "relic_counter_changed":
            lines.append(f"Relic {event.get('source_id') or ''} counter changed to {amount}.")
        elif kind == "relic_deck_choice_required":
            source = event.get("source_id") or ""
            lines.append(f"Relic {source} requires choosing {amount} card(s).")
        elif kind == "room_entered":
            lines.append(
                f"Entered {metadata.get('room_kind', 'room')} room "
                f"on act {metadata.get('act', '?')} floor {metadata.get('floor', '?')}."
            )
        elif kind == "room_completed":
            lines.append(
                f"Completed {metadata.get('room_kind', 'room')} room "
                f"on act {metadata.get('act', '?')} floor {metadata.get('floor', '?')}."
            )
        elif kind == "event_resolved":
            lines.append("Event room finished.")
        elif kind == "reward_skipped":
            lines.append("Reward screen finished with remaining optional rewards skipped.")
    return tuple(lines)


def _inventory_delta_lines(
    label: str,
    before: Sequence[Any],
    after: Sequence[Any],
) -> tuple[str, ...]:
    gained = _ordered_counter_delta(after, before)
    lost = _ordered_counter_delta(before, after)
    lines: list[str] = []
    if gained:
        lines.append(f"{label} gained: {', '.join(gained)}.")
    if lost:
        lines.append(f"{label} lost: {', '.join(lost)}.")
    return tuple(lines)


def _ordered_counter_delta(after: Sequence[Any], before: Sequence[Any]) -> tuple[str, ...]:
    remaining = Counter(str(item) for item in before)
    delta: list[str] = []
    for item in (str(raw) for raw in after):
        if remaining[item] > 0:
            remaining[item] -= 1
        else:
            delta.append(item)
    return tuple(delta)


def _mapping_value_delta_lines(
    label: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[str, ...]:
    changes: list[str] = []
    for key in sorted(set(before) | set(after)):
        old_value = before.get(key, 0)
        new_value = after.get(key, 0)
        if old_value != new_value:
            changes.append(f"{key} {old_value} -> {new_value}")
    if not changes:
        return ()
    return (f"{label} changed: {', '.join(changes)}.",)


def _deck_delta_lines(before_raw: Sequence[Any], after_raw: Sequence[Any]) -> tuple[str, ...]:
    before_cards = tuple(_mapping(card) for card in before_raw)
    after_cards = tuple(_mapping(card) for card in after_raw)
    before_by_id = _cards_by_instance_id(before_cards)
    after_by_id = _cards_by_instance_id(after_cards)
    lines: list[str] = []

    gained = [
        _card_change_label(card)
        for card in after_cards
        if str(card.get("instance_id", "")) not in before_by_id
    ]
    lost = [
        _card_change_label(card)
        for card in before_cards
        if str(card.get("instance_id", "")) not in after_by_id
    ]
    if gained:
        lines.append(f"Deck gained: {', '.join(gained)}.")
    if lost:
        lines.append(f"Deck lost: {', '.join(lost)}.")

    transformed: list[str] = []
    upgraded: list[str] = []
    for instance_id, before_card in before_by_id.items():
        after_card = after_by_id.get(instance_id)
        if after_card is None:
            continue
        before_card_id = str(before_card.get("card_id", ""))
        after_card_id = str(after_card.get("card_id", ""))
        if before_card_id != after_card_id:
            transformed.append(
                f"{_card_change_label(before_card)} -> {_card_change_label(after_card)}"
            )
            continue
        if bool(before_card.get("upgraded", False)) != bool(
            after_card.get("upgraded", False)
        ):
            upgraded.append(_card_change_label(after_card))
    if transformed:
        lines.append(f"Deck transformed: {', '.join(transformed)}.")
    if upgraded:
        lines.append(f"Deck upgraded: {', '.join(upgraded)}.")
    return tuple(lines)


def _cards_by_instance_id(cards: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(card.get("instance_id", "")): card
        for card in cards
        if str(card.get("instance_id", ""))
    }


def _card_change_label(card: Mapping[str, Any]) -> str:
    card_id = str(card.get("card_id") or "unknown")
    name = str(card.get("name") or card_id)
    suffix = "+" if bool(card.get("upgraded", False)) else ""
    if name == card_id:
        return f"{name}{suffix}"
    return f"{name}{suffix} ({card_id})"


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
        "metadata": dict(_mapping(event.get("metadata"))),
        "options": [
            {
                "option_id": str(option.get("option_id", "")),
                "title": str(option.get("title", "")),
                "description": str(option.get("description", "")),
                "disabled": bool(option.get("disabled", False)),
                "metadata": dict(_mapping(option.get("metadata"))),
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
        "gold_skipped": bool(reward.get("gold_skipped", False)),
        "relic_id": _optional_str(reward.get("relic_id")),
        "relic_claimed": bool(reward.get("relic_claimed", False)),
        "relic_skipped": bool(reward.get("relic_skipped", False)),
        "relic_ids": list(_sequence(reward.get("relic_ids"))),
        "claimed_relic_ids": list(_sequence(reward.get("claimed_relic_ids"))),
        "skipped_relic_ids": list(_sequence(reward.get("skipped_relic_ids"))),
        "card_options": list(_sequence(reward.get("card_options"))),
        "card_claimed": bool(reward.get("card_claimed", False)),
        "card_skipped": bool(reward.get("card_skipped", False)),
        "card_option_groups": [
            list(_sequence(group)) for group in _sequence(reward.get("card_option_groups"))
        ],
        "claimed_card_option_group_indices": list(
            _sequence(reward.get("claimed_card_option_group_indices"))
        ),
        "skipped_card_option_group_indices": list(
            _sequence(reward.get("skipped_card_option_group_indices"))
        ),
        "card_ids": list(_sequence(reward.get("card_ids"))),
        "claimed_card_indices": list(_sequence(reward.get("claimed_card_indices"))),
        "skipped_card_indices": list(_sequence(reward.get("skipped_card_indices"))),
        "potion_id": _optional_str(reward.get("potion_id")),
        "potion_claimed": bool(reward.get("potion_claimed", False)),
        "potion_skipped": bool(reward.get("potion_skipped", False)),
        "potion_ids": list(_sequence(reward.get("potion_ids"))),
        "claimed_potion_indices": list(_sequence(reward.get("claimed_potion_indices"))),
        "skipped_potion_indices": list(_sequence(reward.get("skipped_potion_indices"))),
        "metadata": dict(_mapping(reward.get("metadata"))),
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
    hit_count = max(1, _int(monster.get("hit_count")))
    total_damage = _int(monster.get("intent_damage"))
    per_hit_damage = max(0, total_damage // hit_count)
    return {
        "monster_id": str(monster.get("monster_id", "")),
        "name": str(monster.get("name", "")),
        "hp": _int(monster.get("hp")),
        "max_hp": _int(monster.get("max_hp")),
        "block": _int(monster.get("block")),
        "intent": _optional_str(monster.get("intent")),
        "intent_damage": total_damage,
        "intent_damage_per_hit": per_hit_damage,
        "intent_damage_total": total_damage,
        "hit_count": hit_count,
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


def _skip_reward_action_summary(reward: Mapping[str, Any], target_id: str | None) -> str:
    if target_id == "reward:gold":
        return f"Skip reward gold ({_int(reward.get('gold'))})"
    if target_id == "reward:card_options":
        cards = ", ".join(str(card_id) for card_id in _sequence(reward.get("card_options")))
        return f"Skip reward card choices ({cards})" if cards else "Skip reward card choices"
    if target_id == "reward:relic" or str(target_id or "").startswith("reward:relic:"):
        return f"Skip reward relic {_reward_relic_for_target(reward, target_id)}"
    if target_id == "reward:potion" or str(target_id or "").startswith("reward:potion:"):
        return f"Skip reward potion {_reward_potion_for_target(reward, target_id)}"
    if str(target_id or "").startswith("reward:fixed_card:"):
        return f"Skip reward fixed card {_reward_card_for_target(reward, target_id)}"
    if str(target_id or "").startswith("reward:card_group:"):
        return f"Skip reward card group {_reward_card_group_for_target(reward, target_id)}"
    return f"Skip reward {target_id or 'item'}"


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


def _reward_card_group_for_target(reward: Mapping[str, Any], target_id: str | None) -> str:
    parts = (target_id or "").split(":")
    if len(parts) == 3 and parts[:2] == ["reward", "card_group"]:
        groups = _sequence(reward.get("card_option_groups"))
        group_index = _int(parts[2])
        if 0 <= group_index < len(groups):
            group = _sequence(groups[group_index])
            return "[" + ", ".join(str(card_id) for card_id in group) + "]"
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


def _float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
