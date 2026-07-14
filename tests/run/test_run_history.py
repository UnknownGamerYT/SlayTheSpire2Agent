from __future__ import annotations

import json
from html import escape as html_escape

from typer.testing import CliRunner

from sts2sim import legal_actions, new_run, step
from sts2sim.agents import play_strategic_run
from sts2sim.cli.app import app
from sts2sim.engine import MapEdgeState, MapNodeState, MapState, RoomKind, RunPhase
from sts2sim.engine.models import ShopItemState, ShopState
from sts2sim.history import (
    append_history_step,
    record_history_step,
    run_history_html,
    run_history_map_text,
    run_history_summary,
    run_history_summary_html,
    run_history_summary_text,
    start_run_history,
    write_run_history_html,
    write_run_history_map_text,
    write_run_history_summary,
    write_run_history_summary_html,
    write_run_history_summary_text,
)
from sts2sim.learning import collect_random_rollouts


def test_history_records_readable_ancient_and_map_steps() -> None:
    state = new_run(seed=80, character_id="TEST", ascension=0)
    history = start_run_history(state, policy="test")

    ancient_action = next(
        action for action in legal_actions(state) if action.type == "choose_ancient"
    )
    next_state = step(state, ancient_action)
    ancient_step = record_history_step(
        step_index=0,
        before_state=state,
        action=ancient_action,
        after_state=next_state,
    )
    history = append_history_step(history, ancient_step, next_state)

    map_action = next(
        action for action in legal_actions(next_state) if action.type == "choose_node"
    )
    after_map = step(next_state, map_action)
    map_step = record_history_step(
        step_index=1,
        before_state=next_state,
        action=map_action,
        after_state=after_map,
    )
    history = append_history_step(history, map_step, after_map)

    assert "Choose ancient option" in history.steps[0].action_summary
    assert history.steps[0].events
    assert "Choose map node" in history.steps[1].action_summary
    assert history.steps[1].context_before["map"]["reachable"]
    assert history.summary["nodes_chosen"] == 1


def test_history_renders_html_and_text_map_for_chosen_path(tmp_path) -> None:
    state = new_run(seed=80, character_id="TEST", ascension=0)
    history = start_run_history(state, policy="test")
    ancient_action = next(
        action for action in legal_actions(state) if action.type == "choose_ancient"
    )
    after_ancient = step(state, ancient_action)
    history = append_history_step(
        history,
        record_history_step(
            step_index=0,
            before_state=state,
            action=ancient_action,
            after_state=after_ancient,
        ),
        after_ancient,
    )
    map_action = next(
        action for action in legal_actions(after_ancient) if action.type == "choose_node"
    )
    after_map = step(after_ancient, map_action)
    history = append_history_step(
        history,
        record_history_step(
            step_index=1,
            before_state=after_ancient,
            action=map_action,
            after_state=after_map,
        ),
        after_map,
    )

    html = run_history_html(history)
    text_map = run_history_map_text(history)
    html_path = tmp_path / "history.html"
    map_path = tmp_path / "history_map.txt"
    write_run_history_html(history, html_path)
    write_run_history_map_text(history, map_path)

    assert "Map Path" in html
    assert "Timeline" in html
    assert "Choose map node" in html
    assert f":{map_action.target_id}_" in text_map
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert "Legend:" in map_path.read_text(encoding="utf-8")


def test_history_records_combat_card_context_and_events() -> None:
    state = _enter_test_combat()
    play_action = next(action for action in legal_actions(state) if action.type == "play_card")
    next_state = step(state, play_action)

    history_step = record_history_step(
        step_index=0,
        before_state=state,
        action=play_action,
        after_state=next_state,
    )

    assert history_step.action_summary.startswith("Play ")
    assert history_step.context_before["combat"]["hand"]
    assert history_step.context_after["combat"]
    assert history_step.events


def test_history_groups_same_turn_combat_card_actions() -> None:
    state = _enter_test_combat()
    history = start_run_history(state, policy="test")
    first_action = next(action for action in legal_actions(state) if action.type == "play_card")
    after_first = step(state, first_action)
    history = append_history_step(
        history,
        record_history_step(
            step_index=0,
            before_state=state,
            action=first_action,
            after_state=after_first,
        ),
        after_first,
    )
    second_action = next(
        action for action in legal_actions(after_first) if action.type == "play_card"
    )
    after_second = step(after_first, second_action)
    history = append_history_step(
        history,
        record_history_step(
            step_index=1,
            before_state=after_first,
            action=second_action,
            after_state=after_second,
        ),
        after_second,
    )

    html = run_history_html(history)

    assert "Combat Turn 1" in html
    assert "Cards Played This Turn" in html
    assert "Step Details" in html
    assert html.index(html_escape(history.steps[0].action_summary)) < html.index(
        html_escape(history.steps[1].action_summary)
    )
    assert "<h3>Step 0:" not in html


def test_history_renders_multihit_intent_as_per_hit_with_total() -> None:
    state = _enter_test_combat()
    assert state.combat is not None
    monster = state.combat.monsters[0].model_copy(
        update={
            "name": "Seapunk",
            "intent": "Attack",
            "intent_damage": 8,
            "hit_count": 4,
        }
    )
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )

    history = start_run_history(state, policy="test")
    html = run_history_html(history)

    assert "Attack 2x4 (total incoming 8)" in html
    assert "Attack 8x4" not in html
    monster_summary = history.initial["combat"]["monsters"][0]
    assert monster_summary["intent_damage"] == 8
    assert monster_summary["intent_damage_per_hit"] == 2
    assert monster_summary["intent_damage_total"] == 8


def test_history_reports_deck_gain_from_forced_relic_card_reward() -> None:
    state, ancient_action, after_ancient = _find_forced_ancient_card_reward()
    history = start_run_history(state, policy="test")
    history = append_history_step(
        history,
        record_history_step(
            step_index=0,
            before_state=state,
            action=ancient_action,
            after_state=after_ancient,
        ),
        after_ancient,
    )

    reward = after_ancient.reward
    assert reward is not None
    take_card = next(
        action for action in legal_actions(after_ancient) if action.type == "take_reward_card"
    )
    chosen_card_id = _reward_card_id_for_action(reward.card_options, take_card.target_id)
    after_card = step(after_ancient, take_card)
    card_step = record_history_step(
        step_index=1,
        before_state=after_ancient,
        action=take_card,
        after_state=after_card,
    )
    history = append_history_step(history, card_step, after_card)

    html = run_history_html(history)
    assert "Deck gained:" in html
    assert chosen_card_id in html
    assert card_step.context_after["reward"]["card_claimed"] is True
    assert (
        card_step.context_after["player"]["deck_count"]
        == card_step.context_before["player"]["deck_count"] + 1
    )
    assert any(
        event["kind"] == "reward_card_taken"
        and event["metadata"]["card_id"] == chosen_card_id
        for event in card_step.events
    )


def test_history_reports_event_side_and_reward_outcome() -> None:
    state = new_run(
        seed=90,
        character_id="TEST",
        ascension=0,
        source_data={
            "event_id": "WAR_HISTORIAN_REPY",
            "deck": [
                {
                    "card_id": "lantern_key",
                    "name": "Lantern Key",
                    "type": "status",
                    "cost": -1,
                    "target": "self",
                }
            ],
            "relic_pool": (
                {"id": "COMMON_RELIC", "rarity_key": "Common", "pool": "shared"},
                {"id": "UNCOMMON_RELIC", "rarity_key": "Uncommon", "pool": "shared"},
            ),
            "potion_pool": ("fire_potion", "skill_potion"),
        },
    )
    state = _force_next_room(
        state.model_copy(update={"phase": RunPhase.MAP, "ancient": None}),
        RoomKind.EVENT,
    )
    state = step(state, _action(state, "choose_node", "target"))
    event_action = _action(state, "choose_event", "unlock_chest")
    after_event = step(state, event_action)

    history = start_run_history(state, policy="test")
    history = append_history_step(
        history,
        record_history_step(
            step_index=0,
            before_state=state,
            action=event_action,
            after_state=after_event,
        ),
        after_event,
    )
    html = run_history_html(history)

    assert "Event side chosen:" in html
    assert "unlock_chest" in html
    assert "Event reward offered:" in html
    assert "relic" in html
    assert "potion" in html


def test_history_reports_shop_purchase_and_empty_leave() -> None:
    state = _direct_shop_state()
    buy_action = _action(state, "shop_buy", "shop:0")
    after_buy = step(state, buy_action)

    history = start_run_history(state, policy="test")
    history = append_history_step(
        history,
        record_history_step(
            step_index=0,
            before_state=state,
            action=buy_action,
            after_state=after_buy,
        ),
        after_buy,
    )
    html = run_history_html(history)

    assert "Shop pickup: bought relic common anchor for 100 gold." in html
    assert "Shop bought relic anchor for 100 gold." in html
    assert "Relics gained: anchor." in html

    leave_state = _direct_shop_state()
    leave_action = _action(leave_state, "shop_leave")
    after_leave = step(leave_state, leave_action)
    leave_step = record_history_step(
        step_index=0,
        before_state=leave_state,
        action=leave_action,
        after_state=after_leave,
    )

    assert "Left shop without buying anything." in "\n".join(
        run_history_html(
            append_history_step(
                start_run_history(leave_state, policy="test"),
                leave_step,
                after_leave,
            )
        ).splitlines()
    )


def test_history_reports_treasure_available_taken_and_left() -> None:
    state = _treasure_map_state()
    enter_action = _action(state, "choose_node", "target")
    treasure_state = step(state, enter_action)
    assert treasure_state.reward is not None
    relic_id = treasure_state.reward.relic_id

    history = start_run_history(state, policy="test")
    history = append_history_step(
        history,
        record_history_step(
            step_index=0,
            before_state=state,
            action=enter_action,
            after_state=treasure_state,
        ),
        treasure_state,
    )

    after_gold = step(treasure_state, _action(treasure_state, "take_reward_gold", "reward:gold"))
    history = append_history_step(
        history,
        record_history_step(
            step_index=1,
            before_state=treasure_state,
            action=_action(treasure_state, "take_reward_gold", "reward:gold"),
            after_state=after_gold,
        ),
        after_gold,
    )
    after_relic = step(after_gold, _action(after_gold, "take_reward_relic", "reward:relic"))
    history = append_history_step(
        history,
        record_history_step(
            step_index=2,
            before_state=after_gold,
            action=_action(after_gold, "take_reward_relic", "reward:relic"),
            after_state=after_relic,
        ),
        after_relic,
    )
    proceed_action = _action(after_relic, "proceed")
    after_proceed = step(after_relic, proceed_action)
    history = append_history_step(
        history,
        record_history_step(
            step_index=3,
            before_state=after_relic,
            action=proceed_action,
            after_state=after_proceed,
        ),
        after_proceed,
    )

    html = run_history_html(history)
    assert "Treasure opened with" in html
    assert "Reward pickup: took gold" in html
    assert f"Reward pickup: took relic {relic_id}" in html
    assert "Treasure outcome: took" in html
    assert "Treasure left behind" not in html


def test_history_writes_short_route_summary_with_links(tmp_path) -> None:
    state = _treasure_map_state()
    enter_action = _action(state, "choose_node", "target")
    treasure_state = step(state, enter_action)
    after_gold = step(treasure_state, _action(treasure_state, "take_reward_gold", "reward:gold"))
    after_relic = step(after_gold, _action(after_gold, "take_reward_relic", "reward:relic"))
    after_proceed = step(after_relic, _action(after_relic, "proceed"))

    history = start_run_history(state, policy="test")
    previous = state
    for index, (action, current) in enumerate(
        (
            (enter_action, treasure_state),
            (_action(treasure_state, "take_reward_gold", "reward:gold"), after_gold),
            (_action(after_gold, "take_reward_relic", "reward:relic"), after_relic),
            (_action(after_relic, "proceed"), after_proceed),
        )
    ):
        history = append_history_step(
            history,
            record_history_step(
                step_index=index,
                before_state=previous,
                action=action,
                after_state=current,
            ),
            current,
        )
        previous = current

    links = {
        "history": "full.html",
        "history_json": "full.json",
        "map": "map.txt",
    }
    summary = run_history_summary(history, links=links)
    text = run_history_summary_text(history, links=links)
    html = run_history_summary_html(history, links=links)
    json_path = tmp_path / "summary.json"
    text_path = tmp_path / "summary.txt"
    html_path = tmp_path / "summary.html"
    write_run_history_summary(history, json_path, links=links)
    write_run_history_summary_text(history, text_path, links=links)
    write_run_history_summary_html(history, html_path, links=links)

    assert summary["nodes"][0]["kind"] == "treasure"
    assert summary["nodes"][0]["gold_gained"] > 0
    assert summary["nodes"][0]["relics_gained"]
    assert summary["nodes"][0]["links"]["history"] == "full.html#step-0"
    assert "Short run summary" in text
    assert "Took: gold" in text
    assert "open detailed replay" in html
    assert "full.html#step-0" in html
    assert json.loads(json_path.read_text(encoding="utf-8"))["nodes"][0]["kind"] == "treasure"
    assert "Replay: full.html#step-0" in text_path.read_text(encoding="utf-8")
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_strategic_run_returns_and_writes_history(tmp_path) -> None:
    history_path = tmp_path / "history.json"

    result = play_strategic_run(
        seed=81,
        character_id="TEST",
        ascension=0,
        max_steps=2,
        history_path=history_path,
    )

    assert result["history"]["steps"]
    assert history_path.exists()
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    assert payload["policy"] == "strategic_v0"
    assert payload["steps"][0]["action_summary"]


def test_strategic_run_cli_writes_history_output(tmp_path) -> None:
    history_path = tmp_path / "cli_history.json"

    result = CliRunner().invoke(
        app,
        [
            "play-strategic-run",
            "--seed",
            "84",
            "--character",
            "TEST",
            "--max-steps",
            "1",
            "--history-output",
            str(history_path),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(history_path.read_text(encoding="utf-8"))["steps"][0]["action_summary"]


def test_random_rollout_includes_readable_history_by_default() -> None:
    result = collect_random_rollouts(runs=1, max_steps=2, start_seed=82, character_id="TEST")

    history = result.runs[0].history
    assert history is not None
    assert history["summary"]["steps_taken"] == result.runs[0].steps_taken
    assert history["steps"][0]["action_summary"]


def _enter_test_combat():
    state = new_run(
        seed=83,
        character_id="TEST",
        ascension=0,
        source_data={
            "deck": (
                {
                    "id": "TEST_STRIKE",
                    "name": "Test Strike",
                    "type": "Attack",
                    "target": "AnyEnemy",
                    "cost": 1,
                    "damage": 6,
                },
                {
                    "id": "TEST_DEFEND",
                    "name": "Test Defend",
                    "type": "Skill",
                    "target": "Self",
                    "cost": 1,
                    "block": 5,
                },
            ),
            "flags": {"draw_per_turn": 2},
        },
    )
    start = MapNodeState(node_id="start", act=state.act, floor=0, lane=0, kind=RoomKind.START)
    target = MapNodeState(node_id="target", act=state.act, floor=1, lane=0, kind=RoomKind.MONSTER)
    game_map = MapState(
        act=state.act,
        nodes=(start, target),
        edges=(MapEdgeState(from_id=start.node_id, to_id=target.node_id),),
        current_node_id=start.node_id,
        completed_node_ids=(start.node_id,),
    )
    state = state.model_copy(update={"phase": RunPhase.MAP, "map": game_map, "floor": 0})
    return step(
        state,
        next(action for action in legal_actions(state) if action.type == "choose_node"),
    )


def _action(state, action_type: str, target_id: str | None = None):
    return next(
        action
        for action in legal_actions(state)
        if action.type == action_type
        and (target_id is None or action.target_id == target_id)
    )


def _force_next_room(state, room_kind: RoomKind):
    start = MapNodeState(node_id="start", act=state.act, floor=0, lane=0, kind=RoomKind.START)
    target = MapNodeState(node_id="target", act=state.act, floor=1, lane=0, kind=room_kind)
    game_map = MapState(
        act=state.act,
        nodes=(start, target),
        edges=(MapEdgeState(from_id=start.node_id, to_id=target.node_id),),
        current_node_id=start.node_id,
        completed_node_ids=(start.node_id,),
    )
    return state.model_copy(update={"phase": RunPhase.MAP, "map": game_map, "floor": 0})


def _direct_shop_state():
    state = new_run(seed=91, character_id="TEST", ascension=0)
    shop = ShopState(
        node_id="target",
        items=(
            ShopItemState(
                slot_id="shop:0",
                item_id="anchor",
                kind="relic",
                rarity="common",
                price=100,
                base_price=100,
            ),
        ),
    )
    return _force_next_room(
        state.model_copy(
            update={
                "phase": RunPhase.SHOP,
                "ancient": None,
                "shop": shop,
                "player": state.player.model_copy(update={"gold": 300}),
            }
        ),
        RoomKind.SHOP,
    ).model_copy(update={"phase": RunPhase.SHOP, "shop": shop})


def _treasure_map_state():
    state = new_run(
        seed=92,
        character_id="TEST",
        ascension=0,
        source_data={
            "treasure_relic_pool": (
                {"id": "ANCHOR", "name": "Anchor", "rarity_key": "Common", "pool": "shared"},
            ),
            "map_floors": 4,
            "map_width": 1,
        },
    )
    return _force_next_room(
        state.model_copy(update={"phase": RunPhase.MAP, "ancient": None}),
        RoomKind.TREASURE,
    )


def _find_forced_ancient_card_reward():
    for seed in range(500):
        state = new_run(seed=seed, character_id="TEST", ascension=0)
        for ancient_action in legal_actions(state):
            if ancient_action.type != "choose_ancient":
                continue
            after_ancient = step(state, ancient_action)
            reward = after_ancient.reward
            if (
                reward is not None
                and reward.forced
                and reward.card_options
                and len(after_ancient.relics) > len(state.relics)
            ):
                return state, ancient_action, after_ancient
    raise AssertionError("No forced ancient relic card reward found in searched seeds")


def _reward_card_id_for_action(card_options, target_id: str | None) -> str:
    parts = (target_id or "").split(":")
    assert len(parts) == 3
    assert parts[:2] == ["reward", "card"]
    return str(card_options[int(parts[2])])
