from __future__ import annotations

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import RunPhase
from sts2sim.mechanics.campfire import rest_heal_amount


def _action(state, action_type: str, target_id: str | None = None):
    return next(
        action
        for action in legal_actions(state)
        if action.type == action_type and (target_id is None or action.target_id == target_id)
    )


def _card(state, instance_id: str):
    return next(card for card in state.master_deck if card.instance_id == instance_id)


def _play_card(state, instance_id: str):
    return next(
        action
        for action in legal_actions(state)
        if action.type == "play_card" and action.card_instance_id == instance_id
    )


def _enter_campfire(*, ascension: int = 0):
    state = new_run(
        seed=17,
        character_id="TEST",
        ascension=ascension,
        source_data={
            "max_acts": 1,
            "map_floors": 4,
            "map_width": 1,
            "deck": [
                {
                    "instance_id": "strike_1",
                    "card_id": "debug_kill",
                    "name": "Debug Kill",
                    "type": "attack",
                    "cost": 0,
                    "target": "enemy",
                    "effects": {"damage": 999},
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
            ],
            "player": {"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3},
        },
    )

    state = state.model_copy(update={"phase": RunPhase.MAP, "ancient": None})
    state = step(state, _action(state, "choose_node"))
    state = step(state, _play_card(state, "strike_1"))
    state = step(state, _action(state, "proceed"))
    state = step(state, _action(state, "choose_node"))

    assert state.phase.value == "rest"
    return state


def _with_campfire_unlocks(state, *, relics=(), flags=None):
    merged_flags = dict(state.flags)
    if flags:
        merged_flags.update(flags)
    return state.model_copy(
        update={
            "relics": tuple(dict.fromkeys(state.relics + tuple(relics))),
            "flags": merged_flags,
        }
    )


def test_campfire_offers_rest_and_smith_actions() -> None:
    state = _enter_campfire()

    actions = legal_actions(state)
    smith_targets = {action.target_id for action in actions if action.type == "smith"}

    assert any(action.type == "rest" and action.target_id is None for action in actions)
    assert smith_targets == {"strike_1", "defend_1"}


def test_campfire_rest_heals_and_completes_room() -> None:
    state = _enter_campfire(ascension=5)
    state = state.model_copy(update={"player": state.player.model_copy(update={"hp": 40})})

    next_state = step(state, _action(state, "rest"))

    assert next_state.player.hp == 40 + rest_heal_amount(80, ascension_level=5)
    assert next_state.phase.value == "map"
    assert next_state.map is not None
    assert next_state.map.current_node_id in next_state.map.completed_node_ids
    assert next_state.replay_log[-1].events[0].kind == "rest_healed"


def test_campfire_smith_upgrades_target_card_and_completes_room() -> None:
    state = _enter_campfire()

    next_state = step(state, _action(state, "smith", "defend_1"))

    assert _card(next_state, "defend_1").upgraded is True
    assert _card(next_state, "strike_1").upgraded is False
    assert next_state.phase.value == "map"
    assert next_state.map is not None
    assert next_state.map.current_node_id in next_state.map.completed_node_ids
    assert next_state.replay_log[-1].events[0].kind == "card_upgraded"


def test_campfire_recall_sets_ruby_key_and_is_not_offered_again() -> None:
    state = _enter_campfire()

    assert any(action.type == "recall" for action in legal_actions(state))

    next_state = step(state, _action(state, "recall"))
    assert next_state.phase.value == "map"
    assert next_state.flags["has_ruby_key"] is True
    assert next_state.replay_log[-1].events[0].kind == "ruby_key_recalled"

    recalled_state = _with_campfire_unlocks(state, flags={"has_ruby_key": True})
    assert not any(action.type == "recall" for action in legal_actions(recalled_state))


def test_campfire_dig_requires_shovel_and_grants_relic() -> None:
    state = _enter_campfire()

    assert not any(action.type == "dig" for action in legal_actions(state))

    state = _with_campfire_unlocks(
        state,
        relics=("shovel",),
        flags={"campfire_dig_relic_pool": ["anchor", "kunai"]},
    )
    next_state = step(state, _action(state, "dig"))

    assert next_state.phase.value == "map"
    assert next_state.flags["campfire_dig_count"] == 1
    assert {"anchor", "kunai"} & set(next_state.relics)
    assert next_state.replay_log[-1].events[0].kind == "campfire_dug_relic"


def test_campfire_lift_requires_girya_and_caps_at_three() -> None:
    state = _enter_campfire()

    assert not any(action.type == "lift" for action in legal_actions(state))

    state = _with_campfire_unlocks(
        state,
        relics=("girya",),
        flags={"girya_lift_count": 2},
    )
    next_state = step(state, _action(state, "lift"))

    assert next_state.phase.value == "map"
    assert next_state.flags["girya_lift_count"] == 3
    assert next_state.flags["girya_strength_bonus"] == 3
    assert next_state.replay_log[-1].events[0].kind == "girya_lifted"

    capped_state = _with_campfire_unlocks(
        state,
        relics=("girya",),
        flags={"girya_lift_count": 3},
    )
    assert not any(action.type == "lift" for action in legal_actions(capped_state))


def test_campfire_toke_requires_peace_pipe_and_removes_target_card() -> None:
    state = _enter_campfire()

    assert not any(action.type == "toke" for action in legal_actions(state))

    state = _with_campfire_unlocks(state, relics=("peace_pipe",))
    toke_targets = {action.target_id for action in legal_actions(state) if action.type == "toke"}
    assert toke_targets == {"strike_1", "defend_1"}

    next_state = step(state, _action(state, "toke", "defend_1"))

    assert next_state.phase.value == "map"
    assert [card.instance_id for card in next_state.master_deck] == ["strike_1"]
    assert next_state.flags["peace_pipe_removed_card_ids"] == ["defend_1"]
    assert next_state.replay_log[-1].events[0].kind == "card_removed"


def test_eternal_cards_are_not_peace_pipe_or_smith_targets() -> None:
    state = _enter_campfire()
    defend = _card(state, "defend_1")
    eternal = defend.model_copy(update={"custom": {**defend.custom, "eternal": True}})
    state = state.model_copy(
        update={
            "master_deck": tuple(
                eternal if card.instance_id == "defend_1" else card
                for card in state.master_deck
            )
        }
    )
    state = _with_campfire_unlocks(state, relics=("peace_pipe",))

    assert not any(
        action.type == "toke" and action.target_id == "defend_1"
        for action in legal_actions(state)
    )
    assert not any(
        action.type == "smith" and action.target_id == "defend_1"
        for action in legal_actions(state)
    )
