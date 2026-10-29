from __future__ import annotations

import json
from typing import Any

import pytest

from sts2sim import inspect_reward_reroll_fault, legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import AncientOptionState, AncientState, RewardState, RunPhase, RunState
from sts2sim.engine.rng import RunRng, capture_random_state
from sts2sim.mechanics.source_factory_errors import SourceCardFactoryError

CARD_POOL: tuple[dict[str, Any], ...] = (
    *(
        {
            "id": f"{color.upper()}_CARD_{index}",
            "color": color,
            "rarity": "Common",
            "type": "Attack",
            "target": "Enemy",
            "damage": 6,
        }
        for color in ("ironclad", "silent", "regent", "necrobinder", "defect")
        for index in range(3)
    ),
    {"id": "SHARED_CARD", "color": "shared", "rarity": "Common", "type": "Skill"},
    {"id": "COLORLESS_CARD", "color": "colorless", "rarity": "Common", "type": "Skill"},
    {"id": "BASIC_CARD", "color": "silent", "rarity": "Basic", "type": "Attack"},
    {"id": "ANCIENT_CARD", "color": "defect", "rarity": "Ancient", "type": "Skill"},
    {"id": "TOKEN_CARD", "color": "token", "rarity": "Common", "type": "Skill"},
    {"id": "STATUS_CARD", "color": "silent", "rarity": "Common", "type": "Status"},
)
OTHER_CHARACTER_IDS = {
    str(card["id"]).lower()
    for card in CARD_POOL
    if card["color"] in {"silent", "regent", "necrobinder", "defect"}
    and card["rarity"] == "Common"
    and card["type"] == "Attack"
}


def _before_pickup(*, ancient: bool, cards: tuple[dict[str, Any], ...] = CARD_POOL) -> RunState:
    state = new_run(193, "IRONCLAD", 0, source_data={"cards": cards})
    if not ancient:
        return state.model_copy(
            update={
                "phase": RunPhase.REWARD,
                "reward": RewardState(
                    reward_id="kaleidoscope-pickup", source="treasure", relic_id="kaleidoscope"
                ),
            }
        )
    option = AncientOptionState(
        option_id="kaleidoscope",
        name="Kaleidoscope",
        kind="positive_relic",
        relic_id="kaleidoscope",
        metadata={
            "choice": {
                "option_id": "kaleidoscope",
                "kind": "positive_relic",
                "relic_id": "kaleidoscope",
                "fixed_relic_ids": ("kaleidoscope",),
                "card_reward_count": 2,
                "card_reward_size": 3,
                "card_reward_kind": "other_character",
            }
        },
    )
    return state.model_copy(
        update={"ancient": AncientState(act=1, ancient_id="neow", options=(option,))}
    )


def _pickup(state: RunState, *, ancient: bool) -> RunState:
    action_type = "choose_ancient" if ancient else "take_reward_relic"
    return step(
        state, next(action for action in legal_actions(state) if action.type == action_type)
    )


def _groups(state: RunState) -> tuple[tuple[str, ...], ...]:
    assert state.reward is not None
    return tuple(
        group for group in (state.reward.card_options, *state.reward.card_option_groups) if group
    )


@pytest.mark.parametrize("ancient", [False, True])
def test_kaleidoscope_offers_two_usable_rewards_from_other_characters(ancient: bool) -> None:
    state = _pickup(_before_pickup(ancient=ancient), ancient=ancient)
    groups = _groups(state)

    assert "kaleidoscope" in state.relics
    assert len(groups) == 2
    assert all(len(group) == 3 and len(set(group)) == 3 for group in groups)
    assert all(set(group) <= OTHER_CHARACTER_IDS for group in groups)
    assert all(len({identifier.rsplit("_card_", 1)[0] for identifier in group}) == 3
               for group in groups)

    initial_deck_size = len(state.master_deck)
    for _ in range(2):
        state = step(
            state,
            next(action for action in legal_actions(state) if action.type == "take_reward_card"),
        )
    assert len(state.master_deck) == initial_deck_size + 2
    assert all(card.card_id.lower() in OTHER_CHARACTER_IDS for card in state.master_deck[-2:])
    assert all(
        card.type == "attack" and card.effects["sequence"] == [{"damage": 6}]
        for card in state.master_deck[-2:]
    )
    assert not any(action.type == "take_reward_card" for action in legal_actions(state))


@pytest.mark.parametrize("ancient", [False, True])
def test_kaleidoscope_does_not_substitute_own_character_when_source_pool_is_empty(
    ancient: bool,
) -> None:
    cards = tuple(
        card
        for card in CARD_POOL
        if card["id"] not in {value.upper() for value in OTHER_CHARACTER_IDS}
    )
    state = _before_pickup(ancient=ancient, cards=cards)
    before = serialize(state)
    with pytest.raises(SourceCardFactoryError, match="no_valid_rarity"):
        _pickup(state, ancient=ancient)
    assert serialize(state) == before


@pytest.mark.parametrize("ancient", [False, True])
def test_kaleidoscope_rewards_are_identical_after_json_save_reload(ancient: bool) -> None:
    state = _before_pickup(ancient=ancient)
    restored = load_state(json.loads(json.dumps(serialize(state))))

    assert serialize(_pickup(state, ancient=ancient)) == serialize(
        _pickup(restored, ancient=ancient)
    )


@pytest.mark.parametrize("ancient", [False, True])
def test_kaleidoscope_uses_two_pool_shuffles_and_six_full_factories(ancient: bool) -> None:
    state = _before_pickup(ancient=ancient)
    state = state.model_copy(update={"rng": capture_random_state(RunRng(
        "KALEIDOSCOPE", derivation="sts2.v0.107.1.utf16-djb2-splitmix64",
    ))})
    result = _pickup(state, ancient=ancient)
    assert result.rng.streams["niche"].draws == 6
    assert result.rng.streams["rewards"].draws == 18
    assert result.reward.metadata["kaleidoscope_character_pool_groups"] == (
        ("silent", "necrobinder", "regent"), ("necrobinder", "regent", "defect"),
    )


def test_kaleidoscope_persists_explicit_empty_custom_reroll_boundary() -> None:
    state = _before_pickup(ancient=False)
    state = state.model_copy(update={"relics": (*state.relics, "driftwood")})
    result = load_state(serialize(_pickup(state, ancient=False)))
    assert not any(action.type == "reroll_reward" for action in legal_actions(result))
    contexts = result.reward.metadata["card_reroll_contexts"]
    for key in ("primary", "group:0"):
        assert contexts[key]["custom_pool"] is True
        assert tuple(contexts[key]["pools"]) == tuple(contexts[key]["card_ids"]) == ()
        assert contexts[key]["source_fault_reason"] == "empty_custom_card_pool"
        assert contexts[key]["availability"] == "modeled_source_fault"


@pytest.mark.parametrize("group", ["primary", "group:0"])
def test_kaleidoscope_fault_diagnostic_records_exact_source_prefix_without_ui_guess(group):
    state = _before_pickup(ancient=False)
    state = state.model_copy(update={"relics": (*state.relics, "driftwood", "prismatic_gem",
                                                "dingy_rug", "silver_crucible")})
    state = _pickup(state, ancient=False)
    before = serialize(state)
    trace = inspect_reward_reroll_fault(state, group)
    assert serialize(state) == before
    assert inspect_reward_reroll_fault(load_state(before), group) == trace
    assert json.loads(json.dumps(trace))["source_exception"]["type"] == "InvalidOperationException"
    assert trace["normal_transition"] is False
    assert trace["continuation_requires_live_ui_observation"] is True
    assert trace["factory_returned"] is False
    assert trace["after_generated_invoked"] is False
    assert trace["reward_taken_hooks_invoked"] is False
    assert trace["rarity_offset_changed"] is False
    assert trace["rng_after"]["streams"]["rewards"]["draws"] == (
        trace["rng_before"]["streams"]["rewards"]["draws"] + 1)
    for stream, position in trace["rng_before"]["streams"].items():
        if stream != "rewards":
            assert trace["rng_after"]["streams"][stream] == position
    expected = (state.reward.card_options if group == "primary"
                else state.reward.card_option_groups[0])
    assert trace["history_append"] == tuple({"card_id": card, "was_picked": False}
                                            for card in expected)
    partial = trace["partial_reward_model"]
    assert partial["metadata"]["card_reward_rerolled_groups"] == [group]
    if group == "primary":
        assert partial["card_options"] == []
        assert partial["card_option_groups"] == [list(state.reward.card_option_groups[0])]
    else:
        assert partial["card_options"] == list(state.reward.card_options)
        assert partial["card_option_groups"] == [[]]
    # The original normal reward remains usable. Inspecting an oracle fault did
    # not silently take/skip cards, spend its reroll or rerun Crucible callbacks.
    assert len([a for a in legal_actions(state) if a.type == "take_reward_card"]) == 6
    assert not any(a.type == "reroll_reward" for a in legal_actions(state))


def test_kaleidoscope_manual_groups_apply_silver_crucible_once_per_group() -> None:
    from sts2sim.engine.rng import random_from_state
    from sts2sim.engine.source_card_reward_modifiers import apply_card_reward_modifiers
    from sts2sim.engine.source_kaleidoscope import pickup

    state = _before_pickup(ancient=False)
    state = state.model_copy(update={"relics": (*state.relics, "silver_crucible")})
    result, events = pickup(state, random_from_state(state.rng))
    silver = [event for event in events if event.source_id == "silver_crucible"]
    assert len(silver) == 2
    assert all(event.metadata["is_card_reward"] for event in silver)
    assert result.flags["silver_crucible_times_used"] == 2
    repeated, events = apply_card_reward_modifiers(result)
    assert repeated.flags["silver_crucible_times_used"] == 2
    assert events == ()
