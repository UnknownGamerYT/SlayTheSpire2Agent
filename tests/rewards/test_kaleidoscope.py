from __future__ import annotations

import json
from typing import Any

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import AncientOptionState, AncientState, RewardState, RunPhase, RunState

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
        for color in ("ironclad", "silent", "defect")
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
    if card["color"] in {"silent", "defect"}
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
    state = _pickup(_before_pickup(ancient=ancient, cards=cards), ancient=ancient)

    assert _groups(state) == ()
    assert not any(action.type == "take_reward_card" for action in legal_actions(state))


@pytest.mark.parametrize("ancient", [False, True])
def test_kaleidoscope_rewards_are_identical_after_json_save_reload(ancient: bool) -> None:
    state = _before_pickup(ancient=ancient)
    restored = load_state(json.loads(json.dumps(serialize(state))))

    assert serialize(_pickup(state, ancient=ancient)) == serialize(
        _pickup(restored, ancient=ancient)
    )
