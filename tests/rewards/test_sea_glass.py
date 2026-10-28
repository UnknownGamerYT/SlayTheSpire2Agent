from __future__ import annotations

import json
from typing import Any

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import RewardState, RunPhase, RunState

CARD_POOL: tuple[dict[str, Any], ...] = tuple(
    {
        "id": f"{color.upper()}_CARD_{index}",
        "color": color,
        "rarity": "Common",
        "type": "Skill",
        "target": "Self",
        "block": 5,
    }
    for color in ("ironclad", "silent", "defect")
    for index in range(20)
)


def _before_pickup(*, cards: tuple[dict[str, Any], ...] = CARD_POOL) -> RunState:
    state = new_run(195, "IRONCLAD", 0, source_data={"cards": cards})
    return state.model_copy(
        update={
            "phase": RunPhase.REWARD,
            "reward": RewardState(
                reward_id="sea-glass-pickup",
                source="treasure",
                relic_id="sea_glass",
                gold=7,
            ),
        }
    )


def _pickup(state: RunState) -> RunState:
    return step(state, next(a for a in legal_actions(state) if a.type == "take_reward_relic"))


def _take_fixed_card(state: RunState) -> RunState:
    return step(
        state,
        next(
            a
            for a in legal_actions(state)
            if a.type == "take_reward_card" and a.target_id.startswith("reward:fixed_card:")
        ),
    )


@pytest.mark.parametrize("pick_count", [0, 3, 15])
def test_sea_glass_offers_fifteen_cards_from_one_other_character_and_allows_any_number(
    pick_count: int,
) -> None:
    state = _pickup(_before_pickup())
    offered = state.reward.card_ids
    source_color = state.reward.metadata["sea_glass_character_id"]
    assert source_color in {"silent", "defect"}
    assert len(offered) == len(set(offered)) == 15
    assert all(card_id.startswith(f"{source_color}_card_") for card_id in offered)
    assert state.reward.card_options == state.reward.card_option_groups == ()
    assert state.reward.gold == 7

    initial_deck_size = len(state.master_deck)
    for _ in range(pick_count):
        state = _take_fixed_card(state)
    assert len(state.master_deck) == initial_deck_size + pick_count
    assert all(card.card_id.lower() in offered for card in state.master_deck[initial_deck_size:])
    assert sum(a.type == "take_reward_card" for a in legal_actions(state)) == 15 - pick_count
    state = step(state, next(a for a in legal_actions(state) if a.type == "proceed"))
    assert len(state.master_deck) == initial_deck_size + pick_count
    assert state.reward is None


def test_sea_glass_stays_optional_inside_forced_rewards_without_skipping_mandatory_cards() -> None:
    state = _before_pickup()
    state = state.model_copy(
        update={
            "reward": state.reward.model_copy(
                update={
                    "forced": True,
                    "gold": 0,
                    "card_ids": ("IRONCLAD_CARD_0",),
                }
            )
        }
    )
    state = _pickup(state)
    assert not any(a.type == "proceed" for a in legal_actions(state))

    state = _take_fixed_card(state)
    assert state.master_deck[-1].card_id == "IRONCLAD_CARD_0"
    assert any(a.type == "proceed" for a in legal_actions(state))
    assert sum(a.type == "take_reward_card" for a in legal_actions(state)) == 15


def test_sea_glass_selection_and_choices_survive_json_save_reload() -> None:
    initial = _before_pickup()
    restored_initial = load_state(json.loads(json.dumps(serialize(initial))))
    state = _pickup(initial)
    assert serialize(state) == serialize(_pickup(restored_initial))

    state = _take_fixed_card(_take_fixed_card(state))
    restored = load_state(json.loads(json.dumps(serialize(state))))
    assert serialize(_take_fixed_card(state)) == serialize(_take_fixed_card(restored))


def test_sea_glass_keeps_an_unavailable_other_character_pool_explicit() -> None:
    state = _pickup(
        _before_pickup(cards=tuple(card for card in CARD_POOL if card["color"] == "ironclad"))
    )
    assert state.reward.card_ids == ()
    assert any(
        event.kind == "relic_card_reward_blocked"
        and event.metadata["reason"] == "empty_other_character_card_pool"
        for event in state.replay_log[-1].events
    )


def test_sea_glass_uses_available_distinct_cards_when_a_custom_source_has_fewer_than_fifteen() -> (
    None
):
    state = _pickup(_before_pickup(cards=CARD_POOL[20:23]))
    assert len(state.reward.card_ids) == len(set(state.reward.card_ids)) == 3
