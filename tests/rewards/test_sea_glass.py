from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest

from sts2sim import legal_actions, load_state, new_run, serialize, step
from sts2sim.engine import RewardState, RunPhase, RunState
from sts2sim.engine.rng import RunRng, capture_random_state

CARD_POOL: tuple[dict[str, Any], ...] = tuple(
    {
        "id": f"{color.upper()}_{rarity.upper()}_{index}",
        "color": color,
        "rarity": rarity,
        "type": "Skill",
        "target": "Self",
        "block": 5,
    }
    for color in ("ironclad", "silent", "defect")
    for rarity in ("Common", "Uncommon", "Rare")
    for index in range(8)
)
CARD_RARITIES = {str(card["id"]).lower(): str(card["rarity"]).lower() for card in CARD_POOL}


def _before_pickup(*, cards: tuple[dict[str, Any], ...] = CARD_POOL) -> RunState:
    state = new_run(195, "IRONCLAD", 0, source_data={"cards": cards})
    return state.model_copy(
        update={
            "phase": RunPhase.REWARD,
            "flags": {**state.flags, "sea_glass_character_id": "silent"},
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
    assert all(card_id.startswith(f"{source_color}_") for card_id in offered)
    # Public behavior: five cards of each rarity from another character.
    # https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Sea_Glass
    assert Counter(CARD_RARITIES[card_id] for card_id in offered) == {
        "common": 5, "uncommon": 5, "rare": 5,
    }
    assert state.reward.metadata["sea_glass_missing_rarity_counts"] == {}
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
                    "card_ids": ("IRONCLAD_COMMON_0",),
                }
            )
        }
    )
    state = _pickup(state)
    assert not any(a.type == "proceed" for a in legal_actions(state))

    state = _take_fixed_card(state)
    assert state.master_deck[-1].card_id == "IRONCLAD_COMMON_0"
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


def test_sea_glass_reports_missing_rarities_without_filling_with_other_rarities() -> None:
    cards = tuple(
        card for card in CARD_POOL
        if card["color"] == "silent"
        and (card["rarity"] == "Common" or (
            card["rarity"] == "Uncommon" and str(card["id"]).endswith(("_0", "_1"))
        ))
    )
    state = _pickup(_before_pickup(cards=cards + cards))
    offered = state.reward.card_ids
    assert len(offered) == len(set(offered)) == 7
    assert Counter(CARD_RARITIES[card_id] for card_id in offered) == {
        "common": 5, "uncommon": 2,
    }
    assert state.reward.metadata["sea_glass_missing_rarity_counts"] == {
        "uncommon": 3, "rare": 5,
    }
    generated = next(
        event for event in state.replay_log[-1].events
        if event.kind == "relic_card_selection_generated"
    )
    assert generated.metadata["missing_rarity_counts"] == {"uncommon": 3, "rare": 5}
    restored = load_state(json.loads(json.dumps(serialize(state))))
    assert serialize(restored) == serialize(state)


def test_sea_glass_fixed_rarity_sampling_is_independent_of_source_order_and_duplicates() -> None:
    state = _pickup(_before_pickup())
    reordered = _pickup(_before_pickup(cards=tuple(reversed(CARD_POOL)) + CARD_POOL))
    assert state.reward.card_ids == reordered.reward.card_ids
    assert state.reward.metadata == reordered.reward.metadata


def test_source_sea_glass_preserves_preselected_character_and_disables_upgrade_rolls():
    from sts2sim.agent_api import action_space

    before = _before_pickup()
    before = before.model_copy(update={
        "act": 3, "rng": capture_random_state(RunRng("SEAGLASS")),
        "flags": {**before.flags, "sea_glass_character_id": "silent"},
    })
    state = _pickup(before)
    assert state.flags["sea_glass_character_id"] == "silent"
    assert state.reward.metadata["sea_glass_character_id"] == "silent"
    assert all(card.startswith("silent_") for card in state.reward.card_ids)
    flags = state.reward.metadata["fixed_card_upgraded"]
    assert len(flags) == 15
    assert not any(flags)
    assert state.rng.draws == before.rng.draws
    # ForNonCombatWithUniformOdds preserves NoUpgradeRoll through WithFlags.
    assert state.rng.streams["rewards"].draws == 15
    actions = [a for a in action_space(state) if a["type"] == "take_reward_card"]
    for index, upgraded in enumerate(flags):
        descriptor = next(a for a in actions if a["target_id"] == f"reward:fixed_card:{index}")
        assert bool(descriptor["card"].get("upgraded")) == upgraded
    index = 0
    action = next(a for a in legal_actions(state) if a.target_id == f"reward:fixed_card:{index}")
    state = step(load_state(serialize(state)), action)
    assert not state.master_deck[-1].upgraded
