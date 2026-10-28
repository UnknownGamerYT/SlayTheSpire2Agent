"""Item factories use source GetUnlockedCards before selecting reward candidates."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import RewardState, RunPhase, load_state, serialize_state, transitions

SOURCE = json.loads(
    (Path(__file__).parents[1] / "fixtures/stable_card_pool_source.json").read_text()
)
CHARACTERS = ("ironclad", "silent", "regent", "necrobinder", "defect")


def _allowed(pool: str, player_count: int) -> set[str]:
    excluded = "SingleplayerOnly" if player_count > 1 else "MultiplayerOnly"
    return {
        card_id for card_id in SOURCE["pools"][pool]
        if card_id not in SOURCE["epochs"]
        and SOURCE["multiplayer_constraints"].get(card_id) != excluded
    }


def _pickup_state(relic: str, character: str, player_count: int):
    state = new_run("ITEM-PROFILE-2", character.upper(), 0)
    # Supplying the full registered catalog must not bypass GetUnlockedCards.
    rows = transitions._cached_source_rows(state, "cards")
    return state.model_copy(update={
        "phase": RunPhase.REWARD, "ancient": None,
        "reward": RewardState(reward_id="profile", relic_id=relic),
        "flags": {**state.flags, "cards": rows, "revealed_epochs": [],
                  "player_count": player_count},
    })


def _pickup(state):
    action = next(a for a in legal_actions(state) if a.type == "take_reward_relic")
    direct = step(state, action)
    restored = step(load_state(serialize_state(state)), action)
    assert serialize_state(direct) == serialize_state(restored)
    return direct


@pytest.mark.parametrize("character", CHARACTERS)
@pytest.mark.parametrize("player_count", (1, 2))
def test_scroll_boxes_source_pool_honors_epochs_and_player_mode(character, player_count):
    result = _pickup(_pickup_state("scroll_boxes", character, player_count))
    assert result.event is not None and result.event.event_id == "scroll_boxes"
    cards = {card_id for pack in result.event.metadata["card_packs"] for card_id in pack}
    assert cards and cards <= _allowed(character, player_count)


@pytest.mark.parametrize("character", CHARACTERS)
@pytest.mark.parametrize("player_count", (1, 2))
def test_sea_glass_selected_class_pool_honors_epochs_and_player_mode(character, player_count):
    state = _pickup_state("sea_glass", "ironclad", player_count)
    state = state.model_copy(update={"flags": {
        **state.flags, "sea_glass_character_id": character,
    }})
    result = _pickup(state)
    assert result.reward is not None
    offered = result.reward.metadata["sea_glass_card_ids"]
    assert len(offered) == 15
    assert set(offered) <= _allowed(character, player_count)


@pytest.mark.parametrize("character", CHARACTERS)
def test_dusty_tome_setup_uses_registered_source_pool_and_saved_selection(character):
    state = _pickup_state("dusty_tome", character, 1)
    result = _pickup(state)
    selected = result.flags["dusty_tome_card_id"]
    assert selected in _allowed(character, 1)
    card = result.master_deck[-1]
    assert card.card_id.lower() == selected and card.upgraded
    assert selected not in {"break", "suppress", "protector", "meteor_shower", "quadcast"}



def test_registered_transform_catalog_does_not_fill_from_cached_rows():
    from sts2sim.engine.rng import random_from_state, random_stream
    from sts2sim.mechanics.event_flows import EventFlowMarker, EventFlowMarkerKind
    state = _pickup_state('astrolabe', 'ironclad', 1)
    state = state.model_copy(update={'flags': {**state.flags, 'cards': (
        {'id': 'ANGER', 'color': 'ironclad', 'rarity': 'Common'},
        {'id': 'SHRUG_IT_OFF', 'color': 'ironclad', 'rarity': 'Common'},
    )}})
    rng = random_stream(random_from_state(state.rng), 'niche')
    marker = EventFlowMarker(kind=EventFlowMarkerKind.CARD_TRANSFORM)
    for _ in range(20):
        selected = transitions._draw_transformed_card_id(
            state, rng, marker, excluded_card_ids=('bash',))
        assert selected in {'anger', 'shrug_it_off'}


def test_locked_registered_transform_catalog_stays_empty_without_rng_or_fallback():
    from sts2sim.engine.rng import capture_random_state, random_from_state, random_stream
    from sts2sim.mechanics.event_flows import EventFlowMarker, EventFlowMarkerKind
    state = _pickup_state('astrolabe', 'ironclad', 1)
    state = state.model_copy(update={'flags': {**state.flags, 'cards': (
        {'id': 'CRUELTY', 'color': 'ironclad', 'rarity': 'Rare'},
    )}})
    rng = random_stream(random_from_state(state.rng), 'niche')
    before = capture_random_state(rng)
    marker = EventFlowMarker(kind=EventFlowMarkerKind.CARD_TRANSFORM)
    assert transitions._draw_transformed_card_id(
        state, rng, marker, excluded_card_ids=('bash',)) is None
    assert capture_random_state(rng) == before
