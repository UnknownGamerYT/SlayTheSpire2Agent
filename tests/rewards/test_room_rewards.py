from __future__ import annotations

from random import Random

from sts2sim.mechanics.rewards import EncounterType, RewardContext
from sts2sim.mechanics.room_rewards import (
    EventChoice,
    RoomRewardBundle,
    RoomRewardState,
    apply_reward_bundle,
    choose_card_reward,
    discard_potion_reward,
    generate_combat_reward,
    generate_event_choices,
    generate_treasure_reward,
    resolve_event_choice,
)

CARD_POOL = (
    {"id": "strike_plus", "rarity": "common"},
    {"id": "defend_plus", "rarity": "common"},
    {"id": "bash_plus", "rarity": "uncommon"},
    {"id": "limit_break", "rarity": "rare"},
)


def test_treasure_reward_is_seed_deterministic_and_avoids_owned_relics() -> None:
    relic_pool = ("anchor", "shovel", "girya")

    first = generate_treasure_reward(
        Random(10),
        relic_pool,
        owned_relics=("anchor",),
    )
    second = generate_treasure_reward(
        Random(10),
        relic_pool,
        owned_relics=("anchor",),
    )

    assert first == second
    assert first.relic_id in {"shovel", "girya"}


def test_combat_reward_has_stable_gold_and_unique_card_options() -> None:
    context = RewardContext(encounter=EncounterType.NORMAL, act=1, floor=3)

    first = generate_combat_reward(Random(7), CARD_POOL, context=context)
    second = generate_combat_reward(Random(7), CARD_POOL, context=context)

    assert first == second
    assert 10 <= first.gold <= 20
    assert len(first.card_options) == 3
    assert len({option.card_id for option in first.card_options}) == 3


def test_apply_bundle_and_choose_card_reward_update_state() -> None:
    bundle = generate_combat_reward(
        Random(8),
        CARD_POOL,
        relic_pool=("paper_frog",),
        context=RewardContext(encounter=EncounterType.ELITE),
    )
    state = RoomRewardState(hp=40, max_hp=80, gold=5)

    state = apply_reward_bundle(state, bundle)
    chosen = bundle.card_options[0]
    state = choose_card_reward(state, bundle, chosen.card_id)

    assert state.gold >= 30
    assert "paper_frog" in state.relics
    assert chosen.card_id in state.deck


def test_reward_potions_respect_slots_and_can_be_discarded() -> None:
    state = RoomRewardState(
        hp=40,
        max_hp=80,
        potions=("fire_potion", "skill_potion", "foul_potion"),
    )

    full = apply_reward_bundle(state, RoomRewardBundle(potion_id="essence_of_steel"))

    assert full.potions == state.potions

    discarded = discard_potion_reward(full, 1)

    assert discarded.potions == ("fire_potion", "foul_potion")


def test_reward_potion_capacity_uses_ascension_and_slot_relics() -> None:
    reduced = RoomRewardState(
        hp=40,
        max_hp=80,
        ascension_level=11,
        potions=("fire_potion", "skill_potion"),
    )
    choice = EventChoice("take_potion", "Take potion", potion_id="foul_potion")

    assert resolve_event_choice(reduced, choice).state.potions == (
        "fire_potion",
        "skill_potion",
    )

    belted = reduced.__class__(
        hp=reduced.hp,
        max_hp=reduced.max_hp,
        ascension_level=reduced.ascension_level,
        potions=reduced.potions,
        relics=("potion_belt",),
    )

    assert resolve_event_choice(belted, choice).state.potions == (
        "fire_potion",
        "skill_potion",
        "foul_potion",
    )

    buckle = reduced.__class__(
        hp=reduced.hp,
        max_hp=reduced.max_hp,
        ascension_level=reduced.ascension_level,
        potions=reduced.potions,
        relics=("belt_buckle",),
    )
    assert resolve_event_choice(buckle, choice).state.potions == reduced.potions

    holster = reduced.__class__(
        hp=reduced.hp,
        max_hp=reduced.max_hp,
        ascension_level=reduced.ascension_level,
        potions=reduced.potions,
        relics=("phial_holster",),
    )
    assert resolve_event_choice(holster, choice).state.potions == (
        "fire_potion",
        "skill_potion",
        "foul_potion",
    )

    coffer = reduced.__class__(
        hp=reduced.hp,
        max_hp=reduced.max_hp,
        ascension_level=reduced.ascension_level,
        potions=reduced.potions,
        relics=("alchemical_coffer",),
    )
    assert resolve_event_choice(coffer, choice).state.potions == (
        "fire_potion",
        "skill_potion",
        "foul_potion",
    )


def test_event_choice_applies_state_delta() -> None:
    state = RoomRewardState(hp=40, max_hp=80, gold=0)
    choices = generate_event_choices(Random(9), relic_pool=("odd_mushroom",))
    relic_choice = next(choice for choice in choices if choice.choice_id == "blood_for_relic")

    outcome = resolve_event_choice(state, relic_choice)

    assert outcome.state.hp == 32
    assert outcome.state.gold == 0
    assert outcome.state.relics == ("odd_mushroom",)
