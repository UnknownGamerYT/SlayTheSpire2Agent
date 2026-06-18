"""Deterministic helper logic for reward, treasure, and event rooms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from random import Random

from .rewards import (
    EncounterType,
    RewardContext,
    RewardPityState,
    potion_slots_for_ascension,
    roll_card_rarity,
    roll_gold_reward,
    roll_potion_drop,
)

PoolItem = str | Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CardRewardOption:
    card_id: str
    rarity: str
    upgraded: bool = False


@dataclass(frozen=True, slots=True)
class RoomRewardState:
    hp: int
    max_hp: int
    gold: int = 0
    deck: tuple[str, ...] = ()
    relics: tuple[str, ...] = ()
    potions: tuple[str, ...] = ()
    potion_slots: int = 3
    ascension_level: int = 0


@dataclass(frozen=True, slots=True)
class RoomRewardBundle:
    gold: int = 0
    card_options: tuple[CardRewardOption, ...] = ()
    relic_id: str | None = None
    potion_id: str | None = None
    pity_state: RewardPityState = RewardPityState()


@dataclass(frozen=True, slots=True)
class EventChoice:
    choice_id: str
    label: str
    hp_delta: int = 0
    gold_delta: int = 0
    card_id: str | None = None
    relic_id: str | None = None
    potion_id: str | None = None


@dataclass(frozen=True, slots=True)
class EventOutcome:
    choice: EventChoice
    state: RoomRewardState


def generate_treasure_reward(
    rng: Random,
    relic_pool: Sequence[PoolItem],
    *,
    owned_relics: Sequence[str] = (),
) -> RoomRewardBundle:
    """Choose one relic from a chest pool, avoiding owned relics when possible."""

    relic_id = _choose_pool_id(rng, relic_pool, excluded=owned_relics)
    return RoomRewardBundle(relic_id=relic_id)


def generate_combat_reward(
    rng: Random,
    card_pool: Sequence[PoolItem],
    *,
    context: RewardContext = RewardContext(),
    pity_state: RewardPityState = RewardPityState(),
    potion_pool: Sequence[PoolItem] = (),
    relic_pool: Sequence[PoolItem] = (),
    card_count: int = 3,
) -> RoomRewardBundle:
    """Create a deterministic combat reward bundle from supplied pools."""

    gold = roll_gold_reward(rng, context).amount
    card_options, next_pity = generate_card_reward_options(
        rng,
        card_pool,
        count=card_count,
        context=context,
        pity_state=pity_state,
    )
    potion_roll = roll_potion_drop(rng, next_pity)
    potion_id = (
        _choose_pool_id(rng, potion_pool)
        if potion_roll.dropped and potion_pool
        else None
    )
    relic_id = None
    if relic_pool and context.encounter in {EncounterType.ELITE, EncounterType.BOSS}:
        relic_id = _choose_pool_id(rng, relic_pool)

    return RoomRewardBundle(
        gold=gold,
        card_options=card_options,
        relic_id=relic_id,
        potion_id=potion_id,
        pity_state=potion_roll.state,
    )


def generate_card_reward_options(
    rng: Random,
    card_pool: Sequence[PoolItem],
    *,
    count: int = 3,
    context: RewardContext = RewardContext(),
    pity_state: RewardPityState = RewardPityState(),
) -> tuple[tuple[CardRewardOption, ...], RewardPityState]:
    """Roll unique card options using the shared card-rarity helper."""

    selected: list[CardRewardOption] = []
    used_ids: set[str] = set()
    current_pity = pity_state
    for _ in range(max(0, count)):
        rarity_roll = roll_card_rarity(rng, current_pity, context)
        current_pity = rarity_roll.state
        item = _choose_card_item(
            rng,
            card_pool,
            rarity=rarity_roll.rarity.value,
            excluded=used_ids,
        )
        if item is None:
            break
        card_id = _pool_item_id(item)
        used_ids.add(card_id)
        selected.append(
            CardRewardOption(
                card_id=card_id,
                rarity=_pool_item_rarity(item),
                upgraded=_pool_item_bool(item, "upgraded"),
            )
        )
    return tuple(selected), current_pity


def choose_card_reward(
    state: RoomRewardState,
    bundle: RoomRewardBundle,
    card_id: str,
) -> RoomRewardState:
    """Apply one selected card reward to the local reward state."""

    valid_ids = {option.card_id for option in bundle.card_options}
    if card_id not in valid_ids:
        raise ValueError(f"Card reward is not available: {card_id}")
    return replace(state, deck=state.deck + (card_id,))


def skip_card_reward(state: RoomRewardState) -> RoomRewardState:
    """Skip card rewards without changing state."""

    return state


def discard_potion_reward(state: RoomRewardState, slot_index: int) -> RoomRewardState:
    """Discard a potion by slot index from reward/event helper state."""

    if slot_index < 0 or slot_index >= len(state.potions):
        raise ValueError(f"Potion slot is out of range: {slot_index}")
    return replace(
        state,
        potions=tuple(
            potion_id
            for index, potion_id in enumerate(state.potions)
            if index != slot_index
        ),
    )


def apply_reward_bundle(
    state: RoomRewardState,
    bundle: RoomRewardBundle,
) -> RoomRewardState:
    """Apply non-choice reward parts such as gold, relics, and potions."""

    relics = state.relics + ((bundle.relic_id,) if bundle.relic_id else ())
    potions = state.potions
    if bundle.potion_id and len(potions) < _potion_capacity(state, relics=relics):
        potions = potions + (bundle.potion_id,)
    return replace(
        state,
        gold=max(0, state.gold + bundle.gold),
        relics=relics,
        potions=potions,
    )


def generate_event_choices(
    rng: Random,
    *,
    relic_pool: Sequence[PoolItem] = (),
    card_pool: Sequence[PoolItem] = (),
) -> tuple[EventChoice, ...]:
    """Generate a small deterministic placeholder event choice set."""

    gold_amount = rng.randint(25, 75)
    choices = [
        EventChoice(
            choice_id="take_gold",
            label=f"Gain {gold_amount} gold",
            gold_delta=gold_amount,
        ),
        EventChoice(choice_id="leave", label="Leave"),
    ]
    if relic_pool:
        choices.insert(
            1,
            EventChoice(
                choice_id="blood_for_relic",
                label="Lose HP, gain a relic",
                hp_delta=-8,
                relic_id=_choose_pool_id(rng, relic_pool),
            ),
        )
    elif card_pool:
        choices.insert(
            1,
            EventChoice(
                choice_id="take_card",
                label="Gain a card",
                card_id=_choose_pool_id(rng, card_pool),
            ),
        )
    return tuple(choices)


def resolve_event_choice(
    state: RoomRewardState,
    choice: EventChoice,
) -> EventOutcome:
    """Apply an event choice to reward-room state."""

    hp = min(state.max_hp, max(0, state.hp + choice.hp_delta))
    gold = max(0, state.gold + choice.gold_delta)
    deck = state.deck + ((choice.card_id,) if choice.card_id else ())
    relics = state.relics + ((choice.relic_id,) if choice.relic_id else ())
    potions = state.potions
    if choice.potion_id and len(potions) < _potion_capacity(state, relics=relics):
        potions = potions + (choice.potion_id,)
    return EventOutcome(
        choice=choice,
        state=replace(
            state,
            hp=hp,
            gold=gold,
            deck=deck,
            relics=relics,
            potions=potions,
        ),
    )


def _choose_card_item(
    rng: Random,
    pool: Sequence[PoolItem],
    *,
    rarity: str,
    excluded: set[str],
) -> PoolItem | None:
    candidates = [
        item
        for item in pool
        if _pool_item_id(item) not in excluded and _pool_item_rarity(item) == rarity
    ]
    if not candidates:
        candidates = [item for item in pool if _pool_item_id(item) not in excluded]
    if not candidates:
        return None
    return rng.choice(candidates)


def _potion_capacity(
    state: RoomRewardState,
    *,
    relics: Sequence[str] | None = None,
) -> int:
    capacity = potion_slots_for_ascension(state.potion_slots, state.ascension_level)
    owned = {
        _normalized_id(relic_id)
        for relic_id in (state.relics if relics is None else relics)
    }
    if "potion_belt" in owned:
        capacity += 2
    if "alchemical_coffer" in owned:
        capacity += 4
    if "phial_holster" in owned:
        capacity += 1
    return max(0, capacity)


def _normalized_id(value: str) -> str:
    return value.lower().replace("'", "").replace(" ", "_").replace("-", "_")


def _choose_pool_id(
    rng: Random,
    pool: Sequence[PoolItem],
    *,
    excluded: Sequence[str] = (),
) -> str:
    if not pool:
        raise ValueError("Reward pool must not be empty.")
    excluded_set = set(excluded)
    candidates = [item for item in pool if _pool_item_id(item) not in excluded_set]
    if not candidates:
        candidates = list(pool)
    return _pool_item_id(rng.choice(candidates))


def _pool_item_id(item: PoolItem) -> str:
    if isinstance(item, str):
        return item
    value = item.get("card_id", item.get("relic_id", item.get("potion_id", item.get("id"))))
    if value is None:
        raise ValueError(f"Reward pool item is missing an id: {item!r}")
    return str(value)


def _pool_item_rarity(item: PoolItem) -> str:
    if isinstance(item, str):
        return "common"
    value = item.get("rarity", "common")
    return str(value)


def _pool_item_bool(item: PoolItem, key: str) -> bool:
    if isinstance(item, str):
        return False
    return bool(item.get(key, False))
