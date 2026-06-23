"""Bespoke event option primitives.

These helpers model event options whose meaning is richer than the generic
text-derived event room options.  They deliberately operate on
``EventRoomState`` rather than engine ``RunState`` so they can be reused by
tests, tooling, or a future transition layer without importing engine models.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from random import Random

from .combat_rewards import fake_merchant_reward_relic_ids
from .event_rooms import EventOption, EventResolution, EventRoomState, resolve_event_option

PoolItem = str | Mapping[str, object]

LANTERN_KEY_CARD_ID = "lantern_key"
INJURY_CARD_ID = "injury"
HISTORY_COURSE_RELIC_ID = "history_course"
ROYAL_POISON_RELIC_ID = "royal_poison"
FOUL_POTION_ID = "foul_potion"
ENCHANTABLE_CARD_TYPES = frozenset({"attack", "skill", "power"})

SPECIAL_EVENT_IDS = (
    "BATTLEWORN_DUMMY",
    "COLORFUL_PHILOSOPHERS",
    "PUNCH_OFF",
    "DENSE_VEGETATION",
    "INFESTED_AUTOMATON",
    "SELF_HELP_BOOK",
    "THE_FUTURE_OF_POTIONS",
    "THE_LANTERN_KEY",
    "WAR_HISTORIAN_REPY",
    "ROUND_TEA_PARTY",
    "RANWID_THE_ELDER",
    "RELIC_TRADER",
    "POTION_COURIER",
    "FAKE_MERCHANT",
)


@dataclass(frozen=True, slots=True)
class EventSpecialResolution:
    """Resolution wrapper carrying bespoke costs alongside an EventResolution."""

    resolution: EventResolution
    spent_potion_ids: tuple[str, ...] = ()
    spent_relic_ids: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def option(self) -> EventOption:
        return self.resolution.option

    @property
    def state(self) -> EventRoomState:
        return self.resolution.state

    @property
    def gold_delta(self) -> int:
        return self.resolution.gold_delta

    @property
    def hp_delta(self) -> int:
        return self.resolution.hp_delta

    @property
    def heal_amount(self) -> int:
        return self.resolution.heal_amount

    @property
    def max_hp_delta(self) -> int:
        return self.resolution.max_hp_delta

    @property
    def added_card_ids(self) -> tuple[str, ...]:
        return self.resolution.added_card_ids

    @property
    def removed_card_ids(self) -> tuple[str, ...]:
        return self.resolution.removed_card_ids

    @property
    def relic_ids(self) -> tuple[str, ...]:
        return self.resolution.relic_ids

    @property
    def potion_ids(self) -> tuple[str, ...]:
        return self.resolution.potion_ids

    @property
    def upgrade_random_count(self) -> int:
        return self.resolution.upgrade_random_count

    @property
    def transform_random_count(self) -> int:
        return self.resolution.transform_random_count

    @property
    def remove_random_count(self) -> int:
        return self.resolution.remove_random_count

    @property
    def combat_encounter(self) -> str | None:
        return self.resolution.combat_encounter


def battleworn_dummy_options() -> tuple[EventOption, ...]:
    """Return Battleworn Dummy's three fight settings."""

    return (
        EventOption(
            option_id="SETTING_1",
            label="Setting 1",
            description="Fight a 75 HP dummy. Procure 1 random Potion.",
            combat_encounter="battleworn_dummy",
            metadata={
                "monster_id": "training_dummy",
                "monster_hp": 75,
                "post_combat_reward": {
                    "random_potion_count": 1,
                    "card_count": 0,
                    "relic_count": 0,
                },
                "reward_summary": "Procure 1 random potion after the dummy fight.",
            },
        ),
        EventOption(
            option_id="SETTING_2",
            label="Setting 2",
            description="Fight a 150 HP dummy. Upgrade 2 random cards.",
            combat_encounter="battleworn_dummy",
            metadata={
                "monster_id": "training_dummy",
                "monster_hp": 150,
                "post_combat_reward": {
                    "upgrade_random_count": 2,
                    "card_count": 0,
                    "relic_count": 0,
                    "potion_chance_percent": 0,
                },
                "reward_summary": "Upgrade 2 random cards after the dummy fight.",
            },
        ),
        EventOption(
            option_id="SETTING_3",
            label="Setting 3",
            description="Fight a 300 HP dummy. Obtain a random Relic.",
            combat_encounter="battleworn_dummy",
            metadata={
                "monster_id": "training_dummy",
                "monster_hp": 300,
                "post_combat_reward": {
                    "random_relic_count": 1,
                    "card_count": 0,
                    "potion_chance_percent": 0,
                },
                "reward_summary": "Obtain a random relic after the dummy fight.",
            },
        ),
    )


def colorful_philosophers_options() -> tuple[EventOption, ...]:
    """Return Colorful Philosophers' character-filtered card rewards."""

    character_specs = (
        ("DEFECT", "Defect", "defect"),
        ("IRONCLAD", "Ironclad", "ironclad"),
        ("NECROBINDER", "Necrobinder", "necrobinder"),
        ("REGENT", "Regent", "regent"),
        ("SILENT", "Silent", "silent"),
    )
    return tuple(
        EventOption(
            option_id=option_id,
            label=label,
            description=f"Obtain 3 random {label} cards.",
            metadata={
                "card_reward_count": 3,
                "card_reward_character": character_id,
                "card_reward_kind": "character_pool",
            },
        )
        for option_id, label, character_id in character_specs
    )


def punch_off_options() -> tuple[EventOption, ...]:
    """Return Punch Off's immediate nab option and greater-reward fight option."""

    return (
        EventOption(
            option_id="NAB",
            label="Nab",
            description="Add Injury to your Deck. Obtain a random Relic.",
            fixed_card_ids=(INJURY_CARD_ID,),
            random_relic_count=1,
            metadata={"risk": "curse_for_relic"},
        ),
        EventOption(
            option_id="I_CAN_TAKE_THEM",
            label="I Can Take Them",
            description="Fight them for Greater Rewards.",
            combat_encounter="normal",
            metadata={
                "post_combat_reward": {
                    "standard_combat_rewards": True,
                    "random_relic_count": 1,
                    "extra_random_potion_count": 1,
                },
                "reward_summary": (
                    "Normal combat rewards plus an additional potion and random relic."
                ),
            },
        ),
    )


def dense_vegetation_options() -> tuple[EventOption, ...]:
    """Return Dense Vegetation's trudge and rest/fight options."""

    return (
        EventOption(
            option_id="TRUDGE_ON",
            label="Trudge On",
            description="Gain 61-99 Gold. Lose 8 HP.",
            hp_delta=-8,
            metadata={"gold_range": (61, 99)},
        ),
        EventOption(
            option_id="REST",
            label="Rest",
            description="Heal 30% Max HP. Fight some enemies.",
            heal_percent_max_hp=0.30,
            combat_encounter="normal",
            metadata={
                "post_combat_reward": {"standard_combat_rewards": True},
                "reward_summary": "Heal 30% max HP, then fight for standard combat rewards.",
            },
        ),
    )


def infested_automaton_options() -> tuple[EventOption, ...]:
    """Return Infested Automaton's filtered random-card reward options."""

    return (
        EventOption(
            option_id="STUDY",
            label="Study",
            description="Obtain a random Power card.",
            metadata={
                "random_card_count": 1,
                "card_type": "power",
                "card_reward_kind": "filtered_random_card",
            },
        ),
        EventOption(
            option_id="TOUCH_CORE",
            label="Touch Core",
            description="Obtain a random 0-cost card.",
            metadata={
                "random_card_count": 1,
                "card_cost": 0,
                "card_reward_kind": "filtered_random_card",
            },
        ),
    )


def self_help_book_options(
    deck: Sequence[str] | None = None,
) -> tuple[EventOption, ...]:
    """Return Self-Help Book's enchant choices and visible locked markers."""

    deck_card_types = _deck_card_types(deck)
    options: list[EventOption] = []
    any_enchantable = False
    for spec in _SELF_HELP_BOOK_ENCHANT_SPECS:
        if deck_card_types is None:
            options.append(_self_help_book_enchant_option(spec))
            options.append(_self_help_book_locked_option(spec))
            continue
        if spec["card_type"] in deck_card_types:
            any_enchantable = True
            options.append(_self_help_book_enchant_option(spec))
        else:
            options.append(_self_help_book_locked_option(spec))

    no_options_metadata: dict[str, object] = {
        "no_enchantable_cards": True,
        "card_types": tuple(sorted(ENCHANTABLE_CARD_TYPES)),
    }
    if deck_card_types is None or any_enchantable:
        no_options_metadata.update(
            {
                "locked": True,
                "disabled_reason": "requires_no_enchantable_cards",
            }
        )

    options.append(
        EventOption(
            option_id="NO_OPTIONS",
            label="Move On",
            description="You don't have any cards that can be Enchanted.",
            metadata=no_options_metadata,
        )
    )
    return tuple(options)


def future_of_potions_options() -> tuple[EventOption, ...]:
    """Return The Future of Potions?' potion sacrifice reward marker."""

    return (
        EventOption(
            option_id="POTION",
            label="Insert Common Potion",
            description="Lose a Potion. Obtain an Upgraded Common Skill.",
            metadata={
                "required_potion_count": 1,
                "potion_cost_count": 1,
                "random_card_count": 1,
                "card_reward_kind": "filtered_random_card",
                "card_rarity": "common",
                "card_type": "skill",
                "upgraded": True,
                "reward_summary": "Obtain 1 upgraded Common Skill card.",
            },
        ),
    )


def lantern_key_options() -> tuple[EventOption, ...]:
    """Return The Lantern Key's return and keep/fight options."""

    return (
        EventOption(
            option_id="RETURN_THE_KEY",
            label="Return the Key",
            description="Gain 100 Gold.",
            gold_delta=100,
        ),
        EventOption(
            option_id="KEEP_THE_KEY",
            label="Keep the Key",
            description="Fight to obtain the Key.",
            combat_encounter="normal",
            metadata={
                "post_combat_reward": {"fixed_card_ids": (LANTERN_KEY_CARD_ID,)},
                "reward_summary": "Standard combat rewards plus the Lantern Key quest card.",
            },
        ),
    )


def war_historian_repy_options() -> tuple[EventOption, ...]:
    """Return War Historian, Repy's Lantern Key redemption options."""

    return (
        EventOption(
            option_id="UNLOCK_CAGE",
            label="Unlock the Cage",
            description="Lose Lantern Key. Obtain History Course.",
            fixed_relic_ids=(HISTORY_COURSE_RELIC_ID,),
            remove_card_ids=(LANTERN_KEY_CARD_ID,),
            required_card_ids=(LANTERN_KEY_CARD_ID,),
            metadata={
                "multi_lantern_key_bonus": {
                    "required_card_id": LANTERN_KEY_CARD_ID,
                    "required_count": 2,
                    "remove_all_card_ids": (LANTERN_KEY_CARD_ID,),
                    "random_potion_count": 2,
                    "random_relic_count": 2,
                },
            },
        ),
        EventOption(
            option_id="UNLOCK_CHEST",
            label="Unlock the Chest",
            description="Lose Lantern Key. Procure 2 random Potions. Obtain 2 random Relics.",
            random_potion_count=2,
            random_relic_count=2,
            remove_card_ids=(LANTERN_KEY_CARD_ID,),
            required_card_ids=(LANTERN_KEY_CARD_ID,),
            metadata={
                "multi_lantern_key_bonus": {
                    "required_card_id": LANTERN_KEY_CARD_ID,
                    "required_count": 2,
                    "remove_all_card_ids": (LANTERN_KEY_CARD_ID,),
                    "fixed_relic_ids": (HISTORY_COURSE_RELIC_ID,),
                },
            },
        ),
    )


def round_tea_party_options() -> tuple[EventOption, ...]:
    """Return The Round Tea Party's tea and scripted pick-a-fight outcomes."""

    return (
        EventOption(
            option_id="ENJOY_TEA",
            label="Enjoy Your Tea",
            description="Obtain Royal Poison. Heal to full HP.",
            fixed_relic_ids=(ROYAL_POISON_RELIC_ID,),
            heal_percent_max_hp=1.0,
        ),
        EventOption(
            option_id="PICK_FIGHT",
            label="Pick a Fight",
            description="Lose 11 HP. Obtain a random Relic.",
            hp_delta=-11,
            random_relic_count=1,
            metadata={"scripted_event_scene": "round_tea_party_pick_fight"},
        ),
    )


def ranwid_the_elder_options() -> tuple[EventOption, ...]:
    """Return Ranwid the Elder's trade options plus visible locked markers."""

    return (
        EventOption(
            option_id="POTION",
            label="Give Potion",
            description="Give a Potion. Obtain a random Relic.",
            random_relic_count=1,
            metadata={
                "required_potion_count": 1,
                "potion_cost_count": 1,
                "trade_kind": "potion_for_relic",
            },
        ),
        EventOption(
            option_id="POTION_LOCKED",
            label="Locked",
            description="You don't have any Potions that can be given.",
            metadata={
                "locked": True,
                "disabled_reason": "requires_potion",
                "required_potion_count": 1,
            },
        ),
        EventOption(
            option_id="GOLD",
            label="Give 100 Gold",
            description="Give 100 Gold. Obtain a random Relic.",
            random_relic_count=1,
            metadata={
                "required_gold": 100,
                "gold_cost": 100,
                "trade_kind": "gold_for_relic",
            },
        ),
        EventOption(
            option_id="RELIC",
            label="Give Relic",
            description="Give a Relic. Obtain 2 random Relics.",
            random_relic_count=2,
            metadata={
                "required_relic_count": 1,
                "relic_cost_count": 1,
                "trade_kind": "relic_for_two_relics",
            },
        ),
        EventOption(
            option_id="RELIC_LOCKED",
            label="Locked",
            description="You don't have any Relics that can be given.",
            metadata={
                "locked": True,
                "disabled_reason": "requires_relic",
                "required_relic_count": 1,
            },
        ),
    )


def relic_trader_options(
    offered_relic_ids: Sequence[str] = (),
) -> tuple[EventOption, ...]:
    """Return Relic Trader's top/middle/bottom trade slots.

    If ``offered_relic_ids`` are supplied, the slots are treated as already
    rolled fixed offers.  Otherwise each slot records a random relic reward.
    """

    slot_specs = (
        ("TOP", "Take the Top One"),
        ("MIDDLE", "Take the Middle One"),
        ("BOTTOM", "Take the Bottom One"),
    )
    offered = tuple(_normalized_id(relic_id) for relic_id in offered_relic_ids)
    options: list[EventOption] = []
    for index, (option_id, label) in enumerate(slot_specs):
        fixed_relic_ids = offered[index : index + 1]
        metadata: dict[str, object] = {
            "required_relic_count": 1,
            "relic_cost_count": 1,
            "trade_kind": "relic_for_relic",
            "trade_slot": option_id.lower(),
        }
        if fixed_relic_ids:
            metadata["offered_relic_id"] = fixed_relic_ids[0]
        options.append(
            EventOption(
                option_id=option_id,
                label=label,
                description="Trade one of your Relics for a random Relic.",
                fixed_relic_ids=fixed_relic_ids,
                random_relic_count=0 if fixed_relic_ids else 1,
                metadata=metadata,
            )
        )
    return tuple(options)


def potion_courier_options() -> tuple[EventOption, ...]:
    """Return Potion Courier's fixed Foul Potions and uncommon-potion ransack."""

    return (
        EventOption(
            option_id="GRAB_POTIONS",
            label="Grab Potions",
            description="Procure 3 Foul Potions.",
            fixed_potion_ids=(FOUL_POTION_ID, FOUL_POTION_ID, FOUL_POTION_ID),
        ),
        EventOption(
            option_id="RANSACK",
            label="Ransack",
            description="Procure 1 random Uncommon Potion.",
            random_potion_count=1,
            metadata={"potion_rarity": "uncommon"},
        ),
    )


def fake_merchant_summary_option(
    unsold_relic_ids: Sequence[str] = (),
) -> EventOption:
    """Return a synthetic Fake Merchant post-combat reward summary marker."""

    relic_ids = fake_merchant_reward_relic_ids(unsold_relic_ids=unsold_relic_ids)
    return EventOption(
        option_id="SUMMARY",
        label="Fake Merchant Summary",
        description="Winning rewards Fake Merchant's Rug plus all unsold fake relics.",
        fixed_relic_ids=relic_ids,
        metadata={
            "summary_marker": True,
            "post_combat_reward": {"fixed_relic_ids": relic_ids},
            "reward_summary": "Fake Merchant's Rug plus all unsold fake relics.",
        },
    )


def fake_merchant_options(
    unsold_relic_ids: Sequence[str] = (),
) -> tuple[EventOption, ...]:
    """Return Fake Merchant's synthetic summary marker option."""

    return (fake_merchant_summary_option(unsold_relic_ids),)


def special_event_options(
    event_id: str,
    *,
    deck: Sequence[str] | None = None,
    relic_trader_offered_relic_ids: Sequence[str] = (),
    fake_merchant_unsold_relic_ids: Sequence[str] = (),
) -> tuple[EventOption, ...]:
    """Return bespoke option primitives for a supported special event id."""

    key = _event_key(event_id)
    if key == "battleworn_dummy":
        return battleworn_dummy_options()
    if key == "colorful_philosophers":
        return colorful_philosophers_options()
    if key == "punch_off":
        return punch_off_options()
    if key == "dense_vegetation":
        return dense_vegetation_options()
    if key == "infested_automaton":
        return infested_automaton_options()
    if key == "self_help_book":
        return self_help_book_options(deck)
    if key == "the_future_of_potions":
        return future_of_potions_options()
    if key == "the_lantern_key":
        return lantern_key_options()
    if key == "war_historian_repy":
        return war_historian_repy_options()
    if key == "round_tea_party":
        return round_tea_party_options()
    if key == "ranwid_the_elder":
        return ranwid_the_elder_options()
    if key == "relic_trader":
        return relic_trader_options(relic_trader_offered_relic_ids)
    if key == "potion_courier":
        return potion_courier_options()
    if key == "fake_merchant":
        return fake_merchant_options(fake_merchant_unsold_relic_ids)
    raise ValueError(f"Unknown special event id: {event_id}")


def special_event_implementations() -> tuple[dict[str, object], ...]:
    """Return audit metadata for supported bespoke special events."""

    return tuple(
        {
            "event_id": event_id,
            "category": "special",
            "option_ids": tuple(option.option_id for option in special_event_options(event_id)),
            "covers_all_options": True,
            "notes": "Bespoke special-event primitive handler is present.",
        }
        for event_id in SPECIAL_EVENT_IDS
    )


def special_event_room_state(
    event_id: str,
    *,
    hp: int,
    max_hp: int,
    gold: int = 0,
    deck: Sequence[str] | None = None,
    relics: Sequence[str] = (),
    potions: Sequence[str] = (),
    relic_trader_offered_relic_ids: Sequence[str] = (),
    fake_merchant_unsold_relic_ids: Sequence[str] = (),
) -> EventRoomState:
    """Create an EventRoomState populated with bespoke special event options."""

    return EventRoomState(
        event_id=event_id,
        hp=hp,
        max_hp=max_hp,
        gold=gold,
        deck=tuple(deck or ()),
        relics=tuple(relics),
        potions=tuple(potions),
        options=special_event_options(
            event_id,
            deck=deck,
            relic_trader_offered_relic_ids=relic_trader_offered_relic_ids,
            fake_merchant_unsold_relic_ids=fake_merchant_unsold_relic_ids,
        ),
    )


def legal_special_event_option_ids(state: EventRoomState) -> tuple[str, ...]:
    """Return legal bespoke event option ids for the current primitive state."""

    return tuple(
        option.option_id for option in state.options if _special_option_is_legal(option, state)
    )


def available_special_event_option_ids(state: EventRoomState) -> tuple[str, ...]:
    """Alias matching the naming used by other mechanics modules."""

    return legal_special_event_option_ids(state)


def resolve_special_event_option(
    state: EventRoomState,
    option_id: str,
    *,
    rng: Random | None = None,
    relic_pool: Sequence[PoolItem] = (),
    potion_pool: Sequence[PoolItem] = (),
    spent_potion_ids: Sequence[str] = (),
    spent_relic_ids: Sequence[str] = (),
) -> EventSpecialResolution:
    """Resolve a bespoke event option without importing engine RunState.

    The returned ``EventResolution`` applies immediate option effects.  Effects
    marked in option metadata as ``post_combat_reward`` remain metadata for a
    later combat/reward layer.
    """

    option = _option_by_id(state, option_id)
    if option.option_id not in legal_special_event_option_ids(state):
        raise ValueError(f"Special event option is not legal: {option_id}")

    metadata = dict(option.metadata)
    cost_state, spent_potions, spent_relics = _apply_special_costs(
        state,
        option,
        spent_potion_ids=spent_potion_ids,
        spent_relic_ids=spent_relic_ids,
    )
    gold_bonus = _metadata_gold_bonus(metadata, rng)
    effective_option = replace(
        option,
        gold_delta=option.gold_delta + gold_bonus,
    )
    effective_state = _state_with_effective_option(cost_state, effective_option)
    resolution = resolve_event_option(
        effective_state,
        effective_option.option_id,
        rng=rng,
        relic_pool=relic_pool,
        potion_pool=_filtered_potion_pool(effective_option, potion_pool),
    )
    adjusted_resolution = replace(
        resolution,
        gold_delta=resolution.state.gold - state.gold,
        hp_delta=resolution.state.hp - state.hp,
        max_hp_delta=resolution.state.max_hp - state.max_hp,
    )
    return EventSpecialResolution(
        resolution=adjusted_resolution,
        spent_potion_ids=spent_potions,
        spent_relic_ids=spent_relics,
        metadata=metadata,
    )


def _special_option_is_legal(option: EventOption, state: EventRoomState) -> bool:
    metadata = option.metadata
    if metadata.get("locked") or metadata.get("summary_marker"):
        return False

    resolved = {_normalized_id(option_id) for option_id in state.resolved_option_ids}
    if _normalized_id(option.option_id) in resolved:
        return False

    deck = {_normalized_id(card_id) for card_id in state.deck}
    required_cards = {_normalized_id(card_id) for card_id in option.required_card_ids}
    if not required_cards <= deck:
        return False

    relics = {_normalized_id(relic_id) for relic_id in state.relics}
    required_relics = {_normalized_id(relic_id) for relic_id in option.required_relic_ids}
    if not required_relics <= relics:
        return False

    if _metadata_int(metadata, "required_gold") > state.gold:
        return False
    if _metadata_int(metadata, "required_potion_count") > len(state.potions):
        return False
    return _metadata_int(metadata, "required_relic_count") <= len(state.relics)


def _apply_special_costs(
    state: EventRoomState,
    option: EventOption,
    *,
    spent_potion_ids: Sequence[str],
    spent_relic_ids: Sequence[str],
) -> tuple[EventRoomState, tuple[str, ...], tuple[str, ...]]:
    metadata = option.metadata
    next_gold = max(0, state.gold - _metadata_int(metadata, "gold_cost"))
    next_potions, spent_potions = _remove_cost_ids(
        state.potions,
        count=_metadata_int(metadata, "potion_cost_count"),
        requested_ids=spent_potion_ids,
        item_name="potion",
    )
    next_relics, spent_relics = _remove_cost_ids(
        state.relics,
        count=_metadata_int(metadata, "relic_cost_count"),
        requested_ids=spent_relic_ids,
        item_name="relic",
    )
    return (
        replace(state, gold=next_gold, potions=next_potions, relics=next_relics),
        spent_potions,
        spent_relics,
    )


def _remove_cost_ids(
    values: tuple[str, ...],
    *,
    count: int,
    requested_ids: Sequence[str],
    item_name: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if count <= 0:
        return values, ()
    if len(values) < count:
        raise ValueError(f"Not enough {item_name}s to pay event cost.")

    requested = tuple(_normalized_id(item_id) for item_id in requested_ids)
    if requested and len(requested) != count:
        raise ValueError(f"Expected {count} {item_name} cost id(s), got {len(requested)}.")

    remaining = list(values)
    spent: list[str] = []
    targets = requested if requested else tuple(_normalized_id(value) for value in values[:count])
    for target in targets:
        for index, value in enumerate(remaining):
            if _normalized_id(value) != target:
                continue
            spent.append(remaining.pop(index))
            break
        else:
            raise ValueError(f"Event cost {item_name} is not available: {target}")
    return tuple(remaining), tuple(spent)


def _metadata_gold_bonus(metadata: Mapping[str, object], rng: Random | None) -> int:
    value = metadata.get("gold_range")
    if value is None:
        return 0
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise ValueError(f"Invalid event gold range metadata: {value!r}")
    low = int(value[0])
    high = int(value[1])
    if low > high:
        raise ValueError(f"Invalid event gold range: {(low, high)}")
    return low if rng is None else rng.randint(low, high)


def _filtered_potion_pool(
    option: EventOption,
    potion_pool: Sequence[PoolItem],
) -> Sequence[PoolItem]:
    rarity = option.metadata.get("potion_rarity")
    if rarity is None:
        return potion_pool

    target = _normalized_id(str(rarity))
    candidates = [
        item
        for item in potion_pool
        if not isinstance(item, str) and _pool_item_rarity(item) == target
    ]
    return tuple(candidates) if candidates else potion_pool


def _pool_item_rarity(item: Mapping[str, object]) -> str:
    value = item.get("rarity_key", item.get("rarity", ""))
    return _normalized_id(str(value))


def _deck_card_types(deck: Sequence[str] | None) -> frozenset[str] | None:
    if deck is None:
        return None
    card_types = _card_type_by_id()
    if not card_types and deck:
        return None
    return frozenset(
        card_type
        for card_id in deck
        if (card_type := card_types.get(_normalized_id(card_id))) in ENCHANTABLE_CARD_TYPES
    )


@cache
def _card_type_by_id() -> dict[str, str]:
    cards_path = Path(__file__).resolve().parents[3] / "data" / "cache" / "eng" / "cards.json"
    try:
        payload: object = json.loads(cards_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, Sequence) or isinstance(payload, str):
        return {}

    card_types: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        card_id = item.get("id")
        if card_id is None:
            continue
        card_type = item.get("type_key", item.get("type"))
        if card_type is None:
            continue
        card_types[_normalized_id(card_id)] = _normalized_id(card_type)
    return card_types


def _self_help_book_enchant_option(spec: Mapping[str, str]) -> EventOption:
    keyword = spec["enchant_keyword"]
    card_type = spec["card_type"]
    return EventOption(
        option_id=spec["option_id"],
        label=spec["label"],
        description=spec["description"],
        metadata={
            "enchant_keyword": keyword,
            "enchant_amount": 2,
            "card_type": card_type,
            "enchant_card_type": card_type,
        },
    )


def _self_help_book_locked_option(spec: Mapping[str, str]) -> EventOption:
    card_type = spec["card_type"]
    return EventOption(
        option_id=spec["locked_option_id"],
        label="Locked",
        description=spec["locked_description"],
        metadata={
            "locked": True,
            "disabled_reason": f"requires_{card_type}",
            "enchant_keyword": spec["enchant_keyword"],
            "enchant_amount": 2,
            "card_type": card_type,
            "enchant_card_type": card_type,
        },
    )


def _state_with_effective_option(
    state: EventRoomState,
    option: EventOption,
) -> EventRoomState:
    normalized = _normalized_id(option.option_id)
    options = tuple(
        option if _normalized_id(existing.option_id) == normalized else existing
        for existing in state.options
    )
    return replace(state, options=options)


def _option_by_id(state: EventRoomState, option_id: str) -> EventOption:
    normalized = _normalized_id(option_id)
    for option in state.options:
        if _normalized_id(option.option_id) == normalized:
            return option
    raise ValueError(f"Unknown special event option id: {option_id}")


def _metadata_int(metadata: Mapping[str, object], key: str) -> int:
    value = metadata.get(key, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        try:
            return max(0, int(value))
        except ValueError:
            return 0
    return 0


def _event_key(event_id: str) -> str:
    key = _normalized_id(event_id)
    return _EVENT_ALIASES.get(key, key)


def _normalized_id(value: object) -> str:
    return str(value).lower().replace("'", "").replace(" ", "_").replace("-", "_")


_EVENT_ALIASES = {
    "lantern_key": "the_lantern_key",
    "future_of_potions": "the_future_of_potions",
    "war_historian": "war_historian_repy",
    "repy": "war_historian_repy",
    "the_round_tea_party": "round_tea_party",
    "round_tea": "round_tea_party",
    "the_merchant???": "fake_merchant",
    "the_merchant": "fake_merchant",
    "merchant???": "fake_merchant",
}

_SELF_HELP_BOOK_ENCHANT_SPECS = (
    {
        "option_id": "READ_THE_BACK",
        "locked_option_id": "READ_THE_BACK_LOCKED",
        "label": "Read the Back",
        "description": "Choose an Attack to Enchant with Sharp 2.",
        "locked_description": "You don't have any Attacks that can be Enchanted.",
        "enchant_keyword": "Sharp",
        "card_type": "attack",
    },
    {
        "option_id": "READ_PASSAGE",
        "locked_option_id": "READ_PASSAGE_LOCKED",
        "label": "Read a Random Passage",
        "description": "Choose a Skill to Enchant with Nimble 2.",
        "locked_description": "You don't have any Skills that can be Enchanted.",
        "enchant_keyword": "Nimble",
        "card_type": "skill",
    },
    {
        "option_id": "READ_ENTIRE_BOOK",
        "locked_option_id": "READ_ENTIRE_BOOK_LOCKED",
        "label": "Read the Entire Book",
        "description": "Choose a Power to Enchant with Swift 2.",
        "locked_description": "You don't have any Powers that can be Enchanted.",
        "enchant_keyword": "Swift",
        "card_type": "power",
    },
)

__all__ = [
    "FOUL_POTION_ID",
    "HISTORY_COURSE_RELIC_ID",
    "INJURY_CARD_ID",
    "LANTERN_KEY_CARD_ID",
    "ROYAL_POISON_RELIC_ID",
    "SPECIAL_EVENT_IDS",
    "EventSpecialResolution",
    "available_special_event_option_ids",
    "battleworn_dummy_options",
    "colorful_philosophers_options",
    "dense_vegetation_options",
    "fake_merchant_options",
    "fake_merchant_summary_option",
    "future_of_potions_options",
    "infested_automaton_options",
    "lantern_key_options",
    "legal_special_event_option_ids",
    "potion_courier_options",
    "punch_off_options",
    "ranwid_the_elder_options",
    "relic_trader_options",
    "resolve_special_event_option",
    "round_tea_party_options",
    "self_help_book_options",
    "special_event_implementations",
    "special_event_options",
    "special_event_room_state",
    "war_historian_repy_options",
]
