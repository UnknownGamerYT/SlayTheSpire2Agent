"""Pure event-room option primitives.

This module intentionally stays below engine transitions: event options are
plain dataclasses, and resolving an option returns a state-like dataclass with
simple reward/effect fields already applied or marked for a later engine layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from random import Random

PoolItem = str | Mapping[str, object]


@dataclass(frozen=True, slots=True)
class EventOption:
    option_id: str
    label: str = ""
    description: str = ""
    gold_delta: int = 0
    hp_delta: int = 0
    heal_percent_max_hp: float = 0.0
    max_hp_delta: int = 0
    fixed_relic_ids: tuple[str, ...] = ()
    random_relic_count: int = 0
    fixed_potion_ids: tuple[str, ...] = ()
    random_potion_count: int = 0
    fixed_card_ids: tuple[str, ...] = ()
    remove_card_ids: tuple[str, ...] = ()
    required_card_ids: tuple[str, ...] = ()
    required_relic_ids: tuple[str, ...] = ()
    upgrade_random_count: int = 0
    transform_random_count: int = 0
    remove_random_count: int = 0
    combat_encounter: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_id", str(self.option_id))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "gold_delta", int(self.gold_delta))
        object.__setattr__(self, "hp_delta", int(self.hp_delta))
        object.__setattr__(self, "heal_percent_max_hp", max(0.0, float(self.heal_percent_max_hp)))
        object.__setattr__(self, "max_hp_delta", int(self.max_hp_delta))
        object.__setattr__(self, "fixed_relic_ids", _normalized_ids(self.fixed_relic_ids))
        object.__setattr__(self, "fixed_potion_ids", _normalized_ids(self.fixed_potion_ids))
        object.__setattr__(self, "fixed_card_ids", _normalized_ids(self.fixed_card_ids))
        object.__setattr__(self, "remove_card_ids", _normalized_ids(self.remove_card_ids))
        object.__setattr__(self, "required_card_ids", _normalized_ids(self.required_card_ids))
        object.__setattr__(self, "required_relic_ids", _normalized_ids(self.required_relic_ids))
        object.__setattr__(self, "random_relic_count", max(0, int(self.random_relic_count)))
        object.__setattr__(self, "random_potion_count", max(0, int(self.random_potion_count)))
        object.__setattr__(self, "upgrade_random_count", max(0, int(self.upgrade_random_count)))
        object.__setattr__(
            self,
            "transform_random_count",
            max(0, int(self.transform_random_count)),
        )
        object.__setattr__(self, "remove_random_count", max(0, int(self.remove_random_count)))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class EventRoomState:
    event_id: str
    hp: int
    max_hp: int
    gold: int = 0
    deck: tuple[str, ...] = ()
    relics: tuple[str, ...] = ()
    potions: tuple[str, ...] = ()
    options: tuple[EventOption, ...] = ()
    resolved_option_ids: tuple[str, ...] = ()
    upgrade_random_count: int = 0
    transform_random_count: int = 0
    remove_random_count: int = 0
    combat_encounter: str | None = None

    def __post_init__(self) -> None:
        max_hp = max(1, int(self.max_hp))
        object.__setattr__(self, "event_id", str(self.event_id))
        object.__setattr__(self, "max_hp", max_hp)
        object.__setattr__(self, "hp", min(max(0, int(self.hp)), max_hp))
        object.__setattr__(self, "gold", max(0, int(self.gold)))
        object.__setattr__(self, "deck", _normalized_ids(self.deck))
        object.__setattr__(self, "relics", _normalized_ids(self.relics))
        object.__setattr__(self, "potions", _normalized_ids(self.potions))
        object.__setattr__(self, "options", tuple(self.options))
        object.__setattr__(
            self,
            "resolved_option_ids",
            tuple(str(option_id) for option_id in self.resolved_option_ids),
        )
        object.__setattr__(self, "upgrade_random_count", max(0, int(self.upgrade_random_count)))
        object.__setattr__(
            self,
            "transform_random_count",
            max(0, int(self.transform_random_count)),
        )
        object.__setattr__(self, "remove_random_count", max(0, int(self.remove_random_count)))


@dataclass(frozen=True, slots=True)
class EventResolution:
    option: EventOption
    state: EventRoomState
    gold_delta: int = 0
    hp_delta: int = 0
    heal_amount: int = 0
    max_hp_delta: int = 0
    added_card_ids: tuple[str, ...] = ()
    removed_card_ids: tuple[str, ...] = ()
    relic_ids: tuple[str, ...] = ()
    potion_ids: tuple[str, ...] = ()
    upgrade_random_count: int = 0
    transform_random_count: int = 0
    remove_random_count: int = 0
    combat_encounter: str | None = None


EventOutcome = EventResolution


def known_event_options(event_id: str) -> tuple[EventOption, ...]:
    """Return built-in option primitives for a known event id."""

    key = _event_key(event_id)
    options = _KNOWN_EVENT_OPTIONS.get(key)
    if options is None:
        raise ValueError(f"Unknown event id: {event_id}")
    return options


def event_room_state(
    event_id: str,
    *,
    hp: int,
    max_hp: int,
    gold: int = 0,
    deck: Sequence[str] = (),
    relics: Sequence[str] = (),
    potions: Sequence[str] = (),
    options: Sequence[EventOption] | None = None,
) -> EventRoomState:
    """Create an event-room state with known options when no options are supplied."""

    return EventRoomState(
        event_id=event_id,
        hp=hp,
        max_hp=max_hp,
        gold=gold,
        deck=tuple(deck),
        relics=tuple(relics),
        potions=tuple(potions),
        options=tuple(options) if options is not None else known_event_options(event_id),
    )


def legal_event_option_ids(state: EventRoomState) -> tuple[str, ...]:
    """List option ids currently legal for the event-room state."""

    if state.resolved_option_ids:
        return ()
    return tuple(option.option_id for option in state.options if _option_is_legal(option, state))


def available_event_option_ids(state: EventRoomState) -> tuple[str, ...]:
    """Alias matching other mechanics modules' available_* naming."""

    return legal_event_option_ids(state)


def resolve_event_option(
    state: EventRoomState,
    option_id: str,
    *,
    rng: Random | None = None,
    relic_pool: Sequence[PoolItem] = (),
    potion_pool: Sequence[PoolItem] = (),
) -> EventResolution:
    """Resolve a simple event option into a new event-room state."""

    option = _option_by_id(state, option_id)
    if option.option_id not in legal_event_option_ids(state):
        raise ValueError(f"Event option is not legal: {option_id}")
    option = _effective_event_option(state, option)

    next_max_hp = max(1, state.max_hp + option.max_hp_delta)
    hp_after_max = _hp_after_max_hp_change(state.hp, state.max_hp, next_max_hp)
    hp_after_delta = min(max(0, hp_after_max + option.hp_delta), next_max_hp)
    heal_amount = 0
    if option.heal_percent_max_hp > 0:
        requested_heal = _heal_amount(next_max_hp, option.heal_percent_max_hp)
        healed_hp = min(next_max_hp, hp_after_delta + requested_heal)
        heal_amount = healed_hp - hp_after_delta
        hp_after_delta = healed_hp

    next_gold = max(0, state.gold + option.gold_delta)
    deck_after_removal, removed_card_ids = _remove_fixed_cards(
        state.deck,
        option.remove_card_ids,
    )
    added_card_ids = option.fixed_card_ids
    next_deck = deck_after_removal + added_card_ids

    random_relic_ids = _draw_pool_ids(
        rng,
        relic_pool,
        count=option.random_relic_count,
        excluded=state.relics + option.fixed_relic_ids,
        pool_name="relic",
    )
    relic_ids = option.fixed_relic_ids + random_relic_ids

    random_potion_ids = _draw_pool_ids(
        rng,
        potion_pool,
        count=option.random_potion_count,
        pool_name="potion",
    )
    potion_ids = option.fixed_potion_ids + random_potion_ids
    next_combat = option.combat_encounter or state.combat_encounter

    next_state = replace(
        state,
        hp=hp_after_delta,
        max_hp=next_max_hp,
        gold=next_gold,
        deck=next_deck,
        relics=state.relics + relic_ids,
        potions=state.potions + potion_ids,
        resolved_option_ids=state.resolved_option_ids + (option.option_id,),
        upgrade_random_count=state.upgrade_random_count + option.upgrade_random_count,
        transform_random_count=state.transform_random_count + option.transform_random_count,
        remove_random_count=state.remove_random_count + option.remove_random_count,
        combat_encounter=next_combat,
    )
    return EventResolution(
        option=option,
        state=next_state,
        gold_delta=next_gold - state.gold,
        hp_delta=hp_after_delta - state.hp,
        heal_amount=heal_amount,
        max_hp_delta=next_max_hp - state.max_hp,
        added_card_ids=added_card_ids,
        removed_card_ids=removed_card_ids,
        relic_ids=relic_ids,
        potion_ids=potion_ids,
        upgrade_random_count=option.upgrade_random_count,
        transform_random_count=option.transform_random_count,
        remove_random_count=option.remove_random_count,
        combat_encounter=next_combat,
    )


def _option_by_id(state: EventRoomState, option_id: str) -> EventOption:
    key = _normalized_id(option_id)
    for option in state.options:
        if _normalized_id(option.option_id) == key:
            return option
    raise ValueError(f"Unknown event option id: {option_id}")


def _hp_after_max_hp_change(hp: int, max_hp: int, next_max_hp: int) -> int:
    actual_delta = next_max_hp - max(1, max_hp)
    if actual_delta > 0:
        return min(next_max_hp, max(0, hp) + actual_delta)
    return min(max(0, hp), next_max_hp)


def _option_is_legal(option: EventOption, state: EventRoomState) -> bool:
    resolved = {_normalized_id(option_id) for option_id in state.resolved_option_ids}
    if _normalized_id(option.option_id) in resolved:
        return False
    deck = {_normalized_id(card_id) for card_id in state.deck}
    relics = {_normalized_id(relic_id) for relic_id in state.relics}
    return set(option.required_card_ids) <= deck and set(option.required_relic_ids) <= relics


def _effective_event_option(state: EventRoomState, option: EventOption) -> EventOption:
    bonus = _mapping(option.metadata.get("multi_lantern_key_bonus"))
    if not bonus:
        return option
    required_key = _normalized_id(bonus.get("required_card_id", "lantern_key"))
    key_count = sum(1 for card_id in state.deck if _normalized_id(card_id) == required_key)
    if key_count < _int(bonus.get("required_count"), 2):
        return option

    remove_all_ids = _normalized_ids(_sequence(bonus.get("remove_all_card_ids")))
    remove_card_ids = option.remove_card_ids
    if remove_all_ids:
        remove_targets = set(remove_all_ids)
        remove_card_ids = tuple(
            card_id for card_id in state.deck if _normalized_id(card_id) in remove_targets
        )

    return replace(
        option,
        fixed_relic_ids=option.fixed_relic_ids
        + _normalized_ids(_sequence(bonus.get("fixed_relic_ids"))),
        random_relic_count=option.random_relic_count
        + _int(bonus.get("random_relic_count"), 0),
        random_potion_count=option.random_potion_count
        + _int(bonus.get("random_potion_count"), 0),
        remove_card_ids=remove_card_ids,
        metadata={
            **option.metadata,
            "multi_lantern_key_applied": True,
            "multi_lantern_key_count": key_count,
        },
    )


def _remove_fixed_cards(
    deck: tuple[str, ...],
    remove_card_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    next_deck = list(deck)
    removed: list[str] = []
    for card_id in remove_card_ids:
        target = _normalized_id(card_id)
        for index, deck_card_id in enumerate(next_deck):
            if _normalized_id(deck_card_id) != target:
                continue
            removed.append(next_deck.pop(index))
            break
    return tuple(next_deck), tuple(removed)


def _draw_pool_ids(
    rng: Random | None,
    pool: Sequence[PoolItem],
    *,
    count: int,
    excluded: Sequence[str] = (),
    pool_name: str,
) -> tuple[str, ...]:
    if count <= 0:
        return ()

    excluded_ids = {_normalized_id(item_id) for item_id in excluded}
    candidates = [
        item_id
        for item_id in _unique_pool_ids(pool)
        if _normalized_id(item_id) not in excluded_ids
    ]
    if len(candidates) < count:
        raise ValueError(
            f"Event random {pool_name} pool has {len(candidates)} eligible ids, "
            f"needs {count}."
        )
    if rng is None:
        return tuple(candidates[:count])
    return tuple(rng.sample(candidates, count))


def _unique_pool_ids(pool: Sequence[PoolItem]) -> tuple[str, ...]:
    seen: set[str] = set()
    ids: list[str] = []
    for item in pool:
        item_id = _pool_item_id(item)
        if item_id in seen:
            continue
        seen.add(item_id)
        ids.append(item_id)
    return tuple(ids)


def _pool_item_id(item: PoolItem) -> str:
    if isinstance(item, str):
        return _normalized_id(item)
    for key in ("id", "relic_id", "potion_id", "card_id", "item_id"):
        value = item.get(key)
        if value is not None:
            return _normalized_id(str(value))
    raise ValueError(f"Event reward pool item is missing an id: {item!r}")


def _heal_amount(max_hp: int, heal_percent_max_hp: float) -> int:
    fraction = (
        heal_percent_max_hp / 100.0
        if heal_percent_max_hp > 1
        else heal_percent_max_hp
    )
    return max(1, int(max_hp * fraction))


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value)
    return ()


def _int(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        with suppress(ValueError):
            return int(value)
    try:
        numeric = value.__int__  # type: ignore[attr-defined]
    except AttributeError:
        return default
    try:
        return int(numeric())
    except (TypeError, ValueError):
        return default


def _event_key(event_id: str) -> str:
    key = _normalized_id(event_id)
    return _EVENT_ALIASES.get(key, key)


def _normalized_ids(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(_normalized_id(value) for value in values)


def _normalized_id(value: object) -> str:
    return str(value).lower().replace("'", "").replace(" ", "_").replace("-", "_")


_LANTERN_KEY_OPTIONS = (
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
        fixed_card_ids=("lantern_key",),
        combat_encounter="normal",
        metadata={"reward_timing": "post_combat"},
    ),
)

_WAR_HISTORIAN_OPTIONS = (
    EventOption(
        option_id="UNLOCK_CAGE",
        label="Unlock the Cage",
        description="Lose Lantern Key. Obtain History Course.",
        fixed_relic_ids=("history_course",),
        remove_card_ids=("lantern_key",),
        required_card_ids=("lantern_key",),
        metadata={
            "multi_lantern_key_bonus": {
                "required_card_id": "lantern_key",
                "required_count": 2,
                "remove_all_card_ids": ("lantern_key",),
                "random_potion_count": 2,
                "random_relic_count": 2,
            },
        },
    ),
    EventOption(
        option_id="UNLOCK_CHEST",
        label="Unlock the Chest",
        description="Lose Lantern Key. Procure 2 random Potions. Obtain 2 random Relics.",
        random_relic_count=2,
        random_potion_count=2,
        remove_card_ids=("lantern_key",),
        required_card_ids=("lantern_key",),
        metadata={
            "multi_lantern_key_bonus": {
                "required_card_id": "lantern_key",
                "required_count": 2,
                "remove_all_card_ids": ("lantern_key",),
                "fixed_relic_ids": ("history_course",),
            },
        },
    ),
)

_BATTLEWORN_DUMMY_OPTIONS = (
    EventOption(
        option_id="SETTING_1",
        label="Setting 1",
        description="Fight a 75 HP dummy. Procure 1 random Potion.",
        random_potion_count=1,
        combat_encounter="battleworn_dummy",
        metadata={"monster_id": "training_dummy", "monster_hp": 75},
    ),
    EventOption(
        option_id="SETTING_2",
        label="Setting 2",
        description="Fight a 150 HP dummy. Upgrade 2 random cards.",
        upgrade_random_count=2,
        combat_encounter="battleworn_dummy",
        metadata={"monster_id": "training_dummy", "monster_hp": 150},
    ),
    EventOption(
        option_id="SETTING_3",
        label="Setting 3",
        description="Fight a 300 HP dummy. Obtain a random Relic.",
        random_relic_count=1,
        combat_encounter="battleworn_dummy",
        metadata={"monster_id": "training_dummy", "monster_hp": 300},
    ),
)

_DENSE_VEGETATION_OPTIONS = (
    EventOption(
        option_id="TRUDGE_ON",
        label="Trudge On",
        description="Gain 61-99 Gold. Lose 8 HP.",
        hp_delta=-8,
        metadata={"gold_range": (61, 99), "unsupported_effects": ("random_gold",)},
    ),
    EventOption(
        option_id="REST",
        label="Rest",
        description="Heal 30% Max HP. Fight some enemies.",
        heal_percent_max_hp=0.30,
        combat_encounter="normal",
    ),
)

_KNOWN_EVENT_OPTIONS = {
    "the_lantern_key": _LANTERN_KEY_OPTIONS,
    "war_historian_repy": _WAR_HISTORIAN_OPTIONS,
    "battleworn_dummy": _BATTLEWORN_DUMMY_OPTIONS,
    "dense_vegetation": _DENSE_VEGETATION_OPTIONS,
}

_EVENT_ALIASES = {
    "lantern_key": "the_lantern_key",
    "war_historian": "war_historian_repy",
    "repy": "war_historian_repy",
}

__all__ = [
    "EventOption",
    "EventOutcome",
    "EventResolution",
    "EventRoomState",
    "available_event_option_ids",
    "event_room_state",
    "known_event_options",
    "legal_event_option_ids",
    "resolve_event_option",
]
