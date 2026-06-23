"""Pure state machines for multi-page event flows.

The event-room primitives in :mod:`sts2sim.mechanics.event_rooms` model
single-shot options. This module is for bespoke events whose visible options
change across pages, repeat, lock, or defer effects. It intentionally has no
dependency on RunState or engine transitions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from random import Random

PoolItem = str | Mapping[str, object]


class EventFlowMarkerKind(str, Enum):
    """Structured marker for effects a pure flow cannot fully execute."""

    CARD_ADD = "card_add"
    CARD_REWARD = "card_reward"
    CARD_REMOVE = "card_remove"
    CARD_REMOVE_RANDOM = "card_remove_random"
    CARD_TRANSFORM = "card_transform"
    CARD_UPGRADE_ALL = "card_upgrade_all"
    CARD_UPGRADE_RANDOM = "card_upgrade_random"
    CARD_DOWNGRADE_RANDOM = "card_downgrade_random"
    CUSTOM_CARD = "custom_card"
    DELAYED_REWARD = "delayed_reward"
    ENCHANT = "enchant"
    FIXED_RELIC = "fixed_relic"
    FIXED_POTION = "fixed_potion"
    RANDOM_CARD = "random_card"
    RANDOM_POTION = "random_potion"
    RANDOM_RELIC = "random_relic"
    RUN_DEATH = "run_death"
    UNKNOWN = "unknown"
    UNKNOWN_BRANCH = "unknown_branch"


@dataclass(frozen=True, slots=True)
class EventFlowMarker:
    """A state-machine effect marker for the engine or caller to interpret later."""

    kind: EventFlowMarkerKind
    count: int = 1
    item_id: str | None = None
    qualifier: str | None = None
    delay_combat_count: int = 0
    description: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "count", max(0, int(self.count)))
        object.__setattr__(self, "delay_combat_count", max(0, int(self.delay_combat_count)))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class EventFlowOption:
    """One visible event option on a page."""

    option_id: str
    label: str = ""
    description: str = ""
    next_page_id: str | None = None
    terminal: bool = False
    locked: bool = False
    lock_reason: str = ""
    required_gold: int = 0
    gold_delta: int = 0
    hp_delta: int = 0
    max_hp_delta: int = 0
    set_max_hp: int | None = None
    heal_amount: int = 0
    markers: tuple[EventFlowMarker, ...] = ()
    repeatable: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_id", str(self.option_id))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "description", str(self.description))
        if self.next_page_id is not None:
            object.__setattr__(self, "next_page_id", str(self.next_page_id))
        object.__setattr__(self, "terminal", bool(self.terminal))
        object.__setattr__(self, "locked", bool(self.locked))
        object.__setattr__(self, "lock_reason", str(self.lock_reason))
        object.__setattr__(self, "required_gold", max(0, int(self.required_gold)))
        object.__setattr__(self, "gold_delta", int(self.gold_delta))
        object.__setattr__(self, "hp_delta", int(self.hp_delta))
        object.__setattr__(self, "max_hp_delta", int(self.max_hp_delta))
        if self.set_max_hp is not None:
            object.__setattr__(self, "set_max_hp", max(1, int(self.set_max_hp)))
        object.__setattr__(self, "heal_amount", max(0, int(self.heal_amount)))
        object.__setattr__(self, "markers", tuple(self.markers))
        object.__setattr__(self, "repeatable", bool(self.repeatable))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class EventFlowPage:
    """A page in a stepwise event."""

    page_id: str
    description: str = ""
    options: tuple[EventFlowOption, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "page_id", str(self.page_id))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "options", tuple(self.options))


@dataclass(frozen=True, slots=True)
class EventFlowState:
    """Pure event-flow state, independent of engine RunState."""

    event_id: str
    page_id: str = "INITIAL"
    hp: int = 1
    max_hp: int = 1
    gold: int = 0
    terminal: bool = False
    selected_option_ids: tuple[str, ...] = ()
    counters: Mapping[str, int] = field(default_factory=dict)
    data: Mapping[str, object] = field(default_factory=dict)
    markers: tuple[EventFlowMarker, ...] = ()

    def __post_init__(self) -> None:
        max_hp = max(1, int(self.max_hp))
        object.__setattr__(self, "event_id", _event_key(self.event_id))
        object.__setattr__(self, "page_id", str(self.page_id).upper())
        object.__setattr__(self, "max_hp", max_hp)
        object.__setattr__(self, "hp", min(max(0, int(self.hp)), max_hp))
        object.__setattr__(self, "gold", max(0, int(self.gold)))
        object.__setattr__(
            self,
            "selected_option_ids",
            tuple(str(option_id) for option_id in self.selected_option_ids),
        )
        object.__setattr__(
            self,
            "counters",
            {str(key): max(0, int(value)) for key, value in self.counters.items()},
        )
        object.__setattr__(self, "data", dict(self.data))
        object.__setattr__(self, "markers", tuple(self.markers))


@dataclass(frozen=True, slots=True)
class EventFlowResolution:
    """Result of choosing one event-flow option."""

    option: EventFlowOption
    state: EventFlowState
    markers: tuple[EventFlowMarker, ...] = ()
    gold_delta: int = 0
    hp_delta: int = 0
    max_hp_delta: int = 0
    heal_amount: int = 0
    terminal: bool = False


EventFlowOutcome = EventFlowResolution


@dataclass(frozen=True, slots=True)
class EventFlowRewardRequest:
    """A reward marker that needs a later reward-picker or delayed timing layer."""

    reward_kind: str
    count: int = 1
    qualifier: str | None = None
    delay_combat_count: int = 0
    description: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reward_kind", _normalized_id(self.reward_kind))
        object.__setattr__(self, "count", max(0, int(self.count)))
        object.__setattr__(self, "delay_combat_count", max(0, int(self.delay_combat_count)))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class EventFlowBlockedMarker:
    """A marker that was intentionally not applied by the pure marker resolver."""

    marker: EventFlowMarker
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", str(self.reason))


@dataclass(frozen=True, slots=True)
class EventFlowMarkerContext:
    """Minimal state needed to resolve deterministic event-flow markers."""

    deck: tuple[str, ...] = ()
    relics: tuple[str, ...] = ()
    potions: tuple[str, ...] = ()
    upgrade_random_count: int = 0
    upgrade_all_count: int = 0
    remove_random_count: int = 0
    transform_random_count: int = 0
    downgrade_random_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "deck", _normalized_ids(self.deck))
        object.__setattr__(self, "relics", _normalized_ids(self.relics))
        object.__setattr__(self, "potions", _normalized_ids(self.potions))
        object.__setattr__(
            self,
            "upgrade_random_count",
            max(0, int(self.upgrade_random_count)),
        )
        object.__setattr__(self, "upgrade_all_count", max(0, int(self.upgrade_all_count)))
        object.__setattr__(
            self,
            "remove_random_count",
            max(0, int(self.remove_random_count)),
        )
        object.__setattr__(
            self,
            "transform_random_count",
            max(0, int(self.transform_random_count)),
        )
        object.__setattr__(
            self,
            "downgrade_random_count",
            max(0, int(self.downgrade_random_count)),
        )


@dataclass(frozen=True, slots=True)
class EventFlowMarkerApplication:
    """Result of applying event-flow markers to a minimal pure context."""

    context: EventFlowMarkerContext
    added_card_ids: tuple[str, ...] = ()
    removed_card_ids: tuple[str, ...] = ()
    relic_ids: tuple[str, ...] = ()
    potion_ids: tuple[str, ...] = ()
    reward_requests: tuple[EventFlowRewardRequest, ...] = ()
    blocked_markers: tuple[EventFlowBlockedMarker, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "added_card_ids", _normalized_ids(self.added_card_ids))
        object.__setattr__(self, "removed_card_ids", _normalized_ids(self.removed_card_ids))
        object.__setattr__(self, "relic_ids", _normalized_ids(self.relic_ids))
        object.__setattr__(self, "potion_ids", _normalized_ids(self.potion_ids))
        object.__setattr__(self, "reward_requests", tuple(self.reward_requests))
        object.__setattr__(self, "blocked_markers", tuple(self.blocked_markers))


def event_flow_state(
    event_id: str,
    *,
    hp: int,
    max_hp: int,
    gold: int = 0,
    page_id: str = "INITIAL",
    counters: Mapping[str, int] | None = None,
    data: Mapping[str, object] | None = None,
) -> EventFlowState:
    """Create a pure state for a known bespoke event flow."""

    key = _event_key(event_id)
    if key not in _EVENT_PAGE_BUILDERS:
        raise ValueError(f"Unknown event flow id: {event_id}")
    return EventFlowState(
        event_id=key,
        page_id=page_id,
        hp=hp,
        max_hp=max_hp,
        gold=gold,
        counters={} if counters is None else counters,
        data={} if data is None else data,
    )


def current_event_flow_page(state: EventFlowState) -> EventFlowPage:
    """Return the current page with options resolved for the current state."""

    builder = _EVENT_PAGE_BUILDERS.get(state.event_id)
    if builder is None:
        raise ValueError(f"Unknown event flow id: {state.event_id}")
    return builder(state)


def visible_event_flow_option_ids(state: EventFlowState) -> tuple[str, ...]:
    """Return option ids visible on the current page, including locked options."""

    if state.terminal:
        return ()
    return tuple(option.option_id for option in current_event_flow_page(state).options)


def legal_event_flow_option_ids(state: EventFlowState) -> tuple[str, ...]:
    """Return option ids currently selectable by the player/caller."""

    if is_event_flow_terminal(state):
        return ()
    return tuple(
        option.option_id
        for option in current_event_flow_page(state).options
        if not option.locked and state.gold >= option.required_gold
    )


def available_event_flow_option_ids(state: EventFlowState) -> tuple[str, ...]:
    """Alias matching other mechanics modules' available_* naming."""

    return legal_event_flow_option_ids(state)


def is_event_flow_terminal(state: EventFlowState) -> bool:
    """Return whether the flow has reached a terminal page or lethal resolution."""

    if state.terminal:
        return True
    return len(current_event_flow_page(state).options) == 0


def resolve_event_flow_option(
    state: EventFlowState,
    option_id: str,
) -> EventFlowResolution:
    """Apply a visible event-flow option and return the next pure state."""

    option = _option_by_id(current_event_flow_page(state).options, option_id)
    if option.option_id not in legal_event_flow_option_ids(state):
        raise ValueError(f"Event flow option is not legal: {option_id}")

    next_max_hp, hp_after_max = _apply_max_hp(state, option)
    hp_after_delta = min(max(0, hp_after_max + option.hp_delta), next_max_hp)
    hp_after_heal = min(next_max_hp, hp_after_delta + option.heal_amount)
    heal_amount = hp_after_heal - hp_after_delta
    next_gold = max(0, state.gold + option.gold_delta)
    next_page_id = _next_page_id(state, option)
    next_counters = _next_counters(state, option, next_page_id)
    next_data = _next_data(state, option)
    markers = _resolution_markers(state, option, hp_after_heal)
    terminal = option.terminal or hp_after_heal <= 0
    next_state = replace(
        state,
        page_id=next_page_id,
        hp=hp_after_heal,
        max_hp=next_max_hp,
        gold=next_gold,
        terminal=terminal,
        selected_option_ids=state.selected_option_ids + (option.option_id,),
        counters=next_counters,
        data=next_data,
        markers=state.markers + markers,
    )
    return EventFlowResolution(
        option=option,
        state=next_state,
        markers=markers,
        gold_delta=next_gold - state.gold,
        hp_delta=hp_after_heal - state.hp,
        max_hp_delta=next_max_hp - state.max_hp,
        heal_amount=heal_amount,
        terminal=is_event_flow_terminal(next_state),
    )


def resolve_event_flow_markers(
    markers: Sequence[EventFlowMarker],
    context: EventFlowMarkerContext | None = None,
    *,
    rng: Random | None = None,
    relic_pool: Sequence[PoolItem] = (),
    potion_pool: Sequence[PoolItem] = (),
) -> EventFlowMarkerApplication:
    """Apply event-flow markers that can be resolved without RunState.

    Random reward markers always produce a reward request.  When a pool is
    supplied, the same marker also draws deterministic concrete reward ids.
    Unsupported markers are returned as explicit blockers instead of being
    silently ignored.
    """

    current = EventFlowMarkerContext() if context is None else context
    added_card_ids: list[str] = []
    removed_card_ids: list[str] = []
    relic_ids: list[str] = []
    potion_ids: list[str] = []
    reward_requests: list[EventFlowRewardRequest] = []
    blocked_markers: list[EventFlowBlockedMarker] = []

    for marker in markers:
        if marker.kind is EventFlowMarkerKind.CARD_ADD:
            item_id = _marker_item_id(marker)
            if item_id is None:
                blocked_markers.append(_blocked_marker(marker, "missing_card_id"))
                continue
            current = replace(current, deck=current.deck + (item_id,))
            added_card_ids.append(item_id)
        elif marker.kind is EventFlowMarkerKind.CARD_REMOVE:
            item_id = _marker_item_id(marker)
            if item_id is None:
                blocked_markers.append(_blocked_marker(marker, "missing_card_id"))
                continue
            deck_after_removal, removed = _remove_marker_card(current.deck, item_id)
            if not removed:
                blocked_markers.append(_blocked_marker(marker, "card_not_in_deck"))
                continue
            current = replace(current, deck=deck_after_removal)
            removed_card_ids += list(removed)
        elif marker.kind is EventFlowMarkerKind.FIXED_RELIC:
            item_id = _marker_item_id(marker)
            if item_id is None:
                blocked_markers.append(_blocked_marker(marker, "missing_relic_id"))
                continue
            current = replace(current, relics=current.relics + (item_id,))
            relic_ids.append(item_id)
        elif marker.kind is EventFlowMarkerKind.FIXED_POTION:
            item_id = _marker_item_id(marker)
            if item_id is None:
                blocked_markers.append(_blocked_marker(marker, "missing_potion_id"))
                continue
            current = replace(current, potions=current.potions + (item_id,))
            potion_ids.append(item_id)
        elif marker.kind is EventFlowMarkerKind.RANDOM_RELIC:
            reward_requests.append(_reward_request_from_marker(marker))
            drawn = _draw_marker_pool_ids(
                rng,
                relic_pool,
                marker=marker,
                excluded=current.relics + tuple(relic_ids),
                pool_name="relic",
            )
            if drawn is None:
                continue
            if len(drawn) < marker.count:
                blocked_markers.append(_blocked_marker(marker, "random_relic_pool_insufficient"))
                continue
            current = replace(current, relics=current.relics + drawn)
            relic_ids.extend(drawn)
        elif marker.kind is EventFlowMarkerKind.RANDOM_POTION:
            reward_requests.append(_reward_request_from_marker(marker))
            drawn = _draw_marker_pool_ids(
                rng,
                potion_pool,
                marker=marker,
                excluded=(),
                pool_name="potion",
            )
            if drawn is None:
                continue
            if len(drawn) < marker.count:
                blocked_markers.append(_blocked_marker(marker, "random_potion_pool_insufficient"))
                continue
            current = replace(current, potions=current.potions + drawn)
            potion_ids.extend(drawn)
        elif marker.kind is EventFlowMarkerKind.DELAYED_REWARD:
            reward_requests.append(_reward_request_from_marker(marker))
        elif marker.kind is EventFlowMarkerKind.CARD_UPGRADE_RANDOM:
            current = replace(
                current,
                upgrade_random_count=current.upgrade_random_count + marker.count,
            )
        elif marker.kind is EventFlowMarkerKind.CARD_UPGRADE_ALL:
            current = replace(current, upgrade_all_count=current.upgrade_all_count + 1)
        elif marker.kind is EventFlowMarkerKind.CARD_REMOVE_RANDOM:
            current = replace(
                current,
                remove_random_count=current.remove_random_count + marker.count,
            )
        elif marker.kind is EventFlowMarkerKind.CARD_TRANSFORM:
            current = replace(
                current,
                transform_random_count=current.transform_random_count + marker.count,
            )
        elif marker.kind is EventFlowMarkerKind.CARD_DOWNGRADE_RANDOM:
            current = replace(
                current,
                downgrade_random_count=current.downgrade_random_count + marker.count,
            )
        else:
            blocked_markers.append(_blocked_marker(marker, _unsupported_marker_reason(marker)))

    return EventFlowMarkerApplication(
        context=current,
        added_card_ids=tuple(added_card_ids),
        removed_card_ids=tuple(removed_card_ids),
        relic_ids=tuple(relic_ids),
        potion_ids=tuple(potion_ids),
        reward_requests=tuple(reward_requests),
        blocked_markers=tuple(blocked_markers),
    )


def _option_by_id(
    options: Sequence[EventFlowOption],
    option_id: str,
) -> EventFlowOption:
    key = _normalized_id(option_id)
    for option in options:
        if _normalized_id(option.option_id) == key:
            return option
    raise ValueError(f"Unknown event flow option id: {option_id}")


def _marker_item_id(marker: EventFlowMarker) -> str | None:
    if not marker.item_id:
        return None
    item_id = _normalized_id(marker.item_id)
    return item_id or None


def _remove_marker_card(
    deck: tuple[str, ...],
    card_id: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    target = _normalized_id(card_id)
    next_deck = list(deck)
    for index, deck_card_id in enumerate(next_deck):
        if _normalized_id(deck_card_id) != target:
            continue
        removed = next_deck.pop(index)
        return tuple(next_deck), (_normalized_id(removed),)
    return deck, ()


def _reward_request_from_marker(marker: EventFlowMarker) -> EventFlowRewardRequest:
    reward_kind = (
        str(marker.qualifier)
        if marker.kind is EventFlowMarkerKind.DELAYED_REWARD and marker.qualifier
        else marker.kind.value
    )
    return EventFlowRewardRequest(
        reward_kind=reward_kind,
        count=marker.count,
        qualifier=marker.qualifier,
        delay_combat_count=marker.delay_combat_count,
        description=marker.description,
        metadata={
            "source_marker_kind": marker.kind.value,
            **dict(marker.metadata),
        },
    )


def _draw_marker_pool_ids(
    rng: Random | None,
    pool: Sequence[PoolItem],
    *,
    marker: EventFlowMarker,
    excluded: Sequence[str],
    pool_name: str,
) -> tuple[str, ...] | None:
    if marker.count <= 0:
        return ()
    if not pool:
        return None

    excluded_ids = {_normalized_id(item_id) for item_id in excluded}
    candidates = [
        item_id
        for item_id, qualifier in _qualified_pool_ids(pool)
        if item_id not in excluded_ids
        and _pool_qualifier_matches(qualifier, marker.qualifier)
    ]
    if len(candidates) < marker.count:
        return tuple(candidates)
    if rng is None:
        return tuple(candidates[: marker.count])
    return tuple(rng.sample(candidates, marker.count))


def _qualified_pool_ids(pool: Sequence[PoolItem]) -> tuple[tuple[str, str | None], ...]:
    seen: set[str] = set()
    ids: list[tuple[str, str | None]] = []
    for item in pool:
        item_id = _pool_item_id(item)
        if item_id in seen:
            continue
        seen.add(item_id)
        ids.append((item_id, _pool_item_qualifier(item)))
    return tuple(ids)


def _pool_item_id(item: PoolItem) -> str:
    if isinstance(item, str):
        return _normalized_id(item)
    for key in ("id", "relic_id", "potion_id", "card_id", "item_id"):
        value = item.get(key)
        if value is not None:
            return _normalized_id(str(value))
    raise ValueError(f"Event flow reward pool item is missing an id: {item!r}")


def _pool_item_qualifier(item: PoolItem) -> str | None:
    if isinstance(item, str):
        return None
    for key in ("rarity_key", "rarity", "tier", "qualifier"):
        value = item.get(key)
        if value is not None:
            return _normalized_id(str(value))
    return None


def _pool_qualifier_matches(
    pool_qualifier: str | None,
    marker_qualifier: str | None,
) -> bool:
    if marker_qualifier is None:
        return True
    return pool_qualifier == _normalized_id(marker_qualifier)


def _blocked_marker(marker: EventFlowMarker, reason: str) -> EventFlowBlockedMarker:
    return EventFlowBlockedMarker(marker=marker, reason=reason)


def _unsupported_marker_reason(marker: EventFlowMarker) -> str:
    if marker.kind is EventFlowMarkerKind.CUSTOM_CARD:
        return "requires_card_factory"
    if marker.kind in {EventFlowMarkerKind.CARD_REWARD, EventFlowMarkerKind.RANDOM_CARD}:
        return "requires_card_reward_picker"
    if marker.kind is EventFlowMarkerKind.ENCHANT:
        return "requires_card_selection"
    if marker.kind is EventFlowMarkerKind.RUN_DEATH:
        return "requires_run_state"
    if marker.kind is EventFlowMarkerKind.UNKNOWN_BRANCH:
        return "unknown_branch"
    if marker.kind is EventFlowMarkerKind.UNKNOWN:
        return "unknown_marker"
    return "unsupported_marker_kind"


def _apply_max_hp(
    state: EventFlowState,
    option: EventFlowOption,
) -> tuple[int, int]:
    if option.set_max_hp is not None:
        next_max_hp = option.set_max_hp
    else:
        next_max_hp = max(1, state.max_hp + option.max_hp_delta)
    return next_max_hp, min(state.hp, next_max_hp)


def _next_page_id(state: EventFlowState, option: EventFlowOption) -> str:
    if state.event_id == "ABYSSAL_BATHS" and option.option_id in {"IMMERSE", "LINGER"}:
        return _abyssal_linger_next_page(state, option)
    if state.event_id == "TRIAL" and option.option_id == "ACCEPT":
        return _trial_accept_next_page(state)
    return option.next_page_id or state.page_id


def _next_counters(
    state: EventFlowState,
    option: EventFlowOption,
    next_page_id: str,
) -> Mapping[str, int]:
    counters = dict(state.counters)
    if state.event_id == "ENDLESS_CONVEYOR" and option.repeatable:
        counters["endless_conveyor_dishes"] = counters.get("endless_conveyor_dishes", 0) + 1
    if state.event_id == "SLIPPERY_BRIDGE" and state.page_id == "HOLD_ON_LOOP":
        counters["bridge_hold_loop_count"] = counters.get("bridge_hold_loop_count", 0) + 1
    if state.event_id == "TABLET_OF_TRUTH" and option.option_id in {"DECIPHER", "DECIPHER_1"}:
        counters["tablet_decipher_steps"] = counters.get("tablet_decipher_steps", 0) + 1
    if state.event_id == "ABYSSAL_BATHS" and option.option_id in {"IMMERSE", "LINGER"}:
        counters["abyssal_baths_soaks"] = counters.get("abyssal_baths_soaks", 0) + 1
    if next_page_id != state.page_id:
        counters[f"visited_{_normalized_id(next_page_id)}"] = 1
    return counters


def _next_data(
    state: EventFlowState,
    option: EventFlowOption,
) -> Mapping[str, object]:
    data = dict(state.data)
    if state.event_id == "SLIPPERY_BRIDGE" and option.option_id.startswith("HOLD_ON"):
        data.update(_bridge_next_offer_data(state))
    if state.event_id == "TINKER_TIME" and state.page_id == "CHOOSE_CARD_TYPE":
        card_type = option.metadata.get("custom_card_type")
        if card_type is not None:
            data["custom_card_type"] = str(card_type)
    if state.event_id == "TINKER_TIME" and state.page_id == "CHOOSE_RIDER":
        rider_id = option.metadata.get("custom_card_rider")
        if rider_id is not None:
            data["custom_card_rider"] = str(rider_id)
    return data


def _resolution_markers(
    state: EventFlowState,
    option: EventFlowOption,
    next_hp: int,
) -> tuple[EventFlowMarker, ...]:
    markers = option.markers
    if state.event_id == "TINKER_TIME" and state.page_id == "CHOOSE_RIDER":
        markers += (_tinker_custom_card_marker(state, option),)
    if state.event_id == "TRIAL" and option.option_id == "ACCEPT":
        markers += _trial_accept_markers(state)
    if state.event_id == "ABYSSAL_BATHS" and state.page_id == "DEATH_WARNING":
        markers += (
            EventFlowMarker(
                kind=EventFlowMarkerKind.RUN_DEATH,
                description="Choosing to linger after the death warning is lethal.",
            ),
        )
    if next_hp <= 0:
        markers += (
            EventFlowMarker(
                kind=EventFlowMarkerKind.RUN_DEATH,
                description="HP reached 0 while resolving this event option.",
            ),
        )
    return markers


def _abyssal_linger_next_page(
    state: EventFlowState,
    option: EventFlowOption,
) -> str:
    if state.page_id == "DEATH_WARNING":
        return "DEATH_WARNING"

    next_max_hp = state.max_hp + option.max_hp_delta
    next_hp = min(max(0, min(state.hp, next_max_hp) + option.hp_delta), next_max_hp)
    next_page = "IMMERSE" if option.option_id == "IMMERSE" else _ABYSSAL_LINGER_PAGES.get(
        state.page_id,
        "DEATH_WARNING",
    )
    next_damage = _ABYSSAL_LINGER_DAMAGE_BY_PAGE.get(next_page)
    if next_damage is not None and next_hp <= next_damage:
        return "DEATH_WARNING"
    return next_page


def _trial_accept_next_page(state: EventFlowState) -> str:
    case = str(state.data.get("trial_case", "")).upper()
    if case in _TRIAL_CASE_PAGES:
        return case
    return "TRIAL_CASE_SELECTION"


def _trial_accept_markers(state: EventFlowState) -> tuple[EventFlowMarker, ...]:
    case = str(state.data.get("trial_case", "")).upper()
    if case in _TRIAL_CASE_PAGES:
        return ()
    return (
        EventFlowMarker(
            kind=EventFlowMarkerKind.UNKNOWN_BRANCH,
            description="The Trial case is selected outside this pure flow state.",
            metadata={"choices": _TRIAL_CASE_PAGES},
        ),
    )


def _abyssal_baths_page(state: EventFlowState) -> EventFlowPage:
    if state.page_id == "INITIAL":
        return EventFlowPage(
            "INITIAL",
            "Abyssal pools offer a costly soak or a safer abstention.",
            (
                EventFlowOption(
                    option_id="IMMERSE",
                    label="Immerse",
                    description="Gain 2 Max HP. Take 3 damage.",
                    next_page_id="IMMERSE",
                    max_hp_delta=2,
                    hp_delta=-3,
                ),
                EventFlowOption(
                    option_id="ABSTAIN",
                    label="Abstain",
                    description="Heal 10 HP.",
                    next_page_id="ABSTAIN",
                    terminal=True,
                    heal_amount=10,
                ),
            ),
        )
    if state.page_id == "ABSTAIN":
        return _terminal_page("ABSTAIN", "You gather crystalline salts and leave.")
    if state.page_id == "EXIT_BATHS":
        return _terminal_page("EXIT_BATHS", "You hop out of the bath.")
    if state.page_id == "DEATH_WARNING":
        return EventFlowPage(
            "DEATH_WARNING",
            "If you bathe any longer you will die.",
            (
                EventFlowOption(
                    option_id="LINGER",
                    label="Linger",
                    description="If you bathe any longer you will die.",
                    hp_delta=-state.hp,
                    terminal=True,
                ),
                _exit_baths_option(),
            ),
        )

    damage = _ABYSSAL_LINGER_DAMAGE_BY_PAGE.get(state.page_id)
    if damage is None:
        return _terminal_page(state.page_id, "Unknown Abyssal Baths page.")
    return EventFlowPage(
        state.page_id,
        "The bath grows more dangerous.",
        (
            EventFlowOption(
                option_id="LINGER",
                label="Linger",
                description=f"Gain 2 Max HP. Take {damage} damage.",
                max_hp_delta=2,
                hp_delta=-damage,
            ),
            _exit_baths_option(),
        ),
    )


def _exit_baths_option() -> EventFlowOption:
    return EventFlowOption(
        option_id="EXIT_BATHS",
        label="Exit Baths",
        next_page_id="EXIT_BATHS",
        terminal=True,
    )


def _colossal_flower_page(state: EventFlowState) -> EventFlowPage:
    if state.page_id == "INITIAL":
        return EventFlowPage(
            "INITIAL",
            "A colossal flower offers nectar or a painful path deeper.",
            (
                EventFlowOption(
                    option_id="EXTRACT_CURRENT_PRIZE_1",
                    label="Extract Nectar",
                    description="Gain 35 Gold.",
                    next_page_id="EXTRACT_CURRENT_PRIZE",
                    terminal=True,
                    gold_delta=35,
                ),
                EventFlowOption(
                    option_id="REACH_DEEPER_1",
                    label="Reach Deeper",
                    description="Enter deeper. Lose 5 HP.",
                    next_page_id="REACH_DEEPER_1",
                    hp_delta=-5,
                ),
            ),
        )
    if state.page_id == "REACH_DEEPER_1":
        return EventFlowPage(
            "REACH_DEEPER_1",
            "You push past a layer of razor-sharp petals.",
            (
                EventFlowOption(
                    option_id="EXTRACT_CURRENT_PRIZE_2",
                    label="Extract Nectar",
                    description="Gain 75 Gold.",
                    next_page_id="EXTRACT_CURRENT_PRIZE",
                    terminal=True,
                    gold_delta=75,
                ),
                EventFlowOption(
                    option_id="REACH_DEEPER_2",
                    label="Reach Deeper",
                    description="Enter even deeper. Lose 6 HP.",
                    next_page_id="REACH_DEEPER_2",
                    hp_delta=-6,
                ),
            ),
        )
    if state.page_id == "REACH_DEEPER_2":
        return EventFlowPage(
            "REACH_DEEPER_2",
            "The tingling turns into throbbing weakness.",
            (
                EventFlowOption(
                    option_id="EXTRACT_INSTEAD",
                    label="Extract Nectar",
                    description="Gain 135 Gold.",
                    next_page_id="EXTRACT_INSTEAD",
                    terminal=True,
                    gold_delta=135,
                ),
                EventFlowOption(
                    option_id="POLLINOUS_CORE",
                    label="Enter the Center",
                    description="Lose 7 HP. Obtain Pollinous Core.",
                    next_page_id="POLLINOUS_CORE",
                    terminal=True,
                    hp_delta=-7,
                    markers=(
                        EventFlowMarker(
                            kind=EventFlowMarkerKind.FIXED_RELIC,
                            item_id="pollinous_core",
                            description="Obtain Pollinous Core.",
                        ),
                    ),
                ),
            ),
        )
    return _terminal_page(state.page_id, "Colossal Flower terminal page.")


def _endless_conveyor_page(state: EventFlowState) -> EventFlowPage:
    if state.page_id == "INITIAL":
        return EventFlowPage(
            "INITIAL",
            "The chef works beside an endless belt of food.",
            (
                EventFlowOption(
                    option_id="OBSERVE_CHEF",
                    label="Observe the Chef",
                    description="Upgrade a random card in your Deck.",
                    next_page_id="OBSERVE_CHEF",
                    terminal=True,
                    markers=(
                        EventFlowMarker(
                            kind=EventFlowMarkerKind.CARD_UPGRADE_RANDOM,
                            description="Upgrade a random card in your deck.",
                        ),
                    ),
                ),
            ),
        )
    if state.page_id in {"LEAVE", "OBSERVE_CHEF"}:
        return _terminal_page(state.page_id, "Endless Conveyor terminal page.")
    if state.page_id == "GRAB_SOMETHING_OFF_THE_BELT":
        return EventFlowPage("GRAB_SOMETHING_OFF_THE_BELT", "Tasty.", (_leave_conveyor_option(),))
    if state.page_id != "ALL":
        return _terminal_page(state.page_id, "Unknown Endless Conveyor page.")

    if state.gold < 40:
        return EventFlowPage(
            "ALL",
            "The belt keeps moving, but paid dishes require gold.",
            (
                EventFlowOption(
                    option_id="LOCKED",
                    label="Broke",
                    description="You could eat more but you are out of Gold.",
                    locked=True,
                    lock_reason="Requires 40 Gold.",
                    required_gold=40,
                ),
                _leave_conveyor_option(),
            ),
        )

    return EventFlowPage(
        "ALL",
        "Dishes continue passing by on the belt.",
        _CONVEYOR_DISH_OPTIONS + (_leave_conveyor_option(),),
    )


def _leave_conveyor_option() -> EventFlowOption:
    return EventFlowOption(
        option_id="LEAVE",
        label="Leave",
        next_page_id="LEAVE",
        terminal=True,
    )


def _slippery_bridge_page(state: EventFlowState) -> EventFlowPage:
    if state.page_id == "INITIAL":
        return EventFlowPage(
            "INITIAL",
            "Rain and wind buffet a rickety bridge.",
            (
                _bridge_overcome_option(state),
                EventFlowOption(
                    option_id="HOLD_ON_0",
                    label="Hold On",
                    description="Lose 3 HP. The card above is randomized.",
                    next_page_id="HOLD_ON_0",
                    hp_delta=-3,
                    markers=(_bridge_randomized_marker(state),),
                ),
            ),
        )
    if state.page_id == "OVERCOME":
        return _terminal_page("OVERCOME", "I hate bridges.")
    if state.page_id == "HOLD_ON_LOOP":
        loop_count = state.counters.get("bridge_hold_loop_count", 0)
        damage = 11 + loop_count
        return EventFlowPage(
            "HOLD_ON_LOOP",
            "There is no additional bridge text for holding on this long.",
            (
                _bridge_overcome_option(state),
                EventFlowOption(
                    option_id="HOLD_ON_LOOP",
                    label="Hold On",
                    description=f"Lose {damage}+ HP. The card above is randomized.",
                    next_page_id="HOLD_ON_LOOP",
                    hp_delta=-damage,
                    repeatable=True,
                    markers=(_bridge_randomized_marker(state),),
                ),
            ),
        )

    next_option = _BRIDGE_HOLD_OPTIONS.get(state.page_id)
    if next_option is None:
        return _terminal_page(state.page_id, "Unknown Slippery Bridge page.")
    next_option_id, damage, next_page_id = next_option
    return EventFlowPage(
        state.page_id,
        "You keep holding on.",
        (
            _bridge_overcome_option(state),
            EventFlowOption(
                option_id=next_option_id,
                label="Hold On",
                description=f"Lose {damage} HP. The card above is randomized.",
                next_page_id=next_page_id,
                hp_delta=-damage,
                markers=(_bridge_randomized_marker(state),),
            ),
        ),
    )


def _bridge_overcome_option(state: EventFlowState) -> EventFlowOption:
    offer = _bridge_current_offer(state)
    if offer is None:
        marker = EventFlowMarker(
            kind=EventFlowMarkerKind.CARD_REMOVE_RANDOM,
            description="Remove a random card from your deck.",
            metadata={"offer_policy": "no_repeat_until_all_cards_offered"},
        )
        description = "A random card is removed from your Deck."
        metadata: dict[str, object] = {
            "offer_policy": "no_repeat_until_all_cards_offered",
        }
    else:
        marker = EventFlowMarker(
            kind=EventFlowMarkerKind.CARD_REMOVE,
            item_id=offer,
            description=f"Remove {offer} from your deck.",
            metadata={"offer_policy": "no_repeat_until_all_cards_offered"},
        )
        description = f"Remove {offer} from your Deck."
        metadata = {
            "offer_policy": "no_repeat_until_all_cards_offered",
            "offered_card_id": offer,
            "offered_card_ids": _bridge_offered_card_ids(state),
        }
    return EventFlowOption(
        option_id="OVERCOME",
        label="Overcome",
        description=description,
        next_page_id="OVERCOME",
        terminal=True,
        markers=(marker,),
        metadata=metadata,
    )


def _bridge_randomized_marker(state: EventFlowState) -> EventFlowMarker:
    next_offer_data = _bridge_next_offer_data(state)
    metadata: dict[str, object] = {
        "effect": "reroll_visible_card_removal_target",
        "offer_policy": "no_repeat_until_all_cards_offered",
    }
    next_offer = next_offer_data.get("bridge_current_offer")
    if isinstance(next_offer, str) and next_offer:
        metadata["next_offered_card_id"] = next_offer
    return EventFlowMarker(
        kind=EventFlowMarkerKind.UNKNOWN,
        description="The Overcome card-removal target is randomized again.",
        metadata=metadata,
    )


def _bridge_current_offer(state: EventFlowState) -> str | None:
    deck = _bridge_deck(state)
    if not deck:
        return None
    raw_current = state.data.get("bridge_current_offer")
    current = _normalized_id(raw_current) if raw_current is not None else ""
    if current in deck:
        return current
    offered = set(_bridge_offered_card_ids(state))
    for card_id in deck:
        if card_id not in offered:
            return card_id
    return deck[0]


def _bridge_next_offer_data(state: EventFlowState) -> Mapping[str, object]:
    deck = _bridge_deck(state)
    if not deck:
        return {}

    offered: list[str] = list(_bridge_offered_card_ids(state))
    current = _bridge_current_offer(state)
    if current is not None and current not in offered:
        offered.append(current)

    offered_set = set(offered)
    candidates = tuple(card_id for card_id in deck if card_id not in offered_set)
    if candidates:
        next_offer = candidates[0]
    else:
        offered = []
        next_offer = deck[0]
    offered.append(next_offer)
    return {
        "bridge_current_offer": next_offer,
        "bridge_offered_card_ids": tuple(offered),
    }


def _bridge_deck(state: EventFlowState) -> tuple[str, ...]:
    raw_deck = state.data.get("bridge_deck", ())
    if not isinstance(raw_deck, Sequence) or isinstance(raw_deck, (str, bytes, bytearray)):
        return ()
    seen: set[str] = set()
    deck: list[str] = []
    for raw_card_id in raw_deck:
        card_id = _normalized_id(raw_card_id)
        if not card_id or card_id in seen:
            continue
        seen.add(card_id)
        deck.append(card_id)
    return tuple(deck)


def _bridge_offered_card_ids(state: EventFlowState) -> tuple[str, ...]:
    raw_offered = state.data.get("bridge_offered_card_ids", ())
    if not isinstance(raw_offered, Sequence) or isinstance(raw_offered, (str, bytes, bytearray)):
        return ()
    deck = set(_bridge_deck(state))
    offered: list[str] = []
    for raw_card_id in raw_offered:
        card_id = _normalized_id(raw_card_id)
        if not card_id or card_id not in deck or card_id in offered:
            continue
        offered.append(card_id)
    return tuple(offered)


def _tablet_of_truth_page(state: EventFlowState) -> EventFlowPage:
    if state.page_id == "INITIAL":
        return EventFlowPage(
            "INITIAL",
            "A protected vault contains a tablet with decipherable inscriptions.",
            (
                EventFlowOption(
                    option_id="DECIPHER_1",
                    label="Decipher",
                    description="Lose 3 Max HP. Upgrade a random card.",
                    next_page_id="DECIPHER_1",
                    max_hp_delta=-3,
                    markers=(_upgrade_random_marker(),),
                ),
                EventFlowOption(
                    option_id="SMASH",
                    label="Smash",
                    description="Heal 20 HP.",
                    next_page_id="SMASH",
                    terminal=True,
                    heal_amount=20,
                ),
            ),
        )
    if state.page_id == "DECIPHER":
        return EventFlowPage(
            "DECIPHER",
            "You can stop reading the text and leave.",
            (
                EventFlowOption(
                    option_id="GIVE_UP",
                    label="Give Up",
                    description="Stop reading the text and leave.",
                    next_page_id="GIVE_UP",
                    terminal=True,
                ),
            ),
        )

    option = _TABLET_DECIPHER_OPTIONS.get(state.page_id)
    if option is not None:
        return EventFlowPage(state.page_id, "The tablet demands more.", (option,))
    return _terminal_page(state.page_id, "Tablet of Truth terminal page.")


def _trial_page(state: EventFlowState) -> EventFlowPage:
    if state.page_id == "INITIAL":
        return EventFlowPage(
            "INITIAL",
            "You are the Decider for today's Trial.",
            (
                EventFlowOption(
                    option_id="ACCEPT",
                    label="Accept",
                    description="Serve as today's Decider.",
                ),
                EventFlowOption(
                    option_id="REJECT",
                    label="Reject",
                    description="You are not allowed to Reject.",
                    next_page_id="REJECT",
                ),
            ),
        )
    if state.page_id == "REJECT":
        return EventFlowPage(
            "REJECT",
            "The Grand Arbiter threatens lethal punishment.",
            (
                EventFlowOption(
                    option_id="ACCEPT",
                    label="Accept",
                    description="Give in. Serve as today's Decider.",
                ),
                EventFlowOption(
                    option_id="DOUBLE_DOWN",
                    label="Double Down",
                    description="Face lethal repercussions.",
                    next_page_id="DOUBLE_DOWN",
                    terminal=True,
                    hp_delta=-state.hp,
                    markers=(
                        EventFlowMarker(
                            kind=EventFlowMarkerKind.RUN_DEATH,
                            description="Rejecting the Trial twice is lethal.",
                        ),
                    ),
                ),
            ),
        )
    if state.page_id == "TRIAL_CASE_SELECTION":
        return EventFlowPage(
            "TRIAL_CASE_SELECTION",
            "A Trial case must be selected externally or through this pseudo page.",
            (
                _trial_case_selection_option("MERCHANT"),
                _trial_case_selection_option("NOBLE"),
                _trial_case_selection_option("NONDESCRIPT"),
            ),
        )
    if state.page_id in _TRIAL_CASE_OPTIONS:
        return EventFlowPage(
            state.page_id,
            "A case is presented.",
            _TRIAL_CASE_OPTIONS[state.page_id],
        )
    return _terminal_page(state.page_id, "Trial terminal page.")


def _trial_case_selection_option(page_id: str) -> EventFlowOption:
    return EventFlowOption(
        option_id=f"SELECT_{page_id}",
        label=f"Resolve {page_id.title()} case",
        description="Non-player branch selection for pure mechanics testing.",
        next_page_id=page_id,
        markers=(
            EventFlowMarker(
                kind=EventFlowMarkerKind.UNKNOWN_BRANCH,
                description="Resolve the Trial's externally selected case.",
                metadata={"selected_page_id": page_id},
            ),
        ),
    )


def _wongos_page(state: EventFlowState) -> EventFlowPage:
    if state.page_id == "INITIAL":
        return EventFlowPage(
            "INITIAL",
            "Welcome to Wongo's.",
            (
                _wongo_paid_or_locked(
                    state,
                    paid_id="BARGAIN_BIN",
                    locked_id="BARGAIN_BIN_LOCKED",
                    label="Wongo's Bargain Bin",
                    price=100,
                    markers=(
                        EventFlowMarker(
                            kind=EventFlowMarkerKind.RANDOM_RELIC,
                            count=1,
                            qualifier="common",
                            description="Obtain 1 random Common Relic.",
                        ),
                    ),
                ),
                _wongo_paid_or_locked(
                    state,
                    paid_id="FEATURED_ITEM",
                    locked_id="FEATURED_ITEM_LOCKED",
                    label="Wongo's Featured Item",
                    price=200,
                    markers=(
                        EventFlowMarker(
                            kind=EventFlowMarkerKind.RANDOM_RELIC,
                            count=1,
                            description="Obtain a random Relic.",
                        ),
                    ),
                ),
                _wongo_paid_or_locked(
                    state,
                    paid_id="MYSTERY_BOX",
                    locked_id="MYSTERY_BOX_LOCKED",
                    label="Wongo's Mystery Box",
                    price=300,
                    markers=(
                        EventFlowMarker(
                            kind=EventFlowMarkerKind.DELAYED_REWARD,
                            count=3,
                            qualifier="random_relic",
                            delay_combat_count=5,
                            description="Obtain 3 random Relics after 5 combats.",
                        ),
                    ),
                ),
                EventFlowOption(
                    option_id="LEAVE",
                    label="Leave",
                    description="Downgrade a random card.",
                    next_page_id="LEAVE",
                    terminal=True,
                    markers=(
                        EventFlowMarker(
                            kind=EventFlowMarkerKind.CARD_DOWNGRADE_RANDOM,
                            description="Downgrade a random card.",
                        ),
                    ),
                ),
            ),
        )
    return _terminal_page(state.page_id, "Wongo's terminal page.")


def _tinker_time_page(state: EventFlowState) -> EventFlowPage:
    if state.page_id == "INITIAL":
        return EventFlowPage(
            "INITIAL",
            "A tinkerer offers to help create a custom card.",
            (
                EventFlowOption(
                    option_id="CHOOSE_CARD_TYPE",
                    label="Accept",
                    description="Create a custom card to add to your Deck.",
                    next_page_id="CHOOSE_CARD_TYPE",
                ),
            ),
        )
    if state.page_id == "CHOOSE_CARD_TYPE":
        return EventFlowPage(
            "CHOOSE_CARD_TYPE",
            "Choose the base type for the custom card.",
            (
                _tinker_card_type_option("ATTACK", "Weapon", "attack"),
                _tinker_card_type_option("SKILL", "Protector", "skill"),
                _tinker_card_type_option("POWER", "Gadget", "power"),
            ),
        )
    if state.page_id == "CHOOSE_RIDER":
        card_type = str(state.data.get("custom_card_type", "attack")).lower()
        rider_ids = _TINKER_RIDER_IDS_BY_CARD_TYPE.get(
            card_type,
            tuple(_TINKER_RIDER_SPECS),
        )
        return EventFlowPage(
            "CHOOSE_RIDER",
            "Choose the rider text for the custom card.",
            tuple(_tinker_rider_option(option_id) for option_id in rider_ids),
        )
    return _terminal_page(state.page_id, "Tinker Time terminal page.")


def _tinker_card_type_option(
    option_id: str,
    label: str,
    card_type: str,
) -> EventFlowOption:
    return EventFlowOption(
        option_id=option_id,
        label=label,
        description=f"Make a custom {card_type.title()}.",
        next_page_id="CHOOSE_RIDER",
        metadata={"custom_card_type": card_type},
    )


def _tinker_rider_option(option_id: str) -> EventFlowOption:
    spec = _TINKER_RIDER_SPECS[option_id]
    return EventFlowOption(
        option_id=option_id,
        label=str(spec["label"]),
        description=str(spec["description"]),
        next_page_id="DONE",
        terminal=True,
        metadata={
            "custom_card_rider": option_id.lower(),
            "custom_card_rider_effect": spec["effect"],
        },
    )


def _tinker_custom_card_marker(
    state: EventFlowState,
    option: EventFlowOption,
) -> EventFlowMarker:
    card_type = str(state.data.get("custom_card_type", "unknown"))
    rider_id = str(option.metadata.get("custom_card_rider", _normalized_id(option.option_id)))
    rider_spec = dict(_TINKER_RIDER_SPECS.get(option.option_id, {}))
    return EventFlowMarker(
        kind=EventFlowMarkerKind.CUSTOM_CARD,
        item_id=f"tinker_time_{card_type}_{rider_id}",
        qualifier=card_type,
        description="Create a custom Tinker Time card.",
        metadata={
            "card_type": card_type,
            "rider_id": rider_id,
            "rider_effect": option.metadata.get("custom_card_rider_effect"),
            "rider": rider_spec,
            "source_event_id": "TINKER_TIME",
        },
    )


def _wongo_paid_or_locked(
    state: EventFlowState,
    *,
    paid_id: str,
    locked_id: str,
    label: str,
    price: int,
    markers: tuple[EventFlowMarker, ...],
) -> EventFlowOption:
    if state.gold < price:
        return EventFlowOption(
            option_id=locked_id,
            label="Locked",
            description=f"Requires {price} Gold.",
            locked=True,
            lock_reason=f"Requires {price} Gold.",
            required_gold=price,
        )
    return EventFlowOption(
        option_id=paid_id,
        label=label,
        description=f"Pay {price} Gold.",
        next_page_id="AFTER_BUY",
        terminal=True,
        required_gold=price,
        gold_delta=-price,
        markers=markers,
    )


def _upgrade_random_marker(count: int = 1) -> EventFlowMarker:
    return EventFlowMarker(
        kind=EventFlowMarkerKind.CARD_UPGRADE_RANDOM,
        count=count,
        description=f"Upgrade {count} random card(s).",
    )


def _card_add_marker(card_id: str) -> EventFlowMarker:
    return EventFlowMarker(
        kind=EventFlowMarkerKind.CARD_ADD,
        item_id=card_id,
        description=f"Add {card_id} to your deck.",
    )


def _terminal_page(page_id: str, description: str = "") -> EventFlowPage:
    return EventFlowPage(page_id=page_id, description=description, options=())


def _locked_if_short_gold(
    option: EventFlowOption,
    gold: int,
) -> EventFlowOption:
    if gold >= option.required_gold:
        return option
    return replace(
        option,
        locked=True,
        lock_reason=f"Requires {option.required_gold} Gold.",
    )


def _event_key(event_id: str) -> str:
    key = _normalized_id(event_id)
    canonical = _EVENT_ALIASES.get(key, key).upper()
    if canonical in _EVENT_PAGE_BUILDERS:
        return canonical
    return canonical


def _normalized_ids(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(_normalized_id(value) for value in values)


def _normalized_id(value: object) -> str:
    return str(value).lower().replace("'", "").replace(" ", "_").replace("-", "_")


_ABYSSAL_LINGER_DAMAGE_BY_PAGE = {
    "IMMERSE": 4,
    "LINGER1": 5,
    "LINGER2": 6,
    "LINGER3": 7,
    "LINGER4": 8,
    "LINGER5": 9,
    "LINGER6": 10,
    "LINGER7": 11,
    "LINGER8": 12,
    "LINGER9": 13,
}

_ABYSSAL_LINGER_PAGES = {
    "IMMERSE": "LINGER1",
    "LINGER1": "LINGER2",
    "LINGER2": "LINGER3",
    "LINGER3": "LINGER4",
    "LINGER4": "LINGER5",
    "LINGER5": "LINGER6",
    "LINGER6": "LINGER7",
    "LINGER7": "LINGER8",
    "LINGER8": "LINGER9",
    "LINGER9": "DEATH_WARNING",
}

_CONVEYOR_DISH_OPTIONS = tuple(
    _locked_if_short_gold(option, 40)
    for option in (
        EventFlowOption(
            option_id="CAVIAR",
            label="Grab Caviar off the Belt",
            description="Pay 40 Gold. Gain 4 Max HP. Continue feasting!",
            required_gold=40,
            gold_delta=-40,
            max_hp_delta=4,
            repeatable=True,
        ),
        EventFlowOption(
            option_id="CLAM_ROLL",
            label="Grab Clam Roll off the Belt",
            description="Pay 40 Gold. Heal 10 HP. Continue feasting!",
            required_gold=40,
            gold_delta=-40,
            heal_amount=10,
            repeatable=True,
        ),
        EventFlowOption(
            option_id="FRIED_EEL",
            label="Grab Fried Eel off the Belt",
            description="Pay 40 Gold. Add a random Colorless Card.",
            required_gold=40,
            gold_delta=-40,
            repeatable=True,
            markers=(
                EventFlowMarker(
                    kind=EventFlowMarkerKind.RANDOM_CARD,
                    qualifier="colorless",
                    description="Add a random Colorless Card to your deck.",
                ),
            ),
        ),
        EventFlowOption(
            option_id="GOLDEN_FYSH",
            label="Grab Golden Fysh off the Belt",
            description="Lucky winner! Gain 75 Gold.",
            gold_delta=75,
            repeatable=True,
            markers=(
                EventFlowMarker(
                    kind=EventFlowMarkerKind.UNKNOWN,
                    description="Golden Fysh availability is randomized in game.",
                ),
            ),
        ),
        EventFlowOption(
            option_id="JELLY_LIVER",
            label="Grab Jelly Liver off the Belt",
            description="Pay 40 Gold. Transform a card.",
            required_gold=40,
            gold_delta=-40,
            repeatable=True,
            markers=(
                EventFlowMarker(
                    kind=EventFlowMarkerKind.CARD_TRANSFORM,
                    description="Transform a chosen card.",
                ),
            ),
        ),
        EventFlowOption(
            option_id="SEAPUNK_SALAD",
            label="Grab Seapunk Salad off the Belt",
            description="Pay 40 Gold. Add Feeding Frenzy.",
            required_gold=40,
            gold_delta=-40,
            repeatable=True,
            markers=(_card_add_marker("feeding_frenzy"),),
        ),
        EventFlowOption(
            option_id="SPICY_SNAPPY",
            label="Grab Spicy Snappy off the Belt",
            description="Pay 40 Gold. Upgrade a random card.",
            required_gold=40,
            gold_delta=-40,
            repeatable=True,
            markers=(_upgrade_random_marker(),),
        ),
        EventFlowOption(
            option_id="SUSPICIOUS_CONDIMENT",
            label="Grab Suspicious Condiment off the Belt",
            description="Pay 40 Gold. Procure a random Potion.",
            required_gold=40,
            gold_delta=-40,
            repeatable=True,
            markers=(
                EventFlowMarker(
                    kind=EventFlowMarkerKind.RANDOM_POTION,
                    description="Procure a random Potion.",
                ),
            ),
        ),
    )
)

_BRIDGE_HOLD_OPTIONS = {
    "HOLD_ON_0": ("HOLD_ON_1", 4, "HOLD_ON_1"),
    "HOLD_ON_1": ("HOLD_ON_2", 5, "HOLD_ON_2"),
    "HOLD_ON_2": ("HOLD_ON_3", 6, "HOLD_ON_3"),
    "HOLD_ON_3": ("HOLD_ON_4", 7, "HOLD_ON_4"),
    "HOLD_ON_4": ("HOLD_ON_5", 8, "HOLD_ON_5"),
    "HOLD_ON_5": ("HOLD_ON_6", 9, "HOLD_ON_6"),
    "HOLD_ON_6": ("HOLD_ON_LOOP", 10, "HOLD_ON_LOOP"),
}

_TABLET_DECIPHER_OPTIONS = {
    "DECIPHER_1": EventFlowOption(
        option_id="DECIPHER",
        label="Continue Deciphering",
        description="Lose 6 Max HP. Upgrade a random card.",
        next_page_id="DECIPHER_2",
        max_hp_delta=-6,
        markers=(_upgrade_random_marker(),),
    ),
    "DECIPHER_2": EventFlowOption(
        option_id="DECIPHER",
        label="Keep Deciphering",
        description="Lose 12 Max HP. Upgrade a random card.",
        next_page_id="DECIPHER_3",
        max_hp_delta=-12,
        markers=(_upgrade_random_marker(),),
    ),
    "DECIPHER_3": EventFlowOption(
        option_id="DECIPHER",
        label="KEEP DECIPHERING",
        description="Lose 24 Max HP. Upgrade a random card.",
        next_page_id="DECIPHER_4",
        max_hp_delta=-24,
        markers=(_upgrade_random_marker(),),
    ),
    "DECIPHER_4": EventFlowOption(
        option_id="DECIPHER",
        label="Lose Everything",
        description="Set Max HP to 1. Upgrade ALL cards.",
        next_page_id="DECIPHER_5",
        terminal=True,
        set_max_hp=1,
        markers=(
            EventFlowMarker(
                kind=EventFlowMarkerKind.CARD_UPGRADE_ALL,
                description="Upgrade ALL cards.",
            ),
        ),
    ),
}

_TRIAL_CASE_PAGES = ("MERCHANT", "NOBLE", "NONDESCRIPT")

_TRIAL_CASE_OPTIONS = {
    "MERCHANT": (
        EventFlowOption(
            option_id="GUILTY",
            label="DECIDE: Guilty",
            description="Add Regret. Obtain 2 random Relics.",
            next_page_id="MERCHANT_GUILTY",
            terminal=True,
            markers=(
                _card_add_marker("regret"),
                EventFlowMarker(
                    kind=EventFlowMarkerKind.RANDOM_RELIC,
                    count=2,
                    description="Obtain 2 random Relics.",
                ),
            ),
        ),
        EventFlowOption(
            option_id="INNOCENT",
            label="DECIDE: Innocent",
            description="Add Shame. Upgrade 2 cards.",
            next_page_id="MERCHANT_INNOCENT",
            terminal=True,
            markers=(_card_add_marker("shame"), _upgrade_random_marker(2)),
        ),
    ),
    "NOBLE": (
        EventFlowOption(
            option_id="GUILTY",
            label="DECIDE: Guilty",
            description="Heal 10 HP.",
            next_page_id="NOBLE_GUILTY",
            terminal=True,
            heal_amount=10,
        ),
        EventFlowOption(
            option_id="INNOCENT",
            label="DECIDE: Innocent",
            description="Add Regret. Obtain 300 Gold.",
            next_page_id="NOBLE_INNOCENT",
            terminal=True,
            gold_delta=300,
            markers=(_card_add_marker("regret"),),
        ),
    ),
    "NONDESCRIPT": (
        EventFlowOption(
            option_id="GUILTY",
            label="DECIDE: Guilty",
            description="Add Doubt. Gain 2 card rewards.",
            next_page_id="NONDESCRIPT_GUILTY",
            terminal=True,
            markers=(
                _card_add_marker("doubt"),
                EventFlowMarker(
                    kind=EventFlowMarkerKind.CARD_REWARD,
                    count=2,
                    description="Gain 2 card rewards.",
                ),
            ),
        ),
        EventFlowOption(
            option_id="INNOCENT",
            label="DECIDE: Innocent",
            description="Add Doubt. Transform 2 cards.",
            next_page_id="NONDESCRIPT_INNOCENT",
            terminal=True,
            markers=(
                _card_add_marker("doubt"),
                EventFlowMarker(
                    kind=EventFlowMarkerKind.CARD_TRANSFORM,
                    count=2,
                    description="Transform 2 cards.",
                ),
            ),
        ),
    ),
}

_TINKER_RIDER_SPECS = {
    "SAPPING": {
        "label": "Sapping",
        "description": "Apply 2 Weak. Apply 2 Vulnerable.",
        "effect": "apply_debuffs",
        "weak": 2,
        "vulnerable": 2,
    },
    "VIOLENCE": {
        "label": "Violence",
        "description": "Hits 2 additional times.",
        "effect": "additional_hits",
        "additional_hits": 2,
    },
    "CHOKING": {
        "label": "Choking",
        "description": "Whenever you play a card this turn, the enemy loses 6 HP.",
        "effect": "choking",
        "hp_loss_per_card": 6,
    },
    "ENERGIZED": {
        "label": "Energized",
        "description": "Gain 2 Energy.",
        "effect": "gain_energy",
        "energy": 2,
    },
    "WISDOM": {
        "label": "Wisdom",
        "description": "Draw 3 cards.",
        "effect": "draw_cards",
        "draw": 3,
    },
    "CHAOS": {
        "label": "Chaos",
        "description": "Add a random card into your Hand. It's free to play this turn.",
        "effect": "add_random_free_card_to_hand",
        "random_card_count": 1,
    },
    "EXPERTISE": {
        "label": "Expertise",
        "description": "Gain 2 Strength. Gain 2 Dexterity.",
        "effect": "gain_strength_and_dexterity",
        "strength": 2,
        "dexterity": 2,
    },
    "CURIOUS": {
        "label": "Curious",
        "description": "Powers cost 1 Energy less.",
        "effect": "power_cost_reduction",
        "power_cost_reduction": 1,
    },
    "IMPROVEMENT": {
        "label": "Improvement",
        "description": "At the end of combat, Upgrade a random card.",
        "effect": "end_of_combat_upgrade_random",
        "upgrade_random_count": 1,
    },
}

_TINKER_RIDER_IDS_BY_CARD_TYPE = {
    "attack": ("SAPPING", "VIOLENCE", "CHOKING"),
    "skill": ("ENERGIZED", "WISDOM", "CHAOS"),
    "power": ("EXPERTISE", "CURIOUS", "IMPROVEMENT"),
}

_EVENT_ALIASES = {
    "abyssal_baths": "ABYSSAL_BATHS",
    "colossal_flower": "COLOSSAL_FLOWER",
    "endless_conveyor": "ENDLESS_CONVEYOR",
    "slippery_bridge": "SLIPPERY_BRIDGE",
    "tablet_of_truth": "TABLET_OF_TRUTH",
    "tinker_time": "TINKER_TIME",
    "trial": "TRIAL",
    "the_trial": "TRIAL",
    "welcome_to_wongos": "WELCOME_TO_WONGOS",
}

_EVENT_PAGE_BUILDERS = {
    "ABYSSAL_BATHS": _abyssal_baths_page,
    "COLOSSAL_FLOWER": _colossal_flower_page,
    "ENDLESS_CONVEYOR": _endless_conveyor_page,
    "SLIPPERY_BRIDGE": _slippery_bridge_page,
    "TABLET_OF_TRUTH": _tablet_of_truth_page,
    "TINKER_TIME": _tinker_time_page,
    "TRIAL": _trial_page,
    "WELCOME_TO_WONGOS": _wongos_page,
}

__all__ = [
    "EventFlowBlockedMarker",
    "EventFlowMarkerApplication",
    "EventFlowMarkerContext",
    "EventFlowMarker",
    "EventFlowMarkerKind",
    "EventFlowOption",
    "EventFlowOutcome",
    "EventFlowPage",
    "EventFlowRewardRequest",
    "EventFlowResolution",
    "EventFlowState",
    "available_event_flow_option_ids",
    "current_event_flow_page",
    "event_flow_state",
    "is_event_flow_terminal",
    "legal_event_flow_option_ids",
    "resolve_event_flow_markers",
    "resolve_event_flow_option",
    "visible_event_flow_option_ids",
]
