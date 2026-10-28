"""Stable shared event combat entry and Battleworn Dummy's Resume callback.

EventModel's extra rewards are added to ordinary encounter rewards. Only
BattlewornDummyEventEncounter overrides ShouldGiveRewards to false.
"""

from __future__ import annotations

from typing import Any

from sts2sim.content.source_card_pools import NON_UPGRADABLE_CARD_IDS
from sts2sim.content.source_potion_pools import source_potion_options
from sts2sim.mechanics.relic_bag_runtime import (
    draw_source_relic,
    has_source_relic_bag,
    source_relic_rarity,
)
from sts2sim.mechanics.source_rng import is_source_rng

from .errors import EngineError
from .models import EffectEvent, EventOptionState, EventState, RewardState, RunPhase, RunState
from .rng import capture_random_state, random_from_state, random_stream, stable_shuffle

ENCOUNTERS = {
    "battleworn_dummy": "BATTLEWORN_DUMMY_EVENT_ENCOUNTER",
    "dense_vegetation": "DENSE_VEGETATION_EVENT_ENCOUNTER",
    "punch_off": "PUNCH_OFF_EVENT_ENCOUNTER",
    "fake_merchant": "FAKE_MERCHANT_EVENT_ENCOUNTER",
    "the_lantern_key": "MYSTERIOUS_KNIGHT_EVENT_ENCOUNTER",
}


def event_fight_flags(event: EventState, option: EventOptionState) -> dict[str, Any]:
    from . import transitions as t

    event_id = t._normalized_id(event.event_id)
    if event_id not in ENCOUNTERS:
        return {}
    flags: dict[str, Any] = {"combat_encounter_id": ENCOUNTERS[event_id]}
    if event_id == "battleworn_dummy":
        setting_id = t._normalized_id(option.option_id)
        if setting_id not in {"setting_1", "setting_2", "setting_3"}:
            raise EngineError(f"Battleworn Dummy requires a selected setting, got {setting_id}")
        setting = int(setting_id[-1])
        flags.update(
            {
                "battleworn_dummy_setting": setting,
                "combat_monster_ids": (f"BATTLE_FRIEND_V{setting}",),
                "combat_reward_encounter": "event",
                "combat_reward_card_count": 0,
                "combat_reward_relic_count": 0,
                "combat_reward_potion_chance_percent": 0,
            }
        )
    elif event_id == "fake_merchant":
        flags.update(
            {
                "combat_reward_encounter": "normal",
                "fake_merchant_combat": True,
                "combat_reward_gold": 300,
                "combat_reward_gold_is_base": True,
            }
        )
        if "fake_merchant_unsold_relic_ids" in event.metadata:
            flags["fake_merchant_unsold_relic_ids"] = tuple(
                t._normalized_id(value)
                for value in t._string_items(
                    event.metadata["fake_merchant_unsold_relic_ids"],
                )
            )
    return flags


def is_battleworn_fight(state: RunState) -> bool:
    from . import transitions as t

    if t._normalized_id(str(state.flags.get("event_fight_id", ""))) == "battleworn_dummy":
        return True
    if state.combat is None:
        return False
    return bool(state.combat.metadata.get("battleworn_dummy_timed_out")) or any(
        t._normalized_id(str(m.metadata.get("encounter_id", "")))
        == "battleworn_dummy_event_encounter"
        for m in state.combat.monsters
    )


def validate_fake_inventory(
    state: RunState,
    event: EventState,
    flags: dict[str, Any],
) -> None:
    from . import transitions as t

    if t._normalized_id(event.event_id) != "fake_merchant":
        return
    if "fake_merchant_unsold_relic_ids" in event.metadata:
        return
    if "fake_merchant_unsold_relic_ids" in state.flags:
        flags["fake_merchant_unsold_relic_ids"] = state.flags["fake_merchant_unsold_relic_ids"]
        return
    supplied = t._fake_merchant_unsold_relic_ids(event)
    if supplied:
        flags["fake_merchant_unsold_relic_ids"] = supplied
    elif is_source_rng(state.rng):
        raise EngineError("Fake Merchant combat requires its source unsold inventory")


def empty_battleworn_rewards(state: RunState) -> RewardState:
    return RewardState(
        reward_id=f"battleworn_dummy:{len(state.room_history)}",
        source="event",
        metadata={"event_id": "battleworn_dummy", "should_give_encounter_rewards": False},
    )


def _setting(state: RunState) -> int:
    from . import transitions as t

    raw_setting = state.flags.get("battleworn_dummy_setting", 0)
    normalized = t._normalized_id(str(raw_setting)).replace("_", "")
    value = (
        int(normalized.removeprefix("setting"))
        if normalized
        in {
            "1",
            "2",
            "3",
            "setting1",
            "setting2",
            "setting3",
        }
        else 0
    )
    if value in (1, 2, 3):
        return value
    if state.event is not None:
        option = t._normalized_id(str(state.event.resolved_option_id or ""))
        if option in {"setting_1", "setting_2", "setting_3"}:
            return int(option[-1])
    if state.combat is not None:
        source_ids = {
            str(m.metadata.get("source_monster_id", "")).upper() for m in state.combat.monsters
        }
        for candidate in (1, 2, 3):
            if source_ids == {f"BATTLE_FRIEND_V{candidate}"}:
                return candidate
    raise EngineError("Battleworn Dummy resume requires the encounter's selected setting")


def resume_battleworn(state: RunState) -> tuple[RunState, tuple[EffectEvent, ...]] | None:
    from . import transitions as t

    combat = state.combat
    if (
        not is_battleworn_fight(state)
        or combat is None
        or state.phase != RunPhase.REWARD
        or combat.metadata.get("battleworn_resume_complete")
        or t._combat_resolution_is_pending(combat)
    ):
        return None
    timed_out = bool(combat.metadata.get("battleworn_dummy_timed_out"))
    combat = combat.model_copy(
        update={
            "metadata": {
                **combat.metadata,
                "battleworn_resume_complete": True,
            }
        }
    )
    state = state.model_copy(update={"combat": combat})
    event = EffectEvent(
        kind="event_combat_resumed", source_id="battleworn_dummy", metadata={"timed_out": timed_out}
    )
    if timed_out:
        return t._complete_current_room(state.model_copy(update={"reward": None}), (event,))
    setting = _setting(state)
    rng = random_from_state(state.rng)
    if setting == 1:
        pool = source_potion_options(
            state.character_id,
            unlocked_epochs=state.flags.get("revealed_epochs", state.flags.get("unlocked_epochs")),
        )
        if not pool:
            raise EngineError("Battleworn Dummy setting 1 requires unlocked class/shared potions")
        potion_id = random_stream(rng, "rewards").choice(pool)
        reward = RewardState(
            reward_id=f"battleworn_dummy:potion:{len(state.room_history)}",
            source="event",
            potion_id=potion_id,
            metadata={
                "event_id": "battleworn_dummy",
                "setting": 1,
                "potion_slots": t._potion_capacity(state),
                "current_potions": len(state.potions),
                "should_give_encounter_rewards": False,
            },
        )
        return state.model_copy(update={"rng": capture_random_state(rng), "reward": reward}), (
            event,
            *t._reward_generated_events(reward),
        )
    if setting == 2:
        deck = list(state.master_deck)
        candidates = [
            i
            for i, card in enumerate(deck)
            if not card.upgraded and card.card_id.lower() not in NON_UPGRADABLE_CARD_IDS
        ]
        chosen = stable_shuffle(
            candidates,
            random_stream(rng, "BATTLEWORN_DUMMY"),
            key=lambda i: (deck[i].card_id.upper(), int(deck[i].upgraded)),
        )[:2]
        events = [event]
        for index in chosen:
            card = deck[index]
            deck[index] = t._upgrade_card_instance(card)
            events.append(
                EffectEvent(
                    kind="card_upgraded",
                    source_id="battleworn_dummy",
                    target_id=card.instance_id,
                    metadata={"card_id": card.card_id, "setting": 2},
                )
            )
        state = state.model_copy(
            update={
                "master_deck": tuple(deck),
                "rng": capture_random_state(rng),
                "reward": None,
            }
        )
        return t._complete_current_room(state, tuple(events))
    flags = dict(state.flags)
    rewards_rng = random_stream(rng, "rewards")
    if has_source_relic_bag(state):
        relic_id = draw_source_relic(state, flags, source_relic_rarity(rewards_rng))
    elif is_source_rng(rng):
        raise EngineError("Battleworn Dummy setting 3 requires the source player relic bag")
    else:
        selected = t._draw_event_relic_ids_from_pool(
            state, rewards_rng, 1, t._treasure_relic_pool(state)
        )
        if not selected:
            raise EngineError("Battleworn Dummy setting 3 requires an eligible relic")
        relic_id = selected[0]
    state = state.model_copy(
        update={
            "rng": capture_random_state(rng),
            "flags": flags,
            "reward": None,
        }
    )
    state, relic_events = t._add_event_relic(state, relic_id)
    return t._complete_current_room(state, (event,) + relic_events)
