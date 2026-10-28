"""Pure named-act, room-set, Ancient allocation and initial relic-bag helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from random import Random
from typing import Any

from .generation_data import EVENT_EPOCH_IDS, RELIC_EPOCH_IDS, RELIC_POOL_IDS
from .monsters import EncounterDefinition
from .run_encounters import ACT_ENCOUNTER_POOLS, generate_act_encounters

ACT_NUMBERS = {"OVERGROWTH": 1, "UNDERDOCKS": 1, "HIVE": 2, "GLORY": 3}
SHARED_EVENTS = (
    "BRAIN_LEECH",
    "CRYSTAL_SPHERE",
    "DOLL_ROOM",
    "FAKE_MERCHANT",
    "POTION_COURIER",
    "RANWID_THE_ELDER",
    "RELIC_TRADER",
    "ROOM_FULL_OF_CHEESE",
    "SELF_HELP_BOOK",
    "SLIPPERY_BRIDGE",
    "STONE_OF_ALL_TIME",
    "SYMBIOTE",
    "TEA_MASTER",
    "THE_FUTURE_OF_POTIONS",
    "THE_LEGENDS_WERE_TRUE",
    "THIS_OR_THAT",
    "WAR_HISTORIAN_REPY",
    "WELCOME_TO_WONGOS",
)


def selected_act_id(act: int, source: Mapping[str, Any]) -> str:
    flags = source.get("flags", {})
    nested = flags if isinstance(flags, Mapping) else {}
    ids = source.get("act_ids", nested.get("act_ids", {}))
    selected = ids.get(str(act)) if isinstance(ids, Mapping) else None
    if selected is None:
        configured = str(source.get("act_id", nested.get("act_id", ""))).upper()
        if ACT_NUMBERS.get(configured) == act:
            selected = configured
    if selected is None:
        return {1: "OVERGROWTH", 2: "HIVE", 3: "GLORY"}.get(act, "GLORY")
    normalized = str(selected).upper()
    if ACT_NUMBERS.get(normalized) != act:
        raise ValueError(f"Act {act} cannot use named act {selected!r}")
    return normalized


def select_run_act_ids(source: Mapping[str, Any], rng: Random) -> dict[str, str]:
    """ActModel.GetRandomList, with caller overrides after source selection.

    Missing profile facts retain the project's all-unlocked training profile.
    Explicit revealed_epochs/discovered_act_ids reproduce unlock/discovery gates.
    """
    epochs = source.get("revealed_epochs")
    underdocks = epochs is None or "UNDERDOCKS_EPOCH" in epochs
    discovered = source.get("discovered_act_ids")
    if (
        underdocks
        and discovered is not None
        and "UNDERDOCKS" not in {str(value).upper() for value in discovered}
        and int(source.get("player_count", 1)) == 1
    ):
        first = "UNDERDOCKS"
    else:
        first = rng.choice(("OVERGROWTH", "UNDERDOCKS") if underdocks else ("OVERGROWTH",))
    # NextItem still consumes a draw for the singleton acts.
    selected = {"1": first, "2": rng.choice(("HIVE",)), "3": rng.choice(("GLORY",))}
    for number in (1, 2, 3):
        configured = selected_act_id(number, source)
        explicit = (
            (source.get("act_ids") or {}).get(str(number))
            if isinstance(source.get("act_ids"), Mapping)
            else None
        )
        if explicit is not None or str(source.get("act_id", "")).upper() == configured:
            selected[str(number)] = configured
    return selected


def initial_relic_bags(
    relics: Sequence[Mapping[str, Any]],
    character_id: str,
    rng: Random,
    *,
    unlocked_relic_ids: Sequence[str] | None = None,
    revealed_epochs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Populate the shared bag then the player's bag in source rarity order."""
    rows = {str(row.get("id", "")).upper(): row for row in relics}
    allowed = (
        {value.upper() for value in unlocked_relic_ids} if unlocked_relic_ids is not None else None
    )
    shared = RELIC_POOL_IDS["shared"]
    epochs = {value.upper() for value in revealed_epochs} if revealed_epochs is not None else None
    locked = {
        relic_id
        for epoch, ids in RELIC_EPOCH_IDS.items()
        if epochs is not None and epoch not in epochs
        for relic_id in ids
    }

    def build(ids: Sequence[str], *, player: bool, shuffle: bool = True) -> dict[str, list[str]]:
        bag: dict[str, list[str]] = {}
        for relic_id in ids:
            row = rows.get(relic_id)
            if (
                row is None
                or relic_id in locked
                or (allowed is not None and relic_id not in allowed)
            ):
                continue
            rarity = str(row.get("rarity_key", str(row.get("rarity", "")).split(" ")[0]))
            if player and rarity not in {"Common", "Uncommon", "Rare", "Shop"}:
                continue
            bag.setdefault(rarity, []).append(relic_id)
        if shuffle:
            for bucket in bag.values():
                rng.shuffle(bucket)
        return bag

    return {
        "shared_relic_bag_originals": build(shared, player=False, shuffle=False),
        "shared_relic_bag": build(shared, player=False),
        "player_relic_bag": build(
            shared + RELIC_POOL_IDS.get(character_id.lower(), ()), player=True
        ),
    }


def generate_run_room_sets(
    encounters: Sequence[EncounterDefinition],
    acts: Sequence[Mapping[str, Any]],
    rng: Random,
    *,
    act_ids: Mapping[str, str],
    profile: Mapping[str, Any],
    ascension: int = 0,
) -> dict[str, dict[str, Any]]:
    """RunManager.GenerateRooms and ActModel.GenerateRooms draw order."""
    epochs_raw = profile.get("revealed_epochs")
    epochs = {str(value).upper() for value in epochs_raw} if epochs_raw is not None else None
    shared = ["DARV"] if epochs is None or "DARV_EPOCH" in epochs else []
    rng.shuffle(shared)
    allocations: dict[str, list[str]] = {}
    for number in (2, 3):
        count = rng.randrange(len(shared) + 1)
        allocations[str(number)] = shared[:count]
        shared = shared[count:]
    act_rows = {str(row.get("id", "")).upper(): row for row in acts}
    result: dict[str, dict[str, Any]] = {}
    multiplayer = int(profile.get("player_count", 1)) > 1
    for number in (1, 2, 3):
        act_id = act_ids[str(number)]
        row = act_rows.get(act_id, {})
        event_ids = [str(value).upper() for value in row.get("events", ())] + list(SHARED_EVENTS)
        if epochs is not None:
            locked = {
                event
                for epoch, events in EVENT_EPOCH_IDS.items()
                if epoch not in epochs
                for event in events
            }
            event_ids = [event for event in event_ids if event not in locked]
        rng.shuffle(event_ids)
        pools = ACT_ENCOUNTER_POOLS[act_id]
        plan = generate_act_encounters(
            encounters,
            rng,
            act_id=act_id,
            room_count=pools.room_count - int(multiplayer),
            number_of_runs=profile.get("number_of_runs") if not multiplayer else None,
            seen_encounters=(
                profile.get("seen_encounters", () if profile.get("number_of_runs") == 0 else None)
                if not multiplayer
                else None
            ),
        )
        ancient_ids = [str(value).upper() for value in row.get("ancients", ())]
        if epochs is not None:
            ancient_ids = [
                value
                for value in ancient_ids
                if value not in {"NEOW", "OROBAS"} or value + "_EPOCH" in epochs
            ]
        ancient_ids += allocations.get(str(number), [])
        ancient_id = rng.choice(ancient_ids) if ancient_ids else None
        if (
            number == 1
            and act_id == "OVERGROWTH"
            and not multiplayer
            and profile.get("number_of_runs") == 0
        ):
            for index, event_id in enumerate(("BYRDONIS_NEST", "SAPPHIRE_SEED")):
                if event_id in event_ids:
                    previous = event_ids.index(event_id)
                    event_ids[index], event_ids[previous] = event_ids[previous], event_ids[index]
        second_boss = None
        if number == 3 and ascension >= 10:
            remaining = [value for value in pools.boss if value != plan.boss_id]
            second_boss = rng.choice(remaining) if remaining else None
        result[str(number)] = {
            **plan.as_metadata(),
            "act_id": act_id,
            "events": event_ids,
            "ancient_id": ancient_id,
            "second_boss_encounter": second_boss,
            "normal_encounters_visited": 0,
            "elite_encounters_visited": 0,
            "boss_encounters_visited": 0,
            "events_visited": 0,
        }
    return result
