"""Stable v0.107.1 act encounter bags, ordering, and discovery overrides.

Rules: ActModel.GenerateRooms/AddWithoutRepeatingTags and Helpers/GrabBag.cs
in the public v0.107.1 decompilation (Steam build 23811903, game commit
59260271), pinned at zhiyue/sts2-rl-agent@5d874e855ae1d0550d5c30416fadd7dbb0d133b7.
The independent tckmn/sts2-seed-search@cedcb540eca0c5393b883cd847d975df5e97bf64
cleaned-v0.107.1.c and combats.h agree on the bag algorithm and combat tags.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .monsters import EncounterDefinition


class EncounterRng(Protocol):
    def random(self) -> float: ...


@dataclass(frozen=True, slots=True)
class ActEncounterPools:
    weak: tuple[str, ...]
    regular: tuple[str, ...]
    elite: tuple[str, ...]
    boss: tuple[str, ...]
    room_count: int
    weak_count: int
    boss_discovery_order: tuple[str, ...]

    @property
    def all_ids(self) -> frozenset[str]:
        return frozenset(self.weak + self.regular + self.elite + self.boss)


# Order is GenerateAllEncounters filtered by kind, not alphabetic API order or
# BossDiscoveryOrder. In particular Hive's Infested Prisms follows Entomancer.
ACT_ENCOUNTER_POOLS = {
    "OVERGROWTH": ActEncounterPools(
        weak=("FUZZY_WURM_CRAWLER_WEAK", "NIBBITS_WEAK", "SHRINKER_BEETLE_WEAK", "SLIMES_WEAK"),
        regular=(
            "CUBEX_CONSTRUCT_NORMAL",
            "FLYCONID_NORMAL",
            "FOGMOG_NORMAL",
            "INKLETS_NORMAL",
            "MAWLER_NORMAL",
            "NIBBITS_NORMAL",
            "OVERGROWTH_CRAWLERS",
            "RUBY_RAIDERS_NORMAL",
            "SLIMES_NORMAL",
            "SLITHERING_STRANGLER_NORMAL",
            "SNAPPING_JAXFRUIT_NORMAL",
            "VINE_SHAMBLER_NORMAL",
        ),
        elite=("BYGONE_EFFIGY_ELITE", "BYRDONIS_ELITE", "PHROG_PARASITE_ELITE"),
        boss=("CEREMONIAL_BEAST_BOSS", "THE_KIN_BOSS", "VANTOM_BOSS"),
        room_count=15,
        weak_count=3,
        boss_discovery_order=("VANTOM_BOSS", "CEREMONIAL_BEAST_BOSS", "THE_KIN_BOSS"),
    ),
    "UNDERDOCKS": ActEncounterPools(
        weak=("CORPSE_SLUGS_WEAK", "SEAPUNK_WEAK", "SLUDGE_SPINNER_WEAK", "TOADPOLES_WEAK"),
        regular=(
            "CORPSE_SLUGS_NORMAL",
            "CULTISTS_NORMAL",
            "FOSSIL_STALKER_NORMAL",
            "GREMLIN_MERC_NORMAL",
            "HAUNTED_SHIP_NORMAL",
            "LIVING_FOG_NORMAL",
            "PUNCH_CONSTRUCT_NORMAL",
            "SEAPUNK_NORMAL",
            "SEWER_CLAM_NORMAL",
            "TWO_TAILED_RATS_NORMAL",
        ),
        elite=("PHANTASMAL_GARDENERS_ELITE", "SKULKING_COLONY_ELITE", "TERROR_EEL_ELITE"),
        boss=("LAGAVULIN_MATRIARCH_BOSS", "SOUL_FYSH_BOSS", "WATERFALL_GIANT_BOSS"),
        room_count=15,
        weak_count=3,
        boss_discovery_order=("WATERFALL_GIANT_BOSS", "SOUL_FYSH_BOSS", "LAGAVULIN_MATRIARCH_BOSS"),
    ),
    "HIVE": ActEncounterPools(
        weak=("BOWLBUGS_WEAK", "EXOSKELETONS_WEAK", "THIEVING_HOPPER_WEAK", "TUNNELER_WEAK"),
        regular=(
            "BOWLBUGS_NORMAL",
            "CHOMPERS_NORMAL",
            "EXOSKELETONS_NORMAL",
            "HUNTER_KILLER_NORMAL",
            "LOUSE_PROGENITOR_NORMAL",
            "MYTES_NORMAL",
            "OVICOPTER_NORMAL",
            "SLUMBERING_BEETLE_NORMAL",
            "SPINY_TOAD_NORMAL",
            "THE_OBSCURA_NORMAL",
        ),
        elite=("DECIMILLIPEDE_ELITE", "ENTOMANCER_ELITE", "INFESTED_PRISMS_ELITE"),
        boss=("KAISER_CRAB_BOSS", "KNOWLEDGE_DEMON_BOSS", "THE_INSATIABLE_BOSS"),
        room_count=14,
        weak_count=2,
        boss_discovery_order=("THE_INSATIABLE_BOSS", "KNOWLEDGE_DEMON_BOSS", "KAISER_CRAB_BOSS"),
    ),
    "GLORY": ActEncounterPools(
        weak=("DEVOTED_SCULPTOR_WEAK", "SCROLLS_OF_BITING_WEAK", "TURRET_OPERATOR_WEAK"),
        regular=(
            "AXEBOTS_NORMAL",
            "CONSTRUCT_MENAGERIE_NORMAL",
            "FABRICATOR_NORMAL",
            "FROG_KNIGHT_NORMAL",
            "GLOBE_HEAD_NORMAL",
            "OWL_MAGISTRATE_NORMAL",
            "SCROLLS_OF_BITING_NORMAL",
            "SLIMED_BERSERKER_NORMAL",
            "THE_LOST_AND_FORGOTTEN_NORMAL",
        ),
        elite=("KNIGHTS_ELITE", "MECHA_KNIGHT_ELITE", "SOUL_NEXUS_ELITE"),
        boss=("AEONGLASS_BOSS", "QUEEN_BOSS", "TEST_SUBJECT_BOSS"),
        room_count=13,
        weak_count=2,
        boss_discovery_order=("QUEEN_BOSS", "TEST_SUBJECT_BOSS", "AEONGLASS_BOSS"),
    ),
}

ENCOUNTER_TAGS: dict[str, frozenset[str]] = {
    "FUZZY_WURM_CRAWLER_WEAK": frozenset({"crawler"}),
    "NIBBITS_WEAK": frozenset({"nibbit"}),
    "SHRINKER_BEETLE_WEAK": frozenset({"shrinker"}),
    "SLIMES_WEAK": frozenset({"slimes"}),
    "FLYCONID_NORMAL": frozenset({"mushroom", "slimes"}),
    "OVERGROWTH_CRAWLERS": frozenset({"shrinker", "crawler"}),
    "SLIMES_NORMAL": frozenset({"slimes"}),
    "SNAPPING_JAXFRUIT_NORMAL": frozenset({"mushroom"}),
    "CORPSE_SLUGS_WEAK": frozenset({"slugs"}),
    "SEAPUNK_WEAK": frozenset({"seapunk"}),
    "CORPSE_SLUGS_NORMAL": frozenset({"slugs"}),
    "SEAPUNK_NORMAL": frozenset({"seapunk"}),
    "BOWLBUGS_WEAK": frozenset({"workers"}),
    "EXOSKELETONS_WEAK": frozenset({"exoskeletons"}),
    "THIEVING_HOPPER_WEAK": frozenset({"thieves"}),
    "TUNNELER_WEAK": frozenset({"burrower"}),
    "BOWLBUGS_NORMAL": frozenset({"workers"}),
    "CHOMPERS_NORMAL": frozenset({"chomper"}),
    "EXOSKELETONS_NORMAL": frozenset({"exoskeletons"}),
    "SLUMBERING_BEETLE_NORMAL": frozenset({"workers"}),
    "SCROLLS_OF_BITING_WEAK": frozenset({"scrolls"}),
    "SCROLLS_OF_BITING_NORMAL": frozenset({"scrolls"}),
    "KNIGHTS_ELITE": frozenset({"knights"}),
}


@dataclass(frozen=True, slots=True)
class EncounterBagState:
    remaining_ids: tuple[str, ...] = ()
    previous_id: str | None = None


@dataclass(frozen=True, slots=True)
class ActEncounterPlan:
    normal_ids: tuple[str, ...]
    elite_ids: tuple[str, ...]
    boss_id: str | None

    def as_metadata(self) -> dict[str, Any]:
        return {
            "normal_encounters": list(self.normal_ids),
            "elite_encounters": list(self.elite_ids),
            "boss_encounter": self.boss_id,
        }


def weak_count_for_act(act_id: str) -> int:
    pools = ACT_ENCOUNTER_POOLS.get(act_id.upper())
    return pools.weak_count if pools is not None else 0


def encounters_for_act(
    encounters: Sequence[EncounterDefinition],
    act_id: str,
) -> tuple[EncounterDefinition, ...]:
    """Keep the selected named act, including explicitly bound custom rows."""
    normalized_act = act_id.upper()
    pools = ACT_ENCOUNTER_POOLS.get(normalized_act)
    known_ids = pools.all_ids if pools is not None else frozenset()
    return tuple(
        encounter
        for encounter in encounters
        if encounter.encounter_id.upper() in known_ids or encounter.act_id == normalized_act
    )


def shares_encounter_tags(current_id: str, previous_id: str | None) -> bool:
    if previous_id is None:
        return False
    return bool(
        ENCOUNTER_TAGS.get(current_id.upper(), frozenset())
        & ENCOUNTER_TAGS.get(previous_id.upper(), frozenset())
    )


def draw_from_encounter_bag(
    pool: Sequence[EncounterDefinition],
    rng: EncounterRng,
    state: EncounterBagState | None = None,
) -> tuple[EncounterDefinition | None, EncounterBagState]:
    """Draw without replacement, preserving source rejection RNG consumption.

    Refill only an empty bag. Prefer a different encounter with disjoint tags;
    if no remaining entry satisfies that condition, use an unrestricted draw.
    Each rejected sample consumes RNG, as does a one-element bag.
    """
    state = state or EncounterBagState()
    by_id = {encounter.encounter_id: encounter for encounter in pool}
    remaining = list(state.remaining_ids or tuple(by_id))
    remaining = [encounter_id for encounter_id in remaining if encounter_id in by_id]
    if not remaining:
        return None, EncounterBagState(previous_id=state.previous_id)

    def allowed(encounter_id: str) -> bool:
        return encounter_id != state.previous_id and not shares_encounter_tags(
            encounter_id, state.previous_id
        )

    enforce_condition = any(allowed(encounter_id) for encounter_id in remaining)
    while True:
        index = int(rng.random() * len(remaining))
        encounter_id = remaining[index]
        if not enforce_condition or allowed(encounter_id):
            break
    remaining.pop(index)
    return by_id[encounter_id], EncounterBagState(tuple(remaining), encounter_id)


def draw_encounter_sequence(
    pool: Sequence[EncounterDefinition],
    rng: EncounterRng,
    count: int,
    *,
    previous: EncounterDefinition | None = None,
) -> tuple[EncounterDefinition, ...]:
    state = EncounterBagState(previous_id=previous.encounter_id if previous else None)
    chosen: list[EncounterDefinition] = []
    for _ in range(max(0, count)):
        encounter, state = draw_from_encounter_bag(pool, rng, state)
        if encounter is None:
            break
        chosen.append(encounter)
    return tuple(chosen)


def generate_act_encounters(
    encounters: Sequence[EncounterDefinition],
    rng: EncounterRng,
    *,
    act_id: str,
    room_count: int | None = None,
    number_of_runs: int | None = None,
    seen_encounters: Sequence[str] | None = None,
) -> ActEncounterPlan:
    """Generate one act's room set in stable weak/regular/elite/boss order.

    Event shuffling precedes this call and Ancient selection follows it in the
    game. Callers own those draws. Profile overrides apply only when supplied.
    Missing/custom source rows never broaden the pool to unrelated encounters.
    """
    normalized_act = act_id.upper()
    pools = ACT_ENCOUNTER_POOLS.get(normalized_act)
    if pools is None:
        return ActEncounterPlan((), (), None)
    by_id = {encounter.encounter_id.upper(): encounter for encounter in encounters}

    def resolve(ids: tuple[str, ...]) -> tuple[EncounterDefinition, ...]:
        return tuple(by_id[encounter_id] for encounter_id in ids if encounter_id in by_id)

    weak = draw_encounter_sequence(resolve(pools.weak), rng, pools.weak_count)
    regular = draw_encounter_sequence(
        resolve(pools.regular),
        rng,
        max(0, (room_count if room_count is not None else pools.room_count) - pools.weak_count),
        previous=weak[-1] if weak else None,
    )
    elites = draw_encounter_sequence(resolve(pools.elite), rng, 15)
    bosses = resolve(pools.boss)
    boss_id = bosses[int(rng.random() * len(bosses))].encounter_id if bosses else None
    normal_ids = [encounter.encounter_id for encounter in weak + regular]
    elite_ids = [encounter.encounter_id for encounter in elites]
    if seen_encounters is not None:
        seen = {encounter_id.upper() for encounter_id in seen_encounters}
        boss_id = next(
            (
                encounter_id
                for encounter_id in pools.boss_discovery_order
                if encounter_id not in seen and encounter_id in by_id
            ),
            boss_id,
        )
    if normalized_act == "OVERGROWTH" and number_of_runs == 0:
        for index, encounter_id in enumerate(
            (
                "NIBBITS_WEAK",
                "SLIMES_WEAK",
                "SHRINKER_BEETLE_WEAK",
                "INKLETS_NORMAL",
                "MAWLER_NORMAL",
                "RUBY_RAIDERS_NORMAL",
                "NIBBITS_NORMAL",
            )
        ):
            if encounter_id in by_id:
                _swap_to_or_create(normal_ids, encounter_id, index)
        for index, encounter_id in enumerate(("BYRDONIS_ELITE", "PHROG_PARASITE_ELITE")):
            if encounter_id in by_id:
                _swap_to_or_create(elite_ids, encounter_id, index)
    return ActEncounterPlan(tuple(normal_ids), tuple(elite_ids), boss_id)


def _swap_to_or_create(ids: list[str], encounter_id: str, index: int) -> None:
    if index >= len(ids):
        ids.append(encounter_id)
    elif encounter_id in ids:
        previous_index = ids.index(encounter_id)
        ids[index], ids[previous_index] = ids[previous_index], ids[index]
    else:
        ids[index] = encounter_id
