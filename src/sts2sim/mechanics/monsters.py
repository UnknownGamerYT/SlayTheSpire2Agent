"""Source-data driven monster encounters and move selection."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from random import Random
from typing import Any

from .monster_scripts import MONSTER_INITIAL_STATES, MONSTER_STATE_FOLLOWUPS

# AscensionLevel.cs in stable v0.107.1 defines ToughEnemies=8 and
# DeadlyEnemies=9 for every enemy kind (there are no STS1 room thresholds).
NORMAL_DAMAGE_ASCENSION = 9
ELITE_DAMAGE_ASCENSION = 9
BOSS_DAMAGE_ASCENSION = 9
NORMAL_HP_ASCENSION = 8
ELITE_HP_ASCENSION = 8
BOSS_HP_ASCENSION = 8


@dataclass(frozen=True, slots=True)
class MonsterPower:
    power_id: str
    amount: int
    target: str = "self"
    amount_ascension: int | None = None

    def as_metadata(self) -> dict[str, object]:
        return {
            "power_id": self.power_id,
            "amount": self.amount,
            "target": self.target,
            "amount_ascension": self.amount_ascension,
        }


@dataclass(frozen=True, slots=True)
class MonsterMove:
    move_id: str
    name: str
    intent: str
    damage_normal: int = 0
    damage_ascension: int | None = None
    hit_count: int = 1
    block: int = 0
    heal: int = 0
    powers: tuple[MonsterPower, ...] = ()


@dataclass(frozen=True, slots=True)
class MonsterSummonPlan:
    source_monster_id: str
    move_id: str
    summon_monster_ids: tuple[str, ...]
    count: int = 1
    count_policy: str = "fixed"
    max_allies: int | None = None


@dataclass(frozen=True, slots=True)
class AttackPatternBranch:
    move_id: str | None
    weight: float
    repeat: str | None = None
    condition: str | None = None
    max_times: int | None = None


@dataclass(frozen=True, slots=True)
class AttackPatternState:
    state_id: str
    move_id: str | None
    next_selector: str | None
    state_type: str
    branches: tuple[AttackPatternBranch, ...] = ()


@dataclass(frozen=True, slots=True)
class MonsterDefinition:
    monster_id: str
    name: str
    kind: str
    min_hp: int
    max_hp: int
    min_hp_ascension: int | None
    max_hp_ascension: int | None
    moves: tuple[MonsterMove, ...]
    states: tuple[AttackPatternState, ...]
    initial_selector: str | None = None
    innate_powers: tuple[MonsterPower, ...] = ()

    @property
    def move_by_id(self) -> Mapping[str, MonsterMove]:
        return {move.move_id: move for move in self.moves}

    @property
    def state_by_id(self) -> Mapping[str, AttackPatternState]:
        return {state.state_id: state for state in self.states}


@dataclass(frozen=True, slots=True)
class EncounterDefinition:
    encounter_id: str
    name: str
    room_type: str
    monster_ids: tuple[str, ...]
    act_number: int | None = None
    is_weak: bool = False
    act_id: str | None = None


@dataclass(frozen=True, slots=True)
class SpawnedMonster:
    instance_id: str
    source_monster_id: str
    name: str
    hp: int
    max_hp: int
    move: MonsterMove | None
    next_move_id: str | None
    encounter_id: str
    slot_index: int
    innate_powers: tuple[MonsterPower, ...] = ()


_SUMMON_PLANS: tuple[MonsterSummonPlan, ...] = (
    MonsterSummonPlan(
        source_monster_id="FABRICATOR",
        move_id="FABRICATE",
        summon_monster_ids=("GUARDBOT", "NOISEBOT", "ZAPBOT", "STABBOT"),
        count=2,
        max_allies=3,
    ),
    MonsterSummonPlan(
        source_monster_id="FABRICATOR",
        move_id="FABRICATING_STRIKE",
        summon_monster_ids=("ZAPBOT", "STABBOT"),
        max_allies=3,
    ),
    MonsterSummonPlan(
        source_monster_id="FOGMOG",
        move_id="ILLUSION",
        summon_monster_ids=("EYE_WITH_TEETH",),
        max_allies=1,
    ),
    MonsterSummonPlan(
        source_monster_id="LIVING_FOG",
        move_id="BLOAT",
        summon_monster_ids=("GAS_BOMB",),
        count_policy="move_count_plus_one",
        max_allies=6,
    ),
    MonsterSummonPlan(
        source_monster_id="OVICOPTER",
        move_id="LAY_EGGS",
        summon_monster_ids=("TOUGH_EGG",),
        count=3,
        count_policy="fill_to_max_allies",
        max_allies=3,
    ),
    MonsterSummonPlan(
        source_monster_id="THE_OBSCURA",
        move_id="ILLUSION",
        summon_monster_ids=("PARAFRIGHT",),
        max_allies=1,
    ),
    MonsterSummonPlan(
        source_monster_id="TWO_TAILED_RAT",
        move_id="CALL_FOR_BACKUP",
        summon_monster_ids=("TWO_TAILED_RAT",),
        max_allies=5,
    ),
)


def monster_summon_plan(source_monster_id: str, move_id: str) -> MonsterSummonPlan | None:
    normalized_source = _normalized_id(source_monster_id)
    normalized_move = _normalized_id(move_id)
    for plan in _SUMMON_PLANS:
        if (
            _normalized_id(plan.source_monster_id) == normalized_source
            and _normalized_id(plan.move_id) == normalized_move
        ):
            return plan
    return None


def monster_summon_plans_for(source_monster_id: str) -> tuple[MonsterSummonPlan, ...]:
    normalized_source = _normalized_id(source_monster_id)
    return tuple(
        plan
        for plan in _SUMMON_PLANS
        if _normalized_id(plan.source_monster_id) == normalized_source
    )


def build_monster_definitions(raw_monsters: Sequence[Any]) -> Mapping[str, MonsterDefinition]:
    definitions: dict[str, MonsterDefinition] = {}
    for raw_monster in raw_monsters:
        row = _mapping(raw_monster)
        monster_id = _text(row.get("id"))
        if not monster_id:
            continue
        moves = tuple(_monster_move(move) for move in _sequence(row.get("moves")))
        moves = tuple(move for move in moves if move.move_id)
        moves = _source_specific_moves(monster_id, moves)
        innate_powers = tuple(
            _monster_power(power) for power in _sequence(row.get("innate_powers"))
        )
        innate_powers = _source_specific_innate_powers(monster_id, innate_powers)
        min_hp = _int(row.get("min_hp"), 1)
        max_hp = _int(row.get("max_hp"), min_hp)
        min_hp_ascension = _optional_int(row.get("min_hp_ascension"))
        max_hp_ascension = _optional_int(row.get("max_hp_ascension"))
        if row.get("max_hp") is None and max_hp_ascension is None:
            max_hp_ascension = min_hp_ascension
        pattern = _mapping(row.get("attack_pattern"))
        states = tuple(
            _attack_pattern_state(state) for state in _sequence(pattern.get("states"))
        )
        state_ids = {state.state_id for state in states}
        states = tuple(
            replace(state, next_selector=followup)
            if (followup := MONSTER_STATE_FOLLOWUPS.get((monster_id, state.state_id)))
            in state_ids else state
            for state in states
        )
        initial = MONSTER_INITIAL_STATES.get(monster_id)
        if initial not in state_ids:
            initial = _optional_text(pattern.get("initial_move"))
        definition = MonsterDefinition(
            monster_id=monster_id,
            name=_text(row.get("name"), monster_id),
            kind=_text(row.get("type"), "Normal"),
            min_hp=min_hp,
            max_hp=max_hp,
            min_hp_ascension=min_hp_ascension,
            max_hp_ascension=max_hp_ascension,
            moves=moves,
            states=states,
            initial_selector=(
                "HAMMER_UPPERCUT" if _normalized_id(monster_id) == "axebot"
                and any(move.move_id == "HAMMER_UPPERCUT" for move in moves)
                else initial
            ),
            innate_powers=innate_powers,
        )
        definitions[monster_id] = definition
    return definitions


def build_encounter_definitions(raw_encounters: Sequence[Any]) -> tuple[EncounterDefinition, ...]:
    encounters: list[EncounterDefinition] = []
    for raw_encounter in raw_encounters:
        row = _mapping(raw_encounter)
        encounter_id = _text(row.get("id"))
        if not encounter_id:
            continue
        monster_ids = tuple(
            monster_id
            for monster_id in (
                _text(_mapping(monster).get("id")) for monster in _sequence(row.get("monsters"))
            )
            if monster_id
        )
        # Source participant lists include creatures summoned or created on death.
        opening_rosters = {
            "fogmog_normal": ("FOGMOG",),
            "the_obscura_normal": ("THE_OBSCURA",),
            "gremlin_merc_normal": ("GREMLIN_MERC",),
            "phrog_parasite_elite": ("PHROG_PARASITE",),
        }
        expected = opening_rosters.get(_normalized_id(encounter_id))
        if expected is not None:
            allowed = {_normalized_id(item) for item in expected}
            opening = tuple(item for item in monster_ids if _normalized_id(item) in allowed)
            monster_ids = opening or monster_ids
        if not monster_ids:
            continue
        encounters.append(
            EncounterDefinition(
                encounter_id=encounter_id,
                name=_text(row.get("name"), encounter_id),
                room_type=_normalized_room_type(_text(row.get("room_type"), "Monster")),
                monster_ids=monster_ids,
                act_number=_act_number(row.get("act")),
                is_weak=bool(row.get("is_weak", False)),
                act_id=_named_act_id(row.get("act_id", row.get("act"))),
            )
        )
    return tuple(encounters)


def synthetic_encounter(
    *,
    encounter_id: str,
    room_type: str,
    monster_ids: Sequence[str],
    act_number: int | None = None,
) -> EncounterDefinition:
    return EncounterDefinition(
        encounter_id=encounter_id,
        name=encounter_id,
        room_type=_normalized_room_type(room_type),
        monster_ids=tuple(str(monster_id) for monster_id in monster_ids),
        act_number=act_number,
    )


def choose_encounter(
    encounters: Sequence[EncounterDefinition],
    rng: Random,
    *,
    act: int,
    room_type: str,
    preferred_id: str | None = None,
    prefer_weak: bool = False,
    act_id: str | None = None,
    previous_id: str | None = None,
) -> EncounterDefinition | None:
    room_type = _normalized_room_type(room_type)
    if preferred_id:
        preferred = _find_encounter(encounters, preferred_id)
        if preferred is not None:
            return preferred

    from .run_encounters import encounters_for_act, shares_encounter_tags

    eligible = encounters_for_act(encounters, act_id) if act_id is not None else encounters
    candidates = tuple(encounter for encounter in eligible if encounter.room_type == room_type)
    if not candidates:
        return None

    act_candidates = tuple(
        encounter for encounter in candidates if encounter.act_number in (None, act)
    )
    candidates = act_candidates

    weak_candidates = tuple(encounter for encounter in candidates if encounter.is_weak)
    non_weak_candidates = tuple(encounter for encounter in candidates if not encounter.is_weak)
    if prefer_weak and weak_candidates:
        candidates = weak_candidates
    elif not prefer_weak and non_weak_candidates:
        candidates = non_weak_candidates

    if not candidates:
        return None
    different_candidates = tuple(
        encounter
        for encounter in candidates
        if encounter.encounter_id != previous_id
        and not shares_encounter_tags(encounter.encounter_id, previous_id)
    )
    if different_candidates:
        candidates = different_candidates
    return rng.choice(candidates)


def spawn_monsters(
    encounter: EncounterDefinition,
    monster_definitions: Mapping[str, MonsterDefinition],
    rng: Random,
    *,
    ascension_level: int,
) -> tuple[SpawnedMonster, ...]:
    counts = Counter(encounter.monster_ids)
    seen: Counter[str] = Counter()
    spawned: list[SpawnedMonster] = []
    for slot_index, source_monster_id in enumerate(encounter.monster_ids):
        definition = monster_definitions.get(source_monster_id)
        if definition is None:
            continue
        seen[source_monster_id] += 1
        hp = roll_monster_hp(definition, rng, ascension_level=ascension_level)
        current_move = initial_monster_move(
            definition,
            rng,
            slot_index=slot_index,
            ally_count=max(0, len(encounter.monster_ids) - 1),
            is_front=slot_index == 0,
            current_hp=hp,
            max_hp=hp,
        )
        instance_id = source_monster_id
        if counts[source_monster_id] > 1:
            instance_id = f"{source_monster_id}#{seen[source_monster_id]}"
        spawned.append(
            SpawnedMonster(
                instance_id=instance_id,
                source_monster_id=source_monster_id,
                name=definition.name,
                hp=hp,
                max_hp=hp,
                move=current_move,
                next_move_id=None,
                encounter_id=encounter.encounter_id,
                slot_index=slot_index,
                innate_powers=definition.innate_powers,
            )
        )
    return tuple(spawned)


def roll_monster_hp(
    definition: MonsterDefinition,
    rng: Random,
    *,
    ascension_level: int,
) -> int:
    minimum, maximum = monster_hp_range(definition, ascension_level=ascension_level)
    if maximum < minimum:
        maximum = minimum
    return rng.randint(minimum, maximum)


def monster_hp_range(
    definition: MonsterDefinition,
    *,
    ascension_level: int,
) -> tuple[int, int]:
    if (
        _normalized_id(definition.monster_id) == "test_subject"
        and definition.min_hp <= 1
        and definition.max_hp <= 1
    ):
        hp = 111 if ascension_level >= 8 else 100
        return hp, hp

    minimum = definition.min_hp
    maximum = definition.max_hp
    if _uses_ascension_hp(definition, ascension_level):
        minimum = (
            definition.min_hp_ascension
            if definition.min_hp_ascension is not None
            else minimum
        )
        maximum = (
            definition.max_hp_ascension
            if definition.max_hp_ascension is not None
            else maximum
        )
    return max(1, minimum), max(1, maximum)


def monster_move_damage(
    definition: MonsterDefinition,
    move: MonsterMove,
    *,
    ascension_level: int,
) -> int:
    if _uses_ascension_damage(definition, ascension_level) and move.damage_ascension is not None:
        return max(0, move.damage_ascension)
    return max(0, move.damage_normal)


def monster_power_amount(
    definition: MonsterDefinition,
    power: MonsterPower,
    *,
    ascension_level: int,
) -> int:
    threshold = 8 if _normalized_id(power.power_id) in {
        "plating", "curl_up", "skittish", "shriek", "slippery",
    } else 9
    if ascension_level >= threshold and power.amount_ascension is not None:
        return max(0, power.amount_ascension)
    return max(0, power.amount)


def monster_move_for_ascension(
    definition: MonsterDefinition, move: MonsterMove, *, ascension_level: int,
) -> MonsterMove:
    """Resolve omitted stable Block and repeat-count ascension properties."""
    key = (_normalized_id(definition.monster_id), _normalized_id(move.move_id))
    block_rules = {
        ("axe_ruby_raider", "swing_1"): (9, 6),
        ("axe_ruby_raider", "swing_2"): (9, 6),
        ("axebot", "boot_up"): (9, 15),
        ("bowlbug_egg", "bite"): (9, 8),
        ("infested_prism", "radiate"): (9, 13),
        ("infested_prism", "pulsate"): (8, 22),
        ("lagavulin_matriarch", "slash2"): (8, 14),
        ("magi_knight", "power_shield"): (8, 9),
        ("nibbit", "slice"): (8, 6),
        ("seapunk", "bubble_burp"): (8, 8),
        ("the_obscura", "hardening_strike"): (9, 7),
        ("tunneler", "burrow"): (8, 37),
    }
    if key in block_rules:
        threshold, amount = block_rules[key]
        if ascension_level >= threshold:
            move = replace(move, block=amount)
    hit_rules = {
        ("entomancer", "bees"): 8,
        ("exoskeleton", "skitter"): 4,
        ("tracker_ruby_raider", "hounds"): 9,
    }
    if key in hit_rules and ascension_level >= 9:
        move = replace(move, hit_count=hit_rules[key])
    return move


def waterfall_giant_siphon_heal(*, ascension_level: int) -> int:
    """Return the v0.107.x Waterfall Giant Siphon heal amount.

    The source row currently omits this heal payload, so the mechanics layer
    exposes it explicitly for runtime integrations.
    """

    return 15 if ascension_level >= 8 else 10


def initial_monster_move(
    definition: MonsterDefinition,
    rng: Random,
    *,
    slot_index: int = 0,
    ally_count: int | None = None,
    is_front: bool | None = None,
    can_spawn: bool | None = None,
    current_hp: int | None = None,
    max_hp: int | None = None,
    move_counts: Mapping[str, int] | None = None,
) -> MonsterMove | None:
    selector = definition.initial_selector
    if selector is None and definition.states:
        selector = definition.states[0].state_id
    if selector is None and definition.moves:
        selector = definition.moves[0].move_id
    return resolve_monster_move(
        definition,
        selector,
        rng,
        slot_index=slot_index,
        ally_count=ally_count,
        is_front=is_front,
        can_spawn=can_spawn,
        current_hp=current_hp,
        max_hp=max_hp,
        previous_move_id=None,
        move_counts=move_counts or {},
    )


def next_monster_move(
    definition: MonsterDefinition,
    current_move_id: str | None,
    rng: Random,
    *,
    slot_index: int = 0,
    ally_count: int | None = None,
    is_front: bool | None = None,
    can_spawn: bool | None = None,
    current_hp: int | None = None,
    max_hp: int | None = None,
    move_counts: Mapping[str, int] | None = None,
) -> MonsterMove | None:
    if current_move_id is None:
        return initial_monster_move(
            definition,
            rng,
            slot_index=slot_index,
            ally_count=ally_count,
            is_front=is_front,
            can_spawn=can_spawn,
            current_hp=current_hp,
            max_hp=max_hp,
            move_counts=move_counts,
        )

    current_state = _state_for_move(definition, current_move_id)
    selector = _source_specific_next_selector(definition, current_move_id)
    if selector is None:
        selector = current_state.next_selector if current_state is not None else None
    if selector is None:
        selector = _next_cycle_move_id(definition, current_move_id)
    return resolve_monster_move(
        definition,
        selector,
        rng,
        slot_index=slot_index,
        ally_count=ally_count,
        is_front=is_front,
        can_spawn=can_spawn,
        current_hp=current_hp,
        max_hp=max_hp,
        previous_move_id=current_move_id,
        move_counts=move_counts or {},
    )


def move_by_id(definition: MonsterDefinition, move_id: str | None) -> MonsterMove | None:
    if move_id is None:
        return None
    return definition.move_by_id.get(move_id)


def resolve_monster_move(
    definition: MonsterDefinition,
    selector: str | None,
    rng: Random,
    *,
    slot_index: int = 0,
    ally_count: int | None = None,
    is_front: bool | None = None,
    can_spawn: bool | None = None,
    current_hp: int | None = None,
    max_hp: int | None = None,
    previous_move_id: str | None = None,
    move_counts: Mapping[str, int] | None = None,
    depth: int = 0,
) -> MonsterMove | None:
    if not definition.moves:
        return None
    if depth > 8:
        return _fallback_move(definition, rng, previous_move_id=previous_move_id)

    if selector is None:
        return _fallback_move(definition, rng, previous_move_id=previous_move_id)

    direct_move = definition.move_by_id.get(selector)
    if direct_move is not None:
        return direct_move

    state = definition.state_by_id.get(selector)
    if state is None:
        return _fallback_move(definition, rng, previous_move_id=previous_move_id)

    if state.move_id is not None:
        return resolve_monster_move(
            definition,
            state.move_id,
            rng,
            slot_index=slot_index,
            ally_count=ally_count,
            is_front=is_front,
            can_spawn=can_spawn,
            current_hp=current_hp,
            max_hp=max_hp,
            previous_move_id=previous_move_id,
            move_counts=move_counts,
            depth=depth + 1,
        )

    branch = _choose_branch(
        state.branches,
        rng,
        state_type=state.state_type,
        slot_index=slot_index,
        ally_count=ally_count,
        is_front=is_front,
        can_spawn=can_spawn,
        current_hp=current_hp,
        max_hp=max_hp,
        previous_move_id=previous_move_id,
        move_counts=move_counts or {},
    )
    if branch is not None and branch.move_id is not None:
        return resolve_monster_move(
            definition,
            branch.move_id,
            rng,
            slot_index=slot_index,
            ally_count=ally_count,
            is_front=is_front,
            can_spawn=can_spawn,
            current_hp=current_hp,
            max_hp=max_hp,
            previous_move_id=previous_move_id,
            move_counts=move_counts,
            depth=depth + 1,
        )
    return _fallback_move(definition, rng, previous_move_id=previous_move_id)


def next_move_counts(
    move_counts: Mapping[str, int],
    move_id: str | None,
) -> dict[str, int]:
    counts = {str(key): int(value) for key, value in move_counts.items()}
    if move_id:
        counts[move_id] = counts.get(move_id, 0) + 1
    return counts


def _monster_move(raw_move: Any) -> MonsterMove:
    row = _mapping(raw_move)
    damage = _mapping(row.get("damage"))
    return MonsterMove(
        move_id=_text(row.get("id")),
        name=_text(row.get("name"), _text(row.get("id"))),
        intent=_text(row.get("intent"), "Unknown"),
        damage_normal=_int(damage.get("normal"), 0),
        damage_ascension=_optional_int(damage.get("ascension")),
        hit_count=max(1, _int(damage.get("hit_count"), 1)),
        block=max(0, _int(row.get("block"), 0)),
        heal=max(0, _int(row.get("heal"), 0)),
        powers=tuple(_monster_power(power) for power in _sequence(row.get("powers"))),
    )


def _monster_power(raw_power: Any) -> MonsterPower:
    row = _mapping(raw_power)
    return MonsterPower(
        power_id=_text(row.get("power_id")),
        amount=_int(row.get("amount"), 0),
        target=_text(row.get("target"), "self"),
        amount_ascension=_optional_int(row.get("amount_ascension")),
    )


def _source_specific_moves(
    monster_id: str,
    moves: tuple[MonsterMove, ...],
) -> tuple[MonsterMove, ...]:
    normalized_monster_id = _normalized_id(monster_id)
    if normalized_monster_id == "the_forgotten":
        return tuple(
            replace(move, damage_normal=13, damage_ascension=15)
            if _normalized_id(move.move_id) == "dread" and move.damage_normal == 0 else move
            for move in moves
        )
    patched: list[MonsterMove] = []
    scaled_powers = {
        ("skulking_colony", "inertia", "strength"): 4,
        ("myte", "suck", "strength"): 3,
        ("test_subject", "burning_growl", "strength"): 3,
        ("infested_prism", "pulsate", "vital_spark"): 3,
        ("crusher", "adapt", "strength"): 3,
        ("rocket", "charge_up", "strength"): 3,
        ("bowlbug_nectar", "buff", "strength"): 16,
        ("ceremonial_beast", "stamp", "plow"): 160,
        ("ceremonial_beast", "crush", "strength"): 4,
        ("damp_cultist", "incantation", "ritual"): 6,
        ("kin_follower", "power_dance", "strength"): 3,
        ("kin_priest", "ritual", "strength"): 3,
        ("knowledge_demon", "ponder", "strength"): 3,
        ("nibbit", "hiss", "strength"): 3,
        ("ovicopter", "nutritional_paste", "strength"): 4,
        ("phantasmal_gardener", "enlarge", "strength"): 3,
        ("seapunk", "bubble_burp", "strength"): 2,
        ("the_insatiable", "salivate", "strength"): 3,
        ("waterfall_giant", "pressurize", "steam_eruption"): 20,
    }
    for move in moves:
        if (
            normalized_monster_id == "entomancer"
            and _normalized_id(move.move_id) == "pheromone_spit"
        ):
            # The list API flattens both mutually exclusive SpitMove branches.
            patched.append(replace(move, powers=()))
            continue
        powers = tuple(
            replace(power, amount_ascension=scaled_powers[key])
            if (key := (
                normalized_monster_id, _normalized_id(move.move_id), _normalized_id(power.power_id)
            )) in scaled_powers and power.amount_ascension is None
            else power
            for power in move.powers
        )
        patched.append(replace(move, powers=powers))
    return tuple(patched)


def _source_specific_innate_powers(
    monster_id: str,
    powers: tuple[MonsterPower, ...],
) -> tuple[MonsterPower, ...]:
    normalized_monster_id = _normalized_id(monster_id)
    normalized_power_ids = {_normalized_id(power.power_id) for power in powers}
    patched = list(powers)
    if (
        normalized_monster_id == "aeonglass"
        and "withering_presence" not in normalized_power_ids
    ):
        patched.append(MonsterPower("WITHERING_PRESENCE", 6))
    if (
        normalized_monster_id == "skulking_colony"
        and "hardened_shell" not in normalized_power_ids
    ):
        patched.append(MonsterPower("HARDENED_SHELL", 20))
    if normalized_monster_id == "gremlin_merc" and "thievery" not in normalized_power_ids:
        # The list endpoint omits this printed power; the public monster page
        # records Thievery 20 (no ascension change).
        patched.append(MonsterPower("THIEVERY", 20))
    if normalized_monster_id == "axebot" and "stock" not in normalized_power_ids:
        patched.append(MonsterPower("STOCK", 2))
    return tuple(patched)


def _attack_pattern_state(raw_state: Any) -> AttackPatternState:
    row = _mapping(raw_state)
    return AttackPatternState(
        state_id=_text(row.get("id")),
        move_id=_optional_text(row.get("move_id")),
        next_selector=_optional_text(row.get("next")),
        state_type=_text(row.get("type"), "move"),
        branches=tuple(_attack_pattern_branch(branch) for branch in _sequence(row.get("branches"))),
    )


def _attack_pattern_branch(raw_branch: Any) -> AttackPatternBranch:
    row = _mapping(raw_branch)
    return AttackPatternBranch(
        move_id=_optional_text(row.get("move_id")),
        weight=max(0.0, _float(row.get("weight"), 1.0)),
        repeat=_optional_text(row.get("repeat")),
        condition=_optional_text(row.get("condition")),
        max_times=_optional_int(row.get("max_times")),
    )


def _choose_branch(
    branches: Sequence[AttackPatternBranch],
    rng: Random,
    *,
    state_type: str,
    slot_index: int,
    ally_count: int | None,
    is_front: bool | None,
    can_spawn: bool | None,
    current_hp: int | None,
    max_hp: int | None,
    previous_move_id: str | None,
    move_counts: Mapping[str, int],
) -> AttackPatternBranch | None:
    if not branches:
        return None

    conditional = tuple(
        branch
        for branch in branches
        if _condition_matches(
            branch.condition,
            slot_index=slot_index,
            ally_count=ally_count,
            is_front=is_front,
            can_spawn=can_spawn,
            current_hp=current_hp,
            max_hp=max_hp,
            move_counts=move_counts,
        )
        and _branch_available(branch, previous_move_id=previous_move_id, move_counts=move_counts)
    )
    if conditional and _normalized_id(state_type) == "conditional":
        return conditional[0]
    candidates = conditional or tuple(
        branch
        for branch in branches
        if _branch_available(branch, previous_move_id=previous_move_id, move_counts=move_counts)
    )
    if not candidates:
        candidates = tuple(branches)

    total_weight = sum(max(0.0, branch.weight) for branch in candidates)
    if total_weight <= 0:
        return candidates[0]

    needle = rng.random() * total_weight
    upto = 0.0
    for branch in candidates:
        upto += max(0.0, branch.weight)
        if needle <= upto:
            return branch
    return candidates[-1]


def _branch_available(
    branch: AttackPatternBranch,
    *,
    previous_move_id: str | None,
    move_counts: Mapping[str, int],
) -> bool:
    if (
        _normalized_id(branch.repeat or "") == "useonlyonce"
        and branch.move_id is not None
        and move_counts.get(branch.move_id, 0) >= 1
    ):
        return False
    if (
        branch.max_times is not None
        and branch.move_id is not None
        and move_counts.get(branch.move_id, 0) >= branch.max_times
    ):
        return False
    return not (branch.repeat == "CannotRepeat" and branch.move_id == previous_move_id)


def _condition_matches(
    condition: str | None,
    *,
    slot_index: int,
    ally_count: int | None,
    is_front: bool | None,
    can_spawn: bool | None,
    current_hp: int | None,
    max_hp: int | None,
    move_counts: Mapping[str, int],
) -> bool:
    if condition is None:
        return True
    condition = condition.strip()
    slot_names = ("first", "second", "third", "fourth", "fifth", "sixth")
    if "SlotName" in condition:
        slot_name = slot_names[slot_index] if slot_index < len(slot_names) else str(slot_index)
        named_slot = f"wriggler{slot_index + 1}"
        return (
            f'"{slot_name}"' in condition
            or f"'{slot_name}'" in condition
            or f'"{named_slot}"' in condition
            or f"'{named_slot}'" in condition
        )
    formation_match = _formation_condition_matches(
        condition,
        ally_count=ally_count,
        is_front=is_front,
    )
    if formation_match is not None:
        return formation_match
    spawn_match = _spawn_capacity_condition_matches(condition, can_spawn=can_spawn)
    if spawn_match is not None:
        return spawn_match
    hp_match = _hp_threshold_condition_matches(
        condition,
        current_hp=current_hp,
        max_hp=max_hp,
        move_counts=move_counts,
    )
    if hp_match is not None:
        return hp_match
    script_counter_match = _script_counter_condition_matches(
        condition,
        move_counts=move_counts,
    )
    if script_counter_match is not None:
        return script_counter_match
    respawn_counter_match = _respawn_counter_condition_matches(
        condition,
        move_counts=move_counts,
    )
    if respawn_counter_match is not None:
        return respawn_counter_match
    ally_death_match = _ally_death_condition_matches(
        condition,
        move_counts=move_counts,
    )
    if ally_death_match is not None:
        return ally_death_match
    return not condition.startswith("!")


def _formation_condition_matches(
    condition: str,
    *,
    ally_count: int | None,
    is_front: bool | None,
) -> bool | None:
    normalized = re.sub(r"\s+", "", condition).lower()
    if "getallycount()" in normalized:
        if ally_count is None:
            return None
        if "getallycount()>0" in normalized:
            return ally_count > 0
        if "getallycount()==0" in normalized:
            return ally_count == 0
    if "isalone" in normalized:
        if ally_count is None:
            return None
        return ally_count == 0
    if "isfront" in normalized:
        if is_front is None:
            return None
        return (not is_front) if normalized.startswith("!") else is_front
    return None


def _spawn_capacity_condition_matches(
    condition: str,
    *,
    can_spawn: bool | None,
) -> bool | None:
    normalized = _normalized_id(condition)
    if "canlay" not in normalized and "canfabricate" not in normalized:
        return None
    if can_spawn is None:
        return None
    return (not can_spawn) if normalized.startswith("!") else can_spawn


def _hp_threshold_condition_matches(
    condition: str,
    *,
    current_hp: int | None,
    max_hp: int | None,
    move_counts: Mapping[str, int],
) -> bool | None:
    normalized = re.sub(r"\s+", "", condition).lower()
    if "currenthp" not in normalized and "hasbeetlecharged" not in normalized:
        return None
    if current_hp is None or max_hp is None:
        return None

    at_or_above_half = current_hp >= (max_hp / 2)
    below_half = current_hp < (max_hp / 2)
    has_beetle_charged = move_counts.get("BEETLE_CHARGE", 0) > 0
    if "hasbeetlecharged||" in normalized and "currenthp>=base.creature.maxhp/2" in normalized:
        return has_beetle_charged or at_or_above_half
    if (
        normalized.startswith("!hasbeetlecharged&&")
        and "currenthp<base.creature.maxhp/2" in normalized
    ):
        return (not has_beetle_charged) and below_half
    if "currenthp>=base.creature.maxhp/2" in normalized:
        return at_or_above_half
    if "currenthp<base.creature.maxhp/2" in normalized:
        return below_half
    return None


def _script_counter_condition_matches(
    condition: str,
    *,
    move_counts: Mapping[str, int],
) -> bool | None:
    normalized = re.sub(r"\s+", "", condition).lower()
    if "_curseofknowledgecounter" not in normalized:
        return None

    curse_count = max(
        int(count)
        for move_id, count in {"CURSE_OF_KNOWLEDGE": 0, **move_counts}.items()
        if _normalized_id(move_id) == "curse_of_knowledge"
    )
    if "_curseofknowledgecounter<3" in normalized:
        return curse_count < 3
    if "_curseofknowledgecounter>=3" in normalized:
        return curse_count >= 3
    return None


def _respawn_counter_condition_matches(
    condition: str,
    *,
    move_counts: Mapping[str, int],
) -> bool | None:
    normalized = re.sub(r"\s+", "", condition).lower()
    if "respawns" not in normalized:
        return None

    respawns = max(
        int(count)
        for move_id, count in {"RESPAWNS": 0, **move_counts}.items()
        if _normalized_id(move_id) == "respawns"
    )
    if "respawns<2" in normalized:
        return respawns < 2
    if "respawns>=2" in normalized:
        return respawns >= 2
    return None


def _ally_death_condition_matches(
    condition: str,
    *,
    move_counts: Mapping[str, int],
) -> bool | None:
    normalized = re.sub(r"\s+", "", condition).lower()
    if "hasamalgamdied" not in normalized:
        return None
    has_died = move_counts.get("HAS_AMALGAM_DIED", 0) > 0
    return (not has_died) if normalized.startswith("!") else has_died


def _state_for_move(
    definition: MonsterDefinition,
    move_id: str,
) -> AttackPatternState | None:
    for state in definition.states:
        if state.move_id == move_id:
            return state
    return None


def _source_specific_next_selector(
    definition: MonsterDefinition,
    current_move_id: str,
) -> str | None:
    if (
        _normalized_id(definition.monster_id) == "bowlbug_rock"
        and _normalized_id(current_move_id) == "headbutt"
    ):
        # The engine overrides this with DIZZY when Imbalanced set its model
        # flag. Avoid randomly resolving the API's empty POST_HEADBUTT branch.
        return "HEADBUTT"
    if _normalized_id(definition.monster_id) == "axebot":
        return {
            "boot_up": "HAMMER_UPPERCUT",
            "hammer_uppercut": "ONE_TWO",
            "one_two": "HAMMER_UPPERCUT",
        }.get(_normalized_id(current_move_id))
    if _normalized_id(definition.monster_id) == "punch_construct":
        return {
            "ready": "FAST_PUNCH",
            "fast_punch": "STRONG_PUNCH",
            "strong_punch": "READY",
        }.get(_normalized_id(current_move_id))
    if _is_decimillipede_segment(definition.monster_id):
        return {
            "writhe": "CONSTRICT",
            "constrict": "BULK",
            "bulk": "WRITHE",
            "dead": "REATTACH",
            "reattach": "WRITHE",
        }.get(_normalized_id(current_move_id))
    if _normalized_id(definition.monster_id) == "entomancer":
        return {
            "bees": "SPEAR", "spear": "PHEROMONE_SPIT", "pheromone_spit": "BEES",
        }.get(_normalized_id(current_move_id))
    return None


def _next_cycle_move_id(definition: MonsterDefinition, current_move_id: str) -> str | None:
    move_ids = [move.move_id for move in definition.moves]
    if not move_ids:
        return None
    try:
        current_index = move_ids.index(current_move_id)
    except ValueError:
        return move_ids[0]
    return move_ids[(current_index + 1) % len(move_ids)]


def _fallback_move(
    definition: MonsterDefinition,
    rng: Random,
    *,
    previous_move_id: str | None,
) -> MonsterMove | None:
    if not definition.moves:
        return None
    candidates = tuple(move for move in definition.moves if move.move_id != previous_move_id)
    if _is_decimillipede_segment(definition.monster_id):
        candidates = tuple(
            move
            for move in candidates
            if _normalized_id(move.move_id) not in {"dead", "reattach"}
        )
    if not candidates:
        candidates = definition.moves
    return rng.choice(candidates)


def _is_decimillipede_segment(monster_id: str) -> bool:
    return _normalized_id(monster_id).startswith("decimillipede_segment")


def _uses_ascension_damage(definition: MonsterDefinition, ascension_level: int) -> bool:
    return _uses_ascension_damage_kind(definition.kind, ascension_level)


def _uses_ascension_damage_kind(kind: str, ascension_level: int) -> bool:
    threshold = {
        "normal": NORMAL_DAMAGE_ASCENSION,
        "elite": ELITE_DAMAGE_ASCENSION,
        "boss": BOSS_DAMAGE_ASCENSION,
    }.get(kind.lower(), NORMAL_DAMAGE_ASCENSION)
    return ascension_level >= threshold


def _uses_ascension_hp(definition: MonsterDefinition, ascension_level: int) -> bool:
    threshold = {
        "normal": NORMAL_HP_ASCENSION,
        "elite": ELITE_HP_ASCENSION,
        "boss": BOSS_HP_ASCENSION,
    }.get(definition.kind.lower(), NORMAL_HP_ASCENSION)
    return ascension_level >= threshold


def _find_encounter(
    encounters: Sequence[EncounterDefinition],
    encounter_id: str,
) -> EncounterDefinition | None:
    normalized = _normalized_id(encounter_id)
    for encounter in encounters:
        if _normalized_id(encounter.encounter_id) == normalized:
            return encounter
    return None


def _normalized_room_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"normal", "monster", "monsters"}:
        return "monster"
    if normalized in {"elite", "elites"}:
        return "elite"
    if normalized in {"boss", "bosses"}:
        return "boss"
    return normalized


def _act_number(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"\bAct\s+(\d+)\b", str(value), flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def _named_act_id(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().upper()
    for act_id in ("OVERGROWTH", "UNDERDOCKS", "HIVE", "GLORY"):
        if normalized == act_id or normalized.endswith(f"- {act_id}"):
            return act_id
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalized_id(value: str) -> str:
    return value.lower().replace("'", "").replace(" ", "_").replace("-", "_")
