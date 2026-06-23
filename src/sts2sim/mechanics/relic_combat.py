"""Pure combat relic hook helpers.

The combat engine owns state mutation.  This module turns owned relic ids and
small hook contexts into deterministic markers or explicit blocker records that
callers can apply at the correct point in combat.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sts2sim.content.sources import PROVISIONAL_STS2_SOURCE, STS1_COMPAT_SOURCE, SourceRef

CombatRelicInput = str | Mapping[str, object]
StatusMap = Mapping[str, object] | None


class CombatRelicHook(str, Enum):
    START_OF_COMBAT = "start_of_combat"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    CARD_PLAYED = "card_played"
    CARD_EXHAUSTED = "card_exhausted"
    CARD_DISCARDED = "card_discarded"
    CARD_CREATED = "card_created"
    CARD_BLOCK_GAINED = "card_block_gained"
    DAMAGE_DEALT = "damage_dealt"
    DAMAGE_TAKEN = "damage_taken"
    DRAW_PILE_SHUFFLED = "draw_pile_shuffled"
    ENEMY_BLOCK_BROKEN = "enemy_block_broken"
    HAND_EMPTY = "hand_empty"
    MONSTER_KILLED = "monster_killed"
    ORB_CHANNELED = "orb_channeled"
    ORB_PASSIVE_TRIGGERED = "orb_passive_triggered"
    POTION_USED = "potion_used"
    RESOURCE_SPENT = "resource_spent"
    STATUS_APPLIED = "status_applied"
    STATUS_GAINED = "status_gained"
    COMBAT_END = "combat_end"


@dataclass(frozen=True, slots=True)
class CombatRelicContext:
    """Hook context needed by conditional relic helpers."""

    turn_number: int | None = None
    player_hp: int | None = None
    player_max_hp: int | None = None
    player_block: int | None = None
    encounter_type: str | None = None
    card_type: str | None = None
    card_id: str | None = None
    target_id: str | None = None
    player_statuses: Mapping[str, object] = field(default_factory=dict)
    target_statuses: Mapping[str, object] = field(default_factory=dict)
    relic_counters: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "player_statuses", dict(self.player_statuses))
        object.__setattr__(self, "target_statuses", dict(self.target_statuses))
        object.__setattr__(
            self,
            "relic_counters",
            {str(key): int(value) for key, value in self.relic_counters.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class CombatRelicMarker:
    kind: str
    relic_id: str
    hook: CombatRelicHook
    amount: int | None = None
    target_id: str | None = None
    source_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    source: SourceRef = PROVISIONAL_STS2_SOURCE

    def __post_init__(self) -> None:
        if self.source_id is None:
            object.__setattr__(self, "source_id", self.relic_id)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class CombatRelicBlocker:
    hook: CombatRelicHook
    relic_id: str
    reason: str
    source_id: str
    name: str | None = None
    description: str | None = None
    source: SourceRef = PROVISIONAL_STS2_SOURCE


@dataclass(frozen=True, slots=True)
class CombatRelicResolution:
    hook: CombatRelicHook
    markers: tuple[CombatRelicMarker, ...] = ()
    blockers: tuple[CombatRelicBlocker, ...] = ()
    hp_delta: int = 0
    block_delta: int = 0
    energy_delta: int = 0
    source: SourceRef = PROVISIONAL_STS2_SOURCE


@dataclass(frozen=True, slots=True)
class CombatRelicMarkerSpec:
    kind: str
    amount: int | None = None
    target_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    source: SourceRef = PROVISIONAL_STS2_SOURCE


def resolve_combat_relic_hook(
    relics: Sequence[CombatRelicInput],
    hook: CombatRelicHook | str,
    *,
    context: CombatRelicContext | None = None,
    include_blockers: bool = True,
    turn_number: int | None = None,
    player_hp: int | None = None,
    player_max_hp: int | None = None,
    player_block: int | None = None,
    encounter_type: str | None = None,
    card_type: str | None = None,
    card_id: str | None = None,
    target_id: str | None = None,
    player_statuses: StatusMap = None,
    target_statuses: StatusMap = None,
    relic_counters: Mapping[str, int] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> CombatRelicResolution:
    """Resolve all combat relic effects for one hook.

    Unknown relics are only reported as blockers when their description or
    built-in classification indicates that the requested hook is relevant.
    """

    normalized_hook = _combat_hook(hook)
    hook_context = context or CombatRelicContext(
        turn_number=turn_number,
        player_hp=player_hp,
        player_max_hp=player_max_hp,
        player_block=player_block,
        encounter_type=encounter_type,
        card_type=card_type,
        card_id=card_id,
        target_id=target_id,
        player_statuses=player_statuses or {},
        target_statuses=target_statuses or {},
        relic_counters=relic_counters or {},
        metadata=metadata or {},
    )

    markers: list[CombatRelicMarker] = []
    blockers: list[CombatRelicBlocker] = []
    current_hp = hook_context.player_hp
    for relic in _unique_relic_inputs(relics):
        relic_id = combat_relic_content_id(relic)
        resolved = _resolve_supported_relic(relic_id, normalized_hook, hook_context, hp=current_hp)
        if resolved:
            markers.extend(resolved)
            current_hp = _advance_hp(current_hp, resolved)
            continue
        if include_blockers and normalized_hook in inferred_combat_relic_hooks(relic):
            blockers.append(
                combat_relic_blocker(
                    relic,
                    normalized_hook,
                    reason="No pure combat relic helper is registered for this inferred hook.",
                )
            )

    return _resolution_from_markers(normalized_hook, tuple(markers), tuple(blockers))


def start_of_combat(
    relics: Sequence[CombatRelicInput],
    **kwargs: Any,
) -> CombatRelicResolution:
    return resolve_combat_relic_hook(relics, CombatRelicHook.START_OF_COMBAT, **kwargs)


def turn_start(
    relics: Sequence[CombatRelicInput],
    **kwargs: Any,
) -> CombatRelicResolution:
    return resolve_combat_relic_hook(relics, CombatRelicHook.TURN_START, **kwargs)


def turn_end(
    relics: Sequence[CombatRelicInput],
    **kwargs: Any,
) -> CombatRelicResolution:
    return resolve_combat_relic_hook(relics, CombatRelicHook.TURN_END, **kwargs)


def card_played(
    relics: Sequence[CombatRelicInput],
    **kwargs: Any,
) -> CombatRelicResolution:
    return resolve_combat_relic_hook(relics, CombatRelicHook.CARD_PLAYED, **kwargs)


def damage_dealt(
    relics: Sequence[CombatRelicInput],
    **kwargs: Any,
) -> CombatRelicResolution:
    return resolve_combat_relic_hook(relics, CombatRelicHook.DAMAGE_DEALT, **kwargs)


def damage_taken(
    relics: Sequence[CombatRelicInput],
    **kwargs: Any,
) -> CombatRelicResolution:
    return resolve_combat_relic_hook(relics, CombatRelicHook.DAMAGE_TAKEN, **kwargs)


def monster_killed(
    relics: Sequence[CombatRelicInput],
    **kwargs: Any,
) -> CombatRelicResolution:
    return resolve_combat_relic_hook(relics, CombatRelicHook.MONSTER_KILLED, **kwargs)


def combat_end(
    relics: Sequence[CombatRelicInput],
    **kwargs: Any,
) -> CombatRelicResolution:
    return resolve_combat_relic_hook(relics, CombatRelicHook.COMBAT_END, **kwargs)


resolve_start_of_combat_relics = start_of_combat
resolve_turn_start_relics = turn_start
resolve_turn_end_relics = turn_end
resolve_card_played_relics = card_played
resolve_damage_dealt_relics = damage_dealt
resolve_damage_taken_relics = damage_taken
resolve_monster_killed_relics = monster_killed
resolve_combat_end_relics = combat_end


def supported_combat_relic_ids(
    hook: CombatRelicHook | str | None = None,
) -> frozenset[str]:
    """Return relic ids with at least one explicit combat helper."""

    if hook is not None:
        return frozenset(_SUPPORTED_BY_HOOK.get(_combat_hook(hook), frozenset()))
    supported: set[str] = set()
    for relic_ids in _SUPPORTED_BY_HOOK.values():
        supported.update(relic_ids)
    return frozenset(supported)


def unsupported_combat_relic_handlers(
    relics: Sequence[CombatRelicInput],
    *,
    hooks: Sequence[CombatRelicHook | str] | None = None,
) -> tuple[CombatRelicBlocker, ...]:
    """Report inferred combat hooks without bounded helper implementations."""

    blockers: list[CombatRelicBlocker] = []
    requested_hooks = tuple(_combat_hook(hook) for hook in hooks) if hooks is not None else None
    for relic in relics:
        relic_id = combat_relic_content_id(relic)
        inferred_hooks = requested_hooks or inferred_combat_relic_hooks(relic)
        for hook in inferred_hooks:
            if relic_id in _SUPPORTED_BY_HOOK.get(hook, frozenset()):
                continue
            blockers.append(
                combat_relic_blocker(
                    relic,
                    hook,
                    reason="No pure combat relic helper is registered for this inferred hook.",
                )
            )
    return tuple(blockers)


def combat_relic_blocker(
    relic: CombatRelicInput,
    hook: CombatRelicHook | str,
    *,
    reason: str,
) -> CombatRelicBlocker:
    relic_id = combat_relic_content_id(relic)
    return CombatRelicBlocker(
        hook=_combat_hook(hook),
        relic_id=relic_id,
        source_id=relic_source_id(relic),
        name=_content_str(relic, "name"),
        description=_content_str(relic, "description", "description_raw"),
        reason=reason,
    )


def inferred_combat_relic_hooks(
    relic: CombatRelicInput,
) -> tuple[CombatRelicHook, ...]:
    """Infer combat hooks from a source row description or known relic id."""

    relic_id = combat_relic_content_id(relic)
    known = _KNOWN_RELIC_HOOKS.get(relic_id, ())
    description = _content_str(relic, "description", "description_raw")
    if not description:
        return known

    text = _clean_description(description)
    hooks: list[CombatRelicHook] = list(known)

    if (
        "start each combat" in text
        or "start of each combat" in text
        or "start of combat" in text
        or "start of each boss combat" in text
        or "start of each elite combat" in text
        or "while you have no potions" in text
    ):
        hooks.append(CombatRelicHook.START_OF_COMBAT)
    if "start of each turn" in text or "start of your turn" in text:
        hooks.append(CombatRelicHook.TURN_START)
    if "every " in text and " turns" in text:
        hooks.append(CombatRelicHook.TURN_START)
    if "end your turn" in text or "end of your turn" in text or "end of turn" in text:
        hooks.append(CombatRelicHook.TURN_END)
    if (
        "whenever you play" in text
        or "every time you play" in text
        or "first card you play" in text
        or " card you play" in text
        or " attacks you play" in text
        or " skills you play" in text
        or "cost x cards" in text
    ):
        hooks.append(CombatRelicHook.CARD_PLAYED)
    if "exhaust" in text and (
        "whenever you" in text
        or "every " in text
        or "first time" in text
        or "times you" in text
    ):
        hooks.append(CombatRelicHook.CARD_EXHAUSTED)
    if "discard" in text and ("whenever you" in text or "during your turn" in text):
        hooks.append(CombatRelicHook.CARD_DISCARDED)
    if "whenever you create a card" in text:
        hooks.append(CombatRelicHook.CARD_CREATED)
    if "block gained from a card" in text or "gain block from a card" in text:
        hooks.append(CombatRelicHook.CARD_BLOCK_GAINED)
    if "shuffle your draw pile" in text:
        hooks.append(CombatRelicHook.DRAW_PILE_SHUFFLED)
    if "break an enemy" in text and "block" in text:
        hooks.append(CombatRelicHook.ENEMY_BLOCK_BROKEN)
    if "no cards in hand during your turn" in text:
        hooks.append(CombatRelicHook.HAND_EMPTY)
    if "channel" in text and "orb" in text:
        hooks.append(CombatRelicHook.ORB_CHANNELED)
    if "orb triggers its passive" in text or "passive ability of all orbs" in text:
        hooks.append(CombatRelicHook.ORB_PASSIVE_TRIGGERED)
    if "whenever you use a potion" in text:
        hooks.append(CombatRelicHook.POTION_USED)
    if "spent" in text and ("[star" in text or "star" in text):
        hooks.append(CombatRelicHook.RESOURCE_SPENT)
    if "whenever you apply" in text:
        hooks.append(CombatRelicHook.STATUS_APPLIED)
    if "gain strength" in text and "first time" in text:
        hooks.append(CombatRelicHook.STATUS_GAINED)
    triggered_damage = any(
        phrase in text
        for phrase in (
            "at the start",
            "at the end",
            "every time you play",
            "whenever you play",
            "if you end",
            "less damage",
            "less damage to you",
            "damage to you",
            "end of turn",
            "end your turn",
        )
    )
    if "deal" in text and "damage" in text and not triggered_damage:
        hooks.append(CombatRelicHook.DAMAGE_DEALT)
    player_damage_taken = (
        ("take " in text and "damage" in text)
        or "lose hp" in text
        or "less damage to you" in text
        or "damage to you" in text
    )
    enemy_vulnerable_damage_bonus = (
        "enemies with vulnerable take" in text or "enemy with vulnerable take" in text
    )
    if player_damage_taken and not enemy_vulnerable_damage_bonus:
        hooks.append(CombatRelicHook.DAMAGE_TAKEN)
    if "kill an elite" in text or "defeating" in text or "defeated" in text:
        hooks.append(CombatRelicHook.MONSTER_KILLED)
    if "end of combat" in text or "after each combat" in text:
        hooks.append(CombatRelicHook.COMBAT_END)

    return tuple(dict.fromkeys(hooks))


def combat_relic_content_id(relic: CombatRelicInput) -> str:
    """Return a normalized id from a raw id, name, or Codex-style source row."""

    if isinstance(relic, str):
        return _normalized_id(relic)
    value = _first_present(relic, "id", "relic_id", "content_id", "item_id", "name")
    if value is None:
        raise ValueError(f"Relic input is missing an id or name: {relic!r}")
    return _normalized_id(str(value))


def relic_source_id(relic: CombatRelicInput) -> str:
    if isinstance(relic, str):
        return combat_relic_content_id(relic)
    value = _first_present(relic, "id", "relic_id", "content_id", "item_id")
    if value is None:
        value = _first_present(relic, "name")
    return _normalized_id(str(value)) if value is not None else combat_relic_content_id(relic)


def _resolve_supported_relic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
    *,
    hp: int | None,
) -> tuple[CombatRelicMarker, ...]:
    specs = _STATIC_MARKERS_BY_HOOK.get(hook, {}).get(relic_id, ())
    if specs:
        return tuple(_marker_from_spec(relic_id, hook, spec, context, hp=hp) for spec in specs)

    if hook is CombatRelicHook.START_OF_COMBAT:
        return _resolve_start_of_combat_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.TURN_START:
        return _resolve_turn_start_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.TURN_END:
        orichalcum = _resolve_turn_end_dynamic(relic_id, hook, context)
        if orichalcum:
            return orichalcum
        return _resolve_turn_end_other_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.CARD_PLAYED:
        return _resolve_card_played_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.DAMAGE_DEALT:
        return _resolve_damage_dealt_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.DAMAGE_TAKEN:
        return _resolve_damage_taken_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.CARD_EXHAUSTED:
        return _resolve_card_exhausted_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.CARD_DISCARDED:
        return _resolve_card_discarded_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.CARD_CREATED:
        return _resolve_card_created_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.CARD_BLOCK_GAINED:
        return _resolve_card_block_gained_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.DRAW_PILE_SHUFFLED:
        return _resolve_draw_pile_shuffled_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.ENEMY_BLOCK_BROKEN:
        return _resolve_enemy_block_broken_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.HAND_EMPTY:
        return _resolve_hand_empty_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.ORB_CHANNELED:
        return _resolve_orb_channeled_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.ORB_PASSIVE_TRIGGERED:
        return _resolve_orb_passive_triggered_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.POTION_USED:
        return _resolve_potion_used_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.RESOURCE_SPENT:
        return _resolve_resource_spent_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.STATUS_APPLIED:
        return _resolve_status_applied_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.STATUS_GAINED:
        return _resolve_status_gained_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.MONSTER_KILLED:
        return _resolve_monster_killed_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.COMBAT_END:
        return _resolve_combat_end_dynamic(relic_id, hook, context)
    return ()


def _resolve_start_of_combat_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id == "bellows":
        opening_draw_count = _coerce_int(context.metadata.get("opening_draw_count"), default=5)
        amount = min(10, max(0, opening_draw_count))
        if amount <= 0:
            return ()
        return (
            CombatRelicMarker(
                kind="upgrade_draw_pile_cards",
                relic_id=relic_id,
                hook=hook,
                amount=amount,
                target_id="player",
                metadata={"mode": "opening_hand"},
            ),
        )

    if relic_id == "belt_buckle":
        if _coerce_int(context.metadata.get("potion_count"), default=0) > 0:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_status",
                relic_id=relic_id,
                hook=hook,
                amount=2,
                target_id="player",
                metadata={"status": "dexterity", "condition": "no_potions"},
            ),
        )

    if relic_id == "delicate_frond":
        empty_slots_value = _first_present(
            context.metadata,
            "empty_potion_slots",
            "potion_slots_open",
        )
        if empty_slots_value is None:
            if "potion_slots" not in context.metadata or "potion_count" not in context.metadata:
                return ()
            empty_slots = max(
                0,
                _coerce_int(context.metadata.get("potion_slots"))
                - _coerce_int(context.metadata.get("potion_count")),
            )
        else:
            empty_slots = max(0, _coerce_int(empty_slots_value))
        if empty_slots <= 0:
            return ()
        return (
            CombatRelicMarker(
                kind="random_potions_gained",
                relic_id=relic_id,
                hook=hook,
                amount=empty_slots,
                target_id="player",
                metadata={
                    "condition": "empty_potion_slots",
                    "empty_potion_slots": empty_slots,
                },
            ),
        )

    if relic_id == "ember_tea":
        charges = context.relic_counters.get(relic_id, 0)
        if charges <= 0:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_status",
                relic_id=relic_id,
                hook=hook,
                amount=2,
                target_id="player",
                metadata={
                    "status": "strength",
                    "condition": "next_5_combat_charge",
                    "next_counter": max(0, charges - 1),
                },
            ),
        )

    if relic_id == "red_skull":
        if not _player_hp_at_or_below_half(context.player_hp, context.player_max_hp):
            return ()
        return (
            CombatRelicMarker(
                kind="gain_status",
                relic_id=relic_id,
                hook=hook,
                amount=3,
                target_id="player",
                metadata={
                    "status": "strength",
                    "condition": "hp_at_or_below_50_percent",
                },
            ),
        )

    if relic_id == "tea_of_discourtesy":
        charges = context.relic_counters.get(relic_id, 0)
        if charges <= 0:
            return ()
        return (
            CombatRelicMarker(
                kind="shuffle_status_into_draw_pile",
                relic_id=relic_id,
                hook=hook,
                amount=2,
                target_id="player",
                metadata={
                    "card_id": "dazed",
                    "card_type": "status",
                    "target": "self",
                    "condition": "next_combat_charge",
                    "next_counter": max(0, charges - 1),
                },
            ),
        )

    if relic_id == "pantograph":
        if context.encounter_type is not None and _normalized_id(context.encounter_type) != "boss":
            return ()
        return (
            CombatRelicMarker(
                kind="heal_player",
                relic_id=relic_id,
                hook=hook,
                amount=25,
                target_id="player",
                metadata={"condition": "boss_combat"},
                source=STS1_COMPAT_SOURCE,
            ),
        )

    if relic_id == "booming_conch":
        if context.encounter_type is not None and _normalized_id(context.encounter_type) != "elite":
            return ()
        return (
            CombatRelicMarker(
                kind="gain_energy",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"condition": "elite_combat"},
            ),
        )

    if relic_id == "sling_of_courage":
        if context.encounter_type is not None and _normalized_id(context.encounter_type) != "elite":
            return ()
        return (
            CombatRelicMarker(
                kind="gain_status",
                relic_id=relic_id,
                hook=hook,
                amount=2,
                target_id="player",
                metadata={"status": "strength", "condition": "elite_combat"},
            ),
        )

    if relic_id == "bone_tea":
        charges = context.relic_counters.get(relic_id, 0)
        if charges <= 0:
            return ()
        opening_draw_count = _coerce_int(context.metadata.get("opening_draw_count"), default=5)
        amount = min(10, max(0, opening_draw_count))
        if amount <= 0:
            return ()
        return (
            CombatRelicMarker(
                kind="upgrade_draw_pile_cards",
                relic_id=relic_id,
                hook=hook,
                amount=amount,
                target_id="player",
                metadata={
                    "mode": "opening_hand",
                    "condition": "next_combat_charge",
                    "next_counter": max(0, charges - 1),
                },
            ),
        )

    if relic_id != "preserved_insect":
        return ()
    if context.encounter_type is not None and _normalized_id(context.encounter_type) != "elite":
        return ()
    return (
        CombatRelicMarker(
            kind="elite_monster_hp_multiplier",
            relic_id=relic_id,
            hook=hook,
            amount=75,
            target_id="all_enemies",
            metadata={
                "condition": "elite_combat",
                "multiplier_percent": 75,
                "hp_reduction_percent": 25,
            },
            source=STS1_COMPAT_SOURCE,
        ),
    )


def _resolve_turn_start_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id == "bread":
        if context.turn_number == 1:
            return (
                CombatRelicMarker(
                    kind="gain_energy",
                    relic_id=relic_id,
                    hook=hook,
                    amount=-2,
                    target_id="player",
                    metadata={"turn_number": 1},
                ),
            )
        if context.turn_number is None:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_energy",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"condition": "after_first_turn", "turn_number": context.turn_number},
            ),
        )

    if relic_id == "candelabra":
        if context.turn_number != 2:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_energy",
                relic_id=relic_id,
                hook=hook,
                amount=2,
                target_id="player",
                metadata={"turn_number": context.turn_number},
            ),
        )

    if relic_id == "horn_cleat":
        if context.turn_number != 2:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_block",
                relic_id=relic_id,
                hook=hook,
                amount=14,
                target_id="player",
                metadata={"turn_number": context.turn_number},
                source=STS1_COMPAT_SOURCE,
            ),
        )

    if relic_id == "captains_wheel":
        if context.turn_number != 3:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_block",
                relic_id=relic_id,
                hook=hook,
                amount=18,
                target_id="player",
                metadata={"turn_number": context.turn_number},
                source=STS1_COMPAT_SOURCE,
            ),
        )

    if relic_id == "chandelier":
        if context.turn_number != 3:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_energy",
                relic_id=relic_id,
                hook=hook,
                amount=3,
                target_id="player",
                metadata={"turn_number": context.turn_number},
            ),
        )

    if relic_id == "sparkling_rouge":
        if context.turn_number != 3:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_status",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"status": "strength", "turn_number": context.turn_number},
            ),
            CombatRelicMarker(
                kind="gain_status",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"status": "dexterity", "turn_number": context.turn_number},
            ),
        )

    if relic_id == "paels_flesh":
        if context.turn_number is None or context.turn_number < 3:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_energy",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"condition": "turn_3_and_after", "turn_number": context.turn_number},
            ),
        )

    if relic_id == "ring_of_the_drake":
        if context.turn_number is None or context.turn_number > 3:
            return ()
        return (
            CombatRelicMarker(
                kind="draw_cards",
                relic_id=relic_id,
                hook=hook,
                amount=2,
                target_id="player",
                metadata={"condition": "first_3_turns", "turn_number": context.turn_number},
            ),
        )

    if relic_id == "mr_struggles":
        if context.turn_number is None:
            return ()
        return (
            CombatRelicMarker(
                kind="all_damage",
                relic_id=relic_id,
                hook=hook,
                amount=max(0, context.turn_number),
                target_id="all_enemies",
                metadata={"turn_number": context.turn_number},
            ),
        )

    if relic_id == "emotion_chip":
        lost_hp = _metadata_flag(context.metadata, "lost_hp_previous_turn", "lost_hp_last_turn")
        lost_hp = lost_hp or _coerce_int(
            _first_present(
                context.metadata,
                "hp_lost_previous_turn",
                "damage_taken_previous_turn",
            ),
            default=0,
        ) > 0
        if not lost_hp:
            return ()
        return (
            CombatRelicMarker(
                kind="trigger_orb_passive",
                relic_id=relic_id,
                hook=hook,
                target_id="player",
                metadata={"selector": "all", "condition": "lost_hp_previous_turn"},
            ),
        )

    if relic_id == "history_course":
        last_type = _normalized_id(
            str(
                _first_present(
                    context.metadata,
                    "last_played_card_type",
                    "last_card_type",
                )
                or ""
            )
        )
        if last_type not in {"attack", "skill"}:
            return ()
        metadata: dict[str, object] = {
            "selection": "last_played_attack_or_skill",
            "card_type": last_type,
            "condition": "start_of_turn",
        }
        last_card_id = _first_present(
            context.metadata,
            "last_played_card_id",
            "last_card_id",
        )
        if last_card_id is not None:
            metadata["copy_source_card_id"] = _normalized_id(last_card_id)
        last_card_instance_id = _first_present(
            context.metadata,
            "last_played_card_instance_id",
            "last_card_instance_id",
            "copy_source_card_instance_id",
        )
        if last_card_instance_id is not None:
            metadata["copy_source_card_instance_id"] = str(last_card_instance_id)
        last_target_id = _first_present(
            context.metadata,
            "last_played_target_id",
            "last_target_id",
            "copy_source_target_id",
        )
        if last_target_id is not None:
            metadata["copy_source_target_id"] = str(last_target_id)
        last_card_cost = _first_present(
            context.metadata,
            "last_played_card_cost",
            "last_card_cost",
            "copy_source_card_cost",
        )
        if last_card_cost is not None:
            metadata["copy_source_card_cost"] = _coerce_int(last_card_cost)
        last_card = context.metadata.get("last_played_card")
        if isinstance(last_card, Mapping):
            metadata["copy_source_card"] = dict(last_card)
        return (
            CombatRelicMarker(
                kind="play_card_copy",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata=metadata,
            ),
        )

    if relic_id in _PERIODIC_TURN_DRAW:
        period, draw = _PERIODIC_TURN_DRAW[relic_id]
        if context.turn_number is None or context.turn_number % period != 0:
            return ()
        return (
            CombatRelicMarker(
                kind="draw_cards",
                relic_id=relic_id,
                hook=hook,
                amount=draw,
                target_id="player",
                metadata={"period": period, "turn_number": context.turn_number},
            ),
        )

    if relic_id == "seal_of_gold":
        gold = _coerce_int(context.metadata.get("gold"), default=0)
        if gold < 5:
            return ()
        return (
            CombatRelicMarker(
                kind="gold_delta",
                relic_id=relic_id,
                hook=hook,
                amount=-5,
                target_id="player",
                metadata={"condition": "has_at_least_5_gold"},
            ),
            CombatRelicMarker(
                kind="gain_energy",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"condition": "spent_5_gold"},
            ),
        )

    if relic_id not in _PERIODIC_TURN_ENERGY:
        return ()

    period, energy = _PERIODIC_TURN_ENERGY[relic_id]
    counter = context.relic_counters.get(relic_id)
    if counter is not None:
        next_counter = counter + 1
        if next_counter < period:
            return (
                _counter_marker(
                    relic_id,
                    hook,
                    next_counter=next_counter,
                    period=period,
                    source=PROVISIONAL_STS2_SOURCE,
                ),
            )
        return (
            CombatRelicMarker(
                kind="gain_energy",
                relic_id=relic_id,
                hook=hook,
                amount=energy,
                target_id="player",
                metadata={"period": period, "next_counter": 0},
            ),
        )

    if context.turn_number is None:
        return (
            CombatRelicMarker(
                kind="periodic_energy_check",
                relic_id=relic_id,
                hook=hook,
                amount=energy,
                target_id="player",
                metadata={"period": period, "requires": "turn_number_or_relic_counter"},
            ),
        )

    if context.turn_number % period != 0:
        return ()
    return (
        CombatRelicMarker(
            kind="gain_energy",
            relic_id=relic_id,
            hook=hook,
            amount=energy,
            target_id="player",
            metadata={"period": period, "turn_number": context.turn_number},
        ),
    )


def _resolve_turn_end_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id not in _ORICHALCUM_BLOCK:
        return ()
    amount = _ORICHALCUM_BLOCK[relic_id]
    if context.player_block is not None and context.player_block > 0:
        return ()
    metadata: dict[str, object] = {"condition": "player_block_is_zero"}
    if context.player_block is not None:
        metadata["player_block"] = context.player_block
    return (
        CombatRelicMarker(
            kind="gain_block",
            relic_id=relic_id,
            hook=hook,
            amount=amount,
            target_id="player",
            metadata=metadata,
        ),
    )


def _resolve_turn_end_other_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id == "art_of_war":
        attacks = _coerce_int(context.metadata.get("attacks_played_this_turn"), default=0)
        if attacks > 0:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_status",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"status": "next_turn_energy", "condition": "no_attacks_played"},
                source=STS1_COMPAT_SOURCE,
            ),
        )

    if relic_id == "bookmark":
        retained_count = _coerce_int(context.metadata.get("retained_card_count"), default=1)
        if retained_count <= 0:
            return ()
        return (
            CombatRelicMarker(
                kind="reduce_retained_card_cost",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={
                    "selection": "random",
                    "condition": "retained_card",
                    "duration": "until_played",
                },
            ),
        )

    if relic_id == "pocketwatch":
        if "cards_played_this_turn" not in context.metadata:
            return ()
        cards_played = _coerce_int(context.metadata.get("cards_played_this_turn"), default=0)
        if cards_played > 3:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_status",
                relic_id=relic_id,
                hook=hook,
                amount=3,
                target_id="player",
                metadata={
                    "status": "next_turn_draw",
                    "condition": "played_3_or_fewer_cards",
                    "cards_played_this_turn": cards_played,
                },
            ),
        )

    if relic_id == "ripple_basin":
        attacks = _coerce_int(context.metadata.get("attacks_played_this_turn"), default=0)
        if attacks > 0:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_block",
                relic_id=relic_id,
                hook=hook,
                amount=4,
                target_id="player",
                metadata={"condition": "no_attacks_played"},
            ),
        )

    if relic_id == "ringing_triangle":
        if context.turn_number != 1:
            return ()
        return (
            CombatRelicMarker(
                kind="retain_hand",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"mode": "retain_full_hand", "condition": "first_turn"},
            ),
        )

    if relic_id == "cloak_clasp":
        hand_size = _coerce_int(context.metadata.get("hand_size"), default=0)
        if hand_size <= 0:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_block",
                relic_id=relic_id,
                hook=hook,
                amount=hand_size,
                target_id="player",
                metadata={"hand_size": hand_size},
            ),
        )
    if relic_id == "paels_tears":
        energy = _coerce_int(context.metadata.get("energy"), default=0)
        if energy <= 0:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_status",
                relic_id=relic_id,
                hook=hook,
                amount=2,
                target_id="player",
                metadata={"status": "next_turn_energy", "condition": "unspent_energy"},
            ),
        )
    if relic_id == "paels_eye":
        if context.relic_counters.get(relic_id, 0) > 0:
            return ()
        cards_played = _coerce_int(context.metadata.get("cards_played_this_turn"), default=0)
        if cards_played > 0:
            return ()
        hand_size = _coerce_int(context.metadata.get("hand_size"), default=0)
        return (
            CombatRelicMarker(
                kind="exhaust_hand",
                relic_id=relic_id,
                hook=hook,
                amount=hand_size if hand_size > 0 else None,
                target_id="player",
                metadata={
                    "condition": "first_turn_ended_with_no_cards_played",
                    "next_counter": 1,
                },
            ),
            CombatRelicMarker(
                kind="take_extra_turn",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={
                    "condition": "first_turn_ended_with_no_cards_played",
                    "next_counter": 1,
                },
            ),
        )
    if relic_id == "parrying_shield":
        block = context.player_block
        if block is None:
            block = _coerce_int(context.metadata.get("player_block"), default=0)
        if block < 10:
            return ()
        return (
            CombatRelicMarker(
                kind="random_damage",
                relic_id=relic_id,
                hook=hook,
                amount=6,
                target_id="random_enemy",
                metadata={
                    "condition": "block_at_least_10",
                    "player_block": block,
                    "threshold": 10,
                },
            ),
        )
    if relic_id == "ice_cream":
        energy = _coerce_int(context.metadata.get("energy"), default=0)
        if energy <= 0:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_status",
                relic_id=relic_id,
                hook=hook,
                amount=energy,
                target_id="player",
                metadata={
                    "status": "next_turn_energy",
                    "condition": "unspent_energy_conserved",
                    "energy": energy,
                },
            ),
        )
    if relic_id == "sturdy_clamp":
        block = context.player_block
        if block is None:
            block = _coerce_int(context.metadata.get("player_block"), default=0)
        if block <= 0:
            return ()
        persisted = min(10, block)
        return (
            CombatRelicMarker(
                kind="gain_status",
                relic_id=relic_id,
                hook=hook,
                amount=persisted,
                target_id="player",
                metadata={
                    "status": "next_turn_block",
                    "condition": "block_persists",
                    "player_block": block,
                    "limit": 10,
                },
            ),
        )
    if relic_id == "screaming_flagon":
        hand_size = _coerce_int(context.metadata.get("hand_size"), default=0)
        if hand_size > 0:
            return ()
        return (
            CombatRelicMarker(
                kind="all_damage",
                relic_id=relic_id,
                hook=hook,
                amount=20,
                target_id="all_enemies",
                metadata={"condition": "empty_hand"},
            ),
        )
    if relic_id == "stone_calendar":
        if context.turn_number != 7:
            return ()
        return (
            CombatRelicMarker(
                kind="all_damage",
                relic_id=relic_id,
                hook=hook,
                amount=52,
                target_id="all_enemies",
                metadata={"turn_number": context.turn_number},
            ),
        )
    return ()


def _resolve_card_played_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    card_type = _normalized_id(context.card_type or "")
    card_id = _normalized_id(context.card_id or "")
    cards_played_next = _int_context_value(context, "cards_played_this_turn") + 1

    if relic_id == "chemical_x":
        cost = _normalized_id(str(_first_present(context.metadata, "card_cost", "cost") or ""))
        if cost not in {"x", "x_cost"}:
            return ()
        return (
            CombatRelicMarker(
                kind="x_card_effect_bonus",
                relic_id=relic_id,
                hook=hook,
                amount=2,
                target_id="player",
                metadata={"condition": "x_cost_card_played", "increase": 2},
                source=STS1_COMPAT_SOURCE,
            ),
        )

    if relic_id == "throwing_axe":
        if context.relic_counters.get(relic_id, 0) > 0:
            return ()
        if _coerce_int(context.metadata.get("cards_played_this_combat"), default=0) > 0:
            return ()
        metadata: dict[str, object] = {
            "selection": "played_card",
            "condition": "first_card_played_this_combat",
            "next_counter": 1,
        }
        if card_id:
            metadata["copy_source_card_id"] = card_id
        if card_type:
            metadata["copy_source_card_type"] = card_type
        played_card_instance_id = _first_present(
            context.metadata,
            "played_card_instance_id",
            "card_instance_id",
            "copy_source_card_instance_id",
        )
        if played_card_instance_id is not None:
            metadata["copy_source_card_instance_id"] = str(played_card_instance_id)
        if context.target_id:
            metadata["copy_source_target_id"] = context.target_id
        return (
            CombatRelicMarker(
                kind="play_card_again",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata=metadata,
            ),
        )

    if relic_id == "razor_tooth":
        if card_type not in {"attack", "skill"}:
            return ()
        metadata = {
            "condition": "attack_or_skill_played",
            "duration": "combat",
            "card_type": card_type,
        }
        played_card_instance_id = context.metadata.get("played_card_instance_id")
        if played_card_instance_id is not None:
            metadata["played_card_instance_id"] = str(played_card_instance_id)
        return (
            CombatRelicMarker(
                kind="upgrade_played_card",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata=metadata,
            ),
        )

    if relic_id == "unsettling_lamp":
        if context.relic_counters.get(relic_id, 0) > 0:
            return ()
        debuffs_enemy = _metadata_flag(
            context.metadata,
            "debuffs_enemy",
            "applies_debuff",
            "applies_enemy_debuff",
        )
        if not debuffs_enemy:
            return ()
        return (
            CombatRelicMarker(
                kind="modify_debuff_application",
                relic_id=relic_id,
                hook=hook,
                amount=200,
                target_id=context.target_id or "enemy",
                metadata={
                    "condition": "first_debuff_card_this_combat",
                    "operation": "multiply_percent",
                    "multiplier_percent": 200,
                    "next_counter": 1,
                },
            ),
        )

    if relic_id == "bone_flute":
        tags = _normalized_values(context.metadata.get("card_tags"))
        if not (
            _metadata_flag(context.metadata, "osty_attack", "is_osty_attack")
            or "ostyattack" in tags
            or "osty_attack" in tags
        ):
            return ()
        return (
            CombatRelicMarker(
                kind="gain_block",
                relic_id=relic_id,
                hook=hook,
                amount=2,
                target_id="player",
                metadata={"condition": "osty_attack"},
            ),
        )

    if relic_id == "daughter_of_the_wind" and card_type == "attack":
        return (
            CombatRelicMarker(
                kind="gain_block",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"condition": "attack_played"},
            ),
        )
    if relic_id == "brilliant_scarf":
        if cards_played_next != 5:
            return ()
        return (
            CombatRelicMarker(
                kind="make_random_card_free_this_turn",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"condition": "fifth_card_played", "card_count": cards_played_next},
            ),
        )
    if relic_id == "iron_club":
        if cards_played_next % 4 != 0:
            return ()
        return (
            CombatRelicMarker(
                kind="draw_cards",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"card_count": cards_played_next, "period": 4},
            ),
        )
    if relic_id == "helical_dart" and card_id == "shiv":
        return (
            CombatRelicMarker(
                kind="gain_status",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"status": "dexterity", "condition": "shiv_played"},
            ),
            CombatRelicMarker(
                kind="gain_status",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"status": "dexterity_down", "condition": "shiv_played"},
            ),
        )
    if relic_id == "intimidating_helmet":
        if _coerce_int(context.metadata.get("card_cost"), default=0) < 2:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_block",
                relic_id=relic_id,
                hook=hook,
                amount=4,
                target_id="player",
                metadata={"condition": "card_cost_at_least_2"},
            ),
        )
    if relic_id == "ivory_tile":
        if _coerce_int(context.metadata.get("card_cost"), default=0) < 3:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_energy",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"condition": "card_cost_at_least_3"},
            ),
        )
    if relic_id == "letter_opener":
        if card_type != "skill":
            return ()
        next_count = _int_context_value(context, "skills_played_this_turn") + 1
        if next_count % 3 != 0:
            return ()
        return (
            CombatRelicMarker(
                kind="all_damage",
                relic_id=relic_id,
                hook=hook,
                amount=5,
                target_id="all_enemies",
                metadata={"skill_count": next_count, "period": 3},
            ),
        )
    if relic_id == "lost_wisp" and card_type == "power":
        return (
            CombatRelicMarker(
                kind="all_damage",
                relic_id=relic_id,
                hook=hook,
                amount=8,
                target_id="all_enemies",
                metadata={"condition": "power_played"},
            ),
        )
    if relic_id == "music_box":
        if card_type != "attack":
            return ()
        attacks_played = _coerce_int(context.metadata.get("attacks_played_this_turn"), default=0)
        if attacks_played > 0:
            return ()
        return (
            CombatRelicMarker(
                kind="add_card_to_hand",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={
                    "selection": "copy_played_card",
                    "copy_source_card_id": card_id,
                    "card_type": "attack",
                    "ethereal": True,
                    "condition": "first_attack_this_turn",
                },
            ),
        )
    if relic_id == "game_piece" and card_type == "power":
        return (
            CombatRelicMarker(
                kind="draw_cards",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"condition": "power_played"},
            ),
        )
    if relic_id == "mummified_hand" and card_type == "power":
        return (
            CombatRelicMarker(
                kind="make_random_card_free_this_turn",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"condition": "power_played"},
                source=STS1_COMPAT_SOURCE,
            ),
        )
    if relic_id == "permafrost":
        if card_type != "power":
            return ()
        if context.relic_counters.get(relic_id, 0) > 0:
            return ()
        powers_played = _coerce_int(
            context.metadata.get("powers_played_this_combat"),
            default=0,
        )
        if powers_played > 0:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_block",
                relic_id=relic_id,
                hook=hook,
                amount=7,
                target_id="player",
                metadata={"condition": "first_power_this_combat", "next_counter": 1},
            ),
        )
    if relic_id == "nunchaku":
        if card_type != "attack":
            return ()
        counter = context.relic_counters.get(relic_id, 0) + 1
        if counter < 10:
            return (
                CombatRelicMarker(
                    kind="relic_counter_changed",
                    relic_id=relic_id,
                    hook=hook,
                    amount=1,
                    metadata={"counter": counter, "period": 10},
                    source=STS1_COMPAT_SOURCE,
                ),
            )
        return (
            CombatRelicMarker(
                kind="gain_energy",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"period": 10, "next_counter": 0},
                source=STS1_COMPAT_SOURCE,
            ),
        )
    if relic_id == "rainbow_ring":
        if card_type not in {"attack", "skill", "power"}:
            return ()
        played_types = _normalized_values(
            _first_present(
                context.metadata,
                "card_types_played_this_turn",
                "played_card_types_this_turn",
            )
        )
        needed = {"attack", "skill", "power"}
        if needed <= played_types:
            return ()
        if not needed <= (set(played_types) | {card_type}):
            return ()
        metadata = {
            "condition": "attack_skill_power_played_this_turn",
            "card_types": tuple(sorted(needed)),
        }
        return (
            CombatRelicMarker(
                kind="gain_status",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={**metadata, "status": "strength"},
            ),
            CombatRelicMarker(
                kind="gain_status",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={**metadata, "status": "dexterity"},
            ),
        )
    if relic_id == "diamond_diadem":
        return (
            CombatRelicMarker(
                kind="conditional_damage_taken_multiplier",
                relic_id=relic_id,
                hook=hook,
                amount=50,
                target_id="player",
                metadata={"condition": "played_2_or_fewer_cards_this_turn"},
            ),
        )
    if relic_id == "tuning_fork":
        if card_type != "skill":
            return ()
        next_count = _int_context_value(context, "skills_played_this_turn") + 1
        if next_count % 10 != 0:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_block",
                relic_id=relic_id,
                hook=hook,
                amount=7,
                target_id="player",
                metadata={"skill_count": next_count, "period": 10},
            ),
        )

    if relic_id == "kusarigama":
        if card_type != "attack":
            return ()
        next_count = _int_context_value(context, "attacks_played_this_turn") + 1
        if next_count % 3 != 0:
            return ()
        return (
            CombatRelicMarker(
                kind="random_damage",
                relic_id=relic_id,
                hook=hook,
                amount=6,
                target_id="random_enemy",
                metadata={"attack_count": next_count, "period": 3},
            ),
        )

    if relic_id not in _ATTACKS_THIS_TURN_RULES:
        return ()
    if card_type != "attack":
        return ()
    played = _int_context_value(context, "attacks_played_this_turn")
    next_count = played + 1
    period, spec = _ATTACKS_THIS_TURN_RULES[relic_id]
    if next_count % period != 0:
        return ()
    return (
        CombatRelicMarker(
            kind=spec.kind,
            relic_id=relic_id,
            hook=hook,
            amount=spec.amount,
            target_id=spec.target_id,
            metadata={**spec.metadata, "attack_count": next_count, "period": period},
            source=spec.source,
        ),
    )


def _resolve_damage_dealt_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    card_type = _normalized_id(context.card_type or "")
    card_id = _normalized_id(context.card_id or "")
    card_name = _normalized_id(str(context.metadata.get("card_name", "")))

    if relic_id == "paper_phrog":
        if context.target_statuses and _status_amount(context.target_statuses, "vulnerable") <= 0:
            return ()
        metadata: dict[str, object] = {
            "condition": "target_vulnerable",
            "normal_multiplier_percent": 150,
            "multiplier_percent": 175,
        }
        _copy_int_metadata(
            metadata,
            context.metadata,
            "base_damage",
            "base_damage",
            "damage",
            "amount",
        )
        return (
            CombatRelicMarker(
                kind="modify_vulnerable_damage_dealt",
                relic_id=relic_id,
                hook=hook,
                amount=175,
                target_id=context.target_id,
                metadata=metadata,
            ),
        )

    if relic_id == "pen_nib":
        if card_type != "attack":
            return ()
        counter = context.relic_counters.get(relic_id)
        if counter is not None:
            next_counter = counter + 1
            if next_counter < 10:
                return (
                    _counter_marker(
                        relic_id,
                        hook,
                        next_counter=next_counter,
                        period=10,
                        source=STS1_COMPAT_SOURCE,
                    ),
                )
            metadata = {
                "condition": "tenth_attack",
                "operation": "multiply_percent",
                "multiplier_percent": 200,
                "period": 10,
                "next_counter": 0,
            }
            _copy_int_metadata(
                metadata,
                context.metadata,
                "base_damage",
                "base_damage",
                "damage",
                "amount",
            )
            return (
                CombatRelicMarker(
                    kind="modify_card_damage",
                    relic_id=relic_id,
                    hook=hook,
                    amount=200,
                    target_id=context.target_id,
                    metadata=metadata,
                    source=STS1_COMPAT_SOURCE,
                ),
            )
        metadata = {
            "condition": "every_10th_attack",
            "operation": "multiply_percent",
            "multiplier_percent": 200,
            "requires": "relic_counter",
        }
        _copy_int_metadata(
            metadata,
            context.metadata,
            "base_damage",
            "base_damage",
            "damage",
            "amount",
        )
        return (
            CombatRelicMarker(
                kind="modify_card_damage",
                relic_id=relic_id,
                hook=hook,
                amount=200,
                target_id=context.target_id,
                metadata=metadata,
                source=STS1_COMPAT_SOURCE,
            ),
        )

    card_tags = _normalized_values(context.metadata.get("card_tags"))
    is_osty_attack = "ostyattack" in card_tags or "osty_attack" in card_tags

    if card_type != "attack" and not (relic_id == "the_boot" and is_osty_attack):
        return ()

    if relic_id == "the_boot":
        metadata = {
            "condition": "unblocked_attack_damage_at_most_4",
            "operation": "minimum",
            "minimum": 5,
            "threshold": 4,
        }
        if "unblocked_damage" in context.metadata:
            unblocked_damage = _coerce_int(context.metadata.get("unblocked_damage"))
            if unblocked_damage <= 0 or unblocked_damage > 4:
                return ()
            metadata["unblocked_damage"] = unblocked_damage
        if is_osty_attack:
            metadata["card_tag"] = "OstyAttack"
        _copy_int_metadata(
            metadata,
            context.metadata,
            "base_damage",
            "base_damage",
            "damage",
            "amount",
        )
        return (
            CombatRelicMarker(
                kind="modify_card_damage",
                relic_id=relic_id,
                hook=hook,
                amount=5,
                target_id=context.target_id,
                metadata=metadata,
                source=STS1_COMPAT_SOURCE,
            ),
        )

    if relic_id == "vitruvian_minion":
        if "minion" not in card_id and "minion" not in card_name:
            return ()
        metadata = {
            "condition": "card_contains_minion",
            "operation": "multiply_percent",
            "multiplier_percent": 200,
        }
        _copy_int_metadata(
            metadata,
            context.metadata,
            "base_damage",
            "base_damage",
            "damage",
            "amount",
        )
        return (
            CombatRelicMarker(
                kind="modify_card_damage",
                relic_id=relic_id,
                hook=hook,
                amount=200,
                target_id=context.target_id,
                metadata=metadata,
            ),
        )

    if relic_id in {"strike_dummy", "fake_strike_dummy"}:
        if "strike" not in card_id and "strike" not in card_name:
            return ()
        amount = 1 if relic_id == "fake_strike_dummy" else 3
        metadata = {"condition": "card_contains_strike", "operation": "add"}
        _copy_int_metadata(
            metadata,
            context.metadata,
            "base_damage",
            "base_damage",
            "damage",
            "amount",
        )
        return (
            CombatRelicMarker(
                kind="modify_card_damage",
                relic_id=relic_id,
                hook=hook,
                amount=amount,
                target_id=context.target_id,
                metadata=metadata,
            ),
        )

    if relic_id == "miniature_cannon":
        if not bool(context.metadata.get("upgraded", False)):
            return ()
        metadata = {"condition": "upgraded_attack", "operation": "add"}
        _copy_int_metadata(
            metadata,
            context.metadata,
            "base_damage",
            "base_damage",
            "damage",
            "amount",
        )
        return (
            CombatRelicMarker(
                kind="modify_card_damage",
                relic_id=relic_id,
                hook=hook,
                amount=3,
                target_id=context.target_id,
                metadata=metadata,
            ),
        )

    if relic_id == "mystic_lighter":
        if not bool(context.metadata.get("enchanted", False)):
            return ()
        metadata = {"condition": "enchanted_attack", "operation": "add"}
        _copy_int_metadata(
            metadata,
            context.metadata,
            "base_damage",
            "base_damage",
            "damage",
            "amount",
        )
        return (
            CombatRelicMarker(
                kind="modify_card_damage",
                relic_id=relic_id,
                hook=hook,
                amount=9,
                target_id=context.target_id,
                metadata=metadata,
            ),
        )

    return ()


def _resolve_damage_taken_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id == "lizard_tail":
        if context.relic_counters.get(relic_id, 0) > 0:
            return ()
        if context.player_hp is None or context.player_max_hp is None:
            return ()
        hp_loss = _coerce_int(context.metadata.get("hp_loss"), default=0)
        if hp_loss < context.player_hp:
            return ()
        target_hp = max(1, context.player_max_hp // 2)
        return (
            CombatRelicMarker(
                kind="heal_player",
                relic_id=relic_id,
                hook=hook,
                amount=target_hp,
                target_id="player",
                metadata={
                    "condition": "fatal_damage",
                    "target_hp": target_hp,
                    "hp_loss": hp_loss,
                    "next_counter": 1,
                },
                source=STS1_COMPAT_SOURCE,
            ),
        )

    if relic_id == "red_skull":
        if context.player_hp is None or context.player_max_hp is None:
            return ()
        hp_loss = _coerce_int(context.metadata.get("hp_loss"), default=0)
        hp_after_loss = max(0, context.player_hp - hp_loss)
        if _player_hp_at_or_below_half(context.player_hp, context.player_max_hp):
            return ()
        if not _player_hp_at_or_below_half(hp_after_loss, context.player_max_hp):
            return ()
        return (
            CombatRelicMarker(
                kind="gain_status",
                relic_id=relic_id,
                hook=hook,
                amount=3,
                target_id="player",
                metadata={
                    "status": "strength",
                    "condition": "hp_crossed_to_at_or_below_50_percent",
                    "hp_after_loss": hp_after_loss,
                },
            ),
        )

    if relic_id == "centennial_puzzle":
        if context.relic_counters.get(relic_id, 0) > 0:
            return ()
        hp_loss = _coerce_int(context.metadata.get("hp_loss"), default=1)
        if hp_loss <= 0:
            return ()
        return (
            CombatRelicMarker(
                kind="draw_cards",
                relic_id=relic_id,
                hook=hook,
                amount=3,
                target_id="player",
                metadata={"condition": "first_hp_loss_this_combat", "next_counter": 1},
                source=STS1_COMPAT_SOURCE,
            ),
        )

    if relic_id == "demon_tongue":
        if context.relic_counters.get(relic_id, 0) > 0:
            return ()
        turn_owner = context.metadata.get("turn_owner")
        if turn_owner is not None and _normalized_id(turn_owner) not in {"player", "you"}:
            return ()
        if context.metadata.get("on_player_turn") is False:
            return ()
        hp_loss = _coerce_int(context.metadata.get("hp_loss"), default=1)
        heal = _capped_heal(hp_loss, hp=context.player_hp, max_hp=context.player_max_hp)
        if heal <= 0:
            return ()
        return (
            CombatRelicMarker(
                kind="heal_player",
                relic_id=relic_id,
                hook=hook,
                amount=heal,
                target_id="player",
                metadata={
                    "condition": "first_hp_loss_on_player_turn",
                    "hp_loss": hp_loss,
                    "next_counter": 1,
                },
            ),
        )

    if relic_id == "undying_sigil":
        doom = _status_amount(context.target_statuses, "doom")
        if doom <= 0:
            return ()
        attacker_hp = _coerce_int(
            context.metadata.get("attacker_hp", context.metadata.get("target_hp")),
            default=-1,
        )
        if attacker_hp > 0 and doom < attacker_hp:
            return ()
        metadata: dict[str, object] = {
            "condition": "attacker_doom_at_least_hp",
            "multiplier_percent": 50,
        }
        _copy_int_metadata(
            metadata,
            context.metadata,
            "incoming_hp_loss",
            "hp_loss",
            "incoming_hp_loss",
            "damage",
        )
        if attacker_hp > 0:
            metadata["attacker_hp"] = attacker_hp
        metadata["doom"] = doom
        return (
            CombatRelicMarker(
                kind="modify_damage_taken",
                relic_id=relic_id,
                hook=hook,
                amount=50,
                target_id="player",
                metadata=metadata,
            ),
        )

    if relic_id == "odd_mushroom":
        if context.player_statuses and _status_amount(context.player_statuses, "vulnerable") <= 0:
            return ()
        metadata = {
            "condition": "player_vulnerable",
            "normal_multiplier_percent": 150,
            "multiplier_percent": 125,
        }
        _copy_int_metadata(
            metadata,
            context.metadata,
            "incoming_hp_loss",
            "hp_loss",
            "incoming_hp_loss",
            "damage",
        )
        return (
            CombatRelicMarker(
                kind="modify_vulnerable_damage_taken",
                relic_id=relic_id,
                hook=hook,
                amount=125,
                target_id="player",
                metadata=metadata,
                source=STS1_COMPAT_SOURCE,
            ),
        )

    if relic_id == "paper_krane":
        if context.target_statuses and _status_amount(context.target_statuses, "weak") <= 0:
            return ()
        metadata = {
            "condition": "attacker_weak",
            "normal_multiplier_percent": 75,
            "multiplier_percent": 60,
        }
        _copy_int_metadata(
            metadata,
            context.metadata,
            "incoming_hp_loss",
            "hp_loss",
            "incoming_hp_loss",
            "damage",
        )
        return (
            CombatRelicMarker(
                kind="modify_weak_damage_taken",
                relic_id=relic_id,
                hook=hook,
                amount=60,
                target_id="player",
                metadata=metadata,
                source=STS1_COMPAT_SOURCE,
            ),
        )

    if relic_id == "tungsten_rod":
        metadata = {
            "condition": "would_lose_hp",
            "operation": "subtract",
            "reduction": 1,
        }
        _copy_int_metadata(
            metadata,
            context.metadata,
            "incoming_hp_loss",
            "hp_loss",
            "incoming_hp_loss",
            "damage",
        )
        return (
            CombatRelicMarker(
                kind="reduce_hp_loss",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata=metadata,
                source=STS1_COMPAT_SOURCE,
            ),
        )

    if relic_id == "beating_remnant":
        metadata = {
            "condition": "hp_loss_cap_per_turn",
            "operation": "cap_per_turn",
            "cap": 20,
        }
        _copy_int_metadata(
            metadata,
            context.metadata,
            "incoming_hp_loss",
            "hp_loss",
            "incoming_hp_loss",
            "damage",
        )
        _copy_int_metadata(
            metadata,
            context.metadata,
            "hp_lost_this_turn",
            "hp_lost_this_turn",
            "damage_taken_this_turn",
        )
        return (
            CombatRelicMarker(
                kind="cap_hp_loss_per_turn",
                relic_id=relic_id,
                hook=hook,
                amount=20,
                target_id="player",
                metadata=metadata,
            ),
        )

    if relic_id == "self_forming_clay":
        hp_loss = _coerce_int(context.metadata.get("hp_loss"), default=1)
        if hp_loss <= 0:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_status",
                relic_id=relic_id,
                hook=hook,
                amount=3,
                target_id="player",
                metadata={"status": "next_turn_block", "condition": "lost_hp"},
                source=STS1_COMPAT_SOURCE,
            ),
        )

    if relic_id == "diamond_diadem":
        cards_played = _coerce_int(context.metadata.get("cards_played_this_turn"), default=0)
        if cards_played > 2:
            return ()
        metadata = {"condition": "played_2_or_fewer_cards_this_turn"}
        _copy_int_metadata(
            metadata,
            context.metadata,
            "incoming_hp_loss",
            "hp_loss",
            "incoming_hp_loss",
            "damage",
        )
        return (
            CombatRelicMarker(
                kind="modify_damage_taken",
                relic_id=relic_id,
                hook=hook,
                amount=50,
                target_id="player",
                metadata=metadata,
            ),
        )

    return ()


def _resolve_monster_killed_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id == "book_repair_knife":
        death_reason = _normalized_id(
            str(_first_present(context.metadata, "death_reason", "kill_source") or "")
        )
        if death_reason != "doom":
            return ()
        if _metadata_flag(context.metadata, "target_is_minion", "is_minion"):
            return ()
        target_type = _normalized_id(str(context.metadata.get("target_type", "")))
        if target_type == "minion":
            return ()
        heal = _capped_heal(3, hp=context.player_hp, max_hp=context.player_max_hp)
        if heal <= 0:
            return ()
        return (
            CombatRelicMarker(
                kind="heal_player",
                relic_id=relic_id,
                hook=hook,
                amount=heal,
                target_id="player",
                metadata={"condition": "non_minion_enemy_died_to_doom"},
            ),
        )

    if relic_id == "gremlin_horn":
        return (
            CombatRelicMarker(
                kind="gain_energy",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"condition": "enemy_killed"},
                source=STS1_COMPAT_SOURCE,
            ),
            CombatRelicMarker(
                kind="draw_cards",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={"condition": "enemy_killed"},
                source=STS1_COMPAT_SOURCE,
            ),
        )

    if relic_id == "sword_of_stone":
        if not _elite_killed(context):
            return ()
        next_counter = context.relic_counters.get(relic_id, 0) + 1
        metadata: dict[str, object] = {
            "counter": min(next_counter, 5),
            "period": 5,
            "condition": "elite_killed",
        }
        if next_counter >= 5:
            metadata["transform_ready"] = True
        return (
            CombatRelicMarker(
                kind="relic_counter_changed",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata=metadata,
            ),
        )

    if relic_id == "war_hammer":
        if not _elite_killed(context):
            return ()
        return (
            CombatRelicMarker(
                kind="upgrade_deck_cards",
                relic_id=relic_id,
                hook=hook,
                amount=4,
                target_id="player",
                metadata={"selection": "random", "condition": "elite_killed"},
            ),
        )

    if relic_id != "black_star":
        return ()
    if context.encounter_type is not None and _normalized_id(context.encounter_type) != "elite":
        return ()
    return (
        CombatRelicMarker(
            kind="reward_relic_count_delta",
            relic_id=relic_id,
            hook=hook,
            amount=1,
            target_id="reward",
            metadata={"condition": "elite_defeated"},
            source=STS1_COMPAT_SOURCE,
        ),
    )


def _resolve_combat_end_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id == "lava_lamp":
        damage_taken = _coerce_int(
            _first_present(
                context.metadata,
                "damage_taken_this_combat",
                "hp_lost_this_combat",
                "damage_taken",
            ),
            default=0,
        )
        if damage_taken > 0:
            return ()
        return (
            CombatRelicMarker(
                kind="upgrade_card_rewards",
                relic_id=relic_id,
                hook=hook,
                target_id="reward",
                metadata={"condition": "no_damage_taken_this_combat"},
            ),
        )

    if relic_id == "paels_tooth":
        removed_count = context.metadata.get("paels_tooth_removed_count")
        if removed_count is not None and _coerce_int(removed_count) <= 0:
            return ()
        return (
            CombatRelicMarker(
                kind="add_deck_cards",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata={
                    "selection": "random_removed_by_relic",
                    "upgraded": True,
                    "condition": "after_combat",
                },
            ),
        )

    if relic_id != "meat_on_the_bone":
        return ()
    if context.player_hp is None or context.player_max_hp is None:
        return ()
    if context.player_hp * 2 > context.player_max_hp:
        return ()
    return (
        CombatRelicMarker(
            kind="heal_player",
            relic_id=relic_id,
            hook=hook,
            amount=12,
            target_id="player",
            metadata={"condition": "hp_at_or_below_50_percent"},
            source=STS1_COMPAT_SOURCE,
        ),
    )


def _resolve_card_exhausted_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    card_type = _normalized_id(context.card_type or "")
    card_id = _normalized_id(context.card_id or "")

    if relic_id == "burning_sticks":
        if card_type != "skill":
            return ()
        if context.relic_counters.get(relic_id, 0) > 0:
            return ()
        if _coerce_int(context.metadata.get("skills_exhausted_this_combat"), default=0) > 0:
            return ()
        metadata: dict[str, object] = {
            "selection": "copy_exhausted_card",
            "card_type": "skill",
            "condition": "first_skill_exhausted_this_combat",
            "next_counter": 1,
        }
        if card_id:
            metadata["copy_source_card_id"] = card_id
        return (
            CombatRelicMarker(
                kind="add_card_to_hand",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="player",
                metadata=metadata,
            ),
        )

    if relic_id == "charons_ashes":
        return (
            CombatRelicMarker(
                kind="all_damage",
                relic_id=relic_id,
                hook=hook,
                amount=3,
                target_id="all_enemies",
                metadata={"condition": "card_exhausted"},
                source=STS1_COMPAT_SOURCE,
            ),
        )

    if relic_id == "forgotten_soul":
        return (
            CombatRelicMarker(
                kind="random_damage",
                relic_id=relic_id,
                hook=hook,
                amount=1,
                target_id="random_enemy",
                metadata={"condition": "card_exhausted"},
            ),
        )

    if relic_id != "joss_paper":
        return ()

    counter = context.relic_counters.get(relic_id)
    if counter is None:
        counter = _coerce_int(
            _first_present(
                context.metadata,
                "cards_exhausted_since_last_joss_paper",
                "joss_paper_counter",
            ),
            default=0,
        )
    next_counter = counter + 1
    if next_counter < 5:
        return (
            _counter_marker(
                relic_id,
                hook,
                next_counter=next_counter,
                period=5,
                source=PROVISIONAL_STS2_SOURCE,
            ),
        )
    return (
        CombatRelicMarker(
            kind="draw_cards",
            relic_id=relic_id,
            hook=hook,
            amount=1,
            target_id="player",
            metadata={"condition": "fifth_card_exhausted", "period": 5, "next_counter": 0},
        ),
    )


def _resolve_card_discarded_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    turn_owner = context.metadata.get("turn_owner")
    if turn_owner is not None and _normalized_id(turn_owner) not in {"player", "you"}:
        return ()
    if context.metadata.get("on_player_turn") is False:
        return ()

    discarded_count = max(
        1,
        _coerce_int(
            _first_present(
                context.metadata,
                "discarded_count",
                "cards_discarded",
                "card_count",
            ),
            default=1,
        ),
    )

    if relic_id == "tingsha":
        return (
            CombatRelicMarker(
                kind="random_damage",
                relic_id=relic_id,
                hook=hook,
                amount=3 * discarded_count,
                target_id="random_enemy",
                metadata={
                    "condition": "discarded_during_player_turn",
                    "discarded_count": discarded_count,
                    "per_card": 3,
                },
                source=STS1_COMPAT_SOURCE,
            ),
        )

    if relic_id != "tough_bandages":
        return ()
    return (
        CombatRelicMarker(
            kind="gain_block",
            relic_id=relic_id,
            hook=hook,
            amount=3 * discarded_count,
            target_id="player",
            metadata={
                "condition": "discarded_during_player_turn",
                "discarded_count": discarded_count,
                "per_card": 3,
            },
            source=STS1_COMPAT_SOURCE,
        ),
    )


def _resolve_card_created_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id != "regalite":
        return ()
    created_count = max(1, _coerce_int(context.metadata.get("created_count"), default=1))
    return (
        CombatRelicMarker(
            kind="gain_block",
            relic_id=relic_id,
            hook=hook,
            amount=2 * created_count,
            target_id="player",
            metadata={"condition": "card_created", "created_count": created_count, "per_card": 2},
        ),
    )


def _resolve_card_block_gained_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id == "paels_legion":
        if context.relic_counters.get(relic_id, 0) > 0:
            return ()
        return (
            CombatRelicMarker(
                kind="modify_card_block",
                relic_id=relic_id,
                hook=hook,
                amount=200,
                target_id="player",
                metadata={
                    "condition": "card_block_gain",
                    "operation": "multiply_percent",
                    "multiplier_percent": 200,
                    "sleep_turns": 2,
                    "next_counter": 2,
                    "base_block": _coerce_int(context.metadata.get("amount"), default=0),
                },
            ),
        )

    if relic_id != "vambrace":
        return ()
    if context.relic_counters.get(relic_id, 0) > 0:
        return ()
    return (
        CombatRelicMarker(
            kind="modify_card_block",
            relic_id=relic_id,
            hook=hook,
            amount=200,
            target_id="player",
            metadata={
                "condition": "first_card_block_gain_this_combat",
                "operation": "multiply_percent",
                "multiplier_percent": 200,
                "next_counter": 1,
                "base_block": _coerce_int(context.metadata.get("amount"), default=0),
            },
        ),
    )


def _resolve_draw_pile_shuffled_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id != "the_abacus":
        return ()
    return (
        CombatRelicMarker(
            kind="gain_block",
            relic_id=relic_id,
            hook=hook,
            amount=6,
            target_id="player",
            metadata={"condition": "draw_pile_shuffled"},
            source=STS1_COMPAT_SOURCE,
        ),
    )


def _resolve_enemy_block_broken_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id != "hand_drill":
        return ()
    return (
        CombatRelicMarker(
            kind="apply_status",
            relic_id=relic_id,
            hook=hook,
            amount=2,
            target_id=context.target_id or "enemy",
            metadata={"status": "vulnerable", "condition": "enemy_block_broken"},
            source=STS1_COMPAT_SOURCE,
        ),
    )


def _resolve_hand_empty_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id != "unceasing_top":
        return ()
    turn_owner = context.metadata.get("turn_owner")
    if turn_owner is not None and _normalized_id(turn_owner) not in {"player", "you"}:
        return ()
    if context.metadata.get("on_player_turn") is False:
        return ()
    if _coerce_int(context.metadata.get("hand_size"), default=0) > 0:
        return ()
    return (
        CombatRelicMarker(
            kind="draw_cards",
            relic_id=relic_id,
            hook=hook,
            amount=1,
            target_id="player",
            metadata={"condition": "empty_hand_during_player_turn"},
            source=STS1_COMPAT_SOURCE,
        ),
    )


def _resolve_orb_channeled_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id != "metronome":
        return ()
    if _metadata_flag(context.metadata, "metronome_triggered", "already_triggered"):
        return ()

    prior_count = context.relic_counters.get(relic_id)
    if prior_count is None:
        prior_count = _coerce_int(
            _first_present(
                context.metadata,
                "orbs_channeled_this_combat",
                "orb_channel_count",
            ),
            default=0,
        )
    channeled_count = max(1, _coerce_int(context.metadata.get("channeled_count"), default=1))
    next_count = prior_count + channeled_count
    if next_count < 7:
        return (
            _counter_marker(
                relic_id,
                hook,
                next_counter=next_count,
                period=7,
                source=PROVISIONAL_STS2_SOURCE,
            ),
        )
    return (
        CombatRelicMarker(
            kind="all_damage",
            relic_id=relic_id,
            hook=hook,
            amount=30,
            target_id="all_enemies",
            metadata={
                "condition": "first_7_orbs_channeled_this_combat",
                "period": 7,
                "next_counter": 7,
            },
        ),
    )


def _resolve_orb_passive_triggered_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id != "gold_plated_cables":
        return ()
    return (
        CombatRelicMarker(
            kind="trigger_orb_passive",
            relic_id=relic_id,
            hook=hook,
            amount=1,
            target_id="player",
            metadata={"selector": "rightmost", "condition": "rightmost_orb_passive"},
            source=STS1_COMPAT_SOURCE,
        ),
    )


def _resolve_potion_used_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id != "reptile_trinket":
        return ()
    return (
        CombatRelicMarker(
            kind="gain_status",
            relic_id=relic_id,
            hook=hook,
            amount=3,
            target_id="player",
            metadata={"status": "strength", "condition": "potion_used"},
        ),
        CombatRelicMarker(
            kind="gain_status",
            relic_id=relic_id,
            hook=hook,
            amount=3,
            target_id="player",
            metadata={"status": "strength_down", "condition": "potion_used"},
        ),
    )


def _resolve_resource_spent_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    resource = _normalized_id(
        str(_first_present(context.metadata, "resource", "resource_id") or "star")
    )
    if resource not in {"star", "stars"}:
        return ()
    amount_spent = _coerce_int(
        _first_present(
            context.metadata,
            "amount_spent",
            "resource_amount",
            "stars_spent",
        ),
        default=1,
    )
    if amount_spent <= 0:
        return ()

    if relic_id == "galactic_dust":
        block_triggers = amount_spent // 10
        if block_triggers <= 0:
            return ()
        return (
            CombatRelicMarker(
                kind="gain_block",
                relic_id=relic_id,
                hook=hook,
                amount=10 * block_triggers,
                target_id="player",
                metadata={
                    "condition": "stars_spent",
                    "stars_spent": amount_spent,
                    "period": 10,
                    "block_per_period": 10,
                },
            ),
        )

    if relic_id != "mini_regent":
        return ()
    if context.relic_counters.get(relic_id, 0) > 0:
        return ()
    if _coerce_int(context.metadata.get("stars_spent_this_turn"), default=0) > 0:
        return ()
    return (
        CombatRelicMarker(
            kind="gain_status",
            relic_id=relic_id,
            hook=hook,
            amount=1,
            target_id="player",
            metadata={"status": "strength", "condition": "first_star_spent_this_turn"},
        ),
    )


def _resolve_status_applied_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id != "snecko_skull":
        return ()
    status = _normalized_id(str(_first_present(context.metadata, "status", "status_id") or ""))
    if status != "poison":
        return ()
    return (
        CombatRelicMarker(
            kind="apply_status",
            relic_id=relic_id,
            hook=hook,
            amount=1,
            target_id=context.target_id or "enemy",
            metadata={"status": "poison", "condition": "poison_applied"},
            source=STS1_COMPAT_SOURCE,
        ),
    )


def _resolve_status_gained_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id != "ruined_helmet":
        return ()
    if context.relic_counters.get(relic_id, 0) > 0:
        return ()
    status = _normalized_id(str(_first_present(context.metadata, "status", "status_id") or ""))
    if status != "strength":
        return ()
    return (
        CombatRelicMarker(
            kind="modify_status_gain",
            relic_id=relic_id,
            hook=hook,
            amount=200,
            target_id="player",
            metadata={
                "status": "strength",
                "condition": "first_strength_gain_this_combat",
                "operation": "multiply_percent",
                "multiplier_percent": 200,
                "next_counter": 1,
                "base_status_amount": _coerce_int(context.metadata.get("amount"), default=0),
            },
        ),
    )


def _marker_from_spec(
    relic_id: str,
    hook: CombatRelicHook,
    spec: CombatRelicMarkerSpec,
    context: CombatRelicContext,
    *,
    hp: int | None,
) -> CombatRelicMarker:
    amount = spec.amount
    if spec.kind == "heal_player" and amount is not None:
        amount = _capped_heal(amount, hp=hp, max_hp=context.player_max_hp)
    return CombatRelicMarker(
        kind=spec.kind,
        relic_id=relic_id,
        hook=hook,
        amount=amount,
        target_id=spec.target_id,
        metadata=spec.metadata,
        source=spec.source,
    )


def _counter_marker(
    relic_id: str,
    hook: CombatRelicHook,
    *,
    next_counter: int,
    period: int,
    source: SourceRef,
) -> CombatRelicMarker:
    return CombatRelicMarker(
        kind="relic_counter_changed",
        relic_id=relic_id,
        hook=hook,
        amount=1,
        metadata={"counter": next_counter, "period": period},
        source=source,
    )


def _resolution_from_markers(
    hook: CombatRelicHook,
    markers: tuple[CombatRelicMarker, ...],
    blockers: tuple[CombatRelicBlocker, ...],
) -> CombatRelicResolution:
    return CombatRelicResolution(
        hook=hook,
        markers=markers,
        blockers=blockers,
        hp_delta=sum(marker.amount or 0 for marker in markers if marker.kind == "heal_player"),
        block_delta=sum(marker.amount or 0 for marker in markers if marker.kind == "gain_block"),
        energy_delta=sum(marker.amount or 0 for marker in markers if marker.kind == "gain_energy"),
    )


def _advance_hp(
    current_hp: int | None,
    markers: Sequence[CombatRelicMarker],
) -> int | None:
    if current_hp is None:
        return None
    return current_hp + sum(
        marker.amount or 0 for marker in markers if marker.kind == "heal_player"
    )


def _capped_heal(amount: int, *, hp: int | None, max_hp: int | None) -> int:
    if amount <= 0:
        return 0
    if hp is None or max_hp is None:
        return amount
    return max(0, min(amount, max_hp - hp))


def _int_context_value(context: CombatRelicContext, key: str) -> int:
    return _coerce_int(context.metadata.get(key, 0))


def _status_amount(statuses: Mapping[str, object], status_id: str) -> int:
    for key, value in statuses.items():
        if _normalized_id(str(key)) != status_id:
            continue
        return _coerce_int(value)
    return 0


def _metadata_flag(metadata: Mapping[str, object], *keys: str) -> bool:
    for key in keys:
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, str):
            return _normalized_id(value) in {"1", "true", "yes", "y", "on"}
        return bool(value)
    return False


def _copy_int_metadata(
    output: dict[str, object],
    source: Mapping[str, object],
    output_key: str,
    *source_keys: str,
) -> None:
    value = _first_present(source, *source_keys)
    if value is not None:
        output[output_key] = _coerce_int(value)


def _normalized_values(value: object) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset({_normalized_id(value)})
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return frozenset(_normalized_id(item) for item in value)
    return frozenset({_normalized_id(value)})


def _player_hp_at_or_below_half(hp: int | None, max_hp: int | None) -> bool:
    return hp is not None and max_hp is not None and hp * 2 <= max_hp


def _elite_killed(context: CombatRelicContext) -> bool:
    if context.encounter_type is not None and _normalized_id(context.encounter_type) == "elite":
        return True
    return (
        _normalized_id(str(context.metadata.get("encounter_type", ""))) == "elite"
        or _normalized_id(str(context.metadata.get("target_type", ""))) == "elite"
        or _metadata_flag(context.metadata, "target_is_elite", "is_elite")
    )


def _coerce_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _combat_hook(hook: CombatRelicHook | str) -> CombatRelicHook:
    if isinstance(hook, CombatRelicHook):
        return hook
    return CombatRelicHook(_normalized_id(hook))


def _unique_relic_inputs(relics: Sequence[CombatRelicInput]) -> tuple[CombatRelicInput, ...]:
    seen: set[str] = set()
    unique: list[CombatRelicInput] = []
    for relic in relics:
        relic_id = combat_relic_content_id(relic)
        if relic_id in seen:
            continue
        seen.add(relic_id)
        unique.append(relic)
    return tuple(unique)


def _content_str(item: CombatRelicInput, *keys: str) -> str | None:
    if isinstance(item, str):
        return None
    value = _first_present(item, *keys)
    return None if value is None else str(value)


def _first_present(item: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        value = item.get(key)
        if value is not None:
            return value
    return None


def _clean_description(description: str) -> str:
    return (
        description.replace("[gold]", "")
        .replace("[/gold]", "")
        .replace("[blue]", "")
        .replace("[/blue]", "")
        .replace("[green]", "")
        .replace("[/green]", "")
        .replace("[red]", "")
        .replace("[/red]", "")
        .lower()
    )


def _normalized_id(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("'", "")
        .replace(".", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _status_spec(status: str, amount: int, target: str = "player") -> CombatRelicMarkerSpec:
    return CombatRelicMarkerSpec(
        "gain_status",
        amount=amount,
        target_id=target,
        metadata={"status": status},
    )


def _resource_spec(resource: str, amount: int) -> CombatRelicMarkerSpec:
    return CombatRelicMarkerSpec(
        "player_resource",
        amount=amount,
        target_id="player",
        metadata={"resource": resource},
    )


_STATIC_MARKERS_BY_HOOK: Mapping[
    CombatRelicHook,
    Mapping[str, tuple[CombatRelicMarkerSpec, ...]],
] = {
    CombatRelicHook.START_OF_COMBAT: {
        "akabeko": (_status_spec("vigor", 8),),
        "anchor": (CombatRelicMarkerSpec("gain_block", amount=10, target_id="player"),),
        "fake_anchor": (CombatRelicMarkerSpec("gain_block", amount=4, target_id="player"),),
        "bag_of_preparation": (
            CombatRelicMarkerSpec("draw_cards", amount=2, target_id="player"),
        ),
        "big_mushroom": (
            CombatRelicMarkerSpec(
                "draw_cards",
                amount=-2,
                target_id="player",
                metadata={"mode": "opening_draw_reduction"},
            ),
        ),
        "big_hat": (
            CombatRelicMarkerSpec(
                "add_card_to_hand",
                amount=2,
                target_id="player",
                metadata={"selection": "random", "keyword": "ethereal", "ethereal": True},
            ),
        ),
        "blood_vial": (CombatRelicMarkerSpec("heal_player", amount=2, target_id="player"),),
        "fake_blood_vial": (
            CombatRelicMarkerSpec("heal_player", amount=1, target_id="player"),
        ),
        "gambling_chip": (
            CombatRelicMarkerSpec(
                "opening_hand_discard_redraw",
                target_id="player",
                metadata={"selection": "any_number", "draw_equal_to_discarded": True},
            ),
        ),
        "choices_paradox": (
            CombatRelicMarkerSpec(
                "add_card_to_hand",
                amount=1,
                target_id="player",
                metadata={
                    "selection": "choose_one_of_random",
                    "choice_count": 5,
                    "retain_once": True,
                },
            ),
        ),
        "cracked_core": (
            CombatRelicMarkerSpec(
                "channel_orb",
                amount=1,
                target_id="player",
                metadata={"orb": "lightning"},
            ),
        ),
        "data_disk": (_status_spec("focus", 1),),
        "divine_destiny": (_resource_spec("star", 6),),
        "divine_right": (_resource_spec("star", 3),),
        "fencing_manual": (_resource_spec("forge", 10),),
        "festive_popper": (
            CombatRelicMarkerSpec("all_damage", amount=9, target_id="all_enemies"),
        ),
        "gorget": (_status_spec("plated_armor", 4),),
        "infused_core": (
            CombatRelicMarkerSpec(
                "channel_orb",
                amount=3,
                target_id="player",
                metadata={"orb": "lightning", "lightning_damage_bonus": 1},
            ),
        ),
        "lantern": (CombatRelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
        "orange_dough": (
            CombatRelicMarkerSpec(
                "add_card_to_hand",
                amount=2,
                target_id="player",
                metadata={"selection": "random", "card_pool": "colorless"},
            ),
        ),
        "philosophers_stone": (
            CombatRelicMarkerSpec(
                "apply_status",
                amount=1,
                target_id="all_enemies",
                metadata={"status": "strength"},
            ),
        ),
        "phylactery_unbound": (_resource_spec("summon", 5),),
        "petrified_toad": (
            CombatRelicMarkerSpec(
                "procure_potion",
                amount=1,
                target_id="player",
                metadata={"potion_id": "potion_shaped_rock"},
            ),
        ),
        "ring_of_the_snake": (
            CombatRelicMarkerSpec("draw_cards", amount=2, target_id="player"),
        ),
        "runic_capacitor": (
            CombatRelicMarkerSpec("orb_slot_delta", amount=3, target_id="player"),
        ),
        "snecko_eye": (_status_spec("confused", 1),),
        "fake_snecko_eye": (_status_spec("confused", 1),),
        "funerary_mask": (
            CombatRelicMarkerSpec(
                "add_card_to_draw_pile",
                amount=3,
                target_id="player",
                metadata={"card_id": "soul", "card_type": "skill", "target": "self"},
            ),
        ),
        "ninja_scroll": (
            CombatRelicMarkerSpec(
                "add_card_to_hand",
                amount=3,
                target_id="player",
                metadata={"card_id": "shiv", "card_type": "attack", "target": "enemy"},
            ),
        ),
        "symbiotic_virus": (
            CombatRelicMarkerSpec(
                "channel_orb",
                amount=1,
                target_id="player",
                metadata={"orb": "dark"},
            ),
        ),
        "twisted_funnel": (
            CombatRelicMarkerSpec(
                "apply_status",
                amount=4,
                target_id="all_enemies",
                metadata={"status": "poison"},
            ),
        ),
        "toolbox": (
            CombatRelicMarkerSpec(
                "add_card_to_hand",
                amount=1,
                target_id="player",
                metadata={
                    "selection": "choose_one_of_random",
                    "choice_count": 3,
                    "card_pool": "colorless",
                },
                source=STS1_COMPAT_SOURCE,
            ),
        ),
        "vajra": (_status_spec("strength", 1),),
        "vexing_puzzlebox": (
            CombatRelicMarkerSpec(
                "add_card_to_hand",
                amount=1,
                target_id="player",
                metadata={"selection": "random", "free_to_play_this_turn": True},
            ),
        ),
        "sword_of_jade": (_status_spec("strength", 3),),
        "oddly_smooth_stone": (_status_spec("dexterity", 1),),
        "bronze_scales": (_status_spec("thorns", 3),),
        "bag_of_marbles": (
            CombatRelicMarkerSpec(
                "apply_status",
                amount=1,
                target_id="all_enemies",
                metadata={"status": "vulnerable"},
            ),
        ),
        "red_mask": (
            CombatRelicMarkerSpec(
                "apply_status",
                amount=1,
                target_id="all_enemies",
                metadata={"status": "weak"},
            ),
        ),
        "jeweled_mask": (
            CombatRelicMarkerSpec(
                "move_card_type_from_draw_to_hand",
                amount=1,
                target_id="player",
                metadata={"card_type": "power", "free_to_play_this_turn": True},
            ),
        ),
        "power_cell": (
            CombatRelicMarkerSpec(
                "move_zero_cost_cards_to_hand",
                amount=2,
                target_id="player",
                metadata={"free_to_play_this_turn": True},
            ),
        ),
        "radiant_pearl": (
            CombatRelicMarkerSpec(
                "add_card_to_hand",
                amount=1,
                target_id="player",
                metadata={
                    "card_id": "luminesce",
                    "name": "Luminesce",
                    "card_type": "skill",
                    "target": "self",
                    "cost": 0,
                    "exhaust": True,
                },
            ),
        ),
        "stone_cracker": (
            CombatRelicMarkerSpec(
                "upgrade_draw_pile_cards",
                amount=2,
                target_id="player",
                metadata={"mode": "combat_only"},
            ),
        ),
        "blessed_antler": (
            CombatRelicMarkerSpec(
                "shuffle_status_into_draw_pile",
                amount=3,
                target_id="player",
                metadata={"card_id": "dazed", "card_type": "status", "target": "self"},
            ),
        ),
        "royal_poison": (
            CombatRelicMarkerSpec("lose_hp", amount=4, target_id="player"),
        ),
        "very_hot_cocoa": (
            CombatRelicMarkerSpec("gain_energy", amount=4, target_id="player"),
        ),
    },
    CombatRelicHook.TURN_START: {
        "blessed_antler": (CombatRelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
        "bound_phylactery": (_resource_spec("summon", 1),),
        "blood_soaked_rose": (
            CombatRelicMarkerSpec("gain_energy", amount=1, target_id="player"),
        ),
        "crossbow": (
            CombatRelicMarkerSpec(
                "add_card_to_hand",
                amount=1,
                target_id="player",
                metadata={
                    "selection": "random",
                    "card_type": "attack",
                    "free_to_play_this_turn": True,
                },
            ),
        ),
        "brimstone": (
            _status_spec("strength", 2),
            CombatRelicMarkerSpec(
                "apply_status",
                amount=1,
                target_id="all_enemies",
                metadata={"status": "strength"},
            ),
        ),
        "ectoplasm": (CombatRelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
        "fiddle": (
            CombatRelicMarkerSpec("draw_cards", amount=2, target_id="player"),
            _status_spec("no_draw_this_turn", 1),
        ),
        "mercury_hourglass": (
            CombatRelicMarkerSpec("all_damage", amount=3, target_id="all_enemies"),
        ),
        "paels_blood": (CombatRelicMarkerSpec("draw_cards", amount=1, target_id="player"),),
        "philosophers_stone": (
            CombatRelicMarkerSpec("gain_energy", amount=1, target_id="player"),
        ),
        "phylactery_unbound": (_resource_spec("summon", 1),),
        "prismatic_gem": (CombatRelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
        "pumpkin_candle": (CombatRelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
        "sai": (CombatRelicMarkerSpec("gain_block", amount=7, target_id="player"),),
        "snecko_eye": (CombatRelicMarkerSpec("draw_cards", amount=2, target_id="player"),),
        "sozu": (CombatRelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
        "spiked_gauntlets": (
            CombatRelicMarkerSpec("gain_energy", amount=1, target_id="player"),
        ),
        "velvet_choker": (
            CombatRelicMarkerSpec("gain_energy", amount=1, target_id="player"),
            CombatRelicMarkerSpec(
                "turn_card_play_limit",
                amount=6,
                target_id="player",
                metadata={"limit": 6},
            ),
        ),
        "whispering_earring": (
            CombatRelicMarkerSpec("gain_energy", amount=1, target_id="player"),
        ),
        "toasty_mittens": (
            CombatRelicMarkerSpec(
                "exhaust_top_draw_pile",
                amount=1,
                target_id="player",
            ),
            _status_spec("strength", 1),
        ),
    },
    CombatRelicHook.TURN_END: {
        "lunar_pastry": (_resource_spec("star", 1),),
        "runic_pyramid": (
            CombatRelicMarkerSpec(
                "retain_hand",
                amount=1,
                target_id="player",
                metadata={"mode": "retain_full_hand"},
            ),
        ),
    },
    CombatRelicHook.COMBAT_END: {
        "burning_blood": (CombatRelicMarkerSpec("heal_player", amount=6, target_id="player"),),
        "black_blood": (CombatRelicMarkerSpec("heal_player", amount=12, target_id="player"),),
        "chosen_cheese": (
            CombatRelicMarkerSpec("max_hp_delta", amount=1, target_id="player"),
        ),
    },
}

_PERIODIC_TURN_ENERGY = {
    "happy_flower": (3, 1),
    "fake_happy_flower": (5, 1),
}

_PERIODIC_TURN_DRAW = {
    "pendulum": (3, 1),
    "pollinous_core": (4, 2),
}

_ORICHALCUM_BLOCK = {
    "orichalcum": 6,
    "fake_orichalcum": 3,
}

_ATTACKS_THIS_TURN_RULES = {
    "kunai": (
        3,
        CombatRelicMarkerSpec(
            "gain_status",
            amount=1,
            target_id="player",
            metadata={"status": "dexterity"},
        ),
    ),
    "shuriken": (
        3,
        CombatRelicMarkerSpec(
            "gain_status",
            amount=1,
            target_id="player",
            metadata={"status": "strength"},
        ),
    ),
    "ornamental_fan": (
        3,
        CombatRelicMarkerSpec("gain_block", amount=4, target_id="player"),
    ),
}

_KNOWN_RELIC_HOOKS: Mapping[str, tuple[CombatRelicHook, ...]] = {
    "art_of_war": (CombatRelicHook.TURN_END,),
    "beating_remnant": (CombatRelicHook.DAMAGE_TAKEN,),
    "bellows": (CombatRelicHook.START_OF_COMBAT,),
    "black_star": (CombatRelicHook.MONSTER_KILLED,),
    "bookmark": (CombatRelicHook.TURN_END,),
    "bone_tea": (CombatRelicHook.START_OF_COMBAT,),
    "bone_flute": (CombatRelicHook.CARD_PLAYED,),
    "book_repair_knife": (CombatRelicHook.MONSTER_KILLED,),
    "belt_buckle": (CombatRelicHook.START_OF_COMBAT,),
    "big_hat": (CombatRelicHook.START_OF_COMBAT,),
    "booming_conch": (CombatRelicHook.START_OF_COMBAT,),
    "bread": (CombatRelicHook.TURN_START,),
    "brilliant_scarf": (CombatRelicHook.CARD_PLAYED,),
    "burning_sticks": (CombatRelicHook.CARD_EXHAUSTED,),
    "candelabra": (CombatRelicHook.TURN_START,),
    "captains_wheel": (CombatRelicHook.TURN_START,),
    "centennial_puzzle": (CombatRelicHook.DAMAGE_TAKEN,),
    "chandelier": (CombatRelicHook.TURN_START,),
    "charons_ashes": (CombatRelicHook.CARD_EXHAUSTED,),
    "chemical_x": (CombatRelicHook.CARD_PLAYED,),
    "choices_paradox": (CombatRelicHook.START_OF_COMBAT,),
    "cloak_clasp": (CombatRelicHook.TURN_END,),
    "crossbow": (CombatRelicHook.TURN_START,),
    "daughter_of_the_wind": (CombatRelicHook.CARD_PLAYED,),
    "demon_tongue": (CombatRelicHook.DAMAGE_TAKEN,),
    "delicate_frond": (CombatRelicHook.START_OF_COMBAT,),
    "diamond_diadem": (CombatRelicHook.CARD_PLAYED, CombatRelicHook.DAMAGE_TAKEN),
    "ember_tea": (CombatRelicHook.START_OF_COMBAT,),
    "emotion_chip": (CombatRelicHook.TURN_START,),
    "fiddle": (CombatRelicHook.TURN_START,),
    "forgotten_soul": (CombatRelicHook.CARD_EXHAUSTED,),
    "galactic_dust": (CombatRelicHook.RESOURCE_SPENT,),
    "gambling_chip": (CombatRelicHook.START_OF_COMBAT,),
    "game_piece": (CombatRelicHook.CARD_PLAYED,),
    "gold_plated_cables": (CombatRelicHook.ORB_PASSIVE_TRIGGERED,),
    "gremlin_horn": (CombatRelicHook.MONSTER_KILLED,),
    "hand_drill": (CombatRelicHook.ENEMY_BLOCK_BROKEN,),
    "helical_dart": (CombatRelicHook.CARD_PLAYED,),
    "history_course": (CombatRelicHook.TURN_START,),
    "horn_cleat": (CombatRelicHook.TURN_START,),
    "ice_cream": (CombatRelicHook.TURN_END,),
    "intimidating_helmet": (CombatRelicHook.CARD_PLAYED,),
    "iron_club": (CombatRelicHook.CARD_PLAYED,),
    "ivory_tile": (CombatRelicHook.CARD_PLAYED,),
    "joss_paper": (CombatRelicHook.CARD_EXHAUSTED,),
    "kusarigama": (CombatRelicHook.CARD_PLAYED,),
    "lava_lamp": (CombatRelicHook.COMBAT_END,),
    "letter_opener": (CombatRelicHook.CARD_PLAYED,),
    "lizard_tail": (CombatRelicHook.DAMAGE_TAKEN,),
    "lost_wisp": (CombatRelicHook.CARD_PLAYED,),
    "meat_on_the_bone": (CombatRelicHook.COMBAT_END,),
    "metronome": (CombatRelicHook.ORB_CHANNELED,),
    "mini_regent": (CombatRelicHook.RESOURCE_SPENT,),
    "miniature_cannon": (CombatRelicHook.DAMAGE_DEALT,),
    "mr_struggles": (CombatRelicHook.TURN_START,),
    "music_box": (CombatRelicHook.CARD_PLAYED,),
    "mystic_lighter": (CombatRelicHook.DAMAGE_DEALT,),
    "mummified_hand": (CombatRelicHook.CARD_PLAYED,),
    "nunchaku": (CombatRelicHook.CARD_PLAYED,),
    "orange_dough": (CombatRelicHook.START_OF_COMBAT,),
    "paels_eye": (CombatRelicHook.TURN_END,),
    "paels_tears": (CombatRelicHook.TURN_END,),
    "paels_flesh": (CombatRelicHook.TURN_START,),
    "paels_legion": (CombatRelicHook.CARD_BLOCK_GAINED,),
    "paels_tooth": (CombatRelicHook.COMBAT_END,),
    "pantograph": (CombatRelicHook.START_OF_COMBAT,),
    "paper_krane": (CombatRelicHook.DAMAGE_TAKEN,),
    "parrying_shield": (CombatRelicHook.TURN_END,),
    "permafrost": (CombatRelicHook.CARD_PLAYED,),
    "petrified_toad": (CombatRelicHook.START_OF_COMBAT,),
    "philosophers_stone": (
        CombatRelicHook.START_OF_COMBAT,
        CombatRelicHook.TURN_START,
    ),
    "preserved_insect": (CombatRelicHook.START_OF_COMBAT,),
    "pen_nib": (CombatRelicHook.DAMAGE_DEALT,),
    "pocketwatch": (CombatRelicHook.TURN_END,),
    "rainbow_ring": (CombatRelicHook.CARD_PLAYED,),
    "razor_tooth": (CombatRelicHook.CARD_PLAYED,),
    "red_skull": (CombatRelicHook.START_OF_COMBAT, CombatRelicHook.DAMAGE_TAKEN),
    "regalite": (CombatRelicHook.CARD_CREATED,),
    "reptile_trinket": (CombatRelicHook.POTION_USED,),
    "ring_of_the_drake": (CombatRelicHook.TURN_START,),
    "ringing_triangle": (CombatRelicHook.TURN_END,),
    "ripple_basin": (CombatRelicHook.TURN_END,),
    "ruined_helmet": (CombatRelicHook.STATUS_GAINED,),
    "screaming_flagon": (CombatRelicHook.TURN_END,),
    "sling_of_courage": (CombatRelicHook.START_OF_COMBAT,),
    "snecko_skull": (CombatRelicHook.STATUS_APPLIED,),
    "sparkling_rouge": (CombatRelicHook.TURN_START,),
    "stone_calendar": (CombatRelicHook.TURN_END,),
    "strike_dummy": (CombatRelicHook.DAMAGE_DEALT,),
    "sturdy_clamp": (CombatRelicHook.TURN_END,),
    "sword_of_stone": (CombatRelicHook.MONSTER_KILLED,),
    "fake_strike_dummy": (CombatRelicHook.DAMAGE_DEALT,),
    "tea_of_discourtesy": (CombatRelicHook.START_OF_COMBAT,),
    "tuning_fork": (CombatRelicHook.CARD_PLAYED,),
    "tungsten_rod": (CombatRelicHook.DAMAGE_TAKEN,),
    "odd_mushroom": (CombatRelicHook.DAMAGE_TAKEN,),
    "paper_phrog": (CombatRelicHook.DAMAGE_DEALT,),
    "seal_of_gold": (CombatRelicHook.TURN_START,),
    "self_forming_clay": (CombatRelicHook.DAMAGE_TAKEN,),
    "the_boot": (CombatRelicHook.DAMAGE_DEALT,),
    "the_abacus": (CombatRelicHook.DRAW_PILE_SHUFFLED,),
    "throwing_axe": (CombatRelicHook.CARD_PLAYED,),
    "tingsha": (CombatRelicHook.CARD_DISCARDED,),
    "toolbox": (CombatRelicHook.START_OF_COMBAT,),
    "tough_bandages": (CombatRelicHook.CARD_DISCARDED,),
    "undying_sigil": (CombatRelicHook.DAMAGE_TAKEN,),
    "unceasing_top": (CombatRelicHook.HAND_EMPTY,),
    "unsettling_lamp": (CombatRelicHook.CARD_PLAYED,),
    "vambrace": (CombatRelicHook.CARD_BLOCK_GAINED,),
    "vexing_puzzlebox": (CombatRelicHook.START_OF_COMBAT,),
    "vitruvian_minion": (CombatRelicHook.DAMAGE_DEALT,),
    "war_hammer": (CombatRelicHook.MONSTER_KILLED,),
    **{
        relic_id: (CombatRelicHook.TURN_START,)
        for relic_id in _PERIODIC_TURN_ENERGY
    },
    **{
        relic_id: (CombatRelicHook.TURN_START,)
        for relic_id in _PERIODIC_TURN_DRAW
    },
    **{
        relic_id: (CombatRelicHook.TURN_END,)
        for relic_id in _ORICHALCUM_BLOCK
    },
}

_DYNAMIC_MARKER_RELICS_BY_HOOK: Mapping[CombatRelicHook, frozenset[str]] = {
    CombatRelicHook.CARD_BLOCK_GAINED: frozenset({"paels_legion", "vambrace"}),
    CombatRelicHook.CARD_CREATED: frozenset({"regalite"}),
    CombatRelicHook.CARD_DISCARDED: frozenset({"tingsha", "tough_bandages"}),
    CombatRelicHook.CARD_EXHAUSTED: frozenset(
        {"burning_sticks", "charons_ashes", "forgotten_soul", "joss_paper"}
    ),
    CombatRelicHook.CARD_PLAYED: frozenset(
        {
            "chemical_x",
            "kusarigama",
            "razor_tooth",
            "throwing_axe",
            "unsettling_lamp",
        }
    ),
    CombatRelicHook.COMBAT_END: frozenset({"lava_lamp"}),
    CombatRelicHook.DRAW_PILE_SHUFFLED: frozenset({"the_abacus"}),
    CombatRelicHook.ENEMY_BLOCK_BROKEN: frozenset({"hand_drill"}),
    CombatRelicHook.HAND_EMPTY: frozenset({"unceasing_top"}),
    CombatRelicHook.ORB_CHANNELED: frozenset({"metronome"}),
    CombatRelicHook.ORB_PASSIVE_TRIGGERED: frozenset({"gold_plated_cables"}),
    CombatRelicHook.POTION_USED: frozenset({"reptile_trinket"}),
    CombatRelicHook.RESOURCE_SPENT: frozenset({"galactic_dust", "mini_regent"}),
    CombatRelicHook.STATUS_APPLIED: frozenset({"snecko_skull"}),
    CombatRelicHook.STATUS_GAINED: frozenset({"ruined_helmet"}),
    CombatRelicHook.TURN_END: frozenset({"bookmark", "paels_eye", "parrying_shield"}),
    CombatRelicHook.TURN_START: frozenset({"emotion_chip", "history_course"}),
}

_SUPPORTED_BY_HOOK: Mapping[CombatRelicHook, frozenset[str]] = {
    hook: frozenset(
        set(_STATIC_MARKERS_BY_HOOK.get(hook, {}))
        | set(_DYNAMIC_MARKER_RELICS_BY_HOOK.get(hook, frozenset()))
        | (
            set(_PERIODIC_TURN_ENERGY)
            if hook is CombatRelicHook.TURN_START
            else set()
        )
        | (
            set(_PERIODIC_TURN_DRAW)
            if hook is CombatRelicHook.TURN_START
            else set()
        )
        | (set(_ORICHALCUM_BLOCK) if hook is CombatRelicHook.TURN_END else set())
        | (
            set(_ATTACKS_THIS_TURN_RULES)
            if hook is CombatRelicHook.CARD_PLAYED
            else set()
        )
        | (
            {
                "bone_flute",
                "daughter_of_the_wind",
                "brilliant_scarf",
                "diamond_diadem",
                "game_piece",
                "helical_dart",
                "intimidating_helmet",
                "iron_club",
                "ivory_tile",
                "letter_opener",
                "lost_wisp",
                "music_box",
                "mummified_hand",
                "nunchaku",
                "permafrost",
                "rainbow_ring",
                "tuning_fork",
            }
            if hook is CombatRelicHook.CARD_PLAYED
            else set()
        )
        | (
            {
                "fake_strike_dummy",
                "miniature_cannon",
                "mystic_lighter",
                "paper_phrog",
                "pen_nib",
                "strike_dummy",
                "the_boot",
                "vitruvian_minion",
            }
            if hook is CombatRelicHook.DAMAGE_DEALT
            else set()
        )
        | (
            {
                "beating_remnant",
                "centennial_puzzle",
                "demon_tongue",
                "diamond_diadem",
                "lizard_tail",
                "odd_mushroom",
                "paper_krane",
                "red_skull",
                "self_forming_clay",
                "tungsten_rod",
                "undying_sigil",
            }
            if hook is CombatRelicHook.DAMAGE_TAKEN
            else set()
        )
        | (
            {
                "bellows",
                "belt_buckle",
                "bone_tea",
                "booming_conch",
                "delicate_frond",
                "ember_tea",
                "pantograph",
                "preserved_insect",
                "red_skull",
                "sling_of_courage",
                "tea_of_discourtesy",
            }
            if hook is CombatRelicHook.START_OF_COMBAT
            else set()
        )
        | (
            {
                "art_of_war",
                "cloak_clasp",
                "ice_cream",
                "paels_tears",
                "pocketwatch",
                "ringing_triangle",
                "ripple_basin",
                "screaming_flagon",
                "stone_calendar",
                "sturdy_clamp",
            }
            if hook is CombatRelicHook.TURN_END
            else set()
        )
        | (
            {
                "bread",
                "candelabra",
                "captains_wheel",
                "chandelier",
                "horn_cleat",
                "mr_struggles",
                "paels_flesh",
                "ring_of_the_drake",
                "seal_of_gold",
                "sparkling_rouge",
            }
            if hook is CombatRelicHook.TURN_START
            else set()
        )
        | (
            {
                "black_star",
                "book_repair_knife",
                "gremlin_horn",
                "sword_of_stone",
                "war_hammer",
            }
            if hook is CombatRelicHook.MONSTER_KILLED
            else set()
        )
        | (
            {"meat_on_the_bone", "paels_tooth"}
            if hook is CombatRelicHook.COMBAT_END
            else set()
        )
    )
    for hook in CombatRelicHook
}


__all__ = [
    "CombatRelicBlocker",
    "CombatRelicContext",
    "CombatRelicHook",
    "CombatRelicInput",
    "CombatRelicMarker",
    "CombatRelicMarkerSpec",
    "CombatRelicResolution",
    "card_played",
    "combat_end",
    "combat_relic_blocker",
    "combat_relic_content_id",
    "damage_dealt",
    "damage_taken",
    "inferred_combat_relic_hooks",
    "monster_killed",
    "relic_source_id",
    "resolve_card_played_relics",
    "resolve_combat_end_relics",
    "resolve_combat_relic_hook",
    "resolve_damage_dealt_relics",
    "resolve_damage_taken_relics",
    "resolve_monster_killed_relics",
    "resolve_start_of_combat_relics",
    "resolve_turn_end_relics",
    "resolve_turn_start_relics",
    "start_of_combat",
    "supported_combat_relic_ids",
    "turn_end",
    "turn_start",
    "unsupported_combat_relic_handlers",
]
