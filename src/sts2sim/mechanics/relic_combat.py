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
    DAMAGE_DEALT = "damage_dealt"
    DAMAGE_TAKEN = "damage_taken"
    MONSTER_KILLED = "monster_killed"
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
    ):
        hooks.append(CombatRelicHook.CARD_PLAYED)
    if "deal" in text and "damage" in text:
        hooks.append(CombatRelicHook.DAMAGE_DEALT)
    if (
        "take " in text
        and "damage" in text
        or "lose hp" in text
        or "vulnerable" in text
        and "take" in text
    ):
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
        return _resolve_turn_end_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.CARD_PLAYED:
        return _resolve_card_played_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.DAMAGE_DEALT:
        return _resolve_damage_dealt_dynamic(relic_id, hook, context)
    if hook is CombatRelicHook.DAMAGE_TAKEN:
        return _resolve_damage_taken_dynamic(relic_id, hook, context)
    return ()


def _resolve_start_of_combat_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
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


def _resolve_card_played_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id not in _ATTACKS_THIS_TURN_RULES:
        return ()
    if _normalized_id(context.card_type or "") != "attack":
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
    if relic_id != "paper_phrog":
        return ()
    if context.target_statuses and _status_amount(context.target_statuses, "vulnerable") <= 0:
        return ()
    return (
        CombatRelicMarker(
            kind="modify_vulnerable_damage_dealt",
            relic_id=relic_id,
            hook=hook,
            amount=175,
            target_id=context.target_id,
            metadata={
                "condition": "target_vulnerable",
                "normal_multiplier_percent": 150,
                "multiplier_percent": 175,
            },
        ),
    )


def _resolve_damage_taken_dynamic(
    relic_id: str,
    hook: CombatRelicHook,
    context: CombatRelicContext,
) -> tuple[CombatRelicMarker, ...]:
    if relic_id != "odd_mushroom":
        return ()
    if context.player_statuses and _status_amount(context.player_statuses, "vulnerable") <= 0:
        return ()
    return (
        CombatRelicMarker(
            kind="modify_vulnerable_damage_taken",
            relic_id=relic_id,
            hook=hook,
            amount=125,
            target_id="player",
            metadata={
                "condition": "player_vulnerable",
                "normal_multiplier_percent": 150,
                "multiplier_percent": 125,
            },
            source=STS1_COMPAT_SOURCE,
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
        "blood_vial": (CombatRelicMarkerSpec("heal_player", amount=2, target_id="player"),),
        "fake_blood_vial": (
            CombatRelicMarkerSpec("heal_player", amount=1, target_id="player"),
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
        "vajra": (_status_spec("strength", 1),),
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
    },
    CombatRelicHook.TURN_START: {
        "blessed_antler": (CombatRelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
        "blood_soaked_rose": (
            CombatRelicMarkerSpec("gain_energy", amount=1, target_id="player"),
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
        "philosophers_stone": (
            CombatRelicMarkerSpec("gain_energy", amount=1, target_id="player"),
        ),
        "prismatic_gem": (CombatRelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
        "pumpkin_candle": (CombatRelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
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
    },
    CombatRelicHook.COMBAT_END: {
        "burning_blood": (CombatRelicMarkerSpec("heal_player", amount=6, target_id="player"),),
        "black_blood": (CombatRelicMarkerSpec("heal_player", amount=12, target_id="player"),),
    },
}

_PERIODIC_TURN_ENERGY = {
    "happy_flower": (3, 1),
    "fake_happy_flower": (5, 1),
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
    "preserved_insect": (CombatRelicHook.START_OF_COMBAT,),
    "odd_mushroom": (CombatRelicHook.DAMAGE_TAKEN,),
    "paper_phrog": (CombatRelicHook.DAMAGE_DEALT,),
    **{
        relic_id: (CombatRelicHook.TURN_START,)
        for relic_id in _PERIODIC_TURN_ENERGY
    },
    **{
        relic_id: (CombatRelicHook.TURN_END,)
        for relic_id in _ORICHALCUM_BLOCK
    },
}

_SUPPORTED_BY_HOOK: Mapping[CombatRelicHook, frozenset[str]] = {
    hook: frozenset(
        set(_STATIC_MARKERS_BY_HOOK.get(hook, {}))
        | (
            set(_PERIODIC_TURN_ENERGY)
            if hook is CombatRelicHook.TURN_START
            else set()
        )
        | (set(_ORICHALCUM_BLOCK) if hook is CombatRelicHook.TURN_END else set())
        | (
            set(_ATTACKS_THIS_TURN_RULES)
            if hook is CombatRelicHook.CARD_PLAYED
            else set()
        )
        | ({"paper_phrog"} if hook is CombatRelicHook.DAMAGE_DEALT else set())
        | ({"odd_mushroom"} if hook is CombatRelicHook.DAMAGE_TAKEN else set())
        | ({"preserved_insect"} if hook is CombatRelicHook.START_OF_COMBAT else set())
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
