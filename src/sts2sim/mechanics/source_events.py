"""Explicit stable event startup variables and immediate numeric outcomes.

Pinned source: zhiyue/sts2-rl-agent@5d874e855ae1d0550d5c30416fadd7dbb0d133b7,
Models.Events/*.cs CalculateVars and option callbacks. Values are generated once
on entry, before the choice, and stored by the runtime through save/reload.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from random import Random

from .source_rng import source_float


@dataclass(frozen=True)
class EventNumericEffect:
    gold_delta: int = 0
    hp_loss: int = 0
    heal: int = 0
    max_hp_delta: int = 0
    next_page: str | None = None

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EventStartup:
    values: dict[str, int] = field(default_factory=dict)
    effects: dict[str, EventNumericEffect] = field(default_factory=dict)


def source_event_startup(event_id: str, rng: Random, *, max_hp: int, hp: int) -> EventStartup:
    name = event_id.upper()
    if name == "CRYSTAL_SPHERE":
        cost = 50 + rng.randrange(1, 50)
        return EventStartup(
            {"uncover_future_cost": cost}, {"UNCOVER_FUTURE": EventNumericEffect(gold_delta=-cost)}
        )
    if name == "DENSE_VEGETATION":
        gold = rng.randrange(61, 100)
        return EventStartup(
            {"gold": gold, "heal": max_hp * 30 // 100},
            {
                "TRUDGE_ON": EventNumericEffect(gold_delta=gold, hp_loss=8),
                "REST": EventNumericEffect(heal=max_hp * 30 // 100, next_page="REST"),
            },
        )
    if name == "JUNGLE_MAZE_ADVENTURE":
        # C# float->decimal retains seven significant decimal digits; PlayerCmd
        # truncates the final positive decimal gold value to an integer.
        solo = int(Decimal(150) + Decimal(format(source_float(rng, -15, 15), ".7g")))
        together = int(Decimal(50) + Decimal(format(source_float(rng, -15, 15), ".7g")))
        return EventStartup(
            {"solo_gold": solo, "join_forces_gold": together},
            {
                "SOLO_QUEST": EventNumericEffect(gold_delta=solo, hp_loss=18),
                "JOIN_FORCES": EventNumericEffect(gold_delta=together),
            },
        )
    if name == "LOST_WISP":
        gold = 60 + rng.randrange(-15, 16)
        return EventStartup({"gold": gold}, {"SEARCH": EventNumericEffect(gold_delta=gold)})
    if name == "LUMINOUS_CHOIR":
        cost = 149 - rng.randrange(0, 50)
        return EventStartup({"gold": cost}, {"OFFER_TRIBUTE": EventNumericEffect(gold_delta=-cost)})
    if name == "PUNCH_OFF":
        # Gold is consumed by event calculation even though neither option uses
        # the DynamicVar in the stable implementation.
        gold = rng.randrange(91, 99)
        return EventStartup(
            {"gold": gold},
            {
                "I_CAN_TAKE_THEM": EventNumericEffect(next_page="I_CAN_TAKE_THEM"),
            },
        )
    if name == "SPIRALING_WHIRLPOOL":
        heal = max_hp * 33 // 100
        return EventStartup({"heal": heal}, {"DRINK": EventNumericEffect(heal=heal)})
    if name == "SUNKEN_STATUE":
        gold = 111 + rng.randrange(-10, 11)
        return EventStartup(
            {"gold": gold}, {"DIVE_INTO_WATER": EventNumericEffect(gold_delta=gold, hp_loss=7)}
        )
    if name == "SUNKEN_TREASURY":
        first = 60 + rng.randrange(16) - 8
        second = 333 + rng.randrange(61) - 30
        return EventStartup(
            {"small_chest_gold": first, "large_chest_gold": second},
            {
                "FIRST_CHEST": EventNumericEffect(gold_delta=first),
                "SECOND_CHEST": EventNumericEffect(gold_delta=second),
            },
        )
    if name == "THIS_OR_THAT":
        gold = rng.randrange(41, 69)
        return EventStartup(
            {"gold": gold}, {"PLAIN": EventNumericEffect(gold_delta=gold, hp_loss=6)}
        )
    if name == "UNREST_SITE":
        return EventStartup({"heal": max_hp - hp}, {"REST": EventNumericEffect(heal=max_hp - hp)})
    if name == "WHISPERING_HOLLOW":
        cost = 35 + rng.randrange(-9, 10)
        return EventStartup({"gold": cost}, {"GOLD": EventNumericEffect(gold_delta=-cost)})
    return EventStartup()
