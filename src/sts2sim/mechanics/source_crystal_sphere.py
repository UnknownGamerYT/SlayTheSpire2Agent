"""Stable Crystal Sphere board placement and cell reveal rules.

Pinned source: CrystalSphereMinigame/CrystalSphereItem at the stable source
snapshot documented in source_events. Board coordinates iterate x before y.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from random import Random
from typing import Any


@dataclass(frozen=True)
class SphereItem:
    kind: str
    rarity: str = ''
    width: int = 1
    height: int = 1
    x: int = 0
    y: int = 0
    placed: bool = False
    subscriptions: int = 0

    @property
    def cells(self) -> tuple[int, ...]:
        return tuple((self.x + dx) * 11 + self.y + dy
                     for dx in range(self.width) for dy in range(self.height))


@dataclass(frozen=True)
class SphereBoard:
    items: tuple[SphereItem, ...]
    cleared: tuple[int, ...]
    remaining: int
    revealed: tuple[int, ...] = ()
    placed_all: bool = False

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def restore(cls, value: Mapping[str, Any]) -> SphereBoard:
        return cls(items=tuple(SphereItem(**item) for item in value['items']),
                   cleared=tuple(value['cleared']), remaining=int(value['remaining']),
                   revealed=tuple(value.get('revealed', ())),
                   placed_all=bool(value.get('placed_all', False)))


def new_sphere_board(rng: Random, divinations: int) -> SphereBoard:
    cleared = {x * 11 + y for x in range(11) for y in range(11)
               if min(x, 10 - x) + min(y, 10 - y) <= 2}
    occupied: set[int] = set()
    items: list[SphereItem] = []
    templates = (SphereItem('relic', width=4, height=4),
        *(SphereItem('potion', 'common', 1, 3) for _ in range(2)),
        SphereItem('potion', 'rare', 2, 2),
        *(SphereItem('card', rarity, 2, 2) for rarity in ('common', 'uncommon', 'rare')),
        SphereItem('curse', width=2, height=2),
        *(SphereItem('gold') for _ in range(5)),
        *(SphereItem('big_gold', width=2) for _ in range(2)))
    placed_all = False
    # The source retries on the same board and retains prior items/subscriptions.
    for _attempt in range(10):
        placed_all = True
        for template in templates:
            item = template
            if placed_all:
                candidates = []
                for x in range(11):
                    for y in range(11):
                        if x + item.width > 11 or y + item.height > 11:
                            continue
                        cells = {(x + dx) * 11 + y + dy
                                 for dx in range(item.width) for dy in range(item.height)}
                        if not cells & (cleared | occupied):
                            candidates.append((x, y))
                if candidates:
                    x, y = rng.choice(candidates)
                    item = replace(item, x=x, y=y, placed=True)
                    occupied.update(item.cells)
                else:
                    placed_all = False
            items.append(item)
        items = [replace(item, subscriptions=item.subscriptions + 1) for item in items]
        if placed_all:
            break
    return SphereBoard(tuple(items), tuple(sorted(cleared)), divinations,
                       placed_all=placed_all)


def sphere_click(
    board: SphereBoard, x: int, y: int, *, big: bool
) -> tuple[SphereBoard, tuple[int, ...]]:
    if not 0 <= x < 11 or not 0 <= y < 11:
        raise ValueError('sphere cell outside board')
    if board.remaining <= 0 or x * 11 + y in board.cleared:
        raise ValueError('sphere cell is not selectable')
    cells = ([(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
             + [(x + dx, y + dy) for dx in (-1, 1) for dy in (-1, 1)]
             + [(x, y)] if big else [(x, y)])
    cleared = set(board.cleared)
    revealed = list(board.revealed)
    completed: list[int] = []
    for cx, cy in cells:
        if not 0 <= cx < 11 or not 0 <= cy < 11 or cx * 11 + cy in cleared:
            continue
        cell = cx * 11 + cy
        cleared.add(cell)
        for index, item in enumerate(board.items):
            if item.placed and cell in item.cells and set(item.cells) <= cleared:
                completed.append(index)
                revealed.extend([index] * item.subscriptions)
                break
    return replace(board, cleared=tuple(sorted(cleared)), remaining=board.remaining - 1,
                   revealed=tuple(revealed)), tuple(completed)
