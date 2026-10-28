"""Executable supplements for numeric effects omitted by the source list API.

The values here come from the published monster move tables, not from move
names. Both the engine and coverage classifier consume these declarations.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal

CardDestination = Literal["hand", "draw_pile", "discard_pile"]
StatusCardEffect = tuple[str, int, CardDestination]

# Sources: slaythespire.wiki.gg/wiki/Slay_the_Spire_2:<page>
# Chomper, Slimes, Myte, Fabricator, Slimed_Berserker, Phrog_Parasite.
MONSTER_STATUS_CARD_MOVES: dict[tuple[str, str], tuple[StatusCardEffect, ...]] = {
    ("chomper", "screech"): (("dazed", 3, "discard_pile"),),
    ("haunted_ship", "haunt"): (("dazed", 5, "discard_pile"),),
    ("leaf_slime_m", "sticky_shot"): (("slimed", 2, "discard_pile"),),
    ("leaf_slime_s", "goop"): (("slimed", 1, "discard_pile"),),
    ("twig_slime_m", "sticky_shot"): (("slimed", 1, "discard_pile"),),
    ("myte", "toxic"): (("toxic", 2, "hand"),),
    ("noisebot", "noise"): (("dazed", 2, "draw_pile"), ("dazed", 2, "discard_pile")),
    ("slimed_berserker", "vomit_ichor"): (("slimed", 10, "discard_pile"),),
    ("phrog_parasite", "infect"): (("infection", 3, "discard_pile"),),
    ("wriggler", "wriggle"): (("infection", 1, "discard_pile"),),
    ("test_subject", "burning_growl"): (("burn", 3, "discard_pile"),),
}

# Devoted_Sculptor, Shrinker_Beetle, The_Obscura. The source API calls Wail "SAIL".
MONSTER_POWER_MOVES = MappingProxyType({
    ("devoted_sculptor", "forbidden_incantation"): ("ritual", 9, "self"),
    ("shrinker_beetle", "shrinker"): ("shrink", 1, "player"),
    ("the_obscura", "sail"): ("strength", 3, "all_monsters"),
})

# These handlers execute bespoke state changes in transitions.py. An arbitrary
# empty Buff/Status/Debuff source move never qualifies merely because it parses.
EXPLICIT_MONSTER_MOVE_HANDLERS = frozenset({
    ("aeonglass", "increasing_intensity"),
    ("eye_with_teeth", "distract"),
    ("knowledge_demon", "curse_of_knowledge"),
    ("magi_knight", "dampen"),
    ("mecha_knight", "flamethrower"),
    ("soul_fysh", "beckon"),
    ("soul_fysh", "gaze"),
    ("vantom", "dismember"),
    ("the_insatiable", "liquify_ground"),
    ("waterfall_giant", "explode"),
})


def has_scripted_monster_move(monster_id: str, move_id: str) -> bool:
    """Return whether a concrete source supplement or bespoke handler exists."""
    key = (monster_id.lower(), move_id.lower())
    return (
        key in MONSTER_STATUS_CARD_MOVES
        or key in MONSTER_POWER_MOVES
        or key in EXPLICIT_MONSTER_MOVE_HANDLERS
    )


def status_card_move_effects(
    monster_id: str, move_id: str, ascension: int,
) -> tuple[StatusCardEffect, ...]:
    """Resolve documented ascension changes that the list endpoint omits."""
    key = (monster_id.lower(), move_id.lower())
    if key == ("test_subject", "burning_growl") and ascension >= 9:
        return (("burn", 5, "discard_pile"),)
    return MONSTER_STATUS_CARD_MOVES.get(key, ())
