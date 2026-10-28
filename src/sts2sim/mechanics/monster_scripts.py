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
    ("entomancer", "pheromone_spit"),
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


# FollowUpState and starting-state contracts from the pinned v0.107.1 source.
# The list endpoint leaves these edges null; move-list order is not an AI rule.
MONSTER_STATE_FOLLOWUPS = MappingProxyType({
    ('AXEBOT', 'HAMMER_UPPERCUT_MOVE'): 'ONE_TWO_MOVE',
    ('AXEBOT', 'ONE_TWO_MOVE'): 'HAMMER_UPPERCUT_MOVE',
    ('AXEBOT', 'BOOT_UP_MOVE'): 'HAMMER_UPPERCUT_MOVE',
    ('BOWLBUG_ROCK', 'HEADBUTT_MOVE'): 'POST_HEADBUTT',
    ('BOWLBUG_SILK', 'TOXIC_SPIT_MOVE'): 'THRASH_MOVE',
    ('BOWLBUG_SILK', 'THRASH_MOVE'): 'TOXIC_SPIT_MOVE',
    ('BRUTE_RUBY_RAIDER', 'ROAR_MOVE'): 'BEAT_MOVE',
    ('BRUTE_RUBY_RAIDER', 'BEAT_MOVE'): 'ROAR_MOVE',
    ('BYRDONIS', 'SWOOP_MOVE'): 'PECK_MOVE',
    ('BYRDONIS', 'PECK_MOVE'): 'SWOOP_MOVE',
    ('CALCIFIED_CULTIST', 'DARK_STRIKE_MOVE'): 'DARK_STRIKE_MOVE',
    ('CALCIFIED_CULTIST', 'INCANTATION_MOVE'): 'DARK_STRIKE_MOVE',
    ('CHOMPER', 'SCREECH_MOVE'): 'CLAMP_MOVE',
    ('CHOMPER', 'CLAMP_MOVE'): 'SCREECH_MOVE',
    ('CROSSBOW_RUBY_RAIDER', 'RELOAD_MOVE'): 'FIRE_MOVE',
    ('CROSSBOW_RUBY_RAIDER', 'FIRE_MOVE'): 'RELOAD_MOVE',
    ('DAMP_CULTIST', 'DARK_STRIKE_MOVE'): 'DARK_STRIKE_MOVE',
    ('DAMP_CULTIST', 'INCANTATION_MOVE'): 'DARK_STRIKE_MOVE',
    ('DEVOTED_SCULPTOR', 'SAVAGE_MOVE'): 'SAVAGE_MOVE',
    ('DEVOTED_SCULPTOR', 'FORBIDDEN_INCANTATION_MOVE'): 'SAVAGE_MOVE',
    ('ENTOMANCER', 'SPEAR_MOVE'): 'PHEROMONE_SPIT_MOVE',
    ('ENTOMANCER', 'BEES_MOVE'): 'SPEAR_MOVE',
    ('FAT_GREMLIN', 'FLEE_MOVE'): 'FLEE_MOVE',
    ('FAT_GREMLIN', 'SPAWNED_MOVE'): 'FLEE_MOVE',
    ('FUZZY_WURM_CRAWLER', 'INHALE'): 'ACID_GOOP',
    ('FUZZY_WURM_CRAWLER', 'FIRST_ACID_GOOP'): 'INHALE',
    ('HAUNTED_SHIP', 'HAUNT_MOVE'): 'SWIPE_MOVE',
    ('LAGAVULIN_MATRIARCH', 'SLEEP_MOVE'): 'SLEEP_BRANCH',
    ('LEAF_SLIME_M', 'STICKY_SHOT'): 'CLUMP_SHOT',
    ('LEAF_SLIME_M', 'CLUMP_SHOT'): 'STICKY_SHOT',
    ('LOUSE_PROGENITOR', 'CURL_AND_GROW_MOVE'): 'POUNCE_MOVE',
    ('LOUSE_PROGENITOR', 'WEB_CANNON_MOVE'): 'CURL_AND_GROW_MOVE',
    ('PUNCH_CONSTRUCT', 'FAST_PUNCH_MOVE'): 'STRONG_PUNCH_MOVE',
    ('PUNCH_CONSTRUCT', 'READY_MOVE'): 'FAST_PUNCH_MOVE',
    ('SEWER_CLAM', 'JET_MOVE'): 'PRESSURIZE_MOVE',
    ('SEWER_CLAM', 'PRESSURIZE_MOVE'): 'JET_MOVE',
    ('SLIMED_BERSERKER', 'FURIOUS_PUMMELING_MOVE'): 'LEECHING_HUG_MOVE',
    ('SLIMED_BERSERKER', 'VOMIT_ICHOR_MOVE'): 'FURIOUS_PUMMELING_MOVE',
    ('SLUMBERING_BEETLE', 'SNORE_MOVE'): 'SNORE_NEXT',
    ('SNEAKY_GREMLIN', 'TACKLE_MOVE'): 'TACKLE_MOVE',
    ('SNEAKY_GREMLIN', 'SPAWNED_MOVE'): 'TACKLE_MOVE',
    ('THE_FORGOTTEN', 'DREAD'): 'MIASMA',
    ('THE_FORGOTTEN', 'MIASMA'): 'DREAD',
    ('THE_LOST', 'EYE_LASERS'): 'DEBILITATING_SMOG',
    ('THE_LOST', 'DEBILITATING_SMOG'): 'EYE_LASERS',
    ('TOUGH_EGG', 'NIBBLE_MOVE'): 'NIBBLE_MOVE',
    ('TOUGH_EGG', 'HATCH_MOVE'): 'NIBBLE_MOVE',
    ('TRACKER_RUBY_RAIDER', 'HOUNDS_MOVE'): 'HOUNDS_MOVE',
    ('TRACKER_RUBY_RAIDER', 'TRACK_MOVE'): 'HOUNDS_MOVE',
})

MONSTER_INITIAL_STATES = MappingProxyType({
    'BOWLBUG_SILK': 'TOXIC_SPIT_MOVE',
    'BYRDONIS': 'SWOOP_MOVE',
    'CROSSBOW_RUBY_RAIDER': 'RELOAD_MOVE',
    'LEAF_SLIME_M': 'STICKY_SHOT',
    'SEWER_CLAM': 'JET_MOVE',
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
