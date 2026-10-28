"""Opening encounter rosters derived from pinned stable GenerateMonsters rules.

The source list API exposes AllPossibleMonsters, including summons, rather than
an opening roster. Entries here refer to v0.107.1 public source revision
5d874e855ae1d0550d5c30416fadd7dbb0d133b7; see stable_enemy_source_contracts.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class EncounterSpawnSpec:
    monster_id: str
    slot_name: str | None = None
    initial_move_id: str | None = None
    hp_reduction: int = 0


FIXED_ENCOUNTER_ROSTERS = MappingProxyType(
    {
        "AEONGLASS_BOSS": ("AEONGLASS",),
        "AXEBOTS_NORMAL": ("AXEBOT",),
        "BYGONE_EFFIGY_ELITE": ("BYGONE_EFFIGY",),
        "BYRDONIS_ELITE": ("BYRDONIS",),
        "CEREMONIAL_BEAST_BOSS": ("CEREMONIAL_BEAST",),
        "CHOMPERS_NORMAL": ("CHOMPER", "CHOMPER"),
        "CONSTRUCT_MENAGERIE_NORMAL": ("PUNCH_CONSTRUCT", "CUBEX_CONSTRUCT", "CUBEX_CONSTRUCT"),
        "CORPSE_SLUGS_NORMAL": ("CORPSE_SLUG", "CORPSE_SLUG", "CORPSE_SLUG"),
        "CORPSE_SLUGS_WEAK": ("CORPSE_SLUG", "CORPSE_SLUG"),
        "CUBEX_CONSTRUCT_NORMAL": ("CUBEX_CONSTRUCT",),
        "CULTISTS_NORMAL": ("CALCIFIED_CULTIST", "DAMP_CULTIST"),
        "DECIMILLIPEDE_ELITE": (
            "DECIMILLIPEDE_SEGMENT_FRONT",
            "DECIMILLIPEDE_SEGMENT_MIDDLE",
            "DECIMILLIPEDE_SEGMENT_BACK",
        ),
        "DEVOTED_SCULPTOR_WEAK": ("DEVOTED_SCULPTOR",),
        "ENTOMANCER_ELITE": ("ENTOMANCER",),
        "EXOSKELETONS_NORMAL": ("EXOSKELETON", "EXOSKELETON", "EXOSKELETON", "EXOSKELETON"),
        "EXOSKELETONS_WEAK": ("EXOSKELETON", "EXOSKELETON", "EXOSKELETON"),
        "FABRICATOR_NORMAL": ("FABRICATOR",),
        "FAKE_MERCHANT_EVENT_ENCOUNTER": ("FAKE_MERCHANT_MONSTER",),
        "FOGMOG_NORMAL": ("FOGMOG",),
        "FOSSIL_STALKER_NORMAL": ("FOSSIL_STALKER",),
        "FROG_KNIGHT_NORMAL": ("FROG_KNIGHT",),
        "FUZZY_WURM_CRAWLER_WEAK": ("FUZZY_WURM_CRAWLER",),
        "GLOBE_HEAD_NORMAL": ("GLOBE_HEAD",),
        "GREMLIN_MERC_NORMAL": ("GREMLIN_MERC",),
        "HAUNTED_SHIP_NORMAL": ("HAUNTED_SHIP",),
        "HUNTER_KILLER_NORMAL": ("HUNTER_KILLER",),
        "INFESTED_PRISMS_ELITE": ("INFESTED_PRISM",),
        "INKLETS_NORMAL": ("INKLET", "INKLET", "INKLET"),
        "KAISER_CRAB_BOSS": ("CRUSHER", "ROCKET"),
        "KNIGHTS_ELITE": ("FLAIL_KNIGHT", "SPECTRAL_KNIGHT", "MAGI_KNIGHT"),
        "KNOWLEDGE_DEMON_BOSS": ("KNOWLEDGE_DEMON",),
        "LAGAVULIN_MATRIARCH_BOSS": ("LAGAVULIN_MATRIARCH",),
        "LIVING_FOG_NORMAL": ("LIVING_FOG",),
        "LOUSE_PROGENITOR_NORMAL": ("LOUSE_PROGENITOR",),
        "MAWLER_NORMAL": ("MAWLER",),
        "MECHA_KNIGHT_ELITE": ("MECHA_KNIGHT",),
        "MYSTERIOUS_KNIGHT_EVENT_ENCOUNTER": ("MYSTERIOUS_KNIGHT",),
        "MYTES_NORMAL": ("MYTE", "MYTE"),
        "NIBBITS_NORMAL": ("NIBBIT", "NIBBIT"),
        "NIBBITS_WEAK": ("NIBBIT",),
        "OVERGROWTH_CRAWLERS": ("SHRINKER_BEETLE", "FUZZY_WURM_CRAWLER"),
        "OVICOPTER_NORMAL": ("OVICOPTER",),
        "OWL_MAGISTRATE_NORMAL": ("OWL_MAGISTRATE",),
        "PHANTASMAL_GARDENERS_ELITE": (
            "PHANTASMAL_GARDENER",
            "PHANTASMAL_GARDENER",
            "PHANTASMAL_GARDENER",
            "PHANTASMAL_GARDENER",
        ),
        "PHROG_PARASITE_ELITE": ("PHROG_PARASITE",),
        "PUNCH_CONSTRUCT_NORMAL": ("PUNCH_CONSTRUCT",),
        "PUNCH_OFF_EVENT_ENCOUNTER": ("PUNCH_CONSTRUCT", "PUNCH_CONSTRUCT"),
        "QUEEN_BOSS": ("TORCH_HEAD_AMALGAM", "QUEEN"),
        "SCROLLS_OF_BITING_NORMAL": (
            "SCROLL_OF_BITING",
            "SCROLL_OF_BITING",
            "SCROLL_OF_BITING",
            "SCROLL_OF_BITING",
        ),
        "SCROLLS_OF_BITING_WEAK": ("SCROLL_OF_BITING", "SCROLL_OF_BITING", "SCROLL_OF_BITING"),
        "SEAPUNK_NORMAL": ("CALCIFIED_CULTIST", "SEAPUNK"),
        "SEAPUNK_WEAK": ("SEAPUNK",),
        "SEWER_CLAM_NORMAL": ("SEWER_CLAM",),
        "SHRINKER_BEETLE_WEAK": ("SHRINKER_BEETLE",),
        "SKULKING_COLONY_ELITE": ("SKULKING_COLONY",),
        "SLIMED_BERSERKER_NORMAL": ("SLIMED_BERSERKER",),
        "SLUDGE_SPINNER_WEAK": ("SLUDGE_SPINNER",),
        "SLUMBERING_BEETLE_NORMAL": ("BOWLBUG_ROCK", "BOWLBUG_SILK", "SLUMBERING_BEETLE"),
        "SNAPPING_JAXFRUIT_NORMAL": ("SNAPPING_JAXFRUIT", "FLYCONID"),
        "SOUL_FYSH_BOSS": ("SOUL_FYSH",),
        "SOUL_NEXUS_ELITE": ("SOUL_NEXUS",),
        "SPINY_TOAD_NORMAL": ("SPINY_TOAD",),
        "TERROR_EEL_ELITE": ("TERROR_EEL",),
        "TEST_SUBJECT_BOSS": ("TEST_SUBJECT",),
        "THE_ARCHITECT_EVENT_ENCOUNTER": ("ARCHITECT",),
        "THE_INSATIABLE_BOSS": ("THE_INSATIABLE",),
        "THE_KIN_BOSS": ("KIN_FOLLOWER", "KIN_FOLLOWER", "KIN_PRIEST"),
        "THE_LOST_AND_FORGOTTEN_NORMAL": ("THE_LOST", "THE_FORGOTTEN"),
        "THE_OBSCURA_NORMAL": ("THE_OBSCURA",),
        "THIEVING_HOPPER_WEAK": ("THIEVING_HOPPER",),
        "TOADPOLES_WEAK": ("TOADPOLE", "TOADPOLE"),
        "TUNNELER_NORMAL": ("CHOMPER", "TUNNELER"),
        "TUNNELER_WEAK": ("TUNNELER",),
        "TURRET_OPERATOR_WEAK": ("LIVING_SHIELD", "TURRET_OPERATOR"),
        "TWO_TAILED_RATS_NORMAL": ("TWO_TAILED_RAT", "TWO_TAILED_RAT", "TWO_TAILED_RAT"),
        "VANTOM_BOSS": ("VANTOM",),
        "VINE_SHAMBLER_NORMAL": ("VINE_SHAMBLER",),
        "WATERFALL_GIANT_BOSS": ("WATERFALL_GIANT",),
    }
)

_SLOTS = {
    "AXEBOTS_NORMAL": ("front",),
    "DECIMILLIPEDE_ELITE": ("segment1", "segment2", "segment3"),
    "DENSE_VEGETATION_EVENT_ENCOUNTER": ("wriggler1", "wriggler2", "wriggler3", "wriggler4"),
    "EXOSKELETONS_NORMAL": ("first", "second", "third", "fourth"),
    "EXOSKELETONS_WEAK": ("first", "second", "third"),
    "FABRICATOR_NORMAL": ("fabricator",),
    "FAKE_MERCHANT_EVENT_ENCOUNTER": ("merchant",),
    "FOGMOG_NORMAL": ("fogmog",),
    "GREMLIN_MERC_NORMAL": ("merc",),
    "KAISER_CRAB_BOSS": ("crusher", "rocket"),
    "KNIGHTS_ELITE": ("first", "second", "third"),
    "LIVING_FOG_NORMAL": ("livingFog",),
    "MYTES_NORMAL": ("first", "second"),
    "NIBBITS_NORMAL": ("front", "back"),
    "OVICOPTER_NORMAL": ("ovicopter",),
    "PHANTASMAL_GARDENERS_ELITE": ("first", "second", "third", "fourth"),
    "PHROG_PARASITE_ELITE": ("phrog",),
    "QUEEN_BOSS": ("amalgam", "queen"),
    "SLUMBERING_BEETLE_NORMAL": ("first", "second", "third"),
    "THE_KIN_BOSS": ("slot1", "slot2", "leaderSlot"),
    "THE_OBSCURA_NORMAL": ("obscura",),
    "TWO_TAILED_RATS_NORMAL": ("third", "fourth", "fifth"),
    "BOWLBUGS_WEAK": ("odd", "even"),
    "BOWLBUGS_NORMAL": ("first", "middle", "last"),
}


ENCOUNTER_SLOT_ORDER = MappingProxyType({
    "AXEBOTS_NORMAL": ('front',),
    "DECIMILLIPEDE_ELITE": ('segment1', 'segment2', 'segment3'),
    "DENSE_VEGETATION_EVENT_ENCOUNTER": ('wriggler1', 'wriggler2', 'wriggler3', 'wriggler4'),
    "EXOSKELETONS_NORMAL": ('first', 'second', 'third', 'fourth'),
    "EXOSKELETONS_WEAK": ('first', 'second', 'third'),
    "FABRICATOR_NORMAL": ('bot1', 'bot2', 'fabricator', 'bot3', 'bot4'),
    "FAKE_MERCHANT_EVENT_ENCOUNTER": ('merchant',),
    "FOGMOG_NORMAL": ('illusion', 'fogmog'),
    "KAISER_CRAB_BOSS": ('crusher', 'rocket'),
    "LIVING_FOG_NORMAL": ('bomb1', 'bomb2', 'bomb3', 'bomb4', 'bomb5', 'livingFog'),
    "MYTES_NORMAL": ('first', 'second'),
    "NIBBITS_NORMAL": ('front', 'back'),
    "OVICOPTER_NORMAL": ('egg1', 'egg2', 'egg3', 'egg4', 'egg5', 'ovicopter'),
    "PHANTASMAL_GARDENERS_ELITE": ('first', 'second', 'third', 'fourth'),
    "PHROG_PARASITE_ELITE": ('phrog', 'wriggler1', 'wriggler2', 'wriggler3', 'wriggler4'),
    "QUEEN_BOSS": ('amalgam', 'queen'),
    "THE_KIN_BOSS": ('slot1', 'slot2', 'leaderSlot'),
    "THE_OBSCURA_NORMAL": ('illusion', 'obscura'),
    "TWO_TAILED_RATS_NORMAL": ('first', 'second', 'third', 'fourth', 'fifth'),
})


def encounter_spawn_specs(
    encounter_id: str,
    rng: Random,
    *,
    imported_monster_ids: tuple[str, ...] = (),
) -> tuple[EncounterSpawnSpec, ...] | None:
    """Execute the 87 stable opening-roster contracts using encounter-local RNG."""
    key = encounter_id.upper()
    roster = FIXED_ENCOUNTER_ROSTERS.get(key)
    if key == "SLIMES_WEAK":
        small_pool = ["LEAF_SLIME_S", "TWIG_SLIME_S"]
        first = rng.choice(small_pool)
        small_pool.remove(first)
        last = rng.choice(small_pool)
        roster = (first, rng.choice(("LEAF_SLIME_M", "TWIG_SLIME_M")), last)
    elif key == "SLIMES_NORMAL":
        small: tuple[str, ...] = ("LEAF_SLIME_S", "TWIG_SLIME_S")
        # Rng.NextBool is NextInt(2) == 0.
        if rng.randrange(2) != 0:
            small = tuple(reversed(small))
        roster = ("TWIG_SLIME_M", "LEAF_SLIME_M", *small)
    elif key == "FLYCONID_NORMAL":
        roster = (rng.choice(("LEAF_SLIME_M", "TWIG_SLIME_M")), "FLYCONID")
    elif key == "SLITHERING_STRANGLER_NORMAL":
        variant = rng.randrange(3)
        companions = (
            ("SNAPPING_JAXFRUIT",)
            if variant == 0
            else (rng.choice(("LEAF_SLIME_M", "TWIG_SLIME_M")),)
            if variant == 1
            else tuple(rng.choice(("LEAF_SLIME_S", "TWIG_SLIME_S")) for _ in range(2))
        )
        roster = (*companions, "SLITHERING_STRANGLER")
    elif key == "BOWLBUGS_WEAK":
        roster = ("BOWLBUG_ROCK", rng.choice(("BOWLBUG_EGG", "BOWLBUG_NECTAR")))
    elif key == "BOWLBUGS_NORMAL":
        pool = ["BOWLBUG_EGG", "BOWLBUG_SILK", "BOWLBUG_NECTAR"]
        chosen = []
        for _ in range(2):
            monster = rng.choice(pool)
            pool.remove(monster)
            chosen.append(monster)
        roster = ("BOWLBUG_ROCK", *chosen)
    elif key == "RUBY_RAIDERS_NORMAL":
        pool = [
            "AXE_RUBY_RAIDER",
            "ASSASSIN_RUBY_RAIDER",
            "BRUTE_RUBY_RAIDER",
            "CROSSBOW_RUBY_RAIDER",
            "TRACKER_RUBY_RAIDER",
        ]
        chosen = []
        for _ in range(3):
            monster = rng.choice(pool)
            pool.remove(monster)
            chosen.append(monster)
        roster = tuple(chosen)
    elif key == "DENSE_VEGETATION_EVENT_ENCOUNTER":
        roster = ("WRIGGLER",) * 4
    elif key == "BATTLEWORN_DUMMY_EVENT_ENCOUNTER":
        if len(imported_monster_ids) != 1 or imported_monster_ids[0] not in {
            "BATTLE_FRIEND_V1",
            "BATTLE_FRIEND_V2",
            "BATTLE_FRIEND_V3",
        }:
            raise ValueError("Battleworn Dummy encounter requires its selected setting monster")
        roster = imported_monster_ids
    if roster is None:
        return None
    slots = _SLOTS.get(key, ())
    specs = tuple(
        EncounterSpawnSpec(monster_id, slots[i] if i < len(slots) else None)
        for i, monster_id in enumerate(roster)
    )
    if key == "INKLETS_NORMAL":
        specs = tuple(
            replace(spec, initial_move_id="WHIRLWIND" if i == 1 else "JAB")
            for i, spec in enumerate(specs)
        )
    elif key in {"CHOMPERS_NORMAL", "TUNNELER_NORMAL"}:
        specs = tuple(
            replace(
                spec, initial_move_id="SCREECH" if key == "TUNNELER_NORMAL" or i == 1 else "CLAMP"
            )
            if spec.monster_id == "CHOMPER"
            else spec
            for i, spec in enumerate(specs)
        )
    elif key == "THE_KIN_BOSS":
        specs = (
            replace(specs[0], initial_move_id="POWER_DANCE"),
            replace(specs[1], initial_move_id="QUICK_SLASH"),
            specs[2],
        )
    elif key == "PUNCH_OFF_EVENT_ENCOUNTER":
        specs = tuple(
            replace(
                spec,
                initial_move_id="FAST_PUNCH" if i == 0 else "READY",
                hp_reduction=rng.randrange(2, 10),
            )
            for i, spec in enumerate(specs)
        )
    elif key in {
        "CORPSE_SLUGS_NORMAL",
        "CORPSE_SLUGS_WEAK",
        "DECIMILLIPEDE_ELITE",
        "SCROLLS_OF_BITING_NORMAL",
        "SCROLLS_OF_BITING_WEAK",
        "TWO_TAILED_RATS_NORMAL",
    }:
        opening = rng.randrange(3)
        moves = (
            ("WHIP_SLAP", "GLOMP", "GOOP")
            if key.startswith("CORPSE_SLUGS")
            else ("WRITHE", "BULK", "CONSTRICT")
            if key == "DECIMILLIPEDE_ELITE"
            else ("CHOMP", "CHEW", "MORE_TEETH")
            if key.startswith("SCROLLS_OF_BITING")
            else ("SCRATCH", "DISEASE_BITE", "SCREECH")
        )
        specs = tuple(
            replace(spec, initial_move_id=moves[2 if i == 3 else (opening + i) % 3])
            for i, spec in enumerate(specs)
        )
    elif key == "DENSE_VEGETATION_EVENT_ENCOUNTER":
        specs = tuple(
            replace(spec, initial_move_id="NASTY_BITE" if i % 2 == 0 else "WRIGGLE")
            for i, spec in enumerate(specs)
        )
    return specs
