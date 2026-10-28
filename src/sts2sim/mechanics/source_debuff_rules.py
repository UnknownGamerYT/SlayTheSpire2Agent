"""Stable source power types used by Rend's distinct-debuff damage scaling."""

from __future__ import annotations

from collections.abc import Mapping

from .powers import normalize_power_id

# Spire Codex stable /api/powers?lang=eng, captured 2026-10-28.
# Keep the source taxonomy separate from Artifact's application restrictions.
SOURCE_DEBUFF_POWER_IDS = frozenset({
    "BIASED_COGNITION", "BORROWED_TIME", "CHAINS_OF_BINDING", "CONFUSED", "CONQUEROR",
    "CONSTRICT", "CRUSH_UNDER", "DAMPEN", "DARK_SHACKLES", "DEBILITATE", "DEMISE",
    "DISINTEGRATION", "DOOM", "DYING_STAR", "ENFEEBLING_TOUCH", "FLANKING", "FRAIL",
    "HANG", "HEX", "IMBALANCED", "KNOCKDOWN", "MAGIC_BOMB", "MANGLE", "MIND_ROT",
    "MONARCHS_GAZE_STRENGTH_DOWN", "NEUROSURGE", "NO_BLOCK", "NO_DRAW", "NO_ENERGY_GAIN",
    "OBLIVION", "PIERCING_WAIL", "PLOW", "POISON", "RINGING", "SHACKLING_POTION", "SHRIEK",
    "SHRINK", "SIC_EM", "SLOTH", "SLOW", "SMOGGY", "STRANGLE", "SURROUNDED", "TAG_TEAM",
    "TAINTED", "TANGLED", "TENDER", "THE_GAMBIT", "VULNERABLE", "WASTE_AWAY", "WEAK",
    "WRAITH_FORM",
})
_DEBUFF_IDS = frozenset(value.lower().replace("_", "") for value in SOURCE_DEBUFF_POWER_IDS)
# Rend counts negative effective Strength once, rather than each separate
# temporary recovery power. This interaction is documented by the Rend wiki.
_STRENGTH_RECOVERY_IDS = frozenset({"darkshackles", "piercingwail", "shacklingpotion"})
_SIGNED_DEBUFF_IDS = frozenset({"shriek", "shrink"})


def unique_debuff_count(statuses: Mapping[str, int]) -> int:
    amounts: dict[str, int] = {}
    for key, amount in statuses.items():
        normalized = normalize_power_id(key).replace("_", "")
        amounts[normalized] = amounts.get(normalized, 0) + amount
    debuffs = {
        key for key, amount in amounts.items()
        if key in _DEBUFF_IDS - _STRENGTH_RECOVERY_IDS
        and (amount > 0 or key in _SIGNED_DEBUFF_IDS and amount < 0)
    }
    if amounts.get("strength", 0) + amounts.get("temporarystrength", 0) < 0:
        debuffs.add("strength")
    return len(debuffs)
