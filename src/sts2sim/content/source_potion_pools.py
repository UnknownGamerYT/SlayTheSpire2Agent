"""Stable v0.107.1 potion pool membership, order, unlocks and generation gates.

Identifier/metadata projections from zhiyue/sts2-rl-agent@5d874e855ae1d0550d5c30416fadd7dbb0d133b7,
Models.PotionPools/*, Timeline.Epochs/Potion{1,2}Epoch and Models.Potions/*.
Missing profile epochs explicitly assume a fully unlocked profile.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType

SOURCE_POTION_POOLS = MappingProxyType(
    {
        "ironclad": ("blood_potion", "soldiers_stew", "ashwater"),
        "silent": ("poison_potion", "ghost_in_a_jar", "cunning_potion"),
        "regent": ("star_potion", "cosmic_concoction", "kings_courage"),
        "necrobinder": ("potion_of_doom", "pot_of_ghouls", "bone_brew"),
        "defect": ("focus_potion", "essence_of_darkness", "potion_of_capacity"),
        "shared": (
            "attack_potion",
            "beetle_juice",
            "blessing_of_the_forge",
            "block_potion",
            "bottled_potential",
            "clarity",
            "colorless_potion",
            "cure_all",
            "dexterity_potion",
            "distilled_chaos",
            "droplet_of_precognition",
            "duplicator",
            "energy_potion",
            "entropic_brew",
            "explosive_ampoule",
            "fairy_in_a_bottle",
            "fire_potion",
            "flex_potion",
            "fortifier",
            "fruit_juice",
            "fysh_oil",
            "gamblers_brew",
            "gigantification_potion",
            "heart_of_iron",
            "liquid_bronze",
            "liquid_memories",
            "lucky_tonic",
            "mazaleths_gift",
            "orobic_acid",
            "potion_of_binding",
            "powdered_demise",
            "power_potion",
            "radiant_tincture",
            "regen_potion",
            "shackling_potion",
            "ship_in_a_bottle",
            "skill_potion",
            "snecko_oil",
            "speed_potion",
            "stable_serum",
            "strength_potion",
            "swift_potion",
            "touch_of_insanity",
            "vulnerable_potion",
            "weak_potion",
        ),
    }
)

SOURCE_POTION_RARITIES = MappingProxyType(
    {
        "ashwater": "uncommon",
        "attack_potion": "common",
        "beetle_juice": "rare",
        "blessing_of_the_forge": "uncommon",
        "block_potion": "common",
        "blood_potion": "common",
        "bone_brew": "uncommon",
        "bottled_potential": "rare",
        "clarity": "uncommon",
        "colorless_potion": "common",
        "cosmic_concoction": "rare",
        "cunning_potion": "uncommon",
        "cure_all": "uncommon",
        "dexterity_potion": "common",
        "distilled_chaos": "rare",
        "droplet_of_precognition": "rare",
        "duplicator": "uncommon",
        "energy_potion": "common",
        "entropic_brew": "rare",
        "essence_of_darkness": "rare",
        "explosive_ampoule": "common",
        "fairy_in_a_bottle": "rare",
        "fire_potion": "common",
        "flex_potion": "common",
        "focus_potion": "common",
        "fortifier": "uncommon",
        "foul_potion": "event",
        "fruit_juice": "rare",
        "fysh_oil": "uncommon",
        "gamblers_brew": "uncommon",
        "ghost_in_a_jar": "rare",
        "gigantification_potion": "rare",
        "glowwater_potion": "event",
        "heart_of_iron": "uncommon",
        "kings_courage": "uncommon",
        "liquid_bronze": "uncommon",
        "liquid_memories": "rare",
        "lucky_tonic": "rare",
        "mazaleths_gift": "rare",
        "orobic_acid": "rare",
        "poison_potion": "common",
        "pot_of_ghouls": "rare",
        "potion_of_binding": "uncommon",
        "potion_of_capacity": "uncommon",
        "potion_of_doom": "common",
        "potion_shaped_rock": "token",
        "powdered_demise": "uncommon",
        "power_potion": "common",
        "radiant_tincture": "uncommon",
        "regen_potion": "uncommon",
        "shackling_potion": "rare",
        "ship_in_a_bottle": "rare",
        "skill_potion": "common",
        "snecko_oil": "rare",
        "soldiers_stew": "rare",
        "speed_potion": "common",
        "stable_serum": "uncommon",
        "star_potion": "common",
        "strength_potion": "common",
        "swift_potion": "common",
        "touch_of_insanity": "uncommon",
        "vulnerable_potion": "common",
        "weak_potion": "common",
    }
)

POTION_UNLOCK_EPOCHS = MappingProxyType(
    {
        "beetle_juice": "POTION1_EPOCH",
        "mazaleths_gift": "POTION1_EPOCH",
        "droplet_of_precognition": "POTION1_EPOCH",
        "powdered_demise": "POTION2_EPOCH",
        "ship_in_a_bottle": "POTION2_EPOCH",
        "touch_of_insanity": "POTION2_EPOCH",
        "blood_potion": "IRONCLAD4_EPOCH",
        "soldiers_stew": "IRONCLAD4_EPOCH",
        "ashwater": "IRONCLAD4_EPOCH",
        "poison_potion": "SILENT4_EPOCH",
        "ghost_in_a_jar": "SILENT4_EPOCH",
        "cunning_potion": "SILENT4_EPOCH",
        "star_potion": "REGENT4_EPOCH",
        "cosmic_concoction": "REGENT4_EPOCH",
        "kings_courage": "REGENT4_EPOCH",
        "potion_of_doom": "NECROBINDER4_EPOCH",
        "pot_of_ghouls": "NECROBINDER4_EPOCH",
        "bone_brew": "NECROBINDER4_EPOCH",
        "focus_potion": "DEFECT4_EPOCH",
        "essence_of_darkness": "DEFECT4_EPOCH",
        "potion_of_capacity": "DEFECT4_EPOCH",
    }
)

COMBAT_POTION_EXCLUSIONS = frozenset({"fairy_in_a_bottle", "fruit_juice", "regen_potion"})


def source_potion_options(
    character_id: str,
    *,
    unlocked_epochs: Iterable[str] | None = None,
    in_combat: bool = False,
) -> tuple[str, ...]:
    epochs = None if unlocked_epochs is None else {str(epoch).upper() for epoch in unlocked_epochs}
    pool = SOURCE_POTION_POOLS.get(character_id.lower(), ()) + SOURCE_POTION_POOLS["shared"]
    return tuple(
        potion
        for potion in pool
        if (
            epochs is None
            or potion not in POTION_UNLOCK_EPOCHS
            or POTION_UNLOCK_EPOCHS[potion] in epochs
        )
        and (not in_combat or potion not in COMBAT_POTION_EXCLUSIONS)
    )
