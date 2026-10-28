"""Native stable Ancient offer construction, independent of printed descriptions.

Source: zhiyue/sts2-rl-agent@5d874e855ae1d0550d5c30416fadd7dbb0d133b7,
Models.Events/{Neow,Orobas,Pael,Tezcatara,Nonupeipe,Tanx,Vakuu,Darv}.cs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from random import Random
from typing import Any

from .source_rng import float32, source_float

CHARACTERS = ("ironclad", "silent", "regent", "necrobinder", "defect")


def source_unlocked_characters(flags: Mapping[str, Any]) -> tuple[str, ...]:
    explicit = flags.get("unlocked_character_ids")
    if isinstance(explicit, Iterable) and not isinstance(explicit, str | bytes | Mapping):
        values = {str(value).lower() for value in explicit}
        return tuple(character for character in CHARACTERS if character in values)
    if "revealed_epochs" in flags:
        epochs = {str(epoch).upper() for epoch in flags["revealed_epochs"]}
        return tuple(character for character in CHARACTERS if character == "ironclad"
                     or f"{character.upper()}1_EPOCH" in epochs)
    count = int(flags.get("unlocked_character_count", len(CHARACTERS)))
    return CHARACTERS[:max(1, count)]


@dataclass(frozen=True)
class AncientFacts:
    ancient_id: str
    act: int
    character_id: str
    player_count: int = 1
    unlocked_characters: tuple[str, ...] = CHARACTERS
    common_cards: int = 0
    uncommon_cards: int = 0
    enchantable_cards: int = 0
    enchantable_attacks: int = 0
    enchantable_defends: int = 0
    removable_cards: int = 0
    has_basic_strike: bool = False
    has_starter_relic: bool = False
    has_transcendence_starter: bool = False
    has_event_pet: bool = False
    clears_deck: bool = False
    ancient_cards: tuple[str, ...] = ()


@dataclass(frozen=True)
class AncientOffer:
    relic_id: str
    cursed: bool = False
    max_hp_delta: int = 0
    flags: dict[str, object] = field(default_factory=dict)
    locked: bool = False


def generate_source_ancient_offers(
    facts: AncientFacts, rng: Random, rewards_rng: Random
) -> tuple[AncientOffer, ...]:
    name = facts.ancient_id.lower()
    if name == "neow":
        return _neow(facts, rng)
    if name == "orobas":
        other_characters = tuple(c for c in facts.unlocked_characters if c != facts.character_id)
        character = rng.choice(other_characters) if other_characters else facts.character_id
        special = (
            AncientOffer("prismatic_gem")
            if source_float(rng) < float32(0.3333333)
            else (AncientOffer("sea_glass", flags={"sea_glass_character_id": character}))
        )
        first = rng.choice(
            (
                AncientOffer("electric_shrymp"),
                AncientOffer("glass_eye"),
                AncientOffer("sand_castle"),
                special,
            )
        )
        second = AncientOffer(rng.choice(("alchemical_coffer", "driftwood", "radiant_pearl")))
        third = []
        if facts.has_starter_relic:
            third.append(AncientOffer("touch_of_orobas"))
        if facts.has_transcendence_starter:
            third.append(AncientOffer("archaic_tooth"))
        third_offer = rng.choice(third) if third else AncientOffer("", locked=True)
        return first, second, third_offer
    if name == "pael":
        pael_first = rng.choice(("paels_flesh", "paels_horn", "paels_tears"))
        middle = ["paels_wing"]
        if facts.enchantable_defends >= 3:
            middle.append("paels_claw")
        if facts.removable_cards >= 5:
            middle.append("paels_tooth")
        middle = middle * 2 + ["paels_growth"]
        pael_second = rng.choice(middle)
        last = ["paels_eye", "paels_blood"]
        if not facts.has_event_pet:
            last.append("paels_legion")
        return tuple(AncientOffer(relic) for relic in (pael_first, pael_second, rng.choice(last)))
    if name == "tezcatara":
        first_pool = ["very_hot_cocoa", "yummy_cookie"]
        if facts.has_basic_strike:
            first_pool.append("nutritious_soup")
        return tuple(
            AncientOffer(rng.choice(pool))
            for pool in (
                first_pool,
                ["biiig_hug", "storybook", "toasty_mittens"],
                ["golden_compass", "pumpkin_candle", "toy_box", "seal_of_gold"],
            )
        )
    if name in {"nonupeipe", "tanx"}:
        if name == "nonupeipe":
            pool = [
                "blessed_antler",
                "brilliant_scarf",
                "delicate_frond",
                "diamond_diadem",
                "fur_coat",
                "glitter",
                "jewelry_box",
                "looming_fruit",
                "signet_ring",
            ]
            if facts.enchantable_cards >= 4:
                pool.append("beautiful_bracelet")
        else:
            pool = [
                "claws",
                "crossbow",
                "iron_club",
                "meat_cleaver",
                "sai",
                "spiked_gauntlets",
                "tanxs_whistle",
                "throwing_axe",
                "war_hammer",
            ]
            if facts.enchantable_attacks >= 3:
                pool.append("tri_boomerang")
        rng.shuffle(pool)
        return tuple(AncientOffer(relic) for relic in pool[:3])
    if name == "vakuu":
        results = []
        for pool in (
            ["blood_soaked_rose", "whispering_earring", "fiddle"],
            ["preserved_fog", "sere_talon", "distinguished_cape"],
            ["choices_paradox", "music_box", "lords_parasol", "jeweled_mask"],
        ):
            rng.shuffle(pool)
            relic = pool[0]
            results.append(
                AncientOffer(relic, max_hp_delta=-9 if relic == "distinguished_cape" else 0)
            )
        return tuple(results)
    if name == "darv":
        groups: list[tuple[str, ...]] = [
            ("astrolabe",),
            ("black_star",),
            ("calling_bell",),
            ("empty_cage",),
        ]
        if not facts.clears_deck:
            groups.append(("pandoras_box",))
        groups += [("runic_pyramid",), ("snecko_eye",)]
        if facts.act == 2:
            groups.append(("ectoplasm", "sozu"))
        elif facts.act == 3:
            groups.append(("philosophers_stone", "velvet_choker"))
        selected = [AncientOffer(rng.choice(group)) for group in groups]
        rng.shuffle(selected)
        if rng.choice((True, False)):
            flags: dict[str, object] = {}
            if facts.ancient_cards:
                flags["dusty_tome_card_id"] = rewards_rng.choice(facts.ancient_cards)
            return (*selected[:2], AncientOffer("dusty_tome", flags=flags))
        return tuple(selected[:3])
    # DeprecatedAncientEvent produces no options in the stable snapshot.
    return ()


def _neow(facts: AncientFacts, rng: Random) -> tuple[AncientOffer, ...]:
    curses = [
        "cursed_pearl",
        "hefty_tablet",
        "large_capsule",
        "leafy_poultice",
        "neows_bones",
        "precarious_shears",
        "silken_tress",
        "silver_crucible",
    ]
    if facts.player_count != 1:
        curses.remove("silver_crucible")
    curse = rng.choice(curses)
    positives = [
        "arcane_scroll",
        "booming_conch",
        "fishing_rod",
        "golden_pearl",
        "kaleidoscope",
        "lead_paperweight",
        "lost_coffer",
        "massive_scroll",
        "neows_torment",
        "new_leaf",
        "phial_holster",
        "precise_scissors",
        "scroll_boxes",
        "winged_boots",
    ]
    exclusions = {
        "cursed_pearl": "golden_pearl",
        "hefty_tablet": "arcane_scroll",
        "leafy_poultice": "new_leaf",
        "precarious_shears": "precise_scissors",
    }
    if curse in exclusions:
        positives.remove(exclusions[curse])
    if curse != "large_capsule":
        positives.append(rng.choice(("lava_rock", "small_capsule")))
    positives.append(rng.choice(("nutritious_oyster", "stone_humidifier")))
    positives.append(rng.choice(("neows_talisman", "pomander")))
    # IsAllowedAtNeow runs after all three mutually-exclusive pair draws.
    if facts.player_count == 1:
        positives.remove("massive_scroll")
    else:
        positives.remove("winged_boots")
    if len(facts.unlocked_characters) != 5:
        positives.remove("kaleidoscope")
    if facts.common_cards < 4 or facts.uncommon_cards < 2:
        positives.remove("scroll_boxes")
    rng.shuffle(positives)
    return (
        AncientOffer(positives[0]),
        AncientOffer(positives[1]),
        AncientOffer(curse, cursed=True),
    )
