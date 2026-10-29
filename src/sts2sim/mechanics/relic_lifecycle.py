"""Source-backed Ancient relic choices and persistent lifecycle constants."""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

from sts2sim.content.source_card_pools import SOURCE_CARD_POOLS
from sts2sim.content.sources import SourceRef
from sts2sim.engine.rng import random_stream

from .reward_pools import RewardPoolCard
from .rewards import CardRarity

SCROLL_BOXES_SOURCE = SourceRef(
    label="sts2-scroll-boxes-packs",
    version="v0.107.1",
    url="https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Scroll_Boxes",
    notes="Two distinct packs, each one uncommon and two commons; Defect has 1% Claw packs.",
)
TOUCH_OF_OROBAS_SOURCE = SourceRef(
    label="sts2-touch-of-orobas",
    url="https://slaythespire.wiki.gg/wiki/Slay_the_Spire_2:Touch_of_Orobas",
)
ANCIENT_STARTER_RELICS = {
    "burning_blood": "black_blood",
    "ring_of_the_snake": "ring_of_the_drake",
    "divine_right": "divine_destiny",
    "bound_phylactery": "phylactery_unbound",
    "cracked_core": "infused_core",
}

# Source per-combat mutable flags/counters. Saved run counters such as Joss
# Paper, Nunchaku, Pen Nib and Galactic Dust are deliberately preserved.
COMBAT_ONLY_RELIC_COUNTERS = frozenset({
    "belt_buckle",
    "burning_sticks", "centennial_puzzle", "demon_tongue", "metronome", "mini_regent",
    "paels_eye", "paels_legion", "permafrost", "red_skull", "ruined_helmet", "throwing_axe",
    "unsettling_lamp", "vambrace",
})
TURN_ONLY_RELIC_COUNTERS = frozenset({"demon_tongue", "mini_regent"})


def scroll_boxes_packs(
    rng: Random, cards: Sequence[RewardPoolCard], *, character_id: str
) -> tuple[tuple[str, ...], ...]:
    """Roll each pack's Claw exception before its common/common/uncommon draws.

    Empty or incomplete source pools are explicit errors rather than invented
    card contents. Character filtering belongs to the supplied reward pool.
    """
    common = list(dict.fromkeys(card.card_id for card in cards if card.rarity is CardRarity.COMMON))
    uncommon = list(
        dict.fromkeys(card.card_id for card in cards if card.rarity is CardRarity.UNCOMMON)
    )
    if len(common) < 4 or len(uncommon) < 2:
        raise ValueError("Scroll Boxes requires four common and two uncommon source cards.")
    order = {card_id: index for index, card_id in enumerate(
        SOURCE_CARD_POOLS.get(character_id.lower(), ()),
    )}
    common.sort(key=lambda card_id: order.get(card_id, len(order)))
    uncommon.sort(key=lambda card_id: order.get(card_id, len(order)))
    rewards_rng = random_stream(rng, "rewards")
    if character_id.lower() == "defect" and not any(card.card_id == "claw" for card in cards):
        raise ValueError("Defect Scroll Boxes requires the source Claw card.")
    packs = []
    for _ in range(2):
        if character_id.lower() == "defect" and rewards_rng.randrange(100) < 1:
            packs.append(("claw",) * 3)
            continue
        first = rewards_rng.choice(common)
        common.remove(first)
        second = rewards_rng.choice(common)
        common.remove(second)
        third = rewards_rng.choice(uncommon)
        uncommon.remove(third)
        packs.append((first, second, third))
    return tuple(packs)
