"""Source-backed Ancient relic choices and persistent lifecycle constants."""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

from sts2sim.content.sources import SourceRef

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


def scroll_boxes_packs(
    rng: Random, cards: Sequence[RewardPoolCard], *, character_id: str
) -> tuple[tuple[str, ...], ...]:
    """Draw both packs without replacement, then apply Defect's Claw exception.

    Empty or incomplete source pools are explicit errors rather than invented
    card contents. Character filtering belongs to the supplied reward pool.
    """
    common = list(dict.fromkeys(card.card_id for card in cards if card.rarity is CardRarity.COMMON))
    uncommon = list(
        dict.fromkeys(card.card_id for card in cards if card.rarity is CardRarity.UNCOMMON)
    )
    if len(common) < 4 or len(uncommon) < 2:
        raise ValueError("Scroll Boxes requires four common and two uncommon source cards.")
    commons = rng.sample(common, 4)
    uncommons = rng.sample(uncommon, 2)
    packs = [(uncommons[index], *commons[index * 2 : index * 2 + 2]) for index in range(2)]
    if character_id.lower() == "defect":
        if not any(card.card_id == "claw" for card in cards):
            raise ValueError("Defect Scroll Boxes requires the source Claw card.")
        packs = [("claw",) * 3 if rng.random() < 0.01 else pack for pack in packs]
    return tuple(packs)
