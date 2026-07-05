"""Pure Ancient/start-offering helpers.

The transition layer owns RunState mutation. This module keeps Ancient offer
generation and option resolution in small immutable dataclasses so tests,
tooling, and future transition code can reason about act-start choices without
importing engine models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from random import Random
from typing import Any, Protocol, TypeVar

from sts2sim.content.sources import PROVISIONAL_STS2_SOURCE, SourceRef

PoolItem = str | Mapping[str, object]
Seed = int | str
T = TypeVar("T")


class AncientRng(Protocol):
    """Small RNG surface needed by Ancient generation and resolution."""

    def choice(self, seq: Sequence[T]) -> T: ...

    def shuffle(self, x: list[Any]) -> None: ...

    def random(self) -> float: ...


class AncientMarkerKind(str, Enum):
    GOLD = "gold"
    HP = "hp"
    HEAL = "heal"
    MAX_HP = "max_hp"
    POTION_SLOT = "potion_slot"
    CARD_ADD = "card_add"
    CARD_REWARD = "card_reward"
    CARD_REMOVE = "card_remove"
    CARD_REMOVE_RANDOM = "card_remove_random"
    CARD_TRANSFORM_RANDOM = "card_transform_random"
    CARD_UPGRADE_RANDOM = "card_upgrade_random"
    FIXED_RELIC = "fixed_relic"
    RANDOM_RELIC = "random_relic"
    FIXED_POTION = "fixed_potion"
    RANDOM_POTION = "random_potion"
    FLAG_SET = "flag_set"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class AncientMarker:
    """Effect marker for outcomes that a pure helper cannot fully execute."""

    kind: str
    count: int = 1
    amount: int | None = None
    item_id: str | None = None
    qualifier: str | None = None
    description: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        item_id = _normalized_id(self.item_id) if self.item_id is not None else None
        qualifier = str(self.qualifier) if self.qualifier is not None else None
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "count", max(0, int(self.count)))
        object.__setattr__(self, "amount", None if self.amount is None else int(self.amount))
        object.__setattr__(self, "item_id", item_id)
        object.__setattr__(self, "qualifier", qualifier)
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class AncientEffectSpec:
    """Reusable effect payload attached to table-driven Ancient choices."""

    gold_delta: int = 0
    set_gold: int | None = None
    hp_delta: int = 0
    heal_amount: int = 0
    heal_percent_missing_hp: float = 0.0
    max_hp_delta: int = 0
    potion_slot_delta: int = 0
    fixed_card_ids: tuple[str, ...] = ()
    remove_card_ids: tuple[str, ...] = ()
    random_relic_count: int = 0
    fixed_potion_ids: tuple[str, ...] = ()
    random_potion_count: int = 0
    card_reward_count: int = 0
    card_reward_size: int = 3
    card_reward_kind: str | None = None
    upgrade_random_count: int = 0
    transform_random_count: int = 0
    remove_random_count: int = 0
    set_flags: Mapping[str, object] = field(default_factory=dict)
    markers: tuple[AncientMarker, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    source: SourceRef = PROVISIONAL_STS2_SOURCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "gold_delta", int(self.gold_delta))
        if self.set_gold is not None:
            object.__setattr__(self, "set_gold", max(0, int(self.set_gold)))
        object.__setattr__(self, "hp_delta", int(self.hp_delta))
        object.__setattr__(self, "heal_amount", max(0, int(self.heal_amount)))
        object.__setattr__(
            self,
            "heal_percent_missing_hp",
            max(0.0, float(self.heal_percent_missing_hp)),
        )
        object.__setattr__(self, "max_hp_delta", int(self.max_hp_delta))
        object.__setattr__(self, "potion_slot_delta", int(self.potion_slot_delta))
        object.__setattr__(self, "fixed_card_ids", _normalized_ids(self.fixed_card_ids))
        object.__setattr__(self, "remove_card_ids", _normalized_ids(self.remove_card_ids))
        object.__setattr__(self, "random_relic_count", max(0, int(self.random_relic_count)))
        object.__setattr__(self, "fixed_potion_ids", _normalized_ids(self.fixed_potion_ids))
        object.__setattr__(self, "random_potion_count", max(0, int(self.random_potion_count)))
        object.__setattr__(self, "card_reward_count", max(0, int(self.card_reward_count)))
        object.__setattr__(self, "card_reward_size", max(0, int(self.card_reward_size)))
        if self.card_reward_kind is not None:
            object.__setattr__(self, "card_reward_kind", _normalized_id(self.card_reward_kind))
        object.__setattr__(
            self,
            "upgrade_random_count",
            max(0, int(self.upgrade_random_count)),
        )
        object.__setattr__(
            self,
            "transform_random_count",
            max(0, int(self.transform_random_count)),
        )
        object.__setattr__(self, "remove_random_count", max(0, int(self.remove_random_count)))
        object.__setattr__(self, "set_flags", dict(self.set_flags))
        object.__setattr__(self, "markers", tuple(self.markers))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class AncientChoice:
    """One visible Ancient option."""

    option_id: str
    name: str = ""
    kind: str = "choice"
    ancient_id: str | None = None
    act: int | None = None
    description: str = ""
    relic_id: str | None = None
    gold_delta: int = 0
    set_gold: int | None = None
    hp_delta: int = 0
    heal_amount: int = 0
    heal_percent_missing_hp: float = 0.0
    max_hp_delta: int = 0
    potion_slot_delta: int = 0
    fixed_card_ids: tuple[str, ...] = ()
    remove_card_ids: tuple[str, ...] = ()
    fixed_relic_ids: tuple[str, ...] = ()
    random_relic_count: int = 0
    fixed_potion_ids: tuple[str, ...] = ()
    random_potion_count: int = 0
    card_reward_count: int = 0
    card_reward_size: int = 3
    card_reward_kind: str | None = None
    upgrade_random_count: int = 0
    transform_random_count: int = 0
    remove_random_count: int = 0
    set_flags: Mapping[str, object] = field(default_factory=dict)
    required_card_ids: tuple[str, ...] = ()
    required_relic_ids: tuple[str, ...] = ()
    required_gold: int = 0
    markers: tuple[AncientMarker, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    source: SourceRef = PROVISIONAL_STS2_SOURCE

    def __post_init__(self) -> None:
        relic_id = _normalized_id(self.relic_id) if self.relic_id is not None else None
        fixed_relic_ids = _normalized_ids(self.fixed_relic_ids)
        if relic_id is not None and relic_id not in fixed_relic_ids:
            fixed_relic_ids = (relic_id,) + fixed_relic_ids

        object.__setattr__(self, "option_id", str(self.option_id))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "kind", _normalized_id(self.kind))
        if self.ancient_id is not None:
            object.__setattr__(self, "ancient_id", _normalized_id(self.ancient_id))
        if self.act is not None:
            object.__setattr__(self, "act", max(1, int(self.act)))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "relic_id", relic_id)
        object.__setattr__(self, "gold_delta", int(self.gold_delta))
        if self.set_gold is not None:
            object.__setattr__(self, "set_gold", max(0, int(self.set_gold)))
        object.__setattr__(self, "hp_delta", int(self.hp_delta))
        object.__setattr__(self, "heal_amount", max(0, int(self.heal_amount)))
        object.__setattr__(
            self,
            "heal_percent_missing_hp",
            max(0.0, float(self.heal_percent_missing_hp)),
        )
        object.__setattr__(self, "max_hp_delta", int(self.max_hp_delta))
        object.__setattr__(self, "potion_slot_delta", int(self.potion_slot_delta))
        object.__setattr__(self, "fixed_card_ids", _normalized_ids(self.fixed_card_ids))
        object.__setattr__(self, "remove_card_ids", _normalized_ids(self.remove_card_ids))
        object.__setattr__(self, "fixed_relic_ids", fixed_relic_ids)
        object.__setattr__(self, "random_relic_count", max(0, int(self.random_relic_count)))
        object.__setattr__(self, "fixed_potion_ids", _normalized_ids(self.fixed_potion_ids))
        object.__setattr__(self, "random_potion_count", max(0, int(self.random_potion_count)))
        object.__setattr__(self, "card_reward_count", max(0, int(self.card_reward_count)))
        object.__setattr__(self, "card_reward_size", max(0, int(self.card_reward_size)))
        if self.card_reward_kind is not None:
            object.__setattr__(self, "card_reward_kind", _normalized_id(self.card_reward_kind))
        object.__setattr__(
            self,
            "upgrade_random_count",
            max(0, int(self.upgrade_random_count)),
        )
        object.__setattr__(
            self,
            "transform_random_count",
            max(0, int(self.transform_random_count)),
        )
        object.__setattr__(self, "remove_random_count", max(0, int(self.remove_random_count)))
        object.__setattr__(self, "set_flags", dict(self.set_flags))
        object.__setattr__(self, "required_card_ids", _normalized_ids(self.required_card_ids))
        object.__setattr__(self, "required_relic_ids", _normalized_ids(self.required_relic_ids))
        object.__setattr__(self, "required_gold", max(0, int(self.required_gold)))
        object.__setattr__(self, "markers", tuple(self.markers))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class AncientContext:
    """State-like input for act-start Ancient generation and resolution."""

    act: int = 1
    ancient_id: str | None = None
    ascension_level: int = 0
    hp: int = 1
    max_hp: int = 1
    gold: int = 0
    deck: tuple[str, ...] = ()
    relics: tuple[str, ...] = ()
    curses: tuple[str, ...] = ()
    potions: tuple[str, ...] = ()
    flags: Mapping[str, object] = field(default_factory=dict)
    choices: tuple[AncientChoice, ...] = ()
    chosen_option_ids: tuple[str, ...] = ()
    source: SourceRef = PROVISIONAL_STS2_SOURCE

    def __post_init__(self) -> None:
        max_hp = max(1, int(self.max_hp))
        object.__setattr__(self, "act", max(1, int(self.act)))
        if self.ancient_id is not None:
            object.__setattr__(self, "ancient_id", _normalized_id(self.ancient_id))
        object.__setattr__(self, "ascension_level", max(0, int(self.ascension_level)))
        object.__setattr__(self, "max_hp", max_hp)
        object.__setattr__(self, "hp", min(max(0, int(self.hp)), max_hp))
        object.__setattr__(self, "gold", max(0, int(self.gold)))
        object.__setattr__(self, "deck", _normalized_ids(self.deck))
        object.__setattr__(self, "relics", _normalized_ids(self.relics))
        object.__setattr__(self, "curses", _normalized_ids(self.curses))
        object.__setattr__(self, "potions", _normalized_ids(self.potions))
        object.__setattr__(self, "flags", dict(self.flags))
        object.__setattr__(self, "choices", tuple(self.choices))
        object.__setattr__(
            self,
            "chosen_option_ids",
            tuple(str(option_id) for option_id in self.chosen_option_ids),
        )


@dataclass(frozen=True, slots=True)
class AncientResolution:
    """Resolved Ancient choice with applied state-like fields and effect markers."""

    choice: AncientChoice
    state: AncientContext
    gold_delta: int = 0
    hp_delta: int = 0
    heal_amount: int = 0
    max_hp_delta: int = 0
    potion_slot_delta: int = 0
    added_card_ids: tuple[str, ...] = ()
    removed_card_ids: tuple[str, ...] = ()
    relic_ids: tuple[str, ...] = ()
    curse_relic_ids: tuple[str, ...] = ()
    potion_ids: tuple[str, ...] = ()
    card_reward_count: int = 0
    random_relic_count: int = 0
    random_potion_count: int = 0
    upgrade_random_count: int = 0
    transform_random_count: int = 0
    remove_random_count: int = 0
    flags_set: Mapping[str, object] = field(default_factory=dict)
    markers: tuple[AncientMarker, ...] = ()


AncientOutcome = AncientResolution


def _normalized_ids(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(_normalized_id(value) for value in values)


def _normalized_id(value: object) -> str:
    return str(value).lower().replace("'", "").replace(" ", "_").replace("-", "_")


ANCIENT_SOURCE_URL = "https://spire-codex.com/api/mechanics/sections/neow"
OROBAS_PRISMATIC_SOURCE_URL = "https://spire-codex.com/api/mechanics/sections/orobas-prismatic"

NEOW_OFFERING_SOURCE = SourceRef(
    label="spire-codex-neow-offerings",
    url=ANCIENT_SOURCE_URL,
    notes="Starting Ancient relic offerings: two positive relics and one curse relic.",
)
OROBAS_PRISMATIC_SOURCE = SourceRef(
    label="spire-codex-orobas-prismatic",
    url=OROBAS_PRISMATIC_SOURCE_URL,
    notes="Sparse source note: Orobas may include Prismatic Gem in Act 2.",
)

GOLDEN_COMPASS_RELIC_ID = "golden_compass"
PRISMATIC_GEM_RELIC_ID = "prismatic_gem"

ANCIENT_IDS_BY_ACT: Mapping[int, tuple[str, ...]] = {
    1: ("neow",),
    2: ("orobas", "pael", "tezcatara"),
    3: ("nonupeipe", "tanx", "vakuu"),
}
UNPLACED_ANCIENT_IDS = ("darv", "the_architect")

ANCIENT_POSITIVE_RELICS = (
    "arcane_scroll",
    "booming_conch",
    "golden_pearl",
    "lead_paperweight",
    "lost_coffer",
    "massive_scroll",
    "neows_torment",
    "new_leaf",
    "phial_holster",
    "precise_scissors",
    "winged_boots",
)
ANCIENT_POSITIVE_RELIC_PAIRS = (
    ("lava_rock", "small_capsule"),
    ("nutritious_oyster", "stone_humidifier"),
    ("neows_talisman", "pomander"),
)
ANCIENT_CURSE_RELICS = (
    "cursed_pearl",
    "hefty_tablet",
    "large_capsule",
    "leafy_poultice",
    "neows_bones",
    "precarious_shears",
    "scroll_boxes",
    "silver_crucible",
)
ANCIENT_CURSE_BLOCKS: Mapping[str, frozenset[str]] = {
    "cursed_pearl": frozenset({"golden_pearl"}),
    "hefty_tablet": frozenset({"arcane_scroll"}),
    "large_capsule": frozenset({"lava_rock", "small_capsule"}),
    "leafy_poultice": frozenset({"new_leaf"}),
    "precarious_shears": frozenset({"precise_scissors"}),
}
ANCIENT_RELIC_NAMES: Mapping[str, str] = {
    "arcane_scroll": "Arcane Scroll",
    "booming_conch": "Booming Conch",
    "cursed_pearl": "Cursed Pearl",
    "fishing_rod": "Fishing Rod",
    "golden_compass": "Golden Compass",
    "golden_pearl": "Golden Pearl",
    "hefty_tablet": "Hefty Tablet",
    "kaleidoscope": "Kaleidoscope",
    "large_capsule": "Large Capsule",
    "lava_rock": "Lava Rock",
    "lead_paperweight": "Lead Paperweight",
    "leafy_poultice": "Leafy Poultice",
    "lost_coffer": "Lost Coffer",
    "massive_scroll": "Massive Scroll",
    "neows_bones": "Neow's Bones",
    "neows_talisman": "Neow's Talisman",
    "neows_torment": "Neow's Torment",
    "new_leaf": "New Leaf",
    "nutritious_oyster": "Nutritious Oyster",
    "phial_holster": "Phial Holster",
    "pomander": "Pomander",
    "precarious_shears": "Precarious Shears",
    "precise_scissors": "Precise Scissors",
    "prismatic_gem": "Prismatic Gem",
    "pumpkin_candle": "Pumpkin Candle",
    "scroll_boxes": "Scroll Boxes",
    "seal_of_gold": "Seal of Gold",
    "silken_tress": "Silken Tress",
    "silver_crucible": "Silver Crucible",
    "small_capsule": "Small Capsule",
    "stone_humidifier": "Stone Humidifier",
    "winged_boots": "Winged Boots",
}

ANCIENT_RELIC_EFFECTS: Mapping[str, AncientEffectSpec] = {
    "arcane_scroll": AncientEffectSpec(
        card_reward_count=1,
        card_reward_size=1,
        card_reward_kind="rare",
        metadata={"pickup_summary": "Obtain a random Rare card."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "cursed_pearl": AncientEffectSpec(
        gold_delta=333,
        fixed_card_ids=("greed",),
        metadata={"pickup_summary": "Receive Greed and gain 333 Gold."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "golden_compass": AncientEffectSpec(
        set_flags={"golden_compass_act2_map": True},
        markers=(
            AncientMarker(
                kind=AncientMarkerKind.CUSTOM.value,
                item_id="golden_compass",
                qualifier="map_replacement",
                description="Replace the Act 2 map with a special route.",
            ),
        ),
        metadata={"pickup_summary": "Replace the Act 2 map with a special route."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "golden_pearl": AncientEffectSpec(
        gold_delta=150,
        metadata={"pickup_summary": "Gain 150 Gold."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "fishing_rod": AncientEffectSpec(
        markers=(
            AncientMarker(
                kind=AncientMarkerKind.CUSTOM.value,
                item_id="fishing_rod",
                qualifier="normal_combat_upgrade_counter",
                description="Every 3 normal combats, upgrade a random card in Deck.",
                metadata={"combat_type": "normal", "interval": 3, "selection": "random"},
            ),
        ),
        metadata={"pickup_summary": "Every 3 normal combats, upgrade a random card."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "hefty_tablet": AncientEffectSpec(
        card_reward_count=1,
        card_reward_size=3,
        card_reward_kind="rare",
        metadata={"pickup_summary": "Choose 1 of 3 Rare cards."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "kaleidoscope": AncientEffectSpec(
        card_reward_count=2,
        card_reward_size=3,
        card_reward_kind="other_character",
        metadata={"pickup_summary": "Obtain 2 card rewards from other characters."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "large_capsule": AncientEffectSpec(
        random_relic_count=2,
        metadata={"pickup_summary": "Obtain 2 random Relics."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "lead_paperweight": AncientEffectSpec(
        card_reward_count=1,
        card_reward_size=2,
        card_reward_kind="colorless",
        metadata={"pickup_summary": "Choose 1 of 2 Colorless cards."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "leafy_poultice": AncientEffectSpec(
        transform_random_count=1,
        metadata={"pickup_summary": "Transform a Strike/Defend style card.", "filter": "basic"},
        source=NEOW_OFFERING_SOURCE,
    ),
    "lost_coffer": AncientEffectSpec(
        card_reward_count=1,
        random_potion_count=1,
        metadata={"pickup_summary": "Gain a card reward and procure 1 random Potion."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "massive_scroll": AncientEffectSpec(
        card_reward_count=1,
        card_reward_size=3,
        card_reward_kind="multiplayer",
        metadata={"pickup_summary": "Choose 1 of 3 multiplayer cards."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "neows_bones": AncientEffectSpec(
        random_relic_count=2,
        metadata={"pickup_summary": "Gain 2 random Neow Relics."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "neows_talisman": AncientEffectSpec(
        upgrade_random_count=1,
        metadata={"pickup_summary": "Upgrade a Strike/Defend style card.", "filter": "basic"},
        source=NEOW_OFFERING_SOURCE,
    ),
    "neows_torment": AncientEffectSpec(
        fixed_card_ids=("neows_fury",),
        metadata={"pickup_summary": "Add Neow's Fury to your deck."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "new_leaf": AncientEffectSpec(
        transform_random_count=1,
        metadata={"pickup_summary": "Transform 1 card."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "nutritious_oyster": AncientEffectSpec(
        max_hp_delta=11,
        metadata={"pickup_summary": "Raise Max HP by 11."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "phial_holster": AncientEffectSpec(
        potion_slot_delta=1,
        random_potion_count=2,
        metadata={"pickup_summary": "Gain 1 potion slot and procure 2 random Potions."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "pomander": AncientEffectSpec(
        upgrade_random_count=1,
        metadata={"pickup_summary": "Upgrade a card."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "precise_scissors": AncientEffectSpec(
        remove_random_count=1,
        metadata={"pickup_summary": "Remove 1 card from your deck."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "prismatic_gem": AncientEffectSpec(
        markers=(
            AncientMarker(
                kind=AncientMarkerKind.CUSTOM.value,
                item_id="prismatic_gem",
                qualifier="card_reward_all_colors",
                description="Card rewards may contain cards from any color.",
            ),
        ),
        metadata={"pickup_summary": "Card rewards may contain cards from any color."},
        source=OROBAS_PRISMATIC_SOURCE,
    ),
    "pumpkin_candle": AncientEffectSpec(
        markers=(
            AncientMarker(
                kind=AncientMarkerKind.CUSTOM.value,
                item_id="pumpkin_candle",
                qualifier="extinguishing_energy_relic",
                description=(
                    "Gain Energy each turn, extinguish after 5 combats, "
                    "kindle at Rest Sites."
                ),
                metadata={"combat_counter": 5, "campfire_action": "kindle"},
            ),
        ),
        metadata={"pickup_summary": "Gain Energy each turn for 5 combats; can be Kindled."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "scroll_boxes": AncientEffectSpec(
        card_reward_count=1,
        card_reward_kind="skill",
        metadata={"pickup_summary": "Choose 1 of 2 packs of cards to add to your deck."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "seal_of_gold": AncientEffectSpec(
        markers=(
            AncientMarker(
                kind=AncientMarkerKind.CUSTOM.value,
                item_id="seal_of_gold",
                qualifier="gold_for_energy",
                description="At the start of turn, spend 5 Gold to gain Energy.",
                metadata={"gold_cost": 5, "energy": 1},
            ),
        ),
        metadata={"pickup_summary": "Spend 5 Gold at turn start to gain Energy."},
        source=NEOW_OFFERING_SOURCE,
    ),
    "silken_tress": AncientEffectSpec(
        set_gold=0,
        markers=(
            AncientMarker(
                kind=AncientMarkerKind.CUSTOM.value,
                item_id="silken_tress",
                qualifier="first_card_reward_glam",
                description="Enchant all cards in the first card reward with Glam.",
                metadata={"card_reward_index": 1, "enchant_keyword": "glam"},
            ),
        ),
        metadata={
            "pickup_summary": "Lose all Gold; the first card reward is enchanted with Glam."
        },
        source=NEOW_OFFERING_SOURCE,
    ),
    "small_capsule": AncientEffectSpec(
        random_relic_count=1,
        metadata={"pickup_summary": "Obtain a random Relic."},
        source=NEOW_OFFERING_SOURCE,
    ),
}


def ancient_ids_for_act(act: int) -> tuple[str, ...]:
    """Return known Ancient ids for an act."""

    return ANCIENT_IDS_BY_ACT.get(max(1, int(act)), UNPLACED_ANCIENT_IDS)


def default_ancient_id_for_act(act: int) -> str:
    """Return a stable fallback Ancient id for an act."""

    return ancient_ids_for_act(act)[0]


def with_generated_ancient_choices(
    context: AncientContext,
    *,
    rng: AncientRng | None = None,
    seed: Seed | None = None,
) -> AncientContext:
    """Return ``context`` with a stable Ancient id and generated choices."""

    rng_obj = _coerce_rng(rng=rng, seed=seed)
    ancient_id = context.ancient_id or default_ancient_id_for_act(context.act)
    choices = _generate_relic_offering_choices(context, ancient_id, rng_obj)
    return replace(context, ancient_id=ancient_id, choices=choices)


def generate_ancient_choices(
    context: AncientContext,
    *,
    rng: AncientRng | None = None,
    seed: Seed | None = None,
) -> tuple[AncientChoice, ...]:
    """Generate visible Ancient choices from a context and seed/RNG."""

    return with_generated_ancient_choices(context, rng=rng, seed=seed).choices


def legal_ancient_choice_ids(context: AncientContext) -> tuple[str, ...]:
    """Return currently legal Ancient choice ids."""

    chosen = {_normalized_id(option_id) for option_id in context.chosen_option_ids}
    return tuple(
        choice.option_id
        for choice in context.choices
        if _normalized_id(choice.option_id) not in chosen and _choice_is_legal(choice, context)
    )


def available_ancient_choice_ids(context: AncientContext) -> tuple[str, ...]:
    """Alias matching other mechanics modules' available_* naming."""

    return legal_ancient_choice_ids(context)


def resolve_ancient_choice(
    context: AncientContext,
    option_id: str,
    *,
    rng: AncientRng | None = None,
    seed: Seed | None = None,
    relic_pool: Sequence[PoolItem] | None = None,
    potion_pool: Sequence[PoolItem] | None = None,
) -> AncientResolution:
    """Resolve an Ancient choice into deltas, markers, and next context."""

    choice = _choice_by_id(context, option_id)
    if choice.option_id not in legal_ancient_choice_ids(context):
        raise ValueError(f"Ancient choice is not legal: {option_id}")

    rng_obj = (
        _coerce_rng(rng=rng, seed=seed)
        if _needs_random_pool(choice, relic_pool, potion_pool)
        else None
    )
    next_max_hp = max(1, context.max_hp + choice.max_hp_delta)
    hp_after_max = _hp_after_max_hp_change(context.hp, context.max_hp, next_max_hp)
    hp_after_delta = min(max(0, hp_after_max + choice.hp_delta), next_max_hp)
    heal_amount = _choice_heal_amount(choice, next_max_hp, hp_after_delta)
    next_hp = min(next_max_hp, hp_after_delta + heal_amount)
    next_gold = max(
        0,
        choice.set_gold if choice.set_gold is not None else context.gold + choice.gold_delta,
    )

    deck_after_removal, removed_card_ids = _remove_fixed_cards(context.deck, choice.remove_card_ids)
    added_card_ids = choice.fixed_card_ids
    next_deck = deck_after_removal + added_card_ids

    random_relic_ids = _draw_pool_ids(
        rng_obj,
        relic_pool,
        count=choice.random_relic_count,
        excluded=context.relics + choice.fixed_relic_ids,
        pool_name="relic",
    )
    relic_ids = choice.fixed_relic_ids + random_relic_ids
    curse_relic_ids = relic_ids if choice.kind == "curse_relic" else ()

    random_potion_ids = _draw_pool_ids(
        rng_obj,
        potion_pool,
        count=choice.random_potion_count,
        pool_name="potion",
    )
    potion_ids = choice.fixed_potion_ids + random_potion_ids

    next_flags = {**context.flags, **choice.set_flags}
    next_context = replace(
        context,
        hp=next_hp,
        max_hp=next_max_hp,
        gold=next_gold,
        deck=next_deck,
        relics=context.relics + relic_ids,
        curses=context.curses + curse_relic_ids,
        potions=context.potions + potion_ids,
        flags=next_flags,
        chosen_option_ids=context.chosen_option_ids + (choice.option_id,),
    )
    gold_delta = next_gold - context.gold
    hp_delta = next_hp - context.hp
    max_hp_delta = next_max_hp - context.max_hp
    markers = _resolution_markers(
        choice,
        gold_delta=gold_delta,
        hp_delta=hp_delta,
        heal_amount=heal_amount,
        max_hp_delta=max_hp_delta,
        added_card_ids=added_card_ids,
        removed_card_ids=removed_card_ids,
        relic_ids=relic_ids,
        potion_ids=potion_ids,
        random_relic_ids=random_relic_ids,
        random_potion_ids=random_potion_ids,
    )
    return AncientResolution(
        choice=choice,
        state=next_context,
        gold_delta=gold_delta,
        hp_delta=hp_delta,
        heal_amount=heal_amount,
        max_hp_delta=max_hp_delta,
        potion_slot_delta=choice.potion_slot_delta,
        added_card_ids=added_card_ids,
        removed_card_ids=removed_card_ids,
        relic_ids=relic_ids,
        curse_relic_ids=curse_relic_ids,
        potion_ids=potion_ids,
        card_reward_count=choice.card_reward_count,
        random_relic_count=choice.random_relic_count,
        random_potion_count=choice.random_potion_count,
        upgrade_random_count=choice.upgrade_random_count,
        transform_random_count=choice.transform_random_count,
        remove_random_count=choice.remove_random_count,
        flags_set=choice.set_flags,
        markers=markers,
    )


def _generate_relic_offering_choices(
    context: AncientContext,
    ancient_id: str,
    rng: AncientRng,
) -> tuple[AncientChoice, ...]:
    owned = set(context.relics) | set(context.curses)
    curse_candidates = [relic for relic in ANCIENT_CURSE_RELICS if relic not in owned]
    if not curse_candidates:
        curse_candidates = list(ANCIENT_CURSE_RELICS)
    curse_relic = rng.choice(curse_candidates)
    blocked = ANCIENT_CURSE_BLOCKS.get(curse_relic, frozenset())
    positive_pool = _positive_relic_pool(context, ancient_id, blocked=blocked, owned=owned, rng=rng)
    if len(positive_pool) < 2:
        raise ValueError("Ancient positive relic pool has fewer than two eligible choices.")

    rng.shuffle(positive_pool)
    positive_relics = positive_pool[:2]
    choices = [
        _choice_for_relic(
            context,
            ancient_id,
            index=index + 1,
            relic_id=relic_id,
            kind="positive_relic",
            blocked=blocked,
        )
        for index, relic_id in enumerate(positive_relics)
    ]
    choices.append(
        _choice_for_relic(
            context,
            ancient_id,
            index=3,
            relic_id=curse_relic,
            kind="curse_relic",
            blocked=blocked,
        )
    )
    return tuple(choices)


def _positive_relic_pool(
    context: AncientContext,
    ancient_id: str,
    *,
    blocked: frozenset[str],
    owned: set[str],
    rng: AncientRng,
) -> list[str]:
    positive_pool = [
        relic
        for relic in ANCIENT_POSITIVE_RELICS
        if relic not in blocked and relic not in owned
    ]
    if context.act == 2 and GOLDEN_COMPASS_RELIC_ID not in owned:
        positive_pool.append(GOLDEN_COMPASS_RELIC_ID)
    if (
        context.act == 2
        and ancient_id == "orobas"
        and PRISMATIC_GEM_RELIC_ID not in owned
        and rng.random() < (1.0 / 3.0)
    ):
        positive_pool.append(PRISMATIC_GEM_RELIC_ID)

    for relic_pair in ANCIENT_POSITIVE_RELIC_PAIRS:
        pair_candidates = [
            relic
            for relic in relic_pair
            if relic not in blocked and relic not in owned
        ]
        if pair_candidates:
            positive_pool.append(rng.choice(pair_candidates))
    return positive_pool


def _choice_for_relic(
    context: AncientContext,
    ancient_id: str,
    *,
    index: int,
    relic_id: str,
    kind: str,
    blocked: frozenset[str],
) -> AncientChoice:
    effect = ANCIENT_RELIC_EFFECTS.get(relic_id, AncientEffectSpec(source=NEOW_OFFERING_SOURCE))
    name = ANCIENT_RELIC_NAMES.get(relic_id, relic_id.replace("_", " ").title())
    pool = "curse" if kind == "curse_relic" else "positive"
    metadata = {
        "act": context.act,
        "ancient_id": ancient_id,
        "pool": pool,
        "source_url": effect.source.url or ANCIENT_SOURCE_URL,
        "blocked_positive_relic_ids": tuple(sorted(blocked)),
        **effect.metadata,
    }
    return AncientChoice(
        option_id=f"a{context.act}:ancient:{index}",
        name=name,
        kind=kind,
        ancient_id=ancient_id,
        act=context.act,
        description=f"Gain {name}.",
        relic_id=relic_id,
        gold_delta=effect.gold_delta,
        set_gold=effect.set_gold,
        hp_delta=effect.hp_delta,
        heal_amount=effect.heal_amount,
        heal_percent_missing_hp=effect.heal_percent_missing_hp,
        max_hp_delta=effect.max_hp_delta,
        potion_slot_delta=effect.potion_slot_delta,
        fixed_card_ids=effect.fixed_card_ids,
        remove_card_ids=effect.remove_card_ids,
        random_relic_count=effect.random_relic_count,
        fixed_potion_ids=effect.fixed_potion_ids,
        random_potion_count=effect.random_potion_count,
        card_reward_count=effect.card_reward_count,
        card_reward_size=effect.card_reward_size,
        card_reward_kind=effect.card_reward_kind,
        upgrade_random_count=effect.upgrade_random_count,
        transform_random_count=effect.transform_random_count,
        remove_random_count=effect.remove_random_count,
        set_flags=effect.set_flags,
        markers=effect.markers,
        metadata=metadata,
        source=effect.source,
    )


def _choice_by_id(context: AncientContext, option_id: str) -> AncientChoice:
    key = _normalized_id(option_id)
    for choice in context.choices:
        if _normalized_id(choice.option_id) == key:
            return choice
    raise ValueError(f"Unknown Ancient choice id: {option_id}")


def _choice_is_legal(choice: AncientChoice, context: AncientContext) -> bool:
    if context.gold < choice.required_gold:
        return False
    deck = set(context.deck)
    relics = set(context.relics)
    return set(choice.required_card_ids) <= deck and set(choice.required_relic_ids) <= relics


def _choice_heal_amount(choice: AncientChoice, max_hp: int, hp: int) -> int:
    heal = choice.heal_amount
    if choice.heal_percent_missing_hp > 0:
        missing_hp = max(0, max_hp - hp)
        fraction = (
            choice.heal_percent_missing_hp / 100.0
            if choice.heal_percent_missing_hp > 1
            else choice.heal_percent_missing_hp
        )
        heal += int(missing_hp * fraction)
    return min(max(0, heal), max(0, max_hp - hp))


def _hp_after_max_hp_change(hp: int, max_hp: int, next_max_hp: int) -> int:
    actual_delta = next_max_hp - max(1, max_hp)
    if actual_delta > 0:
        return min(next_max_hp, max(0, hp) + actual_delta)
    return min(max(0, hp), next_max_hp)


def _resolution_markers(
    choice: AncientChoice,
    *,
    gold_delta: int,
    hp_delta: int,
    heal_amount: int,
    max_hp_delta: int,
    added_card_ids: tuple[str, ...],
    removed_card_ids: tuple[str, ...],
    relic_ids: tuple[str, ...],
    potion_ids: tuple[str, ...],
    random_relic_ids: tuple[str, ...],
    random_potion_ids: tuple[str, ...],
) -> tuple[AncientMarker, ...]:
    markers: list[AncientMarker] = []
    if gold_delta:
        metadata: dict[str, object] = {}
        if choice.set_gold is not None:
            metadata["set_gold"] = choice.set_gold
        markers.append(_marker(AncientMarkerKind.GOLD, amount=gold_delta, metadata=metadata))
    if max_hp_delta:
        markers.append(_marker(AncientMarkerKind.MAX_HP, amount=max_hp_delta))
    if hp_delta and choice.hp_delta:
        markers.append(_marker(AncientMarkerKind.HP, amount=hp_delta))
    if heal_amount:
        markers.append(_marker(AncientMarkerKind.HEAL, amount=heal_amount))
    if choice.potion_slot_delta:
        markers.append(_marker(AncientMarkerKind.POTION_SLOT, amount=choice.potion_slot_delta))

    for card_id in added_card_ids:
        markers.append(_marker(AncientMarkerKind.CARD_ADD, item_id=card_id))
    for card_id in removed_card_ids:
        markers.append(_marker(AncientMarkerKind.CARD_REMOVE, item_id=card_id))
    for relic_id in choice.fixed_relic_ids:
        markers.append(_marker(AncientMarkerKind.FIXED_RELIC, item_id=relic_id))
    for potion_id in choice.fixed_potion_ids:
        markers.append(_marker(AncientMarkerKind.FIXED_POTION, item_id=potion_id))

    if choice.card_reward_count:
        metadata = {"size": choice.card_reward_size}
        if choice.card_reward_kind is not None:
            metadata["kind"] = choice.card_reward_kind
        markers.append(
            _marker(
                AncientMarkerKind.CARD_REWARD,
                count=choice.card_reward_count,
                metadata=metadata,
            )
        )
    if choice.random_relic_count:
        markers.append(
            _marker(
                AncientMarkerKind.RANDOM_RELIC,
                count=choice.random_relic_count,
                metadata=_resolved_ids_metadata(random_relic_ids),
            )
        )
    if choice.random_potion_count:
        markers.append(
            _marker(
                AncientMarkerKind.RANDOM_POTION,
                count=choice.random_potion_count,
                metadata=_resolved_ids_metadata(random_potion_ids),
            )
        )
    if choice.upgrade_random_count:
        markers.append(
            _marker(AncientMarkerKind.CARD_UPGRADE_RANDOM, count=choice.upgrade_random_count)
        )
    if choice.transform_random_count:
        markers.append(
            _marker(AncientMarkerKind.CARD_TRANSFORM_RANDOM, count=choice.transform_random_count)
        )
    if choice.remove_random_count:
        markers.append(
            _marker(AncientMarkerKind.CARD_REMOVE_RANDOM, count=choice.remove_random_count)
        )
    for key, value in choice.set_flags.items():
        markers.append(
            _marker(
                AncientMarkerKind.FLAG_SET,
                item_id=key,
                metadata={"value": value},
            )
        )
    markers.extend(choice.markers)
    return tuple(markers)


def _marker(
    kind: AncientMarkerKind,
    *,
    count: int = 1,
    amount: int | None = None,
    item_id: str | None = None,
    qualifier: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> AncientMarker:
    return AncientMarker(
        kind=kind.value,
        count=count,
        amount=amount,
        item_id=item_id,
        qualifier=qualifier,
        metadata=dict(metadata or {}),
    )


def _resolved_ids_metadata(ids: tuple[str, ...]) -> Mapping[str, object]:
    return {"resolved_ids": ids} if ids else {}


def _needs_random_pool(
    choice: AncientChoice,
    relic_pool: Sequence[PoolItem] | None,
    potion_pool: Sequence[PoolItem] | None,
) -> bool:
    return (
        (choice.random_relic_count > 0 and relic_pool is not None)
        or (choice.random_potion_count > 0 and potion_pool is not None)
    )


def _draw_pool_ids(
    rng: AncientRng | None,
    pool: Sequence[PoolItem] | None,
    *,
    count: int,
    excluded: Sequence[str] = (),
    pool_name: str,
) -> tuple[str, ...]:
    if count <= 0:
        return ()
    if pool is None:
        return ()

    excluded_ids = {_normalized_id(item_id) for item_id in excluded}
    candidates = [
        item_id
        for item_id in _unique_pool_ids(pool)
        if _normalized_id(item_id) not in excluded_ids
    ]
    if len(candidates) < count:
        raise ValueError(
            f"Ancient random {pool_name} pool has {len(candidates)} eligible ids, needs {count}."
        )
    if rng is None:
        return tuple(candidates[:count])

    available = list(candidates)
    selected: list[str] = []
    for _ in range(count):
        item_id = rng.choice(available)
        selected.append(item_id)
        available.remove(item_id)
    return tuple(selected)


def _remove_fixed_cards(
    deck: tuple[str, ...],
    remove_card_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    next_deck = list(deck)
    removed: list[str] = []
    for card_id in remove_card_ids:
        target = _normalized_id(card_id)
        for index, deck_card_id in enumerate(next_deck):
            if _normalized_id(deck_card_id) != target:
                continue
            removed.append(next_deck.pop(index))
            break
    return tuple(next_deck), tuple(removed)


def _coerce_rng(*, rng: AncientRng | None, seed: Seed | None) -> AncientRng:
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both.")
    if rng is not None:
        return rng
    return Random(0 if seed is None else seed)


def _unique_pool_ids(pool: Sequence[PoolItem]) -> tuple[str, ...]:
    seen: set[str] = set()
    ids: list[str] = []
    for item in pool:
        item_id = _pool_item_id(item)
        if item_id in seen:
            continue
        seen.add(item_id)
        ids.append(item_id)
    return tuple(ids)


def _pool_item_id(item: PoolItem) -> str:
    if isinstance(item, str):
        return _normalized_id(item)
    for key in ("id", "relic_id", "potion_id", "card_id", "item_id"):
        value = item.get(key)
        if value is not None:
            return _normalized_id(str(value))
    raise ValueError(f"Ancient reward pool item is missing an id: {item!r}")


__all__ = [
    "ANCIENT_CURSE_BLOCKS",
    "ANCIENT_CURSE_RELICS",
    "ANCIENT_IDS_BY_ACT",
    "ANCIENT_POSITIVE_RELICS",
    "ANCIENT_POSITIVE_RELIC_PAIRS",
    "ANCIENT_RELIC_EFFECTS",
    "ANCIENT_RELIC_NAMES",
    "AncientChoice",
    "AncientContext",
    "AncientEffectSpec",
    "AncientMarker",
    "AncientMarkerKind",
    "AncientOutcome",
    "AncientResolution",
    "available_ancient_choice_ids",
    "ancient_ids_for_act",
    "default_ancient_id_for_act",
    "generate_ancient_choices",
    "legal_ancient_choice_ids",
    "resolve_ancient_choice",
    "with_generated_ancient_choices",
]
