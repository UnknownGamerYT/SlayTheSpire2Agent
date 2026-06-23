"""Pure relic handler helpers.

The engine owns state mutation; this module only resolves table-driven relic
effects into small deltas and markers that callers can apply later.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

from sts2sim.content.sources import PROVISIONAL_STS2_SOURCE, SourceRef

RelicInput = str | Mapping[str, object]


class RelicHook(str, Enum):
    PICKUP = "pickup"
    SHOP_ENTER = "shop_enter"
    SHOP_PURCHASE = "shop_purchase"
    SHOP_PRICE = "shop_price"
    POTION_CAPACITY = "potion_capacity"
    CAMPFIRE_ENTER = "campfire_enter"
    START_COMBAT = "start_combat"
    START_TURN = "start_turn"
    END_TURN = "end_turn"
    END_COMBAT = "end_combat"


@dataclass(frozen=True, slots=True)
class RelicMarkerSpec:
    kind: str
    amount: int | None = None
    target_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RelicEffectMarker:
    kind: str
    relic_id: str
    amount: int | None = None
    target_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RelicHookRule:
    gold_delta: int = 0
    hp_delta: int = 0
    max_hp_delta: int = 0
    potion_slot_delta: int = 0
    markers: tuple[RelicMarkerSpec, ...] = ()
    source: SourceRef = PROVISIONAL_STS2_SOURCE


@dataclass(frozen=True, slots=True)
class RelicHookResult:
    hook: RelicHook
    relic_id: str
    gold_delta: int = 0
    hp_delta: int = 0
    max_hp_delta: int = 0
    potion_slot_delta: int = 0
    markers: tuple[RelicEffectMarker, ...] = ()
    unsupported: bool = False
    source: SourceRef = PROVISIONAL_STS2_SOURCE


@dataclass(frozen=True, slots=True)
class RelicHookResolution:
    hook: RelicHook
    results: tuple[RelicHookResult, ...] = ()
    gold_delta: int = 0
    hp_delta: int = 0
    max_hp_delta: int = 0
    potion_slot_delta: int = 0
    markers: tuple[RelicEffectMarker, ...] = ()
    source: SourceRef = PROVISIONAL_STS2_SOURCE


@dataclass(frozen=True, slots=True)
class RelicPriceModifier:
    relic_id: str
    multiplier_percent: int | None = None
    fixed_price: int | None = None
    item_kinds: frozenset[str] = field(default_factory=frozenset)
    source: SourceRef = PROVISIONAL_STS2_SOURCE


@dataclass(frozen=True, slots=True)
class RelicPriceResult:
    item_kind: str
    base_price: int
    price: int
    applied_relic_ids: tuple[str, ...] = ()
    multiplier_percent: int = 100
    fixed_price: int | None = None
    source: SourceRef = PROVISIONAL_STS2_SOURCE


@dataclass(frozen=True, slots=True)
class UnsupportedRelicHandler:
    relic_id: str
    name: str | None = None
    unsupported_hooks: tuple[RelicHook, ...] = ()
    description: str | None = None


def _default_relic_hook_rules() -> dict[RelicHook, dict[str, RelicHookRule]]:
    return {
        RelicHook.PICKUP: {
            "arcane_scroll": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "card_reward",
                        amount=1,
                        target_id="player",
                        metadata={"rarity": "rare", "selection": "random"},
                    ),
                ),
            ),
            "astrolabe": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "transform_deck_cards",
                        amount=3,
                        target_id="player",
                        metadata={"selection": "chosen", "upgrade_transformed": True},
                    ),
                ),
            ),
            "archaic_tooth": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "transform_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={
                            "selection": "chosen",
                            "match_card_ids": ("strike", "defend"),
                            "match_name_contains": ("strike", "defend"),
                            "transform_pool": "ancient",
                        },
                    ),
                ),
            ),
            "beautiful_bracelet": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "modify_deck_cards",
                        amount=3,
                        target_id="player",
                        metadata={
                            "selection": "chosen",
                            "custom": {"enchant_keyword": "swift", "enchant_amount": 3},
                        },
                    ),
                ),
            ),
            "biiig_hug": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "remove_deck_cards",
                        amount=4,
                        target_id="player",
                        metadata={"selection": "chosen"},
                    ),
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "shuffle_add_card_to_draw_pile",
                            "card_id": "soot",
                            "needed_subsystem": "draw_pile_shuffle_relic_trigger",
                        },
                    ),
                ),
            ),
            "big_mushroom": RelicHookRule(
                max_hp_delta=20,
                markers=(RelicMarkerSpec("max_hp_delta"),),
            ),
            "bing_bong": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "deck_card_added_duplicate_copy",
                            "needed_subsystem": "deck_add_trigger",
                        },
                    ),
                ),
            ),
            "blood_soaked_rose": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "add_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={"card_id": "enthralled"},
                    ),
                ),
            ),
            "bone_tea": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "relic_counter_changed",
                        amount=1,
                        target_id="player",
                        metadata={"counter": 1},
                    ),
                ),
            ),
            "book_of_five_rings": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "deck_card_added_count_heal",
                            "cards": 5,
                            "heal": 20,
                            "needed_subsystem": "deck_add_trigger",
                        },
                    ),
                ),
            ),
            "bowler_hat": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "gold_gain_bonus_percent",
                            "bonus_percent": 25,
                            "needed_subsystem": "gold_gain_modifier",
                        },
                    ),
                ),
            ),
            "byrdpip": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "add_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={"card_id": "byrd_swoop"},
                    ),
                ),
            ),
            "calling_bell": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "add_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={"card_id": "curse_of_the_bell"},
                    ),
                    RelicMarkerSpec("random_relics_gained", amount=3, target_id="player"),
                ),
            ),
            "cauldron": RelicHookRule(
                markers=(RelicMarkerSpec("random_potions_gained", amount=5),),
            ),
            "circlet": RelicHookRule(
                markers=(RelicMarkerSpec("no_effect"),),
            ),
            "claws": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "transform_deck_cards",
                        amount=6,
                        target_id="player",
                        metadata={"selection": "chosen", "target_card_id": "maul"},
                    ),
                ),
            ),
            "cursed_pearl": RelicHookRule(
                gold_delta=333,
                markers=(RelicMarkerSpec("gold_gained"),),
            ),
            "darkstone_periapt": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "curse_obtained_max_hp_delta_enabled",
                        amount=6,
                        target_id="player",
                        metadata={"trigger": "curse_obtained"},
                    ),
                ),
            ),
            "dingy_rug": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={"passive": "card_rewards_can_include_colorless"},
                    ),
                ),
            ),
            "dollys_mirror": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "duplicate_deck_card",
                        amount=1,
                        target_id="player",
                        metadata={"selection": "chosen"},
                    ),
                ),
            ),
            "dragon_fruit": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "gold_gain_max_hp_delta",
                            "max_hp_delta": 1,
                            "needed_subsystem": "gold_gain_trigger",
                        },
                    ),
                ),
            ),
            "driftwood": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={"passive": "card_reward_reroll_once"},
                    ),
                ),
            ),
            "empty_cage": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "remove_deck_cards",
                        amount=2,
                        target_id="player",
                        metadata={"selection": "chosen"},
                    ),
                ),
            ),
            "electric_shrymp": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "modify_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={
                            "selection": "chosen",
                            "card_type": "skill",
                            "custom": {"enchant_keyword": "imbued"},
                        },
                    ),
                ),
            ),
            "fake_merchants_rug": RelicHookRule(
                markers=(RelicMarkerSpec("no_effect"),),
            ),
            "fake_lees_waffle": RelicHookRule(
                markers=(RelicMarkerSpec("heal_percent_max_hp", amount=10),),
            ),
            "fake_mango": RelicHookRule(
                max_hp_delta=3,
                markers=(RelicMarkerSpec("max_hp_delta"),),
            ),
            "golden_pearl": RelicHookRule(
                gold_delta=150,
                markers=(RelicMarkerSpec("gold_gained"),),
            ),
            "fishing_rod": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "normal_combat_count_random_deck_upgrade",
                            "combat_type": "normal",
                            "interval": 3,
                            "upgrade_count": 1,
                            "selection": "random",
                            "needed_subsystem": "combat_count_relic_trigger",
                        },
                    ),
                ),
            ),
            "golden_compass": RelicHookRule(
                markers=(RelicMarkerSpec("act2_map_replaced"),),
            ),
            "distinguished_cape": RelicHookRule(
                max_hp_delta=-9,
                markers=(
                    RelicMarkerSpec("max_hp_delta"),
                    RelicMarkerSpec(
                        "add_deck_cards",
                        amount=3,
                        target_id="player",
                        metadata={"card_id": "apparition"},
                    ),
                ),
            ),
            "dusty_tome": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "card_reward",
                        amount=1,
                        target_id="player",
                        metadata={"card_pool": "ancient", "selection": "random"},
                    ),
                ),
            ),
            "fragrant_mushroom": RelicHookRule(
                hp_delta=-15,
                markers=(
                    RelicMarkerSpec("hp_delta"),
                    RelicMarkerSpec(
                        "upgrade_deck_cards",
                        amount=2,
                        target_id="player",
                        metadata={"selection": "random"},
                    ),
                ),
            ),
            "fresnel_lens": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "deck_added_block_card_enchanted",
                            "enchant_keyword": "nimble",
                            "enchant_amount": 2,
                            "needed_subsystem": "deck_add_trigger",
                        },
                    ),
                ),
            ),
            "fur_coat": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "mark_map_rooms",
                        amount=7,
                        target_id="map",
                        metadata={
                            "room_kind": "combat",
                            "passive": "marked_combat_enemies_one_hp",
                            "enemy_hp": 1,
                            "needed_subsystem": "map_room_markers",
                        },
                    ),
                ),
            ),
            "ghost_seed": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "modify_deck_cards",
                        target_id="player",
                        metadata={
                            "match_card_ids": ("strike", "defend"),
                            "match_name_contains": ("strike", "defend"),
                            "custom": {"ethereal": True},
                        },
                    ),
                ),
            ),
            "glass_eye": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "card_reward",
                        amount=5,
                        target_id="player",
                        metadata={"rarities": ("common", "common", "uncommon", "uncommon", "rare")},
                    ),
                ),
            ),
            "glitter": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={"passive": "card_rewards_enchanted", "enchant_keyword": "glam"},
                    ),
                ),
            ),
            "gnarled_hammer": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "modify_deck_cards",
                        amount=3,
                        target_id="player",
                        metadata={
                            "selection": "chosen",
                            "card_type": "attack",
                            "custom": {"enchant_keyword": "sharp", "enchant_amount": 3},
                        },
                    ),
                ),
            ),
            "jewelry_box": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "add_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={"card_id": "apotheosis"},
                    ),
                ),
            ),
            "kaleidoscope": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "card_reward",
                        amount=2,
                        target_id="player",
                        metadata={
                            "card_pool": "other_character",
                            "selection": "choose",
                            "needed_subsystem": "cross_character_card_reward",
                        },
                    ),
                ),
            ),
            "hefty_tablet": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "card_reward",
                        amount=1,
                        target_id="player",
                        metadata={"rarity": "rare", "choices": 3, "selection": "choose"},
                    ),
                    RelicMarkerSpec(
                        "add_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={"card_id": "injury"},
                    ),
                ),
            ),
            "juzu_bracelet": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "question_rooms_no_regular_enemy_combats",
                            "needed_subsystem": "event_room_generation",
                        },
                    ),
                ),
            ),
            "kifuda": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "modify_deck_cards",
                        amount=3,
                        target_id="player",
                        metadata={
                            "selection": "chosen",
                            "custom": {"enchant_keyword": "adroit"},
                        },
                    ),
                ),
            ),
            "large_capsule": RelicHookRule(
                markers=(
                    RelicMarkerSpec("random_relics_gained", amount=2, target_id="player"),
                    RelicMarkerSpec(
                        "add_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={"card_id": "strike"},
                    ),
                    RelicMarkerSpec(
                        "add_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={"card_id": "defend"},
                    ),
                ),
            ),
            "lasting_candy": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "card_rewards_gain_additional_power_every_other_combat"
                        },
                    ),
                ),
            ),
            "leafy_poultice": RelicHookRule(
                max_hp_delta=-12,
                markers=(
                    RelicMarkerSpec("max_hp_delta"),
                    RelicMarkerSpec(
                        "transform_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={
                            "selection": "matching",
                            "match_card_ids": ("strike",),
                            "match_name_contains": ("strike",),
                        },
                    ),
                    RelicMarkerSpec(
                        "transform_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={
                            "selection": "matching",
                            "match_card_ids": ("defend",),
                            "match_name_contains": ("defend",),
                        },
                    ),
                ),
            ),
            "lees_waffle": RelicHookRule(
                max_hp_delta=7,
                markers=(RelicMarkerSpec("max_hp_delta"), RelicMarkerSpec("heal_to_full")),
            ),
            "lead_paperweight": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "card_reward",
                        amount=1,
                        target_id="player",
                        metadata={"card_color": "colorless", "choices": 2, "selection": "choose"},
                    ),
                ),
            ),
            "looming_fruit": RelicHookRule(
                max_hp_delta=31,
                markers=(RelicMarkerSpec("max_hp_delta"),),
            ),
            "lost_coffer": RelicHookRule(
                markers=(
                    RelicMarkerSpec("card_reward", amount=1, target_id="player"),
                    RelicMarkerSpec("random_potions_gained", amount=1, target_id="player"),
                ),
            ),
            "massive_scroll": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "card_reward",
                        amount=1,
                        target_id="player",
                        metadata={"card_pool": "multiplayer", "choices": 3, "selection": "choose"},
                    ),
                ),
            ),
            "lucky_fysh": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "deck_card_added_gold",
                            "gold": 15,
                            "needed_subsystem": "deck_add_trigger",
                        },
                    ),
                ),
            ),
            "mango": RelicHookRule(
                max_hp_delta=14,
                markers=(RelicMarkerSpec("max_hp_delta"),),
            ),
            "maw_bank": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "floor_climb_gold_until_shop_spend",
                            "gold": 12,
                            "needed_subsystem": "floor_transition_trigger",
                        },
                    ),
                ),
            ),
            "new_leaf": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "transform_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={"selection": "chosen"},
                    ),
                ),
            ),
            "neows_bones": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "random_relics_gained",
                        amount=2,
                        target_id="player",
                        metadata={"relic_pool": "neow"},
                    ),
                    RelicMarkerSpec("random_curses_gained", amount=1, target_id="player"),
                ),
            ),
            "neows_talisman": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "upgrade_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={"selection": "random", "match_card_ids": ("strike",)},
                    ),
                    RelicMarkerSpec(
                        "upgrade_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={"selection": "random", "match_card_ids": ("defend",)},
                    ),
                ),
            ),
            "neows_torment": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "add_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={"card_id": "neows_fury"},
                    ),
                ),
            ),
            "nutritious_oyster": RelicHookRule(
                max_hp_delta=11,
                markers=(RelicMarkerSpec("max_hp_delta"),),
            ),
            "nutritious_soup": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "modify_deck_cards",
                        amount=3,
                        target_id="player",
                        metadata={
                            "match_card_ids": ("strike",),
                            "match_name_contains": ("strike",),
                            "operation": "add_damage",
                            "custom": {"damage_bonus": 3},
                        },
                    ),
                ),
            ),
            "old_coin": RelicHookRule(
                gold_delta=300,
                markers=(RelicMarkerSpec("old_coin_gold_gained"),),
            ),
            "orrery": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "card_reward",
                        amount=5,
                        target_id="player",
                        metadata={"selection": "choose"},
                    ),
                ),
            ),
            "pear": RelicHookRule(
                max_hp_delta=10,
                markers=(RelicMarkerSpec("max_hp_delta"),),
            ),
            "planisphere": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "event_room_enter_heal",
                            "heal": 5,
                            "needed_subsystem": "room_enter_relic_trigger",
                        },
                    ),
                ),
            ),
            "paels_claw": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "modify_deck_cards",
                        target_id="player",
                        metadata={
                            "match_card_ids": ("defend",),
                            "match_name_contains": ("defend",),
                            "custom": {"enchant_keyword": "goopy"},
                        },
                    ),
                ),
            ),
            "paels_growth": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "modify_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={
                            "selection": "chosen",
                            "custom": {"enchant_keyword": "clone"},
                        },
                    ),
                ),
            ),
            "paels_horn": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "add_deck_cards",
                        amount=2,
                        target_id="player",
                        metadata={"card_id": "relax"},
                    ),
                ),
            ),
            "paels_wing": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={"passive": "sacrifice_card_rewards_for_relic_counter"},
                    ),
                ),
            ),
            "pandoras_box": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "transform_deck_cards",
                        target_id="player",
                        metadata={
                            "selection": "matching",
                            "match_card_ids": ("strike", "defend"),
                            "match_name_contains": ("strike", "defend"),
                        },
                    ),
                ),
            ),
            "pomander": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "upgrade_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={"selection": "chosen"},
                    ),
                ),
            ),
            "precarious_shears": RelicHookRule(
                hp_delta=-16,
                markers=(
                    RelicMarkerSpec("hp_delta"),
                    RelicMarkerSpec(
                        "remove_deck_cards",
                        amount=2,
                        target_id="player",
                        metadata={"selection": "chosen"},
                    ),
                ),
            ),
            "precise_scissors": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "remove_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={"selection": "chosen"},
                    ),
                ),
            ),
            "preserved_fog": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "remove_deck_cards",
                        amount=3,
                        target_id="player",
                        metadata={"selection": "chosen"},
                    ),
                    RelicMarkerSpec(
                        "add_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={"card_id": "folly"},
                    ),
                ),
            ),
            "punch_dagger": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "modify_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={
                            "selection": "chosen",
                            "card_type": "attack",
                            "custom": {"enchant_keyword": "momentum", "enchant_amount": 5},
                        },
                    ),
                ),
            ),
            "royal_stamp": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "modify_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={
                            "selection": "chosen",
                            "match_card_types": ("attack", "skill"),
                            "custom": {"enchant_keyword": "royally_approved"},
                        },
                    ),
                ),
            ),
            "sand_castle": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "upgrade_deck_cards",
                        amount=6,
                        target_id="player",
                        metadata={"selection": "random"},
                    ),
                ),
            ),
            "scroll_boxes": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "card_pack_reward",
                        amount=1,
                        target_id="player",
                        metadata={
                            "choices": 2,
                            "selection": "choose",
                            "needed_subsystem": "card_pack_rewards",
                        },
                    ),
                ),
            ),
            "silken_tress": RelicHookRule(
                markers=(
                    RelicMarkerSpec("set_gold", amount=0, target_id="player"),
                    RelicMarkerSpec(
                        "no_effect",
                        target_id="player",
                        metadata={
                            "passive": "first_card_reward_enchanted",
                            "enchant_keyword": "glam",
                            "card_reward_index": 1,
                            "needed_subsystem": "card_reward_enchant_trigger",
                        },
                    ),
                ),
            ),
            "sea_glass": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "card_reward",
                        amount=15,
                        target_id="player",
                        metadata={
                            "card_pool": "other_character",
                            "selection": "choose_any",
                            "needed_subsystem": "cross_character_card_reward",
                        },
                    ),
                ),
            ),
            "sere_talon": RelicHookRule(
                markers=(
                    RelicMarkerSpec("random_curses_gained", amount=2, target_id="player"),
                    RelicMarkerSpec(
                        "add_deck_cards",
                        amount=3,
                        target_id="player",
                        metadata={"card_id": "wish"},
                    ),
                ),
            ),
            "signet_ring": RelicHookRule(
                gold_delta=999,
                markers=(RelicMarkerSpec("gold_gained"),),
            ),
            "silver_crucible": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "relic_counter_changed",
                        amount=3,
                        target_id="player",
                        metadata={
                            "counter": 3,
                            "passive": "upgrade_first_card_rewards",
                            "first_treasure_empty": True,
                        },
                    ),
                ),
            ),
            "small_capsule": RelicHookRule(
                markers=(RelicMarkerSpec("random_relics_gained", amount=1, target_id="player"),),
            ),
            "storybook": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "add_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={"card_id": "brightest_flame"},
                    ),
                ),
            ),
            "touch_of_orobas": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "replace_starter_relic",
                        amount=1,
                        target_id="player",
                        metadata={
                            "replacement_pool": "ancient_starter_relics",
                            "needed_subsystem": "starter_relic_upgrade",
                        },
                    ),
                ),
            ),
            "toy_box": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "random_relics_gained",
                        amount=4,
                        target_id="player",
                        metadata={"relic_pool": "wax"},
                    ),
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "wax_relic_melts_every_combats",
                            "combats": 3,
                            "needed_subsystem": "combat_count_relic_decay",
                        },
                    ),
                ),
            ),
            "tri_boomerang": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "modify_deck_cards",
                        amount=3,
                        target_id="player",
                        metadata={
                            "selection": "chosen",
                            "card_type": "attack",
                            "custom": {"enchant_keyword": "instinct"},
                        },
                    ),
                ),
            ),
            "strawberry": RelicHookRule(
                max_hp_delta=7,
                markers=(RelicMarkerSpec("max_hp_delta"),),
            ),
            "tanxs_whistle": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "add_deck_cards",
                        amount=1,
                        target_id="player",
                        metadata={"card_id": "whistle"},
                    ),
                ),
            ),
            "white_star": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={"passive": "elite_rare_card_reward_delta"},
                    ),
                ),
            ),
            "wing_charm": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "card_reward_random_card_enchanted",
                            "enchant_keyword": "swift",
                            "enchant_amount": 1,
                        },
                    ),
                ),
            ),
            "winged_boots": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "relic_counter_changed",
                        amount=3,
                        target_id="player",
                        metadata={
                            "counter": 3,
                            "passive": "ignore_map_paths",
                            "needed_subsystem": "map_path_override",
                        },
                    ),
                ),
            ),
            "war_paint": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "upgrade_deck_cards",
                        amount=2,
                        target_id="player",
                        metadata={"selection": "random", "card_type": "skill"},
                    ),
                ),
            ),
            "whetstone": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "upgrade_deck_cards",
                        amount=2,
                        target_id="player",
                        metadata={"selection": "random", "card_type": "attack"},
                    ),
                ),
            ),
            "wongo_customer_appreciation_badge": RelicHookRule(
                markers=(RelicMarkerSpec("no_effect"),),
            ),
            "wongos_mystery_ticket": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "relic_counter_changed",
                        amount=5,
                        target_id="player",
                        metadata={
                            "counter": 5,
                            "passive": "delayed_random_relics_after_combats",
                            "random_relics": 3,
                            "needed_subsystem": "combat_count_delayed_rewards",
                        },
                    ),
                ),
            ),
            "yummy_cookie": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "upgrade_deck_cards",
                        amount=4,
                        target_id="player",
                        metadata={"selection": "chosen"},
                    ),
                ),
            ),
            "potion_belt": RelicHookRule(
                potion_slot_delta=2,
                markers=(RelicMarkerSpec("potion_slots_gained"),),
            ),
            "alchemical_coffer": RelicHookRule(
                potion_slot_delta=4,
                markers=(
                    RelicMarkerSpec(
                        "potion_slots_gained",
                        metadata={"fill_random_potions": 4},
                    ),
                ),
            ),
            "phial_holster": RelicHookRule(
                potion_slot_delta=1,
                markers=(
                    RelicMarkerSpec(
                        "potion_slots_gained",
                        metadata={"fill_random_potions": 2},
                    ),
                ),
            ),
        },
        RelicHook.SHOP_ENTER: {
            "lords_parasol": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "lords_parasol_claimed_shop",
                        target_id="shop",
                        metadata={
                            "engine_handler": "shop_entry_claim_all_non_service_items",
                        },
                    ),
                ),
            ),
            "meal_ticket": RelicHookRule(
                hp_delta=15,
                markers=(RelicMarkerSpec("meal_ticket_healed"),),
            ),
            "the_courier": RelicHookRule(
                markers=(RelicMarkerSpec("shop_restock_enabled"),),
            ),
            "courier": RelicHookRule(
                markers=(RelicMarkerSpec("shop_restock_enabled"),),
            ),
        },
        RelicHook.SHOP_PURCHASE: {
            "maw_bank": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "maw_bank_disabled",
                        target_id="player",
                        metadata={
                            "passive": "floor_climb_gold_until_shop_spend",
                            "needed_subsystem": "shop_purchase_relic_state",
                        },
                    ),
                ),
            ),
            "the_courier": RelicHookRule(
                markers=(RelicMarkerSpec("shop_restock_purchased_item"),),
            ),
            "courier": RelicHookRule(
                markers=(RelicMarkerSpec("shop_restock_purchased_item"),),
            ),
        },
        RelicHook.CAMPFIRE_ENTER: {
            "dream_catcher": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "rest_add_card_reward",
                            "needed_subsystem": "campfire_rest_action",
                        },
                    ),
                ),
            ),
            "eternal_feather": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "rest_site_enter_heal_per_deck_cards",
                            "cards_per_heal": 5,
                            "heal": 3,
                            "needed_subsystem": "campfire_enter_deck_size",
                        },
                    ),
                ),
            ),
            "girya": RelicHookRule(
                markers=(RelicMarkerSpec("campfire_lift_unlocked"),),
            ),
            "meat_cleaver": RelicHookRule(
                markers=(RelicMarkerSpec("campfire_cook_unlocked"),),
            ),
            "miniature_tent": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "campfire_multi_action_unlocked",
                        metadata={
                            "passive": "campfire_choose_any_number_of_options",
                            "needed_subsystem": "campfire_multi_action_resolution",
                        },
                    ),
                ),
            ),
            "peace_pipe": RelicHookRule(
                markers=(RelicMarkerSpec("campfire_toke_unlocked"),),
            ),
            "regal_pillow": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "rest_heal_bonus",
                            "heal": 15,
                            "needed_subsystem": "campfire_rest_action",
                        },
                    ),
                ),
            ),
            "shovel": RelicHookRule(
                markers=(RelicMarkerSpec("campfire_dig_unlocked"),),
            ),
            "stone_humidifier": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "rest_max_hp_delta",
                            "max_hp_delta": 5,
                            "needed_subsystem": "campfire_rest_action",
                        },
                    ),
                ),
            ),
            "tiny_mailbox": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "no_effect",
                        metadata={
                            "passive": "rest_random_potions",
                            "random_potions": 2,
                            "needed_subsystem": "campfire_rest_action",
                        },
                    ),
                ),
            ),
            "venerable_tea_set": RelicHookRule(
                markers=(RelicMarkerSpec("next_combat_energy", amount=2, target_id="player"),),
            ),
            "fake_venerable_tea_set": RelicHookRule(
                markers=(RelicMarkerSpec("next_combat_energy", amount=1, target_id="player"),),
            ),
        },
        RelicHook.START_COMBAT: {
            "akabeko": RelicHookRule(
                markers=(RelicMarkerSpec("gain_status", amount=8, metadata={"status": "vigor"}),),
            ),
            "anchor": RelicHookRule(
                markers=(RelicMarkerSpec("gain_block", amount=10, target_id="player"),),
            ),
            "fake_anchor": RelicHookRule(
                markers=(RelicMarkerSpec("gain_block", amount=4, target_id="player"),),
            ),
            "bag_of_marbles": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "apply_status",
                        amount=1,
                        target_id="all_enemies",
                        metadata={"status": "vulnerable"},
                    ),
                ),
            ),
            "bag_of_preparation": RelicHookRule(
                markers=(RelicMarkerSpec("draw_cards", amount=2, target_id="player"),),
            ),
            "big_mushroom": RelicHookRule(
                markers=(RelicMarkerSpec("draw_cards", amount=-2, target_id="player"),),
            ),
            "blessed_antler": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "shuffle_status_into_draw_pile",
                        amount=3,
                        target_id="player",
                        metadata={"card_id": "dazed"},
                    ),
                ),
            ),
            "blood_vial": RelicHookRule(
                hp_delta=2,
                markers=(RelicMarkerSpec("blood_vial_healed"),),
            ),
            "bronze_scales": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "gain_status",
                        amount=3,
                        target_id="player",
                        metadata={"status": "thorns"},
                    ),
                ),
            ),
            "cracked_core": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "channel_orb",
                        amount=1,
                        target_id="player",
                        metadata={"orb": "lightning"},
                    ),
                ),
            ),
            "data_disk": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "gain_status",
                        amount=1,
                        target_id="player",
                        metadata={"status": "focus"},
                    ),
                ),
            ),
            "divine_destiny": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "player_resource",
                        amount=6,
                        target_id="player",
                        metadata={"resource": "star"},
                    ),
                ),
            ),
            "divine_right": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "player_resource",
                        amount=3,
                        target_id="player",
                        metadata={"resource": "star"},
                    ),
                ),
            ),
            "fencing_manual": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "player_resource",
                        amount=10,
                        target_id="player",
                        metadata={"resource": "forge"},
                    ),
                ),
            ),
            "fake_blood_vial": RelicHookRule(
                hp_delta=1,
                markers=(RelicMarkerSpec("blood_vial_healed"),),
            ),
            "festive_popper": RelicHookRule(
                markers=(RelicMarkerSpec("all_damage", amount=9, target_id="all_enemies"),),
            ),
            "gorget": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "gain_status",
                        amount=4,
                        target_id="player",
                        metadata={"status": "plated_armor"},
                    ),
                ),
            ),
            "infused_core": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "channel_orb",
                        amount=3,
                        target_id="player",
                        metadata={"orb": "lightning", "lightning_damage_bonus": 1},
                    ),
                ),
            ),
            "lantern": RelicHookRule(
                markers=(RelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
            ),
            "jeweled_mask": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "move_card_type_from_draw_to_hand",
                        amount=1,
                        target_id="player",
                        metadata={"card_type": "power", "free_to_play_this_turn": True},
                    ),
                ),
            ),
            "oddly_smooth_stone": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "gain_status",
                        amount=1,
                        target_id="player",
                        metadata={"status": "dexterity"},
                    ),
                ),
            ),
            "philosophers_stone": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "apply_status",
                        amount=1,
                        target_id="all_enemies",
                        metadata={"status": "strength"},
                    ),
                ),
            ),
            "phylactery_unbound": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "player_resource",
                        amount=5,
                        target_id="player",
                        metadata={"resource": "summon"},
                    ),
                ),
            ),
            "power_cell": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "move_zero_cost_cards_to_hand",
                        amount=2,
                        target_id="player",
                        metadata={"free_to_play_this_turn": True},
                    ),
                ),
            ),
            "radiant_pearl": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "add_card_to_hand",
                        amount=1,
                        target_id="player",
                        metadata={
                            "card_id": "luminesce",
                            "card_type": "skill",
                            "target": "self",
                        },
                    ),
                ),
            ),
            "ring_of_the_snake": RelicHookRule(
                markers=(RelicMarkerSpec("draw_cards", amount=2, target_id="player"),),
            ),
            "royal_poison": RelicHookRule(
                markers=(RelicMarkerSpec("lose_hp", amount=4, target_id="player"),),
            ),
            "runic_capacitor": RelicHookRule(
                markers=(RelicMarkerSpec("orb_slot_delta", amount=3, target_id="player"),),
            ),
            "snecko_eye": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "gain_status",
                        amount=1,
                        target_id="player",
                        metadata={"status": "confused"},
                    ),
                ),
            ),
            "fake_snecko_eye": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "gain_status",
                        amount=1,
                        target_id="player",
                        metadata={"status": "confused"},
                    ),
                ),
            ),
            "ninja_scroll": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "add_card_to_hand",
                        amount=3,
                        target_id="player",
                        metadata={"card_id": "shiv", "card_type": "attack", "target": "enemy"},
                    ),
                ),
            ),
            "funerary_mask": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "add_card_to_draw_pile",
                        amount=3,
                        target_id="player",
                        metadata={"card_id": "soul", "card_type": "skill", "target": "self"},
                    ),
                ),
            ),
            "red_mask": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "apply_status",
                        amount=1,
                        target_id="all_enemies",
                        metadata={"status": "weak"},
                    ),
                ),
            ),
            "twisted_funnel": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "apply_status",
                        amount=4,
                        target_id="all_enemies",
                        metadata={"status": "poison"},
                    ),
                ),
            ),
            "symbiotic_virus": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "channel_orb",
                        amount=1,
                        target_id="player",
                        metadata={"orb": "dark"},
                    ),
                ),
            ),
            "vajra": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "gain_status",
                        amount=1,
                        target_id="player",
                        metadata={"status": "strength"},
                    ),
                ),
            ),
            "sword_of_jade": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "gain_status",
                        amount=3,
                        target_id="player",
                        metadata={"status": "strength"},
                    ),
                ),
            ),
            "stone_cracker": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "upgrade_draw_pile_cards",
                        amount=2,
                        target_id="player",
                    ),
                ),
            ),
            "very_hot_cocoa": RelicHookRule(
                markers=(RelicMarkerSpec("gain_energy", amount=4, target_id="player"),),
            ),
        },
        RelicHook.START_TURN: {
            "blessed_antler": RelicHookRule(
                markers=(RelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
            ),
            "bound_phylactery": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "player_resource",
                        amount=1,
                        target_id="player",
                        metadata={"resource": "summon"},
                    ),
                ),
            ),
            "blood_soaked_rose": RelicHookRule(
                markers=(RelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
            ),
            "brimstone": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "gain_status",
                        amount=2,
                        target_id="player",
                        metadata={"status": "strength"},
                    ),
                    RelicMarkerSpec(
                        "apply_status",
                        amount=1,
                        target_id="all_enemies",
                        metadata={"status": "strength"},
                    ),
                ),
            ),
            "ectoplasm": RelicHookRule(
                markers=(RelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
            ),
            "mercury_hourglass": RelicHookRule(
                markers=(RelicMarkerSpec("all_damage", amount=3, target_id="all_enemies"),),
            ),
            "paels_blood": RelicHookRule(
                markers=(RelicMarkerSpec("draw_cards", amount=1, target_id="player"),),
            ),
            "philosophers_stone": RelicHookRule(
                markers=(RelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
            ),
            "phylactery_unbound": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "player_resource",
                        amount=1,
                        target_id="player",
                        metadata={"resource": "summon"},
                    ),
                ),
            ),
            "prismatic_gem": RelicHookRule(
                markers=(RelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
            ),
            "pumpkin_candle": RelicHookRule(
                markers=(RelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
            ),
            "sai": RelicHookRule(
                markers=(RelicMarkerSpec("gain_block", amount=7, target_id="player"),),
            ),
            "seal_of_gold": RelicHookRule(
                markers=(
                    RelicMarkerSpec("gold_delta", amount=-5, target_id="player"),
                    RelicMarkerSpec("gain_energy", amount=1, target_id="player"),
                ),
            ),
            "snecko_eye": RelicHookRule(
                markers=(RelicMarkerSpec("draw_cards", amount=2, target_id="player"),),
            ),
            "sozu": RelicHookRule(
                markers=(RelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
            ),
            "spiked_gauntlets": RelicHookRule(
                markers=(RelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
            ),
            "toasty_mittens": RelicHookRule(
                markers=(
                    RelicMarkerSpec("exhaust_top_draw_pile", amount=1, target_id="player"),
                    RelicMarkerSpec(
                        "gain_status",
                        amount=1,
                        target_id="player",
                        metadata={"status": "strength"},
                    ),
                ),
            ),
            "velvet_choker": RelicHookRule(
                markers=(
                    RelicMarkerSpec("gain_energy", amount=1, target_id="player"),
                    RelicMarkerSpec(
                        "turn_card_play_limit",
                        amount=6,
                        target_id="player",
                    ),
                ),
            ),
            "whispering_earring": RelicHookRule(
                markers=(RelicMarkerSpec("gain_energy", amount=1, target_id="player"),),
            ),
            "pendulum": RelicHookRule(
                markers=(RelicMarkerSpec("draw_cards", amount=1, target_id="player"),),
            ),
            "pollinous_core": RelicHookRule(
                markers=(RelicMarkerSpec("draw_cards", amount=2, target_id="player"),),
            ),
        },
        RelicHook.END_TURN: {
            "cloak_clasp": RelicHookRule(
                markers=(RelicMarkerSpec("gain_block", target_id="player"),),
            ),
            "paels_tears": RelicHookRule(
                markers=(
                    RelicMarkerSpec(
                        "gain_status",
                        amount=2,
                        target_id="player",
                        metadata={"status": "next_turn_energy"},
                    ),
                ),
            ),
            "runic_pyramid": RelicHookRule(
                markers=(RelicMarkerSpec("retain_hand", amount=1, target_id="player"),),
            ),
            "screaming_flagon": RelicHookRule(
                markers=(RelicMarkerSpec("all_damage", amount=20, target_id="all_enemies"),),
            ),
            "stone_calendar": RelicHookRule(
                markers=(RelicMarkerSpec("all_damage", amount=52, target_id="all_enemies"),),
            ),
        },
        RelicHook.END_COMBAT: {
            "burning_blood": RelicHookRule(
                hp_delta=6,
                markers=(RelicMarkerSpec("burning_blood_healed"),),
            ),
            "black_blood": RelicHookRule(
                hp_delta=12,
                markers=(RelicMarkerSpec("black_blood_healed"),),
            ),
            "chosen_cheese": RelicHookRule(
                max_hp_delta=1,
                markers=(RelicMarkerSpec("max_hp_delta", amount=1, target_id="player"),),
            ),
        },
    }


def _default_price_modifiers() -> dict[str, RelicPriceModifier]:
    return {
        "membership_card": RelicPriceModifier("membership_card", multiplier_percent=50),
        "the_courier": RelicPriceModifier("the_courier", multiplier_percent=80),
        "courier": RelicPriceModifier("courier", multiplier_percent=80),
        "smiling_mask": RelicPriceModifier(
            "smiling_mask",
            fixed_price=50,
            item_kinds=frozenset({"card_removal"}),
        ),
    }


DEFAULT_RELIC_HOOK_RULES = _default_relic_hook_rules()
DEFAULT_RELIC_PRICE_MODIFIERS = _default_price_modifiers()
DEFAULT_ENGINE_RELIC_IDS = frozenset(
    {
        "black_star",
        "frozen_egg",
        "molten_egg",
        "toxic_egg",
    }
)
DEFAULT_RELIC_POTION_SLOT_MODIFIERS = {
    "potion_belt": 2,
    "alchemical_coffer": 4,
    "phial_holster": 1,
}


def resolve_relic_pickup(
    relic: RelicInput,
    *,
    hp: int | None = None,
    max_hp: int | None = None,
    rules: Mapping[RelicHook, Mapping[str, RelicHookRule]] = DEFAULT_RELIC_HOOK_RULES,
) -> RelicHookResult:
    """Resolve a single relic pickup into deterministic deltas and markers."""

    relic_id = relic_content_id(relic)
    rule = rules.get(RelicHook.PICKUP, {}).get(relic_id)
    if rule is None:
        return RelicHookResult(
            hook=RelicHook.PICKUP,
            relic_id=relic_id,
            unsupported=True,
        )
    return _result_from_rule(
        relic_id,
        RelicHook.PICKUP,
        rule,
        hp=hp,
        max_hp=max_hp,
    )


def resolve_relic_hook(
    relics: Sequence[RelicInput],
    hook: RelicHook,
    *,
    hp: int | None = None,
    max_hp: int | None = None,
    rules: Mapping[RelicHook, Mapping[str, RelicHookRule]] = DEFAULT_RELIC_HOOK_RULES,
) -> RelicHookResolution:
    """Resolve all relics that have a handler for ``hook``."""

    hook_rules = rules.get(hook, {})
    results: list[RelicHookResult] = []
    current_hp = hp
    for relic_id in _unique_relic_ids(relics):
        rule = hook_rules.get(relic_id)
        if rule is None:
            continue
        result = _result_from_rule(
            relic_id,
            hook,
            rule,
            hp=current_hp,
            max_hp=max_hp,
        )
        results.append(result)
        if current_hp is not None:
            current_hp += result.hp_delta
    return _combine_hook_results(hook, tuple(results))


def apply_relic_price_modifiers(
    base_price: int,
    item_kind: str | Enum,
    relics: Sequence[RelicInput],
    *,
    min_price: int = 0,
    modifiers: Mapping[str, RelicPriceModifier] = DEFAULT_RELIC_PRICE_MODIFIERS,
) -> RelicPriceResult:
    """Apply fixed and percentage shop-price relic modifiers."""

    normalized_kind = _normalized_id(_enum_value(item_kind))
    normalized_base = max(0, int(base_price))
    applied: list[str] = []
    fixed_prices: list[tuple[str, int]] = []
    multiplier_percent = 100

    for relic_id in _unique_relic_ids(relics):
        modifier = modifiers.get(relic_id)
        if modifier is None or not _modifier_applies(modifier, normalized_kind):
            continue
        if modifier.fixed_price is not None:
            fixed_prices.append((relic_id, modifier.fixed_price))
            continue
        if modifier.multiplier_percent is None:
            continue
        applied.append(relic_id)
        multiplier_percent *= max(0, modifier.multiplier_percent)
        multiplier_percent //= 100

    if fixed_prices:
        relic_id, fixed_price = min(fixed_prices, key=lambda item: item[1])
        return RelicPriceResult(
            item_kind=normalized_kind,
            base_price=normalized_base,
            price=max(min_price, fixed_price),
            applied_relic_ids=(relic_id,),
            fixed_price=fixed_price,
        )

    price = normalized_base * multiplier_percent // 100
    return RelicPriceResult(
        item_kind=normalized_kind,
        base_price=normalized_base,
        price=max(min_price, price),
        applied_relic_ids=tuple(applied),
        multiplier_percent=multiplier_percent,
    )


def relic_potion_slot_bonus(
    relics: Sequence[RelicInput],
    *,
    modifiers: Mapping[str, int] = DEFAULT_RELIC_POTION_SLOT_MODIFIERS,
) -> int:
    """Return additional potion slots granted by owned relics."""

    return sum(max(0, int(modifiers.get(relic_id, 0))) for relic_id in _unique_relic_ids(relics))


def supported_relic_ids(
    hook: RelicHook | None = None,
    *,
    rules: Mapping[RelicHook, Mapping[str, RelicHookRule]] = DEFAULT_RELIC_HOOK_RULES,
    price_modifiers: Mapping[str, RelicPriceModifier] = DEFAULT_RELIC_PRICE_MODIFIERS,
    potion_slot_modifiers: Mapping[str, int] = DEFAULT_RELIC_POTION_SLOT_MODIFIERS,
    engine_relic_ids: frozenset[str] = DEFAULT_ENGINE_RELIC_IDS,
) -> frozenset[str]:
    """Return relic ids with at least one explicit helper."""

    if hook is RelicHook.SHOP_PRICE:
        return frozenset(price_modifiers)
    if hook is RelicHook.POTION_CAPACITY:
        return frozenset(potion_slot_modifiers)
    if hook is not None:
        return frozenset(rules.get(hook, {}))

    supported = set(price_modifiers) | set(potion_slot_modifiers) | set(engine_relic_ids)
    for hook_rules in rules.values():
        supported.update(hook_rules)
    return frozenset(supported)


def unsupported_relic_handlers(
    relics: Sequence[RelicInput],
    *,
    hooks: Sequence[RelicHook] | None = None,
    rules: Mapping[RelicHook, Mapping[str, RelicHookRule]] = DEFAULT_RELIC_HOOK_RULES,
    price_modifiers: Mapping[str, RelicPriceModifier] = DEFAULT_RELIC_PRICE_MODIFIERS,
    potion_slot_modifiers: Mapping[str, int] = DEFAULT_RELIC_POTION_SLOT_MODIFIERS,
) -> tuple[UnsupportedRelicHandler, ...]:
    """Report inferred relic hooks without a bounded helper implementation."""

    unsupported: list[UnsupportedRelicHandler] = []
    supported_by_hook = {
        hook: supported_relic_ids(
            hook,
            rules=rules,
            price_modifiers=price_modifiers,
            potion_slot_modifiers=potion_slot_modifiers,
        )
        for hook in RelicHook
    }
    for relic in relics:
        relic_id = relic_content_id(relic)
        needed_hooks = tuple(hooks) if hooks is not None else _inferred_hooks(relic)
        missing = tuple(
            hook
            for hook in needed_hooks
            if relic_id not in supported_by_hook.get(hook, frozenset())
        )
        if missing:
            unsupported.append(
                UnsupportedRelicHandler(
                    relic_id=relic_id,
                    name=_content_str(relic, "name"),
                    unsupported_hooks=missing,
                    description=_content_str(relic, "description", "description_raw"),
                )
            )
    return tuple(unsupported)


def relic_content_id(relic: RelicInput) -> str:
    """Return a normalized relic id from a raw id or Codex-style mapping."""

    if isinstance(relic, str):
        return _normalized_id(relic)
    value = _first_present(relic, "id", "relic_id", "content_id", "item_id")
    if value is None:
        raise ValueError(f"Relic input is missing an id: {relic!r}")
    return _normalized_id(str(value))


def _result_from_rule(
    relic_id: str,
    hook: RelicHook,
    rule: RelicHookRule,
    *,
    hp: int | None,
    max_hp: int | None,
) -> RelicHookResult:
    hp_delta = rule.hp_delta
    if hp_delta > 0 and hp is not None and max_hp is not None:
        hp_delta = max(0, min(hp_delta, max_hp - hp))
    markers = tuple(
        RelicEffectMarker(
            kind=marker.kind,
            relic_id=relic_id,
            amount=_marker_amount(
                marker,
                gold_delta=rule.gold_delta,
                hp_delta=hp_delta,
                max_hp_delta=rule.max_hp_delta,
                potion_slot_delta=rule.potion_slot_delta,
            ),
            target_id=marker.target_id,
            metadata=marker.metadata,
        )
        for marker in rule.markers
    )
    return RelicHookResult(
        hook=hook,
        relic_id=relic_id,
        gold_delta=rule.gold_delta,
        hp_delta=hp_delta,
        max_hp_delta=rule.max_hp_delta,
        potion_slot_delta=rule.potion_slot_delta,
        markers=markers,
        source=rule.source,
    )


def _combine_hook_results(
    hook: RelicHook,
    results: tuple[RelicHookResult, ...],
) -> RelicHookResolution:
    return RelicHookResolution(
        hook=hook,
        results=results,
        gold_delta=sum(result.gold_delta for result in results),
        hp_delta=sum(result.hp_delta for result in results),
        max_hp_delta=sum(result.max_hp_delta for result in results),
        potion_slot_delta=sum(result.potion_slot_delta for result in results),
        markers=tuple(marker for result in results for marker in result.markers),
    )


def _marker_amount(
    marker: RelicMarkerSpec,
    *,
    gold_delta: int,
    hp_delta: int,
    max_hp_delta: int,
    potion_slot_delta: int,
) -> int | None:
    if marker.amount is not None:
        return marker.amount
    for amount in (gold_delta, hp_delta, max_hp_delta, potion_slot_delta):
        if amount:
            return amount
    return None


def _modifier_applies(modifier: RelicPriceModifier, item_kind: str) -> bool:
    return not modifier.item_kinds or item_kind in modifier.item_kinds


def _unique_relic_ids(relics: Sequence[RelicInput]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for relic in relics:
        relic_id = relic_content_id(relic)
        if relic_id in seen:
            continue
        seen.add(relic_id)
        normalized.append(relic_id)
    return tuple(normalized)


def _inferred_hooks(relic: RelicInput) -> tuple[RelicHook, ...]:
    description = _content_str(relic, "description", "description_raw")
    if not description:
        return ()
    text = description.lower()
    hooks: list[RelicHook] = []
    if "upon pickup" in text:
        hooks.append(RelicHook.PICKUP)
    if "enter a shop" in text or "enter a shop room" in text:
        hooks.append(RelicHook.SHOP_ENTER)
    if "merchant" in text or "prices" in text or "discount" in text:
        hooks.append(RelicHook.SHOP_PRICE)
    if (
        "potion slot" in text
        and ("upon pickup" in text or "gain" in text)
        and "empty potion slots" not in text
    ):
        hooks.append(RelicHook.POTION_CAPACITY)
    if "enter a rest site" in text:
        hooks.append(RelicHook.CAMPFIRE_ENTER)
    if "start of each combat" in text or "start each combat" in text:
        hooks.append(RelicHook.START_COMBAT)
    if "start of each turn" in text:
        hooks.append(RelicHook.START_TURN)
    if "end your turn" in text or "end of your turn" in text or "end of turn" in text:
        hooks.append(RelicHook.END_TURN)
    if "end of combat" in text:
        hooks.append(RelicHook.END_COMBAT)
    return tuple(dict.fromkeys(hooks))


def _content_str(item: RelicInput, *keys: str) -> str | None:
    if isinstance(item, str):
        return None
    value = _first_present(item, *keys)
    return None if value is None else str(value)


def _first_present(item: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        value = item.get(key)
        if value is not None:
            return value
    return None


def _enum_value(value: str | Enum) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _normalized_id(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("'", "")
        .replace(" ", "_")
        .replace("-", "_")
    )
