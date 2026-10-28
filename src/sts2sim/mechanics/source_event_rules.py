"""Typed option contracts transcribed from the pinned stable event callbacks.

Source snapshot and names match mechanics.source_events. These rules bypass
printed-text parsing for deck choices and immediate effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .source_events import EventNumericEffect


@dataclass(frozen=True)
class SourceEventRule:
    numeric: EventNumericEffect = field(default_factory=EventNumericEffect)
    selection: str = ""
    count: int = 1
    tag: str = ""
    basic_only: bool = False
    target_card: str = ""
    add_cards: tuple[str, ...] = ()
    add_relics: tuple[str, ...] = ()
    random_relic: bool = False
    relic_pool: tuple[str, ...] = ()
    card_pool: tuple[str, ...] = ()
    required_gold: int = 0
    lose_all_gold: bool = False
    heal_to_full: bool = False
    duplicate_deck: bool = False
    downgrade_random: int = 0
    upgrade_random: int = 0
    stable_upgrade_shuffle: bool = False
    extra_rng_draws: int = 0
    fixed_potion_rewards: tuple[str, ...] = ()
    uniform_potion_reward: bool = False
    potion_rarity: str = ""
    selection_first: bool = False
    enchant: str = ""
    enchant_amount: int = 1
    card_type: str = ""
    card_cost: int | None = None
    generated_count: int = 0
    generated_choose: int = 0
    generated_rarity: str = ""
    generated_color: str = ""
    generated_uniform: bool = False
    generated_upgrade_roll: bool = False
    generated_reward: bool = False
    remove_potion_index: int | None = None
    remove_relic: str = ""
    random_relic_count: int = 1
    doll_offers: int = 0

    def payload(self) -> dict[str, object]:
        return asdict(self)


N = EventNumericEffect
R = SourceEventRule
SOURCE_EVENT_RULES: dict[str, dict[str, SourceEventRule]] = {
    "AMALGAMATOR": {
        "COMBINE_STRIKES": R(
            selection="remove",
            count=2,
            tag="strike",
            basic_only=True,
            add_cards=("ultimate_strike",),
        ),
        "COMBINE_DEFENDS": R(
            selection="remove",
            count=2,
            tag="defend",
            basic_only=True,
            add_cards=("ultimate_defend",),
        ),
    },
    "AROMA_OF_CHAOS": {
        "LET_GO": R(selection="transform"),
        "MAINTAIN_CONTROL": R(selection="upgrade"),
    },
    "BRAIN_LEECH": {
        "SHARE_KNOWLEDGE": R(generated_count=5, generated_choose=1),
        "RIP": R(N(hp_loss=5), generated_count=3, generated_color="colorless",
                 generated_reward=True),
    },
    "INFESTED_AUTOMATON": {
        "STUDY": R(generated_count=1, card_type="power"),
        "TOUCH_CORE": R(generated_count=1, card_cost=0),
    },
    "BUGSLAYER": {
        "EXTERMINATION": R(add_cards=("exterminate",)),
        "SQUASH": R(add_cards=("squash",)),
    },
    "BYRDONIS_NEST": {"EAT": R(N(max_hp_delta=7)), "TAKE": R(add_cards=("byrdonis_egg",))},
    "DOLL_ROOM": {
        "RANDOM": R(relic_pool=("daughter_of_the_wind", "mr_struggles", "bing_bong")),
        "TAKE_SOME_TIME": R(N(hp_loss=5), doll_offers=2),
        "EXAMINE": R(N(hp_loss=15), doll_offers=3),
    },
    "DOORS_OF_LIGHT_AND_DARK": {
        "LIGHT": R(upgrade_random=2, stable_upgrade_shuffle=True),
        "DARK": R(selection="remove"),
    },
    "DROWNING_BEACON": {"CLIMB": R(N(max_hp_delta=-13), add_relics=("fresnel_lens",))},
    "FIELD_OF_MAN_SIZED_HOLES": {
        "RESIST": R(selection="remove", count=2, add_cards=("normality",))
    },
    "GRAVE_OF_THE_FORGOTTEN": {"ACCEPT": R(add_relics=("forgotten_soul",))},
    "HUNGRY_FOR_MUSHROOMS": {
        "BIG_MUSHROOM": R(add_relics=("big_mushroom",)),
        "FRAGRANT_MUSHROOM": R(add_relics=("fragrant_mushroom",)),
    },
    "LOST_WISP": {"CLAIM": R(add_cards=("decay",), add_relics=("lost_wisp",))},
    "LUMINOUS_CHOIR": {
        "REACH_INTO_THE_FLESH": R(selection="remove", count=2, add_cards=("spore_mind",)),
        "OFFER_TRIBUTE": R(random_relic=True),
    },
    "MORPHIC_GROVE": {
        "GROUP": R(selection="transform", count=2, lose_all_gold=True),
        "LONER": R(N(max_hp_delta=5)),
    },
    "POTION_COURIER": {
        "GRAB_POTIONS": R(fixed_potion_rewards=("foul_potion",) * 3),
        "RANSACK": R(uniform_potion_reward=True, potion_rarity="uncommon"),
    },
    "REFLECTIONS": {
        "TOUCH_A_MIRROR": R(downgrade_random=2, upgrade_random=4),
        "SHATTER": R(add_cards=("bad_luck",), duplicate_deck=True),
    },
    "ROOM_FULL_OF_CHEESE": {
        "SEARCH": R(N(hp_loss=14), add_relics=("chosen_cheese",)),
        "GORGE": R(generated_count=8, generated_choose=2,
                   generated_uniform=True, generated_rarity="common"),
    },
    "ROUND_TEA_PARTY": {
        "ENJOY_TEA": R(add_relics=("royal_poison",), heal_to_full=True),
        "PICK_FIGHT": R(N(next_page="PICK_FIGHT")),
        "CONTINUE_FIGHT": R(N(hp_loss=11), random_relic=True),
    },
    "SAPPHIRE_SEED": {"EAT": R(N(heal=9), selection="upgrade"),
                      "PLANT": R(selection="enchant", enchant="Sown")},
    "SPIRALING_WHIRLPOOL": {"OBSERVE": R(selection="enchant", enchant="Spiral")},
    "STONE_OF_ALL_TIME": {
        "LIFT": R(N(max_hp_delta=10), extra_rng_draws=1),
        "PUSH": R(N(hp_loss=6), selection="enchant", enchant="Vigorous",
                  enchant_amount=8, extra_rng_draws=1),
    },
    "SELF_HELP_BOOK": {
        "READ_THE_BACK": R(selection="enchant", enchant="Sharp", enchant_amount=2,
                           card_type="attack"),
        "READ_PASSAGE": R(selection="enchant", enchant="Nimble", enchant_amount=2,
                          card_type="skill"),
        "READ_ENTIRE_BOOK": R(selection="enchant", enchant="Swift", enchant_amount=2,
                              card_type="power"),
    },
    "SPIRIT_GRAFTER": {
        "LET_IT_IN": R(N(heal=25), add_cards=("metamorphosis",)),
        "REJECTION": R(N(hp_loss=10), selection="upgrade", selection_first=True),
    },
    "SUNKEN_STATUE": {"GRAB_SWORD": R(add_relics=("sword_of_stone",))},
    "SUNKEN_TREASURY": {"SECOND_CHEST": R(add_cards=("greed",))},
    "SYMBIOTE": {"KILL_WITH_FIRE": R(selection="transform"),
                 "APPROACH": R(selection="enchant", enchant="Corrupted")},
    "TEA_MASTER": {
        "BONE_TEA": R(N(gold_delta=-50), add_relics=("bone_tea",), required_gold=50),
        "EMBER_TEA": R(N(gold_delta=-150), add_relics=("ember_tea",), required_gold=150),
        "TEA_OF_DISCOURTESY": R(add_relics=("tea_of_discourtesy",)),
    },
    "THE_LANTERN_KEY": {
        "RETURN_THE_KEY": R(N(gold_delta=100)),
        "KEEP_THE_KEY": R(N(next_page="KEEP_THE_KEY")),
    },
    "THE_LEGENDS_WERE_TRUE": {
        "NAB_THE_MAP": R(add_cards=("spoils_map",)),
        "SLOWLY_FIND_AN_EXIT": R(N(hp_loss=8), uniform_potion_reward=True),
    },
    "THIS_OR_THAT": {"ORNATE": R(add_cards=("clumsy",), random_relic=True)},
    "TRASH_HEAP": {
        "DIVE_IN": R(
            N(hp_loss=8),
            relic_pool=("darkstone_periapt", "dream_catcher", "hand_drill", "maw_bank", "the_boot"),
        ),
        "GRAB": R(
            N(gold_delta=100),
            card_pool=(
                "caltrops",
                "clash",
                "distraction",
                "dual_wield",
                "entrench",
                "hello_world",
                "outmaneuver",
                "rebound",
                "rip_and_tear",
                "stack",
            ),
        ),
    },
    "UNREST_SITE": {
        "REST": R(heal_to_full=True, add_cards=("poor_sleep",)),
        "KILL": R(N(max_hp_delta=-8), random_relic=True),
    },
    "WATERLOGGED_SCRIPTORIUM": {
        "BLOODY_INK": R(N(max_hp_delta=6)),
        "PRICKLY_SPONGE": R(N(gold_delta=-99), selection="enchant", count=2,
                            enchant="Steady", required_gold=99),
        "TENTACLE_QUILL": R(N(gold_delta=-55), selection="enchant", enchant="Steady",
                            required_gold=55),
    },
    "WELLSPRING": {
        "BOTTLE": R(uniform_potion_reward=True),
        "BATHE": R(selection="remove", add_cards=("guilty",)),
    },
    "WHISPERING_HOLLOW": {"HUG": R(N(hp_loss=9), selection="transform")},
    "WOOD_CARVINGS": {
        "BIRD": R(selection="transform", basic_only=True, target_card="peck"),
        "TORUS": R(selection="transform", basic_only=True, target_card="toric_toughness"),
    },
    "ZEN_WEAVER": {
        "BREATHING_TECHNIQUES": R(
            N(gold_delta=-50), add_cards=("enlightenment",) * 2, required_gold=50
        ),
        "EMOTIONAL_AWARENESS": R(
            N(gold_delta=-125), selection="remove", required_gold=125, selection_first=True
        ),
        "ARACHNID_ACUPUNCTURE": R(
            N(gold_delta=-250), selection="remove", count=2, required_gold=250, selection_first=True
        ),
    },
}


def source_event_rule(event_id: str, option_id: str) -> SourceEventRule | None:
    return SOURCE_EVENT_RULES.get(event_id.upper(), {}).get(option_id.upper())
