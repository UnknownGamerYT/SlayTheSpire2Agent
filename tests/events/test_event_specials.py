from __future__ import annotations

from random import Random

import pytest

from sts2sim.mechanics.event_specials import (
    FOUL_POTION_ID,
    HISTORY_COURSE_RELIC_ID,
    INJURY_CARD_ID,
    LANTERN_KEY_CARD_ID,
    ROYAL_POISON_RELIC_ID,
    SPECIAL_EVENT_IDS,
    legal_special_event_option_ids,
    resolve_special_event_option,
    special_event_implementations,
    special_event_options,
    special_event_room_state,
)


def test_battleworn_dummy_builds_three_fight_settings_with_reward_markers() -> None:
    state = special_event_room_state("BATTLEWORN_DUMMY", hp=70, max_hp=80)

    assert legal_special_event_option_ids(state) == ("SETTING_1", "SETTING_2", "SETTING_3")

    options = {option.option_id: option for option in state.options}
    assert options["SETTING_1"].combat_encounter == "battleworn_dummy"
    assert options["SETTING_1"].metadata["monster_hp"] == 75
    assert options["SETTING_1"].metadata["post_combat_reward"] == {
        "random_potion_count": 1,
        "card_count": 0,
        "relic_count": 0,
    }
    setting_2_reward = options["SETTING_2"].metadata["post_combat_reward"]
    setting_3_reward = options["SETTING_3"].metadata["post_combat_reward"]
    assert isinstance(setting_2_reward, dict)
    assert isinstance(setting_3_reward, dict)
    assert setting_2_reward["upgrade_random_count"] == 2
    assert setting_3_reward["random_relic_count"] == 1

    outcome = resolve_special_event_option(state, "SETTING_3")

    assert outcome.state.combat_encounter == "battleworn_dummy"
    assert outcome.relic_ids == ()


def test_colorful_philosophers_builds_character_pool_card_rewards() -> None:
    state = special_event_room_state("COLORFUL_PHILOSOPHERS", hp=70, max_hp=80)

    assert "COLORFUL_PHILOSOPHERS" in SPECIAL_EVENT_IDS
    assert legal_special_event_option_ids(state) == (
        "DEFECT",
        "IRONCLAD",
        "NECROBINDER",
        "REGENT",
        "SILENT",
    )

    options = {option.option_id: option for option in state.options}
    assert {
        option_id: options[option_id].metadata
        for option_id in ("DEFECT", "IRONCLAD", "NECROBINDER", "REGENT", "SILENT")
    } == {
        "DEFECT": {
            "card_reward_count": 3,
            "card_reward_character": "defect",
            "card_reward_kind": "character_pool",
        },
        "IRONCLAD": {
            "card_reward_count": 3,
            "card_reward_character": "ironclad",
            "card_reward_kind": "character_pool",
        },
        "NECROBINDER": {
            "card_reward_count": 3,
            "card_reward_character": "necrobinder",
            "card_reward_kind": "character_pool",
        },
        "REGENT": {
            "card_reward_count": 3,
            "card_reward_character": "regent",
            "card_reward_kind": "character_pool",
        },
        "SILENT": {
            "card_reward_count": 3,
            "card_reward_character": "silent",
            "card_reward_kind": "character_pool",
        },
    }

    outcome = resolve_special_event_option(state, "NECROBINDER")

    assert outcome.metadata == options["NECROBINDER"].metadata
    assert outcome.added_card_ids == ()
    assert outcome.state.deck == ()
    assert outcome.state.resolved_option_ids == ("NECROBINDER",)
    assert "NECROBINDER" not in legal_special_event_option_ids(outcome.state)

    implementation = next(
        entry
        for entry in special_event_implementations()
        if entry["event_id"] == "COLORFUL_PHILOSOPHERS"
    )
    assert implementation["option_ids"] == (
        "DEFECT",
        "IRONCLAD",
        "NECROBINDER",
        "REGENT",
        "SILENT",
    )


def test_punch_off_nab_is_immediate_and_fight_marks_greater_rewards() -> None:
    state = special_event_room_state("PUNCH_OFF", hp=50, max_hp=80)

    nab = resolve_special_event_option(
        state,
        "NAB",
        rng=Random(2),
        relic_pool=("anchor", "kunai"),
    )

    assert nab.added_card_ids == (INJURY_CARD_ID,)
    assert nab.state.deck == (INJURY_CARD_ID,)
    assert len(nab.relic_ids) == 1
    assert nab.relic_ids[0] in {"anchor", "kunai"}

    fight = next(option for option in state.options if option.option_id == "I_CAN_TAKE_THEM")
    assert fight.combat_encounter == "normal"
    assert fight.metadata["post_combat_reward"] == {
        "standard_combat_rewards": True,
        "random_relic_count": 1,
        "extra_random_potion_count": 1,
    }


def test_dense_vegetation_trudge_rolls_gold_and_rest_heals_before_fight() -> None:
    state = special_event_room_state("DENSE_VEGETATION", hp=20, max_hp=100, gold=5)

    trudge = resolve_special_event_option(state, "TRUDGE_ON", rng=Random(4))

    assert 61 <= trudge.gold_delta <= 99
    assert trudge.state.gold == 5 + trudge.gold_delta
    assert trudge.hp_delta == -8
    assert trudge.state.hp == 12

    rest = resolve_special_event_option(state, "REST")

    assert rest.heal_amount == 30
    assert rest.state.hp == 50
    assert rest.state.combat_encounter == "normal"


def test_infested_automaton_builds_filtered_random_card_rewards() -> None:
    state = special_event_room_state("INFESTED_AUTOMATON", hp=50, max_hp=80)

    assert "INFESTED_AUTOMATON" in SPECIAL_EVENT_IDS
    assert legal_special_event_option_ids(state) == ("STUDY", "TOUCH_CORE")

    options = {option.option_id: option for option in state.options}
    assert options["STUDY"].metadata == {
        "random_card_count": 1,
        "card_type": "power",
        "card_reward_kind": "filtered_random_card",
    }
    assert options["TOUCH_CORE"].metadata == {
        "random_card_count": 1,
        "card_cost": 0,
        "card_reward_kind": "filtered_random_card",
    }

    studied = resolve_special_event_option(state, "STUDY")
    touched = resolve_special_event_option(state, "TOUCH_CORE")

    assert studied.metadata == options["STUDY"].metadata
    assert studied.added_card_ids == ()
    assert studied.state.deck == ()
    assert studied.state.resolved_option_ids == ("STUDY",)
    assert touched.metadata == options["TOUCH_CORE"].metadata
    assert touched.state.resolved_option_ids == ("TOUCH_CORE",)

    implementation = next(
        entry
        for entry in special_event_implementations()
        if entry["event_id"] == "INFESTED_AUTOMATON"
    )
    assert implementation["option_ids"] == ("STUDY", "TOUCH_CORE")


def test_lantern_key_return_gives_gold_and_keep_marks_quest_card_reward() -> None:
    state = special_event_room_state("THE_LANTERN_KEY", hp=40, max_hp=80)

    returned = resolve_special_event_option(state, "RETURN_THE_KEY")

    assert returned.gold_delta == 100
    assert returned.state.gold == 100

    kept = resolve_special_event_option(state, "KEEP_THE_KEY")

    assert kept.state.combat_encounter == "normal"
    assert kept.added_card_ids == ()
    assert kept.state.deck == ()
    assert kept.option.metadata["post_combat_reward"] == {"fixed_card_ids": (LANTERN_KEY_CARD_ID,)}


def test_war_historian_repy_consumes_lantern_key_for_cage_or_chest() -> None:
    state = special_event_room_state(
        "WAR_HISTORIAN_REPY",
        hp=50,
        max_hp=80,
        deck=("strike", LANTERN_KEY_CARD_ID, "defend"),
    )

    assert legal_special_event_option_ids(state) == ("UNLOCK_CAGE", "UNLOCK_CHEST")

    cage = resolve_special_event_option(state, "UNLOCK_CAGE")

    assert cage.removed_card_ids == (LANTERN_KEY_CARD_ID,)
    assert cage.relic_ids == (HISTORY_COURSE_RELIC_ID,)
    assert cage.state.deck == ("strike", "defend")

    chest = resolve_special_event_option(
        state,
        "UNLOCK_CHEST",
        rng=Random(3),
        relic_pool=("anchor", "kunai", "shovel"),
        potion_pool=("fire_potion", "skill_potion", "foul_potion"),
    )

    assert chest.removed_card_ids == (LANTERN_KEY_CARD_ID,)
    assert len(chest.relic_ids) == 2
    assert len(chest.potion_ids) == 2

    no_key = special_event_room_state("WAR_HISTORIAN_REPY", hp=50, max_hp=80)
    assert legal_special_event_option_ids(no_key) == ()


def test_round_tea_party_models_full_heal_poison_and_scripted_fight_reward() -> None:
    state = special_event_room_state("ROUND_TEA_PARTY", hp=23, max_hp=80)

    tea = resolve_special_event_option(state, "ENJOY_TEA")

    assert tea.heal_amount == 57
    assert tea.state.hp == 80
    assert tea.relic_ids == (ROYAL_POISON_RELIC_ID,)

    fight = resolve_special_event_option(
        state,
        "PICK_FIGHT",
        rng=Random(5),
        relic_pool=("anchor", "kunai"),
    )

    assert fight.hp_delta == -11
    assert fight.state.hp == 12
    assert len(fight.relic_ids) == 1
    assert fight.state.combat_encounter is None


def test_ranwid_the_elder_trade_options_apply_gold_potion_and_relic_costs() -> None:
    state = special_event_room_state(
        "RANWID_THE_ELDER",
        hp=70,
        max_hp=80,
        gold=100,
        potions=("fire_potion", "skill_potion"),
        relics=("starter_relic",),
    )

    assert legal_special_event_option_ids(state) == ("POTION", "GOLD", "RELIC")

    potion_trade = resolve_special_event_option(
        state,
        "POTION",
        spent_potion_ids=("skill_potion",),
        relic_pool=("anchor", "kunai", "shovel"),
    )
    assert potion_trade.spent_potion_ids == ("skill_potion",)
    assert potion_trade.state.potions == ("fire_potion",)
    assert len(potion_trade.relic_ids) == 1

    gold_trade = resolve_special_event_option(
        state,
        "GOLD",
        relic_pool=("anchor", "kunai", "shovel"),
    )
    assert gold_trade.gold_delta == -100
    assert gold_trade.state.gold == 0
    assert len(gold_trade.relic_ids) == 1

    relic_trade = resolve_special_event_option(
        state,
        "RELIC",
        spent_relic_ids=("starter_relic",),
        relic_pool=("anchor", "kunai", "shovel"),
    )
    assert relic_trade.spent_relic_ids == ("starter_relic",)
    assert relic_trade.state.relics == relic_trade.relic_ids
    assert len(relic_trade.relic_ids) == 2

    empty = special_event_room_state("RANWID_THE_ELDER", hp=70, max_hp=80)
    assert legal_special_event_option_ids(empty) == ()


def test_relic_trader_trade_slots_can_use_prerolled_offers() -> None:
    state = special_event_room_state(
        "RELIC_TRADER",
        hp=70,
        max_hp=80,
        relics=("starter_relic",),
        relic_trader_offered_relic_ids=("anchor", "kunai", "shovel"),
    )

    assert legal_special_event_option_ids(state) == ("TOP", "MIDDLE", "BOTTOM")

    trade = resolve_special_event_option(
        state,
        "MIDDLE",
        spent_relic_ids=("starter_relic",),
    )

    assert trade.spent_relic_ids == ("starter_relic",)
    assert trade.relic_ids == ("kunai",)
    assert trade.state.relics == ("kunai",)

    no_relic = special_event_room_state("RELIC_TRADER", hp=70, max_hp=80)
    assert legal_special_event_option_ids(no_relic) == ()


def test_potion_courier_grabs_foul_potions_or_ransacks_uncommon_potion() -> None:
    state = special_event_room_state("POTION_COURIER", hp=70, max_hp=80)

    grabbed = resolve_special_event_option(state, "GRAB_POTIONS")

    assert grabbed.potion_ids == (FOUL_POTION_ID, FOUL_POTION_ID, FOUL_POTION_ID)
    assert grabbed.state.potions == grabbed.potion_ids

    ransacked = resolve_special_event_option(
        state,
        "RANSACK",
        potion_pool=(
            {"id": "fire_potion", "rarity_key": "Common"},
            {"id": "skill_potion", "rarity_key": "Uncommon"},
            {"id": "essence_of_steel", "rarity_key": "Rare"},
        ),
    )

    assert ransacked.potion_ids == ("skill_potion",)


def test_self_help_book_marks_enchant_options_and_locked_variants() -> None:
    options = special_event_options("SELF_HELP_BOOK")
    by_id = {option.option_id: option for option in options}

    assert tuple(by_id) == (
        "READ_THE_BACK",
        "READ_THE_BACK_LOCKED",
        "READ_PASSAGE",
        "READ_PASSAGE_LOCKED",
        "READ_ENTIRE_BOOK",
        "READ_ENTIRE_BOOK_LOCKED",
        "NO_OPTIONS",
    )
    assert by_id["READ_THE_BACK"].metadata == {
        "enchant_keyword": "Sharp",
        "enchant_amount": 2,
        "card_type": "attack",
        "enchant_card_type": "attack",
    }
    assert by_id["READ_PASSAGE"].metadata["enchant_keyword"] == "Nimble"
    assert by_id["READ_PASSAGE"].metadata["card_type"] == "skill"
    assert by_id["READ_ENTIRE_BOOK"].metadata["enchant_keyword"] == "Swift"
    assert by_id["READ_ENTIRE_BOOK"].metadata["card_type"] == "power"
    assert by_id["READ_THE_BACK_LOCKED"].metadata["locked"] is True
    assert by_id["READ_THE_BACK_LOCKED"].metadata["disabled_reason"] == "requires_attack"
    assert by_id["NO_OPTIONS"].metadata["locked"] is True

    state = special_event_room_state("SELF_HELP_BOOK", hp=70, max_hp=80)
    assert legal_special_event_option_ids(state) == (
        "READ_THE_BACK",
        "READ_PASSAGE",
        "READ_ENTIRE_BOOK",
    )


def test_self_help_book_locks_missing_card_types_when_deck_is_known() -> None:
    skill_only = special_event_room_state(
        "SELF_HELP_BOOK",
        hp=70,
        max_hp=80,
        deck=("ACROBATICS",),
    )

    assert legal_special_event_option_ids(skill_only) == ("READ_PASSAGE",)
    by_id = {option.option_id: option for option in skill_only.options}
    assert "READ_THE_BACK" not in by_id
    assert by_id["READ_THE_BACK_LOCKED"].metadata["disabled_reason"] == "requires_attack"
    assert "READ_ENTIRE_BOOK" not in by_id
    assert by_id["READ_ENTIRE_BOOK_LOCKED"].metadata["disabled_reason"] == "requires_power"
    assert by_id["NO_OPTIONS"].metadata["locked"] is True

    empty = special_event_room_state(
        "SELF_HELP_BOOK",
        hp=70,
        max_hp=80,
        deck=(),
    )
    assert legal_special_event_option_ids(empty) == ("NO_OPTIONS",)
    assert {option.option_id for option in empty.options} == {
        "READ_THE_BACK_LOCKED",
        "READ_PASSAGE_LOCKED",
        "READ_ENTIRE_BOOK_LOCKED",
        "NO_OPTIONS",
    }


def test_future_of_potions_requires_potion_and_spends_selected_potion() -> None:
    empty = special_event_room_state("THE_FUTURE_OF_POTIONS", hp=70, max_hp=80)

    assert legal_special_event_option_ids(empty) == ()
    option = empty.options[0]
    assert option.option_id == "POTION"
    assert option.metadata["required_potion_count"] == 1
    assert option.metadata["potion_cost_count"] == 1
    assert option.metadata["card_rarity"] == "common"
    assert option.metadata["card_type"] == "skill"
    assert option.metadata["upgraded"] is True

    state = special_event_room_state(
        "THE_FUTURE_OF_POTIONS",
        hp=70,
        max_hp=80,
        potions=("fire_potion", "skill_potion"),
    )
    assert legal_special_event_option_ids(state) == ("POTION",)

    outcome = resolve_special_event_option(
        state,
        "POTION",
        spent_potion_ids=("skill_potion",),
    )

    assert outcome.spent_potion_ids == ("skill_potion",)
    assert outcome.state.potions == ("fire_potion",)
    assert outcome.metadata["random_card_count"] == 1
    assert outcome.metadata["card_rarity"] == "common"
    assert outcome.metadata["card_type"] == "skill"
    assert outcome.metadata["upgraded"] is True


def test_fake_merchant_builds_summary_marker_without_legal_choice() -> None:
    options = special_event_options(
        "FAKE_MERCHANT",
        fake_merchant_unsold_relic_ids=("fake_anchor", "fake_mango"),
    )
    state = special_event_room_state(
        "FAKE_MERCHANT",
        hp=70,
        max_hp=80,
        fake_merchant_unsold_relic_ids=("fake_anchor", "fake_mango"),
    )

    assert len(options) == 1
    marker = options[0]
    assert marker.option_id == "SUMMARY"
    assert marker.fixed_relic_ids == (
        "fake_merchants_rug",
        "fake_anchor",
        "fake_mango",
    )
    assert marker.metadata["summary_marker"] is True
    assert legal_special_event_option_ids(state) == ()

    with pytest.raises(ValueError, match="not legal"):
        resolve_special_event_option(state, "SUMMARY")
