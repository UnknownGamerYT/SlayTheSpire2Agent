from __future__ import annotations

from random import Random

from sts2sim.mechanics.event_rooms import (
    event_room_state,
    legal_event_option_ids,
    resolve_event_option,
)


def test_lantern_key_keep_option_adds_fixed_quest_card() -> None:
    state = event_room_state("THE_LANTERN_KEY", hp=40, max_hp=80)

    assert legal_event_option_ids(state) == ("RETURN_THE_KEY", "KEEP_THE_KEY")

    outcome = resolve_event_option(state, "KEEP_THE_KEY")

    assert outcome.added_card_ids == ("lantern_key",)
    assert outcome.state.deck == ("lantern_key",)
    assert outcome.state.combat_encounter == "normal"
    assert outcome.state.gold == 0
    assert legal_event_option_ids(outcome.state) == ()


def test_war_historian_cage_and_chest_rewards_consume_lantern_key() -> None:
    state = event_room_state(
        "WAR_HISTORIAN_REPY",
        hp=50,
        max_hp=80,
        deck=("strike", "lantern_key", "defend"),
    )

    assert legal_event_option_ids(state) == ("UNLOCK_CAGE", "UNLOCK_CHEST")

    cage = resolve_event_option(state, "UNLOCK_CAGE")

    assert cage.removed_card_ids == ("lantern_key",)
    assert cage.relic_ids == ("history_course",)
    assert cage.state.deck == ("strike", "defend")
    assert cage.state.relics == ("history_course",)

    chest = resolve_event_option(
        state,
        "UNLOCK_CHEST",
        rng=Random(4),
        relic_pool=("anchor", "kunai", "shovel"),
        potion_pool=("fire_potion", "skill_potion", "foul_potion"),
    )

    assert chest.removed_card_ids == ("lantern_key",)
    assert len(chest.relic_ids) == 2
    assert set(chest.relic_ids) <= {"anchor", "kunai", "shovel"}
    assert len(chest.potion_ids) == 2
    assert set(chest.potion_ids) <= {"fire_potion", "skill_potion", "foul_potion"}
    assert "lantern_key" not in chest.state.deck


def test_war_historian_multi_lantern_keys_unlock_both_rewards_and_clean_duplicates() -> None:
    state = event_room_state(
        "WAR_HISTORIAN_REPY",
        hp=50,
        max_hp=80,
        deck=("lantern_key", "strike", "lantern_key", "defend", "lantern_key"),
    )

    cage = resolve_event_option(
        state,
        "UNLOCK_CAGE",
        rng=Random(6),
        relic_pool=("anchor", "kunai", "shovel"),
        potion_pool=("fire_potion", "skill_potion", "foul_potion"),
    )

    assert cage.removed_card_ids == ("lantern_key", "lantern_key", "lantern_key")
    assert cage.state.deck == ("strike", "defend")
    assert cage.relic_ids[0] == "history_course"
    assert len(cage.relic_ids) == 3
    assert len(cage.potion_ids) == 2
    assert cage.option.metadata["multi_lantern_key_applied"] is True

    chest = resolve_event_option(
        state,
        "UNLOCK_CHEST",
        rng=Random(7),
        relic_pool=("anchor", "kunai", "shovel"),
        potion_pool=("fire_potion", "skill_potion", "foul_potion"),
    )

    assert chest.removed_card_ids == ("lantern_key", "lantern_key", "lantern_key")
    assert chest.relic_ids[0] == "history_course"
    assert len(chest.relic_ids) == 3
    assert len(chest.potion_ids) == 2


def test_war_historian_options_require_lantern_key() -> None:
    state = event_room_state("WAR_HISTORIAN_REPY", hp=50, max_hp=80, deck=("strike",))

    assert legal_event_option_ids(state) == ()


def test_battleworn_dummy_three_options_resolve_their_reward_markers() -> None:
    state = event_room_state("BATTLEWORN_DUMMY", hp=70, max_hp=80)

    assert legal_event_option_ids(state) == ("SETTING_1", "SETTING_2", "SETTING_3")

    potion = resolve_event_option(
        state,
        "SETTING_1",
        rng=Random(1),
        potion_pool=("fire_potion", "skill_potion"),
    )
    upgrade = resolve_event_option(state, "SETTING_2")
    relic = resolve_event_option(
        state,
        "SETTING_3",
        rng=Random(2),
        relic_pool=("anchor", "kunai"),
    )

    assert len(potion.potion_ids) == 1
    assert potion.state.combat_encounter == "battleworn_dummy"
    assert upgrade.upgrade_random_count == 2
    assert upgrade.state.upgrade_random_count == 2
    assert len(relic.relic_ids) == 1
    assert relic.state.combat_encounter == "battleworn_dummy"


def test_dense_vegetation_rest_heals_and_marks_normal_combat() -> None:
    state = event_room_state("DENSE_VEGETATION", hp=20, max_hp=100)

    assert legal_event_option_ids(state) == ("TRUDGE_ON", "REST")

    outcome = resolve_event_option(state, "REST")

    assert outcome.heal_amount == 30
    assert outcome.hp_delta == 30
    assert outcome.state.hp == 50
    assert outcome.state.combat_encounter == "normal"
