from __future__ import annotations

from random import Random

import pytest

from sts2sim.mechanics.event_flows import (
    EventFlowMarker,
    EventFlowMarkerContext,
    EventFlowMarkerKind,
    current_event_flow_page,
    event_flow_state,
    is_event_flow_terminal,
    legal_event_flow_option_ids,
    resolve_event_flow_markers,
    resolve_event_flow_option,
    visible_event_flow_option_ids,
)


def _marker_kinds(markers: object) -> tuple[EventFlowMarkerKind, ...]:
    marker_tuple = markers
    assert isinstance(marker_tuple, tuple)
    return tuple(marker.kind for marker in marker_tuple)


def test_abyssal_baths_linger_progresses_then_exits_terminal() -> None:
    state = event_flow_state("ABYSSAL_BATHS", hp=30, max_hp=70)

    immersed = resolve_event_flow_option(state, "IMMERSE")
    assert immersed.state.page_id == "IMMERSE"
    assert immersed.state.hp == 29
    assert immersed.state.max_hp == 72
    assert legal_event_flow_option_ids(immersed.state) == ("LINGER", "EXIT_BATHS")

    lingered = resolve_event_flow_option(immersed.state, "LINGER")
    assert lingered.state.page_id == "LINGER1"
    assert lingered.state.hp == 27
    assert lingered.state.max_hp == 74

    exited = resolve_event_flow_option(lingered.state, "EXIT_BATHS")
    assert exited.state.page_id == "EXIT_BATHS"
    assert is_event_flow_terminal(exited.state) is True
    assert legal_event_flow_option_ids(exited.state) == ()


def test_abyssal_baths_death_warning_locks_in_lethal_linger_marker() -> None:
    state = event_flow_state("ABYSSAL_BATHS", hp=4, max_hp=70)

    immersed = resolve_event_flow_option(state, "IMMERSE")
    assert immersed.state.page_id == "DEATH_WARNING"
    assert legal_event_flow_option_ids(immersed.state) == ("LINGER", "EXIT_BATHS")

    lethal = resolve_event_flow_option(immersed.state, "LINGER")
    assert lethal.state.hp == 0
    assert is_event_flow_terminal(lethal.state) is True
    assert EventFlowMarkerKind.RUN_DEATH in _marker_kinds(lethal.markers)


def test_colossal_flower_deeper_path_tracks_costs_and_terminal_relic_marker() -> None:
    state = event_flow_state("COLOSSAL_FLOWER", hp=30, max_hp=70)

    first = resolve_event_flow_option(state, "REACH_DEEPER_1")
    second = resolve_event_flow_option(first.state, "REACH_DEEPER_2")
    core = resolve_event_flow_option(second.state, "POLLINOUS_CORE")

    assert first.state.page_id == "REACH_DEEPER_1"
    assert second.state.page_id == "REACH_DEEPER_2"
    assert core.state.hp == 12
    assert core.state.page_id == "POLLINOUS_CORE"
    assert core.terminal is True
    assert core.markers[0].kind is EventFlowMarkerKind.FIXED_RELIC
    assert core.markers[0].item_id == "pollinous_core"


def test_endless_conveyor_repeats_paid_dishes_then_shows_broke_lock() -> None:
    state = event_flow_state("ENDLESS_CONVEYOR", hp=40, max_hp=80, gold=80, page_id="ALL")
    assert "CAVIAR" in legal_event_flow_option_ids(state)

    first = resolve_event_flow_option(state, "CAVIAR")
    second = resolve_event_flow_option(first.state, "CAVIAR")

    assert first.state.page_id == "ALL"
    assert first.state.gold == 40
    assert first.state.hp == 44
    assert first.state.max_hp == 84
    assert second.state.page_id == "ALL"
    assert second.state.gold == 0
    assert second.state.hp == 48
    assert second.state.max_hp == 88
    assert visible_event_flow_option_ids(second.state) == ("LOCKED", "LEAVE")
    assert legal_event_flow_option_ids(second.state) == ("LEAVE",)

    with pytest.raises(ValueError):
        resolve_event_flow_option(second.state, "LOCKED")

    left = resolve_event_flow_option(second.state, "LEAVE")
    assert left.state.page_id == "LEAVE"
    assert is_event_flow_terminal(left.state) is True


def test_slippery_bridge_hold_on_steps_and_loop_damage_escalates() -> None:
    state = event_flow_state("SLIPPERY_BRIDGE", hp=45, max_hp=70)

    current = resolve_event_flow_option(state, "HOLD_ON_0").state
    assert current.page_id == "HOLD_ON_0"
    assert current.hp == 42

    for option_id in (
        "HOLD_ON_1",
        "HOLD_ON_2",
        "HOLD_ON_3",
        "HOLD_ON_4",
        "HOLD_ON_5",
        "HOLD_ON_6",
        "HOLD_ON_LOOP",
    ):
        current = resolve_event_flow_option(current, option_id).state

    assert current.page_id == "HOLD_ON_LOOP"
    assert current.hp == 0
    assert is_event_flow_terminal(current) is True

    loop = event_flow_state("SLIPPERY_BRIDGE", hp=30, max_hp=70, page_id="HOLD_ON_LOOP")
    first_loop = resolve_event_flow_option(loop, "HOLD_ON_LOOP")
    second_loop = resolve_event_flow_option(first_loop.state, "HOLD_ON_LOOP")
    assert first_loop.hp_delta == -11
    assert second_loop.hp_delta == -12


def test_slippery_bridge_offer_targets_do_not_repeat_until_deck_seen() -> None:
    state = event_flow_state(
        "SLIPPERY_BRIDGE",
        hp=45,
        max_hp=70,
        data={"bridge_deck": ("strike", "defend", "bash")},
    )

    first_offer = current_event_flow_page(state).options[0]
    assert first_offer.option_id == "OVERCOME"
    assert first_offer.metadata["offered_card_id"] == "strike"

    first_hold = resolve_event_flow_option(state, "HOLD_ON_0")
    second_offer = current_event_flow_page(first_hold.state).options[0]
    assert second_offer.metadata["offered_card_id"] == "defend"

    second_hold = resolve_event_flow_option(first_hold.state, "HOLD_ON_1")
    third_offer = current_event_flow_page(second_hold.state).options[0]
    assert third_offer.metadata["offered_card_id"] == "bash"

    third_hold = resolve_event_flow_option(second_hold.state, "HOLD_ON_2")
    cycled_offer = current_event_flow_page(third_hold.state).options[0]
    assert cycled_offer.metadata["offered_card_id"] == "strike"

    overcome = resolve_event_flow_option(first_hold.state, "OVERCOME")
    application = resolve_event_flow_markers(
        overcome.markers,
        EventFlowMarkerContext(deck=("strike", "defend", "bash")),
    )
    assert application.removed_card_ids == ("defend",)
    assert application.context.deck == ("strike", "bash")


def test_tablet_of_truth_decipher_chain_locks_out_smash_and_finishes_at_one_max_hp() -> None:
    state = event_flow_state("TABLET_OF_TRUTH", hp=50, max_hp=80)
    assert legal_event_flow_option_ids(state) == ("DECIPHER_1", "SMASH")

    step1 = resolve_event_flow_option(state, "DECIPHER_1")
    assert step1.state.page_id == "DECIPHER_1"
    assert legal_event_flow_option_ids(step1.state) == ("DECIPHER",)

    step2 = resolve_event_flow_option(step1.state, "DECIPHER")
    step3 = resolve_event_flow_option(step2.state, "DECIPHER")
    step4 = resolve_event_flow_option(step3.state, "DECIPHER")
    final = resolve_event_flow_option(step4.state, "DECIPHER")

    assert final.state.page_id == "DECIPHER_5"
    assert final.state.hp == 1
    assert final.state.max_hp == 1
    assert final.markers[0].kind is EventFlowMarkerKind.CARD_UPGRADE_ALL
    assert is_event_flow_terminal(final.state) is True


def test_trial_reject_double_down_is_terminal_and_accept_can_take_supplied_case() -> None:
    state = event_flow_state("TRIAL", hp=20, max_hp=70, data={"trial_case": "NOBLE"})
    rejected = resolve_event_flow_option(state, "REJECT")

    assert rejected.state.page_id == "REJECT"
    assert legal_event_flow_option_ids(rejected.state) == ("ACCEPT", "DOUBLE_DOWN")

    death = resolve_event_flow_option(rejected.state, "DOUBLE_DOWN")
    assert death.state.hp == 0
    assert is_event_flow_terminal(death.state) is True
    assert EventFlowMarkerKind.RUN_DEATH in _marker_kinds(death.markers)

    accepted = resolve_event_flow_option(state, "ACCEPT")
    assert accepted.state.page_id == "NOBLE"

    innocent = resolve_event_flow_option(accepted.state, "INNOCENT")
    assert innocent.state.gold == 300
    assert innocent.state.page_id == "NOBLE_INNOCENT"
    assert innocent.markers[0].kind is EventFlowMarkerKind.CARD_ADD
    assert innocent.markers[0].item_id == "regret"
    assert is_event_flow_terminal(innocent.state) is True


def test_trial_without_supplied_case_exposes_unknown_branch_marker() -> None:
    state = event_flow_state("TRIAL", hp=20, max_hp=70)

    accepted = resolve_event_flow_option(state, "ACCEPT")

    assert accepted.state.page_id == "TRIAL_CASE_SELECTION"
    assert accepted.markers[0].kind is EventFlowMarkerKind.UNKNOWN_BRANCH
    assert legal_event_flow_option_ids(accepted.state) == (
        "SELECT_MERCHANT",
        "SELECT_NOBLE",
        "SELECT_NONDESCRIPT",
    )


def test_wongos_shows_gold_locks_and_mystery_box_delayed_marker() -> None:
    state = event_flow_state("WELCOME_TO_WONGOS", hp=50, max_hp=80, gold=150)

    assert visible_event_flow_option_ids(state) == (
        "BARGAIN_BIN",
        "FEATURED_ITEM_LOCKED",
        "MYSTERY_BOX_LOCKED",
        "LEAVE",
    )
    assert legal_event_flow_option_ids(state) == ("BARGAIN_BIN", "LEAVE")

    with pytest.raises(ValueError):
        resolve_event_flow_option(state, "FEATURED_ITEM_LOCKED")

    bargain = resolve_event_flow_option(state, "BARGAIN_BIN")
    assert bargain.state.gold == 50
    assert bargain.markers[0].kind is EventFlowMarkerKind.RANDOM_RELIC
    assert bargain.markers[0].qualifier == "common"
    assert is_event_flow_terminal(bargain.state) is True

    rich = event_flow_state("WELCOME_TO_WONGOS", hp=50, max_hp=80, gold=300)
    mystery = resolve_event_flow_option(rich, "MYSTERY_BOX")
    assert mystery.state.gold == 0
    assert mystery.markers[0].kind is EventFlowMarkerKind.DELAYED_REWARD
    assert mystery.markers[0].delay_combat_count == 5


def test_tinker_time_builds_custom_card_from_type_and_rider() -> None:
    state = event_flow_state("TINKER_TIME", hp=50, max_hp=80)
    assert visible_event_flow_option_ids(state) == ("CHOOSE_CARD_TYPE",)

    accepted = resolve_event_flow_option(state, "CHOOSE_CARD_TYPE")
    assert accepted.state.page_id == "CHOOSE_CARD_TYPE"
    assert legal_event_flow_option_ids(accepted.state) == ("ATTACK", "SKILL", "POWER")

    card_type = resolve_event_flow_option(accepted.state, "SKILL")
    assert card_type.state.page_id == "CHOOSE_RIDER"
    assert card_type.state.data["custom_card_type"] == "skill"
    assert legal_event_flow_option_ids(card_type.state) == ("ENERGIZED", "WISDOM", "CHAOS")

    rider = resolve_event_flow_option(card_type.state, "WISDOM")
    assert rider.state.page_id == "DONE"
    assert rider.state.data["custom_card_rider"] == "wisdom"
    assert rider.markers[0].kind is EventFlowMarkerKind.CUSTOM_CARD
    assert rider.markers[0].qualifier == "skill"
    assert rider.markers[0].metadata["card_type"] == "skill"
    assert rider.markers[0].metadata["rider_id"] == "wisdom"
    assert rider.markers[0].metadata["rider_effect"] == "draw_cards"
    assert rider.markers[0].metadata["rider"]["draw"] == 3
    assert is_event_flow_terminal(rider.state) is True
    assert legal_event_flow_option_ids(rider.state) == ()


def test_event_flow_marker_application_adds_cards_and_draws_relics() -> None:
    state = event_flow_state("TRIAL", hp=20, max_hp=70, data={"trial_case": "MERCHANT"})
    accepted = resolve_event_flow_option(state, "ACCEPT")
    guilty = resolve_event_flow_option(accepted.state, "GUILTY")

    application = resolve_event_flow_markers(
        guilty.markers,
        EventFlowMarkerContext(deck=("Strike",), relics=("Anchor",)),
        relic_pool=("Anchor", "Kunai", "Shovel", "Bag of Preparation"),
    )

    assert application.added_card_ids == ("regret",)
    assert application.relic_ids == ("kunai", "shovel")
    assert application.context.deck == ("strike", "regret")
    assert application.context.relics == ("anchor", "kunai", "shovel")
    assert len(application.reward_requests) == 1
    assert application.reward_requests[0].reward_kind == "random_relic"
    assert application.reward_requests[0].count == 2
    assert application.blocked_markers == ()


def test_event_flow_marker_application_respects_qualified_random_relic_pools() -> None:
    state = event_flow_state("WELCOME_TO_WONGOS", hp=50, max_hp=80, gold=150)
    bargain = resolve_event_flow_option(state, "BARGAIN_BIN")

    application = resolve_event_flow_markers(
        bargain.markers,
        rng=Random(9),
        relic_pool=(
            {"id": "Anchor", "rarity_key": "Common"},
            {"id": "Kunai", "rarity_key": "Uncommon"},
            {"id": "Shovel", "rarity_key": "Rare"},
        ),
    )

    assert application.relic_ids == ("anchor",)
    assert application.context.relics == ("anchor",)
    assert application.reward_requests[0].qualifier == "common"


def test_event_flow_marker_application_applies_fixed_potions() -> None:
    marker = EventFlowMarker(
        kind=EventFlowMarkerKind.FIXED_POTION,
        item_id="Fire Potion",
        description="Procure Fire Potion.",
    )

    application = resolve_event_flow_markers(
        (marker,),
        EventFlowMarkerContext(potions=("Skill Potion",)),
    )

    assert application.potion_ids == ("fire_potion",)
    assert application.context.potions == ("skill_potion", "fire_potion")
    assert application.blocked_markers == ()


def test_event_flow_marker_application_tracks_upgrade_all_and_delayed_rewards() -> None:
    state = event_flow_state("TABLET_OF_TRUTH", hp=50, max_hp=80)
    current = resolve_event_flow_option(state, "DECIPHER_1").state
    for _ in range(4):
        current_resolution = resolve_event_flow_option(current, "DECIPHER")
        current = current_resolution.state

    application = resolve_event_flow_markers(current_resolution.markers)

    assert application.context.upgrade_all_count == 1

    rich = event_flow_state("WELCOME_TO_WONGOS", hp=50, max_hp=80, gold=300)
    mystery = resolve_event_flow_option(rich, "MYSTERY_BOX")
    delayed = resolve_event_flow_markers(mystery.markers)

    assert delayed.relic_ids == ()
    assert delayed.reward_requests[0].reward_kind == "random_relic"
    assert delayed.reward_requests[0].count == 3
    assert delayed.reward_requests[0].delay_combat_count == 5
    assert delayed.reward_requests[0].metadata["source_marker_kind"] == "delayed_reward"


def test_event_flow_marker_application_blocks_unknown_branches_explicitly() -> None:
    state = event_flow_state("TRIAL", hp=20, max_hp=70)
    accepted = resolve_event_flow_option(state, "ACCEPT")

    application = resolve_event_flow_markers(accepted.markers)

    assert application.blocked_markers
    assert application.blocked_markers[0].marker.kind is EventFlowMarkerKind.UNKNOWN_BRANCH
    assert application.blocked_markers[0].reason == "unknown_branch"
