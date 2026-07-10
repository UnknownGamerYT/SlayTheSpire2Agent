from __future__ import annotations

import pytest

from sts2sim.gymnasium_env import Sts2Env
from sts2sim.learning.rewards import (
    DEFAULT_REWARD_CONFIG,
    LearningRewardTracker,
    aggression_weights,
    deck_delta_summary,
    learning_reward,
    learning_reward_breakdown,
    starter_dependency_summary,
)


def test_reward_defaults_remove_sparse_shortcuts() -> None:
    config = DEFAULT_REWARD_CONFIG

    assert config.step_penalty == 0.0
    assert config.death_penalty == 0.0
    assert config.win_reward == 0.0


@pytest.mark.parametrize(
    ("pressure", "hp_weight", "damage_weight", "prevented_weight"),
    [
        (0.0, 0.09, 0.01, 0.05),
        (0.5, 0.05, 0.05, 0.0325),
        (1.0, 0.01, 0.09, 0.015),
    ],
)
def test_aggression_weights_interpolate_exactly(
    pressure: float,
    hp_weight: float,
    damage_weight: float,
    prevented_weight: float,
) -> None:
    weights = aggression_weights(pressure)

    assert weights["hp_loss_weight"] == hp_weight
    assert weights["enemy_damage_weight"] == damage_weight
    assert weights["prevented_hp_weight"] == prevented_weight


def test_no_direct_death_or_complete_reward() -> None:
    previous = _base_payload(phase="map", hp=40)
    failed = _base_payload(phase="failed", hp=40)
    complete = _base_payload(phase="complete", hp=40)

    assert learning_reward_breakdown(previous, failed).total == 0.0
    assert learning_reward_breakdown(previous, complete).total == 0.0


def test_gold_reward_is_positive_and_capped() -> None:
    previous = _base_payload(gold=0)
    current = _base_payload(gold=999)

    breakdown = learning_reward_breakdown(previous, current)

    assert breakdown.gold_reward == 0.5
    assert breakdown.total == 0.5


def test_reward_screen_state_without_action_descriptor_has_no_direct_reward() -> None:
    previous = _base_payload()
    current = _base_payload()
    current["reward"] = {
        "reward_id": "test",
        "source": "combat",
        "relic_id": "some_relic",
        "relic_claimed": True,
        "card_options": ["strike", "defend", "bash"],
        "card_claimed": True,
        "claimed_card_indices": [0],
        "skipped_card_indices": [1],
    }

    assert learning_reward_breakdown(previous, current).total == 0.0


def test_reward_pickups_receive_small_direct_credit() -> None:
    previous = _base_payload()
    current = _base_payload()

    card = learning_reward_breakdown(
        previous,
        current,
        action_descriptor={"type": "take_reward_card"},
    )
    relic = learning_reward_breakdown(
        previous,
        current,
        action_descriptor={"type": "take_reward_relic"},
    )
    potion = learning_reward_breakdown(
        previous,
        current,
        action_descriptor={"type": "take_reward_potion"},
    )

    assert card.resource_pickup_reward == DEFAULT_REWARD_CONFIG.card_pickup_reward
    assert relic.resource_pickup_reward == DEFAULT_REWARD_CONFIG.relic_pickup_reward
    assert potion.resource_pickup_reward == DEFAULT_REWARD_CONFIG.potion_pickup_reward
    assert relic.resource_pickup_reward < DEFAULT_REWARD_CONFIG.node_progress_reward


def test_card_additions_are_rewarded_by_deck_fit_not_generic_pickup() -> None:
    previous = _base_payload()
    current = _base_payload(
        deck=(
            {
                "card_id": "pommel_strike",
                "type": "attack",
                "cost": 1,
                "effects": {"damage": 9, "draw": 1},
            },
        )
    )

    breakdown = learning_reward_breakdown(
        previous,
        current,
        action_descriptor={"type": "take_reward_card"},
    )

    assert breakdown.resource_pickup_reward == 0.0
    assert breakdown.deck_capability_reward > 0.0


def test_reward_skips_do_not_double_count_pickup_incentives_by_default() -> None:
    previous = _base_payload()

    gold = learning_reward_breakdown(
        previous,
        previous,
        action_descriptor={
            "type": "skip_reward",
            "reward_choice": {"skip_kind": "gold"},
        },
    )
    card = learning_reward_breakdown(
        previous,
        previous,
        action_descriptor={
            "type": "skip_reward",
            "reward_choice": {"skip_kind": "card_options"},
        },
    )
    relic = learning_reward_breakdown(
        _base_payload(
            phase="reward",
            reward={
                "reward_id": "combat:1",
                "source": "combat",
                "relic_id": "anchor",
                "relic_claimed": False,
            },
        ),
        previous,
        action_descriptor={
            "type": "skip_reward",
            "reward_choice": {"skip_kind": "relic"},
        },
    )

    assert gold.reward_skip_penalty == DEFAULT_REWARD_CONFIG.skip_gold_penalty
    assert gold.reward_skip_penalty < 0.0
    assert card.reward_skip_penalty == DEFAULT_REWARD_CONFIG.early_card_skip_penalty
    assert card.reward_skip_penalty < 0.0
    assert relic.reward_skip_penalty == DEFAULT_REWARD_CONFIG.skip_relic_penalty
    assert relic.reward_skip_penalty < 0.0
    assert abs(DEFAULT_REWARD_CONFIG.skip_gold_penalty) > DEFAULT_REWARD_CONFIG.card_pickup_reward


def test_proceeding_with_unclaimed_gold_is_penalized() -> None:
    previous = _base_payload(
        phase="reward",
        reward={
            "reward_id": "combat:1",
            "source": "combat",
            "gold": 19,
            "gold_claimed": False,
        },
    )

    breakdown = learning_reward_breakdown(
        previous,
        previous,
        action_descriptor={"type": "proceed"},
    )

    assert breakdown.reward_skip_penalty == DEFAULT_REWARD_CONFIG.skip_gold_penalty
    assert breakdown.total == DEFAULT_REWARD_CONFIG.skip_gold_penalty


def test_proceeding_with_unclaimed_relic_is_penalized_per_relic() -> None:
    previous = _base_payload(
        phase="reward",
        reward={
            "reward_id": "event:relics",
            "source": "event",
            "relic_ids": ["anchor", "bag_of_marbles"],
            "claimed_relic_ids": ["anchor"],
            "skipped_relic_ids": [],
        },
    )

    breakdown = learning_reward_breakdown(
        previous,
        previous,
        action_descriptor={"type": "proceed"},
    )

    assert breakdown.reward_skip_penalty == DEFAULT_REWARD_CONFIG.skip_relic_penalty
    assert breakdown.total == DEFAULT_REWARD_CONFIG.skip_relic_penalty


def test_multiple_unclaimed_relics_have_diminishing_skip_penalty() -> None:
    previous = _base_payload(
        phase="reward",
        reward={
            "reward_id": "event:relics",
            "source": "event",
            "relic_ids": ["anchor", "bag_of_marbles", "kunai"],
            "claimed_relic_ids": [],
            "skipped_relic_ids": [],
        },
    )

    breakdown = learning_reward_breakdown(
        previous,
        previous,
        action_descriptor={"type": "proceed"},
    )

    expected = DEFAULT_REWARD_CONFIG.skip_relic_penalty * (
        1
        + DEFAULT_REWARD_CONFIG.relic_skip_penalty_decay
        + DEFAULT_REWARD_CONFIG.relic_skip_penalty_decay**2
    )
    linear = DEFAULT_REWARD_CONFIG.skip_relic_penalty * 3
    assert breakdown.reward_skip_penalty == pytest.approx(expected)
    assert abs(breakdown.reward_skip_penalty) < abs(linear)


def test_exclusive_relic_choices_are_not_penalized_as_all_unclaimed() -> None:
    previous = _base_payload(
        phase="reward",
        reward={
            "reward_id": "ancient:relic-choice",
            "source": "ancient",
            "relic_ids": ["anchor", "bag_of_marbles", "kunai"],
            "claimed_relic_ids": [],
            "skipped_relic_ids": [],
            "metadata": {"exclusive_relic_choices": True, "max_relic_choices": 1},
        },
    )

    breakdown = learning_reward_breakdown(
        previous,
        previous,
        action_descriptor={"type": "proceed"},
    )

    assert (
        breakdown.reward_skip_penalty
        == DEFAULT_REWARD_CONFIG.exclusive_relic_choice_skip_penalty
    )
    assert abs(breakdown.reward_skip_penalty) < abs(DEFAULT_REWARD_CONFIG.skip_relic_penalty)


def test_shop_leave_relic_opportunity_cost_respects_affordability() -> None:
    previous = _base_payload(phase="shop", gold=100)
    previous["shop"] = {
        "items": [
            {"kind": "relic", "item_id": "anchor", "price": 80, "purchased": False},
            {"kind": "relic", "item_id": "kunai", "price": 175, "purchased": False},
            {"kind": "relic", "item_id": "shovel", "price": 500, "purchased": False},
            {"kind": "card", "item_id": "strike", "price": 50, "purchased": False},
            {"kind": "relic", "item_id": "bag", "price": 50, "purchased": True},
        ]
    }

    breakdown = learning_reward_breakdown(
        previous,
        previous,
        action_descriptor={"type": "shop_leave"},
    )

    expected = (
        DEFAULT_REWARD_CONFIG.shop_affordable_relic_leave_penalty
        - min(
            DEFAULT_REWARD_CONFIG.shop_unaffordable_relic_penalty_cap,
            75 * abs(DEFAULT_REWARD_CONFIG.shop_unaffordable_relic_shortfall_weight),
        )
    )
    assert breakdown.opportunity_cost_penalty == pytest.approx(expected)


def test_shop_leave_multiple_affordable_relics_use_diminishing_penalty() -> None:
    previous = _base_payload(phase="shop", gold=500)
    previous["shop"] = {
        "items": [
            {"kind": "relic", "item_id": "anchor", "price": 80, "purchased": False},
            {"kind": "relic", "item_id": "kunai", "price": 100, "purchased": False},
            {"kind": "relic", "item_id": "shovel", "price": 120, "purchased": False},
        ]
    }

    breakdown = learning_reward_breakdown(
        previous,
        previous,
        action_descriptor={"type": "shop_leave"},
    )

    expected = DEFAULT_REWARD_CONFIG.shop_affordable_relic_leave_penalty * (
        1
        + DEFAULT_REWARD_CONFIG.shop_affordable_relic_leave_decay
        + DEFAULT_REWARD_CONFIG.shop_affordable_relic_leave_decay**2
    )
    linear = DEFAULT_REWARD_CONFIG.shop_affordable_relic_leave_penalty * 3
    assert breakdown.opportunity_cost_penalty == pytest.approx(expected)
    assert abs(breakdown.opportunity_cost_penalty) < abs(linear)


def test_courier_shop_leave_opportunity_cost_uses_restock_cap() -> None:
    previous = _base_payload(phase="shop", gold=999, relics=("the_courier",))
    previous["shop"] = {
        "items": [
            {"kind": "relic", "item_id": f"relic_{index}", "price": 80, "purchased": False}
            for index in range(6)
        ]
    }

    breakdown = learning_reward_breakdown(
        previous,
        previous,
        action_descriptor={"type": "shop_leave"},
    )

    assert breakdown.opportunity_cost_penalty == pytest.approx(
        -DEFAULT_REWARD_CONFIG.shop_restock_opportunity_penalty_cap
    )


def test_ancient_choice_opportunity_cost_only_penalizes_visibly_lower_branch() -> None:
    previous = _base_payload(phase="ancient")
    previous["ancient"] = {
        "ancient_id": "neow",
        "options": [
            {"option_id": "relic", "relic_id": "anchor", "random_relic_count": 0},
            {"option_id": "upgrade", "upgrade_random_count": 1},
        ],
    }

    lower = learning_reward_breakdown(
        previous,
        previous,
        action_descriptor={
            "type": "choose_ancient",
            "ancient_option": {"option_id": "upgrade", "upgrade_random_count": 1},
        },
    )
    stronger = learning_reward_breakdown(
        previous,
        previous,
        action_descriptor={
            "type": "choose_ancient",
            "ancient_option": {"option_id": "relic", "relic_id": "anchor"},
        },
    )

    assert lower.opportunity_cost_penalty < 0.0
    assert stronger.opportunity_cost_penalty == 0.0


def test_proceeding_after_gold_claimed_is_not_penalized() -> None:
    previous = _base_payload(
        phase="reward",
        reward={
            "reward_id": "combat:1",
            "source": "combat",
            "gold": 19,
            "gold_claimed": True,
        },
    )

    breakdown = learning_reward_breakdown(
        previous,
        previous,
        action_descriptor={"type": "proceed"},
    )

    assert breakdown.reward_skip_penalty == 0.0
    assert breakdown.total == 0.0


def test_deck_capability_reward_credits_mechanical_growth() -> None:
    previous = _base_payload(
        deck=(
            {"card_id": "strike", "type": "attack", "effects": {"damage": 6}},
            {"card_id": "defend", "type": "skill", "effects": {"block": 5}},
        )
    )
    current = _base_payload(
        deck=(
            {"card_id": "strike", "type": "attack", "effects": {"damage": 6}},
            {"card_id": "defend", "type": "skill", "effects": {"block": 5}},
            {
                "card_id": "pommel_strike",
                "type": "attack",
                "effects": {"damage": 9, "draw": 1},
            },
        )
    )

    breakdown = learning_reward_breakdown(previous, current)

    assert breakdown.deck_capability_reward > 0.0
    assert breakdown.deck_capability_reward <= DEFAULT_REWARD_CONFIG.deck_capability_reward_cap


def test_deck_delta_summary_reports_fit_improvements_and_pressure() -> None:
    previous = _base_payload(
        deck=(
            {"card_id": "strike", "type": "attack", "cost": 1, "effects": {"damage": 6}},
            {"card_id": "defend", "type": "skill", "cost": 1, "effects": {"block": 5}},
            {"card_id": "bash", "type": "attack", "cost": 2, "effects": {"damage": 8}},
        )
    )
    current = _base_payload(
        deck=(
            {"card_id": "strike", "type": "attack", "cost": 1, "effects": {"damage": 6}},
            {"card_id": "defend", "type": "skill", "cost": 1, "effects": {"block": 5}},
            {"card_id": "bash", "type": "attack", "cost": 2, "effects": {"damage": 8}},
            {"card_id": "big_hit", "type": "attack", "cost": 3, "effects": {"damage": 20}},
        )
    )

    summary = deck_delta_summary(previous, current)

    assert summary["category_deltas"]["frontload"] > 0.0
    assert summary["problems_worsened"]["energy_heavy"] > 0.0


def test_large_deck_growth_needs_clear_fit_to_receive_deck_reward() -> None:
    previous = _base_payload(deck=_deck_cards(22))
    mediocre_growth = _base_payload(
        deck=(
            *_deck_cards(22),
            {
                "instance_id": "heavy_attack",
                "card_id": "heavy_attack",
                "type": "attack",
                "cost": 2,
                "effects": {"damage": 9},
            },
        )
    )
    useful_growth = _base_payload(
        deck=(
            *_deck_cards(22),
            {
                "instance_id": "pommel",
                "card_id": "pommel_strike",
                "type": "attack",
                "cost": 1,
                "effects": {"damage": 9, "draw": 1},
            },
        )
    )

    mediocre = learning_reward_breakdown(previous, mediocre_growth)
    useful = learning_reward_breakdown(previous, useful_growth)
    mediocre_summary = deck_delta_summary(previous, mediocre_growth)
    useful_summary = deck_delta_summary(previous, useful_growth)

    assert mediocre.deck_capability_reward == 0.0
    assert mediocre_summary["growth_blocked"] is True
    assert useful.deck_capability_reward > 0.0
    assert useful_summary["growth_blocked"] is False
    assert useful_summary["problem_relief"]["low_draw"] > 0.0


def test_curse_burden_penalty_scales_down_in_large_decks() -> None:
    small_before = _base_payload(deck=_deck_cards(10))
    small_after = _base_payload(deck=(*_deck_cards(10), _greed_card()), gold=333)
    large_before = _base_payload(deck=_deck_cards(30))
    large_after = _base_payload(deck=(*_deck_cards(30), _greed_card()), gold=333)

    small = learning_reward_breakdown(small_before, small_after)
    large = learning_reward_breakdown(large_before, large_after)

    assert small.deck_burden_penalty < 0.0
    assert large.deck_burden_penalty < 0.0
    assert abs(large.deck_burden_penalty) < abs(small.deck_burden_penalty)
    assert small.total < large.total


def test_curse_burden_penalty_respects_compensation_and_deck_support() -> None:
    unsupported_before = _base_payload(deck=_deck_cards(10))
    unsupported_after = _base_payload(deck=(*_deck_cards(10), _bad_curse()))
    compensated_after = _base_payload(deck=(*_deck_cards(10), _greed_card()), gold=333)
    supported_before = _base_payload(deck=(*_deck_cards(10), *_draw_exhaust_cards()))
    supported_after = _base_payload(deck=(*_deck_cards(10), *_draw_exhaust_cards(), _bad_curse()))

    unsupported = learning_reward_breakdown(unsupported_before, unsupported_after)
    compensated = learning_reward_breakdown(unsupported_before, compensated_after)
    supported = learning_reward_breakdown(supported_before, supported_after)

    assert unsupported.deck_burden_penalty < compensated.deck_burden_penalty < 0.0
    assert unsupported.deck_burden_penalty < supported.deck_burden_penalty < 0.0


def test_curse_burden_penalty_uses_source_relic_compensation() -> None:
    previous = _base_payload(deck=_deck_cards(10))
    unsupported = learning_reward_breakdown(
        previous,
        _base_payload(deck=(*_deck_cards(10), _bad_curse())),
    )
    calling_bell = learning_reward_breakdown(
        previous,
        _base_payload(
            deck=(*_deck_cards(10), _plain_curse("curse_of_the_bell")),
            relics=("calling_bell",),
        ),
    )

    assert unsupported.deck_burden_penalty < calling_bell.deck_burden_penalty < 0.0


def test_curse_burden_penalty_uses_event_frontloaded_compensation() -> None:
    previous = _base_payload(deck=_deck_cards(10))
    unsupported = learning_reward_breakdown(
        previous,
        _base_payload(deck=(*_deck_cards(10), _plain_curse("doubt"))),
    )
    compensated = learning_reward_breakdown(
        previous,
        _base_payload(
            deck=(*_deck_cards(10), _plain_curse("doubt")),
            reward={
                "card_options": ["pommel_strike", "shrug_it_off", "armaments"],
                "card_option_groups": [["battle_trance", "warcry", "ghostly_armor"]],
            },
        ),
        action_descriptor={
            "type": "choose_event",
            "event_option": {
                "metadata": {
                    "fixed_card_ids": ("doubt",),
                    "card_reward_count": 2,
                }
            },
        },
    )

    assert unsupported.deck_burden_penalty < compensated.deck_burden_penalty < 0.0


def test_eternal_unplayable_burden_is_worse_than_plain_curse() -> None:
    previous = _base_payload(deck=_deck_cards(12))
    plain = learning_reward_breakdown(
        previous,
        _base_payload(deck=(*_deck_cards(12), _bad_curse())),
    )
    eternal = learning_reward_breakdown(
        previous,
        _base_payload(deck=(*_deck_cards(12), _bad_curse(eternal=True))),
    )

    assert eternal.deck_burden_penalty < plain.deck_burden_penalty < 0.0


def test_terminal_starter_similarity_penalizes_unchanged_starter_deck() -> None:
    deck = _starter_deck()

    breakdown = learning_reward_breakdown(
        _base_payload(floor=8, deck=deck),
        _base_payload(phase="failed", floor=8, deck=deck),
    )
    summary = starter_dependency_summary(_base_payload(phase="failed", floor=8, deck=deck))

    assert breakdown.starter_deck_similarity_penalty < 0.0
    assert summary["starter_retention"] == 1.0
    assert summary["starter_share"] == 1.0
    assert summary["dependency_score"] > summary["threshold"]


def test_starter_similarity_penalty_respects_deck_improvements() -> None:
    starter = _starter_deck()
    upgraded = tuple(card | {"upgraded": True} for card in starter)

    unchanged = learning_reward_breakdown(
        _base_payload(floor=8, deck=starter),
        _base_payload(phase="failed", floor=8, deck=starter),
    )
    improved = learning_reward_breakdown(
        _base_payload(floor=8, deck=upgraded),
        _base_payload(phase="failed", floor=8, deck=upgraded, relics=("anchor",)),
    )

    assert improved.starter_deck_similarity_penalty <= 0.0
    assert abs(improved.starter_deck_similarity_penalty) < abs(
        unchanged.starter_deck_similarity_penalty
    )


def test_starter_similarity_penalty_does_not_apply_to_successful_runs() -> None:
    deck = _starter_deck()

    breakdown = learning_reward_breakdown(
        _base_payload(floor=12, deck=deck),
        _base_payload(phase="complete", floor=12, deck=deck),
    )

    assert breakdown.starter_deck_similarity_penalty == 0.0


def test_starter_similarity_penalty_respects_floor_gate() -> None:
    deck = _starter_deck()

    breakdown = learning_reward_breakdown(
        _base_payload(floor=5, deck=deck),
        _base_payload(phase="failed", floor=5, deck=deck),
    )

    assert breakdown.starter_deck_similarity_penalty == 0.0


def test_starter_similarity_penalty_requires_deck_weakness() -> None:
    deck = _starter_deck()
    config = DEFAULT_REWARD_CONFIG.model_copy(update={"starter_deck_problem_threshold": 1.0})

    breakdown = learning_reward_breakdown(
        _base_payload(floor=12, deck=deck),
        _base_payload(phase="failed", floor=12, deck=deck),
        config=config,
    )

    assert breakdown.starter_deck_similarity_penalty == 0.0


def test_starter_dependency_counts_duplicate_starter_cards_as_bloat() -> None:
    deck = (*_starter_deck(), *_deck_cards(4))
    state = _base_payload(phase="failed", floor=12, act=2, deck=deck)
    summary = starter_dependency_summary(state)
    breakdown = learning_reward_breakdown(_base_payload(floor=12, deck=deck), state)

    assert summary["duplicate_starter_count"] == 4
    assert summary["total_starter_weight"] > summary["retained_starter_weight"]
    assert breakdown.starter_deck_similarity_penalty < 0.0


def test_starter_dependency_can_use_tracker_baseline_for_custom_starters() -> None:
    custom_starter = (
        {
            "instance_id": "zap_0",
            "card_id": "zap",
            "type": "attack",
            "effects": {"damage": 4},
        },
        {
            "instance_id": "guard_0",
            "card_id": "guard",
            "type": "skill",
            "effects": {"block": 4},
        },
    )
    tracker = LearningRewardTracker(
        starter_deck_counts={"zap": 1, "guard": 1},
        starter_deck_size=2,
        starter_deck_capability_score=0.0,
    )

    breakdown = learning_reward_breakdown(
        _base_payload(floor=10, deck=custom_starter),
        _base_payload(phase="failed", floor=10, deck=custom_starter),
        tracker=tracker,
    )
    summary = starter_dependency_summary(
        _base_payload(phase="failed", floor=10, deck=custom_starter),
        tracker=tracker,
    )

    assert summary["baseline_counts"] == {"zap": 1, "guard": 1}
    assert summary["starter_retention"] == 1.0
    assert breakdown.starter_deck_similarity_penalty < 0.0


def test_node_progress_reward_is_tiny_and_once_per_tracker() -> None:
    tracker = LearningRewardTracker()
    previous = _base_payload(room_history=("a1:0:0",))
    current = _base_payload(room_history=("a1:0:0", "a1:1:0"))

    first = learning_reward_breakdown(previous, current, tracker=tracker)
    second = learning_reward_breakdown(previous, current, tracker=tracker)

    assert first.node_progress_reward == 0.1
    assert second.node_progress_reward == 0.0


@pytest.mark.parametrize(
    ("kind", "act", "combat_reward", "boss_reward"),
    [
        ("monster", 1, 2.0, 0.0),
        ("elite", 1, 10.0, 0.0),
        ("boss", 1, 0.0, 50.0),
        ("boss", 2, 0.0, 150.0),
        ("boss", 3, 0.0, 400.0),
    ],
)
def test_combat_win_rewards_scale_by_room_and_act(
    kind: str,
    act: int,
    combat_reward: float,
    boss_reward: float,
) -> None:
    previous = _combat_payload(kind=kind, act=act, turn=10)
    current = _base_payload(phase="reward", act=act, floor=1, kind=kind)

    breakdown = learning_reward_breakdown(previous, current)

    assert breakdown.combat_win_reward == combat_reward
    assert breakdown.boss_reward == boss_reward
    assert breakdown.combat_pace_reward == 0.0


def test_fast_combat_bonus_only_applies_before_turn_ten() -> None:
    previous = _combat_payload(kind="elite", turn=2)
    current = _base_payload(phase="reward", kind="elite")

    breakdown = learning_reward_breakdown(previous, current)

    assert breakdown.combat_pace_reward == pytest.approx(2.4)


def test_enemy_hp_progress_cannot_be_refarmed_after_healing() -> None:
    tracker = LearningRewardTracker(max_enemy_hp_progress={"0:test_monster": 20})
    previous = _combat_payload(monster_hp=80)
    current = _combat_payload(monster_hp=70)

    first = learning_reward_breakdown(previous, current, tracker=tracker)
    healed = _combat_payload(monster_hp=90)
    second = learning_reward_breakdown(current, healed, tracker=tracker)
    redamaged = _combat_payload(monster_hp=70)
    third = learning_reward_breakdown(healed, redamaged, tracker=tracker)

    assert first.enemy_hp_progress_reward > 0
    assert second.enemy_hp_progress_reward == 0.0
    assert third.enemy_hp_progress_reward == 0.0


def test_prevented_hp_reward_requires_projected_hp_saved() -> None:
    previous = _combat_payload(player_block=0, intent_damage=12)
    current = _combat_payload(player_block=12, intent_damage=12)

    blocked = learning_reward_breakdown(
        previous,
        current,
        action_descriptor={"preview": {"projected_damage_taken_after_end": 0}},
    )
    unnecessary = learning_reward_breakdown(
        _combat_payload(player_block=20, intent_damage=12),
        _combat_payload(player_block=25, intent_damage=12),
        action_descriptor={"preview": {"projected_damage_taken_after_end": 0}},
    )

    assert blocked.prevented_hp_reward > 0
    assert unnecessary.prevented_hp_reward == 0.0


def test_wasteful_potion_discard_is_contextual() -> None:
    previous = _base_payload(potions=("fire_potion",))

    waste = learning_reward_breakdown(
        previous,
        _base_payload(potions=()),
        action_descriptor={"type": "discard_potion"},
    )
    slot_pressure = learning_reward_breakdown(
        previous,
        _base_payload(potions=()),
        action_descriptor={
            "type": "discard_potion",
            "potion_strategy": {"belt_full": True, "slot_pressure": 1.0},
        },
    )

    assert waste.potion_waste_penalty == -0.5
    assert slot_pressure.potion_waste_penalty == 0.0


def test_env_info_exposes_reward_breakdown() -> None:
    env = Sts2Env(seed=1, character_id="TEST", reward_fn=learning_reward)
    try:
        _observation, info = env.reset()
        action_id = info["action_space"][0]["id"]
        _observation, reward, _terminated, _truncated, info = env.step(action_id)
    finally:
        env.close()

    assert info["reward_breakdown"]["total"] == reward
    assert "aggression_pressure" in info["reward_breakdown"]
    assert "reward_aggression_pressure" in info["agent_memory"]["entries"][0]


def _base_payload(
    *,
    phase: str = "map",
    act: int = 1,
    floor: int = 1,
    hp: int = 50,
    max_hp: int = 80,
    gold: int = 0,
    kind: str = "monster",
    room_history: tuple[str, ...] = (),
    potions: tuple[str, ...] = (),
    relics: tuple[str, ...] = (),
    deck: tuple[dict, ...] = (),
    reward: dict | None = None,
) -> dict:
    payload = {
        "phase": phase,
        "act": act,
        "floor": floor,
        "player": {
            "hp": hp,
            "max_hp": max_hp,
            "block": 0,
            "gold": gold,
            "relics": list(relics),
            "deck": list(deck),
            "deck_count": len(deck),
        },
        "master_deck": list(deck),
        "potions": list(potions),
        "relics": list(relics),
        "room_history": list(room_history),
        "map": {
            "current_node_id": "a1:1:0",
            "completed_node_ids": [],
            "nodes": [
                {
                    "node_id": "a1:1:0",
                    "act": act,
                    "floor": floor,
                    "lane": 0,
                    "kind": kind,
                }
            ],
        },
    }
    if reward is not None:
        payload["reward"] = reward
    return payload


def _combat_payload(
    *,
    kind: str = "monster",
    act: int = 1,
    turn: int = 1,
    hp: int = 50,
    max_hp: int = 80,
    player_block: int = 0,
    monster_hp: int = 80,
    intent_damage: int = 8,
) -> dict:
    payload = _base_payload(phase="combat", act=act, floor=1, hp=hp, max_hp=max_hp, kind=kind)
    payload["combat"] = {
        "turn": turn,
        "player": {"hp": hp, "max_hp": max_hp, "block": player_block, "gold": 0},
        "monsters": [
            {
                "monster_id": "test_monster",
                "name": "Test Monster",
                "hp": monster_hp,
                "max_hp": 100,
                "intent": "attack",
                "intent_damage": intent_damage,
                "hit_count": 1,
                "statuses": {},
                "metadata": {},
            }
        ],
        "metadata": {"room_kind": kind, "combat_id": "test-combat"},
    }
    return payload


def _deck_cards(count: int) -> tuple[dict, ...]:
    return tuple(
        {
            "instance_id": f"strike_{index}",
            "card_id": "strike",
            "type": "attack",
            "effects": {"damage": 6},
        }
        for index in range(count)
    )


def _starter_deck() -> tuple[dict, ...]:
    cards: list[dict] = []
    for index in range(5):
        cards.append(
            {
                "instance_id": f"strike_{index}",
                "card_id": "strike",
                "type": "attack",
                "effects": {"damage": 6},
            }
        )
    for index in range(4):
        cards.append(
            {
                "instance_id": f"defend_{index}",
                "card_id": "defend",
                "type": "skill",
                "effects": {"block": 5},
            }
        )
    cards.append(
        {
            "instance_id": "bash_0",
            "card_id": "bash",
            "type": "attack",
            "effects": {"damage": 8, "apply_status": {"target": "enemy", "vulnerable": 2}},
        }
    )
    return tuple(cards)


def _greed_card() -> dict:
    return {
        "instance_id": "greed_1",
        "card_id": "greed",
        "type": "curse",
        "tags": ["curse", "eternal", "unplayable"],
        "custom": {"eternal": True, "frontloaded_gold": 333},
        "effects": {"noop": {"reason": "frontloaded_gold_curse"}},
    }


def _plain_curse(card_id: str) -> dict:
    return {
        "instance_id": f"{card_id}_1",
        "card_id": card_id,
        "type": "curse",
        "tags": ["curse", "unplayable"],
        "custom": {"unplayable": True},
        "effects": {"noop": {"reason": "curse_burden"}},
    }


def _bad_curse(*, eternal: bool = False) -> dict:
    tags = ["curse", "unplayable"]
    custom = {"unplayable": True}
    if eternal:
        tags.append("eternal")
        custom["eternal"] = True
    return {
        "instance_id": f"bad_curse_{int(eternal)}",
        "card_id": "bad_curse",
        "type": "curse",
        "tags": tags,
        "custom": custom,
        "effects": {"noop": {"reason": "dead_draw"}},
    }


def _draw_exhaust_cards() -> tuple[dict, ...]:
    return (
        {
            "instance_id": "battle_trance_1",
            "card_id": "battle_trance",
            "type": "skill",
            "cost": 0,
            "effects": {"draw": 3},
        },
        {
            "instance_id": "true_grit_1",
            "card_id": "true_grit",
            "type": "skill",
            "cost": 1,
            "effects": {"block": 7, "exhaust_choice": 1},
        },
    )
