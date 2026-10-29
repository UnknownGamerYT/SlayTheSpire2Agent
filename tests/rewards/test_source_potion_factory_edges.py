"""Finite source contracts for PotionFactory batches and forced reward odds."""
from __future__ import annotations

from random import Random

import pytest
from combat.test_card_play_status_runtime import _simple, _state
from combat.test_glam import _play
from rewards.test_relic_deck_choices import _reward_state

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine.rng import RunRng, capture_random_state, random_stream
from sts2sim.mechanics.combat_rewards import CombatRewardContext, roll_combat_potion_drop
from sts2sim.mechanics.rewards import PotionRarity, RewardPityState
from sts2sim.mechanics.shop_rooms import build_basic_shop_inventory
from sts2sim.mechanics.shops import ShopInventoryPlan, ShopItem, ShopItemKind
from sts2sim.mechanics.source_rng import draw_source_potions, float32


class _FirstCommon(Random):
    derivation = "sts2.v0.107.1.utf16-djb2-splitmix64"

    def random(self):
        return .8

    def choice(self, seq):
        return seq[0]


def test_one_source_factory_batch_excludes_each_selected_model():
    options = ("first", "second", "third")
    assert draw_source_potions(_FirstCommon(), options, 3,
                               dict.fromkeys(options, "common")) == options
    assert draw_source_potions(_FirstCommon(), (), 0, {}) == ()


def test_source_merchant_does_not_replace_an_empty_rolled_rarity_with_any_potion():
    class Rare(RunRng):
        def random(self):
            return .05

    plan = ShopInventoryPlan(colored_cards=0, colorless_cards=0, relics=0,
                             potions=1, include_card_removal=False)
    with pytest.raises(ValueError, match="no rare potion"):
        build_basic_shop_inventory(Rare("EMPTY_RARITY"),
            potion_pool=(ShopItem("fire_potion", ShopItemKind.POTION, PotionRarity.COMMON),),
            plan=plan)


def test_phial_holster_public_pickup_does_not_repeat_the_first_model():
    state = _reward_state("phial_holster").model_copy(update={
        "rng": capture_random_state(RunRng("8")),
    })
    action = next(a for a in legal_actions(state) if a.type == "take_reward_relic")
    result = step(state, action)
    assert serialize(result) == serialize(step(load_state(serialize(state)), action))
    # Same four raw draws formerly selected Flex twice when exclusions were lost.
    assert result.potions == ("flex_potion", "power_potion")
    assert result.rng.streams["combat_potion_generation"].draws == 4
    assert result.flags["bonus_potion_slots"] == 1


def test_forced_source_potion_still_draws_and_lowers_unbounded_pity():
    bank = RunRng("FORCED_POTION")
    rng = random_stream(bank, "rewards")
    roll = roll_combat_potion_drop(rng, RewardPityState(potion_current_value=-.5),
        CombatRewardContext("IRONCLAD", owned_relics=("WHITE_BEAST_STATUE",)))
    assert roll.forced and roll.dropped
    assert roll.state.potion_current_value == float32(-.5 - float32(.1))
    assert roll.state.potion_chance_bonus == -10
    assert capture_random_state(bank).streams["rewards"].draws == 1


def test_white_beast_public_victory_populates_one_potion_before_other_rewards():
    kill = _simple("kill", card_type="attack", target="enemy", effects={"damage": 999})
    state = _state(kill, relics=("white_beast_statue",)).model_copy(update={
        "character_id": "IRONCLAD", "rng": capture_random_state(RunRng("8")),
        "flags": {"combat_reward_card_count": 0, "combat_reward_relic_count": 0,
                  "potion_reward_odds": -.5},
    })
    result = _play(state, "kill")
    assert serialize(result) == serialize(_play(load_state(serialize(state)), "kill"))
    assert result.reward.potion_id is not None
    assert result.reward.potion_ids == ()
    assert result.reward.metadata["potion_drop"]["forced"]
    assert result.flags["potion_reward_odds"] == float32(-.5 - float32(.1))
    assert result.rng.streams["rewards"].draws == 4
    assert len([a for a in legal_actions(result) if a.type == "take_reward_potion"]) == 1
    assert not any(event.kind == "reward_extra_potion"
                   for event in result.replay_log[-1].events)
