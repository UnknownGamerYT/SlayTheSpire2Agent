"""Royalties follows source room-reward population and explicit claiming."""
from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _source, _state
from combat.test_glam import _play
from events.test_stable_event_fights import _fight, _kill

from sts2sim import legal_actions, step
from sts2sim.engine import CardEnchantment, RunPhase
from sts2sim.engine.source_royalties import after_combat


def _win(*, upgraded=False, replay=False, relics=(), base_gold=0):
    card = _source("ROYALTIES", upgraded=upgraded)
    if replay:
        card = card.model_copy(update={
            "enchantments": (CardEnchantment(keyword="glam", amount=1),),
        })
    kill = _simple("kill", card_type="attack", target="enemy", effects={"damage": 999})
    state = _state(card, kill, relics=relics)
    state = state.model_copy(update={"flags": {
        **state.flags, "combat_reward_gold": base_gold, "combat_reward_gold_is_base": True,
    }})
    state = _play(_reload(state), "royalties")
    assert state.player.gold == 0
    assert state.player.max_hp == 80
    assert state.player.statuses["royalties"] == (40 if upgraded else 30) * (2 if replay else 1)
    assert not state.combat.metadata.get("combat_triggers")
    return _play(_reload(state), "kill")


@pytest.mark.parametrize("upgraded,amount", [(False, 30), (True, 40)])
def test_royalties_creates_deferred_skippable_gold_without_victory_gain(upgraded, amount):
    state = _win(upgraded=upgraded, relics=("dragon_fruit",))
    assert state.phase == RunPhase.REWARD
    assert state.reward.gold == amount
    assert state.player.gold == 0
    assert state.player.max_hp == 80
    assert state.combat.metadata["source_extra_reward_queue"] == ({
        "kind": "gold", "source_id": "royalties", "amount": amount,
        "was_gold_stolen_back": False,
    },)
    same, events = after_combat(state.combat)
    assert same == state.combat and not events
    state = step(_reload(state), next(a for a in legal_actions(state)
        if a.type == "skip_reward" and "gold" in a.target_id))
    assert state.player.gold == 0
    assert state.player.max_hp == 80


def test_replayed_royalties_stacks_into_one_reward_and_one_on_gain_notification():
    state = _win(replay=True, relics=("bowler_hat", "dragon_fruit"))
    assert state.reward.gold == 60
    assert len(state.reward.metadata["gold_reward_components"]) == 1
    assert state.player.gold == 0
    before = state.rng
    state = step(_reload(state), next(a for a in legal_actions(state)
        if a.type == "take_reward_gold"))
    assert state.player.gold == 75
    assert state.player.max_hp == 81
    assert state.rng == before


def test_royalties_remains_independent_from_base_and_amethyst_and_can_claim_out_of_order():
    state = _win(relics=("bowler_hat", "dragon_fruit", "amethyst_aubergine"), base_gold=11)
    assert [x["amount"] for x in state.reward.metadata["gold_reward_components"]] == [11, 30, 15]
    before = state.rng
    state = step(_reload(state), next(a for a in legal_actions(state)
        if a.type == "take_reward_gold" and a.target_id == "reward:gold:1"))
    assert state.player.gold == 37
    assert state.player.max_hp == 81
    assert state.rng == before


def test_failed_combat_never_queues_or_pays_royalties():
    kill = _simple("fatal", card_type="attack", target="enemy", effects={
        "sequence": [{"damage": 999}, {"hp_loss": 999}],
    })
    state = _play(_state(_source("ROYALTIES"), kill, relics=("dragon_fruit",)), "royalties")
    state = _play(_reload(state), "fatal")
    assert state.phase == RunPhase.FAILED
    assert state.reward is None
    assert state.player.gold == 0
    assert state.player.max_hp == 80
    assert not state.combat.metadata.get("source_royalties_reward_queued")


def test_reward_suppressed_dummy_does_not_offer_royalties_gold():
    state = _fight("battleworn_dummy", source_rng=True)
    player = state.player.model_copy(update={
        "statuses": {**state.player.statuses, "royalties": 30},
    })
    state = state.model_copy(update={"player": player,
        "combat": state.combat.model_copy(update={"player": player})})
    state = _kill(_reload(state))
    assert state.combat.metadata["source_royalties_reward_queued"]
    assert state.player.gold == 0
    assert state.reward.gold == 0
    assert not state.reward.metadata.get("gold_reward_components")
