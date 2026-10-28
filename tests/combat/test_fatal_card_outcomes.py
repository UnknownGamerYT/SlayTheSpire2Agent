from __future__ import annotations

import pytest
from combat.test_combat_engine_monsters import _play_card_targeting, _play_first_card
from combat.test_monster_innate_hooks import _card, _combat, _monster

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.mechanics.card_effects import card_effect_plan


def _fatal(card_id: str, *, upgraded: bool = False, hits: int = 1) -> dict:
    descriptions = {
        "feed": ("raise your Max HP by 3", "raise your Max HP by 4", 10, 1),
        "hand_of_greed": ("gain 20 [gold]Gold[/gold]", "gain 25 [gold]Gold[/gold]", 20, 2),
        "the_hunt": ("gain an additional card reward", "gain an additional card reward", 10, 1),
    }
    base, upgrade, damage, cost = descriptions[card_id]
    return dict(card_effect_plan({
        "id": card_id, "type": "Attack", "cost": cost, "target": "enemy", "damage": damage,
        "hit_count": hits, "description": f"Deal {damage} damage. If [gold]Fatal[/gold], {base}.",
        "upgrade_description": f"Deal {damage} damage. If [gold]Fatal[/gold], {upgrade}.",
        "upgraded": upgraded,
    }).card)


def _fight(card_id: str, *, hp: int = 5, powers: dict | None = None, **kwargs):
    monster = _monster("VICTIM", powers or {})
    monster.update(min_hp=hp, max_hp=hp)
    return _combat(monster, _monster("SURVIVOR", {}), deck=(
        _fatal(card_id, **kwargs), _card("finish", damage=1000),
    ))


@pytest.mark.parametrize("upgraded,gain", [(False, 3), (True, 4)])
def test_feed_fatal_raises_and_heals_max_hp_persistently(upgraded: bool, gain: int) -> None:
    state = _fight("feed", upgraded=upgraded)
    old = state.player.max_hp
    state = _play_card_targeting(state, "feed", state.combat.monsters[0].monster_id)
    assert state.player.max_hp == old + gain
    assert state.player.hp == old + gain
    state = load_state(serialize(state))
    state = _play_first_card(state, "finish")
    assert state.phase.value == "reward"
    assert state.player.max_hp == old + gain


@pytest.mark.parametrize("card_id", ["feed", "hand_of_greed", "the_hunt"])
@pytest.mark.parametrize("hp,powers", [(100, {}), (5, {"MINION": 1}), (5, {"ILLUSION": 1})])
def test_fatal_requires_a_real_nonminion_kill(card_id: str, hp: int, powers: dict) -> None:
    state = _fight(card_id, hp=hp, powers=powers)
    old = state.player
    state = _play_card_targeting(state, card_id, state.combat.monsters[0].monster_id)
    assert state.player.gold == old.gold
    assert state.player.max_hp == old.max_hp
    assert state.combat.metadata.get("fatal_card_reward_count", 0) == 0


@pytest.mark.parametrize("upgraded,gain", [(False, 20), (True, 25)])
def test_hand_of_greed_rewards_only_once_when_early_multihit_is_fatal(
    upgraded: bool, gain: int,
) -> None:
    state = _fight("hand_of_greed", upgraded=upgraded, hits=3)
    old = state.player.gold
    state = _play_card_targeting(state, "hand_of_greed", state.combat.monsters[0].monster_id)
    assert state.player.gold == old + gain
    assert sum(event.amount or 0 for event in state.combat.last_events
               if event.kind == "gold_gained") == gain


def test_hunt_adds_an_actual_skippable_reward_after_save_reload() -> None:
    state = _fight("the_hunt")
    # The source pool is required to draw actual reward groups.
    state = state.model_copy(update={"flags": {**state.flags,
        "combat_reward_card_pool": ("strike", "defend", "bash"),
    }})
    state = _play_card_targeting(state, "the_hunt", state.combat.monsters[0].monster_id)
    assert state.combat.metadata["fatal_card_reward_count"] == 1
    restored = load_state(serialize(state))
    results = [_play_first_card(candidate, "finish") for candidate in (state, restored)]
    assert serialize(results[0]) == serialize(results[1])
    state = results[0]
    assert len(state.reward.card_option_groups) == 1
    assert state.reward.card_option_groups[0]
    state = load_state(serialize(state))
    action = next(action for action in legal_actions(state)
                  if action.type == "take_reward_card"
                  and action.target_id.startswith("reward:card_group:0:"))
    before = len(state.master_deck)
    state = step(state, action)
    assert len(state.master_deck) == before + 1


@pytest.mark.parametrize("hp,expected", [(100, 0), (5, 3)])
def test_sunder_conditional_energy_is_never_also_unconditional(hp: int, expected: int) -> None:
    card = dict(card_effect_plan({
        "id": "SUNDER", "type": "Attack", "cost": 3, "damage": 24, "energy_gain": 3,
        "description": "Deal 24 damage. If this kills an enemy, gain [energy:3].",
    }).card)
    monster = _monster("VICTIM", {})
    monster.update(min_hp=hp, max_hp=hp)
    state = _combat(monster, _monster("SURVIVOR", {}), deck=(card,))
    state = _play_card_targeting(state, "sunder", state.combat.monsters[0].monster_id)
    assert state.combat.player.energy == expected


@pytest.mark.parametrize("monster_id,metadata,powers", [
    ("TEST_SUBJECT", {"test_subject_respawns": 0}, {}),
    ("WATERFALL_GIANT", {}, {"STEAM_ERUPTION": 5}),
    ("DECIMILLIPEDE_SEGMENT_1", {}, {"REATTACH": 25}),
])
def test_temporary_death_phases_do_not_trigger_fatal(
    monster_id: str, metadata: dict, powers: dict,
) -> None:
    monster = _monster(monster_id, powers)
    monster.update(min_hp=5, max_hp=5)
    other_id = "DECIMILLIPEDE_SEGMENT_2" if monster_id.startswith("DECIMILLIPEDE") else "SURVIVOR"
    state = _combat(monster, _monster(other_id, {}), deck=(_fatal("feed"),))
    first = state.combat.monsters[0]
    first = first.model_copy(update={"hp": 5, "metadata": {**first.metadata, **metadata}})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (first, state.combat.monsters[1]),
    })})
    state = _play_card_targeting(state, "feed", first.monster_id)
    assert state.player.max_hp == 80


def test_last_decimillipede_segment_is_fatal() -> None:
    monster = _monster("DECIMILLIPEDE_SEGMENT_1", {"REATTACH": 25})
    monster.update(min_hp=5, max_hp=5)
    state = _combat(monster, deck=(_fatal("feed"),))
    state = _play_first_card(state, "feed")
    assert state.phase.value == "reward"
    assert state.player.max_hp == 83


def test_hunt_final_kill_also_adds_the_extra_reward() -> None:
    monster = _monster("VICTIM", {})
    monster.update(min_hp=5, max_hp=5)
    state = _combat(monster, deck=(_fatal("the_hunt"),))
    state = state.model_copy(update={"flags": {**state.flags,
        "combat_reward_card_pool": ("strike", "defend", "bash"),
    }})
    state = _play_first_card(state, "the_hunt")
    assert state.phase.value == "reward"
    assert len(state.reward.card_option_groups) == 1


def test_sunder_kill_bonus_does_apply_to_minions() -> None:
    card = dict(card_effect_plan({
        "id": "SUNDER", "type": "Attack", "cost": 3, "damage": 24, "energy_gain": 3,
        "description": "Deal 24 damage. If this kills an enemy, gain [energy:3].",
    }).card)
    monster = _monster("VICTIM", {"MINION": 1})
    monster.update(min_hp=5, max_hp=5)
    state = _combat(monster, _monster("SURVIVOR", {}), deck=(card,))
    state = _play_card_targeting(state, "sunder", state.combat.monsters[0].monster_id)
    assert state.combat.player.energy == 3
    assert "energy" not in state.combat.player.resources
