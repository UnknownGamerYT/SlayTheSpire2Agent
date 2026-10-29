"""Stable source tag, phase and value-prop contracts at public action boundaries.

Provenance and actual reviewed method scopes: fixtures/source_item_math.json.
"""
from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _source, _state
from combat.test_glam import _end_turn, _play
from combat.test_stable_item_contracts import _potion
from rewards.test_relic_deck_choices import _choose, _reward_state

from sts2sim import legal_actions, step
from sts2sim.engine import CardEnchantment, OrbState


@pytest.mark.parametrize("tag,name,expected", [
    ("strike", "Plain", 9), ("other", "Misleading Strike", 6),
])
def test_strike_dummy_checks_actual_tag_instead_of_display_text(tag, name, expected):
    card = _simple("custom", name=name, tags=(tag,), card_type="attack", target="enemy",
                   effects={"damage": 6})
    result = _play(_state(card, relics=("strike_dummy",)), "custom")
    assert result.combat.monsters[0].hp == 100 - expected


def test_vitruvian_minion_modifies_tagged_damage_and_block_before_rounding():
    card = _simple("tagged", tags=("minion",), card_type="attack", target="enemy",
                   effects={"damage": 9, "block": 5})
    state = _state(card, relics=("vitruvian_minion",),
                   statuses={"strength": 1, "dexterity": 1, "weak": 2, "frail": 2})
    result = _play(_reload(state), "tagged")
    assert result.combat.monsters[0].hp == 85
    assert result.player.block == 9


def test_vitruvian_minion_does_not_infer_tag_from_name():
    card = _simple("misleading_minion", name="Minion", effects={"block": 5})
    assert _play(_state(card, relics=("vitruvian_minion",)), card.instance_id).player.block == 5


@pytest.mark.parametrize("stacks,expected", [(1, 24), (2, 48)])
def test_shadowmeld_multiplies_unpowered_potion_block_without_dexterity_or_frail(stacks, expected):
    state = _state(_simple("wait"), statuses={"shadowmeld": stacks, "dexterity": 9, "frail": 1})
    result = _potion(_reload(state), "block_potion")
    assert result.player.block == expected


def test_replayed_shadowmeld_stacks_and_modifies_orb_block_before_owner_side_expiry():
    card = _source("SHADOWMELD").model_copy(update={
        "enchantments": (CardEnchantment(keyword="replay", amount=1),),
    })
    state = _play(_state(card), "shadowmeld")
    assert state.player.statuses["shadowmeld"] == 2
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "orbs": (OrbState(orb_id="frost"),), "orb_slots": 1,
    })})
    result = _end_turn(_reload(state))
    orb_event = next(e for e in result.combat.last_events if e.kind == "player_block"
                     and e.metadata.get("orb") == "frost")
    assert orb_event.amount == 8
    assert "shadowmeld" not in result.player.statuses


def test_self_forming_clay_observes_nonlethal_card_hp_loss_once():
    card = _simple("pay", effects={"hp_loss": 4})
    result = _play(_state(card, relics=("self_forming_clay", "tungsten_rod")), "pay")
    assert result.player.hp == 77
    assert result.player.statuses["next_turn_block"] == 3
    assert result.combat.metadata["hp_lost_this_turn"] == 3


def test_precarious_shears_defers_blockable_damage_until_removal_finishes_and_survives_reload():
    state = _reward_state("precarious_shears")
    state = state.model_copy(update={
        "relics": ("tungsten_rod",),
        "player": state.player.model_copy(update={"block": 10}),
    })
    state = step(state, next(a for a in legal_actions(state) if a.type == "take_reward_relic"))
    assert state.player.hp == 60
    state = _choose(_reload(state))
    assert state.player.hp == 60
    result = _choose(_reload(state))
    assert result.player.hp == 55  # 16 unpowered, blockable damage -10 Block -1 Tungsten.
    assert result.player.block == 0
    assert not result.flags.get("source_precarious_shears_damage_pending")
    assert not result.flags.get("pending_relic_deck_choices")
