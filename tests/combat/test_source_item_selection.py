"""Stable0.107.1 relic source selection and stream-use public regressions.

Pinned source5d874e855ae1d0550d5c30416fadd7dbb0d133b7:
SandCastle/WarPaint/Whetstone/WarHammer, SereTalon/NeowsBones,
PandorasBox, PhialHolster, RoyalStamp, StoneCracker/PowerCell/JeweledMask.
"""
from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _state
from combat.test_glam import _play
from combat.test_relic_engine_integration import DUMMY_ENCOUNTERS, DUMMY_MONSTERS, _force_next_room
from rewards.test_relic_deck_choices import _reward_state

from sts2sim import legal_actions, new_run, serialize, step
from sts2sim.engine import CardEnchantment, MapNodeState, MapState, RoomKind
from sts2sim.engine.rng import RunRng, capture_random_state


def _rng():
    return capture_random_state(RunRng(
        "ITEM_SELECTION", derivation="sts2.v0.107.1.utf16-djb2-splitmix64",
    ))


def _pickup(relic_id, cards=(), *, relics=()):
    state = _reward_state(relic_id, cards=cards)
    state = state.model_copy(update={"rng": _rng(), "relics": tuple(relics)})
    action = next(a for a in legal_actions(state) if a.type == "take_reward_relic")
    assert serialize(step(state, action)) == serialize(step(_reload(state), action))
    return step(state, action)


def _opening(relic_id, cards):
    state = new_run(seed="ITEM_SELECTION", character_id="IRONCLAD", ascension=0, source_data={
        "deck": cards, "player": {"hp": 80, "max_hp": 80, "energy": 3,
                                     "max_energy": 3, "resources": {"star": 10}},
        "monsters": DUMMY_MONSTERS, "encounters": DUMMY_ENCOUNTERS,
        "flags": {"draw_per_turn": 0, "combat_reward_card_count": 0},
    })
    state = _force_next_room(state, RoomKind.MONSTER)
    state = state.model_copy(update={"rng": _rng(), "relics": (relic_id,)})
    return step(state, next(a for a in legal_actions(state) if a.type == "choose_node"))


def _draws(state, stream):
    value = state.rng.streams.get(stream)
    return value.draws if value else 0


@pytest.mark.parametrize("relic_id,count,kind", [
    ("war_paint", 2, "skill"), ("whetstone", 2, "attack"), ("sand_castle", 6, "skill"),
])
def test_source_upgrade_relics_sort_full_list_then_niche_shuffle_before_take(relic_id, count, kind):
    cards = tuple(
        _simple(name, card_type=kind, cost=1) for name in ("z", "d", "b", "x", "a", "c", "w")
    )
    first, reverse = _pickup(relic_id, cards), _pickup(relic_id, tuple(reversed(cards)))
    upgraded = {c.instance_id for c in first.master_deck if c.upgraded}
    assert len(upgraded) == count
    assert upgraded == {c.instance_id for c in reverse.master_deck if c.upgraded}
    assert _draws(first, "niche") == 6  # Full7-card Fisher-Yates, independent of Take count.
    assert first.rng == reverse.rng


def test_source_war_hammer_uses_niche_full_shuffle_after_elite_victory():
    attack = _simple("kill", effects={"damage": 100}, card_type="attack", target="enemy")
    state = _state(attack, relics=("war_hammer",))
    node = MapNodeState(node_id="elite", act=1, floor=1, lane=0, kind=RoomKind.ELITE)
    cards = tuple(_simple(name, cost=1) for name in ("z", "d", "b", "x", "a", "c", "w"))
    state = state.model_copy(update={
        "character_id": "IRONCLAD", "rng": _rng(), "master_deck": cards,
        "map": MapState(act=1, nodes=(node,), current_node_id=node.node_id),
    })
    state = _play(state, "kill")
    assert sum(c.upgraded for c in state.master_deck) == 4
    assert _draws(state, "niche") == 6


def test_source_sere_talon_selects_distinct_modifier_curses_before_adding_wishes():
    state = _pickup("sere_talon")
    identifiers = [c.card_id for c in state.master_deck]
    assert len(identifiers) == 5
    assert identifiers[2:] == ["wish"] * 3
    assert len(set(identifiers[:2])) == 2
    assert set(identifiers[:2]) <= {
        "clumsy", "debt", "decay", "doubt", "guilty", "injury", "normality",
        "regret", "shame", "writhe",
    }
    assert _draws(state, "niche") == 2  # Explicit NextItem+Remove, not full shuffle.


@pytest.mark.parametrize("sozu", [False, True])
def test_source_phial_generates_both_potions_before_procurement_and_sozu(sozu):
    state = _pickup("phial_holster", relics=("sozu",) if sozu else ())
    assert state.flags["bonus_potion_slots"] == 1
    assert len(state.potions) == (0 if sozu else 2)
    assert _draws(state, "combat_potion_generation") == 4  # Rarity+item for each potion.
    blocked = [e for e in state.replay_log[-1].events if e.kind == "potion_procurement_blocked"]
    assert len(blocked) == (2 if sozu else 0)


def test_source_pandora_transforms_only_removable_basic_strike_or_defend_on_niche():
    cards = (
        _simple("basic_strike", card_type="attack", tags=("strike",),
                custom={"source_spec": {"rarity": "Basic", "color": "ironclad"}}),
        _simple("basic_defend", tags=("defend",),
                custom={"source_spec": {"rarity": "Basic", "color": "ironclad"}}),
        _simple("perfected_strike", card_type="attack"),
        _simple("misleading_defend_name"),
        _simple("eternal_strike", card_type="attack", tags=("strike",),
                custom={"eternal": True, "source_spec": {"rarity": "Basic", "color": "ironclad"}}),
    )
    state = _pickup("pandoras_box", cards)
    assert [c.instance_id for c in state.master_deck] == [c.instance_id for c in cards]
    assert state.master_deck[0].card_id != "basic_strike"
    assert state.master_deck[1].card_id != "basic_defend"
    assert [c.card_id for c in state.master_deck[2:]] == [c.card_id for c in cards[2:]]
    assert _draws(state, "niche") == 2


def test_source_royal_stamp_consumes_display_shuffle_then_selection_uses_no_rng():
    cards = tuple(_simple(name, card_type="attack") for name in ("z", "d", "b", "x", "a", "c"))
    cards += (_simple("power", card_type="power"), _simple("curse", card_type="curse"),
              _simple("occupied", enchantments=(CardEnchantment(keyword="steady", amount=1),)))
    state = _pickup("royal_stamp", cards)
    assert _draws(state, "niche") == 5
    offered = [a.card_instance_id for a in legal_actions(state) if a.type == "choose_card"]
    assert len(offered) == 6
    assert offered != [c.instance_id for c in cards[:6]]
    before = state.rng
    state = step(_reload(state), next(a for a in legal_actions(state) if a.type == "choose_card"))
    assert state.rng == before
    chosen = next(c for c in state.master_deck if c.instance_id == offered[0])
    assert chosen.enchantments[0].amount == 1
    assert chosen.custom["innate"] and chosen.custom["retain"]


def test_source_stone_cracker_sorts_and_shuffles_all_upgradable_draw_cards():
    cards = tuple({"card_id": name, "type": "skill", "cost": 1, "effects": {"block": 1}}
                  for name in ("z", "d", "b", "x", "a", "c", "w"))
    state = _opening("stone_cracker", cards)
    assert sum(c.upgraded for c in state.combat.draw_pile) == 2
    assert not any(c.upgraded for c in state.master_deck)
    assert _draws(state, "combat_card_selection") == 6


def test_source_power_cell_moves_two_local_zero_cards_without_free_stars_or_draw_events():
    cards = tuple({"card_id": name, "type": "skill", "cost": 0, "star_cost": 3}
                  for name in ("z", "d", "b", "x", "a", "c"))
    state = _opening("power_cell", cards)
    assert len(state.combat.hand) == 2
    assert _draws(state, "combat_card_selection") == 5
    assert not any(c.custom.get("free_to_play_this_turn") for c in state.combat.hand)
    assert not any(e.kind == "card_drawn" for e in state.combat.last_events)
    before = state.player.resources["star"]
    state = _play(state, state.combat.hand[0].instance_id)
    assert state.player.resources["star"] == before - 3


def test_source_jeweled_mask_selects_one_power_with_single_draw_and_move_provenance():
    cards = tuple({"card_id": name, "type": "power", "cost": 2, "star_cost": 3}
                  for name in ("z", "d", "b", "x", "a", "c"))
    state = _opening("jeweled_mask", cards)
    assert len(state.combat.hand) == 1
    assert _draws(state, "combat_card_selection") == 1
    assert state.combat.hand[0].custom["free_to_play_this_turn"]
    assert not any(e.kind == "card_drawn" for e in state.combat.last_events)
    state = _play(state, state.combat.hand[0].instance_id)
    assert state.player.energy == 3
    assert state.player.resources["star"] == 10
