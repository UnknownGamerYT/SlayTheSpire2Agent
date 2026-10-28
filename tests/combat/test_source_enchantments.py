"""Public combat fixtures pinned to stable 0.107.1 Enchantments/*.cs.

Source: Razhuo/sts2-rl-agent 5d874e855ae1d0550d5c30416fadd7dbb0d133b7.
Expected values follow source commands/hook ordering rather than API prose.
"""
from __future__ import annotations

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _source, _state
from combat.test_combat_engine_cards import _enter_combat
from combat.test_glam import _end_turn, _play

from sts2sim.engine.rng import random_from_state, random_stream
from sts2sim.engine.transitions import _enchant_card


def enchanted(name, *, amount=3, damage=4, block=None, **kwargs):
    effects = kwargs.pop("effects", {"damage": damage} if block is None else {"block": block})
    card = _simple(
        name, effects=effects,
        card_type="attack" if block is None else "skill",
        target="enemy" if block is None else "self", **kwargs,
    )
    return _enchant_card(card, name, amount, source_id="source_fixture")


@pytest.mark.parametrize(
    "name,expected", [("sharp", 6), ("instinct", 7), ("corrupted", 6),
                      ("inky", 5), ("tezcataras_ember", 6), ("vigorous", 6)],
)
def test_source_enchantment_arithmetic_precedes_strength_then_weak(name, expected):
    # Source stat enchantments: additive/multiplicative on base4, then Str2,
    # Weak 3/4, final floor. Amount3 is irrelevant to fixed-value models.
    state = _play(_state(enchanted(name), statuses={"strength": 2, "weak": 1}), name)
    assert state.combat.monsters[0].hp == 100 - expected


def test_source_vigorous_applies_to_every_first_play_hit_then_disables_for_replay():
    card = enchanted("vigorous", effects={"repeat_damage": {
        "basis": "energy_spent", "fixed_hits": 2, "amount": 4,
    }}, custom={"replay": 1})
    state = _play(_state(card), "vigorous")
    assert state.combat.monsters[0].hp == 78  # (4+3)*2 + 4*2
    assert state.combat.metadata["source_enchantment_disabled"]["vigorous"] == 1


def test_source_momentum_growth_is_after_main_effect_and_persists_in_reload():
    card = enchanted("momentum", custom={"replay": 1})
    state = _play(_state(card, draw_per_turn=1), "momentum")
    assert state.combat.monsters[0].hp == 89  # first4, replay7
    state = _reload(_end_turn(state))
    state = _play(state, "momentum")
    assert state.combat.monsters[0].hp == 66  # next10, replay13


@pytest.mark.parametrize("name,expected", [("adroit", 12), ("nimble", 10)])
def test_source_adroit_is_separate_block_command_and_nimble_modifies_native_block(name, expected):
    card = enchanted(name, block=5)
    state = _play(_state(card, statuses={"dexterity": 2}), name)
    assert state.player.block == expected  # Adroit:7+5, Nimble:5+3+2


def test_source_goopy_grows_after_each_play_and_updates_deck_version():
    card = enchanted("goopy", amount=1, block=5, custom={"replay": 1})
    state = _play(_state(card), "goopy")
    assert state.player.block == 11
    assert len(state.combat.exhaust_pile) == 1
    assert state.combat.exhaust_pile[0].enchantments[0].amount == 3
    assert state.master_deck[0].enchantments[0].amount == 3
    assert _reload(state).master_deck[0].enchantments[0].amount == 3


def test_source_corrupted_self_damage_bypasses_block_but_honors_intangible_and_buffer():
    for statuses, expected_hp in (({}, 78), ({"intangible": 1}, 79), ({"buffer": 1}, 80)):
        state = _state(enchanted("corrupted"), statuses=statuses)
        player = state.player.model_copy(update={"block": 10})
        state = state.model_copy(update={"player": player, "combat": state.combat.model_copy(
            update={"player": player},
        )})
        state = _play(state, "corrupted")
        assert state.player.hp == expected_hp
        assert state.player.block == 10


def test_source_inky_applies_one_weak_after_damage_and_artifact_prevents_it():
    state = _state(enchanted("inky", amount=9))
    monster = state.combat.monsters[0].model_copy(update={"statuses": {"artifact": 1}})
    state = state.model_copy(update={"combat": state.combat.model_copy(
        update={"monsters": (monster,)},
    )})
    state = _play(state, "inky")
    assert state.combat.monsters[0].hp == 95
    assert not state.combat.monsters[0].statuses.get("artifact", 0)
    assert not state.combat.monsters[0].statuses.get("weak", 0)


@pytest.mark.parametrize("name", ["swift", "sown"])
def test_source_once_per_combat_onplay_is_consumed_before_replays_and_survives_reload(name):
    card = enchanted(name, amount=2, custom={"replay": 1}, cost=1)
    draw = tuple(_simple(f"draw{i}") for i in range(6))
    state = _play(_state(card, draw=draw), name)
    assert state.player.energy == (21 if name == "sown" else 19)
    assert len(state.combat.hand) == (2 if name == "swift" else 0)
    # The enchantment does not reduce the card's native energy cost.
    assert state.combat.discard_pile[0].cost == 1
    assert _reload(state).combat.metadata["source_enchantment_disabled"][name] == 1


def test_source_sown_consumes_charge_even_when_energy_gain_is_forbidden():
    state = _play(_state(enchanted("sown"), statuses={"no_energy_gain_this_turn": 1}), "sown")
    assert state.player.energy == 20
    assert state.combat.metadata["source_enchantment_disabled"]["sown"] == 1


def test_source_slither_randomizes_only_drawn_card_on_energy_cost_stream():
    card = enchanted("slither", cost=2)
    draw = _simple("draw", effects={"draw": 1})
    state = _state(draw, draw=(card,))
    expected = random_stream(random_from_state(state.rng), "combat_energy_costs").randrange(4)
    state = _play(state, "draw")
    drawn = state.combat.hand[0]
    assert drawn.cost == 2  # Base stays intact; local combat override has its own scope.
    assert drawn.custom["source_enchantment_slither_cost"] == expected
    state = _play(_reload(state), "slither")
    assert state.player.energy == 20 - expected


def test_source_perfect_fit_moves_card_to_top_only_when_discard_shuffles():
    perfect = enchanted("perfect_fit")
    draw = _simple("draw", effects={"draw": 1})
    state = _state(draw)
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "discard_pile": (_simple("ordinary1"), perfect, _simple("ordinary2")),
    })})
    state = _play(state, "draw")
    assert state.combat.hand[0].instance_id == "perfect_fit"


def test_source_slumbering_essence_accumulates_in_retained_hand_until_played():
    card = enchanted("slumbering_essence", cost=3, custom={"retain": True})
    state = _state(card)
    state = _reload(_end_turn(_end_turn(state)))
    assert state.combat.hand[0].custom["source_enchantment_slumber_discount"] == 2
    before = state.player.energy
    state = _play(state, "slumbering_essence")
    assert state.player.energy == before - 1
    assert "source_enchantment_slumber_discount" not in state.combat.discard_pile[0].custom


def test_source_spiral_adds_one_replay_independent_of_amount():
    state = _play(_state(enchanted("spiral", amount=9)), "spiral")
    assert state.combat.monsters[0].hp == 92


def test_source_souls_power_removes_local_exhaust_but_global_corruption_still_exhausts():
    card = enchanted("souls_power", exhausts=True)
    state = _play(_state(card), "souls_power")
    assert not state.combat.exhaust_pile
    assert state.combat.discard_pile[0].instance_id == "souls_power"
    card = enchanted("souls_power", block=5, exhausts=True)
    state = _play(_state(card, statuses={"skill_exhaust_on_play": 1}), "souls_power")
    assert state.combat.exhaust_pile[0].instance_id == "souls_power"


@pytest.mark.parametrize("name", ["steady", "royally_approved"])
def test_source_retain_properties_survive_end_turn(name):
    state = _end_turn(_state(enchanted(name)))
    assert state.combat.hand[0].instance_id == name


def test_source_imbued_autoplays_before_opening_hand_draw_and_uses_no_energy():
    state = _enter_combat((
        {"card_id": "imbued", "type": "skill", "target": "self", "cost": 2,
         "effects": {"block": 4}, "enchantments": ({"keyword": "imbued", "amount": 1},)},
        {"card_id": "normal", "type": "skill", "target": "self", "cost": 0},
    ), flags={"draw_per_turn": 1})
    assert state.player.block == 4
    assert state.player.energy == 3
    assert [c.card_id for c in state.combat.hand] == ["normal"]
    assert state.combat.discard_pile[0].card_id == "imbued"
    events = state.combat.last_events
    auto = next(i for i, e in enumerate(events) if e.kind == "imbued_card_triggered")
    draw = next(i for i, e in enumerate(events) if e.kind == "cards_drawn")
    assert auto < draw


def test_source_blade_of_ink_creates_actual_enchanted_shivs():
    state = _play(_state(_source("BLADE_OF_INK")), "blade_of_ink")
    shiv = state.combat.hand[0]
    assert shiv.enchantments[0].keyword == "inky"
    state = _play(state, shiv.instance_id)
    assert state.combat.monsters[0].hp == 95
    assert state.combat.monsters[0].statuses["weak"] == 1


@pytest.mark.parametrize("name", ["clone", "deprecated_enchantment"])
def test_source_marker_only_enchantments_have_no_combat_effect(name):
    state = _play(_state(enchanted(name, amount=9)), name)
    assert state.combat.monsters[0].hp == 96
    assert state.player.energy == 20
    assert state.player.hp == 80
    assert not state.combat.hand


@pytest.mark.parametrize("name,source_id", [
    ("tezcataras_ember", "STRIKE_IRONCLAD"), ("souls_power", "SHIV"),
    ("nimble", "DEFEND_IRONCLAD"),
])
def test_source_on_enchant_properties_reapply_after_public_armaments_upgrade(name, source_id):
    card = _enchant_card(_source(source_id), name, 3, source_id="fixture")
    state = _play(_state(_source("ARMAMENTS", upgraded=True), card), "armaments")
    upgraded = state.combat.hand[0]
    assert upgraded.upgraded
    assert len(upgraded.enchantments) == 1
    if name == "tezcataras_ember":
        assert upgraded.cost == 0
        assert upgraded.custom["eternal"]
    elif name == "souls_power":
        assert not upgraded.exhausts
    state = _play(_reload(state), upgraded.instance_id)
    if name == "nimble":
        assert state.player.block == 16  # Armaments5 + upgraded Defend8 + Nimble3
    elif name == "souls_power":
        assert state.combat.discard_pile[-1].card_id == "shiv"


def test_source_royal_innate_property_affects_public_opening_draw():
    state = _enter_combat((
        {"card_id": "royal", "type": "attack", "target": "enemy", "cost": 1,
         "enchantments": ({"keyword": "royally_approved", "amount": 1},)},
        {"card_id": "ordinary", "type": "skill", "target": "self", "cost": 0},
    ), flags={"draw_per_turn": 1})
    assert [card.card_id for card in state.combat.hand] == ["royal"]
