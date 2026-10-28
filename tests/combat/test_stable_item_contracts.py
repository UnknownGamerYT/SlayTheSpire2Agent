"""Independent v0.107.1 item contracts exercised through public actions.

Source revision, file hashes and compact source projections are retained in
fixtures/stable_item_contracts.json; the game decompilation is not bundled.
"""

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _state
from combat.test_glam import _end_turn, _play

from sts2sim import legal_actions, serialize, step
from sts2sim.agent_api import action_space
from sts2sim.engine import RunPhase


def _potion(state, potion_id, target_id=None):
    state = state.model_copy(update={"potions": state.potions + (potion_id,)})
    action = next(a for a in legal_actions(state) if a.type == "use_potion"
                  and a.payload["potion_id"] == potion_id
                  and (target_id is None or a.target_id == target_id))
    assert any(a["type"] == "use_potion" and a["target_id"] == action.target_id
               for a in action_space(state))
    assert serialize(step(state, action)) == serialize(step(_reload(state), action))
    return step(state, action)


def _enemies(state, *, hp=100, damage=0, statuses=None):
    monster = state.combat.monsters[0]
    monsters = tuple(monster.model_copy(update={
        "monster_id": name, "hp": hp, "max_hp": hp, "intent": "attack",
        "intent_damage": damage, "statuses": dict(statuses or {}),
        "move_id": "ATTACK",
        "metadata": {"damage_per_hit": damage, "source_monster_id": "TEST_ATTACKER"},
    }) for name in ("target", "other"))
    return state.model_copy(update={
        "flags": {**state.flags, "monsters": ({"id": "TEST_ATTACKER", "min_hp": hp,
            "max_hp": hp, "moves": ({"id": "ATTACK", "intent": "Attack",
                                      "damage": {"normal": damage}},)},)},
        "combat": state.combat.model_copy(update={"monsters": monsters}),
    })


def test_glowwater_exhausts_original_hand_then_draws_with_exhaust_hooks_and_reload():
    draw = tuple(_simple(f"draw:{i}") for i in range(12))
    state = _state(_simple("held:a"), _simple("held:b"), draw=draw,
                   relics=("charons_ashes",))
    state = _potion(state, "glowwater_potion")
    assert [c.instance_id for c in state.combat.hand] == [f"draw:{i}" for i in range(10)]
    assert [c.instance_id for c in state.combat.exhaust_pile] == ["held:a", "held:b"]
    assert [c.instance_id for c in state.combat.draw_pile] == ["draw:10", "draw:11"]
    assert state.combat.monsters[0].hp == 94  # Both exhaust events reach the relic.


def test_liquid_memories_discount_expires_on_retained_card_after_turn():
    recovered = _simple("recovered", cost=3, custom={"retain": True})
    state = _state(_simple("unused"))
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "hand": (), "discard_pile": (recovered,),
    })})
    state = _potion(state, "liquid_memories")
    action = next(a for a in legal_actions(state) if a.type == "choose_card")
    state = step(_reload(state), action)
    assert state.combat.hand[0].custom["free_to_play_this_turn"]
    assert not state.combat.hand[0].custom.get("free_to_play_this_combat")
    state = _end_turn(_reload(state))
    assert state.combat.hand[0].cost == 3
    assert not state.combat.hand[0].custom.get("free_to_play_this_turn")


def test_gigantification_survives_turn_end_skips_skills_and_counts_next_attacks():
    attack = _simple("attack", effects={"damage": 4}, card_type="attack", target="enemy",
                     custom={"retain": True})
    skill = _simple("skill", effects={"damage": 2}, target="enemy", custom={"retain": True})
    state = _state(attack, skill)
    state = _potion(_potion(state, "gigantification_potion"), "gigantification_potion")
    state = _end_turn(_reload(state))
    state = _play(state, "skill")
    state = _play(_reload(state), "attack")
    assert state.combat.monsters[0].hp == 86  # 2 skill damage + one triple attack.
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "hand": (attack.model_copy(update={"instance_id": "attack:2"}),),
    })})
    state = _play(_reload(state), "attack:2")
    assert state.combat.monsters[0].hp == 74
    assert not state.combat.metadata.get("next_card_damage_multiplier")


@pytest.mark.parametrize("artifact", [0, 1])
def test_beetle_juice_target_artifact_and_four_enemy_sides(artifact):
    state = _enemies(_state(_simple("unused")), damage=10)
    target = state.combat.monsters[0].model_copy(update={"statuses": {"artifact": artifact}})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (target, state.combat.monsters[1]),
    })})
    state = _potion(state, "beetle_juice", "target")
    assert state.combat.monsters[0].statuses.get("shrink", 0) == (0 if artifact else 4)
    assert not state.combat.monsters[1].statuses
    assert state.combat.monsters[0].statuses.get("artifact", 0) == 0
    for remaining in (3, 2, 1, 0):
        before = state.player.hp
        state = _end_turn(_reload(state))
        assert before - state.player.hp == (20 if artifact else 17)
        assert state.combat.monsters[0].statuses.get("shrink", 0) == (0 if artifact else remaining)
        # Keep the independent fixture alive through all five enemy sides.
        player = state.player.model_copy(update={"hp": 80})
        state = state.model_copy(update={"player": player, "combat": state.combat.model_copy(
            update={"player": player},
        )})
    before = state.player.hp
    state = _end_turn(_reload(state))
    assert before - state.player.hp == 20


def test_beetle_juice_stacks_duration_and_multiplies_weak_before_rounding():
    state = _enemies(_state(_simple("unused")), damage=14, statuses={"weak": 9})
    state = _potion(_potion(state, "beetle_juice", "target"), "beetle_juice", "target")
    assert state.combat.monsters[0].statuses["shrink"] == 8
    before = state.player.hp
    state = _end_turn(_reload(state))
    assert before - state.player.hp == 17  # floor(14*.7*.75)=7, other floor(14*.75)=10.
    assert state.combat.monsters[0].statuses["shrink"] == 7


@pytest.mark.parametrize("artifact", [0, 1])
def test_powdered_demise_targets_one_enemy_stacks_and_waits_for_enemy_side_end(artifact):
    state = _enemies(_state(_simple("unused")), damage=3)
    target = state.combat.monsters[0].model_copy(update={
        "statuses": {"artifact": artifact, "vulnerable": 2},
    })
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (target, state.combat.monsters[1]),
    })})
    state = _potion(_potion(state, "powdered_demise", "target"), "powdered_demise", "target")
    assert [m.hp for m in state.combat.monsters] == [100, 100]
    assert state.combat.monsters[0].statuses["demise"] == (9 if artifact else 18)
    for expected in (100 - (9 if artifact else 18), 100 - (18 if artifact else 36)):
        state = _end_turn(_reload(state))
        assert [m.hp for m in state.combat.monsters] == [expected, 100]
        events = state.combat.last_events
        last_attack = max(i for i, e in enumerate(events) if e.kind == "player_damaged")
        demise = next(i for i, e in enumerate(events) if e.source_id == "demise")
        assert demise > last_attack


@pytest.mark.parametrize("defense,expected", [("intangible", 1), ("slippery", 1),
                                             ("hard_to_kill", 2)])
def test_powdered_demise_is_unblockable_unpowered_damage_with_hp_caps(defense, expected):
    state = _enemies(_state(_simple("unused")), statuses={defense: 2})
    state = _potion(state, "powdered_demise", "target")
    state = _end_turn(_reload(state))
    assert state.combat.monsters[0].hp == 100 - expected
    assert state.combat.monsters[1].hp == 100


@pytest.mark.parametrize("potion_id,amount", [("block_potion", 12), ("ship_in_a_bottle", 10)])
def test_potion_block_is_unpowered_and_does_not_spend_card_block_bonuses(potion_id, amount):
    state = _state(_simple("block", effects={"block": 2}), statuses={
        "dexterity": 9, "frail": 2, "double_block_gain_this_turn": 1,
        "first_card_block_double": 1, "juggernaut": 3, "no_card_block_gain": 2,
    }, relics=("vambrace",))
    state = _potion(state, potion_id)
    assert state.player.block == amount
    assert not state.combat.metadata.get("first_card_block_double_used_turn")
    assert state.combat.monsters[0].hp == 97  # Any-Block power still triggers once.
    assert not any(e.kind == "combat_relic_triggered" and e.source_id == "vambrace"
                   for e in state.combat.last_events)
    state = _play(_reload(state), "block")
    assert state.player.block == amount  # Card Block lock still applies.
    if potion_id == "ship_in_a_bottle":
        state = _end_turn(_reload(state))
        assert state.player.block == 10


@pytest.mark.parametrize("potion_id,stat", [("flex_potion", "strength"),
                                         ("speed_potion", "dexterity")])
@pytest.mark.parametrize("artifact", [0, 1])
def test_temporary_potion_stat_expiration_is_artifact_blockable_at_player_side_end(
    potion_id, stat, artifact,
):
    state = _enemies(_state(_simple("unused"), statuses={"artifact": artifact}), damage=1)
    state = _potion(state, potion_id)
    assert state.player.statuses[stat] == 5
    assert state.player.statuses.get("artifact", 0) == artifact
    state = _end_turn(_reload(state))
    assert state.player.statuses.get(stat, 0) == (5 if artifact else 0)
    assert state.player.statuses.get("artifact", 0) == 0
    if artifact:
        events = state.combat.last_events
        decay = next(i for i, e in enumerate(events) if e.kind == "status_blocked_by_artifact")
        attack = next(i for i, e in enumerate(events) if e.kind == "player_damaged")
        assert decay < attack


def test_flex_expiry_preserves_actual_additional_strength_from_ruined_helmet():
    state = _potion(_state(_simple("unused"), relics=("ruined_helmet",)), "flex_potion")
    assert state.player.statuses["strength"] == 10
    state = _end_turn(_reload(state))
    assert state.player.statuses["strength"] == 5


@pytest.mark.parametrize("potion_id,pool,card_type,ids", [
    ("attack_potion", "ironclad", "attack", ("anger", "bludgeon", "pommel_strike")),
    ("skill_potion", "ironclad", "skill", ("shrug_it_off", "armaments", "bloodletting")),
    ("power_potion", "ironclad", "power", ("barricade", "demon_form", "inflame")),
    ("colorless_potion", "colorless", "skill", ("finesse", "impatience", "discovery")),
])
def test_card_choice_potions_are_optional_exclude_invalid_cards_and_restore_costs(
    potion_id, pool, card_type, ids,
):
    cards = tuple({"id": card_id, "color": pool, "rarity": "Common", "type": card_type,
                   "target": "self", "cost": 2, "effects": {"block": 1}}
                  for card_id in ids)
    cards += (
        {"id": "basic", "color": pool, "rarity": "Basic", "type": card_type},
        {"id": "alchemize", "color": pool, "rarity": "Rare", "type": card_type},
    )
    state = _state(_simple("unused"))
    state = state.model_copy(update={"character_id": "IRONCLAD", "flags": {
        **state.flags, "cards": cards,
    }})
    state = _potion(state, potion_id)
    choice = state.combat.pending_choices[0]
    assert choice.min_choices == 0
    assert {c["card_id"] for c in choice.metadata["generated_cards"]} == set(ids)
    skip = next(a for a in legal_actions(state) if a.type == "proceed")
    assert not step(_reload(state), skip).combat.pending_choices
    action = next(a for a in legal_actions(state) if a.type == "choose_card")
    state = step(_reload(state), action)
    chosen = next(c for c in state.combat.hand if c.instance_id == action.card_instance_id)
    assert chosen.cost == 2
    assert chosen.custom["free_to_play_this_turn"]
    state = _end_turn(_reload(state))
    chosen = next(c for c in state.combat.discard_pile if c.instance_id == chosen.instance_id)
    assert chosen.cost == 2
    assert not chosen.custom.get("free_to_play_this_turn")


def test_cosmic_concoction_adds_distinct_upgraded_colorless_cards():
    ids = ("finesse", "impatience", "discovery")
    cards = tuple({"id": card_id, "color": "colorless", "rarity": "Common", "type": "skill",
                   "target": "self", "cost": 2, "effects": {"block": 1}}
                  for card_id in ids)
    state = _state(_simple("unused"))
    state = state.model_copy(update={"flags": {**state.flags, "cards": cards}})
    state = _potion(state, "cosmic_concoction")
    generated = [c for c in state.combat.hand if c.custom.get("generated")]
    assert {c.card_id for c in generated} == set(ids)
    assert len(generated) == 3
    assert all(c.upgraded and not c.custom.get("free_to_play_this_turn") for c in generated)


@pytest.mark.parametrize("potion_id", ["touch_of_insanity", "liquid_memories"])
def test_free_to_play_potions_remove_star_cost_and_restore_at_printed_duration(potion_id):
    costly = _simple("costly", cost=2, custom={"star_cost": 3, "retain": True})
    free = _simple("already_free")
    state = _state(free, costly)
    player = state.player.model_copy(update={"energy": 0, "max_energy": 0,
                                            "resources": {"star": 0}})
    state = state.model_copy(update={"player": player, "combat": state.combat.model_copy(
        update={"player": player, "hand": (free,) if potion_id == "liquid_memories" else
                (free, costly), "discard_pile": (costly,) if potion_id == "liquid_memories"
                else ()},
    )})
    state = _potion(state, potion_id)
    assert state.combat.pending_choices[0].candidate_ids == ("costly",)
    action = next(a for a in legal_actions(state) if a.type == "choose_card")
    state = step(_reload(state), action)
    assert any(a.type == "play_card" and a.card_instance_id == "costly"
               for a in legal_actions(state))
    state = _end_turn(_reload(state))
    playable = any(a.type == "play_card" and a.card_instance_id == "costly"
                   for a in legal_actions(state))
    assert playable == (potion_id == "touch_of_insanity")


@pytest.mark.parametrize("phase", [RunPhase.MAP, RunPhase.REWARD, RunPhase.EVENT])
@pytest.mark.parametrize("potion_id", ["blood_potion", "fruit_juice", "entropic_brew"])
def test_anytime_potions_are_legal_outside_combat_and_preserve_current_flow(phase, potion_id):
    state = _state(_simple("unused"))
    player = state.player.model_copy(update={"hp": 40})
    state = state.model_copy(update={"player": player, "combat": None, "phase": phase})
    state = _potion(state, potion_id)
    assert state.phase == phase
    assert state.combat is None
    if potion_id == "blood_potion":
        assert (state.player.hp, state.player.max_hp) == (56, 80)
    elif potion_id == "fruit_juice":
        assert (state.player.hp, state.player.max_hp) == (45, 85)
    else:
        assert len(state.potions) == 3
    assert any(e.kind == "potion_used" for e in state.replay_log[-1].events)


def test_combat_only_potions_remain_masked_outside_combat():
    state = _state(_simple("unused"))
    state = state.model_copy(update={"combat": None, "phase": RunPhase.MAP,
                                     "potions": ("fire_potion", "fairy_in_a_bottle")})
    assert not any(a.type == "use_potion" for a in legal_actions(state))


@pytest.mark.parametrize("potion_id", ["block_potion", "fortifier"])
def test_potion_block_respects_source_block_cap(potion_id):
    state = _state(_simple("unused"))
    player = state.player.model_copy(update={"block": 999_999_998})
    state = state.model_copy(update={"player": player, "combat": state.combat.model_copy(
        update={"player": player},
    )})
    state = _potion(state, potion_id)
    assert state.player.block == 999_999_999

