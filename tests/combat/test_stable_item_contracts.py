"""Independent v0.107.1 item contracts exercised through public actions.

Source revision, file hashes and compact source projections are retained in
fixtures/stable_item_contracts.json; the game decompilation is not bundled.
"""

import pytest
from combat.test_card_play_status_runtime import _reload, _simple, _state
from combat.test_glam import _end_turn, _play
from combat.test_relic_engine_integration import _card_spec, _enter_combat

from sts2sim import legal_actions, serialize, step
from sts2sim.agent_api import action_space
from sts2sim.engine import RewardState, RunPhase
from sts2sim.engine.rng import RunRng, capture_random_state
from sts2sim.mechanics.relic_lifecycle import COMBAT_ONLY_RELIC_COUNTERS


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
    assert chosen.cost == 0
    assert chosen.custom["source_local_energy_base"] == 2
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


@pytest.mark.parametrize("damage,remaining", [(0, 10), (15, 5), (25, 0)])
def test_sturdy_clamp_preserves_only_block_remaining_after_enemy_attacks(damage, remaining):
    state = _enemies(_state(_simple("unused"), relics=("sturdy_clamp",)), damage=damage)
    player = state.player.model_copy(update={"block": 20})
    state = state.model_copy(update={"player": player, "combat": state.combat.model_copy(
        update={"player": player, "monsters": state.combat.monsters[:1]},
    )})
    state = _end_turn(_reload(state))
    assert state.player.block == remaining
    assert "next_turn_block" not in state.player.statuses


def test_red_skull_tracks_hp_loss_healing_and_reactivation_between_effect_steps():
    attack = _simple("cross", card_type="attack", target="enemy", effects={
        "sequence": [{"hp_loss": 5}, {"damage": 6}],
    })
    heal = _simple("heal", effects={"heal": 20})
    state = _state(attack, heal, relics=("red_skull",))
    player = state.player.model_copy(update={"hp": 42})
    state = state.model_copy(update={"player": player, "combat": state.combat.model_copy(
        update={"player": player},
    )})
    state = _play(_reload(state), "cross")
    assert state.player.hp == 37
    assert state.player.statuses["strength"] == 3
    assert state.combat.monsters[0].hp == 91
    state = _play(_reload(state), "heal")
    assert state.player.hp == 57
    assert state.player.statuses.get("strength", 0) == 0
    assert state.combat.metadata["relic_counters"]["red_skull"] == 0


def test_red_skull_removal_can_be_blocked_by_artifact_on_healing():
    state = _state(_simple("unused"), relics=("red_skull",),
                   statuses={"strength": 3, "artifact": 1})
    player = state.player.model_copy(update={"hp": 40})
    state = state.model_copy(update={"player": player, "combat": state.combat.model_copy(
        update={"player": player, "metadata": {"relic_counters": {"red_skull": 1}}},
    )})
    state = _potion(state, "blood_potion")
    assert state.player.hp == 56
    assert state.player.statuses["strength"] == 3
    assert state.player.statuses.get("artifact", 0) == 0
    assert state.combat.metadata["relic_counters"]["red_skull"] == 0


def test_fiddle_allows_opening_and_turn_hand_draw_but_blocks_potion_draw():
    state = _enter_combat(tuple(_card_spec(f"card:{i}") for i in range(12)),
                          relics=("fiddle",), draw_per_turn=5)
    assert len(state.combat.hand) == 7
    state = _potion(state, "swift_potion")
    assert len(state.combat.hand) == 7
    assert any(e.source_id == "fiddle" and e.kind == "draw_blocked"
               for e in state.combat.last_events)
    state = _end_turn(_reload(state))
    assert len(state.combat.hand) == 7
    assert not state.player.statuses.get("no_draw_this_turn")


def test_fiddle_allows_enemy_side_draws_from_other_relics():
    cards = tuple(_simple(f"draw:{i}") for i in range(10))
    state = _enemies(_state(_simple("unused"), relics=("fiddle", "centennial_puzzle"),
                            draw=cards, draw_per_turn=1), damage=1)
    state = _end_turn(_reload(state))
    assert len(state.combat.hand) == 6  # 3 from Puzzle, then 1+2 from hand draw.


def test_new_combat_resets_source_ephemeral_relic_counters_and_preserves_saved_counts():
    saved = {"nunchaku": 9, "pen_nib": 9, "joss_paper": 4, "galactic_dust": 9,
             "happy_flower": 1, "lizard_tail": 1, "winged_boots": 2}
    stale = {**dict.fromkeys(COMBAT_ONLY_RELIC_COUNTERS, 9), **saved}
    state = _enter_combat(tuple(_card_spec(f"card:{i}") for i in range(10)),
                          relics=("centennial_puzzle", "ruined_helmet"), draw_per_turn=0,
                          flags={"relic_counters": stale})
    counters = state.combat.metadata["relic_counters"]
    assert not COMBAT_ONLY_RELIC_COUNTERS.intersection(counters)
    assert counters == saved
    loss = _simple("hp_loss", effects={"hp_loss": 3})
    strength = _simple("strength", effects={"apply_status": {"target": "self", "strength": 2}})
    state = state.model_copy(update={"combat": state.combat.model_copy(
        update={"hand": (loss, strength)},
    )})
    state = _play(_reload(state), "hp_loss")
    assert len(state.combat.hand) == 4  # Remaining Strength card plus 3 Puzzle draws.
    state = _play(_reload(state), "strength")
    assert state.player.statuses["strength"] == 4


def test_demon_tongue_restores_first_unpowered_card_hp_loss_once_each_player_turn():
    loss = _simple("loss", effects={"hp_loss": 4})
    second = loss.model_copy(update={"instance_id": "second"})
    state = _state(loss, second, relics=("demon_tongue",))
    state = _play(_reload(state), "loss")
    assert state.player.hp == 80
    state = _play(_reload(state), "second")
    assert state.player.hp == 76
    state = _end_turn(_reload(state))
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"hand": (loss,)})})
    state = _play(_reload(state), "loss")
    assert state.player.hp == 76


def test_mini_regent_activates_once_per_turn_and_galactic_dust_accumulates_small_spends():
    cards = tuple(_simple(f"spend:{i}", custom={"star_cost": cost, "retain": True})
                  for i, cost in enumerate((3, 4, 7, 7)))
    state = _state(*cards, relics=("mini_regent", "galactic_dust"))
    state = _play(_reload(state), "spend:0")
    assert state.player.statuses["strength"] == 1
    assert state.player.block == 0
    assert state.combat.metadata["relic_counters"]["galactic_dust"] == 3
    state = _play(_reload(state), "spend:1")
    assert state.player.statuses["strength"] == 1
    assert state.combat.metadata["relic_counters"]["galactic_dust"] == 7
    state = _play(_reload(state), "spend:2")
    assert state.player.block == 10
    assert state.combat.metadata["relic_counters"]["galactic_dust"] == 4
    state = _end_turn(_reload(state))
    player = state.player.model_copy(update={"resources": {"star": 7}})
    state = state.model_copy(update={"player": player, "combat": state.combat.model_copy(
        update={"player": player},
    )})
    state = _play(_reload(state), "spend:3")
    assert state.player.statuses["strength"] == 2
    assert state.player.block == 10
    assert state.combat.metadata["relic_counters"]["galactic_dust"] == 1


def test_dusty_tome_adds_the_preselected_ancient_card_upgraded_without_reroll():
    state = _state(_simple("unused"))
    state = state.model_copy(update={
        "combat": None, "phase": RunPhase.REWARD,
        "reward": RewardState(reward_id="tome", relic_id="dusty_tome"),
        "rng": capture_random_state(RunRng("TOME")),
        "flags": {**state.flags, "dusty_tome_card_id": "corruption"},
    })
    before = state.rng
    action = next(a for a in legal_actions(state) if a.type == "take_reward_relic")
    state = step(_reload(state), action)
    assert state.master_deck[-1].card_id == "corruption"
    assert state.master_deck[-1].upgraded
    assert state.rng == before


def test_source_alchemize_excludes_combat_restricted_potions_and_uses_own_rng_stream():
    generate = _simple("alchemize", effects={"add_random_potion": 3})
    state = _state(generate)
    state = state.model_copy(update={"rng": capture_random_state(RunRng("ALCHEMIZE")),
                                     "character_id": "IRONCLAD"})
    state = _play(_reload(state), "alchemize")
    assert len(state.potions) == 3
    assert not {"fairy_in_a_bottle", "fruit_juice", "regen_potion"}.intersection(state.potions)
    assert state.rng.draws == 0
    assert state.rng.streams["combat_potion_generation"].draws == 6



def _fairy_attack_state(*, max_hp=80, potions=("fairy_in_a_bottle",), relics=()):
    state = _enemies(_state(_simple("unused"), relics=relics), damage=10)
    player = state.player.model_copy(update={"hp": 1, "max_hp": max_hp})
    monster = state.combat.monsters[0].model_copy(update={"hit_count": 3,
        "intent_damage": 30})
    return state.model_copy(update={"player": player, "potions": potions,
        "flags": {**state.flags, "monsters": ({"id": "TEST_ATTACKER", "min_hp": 100,
            "max_hp": 100, "moves": ({"id": "ATTACK", "intent": "Attack",
                "damage": {"normal": 10, "hit_count": 3}},)},)},
        "combat": state.combat.model_copy(update={"player": player,
            "monsters": (monster,)}),
    })


def test_fairy_revives_between_enemy_hits_and_dispatches_potion_used_with_reload():
    state = _fairy_attack_state(relics=("reptile_trinket", "belt_buckle"))
    assert not any(a.type == "use_potion" for a in legal_actions(state))
    action = next(a for a in legal_actions(state) if a.type == "end_turn")
    assert serialize(step(state, action)) == serialize(step(_reload(state), action))
    state = step(_reload(state), action)
    assert state.player.hp == 4  # 24 restored after hit one, then two more hits.
    assert not state.potions
    assert state.player.statuses["strength"] == 3
    assert state.player.statuses["strength_down"] == 3
    assert state.player.statuses["dexterity"] == 2
    events = state.replay_log[-1].events
    fairy = next(i for i, e in enumerate(events) if e.kind == "potion_passive_triggered")
    assert sum(e.kind == "player_damaged" for e in events[:fairy]) == 1
    assert sum(e.kind == "player_damaged" for e in events[fairy:]) == 2
    assert any(e.kind == "potion_used" and e.source_id == "fairy_in_a_bottle"
               for e in events)


def test_multiple_fairies_are_consumed_per_hit_and_later_hit_can_kill():
    state = _fairy_attack_state(max_hp=20,
                               potions=("fairy_in_a_bottle", "fairy_in_a_bottle"))
    state = _end_turn(_reload(state))
    assert state.phase == RunPhase.FAILED
    assert not state.potions
    assert state.player.hp == 0
    assert sum(e.kind == "potion_passive_triggered"
               for e in state.replay_log[-1].events) == 2


def test_fairy_resolves_between_card_hp_loss_steps_and_skips_lethal_received_power():
    card = _simple("pay_twice", effects={"sequence": [{"hp_loss": 2}, {"hp_loss": 2}]})
    state = _state(card, statuses={"rupture": 3})
    player = state.player.model_copy(update={"hp": 1, "max_hp": 5})
    state = state.model_copy(update={"player": player, "potions": ("fairy_in_a_bottle",),
        "combat": state.combat.model_copy(update={"player": player})})
    state = _play(_reload(state), "pay_twice")
    assert state.phase == RunPhase.FAILED
    assert not state.potions
    assert state.player.statuses.get("strength", 0) == 0


def test_belt_buckle_last_potion_discard_and_use_track_inventory_once():
    state = _state(_simple("unused"), relics=("belt_buckle",))
    state = state.model_copy(update={"potions": ("block_potion",),
        "flags": {**state.flags, "allow_free_potion_discard": True}})
    action = next(a for a in legal_actions(state) if a.type == "discard_potion")
    state = step(_reload(state), action)
    assert state.player.statuses["dexterity"] == 2
    # Procuring a potion removes the actual power; Artifact may block the removal.
    state = _potion(state, "entropic_brew")
    assert state.potions
    assert state.player.statuses.get("dexterity", 0) == 0
    assert state.combat.metadata["relic_counters"]["belt_buckle"] == 0


def test_fairy_lethal_hit_skips_damage_received_relic_then_next_hit_draws_once():
    state = _fairy_attack_state(relics=("centennial_puzzle",))
    draw = tuple(_simple(f"puzzle:{i}") for i in range(6))
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "draw_pile": draw,
    })})
    state = _end_turn(_reload(state))
    assert [c.instance_id for c in state.combat.hand] == [f"puzzle:{i}" for i in range(3)]
    assert state.combat.metadata["relic_counters"]["centennial_puzzle"] == 1
    events = state.replay_log[-1].events
    hits = [i for i, e in enumerate(events) if e.kind == "player_damaged"]
    draw_index = next(i for i, e in enumerate(events) if e.kind == "cards_drawn")
    assert hits[1] < draw_index < hits[2]


def test_lethal_potion_damage_consumes_fairy_and_runs_both_potion_callbacks():
    state = _state(_simple("unused"), relics=("reptile_trinket", "belt_buckle", "demon_tongue"))
    player = state.player.model_copy(update={"hp": 1})
    state = state.model_copy(update={"player": player, "potions": ("fairy_in_a_bottle",),
        "combat": state.combat.model_copy(update={"player": player})})
    state = _potion(state, "foul_potion")
    assert state.player.hp == 24
    assert not state.potions
    assert state.player.statuses["strength"] == 6
    assert state.player.statuses["strength_down"] == 6
    assert state.player.statuses["dexterity"] == 2
    assert not state.combat.metadata["relic_counters"].get("demon_tongue")


def test_clarity_stacks_duration_and_modifies_fiddle_normal_hand_draw():
    draw = tuple(_simple(f"draw:{i}") for i in range(20))
    state = _state(_simple("unused"), draw=draw, relics=("fiddle",))
    state = _potion(_potion(state, "clarity"), "clarity")
    assert state.player.statuses["clarity"] == 6
    assert len(state.combat.hand) == 1  # Direct draws blocked; duration still gained.
    for remaining in range(5, -1, -1):
        state = _end_turn(_reload(state))
        assert len(state.combat.hand) == 3  # Fiddle adds two normal hand draws.
        assert state.player.statuses.get("clarity", 0) == remaining
    state = _end_turn(_reload(state))
    assert len(state.combat.hand) == 2


def test_radiance_stacks_six_turns_of_one_energy_instead_of_three_turns_of_two():
    state = _potion(_potion(_state(_simple("unused")), "radiant_tincture"), "radiant_tincture")
    assert state.player.statuses["radiance"] == 6
    for remaining in range(5, -1, -1):
        state = _end_turn(_reload(state))
        assert state.player.energy == 21
        assert state.player.statuses.get("radiance", 0) == remaining
    state = _end_turn(_reload(state))
    assert state.player.energy == 20


def test_soldiers_stew_uses_source_strike_tag_and_adds_base_replay_without_enchanting():
    tagged = _simple("not_named_strike", effects={"damage": 2}, card_type="attack",
                     target="enemy", tags=("strike",))
    false_match = _simple("strike_in_description", name="Counterstrike", effects={"damage": 2},
                         card_type="attack", target="enemy")
    registered = _simple("localized", card_type="attack", target="enemy", effects={"damage": 2})
    registered = registered.model_copy(update={"card_id": "SOLAR_STRIKE"})
    state = _state(tagged, false_match, registered)
    state = _potion(_potion(state, "soldiers_stew"), "soldiers_stew")
    assert [c.custom.get("base_replay_bonus", 0) for c in state.combat.hand] == [2, 0, 2]
    assert not any(c.enchantments for c in state.combat.hand)
    state = _play(_reload(state), "not_named_strike")
    assert state.combat.monsters[0].hp == 94


def test_snecko_oil_uses_energy_cost_rng_and_restores_cost_after_physical_play():
    card = _simple("reusable", effects={"draw": 0}, cost=2)
    state = _state(card)
    state = state.model_copy(update={"rng": capture_random_state(RunRng(
        "SNECKO", derivation="sts2.v0.107.1.utf16-djb2-splitmix64"))})
    state = _potion(state, "snecko_oil")
    assert {name for name, stream in state.rng.streams.items() if stream.draws} == {
        "combat_energy_costs",
    }
    assert state.rng.streams["combat_energy_costs"].draws == 1
    state = _play(_reload(state), "reusable")
    card = next(c for c in state.combat.discard_pile if c.instance_id == "reusable")
    assert card.cost == 2
    assert "cost_before_randomize" not in card.custom


def test_regen_potion_heals_and_decrements_at_owner_side_end_before_enemy_attack():
    state = _enemies(_state(_simple("unused")), damage=5)
    player = state.player.model_copy(update={"hp": 1})
    state = state.model_copy(update={"player": player,
        "combat": state.combat.model_copy(update={"player": player,
            "monsters": state.combat.monsters[:1]})})
    state = _potion(state, "regen_potion")
    state = _end_turn(_reload(state))
    assert state.phase == RunPhase.COMBAT
    assert state.player.hp == 1
    assert state.player.statuses["regen"] == 4
    events = state.replay_log[-1].events
    assert next(i for i, e in enumerate(events) if e.source_id == "regen") < next(
        i for i, e in enumerate(events) if e.kind == "player_damaged")


def test_heart_of_iron_plating_decays_once_at_owner_turn_start_and_not_per_hit():
    state = _fairy_attack_state(potions=())
    player = state.player.model_copy(update={"hp": 80})
    state = state.model_copy(update={"player": player,
        "combat": state.combat.model_copy(update={"player": player})})
    state = _potion(state, "heart_of_iron")
    state = _end_turn(_reload(state))
    assert state.player.hp == 57  # Seven Block, then three ten-damage hits.
    assert state.player.statuses["plated_armor"] == 6


@pytest.mark.parametrize("free_flag", ["free_to_play_this_turn", "free_to_play_this_combat"])
def test_snecko_latest_energy_override_takes_precedence_over_prior_free_override(free_flag):
    card = _simple("reusable", cost=2, custom={free_flag: True, "star_cost": 2})
    state = _state(card)
    state = state.model_copy(update={"rng": capture_random_state(RunRng(
        "SNECKO", derivation="sts2.v0.107.1.utf16-djb2-splitmix64"))})
    state = _potion(state, "snecko_oil")
    rolled_cost = state.combat.hand[0].cost
    assert rolled_cost > 0
    state = _play(_reload(state), "reusable")
    assert state.player.energy == 20 - rolled_cost
    assert state.player.resources["star"] == 20  # Energy override leaves free Stars intact.


def test_blood_potion_small_max_hp_has_no_minimum_heal_in_or_out_of_combat():
    state = _state(_simple("unused"))
    player = state.player.model_copy(update={"hp": 1, "max_hp": 4})
    state = state.model_copy(update={"player": player,
        "combat": state.combat.model_copy(update={"player": player})})
    assert _potion(state, "blood_potion").player.hp == 1
    state = state.model_copy(update={"phase": RunPhase.MAP, "combat": None})
    assert _potion(state, "blood_potion").player.hp == 1


def test_sozu_entropic_rolls_once_then_stops_at_forbidden_procurement():
    state = _state(_simple("unused"), relics=("sozu",))
    state = state.model_copy(update={"character_id": "IRONCLAD",
        "rng": capture_random_state(RunRng(
            "SOZU", derivation="sts2.v0.107.1.utf16-djb2-splitmix64"))})
    state = _potion(state, "entropic_brew")
    assert not state.potions
    assert state.rng.streams["combat_potion_generation"].draws == 2
    assert any(e.kind == "potion_procurement_blocked" for e in state.replay_log[-1].events)


def test_sozu_claims_forbidden_potion_reward_even_with_full_inventory():
    state = _state(_simple("unused"), relics=("sozu",))
    state = state.model_copy(update={"phase": RunPhase.REWARD, "combat": None,
        "potions": ("fire_potion",) * 3,
        "reward": RewardState(reward_id="potion", source="combat", potion_id="blood_potion")})
    action = next(a for a in legal_actions(state) if a.type == "take_reward_potion")
    state = step(_reload(state), action)
    assert state.potions == ("fire_potion",) * 3
    assert state.reward.potion_claimed
    event = next(e for e in state.replay_log[-1].events if e.kind == "reward_potion_taken")
    assert not event.metadata["procured"]


def test_sozu_blocks_alchemize_procurement_but_preserves_source_factory_draws():
    card = _simple("alchemize_test", effects={"add_random_potion": 1})
    state = _state(card, relics=("sozu",))
    state = state.model_copy(update={"character_id": "IRONCLAD",
        "rng": capture_random_state(RunRng(
            "SOZU", derivation="sts2.v0.107.1.utf16-djb2-splitmix64"))})
    state = _play(_reload(state), "alchemize_test")
    assert not state.potions
    assert state.rng.streams["combat_potion_generation"].draws == 2


def test_confused_randomizes_drawn_costs_for_combat_and_consumes_roll_for_x_card():
    normal = _simple("normal", cost=2)
    x_card = _simple("x_card", cost=-1, custom={"x_cost": True})
    unplayable = _simple("unplayable", cost=None, custom={"unplayable": True})
    state = _state(_simple("unused"), statuses={"confused": 1}, draw=(normal, x_card, unplayable))
    state = state.model_copy(update={"rng": capture_random_state(RunRng(
        "CONFUSED", derivation="sts2.v0.107.1.utf16-djb2-splitmix64"))})
    state = _potion(state, "swift_potion")
    assert state.rng.streams["combat_energy_costs"].draws == 2
    normal = next(c for c in state.combat.hand if c.instance_id == "normal")
    assert normal.custom["source_local_energy_modifiers"][-1]["expiration"] == 4
    assert next(c for c in state.combat.hand if c.instance_id == "x_card").cost == -1
    state = _play(_reload(state), "normal")
    played = next(c for c in state.combat.discard_pile if c.instance_id == "normal")
    assert played.cost == normal.cost


def test_spiked_gauntlets_global_power_cost_applies_after_local_free_cost_and_mask():
    power = _simple("power", cost=3, card_type="power", custom={"free_to_play_this_turn": True})
    state = _state(power, relics=("spiked_gauntlets",))
    player = state.player.model_copy(update={"energy": 0})
    state = state.model_copy(update={"player": player, "combat": state.combat.model_copy(update={
        "player": player, "metadata": {"item_relic_ids": state.relics},
    })})
    assert not any(a.type == "play_card" for a in legal_actions(state))
    player = player.model_copy(update={"energy": 1})
    state = state.model_copy(update={"player": player,
        "combat": state.combat.model_copy(update={"player": player})})
    state = _play(_reload(state), "power")
    assert state.player.energy == 0


def test_paels_flesh_updates_maximum_energy_from_turn_three_and_restores_run_base():
    kill = _simple("kill", effects={"damage": 200}, card_type="attack", target="enemy",
                   custom={"retain": True})
    state = _state(kill, relics=("paels_flesh",))
    state = _end_turn(_reload(state))
    assert (state.player.energy, state.player.max_energy) == (20, 20)
    state = _end_turn(_reload(state))
    assert (state.player.energy, state.player.max_energy) == (21, 21)
    state = _end_turn(_reload(state))
    assert (state.player.energy, state.player.max_energy) == (21, 21)
    state = _play(_reload(state), "kill")
    assert state.phase == RunPhase.REWARD
    assert state.player.max_energy == 20


@pytest.mark.parametrize("relic_id", [
    "blood_soaked_rose", "blessed_antler", "ectoplasm", "sozu", "philosophers_stone",
    "prismatic_gem", "spiked_gauntlets", "velvet_choker", "whispering_earring", "pumpkin_candle",
])
def test_source_energy_relics_modify_maximum_once_at_reset_and_restore_run_base(relic_id):
    state = _enter_combat((), relics=(relic_id,), draw_per_turn=0)
    assert (state.player.energy, state.player.max_energy) == (4, 4)
    state = _end_turn(_reload(state))
    assert (state.player.energy, state.player.max_energy) == (4, 4)


def test_bread_first_turn_loss_and_later_maximum_energy_are_distinct():
    state = _enter_combat((), relics=("bread",), draw_per_turn=0)
    assert (state.player.energy, state.player.max_energy) == (1, 3)
    state = _end_turn(_reload(state))
    assert (state.player.energy, state.player.max_energy) == (4, 4)
    state = _end_turn(_reload(state))
    assert (state.player.energy, state.player.max_energy) == (4, 4)


def test_source_max_energy_stacks_relics_without_repeated_turn_gain_or_zero_candle():
    state = _enter_combat((), relics=("ectoplasm", "prismatic_gem", "pumpkin_candle"),
                          draw_per_turn=0)
    assert (state.player.energy, state.player.max_energy) == (6, 6)
    state = _end_turn(_reload(state))
    assert (state.player.energy, state.player.max_energy) == (6, 6)
    metadata = {**state.combat.metadata, "relic_counters": {
        **state.combat.metadata["relic_counters"], "pumpkin_candle": 0,
    }}
    state = state.model_copy(update={"combat": state.combat.model_copy(
        update={"metadata": metadata},
    )})
    state = _end_turn(_reload(state))
    assert (state.player.energy, state.player.max_energy) == (5, 5)
