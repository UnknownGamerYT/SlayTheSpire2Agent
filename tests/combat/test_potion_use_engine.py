from __future__ import annotations

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import (
    CardInstance,
    MapEdgeState,
    MapNodeState,
    MapState,
    MonsterState,
    RoomKind,
    RunPhase,
)


def _choose_first_ancient(state):
    action = next(action for action in legal_actions(state) if action.type == "choose_ancient")
    return step(state, action)


def _force_next_room(state, room_kind: RoomKind):
    start = MapNodeState(node_id="start", act=state.act, floor=0, lane=0, kind=RoomKind.START)
    target = MapNodeState(node_id="target", act=state.act, floor=1, lane=0, kind=room_kind)
    game_map = MapState(
        act=state.act,
        nodes=(start, target),
        edges=(MapEdgeState(from_id=start.node_id, to_id=target.node_id),),
        current_node_id=start.node_id,
        completed_node_ids=(start.node_id,),
    )
    return state.model_copy(update={"phase": RunPhase.MAP, "map": game_map, "floor": 0})


def _enter_monster_combat(state):
    state = _choose_first_ancient(state)
    state = _force_next_room(state, RoomKind.MONSTER)
    action = next(action for action in legal_actions(state) if action.type == "choose_node")
    return step(state, action)


def _use_potion_action(state, potion_slot: str = "potion:0", target_id: str | None = None):
    return next(
        action
        for action in legal_actions(state)
        if action.type == "use_potion"
        and action.payload.get("potion_slot") == potion_slot
        and (target_id is None or action.target_id == target_id)
    )


def _proceed_action(state):
    return next(action for action in legal_actions(state) if action.type == "proceed")


def _choose_card_action(state, card_instance_id: str | None = None):
    return next(
        action
        for action in legal_actions(state)
        if action.type == "choose_card"
        and (card_instance_id is None or action.card_instance_id == card_instance_id)
    )


def _discard_card_action(state, card_instance_id: str):
    return next(
        action
        for action in legal_actions(state)
        if action.type == "discard_card" and action.card_instance_id == card_instance_id
    )


def _play_card_action(state, card_instance_id: str | None = None):
    return next(
        action
        for action in legal_actions(state)
        if action.type == "play_card"
        and (card_instance_id is None or action.card_instance_id == card_instance_id)
    )


def _end_turn_action(state):
    return next(action for action in legal_actions(state) if action.type == "end_turn")


def _card(
    card_id: str,
    *,
    card_type: str = "skill",
    target: str = "self",
    cost: int = 0,
    damage: int = 0,
    block: int = 0,
) -> CardInstance:
    effects = {}
    if damage:
        effects["damage"] = damage
    if block:
        effects["block"] = block
    return CardInstance(
        instance_id=f"{card_id}:1",
        card_id=card_id,
        name=card_id.replace("_", " ").title(),
        type=card_type,
        target=target,
        cost=cost,
        effects=effects,
    )


def test_fire_potion_is_legal_and_damages_target_from_belt_slot() -> None:
    state = new_run(seed=1300, character_id="TEST", ascension=0)
    state = state.model_copy(update={"potions": ("fire_potion",)})
    state = _enter_monster_combat(state)

    assert state.combat is not None
    monster = state.combat.monsters[0]
    action = _use_potion_action(state, target_id=monster.monster_id)
    state = step(state, action)

    assert state.potions == ()
    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster.hp - 20
    assert [event.kind for event in state.replay_log[-1].events][:2] == [
        "potion_used",
        "monster_damaged",
    ]


def test_toy_ornithopter_heals_when_potion_is_used() -> None:
    state = new_run(seed=1306, character_id="TEST", ascension=0)
    state = state.model_copy(
        update={
            "potions": ("fire_potion",),
            "relics": ("toy_ornithopter",),
            "player": state.player.model_copy(update={"hp": 47, "max_hp": 50}),
        }
    )
    state = _enter_monster_combat(state)

    assert state.combat is not None
    monster_id = state.combat.monsters[0].monster_id
    state = step(state, _use_potion_action(state, target_id=monster_id))

    assert state.combat is not None
    assert state.combat.player.hp == 50
    assert any(
        event.kind == "trigger_potion_use_heal"
        and event.source_id == "toy_ornithopter"
        and event.amount == 3
        for event in state.replay_log[-1].events
    )


def test_block_energy_and_strength_potions_apply_player_effects() -> None:
    state = new_run(seed=1301, character_id="TEST", ascension=0)
    state = state.model_copy(
        update={"potions": ("block_potion", "energy_potion", "strength_potion")}
    )
    state = _enter_monster_combat(state)

    state = step(state, _use_potion_action(state, "potion:0"))
    assert state.combat is not None
    assert state.combat.player.block == 12
    assert state.potions == ("energy_potion", "strength_potion")

    state = step(state, _use_potion_action(state, "potion:0"))
    assert state.combat is not None
    assert state.combat.player.energy == 5

    state = step(state, _use_potion_action(state, "potion:0"))
    assert state.combat is not None
    assert state.combat.player.statuses["strength"] == 2
    assert state.potions == ()


def test_later_same_target_potion_slot_remains_legal_after_slot_mismatch() -> None:
    state = new_run(seed=1311, character_id="TEST", ascension=0)
    state = state.model_copy(
        update={
            "potions": (
                "stable_serum",
                "fire_potion",
                "blessing_of_the_forge",
                "dexterity_potion",
            )
        }
    )
    state = _enter_monster_combat(state)

    action = _use_potion_action(state, "potion:2")
    state = step(state, action)

    assert state.potions == ("stable_serum", "fire_potion", "dexterity_potion")


def test_additional_status_and_targeted_potions_apply_player_and_enemy_effects() -> None:
    state = new_run(seed=1304, character_id="TEST", ascension=0)
    state = state.model_copy(
        update={
            "potions": (
                "focus_potion",
                "liquid_bronze",
                "poison_potion",
                "potion_of_binding",
            )
        }
    )
    state = _enter_monster_combat(state)

    assert state.combat is not None
    monster_id = state.combat.monsters[0].monster_id

    state = step(state, _use_potion_action(state, "potion:0"))
    assert state.combat is not None
    assert state.combat.player.statuses["focus"] == 2

    state = step(state, _use_potion_action(state, "potion:0"))
    assert state.combat is not None
    assert state.combat.player.statuses["thorns"] == 3

    state = step(state, _use_potion_action(state, "potion:0", target_id=monster_id))
    assert state.combat is not None
    assert state.combat.monsters[0].statuses["poison"] == 6

    state = step(state, _use_potion_action(state, "potion:0"))
    assert state.combat is not None
    assert state.combat.monsters[0].statuses["weak"] == 1
    assert state.combat.monsters[0].statuses["vulnerable"] == 1


def test_orb_potions_add_slots_and_channel_dark_for_each_slot() -> None:
    state = new_run(seed=1305, character_id="TEST", ascension=0)
    state = state.model_copy(
        update={"potions": ("potion_of_capacity", "essence_of_darkness", "focus_potion")}
    )
    state = _enter_monster_combat(state)

    assert state.combat is not None
    assert state.combat.orb_slots == 0

    state = step(state, _use_potion_action(state, "potion:0"))
    assert state.combat is not None
    assert state.combat.orb_slots == 2

    state = step(state, _use_potion_action(state, "potion:0"))
    assert state.combat is not None
    assert [orb.orb_id for orb in state.combat.orbs] == ["dark", "dark"]
    assert [orb.value for orb in state.combat.orbs] == [6, 6]

    state = step(state, _use_potion_action(state, "potion:0"))
    assert state.combat is not None
    assert state.combat.player.statuses["focus"] == 2


def test_foul_potion_damages_player_and_all_enemies() -> None:
    state = new_run(seed=1302, character_id="TEST", ascension=0)
    state = state.model_copy(update={"potions": ("foul_potion",)})
    state = _enter_monster_combat(state)

    assert state.combat is not None
    player_hp = state.combat.player.hp
    monster_hp = state.combat.monsters[0].hp
    state = step(state, _use_potion_action(state))

    assert state.potions == ()
    assert state.combat is not None
    assert state.combat.player.hp == player_hp - 12
    assert state.combat.monsters[0].hp == monster_hp - 12


def test_passive_fairy_is_not_manual_use_but_can_be_discarded() -> None:
    state = new_run(seed=1303, character_id="TEST", ascension=0)
    state = state.model_copy(update={"potions": ("fairy_in_a_bottle",)})
    state = _enter_monster_combat(state)

    assert not any(action.type == "use_potion" for action in legal_actions(state))
    assert any(action.type == "discard_potion" for action in legal_actions(state))


def test_fairy_in_a_bottle_revives_and_consumes_passive_potion() -> None:
    state = new_run(seed=1307, character_id="TEST", ascension=0)
    state = state.model_copy(update={"potions": ("fairy_in_a_bottle",)})
    state = _enter_monster_combat(state)
    assert state.combat is not None
    attacker = MonsterState(
        monster_id="fatal_attacker",
        name="Fatal Attacker",
        hp=30,
        max_hp=30,
        intent="attack",
        intent_damage=99,
    )
    player = state.combat.player.model_copy(update={"hp": 10, "max_hp": 80, "block": 0})
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={"hand": (), "monsters": (attacker,), "player": player}
            )
        }
    )

    state = step(state, _end_turn_action(state))

    assert state.phase is RunPhase.COMBAT
    assert state.potions == ()
    assert state.combat is not None
    assert state.combat.player.hp == 24
    assert any(
        event.kind == "potion_passive_triggered"
        and event.source_id == "fairy_in_a_bottle"
        and event.metadata["potion_slot"] == "potion:0"
        for event in state.combat.last_events
    )


def test_only_one_fairy_in_a_bottle_triggers_per_lethal_check() -> None:
    state = new_run(seed=1308, character_id="TEST", ascension=0)
    state = state.model_copy(update={"potions": ("fairy_in_a_bottle", "fairy_in_a_bottle")})
    state = _enter_monster_combat(state)
    assert state.combat is not None
    attacker = MonsterState(
        monster_id="fatal_attacker",
        name="Fatal Attacker",
        hp=30,
        max_hp=30,
        intent="attack",
        intent_damage=99,
    )
    player = state.combat.player.model_copy(update={"hp": 10, "max_hp": 80, "block": 0})
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={"hand": (), "monsters": (attacker,), "player": player}
            )
        }
    )

    state = step(state, _end_turn_action(state))

    assert state.phase is RunPhase.COMBAT
    assert state.potions.count("fairy_in_a_bottle") == 1
    assert state.combat is not None
    assert state.combat.player.hp == 24
    assert [
        event.kind for event in state.combat.last_events
    ].count("potion_passive_triggered") == 1


def test_attack_potion_generates_free_card_choice() -> None:
    state = new_run(seed=1310, character_id="TEST", ascension=0)
    state = state.model_copy(
        update={
            "potions": ("attack_potion",),
            "flags": {
                **state.flags,
                "card_pool": (
                    {
                        "id": "PICK_ATTACK_A",
                        "name": "Pick Attack A",
                        "type": "Attack",
                        "target": "AnyEnemy",
                        "cost": 1,
                        "damage": 3,
                        "color": "test",
                    },
                    {
                        "id": "PICK_ATTACK_B",
                        "name": "Pick Attack B",
                        "type": "Attack",
                        "target": "AnyEnemy",
                        "cost": 2,
                        "damage": 4,
                        "color": "test",
                    },
                    {
                        "id": "PICK_ATTACK_C",
                        "name": "Pick Attack C",
                        "type": "Attack",
                        "target": "AnyEnemy",
                        "cost": 3,
                        "damage": 5,
                        "color": "test",
                    },
                ),
            },
        }
    )
    state = _enter_monster_combat(state)

    state = step(state, _use_potion_action(state))

    assert state.combat is not None
    assert state.potions == ()
    assert len(state.combat.pending_choices) == 1
    choice = state.combat.pending_choices[0]
    assert choice.kind == "move_to_hand"
    assert choice.zone == "generated"
    assert len(choice.candidate_ids) == 3

    chosen_id = choice.candidate_ids[0]
    state = step(state, _choose_card_action(state, chosen_id))

    assert state.combat is not None
    chosen = next(card for card in state.combat.hand if card.instance_id == chosen_id)
    assert chosen.type.value == "attack"
    assert chosen.custom["free_to_play_this_turn"] is True
    assert state.combat.pending_choices == ()


def test_liquid_memories_moves_discarded_card_to_hand_free_for_combat() -> None:
    state = new_run(seed=1311, character_id="TEST", ascension=0)
    state = state.model_copy(update={"potions": ("liquid_memories",)})
    state = _enter_monster_combat(state)
    assert state.combat is not None
    recovered = _card("expensive_attack", card_type="attack", target="enemy", cost=3, damage=6)
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (),
                    "discard_pile": (recovered,),
                    "player": state.combat.player.model_copy(update={"energy": 0}),
                }
            )
        }
    )

    state = step(state, _use_potion_action(state))
    state = step(state, _choose_card_action(state, recovered.instance_id))

    assert state.combat is not None
    assert state.combat.hand[0].instance_id == recovered.instance_id
    assert state.combat.hand[0].custom["free_to_play_this_combat"] is True
    assert any(action.type == "play_card" for action in legal_actions(state))


def test_gamblers_brew_discards_choice_then_draws_that_many_on_proceed() -> None:
    state = new_run(seed=1312, character_id="TEST", ascension=0)
    state = state.model_copy(update={"potions": ("gamblers_brew",)})
    state = _enter_monster_combat(state)
    assert state.combat is not None
    junk_a = _card("junk_a")
    junk_b = _card("junk_b")
    draw_a = _card("draw_a")
    draw_b = _card("draw_b")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (junk_a, junk_b),
                    "draw_pile": (draw_a, draw_b),
                    "discard_pile": (),
                }
            )
        }
    )

    state = step(state, _use_potion_action(state))
    assert state.combat is not None
    assert state.combat.pending_choices[0].min_choices == 0
    state = step(state, _discard_card_action(state, junk_a.instance_id))
    assert state.combat is not None
    assert [card.card_id for card in state.combat.hand] == ["junk_b"]

    state = step(state, _proceed_action(state))

    assert state.combat is not None
    assert [card.card_id for card in state.combat.hand] == ["junk_b", "draw_a"]
    assert [card.card_id for card in state.combat.discard_pile] == ["junk_a"]
    assert state.combat.pending_choices == ()


def test_distilled_chaos_plays_top_three_draw_cards() -> None:
    state = new_run(seed=1313, character_id="TEST", ascension=0)
    state = state.model_copy(update={"potions": ("distilled_chaos",)})
    state = _enter_monster_combat(state)
    assert state.combat is not None
    draw_cards = (
        _card("top_attack_a", card_type="attack", target="enemy", damage=4),
        _card("top_block", block=3),
        _card("top_attack_b", card_type="attack", target="enemy", damage=5),
    )
    monster_hp = state.combat.monsters[0].hp
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={"hand": (), "draw_pile": draw_cards, "discard_pile": ()}
            )
        }
    )

    state = step(state, _use_potion_action(state))

    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 9
    assert state.combat.player.block == 3
    assert state.combat.draw_pile == ()
    assert [card.card_id for card in state.combat.discard_pile] == [
        "top_attack_a",
        "top_block",
        "top_attack_b",
    ]


def test_duplicator_replays_next_card_and_gigantification_triples_next_attack() -> None:
    state = new_run(seed=1314, character_id="TEST", ascension=0)
    state = state.model_copy(update={"potions": ("duplicator",)})
    state = _enter_monster_combat(state)
    assert state.combat is not None
    strike = _card("single_strike", card_type="attack", target="enemy", damage=5)
    monster_hp = state.combat.monsters[0].hp
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"hand": (strike,)})})

    state = step(state, _use_potion_action(state))
    state = step(state, _play_card_action(state, strike.instance_id))

    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 10
    assert any(event.kind == "card_extra_played" for event in state.combat.last_events)

    state = new_run(seed=1315, character_id="TEST", ascension=0)
    state = state.model_copy(update={"potions": ("gigantification_potion",)})
    state = _enter_monster_combat(state)
    assert state.combat is not None
    heavy = _card("heavy_hit", card_type="attack", target="enemy", damage=6)
    monster_hp = state.combat.monsters[0].hp
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"hand": (heavy,)})})

    state = step(state, _use_potion_action(state))
    state = step(state, _play_card_action(state, heavy.instance_id))

    assert state.combat is not None
    assert state.combat.monsters[0].hp == monster_hp - 18
    assert "next_card_damage_multiplier" not in state.combat.metadata


def test_entropic_brew_fills_open_slots_after_consuming_itself() -> None:
    state = new_run(seed=1316, character_id="TEST", ascension=0)
    state = state.model_copy(
        update={
            "potions": ("entropic_brew", "fire_potion"),
            "flags": {**state.flags, "potion_pool": ("skill_potion",)},
        }
    )
    state = _enter_monster_combat(state)

    state = step(state, _use_potion_action(state))

    assert state.potions == ("fire_potion", "skill_potion", "skill_potion")
    assert any(event.kind == "potion_generated" for event in state.combat.last_events)


def test_snecko_oil_draws_then_randomizes_hand_costs_until_turn_end() -> None:
    state = new_run(seed=1317, character_id="TEST", ascension=0)
    state = state.model_copy(update={"potions": ("snecko_oil",)})
    state = _enter_monster_combat(state)
    assert state.combat is not None
    hand_card = _card("held_card", cost=2)
    draw_card = _card("drawn_card", cost=1)
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={"hand": (hand_card,), "draw_pile": (draw_card,), "discard_pile": ()}
            )
        }
    )

    state = step(state, _use_potion_action(state))

    assert state.combat is not None
    assert [card.card_id for card in state.combat.hand] == ["held_card", "drawn_card"]
    assert all("randomized_cost_this_turn" in card.custom for card in state.combat.hand)
    assert all(0 <= card.cost <= 3 for card in state.combat.hand if card.cost is not None)
    assert any(event.kind == "hand_costs_randomized" for event in state.combat.last_events)


def test_bottled_potential_shuffles_piles_then_draws_five() -> None:
    state = new_run(seed=1318, character_id="TEST", ascension=0)
    state = state.model_copy(update={"potions": ("bottled_potential",)})
    state = _enter_monster_combat(state)
    assert state.combat is not None
    cards = tuple(_card(f"card_{index}") for index in range(6))
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": cards[:2],
                    "draw_pile": cards[2:3],
                    "discard_pile": cards[3:],
                }
            )
        }
    )

    state = step(state, _use_potion_action(state))

    assert state.combat is not None
    assert len(state.combat.hand) == 5
    assert len(state.combat.draw_pile) == 1
    assert state.combat.discard_pile == ()
    assert any(event.kind == "combat_cards_shuffled_to_draw" for event in state.combat.last_events)


def test_soldiers_stew_adds_replay_to_strikes_across_combat_piles() -> None:
    state = new_run(seed=1319, character_id="TEST", ascension=0)
    state = state.model_copy(update={"potions": ("soldiers_stew",)})
    state = _enter_monster_combat(state)
    assert state.combat is not None
    cards_by_zone = {
        "hand": (_card("strike_hand", card_type="attack", target="enemy", damage=1),),
        "draw_pile": (_card("strike_draw", card_type="attack", target="enemy", damage=1),),
        "discard_pile": (_card("strike_discard", card_type="attack", target="enemy", damage=1),),
        "exhaust_pile": (_card("strike_exhaust", card_type="attack", target="enemy", damage=1),),
    }
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={**cards_by_zone})}
    )

    state = step(state, _use_potion_action(state))

    assert state.combat is not None
    all_cards = (
        state.combat.hand
        + state.combat.draw_pile
        + state.combat.discard_pile
        + state.combat.exhaust_pile
    )
    assert all(
        any(enchantment.keyword == "replay" for enchantment in card.enchantments)
        for card in all_cards
    )


def test_beetle_juice_reduces_enemy_attack_damage_for_the_turn() -> None:
    state = new_run(seed=1320, character_id="TEST", ascension=0)
    state = state.model_copy(update={"potions": ("beetle_juice",)})
    state = _enter_monster_combat(state)
    assert state.combat is not None
    attacker = MonsterState(
        monster_id="test_attacker",
        name="Test Attacker",
        hp=30,
        max_hp=30,
        intent="attack",
        intent_damage=10,
    )
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (),
                    "monsters": (attacker,),
                    "player": state.combat.player.model_copy(update={"hp": 80, "block": 0}),
                }
            )
        }
    )

    state = step(state, _use_potion_action(state, target_id=attacker.monster_id))
    assert state.combat is not None
    assert state.combat.monsters[0].statuses["temporary_attack_damage_percent"] == -30

    state = step(state, _end_turn_action(state))

    assert state.combat is not None
    assert state.combat.player.hp == 73


def test_resource_and_status_potions_apply_combat_state() -> None:
    state = new_run(seed=1321, character_id="TEST", ascension=0)
    state = state.model_copy(
        update={
            "potions": (
                "bone_brew",
                "kings_courage",
                "star_potion",
                "ghost_in_a_jar",
                "lucky_tonic",
                "mazaleths_gift",
                "stable_serum",
            )
        }
    )
    state = _enter_monster_combat(state)

    state = step(state, _use_potion_action(state, "potion:0"))
    assert state.combat is not None
    assert state.combat.player.resources["summon"] == 15
    assert state.combat.metadata["osty"]["hp"] == 35

    state = step(state, _use_potion_action(state, "potion:0"))
    assert state.combat is not None
    assert state.combat.player.resources["forge"] == 15

    state = step(state, _use_potion_action(state, "potion:0"))
    assert state.combat is not None
    assert state.combat.player.resources["star"] == 3

    state = step(state, _use_potion_action(state, "potion:0"))
    assert state.combat is not None
    assert state.combat.player.statuses["intangible"] == 1

    state = step(state, _use_potion_action(state, "potion:0"))
    assert state.combat is not None
    assert state.combat.player.statuses["buffer"] == 1

    state = step(state, _use_potion_action(state, "potion:0"))
    assert state.combat is not None
    assert state.combat.player.statuses["ritual"] == 1

    state = step(state, _use_potion_action(state, "potion:0"))
    assert state.combat is not None
    assert state.combat.player.statuses["retain_hand"] == 2
    assert state.potions == ()


def test_cure_all_draws_and_glowwater_draws_then_exhausts_hand() -> None:
    state = new_run(seed=1322, character_id="TEST", ascension=0)
    state = state.model_copy(update={"potions": ("cure_all", "glowwater_potion")})
    state = _enter_monster_combat(state)
    assert state.combat is not None
    cure_draws = (_card("cure_draw_a"), _card("cure_draw_b"))
    player = state.combat.player.model_copy(update={"energy": 0})
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={"hand": (), "draw_pile": cure_draws, "player": player}
            )
        }
    )

    state = step(state, _use_potion_action(state, "potion:0"))

    assert state.combat is not None
    assert state.combat.player.energy == 1
    assert [card.card_id for card in state.combat.hand] == ["cure_draw_a", "cure_draw_b"]

    held = _card("held_card")
    glow_draws = (_card("glow_draw_a"), _card("glow_draw_b"))
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (held,),
                    "draw_pile": glow_draws,
                    "discard_pile": (),
                    "exhaust_pile": (),
                }
            )
        }
    )

    state = step(state, _use_potion_action(state, "potion:0"))

    assert state.combat is not None
    assert state.combat.hand == ()
    assert state.combat.draw_pile == ()
    assert [card.card_id for card in state.combat.exhaust_pile] == [
        "held_card",
        "glow_draw_a",
        "glow_draw_b",
    ]
