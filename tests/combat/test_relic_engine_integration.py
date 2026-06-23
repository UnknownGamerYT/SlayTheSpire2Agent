from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import sts2sim.engine.transitions as transitions_module
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
from sts2sim.mechanics.relic_combat import (
    CombatRelicHook,
    CombatRelicMarker,
    CombatRelicResolution,
)
from sts2sim.mechanics.triggers import GameTrigger, TriggerResolution

DUMMY_MONSTERS = (
    {
        "id": "RELIC_DUMMY",
        "name": "Relic Dummy",
        "type": "Normal",
        "min_hp": 50,
        "max_hp": 50,
        "moves": (
            {
                "id": "WAIT",
                "name": "Wait",
                "intent": "Unknown",
            },
        ),
        "attack_pattern": {
            "initial_move": "WAIT",
            "states": ({"id": "WAIT_MOVE", "move_id": "WAIT", "type": "move"},),
        },
    },
)

DUMMY_ENCOUNTERS = (
    {
        "id": "RELIC_DUMMY_ENCOUNTER",
        "name": "Relic Dummy",
        "act": "Act 1 - Overgrowth",
        "room_type": "Monster",
        "is_weak": True,
        "monsters": ({"id": "RELIC_DUMMY"},),
    },
)

REWARD_CARDS = (
    {
        "id": "COMMON_ATTACK",
        "name": "Common Attack",
        "rarity": "Common",
        "color": "test",
        "type": "Attack",
        "target": "AnyEnemy",
        "damage": 6,
    },
    {
        "id": "COMMON_SKILL",
        "name": "Common Skill",
        "rarity": "Common",
        "color": "test",
        "type": "Skill",
        "target": "Self",
        "block": 5,
    },
)

REWARD_RELICS = (
    {"id": "COMMON_RELIC", "name": "Common Relic", "rarity_key": "Common", "pool": "shared"},
    {"id": "UNCOMMON_RELIC", "name": "Uncommon Relic", "rarity_key": "Uncommon", "pool": "shared"},
    {"id": "RARE_RELIC", "name": "Rare Relic", "rarity_key": "Rare", "pool": "shared"},
)


def _card_spec(card_id: str) -> dict[str, Any]:
    return {
        "id": card_id,
        "name": card_id.replace("_", " ").title(),
        "type": "Skill",
        "target": "Self",
        "cost": 0,
    }


def _card(card_id: str) -> CardInstance:
    return CardInstance(
        instance_id=f"{card_id}:1",
        card_id=card_id,
        name=card_id.replace("_", " ").title(),
        type="skill",
        target="self",
        cost=0,
    )


def _direct_card(
    card_id: str,
    *,
    card_type: str = "skill",
    target: str = "self",
    cost: int = 0,
    effects: Mapping[str, Any] | None = None,
    custom: Mapping[str, Any] | None = None,
    exhausts: bool = False,
) -> CardInstance:
    return CardInstance(
        instance_id=f"{card_id}:1",
        card_id=card_id,
        name=card_id.replace("_", " ").title(),
        type=card_type,
        target=target,
        cost=cost,
        effects=dict(effects or {}),
        custom=dict(custom or {}),
        exhausts=exhausts,
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


def _enter_combat(
    deck: Sequence[Mapping[str, Any]],
    *,
    relics: Sequence[str],
    draw_per_turn: int = 5,
    potions: Sequence[str] = (),
    room_kind: RoomKind = RoomKind.MONSTER,
    flags: Mapping[str, Any] | None = None,
    source_data_extra: Mapping[str, Any] | None = None,
):
    source_data = {
        "deck": tuple(dict(card) for card in deck),
        "player": {"hp": 80, "max_hp": 80, "energy": 3, "max_energy": 3},
        "monsters": DUMMY_MONSTERS,
        "encounters": DUMMY_ENCOUNTERS,
        "combat_encounter_id": "RELIC_DUMMY_ENCOUNTER",
        "draw_per_turn": draw_per_turn,
        "cards": REWARD_CARDS,
        "relic_pool": REWARD_RELICS,
        "potion_pool": (
            "fire_potion",
            "skill_potion",
            "essence_of_steel",
            "potion_shaped_rock",
        ),
    }
    if source_data_extra is not None:
        source_data.update(dict(source_data_extra))
    state = new_run(
        seed=9100,
        character_id="TEST",
        ascension=0,
        source_data=source_data,
    )
    state = _choose_first_ancient(state)
    updates: dict[str, Any] = {"relics": tuple(relics), "potions": tuple(potions)}
    if flags is not None:
        updates["flags"] = {**state.flags, **dict(flags)}
    state = state.model_copy(update=updates)
    state = _force_next_room(state, room_kind)
    return step(
        state,
        next(action for action in legal_actions(state) if action.type == "choose_node"),
    )


def _end_turn(state):
    return step(
        state,
        next(action for action in legal_actions(state) if action.type == "end_turn"),
    )


def _play_card(state, card_id: str):
    assert state.combat is not None
    card = next(card for card in state.combat.hand if card.card_id == card_id)
    return step(
        state,
        next(
            action
            for action in legal_actions(state)
            if action.type == "play_card" and action.card_instance_id == card.instance_id
        ),
    )


def _use_potion(state, potion_slot: str = "potion:0", target_id: str | None = None):
    return step(
        state,
        next(
            action
            for action in legal_actions(state)
            if action.type == "use_potion"
            and action.payload.get("potion_slot") == potion_slot
            and (target_id is None or action.target_id == target_id)
        ),
    )


def _all_replay_events(state):
    return tuple(event for entry in state.replay_log for event in entry.events)


def _fake_runtime_relic_marker(
    monkeypatch,
    trigger: GameTrigger,
    *,
    relic_id: str,
    marker_kind: str,
    amount: int,
) -> list[tuple[GameTrigger, Mapping[str, Any]]]:
    original = transitions_module.resolve_game_trigger
    calls: list[tuple[GameTrigger, Mapping[str, Any]]] = []

    def fake_resolve_game_trigger(requested_trigger, *args, **kwargs):
        if requested_trigger is trigger:
            context = kwargs["context"]
            calls.append((trigger, context.metadata))
            marker = CombatRelicMarker(
                kind=marker_kind,
                relic_id=relic_id,
                hook=CombatRelicHook.CARD_PLAYED,
                amount=amount,
                target_id="player",
            )
            return TriggerResolution(
                trigger=trigger,
                combat_relic_resolution=CombatRelicResolution(
                    hook=CombatRelicHook.CARD_PLAYED,
                    markers=(marker,),
                ),
            )
        return original(requested_trigger, *args, **kwargs)

    monkeypatch.setattr(transitions_module, "resolve_game_trigger", fake_resolve_game_trigger)
    return calls


def test_start_combat_relics_mutate_combat_state() -> None:
    deck = tuple(_card_spec(f"card_{index}") for index in range(9))
    state = _enter_combat(
        deck,
        relics=(
            "lantern",
            "gorget",
            "divine_right",
            "fencing_manual",
            "runic_capacitor",
            "infused_core",
            "ring_of_the_snake",
            "snecko_eye",
        ),
    )

    assert state.combat is not None
    assert state.combat.player.energy == 4
    assert state.combat.player.statuses["plated_armor"] == 4
    assert state.combat.player.statuses["confused"] == 1
    assert state.combat.player.resources["star"] == 3
    assert state.combat.player.resources["forge"] == 10
    assert state.combat.orb_slots == 3
    assert [orb.orb_id for orb in state.combat.orbs] == ["lightning", "lightning", "lightning"]
    assert len(state.combat.hand) == 9
    assert any(
        event.kind == "relic_bonus_draw_applied" and event.amount == 4
        for event in state.combat.last_events
    )


def test_more_start_combat_relics_create_cards_and_apply_enemy_statuses() -> None:
    state = _enter_combat(
        (),
        relics=(
            "belt_buckle",
            "ninja_scroll",
            "funerary_mask",
            "twisted_funnel",
            "blessed_antler",
            "royal_poison",
            "very_hot_cocoa",
        ),
        draw_per_turn=0,
    )

    assert state.combat is not None
    assert state.combat.player.hp == 76
    assert state.combat.player.energy == 8
    assert state.combat.player.statuses["dexterity"] == 2
    assert [card.card_id for card in state.combat.hand] == ["shiv", "shiv", "shiv"]
    assert [card.card_id for card in state.combat.draw_pile[:3]] == ["soul", "soul", "soul"]
    assert [card.card_id for card in state.combat.draw_pile[-3:]] == ["dazed", "dazed", "dazed"]
    assert all(card.custom["ethereal"] for card in state.combat.draw_pile[-3:])
    assert state.combat.monsters[0].statuses["poison"] == 4

    with_potion = _enter_combat((), relics=("belt_buckle",), potions=("fire_potion",))
    assert with_potion.combat is not None
    assert "dexterity" not in with_potion.combat.player.statuses


def test_draw_pile_start_relics_move_add_and_upgrade_cards() -> None:
    deck = (
        {"id": "zero_a", "name": "Zero A", "type": "Skill", "target": "Self", "cost": 0},
        {"id": "zero_b", "name": "Zero B", "type": "Attack", "target": "Enemy", "cost": 0},
        {"id": "power_a", "name": "Power A", "type": "Power", "target": "Self", "cost": 1},
        {"id": "upgrade_a", "name": "Upgrade A", "type": "Skill", "target": "Self", "cost": 1},
        {"id": "upgrade_b", "name": "Upgrade B", "type": "Skill", "target": "Self", "cost": 1},
    )
    state = _enter_combat(
        deck,
        relics=("power_cell", "jeweled_mask", "radiant_pearl", "stone_cracker"),
        draw_per_turn=0,
    )

    assert state.combat is not None
    hand_by_id = {card.card_id: card for card in state.combat.hand}
    assert {"zero_a", "zero_b", "power_a", "luminesce"} <= set(hand_by_id)
    assert hand_by_id["zero_a"].custom["free_to_play_this_turn"] is True
    assert hand_by_id["zero_b"].custom["free_to_play_this_turn"] is True
    assert hand_by_id["power_a"].custom["free_to_play_this_turn"] is True
    assert sum(1 for card in state.combat.draw_pile if card.upgraded) == 2
    assert any(event.kind == "draw_pile_cards_moved_to_hand" for event in state.combat.last_events)
    assert any(event.kind == "draw_pile_cards_upgraded" for event in state.combat.last_events)


def test_bellows_upgrades_opening_hand_before_draw() -> None:
    deck = tuple(_card_spec(f"bellows_card_{index}") for index in range(7))
    state = _enter_combat(deck, relics=("bellows",), draw_per_turn=5)

    assert state.combat is not None
    assert len(state.combat.hand) == 5
    assert all(card.upgraded for card in state.combat.hand)
    assert not any(card.upgraded for card in state.combat.draw_pile)
    assert any(
        event.kind == "draw_pile_cards_upgraded" and event.amount == 5
        for event in state.combat.last_events
    )


def test_sling_of_courage_applies_only_in_elite_combat() -> None:
    elite = _enter_combat(
        (),
        relics=("sling_of_courage",),
        draw_per_turn=0,
        room_kind=RoomKind.ELITE,
    )
    hallway = _enter_combat((), relics=("sling_of_courage",), draw_per_turn=0)

    assert elite.combat is not None
    assert hallway.combat is not None
    assert elite.combat.player.statuses["strength"] == 2
    assert "strength" not in hallway.combat.player.statuses


def test_big_mushroom_reduces_opening_draw() -> None:
    deck = tuple(_card_spec(f"card_{index}") for index in range(5))
    state = _enter_combat(deck, relics=("big_mushroom",))

    assert state.combat is not None
    assert len(state.combat.hand) == 3
    assert any(
        event.kind == "relic_bonus_draw_applied" and event.amount == -2
        for event in state.combat.last_events
    )


def test_ring_of_the_drake_draws_on_first_three_turn_starts() -> None:
    deck = tuple(_card_spec(f"drake_card_{index}") for index in range(6))
    state = _enter_combat(deck, relics=("ring_of_the_drake",), draw_per_turn=0)

    assert state.combat is not None
    assert len(state.combat.hand) == 2

    state = _end_turn(state)
    assert state.combat is not None
    assert state.combat.turn == 2
    assert len(state.combat.hand) == 2

    state = _end_turn(state)
    assert state.combat is not None
    assert state.combat.turn == 3
    assert len(state.combat.hand) == 2

    state = _end_turn(state)
    assert state.combat is not None
    assert state.combat.turn == 4
    assert len(state.combat.hand) == 0


def test_bone_tea_charge_upgrades_opening_hand_once() -> None:
    deck = tuple(_card_spec(f"bone_tea_card_{index}") for index in range(6))
    state = _enter_combat(
        deck,
        relics=("bone_tea",),
        flags={"relic_counters": {"bone_tea": 1}},
    )

    assert state.combat is not None
    assert len(state.combat.hand) == 5
    assert all(card.upgraded for card in state.combat.hand)
    assert state.combat.metadata["relic_counters"]["bone_tea"] == 0
    assert any(
        event.kind == "draw_pile_cards_upgraded"
        and event.metadata.get("relic_id") == "bone_tea"
        for event in state.combat.last_events
    )


def test_turn_start_relic_draws_are_applied_after_opening_turn() -> None:
    state = _enter_combat((), relics=("paels_blood",), draw_per_turn=1)
    assert state.combat is not None
    draw_cards = (_card("draw_a"), _card("draw_b"), _card("draw_c"))
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (),
                    "draw_pile": draw_cards,
                    "discard_pile": (),
                    "monsters": (
                        MonsterState(
                            monster_id="quiet_dummy",
                            name="Quiet Dummy",
                            hp=50,
                            max_hp=50,
                            intent="unknown",
                        ),
                    ),
                }
            )
        }
    )

    state = _end_turn(state)

    assert state.combat is not None
    assert [card.card_id for card in state.combat.hand] == ["draw_a", "draw_b"]
    assert any(
        event.kind == "relic_bonus_draw_applied" and event.amount == 1
        for event in state.combat.last_events
    )


def test_turn_timing_relics_mutate_energy_and_block_across_turns() -> None:
    state = _enter_combat(
        (),
        relics=(
            "bread",
            "art_of_war",
            "candelabra",
            "horn_cleat",
            "captains_wheel",
            "chandelier",
        ),
        draw_per_turn=0,
    )

    assert state.combat is not None
    assert state.combat.player.energy == 1

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.turn == 2
    assert state.combat.player.energy == 7
    assert state.combat.player.block == 14

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.turn == 3
    assert state.combat.player.energy == 8
    assert state.combat.player.block == 18


def test_paels_flesh_and_sparkling_rouge_apply_on_turn_three() -> None:
    state = _enter_combat((), relics=("paels_flesh", "sparkling_rouge"), draw_per_turn=0)
    assert state.combat is not None
    assert "strength" not in state.combat.player.statuses

    state = _end_turn(state)
    assert state.combat is not None
    assert state.combat.turn == 2
    assert "strength" not in state.combat.player.statuses

    state = _end_turn(state)
    assert state.combat is not None
    assert state.combat.turn == 3
    assert state.combat.player.energy == 4
    assert state.combat.player.statuses["strength"] == 1
    assert state.combat.player.statuses["dexterity"] == 1


def test_ripple_basin_blocks_when_no_attacks_were_played() -> None:
    state = _enter_combat((), relics=("ripple_basin",), draw_per_turn=0)

    state = _end_turn(state)

    assert state.combat is not None
    assert any(
        event.kind == "player_block"
        and event.source_id == "ripple_basin"
        and event.amount == 4
        for event in state.combat.last_events
    )


def test_pocketwatch_draws_three_extra_cards_after_quiet_turn() -> None:
    deck = tuple(_card_spec(f"watch_card_{index}") for index in range(10))
    state = _enter_combat(deck, relics=("pocketwatch",), draw_per_turn=5)
    assert state.combat is not None

    state = _end_turn(state)

    assert state.combat is not None
    assert len(state.combat.hand) == 8
    assert any(
        event.kind == "status_applied"
        and event.amount == 3
        and event.source_id == "pocketwatch"
        and event.metadata.get("status") == "next_turn_draw"
        for event in state.combat.last_events
    )


def test_card_played_relics_mutate_combat_state() -> None:
    deck = (
        {
            "id": "attack_a",
            "name": "Attack A",
            "type": "Attack",
            "target": "Enemy",
            "cost": 0,
            "damage": 1,
        },
        _card_spec("skill_a"),
        _card_spec("skill_b"),
        _card_spec("skill_c"),
        {"id": "power_a", "name": "Power A", "type": "Power", "target": "Self", "cost": 0},
    )
    state = _enter_combat(
        deck,
        relics=("daughter_of_the_wind", "letter_opener", "lost_wisp"),
        draw_per_turn=5,
    )
    assert state.combat is not None
    starting_hp = state.combat.monsters[0].hp

    state = _play_card(state, "attack_a")
    assert state.combat is not None
    assert state.combat.player.block == 1

    state = _play_card(state, "skill_a")
    state = _play_card(state, "skill_b")
    state = _play_card(state, "skill_c")
    assert state.combat is not None
    assert state.combat.monsters[0].hp == starting_hp - 1 - 5

    state = _play_card(state, "power_a")
    assert state.combat is not None
    assert state.combat.monsters[0].hp == starting_hp - 1 - 5 - 8


def test_power_play_relics_draw_and_make_a_hand_card_free() -> None:
    state = _enter_combat((), relics=("game_piece", "mummified_hand"), draw_per_turn=0)
    assert state.combat is not None
    power = _direct_card("power_a", card_type="power", cost=0)
    costly = _direct_card("costly_skill", cost=2)
    drawn = _direct_card("drawn_skill")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (power, costly),
                    "draw_pile": (drawn,),
                    "discard_pile": (),
                }
            )
        }
    )

    state = _play_card(state, "power_a")

    assert state.combat is not None
    hand_by_id = {card.card_id: card for card in state.combat.hand}
    assert "drawn_skill" in hand_by_id
    assert hand_by_id["costly_skill"].custom["free_to_play_this_turn"] is True
    assert any(
        event.kind == "relic_bonus_draw_applied" and event.amount == 1
        for event in state.combat.last_events
    )
    assert any(event.kind == "card_made_free_this_turn" for event in state.combat.last_events)


def test_iron_club_and_brilliant_scarf_use_total_cards_played_this_turn() -> None:
    state = _enter_combat((), relics=("iron_club", "brilliant_scarf"), draw_per_turn=0)
    assert state.combat is not None
    played_cards = tuple(_direct_card(f"play_{index}") for index in range(5))
    costly = _direct_card("costly_skill", cost=2)
    drawn = _direct_card("drawn_skill")
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": played_cards + (costly,),
                    "draw_pile": (drawn,),
                    "discard_pile": (),
                }
            )
        }
    )

    for index in range(4):
        state = _play_card(state, f"play_{index}")

    assert state.combat is not None
    assert any(card.card_id == "drawn_skill" for card in state.combat.hand)
    assert any(
        event.kind == "relic_bonus_draw_applied" and event.amount == 1
        for event in state.combat.last_events
    )

    state = _play_card(state, "play_4")

    assert state.combat is not None
    hand_by_id = {card.card_id: card for card in state.combat.hand}
    assert hand_by_id["costly_skill"].custom["free_to_play_this_turn"] is True
    assert any(event.kind == "card_made_free_this_turn" for event in state.combat.last_events)


def test_nunchaku_counter_grants_energy_on_tenth_attack() -> None:
    state = _enter_combat((), relics=("nunchaku",), draw_per_turn=0)
    assert state.combat is not None
    attack = _direct_card("quick_hit", card_type="attack", target="enemy", effects={"damage": 1})
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (attack,),
                    "player": state.combat.player.model_copy(update={"energy": 0}),
                    "metadata": {**state.combat.metadata, "relic_counters": {"nunchaku": 9}},
                }
            )
        }
    )

    state = _play_card(state, "quick_hit")

    assert state.combat is not None
    assert state.combat.player.energy == 1
    assert state.combat.metadata["relic_counters"]["nunchaku"] == 0


def test_runic_pyramid_retains_full_hand_at_end_turn() -> None:
    deck = tuple(_card_spec(f"keep_{index}") for index in range(2))
    state = _enter_combat(deck, relics=("runic_pyramid",), draw_per_turn=2)
    assert state.combat is not None
    starting_ids = [card.instance_id for card in state.combat.hand]

    state = _end_turn(state)

    assert state.combat is not None
    assert [card.instance_id for card in state.combat.hand] == starting_ids
    assert state.combat.discard_pile == ()
    assert any(
        event.kind == "hand_retained"
        and event.metadata.get("relic_id") == "runic_pyramid"
        for event in state.combat.last_events
    )


def test_ringing_triangle_retains_full_hand_on_first_turn() -> None:
    deck = tuple(_card_spec(f"triangle_keep_{index}") for index in range(2))
    state = _enter_combat(deck, relics=("ringing_triangle",), draw_per_turn=2)
    assert state.combat is not None
    starting_ids = [card.instance_id for card in state.combat.hand]

    state = _end_turn(state)

    assert state.combat is not None
    assert [card.instance_id for card in state.combat.hand] == starting_ids
    assert state.combat.discard_pile == ()
    assert any(
        event.kind == "hand_retained"
        and event.metadata.get("relic_id") == "ringing_triangle"
        for event in state.combat.last_events
    )


def test_damage_bonus_relics_change_card_damage_math() -> None:
    deck = (
        {
            "id": "pommel_strike",
            "name": "Pommel Strike",
            "type": "Attack",
            "target": "Enemy",
            "cost": 0,
            "damage": 6,
        },
        {
            "id": "upgraded_slash",
            "name": "Upgraded Slash",
            "type": "Attack",
            "target": "Enemy",
            "cost": 0,
            "damage": 6,
            "upgraded": True,
        },
        {
            "id": "enchanted_cut",
            "name": "Enchanted Cut",
            "type": "Attack",
            "target": "Enemy",
            "cost": 0,
            "damage": 6,
            "enchantments": ({"keyword": "Sharp", "amount": 1},),
        },
    )
    state = _enter_combat(
        deck,
        relics=("strike_dummy", "fake_strike_dummy", "miniature_cannon", "mystic_lighter"),
        draw_per_turn=3,
    )
    assert state.combat is not None
    starting_hp = state.combat.monsters[0].hp

    state = _play_card(state, "pommel_strike")
    state = _play_card(state, "upgraded_slash")
    state = _play_card(state, "enchanted_cut")

    assert state.combat is not None
    assert state.combat.monsters[0].hp == starting_hp - 10 - 9 - 15


def test_paper_krane_and_tungsten_rod_reduce_incoming_hp_loss() -> None:
    state = _enter_combat((), relics=("paper_krane", "tungsten_rod"), draw_per_turn=0)
    assert state.combat is not None
    attacker = MonsterState(
        monster_id="weak_attacker",
        name="Weak Attacker",
        hp=30,
        max_hp=30,
        intent="attack",
        intent_damage=10,
        statuses={"weak": 1},
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

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.player.hp == 75
    damage_event = next(
        event for event in state.combat.last_events if event.kind == "player_damaged"
    )
    assert damage_event.metadata["incoming"] == 6
    assert damage_event.metadata["tungsten_rod_reduction"] == 1


def test_diamond_diadem_and_self_forming_clay_react_to_incoming_damage() -> None:
    state = _enter_combat((), relics=("diamond_diadem", "self_forming_clay"), draw_per_turn=0)
    assert state.combat is not None
    attacker = MonsterState(
        monster_id="attacker",
        name="Attacker",
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

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.player.hp == 75
    assert state.combat.player.block == 3
    damage_event = next(
        event for event in state.combat.last_events if event.kind == "player_damaged"
    )
    assert damage_event.metadata["diamond_diadem"] is True
    assert damage_event.metadata["diamond_diadem_reduction"] == 5
    assert any(
        event.kind == "status_applied"
        and event.source_id == "self_forming_clay"
        and event.metadata["status"] == "next_turn_block"
        for event in state.combat.last_events
    )


def test_beating_remnant_caps_hp_loss_per_turn() -> None:
    state = _enter_combat((), relics=("beating_remnant",), draw_per_turn=0)
    assert state.combat is not None
    attacker = MonsterState(
        monster_id="huge_attacker",
        name="Huge Attacker",
        hp=30,
        max_hp=30,
        intent="attack",
        intent_damage=50,
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

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.player.hp == 60
    damage_event = next(
        event for event in state.combat.last_events if event.kind == "player_damaged"
    )
    assert damage_event.amount == 20
    assert damage_event.metadata["beating_remnant_prevented"] == 30
    assert state.combat.metadata["hp_lost_this_turn"] == 0


def test_centennial_puzzle_draws_after_first_combat_hp_loss() -> None:
    state = _enter_combat((), relics=("centennial_puzzle",), draw_per_turn=0)
    assert state.combat is not None
    attacker = MonsterState(
        monster_id="attacker",
        name="Attacker",
        hp=30,
        max_hp=30,
        intent="attack",
        intent_damage=5,
    )
    draw_cards = (
        _direct_card("puzzle_draw_a"),
        _direct_card("puzzle_draw_b"),
        _direct_card("puzzle_draw_c"),
    )
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (),
                    "draw_pile": draw_cards,
                    "monsters": (attacker,),
                    "player": state.combat.player.model_copy(update={"hp": 80, "block": 0}),
                }
            )
        }
    )

    state = _end_turn(state)

    assert state.combat is not None
    assert [card.card_id for card in state.combat.hand] == [
        "puzzle_draw_a",
        "puzzle_draw_b",
        "puzzle_draw_c",
    ]
    assert state.combat.metadata["relic_counters"]["centennial_puzzle"] == 1
    assert any(
        event.kind == "relic_bonus_draw_applied" and event.amount == 3
        for event in state.combat.last_events
    )


def test_gremlin_horn_gains_energy_and_draws_when_monster_is_killed() -> None:
    state = _enter_combat((), relics=("gremlin_horn",), draw_per_turn=0)
    assert state.combat is not None
    attack = _direct_card("kill_attack", card_type="attack", target="enemy", effects={"damage": 5})
    draw_card = _direct_card("horn_draw")
    monsters = (
        MonsterState(
            monster_id="first_target",
            name="First Target",
            hp=5,
            max_hp=5,
            intent="unknown",
        ),
        MonsterState(
            monster_id="second_target",
            name="Second Target",
            hp=30,
            max_hp=30,
            intent="unknown",
        ),
    )
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (attack,),
                    "draw_pile": (draw_card,),
                    "monsters": monsters,
                    "player": state.combat.player.model_copy(update={"energy": 0}),
                }
            )
        }
    )

    state = _play_card(state, "kill_attack")

    assert state.combat is not None
    assert state.combat.player.energy == 1
    assert any(card.card_id == "horn_draw" for card in state.combat.hand)
    assert any(
        event.kind == "relic_bonus_draw_applied" and event.amount == 1
        for event in state.combat.last_events
    )


def test_runtime_discard_relic_hook_applies_returned_marker(monkeypatch) -> None:
    calls = _fake_runtime_relic_marker(
        monkeypatch,
        GameTrigger.CARD_DISCARDED,
        relic_id="tingsha",
        marker_kind="gain_block",
        amount=2,
    )
    state = _enter_combat((), relics=("tingsha",), draw_per_turn=0)
    assert state.combat is not None
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (
                        _direct_card("discarder", effects={"discard_random": 1}),
                        _direct_card("discard_me"),
                    ),
                    "player": state.combat.player.model_copy(update={"block": 0}),
                }
            )
        }
    )

    state = _play_card(state, "discarder")

    assert state.combat is not None
    assert state.combat.player.block == 2
    assert len(calls) == 1
    assert calls[0][1]["reason"] == "random_discard"
    assert calls[0][1]["card_id"] == "discard_me"


def test_runtime_exhaust_shuffle_and_potion_hooks_apply_returned_markers(monkeypatch) -> None:
    exhaust_calls = _fake_runtime_relic_marker(
        monkeypatch,
        GameTrigger.CARD_EXHAUSTED,
        relic_id="charons_ashes",
        marker_kind="gain_energy",
        amount=1,
    )
    state = _enter_combat((), relics=("charons_ashes",), draw_per_turn=0)
    assert state.combat is not None
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (_direct_card("burn_out", exhausts=True),),
                    "player": state.combat.player.model_copy(update={"energy": 0}),
                }
            )
        }
    )

    state = _play_card(state, "burn_out")

    assert state.combat is not None
    assert state.combat.player.energy == 1
    assert len(exhaust_calls) == 1

    shuffle_calls = _fake_runtime_relic_marker(
        monkeypatch,
        GameTrigger.DRAW_PILE_SHUFFLED,
        relic_id="the_abacus",
        marker_kind="gain_block",
        amount=4,
    )
    state = _enter_combat((), relics=("the_abacus",), draw_per_turn=0)
    assert state.combat is not None
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (_direct_card("draw_now", effects={"draw": 1}),),
                    "draw_pile": (),
                    "discard_pile": (_direct_card("reshuffled"),),
                    "player": state.combat.player.model_copy(update={"block": 0}),
                }
            )
        }
    )

    state = _play_card(state, "draw_now")

    assert state.combat is not None
    assert state.combat.player.block == 4
    assert len(shuffle_calls) == 1

    potion_calls = _fake_runtime_relic_marker(
        monkeypatch,
        GameTrigger.POTION_USED,
        relic_id="reptile_trinket",
        marker_kind="gain_energy",
        amount=1,
    )
    state = _enter_combat(
        (),
        relics=("reptile_trinket",),
        potions=("fire_potion",),
        draw_per_turn=0,
    )
    assert state.combat is not None
    target_id = state.combat.monsters[0].monster_id
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={"player": state.combat.player.model_copy(update={"energy": 0})}
            )
        }
    )

    state = _use_potion(state, target_id=target_id)

    assert state.combat is not None
    assert state.combat.player.energy == 1
    assert len(potion_calls) == 1
    assert potion_calls[0][1]["potion_id"] == "fire_potion"


def test_runtime_relic_surfaces_apply_real_hook_effects() -> None:
    state = _enter_combat(
        (),
        relics=(
            "regalite",
            "galactic_dust",
            "mini_regent",
            "vambrace",
            "snecko_skull",
            "unsettling_lamp",
            "throwing_axe",
        ),
        draw_per_turn=0,
    )
    assert state.combat is not None
    hook_card = _direct_card(
        "many_surfaces",
        target="enemy",
        effects={
            "block": 5,
            "apply_status": {"target": "enemy", "poison": 2},
            "add_card_to_hand": {"card_id": "created_card", "type": "skill", "target": "self"},
        },
        custom={"star_cost": 1},
    )
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (hook_card,),
                    "draw_pile": (),
                    "player": state.combat.player.model_copy(
                        update={"resources": {"star": 1}}
                    ),
                }
            )
        }
    )

    state = _play_card(state, "many_surfaces")

    assert state.combat is not None
    assert state.combat.player.resources["star"] == 0
    assert state.combat.player.statuses["strength"] == 1
    assert state.combat.player.block == 19
    assert state.combat.monsters[0].statuses["poison"] == 6
    assert [card.card_id for card in state.combat.hand] == ["created_card", "created_card"]
    assert len({card.instance_id for card in state.combat.hand}) == 2
    assert any(
        event.kind == "card_extra_played"
        and event.metadata.get("relic_id") == "throwing_axe"
        for event in state.combat.last_events
    )
    assert any(
        event.kind == "player_block"
        and event.metadata.get("relic_id") == "vambrace"
        for event in state.combat.last_events
    )
    assert any(
        event.kind == "player_block"
        and event.metadata.get("relic_id") == "regalite"
        for event in state.combat.last_events
    )
    assert all(event.kind != "combat_relic_hook_pending" for event in state.combat.last_events)
    assert all(event.kind != "combat_relic_marker_stubbed" for event in state.combat.last_events)


def test_unceasing_top_draws_when_hand_is_empty_after_card_play() -> None:
    state = _enter_combat((), relics=("unceasing_top",), draw_per_turn=0)
    assert state.combat is not None
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (_direct_card("last_card"),),
                    "draw_pile": (_direct_card("drawn"),),
                }
            )
        }
    )

    state = _play_card(state, "last_card")

    assert state.combat is not None
    assert [card.card_id for card in state.combat.hand] == ["drawn"]
    assert any(
        event.kind == "relic_draw_scheduled"
        and event.metadata.get("relic_id") == "unceasing_top"
        for event in state.combat.last_events
    )
    assert any(event.kind == "relic_bonus_draw_applied" for event in state.combat.last_events)


def test_gambling_chip_opens_optional_discard_redraw_choice_after_opening_draw() -> None:
    state = _enter_combat(
        tuple(_card_spec(f"chip_card_{index}") for index in range(6)),
        relics=("gambling_chip",),
        draw_per_turn=5,
    )

    assert state.combat is not None
    assert len(state.combat.hand) == 5
    assert state.combat.pending_choices
    choice = state.combat.pending_choices[0]
    assert choice.kind == "discard"
    assert choice.source_id == "gambling_chip"
    assert choice.required is False
    assert choice.min_choices == 0
    assert choice.max_choices == len(state.combat.hand)
    assert set(choice.candidate_ids) == {card.instance_id for card in state.combat.hand}
    assert any(action.type == "proceed" for action in legal_actions(state))
    assert all(event.kind != "combat_relic_marker_stubbed" for event in state.combat.last_events)


def test_start_combat_potion_relics_fill_only_open_potion_slots() -> None:
    toad_state = _enter_combat(
        (),
        relics=("petrified_toad",),
        potions=("fire_potion", "skill_potion"),
        draw_per_turn=0,
    )

    assert toad_state.potions == ("fire_potion", "skill_potion", "potion_shaped_rock")
    assert toad_state.combat is not None
    assert all(
        event.kind != "combat_relic_marker_stubbed"
        for event in toad_state.combat.last_events
    )

    frond_state = _enter_combat(
        (),
        relics=("delicate_frond",),
        potions=("fire_potion",),
        draw_per_turn=0,
    )

    assert len(frond_state.potions) == 3
    assert frond_state.potions[0] == "fire_potion"
    assert all(frond_state.potions)
    assert frond_state.combat is not None
    assert all(
        event.kind != "combat_relic_marker_stubbed"
        for event in frond_state.combat.last_events
    )


def test_history_course_plays_copy_of_last_attack_or_skill_at_turn_start() -> None:
    state = _enter_combat(
        (
            {
                "id": "history_hit",
                "name": "History Hit",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 6,
            },
        ),
        relics=("history_course",),
        draw_per_turn=1,
    )
    assert state.combat is not None
    starting_hp = state.combat.monsters[0].hp

    state = _play_card(state, "history_hit")
    assert state.combat is not None
    hp_after_play = state.combat.monsters[0].hp
    assert hp_after_play == starting_hp - 6

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.turn == 2
    assert state.combat.monsters[0].hp == hp_after_play - 6
    assert any(
        event.kind == "card_extra_played"
        and event.metadata.get("relic_id") == "history_course"
        for event in state.combat.last_events
    )
    assert all(event.kind != "combat_relic_marker_stubbed" for event in state.combat.last_events)


def test_war_hammer_upgrades_four_deck_cards_after_elite_kill() -> None:
    state = _enter_combat(
        (
            {
                "id": "elite_kill",
                "name": "Elite Kill",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 999,
            },
            _card_spec("upgrade_candidate_a"),
            _card_spec("upgrade_candidate_b"),
            _card_spec("upgrade_candidate_c"),
            _card_spec("upgrade_candidate_d"),
        ),
        relics=("war_hammer",),
        room_kind=RoomKind.ELITE,
        draw_per_turn=5,
    )

    state = _play_card(state, "elite_kill")

    assert state.phase == RunPhase.REWARD
    assert state.combat is not None
    assert sum(card.upgraded for card in state.master_deck) == 4
    assert any(
        event.kind == "relic_deck_card_upgraded" and event.source_id == "war_hammer"
        for event in state.combat.last_events
    )


def test_paels_tooth_adds_upgraded_removed_card_after_combat() -> None:
    state = _enter_combat(
        (
            {
                "id": "tooth_kill",
                "name": "Tooth Kill",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 999,
            },
        ),
        relics=("paels_tooth",),
        flags={"paels_tooth_removed_card_ids": ("defend",)},
        draw_per_turn=1,
    )

    state = _play_card(state, "tooth_kill")

    assert state.phase == RunPhase.REWARD
    assert state.combat is not None
    assert any(
        card.card_id == "defend" and card.upgraded
        for card in state.master_deck
    )
    assert any(
        event.kind == "relic_deck_card_added" and event.source_id == "paels_tooth"
        for event in state.combat.last_events
    )


def test_black_star_adds_second_elite_reward_relic_through_full_combat() -> None:
    state = _enter_combat(
        (
            {
                "id": "black_star_kill",
                "name": "Black Star Kill",
                "type": "Attack",
                "target": "Enemy",
                "cost": 0,
                "damage": 999,
            },
        ),
        relics=("black_star",),
        room_kind=RoomKind.ELITE,
        draw_per_turn=1,
    )

    state = _play_card(state, "black_star_kill")

    assert state.phase == RunPhase.REWARD
    assert state.reward is not None
    assert state.reward.metadata["encounter"] == "elite"
    assert len(state.reward.relic_ids) == 2


def test_lizard_tail_damage_taken_relic_hook_prevents_fatal_damage_once() -> None:
    state = _enter_combat((), relics=("lizard_tail",), draw_per_turn=0)
    assert state.combat is not None
    attacker = MonsterState(
        monster_id="fatal_attacker",
        name="Fatal Attacker",
        hp=30,
        max_hp=30,
        intent="attack",
        intent_damage=50,
    )
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (),
                    "monsters": (attacker,),
                    "player": state.combat.player.model_copy(update={"hp": 10, "block": 0}),
                }
            )
        }
    )

    state = _end_turn(state)

    assert state.combat is not None
    assert state.combat.player.hp == 40
    assert state.combat.metadata["relic_counters"]["lizard_tail"] == 1
    assert any(
        event.kind == "player_healed"
        and event.source_id == "lizard_tail"
        and event.metadata["condition"] == "fatal_damage"
        for event in state.combat.last_events
    )
