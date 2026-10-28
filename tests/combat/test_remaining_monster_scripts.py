from __future__ import annotations

from test_named_monster_scripts import _combat

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import RunPhase
from sts2sim.engine.transitions import _card_from_spec
from sts2sim.mechanics.monsters import build_encounter_definitions


def _play(state, *, card_type="attack", effects=None, target=None):
    card = _card_from_spec(
        {
            "card_id": "test_play",
            "instance_id": "test_play",
            "type": card_type,
            "target": "enemy" if card_type == "attack" else "self",
            "cost": 0,
            "effects": effects or {"damage": 1},
        },
        1,
    )
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"hand": (card,)})})
    action = next(
        a
        for a in legal_actions(state)
        if a.type == "play_card" and (target is None or a.target_id == target)
    )
    return step(state, action)


def _end(state):
    return step(state, next(a for a in legal_actions(state) if a.type == "end_turn"))


def test_dummy_timeout_ends_room_without_reward_or_victory_effects() -> None:
    state = _combat(
        "BATTLE_FRIEND_V1",
        statuses=(("BATTLEWORN_DUMMY_TIME_LIMIT", 3),),
        moves=({"id": "NOTHING", "intent": "Sleep"},),
    )
    state = state.model_copy(update={"relics": ("burning_blood", "fishing_rod")})
    before = state.player.hp
    state = _end(_end(state))
    assert state.combat.monsters[0].statuses["battleworn_dummy_time_limit"] == 1
    state = load_state(serialize(state))
    state = _end(state)
    assert state.phase is RunPhase.MAP
    assert state.reward is None
    assert state.combat is None
    assert state.player.hp == before
    assert state.map.current_node_id in state.map.completed_node_ids


def test_dummy_killed_before_timeout_generates_normal_victory() -> None:
    state = _combat(
        "BATTLE_FRIEND_V1",
        statuses=(("BATTLEWORN_DUMMY_TIME_LIMIT", 3),),
        moves=({"id": "NOTHING", "intent": "Sleep"},),
    )
    state = _end(_end(state))
    state = _play(state, effects={"damage": 999})
    assert state.phase is RunPhase.REWARD
    assert state.reward is not None


def test_slumber_wakes_after_three_hp_loss_hits_and_removes_plating() -> None:
    state = _combat(
        "SLUMBERING_BEETLE",
        statuses=(("SLUMBER", 3), ("PLATING", 15)),
        moves=(
            {"id": "SNORE", "intent": "Sleep"},
            {
                "id": "ROLL_OUT",
                "intent": "Attack + Buff",
                "damage": {"normal": 16},
                "powers": ({"power_id": "STRENGTH", "amount": 2},),
            },
        ),
    )
    # Three unblocked HP losses, after the source opening Plating Block.
    combat = state.combat
    state = state.model_copy(update={"combat": combat.model_copy(update={
        "monsters": (combat.monsters[0].model_copy(update={"block": 0}),),
    })})
    state = _play(state, effects={"sequence": ({"damage": 1},) * 3})
    beetle = state.combat.monsters[0]
    assert beetle.hp == 97
    assert "slumber" not in beetle.statuses
    assert "plating" not in beetle.statuses
    assert beetle.move_id == "ROLL_OUT"
    assert beetle.intent_damage == 16


def test_blocked_hits_do_not_reduce_slumber_and_three_turns_wake_without_attack() -> None:
    state = _combat(
        "SLUMBERING_BEETLE",
        statuses=(("SLUMBER", 3), ("PLATING", 15)),
        moves=(
            {"id": "SNORE", "intent": "Sleep"},
            {"id": "ROLL_OUT", "intent": "Attack", "damage": {"normal": 16}},
        ),
    )
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "monsters": (state.combat.monsters[0].model_copy(update={"block": 30}),),
                }
            )
        }
    )
    state = _play(state, effects={"sequence": ({"damage": 1},) * 3})
    assert state.combat.monsters[0].statuses["slumber"] == 3
    before = state.combat.player.hp
    state = _end(_end(_end(state)))
    assert state.combat.player.hp == before
    assert state.combat.monsters[0].move_id == "ROLL_OUT"
    assert "slumber" not in state.combat.monsters[0].statuses
    assert "plating" not in state.combat.monsters[0].statuses
    state = _end(state)
    assert state.combat.player.hp == before - 16


def test_galvanic_damage_follows_power_effects_and_uses_new_block() -> None:
    state = _combat("GLOBE_HEAD", statuses=(("GALVANIC", 6),))
    before = state.combat.player.hp
    state = _play(state, card_type="power", effects={"block": 4})
    assert state.combat.player.hp == before - 2
    assert state.combat.player.block == 0
    assert any(e.metadata.get("trigger") == "galvanic" for e in state.combat.last_events)


def test_galvanic_consumes_buffer_from_the_power_just_played() -> None:
    state = _combat("GLOBE_HEAD", statuses=(("GALVANIC", 6),))
    before = state.combat.player.hp
    state = _play(
        state, card_type="power", effects={"apply_status": {"target": "self", "buffer": 1}}
    )
    assert state.combat.player.hp == before
    assert "buffer" not in state.combat.player.statuses
    assert any(e.kind == "buffer_prevented_hp_loss" for e in state.combat.last_events)


def _with_child_definitions(state, ids):
    children = tuple(
        {
            "id": child_id,
            "name": child_id,
            "min_hp": 10,
            "max_hp": 10,
            "moves": (
                {"id": "SPAWNED", "intent": "Stun"},
                {"id": "ATTACK", "intent": "Attack", "damage": {"normal": 6}},
            ),
            "attack_pattern": {
                "initial_move": "SPAWNED",
                "states": (
                    {"id": "SPAWNED_MOVE", "move_id": "SPAWNED", "next": "ATTACK_MOVE"},
                    {"id": "ATTACK_MOVE", "move_id": "ATTACK", "next": "ATTACK_MOVE"},
                ),
            },
        }
        for child_id in ids
    )
    return state.model_copy(
        update={
            "flags": {
                **state.flags,
                "monsters": tuple(state.flags["monsters"]) + children,
            }
        }
    )


def test_infested_death_spawns_four_wrigglers_with_one_inactive_turn() -> None:
    state = _combat("PHROG_PARASITE", statuses=(("INFESTED", 4),))
    state = _with_child_definitions(state, ("WRIGGLER",))
    state = _play(state, effects={"damage": 999})
    assert state.phase is RunPhase.COMBAT
    living = [m for m in state.combat.monsters if m.hp > 0]
    assert len(living) == 4
    assert all(m.metadata["source_monster_id"] == "WRIGGLER" for m in living)
    assert all(m.move_id == "SPAWNED" for m in living)
    before = state.combat.player.hp
    state = load_state(serialize(state))
    state = _end(state)
    assert state.combat.player.hp == before
    state = _end(state)
    assert state.combat.player.hp == before - 24
    assert len(state.combat.monsters) == 5


def test_surprise_death_transfers_stolen_gold_to_fat_gremlin_and_heist_returns_it() -> None:
    state = _combat("GREMLIN_MERC", statuses=(("SURPRISE", 1),))
    state = _with_child_definitions(state, ("FAT_GREMLIN", "SNEAKY_GREMLIN"))
    player = state.combat.player.model_copy(update={"gold": 60})
    state = state.model_copy(
        update={
            "player": player,
            "combat": state.combat.model_copy(
                update={"player": player},
            ),
        }
    )
    state = _end(state)
    assert state.combat.player.gold == 40
    state = _play(state, effects={"damage": 999})
    living = [m for m in state.combat.monsters if m.hp > 0]
    assert len(living) == 2
    fat = next(m for m in living if m.metadata["source_monster_id"] == "FAT_GREMLIN")
    assert fat.metadata["stolen_gold"] == 20
    state = _play(state, effects={"damage": 999}, target=fat.monster_id)
    assert state.combat.player.gold == 60
    assert sum(m.hp > 0 for m in state.combat.monsters) == 1


def test_encounter_participant_lists_exclude_death_only_summons_at_opening() -> None:
    encounters = build_encounter_definitions(
        (
            {
                "id": "GREMLIN_MERC_NORMAL",
                "monsters": tuple(
                    {"id": i}
                    for i in (
                        "FAT_GREMLIN",
                        "GREMLIN_MERC",
                        "SNEAKY_GREMLIN",
                    )
                ),
            },
            {
                "id": "PHROG_PARASITE_ELITE",
                "monsters": (
                    {"id": "PHROG_PARASITE"},
                    {"id": "WRIGGLER"},
                ),
            },
        )
    )
    assert encounters[0].monster_ids == ("GREMLIN_MERC",)
    assert encounters[1].monster_ids == ("PHROG_PARASITE",)


def _crab_pair():
    state = _combat("CRUSHER", statuses=(("BACK_ATTACK_LEFT", 1),))
    left = state.combat.monsters[0]
    right = left.model_copy(update={
        "monster_id": "right_claw", "name": "Rocket", "statuses": {"back_attack_right": 1},
    })
    return state.model_copy(update={"combat": state.combat.model_copy(update={
        "monsters": (left, right), "metadata": {**state.combat.metadata, "player_facing": "left"},
    })})


def test_targeted_card_turns_player_and_back_attack_scales_each_hit() -> None:
    state = _crab_pair()
    state = _play(state, target="right_claw")
    assert state.combat.metadata["player_facing"] == "right"
    assert state.combat.monsters[0].intent_damage == 7 * 3
    assert state.combat.monsters[1].intent_damage == 5 * 3
    before = state.combat.player.hp
    state = _end(state)
    assert state.combat.player.hp == before - (7 + 5) * 3


def test_all_enemy_card_does_not_turn_but_targeted_potion_does() -> None:
    state = _crab_pair()
    card = _card_from_spec({
        "card_id": "aoe", "instance_id": "aoe", "type": "attack", "cost": 0,
        "target": "all_enemies", "effects": {"all_damage": 1},
    }, 1)
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"hand": (card,)})})
    state = step(state, next(a for a in legal_actions(state) if a.type == "play_card"))
    assert state.combat.metadata["player_facing"] == "left"
    state = state.model_copy(update={"potions": ("fire_potion",)})
    state = step(state, next(a for a in legal_actions(state)
                            if a.type == "use_potion" and a.target_id == "right_claw"))
    assert state.combat.metadata["player_facing"] == "right"


def test_back_attack_ends_when_one_claw_dies_and_direction_survives_reload() -> None:
    state = _crab_pair()
    state = load_state(serialize(state))
    assert state.combat.metadata["player_facing"] == "left"
    state = _play(state, target="right_claw", effects={"damage": 999})
    assert state.combat.monsters[0].intent_damage == 5 * 3
    before = state.combat.player.hp
    state = _end(state)
    assert state.combat.player.hp == before - 15
