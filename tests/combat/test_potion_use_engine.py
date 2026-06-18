from __future__ import annotations

from sts2sim import legal_actions, new_run, step
from sts2sim.engine import MapEdgeState, MapNodeState, MapState, RoomKind, RunPhase


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
