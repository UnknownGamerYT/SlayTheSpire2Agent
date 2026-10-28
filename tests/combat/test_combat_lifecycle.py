from __future__ import annotations

import json

import pytest
from combat.test_fishing_rod import _run

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import (
    CardInstance,
    MapEdgeState,
    MapNodeState,
    MapState,
    MonsterState,
    RewardState,
    RoomKind,
    RunPhase,
)
from sts2sim.engine.models import PendingChoiceState
from sts2sim.engine.transitions import _state_after_combat


def _two_fights(*, relics: tuple[str, ...] = ()):
    state = _run(relics=relics)
    finisher = state.master_deck[0].model_copy(update={"effects": {"damage": 999, "max_hp": 3}})
    setup = CardInstance(
        instance_id="setup:1",
        card_id="setup",
        type="skill",
        target="self",
        cost=0,
        effects={
            "sequence": (
                {
                    "apply_status": {
                        "strength": 3,
                        "dexterity": 2,
                        "weak": 2,
                        "poison": 4,
                        "frail": 2,
                        "focus": 2,
                        "vulnerable": 2,
                        "ritual": 1,
                        "next_turn_energy": 4,
                        "metallicize": 2,
                    }
                },
                {"player_resource": {"resource": "star", "amount": 4}},
                {"player_resource": {"resource": "forge", "amount": 2}},
                {"player_resource": {"resource": "summon", "amount": 2}},
                {"player_resource": {"resource": "mantra", "amount": 3}},
                {"player_resource": {"resource": "sandpit", "amount": 2}},
            )
        },
    )
    nodes = (
        MapNodeState(node_id="start", act=1, floor=0, lane=0, kind=RoomKind.START),
        MapNodeState(node_id="first", act=1, floor=1, lane=0, kind=RoomKind.MONSTER),
        MapNodeState(node_id="second", act=1, floor=2, lane=0, kind=RoomKind.MONSTER),
    )
    return state.model_copy(
        update={
            "phase": RunPhase.MAP,
            "player": state.player.model_copy(update={"hp": 70, "max_energy": 4}),
            "master_deck": (setup, finisher, *state.master_deck[1:]),
            "map": MapState(
                nodes=nodes,
                current_node_id="start",
                completed_node_ids=("start",),
                edges=(
                    MapEdgeState(from_id="start", to_id="first"),
                    MapEdgeState(from_id="first", to_id="second"),
                ),
            ),
        }
    )


def _action(state, kind: str, *, card_id: str | None = None):
    return next(
        action
        for action in legal_actions(state)
        if action.type == kind and (card_id is None or action.card_instance_id == card_id)
    )


def _enter(state):
    return step(state, _action(state, "choose_node"))


def _setup(state):
    return step(state, _action(state, "play_card", card_id="setup:1"))


def _finish(state):
    return step(state, _action(state, "play_card", card_id="finisher:1"))


def test_completed_fight_clears_run_combat_state_and_preserves_final_diagnostics() -> None:
    state = _setup(_enter(_two_fights()))
    assert state.player.statuses["strength"] == 3
    assert state.player.statuses["poison"] == 4
    assert state.player.resources["star"] == 4
    assert state.player.resources["mantra"] == 3
    state = _finish(state)

    assert state.phase == RunPhase.REWARD
    assert state.player.statuses == {}
    assert state.player.resources == {}
    assert state.player.block == 0
    assert state.player.energy == state.player.max_energy == 4
    assert state.combat.player.statuses["strength"] == 3
    assert state.combat.player.statuses["poison"] == 4
    assert state.combat.player.resources["star"] == 4

    state = step(state, _action(state, "take_reward_gold"))
    assert state.player.gold == 10
    assert state.combat.player.statuses["strength"] == 3
    assert state.combat.player.gold == 0  # Final combat diagnostics stay unchanged.
    state = load_state(json.loads(json.dumps(serialize(state))))
    state = step(state, _action(state, "proceed"))
    state = _enter(state)

    assert state.phase == RunPhase.COMBAT
    assert state.player.statuses == {}
    assert state.player.resources == {}
    assert state.player.hp == 73
    assert state.player.max_hp == 83
    assert state.player.gold == 10
    assert state.player.max_energy == 4
    assert state.player.energy == 4
    assert len(state.master_deck) == 6


def test_starting_relic_powers_and_regent_resources_are_applied_once_each_fight() -> None:
    relics = (
        "vajra",
        "oddly_smooth_stone",
        "divine_right",
        "fencing_manual",
        "anchor",
        "lantern",
        "fishing_rod",
    )
    state = _enter(_two_fights(relics=relics))
    assert state.player.statuses["strength"] == state.player.statuses["dexterity"] == 1
    assert state.player.resources["star"] == 3
    assert state.player.resources["forge"] == 10
    assert state.player.block == 10
    assert state.player.energy == 5
    state = _finish(_setup(state))
    assert state.flags["relic_counters"]["fishing_rod"] == 1
    state = step(state, _action(state, "proceed"))
    state = _enter(state)

    assert state.player.statuses == {"strength": 1, "dexterity": 1}
    assert state.player.resources.get("star") == 3
    assert state.player.resources.get("forge") == 10
    assert not state.player.resources.get("summon")
    assert not state.player.resources.get("mantra")
    assert state.player.block == 10
    assert state.player.energy == 5
    assert state.combat.metadata["relic_counters"]["fishing_rod"] == 1


def test_failed_combat_cleans_only_run_snapshot_and_does_not_remove_diagnostics() -> None:
    state = _setup(_enter(_two_fights()))
    player = state.player.model_copy(update={"hp": 1, "block": 0})
    monster = MonsterState(monster_id="fatal", hp=99, max_hp=99, intent="attack", intent_damage=999)
    state = state.model_copy(
        update={
            "player": player,
            "combat": state.combat.model_copy(update={"player": player, "monsters": (monster,)}),
        }
    )
    state = step(state, _action(state, "end_turn"))

    assert state.phase == RunPhase.FAILED
    assert state.player.hp == 0
    assert state.player.statuses == state.player.resources == {}
    assert state.combat.player.statuses
    assert state.combat.player.resources


def test_fairy_revival_keeps_combat_powers_and_resources() -> None:
    state = _setup(_enter(_two_fights()))
    player = state.player.model_copy(update={"hp": 1, "block": 0})
    # Poison is removed for this test so only the intended enemy hit is lethal.
    player = player.model_copy(update={"statuses": {"strength": 3, "dexterity": 2}})
    monster = MonsterState(monster_id="fatal", hp=99, max_hp=99, intent="attack", intent_damage=999)
    state = state.model_copy(
        update={
            "player": player,
            "potions": ("fairy_in_a_bottle",),
            "combat": state.combat.model_copy(update={"player": player, "monsters": (monster,)}),
        }
    )
    state = step(state, _action(state, "end_turn"))

    assert state.phase == RunPhase.COMBAT
    assert state.player.hp == 24
    assert state.player.statuses["strength"] == 3
    assert state.player.statuses["dexterity"] == 2
    assert state.player.resources["star"] == 4
    assert not state.potions


@pytest.mark.parametrize("pending", ["choice", "replay"])
def test_terminal_looking_state_is_not_cleaned_while_resolution_is_pending(pending: str) -> None:
    state = _setup(_enter(_two_fights()))
    combat = state.combat.model_copy(update={"monsters": ()})
    if pending == "choice":
        combat = combat.model_copy(
            update={
                "pending_choices": (
                    PendingChoiceState(
                        choice_id="choice", kind="discard", candidate_ids=("finisher:1",)
                    ),
                )
            }
        )
    else:
        combat = combat.model_copy(
            update={
                "metadata": {
                    **combat.metadata,
                    "card_replay_queue": ({"pending": True},),
                }
            }
        )
    next_state = _state_after_combat(state, combat, state.rng)
    assert next_state.player.statuses == combat.player.statuses
    assert next_state.player.resources == combat.player.resources


def test_legacy_reward_room_exit_cleans_old_combat_fields_before_next_fight() -> None:
    state = _finish(_setup(_enter(_two_fights())))
    state = state.model_copy(update={"player": state.combat.player})
    state = load_state(serialize(state))
    state = step(state, _action(state, "proceed"))
    assert state.player.statuses == state.player.resources == {}
    state = _enter(state)
    assert state.player.statuses == state.player.resources == {}


def test_gold_max_hp_passive_cannot_revive_a_failed_combat() -> None:
    state = _enter(_two_fights(relics=("dragon_fruit",)))
    source = CardInstance(
        instance_id="doomed",
        card_id="doomed",
        type="attack",
        target="enemy",
        cost=0,
        effects={"damage": 999, "hp_loss": 999},
    )
    # HeistPower.BeforeDeath queues a room GoldReward before the same card
    # kills the player. CombatManager.CheckWinCondition prioritizes pending
    # loss over victory, so the deferred reward is never offered or claimed.
    monster = MonsterState(
        monster_id="gremlin",
        hp=1,
        max_hp=1,
        metadata={"source_monster_id": "FAT_GREMLIN", "stolen_gold": 5},
    )
    combat = state.combat.model_copy(update={"hand": (source,), "monsters": (monster,)})
    state = state.model_copy(update={"combat": combat})
    gold_before, max_hp_before = state.player.gold, state.player.max_hp
    state = step(state, _action(state, "play_card", card_id="doomed"))

    assert state.phase == RunPhase.FAILED
    assert state.player.gold == gold_before
    assert state.player.max_hp == max_hp_before
    assert state.player.hp == state.combat.player.hp == 0
    assert state.combat.metadata["pending_gold_rewards"] == (
        {"amount": 5, "source_id": "heist", "was_gold_stolen_back": True},)
    assert state.reward is None
    assert not any(event.kind in {"gold_gained", "reward_gold_taken"}
                   or event.source_id == "dragon_fruit"
                   for event in state.replay_log[-1].events)
    restored = load_state(serialize(state))
    assert serialize(restored) == serialize(state)
    assert not legal_actions(restored)
    from sts2sim.engine import Action, ActionType
    from sts2sim.engine.errors import IllegalActionError
    with pytest.raises(IllegalActionError):
        step(restored, Action(type=ActionType.TAKE_REWARD_GOLD, target_id="reward:gold:0"))


def test_dummy_timeout_also_ends_combat_scoped_player_state() -> None:
    from combat.test_named_monster_scripts import _combat

    state = _combat(
        "BATTLE_FRIEND_V1",
        statuses=(("BATTLEWORN_DUMMY_TIME_LIMIT", 3),),
        moves=({"id": "NOTHING", "intent": "Sleep"},),
    )
    player = state.player.model_copy(update={"statuses": {"strength": 3}, "resources": {"star": 4}})
    state = state.model_copy(
        update={
            "player": player,
            "combat": state.combat.model_copy(update={"player": player}),
        }
    )
    for _ in range(3):
        state = step(state, _action(state, "end_turn"))
    assert state.phase == RunPhase.MAP
    assert state.combat is None
    assert state.player.statuses == state.player.resources == {}


def test_scripted_enemy_respawn_keeps_the_current_combats_player_state() -> None:
    from combat.test_combat_engine_monsters import _enter_test_subject_combat, _execute_card

    state = _enter_test_subject_combat(deck=(_execute_card(damage=150),))
    player = state.player.model_copy(update={"statuses": {"strength": 3}, "resources": {"star": 4}})
    state = state.model_copy(
        update={
            "player": player,
            "combat": state.combat.model_copy(update={"player": player}),
        }
    )
    state = step(state, _action(state, "play_card"))
    assert state.phase == RunPhase.COMBAT
    assert state.combat.monsters[0].metadata["test_subject_respawns"] == 1
    assert state.player.statuses == {"strength": 3}
    assert state.player.resources == {"star": 4}


def test_post_combat_deck_pickup_preserves_final_combat_diagnostics() -> None:
    state = _finish(_setup(_enter(_two_fights(relics=("lucky_fysh",)))))
    state = state.model_copy(
        update={
            "reward": RewardState(reward_id="post-combat-card", card_options=("strike",)),
        }
    )
    state = step(state, _action(state, "take_reward_card"))
    assert state.player.statuses == state.player.resources == {}
    assert state.combat.player.statuses["strength"] == 3
    assert state.combat.player.resources["star"] == 4
    assert state.player.gold == 15
    assert len(state.master_deck) == 7
