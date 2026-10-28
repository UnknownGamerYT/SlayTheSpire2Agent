from __future__ import annotations

import pytest

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine import CardInstance, CombatState, MonsterState, PlayerState, RunState
from sts2sim.engine.rng import new_rng_state
from sts2sim.engine.transitions import _energy_cost, _modified_monster_attack_damage


def _card(name: str, kind: str, effects: dict, *, cost: int = 0) -> CardInstance:
    return CardInstance(
        instance_id=name,
        card_id=name,
        type=kind,
        target="enemy" if kind == "attack" else "self",
        cost=cost,
        effects=effects,
        custom={"retain": True},
    )


def _state(power: str | None = None, *, artifact: bool = False) -> RunState:
    source_ids = {"smoggy": "LIVING_FOG", "tangled": "VINE_SHAMBLER", "tender": "HUNTER_KILLER"}
    source_id = source_ids.get(power, "STATUS_WAITER")
    opening_move = "APPLY" if power else "WAIT"
    monster_source = {
        "id": source_id,
        "name": source_id,
        "type": "Normal",
        "min_hp": 500,
        "max_hp": 500,
        "moves": [
            {
                "id": "APPLY",
                "name": "Apply",
                "intent": "Debuff",
                "powers": [
                    {
                        "power_id": power.upper() if power else "SMOGGY",
                        "amount": 1,
                        "target": "player",
                    }
                ],
            },
            {"id": "WAIT", "name": "Wait", "intent": "Unknown"},
        ],
        "attack_pattern": {
            "initial_move": opening_move,
            "states": [
                {"id": "APPLY_STATE", "move_id": "APPLY", "type": "move", "next": "WAIT_STATE"},
                {"id": "WAIT_STATE", "move_id": "WAIT", "type": "move", "next": "WAIT_STATE"},
            ],
        },
    }
    cards = (
        _card("attack:1", "attack", {"damage": 6}, cost=1),
        _card("attack:2", "attack", {"damage": 6}, cost=1),
        _card("skill:1", "skill", {"block": 5}),
        _card("skill:2", "skill", {"block": 5}),
        _card("power:1", "power", {"energy": 1}),
    )
    player = PlayerState(
        hp=200,
        max_hp=200,
        energy=20,
        max_energy=20,
        statuses={"artifact": 1} if artifact else {},
    )
    combat = CombatState(
        player=player,
        hand=cards,
        draw_per_turn=0,
        monsters=(
            MonsterState(
                monster_id=source_id,
                hp=500,
                max_hp=500,
                move_id=opening_move,
                metadata={"source_monster_id": source_id},
            ),
        ),
    )
    return RunState(
        seed=8,
        character_id="TEST",
        rng=new_rng_state(8),
        player=player,
        master_deck=cards,
        combat=combat,
        flags={"monsters": (monster_source,)},
    )


def _end(state: RunState) -> RunState:
    return step(state, next(action for action in legal_actions(state) if action.type == "end_turn"))


def _play(state: RunState, instance_id: str) -> RunState:
    return step(
        state,
        next(
            action
            for action in legal_actions(state)
            if action.type == "play_card" and action.card_instance_id == instance_id
        ),
    )


def test_smoggy_limits_skills_only_and_resets_each_turn_after_real_monster_application() -> None:
    state = _end(_state("smoggy"))
    assert state.player.statuses["smoggy"] == 1

    state = _play(state, "skill:1")
    playable = {
        action.card_instance_id for action in legal_actions(state) if action.type == "play_card"
    }

    assert "skill:2" not in playable
    assert {"attack:1", "power:1"} <= playable
    state = _end(state)
    assert any(action.card_instance_id == "skill:2" for action in legal_actions(state))
    assert state.player.statuses["smoggy"] == 1


def test_smoggy_reserves_current_skill_before_a_nested_automatic_skill_play() -> None:
    state = _end(_state("smoggy"))
    assert state.combat is not None
    first = next(card for card in state.combat.hand if card.instance_id == "skill:1")
    second = next(card for card in state.combat.hand if card.instance_id == "skill:2")
    first = first.model_copy(update={"effects": {"block": 5, "play_top_card": 1}})
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(
                update={
                    "hand": (first,),
                    "draw_pile": (second,),
                }
            )
        }
    )

    result = _play(state, "skill:1")

    assert result.player.block == 5
    assert result.combat is not None
    assert result.combat.metadata["skills_played_this_turn"] == 1
    assert any(card.instance_id == "skill:2" for card in result.combat.discard_pile)
    assert any(
        event.kind == "card_play_blocked" and event.source_id == "smoggy"
        for event in result.replay_log[-1].events
    )


def test_tangled_adds_attack_energy_for_two_complete_player_turns_and_replays() -> None:
    state = _state("tangled")
    assert state.combat is not None
    player = state.player.model_copy(update={"energy": 1, "max_energy": 1})
    state = state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )
    state = _end(state)
    assert state.combat is not None
    assert state.player.statuses["tangled"] == 1
    assert state.combat.metadata["tangled_turns_remaining"] == 2
    assert not any(action.card_instance_id == "attack:1" for action in legal_actions(state))
    assert any(action.card_instance_id == "skill:1" for action in legal_actions(state))
    restored = load_state(serialize(state))

    first = _end(state)
    replayed = _end(restored)

    assert serialize(first) == serialize(replayed)
    assert first.combat is not None
    assert first.combat.metadata["tangled_turns_remaining"] == 1
    assert not any(action.card_instance_id == "attack:1" for action in legal_actions(first))
    final = _end(first)
    assert "tangled" not in final.player.statuses
    assert any(action.card_instance_id == "attack:1" for action in legal_actions(final))


def test_tangled_does_not_override_explicit_free_to_play_cards() -> None:
    state = _end(_state("tangled"))
    assert state.combat is not None
    free = _card("free", "attack", {"damage": 6}, cost=1).model_copy(
        update={"custom": {"free_to_play_this_turn": True}}
    )
    assert _energy_cost(free, 1, combat=state.combat) == 0
    assert _energy_cost(state.combat.hand[0], 20, combat=state.combat) == 2


def test_artifact_blocked_tangled_reapplication_does_not_refresh_its_existing_duration() -> None:
    state = _state("tangled", artifact=True)
    assert state.combat is not None
    player = state.player.model_copy(update={"statuses": {"artifact": 1, "tangled": 1}})
    state = state.model_copy(
        update={
            "player": player,
            "combat": state.combat.model_copy(
                update={
                    "player": player,
                    "metadata": {"tangled_turns_remaining": 2},
                }
            ),
        }
    )

    state = _end(state)

    assert state.combat is not None
    assert state.combat.metadata["tangled_turns_remaining"] == 1
    assert state.player.statuses["tangled"] == 1
    assert "artifact" not in state.player.statuses


def test_tender_reduces_following_attack_and_block_and_restores_temporary_losses() -> None:
    state = _end(_state("tender"))
    assert state.combat is not None
    player = state.player.model_copy(
        update={"statuses": {"tender": 1, "strength": 2, "dexterity": 3}}
    )
    state = state.model_copy(
        update={"player": player, "combat": state.combat.model_copy(update={"player": player})}
    )
    original_hp = state.combat.monsters[0].hp

    state = _play(state, "attack:1")
    assert state.combat is not None
    assert state.combat.monsters[0].hp == original_hp - 8
    assert state.player.statuses["temporary_strength"] == -1
    state = _play(state, "skill:1")
    assert state.player.block == 7  # 5 + 3 Dexterity - 1 temporary Dexterity.
    state = _play(state, "attack:2")
    assert state.combat is not None
    assert state.combat.monsters[0].hp == original_hp - 8 - 6
    assert state.player.statuses["temporary_strength"] == -3
    assert state.player.statuses["temporary_dexterity"] == -3

    state = _end(state)

    assert state.player.statuses["strength"] == 2
    assert state.player.statuses["dexterity"] == 3
    assert state.player.statuses["tender"] == 1
    assert "temporary_strength" not in state.player.statuses
    assert "temporary_dexterity" not in state.player.statuses


@pytest.mark.parametrize("power", ["smoggy", "tangled", "tender"])
def test_artifact_blocks_the_actual_monster_debuff_application(power: str) -> None:
    state = _end(_state(power, artifact=True))
    assert power not in state.player.statuses
    assert "artifact" not in state.player.statuses


def test_monster_vigor_is_per_hit_before_weak_and_is_consumed_after_one_attack() -> None:
    state = _state()
    assert state.combat is not None
    enemy = state.combat.monsters[0].model_copy(
        update={
            "monster_id": "fallback",
            "move_id": None,
            "intent": "Attack",
            "intent_damage": 10,
            "hit_count": 2,
            "statuses": {"vigor": 6},
            "metadata": {},
        }
    )
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (enemy,)})}
    )

    state = _end(state)

    assert state.player.hp == 178  # (5 + 6) x 2 hits.
    assert state.combat is not None
    assert "vigor" not in state.combat.monsters[0].statuses
    assert (
        sum(
            event.metadata.get("status") == "vigor" and event.kind == "status_consumed"
            for event in state.replay_log[-1].events
        )
        == 1
    )
    state = _end(state)
    assert state.player.hp == 168
    assert _modified_monster_attack_damage(10, statuses={"vigor": 6, "weak": 1}) == 12


def test_real_thrashing_attack_consumes_old_vigor_before_gaining_the_next_stack() -> None:
    state = _state()
    assert state.combat is not None
    source = {
        "id": "TERROR_EEL",
        "name": "Terror Eel",
        "type": "Normal",
        "min_hp": 500,
        "max_hp": 500,
        "moves": [
            {
                "id": "THRASH",
                "name": "Thrash",
                "intent": "Attack + Buff",
                "damage": {"normal": 3, "ascension": 4, "hit_count": 3},
                "powers": [{"amount": 6, "power_id": "VIGOR", "target": "self"}],
            },
            {
                "id": "CRASH",
                "name": "Crash",
                "intent": "Attack",
                "damage": {"normal": 16, "ascension": 18},
            },
        ],
        "attack_pattern": {
            "initial_move": "THRASH",
            "states": [
                {"id": "THRASH_STATE", "type": "move", "move_id": "THRASH", "next": "CRASH_STATE"},
                {"id": "CRASH_STATE", "type": "move", "move_id": "CRASH", "next": "CRASH_STATE"},
            ],
        },
    }
    enemy = state.combat.monsters[0].model_copy(
        update={
            "monster_id": "TERROR_EEL",
            "move_id": "THRASH",
            "statuses": {"vigor": 6},
            "metadata": {"source_monster_id": "TERROR_EEL"},
        }
    )
    state = state.model_copy(
        update={
            "combat": state.combat.model_copy(update={"monsters": (enemy,)}),
            "flags": {"monsters": (source,)},
        }
    )

    state = _end(state)

    assert state.player.hp == 173  # Old Vigor: (3 + 6) x 3.
    assert state.combat is not None
    assert state.combat.monsters[0].statuses["vigor"] == 6  # The new Thrash stack remains.
    assert state.combat.monsters[0].intent_damage == 22
    state = _end(state)
    assert state.player.hp == 151  # Crash uses that stack once.
    assert "vigor" not in state.combat.monsters[0].statuses
