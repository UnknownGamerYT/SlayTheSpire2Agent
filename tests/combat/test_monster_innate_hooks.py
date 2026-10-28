from __future__ import annotations

import pytest
from combat.test_combat_engine_monsters import (
    _end_turn_action,
    _enter_source_combat,
    _play_first_card,
    _source_data,
)

from sts2sim import legal_actions, load_state, serialize, step


def _monster(monster_id: str, powers: dict[str, int], *, damage: int = 5, hits: int = 1) -> dict:
    return {
        "id": monster_id,
        "name": monster_id,
        "type": "Normal",
        "min_hp": 100,
        "max_hp": 100,
        "innate_powers": tuple({"power_id": key, "amount": value} for key, value in powers.items()),
        "moves": (
            {
                "id": "HIT",
                "name": "Hit",
                "intent": "Attack",
                "damage": {"normal": damage, "hit_count": hits},
            },
            {"id": "DIZZY", "name": "Dizzy", "intent": "Stun"},
        ),
        "attack_pattern": {
            "initial_move": "HIT",
            "states": (
                {"id": "HIT", "type": "move", "move_id": "HIT", "next": "HIT"},
                {"id": "DIZZY", "type": "move", "move_id": "DIZZY", "next": "HIT"},
            ),
        },
    }


def _card(card_id: str, *, damage: int = 10, effects: dict | None = None) -> dict:
    return {
        "card_id": card_id,
        "type": "attack",
        "target": "enemy",
        "cost": 0,
        "effects": effects if effects is not None else {"damage": damage},
    }


def _combat(*monsters: dict, deck: tuple[dict, ...] = ()):
    encounter = {
        "id": "POWER_TEST",
        "name": "Power Test",
        "room_type": "Monster",
        "is_weak": True,
        "monsters": tuple({"id": monster["id"]} for monster in monsters),
    }
    return _enter_source_combat(_source_data(monsters, (encounter,), "POWER_TEST", deck=deck))


@pytest.mark.parametrize("power,amount", [("TERRITORIAL", 1), ("HIGH_VOLTAGE", 2)])
def test_end_turn_strength_updates_next_intent_and_survives_reload(power: str, amount: int) -> None:
    state = _combat(_monster("GROWING", {power: amount}))
    state = step(state, _end_turn_action(state))
    assert state.combat.monsters[0].statuses["strength"] == amount
    assert state.combat.monsters[0].intent_damage == 5 + amount
    restored = load_state(serialize(state))
    for candidate in (state, restored):
        next_state = step(candidate, _end_turn_action(candidate))
        assert next_state.combat.player.hp == 70 - amount
        assert next_state.combat.monsters[0].statuses["strength"] == amount * 2


def test_suck_counts_unblocked_hits_but_defers_strength_until_attack_finishes() -> None:
    state = _combat(_monster("STALKER", {"SUCK": 3}, damage=3, hits=2))
    state = step(state, _end_turn_action(state))
    assert state.combat.player.hp == 74  # Printed 3x2 is preserved throughout the attack.
    assert state.combat.monsters[0].statuses["strength"] == 6


def test_suck_counts_osty_damage_and_does_not_double_count_a_trampling_hit() -> None:
    state = _combat(_monster("STALKER", {"SUCK": 3}, damage=3, hits=2))
    combat = state.combat.model_copy(
        update={
            "metadata": {
                **state.combat.metadata,
                "osty": {"hp": 4, "max_hp": 4, "alive": True},
            }
        }
    )
    state = state.model_copy(update={"combat": combat})
    state = step(state, _end_turn_action(state))
    assert state.combat.player.hp == 78
    assert state.combat.monsters[0].statuses["strength"] == 6


def test_paper_cuts_counts_only_unblocked_hits_and_clamps_max_hp() -> None:
    state = _combat(_monster("SCROLL", {"PAPER_CUTS": 2}, damage=3, hits=2))
    player = state.combat.player.model_copy(update={"block": 3})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"player": player})})
    state = step(state, _end_turn_action(state))
    assert state.combat.player.hp == 77
    assert state.combat.player.max_hp == 78


@pytest.mark.parametrize("block,stunned", [(5, True), (4, False)])
def test_imbalanced_queues_a_visible_dizzy_move_only_when_attack_fully_blocked(
    block: int, stunned: bool
) -> None:
    state = _combat(_monster("BOWLBUG", {"IMBALANCED": 1}))
    player = state.combat.player.model_copy(update={"block": block})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"player": player})})
    state = step(state, _end_turn_action(state))
    assert state.combat.monsters[0].move_id == ("DIZZY" if stunned else "HIT")
    hp = state.combat.player.hp
    state = step(state, _end_turn_action(state))
    assert state.combat.player.hp == hp - (0 if stunned else 5)
    assert state.combat.monsters[0].move_id == "HIT"


def test_hard_to_kill_caps_incoming_damage_before_block_and_direct_hp_loss() -> None:
    state = _combat(
        _monster("SHELL", {"HARD_TO_KILL": 9}),
        deck=(
            _card("strike", damage=100),
            _card("loss", effects={"enemy_hp_loss": {"target": "enemy", "amount": 100}}),
        ),
    )
    monster = state.combat.monsters[0].model_copy(update={"block": 5})
    state = state.model_copy(
        update={"combat": state.combat.model_copy(update={"monsters": (monster,)})}
    )
    state = _play_first_card(state, "strike")
    assert state.combat.monsters[0].hp == 96
    assert state.combat.monsters[0].block == 0
    state = _play_first_card(state, "loss")
    assert state.combat.monsters[0].hp == 87


def test_hard_to_kill_caps_poison_and_thorns() -> None:
    state = _combat(_monster("SHELL", {"HARD_TO_KILL": 9, "POISON": 20}))
    player = state.combat.player.model_copy(update={"statuses": {"thorns": 30}})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"player": player})})
    state = step(state, _end_turn_action(state))
    assert state.combat.monsters[0].hp == 82
    assert state.combat.monsters[0].statuses["poison"] == 19


def test_hardened_shell_budget_is_shared_across_loss_paths_and_resets_next_turn() -> None:
    state = _combat(
        _monster("COLONY", {"HARDENED_SHELL": 20}),
        deck=(_card("one", damage=12), _card("two", damage=12), _card("three", damage=12)),
    )
    state = _play_first_card(state, "one")
    state = _play_first_card(state, "two")
    state = _play_first_card(state, "three")
    assert state.combat.monsters[0].hp == 80
    state = step(state, _end_turn_action(state))
    state = _play_first_card(state, "one")
    assert state.combat.monsters[0].hp == 68


def test_hardened_shell_resets_before_enemy_turn_poison_and_next_player_turn() -> None:
    state = _combat(
        _monster("COLONY", {"HARDENED_SHELL": 20, "POISON": 5}),
        deck=(_card("hit", damage=20),),
    )
    state = _play_first_card(state, "hit")
    assert state.combat.monsters[0].hp == 80
    state = step(state, _end_turn_action(state))
    assert state.combat.monsters[0].hp == 75
    state = _play_first_card(state, "hit")
    assert state.combat.monsters[0].hp == 55


def test_rampart_applies_on_first_and_later_player_turns_only_to_turret_operator() -> None:
    state = _combat(
        _monster("LIVING_SHIELD", {"RAMPART": 25}),
        _monster("TURRET_OPERATOR", {}),
        _monster("OTHER", {}),
    )
    assert [monster.block for monster in state.combat.monsters] == [0, 25, 0]
    state = step(state, _end_turn_action(state))
    assert [monster.block for monster in state.combat.monsters] == [0, 25, 0]


def test_slow_counts_cards_before_damage_resets_and_does_not_scale_nonattack_damage() -> None:
    state = _combat(
        _monster("EFFIGY", {"SLOW": 1}),
        deck=(
            {"card_id": "wait", "type": "skill", "target": "self", "cost": 0, "effects": {}},
            _card("hit", damage=10),
        ),
    )
    state = _play_first_card(state, "wait")
    state = _play_first_card(state, "hit")
    assert state.combat.monsters[0].hp == 88
    assert state.combat.monsters[0].metadata["slow_cards_played"] == 2
    state = step(state, _end_turn_action(state))
    state = _play_first_card(state, "hit")
    assert state.combat.monsters[0].hp == 77


def test_potions_do_not_trigger_slow_or_receive_its_attack_bonus() -> None:
    state = _combat(_monster("EFFIGY", {"SLOW": 1}), deck=(_card("hit", damage=10),))
    state = state.model_copy(update={"potions": ("fire_potion",)})
    action = next(action for action in legal_actions(state) if action.type == "use_potion")
    state = step(state, action)
    assert state.combat.monsters[0].hp == 80
    assert "slow_cards_played" not in state.combat.monsters[0].metadata
    state = _play_first_card(state, "hit")
    assert state.combat.monsters[0].hp == 69


def test_minions_abandon_when_last_leader_dies() -> None:
    state = _combat(
        _monster("LEADER", {}),
        _monster("MINION", {"MINION": 1}),
        deck=(_card("lethal", damage=100),),
    )
    state = _play_first_card(state, "lethal")
    assert state.phase.value == "reward"
    assert state.combat.monsters[1].metadata["abandoned_combat"] is True
    assert any(event.kind == "monster_escaped" for event in state.combat.last_events)


def test_crab_rage_responds_once_per_defeated_ally() -> None:
    state = _combat(
        _monster("ALLY", {}),
        _monster("CRAB", {"CRAB_RAGE": 1}),
        deck=(_card("lethal", damage=100),),
    )
    state = _play_first_card(state, "lethal")
    crab = state.combat.monsters[1]
    assert crab.statuses["strength"] == 6
    assert crab.block == 99
    assert crab.intent_damage == 11
    state = load_state(serialize(state))
    state = step(state, _end_turn_action(state))
    assert state.combat.monsters[1].statuses["strength"] == 6


def _possessor(monster_id: str) -> dict:
    stat, power, steal_move, attack_move = (
        ("strength", "POSSESS_STRENGTH", "DEBILITATING_SMOG", "EYE_LASERS")
        if monster_id == "THE_LOST"
        else ("dexterity", "POSSESS_SPEED", "MIASMA", "DREAD")
    )
    monster = _monster(monster_id, {power: 1})
    monster["moves"] = (
        {
            "id": steal_move,
            "intent": "Buff + Debuff",
            "powers": ({"power_id": stat.upper(), "target": "self", "amount": 2},),
        },
        {
            "id": attack_move,
            "intent": "Attack",
            "damage": ({"normal": 4, "hit_count": 2} if monster_id == "THE_LOST" else None),
        },
    )
    monster["attack_pattern"] = {
        "initial_move": steal_move,
        "states": (
            {"id": steal_move, "move_id": steal_move, "type": "move", "next": attack_move},
            {"id": attack_move, "move_id": attack_move, "type": "move", "next": steal_move},
        ),
    }
    return monster


@pytest.mark.parametrize(
    "monster_id,stat", [("THE_LOST", "strength"), ("THE_FORGOTTEN", "dexterity")]
)
def test_possess_restores_only_the_stat_actually_stolen_when_holder_dies(
    monster_id: str, stat: str
) -> None:
    state = _combat(_possessor(monster_id), deck=(_card("lethal", damage=1000),))
    state = step(state, _end_turn_action(state))
    assert state.combat.player.statuses[stat] == -2
    assert state.combat.monsters[0].statuses[stat] == 2
    state = load_state(serialize(state))
    state = _play_first_card(state, "lethal")
    assert state.combat.player.statuses.get(stat, 0) == 0
    assert any(event.metadata.get("returned_stolen_stat") for event in state.combat.last_events)


def test_artifact_prevents_stealing_and_does_not_create_a_later_return_bonus() -> None:
    state = _combat(_possessor("THE_LOST"), deck=(_card("lethal", damage=1000),))
    player = state.combat.player.model_copy(update={"statuses": {"artifact": 1}})
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"player": player})})
    state = step(state, _end_turn_action(state))
    assert state.combat.player.statuses.get("strength", 0) == 0
    assert not state.combat.monsters[0].metadata.get("possess_stolen_strength")
    state = _play_first_card(state, "lethal")
    assert state.combat.player.statuses.get("strength", 0) == 0


def test_possess_stat_stealing_stays_two_at_a9_and_forgotten_dread_adds_dexterity() -> None:
    # The source lists Dread without damage: the documented 13/15 base is restored.
    monster = _possessor("THE_FORGOTTEN")
    encounter = {"id": "POSSESS", "room_type": "Monster", "monsters": ({"id": monster["id"]},)}
    data = _source_data((monster,), (encounter,), "POSSESS")
    state = _enter_source_combat(data, ascension=9)
    state = step(state, _end_turn_action(state))
    assert state.combat.player.statuses["dexterity"] == -2
    assert state.combat.monsters[0].statuses["dexterity"] == 2
    assert state.combat.monsters[0].intent_damage == 17
    state = step(state, _end_turn_action(state))
    assert state.combat.player.hp == 63


def test_illusion_revives_on_next_own_turn_with_clean_debuffs_and_no_immediate_attack() -> None:
    state = _combat(
        _monster("PARAFRIGHT", {"ILLUSION": 1, "POISON": 5, "WEAK": 2, "STRENGTH": -3}),
        _monster("LEADER", {}),
        deck=(_card("lethal", damage=1000),),
    )
    state = _play_first_card(state, "lethal")
    illusion = state.combat.monsters[0]
    assert illusion.hp == 0 and illusion.move_id == "REVIVE"
    assert illusion.statuses == {"illusion": 1, "minion": 1}
    state = load_state(serialize(state))
    state = step(state, _end_turn_action(state))
    assert state.combat.monsters[0].hp == 100
    assert state.combat.player.hp == 75  # Only the leader attacked while the illusion revived.
    state = step(state, _end_turn_action(state))
    assert state.combat.player.hp == 65


def test_illusion_cannot_revive_after_the_leader_dies() -> None:
    state = _combat(
        _monster("LEADER", {}),
        _monster("PARAFRIGHT", {"ILLUSION": 1}),
        deck=(_card("lethal", damage=1000),),
    )
    state = _play_first_card(state, "lethal")
    assert state.phase.value == "reward"
    assert state.combat.monsters[1].hp == 0
    assert state.combat.monsters[1].metadata["abandoned_combat"] is True


def test_escape_artist_is_a_visual_timer_and_stops_at_one() -> None:
    state = _combat(_monster("TIMER", {"ESCAPE_ARTIST": 5}, damage=0))
    for remaining in (4, 3, 2, 1, 1):
        state = step(state, _end_turn_action(state))
        assert state.combat.monsters[0].statuses["escape_artist"] == remaining
        assert state.combat.monsters[0].hp > 0


def test_fogmog_opening_roster_does_not_spawn_its_later_illusion_twice() -> None:
    fogmog = _monster("FOGMOG", {})
    fogmog["moves"] = ({"id": "ILLUSION", "intent": "Summon"},)
    fogmog["attack_pattern"] = {
        "initial_move": "ILLUSION",
        "states": ({"id": "ILLUSION", "move_id": "ILLUSION", "type": "move", "next": "ILLUSION"},),
    }
    eye = _monster("EYE_WITH_TEETH", {"ILLUSION": 1})
    encounter = {
        "id": "FOGMOG_NORMAL",
        "room_type": "Monster",
        "monsters": (
            {"id": "FOGMOG"},
            {"id": "EYE_WITH_TEETH"},
        ),
    }
    data = _source_data((fogmog, eye), (encounter,), "FOGMOG_NORMAL")
    state = _enter_source_combat(data)
    assert len(state.combat.monsters) == 1
    state = step(state, _end_turn_action(state))
    assert len(state.combat.monsters) == 2
    assert state.combat.monsters[1].statuses["minion"] == 1
