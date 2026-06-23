from __future__ import annotations

from sts2sim.live_parity import (
    compare_live_step_to_simulator,
    live_state_to_simulator_state,
    map_live_action_to_simulator_action,
    simulator_snapshot_for_compare,
)


def test_live_combat_state_can_be_reconstructed_for_simulator() -> None:
    state = _live_bash_before()

    sim_state = live_state_to_simulator_state(state, seed=0)
    snapshot = simulator_snapshot_for_compare(sim_state)

    assert snapshot["phase"] == "combat"
    assert snapshot["player"]["gold"] == 99
    assert snapshot["relics"] == ["burning_blood"]
    assert snapshot["combat"]["player"]["energy"] == 3
    assert snapshot["combat"]["monsters"][0]["monster_id"] == "NIBBIT_0"
    assert snapshot["combat"]["monsters"][0]["hp"] == 45
    assert snapshot["combat"]["hand"][0]["card_id"] == "BASH"


def test_live_play_card_action_maps_to_simulator_card_instance() -> None:
    state = _live_bash_before()
    sim_state = live_state_to_simulator_state(state, seed=0)

    action = map_live_action_to_simulator_action(
        state,
        {"action": "play_card", "card_index": 0, "target": "NIBBIT_0"},
        sim_state,
    )

    assert action.type.value == "play_card"
    assert action.card_instance_id == "live_hand_001"
    assert action.target_id == "NIBBIT_0"


def test_live_bash_step_matches_simulator_effects() -> None:
    result = compare_live_step_to_simulator(
        before=_live_bash_before(),
        action={"action": "play_card", "card_index": 0, "target": "NIBBIT_0"},
        after=_live_bash_after(),
        seed=0,
    )

    assert result.supported is True
    assert result.mismatch_count == 0
    assert result.mismatches == ()
    assert result.simulator_after["combat"]["monsters"][0]["hp"] == 37
    assert result.simulator_after["combat"]["monsters"][0]["statuses"]["vulnerable"] == 2
    assert result.simulator_after["combat"]["player"]["energy"] == 1
    assert result.simulator_after["combat"]["discard_pile_count"] == 1


def test_true_parity_reports_unsupported_non_combat_actions() -> None:
    result = compare_live_step_to_simulator(
        before={"state_type": "map", "run": {"act": 1}, "player": {"character": "The Ironclad"}},
        action={"action": "choose_map_node", "index": 0},
        after={"state_type": "monster"},
    )

    assert result.supported is False
    assert "combat states only" in result.reason


def _live_bash_before() -> dict[str, object]:
    return {
        "state_type": "monster",
        "run": {"act": 1, "floor": 1, "ascension": 0},
        "battle": {
            "round": 1,
            "turn": "player",
            "is_play_phase": True,
            "enemies": [
                {
                    "entity_id": "NIBBIT_0",
                    "name": "Nibbit",
                    "hp": 45,
                    "max_hp": 45,
                    "block": 0,
                    "status": [],
                    "intents": [{"type": "Attack", "label": "12"}],
                }
            ],
        },
        "player": {
            "character": "The Ironclad",
            "hp": 80,
            "max_hp": 80,
            "block": 0,
            "energy": 3,
            "max_energy": 3,
            "gold": 99,
            "status": [],
            "relics": [{"id": "BURNING_BLOOD", "name": "Burning Blood"}],
            "potions": [],
            "max_potion_slots": 3,
            "hand": [
                {
                    "index": 0,
                    "id": "BASH",
                    "name": "Bash",
                    "type": "Attack",
                    "cost": "2",
                    "target_type": "AnyEnemy",
                    "is_upgraded": False,
                    "can_play": True,
                }
            ],
            "draw_pile": [],
            "discard_pile": [],
            "exhaust_pile": [],
            "draw_pile_count": 0,
            "discard_pile_count": 0,
            "exhaust_pile_count": 0,
        },
    }


def _live_bash_after() -> dict[str, object]:
    after = _live_bash_before()
    battle = after["battle"]
    assert isinstance(battle, dict)
    enemies = battle["enemies"]
    assert isinstance(enemies, list)
    enemy = enemies[0]
    assert isinstance(enemy, dict)
    enemy["hp"] = 37
    enemy["status"] = [{"id": "VULNERABLE_POWER", "name": "Vulnerable", "amount": 2}]
    player = after["player"]
    assert isinstance(player, dict)
    player["energy"] = 1
    player["hand"] = []
    player["discard_pile"] = [{"name": "Bash", "cost": "2"}]
    player["discard_pile_count"] = 1
    return after
