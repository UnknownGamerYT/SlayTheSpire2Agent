from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

from helpers import project_root


def _load_combat_test() -> ModuleType:
    module_path = project_root() / "combat_test.py"
    spec = importlib.util.spec_from_file_location("combat_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["combat_test"] = module
    spec.loader.exec_module(module)
    return module


combat_test = _load_combat_test()


def test_combat_payload_starts_in_playable_combat() -> None:
    state = combat_test.create_combat_state(seed=123, ascension=0)
    payload = combat_test._state_payload(state, seed=123, message="ready")

    assert payload["character_id"] == "IRONCLAD"
    assert payload["ascension"] == 0
    assert payload["phase"] == "combat"
    assert payload["combat"]["turn"] == 1
    assert len(payload["combat"]["hand"]) == 5
    assert len(payload["combat"]["monsters"]) == 1
    assert len(payload["master_deck"]) == 10
    assert "burning_blood" in payload["relics"]
    assert {card["card_id"] for card in payload["master_deck"]} == {
        "STRIKE_IRONCLAD",
        "DEFEND_IRONCLAD",
        "BASH",
    }
    assert {action["type"] for action in payload["actions"]} >= {"play_card", "end_turn"}


def test_combat_payload_can_start_as_other_cached_characters() -> None:
    state = combat_test.create_combat_state(seed=123, ascension=0, character_id="SILENT")
    payload = combat_test._state_payload(state, seed=123, message="ready")

    assert payload["character_id"] == "SILENT"
    assert len(payload["master_deck"]) == 12
    assert "ring_of_the_snake" in payload["relics"]
    assert {card["card_id"] for card in payload["master_deck"]} >= {
        "STRIKE_SILENT",
        "DEFEND_SILENT",
        "NEUTRALIZE",
        "SURVIVOR",
    }


def test_combat_payload_exposes_defect_orbs() -> None:
    state = combat_test.create_combat_state(seed=123, ascension=0, character_id="DEFECT")
    payload = combat_test._state_payload(state, seed=123, message="ready")

    assert payload["character_id"] == "DEFECT"
    assert "cracked_core" in payload["relics"]
    assert payload["combat"]["orb_slots"] == 3
    assert payload["combat"]["orb_slots_open"] == 2
    assert payload["combat"]["orbs"] == [
        {"orb_id": "lightning", "name": "Lightning", "value": 0}
    ]


def test_combat_page_hides_debug_tools_until_toggled() -> None:
    html = combat_test._html_page()

    assert '<body class="debug-open">' not in html
    assert 'id="debug-toggle"' in html
    assert "debug-tools" in html
    assert 'id="pile-list"' in html
    assert 'list="status-options"' in html
    assert 'list="monster-options"' in html
    assert 'list="relic-options"' in html
    assert 'list="potion-options"' in html
    assert 'id="orbs"' in html
    assert 'id="orb-options"' in html
    assert 'id="orb-slots"' in html
    assert html.count('id="card-options"') == 1
    assert 'id="card-results"' not in html
    assert "function cardMatchScore" not in html


def test_combat_payload_includes_debug_dropdown_options() -> None:
    state = combat_test.create_combat_state(seed=123, ascension=0)
    payload = combat_test._state_payload(state, seed=123, message="ready")

    status_ids = {option["id"] for option in payload["status_options"]}
    relic_ids = {option["id"] for option in payload["relic_options"]}
    potion_ids = {option["id"] for option in payload["potion_options"]}
    monster_ids = {option["id"] for option in payload["monster_options"]}
    orb_ids = {option["id"] for option in payload["orb_options"]}

    assert {"strength", "dexterity", "vulnerable", "weak"} <= status_ids
    assert "anchor" in relic_ids
    assert "fire_potion" in potion_ids
    assert "debug_enemy" in monster_ids
    assert {"lightning", "frost", "dark", "plasma", "glass"} <= orb_ids


def test_card_library_can_be_searched_by_applied_status() -> None:
    cards = combat_test._card_library_payload()
    vulnerable_cards = [
        card for card in cards if "vulnerable" in card["statuses"]
    ]

    assert vulnerable_cards
    assert any("vulnerable" in card["search_text"] for card in vulnerable_cards)
    assert any(card["id"] == "assassinate" for card in vulnerable_cards)


def test_debug_actions_can_add_energy_statuses_and_cards() -> None:
    state = combat_test.create_combat_state(seed=124, ascension=0)
    original_hand_size = len(state.combat.hand)
    original_deck_size = len(state.master_deck)

    state, message = combat_test.apply_debug_action(
        state,
        {"action": "toggle_infinite_energy", "enabled": True},
    )
    assert message == "Infinite energy enabled."
    assert state.flags["debug_infinite_energy"] is True
    assert state.player.energy == 99
    assert state.combat.player.energy == 99

    state, _ = combat_test.apply_debug_action(
        state,
        {"action": "add_player_status", "status_id": "Strength", "amount": 2},
    )
    assert state.player.statuses["strength"] == 2
    assert state.combat.player.statuses["strength"] == 2

    state, _ = combat_test.apply_debug_action(
        state,
        {
            "action": "add_card",
            "card_id": "debug_blaster",
            "zone": "hand",
            "cost": 0,
            "damage": 12,
            "retain": True,
        },
    )
    assert len(state.combat.hand) == original_hand_size + 1
    added_card = state.combat.hand[-1]
    assert added_card.effects["damage"] == 12
    assert added_card.custom["retain"] is True

    state, _ = combat_test.apply_debug_action(
        state,
        {"action": "add_card", "card_id": "debug_deck_card", "zone": "deck"},
    )
    assert len(state.master_deck) == original_deck_size + 1


def test_debug_actions_can_channel_and_evoke_orbs() -> None:
    state = combat_test.create_combat_state(seed=124, ascension=0)
    monster_hp = state.combat.monsters[0].hp

    state, _ = combat_test.apply_debug_action(
        state,
        {"action": "set_orb_slots", "slots": 3},
    )
    assert state.combat.orb_slots == 3

    state, _ = combat_test.apply_debug_action(
        state,
        {"action": "channel_orb", "orb_id": "lightning", "amount": 1},
    )
    assert [orb.orb_id for orb in state.combat.orbs] == ["lightning"]

    state, _ = combat_test.apply_debug_action(
        state,
        {
            "action": "evoke_orb",
            "selector": "leftmost",
            "amount": 1,
            "target_id": state.combat.monsters[0].monster_id,
        },
    )
    assert state.combat.orbs == ()
    assert state.combat.monsters[0].hp == monster_hp - 8


def test_debug_card_can_include_orb_effects() -> None:
    state = combat_test.create_combat_state(seed=124, ascension=0)

    state, _ = combat_test.apply_debug_action(
        state,
        {
            "action": "add_card",
            "card_id": "debug_zap",
            "zone": "hand",
            "channel_orb": "frost",
            "orb_slot_delta": 1,
        },
    )

    card = state.combat.hand[-1]
    assert card.effects["channel_orb"] == {"orb": "frost", "amount": 1}
    assert card.effects["orb_slot_delta"] == 1


def test_debug_actions_can_mutate_and_spawn_enemies() -> None:
    state = combat_test.create_combat_state(seed=125, ascension=0)
    monster_id = state.combat.monsters[0].monster_id
    original_hp = state.combat.monsters[0].hp

    state, _ = combat_test.apply_debug_action(
        state,
        {
            "action": "add_enemy_status",
            "monster_id": monster_id,
            "status_id": "Vulnerable",
            "amount": 3,
        },
    )
    assert state.combat.monsters[0].statuses["vulnerable"] == 3

    state, _ = combat_test.apply_debug_action(
        state,
        {"action": "damage_enemy", "monster_id": monster_id, "amount": 4},
    )
    assert state.combat.monsters[0].hp == original_hp - 4

    state, _ = combat_test.apply_debug_action(
        state,
        {"action": "spawn_monster", "monster_id": "debug_sentry", "hp": 22, "damage": 7},
    )
    assert len(state.combat.monsters) == 2
    assert state.combat.monsters[-1].hp == 22
    assert state.combat.monsters[-1].intent_damage == 7

    state, _ = combat_test.apply_debug_action(state, {"action": "kill_all"})
    assert all(monster.hp == 0 for monster in state.combat.monsters)


def test_web_action_payload_can_drive_engine_step() -> None:
    state = combat_test.create_combat_state(seed=126, ascension=0)
    payload = combat_test._state_payload(state, seed=126, message="ready")
    end_turn = next(action for action in payload["actions"] if action["type"] == "end_turn")

    state, message = combat_test.apply_engine_action(state, end_turn)

    assert "not legal" not in message
    assert state.combat.turn == 2
