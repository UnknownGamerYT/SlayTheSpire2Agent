from __future__ import annotations

from combat.test_combat_engine_monsters import _end_turn_action, _play_first_card
from combat.test_monster_innate_hooks import _card, _combat, _monster

from sts2sim import legal_actions, load_state, serialize, step


def _hopper() -> dict:
    monster = _monster("THIEVING_HOPPER", {"ESCAPE_ARTIST": 5})
    ids = ("THIEVERY", "FLUTTER", "HAT_TRICK", "NAB", "ESCAPE")
    monster["moves"] = tuple(
        {
            "id": move,
            "name": move.title(),
            "intent": "Escape" if move == "ESCAPE" else ("Buff" if move == "FLUTTER" else "Attack"),
            "damage": {"normal": 5} if move not in {"FLUTTER", "ESCAPE"} else None,
            "powers": ({"power_id": "FLUTTER", "amount": 5, "target": "self"},)
            if move == "FLUTTER"
            else None,
        }
        for move in ids
    )
    monster["attack_pattern"] = {
        "initial_move": "THIEVERY",
        "states": tuple(
            {"id": move, "move_id": move, "type": "move", "next": ids[min(index + 1, 4)]}
            for index, move in enumerate(ids)
        ),
    }
    return monster


def _deck() -> tuple[dict, ...]:
    return (
        {**_card("keeper", damage=1000), "rarity": "Rare"},
        {
            **_card("victim", damage=14),
            "rarity": "Uncommon",
            "upgraded": True,
            "enchantments": ({"keyword": "glam", "amount": 1},),
        },
        {**_card("ordinary", damage=1), "rarity": "Common"},
    )


def test_thievery_removes_one_preferred_permanent_card_and_is_replayable() -> None:
    state = _combat(_hopper(), deck=_deck())
    restored = load_state(serialize(state))
    results = [step(candidate, _end_turn_action(candidate)) for candidate in (state, restored)]
    assert serialize(results[0]) == serialize(results[1])
    state = results[0]
    assert [card.card_id for card in state.master_deck] == ["keeper", "ordinary"]
    assert state.combat.monsters[0].metadata["stolen_card"]["card_id"] == "victim"
    assert state.combat.monsters[0].statuses["swipe"] == 1
    assert not any(
        card.card_id == "victim"
        for pile in (state.combat.hand, state.combat.draw_pile, state.combat.discard_pile)
        for card in pile
    )


def test_hopper_avoids_imbued_cards_when_unenchanted_cards_exist() -> None:
    deck = list(_deck())
    deck[1]["enchantments"] = ({"keyword": "imbued", "amount": 1},)
    state = _combat(_hopper(), deck=tuple(deck))
    state = step(state, _end_turn_action(state))
    assert state.combat.monsters[0].metadata["stolen_card"]["card_id"] != "victim"
    assert any(card.card_id == "victim" for card in state.master_deck)


def test_killing_hopper_offers_the_stolen_card_with_its_upgrade_and_enchantment() -> None:
    state = _combat(_hopper(), deck=_deck())
    state = step(state, _end_turn_action(state))
    state = _play_first_card(state, "keeper")
    assert state.phase.value == "reward"
    assert state.reward.card_option_groups == (("victim",),)
    state = load_state(serialize(state))
    action = next(
        action
        for action in legal_actions(state)
        if action.type == "take_reward_card" and action.target_id == "reward:card_group:0:0"
    )
    state = step(state, action)
    returned = next(card for card in state.master_deck if card.card_id == "victim")
    assert returned.upgraded is True
    assert returned.effects["damage"] == 14
    assert returned.enchantments[0].keyword == "glam"
    ids = [card.instance_id for card in state.master_deck]
    assert len(ids) == len(set(ids))


def test_player_can_skip_a_stolen_card_reward() -> None:
    state = _combat(_hopper(), deck=_deck())
    state = step(state, _end_turn_action(state))
    state = _play_first_card(state, "keeper")
    action = next(
        action
        for action in legal_actions(state)
        if action.type == "skip_reward" and action.target_id == "reward:card_group:0"
    )
    state = step(state, action)
    assert not any(card.card_id == "victim" for card in state.master_deck)
    assert state.reward.skipped_card_option_group_indices == (0,)


def test_escaping_hopper_keeps_the_stolen_card_and_does_not_award_gold() -> None:
    state = _combat(_hopper(), deck=_deck())
    state = state.model_copy(update={"flags": {**state.flags, "combat_reward_gold": 30}})
    for _ in range(5):
        state = step(state, _end_turn_action(state))
    assert state.phase.value == "reward"
    assert state.reward.gold == 0
    assert state.reward.card_option_groups == ()
    assert not any(card.card_id == "victim" for card in state.master_deck)


def test_no_eligible_card_does_not_softlock_thievery() -> None:
    state = _combat(_hopper())
    state = state.model_copy(update={"master_deck": ()})
    state = step(state, _end_turn_action(state))
    assert state.combat.monsters[0].move_id == "FLUTTER"
    assert "stolen_card" not in state.combat.monsters[0].metadata
    assert any(event.kind == "thieving_hopper_theft_skipped" for event in state.combat.last_events)


def test_thievery_falls_back_to_any_permanent_deck_version_when_no_priority_matches() -> None:
    token = {**_card("token", damage=1), "rarity": "Token"}
    status = {**_card("status", damage=0), "type": "status", "rarity": "Status"}
    state = _combat(_hopper(), deck=(token, status))
    state = step(state, _end_turn_action(state))
    assert state.combat.monsters[0].metadata["stolen_card"]["card_id"] in {"token", "status"}
    assert len(state.master_deck) == 1


def test_flutter_halves_attack_hits_then_stuns_and_advances_the_skipped_move() -> None:
    multi = {
        **_card("keeper", effects={"sequence": tuple({"damage": 4} for _ in range(6))}),
        "rarity": "Rare",
    }
    state = _combat(_hopper(), deck=(multi, _deck()[1]))
    state = step(state, _end_turn_action(state))
    state = step(state, _end_turn_action(state))
    assert state.combat.monsters[0].move_id == "HAT_TRICK"
    state = _play_first_card(state, "keeper")
    assert state.combat.monsters[0].hp == 86  # five half hits, then one full hit.
    assert "flutter" not in state.combat.monsters[0].statuses
    assert state.combat.monsters[0].intent == "Stun"
    assert state.combat.monsters[0].intent_damage == 0
    hp = state.combat.player.hp
    state = step(state, _end_turn_action(state))
    assert state.combat.player.hp == hp
    assert state.combat.monsters[0].move_id == "NAB"


def test_flutter_does_not_reduce_potion_damage_or_lose_a_charge_from_it() -> None:
    monster = _hopper()
    monster["innate_powers"] += ({"power_id": "FLUTTER", "amount": 5},)
    state = _combat(monster)
    state = state.model_copy(update={"potions": ("fire_potion",)})
    action = next(action for action in legal_actions(state) if action.type == "use_potion")
    state = step(state, action)
    assert state.combat.monsters[0].hp == 80
    assert state.combat.monsters[0].statuses["flutter"] == 5


def test_hopper_steals_before_thorns_can_kill_him_during_theft_attack() -> None:
    state = _combat(_hopper(), deck=_deck())
    combat = state.combat
    state = state.model_copy(update={"combat": combat.model_copy(update={
        "player": combat.player.model_copy(update={"statuses": {"thorns": 100}}),
    })})
    state = step(state, _end_turn_action(state))
    assert state.reward.card_option_groups == (("victim",),)
    events = state.combat.last_events
    theft = next(i for i, e in enumerate(events) if e.kind == "thieving_hopper_card_stolen")
    death = next(i for i, e in enumerate(events) if e.kind == "monster_defeated")
    hit = next(i for i, e in enumerate(events) if e.kind == "player_damaged")
    assert theft < death < hit
