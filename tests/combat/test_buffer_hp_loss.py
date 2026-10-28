from __future__ import annotations

import pytest
from test_named_monster_scripts import _combat
from test_remaining_monster_scripts import _end, _play

from sts2sim import legal_actions, load_state, serialize, step
from sts2sim.engine.transitions import (
    _damage_combat_player,
    _damage_player,
    _generated_status_cards,
    _lose_player_hp,
)


def _buffer_fight(*, stacks=1, statuses=None):
    state = _combat("TEST", moves=({"id": "WAIT", "intent": "Sleep"},))
    return state.model_copy(update={"combat": state.combat.model_copy(update={
        "player": state.combat.player.model_copy(update={
            "statuses": {"buffer": stacks, **(statuses or {})},
        }),
    })})


@pytest.mark.parametrize("card_id", ["toxic", "infection", "wither", "beckon"])
def test_buffer_prevents_status_card_damage_and_direct_hp_loss_at_end_turn(card_id) -> None:
    # BUFFER's source description says "times you would lose HP", including
    # Beckon's direct HP loss as well as the three damaging hand statuses.
    state = _buffer_fight()
    cards = _generated_status_cards(state.combat, card_id, 1)
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"hand": cards})})
    original = serialize(state)
    state = _end(load_state(original))
    assert state.combat.player.hp == 80
    assert "buffer" not in state.combat.player.statuses
    event = next(e for e in state.combat.last_events if e.source_id == cards[0].instance_id
                 and e.kind in {"player_damaged", "player_hp_lost"})
    assert event.amount == 0
    assert event.metadata["buffer_prevented_hp_loss"] > 0
    assert load_state(original).combat.player.statuses["buffer"] == 1


def test_two_buffer_stacks_protect_two_hand_cards_then_third_card_damages() -> None:
    state = _buffer_fight(stacks=2)
    cards = tuple(_generated_status_cards(state.combat, key, 1)[0]
                  for key in ("toxic", "infection", "wither"))
    state = state.model_copy(update={"combat": state.combat.model_copy(update={"hand": cards})})
    state = _end(state)
    assert state.combat.player.hp == 77
    assert "buffer" not in state.combat.player.statuses
    events = [e for e in state.combat.last_events if e.kind == "player_damaged"]
    assert [e.amount for e in events] == [0, 0, 3]


def test_self_hp_loss_card_consumes_one_buffer_stack_once_without_spending_block() -> None:
    state = _buffer_fight(stacks=2)
    state = state.model_copy(update={"combat": state.combat.model_copy(update={
        "player": state.combat.player.model_copy(update={"block": 12}),
    })})
    for expected_stacks in (1, 0):
        state = _play(state, card_type="skill", effects={"hp_loss": 3})
        assert state.combat.player.hp == 80
        assert state.combat.player.block == 12
        assert state.combat.player.statuses.get("buffer", 0) == expected_stacks
    state = _play(state, card_type="skill", effects={"hp_loss": 3})
    assert state.combat.player.hp == 77
    assert state.combat.player.block == 12


@pytest.mark.parametrize("damage_path", ["combat", "player"])
@pytest.mark.parametrize("amount,block,relics,hp", [
    (0, 0, (), 80),
    (2, 2, (), 80),
    (1, 0, ("tungsten_rod",), 80),
    (5, 0, (), 0),
])
def test_buffer_is_not_consumed_when_damage_cannot_reduce_hp(
    damage_path, amount, block, relics, hp,
):
    combat = _buffer_fight().combat
    player = combat.player.model_copy(update={"block": block, "hp": hp})
    if damage_path == "combat":
        combat, events = _damage_combat_player(
            combat.model_copy(update={"player": player}), amount, "test", relics=relics,
        )
        player = combat.player
        assert not any(e.kind == "buffer_prevented_hp_loss" for e in events)
    else:
        player, event = _damage_player(player, amount, "test", relics=relics)
        assert event.metadata["buffer_prevented_hp_loss"] == 0
    assert player.hp == hp
    assert player.statuses["buffer"] == 1


@pytest.mark.parametrize("amount,hp", [(0, 80), (5, 0)])
def test_zero_actual_direct_hp_loss_preserves_buffer(amount, hp) -> None:
    player = _buffer_fight().combat.player.model_copy(update={"hp": hp})
    player, event = _lose_player_hp(player, amount, "test")
    assert player.hp == hp
    assert player.statuses["buffer"] == 1
    assert event.amount == 0
    assert event.metadata["buffer_prevented_hp_loss"] == 0


def test_hp_loss_cap_preserves_buffer_and_prevented_loss_does_not_trigger_loss_effects() -> None:
    combat = _buffer_fight(statuses={"plated_armor": 3}).combat
    capped, _ = _damage_combat_player(
        combat.model_copy(update={"metadata": {"hp_lost_this_turn": 20}}),
        9, "test", relics=("beating_remnant", "self_forming_clay"),
    )
    assert capped.player.hp == 80
    assert capped.player.statuses == {"buffer": 1, "plated_armor": 3}
    protected, _ = _damage_combat_player(combat, 9, "test", relics=("self_forming_clay",))
    assert protected.player.hp == 80
    assert protected.player.statuses == {"plated_armor": 3}
    assert protected.metadata.get("hp_lost_this_turn", 0) == 0


def test_vulnerable_and_wrath_only_increase_attack_damage() -> None:
    combat = _buffer_fight(stacks=0, statuses={"vulnerable": 1, "stance_wrath": 1}).combat
    attack, _ = _damage_combat_player(combat, 4, "attack", is_attack=True)
    nonattack, _ = _damage_combat_player(combat, 4, "galvanic")
    status, event = _damage_player(combat.player, 4, "status")
    assert attack.player.hp == 68
    assert nonattack.player.hp == status.hp == 76
    assert event.metadata["incoming"] == 4


def test_vulnerable_increases_attack_card_damage_but_not_fire_potion() -> None:
    state = _buffer_fight(stacks=0)
    monster = state.combat.monsters[0].model_copy(update={"statuses": {"vulnerable": 2}})
    state = state.model_copy(update={
        "potions": ("fire_potion",),
        "combat": state.combat.model_copy(update={"monsters": (monster,)}),
    })
    state = _play(state, effects={"damage": 10})
    assert state.combat.monsters[0].hp == 85
    state = step(state, next(a for a in legal_actions(state) if a.type == "use_potion"))
    assert state.combat.monsters[0].hp == 65
